// Licensed to Elasticsearch B.V. under one or more contributor
// license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright
// ownership. Elasticsearch B.V. licenses this file to you under
// the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package esutil

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/elastic/go-elasticsearch/v9/esapi"
)

// ErrVisibilityTimeout is returned by [VisibilityWaiter.Wait] when the
// deadline expires before every document became searchable. The returned
// result lists the documents that are still missing, so a caller does not have
// to guess.
var ErrVisibilityTimeout = errors.New("esutil: documents did not become visible before the deadline")

// DocumentRef identifies a written document.
type DocumentRef struct {
	Index   string
	ID      string
	Routing string
}

func (d DocumentRef) key() string { return d.Index + "\x00" + d.Routing + "\x00" + d.ID }

// VisibilityWaiterConfig configures a [VisibilityWaiter].
type VisibilityWaiterConfig struct {
	// Client is the Elasticsearch transport (required).
	Client esapi.Transport
	// Index is the default index for DocumentRefs that leave Index empty.
	Index string
	// PollInterval is the delay between refresh-and-search rounds. Defaults
	// to 100ms. It is also the first backoff step.
	PollInterval time.Duration
	// MaxPollInterval caps the exponential growth of the poll interval.
	// Defaults to 1s.
	MaxPollInterval time.Duration
	// DisableRefresh turns off the refresh-before-search step. By default
	// each index is refreshed before it is searched, which is what makes the
	// wait meaningful for documents still sitting in an in-memory buffer.
	// Disable it only when an external refresher already runs.
	DisableRefresh bool
	// RequireCompleteResults makes Wait fail when a participating index could
	// not be searched. When false (the default), the index is reported through
	// Coverage and the remaining documents may still be reported visible.
	RequireCompleteResults bool
}

func (c VisibilityWaiterConfig) withDefaults() VisibilityWaiterConfig {
	if c.PollInterval <= 0 {
		c.PollInterval = 100 * time.Millisecond
	}
	if c.MaxPollInterval <= 0 {
		c.MaxPollInterval = time.Second
	}
	if c.MaxPollInterval < c.PollInterval {
		c.MaxPollInterval = c.PollInterval
	}
	return c
}

// VisibilityResult reports the outcome of one Wait call.
type VisibilityResult struct {
	// Visible documents were found by a search round.
	Visible []DocumentRef
	// Missing documents were not found before the deadline (or before the
	// call returned).
	Missing []DocumentRef
	// Polls is the number of refresh-and-search rounds performed.
	Polls int
	// Elapsed is the wall-clock time spent waiting.
	Elapsed time.Duration
	// Coverage aggregates shard coverage across all polled indices in the
	// final round. An index that could not be searched contributes to
	// IncompleteIndices.
	Coverage VisibilityCoverage
}

// VisibilityCoverage describes search coverage limits observed while waiting.
type VisibilityCoverage struct {
	// IncompleteIndices lists indices that failed to refresh or could not be
	// searched at all.
	IncompleteIndices []string
	// ShardsFailed is the number of shards that failed in the final round.
	ShardsFailed int
	// ShardsSuccessful is the number of shards that responded in the final
	// round.
	ShardsSuccessful int
	// TimedOut is set when the server timed out on the final round.
	TimedOut bool
}

// Complete reports whether every index and shard responded.
func (c VisibilityCoverage) Complete() bool {
	return len(c.IncompleteIndices) == 0 && c.ShardsFailed == 0 && !c.TimedOut
}

// VisibilityWaiter waits until freshly written documents are searchable.
//
// It replaces ad-hoc sleeps: callers pass a context with an explicit deadline,
// and Wait polls the server (optionally refreshing indices) until every
// document is found or the deadline expires. It never waits without a bound.
type VisibilityWaiter struct {
	cfg VisibilityWaiterConfig
}

// NewVisibilityWaiter validates cfg and returns a waiter.
func NewVisibilityWaiter(cfg VisibilityWaiterConfig) (*VisibilityWaiter, error) {
	if cfg.Client == nil {
		return nil, errors.New("esutil: VisibilityWaiterConfig.Client is required")
	}
	return &VisibilityWaiter{cfg: cfg.withDefaults()}, nil
}

// Wait blocks until every document in docs is visible to search, the context
// is done, or the configured condition cannot be satisfied. When the deadline
// expires it returns the documents still missing together with
// ErrVisibilityTimeout. Cancellation returns ctx.Err() wrapped so errors.Is
// works, and is reported as such rather than as a timeout.
func (w *VisibilityWaiter) Wait(ctx context.Context, docs []DocumentRef) (*VisibilityResult, error) {
	start := time.Now()
	result := &VisibilityResult{}

	pending := w.normalize(docs)
	all := make(map[string]DocumentRef, len(pending))
	for k, v := range pending {
		all[k] = v
	}
	if len(pending) == 0 {
		result.Elapsed = time.Since(start)
		return result, nil
	}
	visible := func(found map[string]struct{}) []DocumentRef {
		refs := make(map[string]DocumentRef, len(found))
		for k := range found {
			if ref, ok := all[k]; ok {
				refs[k] = ref
			}
		}
		return sortedRefs(refs)
	}

	interval := w.cfg.PollInterval
	for {
		if err := ctx.Err(); err != nil {
			result.Missing = sortedRefs(pending)
			result.Elapsed = time.Since(start)
			return result, fmt.Errorf("esutil: visibility wait cancelled: %w", err)
		}

		result.Polls++
		found, coverage := w.round(ctx, pending)
		result.Coverage = coverage

		for key := range found {
			delete(pending, key)
		}
		if len(pending) == 0 {
			result.Visible = visible(found)
			result.Elapsed = time.Since(start)
			result.Coverage = coverage
			return result, nil
		}

		if w.cfg.RequireCompleteResults && len(coverage.IncompleteIndices) > 0 {
			result.Missing = sortedRefs(pending)
			result.Elapsed = time.Since(start)
			result.Visible = visible(found)
			return result, fmt.Errorf("%w: indices unavailable: %v", ErrIncompleteResults, coverage.IncompleteIndices)
		}

		timer := time.NewTimer(interval)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			result.Missing = sortedRefs(pending)
			result.Visible = visible(found)
			result.Elapsed = time.Since(start)
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return result, fmt.Errorf("%w: %d document(s) still missing: %w",
					ErrVisibilityTimeout, len(pending), ctx.Err())
			}
			return result, fmt.Errorf("esutil: visibility wait cancelled: %w", ctx.Err())
		}
		timer.Stop()

		interval *= 2
		if interval > w.cfg.MaxPollInterval {
			interval = w.cfg.MaxPollInterval
		}
	}
}

func (w *VisibilityWaiter) normalize(docs []DocumentRef) map[string]DocumentRef {
	pending := make(map[string]DocumentRef, len(docs))
	for _, d := range docs {
		if d.Index == "" {
			d.Index = w.cfg.Index
		}
		pending[d.key()] = d
	}
	return pending
}

type refGroup struct {
	index   string
	routing string
	ids     []string
}

func groupRefs(pending map[string]DocumentRef) []refGroup {
	groups := make(map[string]*refGroup)
	for _, d := range pending {
		gk := d.Index + "\x00" + d.Routing
		g := groups[gk]
		if g == nil {
			g = &refGroup{index: d.Index, routing: d.Routing}
			groups[gk] = g
		}
		g.ids = append(g.ids, d.ID)
	}
	out := make([]refGroup, 0, len(groups))
	for _, g := range groups {
		sort.Strings(g.ids)
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].index != out[j].index {
			return out[i].index < out[j].index
		}
		return out[i].routing < out[j].routing
	})
	return out
}

// round refreshes the pending indices and searches for the pending documents.
func (w *VisibilityWaiter) round(ctx context.Context, pending map[string]DocumentRef) (map[string]struct{}, VisibilityCoverage) {
	found := make(map[string]struct{})
	var coverage VisibilityCoverage

	for _, g := range groupRefs(pending) {
		if err := ctx.Err(); err != nil {
			return found, coverage
		}

		if !w.cfg.DisableRefresh {
			if err := w.refresh(ctx, g.index); err != nil {
				if ctx.Err() != nil {
					return found, coverage
				}
				coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
				continue
			}
		}

		ids, err := json.Marshal(g.ids)
		if err != nil {
			coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
			continue
		}
		body := fmt.Sprintf(`{"size":%d,"_source":false,"track_total_hits":true,"query":{"ids":{"values":%s}}}`,
			len(g.ids), ids)

		req := esapi.SearchRequest{
			Index: []string{g.index},
			Body:  bytesReader([]byte(body)),
		}
		if g.routing != "" {
			req.Routing = []string{g.routing}
		}
		res, err := req.Do(ctx, w.cfg.Client)
		if err != nil {
			if ctx.Err() != nil {
				return found, coverage
			}
			coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
			continue
		}
		status := res.StatusCode
		if status == http.StatusNotFound {
			// The index does not exist yet; treat as unavailable coverage
			// rather than as an error so other groups still progress.
			if res.Body != nil {
				_ = res.Body.Close()
			}
			coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
			continue
		}
		if res.IsError() {
			if res.Body != nil {
				_ = res.Body.Close()
			}
			coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
			continue
		}

		var payload struct {
			TimedOut bool `json:"timed_out"`
			Shards   struct {
				Successful int `json:"successful"`
				Failed     int `json:"failed"`
			} `json:"_shards"`
			Hits struct {
				Hits []struct {
					ID string `json:"_id"`
				} `json:"hits"`
			} `json:"hits"`
		}
		decodeErr := json.NewDecoder(res.Body).Decode(&payload)
		_ = res.Body.Close()
		if decodeErr != nil {
			coverage.IncompleteIndices = append(coverage.IncompleteIndices, g.index)
			continue
		}

		coverage.TimedOut = coverage.TimedOut || payload.TimedOut
		coverage.ShardsFailed += payload.Shards.Failed
		coverage.ShardsSuccessful += payload.Shards.Successful
		for _, h := range payload.Hits.Hits {
			found[DocumentRef{Index: g.index, Routing: g.routing, ID: h.ID}.key()] = struct{}{}
		}
	}

	sort.Strings(coverage.IncompleteIndices)
	return found, coverage
}

func (w *VisibilityWaiter) refresh(ctx context.Context, index string) error {
	req := esapi.IndicesRefreshRequest{Index: []string{index}}
	res, err := req.Do(ctx, w.cfg.Client)
	if err != nil {
		return err
	}
	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}
	if res.IsError() {
		return fmt.Errorf("refresh %q: status %d", index, res.StatusCode)
	}
	return nil
}

func bytesReader(data []byte) *bytes.Reader { return bytes.NewReader(data) }

func sortedRefs(m map[string]DocumentRef) []DocumentRef {
	out := make([]DocumentRef, 0, len(m))
	for _, d := range m {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Index != out[j].Index {
			return out[i].Index < out[j].Index
		}
		if out[i].Routing != out[j].Routing {
			return out[i].Routing < out[j].Routing
		}
		return out[i].ID < out[j].ID
	})
	return out
}
