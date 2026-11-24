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

//go:build !integration
// +build !integration

package esutil

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elastic/go-elasticsearch/v9"
)

// fakeDoc is one document the fake server has accepted.
type fakeDoc struct {
	Index   string
	Routing string
	ID      string
	Body    string
	Version int64
}

func (d fakeDoc) key() string { return d.Index + "\x00" + d.Routing + "\x00" + d.ID }

// fakeBulkRequest records what the server received in one _bulk call.
type fakeBulkRequest struct {
	Actions []fakeAction
}

type fakeAction struct {
	Op          string
	Index       string
	Routing     string
	ID          string
	Body        string
	Version     int64
	VersionType string
}

// fakeES is a small, in-memory stand-in for Elasticsearch. It implements the
// handful of endpoints these tests need: _bulk, _pit open/close, _search and
// _refresh.
type fakeES struct {
	t   *testing.T
	srv *httptest.Server

	mu       sync.Mutex
	docs     map[string]fakeDoc
	autoID   int
	requests []fakeBulkRequest
	// lands counts applications per document key.
	lands map[string]int

	// rateLimited lists document IDs that answer 429 for the given number of
	// attempts.
	rateLimited map[string]int
	// bulkUnavailable makes the next N whole bulk requests answer 503.
	bulkUnavailable int
	// conflicts lists document IDs that always answer 409.
	conflicts map[string]bool

	// PIT state.
	pitSeq      int
	pitValid    map[string]bool
	pitDocs     map[string][]fakeDoc
	refusePIT   int
	refuseCodes int

	// search behavior.
	partialShards bool
	// unavailableIndices are treated as missing: PIT open succeeds only when
	// ignore_unavailable is requested.
	unavailableIndices map[string]bool
}

func newFakeES(t *testing.T) *fakeES {
	t.Helper()
	f := &fakeES{
		t:                  t,
		docs:               map[string]fakeDoc{},
		lands:              map[string]int{},
		rateLimited:        map[string]int{},
		conflicts:          map[string]bool{},
		pitValid:           map[string]bool{},
		pitDocs:            map[string][]fakeDoc{},
		unavailableIndices: map[string]bool{},
	}
	f.srv = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeES) client(t *testing.T) *elasticsearch.Client {
	t.Helper()
	es, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses:    []string{f.srv.URL},
		DisableRetry: true, // replay classification is what we are testing
	})
	if err != nil {
		t.Fatalf("new client: %s", err)
	}
	return es
}

func (f *fakeES) handle(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Elastic-Product", "Elasticsearch")
	path := r.URL.Path

	switch {
	case strings.HasSuffix(path, "/_bulk"):
		f.handleBulk(w, r)
	case strings.HasSuffix(path, "/_pit") && r.Method == http.MethodPost:
		f.handleOpenPIT(w, r)
	case path == "/_pit" && r.Method == http.MethodDelete:
		f.handleClosePIT(w, r)
	case strings.HasSuffix(path, "/_search"):
		f.handleSearch(w, r)
	case strings.HasSuffix(path, "/_refresh"):
		fmt.Fprint(w, `{"_shards":{"total":1,"successful":1,"failed":0}}`)
	default:
		http.NotFound(w, r)
	}
}

func (f *fakeES) handleBulk(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	actions, err := parseBulkRequest(r.Body)
	if err != nil {
		f.t.Errorf("fake bulk: cannot parse request: %s", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	f.requests = append(f.requests, fakeBulkRequest{Actions: actions})

	if f.bulkUnavailable > 0 {
		f.bulkUnavailable--
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, `{"error":{"type":"unavailable_shards_exception","reason":"primary shard is not active"}}`)
		return
	}

	var items bytes.Buffer
	items.WriteByte('[')
	for i, a := range actions {
		if i > 0 {
			items.WriteByte(',')
		}
		if f.conflicts[a.ID] {
			fmt.Fprintf(&items, `{"%s":{"_index":%q,"_id":%q,"status":409,"error":{"type":"version_conflict_engine_exception","reason":"version conflict"}}}`,
				a.Op, a.Index, a.ID)
			continue
		}
		if n := f.rateLimited[a.ID]; n > 0 {
			f.rateLimited[a.ID] = n - 1
			fmt.Fprintf(&items, `{"%s":{"_index":%q,"_id":%q,"status":429,"error":{"type":"es_rejected_execution_exception","reason":"rejected execution"}}}`,
				a.Op, a.Index, a.ID)
			continue
		}

		id := a.ID
		if id == "" {
			f.autoID++
			id = fmt.Sprintf("auto-%d", f.autoID)
		}
		doc := fakeDoc{Index: a.Index, Routing: a.Routing, ID: id, Body: a.Body, Version: a.Version}
		if a.VersionType == "external" {
			if existing, ok := f.docs[doc.key()]; ok && existing.Version >= a.Version {
				fmt.Fprintf(&items, `{"%s":{"_index":%q,"_id":%q,"status":409,"error":{"type":"version_conflict_engine_exception","reason":"current version [%d] is higher or equal to the one provided [%d]"}}}`,
					a.Op, a.Index, a.ID, existing.Version, a.Version)
				continue
			}
		}
		f.docs[doc.key()] = doc
		f.lands[doc.key()]++
		fmt.Fprintf(&items, `{"%s":{"_index":%q,"_id":%q,"_version":1,"result":"created","status":201,"_shards":{"total":1,"successful":1,"failed":0}}}`,
			a.Op, a.Index, id)
	}
	items.WriteByte(']')
	fmt.Fprintf(w, `{"took":1,"errors":false,"items":%s}`, items.String())
}

func (f *fakeES) handleOpenPIT(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.unavailableIndices) > 0 {
		ignore := r.URL.Query().Get("ignore_unavailable") == "true"
		for name := range f.unavailableIndices {
			if !ignore {
				w.WriteHeader(http.StatusNotFound)
				fmt.Fprintf(w, `{"error":{"type":"index_not_found_exception","reason":"no such index [%s]"}}`, name)
				return
			}
		}
	}

	if f.refusePIT > 0 {
		f.refusePIT--
		code := f.refuseCodes
		if code == 0 {
			code = http.StatusBadRequest
		}
		w.WriteHeader(code)
		fmt.Fprint(w, `{"error":{"type":"illegal_argument_exception","reason":"point in time is not supported"}}`)
		return
	}

	f.pitSeq++
	id := fmt.Sprintf("pit-%d", f.pitSeq)
	f.pitValid[id] = true
	snapshot := make([]fakeDoc, 0, len(f.docs))
	for _, d := range f.docs {
		snapshot = append(snapshot, d)
	}
	sort.Slice(snapshot, func(i, j int) bool { return snapshot[i].key() < snapshot[j].key() })
	f.pitDocs[id] = snapshot
	fmt.Fprintf(w, `{"id":%q}`, id)
}

func (f *fakeES) handleClosePIT(w http.ResponseWriter, r *http.Request) {
	var payload struct {
		ID string `json:"id"`
	}
	_ = json.NewDecoder(r.Body).Decode(&payload)
	f.mu.Lock()
	delete(f.pitValid, payload.ID)
	delete(f.pitDocs, payload.ID)
	f.mu.Unlock()
	fmt.Fprint(w, `{"succeeded":true,"num_freed":1}`)
}

func (f *fakeES) handleSearch(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Pit *struct {
			ID string `json:"id"`
		} `json:"pit"`
		SearchAfter []any           `json:"search_after"`
		Query       json.RawMessage `json:"query"`
		Size        int             `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if body.Pit == nil || body.Pit.ID == "" {
		// A plain search is only used by the visibility waiter; it sees
		// everything that has landed.
		f.mu.Lock()
		f.writeSearchResponse(w, f.allDocsLocked(), body.Size, false)
		f.mu.Unlock()
		return
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	snapshot, ok := f.pitDocs[body.Pit.ID]
	if !ok || !f.pitValid[body.Pit.ID] {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":{"type":"search_context_missing_exception","reason":"No search context found for id"}}`)
		return
	}
	f.writeSearchResponse(w, snapshot, body.Size, f.partialShards)
}

func (f *fakeES) allDocsLocked() []fakeDoc {
	out := make([]fakeDoc, 0, len(f.docs))
	for _, d := range f.docs {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].key() < out[j].key() })
	return out
}

func (f *fakeES) writeSearchResponse(w http.ResponseWriter, docs []fakeDoc, size int, partial bool) {
	if size <= 0 || size > len(docs) {
		size = len(docs)
	}
	var hits bytes.Buffer
	hits.WriteByte('[')
	for i := 0; i < size; i++ {
		if i > 0 {
			hits.WriteByte(',')
		}
		d := docs[i]
		fmt.Fprintf(&hits, `{"_index":%q,"_id":%q,"_score":1.0,"_source":%s,"sort":[%d]}`, d.Index, d.ID, d.Body, i)
	}
	hits.WriteByte(']')

	failed := 0
	failures := "[]"
	successful := 1
	if partial {
		failed = 1
		successful = 0
		failures = `[{"index":"orders","shard":1,"reason":{"type":"node_disconnected_exception","reason":"node left the cluster"}}]`
	}
	fmt.Fprintf(w, `{"took":1,"timed_out":false,"_shards":{"total":%d,"successful":%d,"failed":%d,"failures":%s},"hits":{"total":{"value":%d,"relation":"eq"},"hits":%s}}`,
		successful+failed, successful, failed, failures, len(docs), hits.String())
}

func parseBulkRequest(r io.Reader) ([]fakeAction, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), 1<<20)
	var actions []fakeAction
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var meta map[string]json.RawMessage
		if err := json.Unmarshal(line, &meta); err != nil {
			return nil, fmt.Errorf("metadata line: %w", err)
		}
		var action fakeAction
		for op, raw := range meta {
			action.Op = op
			var fields struct {
				Index       string `json:"_index"`
				ID          string `json:"_id"`
				Routing     string `json:"routing"`
				Version     int64  `json:"version"`
				VersionType string `json:"version_type"`
			}
			if err := json.Unmarshal(raw, &fields); err != nil {
				return nil, fmt.Errorf("%s metadata: %w", op, err)
			}
			action.Index = fields.Index
			action.ID = fields.ID
			action.Routing = fields.Routing
			action.Version = fields.Version
			action.VersionType = fields.VersionType
		}
		if action.Op != "delete" && scanner.Scan() {
			action.Body = string(bytes.TrimSpace(scanner.Bytes()))
		}
		actions = append(actions, action)
	}
	return actions, scanner.Err()
}

// hookTransport delegates to a real RoundTripper but lets a test inject a
// transport error before the request is sent.
type hookTransport struct {
	base http.RoundTripper
	hook func(*http.Request) error
}

func (t *hookTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if t.hook != nil {
		if err := t.hook(r); err != nil {
			return nil, err
		}
	}
	return t.base.RoundTrip(r)
}

func TestBulkReplay_PartialRateLimit(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["2"] = 1
	fake.rateLimited["3"] = 2

	client := fake.client(t)
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1, // flush per item so each call is independently observable
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 5,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	var failures atomic.Int32
	for i := 1; i <= 4; i++ {
		id := fmt.Sprint(i)
		if err := indexer.Add(context.Background(), BulkIndexerItem{
			Action:     "index",
			DocumentID: id,
			Body:       strings.NewReader(fmt.Sprintf(`{"id":%q,"content":"v1"}`, id)),
			OnFailure: func(_ context.Context, _ BulkIndexerItem, _ BulkIndexerResponseItem, _ error) {
				failures.Add(1)
			},
		}); err != nil {
			t.Fatalf("add %s: %s", id, err)
		}
	}

	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumAdded != 4 {
		t.Errorf("NumAdded = %d, want 4", stats.NumAdded)
	}
	if stats.NumFlushed != 2 || stats.NumReplayed != 2 {
		t.Errorf("flushed/replayed = %d/%d, want 2/2", stats.NumFlushed, stats.NumReplayed)
	}
	if stats.NumFailed != 0 || stats.NumAbandoned != 0 {
		t.Errorf("failed/abandoned = %d/%d, want 0/0", stats.NumFailed, stats.NumAbandoned)
	}
	if failures.Load() != 0 {
		t.Errorf("OnFailure called %d times, want 0", failures.Load())
	}
	if got := stats.NumFlushed + stats.NumReplayed + stats.NumFailed + stats.NumAbandoned; got != stats.NumAdded {
		t.Errorf("partition sums to %d, want %d", got, stats.NumAdded)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 4 {
		t.Fatalf("landed %d documents, want 4: %+v", len(fake.docs), fake.docs)
	}
	for key, n := range fake.lands {
		if n != 1 {
			t.Errorf("document %q landed %d times, want 1", key, n)
		}
	}
	// Only the rate-limited documents were retried: 4 initial requests plus
	// 2 replay requests, or 3 if both retries happened to become ready
	// together and share a batch.
	if got := len(fake.requests); got < 6 || got > 7 {
		t.Errorf("bulk requests = %d, want 6 or 7", got)
	}
}

func TestBulkReplay_WholeRequestFailureThenSuccess(t *testing.T) {
	fake := newFakeES(t)
	fake.bulkUnavailable = 2

	var attempts atomic.Int32
	base := http.DefaultTransport
	client, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses:    []string{fake.srv.URL},
		DisableRetry: true,
		Transport: &hookTransport{base: base, hook: func(*http.Request) error {
			attempts.Add(1)
			return nil
		}},
	})
	if err != nil {
		t.Fatalf("new client: %s", err)
	}

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 5,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
		OnError: func(context.Context, error) {},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	for i := 1; i <= 3; i++ {
		id := fmt.Sprint(i)
		if err := indexer.Add(context.Background(), BulkIndexerItem{
			Action:     "index",
			DocumentID: id,
			Body:       strings.NewReader(fmt.Sprintf(`{"id":%q}`, id)),
		}); err != nil {
			t.Fatalf("add: %s", err)
		}
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	// The first two items are sent alone (FlushBytes=1) and hit the two 503s;
	// the third succeeds immediately. The two failures then share one replay
	// request, so exactly one item is a first-attempt success.
	if stats.NumFlushed != 1 || stats.NumReplayed != 2 || stats.NumFailed != 0 || stats.NumAbandoned != 0 {
		t.Errorf("stats = %+v, want flushed=1 replayed=2", stats)
	}
	// 3 initial requests, then the two failed items are replayed. If their
	// backoff deadlines coincide they share one replay request (4 total);
	// otherwise each gets its own (5 total). Both are correct.
	if got := attempts.Load(); got < 4 || got > 5 {
		t.Errorf("transport attempts = %d, want 4 or 5", got)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 3 {
		t.Errorf("landed %d docs, want 3", len(fake.docs))
	}
	for key, n := range fake.lands {
		if n != 1 {
			t.Errorf("document %q landed %d times, want 1", key, n)
		}
	}
}

func TestBulkReplay_ConnectionInterrupted(t *testing.T) {
	fake := newFakeES(t)

	var calls atomic.Int32
	base := http.DefaultTransport
	client, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses:    []string{fake.srv.URL},
		DisableRetry: true,
		Transport: &hookTransport{base: base, hook: func(*http.Request) error {
			// Fail the first transport attempt entirely, then succeed.
			if calls.Add(1) == 1 {
				return errors.New("connection reset by peer")
			}
			return nil
		}},
	})
	if err != nil {
		t.Fatalf("new client: %s", err)
	}

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 3,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
		OnError: func(context.Context, error) {},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "index",
		DocumentID: "42",
		Body:       strings.NewReader(`{"content":"original"}`),
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumReplayed != 1 || stats.NumFailed != 0 {
		t.Errorf("stats = %+v, want replayed=1 failed=0", stats)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 1 {
		t.Fatalf("landed %d documents, want 1", len(fake.docs))
	}
	for _, d := range fake.docs {
		if d.Body != `{"content":"original"}` {
			t.Errorf("landed body = %s, want the original content unchanged", d.Body)
		}
	}
}

func TestBulkReplay_VersionConflictIsTerminal(t *testing.T) {
	fake := newFakeES(t)
	fake.conflicts["doc-1"] = true

	client := fake.client(t)
	var failureErr error
	var failureInfo BulkIndexerResponseItem
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 5,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	version := int64(1)
	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:      "index",
		DocumentID:  "doc-1",
		Version:     &version,
		VersionType: "external",
		Body:        strings.NewReader(`{"content":"stale"}`),
		OnFailure: func(_ context.Context, _ BulkIndexerItem, info BulkIndexerResponseItem, err error) {
			failureErr = err
			failureInfo = info
		},
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumFailed != 1 || stats.NumReplayed != 0 || stats.NumAbandoned != 0 {
		t.Errorf("stats = %+v, want one terminal failure and no replay", stats)
	}
	if !errors.Is(failureErr, ErrItemNotRetryable) {
		t.Errorf("OnFailure error = %v, want ErrItemNotRetryable", failureErr)
	}
	if failureInfo.Error.Type != "version_conflict_engine_exception" {
		t.Errorf("failure info type = %q, want version_conflict", failureInfo.Error.Type)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 0 {
		t.Errorf("conflicting document landed: %+v", fake.docs)
	}
	if len(fake.requests) != 1 {
		t.Errorf("bulk requests = %d, want 1 (no replay of a conflict)", len(fake.requests))
	}
}

func TestBulkReplay_SameBatchTwiceNoDuplicates(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay:        &BulkReplayPolicy{MaxAttempts: 3, Backoff: func(int) time.Duration { return time.Millisecond }},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	for run := 0; run < 2; run++ {
		for i := 1; i <= 5; i++ {
			id := fmt.Sprint(i)
			if err := indexer.Add(context.Background(), BulkIndexerItem{
				Action:     "index",
				DocumentID: id,
				Body:       strings.NewReader(fmt.Sprintf(`{"id":%q,"run":%d}`, id, run)),
			}); err != nil {
				t.Fatalf("add: %s", err)
			}
		}
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 5 {
		t.Errorf("landed %d documents, want 5 distinct documents", len(fake.docs))
	}
	for key, d := range fake.docs {
		if !strings.Contains(d.Body, `"run":1`) {
			t.Errorf("document %q content = %s, want the later submission to win", key, d.Body)
		}
	}
	if stats := indexer.Stats(); stats.NumAdded != 10 {
		t.Errorf("NumAdded = %d, want 10", stats.NumAdded)
	}
}

func TestBulkReplay_AutoIDAndCreateAreNotReplayed(t *testing.T) {
	fake := newFakeES(t)
	fake.bulkUnavailable = 1           // fails the first (auto-ID) request
	fake.rateLimited["explicit"] = 100 // the create item is retryable at the item level
	client := fake.client(t)

	var reasons sync.Map
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay:        &BulkReplayPolicy{MaxAttempts: 3, Backoff: func(int) time.Duration { return time.Millisecond }},
		OnError:       func(context.Context, error) {},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action: "index", // no DocumentID: Elasticsearch would mint a new one
		Body:   strings.NewReader(`{"content":"auto"}`),
		OnFailure: func(_ context.Context, _ BulkIndexerItem, _ BulkIndexerResponseItem, err error) {
			reasons.Store("auto", err)
		},
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "create",
		DocumentID: "explicit",
		Body:       strings.NewReader(`{"content":"create"}`),
		OnFailure: func(_ context.Context, _ BulkIndexerItem, _ BulkIndexerResponseItem, err error) {
			reasons.Store("create", err)
		},
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumFailed != 2 || stats.NumAbandoned != 0 || stats.NumReplayed != 0 {
		t.Errorf("stats = %+v, want two terminal failures", stats)
	}
	if stats.NumFlushed+stats.NumReplayed+stats.NumFailed+stats.NumAbandoned != stats.NumAdded {
		t.Errorf("partition does not sum to NumAdded: %+v", stats)
	}
	for _, kind := range []string{"auto", "create"} {
		v, ok := reasons.Load(kind)
		if !ok {
			t.Fatalf("%s item received no OnFailure callback", kind)
		}
		if !errors.Is(v.(error), ErrItemNotReplayable) {
			t.Errorf("%s error = %v, want ErrItemNotReplayable", kind, v)
		}
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 0 {
		t.Errorf("non-replayable items landed: %+v", fake.docs)
	}
}

func TestBulkReplay_AbandonsWhenBudgetExhausted(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["stuck"] = 100

	client := fake.client(t)
	var failureErr error
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 3,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
		OnError: func(context.Context, error) {},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "index",
		DocumentID: "stuck",
		Body:       strings.NewReader(`{"content":"x"}`),
		OnFailure: func(_ context.Context, _ BulkIndexerItem, _ BulkIndexerResponseItem, err error) {
			failureErr = err
		},
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumAbandoned != 1 {
		t.Errorf("NumAbandoned = %d, want 1; stats=%+v", stats.NumAbandoned, stats)
	}
	if !errors.Is(failureErr, ErrReplayExhausted) {
		t.Errorf("OnFailure error = %v, want ErrReplayExhausted", failureErr)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.requests) != 3 {
		t.Errorf("bulk requests = %d, want 3 (MaxAttempts)", len(fake.requests))
	}
}

func TestBulkReplay_GroupsNeverMix(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["1"] = 1
	fake.rateLimited["2"] = 1
	fake.rateLimited["3"] = 1
	fake.rateLimited["4"] = 1

	client := fake.client(t)
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts:  3,
			MaxBatchSize: 2,
			Backoff:      func(int) time.Duration { return time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	items := []BulkIndexerItem{
		{Action: "index", Index: "orders", Routing: "r1", DocumentID: "1", Body: strings.NewReader(`{"n":1}`)},
		{Action: "index", Index: "orders", Routing: "r2", DocumentID: "2", Body: strings.NewReader(`{"n":2}`)},
		{Action: "index", Index: "customers", Routing: "r1", DocumentID: "3", Body: strings.NewReader(`{"n":3}`)},
		{Action: "index", Index: "customers", Routing: "r1", DocumentID: "4", Body: strings.NewReader(`{"n":4}`)},
	}
	for _, item := range items {
		if err := indexer.Add(context.Background(), item); err != nil {
			t.Fatalf("add: %s", err)
		}
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 4 {
		t.Fatalf("landed %d documents, want 4", len(fake.docs))
	}
	// Every request after the first flush must contain items from exactly one
	// (index, routing) group, and each item must carry its own routing.
	for i, req := range fake.requests {
		if len(req.Actions) == 0 {
			t.Fatalf("request %d had no actions", i)
		}
		groups := map[string]bool{}
		for _, a := range req.Actions {
			groups[a.Index+"\x00"+a.Routing] = true
		}
		if len(groups) > 1 {
			t.Errorf("request %d mixed groups: %+v", i, req.Actions)
		}
	}
	if stats := indexer.Stats(); stats.NumAbandoned != 0 || stats.NumFailed != 0 {
		t.Errorf("stats = %+v, want no failures", stats)
	}
}

func TestBulkReplay_OrderingWithinDocument(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    4,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 3,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	for i := 0; i < 20; i++ {
		if err := indexer.Add(context.Background(), BulkIndexerItem{
			Action:     "index",
			DocumentID: "same",
			Body:       strings.NewReader(fmt.Sprintf(`{"n":%d}`, i)),
		}); err != nil {
			t.Fatalf("add: %s", err)
		}
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 1 {
		t.Fatalf("landed %d documents, want 1", len(fake.docs))
	}
	for _, d := range fake.docs {
		if d.Body != `{"n":19}` {
			t.Errorf("final body = %s, want the last submission {\"n\":19}", d.Body)
		}
	}
}

func TestBulkReplay_CloseDrainsInFlight(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["1"] = 1
	fake.rateLimited["2"] = 1

	client := fake.client(t)
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 5,
			MaxElapsed:  time.Second,
			Backoff:     func(int) time.Duration { return 50 * time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	for i := 1; i <= 2; i++ {
		id := fmt.Sprint(i)
		if err := indexer.Add(context.Background(), BulkIndexerItem{
			Action:     "index",
			DocumentID: id,
			Body:       strings.NewReader(fmt.Sprintf(`{"id":%q}`, id)),
		}); err != nil {
			t.Fatalf("add: %s", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := indexer.Close(ctx); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumReplayed != 2 || stats.NumAbandoned != 0 || stats.NumFailed != 0 {
		t.Errorf("stats = %+v, want both items replayed and none abandoned", stats)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 2 {
		t.Errorf("landed %d documents, want 2", len(fake.docs))
	}
}

func TestBulkReplay_CloseDeadlineAbandonsWithReasons(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["stuck"] = 100

	client := fake.client(t)
	var (
		mu      sync.Mutex
		reasons []error
	)
	indexer, newErr := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 100,
			MaxElapsed:  time.Minute,
			Backoff:     func(int) time.Duration { return 100 * time.Millisecond },
		},
		OnError: func(context.Context, error) {},
	})
	if newErr != nil {
		t.Fatalf("new indexer: %s", newErr)
	}

	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "index",
		DocumentID: "stuck",
		Body:       strings.NewReader(`{"content":"x"}`),
		OnFailure: func(_ context.Context, _ BulkIndexerItem, _ BulkIndexerResponseItem, err error) {
			mu.Lock()
			reasons = append(reasons, err)
			mu.Unlock()
		},
	}); err != nil {
		t.Fatalf("add: %s", err)
	}

	// Let the first failure and the retry schedule happen.
	time.Sleep(150 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Millisecond)
	defer cancel()
	closeErr := indexer.Close(ctx)
	if !errors.Is(closeErr, context.DeadlineExceeded) {
		t.Errorf("Close error = %v, want context.DeadlineExceeded", closeErr)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(reasons) != 1 {
		t.Fatalf("OnFailure calls = %d, want exactly 1", len(reasons))
	}
	if !errors.Is(reasons[0], ErrReplayExhausted) {
		t.Errorf("abandon reason = %v, want ErrReplayExhausted", reasons[0])
	}
	stats := indexer.Stats()
	if stats.NumAbandoned != 1 {
		t.Errorf("NumAbandoned = %d, want 1; stats=%+v", stats.NumAbandoned, stats)
	}
}

func TestBulkReplayDisabledKeepsLegacyStats(t *testing.T) {
	fake := newFakeES(t)
	fake.rateLimited["1"] = 100
	client := fake.client(t)

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}
	if err := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "index",
		DocumentID: "1",
		Body:       strings.NewReader(`{"x":1}`),
	}); err != nil {
		t.Fatalf("add: %s", err)
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	stats := indexer.Stats()
	if stats.NumFailed != 1 || stats.NumFlushed != 0 {
		t.Errorf("legacy stats = %+v, want one failure and no replay counters used", stats)
	}
	if stats.NumReplayed != 0 || stats.NumAbandoned != 0 {
		t.Errorf("replay counters must stay zero without a policy: %+v", stats)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.requests) != 1 {
		t.Errorf("bulk requests = %d, want 1 (no replay without a policy)", len(fake.requests))
	}
}

func TestBulkReplay_VersionSemanticsAcrossSubmissions(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay: &BulkReplayPolicy{
			MaxAttempts: 3,
			Backoff:     func(int) time.Duration { return time.Millisecond },
		},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	var conflicts atomic.Int32
	// Submit version 5 first, then the older version 3. Only version 5 may
	// land; the older submission is rejected by version semantics, not by a
	// replay.
	for _, v := range []int64{5, 3} {
		version := v
		if err := indexer.Add(context.Background(), BulkIndexerItem{
			Action:      "index",
			DocumentID:  "doc",
			Version:     &version,
			VersionType: "external",
			Body:        strings.NewReader(fmt.Sprintf(`{"v":%d}`, v)),
			OnFailure: func(_ context.Context, _ BulkIndexerItem, info BulkIndexerResponseItem, _ error) {
				if info.Status == http.StatusConflict {
					conflicts.Add(1)
				}
			},
		}); err != nil {
			t.Fatalf("add v%d: %s", v, err)
		}
	}
	if err := indexer.Close(context.Background()); err != nil {
		t.Fatalf("close: %s", err)
	}

	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.docs) != 1 {
		t.Fatalf("landed %d documents, want 1", len(fake.docs))
	}
	for _, d := range fake.docs {
		if d.Version != 5 {
			t.Errorf("landed version %d, want 5 (the newer submission)", d.Version)
		}
	}
	if conflicts.Load() != 1 {
		t.Errorf("version conflicts reported = %d, want 1", conflicts.Load())
	}
	stats := indexer.Stats()
	if stats.NumFailed != 1 || stats.NumFlushed != 1 {
		t.Errorf("stats = %+v, want one success and one terminal version conflict", stats)
	}
}

func TestBulkReplay_UserCallbackReentrancyDoesNotDeadlock(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	// The callback calls into the indexer. Callbacks must run after the
	// ordering gates are released; otherwise a replayed item whose callback
	// takes the same gate would deadlock the replay loop.
	var indexer BulkIndexer
	flushed := make(chan error, 1)
	indexer, err := NewBulkIndexer(BulkIndexerConfig{
		NumWorkers:    1,
		FlushBytes:    1,
		FlushInterval: time.Hour,
		Index:         "orders",
		Client:        client,
		Replay:        &BulkReplayPolicy{MaxAttempts: 2, Backoff: func(int) time.Duration { return time.Millisecond }},
	})
	if err != nil {
		t.Fatalf("new indexer: %s", err)
	}

	if addErr := indexer.Add(context.Background(), BulkIndexerItem{
		Action:     "index",
		DocumentID: "doc",
		Body:       strings.NewReader(`{"x":1}`),
		OnSuccess: func(context.Context, BulkIndexerItem, BulkIndexerResponseItem) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			flushed <- indexer.Flush(ctx)
		},
	}); addErr != nil {
		t.Fatalf("add: %s", addErr)
	}

	done := make(chan struct{})
	go func() {
		_ = indexer.Close(context.Background())
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Close deadlocked; a user callback re-entered the indexer while gates were held")
	}
	select {
	case err := <-flushed:
		if err != nil && !errors.Is(err, ErrIndexerClosed) {
			t.Errorf("re-entrant Flush error = %v", err)
		}
	default:
		t.Error("OnSuccess callback never ran")
	}
}

func TestBulkReplay_StatsPartitionStableAcrossRuns(t *testing.T) {
	run := func() BulkIndexerStats {
		fake := newFakeES(t)
		fake.rateLimited["1"] = 1
		fake.rateLimited["2"] = 2
		fake.rateLimited["3"] = 100
		fake.conflicts["4"] = true

		indexer, err := NewBulkIndexer(BulkIndexerConfig{
			NumWorkers:    1,
			FlushBytes:    1,
			FlushInterval: time.Hour,
			Index:         "orders",
			Client:        fake.client(t),
			Replay: &BulkReplayPolicy{
				MaxAttempts: 3,
				MaxElapsed:  time.Second,
				Backoff:     func(int) time.Duration { return time.Millisecond },
			},
			OnError: func(context.Context, error) {},
		})
		if err != nil {
			t.Fatalf("new indexer: %s", err)
		}
		for i := 1; i <= 4; i++ {
			id := fmt.Sprint(i)
			if err := indexer.Add(context.Background(), BulkIndexerItem{
				Action:     "index",
				DocumentID: id,
				Body:       strings.NewReader(fmt.Sprintf(`{"id":%q}`, id)),
			}); err != nil {
				t.Fatalf("add: %s", err)
			}
		}
		if err := indexer.Close(context.Background()); err != nil {
			t.Fatalf("close: %s", err)
		}
		return indexer.Stats()
	}

	first := run()
	second := run()

	pick := func(s BulkIndexerStats) [4]uint64 {
		return [4]uint64{s.NumFlushed, s.NumReplayed, s.NumFailed, s.NumAbandoned}
	}
	if pick(first) != pick(second) {
		t.Errorf("stats differ between runs: %v vs %v", pick(first), pick(second))
	}
	sum := first.NumFlushed + first.NumReplayed + first.NumFailed + first.NumAbandoned
	if sum != first.NumAdded {
		t.Errorf("partition sums to %d, want NumAdded %d", sum, first.NumAdded)
	}
	if first.NumAdded != 4 {
		t.Errorf("NumAdded = %d, want 4", first.NumAdded)
	}
}
