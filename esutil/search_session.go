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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/elastic/go-elasticsearch/v9/esapi"
)

// Errors reported by [SearchSession].
var (
	// ErrSnapshotUnavailable is returned when the server refuses to open a
	// point-in-time snapshot (for example an unsupported deployment or an
	// unknown index). The session does not fall back to searching without a
	// snapshot.
	ErrSnapshotUnavailable = errors.New("esutil: snapshot unavailable")

	// ErrSnapshotExpired is returned when a search is rejected because the
	// point-in-time is gone. Callers can choose to call Reopen to rebuild the
	// snapshot; the session never silently retries without one.
	ErrSnapshotExpired = errors.New("esutil: snapshot expired")

	// ErrSnapshotClosed is returned when Search or Reopen is called after
	// Close.
	ErrSnapshotClosed = errors.New("esutil: snapshot session is closed")

	// ErrIncompleteResults is returned by Search when the server reports
	// partial shard coverage and the session requires complete results. The
	// partial hits are still returned alongside the error.
	ErrIncompleteResults = errors.New("esutil: search results are incomplete")
)

// SearchSessionConfig configures a [SearchSession].
type SearchSessionConfig struct {
	// Client is the Elasticsearch transport (required).
	Client esapi.Transport
	// Index is the set of indices or aliases the snapshot covers (required).
	Index []string
	// KeepAlive is how long the snapshot is kept alive on the server between
	// searches. Defaults to 1m; it is refreshed on every search.
	KeepAlive time.Duration
	// RequireCompleteResults makes Search return ErrIncompleteResults when
	// one or more shards failed or the response timed out. When false (the
	// default), partial hits are returned with a Coverage that reports the
	// gap, so a caller can decide.
	RequireCompleteResults bool
	// AllowUnavailableIndices opens the snapshot over the indices that do
	// exist instead of failing when one of them is missing or closed. Rounds
	// then report the gap through Coverage.
	AllowUnavailableIndices bool
	// AllowPartialSearchResults lets a round return results from the shards
	// that responded when others failed, instead of failing the round. When
	// false (the default) the server may still answer partially if it was
	// configured that way; check Coverage either way.
	AllowPartialSearchResults bool
}

func (c SearchSessionConfig) withDefaults() SearchSessionConfig {
	if c.KeepAlive <= 0 {
		c.KeepAlive = time.Minute
	}
	return c
}

// SnapshotHit is one hit returned from a snapshot search.
type SnapshotHit struct {
	Index   string
	ID      string
	Routing string
	Score   *float64
	Source  json.RawMessage
	Sort    []any
}

// ShardFailureInfo describes a shard that could not serve a search round.
type ShardFailureInfo struct {
	Index  string
	Shard  int
	Type   string
	Reason string
}

// SearchCoverage reports how complete a snapshot search round was. A caller
// that must not act on partial data should check Complete.
type SearchCoverage struct {
	ShardsTotal      int
	ShardsSuccessful int
	ShardsFailed     int
	TimedOut         bool
	Failures         []ShardFailureInfo
}

// Complete reports whether every shard responded and the request completed
// before timing out.
func (c SearchCoverage) Complete() bool {
	return !c.TimedOut && c.ShardsFailed == 0 && (c.ShardsTotal == 0 || c.ShardsSuccessful >= c.ShardsTotal)
}

// SnapshotSearchOptions describes one recall round within a snapshot.
type SnapshotSearchOptions struct {
	// Query is the raw Elasticsearch query. A match_all query is used when
	// nil.
	Query json.RawMessage
	// Size is the number of hits to return. Defaults to 10.
	Size int
	// Sort overrides the default "_shard_doc" sort, which is what makes
	// search_after pagination safe on a point-in-time.
	Sort []string
	// Source lists source fields to return. Nil returns the whole source.
	Source []string
	// SearchAfter continues from a previous round. Use the Next field of the
	// previous result.
	SearchAfter []any
}

// SnapshotSearchResult is the outcome of one snapshot search round.
type SnapshotSearchResult struct {
	Hits     []SnapshotHit
	Total    int
	Took     int
	Coverage SearchCoverage
	// PitID is the (possibly refreshed) point-in-time identifier. The session
	// tracks it automatically.
	PitID string
	// Next is the sort value to pass as SearchAfter for the next round. It is
	// nil when the round returned no hits.
	Next []any
}

// SearchSession performs multiple searches over one consistent point-in-time
// snapshot. Documents written after the snapshot was opened are not visible to
// its rounds.
//
// The session is safe for concurrent use; rounds are serialised so the
// refreshed point-in-time identifier is never lost.
type SearchSession struct {
	cfg SearchSessionConfig

	mu       sync.Mutex
	pitID    string
	closed   bool
	openedAt time.Time
	rounds   int
}

// OpenSearchSession opens a point-in-time snapshot over cfg.Index.
//
// A server that refuses to open a snapshot yields ErrSnapshotUnavailable. The
// session never falls back to a plain index search.
func OpenSearchSession(ctx context.Context, cfg SearchSessionConfig) (*SearchSession, error) {
	if cfg.Client == nil {
		return nil, errors.New("esutil: SearchSessionConfig.Client is required")
	}
	if len(cfg.Index) == 0 {
		return nil, errors.New("esutil: SearchSessionConfig.Index is required")
	}
	cfg = cfg.withDefaults()

	s := &SearchSession{cfg: cfg}
	if err := s.open(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *SearchSession) open(ctx context.Context) error {
	req := esapi.OpenPointInTimeRequest{Index: s.cfg.Index, KeepAlive: s.cfg.KeepAlive}
	if s.cfg.AllowUnavailableIndices {
		req.IgnoreUnavailable = esapi.BoolPtr(true)
	}
	res, err := req.Do(ctx, s.cfg.Client)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrSnapshotUnavailable, err)
	}
	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}
	if res.IsError() {
		return &SnapshotError{
			Err:    ErrSnapshotUnavailable,
			Status: res.StatusCode,
			Reason: readReason(res.Body),
		}
	}

	var payload struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return fmt.Errorf("%w: cannot decode open point in time response: %v", ErrSnapshotUnavailable, err)
	}
	if payload.ID == "" {
		return fmt.Errorf("%w: open point in time response did not contain an id", ErrSnapshotUnavailable)
	}

	s.mu.Lock()
	s.pitID = payload.ID
	s.openedAt = time.Now()
	s.rounds = 0
	s.mu.Unlock()
	return nil
}

// SnapshotError describes a rejected snapshot operation.
type SnapshotError struct {
	Err    error
	Status int
	Reason string
}

func (e *SnapshotError) Error() string {
	if e.Reason != "" {
		return fmt.Sprintf("%s (status %d): %s", e.Err.Error(), e.Status, e.Reason)
	}
	return fmt.Sprintf("%s (status %d)", e.Err.Error(), e.Status)
}

// Unwrap exposes the sentinel error.
func (e *SnapshotError) Unwrap() error { return e.Err }

// ID returns the current point-in-time identifier.
func (s *SearchSession) ID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pitID
}

// OpenedAt reports when the current snapshot was opened. It is useful to
// compute how long the snapshot has left before its keep-alive expires.
func (s *SearchSession) OpenedAt() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.openedAt
}

// Rounds returns how many search rounds the current snapshot has served.
func (s *SearchSession) Rounds() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.rounds
}

// Reopen discards the current snapshot and opens a fresh one. It may be called
// after ErrSnapshotExpired to resume retrieval on a new consistent snapshot.
// The old point-in-time is closed on a best-effort basis; failures to close it
// do not prevent reopening.
func (s *SearchSession) Reopen(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return ErrSnapshotClosed
	}
	oldID := s.pitID
	s.mu.Unlock()

	if oldID != "" {
		_ = s.closePIT(ctx, oldID)
	}

	if err := s.open(ctx); err != nil {
		return err
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		_ = s.closePIT(context.Background(), s.ID())
		return ErrSnapshotClosed
	}
	s.mu.Unlock()
	return nil
}

// Search runs one round against the snapshot. It returns ErrSnapshotExpired
// when the point-in-time is no longer available, and ErrIncompleteResults when
// the session requires complete coverage but a shard failed.
func (s *SearchSession) Search(ctx context.Context, opts SnapshotSearchOptions) (*SnapshotSearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, ErrSnapshotClosed
	}
	if s.pitID == "" {
		return nil, fmt.Errorf("%w: session has no point in time", ErrSnapshotUnavailable)
	}

	body, err := s.buildSearchBody(opts)
	if err != nil {
		return nil, err
	}

	searchReq := esapi.SearchRequest{Body: body}
	if s.cfg.AllowPartialSearchResults {
		searchReq.AllowPartialSearchResults = esapi.BoolPtr(true)
	}
	res, err := searchReq.Do(ctx, s.cfg.Client)
	if err != nil {
		return nil, fmt.Errorf("esutil: snapshot search failed: %w", err)
	}
	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}
	if res.IsError() {
		reason := readReason(res.Body)
		if isExpiredSnapshotResponse(res.StatusCode, reason) {
			return nil, &SnapshotError{Err: ErrSnapshotExpired, Status: res.StatusCode, Reason: reason}
		}
		return nil, &SnapshotError{Err: ErrSnapshotUnavailable, Status: res.StatusCode, Reason: reason}
	}

	var payload snapshotSearchPayload
	if err := json.NewDecoder(res.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("esutil: cannot decode snapshot search response: %w", err)
	}

	if payload.PitID != "" {
		s.pitID = payload.PitID
	}
	s.rounds++

	result := &SnapshotSearchResult{
		Took:     payload.Took,
		PitID:    s.pitID,
		Coverage: payload.coverage(),
		Total:    payload.totalHits(),
	}
	for _, h := range payload.Hits.Hits {
		hit := SnapshotHit{
			Index:  h.Index,
			ID:     h.ID,
			Score:  h.Score,
			Source: h.Source,
			Sort:   h.Sort,
		}
		if h.Routing != nil {
			hit.Routing = *h.Routing
		}
		result.Hits = append(result.Hits, hit)
	}
	if n := len(payload.Hits.Hits); n > 0 {
		result.Next = payload.Hits.Hits[n-1].Sort
	}

	if !result.Coverage.Complete() && s.cfg.RequireCompleteResults {
		return result, fmt.Errorf("%w: %d of %d shards failed", ErrIncompleteResults,
			result.Coverage.ShardsFailed, result.Coverage.ShardsTotal)
	}
	return result, nil
}

func (s *SearchSession) buildSearchBody(opts SnapshotSearchOptions) (io.Reader, error) {
	if opts.Size <= 0 {
		opts.Size = 10
	}
	query := opts.Query
	if len(query) == 0 {
		query = json.RawMessage(`{"match_all":{}}`)
	}

	pit := map[string]any{"id": s.pitID}
	if s.cfg.KeepAlive > 0 {
		pit["keep_alive"] = fmt.Sprintf("%dms", s.cfg.KeepAlive.Milliseconds())
	}

	sortValues := opts.Sort
	if len(sortValues) == 0 {
		sortValues = []string{"_shard_doc"}
	}

	body := map[string]any{
		"size":  opts.Size,
		"query": query,
		"sort":  sortValues,
		"pit":   pit,
	}
	if len(opts.Source) > 0 {
		body["_source"] = opts.Source
	}
	if len(opts.SearchAfter) > 0 {
		body["search_after"] = opts.SearchAfter
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("esutil: cannot encode snapshot search body: %w", err)
	}
	return bytesReader(data), nil
}

// Close closes the point-in-time and marks the session unusable. Closing an
// expired or already-gone point-in-time is not an error; it may be called more
// than once.
func (s *SearchSession) Close(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	pitID := s.pitID
	s.pitID = ""
	s.mu.Unlock()

	if pitID == "" {
		return nil
	}
	if err := s.closePIT(ctx, pitID); err != nil && !isMissingPITError(err) {
		return err
	}
	return nil
}

func (s *SearchSession) closePIT(ctx context.Context, id string) error {
	body, err := json.Marshal(map[string]string{"id": id})
	if err != nil {
		return err
	}
	res, err := (esapi.ClosePointInTimeRequest{Body: bytesReader(body)}).Do(ctx, s.cfg.Client)
	if err != nil {
		return fmt.Errorf("esutil: close point in time: %w", err)
	}
	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}
	if res.IsError() {
		return &SnapshotError{Err: ErrSnapshotUnavailable, Status: res.StatusCode, Reason: readReason(res.Body)}
	}
	return nil
}

func isMissingPITError(err error) bool {
	var se *SnapshotError
	return errors.As(err, &se) && se.Status == http.StatusNotFound
}

func isExpiredSnapshotResponse(status int, reason string) bool {
	if status != http.StatusNotFound {
		return false
	}
	for _, marker := range []string{
		"search_context_missing_exception",
		"No search context found",
		"no such point in time",
		"point-in-time",
	} {
		if containsFold(reason, marker) {
			return true
		}
	}
	return false
}

func containsFold(haystack, needle string) bool {
	return len(needle) == 0 || indexFold(haystack, needle) >= 0
}

func indexFold(haystack, needle string) int {
	// Case-insensitive substring search without pulling in extra packages.
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if equalFoldAt(haystack, i, needle) {
			return i
		}
	}
	return -1
}

func equalFoldAt(s string, at int, want string) bool {
	for i := 0; i < len(want); i++ {
		a, b := s[at+i], want[i]
		if 'A' <= a && a <= 'Z' {
			a += 'a' - 'A'
		}
		if 'A' <= b && b <= 'Z' {
			b += 'a' - 'A'
		}
		if a != b {
			return false
		}
	}
	return true
}

func readReason(body io.Reader) string {
	if body == nil {
		return ""
	}
	data, err := io.ReadAll(io.LimitReader(body, 8<<10))
	if err != nil {
		return ""
	}
	var payload struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &payload) == nil && len(payload.Error) > 0 {
		var nested struct {
			Type   string `json:"type"`
			Reason string `json:"reason"`
		}
		if json.Unmarshal(payload.Error, &nested) == nil && (nested.Type != "" || nested.Reason != "") {
			if nested.Type != "" && nested.Reason != "" {
				return nested.Type + ": " + nested.Reason
			}
			return nested.Type + nested.Reason
		}
		var text string
		if json.Unmarshal(payload.Error, &text) == nil {
			return text
		}
	}
	return truncate(string(data), 512)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

type snapshotSearchPayload struct {
	Took     int    `json:"took"`
	TimedOut bool   `json:"timed_out"`
	PitID    string `json:"pit_id"`
	Shards   struct {
		Total      int `json:"total"`
		Successful int `json:"successful"`
		Skipped    int `json:"skipped"`
		Failed     int `json:"failed"`
		Failures   []struct {
			Index  string `json:"index"`
			Shard  int    `json:"shard"`
			Reason struct {
				Type   string `json:"type"`
				Reason string `json:"reason"`
			} `json:"reason"`
		} `json:"failures"`
	} `json:"_shards"`
	Hits struct {
		Total    json.RawMessage `json:"total"`
		MaxScore *float64        `json:"max_score"`
		Hits     []struct {
			Index   string          `json:"_index"`
			ID      string          `json:"_id"`
			Routing *string         `json:"_routing"`
			Score   *float64        `json:"_score"`
			Source  json.RawMessage `json:"_source"`
			Sort    []any           `json:"sort"`
		} `json:"hits"`
	} `json:"hits"`
}

func (p *snapshotSearchPayload) totalHits() int {
	if len(p.Hits.Total) == 0 {
		return len(p.Hits.Hits)
	}
	var object struct {
		Value int `json:"value"`
	}
	if err := json.Unmarshal(p.Hits.Total, &object); err == nil {
		return object.Value
	}
	var number int
	if err := json.Unmarshal(p.Hits.Total, &number); err == nil {
		return number
	}
	return len(p.Hits.Hits)
}

func (p *snapshotSearchPayload) coverage() SearchCoverage {
	c := SearchCoverage{
		ShardsTotal:      p.Shards.Total,
		ShardsSuccessful: p.Shards.Successful,
		ShardsFailed:     p.Shards.Failed,
		TimedOut:         p.TimedOut,
	}
	for _, f := range p.Shards.Failures {
		c.Failures = append(c.Failures, ShardFailureInfo{
			Index:  f.Index,
			Shard:  f.Shard,
			Type:   f.Reason.Type,
			Reason: f.Reason.Reason,
		})
	}
	return c
}
