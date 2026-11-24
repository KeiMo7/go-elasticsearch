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
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elastic/go-elasticsearch/v9"
)

func TestVisibilityWaiter_SucceedsAfterRefresh(t *testing.T) {
	// The document only becomes visible after the second refresh, mirroring
	// the refresh interval lag that caused the original production incident.
	var (
		mu        sync.Mutex
		polls     int
		indexed   = map[string]bool{}
		refreshes int
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Elastic-Product", "Elasticsearch")
		mu.Lock()
		defer mu.Unlock()
		switch {
		case strings.HasSuffix(r.URL.Path, "/_refresh"):
			refreshes++
			if refreshes >= 2 {
				indexed["doc-1"] = true
				indexed["doc-2"] = true
			}
			fmt.Fprint(w, `{"_shards":{"total":1,"successful":1,"failed":0}}`)
		case strings.HasSuffix(r.URL.Path, "/_search"):
			polls++
			var hits []string
			for id := range indexed {
				hits = append(hits, fmt.Sprintf(`{"_index":"orders","_id":%q}`, id))
			}
			fmt.Fprintf(w, `{"took":1,"timed_out":false,"_shards":{"total":1,"successful":1,"failed":0},"hits":{"total":{"value":%d},"hits":[%s]}}`,
				len(hits), strings.Join(hits, ","))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{srv.URL}, DisableRetry: true})
	if err != nil {
		t.Fatalf("client: %s", err)
	}
	waiter, err := NewVisibilityWaiter(VisibilityWaiterConfig{
		Client:       client,
		Index:        "orders",
		PollInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("waiter: %s", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result, err := waiter.Wait(ctx, []DocumentRef{{ID: "doc-1"}, {ID: "doc-2"}})
	if err != nil {
		t.Fatalf("Wait: %s", err)
	}
	if len(result.Visible) != 2 || len(result.Missing) != 0 {
		t.Errorf("visible/missing = %d/%d, want 2/0", len(result.Visible), len(result.Missing))
	}
	if result.Polls < 2 {
		t.Errorf("polls = %d, want at least 2", result.Polls)
	}
	if !result.Coverage.Complete() {
		t.Errorf("coverage = %+v, want complete", result.Coverage)
	}

	mu.Lock()
	defer mu.Unlock()
	if refreshes < 2 {
		t.Errorf("refreshes = %d, want at least 2", refreshes)
	}
}

func TestVisibilityWaiter_DeadlineReportsMissing(t *testing.T) {
	var searches atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Elastic-Product", "Elasticsearch")
		switch {
		case strings.HasSuffix(r.URL.Path, "/_refresh"):
			fmt.Fprint(w, `{"_shards":{"total":1,"successful":1,"failed":0}}`)
		case strings.HasSuffix(r.URL.Path, "/_search"):
			searches.Add(1)
			// doc-1 is visible, doc-2 never is.
			fmt.Fprint(w, `{"took":1,"timed_out":false,"_shards":{"total":1,"successful":1,"failed":0},"hits":{"total":{"value":1},"hits":[{"_index":"orders","_id":"doc-1"}]}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{srv.URL}, DisableRetry: true})
	if err != nil {
		t.Fatalf("client: %s", err)
	}
	waiter, err := NewVisibilityWaiter(VisibilityWaiterConfig{
		Client:       client,
		Index:        "orders",
		PollInterval: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("waiter: %s", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	result, err := waiter.Wait(ctx, []DocumentRef{{ID: "doc-1"}, {ID: "doc-2"}})
	elapsed := time.Since(start)

	if !errors.Is(err, ErrVisibilityTimeout) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want ErrVisibilityTimeout wrapping DeadlineExceeded", err)
	}
	if len(result.Missing) != 1 || result.Missing[0].ID != "doc-2" {
		t.Errorf("missing = %+v, want only doc-2", result.Missing)
	}
	if len(result.Visible) != 1 || result.Visible[0].ID != "doc-1" {
		t.Errorf("visible = %+v, want only doc-1", result.Visible)
	}
	if elapsed > time.Second {
		t.Errorf("Wait took %s; it must honour the deadline", elapsed)
	}
	if searches.Load() < 2 {
		t.Errorf("searches = %d, want the waiter to poll repeatedly", searches.Load())
	}
}

func TestVisibilityWaiter_Cancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Elastic-Product", "Elasticsearch")
		if strings.HasSuffix(r.URL.Path, "/_refresh") {
			fmt.Fprint(w, `{"_shards":{"total":1,"successful":1,"failed":0}}`)
			return
		}
		fmt.Fprint(w, `{"took":1,"_shards":{"total":1,"successful":1,"failed":0},"hits":{"total":{"value":0},"hits":[]}}`)
	}))
	defer srv.Close()

	client, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{srv.URL}, DisableRetry: true})
	if err != nil {
		t.Fatalf("client: %s", err)
	}
	waiter, err := NewVisibilityWaiter(VisibilityWaiterConfig{Client: client, Index: "orders", PollInterval: 5 * time.Millisecond})
	if err != nil {
		t.Fatalf("waiter: %s", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	_, err = waiter.Wait(ctx, []DocumentRef{{ID: "never"}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if errors.Is(err, ErrVisibilityTimeout) {
		t.Errorf("cancellation must not be reported as a visibility timeout")
	}
}

func TestVisibilityWaiter_PartialReplicasReported(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Elastic-Product", "Elasticsearch")
		switch {
		case strings.HasSuffix(r.URL.Path, "/_refresh"):
			fmt.Fprint(w, `{"_shards":{"total":1,"successful":1,"failed":0}}`)
		case strings.HasSuffix(r.URL.Path, "/_search"):
			// doc-1 was found on the shard that did respond; one replica did
			// not. The waiter reports incomplete coverage but still returns
			// what it found.
			fmt.Fprint(w, `{"took":1,"timed_out":false,"_shards":{"total":2,"successful":1,"failed":1,"failures":[{"index":"orders","shard":1,"reason":{"type":"node_disconnected_exception","reason":"node left the cluster"}}]},"hits":{"total":{"value":1},"hits":[{"_index":"orders","_id":"doc-1"}]}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{srv.URL}, DisableRetry: true})
	if err != nil {
		t.Fatalf("client: %s", err)
	}
	waiter, err := NewVisibilityWaiter(VisibilityWaiterConfig{
		Client:       client,
		Index:        "orders",
		PollInterval: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("waiter: %s", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result, err := waiter.Wait(ctx, []DocumentRef{{ID: "doc-1"}})
	if err != nil {
		t.Fatalf("Wait: %s", err)
	}
	if result.Coverage.Complete() {
		t.Fatalf("coverage reported complete despite a failed shard: %+v", result.Coverage)
	}
	if result.Coverage.ShardsFailed != 1 || result.Coverage.ShardsSuccessful != 1 {
		t.Errorf("coverage = %+v, want 1 successful / 1 failed", result.Coverage)
	}
	if len(result.Visible) != 1 {
		t.Errorf("visible = %+v, want the document found on the healthy shard", result.Visible)
	}
}

func TestVisibilityWaiter_RequireCompleteResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Elastic-Product", "Elasticsearch")
		switch {
		case strings.HasSuffix(r.URL.Path, "/_refresh"):
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error":{"type":"unavailable_shards_exception","reason":"no shard available"}}`)
		case strings.HasSuffix(r.URL.Path, "/_search"):
			fmt.Fprint(w, `{"took":1,"_shards":{"total":1,"successful":1,"failed":0},"hits":{"total":{"value":0},"hits":[]}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client, err := elasticsearch.NewClient(elasticsearch.Config{Addresses: []string{srv.URL}, DisableRetry: true})
	if err != nil {
		t.Fatalf("client: %s", err)
	}
	waiter, err := NewVisibilityWaiter(VisibilityWaiterConfig{
		Client:                 client,
		Index:                  "orders",
		PollInterval:           5 * time.Millisecond,
		RequireCompleteResults: true,
	})
	if err != nil {
		t.Fatalf("waiter: %s", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	result, err := waiter.Wait(ctx, []DocumentRef{{ID: "doc-1"}})
	if !errors.Is(err, ErrIncompleteResults) {
		t.Fatalf("error = %v, want ErrIncompleteResults", err)
	}
	if len(result.Coverage.IncompleteIndices) != 1 || result.Coverage.IncompleteIndices[0] != "orders" {
		t.Errorf("incomplete indices = %v, want [orders]", result.Coverage.IncompleteIndices)
	}
}

func TestSearchSession_ConsistentSnapshot(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	// Seed three documents, then open the snapshot.
	for i := 1; i <= 3; i++ {
		if err := putDoc(fake, "orders", "", fmt.Sprint(i)); err != nil {
			t.Fatalf("seed: %s", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}, KeepAlive: time.Minute})
	if err != nil {
		t.Fatalf("open session: %s", err)
	}
	defer func() { _ = session.Close(ctx) }()

	first, err := session.Search(ctx, SnapshotSearchOptions{Size: 10})
	if err != nil {
		t.Fatalf("search 1: %s", err)
	}
	if len(first.Hits) != 3 {
		t.Fatalf("round 1 hits = %d, want 3", len(first.Hits))
	}

	// A write after the snapshot opened must not be visible to later rounds.
	if seedErr := putDoc(fake, "orders", "", "4"); seedErr != nil {
		t.Fatalf("write during snapshot: %s", seedErr)
	}

	second, err := session.Search(ctx, SnapshotSearchOptions{Size: 10})
	if err != nil {
		t.Fatalf("search 2: %s", err)
	}
	if len(second.Hits) != 3 {
		t.Errorf("round 2 hits = %d, want 3 (the snapshot must pin the view)", len(second.Hits))
	}
	for _, hit := range second.Hits {
		if hit.ID == "4" {
			t.Errorf("document written after the snapshot appeared in it")
		}
	}
	if session.Rounds() != 2 {
		t.Errorf("rounds = %d, want 2", session.Rounds())
	}
}

func TestSearchSession_ExpiredSnapshotErrorsAndReopen(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}})
	if err != nil {
		t.Fatalf("open session: %s", err)
	}
	defer func() { _ = session.Close(ctx) }()

	// Expire the snapshot behind the session's back.
	fake.mu.Lock()
	for id := range fake.pitValid {
		delete(fake.pitValid, id)
	}
	fake.mu.Unlock()

	_, err = session.Search(ctx, SnapshotSearchOptions{})
	if !errors.Is(err, ErrSnapshotExpired) {
		t.Fatalf("search error = %v, want ErrSnapshotExpired", err)
	}

	// Rebuilding must produce a fresh, valid snapshot.
	if reopenErr := session.Reopen(ctx); reopenErr != nil {
		t.Fatalf("reopen: %s", reopenErr)
	}
	result, err := session.Search(ctx, SnapshotSearchOptions{})
	if err != nil {
		t.Fatalf("search after reopen: %s", err)
	}
	if result.PitID == "" {
		t.Errorf("reopened session has no point in time id")
	}
}

func TestSearchSession_ServerRefusesSnapshot(t *testing.T) {
	fake := newFakeES(t)
	fake.refusePIT = 1
	fake.refuseCodes = http.StatusBadRequest
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}})
	if err == nil {
		_ = session.Close(ctx)
		t.Fatal("expected OpenSearchSession to fail when the server refuses a snapshot")
	}
	if !errors.Is(err, ErrSnapshotUnavailable) {
		t.Fatalf("error = %v, want ErrSnapshotUnavailable", err)
	}
}

func TestSearchSession_NoSilentFallbackToPlainSearch(t *testing.T) {
	var plainSearches atomic.Int32
	fake := newFakeES(t)
	fake.refusePIT = 100

	// Wrap the server so any search without a PIT is observable; the session
	// must never issue one.
	client, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses:    []string{fake.srv.URL},
		DisableRetry: true,
		Transport: &hookTransport{base: http.DefaultTransport, hook: func(r *http.Request) error {
			if strings.HasSuffix(r.URL.Path, "/_search") {
				plainSearches.Add(1)
			}
			return nil
		}},
	})
	if err != nil {
		t.Fatalf("client: %s", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}}); err == nil {
		t.Fatal("expected an error when the server refuses snapshots")
	}
	if plainSearches.Load() != 0 {
		t.Errorf("session issued %d plain searches; it must not fall back", plainSearches.Load())
	}
}

func TestSearchSession_PartialShardsMarkedIncomplete(t *testing.T) {
	fake := newFakeES(t)
	fake.partialShards = true
	if err := putDoc(fake, "orders", "", "1"); err != nil {
		t.Fatalf("seed: %s", err)
	}
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := OpenSearchSession(ctx, SearchSessionConfig{
		Client:                 client,
		Index:                  []string{"orders"},
		RequireCompleteResults: true,
	})
	if err != nil {
		t.Fatalf("open session: %s", err)
	}
	defer func() { _ = session.Close(ctx) }()

	result, err := session.Search(ctx, SnapshotSearchOptions{Size: 10})
	if !errors.Is(err, ErrIncompleteResults) {
		t.Fatalf("error = %v, want ErrIncompleteResults", err)
	}
	if result == nil || len(result.Hits) != 1 {
		t.Fatalf("partial results should still be returned alongside the error")
	}
	if result.Coverage.Complete() {
		t.Errorf("coverage claims completeness")
	}
	if len(result.Coverage.Failures) != 1 || result.Coverage.Failures[0].Type != "node_disconnected_exception" {
		t.Errorf("failures = %+v, want the shard failure to be reported", result.Coverage.Failures)
	}
}

func TestSearchSession_SearchAfterWithinSnapshot(t *testing.T) {
	fake := newFakeES(t)
	for i := 1; i <= 5; i++ {
		if err := putDoc(fake, "orders", "", fmt.Sprint(i)); err != nil {
			t.Fatalf("seed: %s", err)
		}
	}
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	session, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}})
	if err != nil {
		t.Fatalf("open session: %s", err)
	}
	defer func() { _ = session.Close(ctx) }()

	page1, err := session.Search(ctx, SnapshotSearchOptions{Size: 2})
	if err != nil {
		t.Fatalf("page 1: %s", err)
	}
	if len(page1.Next) == 0 {
		t.Fatal("page 1 returned no sort cursor")
	}
	page2, err := session.Search(ctx, SnapshotSearchOptions{Size: 2, SearchAfter: page1.Next})
	if err != nil {
		t.Fatalf("page 2: %s", err)
	}
	if len(page2.Hits) == 0 {
		t.Error("page 2 returned no hits")
	}
}

func TestSearchSession_CloseIsIdempotent(t *testing.T) {
	fake := newFakeES(t)
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	session, err := OpenSearchSession(ctx, SearchSessionConfig{Client: client, Index: []string{"orders"}})
	if err != nil {
		t.Fatalf("open: %s", err)
	}
	if err := session.Close(ctx); err != nil {
		t.Fatalf("close: %s", err)
	}
	if err := session.Close(ctx); err != nil {
		t.Errorf("second close: %s", err)
	}
	if _, err := session.Search(ctx, SnapshotSearchOptions{}); !errors.Is(err, ErrSnapshotClosed) {
		t.Errorf("search after close = %v, want ErrSnapshotClosed", err)
	}
	if err := session.Reopen(ctx); !errors.Is(err, ErrSnapshotClosed) {
		t.Errorf("reopen after close = %v, want ErrSnapshotClosed", err)
	}
}

// putDoc writes one document through the fake's bulk endpoint.
func putDoc(fake *fakeES, index, routing, id string) error {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	doc := fakeDoc{Index: index, Routing: routing, ID: id, Body: fmt.Sprintf(`{"id":%q}`, id)}
	fake.docs[doc.key()] = doc
	fake.lands[doc.key()]++
	return nil
}

func TestSearchSession_UnavailableIndexReportedNotFatal(t *testing.T) {
	fake := newFakeES(t)
	fake.unavailableIndices["archive"] = true
	if err := putDoc(fake, "orders", "", "1"); err != nil {
		t.Fatalf("seed: %s", err)
	}
	client := fake.client(t)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Without the opt-in the server refuses the snapshot outright.
	if _, err := OpenSearchSession(ctx, SearchSessionConfig{
		Client: client,
		Index:  []string{"orders", "archive"},
	}); !errors.Is(err, ErrSnapshotUnavailable) {
		t.Fatalf("open without opt-in = %v, want ErrSnapshotUnavailable", err)
	}

	// With the opt-in the snapshot opens over what exists, and the round
	// reports the missing shard coverage instead of hiding it.
	session, err := OpenSearchSession(ctx, SearchSessionConfig{
		Client:                  client,
		Index:                   []string{"orders", "archive"},
		AllowUnavailableIndices: true,
	})
	if err != nil {
		t.Fatalf("open with opt-in: %s", err)
	}
	defer func() { _ = session.Close(ctx) }()

	// Make the snapshot itself report a partial shard so Coverage is exercised.
	fake.mu.Lock()
	fake.partialShards = true
	fake.mu.Unlock()

	result, err := session.Search(ctx, SnapshotSearchOptions{Size: 10})
	if err != nil {
		t.Fatalf("search: %s", err)
	}
	if result.Coverage.Complete() {
		t.Fatalf("coverage reported complete with an unavailable index")
	}
	if len(result.Coverage.Failures) == 0 {
		t.Errorf("expected shard failures to be reported: %+v", result.Coverage)
	}
	if session.ID() == "" {
		t.Errorf("session has no point in time id")
	}
}
