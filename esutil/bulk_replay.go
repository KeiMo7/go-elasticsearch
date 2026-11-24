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
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/elastic/elastic-transport-go/v8/elastictransport"
	"github.com/elastic/go-elasticsearch/v9"
	"github.com/elastic/go-elasticsearch/v9/esapi"
)

// Errors reported through BulkIndexerItem.OnFailure when a replay policy is
// configured. They identify why an item was not retried further; the original
// Elasticsearch error, when there is one, is still available in the
// BulkIndexerResponseItem passed alongside.
var (
	// ErrItemNotRetryable is reported for failures that are classified as
	// terminal (for example a mapping error or a version conflict).
	ErrItemNotRetryable = errors.New("esutil: bulk item failure is not retryable")

	// ErrItemNotReplayable is reported for retryable failures on items that
	// cannot be replayed safely, such as auto-ID documents or non-idempotent
	// actions. Replaying them could create duplicates or change the result.
	ErrItemNotReplayable = errors.New("esutil: bulk item is not safely replayable")

	// ErrReplayExhausted is reported when the replay budget (attempts or
	// elapsed time) ran out.
	ErrReplayExhausted = errors.New("esutil: bulk replay budget exhausted")

	// ErrSupersededByNewerDocument is reported for an item that was abandoned
	// because a newer submission for the same document identifier exists.
	ErrSupersededByNewerDocument = errors.New("esutil: bulk item superseded by a newer submission")
)

// BulkReplayPolicy configures per-item replay of retryable bulk failures.
//
// Replay is opt-in: set [BulkIndexerConfig.Replay] to a non-nil policy. When
// replay is enabled:
//
//   - Items failing with a retryable condition (HTTP 429, 408 or 5xx, a
//     transport error, or a retryable Elasticsearch error type) are retried
//     with backoff, up to MaxAttempts attempts and MaxElapsed since the first
//     failure, whichever comes first.
//   - Only items that can be replayed without changing the final content are
//     retried: the item must carry a DocumentID and its action must be
//     idempotent ("index" or "delete"). Auto-ID documents are not replayed
//     because Elasticsearch would assign a new ID; "create" is not replayed
//     because a successful first attempt would turn the retry into a
//     version conflict; "update" is not replayed because scripts may not be
//     idempotent. Use RetryOnConflict for update conflicts.
//   - Version conflicts (409) are never replayed: version semantics decide
//     which submission wins, and a retry would either be a no-op conflict or
//     reorder submissions.
//   - Submissions of the same document identifier (index, routing, ID) are
//     ordered: an item known to be superseded by a newer submission is
//     abandoned instead of being sent, and requests carrying the same
//     document identifier are serialised so an older write cannot land after
//     a newer one.
//   - Stats() partitions every accepted item into NumFlushed (first-attempt
//     success), NumReplayed, NumFailed (terminal failure) or NumAbandoned
//     (budget exhausted or superseded).
//
// A zero value is valid and uses the documented defaults. The policy is
// copied when the indexer is created.
// notifier defers user callbacks until ordering gates have been released, so
// a callback that re-enters the indexer cannot deadlock on a gate.
type notifier struct {
	fns []func()
}

func (n *notifier) add(fn func()) { n.fns = append(n.fns, fn) }

func (n *notifier) run() {
	fns := n.fns
	n.fns = nil
	for _, fn := range fns {
		fn()
	}
}

// reportError queues the indexer-level OnError callback.
func (bi *bulkIndexer) reportError(ctx context.Context, notes *notifier, err error) {
	if bi.config.OnError == nil {
		return
	}
	notes.add(func() { bi.config.OnError(ctx, err) })
}

type BulkReplayPolicy struct {
	// MaxAttempts is the maximum number of attempts per item, including the
	// first. 1 disables retries but keeps ordering, classification and the
	// four-way statistics. Defaults to 3.
	MaxAttempts int

	// MaxElapsed bounds the total time an item may spend being replayed,
	// measured from its first retryable failure. Defaults to 30s.
	MaxElapsed time.Duration

	// Backoff returns the delay before retry number attempt (1-based, so the
	// first retry receives 1). A nil function uses capped exponential backoff
	// starting at 100ms.
	Backoff func(attempt int) time.Duration

	// MaxBatchSize is the maximum number of items grouped into one replay
	// request. Defaults to 1000. Items in a replay request always share the
	// same (index, routing) group.
	MaxBatchSize int
}

func (p BulkReplayPolicy) withDefaults() BulkReplayPolicy {
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = 3
	}
	if p.MaxElapsed <= 0 {
		p.MaxElapsed = 30 * time.Second
	}
	if p.Backoff == nil {
		p.Backoff = defaultReplayBackoff
	}
	if p.MaxBatchSize <= 0 {
		p.MaxBatchSize = 1000
	}
	return p
}

func defaultReplayBackoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	const (
		base    = 100 * time.Millisecond
		ceiling = 2 * time.Second
	)
	d := base
	for i := 1; i < attempt && d < ceiling; i++ {
		d *= 2
	}
	if d > ceiling {
		d = ceiling
	}
	return d
}

// BulkReplayError explains why an item was ultimately not written (or why an
// earlier attempt was abandoned). It is passed to OnFailure as the error
// argument when a replay policy is enabled. Use errors.Is against the exported
// sentinel errors to classify it.
type BulkReplayError struct {
	// Err is one of ErrItemNotRetryable, ErrItemNotReplayable,
	// ErrReplayExhausted or ErrSupersededByNewerDocument.
	Err error
	// Attempts is the number of completed attempts for the item.
	Attempts int
	// Reason is a human-readable explanation.
	Reason string
	// DocumentID, Index and Routing identify the item.
	DocumentID string
	Index      string
	Routing    string
}

func (e *BulkReplayError) Error() string {
	var b strings.Builder
	b.WriteString(e.Err.Error())
	if e.Reason != "" {
		b.WriteString(": ")
		b.WriteString(e.Reason)
	}
	if e.DocumentID != "" {
		fmt.Fprintf(&b, " (document %q", e.DocumentID)
		if e.Index != "" {
			fmt.Fprintf(&b, ", index %q", e.Index)
		}
		if e.Routing != "" {
			fmt.Fprintf(&b, ", routing %q", e.Routing)
		}
		b.WriteString(")")
	}
	if e.Attempts > 1 {
		fmt.Fprintf(&b, " after %d attempts", e.Attempts)
	}
	return b.String()
}

func (e *BulkReplayError) Unwrap() error { return e.Err }

// Bulk action names, kept as constants so the classification and replay rules
// stay in sync.
const (
	actionIndex  = "index"
	actionCreate = "create"
	actionDelete = "delete"
	actionUpdate = "update"
)

// docKey identifies a document submission for ordering and supersession.
type docKey struct {
	index   string
	routing string
	id      string
}

type groupKey struct {
	index   string
	routing string
}

func (bi *bulkIndexer) itemKey(item *BulkIndexerItem) (docKey, bool) {
	if item.DocumentID == "" {
		return docKey{}, false
	}
	index := item.Index
	if index == "" {
		index = bi.config.Index
	}
	return docKey{index: index, routing: item.Routing, id: item.DocumentID}, true
}

func (bi *bulkIndexer) itemGroup(item *BulkIndexerItem) groupKey {
	index := item.Index
	if index == "" {
		index = bi.config.Index
	}
	return groupKey{index: index, routing: item.Routing}
}

// keyRegistry tracks the newest submission sequence for each live document
// identifier. Entries are removed when no submission for the key is pending,
// so memory is bounded by the number of in-flight document identifiers.
type keyRegistry struct {
	mu    sync.Mutex
	state map[docKey]*keyState
	seq   uint64
}

type keyState struct {
	// attempted is the highest sequence handed to the transport. It is the
	// watermark that protects ordering: a lower sequence must not be sent
	// once a higher one has gone out.
	attempted uint64
	pending   int
}

func newKeyRegistry() *keyRegistry {
	return &keyRegistry{state: make(map[docKey]*keyState)}
}

// register assigns a sequence number to a submission and returns it.
func (r *keyRegistry) register(key docKey) uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	ks := r.state[key]
	if ks == nil {
		ks = &keyState{}
		r.state[key] = ks
	}
	ks.pending++
	return r.seq
}

// release removes one pending submission. It is safe to call more than once
// only through replayTask.notified, which guarantees a single final outcome.
func (r *keyRegistry) release(key docKey) {
	r.mu.Lock()
	defer r.mu.Unlock()
	ks := r.state[key]
	if ks == nil {
		return
	}
	ks.pending--
	if ks.pending <= 0 {
		delete(r.state, key)
	}
}

// wouldSend reports whether seq could still be sent without violating the
// attempted watermark. It does not advance the watermark.
func (r *keyRegistry) wouldSend(key docKey, seq uint64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	ks := r.state[key]
	if ks == nil {
		return true
	}
	return seq >= ks.attempted
}

// claim reports whether the submission with sequence seq may be sent, and if
// so advances the attempted watermark. A submission is refused only when a
// strictly newer submission for the same document identifier has already been
// sent, which is exactly the case where sending it could land out of order.
func (r *keyRegistry) claim(key docKey, seq uint64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	ks := r.state[key]
	if ks == nil {
		// The key was released (a final outcome was reached); a late send can
		// no longer race with a newer submission still in the registry.
		return true
	}
	if seq < ks.attempted {
		return false
	}
	ks.attempted = seq
	return true
}

// replayTask is one item's journey through the replay path.
type replayTask struct {
	item         BulkIndexerItem
	attempts     atomic.Int32
	firstFailure time.Time
	readyAt      time.Time

	notified bool // guarded by replayState.mu for callbacks/registry release
}

// replayState owns the replay queue, ordering gates and supersession registry.
type replayState struct {
	policy  BulkReplayPolicy
	keys    *keyRegistry
	indices *bulkIndexer

	mu       sync.Mutex
	queue    []*replayTask
	inflight int
	closed   bool
	stopped  bool
	wake     chan struct{}
	changed  chan struct{} // closed and replaced whenever queue/inflight changes

	gates [replayGateCount]sync.Mutex

	cbMu      sync.Mutex
	cbCond    *sync.Cond
	cbPending []*notifier
	cbClosed  bool

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	cbWG   sync.WaitGroup
}

func newReplayState(policy BulkReplayPolicy, bi *bulkIndexer) *replayState {
	ctx, cancel := context.WithCancel(context.Background())
	r := &replayState{
		policy:  policy.withDefaults(),
		keys:    newKeyRegistry(),
		indices: bi,
		wake:    make(chan struct{}, 1),
		changed: make(chan struct{}),
		ctx:     ctx,
		cancel:  cancel,
	}
	r.cbCond = sync.NewCond(&r.cbMu)
	return r
}

func (r *replayState) start() {
	r.wg.Add(1)
	go r.loop()

	r.cbWG.Add(1)
	go r.callbackLoop()
}

// runCallbacks queues per-item callbacks for the dispatcher goroutine. The
// queue is unbounded, so the replay loop never blocks on a slow callback.
func (r *replayState) runCallbacks(notes *notifier) {
	if notes == nil || len(notes.fns) == 0 {
		return
	}
	r.cbMu.Lock()
	if r.cbClosed {
		// The dispatcher has already drained its queue (only possible after a
		// bounded shutdown). Running inline keeps the callback from being
		// dropped, even though the indexer is stopping.
		r.cbMu.Unlock()
		notes.run()
		return
	}
	r.cbPending = append(r.cbPending, notes)
	r.cbMu.Unlock()
	r.cbCond.Signal()
}

func (r *replayState) callbackLoop() {
	defer r.cbWG.Done()
	for {
		r.cbMu.Lock()
		for len(r.cbPending) == 0 && !r.cbClosed {
			r.cbCond.Wait()
		}
		if len(r.cbPending) == 0 { // closed and drained
			r.cbMu.Unlock()
			return
		}
		notes := r.cbPending
		r.cbPending = nil
		r.cbMu.Unlock()

		for _, n := range notes {
			n.run()
		}
	}
}

// stopCallbacks asks the dispatcher to drain and exit, and waits for it up to
// the caller's deadline. A callback that blocks is not allowed to hold Close
// open forever: the dispatcher keeps running in the background and will still
// deliver every queued callback.
func (r *replayState) stopCallbacks(ctx context.Context) {
	r.cbMu.Lock()
	r.cbClosed = true
	r.cbMu.Unlock()
	r.cbCond.Broadcast()

	done := make(chan struct{})
	go func() {
		r.cbWG.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
	}
}

// signal wakes the replay loop, coalescing notifications.
func (r *replayState) signal() {
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

func (r *replayState) loop() {
	defer func() {
		r.mu.Lock()
		r.stopped = true
		r.notifyLocked()
		r.mu.Unlock()
		r.wg.Done()
	}()

	for {
		batch, ok := r.nextBatch()
		if !ok {
			return
		}
		r.indices.sendReplayBatch(r.ctx, batch)

		r.mu.Lock()
		r.inflight--
		r.notifyLocked()
		r.mu.Unlock()
	}
}

// notifyLocked broadcasts a state change to waitIdle/nextBatch waiters.
// The caller must hold r.mu.
func (r *replayState) notifyLocked() {
	close(r.changed)
	r.changed = make(chan struct{})
}

// waitIdle blocks until the replay queue is empty and no replay request is in
// flight. It returns the context error when ctx expires first.
func (r *replayState) waitIdle(ctx context.Context) error {
	for {
		r.mu.Lock()
		idle := len(r.queue) == 0 && r.inflight == 0
		changed := r.changed
		r.mu.Unlock()
		if idle {
			return nil
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// nextBatch waits until at least one queued task is ready, then returns a
// group of ready tasks belonging to the same (index, routing) group so replay
// requests never mix groups. It returns ok=false once the queue is drained and
// the replayer is closed.
func (r *replayState) nextBatch() ([]*replayTask, bool) {
	for {
		if r.ctx.Err() != nil {
			return nil, false
		}
		r.mu.Lock()
		if len(r.queue) == 0 {
			if r.closed {
				r.stopped = true
				r.notifyLocked()
				r.mu.Unlock()
				return nil, false
			}
			r.mu.Unlock()
			select {
			case <-r.wake:
			case <-r.ctx.Done():
			}
			continue
		}

		now := time.Now()
		earliest := 0
		for i, t := range r.queue {
			if t.readyAt.Before(r.queue[earliest].readyAt) {
				earliest = i
			}
		}
		next := r.queue[earliest].readyAt
		if now.Before(next) {
			r.mu.Unlock()
			timer := time.NewTimer(next.Sub(now))
			select {
			case <-timer.C:
			case <-r.wake:
			case <-r.ctx.Done():
			}
			timer.Stop()
			if r.ctx.Err() != nil {
				return nil, false
			}
			continue
		}

		group := r.indices.itemGroup(&r.queue[earliest].item)
		batch := make([]*replayTask, 0, min(r.policy.MaxBatchSize, len(r.queue)))
		remaining := r.queue[:0]
		for _, t := range r.queue {
			if len(batch) < r.policy.MaxBatchSize &&
				!t.readyAt.After(now) &&
				r.indices.itemGroup(&t.item) == group {
				batch = append(batch, t)
				continue
			}
			remaining = append(remaining, t)
		}
		r.queue = remaining
		r.inflight++
		r.notifyLocked()
		r.mu.Unlock()
		return batch, true
	}
}

// enqueue adds a task for replay. It returns false when the replayer has
// already stopped draining; the caller must then report a final outcome so no
// item is silently dropped.
func (r *replayState) enqueue(t *replayTask) bool {
	r.mu.Lock()
	if r.stopped {
		r.mu.Unlock()
		return false
	}
	r.queue = append(r.queue, t)
	r.notifyLocked()
	r.mu.Unlock()
	r.signal()
	return true
}

// shutdown asks the replayer to drain and exits. If ctx expires first the
// session context is cancelled so in-flight sends fail fast, and any remaining
// queued items are abandoned with an explicit reason rather than dropped.
func (r *replayState) shutdown(ctx context.Context) error {
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	r.signal()

	done := make(chan struct{})
	go func() {
		r.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		r.stopCallbacks(ctx)
		return nil
	case <-ctx.Done():
	}

	r.cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		// A send is not honouring cancellation (custom transport). Abandon
		// the queue below; notify-once guards make this safe.
	}

	r.abandonRemaining(ctx.Err())
	r.stopCallbacks(ctx)
	return ctx.Err()
}

func (r *replayState) abandonRemaining(cause error) {
	for {
		r.mu.Lock()
		if len(r.queue) == 0 {
			r.mu.Unlock()
			return
		}
		batch := r.queue
		r.queue = nil
		r.notifyLocked()
		r.mu.Unlock()

		notes := &notifier{}
		for _, t := range batch {
			r.indices.finishAbandoned(notes, t, &BulkReplayError{
				Err:        ErrReplayExhausted,
				Attempts:   int(t.attempts.Load()),
				Reason:     "indexer closed before the item could be replayed: " + cause.Error(),
				DocumentID: t.item.DocumentID,
				Index:      t.item.Index,
				Routing:    t.item.Routing,
			})
		}
		notes.run()
	}
}

// acquireGates locks the ordering gates for every document key in items and
// returns an unlock function. Gate stripes are acquired in ascending order,
// which prevents deadlocks when batches share documents. Submissions without a
// document ID cannot collide and are not gated.
func (r *replayState) acquireGates(items []*replayTask) func() {
	var stripes []int
	seen := make(map[int]bool, len(items))
	for i := range items {
		key, ok := r.indices.itemKey(&items[i].item)
		if !ok {
			continue
		}
		s := gateIndex(key)
		if !seen[s] {
			seen[s] = true
			stripes = append(stripes, s)
		}
	}
	sort.Ints(stripes)
	for _, s := range stripes {
		r.gates[s].Lock()
	}
	return func() {
		for i := len(stripes) - 1; i >= 0; i-- {
			r.gates[stripes[i]].Unlock()
		}
	}
}

func gateIndex(key docKey) int {
	h := fnv.New32a()
	_, _ = io.WriteString(h, key.index)
	_, _ = h.Write([]byte{0})
	_, _ = io.WriteString(h, key.routing)
	_, _ = h.Write([]byte{0})
	_, _ = io.WriteString(h, key.id)
	return int(h.Sum32() % uint32(replayGateCount))
}

const replayGateCount = 256

// sendReplayBatch re-sends a batch of tasks produced by the replay loop. User
// callbacks are handed to the callback dispatcher instead of running on the
// loop goroutine: a callback may call Flush and block, and the loop must stay
// free to drain retries so that Flush can complete.
func (bi *bulkIndexer) sendReplayBatch(ctx context.Context, tasks []*replayTask) {
	notes := &notifier{}
	bi.sendTasks(ctx, tasks, notes)
	bi.replay.runCallbacks(notes)
}

// flushBufferReplay hands the worker's buffered items to the replay path as
// fresh tasks, then resets the buffer and item slice. The worker's own
// per-item callbacks are invoked later, once each item reaches a final state.
func (w *worker) flushBufferReplay(ctx context.Context) error {
	tasks := make([]*replayTask, 0, len(w.items))
	for i := range w.items {
		item := w.items[i]
		tasks = append(tasks, &replayTask{item: item})
	}

	w.items = nil
	if w.buf.Cap() > w.bi.config.FlushBytes {
		w.buf = bytes.NewBuffer(make([]byte, 0, w.bi.config.FlushBytes))
	} else {
		w.buf.Reset()
	}

	// Callbacks run only after sendTasks released the ordering gates, so a
	// callback that re-enters the indexer never waits on its own gate. (As in
	// the legacy path, callbacks run on the worker goroutine; a callback must
	// not call Flush on the same indexer from inside a flush.)
	notes := &notifier{}
	w.bi.sendTasks(ctx, tasks, notes)
	notes.run()
	return nil
}

// sendTasks encodes and sends tasks, then classifies every response. It is
// used both for first attempts (from a worker) and replays.
func (bi *bulkIndexer) sendTasks(ctx context.Context, tasks []*replayTask, notes *notifier) {
	if len(tasks) == 0 {
		return
	}

	unlock := bi.replay.acquireGates(tasks)
	defer unlock()

	tasks = bi.claimTasks(notes, tasks)
	if len(tasks) == 0 {
		return
	}

	body, bodyLen, encodeErr := bi.encodeTasks(notes, tasks)
	if encodeErr != nil {
		return
	}

	res, elapsed := bi.performBulk(ctx, notes, tasks, body)
	if res == nil {
		return
	}
	bi.flushBytes(bodyLen)
	bi.flushMs(elapsed)
	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}

	var blk BulkIndexerResponse
	if decodeErr := bi.config.Decoder.UnmarshalFromReader(res.Body, &blk); decodeErr != nil {
		decodeErr = fmt.Errorf("flush: error parsing response body: %w", decodeErr)
		bi.handleRequestFailure(ctx, notes, tasks, BulkIndexerResponseItem{}, decodeErr, true)
		return
	}

	bi.processBulkResponse(ctx, notes, tasks, blk)
}

// claimTasks drops submissions that are older than a newer submission already
// sent, and coalesces duplicate submissions inside the batch to the newest one.
// Sending duplicates would leave the final content to the order of entries in
// the request body instead of the order they were added.
func (bi *bulkIndexer) claimTasks(notes *notifier, tasks []*replayTask) []*replayTask {
	newest := make(map[docKey]uint64, len(tasks))
	for _, t := range tasks {
		key, ok := bi.itemKey(&t.item)
		if !ok || !orderingProtected(&t.item) {
			continue
		}
		if seq, seen := newest[key]; !seen || t.item.seq > seq {
			newest[key] = t.item.seq
		}
	}

	kept := tasks[:0]
	var abandoned []*replayTask
	for _, t := range tasks {
		key, ok := bi.itemKey(&t.item)
		if ok && orderingProtected(&t.item) {
			if t.item.seq < newest[key] {
				abandoned = append(abandoned, t)
				continue
			}
			if !bi.replay.keys.claim(key, t.item.seq) {
				abandoned = append(abandoned, t)
				continue
			}
		}
		kept = append(kept, t)
	}

	for _, t := range abandoned {
		bi.finishAbandoned(notes, t, &BulkReplayError{
			Err:        ErrSupersededByNewerDocument,
			Attempts:   int(t.attempts.Load()),
			Reason:     "a newer submission for this document identifier exists; the older one was not sent",
			DocumentID: t.item.DocumentID,
			Index:      t.item.Index,
			Routing:    t.item.Routing,
		})
	}
	return kept
}

// encodeTasks serialises the tasks and records the request attempt. It reports
// a terminal failure for every task when an item body cannot be re-read.
func (bi *bulkIndexer) encodeTasks(notes *notifier, tasks []*replayTask) (io.Reader, int, error) {
	body, bodyLen, err := encodeBulkTasks(tasks)
	if err != nil {
		for _, t := range tasks {
			bi.finishFailed(notes, t, BulkIndexerResponseItem{}, &BulkReplayError{
				Err:        ErrItemNotReplayable,
				Attempts:   int(t.attempts.Load()),
				Reason:     "cannot re-encode item body: " + err.Error(),
				DocumentID: t.item.DocumentID,
				Index:      t.item.Index,
				Routing:    t.item.Routing,
			})
		}
		return nil, 0, err
	}

	for _, t := range tasks {
		t.attempts.Add(1)
	}
	return body, bodyLen, nil
}

// performBulk sends one bulk request and classifies request-level failures.
// It returns nil when the request did not produce a body to decode.
func (bi *bulkIndexer) performBulk(ctx context.Context, notes *notifier, tasks []*replayTask, body io.Reader) (*esapi.Response, time.Duration) {
	atomic.AddUint64(&bi.stats.numRequests, 1)
	req := bi.newBulkRequest(body)
	start := time.Now()
	res, err := req.Do(ctx, bi.config.Client)
	elapsed := time.Since(start)

	if err != nil {
		bi.handleRequestFailure(ctx, notes, tasks, BulkIndexerResponseItem{}, fmt.Errorf("flush: %w", err), true)
		return nil, elapsed
	}

	status := res.StatusCode
	if !res.IsError() {
		return res, elapsed
	}

	if res.Body != nil {
		defer func() { _ = res.Body.Close() }()
	}
	snippet := res.String()
	reqErr := fmt.Errorf("flush: %s", snippet)
	if isRetryableStatus(status) {
		bi.handleRequestFailure(ctx, notes, tasks, BulkIndexerResponseItem{Status: status}, reqErr, true)
		return nil, elapsed
	}

	bi.reportError(ctx, notes, reqErr)
	for _, t := range tasks {
		bi.finishFailed(notes, t, BulkIndexerResponseItem{Status: status}, &BulkReplayError{
			Err:        ErrItemNotRetryable,
			Attempts:   int(t.attempts.Load()),
			Reason:     "bulk request failed with a non-retryable status: " + snippet,
			DocumentID: t.item.DocumentID,
			Index:      t.item.Index,
			Routing:    t.item.Routing,
		})
	}
	return nil, elapsed
}

// flushBytes records bytes sent in a request.
func (bi *bulkIndexer) flushBytes(n int) {
	if n <= 0 {
		return
	}
	atomic.AddUint64(&bi.stats.flushedBytes, uint64(n)) //nolint:gosec // n is positive
}

// flushMs records the elapsed milliseconds of a request.
func (bi *bulkIndexer) flushMs(elapsed time.Duration) {
	if ms := elapsed.Milliseconds(); ms > 0 {
		atomic.AddUint64(&bi.stats.flushedMs, uint64(ms)) //nolint:gosec // elapsed is positive
	}
}

// newBulkRequest builds the configured bulk request around body.
func (bi *bulkIndexer) newBulkRequest(body io.Reader) esapi.BulkRequest {
	req := esapi.BulkRequest{
		Index: bi.config.Index,
		Body:  body,

		Pipeline:            bi.config.Pipeline,
		Refresh:             bi.config.Refresh,
		Source:              bi.config.Source,
		SourceExcludes:      bi.config.SourceExcludes,
		SourceIncludes:      bi.config.SourceIncludes,
		Timeout:             bi.config.Timeout,
		WaitForActiveShards: bi.config.WaitForActiveShards,

		Pretty:     bi.config.Pretty,
		Human:      bi.config.Human,
		ErrorTrace: bi.config.ErrorTrace,
		FilterPath: bi.config.FilterPath,
		Header:     bi.config.Header.Clone(),
	}
	if bi.config.Routing != "" {
		req.Routing = strings.Split(bi.config.Routing, ",")
	}
	if bi.config.RequireAlias {
		req.RequireAlias = &bi.config.RequireAlias
	}
	if transport, ok := bi.config.Client.(elastictransport.Instrumented); ok {
		req.Instrument = transport.InstrumentationEnabled()
	}
	if req.Header == nil {
		req.Header = http.Header{}
	}
	req.Header.Set(elasticsearch.HeaderClientMeta, "h=bp")
	return req
}

// processBulkResponse applies a decoded bulk response to tasks. Missing
// response entries are treated as an ambiguous request: replayable items are
// retried, everything else fails with a clear reason.
func (bi *bulkIndexer) processBulkResponse(ctx context.Context, notes *notifier, tasks []*replayTask, blk BulkIndexerResponse) {
	for i, t := range tasks {
		if i >= len(blk.Items) {
			bi.handleRequestFailure(ctx, notes, tasks[i:], BulkIndexerResponseItem{}, fmt.Errorf(
				"flush: bulk response contains %d items for %d requests", len(blk.Items), len(tasks)), false)
			return
		}

		var (
			op   string
			info BulkIndexerResponseItem
		)
		for k, v := range blk.Items[i] {
			op = k
			info = v
		}

		if info.Error.Type != "" || info.Status > 201 {
			if isRetryableItemFailure(info) {
				bi.scheduleRetry(notes, t, info, &BulkReplayError{
					Err:        ErrItemNotRetryable,
					Attempts:   int(t.attempts.Load()),
					Reason:     bulkItemReason(info),
					DocumentID: t.item.DocumentID,
					Index:      t.item.Index,
					Routing:    t.item.Routing,
				})
			} else {
				bi.finishFailed(notes, t, info, &BulkReplayError{
					Err:        ErrItemNotRetryable,
					Attempts:   int(t.attempts.Load()),
					Reason:     bulkItemReason(info),
					DocumentID: t.item.DocumentID,
					Index:      t.item.Index,
					Routing:    t.item.Routing,
				})
			}
			continue
		}

		bi.finishSuccess(notes, t, info, op)
	}
}

func bulkItemReason(info BulkIndexerResponseItem) string {
	switch {
	case info.Error.Type != "" && info.Error.Reason != "":
		return fmt.Sprintf("Elasticsearch returned %s: %s", info.Error.Type, info.Error.Reason)
	case info.Error.Type != "":
		return "Elasticsearch returned " + info.Error.Type
	default:
		return fmt.Sprintf("Elasticsearch returned status %d", info.Status)
	}
}

// handleRequestFailure classifies items after a request-level failure
// (transport error, retryable HTTP status, or unparseable response). When the
// request is ambiguous, idempotent items are replayed; the rest fail with a
// reason, and no item is counted twice.
func (bi *bulkIndexer) handleRequestFailure(ctx context.Context, notes *notifier, tasks []*replayTask, info BulkIndexerResponseItem, cause error, report bool) {
	if report {
		bi.reportError(ctx, notes, cause)
	}
	for _, t := range tasks {
		ok, reason := replayableAction(&t.item)
		if !ok {
			bi.finishFailed(notes, t, info, &BulkReplayError{
				Err:        ErrItemNotReplayable,
				Attempts:   int(t.attempts.Load()),
				Reason:     reason + " (" + cause.Error() + ")",
				DocumentID: t.item.DocumentID,
				Index:      t.item.Index,
				Routing:    t.item.Routing,
			})
			continue
		}
		bi.scheduleRetry(notes, t, info, &BulkReplayError{
			Err:        ErrItemNotRetryable,
			Attempts:   int(t.attempts.Load()),
			Reason:     cause.Error(),
			DocumentID: t.item.DocumentID,
			Index:      t.item.Index,
			Routing:    t.item.Routing,
		})
	}
}

// scheduleRetry either enqueues a retry for t or, when the item cannot be
// retried safely or the budget is spent, reports the final outcome.
func (bi *bulkIndexer) scheduleRetry(notes *notifier, t *replayTask, info BulkIndexerResponseItem, cause *BulkReplayError) {
	r := bi.replay
	now := time.Now()
	if t.firstFailure.IsZero() {
		t.firstFailure = now
	}

	if ok, reason := replayableAction(&t.item); !ok {
		cause.Err = ErrItemNotReplayable
		cause.Reason = reason + ": " + cause.Reason
		bi.finishFailed(notes, t, info, cause)
		return
	}

	if r.ctx.Err() != nil {
		cause.Err = ErrReplayExhausted
		cause.Reason = "indexer is stopping: " + cause.Reason
		bi.finishAbandoned(notes, t, cause)
		return
	}

	if int(t.attempts.Load()) >= r.policy.MaxAttempts {
		cause.Err = ErrReplayExhausted
		cause.Reason = fmt.Sprintf("maximum attempts (%d) reached: %s", r.policy.MaxAttempts, cause.Reason)
		bi.finishAbandoned(notes, t, cause)
		return
	}

	backoff := r.policy.Backoff(int(t.attempts.Load()))
	if r.policy.MaxElapsed > 0 && now.Sub(t.firstFailure)+backoff >= r.policy.MaxElapsed {
		cause.Err = ErrReplayExhausted
		cause.Reason = fmt.Sprintf("elapsed budget (%s) would be exceeded: %s", r.policy.MaxElapsed, cause.Reason)
		bi.finishAbandoned(notes, t, cause)
		return
	}

	if key, ok := bi.itemKey(&t.item); ok && orderingProtected(&t.item) && !r.keys.wouldSend(key, t.item.seq) {
		cause.Err = ErrSupersededByNewerDocument
		cause.Reason = "a newer submission for this document identifier has already been sent"
		bi.finishAbandoned(notes, t, cause)
		return
	}

	t.readyAt = now.Add(backoff)
	if !r.enqueue(t) {
		cause.Err = ErrReplayExhausted
		cause.Reason = "replay queue is closed: " + cause.Reason
		bi.finishAbandoned(notes, t, cause)
	}
}

// orderingProtected reports whether the submission is ordered purely by
// arrival. When an explicit version is supplied, Elasticsearch enforces
// version semantics and decides which submission wins, so the indexer must not
// abandon a lower-sequence submission on its own.
func orderingProtected(item *BulkIndexerItem) bool {
	return item.Version == nil
}

// replayableAction reports whether an action may be replayed without changing
// the final document set or content.
func replayableAction(item *BulkIndexerItem) (bool, string) {
	if item.DocumentID == "" {
		return false, "document has no ID; Elasticsearch would assign a new one, duplicating the document"
	}
	switch item.Action {
	case actionIndex, actionDelete:
		return true, ""
	case actionCreate:
		return false, "create is not idempotent: if the first attempt landed, a retry would conflict"
	case actionUpdate:
		return false, "update may run a non-idempotent script; use RetryOnConflict for update conflicts"
	default:
		return false, fmt.Sprintf("action %q is not known to be idempotent", item.Action)
	}
}

// isRetryableStatus classifies whole-request HTTP statuses.
func isRetryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests:
		return true
	}
	return status >= 500 && status <= 599
}

// isRetryableItemFailure classifies a single response item.
func isRetryableItemFailure(info BulkIndexerResponseItem) bool {
	if isRetryableStatus(info.Status) {
		return true
	}
	switch info.Error.Type {
	case "es_rejected_execution_exception",
		"circuit_breaking_exception",
		"cluster_block_exception",
		"unavailable_shards_exception",
		"no_shard_available_action_exception",
		"primary_shard_not_allocated_exception",
		"node_not_connected_exception",
		"node_disconnected_exception",
		"process_cluster_event_timeout_exception",
		"timeout_exception":
		return true
	}
	// Version conflicts are intentionally terminal: version semantics decide
	// the winner, and replaying would either be a no-op or reorder writes.
	return false
}

// finishSuccess records a successful item and invokes OnSuccess once.
func (bi *bulkIndexer) finishSuccess(notes *notifier, t *replayTask, info BulkIndexerResponseItem, op string) {
	if !bi.markNotified(t) {
		return
	}
	if t.attempts.Load() > 1 {
		atomic.AddUint64(&bi.stats.numReplayed, 1)
	} else {
		atomic.AddUint64(&bi.stats.numFlushed, 1)
	}
	switch op {
	case actionIndex:
		atomic.AddUint64(&bi.stats.numIndexed, 1)
	case actionCreate:
		atomic.AddUint64(&bi.stats.numCreated, 1)
	case actionDelete:
		atomic.AddUint64(&bi.stats.numDeleted, 1)
	case actionUpdate:
		atomic.AddUint64(&bi.stats.numUpdated, 1)
	}
	if t.item.OnSuccess != nil {
		notes.add(func() { t.item.OnSuccess(t.item.ctx, t.item, info) })
	}
	bi.releaseKey(t)
}

// finishFailed records a terminal failure and invokes OnFailure once.
func (bi *bulkIndexer) finishFailed(notes *notifier, t *replayTask, info BulkIndexerResponseItem, cause error) {
	if !bi.markNotified(t) {
		return
	}
	atomic.AddUint64(&bi.stats.numFailed, 1)
	if t.item.OnFailure != nil {
		notes.add(func() { t.item.OnFailure(t.item.ctx, t.item, info, cause) })
	}
	bi.releaseKey(t)
}

// finishAbandoned records an item given up on and invokes OnFailure once.
func (bi *bulkIndexer) finishAbandoned(notes *notifier, t *replayTask, cause error) {
	if !bi.markNotified(t) {
		return
	}
	atomic.AddUint64(&bi.stats.numAbandoned, 1)
	if t.item.OnFailure != nil {
		notes.add(func() { t.item.OnFailure(t.item.ctx, t.item, BulkIndexerResponseItem{}, cause) })
	}
	bi.releaseKey(t)
}

func (bi *bulkIndexer) markNotified(t *replayTask) bool {
	bi.replay.mu.Lock()
	defer bi.replay.mu.Unlock()
	if t.notified {
		return false
	}
	t.notified = true
	return true
}

func (bi *bulkIndexer) releaseKey(t *replayTask) {
	if key, ok := bi.itemKey(&t.item); ok {
		bi.replay.keys.release(key)
	}
}

// releaseKey deregisters a submission that never reached a replayTask.
func (r *keyRegistry) releaseKey(bi *bulkIndexer, item *BulkIndexerItem) {
	if key, ok := bi.itemKey(item); ok {
		r.release(key)
	}
}

// encodeBulkTasks builds a bulk request body from tasks, replaying each item's
// original metadata (which carries its own _index and routing) and body.
func encodeBulkTasks(tasks []*replayTask) (io.Reader, int, error) {
	var buf bytes.Buffer
	for _, t := range tasks {
		meta := t.item.meta.Bytes()
		buf.Write(meta)
		if t.item.Body != nil {
			if _, err := t.item.Body.Seek(0, io.SeekStart); err != nil {
				return nil, 0, fmt.Errorf("seek body of document %q: %w", t.item.DocumentID, err)
			}
			if _, err := buf.ReadFrom(t.item.Body); err != nil {
				return nil, 0, fmt.Errorf("read body of document %q: %w", t.item.DocumentID, err)
			}
			buf.WriteByte('\n')
		}
	}
	n := buf.Len()
	return bytes.NewReader(buf.Bytes()), n, nil
}
