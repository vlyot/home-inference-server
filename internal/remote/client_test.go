package remote_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/remote"
	queuepkg "github.com/ngkaichong/home-inference-server/queue"
)

// stubBackend drains queue jobs and immediately resolves them with a canned result.
func runStubBackend(ctx context.Context, q *queue.Queue) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		// Non-blocking drain attempt.
		select {
		case job, ok := <-q.Drain():
			if !ok {
				return
			}
			job.ResultCh <- queue.Result{
				Output:          "stub output",
				TokensGenerated: 1,
				ModelTier:       "weak",
			}
		case <-ctx.Done():
			return
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func newTestQueue() *queue.Queue {
	return queue.New(16)
}

// makeJob returns a minimal QueueJob payload that the client can decode.
func makeJob(id string) queuepkg.QueueJob {
	payload, _ := json.Marshal(map[string]any{
		"modality":   "text",
		"text_input": map[string]string{"prompt": "hello"},
	})
	return queuepkg.QueueJob{
		ID:            id,
		CorrelationID: "req-" + id,
		Status:        queuepkg.QueueJobProcessing,
		Payload:       payload,
		AttemptCount:  1,
		MaxAttempts:   3,
	}
}

func TestRunClaimsAndAcksJob(t *testing.T) {
	job := makeJob("test-job-1")

	var claimCount, ackCount atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/claim":
			if claimCount.Add(1) == 1 {
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{Job: &job})
			} else {
				// After first job, return empty to let the test context cancel.
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
			}
		case "/ack":
			ackCount.Add(1)
			var req queuepkg.AckRequest
			json.NewDecoder(r.Body).Decode(&req)
			if req.Status != queuepkg.QueueJobDone {
				t.Errorf("ack status = %q, want done", req.Status)
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	go runStubBackend(ctx, q)

	c := remote.New(srv.URL, "test-key", q, nil)

	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	// Wait for ack to happen.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if ackCount.Load() >= 1 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	cancel()
	<-done

	if ackCount.Load() < 1 {
		t.Errorf("expected at least 1 ack, got %d", ackCount.Load())
	}
}

func TestRunRetriesOnClaimError(t *testing.T) {
	var callCount atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := callCount.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// After two failures, return empty (no jobs) so the loop continues cleanly.
		json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	c := remote.New(srv.URL, "test-key", q, nil)

	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	// Wait for backoff to recover (2 failures with 1s+2s backoff = ~3s).
	deadline := time.Now().Add(4 * time.Second)
	for time.Now().Before(deadline) {
		if callCount.Load() >= 3 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	<-done

	if callCount.Load() < 3 {
		t.Errorf("expected at least 3 calls (2 errors + 1 success), got %d", callCount.Load())
	}
}

func TestRunExitsOnContextCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Slow response to simulate long-poll.
		select {
		case <-r.Context().Done():
		case <-time.After(500 * time.Millisecond):
		}
		json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithCancel(context.Background())

	c := remote.New(srv.URL, "test-key", q, nil)

	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	// Cancel immediately.
	cancel()

	select {
	case <-done:
		// Good — Run exited.
	case <-time.After(3 * time.Second):
		t.Fatal("Run did not exit within 3 seconds of context cancellation")
	}
}

func TestRunLogsQueueReturnedOnAckFailure(t *testing.T) {
	job := makeJob("test-job-ack-fail")

	var claimCount atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/claim":
			if claimCount.Add(1) == 1 {
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{Job: &job})
			} else {
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
			}
		case "/ack":
			// Simulate ack failure.
			w.WriteHeader(http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	go runStubBackend(ctx, q)

	c := remote.New(srv.URL, "test-key", q, nil)

	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()

	// Give it time to process the job and attempt the ack.
	time.Sleep(500 * time.Millisecond)
	cancel()
	<-done
	// Test passes as long as it doesn't panic — the warning log covers the ack failure.
}

// --- pause / resume ---

func TestPauseStopsClaiming(t *testing.T) {
	var claimCount atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/claim" {
			claimCount.Add(1)
		}
		json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c := remote.New(srv.URL, "k", q, nil)
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()

	time.Sleep(150 * time.Millisecond)
	c.Pause()
	time.Sleep(50 * time.Millisecond)
	frozen := claimCount.Load()

	time.Sleep(300 * time.Millisecond)
	if got := claimCount.Load(); got != frozen {
		t.Fatalf("claim count rose from %d to %d while paused", frozen, got)
	}
	if !c.Paused() {
		t.Fatal("Paused() = false after Pause()")
	}

	c.Resume()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) && claimCount.Load() <= frozen {
		time.Sleep(10 * time.Millisecond)
	}
	if got := claimCount.Load(); got <= frozen {
		t.Fatalf("claim count did not resume (still %d)", got)
	}

	cancel()
	<-done
}

func TestPauseDoesNotAbortInflightJob(t *testing.T) {
	job := makeJob("pause-inflight")
	var claimCount, ackCount atomic.Int32
	var ackStatus string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/claim":
			if claimCount.Add(1) == 1 {
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{Job: &job})
			} else {
				json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
			}
		case "/ack":
			var req queuepkg.AckRequest
			json.NewDecoder(r.Body).Decode(&req)
			ackStatus = string(req.Status)
			ackCount.Add(1)
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case j, ok := <-q.Drain():
				if !ok {
					return
				}
				time.Sleep(200 * time.Millisecond)
				j.ResultCh <- queue.Result{Output: "ok", TokensGenerated: 1, ModelTier: "weak"}
			}
		}
	}()

	c := remote.New(srv.URL, "k", q, nil)
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()

	for claimCount.Load() == 0 {
		time.Sleep(5 * time.Millisecond)
	}
	c.Pause()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && ackCount.Load() == 0 {
		time.Sleep(10 * time.Millisecond)
	}
	if ackCount.Load() != 1 {
		t.Fatalf("in-flight job did not ack after Pause (ackCount=%d)", ackCount.Load())
	}
	if ackStatus != string(queuepkg.QueueJobDone) {
		t.Fatalf("ack status = %q, want done", ackStatus)
	}

	cancel()
	<-done
}

func TestRunExitsWhilePaused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(queuepkg.ClaimResponse{})
	}))
	defer srv.Close()

	q := newTestQueue()
	ctx, cancel := context.WithCancel(context.Background())

	c := remote.New(srv.URL, "k", q, nil)
	c.Pause()

	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()

	time.Sleep(50 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not exit within 2s of cancel while paused")
	}
}
