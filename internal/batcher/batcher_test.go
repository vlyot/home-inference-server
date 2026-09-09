package batcher_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/batcher"
	"github.com/ngkaichong/home-inference-server/internal/queue"
)

func makeJob(id string) queue.Job {
	return queue.Job{
		ID:       id,
		Req:      backend.Request{RequestID: id, Modality: backend.ModalityKindText},
		ResultCh: make(chan queue.Result, 1),
	}
}

func fillQueue(t *testing.T, q *queue.Queue, n int) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < n; i++ {
		if err := q.Enqueue(ctx, makeJob("job")); err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
	}
}

func TestFlushOnMaxBatchSize(t *testing.T) {
	q := queue.New(16)
	fillQueue(t, q, 4)

	cfg := batcher.Config{MaxBatchSize: 4, WindowDuration: 10 * time.Second}
	var mu sync.Mutex
	var got [][]queue.Job

	dispatch := func(batch []queue.Job) {
		mu.Lock()
		got = append(got, batch)
		mu.Unlock()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	b := batcher.New(cfg, q.Drain(), dispatch)
	go b.Run(ctx)

	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || len(got[0]) != 4 {
		t.Fatalf("expected 1 batch of 4, got %v batches", len(got))
	}
}

func TestFlushOnWindowExpiry(t *testing.T) {
	q := queue.New(16)
	fillQueue(t, q, 2) // below MaxBatchSize=8

	cfg := batcher.Config{MaxBatchSize: 8, WindowDuration: 30 * time.Millisecond}
	var mu sync.Mutex
	var got [][]queue.Job

	dispatch := func(batch []queue.Job) {
		mu.Lock()
		got = append(got, batch)
		mu.Unlock()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	b := batcher.New(cfg, q.Drain(), dispatch)
	go b.Run(ctx)

	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if len(got) == 0 || len(got[0]) != 2 {
		t.Fatalf("expected 1 batch of 2 after window, got batches: %v", got)
	}
}

func TestActiveBatchSizeIsZeroWhenIdle(t *testing.T) {
	q := queue.New(16)
	cfg := batcher.Config{MaxBatchSize: 8, WindowDuration: 5 * time.Second}
	b := batcher.New(cfg, q.Drain(), func(_ []queue.Job) {})
	if got := b.ActiveBatchSize(); got != 0 {
		t.Fatalf("expected 0 before Run, got %d", got)
	}
}

func TestActiveBatchSizeIncrementsAndResets(t *testing.T) {
	q := queue.New(16)

	cfg := batcher.Config{MaxBatchSize: 4, WindowDuration: 10 * time.Second}

	flushed := make(chan struct{}, 1)
	b := batcher.New(cfg, q.Drain(), func(batch []queue.Job) {
		flushed <- struct{}{}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go b.Run(ctx)

	// Enqueue 4 jobs (triggers flush on max batch size).
	ctx2 := context.Background()
	for i := 0; i < 4; i++ {
		if err := q.Enqueue(ctx2, makeJob("j")); err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
	}

	select {
	case <-flushed:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("flush did not fire")
	}

	// After flush the counter must be 0.
	time.Sleep(10 * time.Millisecond)
	if got := b.ActiveBatchSize(); got != 0 {
		t.Fatalf("expected 0 after flush, got %d", got)
	}
}

func TestExitsOnCtxCancelled(t *testing.T) {
	q := queue.New(16)
	cfg := batcher.Config{MaxBatchSize: 8, WindowDuration: 5 * time.Second}

	dispatch := func(_ []queue.Job) {}

	ctx, cancel := context.WithCancel(context.Background())
	b := batcher.New(cfg, q.Drain(), dispatch)

	done := make(chan struct{})
	go func() {
		b.Run(ctx)
		close(done)
	}()

	cancel()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("batcher did not exit after context cancellation")
	}
}
