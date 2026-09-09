package queue_test

import (
	"context"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/queue"
)

func makeJob(id string) queue.Job {
	return queue.Job{
		ID: id,
		Req: backend.Request{
			RequestID: id,
			Modality:  backend.ModalityKindText,
		},
		ResultCh: make(chan queue.Result, 1),
	}
}

func TestEnqueueDequeueOrder(t *testing.T) {
	q := queue.New(4)
	ctx := context.Background()

	ids := []string{"a", "b", "c"}
	for _, id := range ids {
		if err := q.Enqueue(ctx, makeJob(id)); err != nil {
			t.Fatalf("Enqueue(%q): %v", id, err)
		}
	}

	drain := q.Drain()
	for _, want := range ids {
		got := <-drain
		if got.ID != want {
			t.Errorf("got job %q, want %q", got.ID, want)
		}
	}
}

func TestEnqueueBlocksWhenFull(t *testing.T) {
	q := queue.New(1)
	ctx := context.Background()

	if err := q.Enqueue(ctx, makeJob("first")); err != nil {
		t.Fatal(err)
	}

	// Second enqueue should block until drained.
	enqueued := make(chan error, 1)
	go func() {
		enqueued <- q.Enqueue(ctx, makeJob("second"))
	}()

	select {
	case <-enqueued:
		t.Fatal("Enqueue returned before drain — should have blocked")
	case <-time.After(20 * time.Millisecond):
	}

	// Drain first job; second should now enqueue.
	<-q.Drain()
	select {
	case err := <-enqueued:
		if err != nil {
			t.Fatalf("Enqueue after drain: %v", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Enqueue still blocked after drain")
	}
}

func TestEnqueueCancelledContext(t *testing.T) {
	q := queue.New(1)
	ctx, cancel := context.WithCancel(context.Background())

	if err := q.Enqueue(ctx, makeJob("fill")); err != nil {
		t.Fatal(err)
	}

	cancel()
	err := q.Enqueue(ctx, makeJob("blocked"))
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestCloseExitsBatcher(t *testing.T) {
	q := queue.New(2)
	q.Close()

	// drain channel should be closed; receive should return zero value + ok=false
	_, ok := <-q.Drain()
	if ok {
		t.Fatal("expected closed channel after Close()")
	}
}
