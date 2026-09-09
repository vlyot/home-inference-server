package dispatcher

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
)

// slowBackend sleeps per Infer and tracks peak concurrent occupancy.
type slowBackend struct {
	delay    time.Duration
	inFlight atomic.Int32
	peak     atomic.Int32
}

func (s *slowBackend) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (s *slowBackend) Ready() bool                    { return true }
func (s *slowBackend) Shutdown(context.Context) error { return nil }
func (s *slowBackend) Infer(ctx context.Context, _ backend.Request) (backend.Response, error) {
	n := s.inFlight.Add(1)
	for {
		p := s.peak.Load()
		if n <= p || s.peak.CompareAndSwap(p, n) {
			break
		}
	}
	defer s.inFlight.Add(-1)
	select {
	case <-time.After(s.delay):
	case <-ctx.Done():
		return backend.Response{}, ctx.Err()
	}
	return backend.Response{Output: "ok"}, nil
}

type panicBackend struct{}

func (panicBackend) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (panicBackend) Ready() bool                    { return true }
func (panicBackend) Shutdown(context.Context) error { return nil }
func (panicBackend) Infer(context.Context, backend.Request) (backend.Response, error) {
	panic("boom")
}

func newJob(id string) queue.Job {
	return queue.Job{
		ID:       id,
		Req:      backend.Request{RequestID: id, Modality: backend.ModalityKindText},
		ResultCh: make(chan queue.Result, 1),
	}
}

func TestRunProcessesEveryJobInBatch(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: &slowBackend{},
	})
	d := New(r, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	batch := make([]queue.Job, 6)
	for i := range batch {
		batch[i] = newJob("j")
	}
	if !d.Submit(batch) {
		t.Fatal("Submit returned false on an empty pool")
	}
	for i, job := range batch {
		select {
		case res := <-job.ResultCh:
			if res.Err != nil {
				t.Fatalf("job %d: %v", i, res.Err)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("job %d never completed", i)
		}
	}
}

func TestConcurrentBatchesRespectPoolSize(t *testing.T) {
	sb := &slowBackend{delay: 50 * time.Millisecond}
	r := router.New(map[backend.ModalityKind]backend.Backend{backend.ModalityKindText: sb})
	d := New(r, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		job := newJob("j")
		d.Submit([]queue.Job{job})
		wg.Add(1)
		go func() { defer wg.Done(); <-job.ResultCh }()
	}
	wg.Wait()

	if peak := sb.peak.Load(); peak > 2 {
		t.Fatalf("peak concurrency = %d; want <= 2 (pool size)", peak)
	}
}

func TestSubmitNonBlockingWhenBufferFull(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: &slowBackend{delay: time.Second},
	})
	d := New(r, 1) // buffer cap = 1*4 = 4
	// Do NOT start Run: nothing drains d.batches.
	accepted := 0
	for i := 0; i < 50; i++ {
		if d.Submit([]queue.Job{newJob("j")}) {
			accepted++
		}
	}
	if accepted == 50 {
		t.Fatal("Submit never returned false despite no drainer")
	}
	if accepted == 0 {
		t.Fatal("Submit never accepted anything")
	}
}

func TestInFlightGaugeRisesAndFalls(t *testing.T) {
	sb := &slowBackend{delay: 100 * time.Millisecond}
	r := router.New(map[backend.ModalityKind]backend.Backend{backend.ModalityKindText: sb})
	d := New(r, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	job := newJob("j")
	d.Submit([]queue.Job{job})

	deadline := time.After(time.Second)
	for d.InFlight() == 0 {
		select {
		case <-deadline:
			t.Fatal("InFlight never rose above 0")
		default:
			time.Sleep(2 * time.Millisecond)
		}
	}
	<-job.ResultCh
	for i := 0; i < 100 && d.InFlight() != 0; i++ {
		time.Sleep(5 * time.Millisecond)
	}
	if d.InFlight() != 0 {
		t.Fatalf("InFlight = %d after drain; want 0", d.InFlight())
	}
}

func TestRunExitsOnContextCancel(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: &slowBackend{},
	})
	d := New(r, 3)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { d.Run(ctx); close(done) }()
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not return after context cancellation")
	}
}

func TestDispatchOnePanicRecovered(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: panicBackend{},
	})
	d := New(r, 2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go d.Run(ctx)

	bad, good := newJob("bad"), newJob("good")
	d.Submit([]queue.Job{bad, good})

	for _, job := range []queue.Job{bad, good} {
		select {
		case res := <-job.ResultCh:
			if job.ID == "bad" && res.Err == nil {
				t.Fatal("panicking job should have produced an error result")
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("job %s never completed (panic wedged the worker?)", job.ID)
		}
	}
}
