// Package dispatcher is the bounded worker pool that feeds the router. The
// batcher hands it whole batches; N worker goroutines pull batches and run each
// job through router.DispatchOne, so up to N inferences proceed concurrently
// (the hard ceiling is enforced downstream by vram.Backend's semaphore, which
// also covers the streaming path). This replaced an earlier unbounded
// `go router.Dispatch(batch)` fan-out.
package dispatcher

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"

	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/logschema"
)

// Dispatcher runs a fixed pool of workers that drain batches into the router.
type Dispatcher struct {
	router   *router.Router
	n        int
	batches  chan []queue.Job
	inFlight atomic.Int32
	wg       sync.WaitGroup
}

// New builds a Dispatcher with n workers (clamped to >= 1). The batch buffer is
// sized to a small multiple of n so a brief burst of flushes from the batcher
// doesn't block its goroutine; sustained backpressure is the job queue's job.
func New(r *router.Router, n int) *Dispatcher {
	if n < 1 {
		n = 1
	}
	return &Dispatcher{
		router:  r,
		n:       n,
		batches: make(chan []queue.Job, n*4),
	}
}

// Run starts the worker pool and blocks until ctx is cancelled, then waits for
// in-flight jobs to finish. Intended to be called as `go d.Run(ctx)`.
func (d *Dispatcher) Run(ctx context.Context) {
	for i := 0; i < d.n; i++ {
		d.wg.Add(1)
		go d.worker(ctx)
	}
	<-ctx.Done()
	d.wg.Wait()
}

func (d *Dispatcher) worker(ctx context.Context) {
	defer d.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case batch := <-d.batches:
			for _, job := range batch {
				d.inFlight.Add(1)
				d.router.DispatchOne(ctx, job, len(batch))
				d.inFlight.Add(-1)
			}
		}
	}
}

// Submit hands a flushed batch to the pool without blocking the caller (the
// batcher goroutine). Returns false if the batch buffer is full — the caller
// then dispatches the batch inline so no job is dropped.
func (d *Dispatcher) Submit(batch []queue.Job) bool {
	select {
	case d.batches <- batch:
		slog.Info("batch dispatched",
			slog.String(logschema.FieldEvent, string(logschema.EventBatchDispatched)),
			slog.Int(logschema.FieldBatchSize, len(batch)),
			slog.Int(logschema.FieldInFlight, d.InFlight()),
			slog.Int(logschema.FieldMaxParallel, d.n),
		)
		return true
	default:
		return false
	}
}

// InFlight is the number of jobs currently executing in router.DispatchOne.
func (d *Dispatcher) InFlight() int { return int(d.inFlight.Load()) }

// MaxParallel is the worker count.
func (d *Dispatcher) MaxParallel() int { return d.n }
