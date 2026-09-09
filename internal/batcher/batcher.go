package batcher

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/queue"
)

// Config controls when a batch is flushed.
type Config struct {
	MaxBatchSize   int
	WindowDuration time.Duration
}

// DefaultConfig returns production-default batcher settings.
func DefaultConfig() Config {
	return Config{
		MaxBatchSize:   8,
		WindowDuration: 20 * time.Millisecond,
	}
}

// Batcher accumulates jobs from a queue and flushes them in batches.
type Batcher struct {
	cfg      Config
	drain    <-chan queue.Job
	dispatch func([]queue.Job)
	active   atomic.Int32
}

// New creates a Batcher. dispatch is called with each completed batch.
func New(cfg Config, drain <-chan queue.Job, dispatch func([]queue.Job)) *Batcher {
	return &Batcher{cfg: cfg, drain: drain, dispatch: dispatch}
}

// ActiveBatchSize returns the number of jobs in the current in-progress batch.
// Safe to call from any goroutine.
func (b *Batcher) ActiveBatchSize() int {
	return int(b.active.Load())
}

// Run is the main batching loop. It exits when ctx is cancelled and the drain
// channel is empty (or closed).
func (b *Batcher) Run(ctx context.Context) {
	batch := make([]queue.Job, 0, b.cfg.MaxBatchSize)
	timer := time.NewTimer(b.cfg.WindowDuration)
	defer timer.Stop()

	flush := func() {
		if len(batch) == 0 {
			return
		}
		b.dispatch(batch)
		batch = make([]queue.Job, 0, b.cfg.MaxBatchSize)
		b.active.Store(0)
		// Reset timer after flush so the next window starts fresh.
		if !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
		timer.Reset(b.cfg.WindowDuration)
	}

	for {
		select {
		case job, ok := <-b.drain:
			if !ok {
				// Channel closed: flush remaining jobs and exit.
				flush()
				return
			}
			batch = append(batch, job)
			b.active.Store(int32(len(batch)))
			if len(batch) >= b.cfg.MaxBatchSize {
				flush()
			}

		case <-timer.C:
			flush()
			timer.Reset(b.cfg.WindowDuration)

		case <-ctx.Done():
			// Drain any jobs already in the channel before exiting.
		drain:
			for {
				select {
				case job, ok := <-b.drain:
					if !ok {
						break drain
					}
					batch = append(batch, job)
					b.active.Store(int32(len(batch)))
					if len(batch) >= b.cfg.MaxBatchSize {
						flush()
					}
				default:
					break drain
				}
			}
			flush()
			return
		}
	}
}
