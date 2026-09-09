package queue

import (
	"context"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
)

// Result carries the outcome of a single inference job back to the HTTP handler.
type Result struct {
	Output          string
	Reasoning       string
	TokensGenerated int
	ModelTier       string
	QualityDegraded bool
	TokPerSecSample float64
	StartedAt       time.Time
	Err             error
}

// Job is one unit of work flowing through the pipeline.
type Job struct {
	ID            string
	CorrelationID string
	EnqueuedAt    time.Time
	Req           backend.Request
	ResultCh      chan Result
}

// Queue is a goroutine-safe bounded job queue backed by a buffered channel.
type Queue struct {
	ch     chan Job
	doneCh chan struct{}
}

// New returns a Queue with the given capacity. Enqueue blocks when full.
func New(capacity int) *Queue {
	return &Queue{
		ch:     make(chan Job, capacity),
		doneCh: make(chan struct{}),
	}
}

// Enqueue adds a job. Blocks until space is available or ctx is done.
func (q *Queue) Enqueue(ctx context.Context, job Job) error {
	select {
	case q.ch <- job:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Drain returns the read side of the internal channel for the batcher to consume.
func (q *Queue) Drain() <-chan Job {
	return q.ch
}

// Depth returns the current number of pending jobs.
func (q *Queue) Depth() int {
	return len(q.ch)
}

// Close signals that no more jobs will be enqueued; the batcher loop exits after
// draining remaining items.
func (q *Queue) Close() {
	close(q.ch)
}
