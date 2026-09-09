package router

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/logschema"
)

// Router dispatches batches of jobs to the appropriate backend by modality and
// writes each result back to the job's ResultCh.
type Router struct {
	backends map[backend.ModalityKind]backend.Backend
}

// New builds a Router from a map of modality → backend.
func New(backends map[backend.ModalityKind]backend.Backend) *Router {
	return &Router{backends: backends}
}

// sendResult delivers a result to the job's channel. A nil ResultCh (a job with
// no local waiter) is a no-op rather than a permanent block on a nil channel.
func sendResult(job queue.Job, res queue.Result) {
	if job.ResultCh == nil {
		return
	}
	job.ResultCh <- res
}

// Dispatch processes every job in the batch sequentially, routing each to the
// matching backend and writing the result (or error) to the job's ResultCh.
// A panic in a backend is recovered per job so it cannot kill the batcher.
// The hot path (internal/dispatcher) calls DispatchOne directly, one job per
// pool worker; Dispatch is retained for internal/remote and tests.
func (r *Router) Dispatch(ctx context.Context, batch []queue.Job) {
	batchSize := len(batch)
	for _, job := range batch {
		r.DispatchOne(ctx, job, batchSize)
	}
}

// DispatchOne routes a single job to the matching backend and writes the result
// (or error) to job.ResultCh. A backend panic is recovered here so one bad job
// cannot kill the caller's goroutine. Safe to call concurrently — the Router
// holds no per-job state and each job owns its own result channel.
func (r *Router) DispatchOne(ctx context.Context, job queue.Job, batchSize int) {
	defer func() {
		if p := recover(); p != nil {
			slog.Error("router: recovered from backend panic",
				slog.String(logschema.FieldCorrelationID, job.CorrelationID),
				slog.String(logschema.FieldRequestID, job.ID),
				"panic", p,
			)
			sendResult(job, queue.Result{Err: &backend.BackendError{
				Code:    "internal_error",
				Message: fmt.Sprintf("backend panic: %v", p),
			}})
		}
	}()

	slog.Info("job dispatched",
		slog.String(logschema.FieldEvent, string(logschema.EventJobDispatched)),
		slog.String(logschema.FieldCorrelationID, job.CorrelationID),
		slog.String(logschema.FieldRequestID, job.ID),
		slog.Int(logschema.FieldBatchSize, batchSize),
	)
	b, ok := r.backends[job.Req.Modality]
	if !ok {
		sendResult(job, queue.Result{
			Err: &backend.BackendError{
				Code:    "invalid_modality",
				Message: fmt.Sprintf("no backend registered for modality %q", job.Req.Modality),
			},
		})
		return
	}

	startedAt := time.Now()
	resp, err := b.Infer(ctx, job.Req)
	if err != nil {
		sendResult(job, queue.Result{Err: err, StartedAt: startedAt})
		return
	}

	sendResult(job, queue.Result{
		Output:          resp.Output,
		Reasoning:       resp.Reasoning,
		TokensGenerated: resp.TokensGenerated,
		ModelTier:       resp.ModelTier,
		QualityDegraded: resp.QualityDegraded,
		TokPerSecSample: resp.TokPerSecSample,
		StartedAt:       startedAt,
	})
}
