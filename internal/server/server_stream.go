package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// handleInferStream handles POST /v1/infer when req.Stream == true.
// It bypasses the queue/batcher and proxies SSE chunks directly from the
// backend to the client. Falls back to non-streaming if the backend does not
// implement backend.Streamer.
func (s *Server) handleInferStream(w http.ResponseWriter, r *http.Request, req api.InferRequest, backendReq backend.Request) {
	// Under pressure, hand the request to the durable relay and return 202
	// (not an SSE stream — the caller polls the relay's /result/{id}).
	if s.deferToRelay(w, r, req, backendReq) {
		return
	}

	// Resolve the backend for this modality.
	s.backendsMu.RLock()
	b, ok := s.backends[backend.ModalityKind(req.Modality)]
	s.backendsMu.RUnlock()

	streamer, canStream := b.(backend.Streamer)
	if !ok || !canStream {
		// Backend not registered or doesn't support streaming — fall back.
		s.handleInferBlocking(w, r, req, backendReq)
		return
	}

	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		// ResponseWriter doesn't support flushing (e.g. in some test harnesses).
		s.handleInferBlocking(w, r, req, backendReq)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")

	enqueuedAt := time.Now()

	slog.Info("job queued (stream)",
		slog.String(logschema.FieldEvent, string(logschema.EventJobQueued)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
	)

	startedAt := time.Now()

	writeSSEChunk := func(v any) {
		data, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
	// writeDone emits the terminal chunk followed by the "data: [DONE]" sentinel
	// line that the docs promise and OpenAI-style clients expect.
	writeDone := func(chunk api.StreamChunk) {
		chunk.Done = true
		writeSSEChunk(chunk)
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}

	contentEmitted := false
	reasoningEmitted := false
	resp, err := streamer.InferStream(r.Context(), backendReq, func(kind backend.ChunkKind, delta string) {
		chunk := api.StreamChunk{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
		}
		if kind == backend.ChunkReasoning {
			reasoningEmitted = true
			chunk.Reasoning = delta
		} else {
			contentEmitted = true
			chunk.Delta = delta
		}
		writeSSEChunk(chunk)
	})

	finishedAt := time.Now()
	durationMS := finishedAt.Sub(enqueuedAt).Milliseconds()
	queueWaitMS := startedAt.Sub(enqueuedAt).Milliseconds()
	inferenceMS := finishedAt.Sub(startedAt).Milliseconds()

	if err != nil {
		var be *backend.BackendError
		errCode := api.ErrCodeInternal
		if isBackendErr(err, &be) {
			errCode = be.Code
		}
		slog.Info("job failed (stream)",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldRequestID, backendReq.RequestID),
			slog.String(logschema.FieldErrorCode, errCode),
		)
		s.recordJob(types.JobEntry{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusFailed,
			Modality:      string(req.Modality),
			Priority:      req.Priority,
			MinTier:       req.MinTier,
			EnqueuedAt:    enqueuedAt,
			StartedAt:     startedAt,
			FinishedAt:    finishedAt,
			DurationMS:    durationMS,
			QueueWaitMS:   queueWaitMS,
			InferenceMS:   inferenceMS,
		})
		// Signal error in the stream via a done chunk carrying the error code.
		writeDone(api.StreamChunk{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Error:         errCode,
		})
		return
	}

	// A completion that returned no error but produced no visible answer is a
	// failure from the caller's point of view. Two distinct cases:
	//   - nothing at all (no content, no reasoning, no counted tokens): the
	//     subprocess may still have been loading — ErrCodeInternal.
	//   - reasoning happened (a reasoning-capable model's "thinking" phase ran
	//     and consumed tokens) but no content ever followed: the model ran out
	//     of its token budget before answering — ErrCodeReasoningExhausted, a
	//     distinct signal so the client can say what actually happened rather
	//     than a generic "no output".
	if !contentEmitted {
		errCode := api.ErrCodeInternal
		if reasoningEmitted {
			errCode = api.ErrCodeReasoningExhausted
		}
		slog.Info("job failed (stream)",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldRequestID, backendReq.RequestID),
			slog.String(logschema.FieldErrorCode, errCode),
		)
		s.recordJob(types.JobEntry{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusFailed,
			Modality:      string(req.Modality),
			ModelTier:     resp.ModelTier,
			Priority:      req.Priority,
			MinTier:       req.MinTier,
			EnqueuedAt:    enqueuedAt,
			StartedAt:     startedAt,
			FinishedAt:    finishedAt,
			DurationMS:    durationMS,
			QueueWaitMS:   queueWaitMS,
			InferenceMS:   inferenceMS,
		})
		writeDone(api.StreamChunk{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Error:         errCode,
		})
		return
	}

	if resp.QualityDegraded {
		w.Header().Set(api.HeaderQualityDegraded, "true")
	}

	writeDone(api.StreamChunk{
		RequestID:       backendReq.RequestID,
		CorrelationID:   req.CorrelationID,
		TokensGenerated: resp.TokensGenerated,
		TokensPerSec:    resp.TokPerSecSample,
		QueueWaitMS:     queueWaitMS,
		InferenceMS:     inferenceMS,
		DurationMS:      durationMS,
		ModelTier:       resp.ModelTier,
	})

	slog.Info("job done (stream)",
		slog.String(logschema.FieldEvent, string(logschema.EventJobDone)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
		slog.Int64(logschema.FieldDurationMS, durationMS),
		slog.Int(logschema.FieldTokensGenerated, resp.TokensGenerated),
		slog.String(logschema.FieldModelTier, resp.ModelTier),
	)

	s.recordJob(types.JobEntry{
		RequestID:       backendReq.RequestID,
		CorrelationID:   req.CorrelationID,
		Source:          types.JobSourceLocal,
		Status:          types.JobStatusDone,
		Modality:        string(req.Modality),
		ModelTier:       resp.ModelTier,
		Priority:        req.Priority,
		MinTier:         req.MinTier,
		EnqueuedAt:      enqueuedAt,
		StartedAt:       startedAt,
		FinishedAt:      finishedAt,
		DurationMS:      durationMS,
		QueueWaitMS:     queueWaitMS,
		InferenceMS:     inferenceMS,
		TokensGenerated: resp.TokensGenerated,
	})
}

// isBackendErr is a local alias used by server_stream.go to avoid shadowing
// the errors.As helper in server.go.
func isBackendErr(err error, out **backend.BackendError) bool {
	be, ok := err.(*backend.BackendError)
	if ok {
		*out = be
	}
	return ok
}
