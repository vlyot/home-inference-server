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

	// Salvaging partial content only makes sense for free-form prose: half a
	// sentence is still a real, readable answer. It does NOT make sense when
	// response_format constrained the output to JSON — a stream cut off
	// mid-object is not "an incomplete answer a caller could use", it's
	// invalid JSON that will fail to parse, and a client trusting Truncated
	// as "safe to use" would break on it. Structured-output callers (the
	// primary consumers of this API — see docs) need an unambiguous failure
	// here, not a partial payload dressed up as a soft success.
	salvageable := len(req.ResponseFormat) == 0

	if err != nil && contentEmitted && salvageable {
		// The backend failed AFTER already streaming real content (observed on
		// real hardware: the Gemma-4 subprocess can crash mid-generation from
		// an unresolved upstream llama.cpp bug — see roadmap). Every prior
		// Delta chunk the client received is genuine model output, not
		// garbage from a failed request — discarding the whole turn (as the
		// Error path does) would throw away a real, if incomplete, answer the
		// user already saw stream in. Salvage it: end the stream as Truncated,
		// not Error, so the client keeps what arrived instead of dropping it.
		var be *backend.BackendError
		errCode := api.ErrCodeInternal
		if isBackendErr(err, &be) {
			errCode = be.Code
		}
		slog.Info("job truncated (stream)",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldRequestID, backendReq.RequestID),
			slog.String(logschema.FieldErrorCode, errCode),
		)
		s.recordJob(types.JobEntry{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusDone,
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
			DurationMS:    durationMS,
			QueueWaitMS:   queueWaitMS,
			InferenceMS:   inferenceMS,
			ModelTier:     resp.ModelTier,
			Truncated:     true,
		})
		return
	}

	if err != nil {
		var be *backend.BackendError
		errCode := api.ErrCodeInternal
		if isBackendErr(err, &be) {
			errCode = be.Code
		}
		if contentEmitted {
			// Reaches here only when !salvageable (response_format was set) —
			// real content streamed but had to be discarded as unparseable
			// partial JSON rather than salvaged. Worth a distinct log line:
			// this is the "the crash mitigation didn't apply here, and here's
			// why" case, not a from-the-start failure.
			slog.Info("job failed (stream) — partial content discarded, not salvageable (response_format set)",
				slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
				slog.String(logschema.FieldCorrelationID, req.CorrelationID),
				slog.String(logschema.FieldRequestID, backendReq.RequestID),
				slog.String(logschema.FieldErrorCode, errCode),
			)
		} else {
			slog.Info("job failed (stream)",
				slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
				slog.String(logschema.FieldCorrelationID, req.CorrelationID),
				slog.String(logschema.FieldRequestID, backendReq.RequestID),
				slog.String(logschema.FieldErrorCode, errCode),
			)
		}
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
