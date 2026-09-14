package server

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/websearch"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// ToolSearcher resolves the server's built-in "web_search" tool. Implemented
// by *websearch.Client; an interface here (rather than a concrete dependency)
// matches the existing CPUMetrics/PressureSource/VRAMSource optional-dependency
// pattern and keeps this package's tests free of a real HTTP client.
type ToolSearcher interface {
	Search(ctx context.Context, query string) ([]websearch.Result, error)
}

// SetToolSearcher wires the web_search tool's resolver. Nil (the default)
// means any request carrying tools gets ErrCodeToolUnavailable for that call.
func (s *Server) SetToolSearcher(t ToolSearcher) { s.toolSearcher = t }

// maxToolHops bounds the orchestration loop to exactly one grounding
// round-trip: call, resolve tool(s), call again for the final answer. Keeps
// the feature simple and its latency/cost predictable — see roadmap.md.
const maxToolHops = 1

// handleInferWithTools runs the tool-calling orchestration loop: it bypasses
// the queue/batcher (same rationale as handleInferStream — a multi-step
// backend interaction doesn't fit the single-job queue model), calls the
// backend directly, resolves any tool call the model makes, and re-calls the
// backend once more with the results appended before returning the final
// answer in the same api.InferResponse shape handleInferBlocking uses.
func (s *Server) handleInferWithTools(w http.ResponseWriter, r *http.Request, req api.InferRequest, backendReq backend.Request) {
	enqueuedAt := time.Now()

	if s.deferToRelay(w, r, req, backendReq) {
		return
	}

	s.backendsMu.RLock()
	b, ok := s.backends[backend.ModalityKind(req.Modality)]
	s.backendsMu.RUnlock()
	if !ok {
		writeError(w, http.StatusServiceUnavailable, api.ErrCodeUnavailable, "no backend registered for modality", req.CorrelationID)
		return
	}

	startedAt := time.Now()
	resp, toolCalls, err := s.runWithTools(r.Context(), b, backendReq)
	finishedAt := time.Now()
	durationMS := finishedAt.Sub(enqueuedAt).Milliseconds()
	queueWaitMS := startedAt.Sub(enqueuedAt).Milliseconds()
	inferenceMS := finishedAt.Sub(startedAt).Milliseconds()

	if err != nil {
		var be *backend.BackendError
		errCode := api.ErrCodeInternal
		status := http.StatusInternalServerError
		if errors.As(err, &be) {
			errCode = be.Code
			status = backendErrToHTTPStatus(be.Code)
			writeError(w, status, be.Code, be.Message, req.CorrelationID)
		} else {
			writeError(w, status, errCode, err.Error(), req.CorrelationID)
		}
		slog.Info("job failed",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldRequestID, backendReq.RequestID),
			slog.Int64(logschema.FieldDurationMS, durationMS),
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
		return
	}

	slog.Info("job done",
		slog.String(logschema.FieldEvent, string(logschema.EventJobDone)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
		slog.Int64(logschema.FieldDurationMS, durationMS),
		slog.Int(logschema.FieldTokensGenerated, resp.TokensGenerated),
		slog.String(logschema.FieldModelTier, resp.ModelTier),
		slog.Int("tool_calls", len(toolCalls)),
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

	if resp.QualityDegraded {
		w.Header().Set(api.HeaderQualityDegraded, "true")
	}
	writeJSON(w, http.StatusOK, api.InferResponse{
		RequestID:       backendReq.RequestID,
		CorrelationID:   req.CorrelationID,
		Modality:        req.Modality,
		Output:          resp.Output,
		Reasoning:       resp.Reasoning,
		TokensGenerated: resp.TokensGenerated,
		TokensPerSec:    resp.TokPerSecSample,
		QueueWaitMS:     queueWaitMS,
		InferenceMS:     inferenceMS,
		DurationMS:      durationMS,
		ModelTier:       resp.ModelTier,
		FinishedAt:      finishedAt,
		ToolCalls:       toolCalls,
		Truncated:       resp.LengthLimited,
	})
}

// runWithTools calls b.Infer with req.Tools attached. If the response carries
// no tool calls, it is returned unchanged (the common case — most prompts
// don't trigger a tool). If it carries tool calls, each is resolved via
// executeToolCall, a "tool" role message with the result (or an error string)
// is appended per call, and b.Infer is called exactly once more — with Tools
// cleared, bounding the loop to maxToolHops round-trips — to produce the
// final grounded answer.
func (s *Server) runWithTools(ctx context.Context, b backend.Backend, req backend.Request) (backend.Response, []api.ToolCallSummary, error) {
	resp, err := b.Infer(ctx, req)
	if err != nil {
		return backend.Response{}, nil, err
	}
	if len(resp.ToolCalls) == 0 {
		return resp, nil, nil
	}

	summaries := make([]api.ToolCallSummary, 0, len(resp.ToolCalls))
	messages := append([]backend.Message{}, req.Messages...)
	messages = append(messages, backend.Message{Role: "assistant", ToolCalls: resp.ToolCalls})

	for _, call := range resp.ToolCalls {
		toolMsg, summary := s.executeToolCall(ctx, call)
		messages = append(messages, toolMsg)
		summaries = append(summaries, summary)
	}

	nextReq := req
	nextReq.Messages = messages
	nextReq.Tools = nil // bound to maxToolHops: no further tool offers on the follow-up call

	final, err := b.Infer(ctx, nextReq)
	if err != nil {
		return backend.Response{}, summaries, err
	}
	return final, summaries, nil
}

// runWithToolsStream is the streaming sibling of runWithTools, same
// maxToolHops bound. Passes the SAME chunkFn to both InferStream calls,
// unmodified: verified live against the real model that a tool-calling turn
// streams its reasoning_content live and never emits content, while
// llamacpp.chatCompleteStream already withholds tool-call JSON fragments
// from reaching chunkFn at all (see streamToolCallAccumulator) — so there is
// nothing to buffer or discard. If the first hop's Response.ToolCalls is
// empty, the turn was a plain answer whose content already streamed live via
// that same call, so it is returned unchanged, exactly like runWithTools.
// onToolCall, if non-nil, is invoked once per resolved call, after
// executeToolCall returns but before the second InferStream call begins —
// the HTTP layer uses this to emit a StreamChunk.ToolCall notification
// ahead of the second hop's content so a client can show a "searching the
// web…" state before real tokens resume.
func (s *Server) runWithToolsStream(ctx context.Context, streamer backend.Streamer, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string), onToolCall func(api.ToolCallSummary)) (backend.Response, []api.ToolCallSummary, error) {
	resp, err := streamer.InferStream(ctx, req, chunkFn)
	if err != nil {
		return backend.Response{}, nil, err
	}
	if len(resp.ToolCalls) == 0 {
		return resp, nil, nil
	}

	summaries := make([]api.ToolCallSummary, 0, len(resp.ToolCalls))
	messages := append([]backend.Message{}, req.Messages...)
	messages = append(messages, backend.Message{Role: "assistant", ToolCalls: resp.ToolCalls})

	for _, call := range resp.ToolCalls {
		toolMsg, summary := s.executeToolCall(ctx, call)
		messages = append(messages, toolMsg)
		summaries = append(summaries, summary)
		if onToolCall != nil {
			onToolCall(summary)
		}
	}

	nextReq := req
	nextReq.Messages = messages
	nextReq.Tools = nil // bound to maxToolHops: no further tool offers on the follow-up call

	final, err := streamer.InferStream(ctx, nextReq, chunkFn)
	if err != nil {
		return backend.Response{}, summaries, err
	}
	return final, summaries, nil
}

// executeToolCall dispatches one model-requested tool call by name. Only
// "web_search" is registered; any other name (a model/config mismatch, since
// the model can only call tools it was offered) gets a synthesized error
// result rather than failing the request — the model gets a chance to say it
// couldn't complete the action.
func (s *Server) executeToolCall(ctx context.Context, call backend.ToolCall) (backend.Message, api.ToolCallSummary) {
	summary := api.ToolCallSummary{Name: call.Name, Arguments: call.Arguments}

	if call.Name != "web_search" {
		summary.Error = "tool not available"
		return backend.Message{
			Role:       "tool",
			Content:    searchFailureContent("tool not available"),
			ToolCallID: call.ID,
			Name:       call.Name,
		}, summary
	}

	if s.toolSearcher == nil {
		summary.Error = "search backend not configured"
		return backend.Message{
			Role:       "tool",
			Content:    searchFailureContent("search is not available right now"),
			ToolCallID: call.ID,
			Name:       call.Name,
		}, summary
	}

	var args struct {
		Query string `json:"query"`
	}
	_ = json.Unmarshal([]byte(call.Arguments), &args)

	results, err := s.toolSearcher.Search(ctx, args.Query)
	if err != nil {
		var be *backend.BackendError
		msg := "search failed"
		if errors.As(err, &be) {
			msg = be.Message
		}
		summary.Error = msg
		return backend.Message{
			Role:       "tool",
			Content:    searchFailureContent(msg),
			ToolCallID: call.ID,
			Name:       call.Name,
		}, summary
	}

	content, _ := json.Marshal(map[string]any{"results": results})
	return backend.Message{
		Role:       "tool",
		Content:    string(content),
		ToolCallID: call.ID,
		Name:       call.Name,
	}, summary
}

// searchFailureContent builds the role:"tool" content for a web_search call
// that could not be completed, for any reason (unknown tool, no searcher
// configured, or the search itself erroring/rate-limiting). Beyond the raw
// error, it explicitly instructs the model to frame its answer as coming
// from training data rather than current information — the model asked to
// search specifically because it judged the question needed up-to-date
// facts, so silently falling back to an unqualified answer would misstate
// how current that answer actually is. This instruction only ever reaches a
// conversation that already contains a web_search call — a request with no
// tools, or one whose tools go unused, is completely unaffected.
func searchFailureContent(errMsg string) string {
	data, _ := json.Marshal(struct {
		Error       string `json:"error"`
		Instruction string `json:"instruction"`
	}{
		Error:       errMsg,
		Instruction: "The web search failed, so you have no current information for this query. Answer using only what you already know, and say so explicitly — e.g. start with \"Based on my last update...\" or \"I wasn't able to search, but as of my training...\" — rather than presenting your answer as current or search-verified.",
	})
	return string(data)
}
