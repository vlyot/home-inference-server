package api

import "time"

// InferResponse is the successful response body from POST /v1/infer.
// Not used for streaming responses — see StreamChunk for that.
type InferResponse struct {
	RequestID     string   `json:"request_id"`
	CorrelationID string   `json:"correlation_id"`
	Modality      Modality `json:"modality"`
	Output        string   `json:"output"`
	// Reasoning is a reasoning-capable model's (e.g. Gemma's "thinking" mode)
	// chain-of-thought text, kept separate from Output. Empty when the model
	// produced none.
	Reasoning       string `json:"reasoning,omitempty"`
	TokensGenerated int    `json:"tokens_generated"`
	// TokensPerSec is the observed generation speed for this job.
	TokensPerSec float64 `json:"tokens_per_sec"`
	// QueueWaitMS is time spent in the queue before inference started.
	QueueWaitMS int64 `json:"queue_wait_ms"`
	// InferenceMS is time spent in active inference (total - queue wait).
	InferenceMS int64 `json:"inference_ms"`
	// DurationMS is total wall-clock time from enqueue to result ready.
	DurationMS int64 `json:"duration_ms"`
	// ModelTier is the tier that actually handled this job (may differ from
	// requested tier due to fallback).
	ModelTier  string    `json:"model_tier"`
	FinishedAt time.Time `json:"finished_at"`
	// Deferred is true if this request was held in the local defer queue
	// before being served.
	Deferred bool `json:"deferred,omitempty"`
}

// StreamChunk is one Server-Sent Event payload for streaming responses.
// The final chunk has Done=true and carries the summary fields.
type StreamChunk struct {
	RequestID     string `json:"request_id"`
	CorrelationID string `json:"correlation_id"`
	// Delta is the partial visible-answer text for this chunk.
	Delta string `json:"delta,omitempty"`
	// Reasoning is a reasoning-capable model's (e.g. Gemma's "thinking" mode)
	// partial chain-of-thought text for this chunk. Mutually exclusive with
	// Delta on any given chunk — never both set. Intended to be rendered
	// separately (e.g. a collapsible "Thinking…" section), not appended to
	// the visible answer.
	Reasoning string `json:"reasoning,omitempty"`
	// Done signals the end of the stream. When true, summary fields are populated.
	Done            bool    `json:"done"`
	TokensGenerated int     `json:"tokens_generated,omitempty"`
	TokensPerSec    float64 `json:"tokens_per_sec,omitempty"`
	QueueWaitMS     int64   `json:"queue_wait_ms,omitempty"`
	InferenceMS     int64   `json:"inference_ms,omitempty"`
	DurationMS      int64   `json:"duration_ms,omitempty"`
	ModelTier       string  `json:"model_tier,omitempty"`
	Deferred        bool    `json:"deferred,omitempty"`
	// Error is set on the final chunk when inference failed. It carries one of
	// the ErrCode* tokens; Delta and the summary fields are empty in that case.
	Error string `json:"error,omitempty"`
	// Truncated is set on the final chunk when the stream ended early because
	// the backend failed AFTER some real content was already delivered (e.g.
	// the model subprocess crashed mid-generation). Distinct from Error: this
	// is not a failed request — every prior Delta chunk is genuine model
	// output and should be kept, just incomplete. Error and Truncated are
	// mutually exclusive; a client should treat Truncated the same as a
	// normal Done (persist what streamed) while surfacing that it was cut
	// short, rather than discarding the turn as it would on Error.
	Truncated bool `json:"truncated,omitempty"`
}

// PressureSnapshot is the response body for GET /v1/status/pressure.
// It provides a lightweight, real-time view of system load without job history.
type PressureSnapshot struct {
	Timestamp   time.Time `json:"timestamp"`
	VRAMAvailMB int64     `json:"vram_avail_mb"`
	// SelfVRAMMB is VRAM held by this server's own subprocesses; OtherVRAMMB is
	// total VRAM minus free minus self (what other applications are using).
	SelfVRAMMB  int64   `json:"self_vram_mb"`
	OtherVRAMMB int64   `json:"other_vram_mb"`
	CPUPct      float64 `json:"cpu_pct"`
	TokPerSec   float64 `json:"tok_per_sec"`
	ActiveTier  string  `json:"active_tier"`
	// InFlight is the number of requests currently executing inference;
	// MaxParallel is the concurrent-inference ceiling (llama-server --parallel N).
	InFlight    int `json:"in_flight"`
	MaxParallel int `json:"max_parallel"`
}

// HealthResponse is the body returned by GET /healthz.
type HealthResponse struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	UptimeS int64  `json:"uptime_s"`
	// State is "ok" normally or "draining" after POST /v1/admin/drain. /healthz
	// stays 200 either way (liveness); this field is the readiness signal.
	State string `json:"state"`
}

// AdminStateResponse is the body returned by the /v1/admin/* control endpoints.
type AdminStateResponse struct {
	State      string `json:"state"` // "ok" | "draining" | "restarting"
	InFlight   int    `json:"in_flight"`
	QueueDepth int    `json:"queue_depth"`
}

// Conversation is a stored chat conversation, returned by GET /v1/chats/{id}
// and accepted (partially) by PUT /v1/chats/{id}.
type Conversation struct {
	ID        string        `json:"id"`
	Title     string        `json:"title"`
	Messages  []ChatMessage `json:"messages"`
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
}

// ConversationMeta is the list-view summary returned by GET /v1/chats.
type ConversationMeta struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	TurnCount int       `json:"turn_count"`
}

// TokenizeRequest is the body for POST /v1/tokenize.
type TokenizeRequest struct {
	Text string `json:"text"`
}

// TokenizeResponse is the body returned by POST /v1/tokenize. When no model is
// loaded, Tokens is a whitespace-word approximation and NCtx is the configured
// context size (see ModelLoaded).
type TokenizeResponse struct {
	Tokens int `json:"tokens"`
	NCtx   int `json:"n_ctx"`
	// ModelLoaded is false when the counts are a cheap estimate served without
	// starting a model. Exact once a model is resident.
	ModelLoaded bool `json:"model_loaded"`
}

// ModelPropsResponse is the body returned by GET /v1/model/props.
type ModelPropsResponse struct {
	NCtx int `json:"n_ctx"`
	// ModelLoaded is false when NCtx is the configured default served without
	// starting a model.
	ModelLoaded bool `json:"model_loaded"`
}

// DescribeResponse is the body returned by POST /v1/describe.
type DescribeResponse struct {
	// Description is the vision model's literal inventory of the image.
	Description string `json:"description"`
	// ModelTier is the tier that produced the description (e.g. "weak").
	ModelTier string `json:"model_tier"`
}

// ErrorResponse is the body returned on any 4xx or 5xx response.
type ErrorResponse struct {
	// Code is a machine-readable error token matching one of the ErrCode* constants.
	Code          string `json:"code"`
	Message       string `json:"message"`
	CorrelationID string `json:"correlation_id,omitempty"`
}
