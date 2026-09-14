package api

// Modality identifies which inference backend handles a request.
type Modality string

const (
	ModalityText   Modality = "text"
	ModalityVision Modality = "vision"
)

const (
	HeaderAPIKey        = "X-API-Key"
	HeaderAuthorization = "Authorization"
	HeaderCorrelationID = "X-Correlation-ID"
	HeaderRequestID     = "X-Request-ID"
)

const (
	PathInfer          = "/v1/infer"
	PathStatus         = "/v1/status"
	PathStatusPressure = "/v1/status/pressure"
	PathHealth         = "/healthz"
	PathChats          = "/v1/chats"
	PathTokenize       = "/v1/tokenize"
	PathModelProps     = "/v1/model/props"
	PathLogs           = "/v1/logs"
	// PathDescribe runs only the vision perception hop (SmolVLM2): image in,
	// literal text description out. Used by the chat UI to fold an image into a
	// text conversation without a full vision inference per follow-up.
	PathDescribe = "/v1/describe"
	// Admin control surface (loopback-only). PathAdmin alone is a GET state probe.
	PathAdmin        = "/v1/admin"
	PathAdminDrain   = "/v1/admin/drain"
	PathAdminResume  = "/v1/admin/resume"
	PathAdminRestart = "/v1/admin/restart"
)

// Server lifecycle states reported on /healthz and /v1/status.
const (
	StateOK       = "ok"
	StateDraining = "draining"
)

const (
	ErrCodeInvalidRequest  = "invalid_request"
	ErrCodeInvalidModality = "invalid_modality"
	ErrCodeUnauthorized    = "unauthorized"
	ErrCodeOverloaded      = "overloaded"
	ErrCodeModelLoadFailed = "model_load_failed"
	ErrCodeInternal        = "internal_error"
	ErrCodeUnavailable     = "unavailable"
	ErrCodeTimeout         = "timeout"
	ErrCodeRateLimited     = "rate_limited"
	ErrCodeQuotaExceeded   = "quota_exceeded"
	ErrCodeQueueFull       = "queue_full"
	// ErrCodeReasoningExhausted marks a stream that spent its entire token
	// budget on a reasoning-capable model's "thinking" phase (see
	// StreamChunk.Reasoning) and never produced a visible answer. Distinct
	// from ErrCodeInternal so a client can tell "nothing happened" apart from
	// "the model reasoned but ran out of budget before answering".
	ErrCodeReasoningExhausted = "reasoning_exhausted"
	// ErrCodeNotImplemented marks a documented capability that is not wired up
	// in this build (currently: vision inference — see the model roster).
	ErrCodeNotImplemented = "not_implemented"
	// ErrCodeNotFound marks a lookup miss distinct from a bad request: a result
	// id that is unknown, belongs to another caller, or has aged past its TTL.
	ErrCodeNotFound = "not_found"
	// ErrCodeDraining is returned for a new inference request while the server is
	// draining for maintenance (POST /v1/admin/drain). Read-only endpoints keep
	// working; retry after /v1/admin/resume or a restart.
	ErrCodeDraining = "draining"
	// ErrCodeInvalidGrammar marks a request whose response_format could not be
	// compiled to a sampling grammar by llama-server (malformed JSON Schema,
	// unsupported construct). Returned as HTTP 400 — the caller must fix the
	// schema.
	ErrCodeInvalidGrammar = "invalid_grammar"
	// ErrCodeToolUnavailable marks a tool call that could not be resolved
	// because its backend (e.g. the Tavily search API) was unreachable or
	// returned an unexpected error. The request itself does not fail: the
	// model receives an error string as the tool result and gets a chance to
	// say so in its answer.
	ErrCodeToolUnavailable = "tool_unavailable"
	// ErrCodeToolNotSupported marks a request whose modality/tier has no
	// tool-calling chat template (currently: any request with vision_input) —
	// returned as HTTP 400, the caller must drop tools or the image.
	ErrCodeToolNotSupported = "tool_not_supported"
	// ErrCodeToolRateLimited marks a tool call whose backend responded with a
	// rate-limit error (e.g. Tavily's keyless access cap — HTTP 429). Distinct
	// from ErrCodeToolUnavailable because the remedy differs: wait, or add a
	// free API key, rather than "the backend is down". Like the other tool_*
	// codes, this never fails the request — it appears only inside a resolved
	// tool_calls[].error field.
	ErrCodeToolRateLimited = "tool_rate_limited"
)

const (
	HeaderQualityDegraded = "X-Quality-Degraded"
)

const (
	PriorityHigh   = "high"
	PriorityNormal = "normal"
	PriorityLow    = "low"
)

const (
	MinTierStrong = "strong"
	MinTierMid    = "mid"
	MinTierWeak   = "weak"
)
