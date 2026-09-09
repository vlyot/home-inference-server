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
