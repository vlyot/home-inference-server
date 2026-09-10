package backend

import (
	"context"
	"encoding/json"
)

// ModalityKind identifies which type of inference a Backend handles.
type ModalityKind string

const (
	ModalityKindText   ModalityKind = "text"
	ModalityKindVision ModalityKind = "vision"
)

// Message is a single conversation turn passed through to the model's
// chat-completions endpoint.
type Message struct {
	Role    string // "system" | "user" | "assistant"
	Content string
}

// Request is the backend-internal inference request. The router translates
// api.InferRequest into this type before dispatching, keeping api and backend
// packages free of each other's concerns.
type Request struct {
	CorrelationID string
	RequestID     string
	Modality      ModalityKind
	// Prompt is the combined prompt string.
	// For text: from TextInput.Prompt.
	// For vision: from VisionInput.Prompt.
	Prompt string
	// Messages, when non-empty, is the full conversation to send to the model's
	// chat-completions endpoint (chat template applied by llama-server). Takes
	// precedence over Prompt; Prompt remains the path for single-shot text
	// completions and vision.
	Messages []Message
	// ImageData is the decoded image bytes for vision requests. Nil for text.
	ImageData   []byte
	MaxTokens   int
	Temperature float32
	// Priority is the caller-supplied urgency ("high", "normal", "low").
	// Empty string is treated as "normal".
	Priority string
	// MinTier is the minimum acceptable model quality floor ("strong", "mid", "weak").
	// Empty string is treated as "weak" (no floor enforced).
	MinTier string
	// PreferredTier, when set, pins tier selection to exactly this tier
	// ("strong", "mid", or "weak") instead of the default "largest tier that
	// fits in free VRAM" — e.g. deliberately running the small model to leave
	// VRAM headroom for other GPU work. Unlike MinTier (a floor, never
	// upgraded away from), this is an exact pin: the speed-floor cascade does
	// not override it. If the pinned tier's inner backend isn't registered,
	// the request errors rather than silently falling back to another tier.
	PreferredTier string
	// Stream requests a streaming SSE response instead of a single blocking JSON body.
	Stream bool
	// ResponseFormat, when set, is forwarded verbatim as the chat-completions
	// "response_format" so llama-server constrains sampling to the schema. Only
	// the Messages path honours it. Nil = unconstrained.
	ResponseFormat json.RawMessage
}

// Response is what a Backend returns after inference completes.
type Response struct {
	Output          string
	TokensGenerated int
	// Reasoning holds a reasoning-capable model's chain-of-thought text (e.g.
	// Gemma's "thinking" mode), when the model emitted any. Empty for models
	// or requests that produced none. Kept separate from Output so callers can
	// render it distinctly rather than mixing it into the visible answer.
	Reasoning string
	// ModelTier is the tier label of the model that ran this job.
	ModelTier string
	// QualityDegraded is true when a high-priority request was served by a
	// weaker tier than requested (e.g. strong requested but only weak available).
	// The HTTP layer sets X-Quality-Degraded: true when this is set.
	QualityDegraded bool
	// TokPerSecSample is the tokens/sec measured for this single completion.
	// Set by llamacpp.Backend from llama-server's timings block; 0 if unavailable.
	// Used to update the rolling average on the backend — not forwarded to callers.
	TokPerSecSample float64
}

// ChunkKind identifies what a streamed delta represents.
type ChunkKind int

const (
	// ChunkContent is a normal, user-visible answer token.
	ChunkContent ChunkKind = iota
	// ChunkReasoning is a reasoning-model "thinking" token (e.g. Gemma's
	// reasoning_content), meant to be rendered separately from the answer.
	ChunkReasoning
)

// Measurable is an optional interface that inner backends may implement to
// expose their rolling tokens-per-second rate. vram.Backend uses this to
// enforce priority speed floors without coupling to the llamacpp package.
type Measurable interface {
	TokPerSec() float64
}

// Streamer is an optional interface for backends that support SSE streaming.
// The server checks for this interface when req.Stream is true; if absent,
// the streaming request falls back to non-streaming.
type Streamer interface {
	// InferStream runs inference and calls chunkFn for each token delta as
	// it is produced, tagged with its ChunkKind. Returns the final summary
	// (tokens, timing) when done.
	InferStream(ctx context.Context, req Request, chunkFn func(kind ChunkKind, delta string)) (Response, error)
}

// Backend is the interface every modality implementation must satisfy.
// It is safe for concurrent calls from multiple goroutines.
//
// Responsibilities:
//   - Check VRAM headroom and select a model tier
//   - Load the model if not already loaded
//   - Run inference (possibly batched internally)
//   - Return the result or a BackendError
type Backend interface {
	// Modality returns the ModalityKind this backend handles.
	// Used by the router to build its dispatch table.
	Modality() ModalityKind

	// Infer runs a single inference request. The context carries the
	// deadline/cancellation signal from the HTTP handler.
	// Errors are BackendError so the HTTP layer can set the correct status
	// code without knowing backend internals.
	Infer(ctx context.Context, req Request) (Response, error)

	// Ready returns true if the backend can currently accept requests.
	Ready() bool

	// Shutdown signals the backend to finish in-flight requests and release
	// resources. Blocks until clean.
	Shutdown(ctx context.Context) error
}

// BackendError is returned by Backend.Infer to carry a machine-readable code
// alongside the error message. The HTTP layer uses errors.As to extract the
// code and map it to an api.ErrorResponse without a type switch.
type BackendError struct {
	// Code matches one of the api.ErrCode* constants.
	Code    string
	Message string
	Cause   error
}

func (e *BackendError) Error() string {
	if e.Cause != nil {
		return e.Message + ": " + e.Cause.Error()
	}
	return e.Message
}

func (e *BackendError) Unwrap() error { return e.Cause }
