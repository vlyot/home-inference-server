package api

// InferRequest is the inbound body for POST /v1/infer.
// Exactly one of TextInput or VisionInput must be non-nil, matching Modality.
type InferRequest struct {
	// CorrelationID is set by the caller and propagated through every log event.
	// If empty, the server generates one (format: "req-" + uuid4 without hyphens).
	CorrelationID string       `json:"correlation_id,omitempty"`
	Modality      Modality     `json:"modality"`
	TextInput     *TextInput   `json:"text_input,omitempty"`
	VisionInput   *VisionInput `json:"vision_input,omitempty"`
	// MaxTokens caps output length. 0 means server default.
	MaxTokens int `json:"max_tokens,omitempty"`
	// Temperature controls sampling randomness. 0.0 uses server default.
	Temperature float32 `json:"temperature,omitempty"`
	// SystemPrompt is a convenience field — equivalent to prepending a
	// role=system ChatMessage. Ignored if TextInput.Messages already contains
	// a system entry.
	SystemPrompt string `json:"system_prompt,omitempty"`
	// Stream requests Server-Sent Events delivery instead of a single JSON
	// response body. Each event is a partial token chunk; a final [DONE] event
	// signals completion.
	Stream bool `json:"stream,omitempty"`
	// TimeoutMS is a caller-specified deadline in milliseconds. If inference
	// does not complete within this window the server cancels and returns
	// ErrCodeTimeout. 0 means the server's default timeout applies.
	TimeoutMS int `json:"timeout_ms,omitempty"`
	// Priority controls urgency and drain order from the defer queue.
	// Values: "high" | "normal" | "low". Default: "normal".
	Priority string `json:"priority,omitempty"`
	// MinTier sets the quality floor. If the tier the router would serve on
	// is below MinTier, the request is deferred instead of served degraded.
	// Values: "strong" | "mid" | "weak". Default: "weak".
	MinTier string `json:"min_tier,omitempty"`
	// PreferredTier, when set, pins the request to exactly this model tier
	// instead of the default "largest tier that fits in free VRAM" — e.g. to
	// deliberately run the small/fast model and leave VRAM headroom for other
	// GPU work. Values: "strong" | "mid" | "weak". Empty means automatic
	// selection (default). Unlike min_tier, this is an exact pin, not a floor.
	PreferredTier string `json:"preferred_tier,omitempty"`
}

// TextInput carries a prompt for completion or a message list for chat.
// Exactly one of Prompt or Messages must be set.
type TextInput struct {
	Prompt   string        `json:"prompt,omitempty"`
	Messages []ChatMessage `json:"messages,omitempty"`
}

// ChatMessage is a single turn in a conversation.
type ChatMessage struct {
	// Role is "system", "user", or "assistant".
	Role    string `json:"role"`
	Content string `json:"content"`
}

// VisionInput carries an image plus an associated text prompt.
// Exactly one of ImageBase64 or ImageURL must be set.
type VisionInput struct {
	ImageBase64 string `json:"image_base64,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	Prompt      string `json:"prompt"`
}
