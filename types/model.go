package types

// ModelTierLabel is a human-readable tier identifier, ordered small → large.
type ModelTierLabel string

const (
	TierWeak   ModelTierLabel = "weak"
	TierMid    ModelTierLabel = "mid"
	TierStrong ModelTierLabel = "strong"
)

// ModelDescriptor describes one entry in the tiered model roster.
type ModelDescriptor struct {
	TierLabel ModelTierLabel `json:"tier_label"`
	Name      string         `json:"name"`
	FilePath  string         `json:"file_path"`
	// MMProjPath is the vision projector (mmproj) GGUF for a multimodal model.
	// When non-empty it is passed to llama-server as --mmproj, enabling image
	// input on that subprocess. Empty for text-only models.
	MMProjPath string `json:"mmproj_path,omitempty"`
	// RequiredVRAMMB is the expected VRAM footprint for this model in megabytes.
	RequiredVRAMMB int64  `json:"required_vram_mb"`
	Modality       string `json:"modality"`
	// GPULayers is the number of transformer layers to offload to GPU via
	// llama-server --n-gpu-layers. -1 means offload all layers (full GPU).
	// 0 means CPU-only. Normally computed dynamically at load time by
	// vram.ComputeGPULayers; may be set explicitly in the roster.
	GPULayers int `json:"gpu_layers"`
	// Port is the localhost port for this tier's llama-server subprocess.
	// Assign distinct ports per tier (e.g. 8090/8091/8092) to avoid collisions.
	// Falls back to 8090 if zero.
	Port int `json:"port"`
	// TotalLayers is the model's transformer block count. It converts an
	// --n-gpu-layers count into a GPU-resident weight fraction for the
	// partial-offload fit test. 0 falls back to vram.defaultTotalLayers.
	TotalLayers int `json:"total_layers"`
	// KVOverheadMB is the measured non-weight VRAM cost at --ctx-size 4096
	// with a single context (KV cache + compute buffers + CUDA context), read
	// from llama-server startup logs. Retained as a single-context shorthand:
	// when KVCacheMB/KVFixedMB are both zero, callers treat the whole value as
	// per-slot KV cache (see kvParts).
	KVOverheadMB int64 `json:"kv_overhead_mb"`
	// KVCacheMB is the per-slot KV-cache VRAM cost at --ctx-size 4096. It scales
	// linearly with llama-server's --parallel N (N concurrent contexts each hold
	// their own KV cache). Read from llama-server startup logs.
	KVCacheMB int64 `json:"kv_cache_mb"`
	// KVFixedMB is the non-weight VRAM cost that does NOT scale with --parallel:
	// compute buffers plus the CUDA context. Read from llama-server startup logs.
	KVFixedMB int64 `json:"kv_fixed_mb"`

	// VisionRequiredVRAMMB, VisionKVCacheMB and VisionKVFixedMB are this
	// descriptor's counterparts for a load spawned WITH --mmproj (an image
	// present in the request). They are independent of the text-mode fields
	// above — for most models the weight is identical either way and only the
	// KV/fixed terms grow (the mmproj's own GPU-resident cost lives in
	// VisionKVFixedMB), but keeping them separate means a re-quantized vision
	// variant can differ in weight size too. Zero/unset alongside an empty
	// MMProjPath for a text-only tier.
	VisionRequiredVRAMMB int64 `json:"vision_required_vram_mb,omitempty"`
	VisionKVCacheMB      int64 `json:"vision_kv_cache_mb,omitempty"`
	VisionKVFixedMB      int64 `json:"vision_kv_fixed_mb,omitempty"`

	// RequireChatTemplate forces every request to this tier through
	// llama-server's /v1/chat/completions endpoint (the model's own
	// GGUF-embedded jinja chat template applied), even a plain
	// text_input.prompt request with no Messages that would otherwise take
	// the templateless /completion endpoint. Added for LFM2-VL-3B (Phase
	// 13f): a raw prompt-only completion on this model produced noticeably
	// degraded output — repetition loops, or a quiz-continuation style
	// ("A) Paris B) London...") instead of a direct instruct-style answer —
	// while the exact same prompt through the chat-template path answered
	// correctly and concisely every time. Some instruct-tuned models depend
	// on their chat template's special tokens/framing much more than others;
	// this flag lets a roster entry opt into "always template" without
	// changing behaviour for tiers that don't need it (Gemma-4 mid/strong
	// tolerate raw completions fine and are unaffected).
	RequireChatTemplate bool `json:"require_chat_template,omitempty"`
}

// HasVision reports whether this tier has a vision-capable (--mmproj) variant.
func (d ModelDescriptor) HasVision() bool {
	return d.MMProjPath != ""
}

// KVParts returns the descriptor's per-slot KV-cache cost and its fixed
// (concurrency-independent) overhead. When KVCacheMB/KVFixedMB are unset it
// falls back to treating KVOverheadMB entirely as per-slot KV cache, which
// over-budgets slightly under --parallel N but never under-budgets.
func (d ModelDescriptor) KVParts() (cacheMB, fixedMB int64) {
	if d.KVCacheMB > 0 || d.KVFixedMB > 0 {
		return d.KVCacheMB, d.KVFixedMB
	}
	return d.KVOverheadMB, 0
}

// VisionKVParts is KVParts' counterpart for a load spawned WITH --mmproj.
func (d ModelDescriptor) VisionKVParts() (cacheMB, fixedMB int64) {
	return d.VisionKVCacheMB, d.VisionKVFixedMB
}

// WeightMB returns the GPU-resident weight footprint for the given mode:
// VisionRequiredVRAMMB when needsVision and this tier has a vision variant,
// RequiredVRAMMB otherwise.
func (d ModelDescriptor) WeightMB(needsVision bool) int64 {
	if needsVision && d.HasVision() {
		return d.VisionRequiredVRAMMB
	}
	return d.RequiredVRAMMB
}
