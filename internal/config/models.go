package config

import (
	"fmt"
	"path/filepath"
	"sort"

	"github.com/ngkaichong/home-inference-server/types"
)

// BuiltinModels is the tiered model roster used when config.json lists no
// models. FilePath/MMProjPath are bare filenames, joined to models_dir by
// Roster. RequiredVRAMMB is the GGUF weight size; GPULayers is computed
// dynamically at load time. TotalLayers, KVCacheMB and KVFixedMB are read from
// llama-server startup logs (--ctx-size 4096) and feed the partial-offload fit
// test in vram.selectAndLoad. The GPU-resident footprint of a full-offload load
// is RequiredVRAMMB + KVCacheMB*N + KVFixedMB (N = max_parallel); the eviction
// loop and selectAndLoad both size pressure against that sum via GPUResidentMB,
// never against RequiredVRAMMB alone. KVCacheMB (per --parallel slot) vs
// KVFixedMB (compute buffers + CUDA context, constant) split from the old single
// KVOverheadMB — the cache portion is the ctx-proportional part, roughly two
// thirds. Tighten from slog.Debug llama-server startup lines on the GPU.
//
// Every tier is a text.Backend; a tier whose MMProjPath is set (currently only
// weak) ALSO accepts modality: "vision" requests — vram.Backend restricts
// vision selection to HasVision() tiers and spawns that tier's subprocess with
// --mmproj only for the request that actually needs it (see
// types.ModelDescriptor.WeightMB / VisionKVParts and
// llamacpp.Backend.SetNextNeedsVision). There is no separate vision-only
// roster or backend.
func BuiltinModels() []types.ModelDescriptor {
	return []types.ModelDescriptor{
		{
			TierLabel: types.TierWeak,
			Name:      "lfm2-vl-3b-instruct",
			FilePath:  "lfm2-vl-3b-instruct-q4_k_m.gguf",
			// MMProjPath makes this tier vision-capable. It is passed to
			// llama-server ONLY on a load that a vision request triggers
			// (llamacpp.Backend.ensureRunning consults SetNextNeedsVision) — a
			// plain text load never touches --mmproj or the VRAM it costs. See
			// HasVision(), WeightMB(), VisionKVParts().
			//
			// Swapped from qwen2.5-vl-3b-instruct (Phase 13f): Qwen2.5-VL-3B
			// reliably hallucinated a different confident-but-wrong identification
			// on the same test image across repeated runs once prompted to commit
			// to identifications ("Coca-Cola", "Amazon Alexa logo", "Raspberry Pi"
			// — never the same answer twice). LFM2-VL-3B (Liquid AI, SigLIP2 NaFlex
			// encoder with real image-token reduction) passed the same repeated
			// test cleanly: consistent, calibrated answers, no hallucinated brand
			// names, and a real quality edge over its own 1.6B sibling (MMMU 48.4
			// vs 40.56, Liquid's published numbers) while measuring LOWER resident
			// VRAM than the Qwen2.5-VL-3B setup it replaces. Two other candidate
			// families were tested and rejected on real hardware first — see
			// roadmap.md: Qwen3-VL (2B and 4B both measured ~6 GB resident
			// regardless of text-model size, the vision encoder's fixed cost
			// dominates, not the text backbone) and Gemma-4-E2B-vision (burns its
			// entire token budget in reasoning mode, then falsely claims no image
			// was provided).
			MMProjPath:     "mmproj-lfm2-vl-3b-instruct-q8_0.gguf",
			RequiredVRAMMB: 1568, // Q4_K_M text weights, measured file size (1567.7 MB)
			Modality:       "text",
			TotalLayers:    30, // lfm2.block_count from the GGUF metadata
			KVCacheMB:      40,
			KVFixedMB:      182,
			// Vision-mode constants, measured on the real GPU (RTX 4060 Ti 8 GB):
			// text-only load (no --mmproj) resident ~1830 MB at --parallel 2,
			// ~1939 MB at --parallel 1; vision-mode load (with --mmproj) resident
			// ~2450 MB at --parallel 2, ~2630 MB at --parallel 1. The per-slot
			// swing between parallel=1 and parallel=2 was smaller than the ~100 MB
			// run-to-run measurement noise floor (LFM2's hybrid conv+attention
			// architecture appears to carry very little KV-cache pressure), so
			// KVCacheMB/VisionKVCacheMB are a small conservative nonzero floor
			// rather than a value derived from a clean per-slot delta, and
			// VisionKVFixedMB is sized off the HIGHER of the two vision-mode
			// readings (2630, not 2450) to stay on the safe (over-, not under-)
			// booking side per this project's convention. See roadmap.md for the
			// full measurement log.
			VisionRequiredVRAMMB: 1568,
			VisionKVCacheMB:      40,
			VisionKVFixedMB:      982,
			Port:                 8093,
			// RequireChatTemplate: LFM2-VL-3B produced noticeably degraded output
			// on a raw prompt-only /completion request (repetition loops, or a
			// quiz-continuation like "A) Paris B) London..." instead of a direct
			// answer) — the same prompt through the chat-template path answered
			// correctly every time. See types.ModelDescriptor.RequireChatTemplate.
			RequireChatTemplate: true,
		},
		{
			TierLabel:      types.TierMid,
			Name:           "gemma-4-e2b-q4_k_m",
			FilePath:       "gemma-4-e2b-q4_k_m.gguf",
			RequiredVRAMMB: 2963,
			Modality:       "text",
			TotalLayers:    30,
			KVCacheMB:      450,
			KVFixedMB:      250,
			Port:           8091,
		},
		{
			TierLabel:      types.TierStrong,
			Name:           "gemma-4-e4b-q4_k_m",
			FilePath:       "gemma-4-e4b-q4_k_m.gguf",
			RequiredVRAMMB: 4747,
			Modality:       "text",
			TotalLayers:    34,
			KVCacheMB:      580,
			KVFixedMB:      320,
			Port:           8092,
		},
	}
}

// tierOrder ranks the roster slots small → large; vram.Backend expects the
// roster in this order.
var tierOrder = map[types.ModelTierLabel]int{types.TierWeak: 0, types.TierMid: 1, types.TierStrong: 2}

// defaultTierPort is the llama-server port a custom model gets when its config
// entry leaves port unset.
var defaultTierPort = map[types.ModelTierLabel]int{types.TierWeak: 8093, types.TierMid: 8091, types.TierStrong: 8092}

// Roster returns c.Models (or BuiltinModels when none are configured) with
// relative file paths joined to ModelsDir, missing ports filled per tier, and
// entries sorted weak → strong. It rejects entries vram.Backend can't serve.
func (c Config) Roster() ([]types.ModelDescriptor, error) {
	src := c.Models
	if len(src) == 0 {
		src = BuiltinModels()
	}
	out := make([]types.ModelDescriptor, len(src))
	copy(out, src)

	seenTier := map[types.ModelTierLabel]bool{}
	seenPort := map[int]string{}
	for i := range out {
		d := &out[i]
		if _, ok := tierOrder[d.TierLabel]; !ok {
			return nil, fmt.Errorf("model %q: tier_label %q must be weak, mid or strong", d.Name, d.TierLabel)
		}
		if seenTier[d.TierLabel] {
			return nil, fmt.Errorf("two models use tier_label %q; each tier holds one model", d.TierLabel)
		}
		seenTier[d.TierLabel] = true
		if d.Name == "" || d.FilePath == "" {
			return nil, fmt.Errorf("model in tier %q needs both name and file_path", d.TierLabel)
		}
		if d.Port == 0 {
			d.Port = defaultTierPort[d.TierLabel]
		}
		if other, dup := seenPort[d.Port]; dup {
			return nil, fmt.Errorf("models %q and %q both use port %d", other, d.Name, d.Port)
		}
		seenPort[d.Port] = d.Name
		if d.Modality == "" {
			d.Modality = "text"
		}
		d.FilePath = c.modelPath(d.FilePath)
		if d.MMProjPath != "" {
			d.MMProjPath = c.modelPath(d.MMProjPath)
		}
	}
	sort.Slice(out, func(i, j int) bool { return tierOrder[out[i].TierLabel] < tierOrder[out[j].TierLabel] })
	return out, nil
}

func (c Config) modelPath(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(c.ModelsDir, p)
}

// FilterInstalled drops roster entries whose model file doesn't exist, and
// keeps an entry whose vision projector is missing as text-only. notes
// describes each change for the startup log.
func FilterInstalled(roster []types.ModelDescriptor, exists func(string) bool) (kept []types.ModelDescriptor, notes []string) {
	for _, d := range roster {
		if !exists(d.FilePath) {
			notes = append(notes, fmt.Sprintf("skipped %s model %q: %s not found", d.TierLabel, d.Name, d.FilePath))
			continue
		}
		if d.MMProjPath != "" && !exists(d.MMProjPath) {
			notes = append(notes, fmt.Sprintf("%s model %q is text-only: vision projector %s not found", d.TierLabel, d.Name, d.MMProjPath))
			d.MMProjPath = ""
			d.VisionRequiredVRAMMB, d.VisionKVCacheMB, d.VisionKVFixedMB = 0, 0, 0
		}
		kept = append(kept, d)
	}
	return kept, notes
}
