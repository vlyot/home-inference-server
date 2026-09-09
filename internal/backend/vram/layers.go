package vram

import "github.com/ngkaichong/home-inference-server/types"

// defaultTotalLayers is the assumed transformer layer count used for proportional
// GPU offload when exact layer count is unknown. llama-server clips to the real
// maximum, so overestimating is safe.
const defaultTotalLayers = 32

// ScaledKVOverheadMB is the non-weight VRAM a load of d occupies when
// llama-server runs with `slots` concurrent contexts (--parallel N): the
// per-slot KV cache multiplied by the slot count, plus the fixed compute/CUDA
// overhead. slots < 1 is treated as 1. Exported so llamacpp can size its own
// resident-VRAM estimate the same way the fit/eviction math does.
func ScaledKVOverheadMB(d types.ModelDescriptor, slots int) int64 {
	if slots < 1 {
		slots = 1
	}
	cache, fixed := d.KVParts()
	return cache*int64(slots) + fixed
}

// ComputeGPULayers decides how many transformer layers to offload to GPU based
// on available VRAM headroom relative to the model's weight.
//
// Returns:
//   - -1  — offload all layers (full GPU): headroom ≥ modelWeightMB * 1.1
//   - 0   — CPU-only: headroom < modelWeightMB * 0.2
//   - N   — proportional: int(headroom/modelWeightMB * defaultTotalLayers)
func ComputeGPULayers(modelWeightMB, availableVRAMMB int64) int {
	if modelWeightMB <= 0 {
		return -1
	}
	if availableVRAMMB < 0 {
		// NVML unavailable: treat as unlimited, full GPU offload.
		return -1
	}

	fullThreshold := modelWeightMB + modelWeightMB/10 // modelWeightMB * 1.1
	cpuOnlyThreshold := modelWeightMB / 5             // modelWeightMB * 0.2

	if availableVRAMMB >= fullThreshold {
		return -1
	}
	if availableVRAMMB < cpuOnlyThreshold {
		return 0
	}

	ratio := float64(availableVRAMMB) / float64(modelWeightMB)
	layers := int(ratio * float64(defaultTotalLayers))
	if layers < 1 {
		layers = 1
	}
	return layers
}

// FitLayers returns the largest --n-gpu-layers value whose GPU-resident
// footprint (weight fraction + kvOverheadMB) fits within availableVRAMMB after a
// bufferPct% safety margin. kvOverheadMB is the concurrency-scaled non-weight
// cost (see ScaledKVOverheadMB) — pass N* the per-slot KV cache plus fixed
// overhead when llama-server runs with --parallel N.
//
//   - -1  full GPU: every layer fits
//   - 0   CPU-only: not even a single layer's worth of weights fits alongside
//     the overhead (the caller decides whether the overhead alone still fits)
//   - N   the partial count that fills the weights budget
//
// totalLayers <= 0 falls back to defaultTotalLayers.
func FitLayers(weightMB, kvOverheadMB, availableVRAMMB int64, totalLayers, bufferPct int) int {
	if weightMB <= 0 || availableVRAMMB < 0 {
		return -1
	}
	total := totalLayers
	if total <= 0 {
		total = defaultTotalLayers
	}
	// Budget for weights = free VRAM shrunk by the buffer, minus the fixed
	// overhead that always sits on the GPU.
	budget := availableVRAMMB*100/int64(100+bufferPct) - kvOverheadMB
	if budget >= weightMB {
		return -1
	}
	if budget <= 0 {
		return 0
	}
	layers := int(budget * int64(total) / weightMB)
	if layers < 1 {
		return 0
	}
	if layers >= total {
		return -1
	}
	return layers
}

// GPUResidentMB estimates how much VRAM a load with the given GPU/CPU layer
// split actually occupies: the on-GPU fraction of the weights plus the model's
// non-weight KV-cache / compute-buffer overhead. kvOverheadMB is the
// concurrency-scaled overhead (see ScaledKVOverheadMB).
//
//   - gpuLayers < 0  (full GPU) → weightMB + kvOverheadMB
//   - gpuLayers == 0 (CPU only) → kvOverheadMB (weights in system RAM; the GPU
//     still holds a compute/KV buffer)
//   - otherwise → weightMB * gpuLayers/totalLayers + kvOverheadMB
//
// totalLayers <= 0 falls back to defaultTotalLayers.
func GPUResidentMB(weightMB, kvOverheadMB int64, gpuLayers, totalLayers int) int64 {
	total := totalLayers
	if total <= 0 {
		total = defaultTotalLayers
	}
	switch {
	case gpuLayers < 0:
		return weightMB + kvOverheadMB
	case gpuLayers == 0:
		return kvOverheadMB
	default:
		if gpuLayers > total {
			gpuLayers = total
		}
		return weightMB*int64(gpuLayers)/int64(total) + kvOverheadMB
	}
}
