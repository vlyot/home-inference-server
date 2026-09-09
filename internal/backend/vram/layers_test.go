package vram

import (
	"testing"

	"github.com/ngkaichong/home-inference-server/types"
)

func TestComputeGPULayers_FullGPU(t *testing.T) {
	// headroom (8000) >= modelWeight (4000) * 1.1 (4400) → full GPU
	got := ComputeGPULayers(4000, 8000)
	if got != -1 {
		t.Errorf("got %d; want -1 (full GPU)", got)
	}
}

func TestComputeGPULayers_ExactBoundary(t *testing.T) {
	// headroom == modelWeight * 1.1 exactly → full GPU
	// modelWeight=1000, threshold=1100, headroom=1100
	got := ComputeGPULayers(1000, 1100)
	if got != -1 {
		t.Errorf("got %d; want -1 (full GPU at exact boundary)", got)
	}
}

func TestComputeGPULayers_CPUOnly(t *testing.T) {
	// headroom (100) < modelWeight (4000) * 0.2 (800) → CPU only
	got := ComputeGPULayers(4000, 100)
	if got != 0 {
		t.Errorf("got %d; want 0 (CPU-only)", got)
	}
}

func TestComputeGPULayers_Proportional(t *testing.T) {
	// headroom = 50% of modelWeight → ~16 layers (of 32)
	// modelWeight=4000, headroom=2000 → ratio=0.5 → 0.5*32=16
	got := ComputeGPULayers(4000, 2000)
	if got != 16 {
		t.Errorf("got %d; want 16", got)
	}
}

func TestComputeGPULayers_NVMLUnavailable(t *testing.T) {
	// availableVRAMMB=-1 means NVML unavailable → full GPU
	got := ComputeGPULayers(4000, -1)
	if got != -1 {
		t.Errorf("got %d; want -1 (NVML unavailable → full GPU)", got)
	}
}

func TestComputeGPULayers_ZeroModelWeight(t *testing.T) {
	// modelWeightMB=0 → guard against division, return full GPU
	got := ComputeGPULayers(0, 4000)
	if got != -1 {
		t.Errorf("got %d; want -1 (zero model weight)", got)
	}
}

func TestGPUResidentMB_FullGPU(t *testing.T) {
	// gpuLayers -1 → whole weight + overhead on GPU
	got := GPUResidentMB(4000, 900, -1, 34)
	if got != 4900 {
		t.Errorf("got %d; want 4900", got)
	}
}

func TestGPUResidentMB_CPUOnly(t *testing.T) {
	// gpuLayers 0 → only the KV/compute overhead is on GPU
	got := GPUResidentMB(4000, 900, 0, 34)
	if got != 900 {
		t.Errorf("got %d; want 900", got)
	}
}

func TestGPUResidentMB_HalfSplit(t *testing.T) {
	// 17 of 34 layers → half the weight + overhead: 2000 + 900
	got := GPUResidentMB(4000, 900, 17, 34)
	if got != 2900 {
		t.Errorf("got %d; want 2900", got)
	}
}

func TestGPUResidentMB_TotalLayersFallback(t *testing.T) {
	// totalLayers 0 → uses defaultTotalLayers (32): 16/32 of 3200 + 200
	got := GPUResidentMB(3200, 200, 16, 0)
	if got != 1800 {
		t.Errorf("got %d; want 1800 (fallback total=32)", got)
	}
}

func TestGPUResidentMB_ClampsLayersToTotal(t *testing.T) {
	// gpuLayers above totalLayers behaves like full offload of the weights
	got := GPUResidentMB(4000, 900, 99, 34)
	if got != 4900 {
		t.Errorf("got %d; want 4900 (clamped)", got)
	}
}

func TestFitLayers_FullGPU(t *testing.T) {
	// 8000 free, weight 4000 + overhead 900, 10% buffer: whole model fits
	got := FitLayers(4000, 900, 8000, 32, 10)
	if got != -1 {
		t.Errorf("got %d; want -1 (full GPU)", got)
	}
}

func TestFitLayers_PartialFillsWeightsBudget(t *testing.T) {
	// 4000 free, 10% buffer → ~3636 for GPU; minus 900 overhead → 2736 for
	// weights; 2736/4000 * 32 ≈ 21 layers
	got := FitLayers(4000, 900, 4000, 32, 10)
	if got < 18 || got > 24 {
		t.Errorf("got %d; want ~21 partial layers", got)
	}
}

func TestFitLayers_CPUOnlyWhenBudgetBelowOneLayer(t *testing.T) {
	// 1000 free, 10% buffer → ~909; minus 900 overhead → 9 for weights → 0 layers
	got := FitLayers(4000, 900, 1000, 32, 10)
	if got != 0 {
		t.Errorf("got %d; want 0 (CPU-only)", got)
	}
}

func TestFitLayers_CPUOnlyWhenOverheadExceedsBudget(t *testing.T) {
	// free below the overhead itself → 0 (caller then checks if overhead fits)
	got := FitLayers(4000, 900, 500, 32, 10)
	if got != 0 {
		t.Errorf("got %d; want 0", got)
	}
}

func TestFitLayers_TotalLayersFallback(t *testing.T) {
	got := FitLayers(3200, 200, 1800, 0, 0) // budget 1600 for weights; 1600/3200*32 = 16
	if got != 16 {
		t.Errorf("got %d; want 16 (fallback total=32)", got)
	}
}

// --- per-slot KV scaling ---

func TestScaledKVOverheadMB_ScalesCacheNotFixed(t *testing.T) {
	d := types.ModelDescriptor{KVCacheMB: 500, KVFixedMB: 300}
	if got := ScaledKVOverheadMB(d, 1); got != 800 {
		t.Errorf("slots=1: got %d; want 800", got)
	}
	if got := ScaledKVOverheadMB(d, 2); got != 1300 {
		t.Errorf("slots=2: got %d; want 1300 (2*500 + 300)", got)
	}
	if got := ScaledKVOverheadMB(d, 0); got != 800 {
		t.Errorf("slots=0 clamps to 1: got %d; want 800", got)
	}
}

func TestKVParts_FallbackWhenOnlyOverheadSet(t *testing.T) {
	d := types.ModelDescriptor{KVOverheadMB: 700}
	cache, fixed := d.KVParts()
	if cache != 700 || fixed != 0 {
		t.Errorf("KVParts() = (%d, %d); want (700, 0)", cache, fixed)
	}
	if got := ScaledKVOverheadMB(d, 3); got != 2100 {
		t.Errorf("slots=3 with legacy overhead: got %d; want 2100", got)
	}
}

func TestFitLayers_TighterWithMoreSlots(t *testing.T) {
	// weight 3000, per-slot cache 600, fixed 300, 12000 MB free, 10% buffer.
	// slots=1: overhead 900, budget = 12000/1.1 - 900 ≈ 9909 >= 3000 → -1 (full).
	one := FitLayers(3000, ScaledKVOverheadMB(types.ModelDescriptor{KVCacheMB: 600, KVFixedMB: 300}, 1), 4200, 32, 10)
	// slots=1 with only 4200 free: budget = 4200/1.1 - 900 ≈ 2918 < 3000 → partial.
	if one == 0 || one == -1 {
		t.Fatalf("slots=1: got %d; want a partial split", one)
	}
	two := FitLayers(3000, ScaledKVOverheadMB(types.ModelDescriptor{KVCacheMB: 600, KVFixedMB: 300}, 2), 4200, 32, 10)
	// slots=2: overhead 1500, budget = 4200/1.1 - 1500 ≈ 2318 → fewer layers than slots=1.
	if two >= one {
		t.Errorf("slots=2 (%d layers) should fit fewer than slots=1 (%d layers)", two, one)
	}
}
