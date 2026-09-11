package vram

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/types"
)

// testRoster returns a 3-tier roster (weak=1000, mid=3000, strong=6000 MB weight)
// with realistic per-tier KV/compute overhead and layer counts.
func testRoster() []types.ModelDescriptor {
	return []types.ModelDescriptor{
		{TierLabel: types.TierWeak, Name: "weak-model", RequiredVRAMMB: 1000, KVOverheadMB: 300, TotalLayers: 32},
		{TierLabel: types.TierMid, Name: "mid-model", RequiredVRAMMB: 3000, KVOverheadMB: 600, TotalLayers: 32},
		{TierLabel: types.TierStrong, Name: "strong-model", RequiredVRAMMB: 6000, KVOverheadMB: 900, TotalLayers: 32},
	}
}

func fastOpts() Options {
	return Options{
		BufferPct:           10,
		EvictionIdleTimeout: 100 * time.Millisecond,
		EvictionInterval:    20 * time.Millisecond,
	}
}

// stubInners returns an inners map with zero-latency stub backends for all tiers.
func stubInners(latency time.Duration) map[types.ModelTierLabel]backend.Backend {
	return map[types.ModelTierLabel]backend.Backend{
		types.TierWeak:   stub.New(backend.ModalityKindText, latency),
		types.TierMid:    stub.New(backend.ModalityKindText, latency),
		types.TierStrong: stub.New(backend.ModalityKindText, latency),
	}
}

// typedStubInners returns the map plus the concrete *stub.StubBackend per tier
// so a test can read PeakInFlight().
func typedStubInners(latency time.Duration) (map[types.ModelTierLabel]backend.Backend, map[types.ModelTierLabel]*stub.StubBackend) {
	typed := map[types.ModelTierLabel]*stub.StubBackend{
		types.TierWeak:   stub.New(backend.ModalityKindText, latency),
		types.TierMid:    stub.New(backend.ModalityKindText, latency),
		types.TierStrong: stub.New(backend.ModalityKindText, latency),
	}
	generic := make(map[types.ModelTierLabel]backend.Backend, len(typed))
	for k, v := range typed {
		generic[k] = v
	}
	return generic, typed
}

func newTestBackend(freeMB int64, opts Options) (*Backend, *MockVRAMProvider) {
	return newTestBackendWithLatency(freeMB, opts, 0)
}

func newTestBackendWithLatency(freeMB int64, opts Options, latency time.Duration) (*Backend, *MockVRAMProvider) {
	mock := &MockVRAMProvider{FreeMB: freeMB}
	b := New(backend.ModalityKindText, testRoster(), mock, stubInners(latency), opts)
	return b, mock
}

// --- Tier selection ---

func TestSelectsLargestFittingTier(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// 8000 MB - 10% buffer = 7200; strong needs 6000 → should fit
	if resp.ModelTier != string(types.TierStrong) {
		t.Errorf("expected strong tier, got %q", resp.ModelTier)
	}
}

// TestPreferredTierPinsToWeakDespiteAmpleVRAM is the regression test for the
// "priority dropdown doesn't actually pick the model" issue: with 8000 MB
// free (plenty for strong), a request that sets PreferredTier="weak" must
// still load the weak tier, not the largest tier that fits.
func TestPreferredTierPinsToWeakDespiteAmpleVRAM(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{
		RequestID: "r1", CorrelationID: "c1", PreferredTier: "weak",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("expected weak tier (pinned), got %q", resp.ModelTier)
	}
}

// TestPreferredTierEvictsAlreadyLoadedLargerTier confirms pinning to a
// smaller tier evicts a currently-loaded larger one, actually freeing VRAM —
// not just "permitting" the smaller tier while leaving the big one resident.
func TestPreferredTierEvictsAlreadyLoadedLargerTier(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	// First request loads strong (the default, largest-fitting choice).
	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("first infer: %v", err)
	}
	if resp.ModelTier != string(types.TierStrong) {
		t.Fatalf("setup: expected strong loaded first, got %q", resp.ModelTier)
	}

	// Second request pins weak — must evict strong and load weak.
	resp, err = b.Infer(context.Background(), backend.Request{
		RequestID: "r2", CorrelationID: "c2", PreferredTier: "weak",
	})
	if err != nil {
		t.Fatalf("second infer: %v", err)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("expected weak tier after pin, got %q", resp.ModelTier)
	}
	if got := b.LoadedTier(); got != string(types.TierWeak) {
		t.Errorf("LoadedTier() = %q; want weak", got)
	}
}

// TestPreferredTierUnknownLabelReturnsInvalidRequest guards against a typo'd
// or stale client-supplied tier name silently falling back to automatic
// selection instead of surfacing a clear error.
func TestPreferredTierUnknownLabelReturnsInvalidRequest(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{
		RequestID: "r1", CorrelationID: "c1", PreferredTier: "ultra",
	})
	if err == nil {
		t.Fatal("expected an error for an unknown preferred_tier, got nil")
	}
	be, ok := err.(*backend.BackendError)
	if !ok || be.Code != "invalid_request" {
		t.Errorf("err = %v; want invalid_request BackendError", err)
	}
}

// TestPreferredTierNotOverriddenBySpeedFloor confirms a pin is never cascaded
// away from by the speed-floor check, even when the pinned tier measures
// below the priority's floor — a pin means "this tier, full stop."
func TestPreferredTierNotOverriddenBySpeedFloor(t *testing.T) {
	inners := measurableInners(1 /* weak: far below any floor */, 20, 20)
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 8000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{
		RequestID: "r1", CorrelationID: "c1", PreferredTier: "weak", Priority: "high",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("expected weak tier to stay pinned despite failing the speed floor, got %q", resp.ModelTier)
	}
}

func TestInferStreamFallsBackToSingleDelta(t *testing.T) {
	// measurableStub does not implement backend.Streamer, so InferStream must run
	// one Infer call and emit the whole output as a single content delta.
	mock := &MockVRAMProvider{FreeMB: 8000}
	b := New(backend.ModalityKindText, testRoster(), mock, measurableInners(10, 10, 10), fastOpts())
	defer b.Shutdown(context.Background())

	var deltas []string
	resp, err := b.InferStream(context.Background(), backend.Request{RequestID: "s1", CorrelationID: "c1"}, func(kind backend.ChunkKind, d string) {
		deltas = append(deltas, d)
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(deltas) != 1 || deltas[0] != "stub ["+string(types.TierStrong)+"]" {
		t.Errorf("deltas = %v; want one delta 'stub [strong]'", deltas)
	}
	if resp.ModelTier != string(types.TierStrong) {
		t.Errorf("model tier = %q; want strong", resp.ModelTier)
	}
}

func TestInferStreamEnforcesMinTier(t *testing.T) {
	// 800 MB free: strong (weight 6000 + overhead 900) cannot fit even a single
	// GPU layer or CPU-only, and min_tier=strong forbids cascading lower → reject.
	b, _ := newTestBackend(800, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.InferStream(context.Background(), backend.Request{
		RequestID: "s1", CorrelationID: "c1", MinTier: string(types.TierStrong),
	}, func(backend.ChunkKind, string) {})
	if err == nil {
		t.Fatal("expected min-tier rejection, got nil")
	}
	be, ok := err.(*backend.BackendError)
	if !ok || be.Code != "overloaded" {
		t.Errorf("err = %v; want overloaded BackendError", err)
	}
}

func TestSelectsStrongViaPartialOffloadUnderModeratePressure(t *testing.T) {
	// 5000 MB free: strong's full weight (6000) doesn't fit, but a partial GPU
	// split does — the big model still gets served, not dropped.
	b, _ := newTestBackend(5000, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierStrong) {
		t.Errorf("expected strong tier via partial offload, got %q", resp.ModelTier)
	}
	if l := b.LoadedModel().GPULayers; l <= 0 || l == -1 {
		t.Errorf("GPULayers = %d; want a partial count", l)
	}
}

func TestSelectsSmallestTierThatFitsOnGPUUnderHeavyPressure(t *testing.T) {
	// 700 MB free: neither strong nor mid can fit even one GPU layer alongside
	// their overhead; weak (weight 1000, overhead 300) fits a partial split.
	b, _ := newTestBackend(700, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("expected weak tier, got %q", resp.ModelTier)
	}
}

func TestReturnsOverloadedWhenEvenCPUOnlyDoesNotFit(t *testing.T) {
	// 200 MB free — below the weak tier's KV/compute overhead (300 + 10%),
	// so not even a CPU-only weight load fits.
	b, _ := newTestBackend(200, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var be *backend.BackendError
	if !isBackendErr(err, &be) || be.Code != "overloaded" {
		t.Errorf("expected overloaded BackendError, got %v", err)
	}
}

func TestAdmitsWeakViaCPUOffloadUnderHeavyPressure(t *testing.T) {
	// 600 MB free — below every tier's full weight, but the weak tier's
	// CPU-only footprint (300 overhead + 10% = 330) fits.
	b, _ := newTestBackend(600, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("expected weak tier via CPU offload, got %q", resp.ModelTier)
	}
}

func TestSkipsReloadWhenSameTierAlreadyLoaded(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	// First call loads large.
	resp1, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("first infer error: %v", err)
	}
	// Second call should reuse loaded tier without evict/reload.
	resp2, err := b.Infer(context.Background(), backend.Request{RequestID: "r2", CorrelationID: "c2"})
	if err != nil {
		t.Fatalf("second infer error: %v", err)
	}
	if resp1.ModelTier != resp2.ModelTier {
		t.Errorf("expected same tier on second call, got %q → %q", resp1.ModelTier, resp2.ModelTier)
	}
}

// --- OOM handling ---

func TestRetriesWithSmallerTierOnOOM(t *testing.T) {
	opts := fastOpts()
	// OOM only the large tier; medium and small succeed.
	opts.OOMSimulator = func(d types.ModelDescriptor) bool {
		return d.TierLabel == types.TierStrong
	}
	b, _ := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierMid) {
		t.Errorf("expected fallback to mid, got %q", resp.ModelTier)
	}
}

func TestReturnsOverloadedWhenAllTiersOOM(t *testing.T) {
	opts := fastOpts()
	opts.OOMSimulator = func(_ types.ModelDescriptor) bool { return true }
	b, _ := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var be *backend.BackendError
	if !isBackendErr(err, &be) || be.Code != "overloaded" {
		t.Errorf("expected overloaded BackendError, got %v", err)
	}
}

// --- Eviction ---

func TestEvictsModelAfterIdleTimeout(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 30 * time.Millisecond
	opts.EvictionInterval = 10 * time.Millisecond

	b, _ := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("infer error: %v", err)
	}
	if b.LoadedModel() == nil {
		t.Fatal("expected model to be loaded after infer")
	}

	// Wait longer than idle timeout + a couple eviction intervals.
	time.Sleep(100 * time.Millisecond)

	if b.LoadedModel() != nil {
		t.Error("expected model to be evicted after idle timeout")
	}
}

// countingSweeper records Sweep() calls for the post-eviction-sweep test.
type countingSweeper struct{ n atomic.Int32 }

func (c *countingSweeper) Sweep() int { c.n.Add(1); return 0 }

func TestPostIdleEvictionSweepsWithinOneTick(t *testing.T) {
	sw := &countingSweeper{}
	opts := fastOpts()
	opts.EvictionIdleTimeout = 30 * time.Millisecond
	opts.EvictionInterval = 10 * time.Millisecond
	opts.Reaper = sw

	b, _ := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer error: %v", err)
	}

	// Wait for the idle eviction + one more tick.
	time.Sleep(90 * time.Millisecond)

	if b.LoadedModel() != nil {
		t.Fatal("expected model evicted after idle timeout")
	}
	if sw.n.Load() == 0 {
		t.Error("expected a reaper sweep right after the idle eviction, got none")
	}
}

func TestEvictsProactivelyOnVRAMPressure(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second // long timeout — won't trigger
	opts.EvictionInterval = 10 * time.Millisecond

	b, mock := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("infer error: %v", err)
	}
	if b.LoadedModel() == nil {
		t.Fatal("expected model loaded after infer")
	}

	// Simulate external VRAM pressure: drop available well below large model's requirement.
	mock.SetFreeMB(100)

	// Wait a few eviction ticks.
	time.Sleep(80 * time.Millisecond)

	if b.LoadedModel() != nil {
		t.Error("expected proactive eviction under VRAM pressure")
	}
}

func TestDoesNotEvictWhileInferInFlight(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 5 * time.Millisecond
	opts.EvictionInterval = 5 * time.Millisecond

	// Use a slow inner backend so the eviction loop fires while the request is active.
	b, _ := newTestBackendWithLatency(8000, opts, 150*time.Millisecond)
	defer b.Shutdown(context.Background())

	// Preload a model.
	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r-pre", CorrelationID: "c-pre"})
	if err != nil {
		t.Fatalf("preload error: %v", err)
	}

	// Start a slow request; check that model is still loaded mid-flight.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		b.Infer(context.Background(), backend.Request{RequestID: "r2", CorrelationID: "c2"}) //nolint:errcheck
	}()

	// Let eviction loop run a few times while the request is in-flight.
	time.Sleep(60 * time.Millisecond)
	if b.LoadedModel() == nil {
		t.Error("model must not be evicted while a request is in-flight")
	}

	wg.Wait()
}

// --- Contention detection ---

func TestContencionDetectedDoesNotAbortRequest(t *testing.T) {
	opts := fastOpts()
	b, mock := newTestBackend(8000, opts)
	defer b.Shutdown(context.Background())

	// Preload large tier.
	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("first infer error: %v", err)
	}

	// Drop VRAM below large-tier's contention threshold (6000/2=3000 MB)
	// but still enough for the small tier (1000 MB) to be selected.
	mock.SetFreeMB(2000)

	// Request must still succeed despite contention.
	_, err = b.Infer(context.Background(), backend.Request{RequestID: "r2", CorrelationID: "c2"})
	if err != nil {
		t.Errorf("contention must not abort request; got error: %v", err)
	}
}

// --- Lifecycle ---

func TestShutdownStopsEvictionAndEvictsModel(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())

	_, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("infer error: %v", err)
	}

	if err := b.Shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown error: %v", err)
	}
	if b.LoadedModel() != nil {
		t.Error("expected model to be nil after Shutdown")
	}
}

func TestReadyAlwaysTrue(t *testing.T) {
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	if !b.Ready() {
		t.Error("Ready() should return true before any infer")
	}
	b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}) //nolint:errcheck
	if !b.Ready() {
		t.Error("Ready() should return true after infer")
	}
}

// --- Unlimited VRAM (NVMLProvider returns -1) ---

func TestUnlimitedVRAMSelectsLargestTier(t *testing.T) {
	// FreeMB=-1 means unlimited; should always pick strong.
	b, _ := newTestBackend(-1, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierStrong) {
		t.Errorf("expected strong tier with unlimited VRAM, got %q", resp.ModelTier)
	}
}

// --- min_tier enforcement ---

func TestInfer_MinTierRejection(t *testing.T) {
	// 700 MB free → mid (weight 3000 + overhead 600) can't fit even one GPU
	// layer or CPU-only; min_tier=mid forbids dropping to weak → rejected.
	b, _ := newTestBackend(700, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{
		RequestID:     "r1",
		CorrelationID: "c1",
		MinTier:       "mid",
	})
	if err == nil {
		t.Fatal("expected overloaded error for min_tier rejection, got nil")
	}
	var be *backend.BackendError
	if !isBackendErr(err, &be) || be.Code != "overloaded" {
		t.Errorf("expected overloaded BackendError, got %v", err)
	}
}

func TestInfer_MinTierSatisfied(t *testing.T) {
	// 8000 MB → strong selected; min_tier=mid → strong >= mid, should pass
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{
		RequestID:     "r1",
		CorrelationID: "c1",
		MinTier:       "mid",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ModelTier != string(types.TierStrong) {
		t.Errorf("expected strong tier, got %q", resp.ModelTier)
	}
}

// --- speed floor ---

func TestInfer_SpeedFloorSkipped_NoSamples(t *testing.T) {
	// tokPerSec==0 (no samples yet) → speed floor not enforced regardless of priority
	b, _ := newTestBackend(8000, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{
		RequestID:     "r1",
		CorrelationID: "c1",
		Priority:      "high",
	})
	if err != nil {
		t.Fatalf("speed floor must not fire with zero tok/sec samples: %v", err)
	}
}

// --- quality degraded ---

func TestInfer_HighPriorityWeakTier_QualityDegraded(t *testing.T) {
	// 700 MB free → only weak is admittable (strong/mid can't fit any GPU layer);
	// high priority landing on weak → QualityDegraded=true
	b, _ := newTestBackend(700, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{
		RequestID:     "r1",
		CorrelationID: "c1",
		Priority:      "high",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !resp.QualityDegraded {
		t.Error("expected QualityDegraded=true for high-priority on weak tier")
	}
}

func TestInfer_NormalPriorityWeakTier_NotDegraded(t *testing.T) {
	// normal priority on weak tier → QualityDegraded stays false
	b, _ := newTestBackend(1500, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{
		RequestID:     "r1",
		CorrelationID: "c1",
		Priority:      "normal",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.QualityDegraded {
		t.Error("expected QualityDegraded=false for normal priority")
	}
}

// --- helpers ---

func isBackendErr(err error, out **backend.BackendError) bool {
	if err == nil {
		return false
	}
	type unwrapper interface{ Unwrap() error }
	cur := err
	for cur != nil {
		if be, ok := cur.(*backend.BackendError); ok {
			*out = be
			return true
		}
		if u, ok := cur.(unwrapper); ok {
			cur = u.Unwrap()
		} else {
			break
		}
	}
	return false
}

// --- partial-offload selection + speed-floor cascade + lifecycle ---

// measurableStub is an inner backend with a controllable rolling tok/sec and a
// Shutdown counter, for cascade and eviction-teardown tests.
type measurableStub struct {
	tier       types.ModelTierLabel
	tps        float64
	resident   int64
	mu         sync.Mutex
	shutdowns  int
	restarting bool
}

func (m *measurableStub) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (m *measurableStub) Ready() bool                    { return true }
func (m *measurableStub) TokPerSec() float64             { return m.tps }
func (m *measurableStub) ResidentMB() int64              { return m.resident }
func (m *measurableStub) Restarting() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.restarting
}
func (m *measurableStub) setRestarting(v bool) {
	m.mu.Lock()
	m.restarting = v
	m.mu.Unlock()
}
func (m *measurableStub) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	return backend.Response{Output: "stub [" + string(m.tier) + "]", ModelTier: string(m.tier)}, nil
}
func (m *measurableStub) Shutdown(_ context.Context) error {
	m.mu.Lock()
	m.shutdowns++
	m.mu.Unlock()
	return nil
}
func (m *measurableStub) shutdownCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.shutdowns
}

func measurableInners(weakTPS, midTPS, strongTPS float64) map[types.ModelTierLabel]backend.Backend {
	return map[types.ModelTierLabel]backend.Backend{
		types.TierWeak:   &measurableStub{tier: types.TierWeak, tps: weakTPS, resident: 500},
		types.TierMid:    &measurableStub{tier: types.TierMid, tps: midTPS, resident: 2000},
		types.TierStrong: &measurableStub{tier: types.TierStrong, tps: strongTPS, resident: 4000},
	}
}

func TestSelectAdmitsStrongViaPartialSplitUnderPressure(t *testing.T) {
	// 5500 MB free: strong weight 6000 doesn't fully fit, but a partial split's
	// GPU-resident footprint (weight fraction + 900 overhead) does.
	mock := &MockVRAMProvider{FreeMB: 5500}
	b := New(backend.ModalityKindText, testRoster(), mock, stubInners(0), fastOpts())
	defer b.Shutdown(context.Background())

	desc, err := b.selectAndLoad(context.Background(), nil, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc.TierLabel != types.TierStrong {
		t.Fatalf("tier = %q; want strong via partial split", desc.TierLabel)
	}
	snap := b.LoadedModel()
	if snap.GPULayers <= 0 || snap.GPULayers == -1 {
		t.Errorf("GPULayers = %d; want a partial count (1..31)", snap.GPULayers)
	}
}

func TestSpeedFloorCascadesToLowerTier(t *testing.T) {
	// Plenty of VRAM, but strong's rolling tok/s (3) is below the normal floor (4).
	inners := measurableInners(20, 20, 3)
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, _, err := b.selectInner(context.Background(), backend.Request{Priority: "normal", CorrelationID: "c1"}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc.TierLabel != types.TierMid {
		t.Errorf("tier = %q; want mid after speed-floor cascade", desc.TierLabel)
	}
}

func TestSpeedFloorCascadeStopsAtMinTier(t *testing.T) {
	inners := measurableInners(20, 20, 3)
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, _, err := b.selectInner(context.Background(), backend.Request{
		Priority: "normal", MinTier: string(types.TierStrong), CorrelationID: "c1",
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc.TierLabel != types.TierStrong {
		t.Errorf("tier = %q; want strong (min_tier caps the cascade)", desc.TierLabel)
	}
}

func TestSpeedFloorCascadeBottomsOutServesDegraded(t *testing.T) {
	// Every tier is below the high floor (8); with no min_tier the cascade
	// bottoms out on weak and serves rather than erroring.
	inners := measurableInners(2, 2, 2)
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, _, err := b.selectInner(context.Background(), backend.Request{Priority: "high", CorrelationID: "c1"}, nil)
	if err != nil {
		t.Fatalf("expected degraded service, got error: %v", err)
	}
	if desc.TierLabel != types.TierWeak {
		t.Errorf("tier = %q; want weak at the bottom of the cascade", desc.TierLabel)
	}
}

func TestEvictShutsDownInner(t *testing.T) {
	inners := measurableInners(20, 20, 20)
	opts := fastOpts()
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, opts)

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer: %v", err)
	}
	b.mu.Lock()
	toStop := b.evictLocked()
	b.mu.Unlock()
	stopInner(toStop)

	strong := inners[types.TierStrong].(*measurableStub)
	if strong.shutdownCount() != 1 {
		t.Errorf("strong inner Shutdown count = %d; want 1", strong.shutdownCount())
	}
	_ = b.Shutdown(context.Background())
}

// TestEvictionIsSynchronousBeforeReselect guards the fix for the eviction/
// reload race: evictLocked must hand the inner to stopInner, and stopInner
// must fully finish (the subprocess is actually down) before the caller does
// anything that could let a concurrent request observe loadedTier == nil and
// race a fresh load against the still-shutting-down old subprocess. Simulated
// here with a slow inner Shutdown: by the time stopInner returns, the
// shutdown must already be complete, not merely started in the background.
func TestEvictionIsSynchronousBeforeReselect(t *testing.T) {
	inners := measurableInners(20, 20, 20)
	strong := inners[types.TierStrong].(*measurableStub)
	slow := &slowShutdownStub{measurableStub: strong, delay: 100 * time.Millisecond}
	inners[types.TierStrong] = slow

	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second // manual evict only
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer: %v", err)
	}

	b.mu.Lock()
	toStop := b.evictLocked()
	b.mu.Unlock()

	if slow.shutdownCount() != 0 {
		t.Fatal("Shutdown must not have run yet — evictLocked only clears bookkeeping")
	}
	stopInner(toStop)
	if slow.shutdownCount() != 1 {
		t.Error("stopInner must block until the slow inner Shutdown completes")
	}
}

// slowShutdownStub wraps measurableStub with an artificial delay in Shutdown,
// so a test can distinguish "shutdown kicked off" from "shutdown completed".
type slowShutdownStub struct {
	*measurableStub
	delay time.Duration
}

func (s *slowShutdownStub) Shutdown(ctx context.Context) error {
	time.Sleep(s.delay)
	return s.measurableStub.Shutdown(ctx)
}

func TestTierSwitchShutsDownPreviousInner(t *testing.T) {
	inners := measurableInners(20, 20, 20)
	mock := &MockVRAMProvider{FreeMB: 12000}
	b := New(backend.ModalityKindText, testRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("first infer: %v", err)
	}
	// Drop VRAM so strong no longer fits even partially → switch to mid.
	mock.SetFreeMB(3500)
	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r2", CorrelationID: "c2"}); err != nil {
		t.Fatalf("second infer: %v", err)
	}

	strong := inners[types.TierStrong].(*measurableStub)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if strong.shutdownCount() > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if strong.shutdownCount() < 1 {
		t.Errorf("strong inner never shut down on tier switch")
	}
}

func TestSelfVRAMMB_SumsResidentInners(t *testing.T) {
	inners := measurableInners(20, 20, 20) // resident 500 + 2000 + 4000
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	if got := b.SelfVRAMMB(); got != 6500 {
		t.Errorf("SelfVRAMMB = %d; want 6500", got)
	}
}

// --- eviction: pressure check no longer evicts a model's own allocation ---

func TestDoesNotEvictOwnAllocationUnderNormalHeadroom(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second // don't let idle-timeout interfere
	opts.EvictionInterval = 10 * time.Millisecond

	// 12000 MB free at load time → strong (weight 6000 + KV 900) loads full-GPU.
	b, mock := newTestBackend(12000, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer error: %v", err)
	}
	// Now NVML reports only what's left after the strong model's ~6900 MB:
	// 12000 - 6900 = 5100 MB free. That's low, but it's *our* allocation — the
	// model still fits on an effective-headroom basis, so it must NOT be evicted.
	mock.SetFreeMB(5100)

	time.Sleep(80 * time.Millisecond)

	if b.LoadedModel() == nil {
		t.Error("model evicted despite fitting on effective headroom — pressure check regressed")
	}
}

func TestEvictsWhenExternalProcessCollapsesVRAM(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second
	opts.EvictionInterval = 10 * time.Millisecond

	b, mock := newTestBackend(12000, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer error: %v", err)
	}
	// A real external grab: free VRAM collapses below the absolute floor and the
	// model no longer fits even counting its own allocation as reclaimable.
	mock.SetFreeMB(300)

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if b.LoadedModel() == nil {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Error("expected eviction when external process collapsed VRAM below the floor")
}

func TestNoEvictReloadLoopWhenModelJustFits(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second
	opts.EvictionInterval = 5 * time.Millisecond

	inners := measurableInners(20, 20, 20)
	mock := &MockVRAMProvider{FreeMB: 12000}
	b := New(backend.ModalityKindText, testRoster(), mock, inners, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer error: %v", err)
	}
	// Simulate steady-state: free VRAM is just what's left after our own model.
	mock.SetFreeMB(5200)

	time.Sleep(100 * time.Millisecond)

	strong := inners[types.TierStrong].(*measurableStub)
	if n := strong.shutdownCount(); n != 0 {
		t.Errorf("strong inner shut down %d times in steady state — evict/reload storm not fixed", n)
	}
	if b.LoadedModel() == nil {
		t.Error("model unexpectedly evicted in steady state")
	}
}

func TestEvictionSkippedWhileInnerRestarting(t *testing.T) {
	opts := fastOpts()
	opts.EvictionIdleTimeout = 10 * time.Second
	opts.EvictionInterval = 5 * time.Millisecond

	inners := measurableInners(20, 20, 20)
	mock := &MockVRAMProvider{FreeMB: 12000}
	b := New(backend.ModalityKindText, testRoster(), mock, inners, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"}); err != nil {
		t.Fatalf("infer error: %v", err)
	}

	strong := inners[types.TierStrong].(*measurableStub)
	strong.setRestarting(true)
	// Genuine external pressure that would otherwise evict.
	mock.SetFreeMB(200)

	time.Sleep(60 * time.Millisecond)

	if n := strong.shutdownCount(); n != 0 {
		t.Errorf("inner shut down %d times while mid-restart — eviction guard missing", n)
	}

	strong.setRestarting(false)
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if strong.shutdownCount() > 0 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Error("expected eviction to proceed once restart cleared")
}

func TestEffectiveHeadroomFitsMatchesInlineFormula(t *testing.T) {
	d := &types.ModelDescriptor{RequiredVRAMMB: 4000, KVCacheMB: 600, KVFixedMB: 300, TotalLayers: 32}
	cases := []struct {
		availRaw int64
		layers   int
		slots    int
	}{
		{availRaw: 100, layers: -1, slots: 1},
		{availRaw: 1000, layers: -1, slots: 2},
		{availRaw: 5000, layers: -1, slots: 1},
		{availRaw: 500, layers: 0, slots: 2},
		{availRaw: 2000, layers: 16, slots: 1},
	}
	for _, c := range cases {
		resident := GPUResidentMB(d.RequiredVRAMMB, ScaledKVOverheadMB(*d, c.slots, false), c.layers, d.TotalLayers)
		want := fitsWithBuffer(resident, c.availRaw+resident, 10)
		got := effectiveHeadroomFits(d, c.layers, false, c.slots, c.availRaw, 10)
		if got != want {
			t.Errorf("availRaw=%d layers=%d slots=%d: effectiveHeadroomFits=%v; inline=%v", c.availRaw, c.layers, c.slots, got, want)
		}
	}
}

// --- concurrency ---

func TestSemaphoreCapsConcurrentInfer(t *testing.T) {
	opts := fastOpts()
	opts.MaxParallel = 2
	generic, typed := typedStubInners(40 * time.Millisecond)
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000}, generic, opts)
	defer b.Shutdown(context.Background())

	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = b.Infer(context.Background(), backend.Request{RequestID: "r", CorrelationID: "c"})
		}()
	}
	wg.Wait()

	// Ample VRAM → all requests land on strong. Its stub must never have seen
	// more than MaxParallel concurrent calls.
	if peak := typed[types.TierStrong].PeakInFlight(); peak > 2 {
		t.Fatalf("strong inner peak concurrency = %d; want <= 2 (MaxParallel)", peak)
	}
}

func TestSemaphoreReleasedOnError(t *testing.T) {
	opts := fastOpts()
	opts.MaxParallel = 1
	// No inner registered for any tier → selectInner returns internal_error.
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000},
		map[types.ModelTierLabel]backend.Backend{}, opts)
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1"}); err == nil {
		t.Fatal("expected an error from a backend with no inners")
	}
	if got := b.InFlight(); got != 0 {
		t.Fatalf("InFlight = %d after a failed call; semaphore leaked", got)
	}
	// A second call must still be admitted (slot was released).
	done := make(chan struct{})
	go func() { _, _ = b.Infer(context.Background(), backend.Request{RequestID: "r2"}); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("second Infer blocked; semaphore was not released after the error")
	}
}

func TestInferStreamAlsoAcquiresSemaphore(t *testing.T) {
	opts := fastOpts()
	opts.MaxParallel = 1
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000},
		stubInners(60*time.Millisecond), opts)
	defer b.Shutdown(context.Background())

	streamStarted := make(chan struct{})
	streamDone := make(chan struct{})
	go func() {
		close(streamStarted)
		_, _ = b.InferStream(context.Background(), backend.Request{RequestID: "s1"},
			func(backend.ChunkKind, string) {})
		close(streamDone)
	}()
	<-streamStarted
	time.Sleep(10 * time.Millisecond) // let the stream acquire the single slot

	inferReturned := make(chan struct{})
	go func() {
		_, _ = b.Infer(context.Background(), backend.Request{RequestID: "i1"})
		close(inferReturned)
	}()

	select {
	case <-inferReturned:
		t.Fatal("Infer completed while a stream held the only slot")
	case <-time.After(20 * time.Millisecond):
	}
	<-streamDone
	select {
	case <-inferReturned:
	case <-time.After(time.Second):
		t.Fatal("Infer never ran after the stream released the slot")
	}
}

func TestConcurrentSelectDifferentTiersNoForceTierRace(t *testing.T) {
	opts := fastOpts()
	opts.MaxParallel = 4
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 12000},
		stubInners(10*time.Millisecond), opts)
	defer b.Shutdown(context.Background())

	var wg sync.WaitGroup
	got := make([]string, 20)
	for i := 0; i < 20; i++ {
		wg.Add(1)
		tier := types.TierStrong
		if i%2 == 0 {
			tier = types.TierWeak
		}
		idx := i
		go func() {
			defer wg.Done()
			resp, err := b.Infer(context.Background(), backend.Request{
				RequestID: "r", PreferredTier: string(tier),
			})
			if err != nil {
				t.Errorf("req %d (%s): %v", idx, tier, err)
				return
			}
			got[idx] = resp.ModelTier
		}()
	}
	wg.Wait()

	for i, tier := range got {
		want := string(types.TierStrong)
		if i%2 == 0 {
			want = string(types.TierWeak)
		}
		if tier != want {
			t.Errorf("req %d: served tier %q; want %q (forceTier race?)", i, tier, want)
		}
	}
}

// TestConcurrentColdInferLoadsModelOnce is the regression for the concurrent
// selectAndLoad race: with MaxParallel > 1, N requests hitting a cold backend
// must trigger exactly one model load, not N.
func TestConcurrentColdInferLoadsModelOnce(t *testing.T) {
	opts := fastOpts()
	opts.MaxParallel = 4
	var loads int32
	opts.OOMSimulator = func(types.ModelDescriptor) bool {
		atomic.AddInt32(&loads, 1)        // spy: never returns true, just counts loadModel calls
		time.Sleep(20 * time.Millisecond) // widen the load window so an unsynchronised race shows
		return false
	}
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 20000},
		stubInners(15*time.Millisecond), opts)
	defer b.Shutdown(context.Background())

	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r", CorrelationID: "c"}); err != nil {
				t.Errorf("infer: %v", err)
			}
		}()
	}
	wg.Wait()

	if n := atomic.LoadInt32(&loads); n != 1 {
		t.Fatalf("loadModel ran %d times for a cold burst; want 1 (selectAndLoad race)", n)
	}
}

// erroringInner is a backend whose Shutdown reports incomplete teardown.
type erroringInner struct{ shutdownErr error }

func (e *erroringInner) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (e *erroringInner) Ready() bool                    { return true }
func (e *erroringInner) Infer(context.Context, backend.Request) (backend.Response, error) {
	return backend.Response{Output: "x"}, nil
}
func (e *erroringInner) Shutdown(context.Context) error { return e.shutdownErr }

func TestStopInner_ReturnsShutdownError(t *testing.T) {
	want := errors.New("subprocess pid 123 survived stop+kill")
	if got := stopInner(&erroringInner{shutdownErr: want}); got != want {
		t.Fatalf("stopInner returned %v; want %v", got, want)
	}
	if got := stopInner(nil); got != nil {
		t.Fatalf("stopInner(nil) = %v; want nil", got)
	}
	if got := stopInner(&erroringInner{shutdownErr: nil}); got != nil {
		t.Fatalf("stopInner with clean Shutdown = %v; want nil", got)
	}
}

// --- Vision mode ---

// visionAwareStub is a backend.Backend that also implements gpuLayerSink and
// visionSink, so tests can observe what loadModel pushes to it before a spawn
// — mirroring what *llamacpp.Backend does in production.
type visionAwareStub struct {
	mu          sync.Mutex
	shutdowns   int
	lastVision  bool
	visionCalls int
	lastReq     backend.Request
}

func (v *visionAwareStub) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (v *visionAwareStub) Ready() bool                    { return true }
func (v *visionAwareStub) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	v.mu.Lock()
	v.lastReq = req
	v.mu.Unlock()
	return backend.Response{Output: "ok", TokPerSecSample: 10}, nil
}
func (v *visionAwareStub) Shutdown(context.Context) error {
	v.mu.Lock()
	v.shutdowns++
	v.mu.Unlock()
	return nil
}
func (v *visionAwareStub) SetNextGPULayers(int) {}
func (v *visionAwareStub) SetNextNeedsVision(needsVision bool) {
	v.mu.Lock()
	v.lastVision = needsVision
	v.visionCalls++
	v.mu.Unlock()
}
func (v *visionAwareStub) snapshot() (shutdowns, visionCalls int, lastVision bool) {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.shutdowns, v.visionCalls, v.lastVision
}

func (v *visionAwareStub) lastRequest() backend.Request {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.lastReq
}

// visionRoster is a 3-tier roster where only weak has a vision variant,
// mirroring the real weak-tier-only vision deployment.
func visionRoster() []types.ModelDescriptor {
	return []types.ModelDescriptor{
		{
			TierLabel: types.TierWeak, Name: "weak-vl", RequiredVRAMMB: 1000,
			MMProjPath: "mmproj.gguf", TotalLayers: 32,
			KVCacheMB: 100, KVFixedMB: 200,
			VisionRequiredVRAMMB: 1000, VisionKVCacheMB: 100, VisionKVFixedMB: 800,
		},
		{TierLabel: types.TierMid, Name: "mid-model", RequiredVRAMMB: 3000, KVCacheMB: 200, KVFixedMB: 400, TotalLayers: 32},
		{TierLabel: types.TierStrong, Name: "strong-model", RequiredVRAMMB: 6000, KVCacheMB: 300, KVFixedMB: 600, TotalLayers: 32},
	}
}

func visionInners() (map[types.ModelTierLabel]backend.Backend, *visionAwareStub) {
	weak := &visionAwareStub{}
	return map[types.ModelTierLabel]backend.Backend{
		types.TierWeak:   weak,
		types.TierMid:    &visionAwareStub{},
		types.TierStrong: &visionAwareStub{},
	}, weak
}

func TestSelectInner_TextRequest_IgnoresVisionOnlyConstants(t *testing.T) {
	inners, weak := visionInners()
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, _, err := b.selectInner(context.Background(), backend.Request{Prompt: "hi", PreferredTier: "weak"}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc.TierLabel != types.TierWeak {
		t.Fatalf("tier = %q; want weak (pinned)", desc.TierLabel)
	}
	if b.loadedVision {
		t.Error("loadedVision = true for a text request; want false")
	}
	_, calls, lastVision := weak.snapshot()
	if calls != 1 || lastVision {
		t.Errorf("SetNextNeedsVision calls=%d lastVision=%v; want 1 call with false", calls, lastVision)
	}
}

func TestSelectInner_VisionRequest_RestrictsToVisionCapableTiers(t *testing.T) {
	inners, _ := visionInners()
	// Ample VRAM: without the vision restriction, strong (largest) would win.
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, _, err := b.selectInner(context.Background(), backend.Request{ImageData: []byte{1}}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc.TierLabel != types.TierWeak {
		t.Fatalf("tier = %q; want weak (the only HasVision() tier)", desc.TierLabel)
	}
	if !b.loadedVision {
		t.Error("loadedVision = false after a vision load; want true")
	}
}

func TestSelectInner_VisionRequest_NoVisionTierFits_ReturnsOverloaded(t *testing.T) {
	inners, _ := visionInners()
	// Too little VRAM even for weak's vision-mode CPU-only overhead (1000).
	mock := &MockVRAMProvider{FreeMB: 50}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	_, _, err := b.selectInner(context.Background(), backend.Request{ImageData: []byte{1}}, nil)
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "overloaded" {
		t.Fatalf("err = %v; want overloaded", err)
	}
}

func TestSelectInner_PreferredTierVisionRequest_NonVisionTier_Returns400(t *testing.T) {
	inners, _ := visionInners()
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	_, _, err := b.selectInner(context.Background(), backend.Request{ImageData: []byte{1}, PreferredTier: "mid"}, nil)
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "invalid_request" {
		t.Fatalf("err = %v; want invalid_request", err)
	}
}

func TestSelectAndLoad_ModeSwitchForcesEvictAndReload(t *testing.T) {
	inners, weak := visionInners()
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	// Load weak in text mode.
	if _, err := b.selectAndLoad(context.Background(), &types.ModelDescriptor{TierLabel: types.TierWeak}, false); err != nil {
		t.Fatalf("text load: %v", err)
	}
	if b.loadedVision {
		t.Fatal("loadedVision = true after a text load")
	}

	// A vision request for the same tier must evict + reload in vision mode.
	if _, err := b.selectAndLoad(context.Background(), &types.ModelDescriptor{TierLabel: types.TierWeak}, true); err != nil {
		t.Fatalf("vision load: %v", err)
	}
	if !b.loadedVision {
		t.Fatal("loadedVision = false after a vision load; want true")
	}
	shutdowns, calls, lastVision := weak.snapshot()
	if shutdowns != 1 {
		t.Errorf("shutdowns = %d; want 1 (evicted once for the mode switch)", shutdowns)
	}
	if calls != 2 || !lastVision {
		t.Errorf("SetNextNeedsVision calls=%d lastVision=%v; want 2 calls, last true", calls, lastVision)
	}
}

func TestSelectAndLoad_WarmVisionModeReusedForSecondVisionRequest(t *testing.T) {
	inners, weak := visionInners()
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	if _, err := b.selectAndLoad(context.Background(), &types.ModelDescriptor{TierLabel: types.TierWeak}, true); err != nil {
		t.Fatalf("first vision load: %v", err)
	}
	if _, err := b.selectAndLoad(context.Background(), &types.ModelDescriptor{TierLabel: types.TierWeak}, true); err != nil {
		t.Fatalf("second vision load: %v", err)
	}
	shutdowns, _, _ := weak.snapshot()
	if shutdowns != 0 {
		t.Errorf("shutdowns = %d; want 0 (warm vision-mode tier reused, no reload)", shutdowns)
	}
}

func TestEffectiveHeadroomFits_UsesLoadedVisionFlagForResidentEstimate(t *testing.T) {
	d := &types.ModelDescriptor{
		RequiredVRAMMB: 1000, KVCacheMB: 100, KVFixedMB: 200, TotalLayers: 32,
		MMProjPath:           "mmproj.gguf",
		VisionRequiredVRAMMB: 1000, VisionKVCacheMB: 100, VisionKVFixedMB: 3000,
	}
	// effectiveHeadroomFits checks resident*bufferPct% <= availRaw (the
	// resident amount itself is credited back into availRaw either way — see
	// fitsWithBuffer). Full-GPU resident: text = 1000+300=1300 (10% = 130);
	// vision = 1000+3100=4100 (10% = 410). At availRaw=200 the text margin
	// fits but the (much larger) vision margin does not.
	textFits := effectiveHeadroomFits(d, -1, false, 1, 200, 10)
	visionFits := effectiveHeadroomFits(d, -1, true, 1, 200, 10)
	if !textFits {
		t.Error("text mode's smaller resident footprint should fit at availRaw=200")
	}
	if visionFits {
		t.Error("vision mode's larger resident footprint (4100 MB) should NOT fit at availRaw=200")
	}
}

func TestDescribe_RoutesThroughNormalSelectAndLoad(t *testing.T) {
	inners, weak := visionInners()
	mock := &MockVRAMProvider{FreeMB: 10000}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	desc, tier, err := b.Describe(context.Background(), []byte{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc != "ok" {
		t.Errorf("description = %q; want the inner backend's Output", desc)
	}
	if tier != string(types.TierWeak) {
		t.Errorf("tier = %q; want weak", tier)
	}
	if !b.loadedVision {
		t.Error("Describe should have loaded the weak tier in vision mode")
	}
	_, calls, lastVision := weak.snapshot()
	if calls == 0 || !lastVision {
		t.Errorf("SetNextNeedsVision calls=%d lastVision=%v; want at least 1 call, true", calls, lastVision)
	}
	if weak.lastRequest().SystemPrompt == "" {
		t.Error("Describe's request should carry a non-empty SystemPrompt (regression guard for the refusal bug: an image with no system turn causes a small VLM to refuse)")
	}
}

func TestDescribe_PropagatesBackendError(t *testing.T) {
	inners, _ := visionInners()
	// No VRAM at all: even weak's vision-mode CPU-only overhead won't fit.
	mock := &MockVRAMProvider{FreeMB: 10}
	b := New(backend.ModalityKindText, visionRoster(), mock, inners, fastOpts())
	defer b.Shutdown(context.Background())

	_, _, err := b.Describe(context.Background(), []byte{1})
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "overloaded" {
		t.Fatalf("err = %v; want overloaded, got %v", be, err)
	}
}
