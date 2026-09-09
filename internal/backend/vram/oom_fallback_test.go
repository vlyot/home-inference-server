package vram

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/types"
)

// loadFailStub returns a model_load_failed BackendError until failN Infer calls
// have been made, then succeeds. Also records SetNextGPULayers.
type loadFailStub struct {
	tier      types.ModelTierLabel
	failFirst int32
	calls     int32
	lastLayer atomic.Int64
}

func (s *loadFailStub) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (s *loadFailStub) Ready() bool                    { return true }
func (s *loadFailStub) SetNextGPULayers(n int)         { s.lastLayer.Store(int64(n)) }
func (s *loadFailStub) Shutdown(context.Context) error { return nil }
func (s *loadFailStub) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	n := atomic.AddInt32(&s.calls, 1)
	if n <= s.failFirst {
		return backend.Response{}, &backend.BackendError{Code: "model_load_failed", Message: "cuda oom"}
	}
	return backend.Response{Output: "ok [" + string(s.tier) + "]", ModelTier: string(s.tier)}, nil
}

func TestRealOOMFallsBackToSmallerTier(t *testing.T) {
	strong := &loadFailStub{tier: types.TierStrong, failFirst: 99} // always fails
	mid := &loadFailStub{tier: types.TierMid, failFirst: 0}        // succeeds
	weak := &loadFailStub{tier: types.TierWeak, failFirst: 0}
	inners := map[types.ModelTierLabel]backend.Backend{
		types.TierWeak: weak, types.TierMid: mid, types.TierStrong: strong,
	}
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 20000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	resp, err := b.Infer(context.Background(), backend.Request{RequestID: "r1", CorrelationID: "c1"})
	if err != nil {
		t.Fatalf("expected a successful fallback, got %v", err)
	}
	if resp.ModelTier != string(types.TierMid) {
		t.Fatalf("served on tier %q, want mid after strong's OOM", resp.ModelTier)
	}
	if atomic.LoadInt32(&strong.calls) == 0 {
		t.Errorf("strong tier was never attempted")
	}
}

func TestOOMFallbackRespectsMinTier(t *testing.T) {
	strong := &loadFailStub{tier: types.TierStrong, failFirst: 99}
	mid := &loadFailStub{tier: types.TierMid, failFirst: 0}
	weak := &loadFailStub{tier: types.TierWeak, failFirst: 0}
	inners := map[types.ModelTierLabel]backend.Backend{
		types.TierWeak: weak, types.TierMid: mid, types.TierStrong: strong,
	}
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 20000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	_, err := b.Infer(context.Background(), backend.Request{
		RequestID: "r1", CorrelationID: "c1", MinTier: string(types.TierStrong),
	})
	if err == nil {
		t.Fatal("expected an error: strong OOMs and min_tier=strong forbids cascading")
	}
}

func TestSelectAndLoadPushesLayerCountToInner(t *testing.T) {
	weak := &loadFailStub{tier: types.TierWeak, failFirst: 0}
	mid := &loadFailStub{tier: types.TierMid, failFirst: 0}
	strong := &loadFailStub{tier: types.TierStrong, failFirst: 0}
	inners := map[types.ModelTierLabel]backend.Backend{
		types.TierWeak: weak, types.TierMid: mid, types.TierStrong: strong,
	}
	// Ample VRAM → strong is selected at full offload (-1).
	b := New(backend.ModalityKindText, testRoster(), &MockVRAMProvider{FreeMB: 40000}, inners, fastOpts())
	defer b.Shutdown(context.Background())

	if _, err := b.Infer(context.Background(), backend.Request{RequestID: "r1"}); err != nil {
		t.Fatal(err)
	}
	if got := strong.lastLayer.Load(); got != -1 {
		t.Fatalf("strong inner got SetNextGPULayers(%d), want -1 (full GPU)", got)
	}
}
