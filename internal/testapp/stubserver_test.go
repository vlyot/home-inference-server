package testapp_test

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/backend/vram"
	"github.com/ngkaichong/home-inference-server/internal/batcher"
	"github.com/ngkaichong/home-inference-server/internal/chatstore"
	"github.com/ngkaichong/home-inference-server/internal/dispatcher"
	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/types"
)

// cpuZero backs server.SetPressureSources with a zero CPU reading.
type cpuZero struct{}

func (cpuZero) Pct() float64 { return 0 }

// fakeTokenizer backs /v1/tokenize + /v1/model/props in the stub server.
type fakeTokenizer struct{}

func (fakeTokenizer) Tokenize(_ context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}
func (fakeTokenizer) NCtx(context.Context) (int, error) { return 4096, nil }

// stubRoster mirrors the three text tiers so vram.Backend's tier-selection,
// preferred_tier and min_tier logic are exercised (with stub inners).
func stubRoster() []types.ModelDescriptor {
	return []types.ModelDescriptor{
		{TierLabel: types.TierWeak, Name: "stub-weak", RequiredVRAMMB: 1000, TotalLayers: 32, KVOverheadMB: 200},
		{TierLabel: types.TierMid, Name: "stub-mid", RequiredVRAMMB: 3000, TotalLayers: 32, KVOverheadMB: 400},
		{TierLabel: types.TierStrong, Name: "stub-strong", RequiredVRAMMB: 5000, TotalLayers: 32, KVOverheadMB: 600},
	}
}

// startStubServer builds the real server pipeline (queue → batcher → router →
// vram.Backend → stub inner) so the conformance suite runs with no GPU and no
// llama-server while still exercising tier selection.
func startStubServer(t *testing.T) string {
	t.Helper()

	roster := stubRoster()
	inners := map[types.ModelTierLabel]backend.Backend{
		types.TierWeak:   stub.New(backend.ModalityKindText, 0),
		types.TierMid:    stub.New(backend.ModalityKindText, 0),
		types.TierStrong: stub.New(backend.ModalityKindText, 0),
	}
	opts := vram.DefaultOptions()
	opts.MaxParallel = 2
	textBackend := vram.New(backend.ModalityKindText, roster, &vram.MockVRAMProvider{FreeMB: 8000},
		inners, opts)
	t.Cleanup(func() { _ = textBackend.Shutdown(context.Background()) })

	q := queue.New(64)
	backends := map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   textBackend,
		backend.ModalityKindVision: stub.New(backend.ModalityKindVision, 0),
	}
	r := router.New(backends)

	ctx, cancel := context.WithCancel(context.Background())
	disp := dispatcher.New(r, 2)
	go disp.Run(ctx)
	b := batcher.New(batcher.Config{MaxBatchSize: 8, WindowDuration: 5 * time.Millisecond}, q.Drain(), func(batch []queue.Job) {
		if !disp.Submit(batch) {
			go r.Dispatch(ctx, batch)
		}
	})
	go b.Run(ctx)

	lb := logbuf.NewBuffer(1000, time.Hour)

	srv := server.New(q, func() types.ServerStatus {
		return types.ServerStatus{
			QueueDepth:      q.Depth(),
			ActiveBatchSize: b.ActiveBatchSize(),
			InFlight:        disp.InFlight(),
			MaxParallel:     disp.MaxParallel(),
			AvailableVRAMMB: -1,
			Version:         "stub",
		}
	}, "stub", time.Now())
	srv.SetBackends(backends)
	srv.SetPressureSources(cpuZero{}, textBackend)
	srv.SetTokenizer(fakeTokenizer{})
	srv.SetLogSource(lb)

	store, err := chatstore.Open(t.TempDir())
	if err != nil {
		t.Fatalf("chatstore: %v", err)
	}
	srv.SetChatStore(store)

	ts := httptest.NewServer(srv)
	t.Cleanup(func() { cancel(); ts.Close() })
	return ts.URL
}
