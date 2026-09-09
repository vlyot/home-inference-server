// Package vram implements a VRAM-aware inference backend.
// It wraps a tiered model roster, queries available GPU headroom on each
// request, selects the largest model tier that fits within the headroom
// minus a configurable buffer, and retries with a smaller tier on OOM.
// Idle models are proactively evicted by a background goroutine.
package vram

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// Options configures the VRAM-aware backend.
type Options struct {
	// BufferPct is the percentage of required VRAM reserved as a safety margin
	// to account for driver overhead and OS-level fragmentation. Default: 10.
	BufferPct int

	// EvictionIdleTimeout is how long a loaded model may be idle before the
	// eviction loop considers removing it. Default: 20s.
	EvictionIdleTimeout time.Duration

	// EvictionInterval is how often the eviction loop runs. Default: 10s.
	EvictionInterval time.Duration

	// OOMSimulator is an optional hook called inside loadModel to force OOM
	// in tests. If it returns true, loadModel returns an OOM error.
	// Nil in production.
	OOMSimulator func(tier types.ModelDescriptor) bool

	// NotifyCh receives human-readable alert strings (e.g. quality-degraded
	// events) for display in the system tray tooltip. Non-blocking send; drop
	// if the channel is full. Nil disables notifications.
	NotifyCh chan<- string

	// Reaper, when set, is swept from the eviction loop (while no model is
	// loaded, and right after every eviction) to kill any stray llama-server
	// subprocess left on a roster port. Sweep returns the count it killed.
	Reaper interface{ Sweep() int }

	// MaxParallel is the number of concurrent inference requests permitted
	// through this backend, matching llama-server's --parallel N. It gates
	// Infer/InferStream via a semaphore and scales the per-slot KV-cache term
	// in the VRAM fit/eviction math. Default: 2. Values < 1 are clamped to 1.
	MaxParallel int
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() Options {
	return Options{
		BufferPct:           10,
		EvictionIdleTimeout: 20 * time.Second,
		EvictionInterval:    10 * time.Second,
		MaxParallel:         2,
	}
}

// oomError is a sentinel for OOM-on-load failures so the retry path can
// distinguish them from other loadModel errors.
type oomError struct{ tier string }

func (e *oomError) Error() string { return fmt.Sprintf("OOM loading %s tier", e.tier) }

// Backend is a VRAM-aware inference backend. It satisfies backend.Backend.
type Backend struct {
	modality backend.ModalityKind
	// roster is sorted small→large; selectTier walks it in reverse.
	roster []types.ModelDescriptor
	vram   VRAMProvider
	opts   Options
	cancel context.CancelFunc
	// inners maps TierLabel → the backend that actually runs inference for that tier.
	// Populated at construction; one entry per roster entry.
	inners map[types.ModelTierLabel]backend.Backend

	// sem bounds concurrent Infer/InferStream calls to opts.MaxParallel,
	// matching llama-server's --parallel N. Buffered channel of that capacity.
	sem chan struct{}

	// loadMu serialises tier selection + load. Held across selectAndLoad's
	// decision-and-load so two concurrent requests (now possible with
	// MaxParallel > 1) can't both pass the "nothing loaded" check and spawn two
	// llama-server subprocesses. It does NOT cover inference itself — once a
	// tier is resident, N requests run through it concurrently.
	loadMu sync.Mutex

	mu           sync.Mutex
	loadedTier   *types.ModelDescriptor
	loadedAt     time.Time
	loadedLayers int // --n-gpu-layers chosen for the loaded tier (-1 full, 0 CPU)
	lastUsed     time.Time
	// inflight tracks how many Infer calls are active, so eviction can skip
	// while requests are in-flight.
	inflight int
	// contentionLogged is true while an external-pressure episode is ongoing, so
	// detectContention logs once per episode rather than per request.
	contentionLogged bool
}

// New constructs a Backend. roster must contain at least one descriptor and
// be sorted small→large by RequiredVRAMMB; New re-sorts to guarantee this.
// inners maps each TierLabel to the backend that performs real inference for
// that tier (e.g. a llamacpp.Backend). For testing, pass stub backends.
func New(
	modality backend.ModalityKind,
	roster []types.ModelDescriptor,
	vram VRAMProvider,
	inners map[types.ModelTierLabel]backend.Backend,
	opts Options,
) *Backend {
	r := make([]types.ModelDescriptor, len(roster))
	copy(r, roster)
	sort.Slice(r, func(i, j int) bool {
		return r[i].RequiredVRAMMB < r[j].RequiredVRAMMB
	})
	if opts.MaxParallel < 1 {
		opts.MaxParallel = 1
	}
	b := &Backend{
		modality: modality,
		roster:   r,
		vram:     vram,
		opts:     opts,
		inners:   inners,
		sem:      make(chan struct{}, opts.MaxParallel),
	}
	ctx, cancel := context.WithCancel(context.Background())
	b.cancel = cancel
	go b.runEviction(ctx)
	return b
}

func (b *Backend) Modality() backend.ModalityKind { return b.modality }

func (b *Backend) Ready() bool { return true }

// Infer selects the best-fit model tier, loads it if needed, detects
// contention, and runs inference. Concurrent calls are bounded to
// opts.MaxParallel by b.sem, matching llama-server's --parallel N.
func (b *Backend) Infer(ctx context.Context, req backend.Request) (backend.Response, error) {
	select {
	case b.sem <- struct{}{}:
		defer func() { <-b.sem }()
	case <-ctx.Done():
		return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "context cancelled before an inference slot was free", Cause: ctx.Err()}
	}

	b.mu.Lock()
	b.inflight++
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.inflight--
		b.mu.Unlock()
	}()

	slog.Info("job inferring",
		slog.String(logschema.FieldEvent, string(logschema.EventJobInferring)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, req.RequestID),
		slog.String(logschema.FieldModality, string(req.Modality)),
	)

	// Select, infer, and — on a real load failure / OOM at subprocess start —
	// evict and cascade to the next-smaller tier once (bounded by the roster).
	// forcedNext carries the cascade target between iterations (nil = free choice).
	var forcedNext *types.ModelDescriptor
	for attempt := 0; attempt < len(b.roster); attempt++ {
		desc, inner, err := b.selectInner(ctx, req, forcedNext)
		if err != nil {
			return backend.Response{}, err
		}

		resp, err := inner.Infer(ctx, req)
		if err == nil {
			resp.ModelTier = string(desc.TierLabel)
			b.markDegraded(&resp, req, desc)
			return resp, nil
		}

		next := b.oomFallbackTier(req, desc, err)
		if next == nil {
			return backend.Response{}, err
		}
		slog.Warn("model load failed, cascading to smaller tier",
			slog.String(logschema.FieldEvent, string(logschema.EventModelOOM)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String("from", string(desc.TierLabel)),
			slog.String("to", string(next.TierLabel)),
		)
		b.mu.Lock()
		toStop := b.evictLocked()
		b.mu.Unlock()
		stopInner(toStop)
		forcedNext = next
		slog.Info("model fallback",
			slog.String(logschema.FieldEvent, string(logschema.EventModelFallback)),
			slog.String(logschema.FieldModelTier, string(next.TierLabel)),
		)
	}
	// Roster exhausted without a working tier.
	return backend.Response{}, &backend.BackendError{Code: "overloaded", Message: "all model tiers failed to load"}
}

// oomFallbackTier returns the next-smaller tier to try when err is a real
// load/OOM failure, or nil when the error is not retryable, a pin forbids
// cascading, or a min_tier floor / the smallest tier has been reached.
func (b *Backend) oomFallbackTier(req backend.Request, current *types.ModelDescriptor, err error) *types.ModelDescriptor {
	be, ok := err.(*backend.BackendError)
	if !ok || (be.Code != "model_load_failed" && be.Code != "oom") {
		return nil
	}
	if req.PreferredTier != "" {
		return nil // an explicit pin does not cascade
	}
	next := b.tierBelow(current.TierLabel)
	if next == nil {
		return nil
	}
	if req.MinTier != "" && tierRank(next.TierLabel) < tierRank(types.ModelTierLabel(req.MinTier)) {
		return nil
	}
	return next
}

// InferStream selects a tier (running the same min-tier and speed-floor checks
// as Infer), then delegates to the inner backend's streaming path. If the inner
// backend does not implement backend.Streamer, it falls back to a single Infer
// call and emits the whole output as one delta. Implements backend.Streamer.
func (b *Backend) InferStream(ctx context.Context, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	select {
	case b.sem <- struct{}{}:
		defer func() { <-b.sem }()
	case <-ctx.Done():
		return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "context cancelled before an inference slot was free", Cause: ctx.Err()}
	}

	b.mu.Lock()
	b.inflight++
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.inflight--
		b.mu.Unlock()
	}()

	slog.Info("job inferring (stream)",
		slog.String(logschema.FieldEvent, string(logschema.EventJobInferring)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, req.RequestID),
		slog.String(logschema.FieldModality, string(req.Modality)),
	)

	desc, inner, err := b.selectInner(ctx, req, nil)
	if err != nil {
		return backend.Response{}, err
	}

	var resp backend.Response
	if streamer, ok := inner.(backend.Streamer); ok {
		resp, err = streamer.InferStream(ctx, req, chunkFn)
	} else {
		resp, err = inner.Infer(ctx, req)
		if err == nil {
			if resp.Reasoning != "" {
				chunkFn(backend.ChunkReasoning, resp.Reasoning)
			}
			if resp.Output != "" {
				chunkFn(backend.ChunkContent, resp.Output)
			}
		}
	}
	if err != nil {
		return backend.Response{}, err
	}
	resp.ModelTier = string(desc.TierLabel)
	b.markDegraded(&resp, req, desc)
	return resp, nil
}

// selectInner runs tier selection, the min-tier check, and the speed-floor
// cascade shared by Infer and InferStream, returning the chosen descriptor and
// its inner backend. On a speed-floor miss it drops one tier and re-selects,
// bounded by the roster size and capped by min_tier; if it bottoms out it
// serves on the smallest allowed tier (degraded) rather than erroring.
//
// If req.PreferredTier is set, it pins selection to exactly that tier for the
// whole call — validated up front, re-asserted every loop iteration so the
// speed-floor cascade can never move off it, and the cascade itself is
// skipped entirely once pinned (a pin means "this tier, full stop").
//
// initialForced, when non-nil, seeds the first selectAndLoad with a specific
// tier (the OOM cascade in Infer uses it to retry on the next-smaller tier). A
// PreferredTier pin takes precedence over it.
func (b *Backend) selectInner(ctx context.Context, req backend.Request, initialForced *types.ModelDescriptor) (*types.ModelDescriptor, backend.Backend, error) {
	var pinned *types.ModelDescriptor
	if req.PreferredTier != "" {
		desc, ok := b.tierByLabel(types.ModelTierLabel(req.PreferredTier))
		if !ok {
			return nil, nil, &backend.BackendError{
				Code:    "invalid_request",
				Message: fmt.Sprintf("preferred_tier %q is not in the model roster", req.PreferredTier),
			}
		}
		pinned = desc
	}

	// forced pins the next selectAndLoad to a specific tier: the pinned tier for
	// a PreferredTier request, the OOM-cascade seed, or the next-smaller tier
	// chosen by the speed-floor cascade below. Carried as a local (not a shared
	// field) so concurrent selectInner calls can't clobber one another's choice.
	forced := pinned
	if forced == nil {
		forced = initialForced
	}

	for attempt := 0; attempt < len(b.roster); attempt++ {
		desc, err := b.selectAndLoad(ctx, forced)
		if err != nil {
			return nil, nil, err
		}

		if req.MinTier != "" {
			if err := enforceMinTier(desc.TierLabel, types.ModelTierLabel(req.MinTier)); err != nil {
				return nil, nil, err
			}
		}

		b.detectContention(desc)

		inner, ok := b.inners[desc.TierLabel]
		if !ok {
			return nil, nil, &backend.BackendError{
				Code:    "internal_error",
				Message: fmt.Sprintf("no inner backend registered for tier %q", desc.TierLabel),
			}
		}

		if pinned != nil {
			// Pinned: never cascade away from the requested tier.
			return desc, inner, nil
		}

		m, measurable := inner.(backend.Measurable)
		if !measurable {
			return desc, inner, nil
		}
		tps := m.TokPerSec()
		if checkSpeedFloor(req.Priority, tps) == nil {
			return desc, inner, nil
		}

		next := b.tierBelow(desc.TierLabel)
		belowMin := req.MinTier != "" && next != nil &&
			tierRank(next.TierLabel) < tierRank(types.ModelTierLabel(req.MinTier))
		if next == nil || belowMin {
			slog.Info("speed floor: no lower tier, serving degraded",
				slog.String(logschema.FieldEvent, string(logschema.EventSpeedFloor)),
				slog.String(logschema.FieldCorrelationID, req.CorrelationID),
				slog.String("priority", req.Priority),
				slog.Float64(logschema.FieldTokensPerSec, tps),
			)
			return desc, inner, nil
		}

		slog.Info("speed floor: cascading to lower tier",
			slog.String(logschema.FieldEvent, string(logschema.EventSpeedFloor)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String("priority", req.Priority),
			slog.Float64(logschema.FieldTokensPerSec, tps),
			slog.String("from", string(desc.TierLabel)),
			slog.String("to", string(next.TierLabel)),
		)
		forced = next
	}

	// Roster exhausted (should be unreachable — the loop returns on the last
	// tier). Fall back to a plain selection.
	desc, err := b.selectAndLoad(ctx, forced)
	if err != nil {
		return nil, nil, err
	}
	inner := b.inners[desc.TierLabel]
	return desc, inner, nil
}

// tierBelow returns the next-smaller roster entry (roster is sorted small→large),
// or nil when label is already the smallest tier.
func (b *Backend) tierBelow(label types.ModelTierLabel) *types.ModelDescriptor {
	for i, d := range b.roster {
		if d.TierLabel == label {
			if i == 0 {
				return nil
			}
			return &b.roster[i-1]
		}
	}
	return nil
}

// markDegraded sets QualityDegraded and emits the event when a high-priority
// request landed on the weak tier.
func (b *Backend) markDegraded(resp *backend.Response, req backend.Request, desc *types.ModelDescriptor) {
	if req.Priority == "high" && desc.TierLabel == types.TierWeak {
		resp.QualityDegraded = true
		slog.Info("quality degraded",
			slog.String(logschema.FieldEvent, string(logschema.EventQualityDegraded)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldModelTier, string(desc.TierLabel)),
		)
		b.notify("Quality Degraded: high-priority request served on weak tier")
	}
}

// notify sends a message to opts.NotifyCh without blocking.
func (b *Backend) notify(msg string) {
	if b.opts.NotifyCh == nil {
		return
	}
	select {
	case b.opts.NotifyCh <- msg:
	default:
	}
}

// Shutdown stops the eviction loop, evicts any loaded model, and shuts down
// all inner backends.
func (b *Backend) Shutdown(ctx context.Context) error {
	b.cancel()
	b.mu.Lock()
	b.loadedTier = nil
	b.loadedLayers = 0
	b.loadedAt = time.Time{}
	b.mu.Unlock()

	for _, inner := range b.inners {
		inner.Shutdown(ctx) //nolint:errcheck
	}
	return nil
}

// selectAndLoad picks a tier that fits after partial GPU/CPU offload and ensures
// it is loaded. Must NOT be called while holding mu.
//
// Admission: for each tier (largest first) it computes the GPU/CPU layer split
// for the current free VRAM and admits the tier when the resulting GPU-resident
// footprint plus a BufferPct% margin fits. A tier is only rejected outright when
// even a CPU-only load (weights in system RAM) won't fit its KV/compute buffer.
// A non-nil forced (from a PreferredTier pin, the OOM cascade, or the speed-floor
// cascade in selectInner) pins the choice to that tier.
func (b *Backend) selectAndLoad(ctx context.Context, forced *types.ModelDescriptor) (*types.ModelDescriptor, error) {
	// Serialise the whole select-and-load decision: with MaxParallel > 1 two
	// concurrent requests could otherwise both observe "nothing loaded", both
	// evict, and both spawn a subprocess. The common case (a warm model that
	// still fits) returns before any load, so this lock is uncontended once a
	// tier is resident.
	b.loadMu.Lock()
	defer b.loadMu.Unlock()

	// If a model is already loaded (and no cascade is forcing a different tier),
	// check if it can still be reused. NVML free VRAM excludes the loaded model's
	// own allocation, so add its GPU-resident footprint back to get "effective
	// headroom" — what VRAM would look like if we reloaded right now.
	b.mu.Lock()
	if b.loadedTier != nil && (forced == nil || forced.TierLabel == b.loadedTier.TierLabel) {
		loaded := b.loadedTier
		loadedLayers := b.loadedLayers
		b.mu.Unlock()

		availRaw, err := b.vram.AvailableMB()
		if err != nil {
			return nil, &backend.BackendError{Code: "unavailable", Message: "VRAM query failed", Cause: err}
		}
		if availRaw < 0 {
			b.mu.Lock()
			b.lastUsed = time.Now()
			b.mu.Unlock()
			return loaded, nil
		}
		if effectiveHeadroomFits(loaded, loadedLayers, b.opts.MaxParallel, availRaw, b.opts.BufferPct) {
			b.mu.Lock()
			b.lastUsed = time.Now()
			b.mu.Unlock()
			return loaded, nil
		}
		slog.Info("loaded model no longer fits, evicting for reselection",
			slog.String(logschema.FieldEvent, string(logschema.EventModelEvicting)),
			slog.String(logschema.FieldModelTier, string(loaded.TierLabel)),
			slog.Int64(logschema.FieldVRAMAvailMB, availRaw),
		)
		b.mu.Lock()
		toStop := b.evictLocked()
		b.mu.Unlock()
		stopInner(toStop)
	} else {
		b.mu.Unlock()
	}

	availRaw, err := b.vram.AvailableMB()
	if err != nil {
		return nil, &backend.BackendError{Code: "unavailable", Message: "VRAM query failed", Cause: err}
	}
	unlimited := availRaw < 0

	slog.Info("vram check",
		slog.String(logschema.FieldEvent, string(logschema.EventVRAMCheck)),
		slog.Int64(logschema.FieldVRAMAvailMB, availRaw),
	)

	var chosen *types.ModelDescriptor
	var chosenLayers int
	matches := func(d types.ModelDescriptor) bool {
		return forced == nil || d.TierLabel == forced.TierLabel
	}

	switch {
	case unlimited:
		for i := len(b.roster) - 1; i >= 0; i-- {
			if matches(b.roster[i]) {
				chosen, chosenLayers = &b.roster[i], -1
				break
			}
		}
	default:
		// Pass 1, large→small: prefer the largest tier that fits with at least a
		// partial GPU offload — a bigger model on GPU beats a smaller one.
		for i := len(b.roster) - 1; i >= 0 && chosen == nil; i-- {
			d := b.roster[i]
			if !matches(d) {
				continue
			}
			layers := FitLayers(d.RequiredVRAMMB, ScaledKVOverheadMB(d, b.opts.MaxParallel), availRaw, d.TotalLayers, b.opts.BufferPct)
			if layers != 0 {
				chosen, chosenLayers = &b.roster[i], layers
			}
		}
		// Pass 2, small→large: nothing fits on GPU — fall back to a CPU-only load
		// of the smallest tier whose overhead still fits (smaller = faster on CPU).
		for i := 0; i < len(b.roster) && chosen == nil; i++ {
			d := b.roster[i]
			if !matches(d) {
				continue
			}
			if fitsWithBuffer(ScaledKVOverheadMB(d, b.opts.MaxParallel), availRaw, b.opts.BufferPct) {
				chosen, chosenLayers = &b.roster[i], 0
			}
		}
	}
	if chosen == nil {
		return nil, &backend.BackendError{
			Code:    "overloaded",
			Message: fmt.Sprintf("no model tier fits within %d MB VRAM even with CPU offload", availRaw),
		}
	}

	b.mu.Lock()
	if b.loadedTier != nil && b.loadedTier.TierLabel == chosen.TierLabel {
		b.lastUsed = time.Now()
		b.mu.Unlock()
		return b.loadedTier, nil
	}
	toStop := b.evictLocked()
	b.mu.Unlock()
	stopInner(toStop)

	// Attempt to load chosen tier; retry with next-smaller on OOM.
	chosenIdx := b.rosterIndex(chosen.TierLabel)
	for idx := chosenIdx; idx >= 0; idx-- {
		d := &b.roster[idx]
		layers := chosenLayers
		if idx != chosenIdx && !unlimited {
			layers = FitLayers(d.RequiredVRAMMB, ScaledKVOverheadMB(*d, b.opts.MaxParallel), availRaw, d.TotalLayers, b.opts.BufferPct)
		}
		loadErr := b.loadModel(d, layers)
		if loadErr == nil {
			return d, nil
		}
		var oom *oomError
		if !errors.As(loadErr, &oom) {
			return nil, &backend.BackendError{Code: "unavailable", Message: "model load failed", Cause: loadErr}
		}
		slog.Warn("model OOM",
			slog.String(logschema.FieldEvent, string(logschema.EventModelOOM)),
			slog.String(logschema.FieldModelTier, string(d.TierLabel)),
		)
		if idx > 0 {
			slog.Info("model fallback",
				slog.String(logschema.FieldEvent, string(logschema.EventModelFallback)),
				slog.String(logschema.FieldModelTier, string(b.roster[idx-1].TierLabel)),
			)
		}
	}
	return nil, &backend.BackendError{Code: "overloaded", Message: "all model tiers OOM"}
}

// fitsWithBuffer reports whether resident MB plus a pct% safety margin fits
// within avail MB.
func fitsWithBuffer(resident, avail int64, pct int) bool {
	return resident+resident*int64(pct)/100 <= avail
}

// effectiveHeadroomFits reports whether the currently loaded model would still
// fit if it were reloaded right now. NVML free VRAM (availRaw) already excludes
// the loaded model's own allocation, so its GPU-resident footprint is added back
// to get "effective headroom" — the headroom a fresh load would actually see.
// Shared by selectAndLoad's reuse check and the eviction loop's pressure check
// so the two never disagree about whether a healthy model is under pressure.
// slots is the --parallel N concurrency the KV cache is sized for.
func effectiveHeadroomFits(loaded *types.ModelDescriptor, loadedLayers, slots int, availRaw int64, bufferPct int) bool {
	resident := GPUResidentMB(loaded.RequiredVRAMMB, ScaledKVOverheadMB(*loaded, slots), loadedLayers, loaded.TotalLayers)
	return fitsWithBuffer(resident, availRaw+resident, bufferPct)
}

// gpuLayerSink is implemented by an inner backend (llamacpp.Backend) that can be
// told exactly how many layers to offload, so the count the subprocess actually
// runs with matches the FitLayers value selectAndLoad admitted the tier on.
type gpuLayerSink interface{ SetNextGPULayers(int) }

// loadModel records that we've selected this tier (and the GPU layer split
// chosen for it), pushes that split to the inner backend, and applies the
// OOMSimulator hook for tests. Actual subprocess startup happens lazily inside
// the inner backend on the first Infer call.
func (b *Backend) loadModel(d *types.ModelDescriptor, gpuLayers int) error {
	if b.opts.OOMSimulator != nil && b.opts.OOMSimulator(*d) {
		return &oomError{tier: string(d.TierLabel)}
	}

	if sink, ok := b.inners[d.TierLabel].(gpuLayerSink); ok {
		sink.SetNextGPULayers(gpuLayers)
	}

	now := time.Now()
	b.mu.Lock()
	b.loadedTier = d
	b.loadedLayers = gpuLayers
	b.loadedAt = now
	b.lastUsed = now
	b.mu.Unlock()

	return nil
}

// evictLocked evicts the currently loaded model, shutting down its subprocess
// so it stops holding VRAM and RAM. Caller must hold mu; the inner Shutdown is
// run in a goroutine so the lock is not held across a process kill.
// evictLocked clears the loaded-tier bookkeeping and returns the inner backend
// whose subprocess still needs to be stopped (nil if none). Callers MUST hold
// b.mu, and MUST call stopInner with the returned value AFTER releasing b.mu —
// stopping the subprocess is synchronous and slow (SIGINT + wait), so it must
// not race a concurrent selectAndLoad that has already seen loadedTier == nil.
func (b *Backend) evictLocked() backend.Backend {
	if b.loadedTier == nil {
		return nil
	}
	tier := b.loadedTier.TierLabel
	slog.Info("model evicting",
		slog.String(logschema.FieldEvent, string(logschema.EventModelEvicting)),
		slog.String(logschema.FieldModelTier, string(tier)),
	)
	inner := b.inners[tier]
	b.loadedTier = nil
	b.loadedLayers = 0
	b.loadedAt = time.Time{}
	slog.Info("model evicted",
		slog.String(logschema.FieldEvent, string(logschema.EventModelEvicted)),
	)
	return inner
}

// stopInner synchronously shuts down an inner backend returned by evictLocked.
// Safe to call with nil. Must NOT be called while holding b.mu. Returns the
// inner's Shutdown error (logged here) so the eviction loop can force a reaper
// sweep when a subprocess teardown was incomplete.
func stopInner(inner backend.Backend) error {
	if inner == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err := inner.Shutdown(ctx)
	if err != nil {
		slog.Error("vram: inner backend shutdown reported incomplete teardown — will sweep", slog.Any("err", err))
	}
	return err
}

// detectContention checks whether available VRAM has dropped below the
// expected headroom for the currently loaded model and logs a warning.
// It never blocks or aborts the request.
func (b *Backend) detectContention(desc *types.ModelDescriptor) {
	avail, err := b.vram.AvailableMB()
	if err != nil || avail < 0 {
		return
	}
	// Fire only on genuine external pressure: the loaded model would no longer
	// fit even after crediting back its own allocation (effective headroom), and
	// free VRAM has collapsed below the absolute floor. The old
	// avail < RequiredVRAMMB/2 heuristic fired on almost every request right
	// after a load on an 8 GB card (our own model legitimately consumed the VRAM).
	b.mu.Lock()
	loadedLayers := b.loadedLayers
	logged := b.contentionLogged
	b.mu.Unlock()

	underPressure := !effectiveHeadroomFits(desc, loadedLayers, b.opts.MaxParallel, avail, b.opts.BufferPct) &&
		avail < externalPressureFloorMB

	if underPressure && !logged {
		slog.Warn("contention detected",
			slog.String(logschema.FieldEvent, string(logschema.EventContention)),
			slog.Int64(logschema.FieldVRAMAvailMB, avail),
			slog.Int64(logschema.FieldVRAMRequiredMB,
				GPUResidentMB(desc.RequiredVRAMMB, ScaledKVOverheadMB(*desc, b.opts.MaxParallel), loadedLayers, desc.TotalLayers)),
		)
	}
	b.mu.Lock()
	b.contentionLogged = underPressure
	b.mu.Unlock()
}

// tierByLabel returns the roster descriptor for label, or ok=false if no such
// tier exists — used to validate an untrusted, caller-supplied tier label
// (e.g. req.PreferredTier) without panicking.
func (b *Backend) tierByLabel(label types.ModelTierLabel) (*types.ModelDescriptor, bool) {
	for i, d := range b.roster {
		if d.TierLabel == label {
			return &b.roster[i], true
		}
	}
	return nil, false
}

// rosterIndex returns the index of the tier with the given label.
// Panics if not found — roster is validated at construction.
func (b *Backend) rosterIndex(label types.ModelTierLabel) int {
	for i, d := range b.roster {
		if d.TierLabel == label {
			return i
		}
	}
	panic(fmt.Sprintf("vram: tier %q not found in roster", label))
}

// tierRank maps a tier label to a numeric rank for comparison (higher = better).
func tierRank(label types.ModelTierLabel) int {
	switch label {
	case types.TierStrong:
		return 2
	case types.TierMid:
		return 1
	default: // TierWeak or unrecognised
		return 0
	}
}

// enforceMinTier returns an overloaded BackendError if the selected tier is
// below the caller's requested minimum quality floor.
func enforceMinTier(selected, minTier types.ModelTierLabel) error {
	if tierRank(selected) < tierRank(minTier) {
		return &backend.BackendError{
			Code:    "overloaded",
			Message: fmt.Sprintf("selected tier %q is below min_tier %q", selected, minTier),
		}
	}
	return nil
}

// checkSpeedFloor returns an overloaded BackendError if the current tok/sec
// rate is known (> 0) and falls below the floor required for the given priority.
// If tokPerSec == 0 (no samples yet) the check is skipped — benefit of the doubt.
func checkSpeedFloor(priority string, tokPerSec float64) error {
	if tokPerSec == 0 {
		return nil
	}
	var floor float64
	switch priority {
	case "high":
		floor = 8
	case "normal":
		floor = 4
	default:
		return nil
	}
	if tokPerSec < floor {
		return &backend.BackendError{
			Code:    "overloaded",
			Message: fmt.Sprintf("tok/sec %.1f below floor %.0f for priority %q", tokPerSec, floor, priority),
		}
	}
	return nil
}

// LoadedModelSnapshot is a point-in-time view of the currently loaded model.
type LoadedModelSnapshot struct {
	Descriptor types.ModelDescriptor
	LoadedAt   time.Time
	// GPULayers is the --n-gpu-layers value in effect: -1 full GPU, 0 CPU-only,
	// or a partial count. TotalLayers is the model's transformer block count.
	GPULayers   int
	TotalLayers int
}

// residentReporter is implemented by inner backends that can report the VRAM
// their running subprocess currently occupies.
type residentReporter interface {
	ResidentMB() int64
}

// busyReporter is implemented by inner backends that can be mid-lifecycle
// (e.g. a llamacpp.Backend restarting its subprocess after a connection error).
// The eviction loop must not tear such a backend down: shutting down a process
// that is being replaced races the restart and leaves a dead port behind.
type busyReporter interface {
	Restarting() bool
}

// innerBusy reports whether the inner backend for tier must not be evicted right
// now because it is mid-restart. A missing entry or a backend that doesn't
// implement busyReporter is treated as not busy.
func (b *Backend) innerBusy(tier types.ModelTierLabel) bool {
	inner, ok := b.inners[tier]
	if !ok {
		return false
	}
	br, ok := inner.(busyReporter)
	return ok && br.Restarting()
}

// InFlight is the number of inference requests currently holding a slot in the
// concurrency semaphore (0..MaxParallel). Satisfies server.PressureSource.
func (b *Backend) InFlight() int { return len(b.sem) }

// MaxParallel is the configured concurrent-inference ceiling, matching
// llama-server's --parallel N. Satisfies server.PressureSource.
func (b *Backend) MaxParallel() int { return cap(b.sem) }

// SelfVRAMMB is the total VRAM currently held by this backend's own running
// subprocesses. Used for the dashboard's server-vs-other-apps breakdown.
func (b *Backend) SelfVRAMMB() int64 {
	var sum int64
	for _, inner := range b.inners {
		if r, ok := inner.(residentReporter); ok {
			sum += r.ResidentMB()
		}
	}
	return sum
}

// TokPerSec returns the tok/sec from the currently loaded tier's inner backend,
// or 0 if no model is loaded or the inner backend doesn't implement Measurable.
// Satisfies backend.Measurable and server.PressureSource.
func (b *Backend) TokPerSec() float64 {
	b.mu.Lock()
	loaded := b.loadedTier
	b.mu.Unlock()
	if loaded == nil {
		return 0
	}
	inner, ok := b.inners[loaded.TierLabel]
	if !ok {
		return 0
	}
	m, ok := inner.(backend.Measurable)
	if !ok {
		return 0
	}
	return m.TokPerSec()
}

// tokenizer is the subset of llamacpp.Backend used for the context-meter proxy.
type tokenizer interface {
	Tokenize(ctx context.Context, text string) (int, error)
	NCtx(ctx context.Context) (int, error)
}

// tokenizerInner returns an inner backend that can tokenize: the currently
// loaded tier if it supports it, otherwise the largest roster tier (last entry,
// roster is sorted small→large).
// tokenizerInner returns the currently-loaded tier's inner if it can tokenize.
// It deliberately does NOT fall back to an unloaded roster tier: proxying to one
// would trigger a full model spawn (30–120 s, VRAM) just to answer a token
// count. Callers get ErrNoModelLoaded and should serve a cheap static answer.
func (b *Backend) tokenizerInner() tokenizer {
	b.mu.Lock()
	loaded := b.loadedTier
	b.mu.Unlock()
	if loaded != nil {
		if tk, ok := b.inners[loaded.TierLabel].(tokenizer); ok {
			return tk
		}
	}
	return nil
}

// ErrNoModelLoaded is returned by Tokenize/NCtx when no model is resident, so
// the HTTP layer can answer with a cached default instead of forcing a load.
var ErrNoModelLoaded = &backend.BackendError{Code: "unavailable", Message: "no model is loaded"}

// Tokenize counts tokens using the currently-loaded inner backend. Returns
// ErrNoModelLoaded (never starts a model) if none is resident.
func (b *Backend) Tokenize(ctx context.Context, text string) (int, error) {
	tk := b.tokenizerInner()
	if tk == nil {
		return 0, ErrNoModelLoaded
	}
	return tk.Tokenize(ctx, text)
}

// NCtx returns the loaded model's context size. Returns ErrNoModelLoaded (never
// starts a model) if none is resident.
func (b *Backend) NCtx(ctx context.Context) (int, error) {
	tk := b.tokenizerInner()
	if tk == nil {
		return 0, ErrNoModelLoaded
	}
	return tk.NCtx(ctx)
}

// LoadedTier returns the tier label of the currently loaded model, or "" if
// nothing is loaded. Satisfies server.PressureSource.
func (b *Backend) LoadedTier() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.loadedTier == nil {
		return ""
	}
	return string(b.loadedTier.TierLabel)
}

// LoadedModel returns a snapshot of the currently loaded model, or nil if
// nothing is loaded. Safe to call from any goroutine.
func (b *Backend) LoadedModel() *LoadedModelSnapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.loadedTier == nil {
		return nil
	}
	total := b.loadedTier.TotalLayers
	if total <= 0 {
		total = defaultTotalLayers
	}
	return &LoadedModelSnapshot{
		Descriptor:  *b.loadedTier,
		LoadedAt:    b.loadedAt,
		GPULayers:   b.loadedLayers,
		TotalLayers: total,
	}
}
