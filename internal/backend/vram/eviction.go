package vram

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/logschema"
)

// externalPressureFloorMB is the free-VRAM level below which the idle eviction
// loop treats the situation as a genuine external grab worth evicting for. It is
// sized to the largest tier's KV/compute buffer: once free VRAM drops under
// this, even a CPU-only reload's on-GPU buffer is at risk. Above it, low free
// VRAM is almost always just our own resident model and must not trigger an
// evict (see effectiveHeadroomFits).
const externalPressureFloorMB = 800

// runEviction is the background eviction loop. It ticks every EvictionInterval
// and evicts the loaded model if either:
//
//	(a) the model has been idle longer than EvictionIdleTimeout, or
//	(b) available VRAM has dropped below what the loaded model needs (external pressure).
//
// In-flight requests are never interrupted: the loop skips its eviction check
// while inflight > 0.
// reaperSweepEvery throttles the reaper: it shells out to netstat/tasklist,
// which can take seconds, so run it at most once per N eviction ticks.
const reaperSweepEvery = 6

func (b *Backend) runEviction(ctx context.Context) {
	ticker := time.NewTicker(b.opts.EvictionInterval)
	defer ticker.Stop()

	var sweeping atomic.Bool
	tick := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			tick++
			b.mu.Lock()
			idle := b.loadedTier == nil
			busy := b.inflight > 0
			b.mu.Unlock()

			if idle {
				// Nothing of ours should be listening on a roster port — sweep
				// away any stray subprocess a crash or failed evict left behind.
				// Off the loop goroutine and throttled so a slow sweep can't
				// stall eviction.
				if b.opts.Reaper != nil && tick%reaperSweepEvery == 0 && sweeping.CompareAndSwap(false, true) {
					go func() {
						defer sweeping.Store(false)
						b.opts.Reaper.Sweep()
					}()
				}
				continue
			}
			if busy {
				continue
			}

			b.mu.Lock()
			if b.loadedTier == nil || b.inflight > 0 {
				b.mu.Unlock()
				continue
			}

			idleFor := time.Since(b.lastUsed)
			loaded := b.loadedTier
			loadedLayers := b.loadedLayers
			loadedVision := b.loadedVision

			if idleFor >= b.opts.EvictionIdleTimeout {
				toStop := b.evictLocked()
				b.mu.Unlock()
				_ = stopInner(toStop)
				// Always sweep right after an idle eviction — this is the exact
				// scenario where a failed teardown can leave an orphan holding
				// a roster port.
				b.maybeSweepAfterEvict(true, &sweeping)
				continue
			}

			// Check for external VRAM pressure while idle.
			b.mu.Unlock()

			avail, err := b.vram.AvailableMB()
			if err != nil || avail < 0 {
				continue
			}

			// Evict for external pressure only when BOTH hold:
			//   1. the loaded model would no longer fit even after counting its
			//      own allocation as reclaimable (effective headroom), and
			//   2. free VRAM has collapsed below an absolute floor — a real
			//      external grab, not just our own resident footprint.
			// Without (1) a healthy loaded model looks starved on every tick
			// (NVML free VRAM already excludes its allocation), producing an
			// endless load/evict/reload storm.
			headroomFits := effectiveHeadroomFits(loaded, loadedLayers, loadedVision, b.opts.MaxParallel, avail, b.opts.BufferPct)
			if !headroomFits && avail < externalPressureFloorMB {
				resident := GPUResidentMB(loaded.WeightMB(loadedVision), ScaledKVOverheadMB(*loaded, b.opts.MaxParallel, loadedVision), loadedLayers, loaded.TotalLayers)
				slog.Warn("vram pressure detected",
					slog.String(logschema.FieldEvent, string(logschema.EventVRAMPressure)),
					slog.Int64(logschema.FieldVRAMAvailMB, avail),
					slog.Int64(logschema.FieldVRAMRequiredMB, resident),
				)
				b.mu.Lock()
				// Re-check under lock: another goroutine may have evicted or
				// started a request, or the inner backend may be mid-restart.
				var toStop backend.Backend
				if b.loadedTier != nil && b.inflight == 0 && !b.innerBusy(b.loadedTier.TierLabel) {
					toStop = b.evictLocked()
				}
				b.mu.Unlock()
				if toStop != nil {
					err := stopInner(toStop)
					b.maybeSweepAfterEvict(err != nil, &sweeping)
				}
			}
		}
	}
}

// maybeSweepAfterEvict runs a reaper sweep off the eviction goroutine when
// force is set (idle-timeout eviction) or a subprocess teardown reported an
// error. Guarded by the shared `sweeping` atomic so at most one sweep is ever
// in flight (with the periodic tick%reaperSweepEvery sweep). Non-blocking.
func (b *Backend) maybeSweepAfterEvict(force bool, sweeping *atomic.Bool) {
	if b.opts.Reaper == nil || !force {
		return
	}
	if !sweeping.CompareAndSwap(false, true) {
		return
	}
	go func() {
		defer sweeping.Store(false)
		if n := b.opts.Reaper.Sweep(); n > 0 {
			slog.Warn("vram: post-eviction sweep killed a lingering subprocess", slog.Int("count", n))
		}
	}()
}
