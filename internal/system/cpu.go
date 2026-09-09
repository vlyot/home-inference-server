// Package system provides host-level metrics (CPU utilisation, etc.) for
// pressure-aware routing decisions.
package system

import (
	"context"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
)

const (
	pollInterval = time.Second
	ringSize     = 5
)

// CPUMonitor polls overall (whole-host, all cores) CPU utilisation once per
// second and exposes a rolling average over the last 5 samples.
//
// Note: Pct() is a whole-host figure. The pressure policy's "CPU > 90%" defer
// trigger is deliberately conservative — on a many-core desktop, GPU inference
// pins only a couple of cores so whole-host CPU rarely approaches 90%. The VRAM
// floor (< 500 MB) is the primary defer signal; CPU is a secondary guard.
// Pct() also returns 0 until the first sample lands (~1 s after start); callers
// should treat 0 as "unknown, not under pressure".
type CPUMonitor struct {
	mu   sync.Mutex
	ring [ringSize]float64
	n    int // total samples collected (capped at ringSize for avg)
	head int // index of the next write slot
}

// NewCPUMonitor starts the polling goroutine and returns the monitor.
// The goroutine stops when ctx is cancelled.
func NewCPUMonitor(ctx context.Context) *CPUMonitor {
	m := &CPUMonitor{}
	go m.run(ctx)
	return m
}

func (m *CPUMonitor) run(ctx context.Context) {
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pcts, err := cpu.Percent(0, false)
			if err != nil || len(pcts) == 0 {
				continue
			}
			m.mu.Lock()
			m.ring[m.head] = pcts[0]
			m.head = (m.head + 1) % ringSize
			if m.n < ringSize {
				m.n++
			}
			m.mu.Unlock()
		}
	}
}

// Pct returns the average CPU utilisation over the last up-to-5 samples.
// Returns 0 if no samples have been collected yet (see Ready).
func (m *CPUMonitor) Pct() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < m.n; i++ {
		sum += m.ring[i]
	}
	return sum / float64(m.n)
}

// Ready reports whether at least one sample has been collected. Until then Pct()
// returns a placeholder 0, which callers should not read as "idle".
func (m *CPUMonitor) Ready() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.n > 0
}
