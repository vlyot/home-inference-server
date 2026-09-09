package vram

import "sync/atomic"

// VRAMProvider abstracts VRAM headroom queries so the backend can be tested
// without a physical GPU. The real implementation (NVMLProvider, Windows only)
// queries NVML; on other platforms it is a stub that reports "unknown"; the mock
// returns a configurable value for unit tests.
type VRAMProvider interface {
	// AvailableMB returns free VRAM in megabytes.
	// Returns -1 if the query is unsupported (no GPU, NVML unavailable, or a
	// non-Windows build).
	AvailableMB() (int64, error)
}

// nvmlMemory matches the nvmlMemory_t C struct layout used by nvml.dll.
type nvmlMemory struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// MockVRAMProvider is used in tests. FreeMB is the initial reading; tests that
// change the free VRAM mid-run must use SetFreeMB (not assign FreeMB directly)
// so the concurrent eviction-loop reader stays race-free.
type MockVRAMProvider struct {
	FreeMB   int64
	Err      error
	override atomic.Int64 // 0 = use FreeMB; otherwise the live value + 1
}

// SetFreeMB updates the reported free VRAM from any goroutine.
func (m *MockVRAMProvider) SetFreeMB(v int64) { m.override.Store(v + 1) }

func (m *MockVRAMProvider) AvailableMB() (int64, error) {
	if o := m.override.Load(); o != 0 {
		return o - 1, m.Err
	}
	return m.FreeMB, m.Err
}
