//go:build !windows

package vram

// NVMLProvider on non-Windows platforms is a stub: this project's GPU headroom
// path is implemented via nvml.dll (Windows only). Both queries report "unknown"
// (-1), which the selection logic treats as "unlimited VRAM, full GPU offload".
// This keeps `go build ./...` and the test suite working on Linux/macOS CI.
type NVMLProvider struct{}

func (NVMLProvider) AvailableMB() (int64, error) { return -1, nil }

func (NVMLProvider) TotalMB() (int64, error) { return -1, nil }
