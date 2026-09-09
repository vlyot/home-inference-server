package vram

import (
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

// nvmlDLL is a lazily-initialised handle to nvml.dll, shared across all
// NVMLProvider calls. Errors on first load are cached so we fall back to -1
// without retrying on every request.
var (
	nvmlOnce sync.Once
	nvmlProc struct {
		init       *syscall.Proc
		shutdown   *syscall.Proc
		getCount   *syscall.Proc
		getHandle  *syscall.Proc
		getMemInfo *syscall.Proc
	}
	nvmlLoadErr error
)

func loadNVML() error {
	nvmlOnce.Do(func() {
		dll, err := syscall.LoadDLL("nvml.dll")
		if err != nil {
			nvmlLoadErr = fmt.Errorf("nvml.dll not found: %w", err)
			return
		}
		for _, entry := range []struct {
			name string
			dst  **syscall.Proc
		}{
			{"nvmlInit_v2", &nvmlProc.init},
			{"nvmlShutdown", &nvmlProc.shutdown},
			{"nvmlDeviceGetCount_v2", &nvmlProc.getCount},
			{"nvmlDeviceGetHandleByIndex_v2", &nvmlProc.getHandle},
			{"nvmlDeviceGetMemoryInfo", &nvmlProc.getMemInfo},
		} {
			p, err := dll.FindProc(entry.name)
			if err != nil {
				nvmlLoadErr = fmt.Errorf("nvml.dll missing %s: %w", entry.name, err)
				return
			}
			*entry.dst = p
		}
		if ret, _, _ := nvmlProc.init.Call(); ret != 0 {
			nvmlLoadErr = fmt.Errorf("nvmlInit_v2 returned %d", ret)
		}
	})
	return nvmlLoadErr
}

// NVMLProvider queries the system GPU via nvml.dll (Windows).
// Falls back to -1 (unlimited) if nvml.dll is unavailable — allows the server
// to run on machines without a GPU.
type NVMLProvider struct{}

func (NVMLProvider) AvailableMB() (int64, error) {
	if err := loadNVML(); err != nil {
		return -1, nil //nolint:nilerr // -1 signals "treat as unlimited"
	}

	var handle uintptr
	if ret, _, _ := nvmlProc.getHandle.Call(0, uintptr(unsafe.Pointer(&handle))); ret != 0 {
		return -1, nil
	}

	var mem nvmlMemory
	if ret, _, _ := nvmlProc.getMemInfo.Call(handle, uintptr(unsafe.Pointer(&mem))); ret != 0 {
		return -1, nil
	}

	return int64(mem.Free >> 20), nil
}

// TotalMB returns the GPU's total VRAM in megabytes, or -1 if NVML is
// unavailable.
func (NVMLProvider) TotalMB() (int64, error) {
	if err := loadNVML(); err != nil {
		return -1, nil //nolint:nilerr
	}
	var handle uintptr
	if ret, _, _ := nvmlProc.getHandle.Call(0, uintptr(unsafe.Pointer(&handle))); ret != 0 {
		return -1, nil
	}
	var mem nvmlMemory
	if ret, _, _ := nvmlProc.getMemInfo.Call(handle, uintptr(unsafe.Pointer(&mem))); ret != 0 {
		return -1, nil
	}
	return int64(mem.Total >> 20), nil
}
