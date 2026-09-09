//go:build windows

// nvml-check verifies that nvml.dll can be loaded and queried for real VRAM.
// Uses Windows syscall — no CGo or external dependencies required.
package main

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// nvmlMemory matches the C struct nvmlMemory_t layout.
type nvmlMemory struct {
	Total uint64
	Free  uint64
	Used  uint64
}

func main() {
	dll, err := syscall.LoadDLL("nvml.dll")
	if err != nil {
		fmt.Fprintf(os.Stderr, "LoadDLL(nvml.dll) failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "Ensure NVIDIA drivers are installed and nvml.dll is in System32.\n")
		os.Exit(1)
	}
	defer dll.Release()

	init_, err := dll.FindProc("nvmlInit_v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlInit_v2) failed: %v\n", err)
		os.Exit(1)
	}
	shutdown, err := dll.FindProc("nvmlShutdown")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlShutdown) failed: %v\n", err)
		os.Exit(1)
	}
	getCount, err := dll.FindProc("nvmlDeviceGetCount_v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlDeviceGetCount_v2) failed: %v\n", err)
		os.Exit(1)
	}
	getHandle, err := dll.FindProc("nvmlDeviceGetHandleByIndex_v2")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlDeviceGetHandleByIndex_v2) failed: %v\n", err)
		os.Exit(1)
	}
	getName, err := dll.FindProc("nvmlDeviceGetName")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlDeviceGetName) failed: %v\n", err)
		os.Exit(1)
	}
	getMemInfo, err := dll.FindProc("nvmlDeviceGetMemoryInfo")
	if err != nil {
		fmt.Fprintf(os.Stderr, "FindProc(nvmlDeviceGetMemoryInfo) failed: %v\n", err)
		os.Exit(1)
	}

	if ret, _, _ := init_.Call(); ret != 0 {
		fmt.Fprintf(os.Stderr, "nvmlInit_v2 returned error code %d\n", ret)
		os.Exit(1)
	}
	defer shutdown.Call()

	var count uint32
	if ret, _, _ := getCount.Call(uintptr(unsafe.Pointer(&count))); ret != 0 {
		fmt.Fprintf(os.Stderr, "nvmlDeviceGetCount_v2 returned error code %d\n", ret)
		os.Exit(1)
	}

	for i := uint32(0); i < count; i++ {
		var handle uintptr
		if ret, _, _ := getHandle.Call(uintptr(i), uintptr(unsafe.Pointer(&handle))); ret != 0 {
			fmt.Fprintf(os.Stderr, "GPU %d: GetHandleByIndex returned error code %d\n", i, ret)
			continue
		}

		nameBuf := make([]byte, 96)
		getName.Call(handle, uintptr(unsafe.Pointer(&nameBuf[0])), uintptr(len(nameBuf)))
		name := ""
		for j, b := range nameBuf {
			if b == 0 {
				name = string(nameBuf[:j])
				break
			}
		}

		var mem nvmlMemory
		if ret, _, _ := getMemInfo.Call(handle, uintptr(unsafe.Pointer(&mem))); ret != 0 {
			fmt.Fprintf(os.Stderr, "GPU %d: GetMemoryInfo returned error code %d\n", i, ret)
			continue
		}

		fmt.Printf("GPU %d: %s\n", i, name)
		fmt.Printf("  Total VRAM : %d MB\n", mem.Total>>20)
		fmt.Printf("  Used VRAM  : %d MB\n", mem.Used>>20)
		fmt.Printf("  Free VRAM  : %d MB\n", mem.Free>>20)
	}
}
