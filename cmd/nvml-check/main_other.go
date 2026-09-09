//go:build !windows

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "nvml-check is Windows-only (queries nvml.dll).")
	os.Exit(1)
}
