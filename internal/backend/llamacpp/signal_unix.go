//go:build !windows

package llamacpp

import (
	"os"
	"os/exec"
	"syscall"
)

// gracefulStop on Unix sends SIGINT so llama-server can shut down cleanly. The
// error return mirrors the Windows implementation (where the forced kill can
// fail); on Unix delivering SIGINT effectively never errors for our own child.
func gracefulStop(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Signal(syscall.SIGINT)
}

// pidAlive reports whether a PID is a running process (Unix): signal 0 probes
// existence without delivering anything.
func pidAlive(pid int) bool {
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}
