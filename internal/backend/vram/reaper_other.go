//go:build !windows

package vram

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
)

type defaultPortOwner struct{}

// pidOnPort uses `lsof -ti tcp:<port> -sTCP:LISTEN`.
func (defaultPortOwner) pidOnPort(port int) int {
	out, err := exec.Command("lsof", "-ti", "tcp:"+strconv.Itoa(port), "-sTCP:LISTEN").Output()
	if err != nil {
		return 0
	}
	first := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
	pid, _ := strconv.Atoi(first)
	return pid
}

// imageName reads /proc/<pid>/comm (Linux) or falls back to `ps`.
func (defaultPortOwner) imageName(pid int) string {
	if b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/comm"); err == nil {
		return strings.ToLower(strings.TrimSpace(string(b)))
	}
	out, err := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "comm=").Output()
	if err != nil {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(string(out)))
}

func (defaultPortOwner) kill(pid int) error {
	p, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return p.Signal(syscall.SIGKILL)
}
