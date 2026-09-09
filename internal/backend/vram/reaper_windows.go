//go:build windows

package vram

import (
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/windows"
)

type defaultPortOwner struct{}

// pidOnPort parses `netstat -ano` for a LISTENING socket on 127.0.0.1:<port>.
// (Left as a shell-out — it identified the orphan correctly in the incident and
// GetExtendedTcpTable is a much larger surface.)
func (defaultPortOwner) pidOnPort(port int) int {
	out, err := exec.Command("netstat", "-ano", "-p", "tcp").Output()
	if err != nil {
		return 0
	}
	suffix := ":" + strconv.Itoa(port)
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 5 || !strings.EqualFold(f[3], "LISTENING") {
			continue
		}
		if !strings.HasSuffix(f[1], suffix) {
			continue
		}
		if pid, err := strconv.Atoi(f[4]); err == nil {
			return pid
		}
	}
	return 0
}

// imageName returns the lowercased executable base name for a PID via the
// QueryFullProcessImageName syscall. A `tasklist` outage can no longer forge a
// false "" for a live process (the bug that let the Reaper skip a real orphan).
func (defaultPortOwner) imageName(pid int) string {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return ""
	}
	defer windows.CloseHandle(h)

	buf := make([]uint16, windows.MAX_PATH)
	n := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(h, 0, &buf[0], &n); err != nil {
		return ""
	}
	return strings.ToLower(filepath.Base(windows.UTF16ToString(buf[:n])))
}

// kill terminates a PID via TerminateProcess. Access-denied here still means the
// stray was started by a more-privileged session — surfaced as a real Win32
// error rather than a `taskkill` exit code.
func (defaultPortOwner) kill(pid int) error {
	h, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	return windows.TerminateProcess(h, 1)
}
