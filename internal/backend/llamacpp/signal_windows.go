//go:build windows

package llamacpp

import (
	"errors"
	"log/slog"
	"os/exec"
	"time"

	"golang.org/x/sys/windows"
)

// stillActive is Win32 STILL_ACTIVE (259) — GetExitCodeProcess returns this for
// a process that has not exited. x/sys/windows has no named constant for it.
const stillActive = 259

// pidAlive reports whether pid is a running process (Windows). It uses a
// PROCESS_QUERY_LIMITED_INFORMATION handle + GetExitCodeProcess — no shell-out,
// no signal — so a flaky `tasklist` can no longer make a dead process look
// alive (the bug that froze self_vram_mb on the dashboard for 24 minutes).
func pidAlive(pid int) bool {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		// The PID does not exist (recycled or never was) — must not be reported
		// alive as some stranger's process. Access-denied on our own
		// same-session child effectively cannot happen; if it somehow did, the
		// PID has almost certainly been recycled.
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) || errors.Is(err, windows.ERROR_ACCESS_DENIED) {
			return false
		}
		return true // genuinely ambiguous OS error — stay conservative
	}
	defer windows.CloseHandle(h)

	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return true
	}
	return code == stillActive
}

// gracefulStop gives llama-server a short window to finish, then force-kills it
// via TerminateProcess. llama-server on Windows has no console and ignores
// WM_CLOSE, so there is no polite signal to send — the grace is just a timed
// wait before the kill. The result is logged; the job object (closed by
// process.Stop) is the tree-kill mechanism, so no `taskkill /T` is needed.
func gracefulStop(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	pid := cmd.Process.Pid
	time.Sleep(750 * time.Millisecond)

	err := terminatePID(pid)
	switch {
	case err == nil:
		slog.Debug("llamacpp: subprocess terminated", "pid", pid)
		return nil
	case pidGone(err):
		return nil
	default:
		slog.Warn("llamacpp: forced kill of subprocess failed", "pid", pid, "err", err)
		return err
	}
}
