//go:build windows

package llamacpp

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

func spawnSleeper(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(sleeperExe(), sleeperArgs()...)
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn sleeper: %v", err)
	}
	t.Cleanup(func() { _ = cmd.Process.Kill(); _ = cmd.Wait() })
	return cmd
}

func TestPidAlive_TrueForRunningProcess(t *testing.T) {
	cmd := spawnSleeper(t)
	if !pidAlive(cmd.Process.Pid) {
		t.Fatal("pidAlive = false for a freshly spawned running process")
	}
}

func TestPidAlive_FalseAfterExit(t *testing.T) {
	cmd := exec.Command(os.Getenv("COMSPEC"), "/c", "exit", "0")
	if cmd.Path == "" {
		cmd = exec.Command("cmd", "/c", "exit", "0")
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	pid := cmd.Process.Pid
	_ = cmd.Wait()
	// Give the OS a beat to tear the process object down.
	deadline := time.Now().Add(time.Second)
	for pidAlive(pid) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if pidAlive(pid) {
		t.Fatal("pidAlive = true for a process that has exited")
	}
}

func TestPidAlive_FalseForImpossiblePID(t *testing.T) {
	if pidAlive(0x7FFFFFFE) {
		t.Fatal("pidAlive = true for a PID that cannot exist")
	}
}

func TestTerminatePID_KillsProcess(t *testing.T) {
	cmd := spawnSleeper(t)
	pid := cmd.Process.Pid
	if err := terminatePID(pid); err != nil {
		t.Fatalf("terminatePID: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for pidAlive(pid) && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if pidAlive(pid) {
		t.Fatal("process still alive 2s after terminatePID")
	}
	// terminatePID on a now-dead pid returns a "gone" error the callers ignore.
	if err := terminatePID(pid); err != nil && !pidGone(err) {
		t.Fatalf("terminatePID on a dead pid: unexpected err %v", err)
	}
}
