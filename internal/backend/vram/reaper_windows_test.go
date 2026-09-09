//go:build windows

package vram

import (
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestImageName_CurrentProcess(t *testing.T) {
	name := defaultPortOwner{}.imageName(os.Getpid())
	if name == "" {
		t.Fatal("imageName returned empty for the current process")
	}
	if !strings.HasSuffix(name, ".exe") {
		t.Errorf("imageName = %q; want an .exe base name", name)
	}
	if name != strings.ToLower(name) {
		t.Errorf("imageName = %q; want lowercased", name)
	}
}

func TestImageName_ImpossiblePID(t *testing.T) {
	if got := (defaultPortOwner{}).imageName(0x7FFFFFFE); got != "" {
		t.Fatalf("imageName for an impossible PID = %q; want \"\"", got)
	}
}

func TestKill_TerminatesSpawnedProcess(t *testing.T) {
	exe := os.Getenv("COMSPEC")
	if exe == "" {
		exe = "cmd"
	}
	cmd := exec.Command(exe, "/c", "pause")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	pid := cmd.Process.Pid
	waited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(waited) }()
	t.Cleanup(func() { _ = cmd.Process.Kill(); <-waited })

	if err := (defaultPortOwner{}).kill(pid); err != nil {
		t.Fatalf("kill: %v", err)
	}
	owner := defaultPortOwner{}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if owner.imageName(pid) == "" {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("process still present 2s after kill")
}
