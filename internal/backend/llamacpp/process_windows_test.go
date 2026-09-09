//go:build windows

package llamacpp

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// startVerifiableProcess spawns a real killable stand-in via process.Start
// (against a fake /health + /props so Start returns), so Stop has a genuine
// cmd/Process to verify against.
func startVerifiableProcess(t *testing.T) *process {
	t.Helper()
	port := freePort(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/props", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	fake := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", port), Handler: mux}
	go fake.ListenAndServe()
	t.Cleanup(func() { fake.Close() })

	p := &process{exe: sleeperExe(), testArgs: sleeperArgs(), port: port, tier: "test"}
	callerCtx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	if err := p.Start(callerCtx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() {
		if p.cmd != nil && p.cmd.Process != nil {
			_ = p.cmd.Process.Kill()
		}
	})
	return p
}

func TestStop_VerifiesProcessGone(t *testing.T) {
	p := startVerifiableProcess(t)
	pid := p.cmd.Process.Pid

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	if err := p.Stop(ctx); err != nil {
		t.Fatalf("Stop returned an error for a killable process: %v", err)
	}
	if pidAlive(pid) {
		t.Fatal("Stop returned nil but the OS process is still alive")
	}
}

func TestStop_ReturnsErrorWhenProcessSurvives(t *testing.T) {
	p := startVerifiableProcess(t)

	// Force the verify loop to always see the process as alive, and neuter the
	// escalation kill so it can't actually help.
	origAlive, origKill := pidAliveFn, forceKillFn
	pidAliveFn = func(int) bool { return true }
	forceKillFn = func(*os.Process) error { return nil }
	t.Cleanup(func() { pidAliveFn, forceKillFn = origAlive, origKill })

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	err := p.Stop(ctx)
	if err == nil {
		t.Fatal("Stop returned nil for a process that never dies")
	}
	if !strings.Contains(err.Error(), "survived stop+kill") {
		t.Fatalf("unexpected error shape: %v", err)
	}

	// Real cleanup — the escalation was neutered above.
	_ = p.cmd.Process.Kill()
}

func TestStop_NoJobObject_StillVerifies(t *testing.T) {
	p := startVerifiableProcess(t)
	p.jobClose = nil
	p.noJobObject = true

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	if err := p.Stop(ctx); err != nil {
		t.Fatalf("Stop with no job object errored on a killable process: %v", err)
	}
	if pidAlive(p.cmd.Process.Pid) {
		t.Fatal("process survived Stop when jobClose was nil")
	}
}
