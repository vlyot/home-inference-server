package llamacpp

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestSubprocessSurvivesCallerContextCancellation is the regression test for
// the bug that made every request reload the model: process.Start used to
// spawn the subprocess with exec.CommandContext(ctx, ...) where ctx was the
// caller's context — typically a single inbound HTTP request's context. The
// moment that one request's handler returned and its context was cancelled,
// Go's exec package killed the subprocess out from under every other
// request, forcing a reload on effectively every subsequent call to
// ensureRunning.
//
// This starts a real long-lived stand-in process against a fake /health +
// /props server (so Start returns successfully, exactly as it would once
// llama-server finishes loading), then cancels the ctx that was passed to
// Start — simulating the HTTP handler that triggered the load returning —
// and asserts the subprocess is still alive well afterward. It must be: the
// subprocess serves every future request, not just the one that started it.
func TestSubprocessSurvivesCallerContextCancellation(t *testing.T) {
	port := freePort(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/props", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	fakeHealth := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", port), Handler: mux}
	go fakeHealth.ListenAndServe()
	t.Cleanup(func() { fakeHealth.Close() })

	p := &process{exe: sleeperExe(), testArgs: sleeperArgs(), port: port}
	// However the test exits — pass, fail, or fatal — make sure the spawned
	// process doesn't leak. t.Cleanup runs even after t.Fatal.
	t.Cleanup(func() {
		stopCtx, stopCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer stopCancel()
		_ = p.Stop(stopCtx)
	})

	callerCtx, cancel := context.WithCancel(context.Background())
	if err := p.Start(callerCtx); err != nil {
		cancel()
		t.Fatalf("Start failed: %v", err)
	}
	pid := p.cmd.Process.Pid

	// Simulate the HTTP handler that triggered ensureRunning/Start returning —
	// exactly what cancels a request's context in the real server.
	cancel()

	// Give exec's context-watcher goroutine, if it were (wrongly) tied to
	// callerCtx, ample time to kill the process, then confirm it survived.
	time.Sleep(2 * time.Second)

	if !processAlive(pid) {
		t.Fatal("subprocess exited when the caller ctx was cancelled after a " +
			"successful Start — its lifetime is wrongly tied to the caller's " +
			"context instead of an internal runCtx")
	}
}

// TestStart_SpawnFailureLogsError covers the startup-visibility gap:
// when the server is launched headless by Task Scheduler and llama-server.exe
// or a model file is missing, exec.Cmd.Start fails. That error propagates up to
// the HTTP handler, but nothing logs it at Error level and nothing fires until
// the first inference request arrives — so a headless boot failure leaves
// logs\server.log empty. Start must emit an actionable slog.Error naming the
// offending paths, while still returning the same wrapped error to callers.
func TestStart_SpawnFailureLogsError(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelError})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	const missingModel = `Z:\definitely\missing\model.gguf`
	p := &process{
		exe:       `Z:\definitely\missing\llama-server.exe`,
		modelPath: missingModel,
		port:      freePort(t),
		testArgs:  []string{"--model", missingModel},
	}

	err := p.Start(context.Background())
	if err == nil {
		t.Fatal("Start should return an error when the executable does not exist")
	}
	if !strings.Contains(err.Error(), "llamacpp: spawn failed") {
		t.Fatalf("wrapped error changed shape: %v", err)
	}

	logged := buf.String()
	if !strings.Contains(logged, "subprocess spawn failed") {
		t.Fatalf("expected an Error log about the spawn failure, got: %q", logged)
	}
	if !strings.Contains(logged, missingModel) {
		t.Fatalf("Error log should name the model path so the operator knows what is missing, got: %q", logged)
	}
}

// freePort asks the OS for an unused TCP port on localhost.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("could not find a free port: %v", err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

// sleeperExe returns a harmless executable this test can spawn as a stand-in
// for llama-server, without requiring the real binary.
func sleeperExe() string {
	if path, err := exec.LookPath("powershell.exe"); err == nil {
		return path
	}
	return os.Getenv("COMSPEC") // cmd.exe fallback
}

// sleeperArgs returns args that make the stand-in executable sleep long
// enough for the test to observe it surviving a cancelled caller ctx.
func sleeperArgs() []string {
	return []string{"-NoProfile", "-Command", "Start-Sleep -Seconds 30"}
}

// processAlive reports whether pid is a live OS process. os.FindProcess and
// Process.Signal are unreliable liveness checks on Windows (FindProcess
// always succeeds; Signal doesn't support a no-op probe signal), so this
// shells out to tasklist, which only lists processes that actually exist.
func processAlive(pid int) bool {
	out, err := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH").Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), strconv.Itoa(pid))
}
