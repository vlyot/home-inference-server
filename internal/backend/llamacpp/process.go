package llamacpp

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync/atomic"
	"time"
)

// ctxSize is the llama-server --ctx-size (the shared budget; with --parallel N
// each slot gets ctxSize/N). The roster's KVCacheMB/KVFixedMB estimates
// (types.ModelDescriptor) are hand-tuned for exactly this value — raising it
// grows the KV cache and makes those estimates (and the fit/eviction math built
// on them) wrong. Overridable via HIS_CTX_SIZE with a warning; see cmd/server.
const ctxSize = 4096

// ctxSizeValue returns the effective --ctx-size, honouring a HIS_CTX_SIZE
// override.
func ctxSizeValue() int {
	if v := os.Getenv("HIS_CTX_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return ctxSize
}

// process manages a single llama-server.exe subprocess.
type process struct {
	exe         string
	modelPath   string
	mmprojPath  string // llama-server --mmproj (vision projector); "" = text-only
	tier        string
	gpuLayers   int
	port        int
	parallel    int  // llama-server --parallel N (concurrent request slots); >= 1
	noJobObject bool // true when assignToJobObject failed at Start (no kill-on-close protection)
	cmd         *exec.Cmd
	jobClose    func()
	runCancel   context.CancelFunc // cancels runCtx; called by Stop/kill to end the subprocess
	stopped     atomic.Bool        // set by Stop/kill so Alive() reports false immediately

	// testArgs, when non-nil, replaces the llama-server flag set built in
	// Start. Used only by tests to spawn a harmless stand-in executable.
	testArgs []string
}

// pidAliveFn / forceKillFn are indirection seams so Stop's verify-and-escalate
// loop can be exercised in tests without a real OS process. Production wiring is
// the real pidAlive and (*exec.Cmd).Process.Kill.
var (
	pidAliveFn  = pidAlive
	forceKillFn = func(p *os.Process) error { return p.Kill() }
)

func newProcess(exe, modelPath, mmprojPath, tier string, gpuLayers, port, parallel int) *process {
	if parallel < 1 {
		parallel = 1
	}
	return &process{
		exe:        exe,
		modelPath:  modelPath,
		mmprojPath: mmprojPath,
		tier:       tier,
		gpuLayers:  gpuLayers,
		port:       port,
		parallel:   parallel,
	}
}

// spawnArgs is the llama-server flag set Start passes to exec. Split out so a
// test can assert the flags without spawning the process.
func (p *process) spawnArgs() []string {
	if p.testArgs != nil {
		return p.testArgs
	}
	args := []string{
		"--model", p.modelPath,
		"--n-gpu-layers", strconv.Itoa(p.gpuLayers),
		"--port", strconv.Itoa(p.port),
		"--ctx-size", strconv.Itoa(ctxSizeValue()),
		"--parallel", strconv.Itoa(p.parallel),
		"--cont-batching",
		"--no-mmap",
		// --flash-attn off: llama-server's default ('auto', flash attention on
		// for CUDA) silently crashes the subprocess a few seconds into
		// generation — no CUDA error, no log line, the process just
		// disappears — specifically under a tight VRAM margin that forces
		// partial GPU/CPU layer offload (e.g. a large tier reloading right
		// after evicting a smaller one leaves too little free VRAM for a full
		// GPU load). Reproduced on real hardware at ~30% failure rate with
		// flash attention on 'auto'; 0/10 failures across two batches with it
		// forced off. Matches a known community pattern for Gemma-family
		// models under partial CUDA offload (e.g. ggml-org/llama.cpp#21401,
		// #22483) — flash attention is the first thing to disable when a
		// partially-offloaded llama-server crashes silently mid-generation.
		"--flash-attn", "off",
	}
	if p.mmprojPath != "" {
		args = append(args, "--mmproj", p.mmprojPath)
	}
	return args
}

// Start spawns llama-server and waits until its /health endpoint returns 200.
// Times out after 120 seconds (cold model load from disk can be slow).
//
// The subprocess's lifetime is intentionally NOT tied to ctx: ctx here is
// typically an inbound HTTP request's context (e.g. from ensureRunning), which
// is cancelled the moment that single request's handler returns. The
// subprocess must keep running long after that — for every subsequent request
// — until Stop or kill is called. A background runCtx (cancelled only by this
// process's own Stop/kill) is used for exec.CommandContext instead; the
// caller's ctx is still honored for the startup health-poll deadline below via
// ctx.Err().
func (p *process) Start(ctx context.Context) error {
	runCtx, cancel := context.WithCancel(context.Background())
	p.runCancel = cancel

	p.cmd = exec.CommandContext(runCtx, p.exe, p.spawnArgs()...)

	stdout, _ := p.cmd.StdoutPipe()
	stderr, _ := p.cmd.StderrPipe()

	if err := p.cmd.Start(); err != nil {
		cancel()
		slog.Error("llamacpp: subprocess spawn failed", "exe", p.exe, "model", p.modelPath, "err", err)
		return fmt.Errorf("llamacpp: spawn failed: %w", err)
	}

	if closeFn, err := assignToJobObject(p.cmd); err != nil {
		slog.Error("llamacpp: job object unavailable — subprocess tree will not be force-killed on job close; Stop verifies liveness directly", "err", err)
		p.noJobObject = true
	} else {
		p.jobClose = closeFn
	}

	go p.pipeLogs(stdout)
	go p.pipeLogs(stderr)

	base := fmt.Sprintf("http://127.0.0.1:%d", p.port)
	healthURL := base + "/health"
	propsURL := base + "/props"
	deadline := time.Now().Add(120 * time.Second)
	client := &http.Client{Timeout: 2 * time.Second}

	// Newer llama-server answers /health 200 as soon as its HTTP listener is up,
	// while /v1/* and /props still 503 until the model weights finish loading.
	// Require two consecutive /health 200s a poll apart AND one successful /props
	// before declaring ready, so the first real request can't race a 503.
	healthyStreak := 0
	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			p.kill()
			return ctx.Err()
		}
		if resp, err := client.Get(healthURL); err == nil {
			ok := resp.StatusCode == http.StatusOK
			resp.Body.Close()
			if ok {
				healthyStreak++
				if healthyStreak >= 2 && p.propsReady(client, propsURL) {
					return nil
				}
			} else {
				healthyStreak = 0
			}
		} else {
			healthyStreak = 0
		}
		time.Sleep(500 * time.Millisecond)
	}

	p.kill()
	return fmt.Errorf("llamacpp: server did not become healthy within 120s on port %d", p.port)
}

// propsReady reports whether GET /props returns 200 — the signal that the model
// is loaded and serving, not just that the HTTP listener is up.
func (p *process) propsReady(client *http.Client, propsURL string) bool {
	resp, err := client.Get(propsURL)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Stop asks the subprocess to exit (SIGINT on Unix; a timed grace then
// TerminateProcess on Windows), waits for it, then VERIFIES the OS process is
// actually gone — polling pidAlive, escalating to a direct kill + closing the
// job object, and returning an error if the PID still survives. A nil return
// therefore means "verified dead"; a non-nil return means the caller must
// assume VRAM is still held and sweep. This closes the silent-success hole
// where cmd.Wait() reaped the parent while a re-parented child lived on.
func (p *process) Stop(ctx context.Context) error {
	p.stopped.Store(true)
	if p.runCancel != nil {
		defer p.runCancel()
	}
	if p.cmd == nil || p.cmd.Process == nil {
		return nil
	}
	pid := p.cmd.Process.Pid

	_ = gracefulStop(p.cmd) // its own failure is logged; the verify loop is the backstop

	done := make(chan error, 1)
	go func() { done <- p.cmd.Wait() }()

	select {
	case <-ctx.Done():
		p.kill()
	case werr := <-done:
		if p.jobClose != nil {
			p.jobClose()
			p.jobClose = nil
		}
		if werr != nil {
			slog.Debug("llama-server exited after stop", "tier", p.tier, "err", werr)
		}
	}

	// Verify the OS process is gone.
	if p.waitGone(pid, 2*time.Second) {
		return nil
	}

	// Escalate: direct kill + close the job object, then re-check briefly.
	slog.Warn("llamacpp: subprocess still alive after stop — escalating", "pid", pid, "tier", p.tier)
	_ = forceKillFn(p.cmd.Process)
	if p.jobClose != nil {
		p.jobClose()
		p.jobClose = nil
	}
	if p.waitGone(pid, 500*time.Millisecond) {
		slog.Warn("llamacpp: subprocess killed only after escalation past gracefulStop", "pid", pid, "tier", p.tier)
		return nil
	}
	return fmt.Errorf("llamacpp: subprocess pid %d survived stop+kill for tier %s", pid, p.tier)
}

// waitGone polls pidAlive for up to d and reports whether the process exited.
func (p *process) waitGone(pid int, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		if !pidAliveFn(pid) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func (p *process) kill() {
	p.stopped.Store(true)
	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	if p.jobClose != nil {
		p.jobClose()
		p.jobClose = nil
	}
	if p.runCancel != nil {
		p.runCancel()
	}
}

func (p *process) Port() int { return p.port }

// PID returns the subprocess OS process id, or 0 if it is not running.
func (p *process) PID() int {
	if p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// Alive reports whether the OS process is still running. Used to catch a
// subprocess killed out-of-band (reaper, OS, an external kill) so callers stop
// reporting VRAM it no longer holds. Conservative: false once Stop/kill has run
// or our Wait() goroutine has reaped the process, and otherwise an OS liveness
// probe.
func (p *process) Alive() bool {
	if p.cmd == nil || p.cmd.Process == nil {
		return false
	}
	if p.stopped.Load() || p.cmd.ProcessState != nil {
		return false
	}
	return pidAlive(p.cmd.Process.Pid)
}

// pipeLogs forwards each subprocess output line to the debug log. llama-server
// prints its layer count, KV-cache size and compute-buffer size here.
func (p *process) pipeLogs(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 256*1024)
	for sc.Scan() {
		slog.Debug("llama-server", "tier", p.tier, "port", p.port, "line", sc.Text())
	}
}
