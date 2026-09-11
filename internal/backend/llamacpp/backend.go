// Package llamacpp implements a backend.Backend that manages a llama-server
// subprocess and communicates with it via its HTTP completion API.
package llamacpp

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/vram"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

const tokRingSize = 10

// Backend implements backend.Backend by managing a llama-server subprocess.
// It is safe for concurrent calls; only one subprocess runs at a time.
type Backend struct {
	modality    backend.ModalityKind
	desc        types.ModelDescriptor
	exe         string
	vramp       vram.VRAMProvider // nil = always use full GPU (-1)
	maxParallel int               // llama-server --parallel N; >= 1

	// active counts inference calls currently past ensureRunning. Read when a
	// completion finishes to normalise its per-slot tok/sec sample back to a
	// single-stream-equivalent rate (llama-server time-slices N streams evenly).
	active atomic.Int32

	// startMu serialises subprocess spawns. It is held across the slow
	// proc.Start (up to 120 s health poll) so two callers don't race a spawn,
	// but b.mu is NOT — cheap accessors (Ready/TokPerSec/RunningPID/ResidentMB/
	// Restarting) stay responsive during a cold start.
	startMu sync.Mutex

	mu         sync.Mutex
	proc       *process
	baseURL    string
	ready      bool
	starting   bool  // true while a spawn is in progress under startMu
	restarting bool  // true while the restart path is replacing a dead subprocess
	residentMB int64 // estimated GPU VRAM the running subprocess holds; 0 when stopped
	nCtx       int   // memoised from /props; 0 until first NCtx call, reset on Shutdown
	// nextGPULayers is the --n-gpu-layers value to spawn with, set by vram.Backend
	// (which computes it via FitLayers) so bookkeeping and the subprocess agree.
	// When nextGPULayersSet is false, ensureRunning falls back to ComputeGPULayers.
	nextGPULayers    int
	nextGPULayersSet bool
	// nextNeedsVision is whether the NEXT spawn should run WITH --mmproj, set by
	// vram.Backend (via SetNextNeedsVision) before every Infer/InferStream.
	// runningWithVision is which mode the CURRENTLY RUNNING subprocess (if any)
	// was actually spawned with. ensureRunning compares the two and forces an
	// evict + respawn on mismatch.
	nextNeedsVision   bool
	runningWithVision bool

	// Rolling tok/sec average over the last tokRingSize completions.
	tokRing [tokRingSize]float64
	tokN    int // total samples, capped at tokRingSize for average
	tokHead int // next write slot
}

// New constructs a Backend. Pass a non-nil vramp to enable dynamic GPU layer
// computation at subprocess start time; pass nil to always use full GPU offload.
// maxParallel is the llama-server --parallel N the subprocess is spawned with
// (values < 1 are clamped to 1). The subprocess is not started until the first
// call to Infer (lazy start).
func New(modality backend.ModalityKind, desc types.ModelDescriptor, exe string, vramp vram.VRAMProvider, maxParallel int) *Backend {
	if maxParallel < 1 {
		maxParallel = 1
	}
	return &Backend{
		modality:    modality,
		desc:        desc,
		exe:         exe,
		vramp:       vramp,
		maxParallel: maxParallel,
	}
}

func (b *Backend) Modality() backend.ModalityKind { return b.modality }

// SetNextGPULayers records the --n-gpu-layers the next spawn should use. Called
// by vram.Backend so the subprocess runs with the same split the VRAM
// bookkeeping assumes. Takes effect on the next (re)load, not a running process.
func (b *Backend) SetNextGPULayers(n int) {
	b.mu.Lock()
	b.nextGPULayers = n
	b.nextGPULayersSet = true
	b.mu.Unlock()
}

// SetNextNeedsVision records whether the next spawn should run WITH --mmproj
// (true) or without it (false). Called by vram.Backend before every
// Infer/InferStream so ensureRunning can tell whether the CURRENTLY RUNNING
// subprocess (if any) already matches — a mismatch forces an evict + respawn
// before the request is served. Takes effect on the next ensureRunning call,
// not a running process.
func (b *Backend) SetNextNeedsVision(v bool) {
	b.mu.Lock()
	b.nextNeedsVision = v
	b.mu.Unlock()
}

func (b *Backend) Ready() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.ready
}

// Restarting reports whether the backend is mid-way through replacing a dead
// subprocess after a connection error. The VRAM eviction loop consults this so
// it never tears down a process that is already being restarted.
func (b *Backend) Restarting() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.restarting
}

// Infer ensures the subprocess is running, then calls the completion API.
// On connection failure it attempts one restart before returning an error.
func (b *Backend) Infer(ctx context.Context, req backend.Request) (backend.Response, error) {
	baseURL, err := b.ensureRunning(ctx)
	if err != nil {
		return backend.Response{}, err
	}

	concurrency := int(b.active.Add(1))
	defer b.active.Add(-1)

	resp, err := b.runComplete(ctx, baseURL, req)
	if err != nil {
		// One restart attempt on connection error (process may have crashed).
		be, ok := err.(*backend.BackendError)
		if ok && be.Code == "unavailable" {
			newURL, restartErr := b.restartSubprocess(ctx, req.CorrelationID)
			if restartErr != nil {
				return backend.Response{}, restartErr
			}
			resp, err = b.runComplete(ctx, newURL, req)
		}
		if err != nil {
			return backend.Response{}, err
		}
	}

	b.recordTokSample(resp.TokPerSecSample, concurrency)
	resp.ModelTier = string(b.desc.TierLabel)

	return resp, nil
}

// recordTokSample folds one completion's per-stream tok/sec into the rolling
// ring, first normalising it back to a single-stream-equivalent rate.
// llama-server time-slices `concurrency` in-flight streams roughly evenly, so a
// solo request's sample is trusted as-is while an N-way concurrent sample is
// multiplied by N — otherwise the speed-floor cascade (defined against
// single-stream rates) would trip on every request under load.
func (b *Backend) recordTokSample(sample float64, concurrency int) {
	if sample <= 0 {
		return
	}
	if concurrency < 1 {
		concurrency = 1
	}
	normalised := sample * float64(concurrency)
	b.mu.Lock()
	b.tokRing[b.tokHead] = normalised
	b.tokHead = (b.tokHead + 1) % tokRingSize
	if b.tokN < tokRingSize {
		b.tokN++
	}
	b.mu.Unlock()
}

// InferStream runs inference and streams token deltas to chunkFn as they
// arrive, tagged as ChunkContent or ChunkReasoning. Implements backend.Streamer.
func (b *Backend) InferStream(ctx context.Context, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	baseURL, err := b.ensureRunning(ctx)
	if err != nil {
		return backend.Response{}, err
	}

	concurrency := int(b.active.Add(1))
	defer b.active.Add(-1)

	emitted := false
	wrapped := func(kind backend.ChunkKind, delta string) {
		emitted = true
		chunkFn(kind, delta)
	}

	resp, err := b.runCompleteStream(ctx, baseURL, req, wrapped)
	if err != nil {
		// One restart attempt on connection error, but only if nothing has been
		// streamed yet — restarting mid-stream would duplicate tokens.
		be, ok := err.(*backend.BackendError)
		if ok && be.Code == "unavailable" && !emitted {
			newURL, restartErr := b.restartSubprocess(ctx, req.CorrelationID)
			if restartErr != nil {
				return backend.Response{}, restartErr
			}
			resp, err = b.runCompleteStream(ctx, newURL, req, wrapped)
		}
		if err != nil {
			// Even on failure, carry ModelTier: a caller that already streamed
			// real content before this error (e.g. the mid-generation crash
			// server_stream.go salvages as Truncated) needs to know which
			// tier produced it, for logging/job-history purposes.
			resp.ModelTier = string(b.desc.TierLabel)
			return resp, err
		}
	}

	b.recordTokSample(resp.TokPerSecSample, concurrency)
	resp.ModelTier = string(b.desc.TierLabel)

	return resp, nil
}

// runComplete dispatches to the chat-completions endpoint when the request
// carries a message list, an image, or this tier's descriptor sets
// RequireChatTemplate (the /completion endpoint has no chat template and no
// --mmproj image path), otherwise the single-prompt endpoint. A prompt-only
// image request is promoted to a one-turn user message so toChatMessages can
// attach the image_url part.
func (b *Backend) runComplete(ctx context.Context, baseURL string, req backend.Request) (backend.Response, error) {
	if req := chatShape(req, b.desc.RequireChatTemplate); len(req.Messages) > 0 {
		req.MaxTokens = b.resolveChatMaxTokens(ctx, baseURL, req)
		return chatComplete(ctx, baseURL, req)
	}
	return complete(ctx, baseURL, req)
}

// runCompleteStream is the streaming counterpart of runComplete. The plain
// /completion endpoint has no chat template and therefore no reasoning
// concept, so its deltas are always tagged ChunkContent.
func (b *Backend) runCompleteStream(ctx context.Context, baseURL string, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	if req := chatShape(req, b.desc.RequireChatTemplate); len(req.Messages) > 0 {
		req.MaxTokens = b.resolveChatMaxTokens(ctx, baseURL, req)
		return chatCompleteStream(ctx, baseURL, req, chunkFn)
	}
	return completeStream(ctx, baseURL, req, func(delta string) {
		chunkFn(backend.ChunkContent, delta)
	})
}

// chatShape returns req unchanged when it already has Messages, and otherwise
// wraps its Prompt as a chat message list — always when forceTemplate is set
// (this tier's ModelDescriptor.RequireChatTemplate), or when the request
// carries an image — so the chat/completions (and --mmproj, for an image)
// path is used. A plain prompt-only text request on a tier that does NOT
// require the template is left alone so it still takes the plain
// /completion endpoint (cheaper: no /tokenize round trip for max-tokens
// resolution).
//
// A non-empty SystemPrompt is prepended as its own {role: "system"} turn
// rather than concatenated into the user turn's text: llama-server's jinja
// chat template (the model's own GGUF-embedded template, used by default)
// only recognises a real system-role message, and a small vision-language
// model given an image with no system turn at all tends to fall back to a
// generic "I cannot see images" text-completion refusal instead of answering
// — a known small-VLM failure mode, not a wording problem in the user prompt.
func chatShape(req backend.Request, forceTemplate bool) backend.Request {
	if len(req.Messages) > 0 {
		return req
	}
	if len(req.ImageData) == 0 && !forceTemplate {
		return req
	}
	if req.SystemPrompt != "" {
		req.Messages = []backend.Message{
			{Role: "system", Content: req.SystemPrompt},
			{Role: "user", Content: req.Prompt},
		}
	} else {
		req.Messages = []backend.Message{{Role: "user", Content: req.Prompt}}
	}
	return req
}

// chatMaxTokensSafetyMarginPct reserves this percentage of the model's total
// context as headroom below the computed cap — chat-template turn markers,
// tokenizer estimation error (the /tokenize count is on raw joined content,
// not the exact templated prompt), and a little slack so the reply is never
// truncated by a hair.
const chatMaxTokensSafetyMarginPct = 10

// resolveChatMaxTokens returns req.MaxTokens unchanged if the caller specified
// one explicitly; otherwise it computes how many tokens are actually left in
// the model's context window after the prompt, so a reasoning-capable model
// gets the full remaining budget instead of an arbitrary fixed cap that can
// run out mid-thought (see defaultChatMaxTokens) or, on a short conversation,
// leave most of the context unused. Falls back to defaultChatMaxTokens if
// n_ctx or the prompt token count can't be determined (e.g. transient
// tokenize failure) — never blocks a request over this.
func (b *Backend) resolveChatMaxTokens(ctx context.Context, baseURL string, req backend.Request) int {
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}

	nCtx, err := b.NCtx(ctx)
	if err != nil || nCtx <= 0 {
		return defaultChatMaxTokens
	}

	var joined strings.Builder
	for i, m := range req.Messages {
		if i > 0 {
			joined.WriteByte('\n')
		}
		joined.WriteString(m.Content)
	}
	promptTokens, err := tokenize(ctx, baseURL, joined.String())
	if err != nil {
		return defaultChatMaxTokens
	}
	// Chat-template turn markers (role tags, BOS/EOS per turn) aren't counted
	// by tokenizing raw joined content; a fixed per-message allowance covers
	// them, matching the estimate the chat UI itself uses client-side.
	promptTokens += 8 * len(req.Messages)

	margin := nCtx * chatMaxTokensSafetyMarginPct / 100
	remaining := nCtx - promptTokens - margin
	if remaining < 1 {
		remaining = 1
	}
	return remaining
}

// Tokenize returns the number of tokens the model's tokenizer produces for text.
// Starts the subprocess lazily if it is not already running, and retries once if
// the subprocess has died since it was last used.
func (b *Backend) Tokenize(ctx context.Context, text string) (int, error) {
	baseURL, err := b.ensureRunning(ctx)
	if err != nil {
		return 0, err
	}
	n, err := tokenize(ctx, baseURL, text)
	if be, ok := err.(*backend.BackendError); ok && be.Code == "unavailable" {
		b.mu.Lock()
		b.ready = false
		b.proc = nil
		b.mu.Unlock()
		if baseURL, err = b.ensureRunning(ctx); err != nil {
			return 0, err
		}
		return tokenize(ctx, baseURL, text)
	}
	return n, err
}

// NCtx returns the model's context window size, memoised for the subprocess
// lifetime. Starts the subprocess lazily if it is not already running.
func (b *Backend) NCtx(ctx context.Context) (int, error) {
	b.mu.Lock()
	if b.nCtx > 0 {
		n := b.nCtx
		b.mu.Unlock()
		return n, nil
	}
	b.mu.Unlock()

	baseURL, err := b.ensureRunning(ctx)
	if err != nil {
		return 0, err
	}
	n, err := props(ctx, baseURL)
	if err != nil {
		return 0, err
	}
	b.mu.Lock()
	b.nCtx = n
	b.mu.Unlock()
	return n, nil
}

// Shutdown stops the subprocess gracefully.
func (b *Backend) Shutdown(ctx context.Context) error {
	b.mu.Lock()
	proc := b.proc
	b.proc = nil
	b.ready = false
	b.residentMB = 0
	b.nCtx = 0
	b.mu.Unlock()

	if proc == nil {
		return nil
	}

	slog.Info("llamacpp: shutting down subprocess",
		slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
	)
	err := proc.Stop(ctx)
	if err != nil {
		slog.Error("llamacpp: subprocess teardown incomplete — VRAM may still be held",
			slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
			slog.Any("err", err),
		)
	}
	return err
}

// ResidentMB returns the estimated GPU VRAM the running subprocess currently
// holds, or 0 when it is not running. It checks the OS process is actually alive
// so a subprocess killed out-of-band (the reaper, the OS, an external kill)
// doesn't leave a stale figure on the dashboard.
func (b *Backend) ResidentMB() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.ready || b.proc == nil {
		return 0
	}
	if !b.proc.Alive() {
		slog.Warn("llamacpp: resident VRAM cleared — subprocess no longer alive",
			slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
			slog.Int64("was_mb", b.residentMB),
		)
		b.ready = false
		b.residentMB = 0
		return 0
	}
	return b.residentMB
}

// RunningPID returns the subprocess OS process id, or 0 when it is not running.
func (b *Backend) RunningPID() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.proc == nil {
		return 0
	}
	return b.proc.PID()
}

// TokPerSec returns the rolling average tokens-per-second over the last
// tokRingSize completions. Returns 0 if no completions have been recorded yet.
// Implements backend.Measurable.
func (b *Backend) TokPerSec() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.tokN == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < b.tokN; i++ {
		sum += b.tokRing[i]
	}
	return sum / float64(b.tokN)
}

// restartSubprocess handles a single connection-error recovery: it marks the
// backend as restarting (so the eviction loop leaves it alone), stops the old
// subprocess and waits for it to exit so the port is free, then starts a fresh
// one. The restarting flag is always cleared before returning.
func (b *Backend) restartSubprocess(ctx context.Context, correlationID string) (string, error) {
	slog.Warn("llamacpp: connection error, attempting restart",
		slog.String(logschema.FieldCorrelationID, correlationID),
	)

	b.mu.Lock()
	old := b.proc
	b.ready = false
	b.proc = nil
	b.restarting = true
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		b.restarting = false
		b.mu.Unlock()
	}()

	if old != nil {
		stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = old.Stop(stopCtx)
		cancel()
	}

	return b.ensureRunning(ctx)
}

// probeHealthy does a fast GET on {baseURL}/health and reports whether the
// subprocess answered 200. Used to catch a stale ready flag pointing at a
// subprocess that has since died (e.g. reaped after a failed eviction).
func probeHealthy(ctx context.Context, baseURL string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/health", nil)
	if err != nil {
		return false
	}
	client := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// ensureRunning starts the subprocess if it is not already running, or
// restarts it if the running process was spawned in the wrong mode (with or
// without --mmproj) for what the next request needs. Returns the base URL to
// use for HTTP requests. b.mu is held only for the short state checks/writes;
// the slow proc.Start runs under startMu with b.mu released, so status
// accessors stay responsive during a cold load.
func (b *Backend) ensureRunning(ctx context.Context) (string, error) {
	b.mu.Lock()
	if b.ready {
		proc, baseURL := b.proc, b.baseURL
		modeMismatch := b.runningWithVision != b.nextNeedsVision
		b.mu.Unlock()

		if modeMismatch {
			slog.Info("llamacpp: vision mode changed, restarting subprocess",
				slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
			)
			if proc != nil {
				stopCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				_ = proc.Stop(stopCtx)
				cancel()
				// Confirming the OS process exited isn't enough on its own: the
				// NVIDIA driver releases a CUDA context's VRAM and tears it down
				// asynchronously relative to process exit. Cheap defensive
				// insurance mirroring vram.Backend's waitForVRAMReclaim (which
				// polls real VRAM headroom via its VRAMProvider — not available
				// at this layer, hence a short fixed pause instead). This did
				// NOT turn out to be the cause of the specific crash that
				// prompted it — see waitForVRAMReclaim's doc comment and the
				// roadmap entry for the actual root cause (--flash-attn) — but
				// it is genuine, community-documented behaviour worth guarding
				// against regardless.
				time.Sleep(300 * time.Millisecond)
			}
			b.mu.Lock()
			b.ready = false
			b.proc = nil
			b.residentMB = 0
		} else if proc == nil || probeHealthy(ctx, baseURL) {
			// A live proc handle plus a ready flag can still point at a dead
			// subprocess if it was killed out from under us. Probe before trusting.
			return baseURL, nil
		} else {
			slog.Info("llamacpp: subprocess no longer healthy, reloading",
				slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
			)
			b.mu.Lock()
			b.ready = false
			b.proc = nil
		}
	}
	b.mu.Unlock()

	// Serialise the spawn itself.
	b.startMu.Lock()
	defer b.startMu.Unlock()

	// Another goroutine may have completed the spawn while we waited on startMu
	// — but only if it landed in the mode this call still wants.
	b.mu.Lock()
	if b.ready && b.runningWithVision == b.nextNeedsVision {
		url := b.baseURL
		b.mu.Unlock()
		return url, nil
	}
	b.starting = true
	needsVision := b.nextNeedsVision
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.starting = false
		b.mu.Unlock()
	}()

	port := b.desc.Port
	if port == 0 {
		port = 8090
	}
	b.mu.Lock()
	gpuLayers, haveLayers := b.nextGPULayers, b.nextGPULayersSet
	b.mu.Unlock()
	if !haveLayers {
		// Standalone use (no vram.Backend driving us): compute a split here.
		gpuLayers = -1
		if b.vramp != nil {
			if availMB, err := b.vramp.AvailableMB(); err == nil {
				gpuLayers = vram.ComputeGPULayers(b.desc.WeightMB(needsVision), availMB)
			}
		}
	}
	resident := vram.GPUResidentMB(b.desc.WeightMB(needsVision), vram.ScaledKVOverheadMB(b.desc, b.maxParallel, needsVision), gpuLayers, b.desc.TotalLayers)
	mmprojPath := ""
	if needsVision {
		mmprojPath = b.desc.MMProjPath
	}
	proc := newProcess(b.exe, b.desc.FilePath, mmprojPath, string(b.desc.TierLabel), gpuLayers, port, b.maxParallel)

	slog.Info("model loading",
		slog.String(logschema.FieldEvent, string(logschema.EventModelLoading)),
		slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
		slog.Int64(logschema.FieldVRAMRequiredMB, b.desc.WeightMB(needsVision)),
		slog.Int("gpu_layers", gpuLayers),
		slog.Int("total_layers", b.desc.TotalLayers),
		slog.Int("parallel", b.maxParallel),
		slog.Bool("vision", needsVision),
		slog.Int64("vram_resident_mb", resident),
	)

	if err := proc.Start(ctx); err != nil {
		return "", &backend.BackendError{
			Code:    "model_load_failed",
			Message: fmt.Sprintf("llamacpp: failed to start %s", b.desc.Name),
			Cause:   err,
		}
	}

	b.mu.Lock()
	b.proc = proc
	b.baseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	b.ready = true
	b.runningWithVision = needsVision
	b.residentMB = resident
	b.mu.Unlock()

	slog.Info("model loaded",
		slog.String(logschema.FieldEvent, string(logschema.EventModelLoaded)),
		slog.String(logschema.FieldModelTier, string(b.desc.TierLabel)),
	)

	return b.baseURL, nil
}
