package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ngkaichong/home-inference-server/assets"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/llamacpp"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/backend/vram"
	"github.com/ngkaichong/home-inference-server/internal/batcher"
	"github.com/ngkaichong/home-inference-server/internal/chatstore"
	"github.com/ngkaichong/home-inference-server/internal/dashboard"
	"github.com/ngkaichong/home-inference-server/internal/dispatcher"
	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/relayenqueue"
	"github.com/ngkaichong/home-inference-server/internal/remote"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/internal/server"
	isystem "github.com/ngkaichong/home-inference-server/internal/system"
	"github.com/ngkaichong/home-inference-server/internal/tray"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

type appConfig struct {
	RailwayQueueURL string `json:"railway_queue_url"`
	RailwayAPIKey   string `json:"railway_api_key"`
}

func loadConfig() appConfig {
	var cfg appConfig
	if data, err := os.ReadFile("config.json"); err == nil {
		if err := json.Unmarshal(data, &cfg); err != nil {
			slog.Warn("config.json parse error", "err", err)
			cfg = appConfig{}
		}
	}
	// Environment variables win over config.json — the autostart deploy
	// (deploy/windows/run.cmd) sets these instead of shipping a config file.
	cfg.RailwayQueueURL = envStr("RAILWAY_QUEUE_URL", cfg.RailwayQueueURL)
	cfg.RailwayAPIKey = envStr("RAILWAY_QUEUE_API_KEY", cfg.RailwayAPIKey)
	return cfg
}

const version = "0.12.0"

// defaultMaxParallel is the concurrent-inference degree: llama-server's
// --parallel N, the vram.Backend semaphore capacity, and the dispatcher's
// worker count. Overridable via HIS_MAX_PARALLEL. Sized for an 8 GB card where
// 2 KV caches fit alongside a partially-offloaded strong tier.
const defaultMaxParallel = 2

// llamaExe is the path to the llama-server binary.
const llamaExe = `C:\llama\llama-server.exe`

// defaultRoster is the tiered model roster. RequiredVRAMMB is the GGUF weight
// size; GPULayers is computed dynamically at load time. TotalLayers, KVCacheMB
// and KVFixedMB are read from llama-server startup logs (--ctx-size 4096) and
// feed the partial-offload fit test in vram.selectAndLoad. The GPU-resident
// footprint of a full-offload load is RequiredVRAMMB + KVCacheMB*N + KVFixedMB
// (N = HIS_MAX_PARALLEL); the eviction loop and selectAndLoad both size pressure
// against that sum via GPUResidentMB, never against RequiredVRAMMB alone.
// KVCacheMB (per --parallel slot) vs KVFixedMB (compute buffers + CUDA context,
// constant) split from the old single KVOverheadMB — the cache portion is the
// ctx-proportional part, roughly two thirds. Tighten from slog.Debug
// llama-server startup lines on the GPU.
//
// Every tier is a text.Backend; a tier whose MMProjPath is set (currently only
// weak) ALSO accepts modality: "vision" requests — vram.Backend restricts
// vision selection to HasVision() tiers and spawns that tier's subprocess with
// --mmproj only for the request that actually needs it (see
// types.ModelDescriptor.WeightMB / VisionKVParts and
// llamacpp.Backend.SetNextNeedsVision). There is no separate vision-only
// roster or backend.
var defaultRoster = []types.ModelDescriptor{
	{
		TierLabel: types.TierWeak,
		Name:      "qwen2.5-vl-3b-instruct",
		FilePath:  `models\qwen2.5-vl-3b-instruct-q4_k_m.gguf`,
		// MMProjPath makes this tier vision-capable. It is passed to
		// llama-server ONLY on a load that a vision request triggers
		// (llamacpp.Backend.ensureRunning consults SetNextNeedsVision) — a
		// plain text load never touches --mmproj or the VRAM it costs. See
		// HasVision(), WeightMB(), VisionKVParts().
		MMProjPath:     `models\mmproj-qwen2.5-vl-3b-instruct-q8_0.gguf`,
		RequiredVRAMMB: 1840, // Q4_K_M text weights, measured file size
		Modality:       "text",
		TotalLayers:    36, // qwen2vl.block_count from the GGUF metadata
		KVCacheMB:      320,
		KVFixedMB:      180,
		// Vision-mode constants: same weights (native Qwen2.5-VL, no separate
		// vision checkpoint) plus the ~805 MB mmproj and its own compute
		// buffers. Measured on the GPU: a full-GPU vision load's estimate
		// (weight 1840 + KVFixedMB 1600 + KVCacheMB*2 slots = 4080) tracked a
		// real nvidia-smi peak of ~5000 MB including the ~1760 MB idle
		// baseline (~3240 MB resident) — conservative in the safe direction
		// (over-, not under-, booked). See roadmap.md for the full run.
		VisionRequiredVRAMMB: 1840,
		VisionKVCacheMB:      320,
		VisionKVFixedMB:      1600,
		Port:                 8093,
	},
	{
		TierLabel:      types.TierMid,
		Name:           "gemma-4-e2b-q4_k_m",
		FilePath:       `models\gemma-4-e2b-q4_k_m.gguf`,
		RequiredVRAMMB: 2963,
		Modality:       "text",
		TotalLayers:    30,
		KVCacheMB:      450,
		KVFixedMB:      250,
		Port:           8091,
	},
	{
		TierLabel:      types.TierStrong,
		Name:           "gemma-4-e4b-q4_k_m",
		FilePath:       `models\gemma-4-e4b-q4_k_m.gguf`,
		RequiredVRAMMB: 4747,
		Modality:       "text",
		TotalLayers:    34,
		KVCacheMB:      580,
		KVFixedMB:      320,
		Port:           8092,
	},
}

// rosterPorts is the set of localhost ports our llama-server subprocesses use;
// the reaper only ever kills stray llama-server processes on one of these.
var rosterPorts = func() []int {
	ports := make([]int, 0, len(defaultRoster))
	for _, d := range defaultRoster {
		ports = append(ports, d.Port)
	}
	return ports
}()

func main() {
	logBuf := logbuf.NewBuffer(5000, 24*time.Hour)
	logOpts := &slog.HandlerOptions{}
	if os.Getenv("HIS_DEBUG_LOG") != "" {
		logOpts.Level = slog.LevelDebug
	}
	logger := slog.New(logBuf.Handler(slog.NewJSONHandler(os.Stdout, logOpts)))
	slog.SetDefault(logger)

	var flagChatsDir string
	var flagBackend string
	flag.StringVar(&flagChatsDir, "chats-dir", "", "directory for persisted chat conversations (default: %LOCALAPPDATA%\\home-inference-server\\chats)")
	flag.StringVar(&flagBackend, "backend", "llamacpp", "inference backend: 'llamacpp' (real, needs the GPU + models) or 'stub' (fake, for CI / the conformance suite)")
	flag.Parse()

	cfg := loadConfig()
	port := envStr("PORT", "8080")
	maxParallel := envInt("HIS_MAX_PARALLEL", defaultMaxParallel)
	if maxParallel < 1 {
		maxParallel = 1
	}

	nvmlProvider := vram.NVMLProvider{}

	// vramProvider is what the tier-selection logic queries for headroom. In stub
	// mode it is a fixed, ample value so the pipeline (queue → batcher → router →
	// vram.Backend → inner) runs end to end without a GPU.
	var vramProvider vram.VRAMProvider = nvmlProvider
	if flagBackend == "stub" {
		vramProvider = &vram.MockVRAMProvider{FreeMB: 8000}
	}

	// Build one inner backend per roster entry, keyed by tier label. A tier
	// whose MMProjPath is set (HasVision()) serves both modalities through the
	// same subprocess identity — vram.Backend decides per-request whether to
	// spawn it with --mmproj (see SetNextNeedsVision).
	var llamaInners []*llamacpp.Backend
	inners := make(map[types.ModelTierLabel]backend.Backend, len(defaultRoster))
	if flagBackend == "stub" {
		for _, desc := range defaultRoster {
			inners[desc.TierLabel] = stub.New(backend.ModalityKindText, 0)
		}
		slog.Warn("running with the STUB inference backend — responses are fake")
	} else {
		for _, desc := range defaultRoster {
			lb := llamacpp.New(backend.ModalityKindText, desc, llamaExe, nvmlProvider, maxParallel)
			inners[desc.TierLabel] = lb
			llamaInners = append(llamaInners, lb)
		}
	}

	// Reaper kills any stray llama-server left on a roster port by a previous
	// crash or a failed eviction. "owned" is the live set of our subprocess PIDs.
	reaper := vram.NewReaper(rosterPorts, func() map[int]bool {
		owned := make(map[int]bool, len(llamaInners))
		for _, lb := range llamaInners {
			if pid := lb.RunningPID(); pid != 0 {
				owned[pid] = true
			}
		}
		return owned
	})
	if flagBackend != "stub" {
		if n := reaper.Sweep(); n > 0 { // clear orphans from a previous run before we start
			slog.Warn("startup: reaped orphaned subprocess(es) from a previous run", slog.Int("count", n))
		}
	}

	q := queue.New(256)

	notifyCh := make(chan string, 4)

	opts := vram.DefaultOptions()
	opts.NotifyCh = notifyCh
	opts.MaxParallel = maxParallel
	if flagBackend != "stub" {
		opts.Reaper = reaper
	}

	vramBackend := vram.New(backend.ModalityKindText, defaultRoster, vramProvider, inners, opts)

	backends := map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   vramBackend,
		backend.ModalityKindVision: vramBackend, // same instance — selectInner branches on req.ImageData
	}

	r := router.New(backends)

	startedAt := time.Now()

	// The bounded worker pool: maxParallel workers drain flushed batches into
	// the router concurrently. The hard concurrency ceiling is enforced one
	// level down by vram.Backend's semaphore (which also gates the streaming
	// path that bypasses the router).
	disp := dispatcher.New(r, maxParallel)

	batchCfg := batcher.DefaultConfig()
	b := batcher.New(batchCfg, q.Drain(), func(batch []queue.Job) {
		// Hand the batch to the pool without blocking the batcher goroutine. If
		// the pool's intake buffer is momentarily full, dispatch inline rather
		// than drop the batch.
		if !disp.Submit(batch) {
			go r.Dispatch(context.Background(), batch)
		}
	})

	srv := server.New(q, func() types.ServerStatus {
		availMB, _ := nvmlProvider.AvailableMB()

		var loaded *types.LoadedModel
		if snap := vramBackend.LoadedModel(); snap != nil {
			loaded = &types.LoadedModel{
				Descriptor:  snap.Descriptor,
				LoadedAt:    snap.LoadedAt,
				GPULayers:   snap.GPULayers,
				TotalLayers: snap.TotalLayers,
			}
		}

		return types.ServerStatus{
			Timestamp:       time.Now(),
			QueueDepth:      q.Depth(),
			ActiveBatchSize: b.ActiveBatchSize(),
			InFlight:        disp.InFlight(),
			MaxParallel:     disp.MaxParallel(),
			LoadedModel:     loaded,
			AvailableVRAMMB: availMB,
			UptimeSince:     startedAt,
			Version:         version,
		}
	}, version, startedAt)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cpuMon := isystem.NewCPUMonitor(ctx)
	srv.SetPressureSources(cpuMon, vramBackend)
	srv.SetBackends(backends)
	if flagBackend == "stub" {
		srv.SetTokenizer(stubTokenizer{})
		srv.SetDescriber(stubDescriber{})
	} else {
		srv.SetTokenizer(vramBackend)
		srv.SetDescriber(vramBackend)
	}
	srv.SetLogSource(logBuf)
	if inferTOMS := envInt("HIS_INFER_TIMEOUT_MS", 0); inferTOMS > 0 {
		srv.SetInferTimeout(time.Duration(inferTOMS) * time.Millisecond)
	}

	chatsDir := flagChatsDir
	if chatsDir == "" {
		chatsDir = os.Getenv("CHATS_DIR")
	}
	if chatsDir == "" {
		chatsDir = filepath.Join(os.Getenv("LOCALAPPDATA"), "home-inference-server", "chats")
	}
	if store, err := chatstore.Open(chatsDir); err != nil {
		slog.Warn("chat persistence disabled", "dir", chatsDir, "err", err)
	} else {
		srv.SetChatStore(store)
		slog.Info("chat persistence enabled", "dir", chatsDir)
	}

	go disp.Run(ctx)
	go b.Run(ctx)
	slog.Info("inference worker pool started", slog.Int(logschema.FieldMaxParallel, maxParallel))

	restartCh := make(chan struct{}, 1)
	srv.SetRestartChannel(restartCh)

	if cfg.RailwayQueueURL != "" {
		remoteClient := remote.New(cfg.RailwayQueueURL, cfg.RailwayAPIKey, q, srv.RecordJob)
		go remoteClient.Run(ctx)
		srv.SetPauseSink(remoteClient) // admin drain/resume also pauses relay claiming
		// The same relay is where the local server hands off requests it can't
		// serve under pressure (durable deferral). No relay ⇒ the server never
		// defers; it serves, cascades tiers, or 503s from the queue.
		srv.SetRelayEnqueuer(relayenqueue.New(cfg.RailwayQueueURL, cfg.RailwayAPIKey), nvmlProvider)
		slog.Info("railway offline queue enabled", "url", cfg.RailwayQueueURL)
	} else {
		slog.Warn("no railway_queue_url (config.json or RAILWAY_QUEUE_URL) — offline queue + durable deferral disabled, LAN-only mode")
	}

	dashHandler := dashboard.New(assets.FS)

	mux := http.NewServeMux()
	mux.Handle("/v1/", srv)
	mux.Handle("/healthz", srv)
	mux.Handle("/", dashHandler)

	// Optional LAN hardening (all off by default — the local API is
	// unauthenticated unless HIS_API_KEY is set). CORS wraps auth so an OPTIONS
	// preflight isn't rejected by the key check.
	var handler http.Handler = mux
	handler = server.APIKeyAuth(os.Getenv("HIS_API_KEY"), handler)
	if origins := os.Getenv("HIS_CORS_ORIGINS"); origins != "" {
		handler = server.CORS(strings.Split(origins, ","), handler)
		slog.Info("CORS enabled", "origins", origins)
	}

	bindAddr := os.Getenv("HIS_BIND_ADDR")
	if bindAddr == "" {
		bindAddr = fmt.Sprintf(":%s", port)
	}

	dashboardURL := fmt.Sprintf("http://localhost:%s", port)
	go tray.Run(ctx, dashboardURL, notifyCh)

	httpServer := &http.Server{
		Addr:              bindAddr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// ReadTimeout / WriteTimeout are intentionally 0: /v1/infer streams SSE
		// and long-poll-style requests must not be cut off mid-response. The
		// per-request inference deadline (server.defaultInferTimeout /
		// HIS_INFER_TIMEOUT_MS) bounds work instead.
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		slog.Info("server starting",
			slog.String(logschema.FieldEvent, string(logschema.EventServerStart)),
			"addr", httpServer.Addr, "version", version, "dashboard", dashboardURL,
		)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("http server error", "err", err)
			os.Exit(1)
		}
	}()

	restart := false
	select {
	case <-sigCh:
		slog.Info("shutdown signal received")
	case <-restartCh:
		restart = true
		slog.Info("restart requested via /v1/admin/restart")
	}
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		slog.Error("http shutdown error", "err", err)
	}

	vramBackend.Shutdown(shutdownCtx) //nolint:errcheck
	q.Close()
	slog.Info("server stopped", slog.String(logschema.FieldEvent, string(logschema.EventServerStop)))

	if restart {
		writeRestartSentinel()
		os.Exit(0) // the run.cmd loop sees restart.flag and re-execs the binary
	}
}

// writeRestartSentinel drops a restart.flag next to the executable so the
// run.cmd relaunch loop knows to re-exec after this process exits. Best-effort:
// a write failure just means the loop won't relaunch and the operator restarts
// by hand — it never blocks the exit.
func writeRestartSentinel() {
	exe, err := os.Executable()
	if err != nil {
		slog.Error("restart: cannot locate executable for sentinel", "err", err)
		return
	}
	flag := filepath.Join(filepath.Dir(exe), "restart.flag")
	if err := os.WriteFile(flag, []byte(time.Now().Format(time.RFC3339)+"\n"), 0o644); err != nil {
		slog.Error("restart: cannot write sentinel", "path", flag, "err", err)
		return
	}
	slog.Info("restart: sentinel written", "path", flag)
}

func envStr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// stubTokenizer backs /v1/tokenize and /v1/model/props when running with
// --backend=stub, so the conformance suite can exercise those routes with no
// llama-server. It approximates token count as whitespace-separated words.
type stubTokenizer struct{}

func (stubTokenizer) Tokenize(_ context.Context, text string) (int, error) {
	return len(strings.Fields(text)), nil
}
func (stubTokenizer) NCtx(context.Context) (int, error) { return 4096, nil }

// stubDescriber backs /v1/describe under --backend=stub so the conformance
// suite can exercise the route with no vision model.
type stubDescriber struct{}

func (stubDescriber) Describe(_ context.Context, _ []byte) (string, string, error) {
	return "a stub image description", "weak", nil
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
		slog.Warn("ignoring non-integer env var", "key", key, "value", v)
	}
	return fallback
}
