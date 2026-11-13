package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
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
	"github.com/ngkaichong/home-inference-server/internal/config"
	"github.com/ngkaichong/home-inference-server/internal/dashboard"
	"github.com/ngkaichong/home-inference-server/internal/dispatcher"
	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/relayenqueue"
	"github.com/ngkaichong/home-inference-server/internal/remote"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/internal/setup"
	isystem "github.com/ngkaichong/home-inference-server/internal/system"
	"github.com/ngkaichong/home-inference-server/internal/tray"
	"github.com/ngkaichong/home-inference-server/internal/websearch"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// version is overridden at release build time via -ldflags "-X main.version=…".
var version = "0.13.0"

// startupRoster builds the roster this run serves. The stub backend keeps the
// full configured roster (it needs no files); the real backend drops tiers whose
// GGUF isn't on disk and refuses to start with none.
func startupRoster(cfg config.Config, backendKind string, exists func(string) bool) ([]types.ModelDescriptor, error) {
	roster, err := cfg.Roster()
	if err != nil {
		return nil, err
	}
	if backendKind == "stub" {
		return roster, nil
	}
	kept, notes := config.FilterInstalled(roster, exists)
	for _, n := range notes {
		slog.Warn("model roster: " + n)
	}
	if len(kept) == 0 {
		var want []string
		for _, d := range roster {
			want = append(want, filepath.Base(d.FilePath))
		}
		dir, _ := filepath.Abs(cfg.ModelsDir)
		return nil, fmt.Errorf("no model files found in %s; expected at least one of: %s",
			dir, strings.Join(want, ", "))
	}
	return kept, nil
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "setup" {
		fs := flag.NewFlagSet("setup", flag.ExitOnError)
		path := fs.String("config", "", "where to write config.json (default: the per-user config folder)")
		_ = fs.Parse(os.Args[2:]) // ExitOnError: Parse exits on a bad flag
		if err := setup.Run(os.Stdin, os.Stdout, setup.OSEnv(*path)); err != nil {
			exitf("setup: %v", err)
		}
		return
	}

	logBuf := logbuf.NewBuffer(5000, 24*time.Hour)
	logOpts := &slog.HandlerOptions{}
	if os.Getenv("HIS_DEBUG_LOG") != "" {
		logOpts.Level = slog.LevelDebug
	}
	logger := slog.New(logBuf.Handler(slog.NewJSONHandler(os.Stdout, logOpts)))
	slog.SetDefault(logger)

	var flagChatsDir, flagBackend, flagConfig string
	flag.StringVar(&flagConfig, "config", "", "path to config.json (default: ./config.json if present, else the per-user config folder)")
	flag.StringVar(&flagChatsDir, "chats-dir", "", "directory for persisted chat conversations (overrides config chats_dir)")
	flag.StringVar(&flagBackend, "backend", "llamacpp", "inference backend: 'llamacpp' (real, needs the GPU + models) or 'stub' (fake, for CI / the conformance suite)")
	flag.Parse()

	osEnv := config.OSEnv()
	cfgPath := config.Locate(flagConfig, osEnv)
	cfg, unknownKeys, err := config.Load(cfgPath, osEnv)
	if err != nil {
		exitf("config: %v", err)
	}
	slog.Info("config loaded", "path", cfgPath, "exists", config.FileExists(cfgPath))
	for _, k := range unknownKeys {
		slog.Warn("config: ignoring unknown key", "path", cfgPath, "key", k)
	}
	if flagChatsDir != "" {
		cfg.ChatsDir = flagChatsDir
	}
	maxParallel := cfg.MaxParallel
	if cfg.CtxSize > 0 {
		llamacpp.SetCtxSize(cfg.CtxSize)
		if cfg.CtxSize != llamacpp.DefaultCtxSize {
			slog.Warn("ctx_size differs from the 4096 the roster's KV estimates were measured at; VRAM fit math may be off",
				slog.Int("ctx_size", cfg.CtxSize))
		}
	}

	roster, err := startupRoster(cfg, flagBackend, config.FileExists)
	if err != nil {
		exitf("models: %v", err)
	}
	if flagBackend != "stub" && !config.FileExists(cfg.EnginePath) {
		exitf("llama-server not found (engine_path %q). Install it (macOS: brew install llama.cpp) or set engine_path in %s", cfg.EnginePath, cfgPath)
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
	inners := make(map[types.ModelTierLabel]backend.Backend, len(roster))
	if flagBackend == "stub" {
		for _, desc := range roster {
			inners[desc.TierLabel] = stub.New(backend.ModalityKindText, 0)
		}
		slog.Warn("running with the STUB inference backend — responses are fake")
	} else {
		for _, desc := range roster {
			lb := llamacpp.New(backend.ModalityKindText, desc, cfg.EnginePath, nvmlProvider, maxParallel)
			inners[desc.TierLabel] = lb
			llamaInners = append(llamaInners, lb)
		}
	}

	// Reaper kills any stray llama-server left on a roster port by a previous
	// crash or a failed eviction. "owned" is the live set of our subprocess PIDs.
	reaper := vram.NewReaper(rosterPorts(roster), func() map[int]bool {
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

	vramBackend := vram.New(backend.ModalityKindText, roster, vramProvider, inners, opts)

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
	if cfg.InferTimeoutMS > 0 {
		srv.SetInferTimeout(time.Duration(cfg.InferTimeoutMS) * time.Millisecond)
	}

	srv.SetToolSearcher(websearch.NewClient(10 * time.Second))

	chatsDir := cfg.ChatsDir
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
	handler = server.APIKeyAuth(cfg.APIKey, handler)
	if len(cfg.CORSOrigins) > 0 {
		handler = server.CORS(cfg.CORSOrigins, handler)
		slog.Info("CORS enabled", "origins", strings.Join(cfg.CORSOrigins, ","))
	}

	bindAddr := cfg.ListenAddr()
	dashboardURL := fmt.Sprintf("http://localhost:%d", cfg.Port)
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

// rosterPorts is the set of localhost ports our llama-server subprocesses use;
// the reaper only ever kills stray llama-server processes on one of these.
func rosterPorts(roster []types.ModelDescriptor) []int {
	ports := make([]int, 0, len(roster))
	for _, d := range roster {
		ports = append(ports, d.Port)
	}
	return ports
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "home-inference-server: "+format+"\n", args...)
	os.Exit(1)
}
