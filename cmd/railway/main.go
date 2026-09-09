package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/railwayq"
	"github.com/ngkaichong/home-inference-server/internal/stackauth"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	dbURL := envStr("DATABASE_URL", "")
	if dbURL == "" {
		slog.Error("DATABASE_URL is required")
		os.Exit(1)
	}
	port := envStr("PORT", "8081")

	cfg, err := railwayq.LoadConfig()
	if err != nil {
		slog.Error("config error", "err", err)
		os.Exit(1)
	}

	dbURL = railwayq.NormalizeDSN(dbURL)

	db, err := railwayq.Open(dbURL)
	if err != nil {
		slog.Error("failed to open database", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	limiter := railwayq.NewLimiter(cfg.RatePerMin)
	evictorStop := make(chan struct{})
	go limiter.StartEvictor(evictorStop)
	go railwayq.StartTTLSweeper(ctx, db, cfg.ResultTTL)

	var verifier *stackauth.Verifier
	if cfg.StackJWKSURL != "" {
		verifier = stackauth.New(cfg.StackJWKSURL, cfg.StackProjectID)
	}

	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%s", port),
		Handler: railwayq.New(db, cfg, limiter, verifier),
	}

	go func() {
		slog.Info("railway relay server starting", "addr", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("http server error", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutdown signal received")
	close(evictorStop)

	shutCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutCtx); err != nil {
		slog.Error("http shutdown error", "err", err)
	}
	slog.Info("railway relay server stopped")
}

func envStr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
