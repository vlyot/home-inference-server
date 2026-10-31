package railwayq

import (
	"context"
	"database/sql"
	"net/http"

	"github.com/ngkaichong/home-inference-server/assets"
	"github.com/ngkaichong/home-inference-server/internal/apps"
	"github.com/ngkaichong/home-inference-server/internal/dashboard"
	"github.com/ngkaichong/home-inference-server/internal/stackauth"
)

// New returns an http.Handler with the relay API routes wrapped in identity
// authentication (Neon Auth / Stack Auth Bearer token, gated by the invite
// list, or the static X-API-Key) plus per-identity rate limiting and daily
// quota on /enqueue. verifier may be nil to disable the Bearer path.
//
// /docs mirrors the same docs.html served locally by the home server's own
// /docs (see internal/dashboard) — this relay is reachable from the public
// internet, so it doubles as a public API reference other clients or AI
// agents can read without needing loopback access to the home machine.
// dashboard.New's routing table covers more paths (/, /chat, /admin, ...)
// than are meaningful here; mounting it at /docs is enough — it internally
// maps that exact path to the embedded docs.html file.
//
// /apps/ask is the public route for browser-only apps: a publishable key
// selects the app's server-side spec and the relay builds the prompt (see
// docs/public-apps.md). knowledge loads apps' knowledge files; nil leaves any
// app with a knowledge_url answering 503 unavailable.
func New(db *sql.DB, cfg Config, limiter *Limiter, verifier *stackauth.Verifier, knowledge *apps.Registry) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/enqueue", handleEnqueue(db, cfg))
	mux.HandleFunc("/claim", handleClaim(db))
	mux.HandleFunc("/ack", handleAck(db))
	mux.HandleFunc("/result/", handleResult(db))
	mux.HandleFunc("/status", handleStatus(db))
	mux.HandleFunc("/members/me", handleMembersMe(db))
	mux.HandleFunc("DELETE /job/{id}", handleCancel(db))
	mux.HandleFunc("/apps/ask", handleAsk(cfg, dbAskStore{db}, limiter, func(ctx context.Context, url string) *apps.Index {
		if knowledge == nil {
			return nil
		}
		return knowledge.Index(ctx, url)
	}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.Handle("/docs", dashboard.New(assets.FS))
	return identityMiddleware(db, cfg, limiter, verifier, mux)
}
