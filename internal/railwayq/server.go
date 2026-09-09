package railwayq

import (
	"database/sql"
	"net/http"

	"github.com/ngkaichong/home-inference-server/internal/stackauth"
)

// New returns an http.Handler with the relay API routes wrapped in identity
// authentication (Neon Auth / Stack Auth Bearer token, gated by the invite
// list, or the static X-API-Key) plus per-identity rate limiting and daily
// quota on /enqueue. verifier may be nil to disable the Bearer path.
func New(db *sql.DB, cfg Config, limiter *Limiter, verifier *stackauth.Verifier) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/enqueue", handleEnqueue(db, cfg))
	mux.HandleFunc("/claim", handleClaim(db))
	mux.HandleFunc("/ack", handleAck(db))
	mux.HandleFunc("/result/", handleResult(db))
	mux.HandleFunc("/status", handleStatus(db))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return identityMiddleware(db, cfg, limiter, verifier, mux)
}
