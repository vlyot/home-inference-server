package railwayq

import (
	"context"
	"crypto/subtle"
	"database/sql"
	"errors"
	"log/slog"
	"net/http"
	"strings"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/apps"
	"github.com/ngkaichong/home-inference-server/internal/stackauth"
	"github.com/ngkaichong/home-inference-server/logschema"
)

type ctxKey int

const identityCtxKey ctxKey = 0

// identityFrom returns the resolved caller identity stashed by identityMiddleware.
func identityFrom(ctx context.Context) string {
	id, _ := ctx.Value(identityCtxKey).(string)
	return id
}

// OwnerIdentity is the identity of the static X-API-Key (QUEUE_API_KEY): the
// owner's own clients and the PC worker. It is the only identity allowed on
// the worker routes.
const OwnerIdentity = "apikey"

// workerPaths are the PC worker's routes. Claiming a job reads its prompt and
// acking one writes its answer, so only the owner key may call them — an
// invited member or an app key must never be able to impersonate the PC.
var workerPaths = map[string]bool{
	"/claim": true,
	"/ack":   true,
}

// resolveIdentity extracts the caller identity from a request:
//   - a valid "Authorization: Bearer <stack-auth-token>" whose subject is on
//     the invite list yields "user:<stack-user-id>";
//   - an X-API-Key secret app key ("sk_…") yields "app:<id>" and its App, for
//     the app's own limits;
//   - an X-API-Key matching QUEUE_API_KEY yields OwnerIdentity.
//
// A publishable app key ("pk_…") is refused here: it only works on the public
// /apps/ask route. The returned status is the HTTP code to send when ok is
// false: 401 for a missing/invalid credential, 403 for a valid token whose
// user is not an invited member.
func resolveIdentity(ctx context.Context, r *http.Request, cfg Config, db *sql.DB, verifier *stackauth.Verifier) (identity string, app *App, status int, ok bool) {
	authz := r.Header.Get(api.HeaderAuthorization)
	if strings.HasPrefix(authz, "Bearer ") && verifier != nil {
		token := strings.TrimSpace(authz[len("Bearer "):])
		claims, err := verifier.Verify(ctx, token)
		if err != nil {
			return "", nil, http.StatusUnauthorized, false
		}
		allowed, err := IsAllowedMember(ctx, db, claims.Subject)
		if err != nil {
			return "", nil, http.StatusInternalServerError, false
		}
		if !allowed {
			return "", nil, http.StatusForbidden, false
		}
		return "user:" + claims.Subject, nil, 0, true
	}

	got := r.Header.Get(api.HeaderAPIKey)
	if IsAppKey(got) {
		a, err := LookupKey(ctx, db, got)
		if errors.Is(err, ErrNotFound) || (err == nil && a.Kind != apps.KeySecret) {
			return "", nil, http.StatusUnauthorized, false
		}
		if err != nil {
			return "", nil, http.StatusInternalServerError, false
		}
		return a.Identity(), &a, 0, true
	}
	if cfg.StaticAPIKey != "" && got != "" &&
		subtle.ConstantTimeCompare([]byte(got), []byte(cfg.StaticAPIKey)) == 1 {
		return OwnerIdentity, nil, 0, true
	}
	return "", nil, http.StatusUnauthorized, false
}

// publicPaths lists routes that bypass identity resolution entirely: the
// liveness check, and the mirrored API docs page (see dashboard.New) — the
// docs are reference material meant to be readable by any client or AI
// agent hitting this relay's public URL, not a member-gated capability — and
// /apps/ask, which authenticates its own publishable key and carries per-app
// limits (see ask.go), and /apps/status, which says only whether the PC is
// online (see status_public.go).
var publicPaths = map[string]bool{
	"/healthz":     true,
	"/docs":        true,
	"/apps/ask":    true,
	"/apps/status": true,
}

// identityMiddleware authenticates every request (except publicPaths), keeps
// everyone but the owner off the worker routes and, for /enqueue, enforces
// the per-identity rate limit and daily quota — the app's own limits for a
// secret app key, the global ones otherwise.
func identityMiddleware(db *sql.DB, cfg Config, limiter *Limiter, verifier *stackauth.Verifier, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if publicPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}

		identity, app, status, ok := resolveIdentity(r.Context(), r, cfg, db, verifier)
		if !ok {
			switch status {
			case http.StatusForbidden:
				writeErr(w, http.StatusForbidden, api.ErrCodeUnauthorized, "not an invited member")
			case http.StatusInternalServerError:
				writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "credential check failed")
			default:
				writeErr(w, http.StatusUnauthorized, api.ErrCodeUnauthorized, "missing or invalid credentials")
			}
			return
		}

		if workerPaths[r.URL.Path] && identity != OwnerIdentity {
			writeErr(w, http.StatusForbidden, api.ErrCodeUnauthorized, "worker routes need the owner key")
			return
		}

		if r.URL.Path == "/enqueue" {
			perMin, quota := cfg.RatePerMin, cfg.DailyQuota
			if app != nil {
				perMin, quota = app.Spec.Limits.PerMin, app.Spec.Limits.PerDay
			}
			if !limiter.AllowAt(identity, perMin) {
				slog.Warn("relay rate limit hit",
					slog.String(logschema.FieldEvent, string(logschema.EventRateLimited)),
					slog.String(logschema.FieldIdentity, identity),
				)
				writeErr(w, http.StatusTooManyRequests, api.ErrCodeRateLimited, "rate limit exceeded, retry shortly")
				return
			}
			used, err := UsageToday(r.Context(), db, identity)
			if err != nil {
				slog.Error("usage check failed", "err", err, slog.String(logschema.FieldIdentity, identity))
				writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "usage check failed")
				return
			}
			if used >= quota {
				slog.Warn("relay daily quota hit",
					slog.String(logschema.FieldEvent, string(logschema.EventQuotaExceeded)),
					slog.String(logschema.FieldIdentity, identity),
					slog.Int(logschema.FieldCount, used),
				)
				writeErr(w, http.StatusTooManyRequests, api.ErrCodeQuotaExceeded, "daily quota exhausted, resets at UTC midnight")
				return
			}
		}

		ctx := context.WithValue(r.Context(), identityCtxKey, identity)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
