package railwayq

import (
	"context"
	"crypto/subtle"
	"database/sql"
	"log/slog"
	"net/http"
	"strings"

	"github.com/ngkaichong/home-inference-server/api"
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

// resolveIdentity extracts the caller identity from a request. A valid
// "Authorization: Bearer <stack-auth-token>" whose subject is on the invite
// list yields "user:<stack-user-id>"; a matching X-API-Key yields "apikey".
//
// The returned status is the HTTP code to send when ok is false: 401 for a
// missing/invalid credential, 403 for a valid token whose user is not an
// invited member.
func resolveIdentity(ctx context.Context, r *http.Request, cfg Config, db *sql.DB, verifier *stackauth.Verifier) (identity string, status int, ok bool) {
	authz := r.Header.Get(api.HeaderAuthorization)
	if strings.HasPrefix(authz, "Bearer ") && verifier != nil {
		token := strings.TrimSpace(authz[len("Bearer "):])
		claims, err := verifier.Verify(ctx, token)
		if err != nil {
			return "", http.StatusUnauthorized, false
		}
		allowed, err := IsAllowedMember(ctx, db, claims.Subject)
		if err != nil {
			return "", http.StatusInternalServerError, false
		}
		if !allowed {
			return "", http.StatusForbidden, false
		}
		return "user:" + claims.Subject, 0, true
	}

	if cfg.StaticAPIKey != "" {
		if got := r.Header.Get(api.HeaderAPIKey); got != "" &&
			subtle.ConstantTimeCompare([]byte(got), []byte(cfg.StaticAPIKey)) == 1 {
			return "apikey", 0, true
		}
	}
	return "", http.StatusUnauthorized, false
}

// identityMiddleware authenticates every request (except /healthz) and, for
// /enqueue, enforces the per-identity rate limit and daily quota.
func identityMiddleware(db *sql.DB, cfg Config, limiter *Limiter, verifier *stackauth.Verifier, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}

		identity, status, ok := resolveIdentity(r.Context(), r, cfg, db, verifier)
		if !ok {
			switch status {
			case http.StatusForbidden:
				writeErr(w, http.StatusForbidden, api.ErrCodeUnauthorized, "not an invited member")
			case http.StatusInternalServerError:
				writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "membership check failed")
			default:
				writeErr(w, http.StatusUnauthorized, api.ErrCodeUnauthorized, "missing or invalid credentials")
			}
			return
		}

		if r.URL.Path == "/enqueue" {
			if !limiter.Allow(identity) {
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
			if used >= cfg.DailyQuota {
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
