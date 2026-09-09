package server

import (
	"net"
	"net/http"
	"strings"

	"github.com/ngkaichong/home-inference-server/api"
)

// APIKeyAuth gates /v1/* on X-API-Key when key is non-empty. Loopback callers
// (127.0.0.0/8, ::1) are exempt so the local dashboard/chat/dev keep working.
// A missing/wrong key returns 401 unauthorized. When key == "" the middleware
// is a pass-through (the default — the local API is unauthenticated).
func APIKeyAuth(key string, next http.Handler) http.Handler {
	if key == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1/") || IsLoopback(r.RemoteAddr) {
			next.ServeHTTP(w, r)
			return
		}
		if r.Header.Get(api.HeaderAPIKey) == key {
			next.ServeHTTP(w, r)
			return
		}
		writeError(w, http.StatusUnauthorized, api.ErrCodeUnauthorized, "missing or invalid X-API-Key", "")
	})
}

// IsLoopback reports whether remoteAddr (an http.Request.RemoteAddr, i.e. the
// real TCP peer — not a spoofable header) is a loopback address (127.0.0.0/8 or
// ::1). Gates the local dashboard exemption and the loopback-only admin surface.
func IsLoopback(remoteAddr string) bool {
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// CORS adds Access-Control-* headers and answers OPTIONS preflight when the
// request Origin is in allowOrigins (or allowOrigins contains "*"). An empty
// allowOrigins slice disables CORS entirely (the default).
func CORS(allowOrigins []string, next http.Handler) http.Handler {
	if len(allowOrigins) == 0 {
		return next
	}
	allowAll := false
	set := map[string]bool{}
	for _, o := range allowOrigins {
		o = strings.TrimSpace(o)
		if o == "*" {
			allowAll = true
		} else if o != "" {
			set[o] = true
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && (allowAll || set[origin]) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, "+api.HeaderAPIKey+", "+api.HeaderCorrelationID)
			w.Header().Set("Access-Control-Max-Age", "600")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
