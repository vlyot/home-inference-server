// Package dashboard serves the read-only web dashboard embedded in the binary.
package dashboard

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
)

// New returns an http.Handler that serves dashboard.html on GET / and
// GET /dashboard, docs.html on GET /docs, chat.html on GET /chat,
// logs.html on GET /logs, and admin.html on GET /admin (the loopback-only
// drain/resume/restart control page — its buttons only work when opened on the
// server itself). GET /vendor/* serves embedded third-party assets directly by
// path. All other paths return 404.
func New(files embed.FS) http.Handler {
	sub, err := fs.Sub(files, "web")
	if err != nil {
		panic("dashboard: missing embedded web/ directory: " + err.Error())
	}
	fileServer := http.FileServer(http.FS(sub))

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		switch {
		case r.URL.Path == "/" || r.URL.Path == "/dashboard":
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/dashboard.html"
			fileServer.ServeHTTP(w, r2)
		case r.URL.Path == "/docs":
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/docs.html"
			fileServer.ServeHTTP(w, r2)
		case r.URL.Path == "/chat":
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/chat.html"
			fileServer.ServeHTTP(w, r2)
		case r.URL.Path == "/logs":
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/logs.html"
			fileServer.ServeHTTP(w, r2)
		case r.URL.Path == "/admin":
			r2 := r.Clone(r.Context())
			r2.URL.Path = "/admin.html"
			fileServer.ServeHTTP(w, r2)
		case strings.HasPrefix(r.URL.Path, "/vendor/"):
			fileServer.ServeHTTP(w, r)
		default:
			http.NotFound(w, r)
		}
	})
}
