package railwayq

import (
	"context"
	"net/http"
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

// publicStatusTTL is how long one database lookup answers GET /apps/status,
// so a flood of visitors costs at most one query per TTL.
const publicStatusTTL = 15 * time.Second

// PublicStatusResponse is the body of GET /apps/status: only whether the PC
// is online. Queue depth and exact poll times stay behind the member-only
// /status.
type PublicStatusResponse struct {
	Online    bool      `json:"online"`
	CheckedAt time.Time `json:"checked_at"`
}

// handlePublicStatus serves GET /apps/status to any origin without a key, so
// a public site can show whether its chat has a model behind it. check reports
// whether the PC is connected; now is the clock, injectable for tests.
func handlePublicStatus(check func(context.Context) (bool, error), now func() time.Time) http.HandlerFunc {
	var (
		mu     sync.Mutex
		cached *PublicStatusResponse
	)

	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeErr(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "use GET")
			return
		}

		mu.Lock()
		defer mu.Unlock()
		if cached == nil || now().Sub(cached.CheckedAt) >= publicStatusTTL {
			online, err := check(r.Context())
			if err != nil {
				writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "status lookup failed")
				return
			}
			cached = &PublicStatusResponse{Online: online, CheckedAt: now().UTC()}
		}
		w.Header().Set("Cache-Control", "public, max-age=15")
		writeJSON(w, http.StatusOK, cached)
	}
}
