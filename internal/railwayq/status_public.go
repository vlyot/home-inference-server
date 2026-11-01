package railwayq

import (
	"net/http"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

// PublicStatusResponse is the body of GET /apps/status: only whether the PC
// is online. Queue depth and exact poll times stay behind the member-only
// /status.
type PublicStatusResponse struct {
	Online    bool      `json:"online"`
	CheckedAt time.Time `json:"checked_at"`
}

// handlePublicStatus serves GET /apps/status to any origin without a key, so
// a public site can show whether its chat has a model behind it. online
// reports whether the PC is connected at a given time, from memory, so a
// visitor watching the page never wakes the database; now is the clock,
// injectable for tests.
func handlePublicStatus(online func(time.Time) bool, now func() time.Time) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeErr(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "use GET")
			return
		}
		t := now()
		w.Header().Set("Cache-Control", "public, max-age=15")
		writeJSON(w, http.StatusOK, PublicStatusResponse{Online: online(t), CheckedAt: t.UTC()})
	}
}
