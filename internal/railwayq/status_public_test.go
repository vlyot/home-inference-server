package railwayq

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func getPublicStatus(t *testing.T, h http.Handler, method string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, "/apps/status", nil))
	return rec
}

func decodePublicStatus(t *testing.T, rec *httptest.ResponseRecorder) PublicStatusResponse {
	t.Helper()
	var body PublicStatusResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return body
}

func TestPublicStatus_FollowsThePCsLastRequest(t *testing.T) {
	pc := &pcPresence{}
	now := time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)
	h := handlePublicStatus(pc.online, func() time.Time { return now })

	if body := decodePublicStatus(t, getPublicStatus(t, h, http.MethodGet)); body.Online {
		t.Fatal("online before the PC ever polled")
	}

	pc.seen(now)
	now = now.Add(59 * time.Second)
	rec := getPublicStatus(t, h, http.MethodGet)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	body := decodePublicStatus(t, rec)
	if !body.Online || !body.CheckedAt.Equal(now) {
		t.Fatalf("body = %+v, want online, checked at %v", body, now)
	}
	if got := rec.Header().Get("Cache-Control"); got != "public, max-age=15" {
		t.Fatalf("Cache-Control = %q", got)
	}

	now = now.Add(time.Second) // 60 s since the last poll
	if body := decodePublicStatus(t, getPublicStatus(t, h, http.MethodGet)); body.Online {
		t.Fatal("still online 60 s after the last poll")
	}
}

func TestPublicStatus_NoAuthWildcardCORS(t *testing.T) {
	online := func(time.Time) bool { return true }
	h := identityMiddleware(nil, Config{StaticAPIKey: "secret"}, NewLimiter(60), nil, handlePublicStatus(online, time.Now))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/apps/status", nil)
	req.Header.Set("Origin", "https://vlyot.github.io")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 without credentials", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Fatalf("Access-Control-Allow-Origin = %q, want *", got)
	}
}

func TestPublicStatus_RejectsNonGET(t *testing.T) {
	calls := 0
	online := func(time.Time) bool { calls++; return true }
	rec := getPublicStatus(t, handlePublicStatus(online, time.Now), http.MethodPost)
	if rec.Code != http.StatusMethodNotAllowed || calls != 0 {
		t.Fatalf("status = %d, calls = %d; want 405 and no check", rec.Code, calls)
	}
}
