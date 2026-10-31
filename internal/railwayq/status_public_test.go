package railwayq

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type statusCheck struct {
	online bool
	err    error
	calls  int
}

func (c *statusCheck) check(context.Context) (bool, error) {
	c.calls++
	return c.online, c.err
}

func getPublicStatus(t *testing.T, h http.Handler, method string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(method, "/apps/status", nil))
	return rec
}

func TestPublicStatus_ReportsOnlineFromCheck(t *testing.T) {
	c := &statusCheck{online: true}
	rec := getPublicStatus(t, handlePublicStatus(c.check, time.Now), http.MethodGet)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body PublicStatusResponse
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if !body.Online || body.CheckedAt.IsZero() {
		t.Fatalf("body = %+v, want online with a check time", body)
	}
}

func TestPublicStatus_CachesForFifteenSeconds(t *testing.T) {
	c := &statusCheck{online: false}
	now := time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)
	h := handlePublicStatus(c.check, func() time.Time { return now })

	getPublicStatus(t, h, http.MethodGet)
	now = now.Add(14 * time.Second)
	getPublicStatus(t, h, http.MethodGet)
	if c.calls != 1 {
		t.Fatalf("calls within the TTL = %d, want 1", c.calls)
	}
	now = now.Add(time.Second)
	getPublicStatus(t, h, http.MethodGet)
	if c.calls != 2 {
		t.Fatalf("calls after the TTL = %d, want 2", c.calls)
	}
}

func TestPublicStatus_NoAuthWildcardCORS(t *testing.T) {
	c := &statusCheck{online: true}
	h := identityMiddleware(nil, Config{StaticAPIKey: "secret"}, NewLimiter(60), nil, handlePublicStatus(c.check, time.Now))
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
	c := &statusCheck{}
	rec := getPublicStatus(t, handlePublicStatus(c.check, time.Now), http.MethodPost)
	if rec.Code != http.StatusMethodNotAllowed || c.calls != 0 {
		t.Fatalf("status = %d, calls = %d; want 405 and no lookup", rec.Code, c.calls)
	}
}

func TestPublicStatus_LookupErrorIsNotCached(t *testing.T) {
	c := &statusCheck{err: errors.New("db down")}
	h := handlePublicStatus(c.check, time.Now)
	if rec := getPublicStatus(t, h, http.MethodGet); rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	c.err, c.online = nil, true
	if rec := getPublicStatus(t, h, http.MethodGet); rec.Code != http.StatusOK || c.calls != 2 {
		t.Fatalf("status = %d, calls = %d; want a fresh lookup after an error", rec.Code, c.calls)
	}
}
