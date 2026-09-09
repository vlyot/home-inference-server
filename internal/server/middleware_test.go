package server_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/server"
)

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
}

func TestAPIKeyPassthroughWhenUnset(t *testing.T) {
	ts := httptest.NewServer(server.APIKeyAuth("", okHandler()))
	defer ts.Close()
	resp, _ := http.Get(ts.URL + "/v1/status")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (no key configured)", resp.StatusCode)
	}
}

func TestAPIKeyExemptForLoopback(t *testing.T) {
	// httptest server is reached over loopback, so no key is needed.
	ts := httptest.NewServer(server.APIKeyAuth("secret", okHandler()))
	defer ts.Close()
	resp, _ := http.Get(ts.URL + "/v1/status")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (loopback exempt)", resp.StatusCode)
	}
}

func TestAPIKeyEnforcedForNonLoopback(t *testing.T) {
	h := server.APIKeyAuth("secret", okHandler())
	// Simulate a remote client by setting RemoteAddr directly.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/status", nil)
	req.RemoteAddr = "203.0.113.7:5555"
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("no key from remote: status = %d, want 401", rec.Code)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/v1/status", nil)
	req.RemoteAddr = "203.0.113.7:5555"
	req.Header.Set(api.HeaderAPIKey, "secret")
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("correct key from remote: status = %d, want 200", rec.Code)
	}
}

func TestAPIKeyDoesNotGuardNonV1(t *testing.T) {
	h := server.APIKeyAuth("secret", okHandler())
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.RemoteAddr = "203.0.113.7:5555"
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/healthz from remote: status = %d, want 200 (not guarded)", rec.Code)
	}
}

func TestCORSDisabledByDefault(t *testing.T) {
	ts := httptest.NewServer(server.CORS(nil, okHandler()))
	defer ts.Close()
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/v1/status", nil)
	req.Header.Set("Origin", "https://app.example")
	resp, _ := http.DefaultClient.Do(req)
	if resp.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("CORS header present with no origins configured")
	}
}

func TestCORSPreflightAndAllowlist(t *testing.T) {
	h := server.CORS([]string{"https://app.example"}, okHandler())
	ts := httptest.NewServer(h)
	defer ts.Close()

	// Allowed origin, preflight.
	req, _ := http.NewRequest(http.MethodOptions, ts.URL+"/v1/infer", nil)
	req.Header.Set("Origin", "https://app.example")
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("preflight status = %d, want 204", resp.StatusCode)
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "https://app.example" {
		t.Fatalf("Allow-Origin = %q", resp.Header.Get("Access-Control-Allow-Origin"))
	}
	if !strings.Contains(resp.Header.Get("Access-Control-Allow-Headers"), api.HeaderAPIKey) {
		t.Fatalf("Allow-Headers missing %s", api.HeaderAPIKey)
	}

	// Disallowed origin gets no CORS header.
	req, _ = http.NewRequest(http.MethodGet, ts.URL+"/v1/status", nil)
	req.Header.Set("Origin", "https://evil.example")
	resp, _ = http.DefaultClient.Do(req)
	if resp.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("Allow-Origin set for a disallowed origin")
	}
}

func TestIsLoopbackExported(t *testing.T) {
	cases := map[string]bool{
		"127.0.0.1:5555":   true,
		"[::1]:80":         true,
		"127.9.9.9:1":      true,
		"192.168.1.9:80":   false,
		"203.0.113.7:5555": false,
		"10.0.0.1":         false,
	}
	for addr, want := range cases {
		if got := server.IsLoopback(addr); got != want {
			t.Errorf("IsLoopback(%q) = %v, want %v", addr, got, want)
		}
	}
}
