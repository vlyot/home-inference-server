package websearch

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
)

// withTestURL points tavilySearchURL at srv.URL for the duration of the
// calling test, restoring the real Tavily URL afterward.
func withTestURL(t *testing.T, srv *httptest.Server) {
	t.Helper()
	orig := tavilySearchURL
	tavilySearchURL = srv.URL
	t.Cleanup(func() { tavilySearchURL = orig })
}

func TestSearch_ReturnsParsedResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Tavily-Access-Mode"); got != "keyless" {
			t.Errorf("X-Tavily-Access-Mode = %q, want keyless", got)
		}
		var body struct {
			Query      string `json:"query"`
			MaxResults int    `json:"max_results"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Query != "golang" {
			t.Errorf("query = %q, want golang", body.Query)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"results":[
			{"title":"A","url":"https://a.example","content":"about a"},
			{"title":"B","url":"https://b.example","content":"about b"},
			{"title":"C","url":"https://c.example","content":"about c"},
			{"title":"D","url":"https://d.example","content":"about d"},
			{"title":"E","url":"https://e.example","content":"about e"},
			{"title":"F","url":"https://f.example","content":"about f"}
		]}`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	results, err := c.Search(context.Background(), "golang")
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != maxResults {
		t.Fatalf("len(results) = %d, want %d (capped)", len(results), maxResults)
	}
	if results[0] != (Result{Title: "A", URL: "https://a.example", Content: "about a"}) {
		t.Errorf("results[0] = %+v, want A", results[0])
	}
}

func TestSearch_EmptyResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	results, err := c.Search(context.Background(), "nothing")
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("len(results) = %d, want 0", len(results))
	}
}

func TestSearch_NonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	_, err := c.Search(context.Background(), "q")
	var be *backend.BackendError
	if !isBackendErr(err, &be) {
		t.Fatalf("err = %v, want *backend.BackendError", err)
	}
	if be.Code != "tool_unavailable" {
		t.Errorf("Code = %q, want tool_unavailable", be.Code)
	}
}

func TestSearch_RateLimited(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":"Your request has been blocked due to excessive requests. Please reduce the rate of requests."}`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	_, err := c.Search(context.Background(), "q")
	var be *backend.BackendError
	if !isBackendErr(err, &be) {
		t.Fatalf("err = %v, want *backend.BackendError", err)
	}
	if be.Code != "tool_rate_limited" {
		t.Errorf("Code = %q, want tool_rate_limited", be.Code)
	}
	if be.Message != "Your request has been blocked due to excessive requests. Please reduce the rate of requests." {
		t.Errorf("Message = %q, want Tavily's error text surfaced verbatim", be.Message)
	}
}

func TestSearch_RateLimitedWithUnparsableBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`not json`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	_, err := c.Search(context.Background(), "q")
	var be *backend.BackendError
	if !isBackendErr(err, &be) {
		t.Fatalf("err = %v, want *backend.BackendError", err)
	}
	if be.Code != "tool_rate_limited" {
		t.Errorf("Code = %q, want tool_rate_limited (even when the body can't be parsed)", be.Code)
	}
	if be.Message == "" {
		t.Error("Message should fall back to a generic rate-limit message when the body doesn't parse")
	}
}

// TestSearch_RateLimitedKeylessEnvelope covers the keyless-specific error
// shape confirmed from Tavily's own Python SDK source
// (tavily/tavily.py:_raise_keyless_envelope) — a nested "error" object with
// "window" and "retry_after_seconds", distinct from the flat {"error":
// "<string>"} shape documented on the general rate-limits page.
func TestSearch_RateLimitedKeylessEnvelope(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"code":"keyless_limit_exceeded","message":"Keyless quota exhausted","window":"1h","retry_after_seconds":3200,"next_actions":["sign up for a free API key"]}}`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	_, err := c.Search(context.Background(), "q")
	var be *backend.BackendError
	if !isBackendErr(err, &be) {
		t.Fatalf("err = %v, want *backend.BackendError", err)
	}
	if be.Code != "tool_rate_limited" {
		t.Errorf("Code = %q, want tool_rate_limited", be.Code)
	}
	for _, want := range []string{"Keyless quota exhausted", "1h", "3200"} {
		if !strings.Contains(be.Message, want) {
			t.Errorf("Message = %q, want it to contain %q", be.Message, want)
		}
	}
}

// flakyThenOKListener wraps a net.Listener so its first Accept fails
// (simulating a connection reset before the server ever sees the request),
// then behaves normally — used to test Search's one-retry-on-connection-failure
// behavior without a real flaky external dependency.
type flakyThenOKListener struct {
	net.Listener
	failedOnce bool
}

func (l *flakyThenOKListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	if !l.failedOnce {
		l.failedOnce = true
		conn.Close() // reset before any HTTP response is written
		return l.Accept()
	}
	return conn, nil
}

func TestSearch_RetriesOnceOnConnectionFailure(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	flaky := &flakyThenOKListener{Listener: ln}

	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"results":[{"title":"ok","url":"https://x","content":"y"}]}`))
	}))
	srv.Listener = flaky
	srv.Start()
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	results, err := c.Search(context.Background(), "q")
	if err != nil {
		t.Fatalf("Search returned error after retry: %v", err)
	}
	if len(results) != 1 || results[0].Title != "ok" {
		t.Errorf("results = %+v, want one result titled 'ok'", results)
	}
}

func TestSearch_DoesNotRetryOnNonOKStatus(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()
	withTestURL(t, srv)

	c := NewClient(5 * time.Second)
	start := time.Now()
	_, err := c.Search(context.Background(), "q")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error")
	}
	if hits != 1 {
		t.Errorf("server hit %d times, want 1 (a non-OK status must not trigger a retry)", hits)
	}
	if elapsed >= retryWait {
		t.Errorf("Search took %v, want well under retryWait (%v) — it should not have waited to retry", elapsed, retryWait)
	}
}

func TestSearch_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.Write([]byte(`{"results":[]}`))
	}))
	defer srv.Close()
	withTestURL(t, srv)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	c := NewClient(5 * time.Second)
	_, err := c.Search(ctx, "q")
	var be *backend.BackendError
	if !isBackendErr(err, &be) {
		t.Fatalf("err = %v, want *backend.BackendError", err)
	}
	if be.Code != "timeout" {
		t.Errorf("Code = %q, want timeout", be.Code)
	}
}

func isBackendErr(err error, out **backend.BackendError) bool {
	be, ok := err.(*backend.BackendError)
	if ok {
		*out = be
	}
	return ok
}
