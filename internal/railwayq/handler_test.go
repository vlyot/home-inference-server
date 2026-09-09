package railwayq_test

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/railwayq"
	"github.com/ngkaichong/home-inference-server/internal/stackauth"
	"github.com/ngkaichong/home-inference-server/queue"
)

const (
	testAPIKey    = "test-static-key"
	testProjectID = "test-stack-project"
)

// issuer is a process-wide ES256 token minter; its Verifier is wired into every
// test server so Bearer tokens it signs are accepted.
var issuer = stackauth.NewTestIssuer(testProjectID)

func testConfig() railwayq.Config {
	return railwayq.Config{
		ResultTTL:       time.Hour,
		MaxPendingDepth: 50,
		RatePerMin:      100,
		DailyQuota:      1000,
		MaxBodyBytes:    256,
		StackJWKSURL:    "http://unused.test/jwks",
		StackProjectID:  testProjectID,
		StaticAPIKey:    testAPIKey,
	}
}

func newTestServer(t *testing.T) (http.Handler, railwayq.Config, *sql.DB) {
	t.Helper()
	return newTestServerCfg(t, testConfig())
}

func newTestServerCfg(t *testing.T, cfg railwayq.Config) (http.Handler, railwayq.Config, *sql.DB) {
	t.Helper()
	db := openTestDB(t)
	return railwayq.New(db, cfg, railwayq.NewLimiter(cfg.RatePerMin), issuer.Verifier), cfg, db
}

// invite registers a Stack Auth subject in allowed_members on db and returns a
// Bearer header carrying a freshly signed token for it.
func invite(t *testing.T, db *sql.DB, sub string) string {
	t.Helper()
	_, err := db.ExecContext(context.Background(), `
		INSERT INTO allowed_members (stack_user_id, email) VALUES ($1, $2)
		ON CONFLICT (stack_user_id) DO NOTHING`, sub, sub+"@example.com")
	if err != nil {
		t.Fatalf("invite %s: %v", sub, err)
	}
	return "Bearer " + issuer.Token(sub, sub+"@example.com")
}

func enqueueBody(corr string) *bytes.Reader {
	b, _ := json.Marshal(queue.EnqueueRequest{
		CorrelationID: corr,
		Payload:       json.RawMessage(`{"modality":"text","text_input":{"prompt":"hi"}}`),
	})
	return bytes.NewReader(b)
}

func TestHealthzBypassesAuth(t *testing.T) {
	srv, _, _ := newTestServer(t)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("healthz = %d, want 200", rr.Code)
	}
}

func TestEnqueueRequiresAuth(t *testing.T) {
	srv, _, _ := newTestServer(t)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody("c1")))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("no-auth enqueue = %d, want 401", rr.Code)
	}
}

func TestEnqueueRejectsUninvitedMember(t *testing.T) {
	srv, _, _ := newTestServer(t)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody("c-uninvited"))
	// Valid token, but the subject was never added to allowed_members.
	req.Header.Set("Authorization", "Bearer "+issuer.Token("stranger", "stranger@example.com"))
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("uninvited enqueue = %d, want 403", rr.Code)
	}
}

func TestEnqueueAcceptsBearerAndKey(t *testing.T) {
	srv, _, db := newTestServer(t)
	bearerHdr := invite(t, db, "user-jwt")

	for _, tc := range []struct{ name, hdr, val string }{
		{"jwt", "Authorization", bearerHdr},
		{"apikey", "X-API-Key", testAPIKey},
	} {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody("c-"+tc.name))
		req.Header.Set(tc.hdr, tc.val)
		srv.ServeHTTP(rr, req)
		if rr.Code != http.StatusCreated {
			t.Fatalf("%s enqueue = %d, want 201 (body %s)", tc.name, rr.Code, rr.Body)
		}
		var resp queue.EnqueueResponse
		json.Unmarshal(rr.Body.Bytes(), &resp) //nolint:errcheck
		if resp.ID == "" {
			t.Fatalf("%s enqueue returned empty id", tc.name)
		}
	}
}

func TestEnqueueRejectsOversizeBody(t *testing.T) {
	srv, cfg, _ := newTestServer(t)
	big := strings.Repeat("x", 1024)
	b, _ := json.Marshal(queue.EnqueueRequest{
		CorrelationID: "big",
		Payload:       json.RawMessage(fmt.Sprintf(`{"p":%q}`, big)),
	})
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/enqueue", bytes.NewReader(b))
	req.Header.Set("X-API-Key", cfg.StaticAPIKey)
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize enqueue = %d, want 413", rr.Code)
	}
}

func TestEnqueueRejectsWhenQueueFull(t *testing.T) {
	cfg := testConfig()
	cfg.MaxPendingDepth = 3
	srv, _, _ := newTestServerCfg(t, cfg)
	for i := 0; i < 3; i++ {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody(fmt.Sprintf("full-%d", i)))
		req.Header.Set("X-API-Key", cfg.StaticAPIKey)
		srv.ServeHTTP(rr, req)
		if rr.Code != http.StatusCreated {
			t.Fatalf("setup enqueue %d = %d", i, rr.Code)
		}
	}
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody("overflow"))
	req.Header.Set("X-API-Key", cfg.StaticAPIKey)
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("queue-full enqueue = %d, want 503", rr.Code)
	}
}

func TestEnqueueRateLimited(t *testing.T) {
	cfg := testConfig()
	cfg.RatePerMin = 5 // burst 5
	srv, _, db := newTestServerCfg(t, cfg)
	bearerHdr := invite(t, db, "rl-user")
	code := 0
	for i := 0; i < 8; i++ {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody(fmt.Sprintf("rl-%d", i)))
		req.Header.Set("Authorization", bearerHdr)
		srv.ServeHTTP(rr, req)
		code = rr.Code
		if code == http.StatusTooManyRequests {
			break
		}
	}
	if code != http.StatusTooManyRequests {
		t.Fatalf("expected a 429 within the burst window, last code %d", code)
	}
}

func TestResultRequiresAuth(t *testing.T) {
	srv, _, _ := newTestServer(t)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/result/abc", nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("no-auth result = %d, want 401", rr.Code)
	}
}

func TestResultForeignJobIs404(t *testing.T) {
	srv, _, db := newTestServer(t)
	aliceHdr := invite(t, db, "alice")
	bobHdr := invite(t, db, "bob")

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/enqueue", enqueueBody("owned-job"))
	req.Header.Set("Authorization", aliceHdr)
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("setup enqueue = %d (%s)", rr.Code, rr.Body)
	}
	var resp queue.EnqueueResponse
	json.Unmarshal(rr.Body.Bytes(), &resp) //nolint:errcheck

	// Alice can read it.
	rr = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/result/"+resp.ID, nil)
	req.Header.Set("Authorization", aliceHdr)
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("owner result = %d, want 200", rr.Code)
	}

	// Bob cannot.
	rr = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/result/"+resp.ID, nil)
	req.Header.Set("Authorization", bobHdr)
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("foreign result = %d, want 404", rr.Code)
	}
	var er api.ErrorResponse
	json.Unmarshal(rr.Body.Bytes(), &er)
	if er.Code != api.ErrCodeNotFound {
		t.Errorf("foreign result code = %q, want %q", er.Code, api.ErrCodeNotFound)
	}
}

func TestStatusReportsPCConnected(t *testing.T) {
	srv, cfg, db := newTestServer(t)

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	req.Header.Set("X-API-Key", cfg.StaticAPIKey)
	srv.ServeHTTP(rr, req)
	var st queue.QueueStatusResponse
	json.Unmarshal(rr.Body.Bytes(), &st) //nolint:errcheck
	if st.PCConnected {
		t.Fatalf("expected pc_connected=false with no claims")
	}

	db.Exec(`INSERT INTO jobs (id, correlation_id, status, payload, claimed_at, attempt_count, max_attempts)
	         VALUES ('c-recent','r','processing','{}',NOW(),1,3)`) //nolint:errcheck
	rr = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodGet, "/status", nil)
	req.Header.Set("X-API-Key", cfg.StaticAPIKey)
	srv.ServeHTTP(rr, req)
	json.Unmarshal(rr.Body.Bytes(), &st) //nolint:errcheck
	if !st.PCConnected {
		t.Fatalf("expected pc_connected=true after a recent claim")
	}
}
