package railwayq_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/internal/apps"
	"github.com/ngkaichong/home-inference-server/internal/railwayq"
	"github.com/ngkaichong/home-inference-server/queue"
)

func registerApp(t *testing.T, db *sql.DB, id string, mod func(*apps.Spec)) apps.Spec {
	t.Helper()
	s := apps.Spec{ID: id, Origins: []string{"https://a.test"}, SystemPrompt: "Be brief."}
	if mod != nil {
		mod(&s)
	}
	s.ApplyDefaults()
	if err := railwayq.UpsertApp(context.Background(), db, s); err != nil {
		t.Fatalf("upsert %s: %v", id, err)
	}
	return s
}

func issue(t *testing.T, db *sql.DB, appID, kind string) string {
	t.Helper()
	key, err := railwayq.IssueKey(context.Background(), db, appID, kind)
	if err != nil {
		t.Fatalf("issue %s key: %v", kind, err)
	}
	return key
}

func TestAppKeyLookupIgnoresRevokedAndDisabled(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	registerApp(t, db, "demo", nil)
	key := issue(t, db, "demo", apps.KeyPublishable)
	if !strings.HasPrefix(key, "pk_") {
		t.Fatalf("key %q lacks pk_ prefix", key)
	}

	app, err := railwayq.LookupKey(ctx, db, key)
	if err != nil || app.Spec.ID != "demo" || app.Kind != apps.KeyPublishable || app.Identity() != "app:demo" {
		t.Fatalf("lookup = %+v, %v", app, err)
	}

	if err := railwayq.SetAppEnabled(ctx, db, "demo", false); err != nil {
		t.Fatal(err)
	}
	if _, err := railwayq.LookupKey(ctx, db, key); !errors.Is(err, railwayq.ErrNotFound) {
		t.Fatalf("disabled app lookup err = %v, want ErrNotFound", err)
	}
	railwayq.SetAppEnabled(ctx, db, "demo", true) //nolint:errcheck

	if err := railwayq.RevokeKey(ctx, db, key[:12]); err != nil {
		t.Fatal(err)
	}
	if _, err := railwayq.LookupKey(ctx, db, key); !errors.Is(err, railwayq.ErrNotFound) {
		t.Fatalf("revoked key lookup err = %v, want ErrNotFound", err)
	}
	if err := railwayq.RevokeKey(ctx, db, key[:12]); !errors.Is(err, railwayq.ErrNotFound) {
		t.Fatalf("second revoke err = %v, want ErrNotFound", err)
	}
}

func TestUpsertAppReplacesSpecAndKeepsKeys(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	registerApp(t, db, "demo", nil)
	key := issue(t, db, "demo", apps.KeySecret)
	registerApp(t, db, "demo", func(s *apps.Spec) { s.SystemPrompt = "Be thorough." })

	app, err := railwayq.LookupKey(ctx, db, key)
	if err != nil || app.Spec.SystemPrompt != "Be thorough." {
		t.Fatalf("after re-upsert lookup = %+v, %v", app, err)
	}
	list, err := railwayq.ListApps(ctx, db)
	if err != nil || len(list) != 1 || len(list[0].Keys) != 1 || list[0].Keys[0].Prefix != key[:12] {
		t.Fatalf("list = %+v, %v", list, err)
	}
}

func apiKeyReq(method, path, key string, body *strings.Reader) *http.Request {
	var req *http.Request
	if body == nil {
		req = httptest.NewRequest(method, path, nil)
	} else {
		req = httptest.NewRequest(method, path, body)
	}
	if key != "" {
		req.Header.Set("X-API-Key", key)
	}
	return req
}

func TestSecretAppKeyEnqueuesUnderAppIdentityAndLimits(t *testing.T) {
	srv, _, db := newTestServer(t)
	registerApp(t, db, "demo", func(s *apps.Spec) { s.Limits.PerDay = 1 })
	key := issue(t, db, "demo", apps.KeySecret)

	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, apiKeyReq(http.MethodPost, "/enqueue", key, strings.NewReader(`{"correlation_id":"c1","payload":{"modality":"text"}}`)))
	if rr.Code != http.StatusCreated {
		t.Fatalf("sk enqueue = %d %s", rr.Code, rr.Body)
	}
	var resp queue.EnqueueResponse
	json.Unmarshal(rr.Body.Bytes(), &resp) //nolint:errcheck
	var identity string
	db.QueryRow(`SELECT identity FROM jobs WHERE id = $1`, resp.ID).Scan(&identity) //nolint:errcheck
	if identity != "app:demo" {
		t.Fatalf("job identity = %q, want app:demo", identity)
	}

	rr = httptest.NewRecorder()
	srv.ServeHTTP(rr, apiKeyReq(http.MethodPost, "/enqueue", key, strings.NewReader(`{"correlation_id":"c2","payload":{"modality":"text"}}`)))
	if rr.Code != http.StatusTooManyRequests {
		t.Fatalf("second sk enqueue over per_day 1 = %d, want 429", rr.Code)
	}
}

func TestPublishableKeyCannotUseRawContract(t *testing.T) {
	srv, _, db := newTestServer(t)
	registerApp(t, db, "demo", nil)
	key := issue(t, db, "demo", apps.KeyPublishable)
	for _, path := range []string{"/enqueue", "/status"} {
		rr := httptest.NewRecorder()
		srv.ServeHTTP(rr, apiKeyReq(http.MethodPost, path, key, strings.NewReader(`{"correlation_id":"c","payload":{}}`)))
		if rr.Code != http.StatusUnauthorized {
			t.Fatalf("pk on %s = %d, want 401", path, rr.Code)
		}
	}
}

func TestWorkerRoutesNeedOwnerKey(t *testing.T) {
	srv, _, db := newTestServer(t)
	registerApp(t, db, "demo", nil)
	sk := issue(t, db, "demo", apps.KeySecret)
	bearer := invite(t, db, "member")

	for _, path := range []string{"/claim", "/ack"} {
		for name, set := range map[string]func(*http.Request){
			"member": func(r *http.Request) { r.Header.Set("Authorization", bearer) },
			"sk":     func(r *http.Request) { r.Header.Set("X-API-Key", sk) },
		} {
			rr := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"worker_id":"w"}`))
			set(req)
			srv.ServeHTTP(rr, req)
			if rr.Code != http.StatusForbidden {
				t.Errorf("%s on %s = %d, want 403", name, path, rr.Code)
			}
		}
	}

	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, apiKeyReq(http.MethodPost, "/ack", testAPIKey, strings.NewReader(`{}`)))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("owner key on /ack = %d, want past auth (400 for the empty body)", rr.Code)
	}
}

func TestCancelJobEndpoint(t *testing.T) {
	srv, _, db := newTestServer(t)
	ctx := context.Background()
	registerApp(t, db, "demo", nil)
	sk := issue(t, db, "demo", apps.KeySecret)
	railwayq.Enqueue(ctx, db, newJobWithIdentity("mine", "app:demo"))     //nolint:errcheck
	railwayq.Enqueue(ctx, db, newJobWithIdentity("theirs", "user:alice")) //nolint:errcheck
	railwayq.Enqueue(ctx, db, newJobWithIdentity("claimed", "app:demo"))  //nolint:errcheck
	db.Exec(`UPDATE jobs SET status = 'processing' WHERE id = 'claimed'`) //nolint:errcheck

	for _, tc := range []struct {
		id   string
		want int
	}{{"mine", http.StatusNoContent}, {"mine", http.StatusNotFound}, {"theirs", http.StatusNotFound}, {"claimed", http.StatusConflict}} {
		rr := httptest.NewRecorder()
		srv.ServeHTTP(rr, apiKeyReq(http.MethodDelete, "/job/"+tc.id, sk, nil))
		if rr.Code != tc.want {
			t.Errorf("DELETE /job/%s = %d, want %d", tc.id, rr.Code, tc.want)
		}
	}
	var n int
	db.QueryRow(`SELECT COUNT(*) FROM jobs WHERE id = 'theirs'`).Scan(&n) //nolint:errcheck
	if n != 1 {
		t.Fatal("another identity's job was deleted")
	}
}
