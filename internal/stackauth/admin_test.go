package stackauth

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// testAdminClient points an AdminClient at a local httptest.Server instead
// of the real Stack Auth API.
func testAdminClient(t *testing.T, handler http.HandlerFunc) *AdminClient {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c := NewAdminClient("proj-123", "ssk-test")
	c.http = srv.Client()
	// admin.go's adminBaseURL is a package constant; tests need requests to
	// land on srv instead, so do() is exercised through a client whose
	// requests we redirect via a custom RoundTripper pointed at srv's URL.
	base := srv.URL
	c.http.Transport = rewriteHostTransport{base: base}
	return c
}

// rewriteHostTransport rewrites every request's scheme+host to base,
// preserving path and query, so tests can point AdminClient's hardcoded
// adminBaseURL at an httptest.Server.
type rewriteHostTransport struct{ base string }

func (t rewriteHostTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	target, err := req.URL.Parse(t.base + req.URL.Path + "?" + req.URL.RawQuery)
	if err != nil {
		return nil, err
	}
	req = req.Clone(req.Context())
	req.URL = target
	req.Host = ""
	return http.DefaultTransport.RoundTrip(req)
}

func TestFindUserByEmailMatchFound(t *testing.T) {
	c := testAdminClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Stack-Secret-Server-Key") != "ssk-test" {
			t.Errorf("missing secret key header")
		}
		if r.URL.Query().Get("query") != "user@example.com" {
			t.Errorf("query = %q", r.URL.Query().Get("query"))
		}
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"items": []map[string]any{
				{"id": "other-id", "primary_email": "someone-else@example.com"},
				{"id": "user-id-1", "primary_email": "User@Example.com"},
			},
		})
	})
	id, err := c.FindUserByEmail(context.Background(), "user@example.com")
	if err != nil {
		t.Fatalf("FindUserByEmail: %v", err)
	}
	if id != "user-id-1" {
		t.Fatalf("id = %q, want user-id-1 (case-insensitive match)", id)
	}
}

func TestFindUserByEmailNoMatch(t *testing.T) {
	c := testAdminClient(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"items": []map[string]any{}}) //nolint:errcheck
	})
	id, err := c.FindUserByEmail(context.Background(), "nobody@example.com")
	if err != nil {
		t.Fatalf("FindUserByEmail: %v", err)
	}
	if id != "" {
		t.Fatalf("id = %q, want empty", id)
	}
}

func TestCreateUserSendsExpectedRequestAndReturnsID(t *testing.T) {
	c := testAdminClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.Header.Get("X-Stack-Access-Type") != "server" {
			t.Errorf("missing access-type header")
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
		if body["primary_email"] != "new@example.com" {
			t.Errorf("primary_email = %v", body["primary_email"])
		}
		if body["password"] != "temp-pw-123" {
			t.Errorf("password = %v", body["password"])
		}
		if body["display_name"] != "New Person" {
			t.Errorf("display_name = %v", body["display_name"])
		}
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]any{"id": "new-id-1"}) //nolint:errcheck
	})
	id, err := c.CreateUser(context.Background(), "new@example.com", "New Person", "temp-pw-123")
	if err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if id != "new-id-1" {
		t.Fatalf("id = %q, want new-id-1", id)
	}
}

func TestCreateUserOmitsBlankDisplayName(t *testing.T) {
	c := testAdminClient(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
		if _, ok := body["display_name"]; ok {
			t.Errorf("display_name present with blank input: %v", body["display_name"])
		}
		json.NewEncoder(w).Encode(map[string]any{"id": "new-id-2"}) //nolint:errcheck
	})
	if _, err := c.CreateUser(context.Background(), "new2@example.com", "", "pw"); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
}

func TestAdminClientNonOKStatusReturnsError(t *testing.T) {
	c := testAdminClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusConflict)
		w.Write([]byte(`{"code":"EMAIL_ALREADY_EXISTS"}`)) //nolint:errcheck
	})
	_, err := c.CreateUser(context.Background(), "dupe@example.com", "", "pw")
	if err == nil {
		t.Fatal("expected error on 409 response")
	}
}
