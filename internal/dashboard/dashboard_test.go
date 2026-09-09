package dashboard_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/assets"
	"github.com/ngkaichong/home-inference-server/internal/dashboard"
)

func TestGetRootReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET / want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("want text/html Content-Type, got %q", ct)
	}
}

func TestGetDashboardReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/dashboard", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /dashboard want 200, got %d", w.Code)
	}
}

func TestGetDocsReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/docs", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /docs want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("want text/html, got %q", ct)
	}
}

func TestGetChatReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/chat", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /chat want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("want text/html, got %q", ct)
	}
}

func TestGetLogsReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/logs", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /logs want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("want text/html, got %q", ct)
	}
}

func TestGetVendorMarkedReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/vendor/marked.min.js", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /vendor/marked.min.js want 200, got %d", w.Code)
	}
}

func TestGetVendorDompurifyReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/vendor/dompurify.min.js", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("GET /vendor/dompurify.min.js want 200, got %d", w.Code)
	}
}

func TestGetVendorUnknownFileReturns404(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/vendor/nonexistent.js", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Fatalf("GET /vendor/nonexistent.js want 404, got %d", w.Code)
	}
}

func TestGetChatHTMLRawPathReturns404(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/chat.html", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Fatalf("GET /chat.html want 404, got %d", w.Code)
	}
}

func TestUnknownPathReturns404(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/unknown", nil)
	h.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Fatalf("GET /unknown want 404, got %d", w.Code)
	}
}

func TestGetAdminReturns200(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/admin", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("GET /admin want 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("want text/html, got %q", ct)
	}
}

func TestGetAdminHTMLRawPathReturns404(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/admin.html", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("GET /admin.html want 404, got %d", w.Code)
	}
}

func TestPostAdminPageReturns404(t *testing.T) {
	h := dashboard.New(assets.FS)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/admin", nil))
	if w.Code != http.StatusNotFound {
		t.Fatalf("POST /admin (dashboard handler) want 404, got %d", w.Code)
	}
}

func TestAdminHTMLHasControlIDs(t *testing.T) {
	b, err := assets.FS.ReadFile("web/admin.html")
	if err != nil {
		t.Fatalf("read admin.html: %v", err)
	}
	body := string(b)
	for _, id := range []string{`id="btn-drain"`, `id="btn-resume"`, `id="btn-restart"`, `id="state-badge"`, `id="admin-notice"`} {
		if !strings.Contains(body, id) {
			t.Errorf("admin.html missing %s", id)
		}
	}
}
