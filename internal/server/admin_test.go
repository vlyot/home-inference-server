package server_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
)

// fakePauseSink records Pause/Resume calls for the drain-sink assertions.
type fakePauseSink struct {
	pauses  atomic.Int32
	resumes atomic.Int32
}

func (f *fakePauseSink) Pause()  { f.pauses.Add(1) }
func (f *fakePauseSink) Resume() { f.resumes.Add(1) }

func adminPost(t *testing.T, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestAdminDrainRejectsNewInfer(t *testing.T) {
	h := newHarness(t, 0)

	if resp := adminPost(t, h.ts.URL+api.PathAdminDrain); resp.StatusCode != http.StatusOK {
		t.Fatalf("drain: status = %d, want 200", resp.StatusCode)
	}

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("infer while drained: status = %d, want 503", resp.StatusCode)
	}
	var er api.ErrorResponse
	_ = json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeDraining {
		t.Fatalf("error code = %q, want %q", er.Code, api.ErrCodeDraining)
	}
}

func TestAdminResumeRestoresInfer(t *testing.T) {
	h := newHarness(t, 0)
	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()
	adminPost(t, h.ts.URL+api.PathAdminResume).Body.Close()

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("infer after resume: status = %d, want 200", resp.StatusCode)
	}
}

func TestAdminDrainDoesNotBlockReadOnly(t *testing.T) {
	h := newHarness(t, 0)
	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()

	for _, path := range []string{api.PathHealth, api.PathStatus} {
		resp, err := http.Get(h.ts.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s while drained: status = %d, want 200", path, resp.StatusCode)
		}
	}
}

func TestAdminHealthReportsDrainingState(t *testing.T) {
	h := newHarness(t, 0)

	get := func() string {
		resp, err := http.Get(h.ts.URL + api.PathHealth)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var hr api.HealthResponse
		_ = json.NewDecoder(resp.Body).Decode(&hr)
		return hr.State
	}

	if got := get(); got != api.StateOK {
		t.Fatalf("state before drain = %q, want %q", got, api.StateOK)
	}
	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()
	if got := get(); got != api.StateDraining {
		t.Fatalf("state after drain = %q, want %q", got, api.StateDraining)
	}
	adminPost(t, h.ts.URL+api.PathAdminResume).Body.Close()
	if got := get(); got != api.StateOK {
		t.Fatalf("state after resume = %q, want %q", got, api.StateOK)
	}
}

func TestAdminStatusReportsDrainingState(t *testing.T) {
	h := newHarness(t, 0)
	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()

	resp, err := http.Get(h.ts.URL + api.PathStatus)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var sr struct {
		State string `json:"state"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&sr)
	if sr.State != api.StateDraining {
		t.Fatalf("/v1/status state = %q, want %q", sr.State, api.StateDraining)
	}
}

func TestAdminEndpointsRequireLoopback(t *testing.T) {
	h := newHarness(t, 0)
	for _, path := range []string{api.PathAdmin, api.PathAdminDrain, api.PathAdminResume, api.PathAdminRestart} {
		rec := httptest.NewRecorder()
		method := http.MethodPost
		if path == api.PathAdmin {
			method = http.MethodGet
		}
		req := httptest.NewRequest(method, path, nil)
		req.RemoteAddr = "203.0.113.7:5555"
		h.srv.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotFound {
			t.Errorf("%s from non-loopback: status = %d, want 404", path, rec.Code)
		}
	}
}

func TestAdminRestartSignalsChannel(t *testing.T) {
	h := newHarness(t, 0)
	ch := make(chan struct{}, 1)
	h.srv.SetRestartChannel(ch)

	resp := adminPost(t, h.ts.URL+api.PathAdminRestart)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("restart: status = %d, want 200", resp.StatusCode)
	}
	select {
	case <-ch:
	default:
		t.Fatal("restart channel received no signal")
	}
	if !h.srv.Draining() {
		t.Error("restart should have set the drain flag")
	}
}

func TestAdminRestartWithoutChannelReturns501(t *testing.T) {
	h := newHarness(t, 0) // no SetRestartChannel
	resp := adminPost(t, h.ts.URL+api.PathAdminRestart)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("restart without channel: status = %d, want 501", resp.StatusCode)
	}
}

func TestAdminDrainCallsPauseSink(t *testing.T) {
	h := newHarness(t, 0)
	sink := &fakePauseSink{}
	h.srv.SetPauseSink(sink)

	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()
	adminPost(t, h.ts.URL+api.PathAdminResume).Body.Close()

	if p := sink.pauses.Load(); p != 1 {
		t.Errorf("Pause calls = %d, want 1", p)
	}
	if r := sink.resumes.Load(); r != 1 {
		t.Errorf("Resume calls = %d, want 1", r)
	}
}

func TestAdminDrainIdempotent(t *testing.T) {
	h := newHarness(t, 0)
	sink := &fakePauseSink{}
	h.srv.SetPauseSink(sink)

	for i := 0; i < 3; i++ {
		resp := adminPost(t, h.ts.URL+api.PathAdminDrain)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("drain #%d: status = %d, want 200", i, resp.StatusCode)
		}
	}
	if p := sink.pauses.Load(); p != 1 {
		t.Errorf("Pause called %d times for repeated drains, want 1", p)
	}
}

func TestAdminWrongMethod(t *testing.T) {
	h := newHarness(t, 0)
	resp, err := http.Get(h.ts.URL + api.PathAdminDrain) // drain is POST-only
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/admin/drain: status = %d, want 405", resp.StatusCode)
	}
	if allow := resp.Header.Get("Allow"); allow != http.MethodPost {
		t.Errorf("Allow header = %q, want POST", allow)
	}
}

func TestAdminGetStateProbe(t *testing.T) {
	h := newHarness(t, 0)
	resp, err := http.Get(h.ts.URL + api.PathAdmin)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/admin: status = %d, want 200", resp.StatusCode)
	}
	var ar api.AdminStateResponse
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		t.Fatal(err)
	}
	if ar.State != api.StateOK {
		t.Errorf("state = %q, want %q", ar.State, api.StateOK)
	}
}

func TestInferStreamAlsoDrained(t *testing.T) {
	h := newHarness(t, 0)
	adminPost(t, h.ts.URL+api.PathAdminDrain).Body.Close()

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "hi"},
		Stream:    true,
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("streaming infer while drained: status = %d, want 503", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct == "text/event-stream" {
		t.Error("drained streaming request should not open an SSE stream")
	}
}
