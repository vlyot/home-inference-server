package server_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/batcher"
	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// harness wires up a full in-process pipeline for testing.
type harness struct {
	srv    *server.Server
	ts     *httptest.Server
	cancel context.CancelFunc
}

func newHarness(t *testing.T, latency time.Duration) *harness {
	t.Helper()
	return newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: stub.New(backend.ModalityKindText, latency),
	})
}

func newMultiModalHarness(t *testing.T, latency time.Duration) *harness {
	t.Helper()
	return newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   stub.New(backend.ModalityKindText, latency),
		backend.ModalityKindVision: &fakeVisionBackend{output: "a description of the image"},
	})
}

// fakeVisionBackend stands in for the SmolVLM2→Gemma pipeline in server tests.
// It returns a canned 200 response, or a degraded one when degraded is set, or
// an error when inferErr is set. recordedReq captures the last request seen.
type fakeVisionBackend struct {
	output      string
	degraded    bool
	inferErr    error
	recordedReq backend.Request
}

func (f *fakeVisionBackend) Modality() backend.ModalityKind   { return backend.ModalityKindVision }
func (f *fakeVisionBackend) Ready() bool                      { return true }
func (f *fakeVisionBackend) Shutdown(_ context.Context) error { return nil }
func (f *fakeVisionBackend) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	f.recordedReq = req
	if f.inferErr != nil {
		return backend.Response{}, f.inferErr
	}
	return backend.Response{
		Output:          f.output,
		TokensGenerated: 5,
		ModelTier:       "weak",
		QualityDegraded: f.degraded,
	}, nil
}

// recordingBackend captures the backend.Request it is handed and returns a
// canned response or error. Used to assert request-translation behaviour
// (response_format passthrough) and error mapping at the HTTP layer.
type recordingBackend struct {
	modality    backend.ModalityKind
	recordedReq backend.Request
	inferErr    error
}

func (r *recordingBackend) Modality() backend.ModalityKind   { return r.modality }
func (r *recordingBackend) Ready() bool                      { return true }
func (r *recordingBackend) Shutdown(_ context.Context) error { return nil }
func (r *recordingBackend) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	r.recordedReq = req
	if r.inferErr != nil {
		return backend.Response{}, r.inferErr
	}
	return backend.Response{Output: "ok", TokensGenerated: 1, ModelTier: "weak"}, nil
}

func newHarnessWithBackends(t *testing.T, backends map[backend.ModalityKind]backend.Backend) *harness {
	t.Helper()
	q := queue.New(256)
	r := router.New(backends)

	var srv *server.Server
	cfg := batcher.Config{MaxBatchSize: 8, WindowDuration: 10 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	b := batcher.New(cfg, q.Drain(), func(batch []queue.Job) {
		r.Dispatch(ctx, batch)
	})
	go b.Run(ctx)

	srv = server.New(q, func() types.ServerStatus {
		return types.ServerStatus{
			Timestamp:       time.Now(),
			QueueDepth:      q.Depth(),
			AvailableVRAMMB: -1,
		}
	}, "test", time.Now())

	ts := httptest.NewServer(srv)
	t.Cleanup(func() {
		cancel()
		ts.Close()
	})

	return &harness{srv: srv, ts: ts, cancel: cancel}
}

func post(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestInferReturns200WithOutput(t *testing.T) {
	h := newHarness(t, 0)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "hello"},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var ir api.InferResponse
	if err := json.NewDecoder(resp.Body).Decode(&ir); err != nil {
		t.Fatal(err)
	}
	if ir.Output == "" {
		t.Error("output field is empty")
	}
	if ir.RequestID == "" {
		t.Error("request_id field is empty")
	}
}

func TestInferMissingModalityReturns400(t *testing.T) {
	h := newHarness(t, 0)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		TextInput: &api.TextInput{Prompt: "hello"},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeInvalidRequest {
		t.Errorf("want code %q, got %q", api.ErrCodeInvalidRequest, er.Code)
	}
}

func TestStatusReturnsQueueDepthAndHistory(t *testing.T) {
	h := newHarness(t, 5*time.Millisecond)

	// Fire one inference to populate history.
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "status test"},
	})
	resp.Body.Close()

	time.Sleep(50 * time.Millisecond)

	statusResp, err := http.Get(h.ts.URL + api.PathStatus)
	if err != nil {
		t.Fatal(err)
	}
	defer statusResp.Body.Close()

	if statusResp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", statusResp.StatusCode)
	}
	var ss types.ServerStatus
	if err := json.NewDecoder(statusResp.Body).Decode(&ss); err != nil {
		t.Fatal(err)
	}
	if len(ss.RecentJobs) == 0 {
		t.Error("expected at least one job in recent_jobs")
	}
}

func TestHealthz(t *testing.T) {
	h := newHarness(t, 0)
	resp, err := http.Get(h.ts.URL + api.PathHealth)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
}

func TestConcurrentRequests(t *testing.T) {
	for _, n := range []int{1, 2, 4, 8} {
		n := n
		t.Run("batch"+string(rune('0'+n)), func(t *testing.T) {
			h := newHarness(t, 10*time.Millisecond)
			var wg sync.WaitGroup
			errs := make(chan error, n)

			for i := 0; i < n; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
						Modality:  api.ModalityText,
						TextInput: &api.TextInput{Prompt: "concurrent"},
					})
					resp.Body.Close()
					if resp.StatusCode != http.StatusOK {
						errs <- nil
					}
				}()
			}
			wg.Wait()
			close(errs)
			for e := range errs {
				if e != nil {
					t.Error(e)
				}
			}
		})
	}
}

func TestVisionInfer_RunsPipelineAndReturnsOutput(t *testing.T) {
	h := newMultiModalHarness(t, 0)
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-image-bytes"))
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality: api.ModalityVision,
		VisionInput: &api.VisionInput{
			ImageBase64: imgData,
			Prompt:      "what is this?",
		},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var ir api.InferResponse
	json.NewDecoder(resp.Body).Decode(&ir)
	if ir.Output == "" {
		t.Error("expected non-empty vision output")
	}
	if ir.Modality != api.ModalityVision {
		t.Errorf("modality = %q; want vision", ir.Modality)
	}
}

func TestVisionInfer_ImageURLReturns400(t *testing.T) {
	h := newMultiModalHarness(t, 0)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality: api.ModalityVision,
		VisionInput: &api.VisionInput{
			ImageURL: "https://example.com/photo.jpg",
			Prompt:   "what is this?",
		},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeInvalidRequest || !strings.Contains(er.Message, "image_base64") {
		t.Errorf("er = %+v; want invalid_request mentioning image_base64", er)
	}
}

func TestVisionInfer_QualityDegradedHeaderWhenPipelineDegrades(t *testing.T) {
	h := newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   stub.New(backend.ModalityKindText, 0),
		backend.ModalityKindVision: &fakeVisionBackend{output: "raw description", degraded: true},
	})
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:    api.ModalityVision,
		VisionInput: &api.VisionInput{ImageBase64: imgData, Prompt: "describe"},
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get(api.HeaderQualityDegraded); got != "true" {
		t.Errorf("X-Quality-Degraded = %q; want true", got)
	}
}

func TestVisionMissingPromptReturns400(t *testing.T) {
	h := newMultiModalHarness(t, 0)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:    api.ModalityVision,
		VisionInput: &api.VisionInput{ImageBase64: "aGk="},
	})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestWrongMethodReturns405(t *testing.T) {
	h := newHarness(t, 0)
	for _, tc := range []struct {
		method, path string
	}{
		{http.MethodGet, api.PathInfer},
		{http.MethodDelete, api.PathStatus},
		{http.MethodPost, api.PathHealth},
		{http.MethodGet, api.PathTokenize},
	} {
		req, _ := http.NewRequest(tc.method, h.ts.URL+tc.path, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("%s %s: status = %d, want 405", tc.method, tc.path, resp.StatusCode)
		}
		if resp.Header.Get("Allow") == "" {
			t.Errorf("%s %s: missing Allow header", tc.method, tc.path)
		}
	}
}

func TestVisionMissingInputReturns400(t *testing.T) {
	h := newMultiModalHarness(t, 0)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality: api.ModalityVision,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeInvalidRequest {
		t.Errorf("want code %q, got %q", api.ErrCodeInvalidRequest, er.Code)
	}
}

func TestMixedModalityConcurrent(t *testing.T) {
	h := newMultiModalHarness(t, 5*time.Millisecond)
	imgData := base64.StdEncoding.EncodeToString([]byte("img"))

	var wg sync.WaitGroup
	textTiers := make(chan string, 2)
	visionCodes := make(chan int, 2)

	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
				Modality:  api.ModalityText,
				TextInput: &api.TextInput{Prompt: "hello"},
			})
			defer resp.Body.Close()
			var ir api.InferResponse
			json.NewDecoder(resp.Body).Decode(&ir)
			textTiers <- ir.ModelTier
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
				Modality: api.ModalityVision,
				VisionInput: &api.VisionInput{
					ImageBase64: imgData,
					Prompt:      "describe",
				},
			})
			defer resp.Body.Close()
			visionCodes <- resp.StatusCode
		}()
	}
	wg.Wait()
	close(textTiers)
	close(visionCodes)

	textOK := 0
	for tier := range textTiers {
		if strings.HasPrefix(tier, "weak") {
			textOK++
		}
	}
	if textOK != 2 {
		t.Errorf("expected 2 text results on the weak tier, got %d", textOK)
	}
	visOK := 0
	for code := range visionCodes {
		if code == http.StatusOK {
			visOK++
		}
	}
	if visOK != 2 {
		t.Errorf("expected 2 vision requests to return 200, got %d", visOK)
	}
}

// --- pressure endpoint ---

func TestHandlePressure_Returns200(t *testing.T) {
	h := newHarness(t, 0)
	resp, err := http.Get(h.ts.URL + api.PathStatusPressure)
	if err != nil {
		t.Fatalf("GET pressure: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	var snap api.PressureSnapshot
	if err := json.NewDecoder(resp.Body).Decode(&snap); err != nil {
		t.Fatalf("decode PressureSnapshot: %v", err)
	}
	if snap.Timestamp.IsZero() {
		t.Error("expected non-zero Timestamp in PressureSnapshot")
	}
}

// --- quality-degraded header ---

// mockQualityDegradedBackend always returns QualityDegraded=true.
type mockQualityDegradedBackend struct{}

func (m *mockQualityDegradedBackend) Modality() backend.ModalityKind   { return backend.ModalityKindText }
func (m *mockQualityDegradedBackend) Ready() bool                      { return true }
func (m *mockQualityDegradedBackend) Shutdown(_ context.Context) error { return nil }
func (m *mockQualityDegradedBackend) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	return backend.Response{
		Output:          "degraded output",
		TokensGenerated: 1,
		ModelTier:       "weak",
		QualityDegraded: true,
	}, nil
}

func TestInfer_QualityDegradedHeader(t *testing.T) {
	h := newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: &mockQualityDegradedBackend{},
	})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "test"},
		Priority:  "high",
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if got := resp.Header.Get(api.HeaderQualityDegraded); got != "true" {
		t.Errorf("expected X-Quality-Degraded: true, got %q", got)
	}
}

// --- structured output (response_format) ---

func TestInfer_ResponseFormatForwardedToBackend(t *testing.T) {
	rec := &recordingBackend{modality: backend.ModalityKindText}
	h := newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: rec,
	})

	rf := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}`)
	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:       api.ModalityText,
		TextInput:      &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "give me json"}}},
		ResponseFormat: rf,
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if string(rec.recordedReq.ResponseFormat) != string(rf) {
		t.Errorf("backend saw response_format %q; want %q", rec.recordedReq.ResponseFormat, rf)
	}
}

func TestInfer_InvalidGrammarMapsTo400(t *testing.T) {
	rec := &recordingBackend{
		modality: backend.ModalityKindText,
		inferErr: &backend.BackendError{Code: api.ErrCodeInvalidGrammar, Message: "bad schema"},
	}
	h := newHarnessWithBackends(t, map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: rec,
	})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:       api.ModalityText,
		TextInput:      &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "x"}}},
		ResponseFormat: json.RawMessage(`{"type":"json_schema"}`),
	})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeInvalidGrammar {
		t.Errorf("code = %q; want invalid_grammar", er.Code)
	}
}

func TestHandleLogsDisabled(t *testing.T) {
	h := newHarness(t, 0)
	resp, err := http.Get(h.ts.URL + api.PathLogs)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("want 501 when no log source, got %d", resp.StatusCode)
	}
}

func TestHandleLogsReturnsRecords(t *testing.T) {
	h := newHarness(t, 0)
	buf := logbuf.NewBuffer(100, time.Hour)
	lg := slog.New(buf.Handler(slog.NewJSONHandler(io.Discard, nil)))
	lg.Info("info line")
	lg.Error("boom", slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)))
	h.srv.SetLogSource(buf)

	resp, err := http.Get(h.ts.URL + api.PathLogs + "?level=ERROR")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var body struct {
		Logs []logbuf.Record `json:"logs"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Logs) != 1 {
		t.Fatalf("want 1 ERROR record, got %d", len(body.Logs))
	}
	if body.Logs[0].Event != string(logschema.EventJobFailed) {
		t.Fatalf("want job.failed event, got %q", body.Logs[0].Event)
	}
}

func TestHandleLogsSinceClamped(t *testing.T) {
	h := newHarness(t, 0)
	h.srv.SetLogSource(logbuf.NewBuffer(10, time.Hour))
	resp, err := http.Get(h.ts.URL + api.PathLogs + "?since=999h")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200 (since clamped, not rejected), got %d", resp.StatusCode)
	}
}
