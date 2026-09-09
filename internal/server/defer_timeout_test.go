package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/batcher"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/types"
)

// fakeRelay records the last Enqueue call and can be told to fail.
type fakeRelay struct {
	fail       bool
	calls      int32
	lastCorrID string
	lastBody   string
}

func (f *fakeRelay) Enqueue(_ context.Context, corrID string, payload json.RawMessage) (string, error) {
	atomic.AddInt32(&f.calls, 1)
	f.lastCorrID = corrID
	f.lastBody = string(payload)
	if f.fail {
		return "", context.DeadlineExceeded
	}
	return "relay-job-1", nil
}
func (f *fakeRelay) ResultURL(id string) string { return "https://relay.example/result/" + id }

type staticVRAM struct{ mb int64 }

func (s staticVRAM) AvailableMB() (int64, error) { return s.mb, nil }

type staticCPU struct{ pct float64 }

func (s staticCPU) Pct() float64 { return s.pct }

func bytesReader(b []byte) io.Reader { return bytes.NewReader(b) }

func newDeferHarness(t *testing.T, relay server.RelayEnqueuer, vramMB int64, cpuPct float64) *httptest.Server {
	t.Helper()
	q := queue.New(16)
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: stub.New(backend.ModalityKindText, 0),
	})
	ctx, cancel := context.WithCancel(context.Background())
	b := batcher.New(batcher.Config{MaxBatchSize: 4, WindowDuration: 5 * time.Millisecond}, q.Drain(), func(batch []queue.Job) {
		go r.Dispatch(ctx, batch)
	})
	go b.Run(ctx)

	srv := server.New(q, func() types.ServerStatus {
		return types.ServerStatus{AvailableVRAMMB: vramMB}
	}, "test", time.Now())
	srv.SetPressureSources(staticCPU{cpuPct}, nil)
	if relay != nil {
		srv.SetRelayEnqueuer(relay, staticVRAM{vramMB})
	}

	ts := httptest.NewServer(srv)
	t.Cleanup(func() { cancel(); ts.Close() })
	return ts
}

func TestDeferPushesToRelayReturns202WithResultURL(t *testing.T) {
	relay := &fakeRelay{}
	ts := newDeferHarness(t, relay, 100, 95) // under pressure

	body, _ := json.Marshal(api.InferRequest{
		CorrelationID: "corr-x",
		Modality:      api.ModalityText,
		Priority:      "normal",
		TextInput:     &api.TextInput{Prompt: "hi"},
	})
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("want 202, got %d", resp.StatusCode)
	}
	var got map[string]any
	json.NewDecoder(resp.Body).Decode(&got)
	if got["deferred"] != true {
		t.Errorf("deferred = %v", got["deferred"])
	}
	if got["request_id"] != "relay-job-1" {
		t.Errorf("request_id = %v", got["request_id"])
	}
	if got["result_url"] != "https://relay.example/result/relay-job-1" {
		t.Errorf("result_url = %v", got["result_url"])
	}
	if relay.calls != 1 {
		t.Errorf("relay Enqueue calls = %d", relay.calls)
	}
	if relay.lastCorrID != "corr-x" {
		t.Errorf("relay got corrID %q", relay.lastCorrID)
	}
}

func TestDeferRelayUnreachableReturns503(t *testing.T) {
	relay := &fakeRelay{fail: true}
	ts := newDeferHarness(t, relay, 100, 95)

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		Priority:  "normal",
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("want 503, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeOverloaded {
		t.Errorf("code = %q, want overloaded", er.Code)
	}
}

func TestNoRelayConfiguredNeverDefers(t *testing.T) {
	ts := newDeferHarness(t, nil, 100, 95) // pressure, but no relay

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		Priority:  "normal",
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200 (served, not deferred), got %d", resp.StatusCode)
	}
}

func TestHighPriorityBypassesDeferral(t *testing.T) {
	relay := &fakeRelay{}
	ts := newDeferHarness(t, relay, 100, 95)

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		Priority:  "high",
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200 for high priority, got %d", resp.StatusCode)
	}
	if relay.calls != 0 {
		t.Errorf("high-priority request should not hit the relay; calls = %d", relay.calls)
	}
}

func TestStreamDefersToRelayReturns202(t *testing.T) {
	relay := &fakeRelay{}
	ts := newDeferHarness(t, relay, 100, 95) // under pressure

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		Priority:  "normal",
		Stream:    true,
		TextInput: &api.TextInput{Prompt: "hi"},
	})
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("want 202, got %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct == "text/event-stream" {
		t.Fatalf("deferred stream should not be an SSE response, got Content-Type %q", ct)
	}
	var got map[string]any
	json.NewDecoder(resp.Body).Decode(&got)
	if got["deferred"] != true || got["result_url"] == nil {
		t.Fatalf("body = %v", got)
	}
	if relay.calls != 1 {
		t.Fatalf("relay calls = %d", relay.calls)
	}
}

func TestDefaultInferTimeoutApplied(t *testing.T) {
	q := queue.New(16)
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: stub.New(backend.ModalityKindText, 2*time.Second),
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b := batcher.New(batcher.Config{MaxBatchSize: 4, WindowDuration: 5 * time.Millisecond}, q.Drain(), func(batch []queue.Job) {
		go r.Dispatch(ctx, batch)
	})
	go b.Run(ctx)

	srv := server.New(q, func() types.ServerStatus { return types.ServerStatus{AvailableVRAMMB: -1} }, "test", time.Now())
	srv.SetInferTimeout(80 * time.Millisecond)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "slow"},
	})
	start := time.Now()
	resp, err := http.Post(ts.URL+api.PathInfer, "application/json", bytesReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if time.Since(start) > time.Second {
		t.Fatalf("request did not honour the 80ms server default timeout (took %s)", time.Since(start))
	}
	// The handler's own ctx deadline fires first → 408 timeout, recorded as
	// a cancelled job.
	if resp.StatusCode != http.StatusRequestTimeout {
		t.Fatalf("status = %d, want 408", resp.StatusCode)
	}

	sresp, err := http.Get(ts.URL + api.PathStatus)
	if err != nil {
		t.Fatal(err)
	}
	defer sresp.Body.Close()
	var st types.ServerStatus
	json.NewDecoder(sresp.Body).Decode(&st)
	if len(st.RecentJobs) == 0 || st.RecentJobs[0].Status != types.JobStatusCancelled {
		t.Fatalf("expected a recorded 'cancelled' job, got %+v", st.RecentJobs)
	}
}
