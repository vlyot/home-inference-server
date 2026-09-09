package testapp_test

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/hisclient"
	"github.com/ngkaichong/home-inference-server/internal/testapp"
)

// TestConformanceAgainstStub always runs: it starts the real server pipeline
// wired to stub inference backends (no GPU) and asserts every documented
// endpoint behaves.
func TestConformanceAgainstStub(t *testing.T) {
	addr := startStubServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	c := hisclient.New(addr)
	res := testapp.Run(ctx, c, testapp.Caps{StubBackend: true})

	res.Print(testWriter{t})
	for _, f := range res.Failed {
		t.Errorf("%s: %s", f.Name, f.Err)
	}
}

// TestConformanceConcurrentInferAgainstStub fires several /v1/infer requests at
// once and asserts they all succeed within the deadline, that /v1/status/pressure
// reports a numeric max_parallel, and that in_flight drains back to zero.
func TestConformanceConcurrentInferAgainstStub(t *testing.T) {
	addr := startStubServer(t)
	c := hisclient.New(addr)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	pre, err := c.Pressure(ctx)
	if err != nil {
		t.Fatalf("pressure: %v", err)
	}
	if pre.MaxParallel < 1 {
		t.Fatalf("pressure.max_parallel = %d; want >= 1", pre.MaxParallel)
	}

	const n = 8
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = c.Infer(ctx, api.InferRequest{
				Modality:  api.ModalityText,
				TextInput: &api.TextInput{Prompt: "hi"},
				Priority:  "normal",
				MaxTokens: 8,
			})
		}(i)
	}
	wg.Wait()
	for i, e := range errs {
		if e != nil {
			t.Errorf("concurrent infer %d: %v", i, e)
		}
	}

	post, err := c.Pressure(ctx)
	if err != nil {
		t.Fatalf("pressure after: %v", err)
	}
	if post.InFlight != 0 {
		t.Errorf("pressure.in_flight = %d after all requests completed; want 0", post.InFlight)
	}
}

// TestConformanceAdminDrainCycleAgainstStub exercises the loopback admin control
// surface end to end: drain -> /v1/infer 503 draining, /v1/status state:draining
// -> resume -> /v1/infer served again.
func TestConformanceAdminDrainCycleAgainstStub(t *testing.T) {
	addr := startStubServer(t)
	c := hisclient.New(addr)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	infer := func() (*hisclient.InferResult, error) {
		return c.Infer(ctx, api.InferRequest{
			Modality:  api.ModalityText,
			TextInput: &api.TextInput{Prompt: "hi"},
			MaxTokens: 8,
		})
	}

	if _, err := infer(); err != nil {
		t.Fatalf("infer before drain: %v", err)
	}

	code, _, err := c.RawPost(ctx, api.PathAdminDrain, "")
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if code != 200 {
		t.Fatalf("drain: status = %d, want 200", code)
	}

	if _, err := infer(); err == nil {
		t.Fatal("infer while drained: want an error, got nil")
	} else if apiErr, ok := hisclient.AsAPIError(err); !ok || apiErr.Code != api.ErrCodeDraining {
		t.Fatalf("infer while drained: err = %v, want %q", err, api.ErrCodeDraining)
	}

	st, err := c.Status(ctx)
	if err != nil {
		t.Fatalf("status while drained: %v", err)
	}
	if st["state"] != api.StateDraining {
		t.Fatalf("status.state = %v, want %q", st["state"], api.StateDraining)
	}

	code, _, err = c.RawPost(ctx, api.PathAdminResume, "")
	if err != nil || code != 200 {
		t.Fatalf("resume: status = %d err = %v", code, err)
	}

	if _, err := infer(); err != nil {
		t.Fatalf("infer after resume: %v", err)
	}
}

// TestConformanceAgainstLiveServer runs only when HIS_TESTAPP_ADDR points at a
// running server (real backend). HIS_API_KEY, if set, is sent and the auth
// checks run.
func TestConformanceAgainstLiveServer(t *testing.T) {
	addr := os.Getenv("HIS_TESTAPP_ADDR")
	if addr == "" {
		t.Skip("set HIS_TESTAPP_ADDR to run the conformance suite against a live server")
	}
	key := os.Getenv("HIS_API_KEY")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	opts := []hisclient.Option{}
	if key != "" {
		opts = append(opts, hisclient.WithAPIKey(key))
	}
	c := hisclient.New(addr, opts...)
	res := testapp.Run(ctx, c, testapp.Caps{StubBackend: os.Getenv("HIS_TESTAPP_STUB") == "1", APIKey: key})

	res.Print(testWriter{t})
	for _, f := range res.Failed {
		t.Errorf("%s: %s", f.Name, f.Err)
	}
}

// testWriter adapts *testing.T to io.Writer for Result.Print.
type testWriter struct{ t *testing.T }

func (w testWriter) Write(p []byte) (int, error) {
	w.t.Log("\n" + string(p))
	return len(p), nil
}
