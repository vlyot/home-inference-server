package router_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	istub "github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
)

// visionErrBackend is a vision-modality backend that always returns a fixed
// BackendError — used to prove the router dispatches vision jobs to whatever is
// registered under ModalityKindVision and surfaces its error.
type visionErrBackend struct{ code string }

func (v visionErrBackend) Modality() backend.ModalityKind { return backend.ModalityKindVision }
func (v visionErrBackend) Ready() bool                    { return true }
func (v visionErrBackend) Shutdown(context.Context) error { return nil }
func (v visionErrBackend) Infer(context.Context, backend.Request) (backend.Response, error) {
	return backend.Response{}, &backend.BackendError{Code: v.code, Message: "vision backend stub"}
}

func makeJob(id string, modality backend.ModalityKind) queue.Job {
	return queue.Job{
		ID: id,
		Req: backend.Request{
			RequestID: id,
			Modality:  modality,
		},
		ResultCh: make(chan queue.Result, 1),
	}
}

func TestRoutesTextJobToStub(t *testing.T) {
	b := istub.New(backend.ModalityKindText, 0)
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: b,
	})

	job := makeJob("req1", backend.ModalityKindText)
	r.Dispatch(context.Background(), []queue.Job{job})

	result := <-job.ResultCh
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if result.Output == "" {
		t.Fatal("expected non-empty output")
	}
}

func TestNoBackendForModality(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{})

	job := makeJob("req2", backend.ModalityKindText)
	r.Dispatch(context.Background(), []queue.Job{job})

	result := <-job.ResultCh
	if result.Err == nil {
		t.Fatal("expected error for unregistered modality")
	}
	var be *backend.BackendError
	if !errors.As(result.Err, &be) {
		t.Fatalf("expected BackendError, got %T", result.Err)
	}
}

func TestResultWrittenToResultCh(t *testing.T) {
	b := istub.New(backend.ModalityKindText, 5*time.Millisecond)
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: b,
	})

	job := makeJob("req3", backend.ModalityKindText)
	r.Dispatch(context.Background(), []queue.Job{job})

	select {
	case result := <-job.ResultCh:
		if result.ModelTier != "weak" {
			t.Errorf("expected tier 'weak', got %q", result.ModelTier)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for result")
	}
}

func TestVisionJobRoutesToVisionBackend(t *testing.T) {
	vb := visionErrBackend{code: "not_implemented"}
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   istub.New(backend.ModalityKindText, 0),
		backend.ModalityKindVision: vb,
	})

	job := makeJob("vis1", backend.ModalityKindVision)
	r.Dispatch(context.Background(), []queue.Job{job})

	result := <-job.ResultCh
	// Vision is not implemented in this build — the request routes to the vision
	// backend, which returns a not_implemented BackendError.
	var be *backend.BackendError
	if !errors.As(result.Err, &be) || be.Code != "not_implemented" {
		t.Fatalf("expected not_implemented BackendError, got %v", result.Err)
	}
}

func TestRoutesMixedBatchToCorrectBackends(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText:   istub.New(backend.ModalityKindText, 0),
		backend.ModalityKindVision: visionErrBackend{code: "not_implemented"},
	})

	textJob := makeJob("txt1", backend.ModalityKindText)
	visJob := makeJob("vis2", backend.ModalityKindVision)

	r.Dispatch(context.Background(), []queue.Job{textJob, visJob})

	textResult := <-textJob.ResultCh
	if textResult.Err != nil {
		t.Fatalf("text job error: %v", textResult.Err)
	}
	if textResult.ModelTier != "weak" {
		t.Errorf("text job: expected tier 'weak', got %q", textResult.ModelTier)
	}

	visResult := <-visJob.ResultCh
	var be *backend.BackendError
	if !errors.As(visResult.Err, &be) || be.Code != "not_implemented" {
		t.Fatalf("vision job: expected not_implemented BackendError, got %v", visResult.Err)
	}
}
