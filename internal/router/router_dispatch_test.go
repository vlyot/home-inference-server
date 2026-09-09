package router_test

import (
	"context"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	istub "github.com/ngkaichong/home-inference-server/internal/backend/stub"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/router"
)

// panicBackend panics inside Infer to prove the router recovers per job.
type panicBackend struct{}

func (panicBackend) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (panicBackend) Infer(context.Context, backend.Request) (backend.Response, error) {
	panic("boom")
}
func (panicBackend) Ready() bool                    { return true }
func (panicBackend) Shutdown(context.Context) error { return nil }

func TestDispatchOneRoutesByModality(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: istub.New(backend.ModalityKindText, 0),
	})
	job := queue.Job{
		ID:       "x",
		Req:      backend.Request{RequestID: "x", Modality: backend.ModalityKindText},
		ResultCh: make(chan queue.Result, 1),
	}
	r.DispatchOne(context.Background(), job, 1)
	select {
	case res := <-job.ResultCh:
		if res.Err != nil {
			t.Fatalf("unexpected error: %v", res.Err)
		}
	case <-time.After(time.Second):
		t.Fatal("DispatchOne produced no result")
	}
}

func TestDispatchNilResultChIsNoOp(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: istub.New(backend.ModalityKindText, 0),
	})
	job := queue.Job{ID: "no-waiter", Req: backend.Request{RequestID: "no-waiter", Modality: backend.ModalityKindText}}

	done := make(chan struct{})
	go func() {
		r.Dispatch(context.Background(), []queue.Job{job})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Dispatch blocked on a nil ResultCh")
	}
}

func TestDispatchPanicInBackendDoesNotWedge(t *testing.T) {
	r := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: panicBackend{},
	})
	bad := queue.Job{ID: "p", Req: backend.Request{RequestID: "p", Modality: backend.ModalityKindText}, ResultCh: make(chan queue.Result, 1)}
	good := queue.Job{ID: "g", Req: backend.Request{RequestID: "g", Modality: backend.ModalityKindText}, ResultCh: make(chan queue.Result, 1)}
	// A healthy backend for the second job.
	r2 := router.New(map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: istub.New(backend.ModalityKindText, 0),
	})

	r.Dispatch(context.Background(), []queue.Job{bad})
	res := <-bad.ResultCh
	if res.Err == nil {
		t.Fatal("expected an error result from a panicking backend")
	}

	r2.Dispatch(context.Background(), []queue.Job{good})
	if res := <-good.ResultCh; res.Err != nil {
		t.Fatalf("subsequent job failed after a panic: %v", res.Err)
	}
}
