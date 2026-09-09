package stub_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/stub"
)

func TestOutputContainsRequestID(t *testing.T) {
	b := stub.New(backend.ModalityKindText, 0)
	req := backend.Request{RequestID: "myreq123", Modality: backend.ModalityKindText}

	resp, err := b.Infer(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(resp.Output, "myreq123") {
		t.Errorf("output %q does not contain request ID", resp.Output)
	}
}

func TestCancellationHonoured(t *testing.T) {
	b := stub.New(backend.ModalityKindText, 5*time.Second)
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		_, err := b.Infer(ctx, backend.Request{RequestID: "x", Modality: backend.ModalityKindText})
		done <- err
	}()

	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error from cancelled context, got nil")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Infer did not return after context cancellation")
	}
}
