package stub

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
)

// StubBackend simulates inference with a configurable sleep latency.
// It satisfies backend.Backend, backend.Streamer and backend.Measurable so the
// full server pipeline (including the SSE path) can be exercised without a GPU —
// used by cmd/server --backend=stub and the conformance suite.
type StubBackend struct {
	modality backend.ModalityKind
	latency  time.Duration

	inFlight atomic.Int32 // requests currently sleeping in Infer/InferStream
	peak     atomic.Int32 // high-water mark of inFlight, for concurrency tests
}

// New returns a StubBackend that sleeps latency per request.
func New(modality backend.ModalityKind, latency time.Duration) *StubBackend {
	return &StubBackend{modality: modality, latency: latency}
}

// enter/leave track concurrent occupancy so a test can assert overlap.
func (s *StubBackend) enter() int32 {
	n := s.inFlight.Add(1)
	for {
		p := s.peak.Load()
		if n <= p || s.peak.CompareAndSwap(p, n) {
			return n
		}
	}
}

func (s *StubBackend) leave() { s.inFlight.Add(-1) }

// InFlight is the number of requests currently in Infer/InferStream.
func (s *StubBackend) InFlight() int { return int(s.inFlight.Load()) }

// PeakInFlight is the highest concurrent occupancy seen since construction.
func (s *StubBackend) PeakInFlight() int { return int(s.peak.Load()) }

// MaxParallel is a fixed advertised ceiling for the stub.
func (s *StubBackend) MaxParallel() int { return 2 }

func (s *StubBackend) Modality() backend.ModalityKind {
	return s.modality
}

func (s *StubBackend) Infer(ctx context.Context, req backend.Request) (backend.Response, error) {
	s.enter()
	defer s.leave()
	select {
	case <-time.After(s.latency):
	case <-ctx.Done():
		return backend.Response{}, &backend.BackendError{
			Code:    "unavailable",
			Message: "request cancelled",
			Cause:   ctx.Err(),
		}
	}
	return backend.Response{
		Output:          stubOutput(req),
		TokensGenerated: 4,
		ModelTier:       "weak",
		TokPerSecSample: 12,
	}, nil
}

// InferStream emits the stub output as a few content deltas so streaming
// clients have something to parse. Implements backend.Streamer.
func (s *StubBackend) InferStream(ctx context.Context, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	s.enter()
	defer s.leave()
	select {
	case <-time.After(s.latency):
	case <-ctx.Done():
		return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "request cancelled", Cause: ctx.Err()}
	}
	for _, part := range chunkWords(stubOutput(req)) {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "request cancelled", Cause: ctx.Err()}
		}
		chunkFn(backend.ChunkContent, part)
	}
	return backend.Response{
		TokensGenerated: 4,
		ModelTier:       "weak",
		TokPerSecSample: 12,
	}, nil
}

// TokPerSec implements backend.Measurable.
func (s *StubBackend) TokPerSec() float64 { return 12 }

func (s *StubBackend) Ready() bool { return true }

func (s *StubBackend) Shutdown(_ context.Context) error { return nil }

func stubOutput(req backend.Request) string {
	if len(req.Messages) > 0 {
		return fmt.Sprintf("stub reply [%s] to %d messages", req.RequestID, len(req.Messages))
	}
	return fmt.Sprintf("stub output [%s]", req.RequestID)
}

func chunkWords(s string) []string {
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return []string{s}
	}
	out := make([]string, len(fields))
	for i, f := range fields {
		if i == 0 {
			out[i] = f
		} else {
			out[i] = " " + f
		}
	}
	return out
}
