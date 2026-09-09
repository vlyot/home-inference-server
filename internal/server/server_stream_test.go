package server_test

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/types"
)

// streamStub is a backend.Streamer whose streaming output is fully scripted:
// it emits each delta in deltas (as ChunkContent) then each entry in reasoning
// (as ChunkReasoning), then reports tokens as TokensGenerated.
type streamStub struct {
	deltas    []string
	reasoning []string
	tokens    int
}

func (s *streamStub) Modality() backend.ModalityKind   { return backend.ModalityKindText }
func (s *streamStub) Ready() bool                      { return true }
func (s *streamStub) Shutdown(_ context.Context) error { return nil }
func (s *streamStub) Infer(_ context.Context, _ backend.Request) (backend.Response, error) {
	return backend.Response{
		Output:          strings.Join(s.deltas, ""),
		Reasoning:       strings.Join(s.reasoning, ""),
		TokensGenerated: s.tokens,
		ModelTier:       "weak",
	}, nil
}
func (s *streamStub) InferStream(_ context.Context, _ backend.Request, chunkFn func(backend.ChunkKind, string)) (backend.Response, error) {
	for _, r := range s.reasoning {
		chunkFn(backend.ChunkReasoning, r)
	}
	for _, d := range s.deltas {
		chunkFn(backend.ChunkContent, d)
	}
	return backend.Response{TokensGenerated: s.tokens, ModelTier: "weak"}, nil
}

func streamHarness(t *testing.T, stub *streamStub) *harness {
	t.Helper()
	backends := map[backend.ModalityKind]backend.Backend{
		backend.ModalityKindText: stub,
	}
	h := newHarnessWithBackends(t, backends)
	h.srv.SetBackends(backends) // enable the SSE dispatch path
	return h
}

// readSSEChunks reads a text/event-stream body into decoded StreamChunks.
// The terminal "data: [DONE]" sentinel line is recognised and skipped.
func readSSEChunks(t *testing.T, r *http.Response) []api.StreamChunk {
	t.Helper()
	var chunks []api.StreamChunk
	sawDone := false
	sc := bufio.NewScanner(r.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			continue
		}
		var c api.StreamChunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad SSE chunk %q: %v", line, err)
		}
		chunks = append(chunks, c)
	}
	if !sawDone {
		t.Errorf("stream did not end with a 'data: [DONE]' sentinel line")
	}
	return chunks
}

func TestStreamZeroOutputProducesErrorChunk(t *testing.T) {
	h := streamHarness(t, &streamStub{deltas: nil, tokens: 0})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
	})
	defer resp.Body.Close()

	chunks := readSSEChunks(t, resp)
	if len(chunks) != 1 {
		t.Fatalf("want exactly 1 chunk, got %d: %+v", len(chunks), chunks)
	}
	last := chunks[0]
	if !last.Done {
		t.Error("terminal chunk not marked done")
	}
	if last.Error != api.ErrCodeInternal {
		t.Errorf("error = %q; want %q", last.Error, api.ErrCodeInternal)
	}
}

func TestStreamNormalOutputStillEndsWithSuccessChunk(t *testing.T) {
	h := streamHarness(t, &streamStub{deltas: []string{"Hel", "lo"}, tokens: 2})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
	})
	defer resp.Body.Close()

	chunks := readSSEChunks(t, resp)
	if len(chunks) < 2 {
		t.Fatalf("want deltas + done chunk, got %d", len(chunks))
	}
	done := chunks[len(chunks)-1]
	if !done.Done || done.Error != "" {
		t.Errorf("terminal chunk = %+v; want done with no error", done)
	}
	if done.TokensGenerated != 2 {
		t.Errorf("tokens_generated = %d; want 2", done.TokensGenerated)
	}
}

func TestStreamZeroOutputRecordsFailedJob(t *testing.T) {
	stub := &streamStub{deltas: nil, tokens: 0}
	h := streamHarness(t, stub)

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
	})
	resp.Body.Close()

	deadline := time.Now().Add(time.Second)
	var jobs []types.JobEntry
	for time.Now().Before(deadline) {
		jobs = h.srv.RecentJobs()
		if len(jobs) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(jobs) == 0 {
		t.Fatal("no job recorded for zero-output stream")
	}
	if jobs[0].Status != types.JobStatusFailed {
		t.Errorf("job status = %q; want %q", jobs[0].Status, types.JobStatusFailed)
	}
}

// TestStreamReasoningOnlyProducesReasoningExhaustedError is the regression
// test for the bug where a reasoning-capable model (e.g. Gemma's "thinking"
// mode) could spend its entire token budget reasoning and never answer:
// tokens_generated was non-zero so the old !deltaEmitted && TokensGenerated==0
// check never fired, silently reporting success with an empty answer.
func TestStreamReasoningOnlyProducesReasoningExhaustedError(t *testing.T) {
	h := streamHarness(t, &streamStub{reasoning: []string{"thinking...", "still thinking..."}, tokens: 512})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
	})
	defer resp.Body.Close()

	chunks := readSSEChunks(t, resp)
	if len(chunks) == 0 {
		t.Fatal("want at least the reasoning chunks + a terminal chunk")
	}
	for _, c := range chunks[:len(chunks)-1] {
		if c.Reasoning == "" || c.Delta != "" {
			t.Errorf("non-terminal chunk = %+v; want reasoning-only", c)
		}
	}
	last := chunks[len(chunks)-1]
	if !last.Done {
		t.Error("terminal chunk not marked done")
	}
	if last.Error != api.ErrCodeReasoningExhausted {
		t.Errorf("error = %q; want %q", last.Error, api.ErrCodeReasoningExhausted)
	}
}

// TestStreamReasoningThenContentSucceeds covers the normal reasoning-model
// path: reasoning chunks stream first, then real content, and the terminal
// chunk reports success — no error, despite reasoning chunks having been
// emitted along the way.
func TestStreamReasoningThenContentSucceeds(t *testing.T) {
	h := streamHarness(t, &streamStub{
		reasoning: []string{"let me think..."},
		deltas:    []string{"The answer", " is 4."},
		tokens:    10,
	})

	resp := post(t, h.ts.URL+api.PathInfer, api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
	})
	defer resp.Body.Close()

	chunks := readSSEChunks(t, resp)
	if len(chunks) != 4 { // 1 reasoning + 2 content + 1 terminal
		t.Fatalf("want 4 chunks, got %d: %+v", len(chunks), chunks)
	}
	if chunks[0].Reasoning != "let me think..." {
		t.Errorf("chunk 0 = %+v; want the reasoning text", chunks[0])
	}
	done := chunks[len(chunks)-1]
	if !done.Done || done.Error != "" {
		t.Errorf("terminal chunk = %+v; want done with no error", done)
	}
}
