package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/websearch"
)

// scriptedToolBackend answers the Nth Infer call with responses[N] (clamped
// to the last entry), simulating a model that first calls a tool then answers
// once the tool result is appended to the conversation.
type scriptedToolBackend struct {
	responses []backend.Response
	calls     int
}

func (b *scriptedToolBackend) Modality() backend.ModalityKind   { return backend.ModalityKindText }
func (b *scriptedToolBackend) Ready() bool                      { return true }
func (b *scriptedToolBackend) Shutdown(_ context.Context) error { return nil }
func (b *scriptedToolBackend) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	resp := b.scriptedResponse(req)
	return resp, nil
}

// InferStream implements backend.Streamer so scriptedToolBackend can also
// exercise the streaming-with-tools path. Emits the scripted response's
// Reasoning (as ChunkReasoning) then Output (as ChunkContent) via chunkFn,
// matching the real verified sequence (reasoning streams, then either
// content or — on the first hop — nothing, since Output is empty when
// ToolCalls is set).
func (b *scriptedToolBackend) InferStream(_ context.Context, req backend.Request, chunkFn func(backend.ChunkKind, string)) (backend.Response, error) {
	resp := b.scriptedResponse(req)
	if resp.Reasoning != "" {
		chunkFn(backend.ChunkReasoning, resp.Reasoning)
	}
	if resp.Output != "" {
		chunkFn(backend.ChunkContent, resp.Output)
	}
	return resp, nil
}

// scriptedResponse returns responses[calls] (clamped to the last entry) and
// increments calls. Reflects the injected search result content into the
// final answer so a test can assert the model actually "saw" grounded
// content, matching how a real chat-completions call would incorporate the
// role:"tool" message.
func (b *scriptedToolBackend) scriptedResponse(req backend.Request) backend.Response {
	idx := b.calls
	if idx >= len(b.responses) {
		idx = len(b.responses) - 1
	}
	b.calls++
	resp := b.responses[idx]
	if idx == len(b.responses)-1 && len(req.Messages) > 0 {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == "tool" {
			resp.Output = resp.Output + " [saw: " + last.Content + "]"
		}
	}
	return resp
}

// fakeToolSearcher is a scripted websearch-shaped ToolSearcher for the e2e test.
type fakeToolSearcher struct {
	results []websearch.Result
	err     error
}

func (f *fakeToolSearcher) Search(_ context.Context, _ string) ([]websearch.Result, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

func TestServer_InferWithToolsEndToEnd(t *testing.T) {
	b := &scriptedToolBackend{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"latest news"}`}}},
		{Output: "Here is a grounded answer."},
	}}
	h := streamHarnessWithBackend(t, b)
	h.srv.SetToolSearcher(&fakeToolSearcher{results: []websearch.Result{{Title: "Headline", URL: "https://x", Content: "breaking news"}}})

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "what's the latest news"}}},
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "web_search", Description: "search the web"},
		}},
	})

	resp, err := http.Post(h.ts.URL+api.PathInfer, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/infer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var out api.InferResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Output == "" || out.Output == "Here is a grounded answer." {
		t.Errorf("Output = %q, want it to reflect the injected search result", out.Output)
	}
	if b.calls != 2 {
		t.Errorf("backend called %d times, want 2", b.calls)
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].Name != "web_search" {
		t.Errorf("ToolCalls = %+v, want one web_search entry", out.ToolCalls)
	}
	if out.TokensGenerated < 0 { // exercised, not a meaningful bound — the stub doesn't set it
		t.Errorf("TokensGenerated = %d", out.TokensGenerated)
	}
	if out.DurationMS < 0 || out.InferenceMS < 0 || out.QueueWaitMS < 0 {
		t.Errorf("timing fields should be non-negative: duration=%d inference=%d queue_wait=%d", out.DurationMS, out.InferenceMS, out.QueueWaitMS)
	}
}

func TestHandleInfer_ToolsWithVisionRejected(t *testing.T) {
	h := newMultiModalHarness(t, 0)

	body, _ := json.Marshal(api.InferRequest{
		Modality:    api.ModalityVision,
		VisionInput: &api.VisionInput{Prompt: "what is this", ImageBase64: "aGVsbG8="},
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "web_search"},
		}},
	})

	resp, err := http.Post(h.ts.URL+api.PathInfer, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/infer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}

	var out api.ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if out.Code != api.ErrCodeToolNotSupported {
		t.Errorf("Code = %q, want %q", out.Code, api.ErrCodeToolNotSupported)
	}
}

func TestHandleInfer_ToolsWithoutMessagesRejected(t *testing.T) {
	h := newHarness(t, 0)

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Prompt: "hello"},
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "web_search"},
		}},
	})

	resp, err := http.Post(h.ts.URL+api.PathInfer, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/infer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

// TestHandleInfer_ToolsWithStreamNoLongerRejected replaces the old
// TestHandleInfer_ToolsWithStreamRejected (which asserted 501) now that
// streaming + tools is a real, supported combination.
func TestHandleInfer_ToolsWithStreamNoLongerRejected(t *testing.T) {
	b := &scriptedToolBackend{responses: []backend.Response{{Output: "a plain streamed answer"}}}
	h := streamHarnessWithBackend(t, b)

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "hi"}}},
		Stream:    true,
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "web_search"},
		}},
	})

	resp, err := http.Post(h.ts.URL+api.PathInfer, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/infer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
}

// TestServer_InferStreamWithToolsEndToEnd exercises the full HTTP round-trip
// for stream:true + tools:[web_search]: the model calls the tool on hop one
// (streaming its reasoning live), the server resolves the search, then hop
// two's grounded answer streams live too — plus one ToolCall notification
// chunk emitted between the two hops.
func TestServer_InferStreamWithToolsEndToEnd(t *testing.T) {
	b := &scriptedToolBackend{responses: []backend.Response{
		{Reasoning: "I should search for this.", ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"latest news"}`}}},
		{Output: "Here is a grounded answer."},
	}}
	h := streamHarnessWithBackend(t, b)
	h.srv.SetToolSearcher(&fakeToolSearcher{results: []websearch.Result{{Title: "Headline", URL: "https://x", Content: "breaking news"}}})

	body, _ := json.Marshal(api.InferRequest{
		Modality:  api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: "what's the latest news"}}},
		Stream:    true,
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "web_search", Description: "search the web"},
		}},
	})

	resp, err := http.Post(h.ts.URL+api.PathInfer, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /v1/infer: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	chunks := readSSEChunks(t, resp)
	if len(chunks) == 0 {
		t.Fatal("no chunks received")
	}

	var sawReasoning, sawToolCallNotification, sawGroundedContent bool
	var toolCallName string
	for _, c := range chunks {
		if c.Reasoning != "" {
			sawReasoning = true
		}
		if c.ToolCall != nil {
			sawToolCallNotification = true
			toolCallName = c.ToolCall.Name
		}
		if strings.Contains(c.Delta, "grounded answer") {
			sawGroundedContent = true
		}
	}
	if !sawReasoning {
		t.Error("expected the first hop's reasoning to have streamed as a Reasoning chunk")
	}
	if !sawToolCallNotification || toolCallName != "web_search" {
		t.Errorf("expected a ToolCall notification chunk for web_search, got name=%q seen=%v", toolCallName, sawToolCallNotification)
	}
	if !sawGroundedContent {
		t.Error("expected the second hop's grounded answer to have streamed as Delta chunks")
	}

	final := chunks[len(chunks)-1]
	if !final.Done {
		t.Error("final chunk should have Done=true")
	}
	if final.Error != "" {
		t.Errorf("final chunk Error = %q, want empty", final.Error)
	}
	if b.calls != 2 {
		t.Errorf("backend called %d times, want 2", b.calls)
	}
}
