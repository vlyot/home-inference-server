package llamacpp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/backend"
)

func TestChatComplete_ForwardsTools(t *testing.T) {
	var received chatReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&received)
		resp := chatResp{}
		resp.Choices = make([]struct {
			Message struct {
				Content          string         `json:"content"`
				ReasoningContent string         `json:"reasoning_content"`
				ToolCalls        []wireToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}, 1)
		resp.Choices[0].Message.Content = "ok"
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	req := backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "search this"}},
		Tools: []backend.Tool{{
			Type: "function",
			Function: backend.ToolFunction{
				Name:        "web_search",
				Description: "Search the web",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
		}},
	}

	if _, err := chatComplete(context.Background(), srv.URL, req); err != nil {
		t.Fatalf("chatComplete returned error: %v", err)
	}

	if len(received.Tools) != 1 {
		t.Fatalf("len(received.Tools) = %d, want 1", len(received.Tools))
	}
	if received.Tools[0].Function.Name != "web_search" {
		t.Errorf("tool name = %q, want web_search", received.Tools[0].Function.Name)
	}
	if received.Tools[0].Type != "function" {
		t.Errorf("tool type = %q, want function", received.Tools[0].Type)
	}
}

func TestChatComplete_ParsesToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := chatResp{}
		resp.Choices = make([]struct {
			Message struct {
				Content          string         `json:"content"`
				ReasoningContent string         `json:"reasoning_content"`
				ToolCalls        []wireToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}, 1)
		resp.Choices[0].FinishReason = "tool_calls"
		resp.Choices[0].Message.ToolCalls = []wireToolCall{{
			ID:   "call_1",
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: "web_search", Arguments: `{"query":"weather tokyo"}`},
		}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	req := backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "what's the weather in tokyo"}},
		Tools: []backend.Tool{{
			Type:     "function",
			Function: backend.ToolFunction{Name: "web_search"},
		}},
	}

	resp, err := chatComplete(context.Background(), srv.URL, req)
	if err != nil {
		t.Fatalf("chatComplete returned error: %v", err)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(resp.ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	got := resp.ToolCalls[0]
	if got.ID != "call_1" || got.Name != "web_search" || got.Arguments != `{"query":"weather tokyo"}` {
		t.Errorf("ToolCalls[0] = %+v, want {ID:call_1 Name:web_search Arguments:{\"query\":\"weather tokyo\"}}", got)
	}
}

// TestChatComplete_LengthFinishReasonSetsLengthLimited is the non-streaming
// counterpart of TestChatCompleteStream_LengthFinishReasonSetsLengthLimited —
// same finish_reason:"length" gap, blocking /v1/infer path.
func TestChatComplete_LengthFinishReasonSetsLengthLimited(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := chatResp{}
		resp.Choices = make([]struct {
			Message struct {
				Content          string         `json:"content"`
				ReasoningContent string         `json:"reasoning_content"`
				ToolCalls        []wireToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}, 1)
		resp.Choices[0].Message.Content = "The answer starts here and then"
		resp.Choices[0].FinishReason = "length"
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	req := backend.Request{Messages: []backend.Message{{Role: "user", Content: "hi"}}}
	resp, err := chatComplete(context.Background(), srv.URL, req)
	if err != nil {
		t.Fatalf("chatComplete returned error: %v", err)
	}
	if !resp.LengthLimited {
		t.Error("LengthLimited = false for finish_reason \"length\", want true")
	}
}

// sseHandler writes each line in lines as its own "data: <line>\n\n" SSE
// event, followed by the terminal "data: [DONE]\n\n" sentinel.
func sseHandler(lines []string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, l := range lines {
			w.Write([]byte("data: " + l + "\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}
}

func TestChatCompleteStream_ForwardsTools(t *testing.T) {
	var received chatReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&received)
		sseHandler([]string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}`})(w, r)
	}))
	defer srv.Close()

	req := backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "search this"}},
		Tools: []backend.Tool{{
			Type:     "function",
			Function: backend.ToolFunction{Name: "web_search", Description: "Search the web"},
		}},
	}

	if _, err := chatCompleteStream(context.Background(), srv.URL, req, func(backend.ChunkKind, string) {}); err != nil {
		t.Fatalf("chatCompleteStream returned error: %v", err)
	}
	if len(received.Tools) != 1 || received.Tools[0].Function.Name != "web_search" {
		t.Fatalf("received.Tools = %+v, want one web_search tool", received.Tools)
	}
}

// TestChatCompleteStream_ReasoningStreamsLiveBeforeToolCall pins the
// real, live-verified SSE sequence for a tool-calling turn: several
// reasoning_content deltas, then incremental tool_calls deltas (arguments
// split across many chunks, matching what the real model actually streams),
// then finish_reason:"tool_calls". Asserts the reasoning deltas DO reach
// chunkFn as ChunkReasoning (live, not discarded) while the tool_calls
// deltas do not reach chunkFn at all.
func TestChatCompleteStream_ReasoningStreamsLiveBeforeToolCall(t *testing.T) {
	srv := httptest.NewServer(sseHandler([]string{
		`{"choices":[{"delta":{"reasoning_content":"The"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning_content":" user"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"reasoning_content":" wants search."},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{"}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"query"}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\"go news\"}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
	}))
	defer srv.Close()

	var reasoningSeen strings.Builder
	var contentCalls, toolLikeCalls int
	chunkFn := func(kind backend.ChunkKind, delta string) {
		switch kind {
		case backend.ChunkReasoning:
			reasoningSeen.WriteString(delta)
		case backend.ChunkContent:
			contentCalls++
			if strings.Contains(delta, "query") || strings.Contains(delta, "web_search") {
				toolLikeCalls++
			}
		}
	}

	req := backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "recent go news"}},
		Tools:    []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	}
	resp, err := chatCompleteStream(context.Background(), srv.URL, req, chunkFn)
	if err != nil {
		t.Fatalf("chatCompleteStream returned error: %v", err)
	}

	if reasoningSeen.String() != "The user wants search." {
		t.Errorf("reasoning seen by chunkFn = %q, want the full reasoning trace forwarded live", reasoningSeen.String())
	}
	if contentCalls != 0 {
		t.Errorf("chunkFn received %d ChunkContent calls, want 0 — tool-call fragments must never reach chunkFn", contentCalls)
	}
	if toolLikeCalls != 0 {
		t.Error("a tool-call JSON fragment leaked through to chunkFn")
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("len(resp.ToolCalls) = %d, want 1", len(resp.ToolCalls))
	}
	got := resp.ToolCalls[0]
	if got.ID != "call_1" || got.Name != "web_search" || got.Arguments != `{"query":"go news"}` {
		t.Errorf("ToolCalls[0] = %+v, want the fragments correctly accumulated", got)
	}
}

func TestChatCompleteStream_PlainAnswerStillStreamsNormally(t *testing.T) {
	srv := httptest.NewServer(sseHandler([]string{
		`{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{"content":" there"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}`,
	}))
	defer srv.Close()

	var content strings.Builder
	chunkFn := func(kind backend.ChunkKind, delta string) {
		if kind == backend.ChunkContent {
			content.WriteString(delta)
		}
	}

	req := backend.Request{Messages: []backend.Message{{Role: "user", Content: "hi"}}}
	resp, err := chatCompleteStream(context.Background(), srv.URL, req, chunkFn)
	if err != nil {
		t.Fatalf("chatCompleteStream returned error: %v", err)
	}
	if content.String() != "Hello there" {
		t.Errorf("content = %q, want %q", content.String(), "Hello there")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %+v, want none for a plain answer", resp.ToolCalls)
	}
	if resp.LengthLimited {
		t.Error("LengthLimited = true for finish_reason \"stop\", want false")
	}
}

// TestChatCompleteStream_LengthFinishReasonSetsLengthLimited pins the gap
// found while investigating a user report of a chat UI answer stopping
// mid-sentence with no error surfaced anywhere: llama-server's
// finish_reason:"length" (the model was still generating when it hit its
// max_tokens budget) was being parsed into chatStreamChunk.FinishReason but
// never read, so a genuinely cut-short answer looked identical to a normal
// completion end-to-end. This asserts the streamed content still reaches
// chunkFn (a length-limited answer's partial text is real and must be kept)
// while resp.LengthLimited reports the cutoff so callers can flag it.
func TestChatCompleteStream_LengthFinishReasonSetsLengthLimited(t *testing.T) {
	srv := httptest.NewServer(sseHandler([]string{
		`{"choices":[{"delta":{"content":"The answer starts here and then"},"finish_reason":null}]}`,
		`{"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"completion_tokens":8}}`,
	}))
	defer srv.Close()

	var content strings.Builder
	chunkFn := func(kind backend.ChunkKind, delta string) {
		if kind == backend.ChunkContent {
			content.WriteString(delta)
		}
	}

	req := backend.Request{Messages: []backend.Message{{Role: "user", Content: "hi"}}}
	resp, err := chatCompleteStream(context.Background(), srv.URL, req, chunkFn)
	if err != nil {
		t.Fatalf("chatCompleteStream returned error: %v", err)
	}
	if content.String() != "The answer starts here and then" {
		t.Errorf("content = %q, want the partial text to still stream", content.String())
	}
	if !resp.LengthLimited {
		t.Error("LengthLimited = false for finish_reason \"length\", want true")
	}
}
