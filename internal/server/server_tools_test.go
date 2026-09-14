package server

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/websearch"
)

// toolCallBackend is a scripted backend.Backend for the tool-calling
// orchestration loop: the Nth call to Infer returns responses[N] (or the last
// entry if there are more calls than scripted responses). callCount and
// lastReq record what actually happened so tests can assert on it.
type toolCallBackend struct {
	responses []backend.Response
	err       error // returned on every call if set (takes precedence)
	calls     int
	lastReq   backend.Request
}

func (b *toolCallBackend) Modality() backend.ModalityKind   { return backend.ModalityKindText }
func (b *toolCallBackend) Ready() bool                      { return true }
func (b *toolCallBackend) Shutdown(_ context.Context) error { return nil }
func (b *toolCallBackend) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	b.lastReq = req
	idx := b.calls
	if idx >= len(b.responses) {
		idx = len(b.responses) - 1
	}
	b.calls++
	if b.err != nil {
		return backend.Response{}, b.err
	}
	return b.responses[idx], nil
}

// fakeSearcher is a scripted ToolSearcher.
type fakeSearcher struct {
	results []websearch.Result
	err     error
	calls   int
	lastQ   string
}

func (f *fakeSearcher) Search(_ context.Context, query string) ([]websearch.Result, error) {
	f.calls++
	f.lastQ = query
	if f.err != nil {
		return nil, f.err
	}
	return f.results, nil
}

func TestRunWithTools_NoToolCallPassesThrough(t *testing.T) {
	b := &toolCallBackend{responses: []backend.Response{{Output: "a plain answer"}}}
	s := &Server{}

	resp, summaries, err := s.runWithTools(context.Background(), b, backend.Request{})
	if err != nil {
		t.Fatalf("runWithTools returned error: %v", err)
	}
	if resp.Output != "a plain answer" {
		t.Errorf("Output = %q, want %q", resp.Output, "a plain answer")
	}
	if len(summaries) != 0 {
		t.Errorf("len(summaries) = %d, want 0", len(summaries))
	}
	if b.calls != 1 {
		t.Errorf("backend called %d times, want 1 (search should never run)", b.calls)
	}
}

func TestRunWithTools_ResolvesWebSearchAndReturnsGroundedAnswer(t *testing.T) {
	b := &toolCallBackend{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"weather tokyo"}`}}},
		{Output: "It's sunny in Tokyo, grounded by search."},
	}}
	searcher := &fakeSearcher{results: []websearch.Result{{Title: "Tokyo Weather", URL: "https://x", Content: "sunny"}}}
	s := &Server{}
	s.SetToolSearcher(searcher)

	resp, summaries, err := s.runWithTools(context.Background(), b, backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "what's the weather in tokyo"}},
		Tools:    []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	})
	if err != nil {
		t.Fatalf("runWithTools returned error: %v", err)
	}
	if resp.Output != "It's sunny in Tokyo, grounded by search." {
		t.Errorf("Output = %q", resp.Output)
	}
	if b.calls != 2 {
		t.Fatalf("backend called %d times, want 2 (one-hop bound)", b.calls)
	}
	if searcher.calls != 1 {
		t.Errorf("searcher called %d times, want 1", searcher.calls)
	}
	if searcher.lastQ != "weather tokyo" {
		t.Errorf("search query = %q, want %q", searcher.lastQ, "weather tokyo")
	}
	if len(summaries) != 1 || summaries[0].Name != "web_search" || summaries[0].Error != "" {
		t.Errorf("summaries = %+v, want one clean web_search summary", summaries)
	}
	// The follow-up call must not re-offer tools (bounds the loop to one hop).
	if len(b.lastReq.Tools) != 0 {
		t.Errorf("follow-up request still carried Tools: %+v", b.lastReq.Tools)
	}
}

func TestRunWithTools_SearchBackendUnavailable(t *testing.T) {
	b := &toolCallBackend{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"x"}`}}},
		{Output: "I could not search, but here's what I know."},
	}}
	searcher := &fakeSearcher{err: &backend.BackendError{Code: "tool_unavailable", Message: "search backend unreachable"}}
	s := &Server{}
	s.SetToolSearcher(searcher)

	resp, summaries, err := s.runWithTools(context.Background(), b, backend.Request{
		Tools: []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	})
	if err != nil {
		t.Fatalf("runWithTools returned error: %v (should degrade gracefully, not fail)", err)
	}
	if b.calls != 2 {
		t.Fatalf("backend called %d times, want 2 (second call still happens)", b.calls)
	}
	if len(summaries) != 1 || summaries[0].Error == "" {
		t.Fatalf("summaries = %+v, want one summary with a non-empty Error", summaries)
	}
	if resp.Output == "" {
		t.Errorf("expected a final answer despite search failure")
	}

	// The role:"tool" message the model actually sees must explicitly tell
	// it to frame the fallback answer as training-data-based, not current —
	// this is the graceful-degradation behavior itself, not just that an
	// error string exists somewhere.
	toolMsg := lastToolMessage(t, b.lastReq)
	if !strings.Contains(toolMsg, "search backend unreachable") {
		t.Errorf("tool message = %q, want it to include the underlying error", toolMsg)
	}
	if !strings.Contains(toolMsg, "Based on my last update") {
		t.Errorf("tool message = %q, want it to instruct the model to caveat its answer as not current", toolMsg)
	}
}

// lastToolMessage returns the Content of the last role:"tool" message in
// req.Messages, failing the test if there isn't one.
func lastToolMessage(t *testing.T, req backend.Request) string {
	t.Helper()
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "tool" {
			return req.Messages[i].Content
		}
	}
	t.Fatalf("no role:\"tool\" message found in %+v", req.Messages)
	return ""
}

func TestRunWithTools_SearchRateLimited(t *testing.T) {
	b := &toolCallBackend{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"x"}`}}},
		{Output: "Based on my last update, ..."},
	}}
	searcher := &fakeSearcher{err: &backend.BackendError{
		Code:    "tool_rate_limited",
		Message: "rate limited (limit resets every 1h); retry after 3200s",
	}}
	s := &Server{}
	s.SetToolSearcher(searcher)

	resp, summaries, err := s.runWithTools(context.Background(), b, backend.Request{
		Tools: []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	})
	if err != nil {
		t.Fatalf("runWithTools returned error: %v (should degrade gracefully, not fail)", err)
	}
	if len(summaries) != 1 || !strings.Contains(summaries[0].Error, "retry after 3200s") {
		t.Fatalf("summaries = %+v, want the retry-after timing surfaced", summaries)
	}
	if resp.Output == "" {
		t.Errorf("expected a final answer despite the rate limit")
	}

	toolMsg := lastToolMessage(t, b.lastReq)
	if !strings.Contains(toolMsg, "retry after 3200s") {
		t.Errorf("tool message = %q, want the retry timing passed through to the model", toolMsg)
	}
	if !strings.Contains(toolMsg, "Based on my last update") {
		t.Errorf("tool message = %q, want the graceful-degradation instruction on a rate limit too, not just a hard failure", toolMsg)
	}
}

func TestRunWithTools_UnknownToolName(t *testing.T) {
	b := &toolCallBackend{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "send_email", Arguments: `{}`}}},
		{Output: "I can't send emails."},
	}}
	s := &Server{} // no ToolSearcher wired — irrelevant, send_email isn't web_search anyway

	_, summaries, err := s.runWithTools(context.Background(), b, backend.Request{})
	if err != nil {
		t.Fatalf("runWithTools returned error: %v", err)
	}
	if len(summaries) != 1 || summaries[0].Name != "send_email" || summaries[0].Error != "tool not available" {
		t.Errorf("summaries = %+v, want one send_email summary with 'tool not available'", summaries)
	}
	if b.calls != 2 {
		t.Errorf("backend called %d times, want 2 (request should not hard-fail)", b.calls)
	}
}

func TestRunWithTools_FirstCallError(t *testing.T) {
	wantErr := &backend.BackendError{Code: "unavailable", Message: "boom"}
	b := &toolCallBackend{err: wantErr}
	s := &Server{}

	_, _, err := s.runWithTools(context.Background(), b, backend.Request{})
	if !errors.Is(err, error(wantErr)) && err != wantErr {
		t.Errorf("err = %v, want %v", err, wantErr)
	}
}

// toolCallStreamer is the streaming counterpart of toolCallBackend: scripted
// responses per call, and it actually invokes chunkFn with each response's
// Output as one ChunkContent delta (or Reasoning as one ChunkReasoning delta,
// if set) so tests can assert what a real caller would have seen streamed.
type toolCallStreamer struct {
	responses []backend.Response
	err       error
	calls     int
	lastReq   backend.Request
}

func (b *toolCallStreamer) Modality() backend.ModalityKind   { return backend.ModalityKindText }
func (b *toolCallStreamer) Ready() bool                      { return true }
func (b *toolCallStreamer) Shutdown(_ context.Context) error { return nil }
func (b *toolCallStreamer) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	panic("toolCallStreamer.Infer should not be called by the streaming orchestration path")
}
func (b *toolCallStreamer) InferStream(_ context.Context, req backend.Request, chunkFn func(backend.ChunkKind, string)) (backend.Response, error) {
	b.lastReq = req
	idx := b.calls
	if idx >= len(b.responses) {
		idx = len(b.responses) - 1
	}
	b.calls++
	if b.err != nil {
		return backend.Response{}, b.err
	}
	resp := b.responses[idx]
	if resp.Reasoning != "" {
		chunkFn(backend.ChunkReasoning, resp.Reasoning)
	}
	if resp.Output != "" {
		chunkFn(backend.ChunkContent, resp.Output)
	}
	return resp, nil
}

func TestRunWithToolsStream_NoToolCallStreamsDirectly(t *testing.T) {
	b := &toolCallStreamer{responses: []backend.Response{{Output: "a plain answer"}}}
	s := &Server{}

	var streamedContent strings.Builder
	chunkFn := func(kind backend.ChunkKind, delta string) {
		if kind == backend.ChunkContent {
			streamedContent.WriteString(delta)
		}
	}

	resp, summaries, err := s.runWithToolsStream(context.Background(), b, backend.Request{}, chunkFn, nil)
	if err != nil {
		t.Fatalf("runWithToolsStream returned error: %v", err)
	}
	if resp.Output != "a plain answer" {
		t.Errorf("Output = %q, want %q", resp.Output, "a plain answer")
	}
	if streamedContent.String() != "a plain answer" {
		t.Errorf("streamed content = %q, want it to have reached chunkFn live", streamedContent.String())
	}
	if len(summaries) != 0 {
		t.Errorf("len(summaries) = %d, want 0", len(summaries))
	}
	if b.calls != 1 {
		t.Errorf("streamer called %d times, want 1 (search should never run)", b.calls)
	}
}

func TestRunWithToolsStream_ResolvesWebSearchAndStreamsGroundedAnswer(t *testing.T) {
	b := &toolCallStreamer{responses: []backend.Response{
		{Reasoning: "The user wants current info, I should search.", ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"weather tokyo"}`}}},
		{Output: "It's sunny in Tokyo, grounded by search."},
	}}
	searcher := &fakeSearcher{results: []websearch.Result{{Title: "Tokyo Weather", URL: "https://x", Content: "sunny"}}}
	s := &Server{}
	s.SetToolSearcher(searcher)

	var streamedReasoning, streamedContent strings.Builder
	chunkFn := func(kind backend.ChunkKind, delta string) {
		switch kind {
		case backend.ChunkReasoning:
			streamedReasoning.WriteString(delta)
		case backend.ChunkContent:
			streamedContent.WriteString(delta)
		}
	}

	var onToolCallSeen []api.ToolCallSummary
	resp, summaries, err := s.runWithToolsStream(context.Background(), b, backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "what's the weather in tokyo"}},
		Tools:    []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	}, chunkFn, func(summary api.ToolCallSummary) {
		onToolCallSeen = append(onToolCallSeen, summary)
	})
	if err != nil {
		t.Fatalf("runWithToolsStream returned error: %v", err)
	}
	if resp.Output != "It's sunny in Tokyo, grounded by search." {
		t.Errorf("Output = %q", resp.Output)
	}
	if streamedReasoning.String() != "The user wants current info, I should search." {
		t.Errorf("streamed reasoning = %q, want the first hop's reasoning to have streamed live", streamedReasoning.String())
	}
	if streamedContent.String() != "It's sunny in Tokyo, grounded by search." {
		t.Errorf("streamed content = %q, want the second hop's answer to have streamed live", streamedContent.String())
	}
	if b.calls != 2 {
		t.Fatalf("streamer called %d times, want 2 (one-hop bound)", b.calls)
	}
	if searcher.calls != 1 {
		t.Errorf("searcher called %d times, want 1", searcher.calls)
	}
	if len(summaries) != 1 || summaries[0].Name != "web_search" || summaries[0].Error != "" {
		t.Errorf("summaries = %+v, want one clean web_search summary", summaries)
	}
	if len(onToolCallSeen) != 1 || onToolCallSeen[0].Name != "web_search" {
		t.Errorf("onToolCall callback saw %+v, want one web_search notification", onToolCallSeen)
	}
	if len(b.lastReq.Tools) != 0 {
		t.Errorf("follow-up request still carried Tools: %+v", b.lastReq.Tools)
	}
}

func TestRunWithToolsStream_SearchFailureStillStreamsFinalAnswer(t *testing.T) {
	b := &toolCallStreamer{responses: []backend.Response{
		{ToolCalls: []backend.ToolCall{{ID: "call_1", Name: "web_search", Arguments: `{"query":"x"}`}}},
		{Output: "Based on my last update, I could not search."},
	}}
	searcher := &fakeSearcher{err: &backend.BackendError{Code: "tool_unavailable", Message: "search backend unreachable"}}
	s := &Server{}
	s.SetToolSearcher(searcher)

	var streamedContent strings.Builder
	chunkFn := func(kind backend.ChunkKind, delta string) {
		if kind == backend.ChunkContent {
			streamedContent.WriteString(delta)
		}
	}

	resp, summaries, err := s.runWithToolsStream(context.Background(), b, backend.Request{
		Tools: []backend.Tool{{Type: "function", Function: backend.ToolFunction{Name: "web_search"}}},
	}, chunkFn, nil)
	if err != nil {
		t.Fatalf("runWithToolsStream returned error: %v (should degrade gracefully, not fail)", err)
	}
	if b.calls != 2 {
		t.Fatalf("streamer called %d times, want 2 (second call still happens)", b.calls)
	}
	if len(summaries) != 1 || summaries[0].Error == "" {
		t.Fatalf("summaries = %+v, want one summary with a non-empty Error", summaries)
	}
	if streamedContent.String() != "Based on my last update, I could not search." {
		t.Errorf("streamed content = %q, want the graceful-degradation answer to have streamed live", streamedContent.String())
	}
	if resp.Output == "" {
		t.Errorf("expected a final answer despite search failure")
	}
}
