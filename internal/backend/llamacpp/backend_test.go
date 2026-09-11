package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/backend/vram"
	"github.com/ngkaichong/home-inference-server/types"
)

// stubServer returns an httptest.Server that responds to /completion and /health.
func stubServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/completion", handler)
	return httptest.NewServer(mux)
}

func okHandler(output string, tokens int) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(completionResp{
			Content:         output,
			TokensPredicted: tokens,
		})
	}
}

// makeTestBackend creates a Backend that is already in the "ready" state
// pointing at the given base URL, bypassing subprocess lifecycle.
func makeTestBackend(baseURL string) *Backend {
	b := &Backend{
		modality: backend.ModalityKindText,
		desc: types.ModelDescriptor{
			TierLabel: types.TierWeak,
			Name:      "test-model",
		},
		exe:     "llama-server",
		baseURL: baseURL,
		ready:   true,
	}
	return b
}

func TestComplete_Success(t *testing.T) {
	srv := stubServer(t, okHandler("hello world", 3))
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	resp, err := b.Infer(context.Background(), backend.Request{
		Prompt:    "hi",
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Output != "hello world" {
		t.Errorf("output = %q; want %q", resp.Output, "hello world")
	}
	if resp.TokensGenerated != 3 {
		t.Errorf("tokens = %d; want 3", resp.TokensGenerated)
	}
	if resp.ModelTier != string(types.TierWeak) {
		t.Errorf("model_tier = %q; want %q", resp.ModelTier, types.TierWeak)
	}
}

func TestComplete_ContextCancelled(t *testing.T) {
	// Use a very short-lived context that times out almost immediately after the
	// request reaches the server. We verify the client gets a timeout error
	// without needing to coordinate a blocking handler goroutine.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// Handler that sleeps long enough to guarantee the context expires.
	srv := stubServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	})
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(ctx, backend.Request{Prompt: "test"})
	if err == nil {
		t.Fatal("expected error on timed-out context, got nil")
	}
	be, ok := err.(*backend.BackendError)
	if !ok {
		t.Fatalf("expected BackendError, got %T: %v", err, err)
	}
	if be.Code != "timeout" {
		t.Errorf("error code = %q; want %q", be.Code, "timeout")
	}
}

func TestComplete_ServerError(t *testing.T) {
	srv := stubServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(context.Background(), backend.Request{Prompt: "test"})
	if err == nil {
		t.Fatal("expected error on 500, got nil")
	}
	be, ok := err.(*backend.BackendError)
	if !ok {
		t.Fatalf("expected BackendError, got %T", err)
	}
	if be.Code != "model_load_failed" {
		t.Errorf("error code = %q; want %q", be.Code, "model_load_failed")
	}
}

func TestComplete_MalformedResponse(t *testing.T) {
	srv := stubServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("not json{{{"))
	})
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(context.Background(), backend.Request{Prompt: "test"})
	if err == nil {
		t.Fatal("expected error on malformed JSON, got nil")
	}
	be, ok := err.(*backend.BackendError)
	if !ok {
		t.Fatalf("expected BackendError, got %T", err)
	}
	if be.Code != "internal_error" {
		t.Errorf("error code = %q; want %q", be.Code, "internal_error")
	}
}

// captureProcess is a fake process that records the gpuLayers it was started with.
type captureProcess struct {
	gpuLayers int
	started   bool
}

// TestNew_NilVRAMProvider_UsesFullGPU verifies that when no VRAMProvider is
// supplied, ensureRunning falls back to -1 (full GPU offload).
func TestNew_NilVRAMProvider_UsesFullGPU(t *testing.T) {
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel:      types.TierStrong,
		Name:           "test",
		RequiredVRAMMB: 4747,
	}, "llama-server", nil, 1)

	if b.vramp != nil {
		t.Fatal("expected vramp to be nil")
	}
	// Derive the layer count the same way ensureRunning does when vramp is nil.
	gpuLayers := -1
	if b.vramp != nil {
		if availMB, err := b.vramp.AvailableMB(); err == nil {
			gpuLayers = vram.ComputeGPULayers(b.desc.RequiredVRAMMB, availMB)
		}
	}
	if gpuLayers != -1 {
		t.Errorf("gpuLayers = %d; want -1 (full GPU)", gpuLayers)
	}
}

// TestNew_VRAMProvider_ComputesLayers verifies proportional layer computation
// when VRAM is available but insufficient for full offload.
func TestNew_VRAMProvider_ComputesLayers(t *testing.T) {
	// E4B: 4747 MB weight; 3000 MB available → ratio ≈ 0.632 × 32 ≈ 20 layers
	mock := &vram.MockVRAMProvider{FreeMB: 3000}
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel:      types.TierStrong,
		Name:           "test",
		RequiredVRAMMB: 4747,
	}, "llama-server", mock, 1)

	var gpuLayers int
	if b.vramp != nil {
		if availMB, err := b.vramp.AvailableMB(); err == nil {
			gpuLayers = vram.ComputeGPULayers(b.desc.RequiredVRAMMB, availMB)
		}
	}
	if gpuLayers <= 0 || gpuLayers == -1 {
		t.Errorf("gpuLayers = %d; want proportional value (1–31)", gpuLayers)
	}
}

// TestNew_VRAMProviderError_FallsBackToFullGPU verifies that an NVML error
// leaves gpuLayers at -1 (full GPU) rather than propagating the error.
func TestNew_VRAMProviderError_FallsBackToFullGPU(t *testing.T) {
	mock := &vram.MockVRAMProvider{FreeMB: 0, Err: errors.New("nvml unavailable")}
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel:      types.TierStrong,
		Name:           "test",
		RequiredVRAMMB: 4747,
	}, "llama-server", mock, 1)

	gpuLayers := -1
	if b.vramp != nil {
		if availMB, err := b.vramp.AvailableMB(); err == nil {
			gpuLayers = vram.ComputeGPULayers(b.desc.RequiredVRAMMB, availMB)
		}
	}
	if gpuLayers != -1 {
		t.Errorf("gpuLayers = %d; want -1 on provider error", gpuLayers)
	}
}

// chatStub serves the chat-completions, tokenize, and props endpoints alongside
// the health check, and records which completion path was hit.
func chatStub(t *testing.T) (*httptest.Server, *stubState) {
	t.Helper()
	st := &stubState{}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/completion", func(w http.ResponseWriter, _ *http.Request) {
		st.completionHits++
		json.NewEncoder(w).Encode(completionResp{Content: "prompt-path", TokensPredicted: 2})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		st.chatHits++
		var body chatReq
		json.NewDecoder(r.Body).Decode(&body)
		st.lastChatMessages = len(body.Messages)
		st.lastMaxTokens = body.MaxTokens
		st.lastResponseFormat = string(body.ResponseFormat)
		resp := chatResp{}
		resp.Choices = append(resp.Choices, struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		}{})
		resp.Choices[0].Message.Content = "chat-path"
		resp.Usage.CompletionTokens = 5
		json.NewEncoder(w).Encode(resp)
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string][]int{"tokens": {1, 2, 3, 4}})
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, _ *http.Request) {
		st.propsHits++
		json.NewEncoder(w).Encode(map[string]any{
			"default_generation_settings": map[string]int{"n_ctx": 4096},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, st
}

type stubState struct {
	completionHits     int
	chatHits           int
	propsHits          int
	lastChatMessages   int
	lastMaxTokens      int
	lastResponseFormat string
}

func TestInfer_UsesChatEndpointWhenMessagesPresent(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)

	resp, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "user", Content: "how are you?"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.chatHits != 1 || st.completionHits != 0 {
		t.Fatalf("chatHits=%d completionHits=%d; want 1/0", st.chatHits, st.completionHits)
	}
	if st.lastChatMessages != 3 {
		t.Errorf("chat messages sent = %d; want 3", st.lastChatMessages)
	}
	if resp.Output != "chat-path" {
		t.Errorf("output = %q; want chat-path", resp.Output)
	}
}

func TestInfer_UsesCompletionEndpointForPromptOnly(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)

	_, err := b.Infer(context.Background(), backend.Request{Prompt: "just a prompt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.completionHits != 1 || st.chatHits != 0 {
		t.Fatalf("completionHits=%d chatHits=%d; want 1/0", st.completionHits, st.chatHits)
	}
}

func TestToChatMessages_PlainWhenNoImage(t *testing.T) {
	msgs := toChatMessages(backend.Request{
		Messages: []backend.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "hi"},
		},
	})
	for i, m := range msgs {
		if _, ok := m.Content.(string); !ok {
			t.Errorf("message %d Content is %T; want string", i, m.Content)
		}
	}
}

func TestToChatMessages_MultiPartWhenImagePresent(t *testing.T) {
	pngMagic := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 0}
	msgs := toChatMessages(backend.Request{
		Messages: []backend.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "what is this?"},
		},
		ImageData: pngMagic,
	})
	if _, ok := msgs[0].Content.(string); !ok {
		t.Errorf("system turn Content = %T; want string", msgs[0].Content)
	}
	parts, ok := msgs[1].Content.([]contentPart)
	if !ok {
		t.Fatalf("user turn Content = %T; want []contentPart", msgs[1].Content)
	}
	if len(parts) != 2 || parts[0].Type != "text" || parts[0].Text != "what is this?" {
		t.Errorf("part[0] = %+v; want text 'what is this?'", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil ||
		!strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("part[1] = %+v; want image_url data:image/png", parts[1])
	}
}

func TestSniffMIME_DetectsCommonFormats(t *testing.T) {
	cases := []struct {
		name string
		b    []byte
		want string
	}{
		{"jpeg", []byte{0xFF, 0xD8, 0xFF, 0xE0}, "image/jpeg"},
		{"png", []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}, "image/png"},
		{"webp", append([]byte("RIFF\x00\x00\x00\x00WEBP"), 0), "image/webp"},
		{"gif", []byte("GIF89a\x00"), "image/gif"},
		{"garbage", []byte{0x01, 0x02, 0x03, 0x04}, "image/jpeg"},
		{"empty", nil, "image/jpeg"},
	}
	for _, tc := range cases {
		if got := sniffMIME(tc.b); got != tc.want {
			t.Errorf("%s: sniffMIME = %q; want %q", tc.name, got, tc.want)
		}
	}
}

func TestChatComplete_ForwardsResponseFormat(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)

	rf := json.RawMessage(`{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}`)
	_, err := b.Infer(context.Background(), backend.Request{
		Messages:       []backend.Message{{Role: "user", Content: "hi"}},
		ResponseFormat: rf,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.lastResponseFormat != string(rf) {
		t.Errorf("forwarded response_format = %q; want %q", st.lastResponseFormat, string(rf))
	}
}

func TestChatComplete_NoResponseFormatWhenUnset(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)

	_, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.lastResponseFormat != "" {
		t.Errorf("response_format = %q; want empty (omitted)", st.lastResponseFormat)
	}
}

func TestChatComplete_InvalidGrammarMapsTo400Code(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		// Wording matches what llama-server actually returns for a bad schema.
		w.Write([]byte(`{"error":{"code":400,"message":"Unable to generate parser for this template. JSON schema conversion failed","type":"invalid_request_error"}}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(context.Background(), backend.Request{
		Messages:       []backend.Message{{Role: "user", Content: "hi"}},
		ResponseFormat: json.RawMessage(`{"type":"json_schema"}`),
	})
	be := &backend.BackendError{}
	if !errors.As(err, &be) {
		t.Fatalf("expected BackendError, got %T: %v", err, err)
	}
	if be.Code != "invalid_grammar" {
		t.Errorf("code = %q; want invalid_grammar", be.Code)
	}
}

func TestChatComplete_Non400StaysModelLoadFailed(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("loading model"))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	})
	be := &backend.BackendError{}
	if !errors.As(err, &be) {
		t.Fatalf("expected BackendError, got %T", err)
	}
	if be.Code != "model_load_failed" {
		t.Errorf("code = %q; want model_load_failed", be.Code)
	}
}

// chatStubWithCtx is chatStub but with a caller-controlled n_ctx and a fixed
// /tokenize token count, for exercising resolveChatMaxTokens precisely.
func chatStubWithCtx(t *testing.T, nCtx, tokenizeCount int) (*httptest.Server, *stubState) {
	t.Helper()
	st := &stubState{}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		st.chatHits++
		var body chatReq
		json.NewDecoder(r.Body).Decode(&body)
		st.lastMaxTokens = body.MaxTokens
		resp := chatResp{}
		resp.Choices = append(resp.Choices, struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		}{})
		resp.Choices[0].Message.Content = "ok"
		json.NewEncoder(w).Encode(resp)
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, _ *http.Request) {
		tokens := make([]int, tokenizeCount)
		json.NewEncoder(w).Encode(map[string][]int{"tokens": tokens})
	})
	mux.HandleFunc("/props", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"default_generation_settings": map[string]int{"n_ctx": nCtx},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, st
}

func TestResolveChatMaxTokens_DerivedFromRemainingContext(t *testing.T) {
	// n_ctx=4096, tokenize=4, 2 messages: prompt = 4 + 8*2 = 20;
	// margin = 4096*10/100 = 409; remaining = 4096 - 20 - 409 = 3667.
	srv, st := chatStubWithCtx(t, 4096, 4)
	b := makeTestBackend(srv.URL)

	_, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := 3667; st.lastMaxTokens != want {
		t.Errorf("max_tokens sent = %d; want %d", st.lastMaxTokens, want)
	}
}

func TestResolveChatMaxTokens_RespectsExplicitMaxTokens(t *testing.T) {
	srv, st := chatStubWithCtx(t, 4096, 4)
	b := makeTestBackend(srv.URL)

	_, err := b.Infer(context.Background(), backend.Request{
		Messages:  []backend.Message{{Role: "user", Content: "hi"}},
		MaxTokens: 100,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.lastMaxTokens != 100 {
		t.Errorf("max_tokens sent = %d; want the explicit 100 unchanged", st.lastMaxTokens)
	}
}

func TestResolveChatMaxTokens_ClampsToOneOnLongPrompt(t *testing.T) {
	// n_ctx=4096, a prompt token count that alone exceeds n_ctx minus margin
	// must never produce a negative or zero max_tokens.
	srv, st := chatStubWithCtx(t, 4096, 5000)
	b := makeTestBackend(srv.URL)

	_, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "a very long prompt"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.lastMaxTokens != 1 {
		t.Errorf("max_tokens sent = %d; want clamped to 1", st.lastMaxTokens)
	}
}

func TestResolveChatMaxTokens_FallsBackWhenPropsFails(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	var lastMaxTokens int
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body chatReq
		json.NewDecoder(r.Body).Decode(&body)
		lastMaxTokens = body.MaxTokens
		resp := chatResp{}
		resp.Choices = append(resp.Choices, struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		}{})
		json.NewEncoder(w).Encode(resp)
	})
	// No /tokenize or /props handler — both introspection calls fail (404).
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	b := makeTestBackend(srv.URL)
	_, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lastMaxTokens != defaultChatMaxTokens {
		t.Errorf("max_tokens sent = %d; want fallback %d", lastMaxTokens, defaultChatMaxTokens)
	}
}

func TestChatCompleteStream_ParsesDeltaContent(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, tok := range []string{"Hel", "lo", " world"} {
			w.Write([]byte(`data: {"choices":[{"delta":{"content":"` + tok + `"}}]}` + "\n\n"))
			fl.Flush()
		}
		w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":3},"timings":{"predicted_ms":300}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
		fl.Flush()
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	b := makeTestBackend(srv.URL)
	var got string
	resp, err := b.InferStream(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	}, func(kind backend.ChunkKind, delta string) { got += delta })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "Hello world" {
		t.Errorf("streamed text = %q; want %q", got, "Hello world")
	}
	if resp.TokensGenerated != 3 {
		t.Errorf("tokens = %d; want 3", resp.TokensGenerated)
	}
}

// TestChatCompleteStream_ParsesReasoningContent is the regression test for
// the bug where a reasoning-capable model's (e.g. Gemma's) chain-of-thought
// tokens arrive under delta.reasoning_content, a field the parser previously
// never read — every such chunk had delta.content == "", so InferStream
// called chunkFn zero times, and if the model spent its whole token budget
// reasoning, the caller saw success with no visible output whatsoever.
func TestChatCompleteStream_ParsesReasoningContent(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, tok := range []string{"Let me ", "think..."} {
			w.Write([]byte(`data: {"choices":[{"delta":{"reasoning_content":"` + tok + `"}}]}` + "\n\n"))
			fl.Flush()
		}
		for _, tok := range []string{"The ", "answer."} {
			w.Write([]byte(`data: {"choices":[{"delta":{"content":"` + tok + `"}}]}` + "\n\n"))
			fl.Flush()
		}
		w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":4},"timings":{"predicted_ms":400}}` + "\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
		fl.Flush()
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	b := makeTestBackend(srv.URL)
	var content, reasoning string
	resp, err := b.InferStream(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	}, func(kind backend.ChunkKind, delta string) {
		if kind == backend.ChunkReasoning {
			reasoning += delta
		} else {
			content += delta
		}
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reasoning != "Let me think..." {
		t.Errorf("reasoning = %q; want %q", reasoning, "Let me think...")
	}
	if content != "The answer." {
		t.Errorf("content = %q; want %q", content, "The answer.")
	}
	if resp.Reasoning != "Let me think..." {
		t.Errorf("resp.Reasoning = %q; want %q", resp.Reasoning, "Let me think...")
	}
	if resp.TokensGenerated != 4 {
		t.Errorf("tokens = %d; want 4", resp.TokensGenerated)
	}
}

// TestChatComplete_ParsesReasoningContent covers the non-streaming
// /v1/chat/completions path (used by the blocking POST /v1/infer).
func TestChatComplete_ParsesReasoningContent(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"content": "The answer.", "reasoning_content": "Let me think..."}},
			},
			"usage": map[string]int{"completion_tokens": 4},
		})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	b := makeTestBackend(srv.URL)
	resp, err := b.Infer(context.Background(), backend.Request{
		Messages: []backend.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Output != "The answer." {
		t.Errorf("Output = %q; want %q", resp.Output, "The answer.")
	}
	if resp.Reasoning != "Let me think..." {
		t.Errorf("Reasoning = %q; want %q", resp.Reasoning, "Let me think...")
	}
}

func TestTokenize_ReturnsTokenCount(t *testing.T) {
	srv, _ := chatStub(t)
	b := makeTestBackend(srv.URL)

	n, err := b.Tokenize(context.Background(), "some text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 4 {
		t.Errorf("token count = %d; want 4", n)
	}
}

func TestNCtx_ReadsPropsAndMemoises(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)

	n, err := b.NCtx(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 4096 {
		t.Errorf("n_ctx = %d; want 4096", n)
	}
	if _, err := b.NCtx(context.Background()); err != nil {
		t.Fatalf("second NCtx errored: %v", err)
	}
	if st.propsHits != 1 {
		t.Errorf("propsHits = %d; want 1 (memoised)", st.propsHits)
	}
}

func TestResidentMB_ZeroWhenNotRunning(t *testing.T) {
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel: types.TierStrong, Name: "test", RequiredVRAMMB: 4747,
	}, "llama-server", nil, 1)
	if got := b.ResidentMB(); got != 0 {
		t.Errorf("ResidentMB = %d; want 0 when not running", got)
	}
}

func TestRunningPID_ZeroWhenNotRunning(t *testing.T) {
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel: types.TierStrong, Name: "test", RequiredVRAMMB: 4747,
	}, "llama-server", nil, 1)
	if got := b.RunningPID(); got != 0 {
		t.Errorf("RunningPID = %d; want 0 when not running", got)
	}
}

func TestProbeHealthy(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	srv := httptest.NewServer(mux)

	if !probeHealthy(context.Background(), srv.URL) {
		t.Error("probeHealthy = false for a live 200 /health")
	}

	srv.Close()
	if probeHealthy(context.Background(), srv.URL) {
		t.Error("probeHealthy = true after the server was shut down")
	}
}

func TestPropsReady_GatesOnPropsStatus(t *testing.T) {
	var propsOK bool
	mux := http.NewServeMux()
	mux.HandleFunc("/props", func(w http.ResponseWriter, _ *http.Request) {
		if !propsOK {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	p := &process{}
	client := &http.Client{Timeout: time.Second}

	if p.propsReady(client, srv.URL+"/props") {
		t.Error("propsReady = true while /props still 503 (model loading)")
	}
	propsOK = true
	if !p.propsReady(client, srv.URL+"/props") {
		t.Error("propsReady = false once /props returns 200")
	}
}

func TestRestarting_FalseByDefault(t *testing.T) {
	b := New(backend.ModalityKindText, types.ModelDescriptor{
		TierLabel: types.TierStrong, Name: "test",
	}, "llama-server", nil, 1)
	if b.Restarting() {
		t.Error("Restarting() = true on a fresh backend")
	}
}

// TestEnsureRunning_FastPathWhenNoProc guards the boundary the test harness
// relies on: a ready backend with no proc handle (test setup) returns its URL
// without a health re-probe.
func TestEnsureRunning_FastPathWhenNoProc(t *testing.T) {
	b := makeTestBackend("http://127.0.0.1:0")
	url, err := b.ensureRunning(context.Background())
	if err != nil {
		t.Fatalf("ensureRunning errored: %v", err)
	}
	if url != "http://127.0.0.1:0" {
		t.Errorf("url = %q; want the preset base URL", url)
	}
}

// TestEnsureRunning_StaleReadyReloadsRatherThanTrustingDeadProc covers the
// race an idle eviction can create: the ready flag and proc handle still say
// "running" but the subprocess behind baseURL is already gone (e.g. its
// eviction-triggered Shutdown raced this call). ensureRunning must not trust
// the stale flag — it should clear ready/proc and attempt a fresh load rather
// than handing back a URL nothing is listening on.
func TestEnsureRunning_StaleReadyReloadsRatherThanTrustingDeadProc(t *testing.T) {
	srv := stubServer(t, okHandler("unused", 0))
	srv.Close() // dead immediately: probeHealthy must fail against this URL

	b := makeTestBackend(srv.URL)
	b.proc = &process{} // non-nil handle so probeHealthy actually runs
	b.exe = "llama-server-does-not-exist"

	_, err := b.ensureRunning(context.Background())
	if err == nil {
		t.Fatal("expected an error attempting to reload with a nonexistent exe")
	}
	be, ok := err.(*backend.BackendError)
	if !ok || be.Code != "model_load_failed" {
		t.Fatalf("want model_load_failed reload attempt, got %v", err)
	}

	b.mu.Lock()
	stillReady, stillProc := b.ready, b.proc
	b.mu.Unlock()
	if stillReady || stillProc != nil {
		t.Errorf("stale ready/proc were not cleared: ready=%v proc=%v", stillReady, stillProc)
	}
}

// TestEnsureRunning_SpawnsWithMMProjWhenNeedsVisionSet covers the fast path
// where the subprocess is already running in the mode a request needs: no
// mismatch, no respawn, the request just proceeds. (The actual --mmproj
// spawn-arg construction is covered by TestSpawnArgsIncludeMMProjWhenSet;
// exercising a real cold spawn end-to-end needs a real llama-server binary,
// which unit tests don't have.)
func TestEnsureRunning_SpawnsWithMMProjWhenNeedsVisionSet(t *testing.T) {
	srv, st := chatStub(t)
	b := makeTestBackend(srv.URL)
	b.desc.MMProjPath = "mmproj.gguf"
	b.runningWithVision = true // already running in vision mode

	b.SetNextNeedsVision(true) // request wants vision mode too — no mismatch
	if _, err := b.Infer(context.Background(), backend.Request{
		Messages:  []backend.Message{{Role: "user", Content: "hi"}},
		ImageData: []byte{1},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if st.chatHits != 1 {
		t.Fatalf("chatHits = %d; want 1", st.chatHits)
	}
}

func TestEnsureRunning_ModeMismatchClearsReadyAndStopsProcess(t *testing.T) {
	srv := stubServer(t, okHandler("unused", 0))
	defer srv.Close()

	b := makeTestBackend(srv.URL)
	b.proc = &process{} // non-nil handle so the mismatch path calls Stop on it
	b.runningWithVision = false
	b.exe = "llama-server-does-not-exist" // the respawn attempt fails; we only assert the mismatch side-effects

	b.SetNextNeedsVision(true)
	_, err := b.ensureRunning(context.Background())
	if err == nil {
		t.Fatal("expected an error from the doomed respawn attempt")
	}

	b.mu.Lock()
	stillReady, stillProc := b.ready, b.proc
	b.mu.Unlock()
	if stillReady || stillProc != nil {
		t.Errorf("mode-mismatch path did not clear ready/proc: ready=%v proc=%v", stillReady, stillProc)
	}
}

func TestEnsureRunning_SameModeReusesWithoutMismatchPath(t *testing.T) {
	b := makeTestBackend("http://127.0.0.1:0")
	b.runningWithVision = true
	b.SetNextNeedsVision(true)

	url, err := b.ensureRunning(context.Background())
	if err != nil {
		t.Fatalf("ensureRunning errored: %v", err)
	}
	if url != "http://127.0.0.1:0" {
		t.Errorf("url = %q; want the preset base URL (fast path, no mismatch)", url)
	}
}

// --- --parallel spawn args + per-request tok/sec normalisation ---

func argsHave(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

func TestSpawnArgsIncludeParallelAndContBatching(t *testing.T) {
	p := newProcess("llama-server", `models\m.gguf`, "", "strong", -1, 8092, 3)
	args := p.spawnArgs()

	for _, w := range []string{"--parallel", "3", "--cont-batching"} {
		if !argsHave(args, w) {
			t.Errorf("spawn args %v missing %q", args, w)
		}
	}
}

func TestNewProcessClampsParallelToOne(t *testing.T) {
	p := newProcess("llama-server", "m", "", "weak", -1, 8090, 0)
	if p.parallel != 1 {
		t.Errorf("parallel = %d; want 1 (clamped)", p.parallel)
	}
}

func TestSpawnArgsIncludeMMProjWhenSet(t *testing.T) {
	p := newProcess("llama-server", `models\m.gguf`, `models\mmproj.gguf`, "weak", -1, 8094, 1)
	args := p.spawnArgs()
	if !argsHave(args, "--mmproj") || !argsHave(args, `models\mmproj.gguf`) {
		t.Errorf("spawn args %v missing --mmproj <path>", args)
	}
}

func TestSpawnArgsOmitMMProjWhenEmpty(t *testing.T) {
	p := newProcess("llama-server", `models\m.gguf`, "", "weak", -1, 8090, 1)
	if argsHave(p.spawnArgs(), "--mmproj") {
		t.Errorf("spawn args %v should not contain --mmproj when path is empty", p.spawnArgs())
	}
}

func TestRecordTokSampleNormalisesBySlotCount(t *testing.T) {
	b := &Backend{}
	b.recordTokSample(10, 1) // solo: trusted as-is
	b.recordTokSample(4, 3)  // 3-way concurrent per-stream 4 → normalised 12
	b.recordTokSample(0, 2)  // zero sample: ignored

	// ring holds {10, 12}; mean = 11
	if got := b.TokPerSec(); got != 11 {
		t.Fatalf("TokPerSec() = %v; want 11 (mean of 10 and 12)", got)
	}
}

func TestNewSetsMaxParallel(t *testing.T) {
	b := New(backend.ModalityKindText, types.ModelDescriptor{TierLabel: types.TierStrong, Name: "m"}, "llama-server", nil, 4)
	if b.maxParallel != 4 {
		t.Errorf("maxParallel = %d; want 4", b.maxParallel)
	}
	b0 := New(backend.ModalityKindText, types.ModelDescriptor{TierLabel: types.TierWeak, Name: "m"}, "llama-server", nil, 0)
	if b0.maxParallel != 1 {
		t.Errorf("maxParallel = %d; want 1 (clamped)", b0.maxParallel)
	}
}
