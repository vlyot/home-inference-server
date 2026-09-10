package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
)

var httpClient = &http.Client{Timeout: 0} // no timeout — callers use ctx deadline

// grammarErrorSignatures are substrings llama-server puts in a 400 body when it
// cannot turn the request's response_format into a sampling grammar. Matched
// case-insensitively.
var grammarErrorSignatures = []string{
	"json schema conversion failed",
	"unable to generate parser",
	"grammar",
	"json_schema",
	"response_format",
}

// chatErrorFromStatus turns a non-2xx chat/completions response into a
// BackendError. A 400 whose body indicates the response_format schema could not
// be compiled is surfaced as ErrCodeInvalidGrammar (the caller's schema is at
// fault, HTTP 400); anything else stays model_load_failed (503). The body is
// read once, capped, and closed by the caller's defer.
func chatErrorFromStatus(resp *http.Response) *backend.BackendError {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if resp.StatusCode == http.StatusBadRequest {
		low := strings.ToLower(string(body))
		for _, sig := range grammarErrorSignatures {
			if strings.Contains(low, sig) {
				return &backend.BackendError{
					Code:    "invalid_grammar",
					Message: "response_format could not be compiled to a grammar: " + strings.TrimSpace(string(body)),
				}
			}
		}
	}
	return &backend.BackendError{
		Code:    "model_load_failed",
		Message: fmt.Sprintf("llama-server returned HTTP %d", resp.StatusCode),
	}
}

// defaultCompletionMaxTokens caps a single-shot /completion request (no chat
// template, no reasoning) when the caller doesn't specify one.
const defaultCompletionMaxTokens = 512

// defaultChatMaxTokens is the fallback cap for a /v1/chat/completions request
// when the caller didn't specify one AND Backend.resolveChatMaxTokens
// couldn't compute a context-derived cap (e.g. a transient /props or
// /tokenize failure). The normal case doesn't use this constant at all — it
// uses whatever is actually left in the model's context window, because a
// fixed cap can't know how long the conversation already is: a
// reasoning-capable model (e.g. Gemma's "thinking" mode) spends part of its
// budget on chain-of-thought before producing a visible answer, and the old
// fixed 512 was observed to be fully consumed by reasoning alone on a
// moderately open-ended prompt, leaving zero tokens for the actual response.
const defaultChatMaxTokens = 2048

type completionReq struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	Stream      bool    `json:"stream"`
}

type completionResp struct {
	Content         string `json:"content"`
	TokensPredicted int    `json:"tokens_predicted"`
	// timings block returned by llama-server
	Timings struct {
		PredictedMS float64 `json:"predicted_ms"`
	} `json:"timings"`
}

// complete sends a single non-streaming completion request to a running
// llama-server at baseURL and returns the result as a backend.Response.
func complete(ctx context.Context, baseURL string, req backend.Request) (backend.Response, error) {
	body := completionReq{
		Prompt:      req.Prompt,
		NPredict:    req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      false,
	}
	if body.NPredict == 0 {
		body.NPredict = defaultCompletionMaxTokens
	}

	data, err := json.Marshal(body)
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "marshal request", Cause: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/completion", bytes.NewReader(data))
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "build request", Cause: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "request cancelled or timed out", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()
	elapsed := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		return backend.Response{}, &backend.BackendError{
			Code:    "model_load_failed",
			Message: fmt.Sprintf("llama-server returned HTTP %d", resp.StatusCode),
		}
	}

	var cr completionResp
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "decode response", Cause: err}
	}

	_ = elapsed

	var tokPerSec float64
	if cr.Timings.PredictedMS > 0 && cr.TokensPredicted > 0 {
		tokPerSec = float64(cr.TokensPredicted) / (cr.Timings.PredictedMS / 1000)
	}

	return backend.Response{
		Output:          cr.Content,
		TokensGenerated: cr.TokensPredicted,
		TokPerSecSample: tokPerSec,
	}, nil
}

// streamChunk is the per-token SSE payload from llama-server's streaming endpoint.
type streamChunk struct {
	Content         string `json:"content"`
	Stop            bool   `json:"stop"`
	TokensPredicted int    `json:"tokens_predicted"`
	Timings         struct {
		PredictedMS float64 `json:"predicted_ms"`
	} `json:"timings"`
}

// completeStream POSTs to /completion with stream:true and calls chunkFn for
// each token delta. Returns the final summary (token count, timing) when done.
func completeStream(ctx context.Context, baseURL string, req backend.Request, chunkFn func(delta string)) (backend.Response, error) {
	body := completionReq{
		Prompt:      req.Prompt,
		NPredict:    req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
	}
	if body.NPredict == 0 {
		body.NPredict = defaultCompletionMaxTokens
	}

	data, err := json.Marshal(body)
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "marshal stream request", Cause: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/completion", bytes.NewReader(data))
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "build stream request", Cause: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "stream request cancelled or timed out", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return backend.Response{}, &backend.BackendError{
			Code:    "model_load_failed",
			Message: fmt.Sprintf("llama-server returned HTTP %d", resp.StatusCode),
		}
	}

	scanner := bufio.NewScanner(resp.Body)
	var finalChunk streamChunk
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if chunk.Content != "" {
			chunkFn(chunk.Content)
		}
		if chunk.Stop {
			finalChunk = chunk
		}
	}

	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "stream interrupted", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "stream read error", Cause: err}
	}

	var tokPerSec float64
	if finalChunk.Timings.PredictedMS > 0 && finalChunk.TokensPredicted > 0 {
		tokPerSec = float64(finalChunk.TokensPredicted) / (finalChunk.Timings.PredictedMS / 1000)
	}

	return backend.Response{
		TokensGenerated: finalChunk.TokensPredicted,
		TokPerSecSample: tokPerSec,
	}, nil
}

// ── chat-completions path (multi-turn) ─────────────────────────────────────────

// chatMessage.Content is either a plain string (text turn) or a []contentPart
// (a turn carrying an image, per the OpenAI multi-part content shape).
type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// contentPart is one element of a multi-part message: a text span or an image.
type contentPart struct {
	Type     string    `json:"type"` // "text" | "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"` // "data:<mime>;base64,<...>"
}

// sniffMIME picks an image MIME type from the leading magic bytes. Defaults to
// image/jpeg for anything unrecognised — llama-server's image loader sniffs the
// real format itself; the data-URI label is advisory.
func sniffMIME(b []byte) string {
	switch {
	case len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF:
		return "image/jpeg"
	case len(b) >= 8 && string(b[:8]) == "\x89PNG\r\n\x1a\n":
		return "image/png"
	case len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WEBP":
		return "image/webp"
	case len(b) >= 6 && (string(b[:6]) == "GIF87a" || string(b[:6]) == "GIF89a"):
		return "image/gif"
	default:
		return "image/jpeg"
	}
}

type chatReq struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float32       `json:"temperature,omitempty"`
	Stream      bool          `json:"stream"`
	// ResponseFormat is forwarded verbatim from the caller. llama-server
	// compiles a {"type":"json_schema",...} or {"type":"json_object"} value to
	// a GBNF sampling grammar. Nil / omitted leaves generation unconstrained.
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
}

type chatResp struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
			// ReasoningContent carries a reasoning-capable model's (e.g.
			// Gemma's) chain-of-thought text, kept separate from Content.
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Timings struct {
		PredictedMS float64 `json:"predicted_ms"`
	} `json:"timings"`
}

// toChatMessages converts the backend message list into the wire shape. With no
// image, every turn's Content is a plain string. With an image, the last user
// turn's Content becomes a []contentPart: its text followed by an image_url
// data-URI part — the shape llama-server's --mmproj path expects.
func toChatMessages(req backend.Request) []chatMessage {
	out := make([]chatMessage, len(req.Messages))
	for i, m := range req.Messages {
		out[i] = chatMessage{Role: m.Role, Content: m.Content}
	}
	if len(req.ImageData) == 0 {
		return out
	}
	lastUser := -1
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == "user" {
			lastUser = i
			break
		}
	}
	if lastUser < 0 {
		return out
	}
	text, _ := out[lastUser].Content.(string)
	dataURI := "data:" + sniffMIME(req.ImageData) + ";base64," +
		base64.StdEncoding.EncodeToString(req.ImageData)
	out[lastUser].Content = []contentPart{
		{Type: "text", Text: text},
		{Type: "image_url", ImageURL: &imageURL{URL: dataURI}},
	}
	return out
}

// chatComplete sends a non-streaming /v1/chat/completions request.
func chatComplete(ctx context.Context, baseURL string, req backend.Request) (backend.Response, error) {
	body := chatReq{
		Model:          "local",
		Messages:       toChatMessages(req),
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		Stream:         false,
		ResponseFormat: req.ResponseFormat,
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = defaultChatMaxTokens
	}

	data, err := json.Marshal(body)
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "marshal chat request", Cause: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "build chat request", Cause: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "request cancelled or timed out", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return backend.Response{}, chatErrorFromStatus(resp)
	}

	var cr chatResp
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "decode chat response", Cause: err}
	}
	if len(cr.Choices) == 0 {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "chat response had no choices"}
	}

	var tokPerSec float64
	if cr.Timings.PredictedMS > 0 && cr.Usage.CompletionTokens > 0 {
		tokPerSec = float64(cr.Usage.CompletionTokens) / (cr.Timings.PredictedMS / 1000)
	}

	return backend.Response{
		Output:          cr.Choices[0].Message.Content,
		Reasoning:       cr.Choices[0].Message.ReasoningContent,
		TokensGenerated: cr.Usage.CompletionTokens,
		TokPerSecSample: tokPerSec,
	}, nil
}

type chatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			// ReasoningContent carries a reasoning-capable model's (e.g.
			// Gemma's) chain-of-thought text, streamed separately from
			// Content — see backend.ChunkReasoning.
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Timings struct {
		PredictedMS float64 `json:"predicted_ms"`
	} `json:"timings"`
}

// chatCompleteStream streams /v1/chat/completions, calling chunkFn per delta,
// tagged as ChunkContent or ChunkReasoning.
func chatCompleteStream(ctx context.Context, baseURL string, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	body := struct {
		chatReq
		StreamOptions map[string]bool `json:"stream_options"`
	}{
		chatReq: chatReq{
			Model:          "local",
			Messages:       toChatMessages(req),
			MaxTokens:      req.MaxTokens,
			Temperature:    req.Temperature,
			Stream:         true,
			ResponseFormat: req.ResponseFormat,
		},
		StreamOptions: map[string]bool{"include_usage": true},
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = defaultChatMaxTokens
	}

	data, err := json.Marshal(body)
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "marshal chat stream request", Cause: err}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "build chat stream request", Cause: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "stream request cancelled or timed out", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return backend.Response{}, chatErrorFromStatus(resp)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var tokens int
	var predictedMS float64
	var reasoning strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			if c := chunk.Choices[0].Delta.Content; c != "" {
				chunkFn(backend.ChunkContent, c)
			}
			if r := chunk.Choices[0].Delta.ReasoningContent; r != "" {
				reasoning.WriteString(r)
				chunkFn(backend.ChunkReasoning, r)
			}
		}
		if chunk.Usage.CompletionTokens > 0 {
			tokens = chunk.Usage.CompletionTokens
		}
		if chunk.Timings.PredictedMS > 0 {
			predictedMS = chunk.Timings.PredictedMS
		}
	}

	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil {
			return backend.Response{}, &backend.BackendError{Code: "timeout", Message: "stream interrupted", Cause: ctx.Err()}
		}
		return backend.Response{}, &backend.BackendError{Code: "internal_error", Message: "stream read error", Cause: err}
	}

	var tokPerSec float64
	if predictedMS > 0 && tokens > 0 {
		tokPerSec = float64(tokens) / (predictedMS / 1000)
	}

	return backend.Response{
		Reasoning:       reasoning.String(),
		TokensGenerated: tokens,
		TokPerSecSample: tokPerSec,
	}, nil
}

// ── introspection: /tokenize and /props ───────────────────────────────────────

var introspectClient = &http.Client{Timeout: 10 * time.Second}

// tokenize returns the number of tokens llama-server's tokenizer produces for text.
func tokenize(ctx context.Context, baseURL, text string) (int, error) {
	data, err := json.Marshal(map[string]string{"content": text})
	if err != nil {
		return 0, &backend.BackendError{Code: "internal_error", Message: "marshal tokenize request", Cause: err}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/tokenize", bytes.NewReader(data))
	if err != nil {
		return 0, &backend.BackendError{Code: "internal_error", Message: "build tokenize request", Cause: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := introspectClient.Do(httpReq)
	if err != nil {
		return 0, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, &backend.BackendError{Code: "model_load_failed", Message: fmt.Sprintf("tokenize returned HTTP %d", resp.StatusCode)}
	}

	var tr struct {
		Tokens []int `json:"tokens"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
		return 0, &backend.BackendError{Code: "internal_error", Message: "decode tokenize response", Cause: err}
	}
	return len(tr.Tokens), nil
}

// props returns the model's context size (n_ctx) from llama-server's /props.
func props(ctx context.Context, baseURL string) (int, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/props", nil)
	if err != nil {
		return 0, &backend.BackendError{Code: "internal_error", Message: "build props request", Cause: err}
	}

	resp, err := introspectClient.Do(httpReq)
	if err != nil {
		return 0, &backend.BackendError{Code: "unavailable", Message: "llama-server unreachable", Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, &backend.BackendError{Code: "model_load_failed", Message: fmt.Sprintf("props returned HTTP %d", resp.StatusCode)}
	}

	var pr struct {
		NCtx                      int `json:"n_ctx"`
		DefaultGenerationSettings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		return 0, &backend.BackendError{Code: "internal_error", Message: "decode props response", Cause: err}
	}
	if pr.DefaultGenerationSettings.NCtx > 0 {
		return pr.DefaultGenerationSettings.NCtx, nil
	}
	return pr.NCtx, nil
}
