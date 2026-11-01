// Package hosted answers public-app questions with a hosted model while the
// PC is offline. It speaks OpenRouter's chat completions API and makes one
// attempt per question: the relay turns any failure into "unavailable", and
// the client app falls back to its own offline search.
package hosted

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ngkaichong/home-inference-server/api"
)

// DefaultModel is OpenRouter's free Nemotron 3 Ultra. It held every rule in
// a prompt-injection check against the portfolio's real prompt, and it isn't
// served from the shared Google pool that kept the free Gemma 4 models
// rate-limited.
const DefaultModel = "nvidia/nemotron-3-ultra-550b-a55b:free"

const (
	endpoint      = "https://openrouter.ai/api/v1/chat/completions"
	maxReplyBytes = 256 * 1024
)

// ErrEmpty means the model replied without any text.
var ErrEmpty = errors.New("hosted: empty reply")

// Client calls one hosted model.
type Client struct {
	key   string
	model string
	url   string
	http  *http.Client
}

// New returns a client for model, or nil when key is empty: no key means no
// hosted fallback.
func New(key, model string) *Client {
	if key == "" {
		return nil
	}
	return &Client{key: key, model: model, url: endpoint, http: &http.Client{}}
}

// Model is the model the client calls.
func (c *Client) Model() string { return c.model }

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float32   `json:"temperature"`
	Reasoning   reasoning `json:"reasoning"`
}

// reasoning is off: the relay only ever shows the final answer, and a
// reasoning trace is where safety-tuned models leak what they refuse.
type reasoning struct {
	Enabled bool `json:"enabled"`
}

type completionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Code int `json:"code"`
	} `json:"error"`
}

// Answer sends a request built by apps.BuildRequest (system prompt plus the
// framed question) and returns the model's reply. Errors carry only status
// codes, never the question. ctx bounds the whole call.
func (c *Client) Answer(ctx context.Context, req api.InferRequest) (string, error) {
	prompt := ""
	if req.TextInput != nil {
		prompt = req.TextInput.Prompt
	}
	body, err := json.Marshal(completionRequest{
		Model: c.model,
		Messages: []message{
			{Role: "system", Content: req.SystemPrompt},
			{Role: "user", Content: prompt},
		},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		return "", err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.key)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := c.http.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", errors.New("hosted: request failed")
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return "", fmt.Errorf("hosted: status %d", res.StatusCode)
	}

	var out completionResponse
	if err := json.NewDecoder(io.LimitReader(res.Body, maxReplyBytes)).Decode(&out); err != nil {
		return "", errors.New("hosted: unreadable reply")
	}
	// OpenRouter can report an upstream failure inside a 200.
	if out.Error != nil {
		return "", fmt.Errorf("hosted: upstream error %d", out.Error.Code)
	}
	if len(out.Choices) == 0 || strings.TrimSpace(out.Choices[0].Message.Content) == "" {
		return "", ErrEmpty
	}
	return out.Choices[0].Message.Content, nil
}
