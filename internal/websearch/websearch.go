// Package websearch is an HTTP client for Tavily's keyless search API, used
// to resolve the server's built-in "web_search" tool. Keyless access
// (X-Tavily-Access-Mode: keyless) requires no account, API key, or card —
// see https://docs.tavily.com/documentation/keyless. Free-tier/keyless
// traffic is rate-limited by Tavily; a sustained high-volume deployment
// should get a free API key (docs.tavily.com) and pass it as a Bearer token
// instead, but that is not needed for this server's expected call volume.
package websearch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
)

// tavilySearchURL is Tavily's search endpoint. Self-hosting/on-prem is not a
// concern here — this is a hosted API, not something this project deploys.
// Var (not const) so tests can point it at an httptest server.
var tavilySearchURL = "https://api.tavily.com/search"

// maxResults caps how many results Search returns, keeping the tool-response
// payload handed back to the model small.
const maxResults = 5

// Result is one search hit.
type Result struct {
	Title   string
	URL     string
	Content string
}

// Client queries Tavily's search API.
type Client struct {
	http *http.Client
}

// NewClient returns a Client using keyless access — no API key required.
// Mirrors the timeout-scoped client pattern already used for llama-server
// calls (internal/backend/llamacpp/client.go).
func NewClient(timeout time.Duration) *Client {
	return &Client{http: &http.Client{Timeout: timeout}}
}

// retryWait is how long Search waits before its one retry on a
// connection-level failure.
const retryWait = 1500 * time.Millisecond

// errConnFailed marks a connection-level failure (Tavily unreachable) as
// distinct from a non-2xx response or a decode error — both of the latter
// mean Tavily answered but something about the request/response itself is
// wrong, so retrying wouldn't help. Wrapped inside the *backend.BackendError
// Search/doSearch return (still Code "tool_unavailable" either way — this is
// an internal signal for the retry decision, not part of the public error
// contract).
var errConnFailed = errors.New("search backend unreachable")

// Search queries Tavily and returns up to maxResults results. Returns a
// *backend.BackendError with code "tool_unavailable" on any failure to reach
// or parse a response from the search backend, and "timeout" if ctx expires —
// the caller (the tool-calling orchestration loop) treats both as "search
// failed, tell the model" rather than failing the whole request.
//
// Retries once, after retryWait, on a connection-level failure only.
func (c *Client) Search(ctx context.Context, query string) ([]Result, error) {
	results, err := c.doSearch(ctx, query)
	if err == nil || !errors.Is(err, errConnFailed) {
		return results, err
	}

	select {
	case <-time.After(retryWait):
	case <-ctx.Done():
		return nil, &backend.BackendError{Code: "timeout", Message: "search request cancelled or timed out", Cause: ctx.Err()}
	}

	return c.doSearch(ctx, query)
}

func (c *Client) doSearch(ctx context.Context, query string) ([]Result, error) {
	body, err := json.Marshal(struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}{Query: query, MaxResults: maxResults})
	if err != nil {
		return nil, &backend.BackendError{Code: "internal_error", Message: "build search request", Cause: err}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tavilySearchURL, bytes.NewReader(body))
	if err != nil {
		return nil, &backend.BackendError{Code: "internal_error", Message: "build search request", Cause: err}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Tavily-Access-Mode", "keyless")

	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, &backend.BackendError{Code: "timeout", Message: "search request cancelled or timed out", Cause: ctx.Err()}
		}
		return nil, &backend.BackendError{Code: "tool_unavailable", Message: "search backend unreachable", Cause: errConnFailed}
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, rateLimitError(resp.Body)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, &backend.BackendError{
			Code:    "tool_unavailable",
			Message: fmt.Sprintf("search backend returned HTTP %d", resp.StatusCode),
		}
	}

	var sr struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, &backend.BackendError{Code: "tool_unavailable", Message: "decode search response", Cause: err}
	}

	n := len(sr.Results)
	if n > maxResults {
		n = maxResults
	}
	out := make([]Result, n)
	for i := 0; i < n; i++ {
		out[i] = Result{Title: sr.Results[i].Title, URL: sr.Results[i].URL, Content: sr.Results[i].Content}
	}
	return out, nil
}

// rateLimitError builds the BackendError for a 429 response, reading body
// (capped, since this is an error path with no legitimate reason to be
// large) and trying two known Tavily shapes in turn:
//
//  1. The keyless-specific envelope confirmed straight from Tavily's own
//     Python SDK source (tavily/tavily.py, _raise_keyless_envelope) —
//     {"error": {"code", "message", "window", "retry_after_seconds", ...}}.
//     This is the shape actually returned to keyless callers; "window" and
//     "retry_after_seconds" are server-filled and not documented anywhere
//     other than this SDK source, so surfacing them here is the only way a
//     caller (or the model, via the tool-result content) learns how long
//     the throttle lasts.
//  2. The general rate-limits page's documented shape,
//     {"error": "<plain string>"} — used for authenticated (non-keyless)
//     429s, and kept as a fallback in case Tavily ever returns it to a
//     keyless caller too.
//
// Falls back to a generic message if neither parses — a malformed error
// body shouldn't produce a worse error than a well-formed one.
func rateLimitError(body io.Reader) *backend.BackendError {
	data, _ := io.ReadAll(io.LimitReader(body, 4096))

	var envelope struct {
		Error struct {
			Code              string `json:"code"`
			Message           string `json:"message"`
			Window            string `json:"window"`
			RetryAfterSeconds *int   `json:"retry_after_seconds"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &envelope); err == nil && envelope.Error.Code != "" {
		msg := envelope.Error.Message
		if msg == "" {
			msg = "rate limited"
		}
		if envelope.Error.Window != "" {
			msg += fmt.Sprintf(" (limit resets every %s)", envelope.Error.Window)
		}
		if envelope.Error.RetryAfterSeconds != nil {
			msg += fmt.Sprintf("; retry after %ds", *envelope.Error.RetryAfterSeconds)
		}
		return &backend.BackendError{Code: "tool_rate_limited", Message: msg}
	}

	var flat struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &flat); err == nil && flat.Error != "" {
		return &backend.BackendError{Code: "tool_rate_limited", Message: flat.Error}
	}

	return &backend.BackendError{Code: "tool_rate_limited", Message: "rate limited (keyless access is capped — consider a free Tavily API key)"}
}
