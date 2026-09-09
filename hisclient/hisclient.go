// Package hisclient is a small, dependency-free Go client for the Home
// Inference Server's local HTTP API (the /v1/* endpoints documented at GET
// /docs). It is the reference SDK: future apps can import it instead of
// hand-rolling requests, and the conformance suite in internal/testapp uses it
// to prove the server honours its documented contract.
package hisclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

// Client talks to one Home Inference Server instance.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey sends X-API-Key on every request (needed when the server runs with
// HIS_API_KEY set and the client is not on loopback).
func WithAPIKey(key string) Option { return func(c *Client) { c.apiKey = key } }

// WithHTTPClient supplies a custom *http.Client (timeouts, transport, etc.).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

// New builds a Client for the server at baseURL (e.g. http://localhost:8080).
func New(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// BaseURL is the server root this client targets, without a trailing slash.
func (c *Client) BaseURL() string { return c.baseURL }

// RawGet issues a bare GET and returns the status code and body. Used by the
// conformance suite to assert wire-level behaviour (e.g. 405 on a wrong method).
func (c *Client) RawGet(ctx context.Context, path string) (int, string, error) {
	req, err := c.newRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return 0, "", err
	}
	return c.raw(req)
}

// RawPost issues a bare POST with a raw (possibly invalid) JSON body and returns
// the status code and body.
func (c *Client) RawPost(ctx context.Context, path, body string) (int, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, strings.NewReader(body))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set(api.HeaderAPIKey, c.apiKey)
	}
	return c.raw(req)
}

func (c *Client) raw(req *http.Request) (int, string, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	return resp.StatusCode, string(b), nil
}

// APIError is returned when the server responds with a 4xx/5xx and an
// api.ErrorResponse body, or when a stream ends with an error chunk.
type APIError struct {
	HTTPStatus int
	Code       string
	Message    string
}

func (e *APIError) Error() string {
	if e.HTTPStatus > 0 {
		return fmt.Sprintf("his: %d %s: %s", e.HTTPStatus, e.Code, e.Message)
	}
	return fmt.Sprintf("his: %s: %s", e.Code, e.Message)
}

// AsAPIError extracts an *APIError from err, if present.
func AsAPIError(err error) (*APIError, bool) {
	if e, ok := err.(*APIError); ok {
		return e, true
	}
	return nil, false
}

func (c *Client) newRequest(ctx context.Context, method, path string, body []byte) (*http.Request, error) {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, r)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set(api.HeaderAPIKey, c.apiKey)
	}
	return req, nil
}

// doJSON sends a request and decodes a 2xx JSON body into out (may be nil).
// A non-2xx response is turned into an *APIError.
func (c *Client) doJSON(req *http.Request, out any) (http.Header, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.Header, c.errorFrom(resp)
	}
	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return resp.Header, fmt.Errorf("his: decode %s: %w", req.URL.Path, err)
		}
	}
	return resp.Header, nil
}

func (c *Client) errorFrom(resp *http.Response) error {
	var er api.ErrorResponse
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	_ = json.Unmarshal(raw, &er)
	if er.Code == "" {
		er.Code = "http_" + strconv.Itoa(resp.StatusCode)
		er.Message = strings.TrimSpace(string(raw))
	}
	return &APIError{HTTPStatus: resp.StatusCode, Code: er.Code, Message: er.Message}
}

// InferResult bundles the response body with the response headers so callers
// can read X-Quality-Degraded.
type InferResult struct {
	api.InferResponse
	QualityDegraded bool
}

// Infer performs a blocking POST /v1/infer.
func (c *Client) Infer(ctx context.Context, in api.InferRequest) (*InferResult, error) {
	body, _ := json.Marshal(in)
	req, err := c.newRequest(ctx, http.MethodPost, api.PathInfer, body)
	if err != nil {
		return nil, err
	}
	var ir api.InferResponse
	hdr, err := c.doJSON(req, &ir)
	if err != nil {
		return nil, err
	}
	return &InferResult{InferResponse: ir, QualityDegraded: hdr.Get(api.HeaderQualityDegraded) == "true"}, nil
}

// StreamHandler receives each SSE chunk as it arrives. Return a non-nil error to
// stop consuming early.
type StreamHandler func(api.StreamChunk) error

// InferStream performs a streaming POST /v1/infer. It parses the SSE frames,
// invokes onChunk for each, and returns the final (done) chunk. It tolerates
// both terminators the server may send: a chunk with done=true and a literal
// "data: [DONE]" line. A done chunk carrying an error field is surfaced as an
// *APIError.
func (c *Client) InferStream(ctx context.Context, in api.InferRequest, onChunk StreamHandler) (*api.StreamChunk, error) {
	in.Stream = true
	body, _ := json.Marshal(in)
	req, err := c.newRequest(ctx, http.MethodPost, api.PathInfer, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusAccepted {
		// Deferred: the server returns a plain JSON body, not a stream.
		var got map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&got)
		return nil, &APIError{HTTPStatus: 202, Code: "deferred", Message: fmt.Sprintf("%v", got["result_url"])}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, c.errorFrom(resp)
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	var final *api.StreamChunk
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk api.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if onChunk != nil {
			if err := onChunk(chunk); err != nil {
				return &chunk, err
			}
		}
		if chunk.Done {
			c := chunk
			final = &c
			break
		}
	}
	if err := sc.Err(); err != nil {
		return final, fmt.Errorf("his: stream read: %w", err)
	}
	if final != nil && final.Error != "" {
		return final, &APIError{Code: final.Error, Message: "stream ended with an error"}
	}
	return final, nil
}

// Status fetches GET /v1/status.
func (c *Client) Status(ctx context.Context) (map[string]any, error) {
	return c.getMap(ctx, api.PathStatus)
}

// Pressure fetches GET /v1/status/pressure.
func (c *Client) Pressure(ctx context.Context) (*api.PressureSnapshot, error) {
	req, err := c.newRequest(ctx, http.MethodGet, api.PathStatusPressure, nil)
	if err != nil {
		return nil, err
	}
	var p api.PressureSnapshot
	if _, err := c.doJSON(req, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// Health fetches GET /healthz.
func (c *Client) Health(ctx context.Context) (*api.HealthResponse, error) {
	req, err := c.newRequest(ctx, http.MethodGet, api.PathHealth, nil)
	if err != nil {
		return nil, err
	}
	var h api.HealthResponse
	if _, err := c.doJSON(req, &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// Logs fetches GET /v1/logs. Empty level/event/since disable that filter.
func (c *Client) Logs(ctx context.Context, level, event, since string, limit int) (map[string]any, error) {
	q := url.Values{}
	if level != "" {
		q.Set("level", level)
	}
	if event != "" {
		q.Set("event", event)
	}
	if since != "" {
		q.Set("since", since)
	}
	if limit > 0 {
		q.Set("limit", strconv.Itoa(limit))
	}
	return c.getMap(ctx, api.PathLogs+"?"+q.Encode())
}

// Tokenize calls POST /v1/tokenize.
func (c *Client) Tokenize(ctx context.Context, text string) (*api.TokenizeResponse, error) {
	body, _ := json.Marshal(api.TokenizeRequest{Text: text})
	req, err := c.newRequest(ctx, http.MethodPost, api.PathTokenize, body)
	if err != nil {
		return nil, err
	}
	var tr api.TokenizeResponse
	if _, err := c.doJSON(req, &tr); err != nil {
		return nil, err
	}
	return &tr, nil
}

// ModelProps calls GET /v1/model/props.
func (c *Client) ModelProps(ctx context.Context) (*api.ModelPropsResponse, error) {
	req, err := c.newRequest(ctx, http.MethodGet, api.PathModelProps, nil)
	if err != nil {
		return nil, err
	}
	var mp api.ModelPropsResponse
	if _, err := c.doJSON(req, &mp); err != nil {
		return nil, err
	}
	return &mp, nil
}

func (c *Client) getMap(ctx context.Context, path string) (map[string]any, error) {
	req, err := c.newRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	if _, err := c.doJSON(req, &out); err != nil {
		return nil, err
	}
	return out, nil
}
