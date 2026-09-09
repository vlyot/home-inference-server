package hisclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/queue"
)

// RelayClient talks to the hosted relay (the Railway queue-api service), which
// F&F apps and CLI tools use when they are not on the PC's LAN. Auth is the
// static X-API-Key.
type RelayClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// NewRelay builds a RelayClient for baseURL (the relay root URL) with the relay
// API key.
func NewRelay(baseURL, apiKey string) *RelayClient {
	return &RelayClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *RelayClient) do(ctx context.Context, method, path string, body []byte, out any) error {
	var r io.Reader
	if body != nil {
		r = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, r)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set(api.HeaderAPIKey, c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var er api.ErrorResponse
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		_ = json.Unmarshal(raw, &er)
		if er.Code == "" {
			er.Message = strings.TrimSpace(string(raw))
		}
		return &APIError{HTTPStatus: resp.StatusCode, Code: er.Code, Message: er.Message}
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}

// Enqueue posts an inference request to the relay and returns the job id.
func (c *RelayClient) Enqueue(ctx context.Context, correlationID string, payload api.InferRequest) (string, error) {
	pb, _ := json.Marshal(payload)
	body, _ := json.Marshal(queue.EnqueueRequest{CorrelationID: correlationID, Payload: pb})
	var out queue.EnqueueResponse
	if err := c.do(ctx, http.MethodPost, "/enqueue", body, &out); err != nil {
		return "", err
	}
	if out.ID == "" {
		return "", fmt.Errorf("his: relay returned an empty job id")
	}
	return out.ID, nil
}

// Result fetches GET /result/{id} on the relay.
func (c *RelayClient) Result(ctx context.Context, jobID string) (*queue.ResultResponse, error) {
	var out queue.ResultResponse
	if err := c.do(ctx, http.MethodGet, "/result/"+jobID, nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Status fetches GET /status on the relay.
func (c *RelayClient) Status(ctx context.Context) (*queue.QueueStatusResponse, error) {
	var out queue.QueueStatusResponse
	if err := c.do(ctx, http.MethodGet, "/status", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
