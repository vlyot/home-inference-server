// Package relayenqueue is a minimal client for the hosted relay's POST /enqueue
// endpoint. The local inference server uses it to hand off a request it cannot
// serve right now (VRAM/CPU pressure) to the durable Railway queue instead of
// holding it in memory — so a deferred request survives a restart or crash and
// is picked back up by the normal internal/remote /claim worker loop.
package relayenqueue

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

// Client posts jobs to the relay's /enqueue endpoint with the shared API key.
type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

// New builds a Client. baseURL is the relay root (e.g.
// https://queue-api-production-xxxx.up.railway.app); apiKey is the relay's
// QUEUE_API_KEY. A nil return is never produced — callers gate on baseURL != "".
func New(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 15 * time.Second},
	}
}

// BaseURL is the relay root this client posts to, without a trailing slash.
func (c *Client) BaseURL() string { return c.baseURL }

// Enqueue posts one job and returns the relay-assigned job id. The relay scopes
// result retrieval to the identity it resolves from the credential: an X-API-Key
// request (which this client always sends) is identity "apikey", so a caller
// retrieves the result from the relay's GET /result/{id} using the same relay
// API key. An error means the relay was unreachable or rejected the job; the
// caller should surface it as 503.
func (c *Client) Enqueue(ctx context.Context, correlationID string, payload json.RawMessage) (string, error) {
	body, err := json.Marshal(queue.EnqueueRequest{
		CorrelationID: correlationID,
		Payload:       payload,
	})
	if err != nil {
		return "", fmt.Errorf("marshal enqueue request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/enqueue", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(api.HeaderAPIKey, c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("relay /enqueue unreachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return "", fmt.Errorf("relay /enqueue returned %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}

	var out queue.EnqueueResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode enqueue response: %w", err)
	}
	if out.ID == "" {
		return "", fmt.Errorf("relay /enqueue returned an empty id")
	}
	return out.ID, nil
}

// ResultURL is the relay endpoint a caller polls for a deferred job's result.
func (c *Client) ResultURL(jobID string) string {
	return c.baseURL + "/result/" + jobID
}
