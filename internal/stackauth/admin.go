package stackauth

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// adminBaseURL is Stack Auth's server-access-type REST API. The docs domain
// (docs.stack-auth.com) now redirects to hexclave.com — Hexclave is the
// umbrella company, Stack Auth the product — but this REST host and the
// X-Stack-* headers remain the documented, working API.
const adminBaseURL = "https://api.hexclave.com/api/v1"

// AdminClient calls Stack Auth's server-access-type REST API to provision
// accounts directly. Distinct from Verifier, which only validates tokens
// callers present — this makes outbound calls authenticated with the
// project's secret server key.
type AdminClient struct {
	projectID string
	secretKey string
	http      *http.Client
}

// NewAdminClient returns an AdminClient for the given Stack Auth project,
// authenticated with its secret server key.
func NewAdminClient(projectID, secretKey string) *AdminClient {
	return &AdminClient{
		projectID: projectID,
		secretKey: secretKey,
		http:      &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *AdminClient) do(ctx context.Context, method, path string, body any, out any) error {
	var reqBody io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, adminBaseURL+path, reqBody)
	if err != nil {
		return err
	}
	req.Header.Set("X-Stack-Access-Type", "server")
	req.Header.Set("X-Stack-Project-Id", c.projectID)
	req.Header.Set("X-Stack-Secret-Server-Key", c.secretKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("stackauth: %s %s: %d: %s", method, path, resp.StatusCode, snippet)
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// FindUserByEmail looks up an existing account by exact primary_email
// (case-insensitive). Returns "", nil if no match — this is not an error,
// just "doesn't exist yet".
//
// GET /users?query=<email> is free-text search (also matches display name
// and an exact-match user id) — the query param is not a precise email
// filter, so this filters the returned items for an exact,
// case-insensitive primary_email match itself rather than trusting search
// relevance.
func (c *AdminClient) FindUserByEmail(ctx context.Context, email string) (string, error) {
	var out struct {
		Items []struct {
			ID           string `json:"id"`
			PrimaryEmail string `json:"primary_email"`
		} `json:"items"`
	}
	path := "/users?query=" + url.QueryEscape(email)
	if err := c.do(ctx, http.MethodGet, path, nil, &out); err != nil {
		return "", err
	}
	for _, item := range out.Items {
		if strings.EqualFold(item.PrimaryEmail, email) {
			return item.ID, nil
		}
	}
	return "", nil
}

// CreateUser provisions a new Stack Auth account directly — no sign-up flow
// involved. Returns the new account's Stack Auth user id.
func (c *AdminClient) CreateUser(ctx context.Context, email, displayName, password string) (string, error) {
	body := map[string]any{
		"primary_email":              email,
		"primary_email_auth_enabled": true,
		"password":                   password,
	}
	if displayName != "" {
		body["display_name"] = displayName
	}
	var out struct {
		ID string `json:"id"`
	}
	if err := c.do(ctx, http.MethodPost, "/users", body, &out); err != nil {
		return "", err
	}
	return out.ID, nil
}
