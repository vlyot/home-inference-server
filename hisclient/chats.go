package hisclient

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/ngkaichong/home-inference-server/api"
)

// chatBody is the create/update payload accepted by POST/PUT /v1/chats*.
type chatBody struct {
	Title    string            `json:"title"`
	Messages []api.ChatMessage `json:"messages"`
}

// ListChats fetches GET /v1/chats.
func (c *Client) ListChats(ctx context.Context) ([]api.ConversationMeta, error) {
	req, err := c.newRequest(ctx, http.MethodGet, api.PathChats, nil)
	if err != nil {
		return nil, err
	}
	var out []api.ConversationMeta
	if _, err := c.doJSON(req, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// CreateChat POSTs a new conversation to /v1/chats and returns its metadata
// (the server assigns the id).
func (c *Client) CreateChat(ctx context.Context, title string, messages []api.ChatMessage) (*api.ConversationMeta, error) {
	body, _ := json.Marshal(chatBody{Title: title, Messages: messages})
	req, err := c.newRequest(ctx, http.MethodPost, api.PathChats, body)
	if err != nil {
		return nil, err
	}
	var meta api.ConversationMeta
	if _, err := c.doJSON(req, &meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// GetChat fetches GET /v1/chats/{id}.
func (c *Client) GetChat(ctx context.Context, id string) (*api.Conversation, error) {
	req, err := c.newRequest(ctx, http.MethodGet, api.PathChats+"/"+id, nil)
	if err != nil {
		return nil, err
	}
	var conv api.Conversation
	if _, err := c.doJSON(req, &conv); err != nil {
		return nil, err
	}
	return &conv, nil
}

// UpdateChat replaces the conversation at /v1/chats/{id}.
func (c *Client) UpdateChat(ctx context.Context, id, title string, messages []api.ChatMessage) (*api.ConversationMeta, error) {
	body, _ := json.Marshal(chatBody{Title: title, Messages: messages})
	req, err := c.newRequest(ctx, http.MethodPut, api.PathChats+"/"+id, body)
	if err != nil {
		return nil, err
	}
	var meta api.ConversationMeta
	if _, err := c.doJSON(req, &meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// DeleteChat removes the conversation at /v1/chats/{id}. A missing id is not an
// error (the server returns 204 either way).
func (c *Client) DeleteChat(ctx context.Context, id string) error {
	req, err := c.newRequest(ctx, http.MethodDelete, api.PathChats+"/"+id, nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return c.errorFrom(resp)
	}
	return nil
}
