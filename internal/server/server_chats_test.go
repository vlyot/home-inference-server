package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/types"
)

// fakeChatStore is an in-memory ChatStore for handler tests.
type fakeChatStore struct {
	convs map[string]api.Conversation
}

func newFakeChatStore() *fakeChatStore {
	return &fakeChatStore{convs: map[string]api.Conversation{}}
}

func (f *fakeChatStore) List() []api.ConversationMeta {
	out := make([]api.ConversationMeta, 0, len(f.convs))
	for _, c := range f.convs {
		out = append(out, api.ConversationMeta{ID: c.ID, Title: c.Title, UpdatedAt: c.UpdatedAt})
	}
	return out
}

func (f *fakeChatStore) Get(id string) (api.Conversation, bool) {
	c, ok := f.convs[id]
	return c, ok
}

func (f *fakeChatStore) Save(c api.Conversation) (api.ConversationMeta, error) {
	if existing, ok := f.convs[c.ID]; ok {
		c.CreatedAt = existing.CreatedAt
	} else {
		c.CreatedAt = time.Now()
	}
	c.UpdatedAt = time.Now()
	if c.Title == "" {
		c.Title = "untitled"
	}
	f.convs[c.ID] = c
	return api.ConversationMeta{ID: c.ID, Title: c.Title, UpdatedAt: c.UpdatedAt}, nil
}

func (f *fakeChatStore) Delete(id string) error {
	delete(f.convs, id)
	return nil
}

// fakeTokenizer implements server.Tokenizer.
type fakeTokenizer struct {
	tokErr error
	nctErr error
}

func (f fakeTokenizer) Tokenize(_ context.Context, text string) (int, error) {
	if f.tokErr != nil {
		return 0, f.tokErr
	}
	return len(text), nil
}

func (f fakeTokenizer) NCtx(_ context.Context) (int, error) {
	if f.nctErr != nil {
		return 0, f.nctErr
	}
	return 4096, nil
}

func bareServer() *server.Server {
	q := queue.New(8)
	return server.New(q, func() types.ServerStatus {
		return types.ServerStatus{Timestamp: time.Now(), AvailableVRAMMB: -1}
	}, "test", time.Now())
}

func doReq(t *testing.T, srv *server.Server, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var r *http.Request
	if body != nil {
		b, _ := json.Marshal(body)
		r = httptest.NewRequest(method, path, bytes.NewReader(b))
	} else {
		r = httptest.NewRequest(method, path, nil)
	}
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	return w
}

func TestChatsListEmpty(t *testing.T) {
	srv := bareServer()
	srv.SetChatStore(newFakeChatStore())

	w := doReq(t, srv, http.MethodGet, api.PathChats, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; want 200", w.Code)
	}
	var list []api.ConversationMeta
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(list) != 0 {
		t.Errorf("list len = %d; want 0", len(list))
	}
}

func TestChatPutCreatesThenListShows(t *testing.T) {
	srv := bareServer()
	srv.SetChatStore(newFakeChatStore())

	w := doReq(t, srv, http.MethodPut, api.PathChats+"/c1", map[string]any{
		"messages": []api.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("PUT status = %d; want 200", w.Code)
	}

	w = doReq(t, srv, http.MethodGet, api.PathChats, nil)
	var list []api.ConversationMeta
	json.Unmarshal(w.Body.Bytes(), &list)
	if len(list) != 1 || list[0].ID != "c1" {
		t.Errorf("list = %+v; want one entry id=c1", list)
	}
}

func TestChatGetReturnsConversation(t *testing.T) {
	srv := bareServer()
	store := newFakeChatStore()
	srv.SetChatStore(store)
	store.Save(api.Conversation{ID: "g1", Messages: []api.ChatMessage{{Role: "user", Content: "yo"}}})

	w := doReq(t, srv, http.MethodGet, api.PathChats+"/g1", nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; want 200", w.Code)
	}
	var c api.Conversation
	json.Unmarshal(w.Body.Bytes(), &c)
	if c.ID != "g1" || len(c.Messages) != 1 {
		t.Errorf("conversation = %+v", c)
	}
}

func TestChatGetUnknown404(t *testing.T) {
	srv := bareServer()
	srv.SetChatStore(newFakeChatStore())

	w := doReq(t, srv, http.MethodGet, api.PathChats+"/missing", nil)
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d; want 404", w.Code)
	}
}

func TestChatPutUpdatesExisting(t *testing.T) {
	srv := bareServer()
	store := newFakeChatStore()
	srv.SetChatStore(store)

	doReq(t, srv, http.MethodPut, api.PathChats+"/u1", map[string]any{
		"messages": []api.ChatMessage{{Role: "user", Content: "one"}},
	})
	w := doReq(t, srv, http.MethodPut, api.PathChats+"/u1", map[string]any{
		"messages": []api.ChatMessage{
			{Role: "user", Content: "one"},
			{Role: "assistant", Content: "two"},
		},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("second PUT status = %d; want 200", w.Code)
	}
	c, _ := store.Get("u1")
	if len(c.Messages) != 2 {
		t.Errorf("messages len = %d; want 2 after update", len(c.Messages))
	}
}

func TestChatDelete204ThenGet404(t *testing.T) {
	srv := bareServer()
	store := newFakeChatStore()
	srv.SetChatStore(store)
	store.Save(api.Conversation{ID: "d1"})

	w := doReq(t, srv, http.MethodDelete, api.PathChats+"/d1", nil)
	if w.Code != http.StatusNoContent {
		t.Fatalf("DELETE status = %d; want 204", w.Code)
	}
	w = doReq(t, srv, http.MethodGet, api.PathChats+"/d1", nil)
	if w.Code != http.StatusNotFound {
		t.Errorf("GET after DELETE = %d; want 404", w.Code)
	}
}

func TestChatEndpointsNilStore404(t *testing.T) {
	srv := bareServer() // no SetChatStore

	for _, tc := range []struct {
		method, path string
	}{
		{http.MethodGet, api.PathChats},
		{http.MethodGet, api.PathChats + "/x"},
		{http.MethodPut, api.PathChats + "/x"},
		{http.MethodDelete, api.PathChats + "/x"},
	} {
		w := doReq(t, srv, tc.method, tc.path, nil)
		if w.Code != http.StatusNotFound {
			t.Errorf("%s %s = %d; want 404 with nil store", tc.method, tc.path, w.Code)
		}
	}
}

func TestTokenizeProxyReturnsCountAndNCtx(t *testing.T) {
	srv := bareServer()
	srv.SetTokenizer(fakeTokenizer{})

	w := doReq(t, srv, http.MethodPost, api.PathTokenize, api.TokenizeRequest{Text: "abcde"})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; want 200", w.Code)
	}
	var resp api.TokenizeResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Tokens != 5 || resp.NCtx != 4096 {
		t.Errorf("resp = %+v; want tokens=5 n_ctx=4096", resp)
	}
}

func TestModelPropsProxyReturnsNCtx(t *testing.T) {
	srv := bareServer()
	srv.SetTokenizer(fakeTokenizer{})

	w := doReq(t, srv, http.MethodGet, api.PathModelProps, nil)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; want 200", w.Code)
	}
	var resp api.ModelPropsResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.NCtx != 4096 {
		t.Errorf("n_ctx = %d; want 4096", resp.NCtx)
	}
}

func TestTokenizeProxyNilTokenizer501(t *testing.T) {
	srv := bareServer() // no SetTokenizer

	w := doReq(t, srv, http.MethodPost, api.PathTokenize, api.TokenizeRequest{Text: "x"})
	if w.Code != http.StatusNotImplemented {
		t.Errorf("POST /v1/tokenize = %d; want 501", w.Code)
	}
	w = doReq(t, srv, http.MethodGet, api.PathModelProps, nil)
	if w.Code != http.StatusNotImplemented {
		t.Errorf("GET /v1/model/props = %d; want 501", w.Code)
	}
}

func TestTokenizeProxyNoModelReturnsCheapEstimate(t *testing.T) {
	// When the tokenizer can't answer (no model loaded), the server serves a
	// whitespace-word estimate + default n_ctx rather than 503 / a cold spawn.
	srv := bareServer()
	srv.SetTokenizer(fakeTokenizer{tokErr: errors.New("no model is loaded")})

	w := doReq(t, srv, http.MethodPost, api.PathTokenize, api.TokenizeRequest{Text: "one two three"})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d; want 200", w.Code)
	}
	var tr api.TokenizeResponse
	json.Unmarshal(w.Body.Bytes(), &tr)
	if tr.ModelLoaded {
		t.Errorf("model_loaded = true; want false")
	}
	if tr.Tokens != 3 || tr.NCtx != 4096 {
		t.Errorf("tokens=%d n_ctx=%d; want 3, 4096", tr.Tokens, tr.NCtx)
	}
}

// --- TranslateInferRequest multi-turn behaviour ---

func TestTranslateInferRequestCarriesAllMessages(t *testing.T) {
	req := api.InferRequest{
		Modality: api.ModalityText,
		TextInput: &api.TextInput{Messages: []api.ChatMessage{
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "b"},
			{Role: "user", Content: "c"},
		}},
	}
	got := server.TranslateInferRequest(req, "rid")
	if len(got.Messages) != 3 {
		t.Fatalf("messages len = %d; want 3", len(got.Messages))
	}
	if got.Messages[0].Content != "a" || got.Messages[2].Content != "c" {
		t.Errorf("order not preserved: %+v", got.Messages)
	}
	if got.Prompt != "" {
		t.Errorf("Prompt = %q; want empty when messages present", got.Prompt)
	}
}

func TestTranslateInferRequestPrependsSystemPrompt(t *testing.T) {
	req := api.InferRequest{
		Modality:     api.ModalityText,
		SystemPrompt: "be terse",
		TextInput: &api.TextInput{Messages: []api.ChatMessage{
			{Role: "user", Content: "hi"},
		}},
	}
	got := server.TranslateInferRequest(req, "rid")
	if len(got.Messages) != 2 || got.Messages[0].Role != "system" || got.Messages[0].Content != "be terse" {
		t.Errorf("system prompt not prepended: %+v", got.Messages)
	}
}

func TestTranslateInferRequestSkipsSystemPrependWhenPresent(t *testing.T) {
	req := api.InferRequest{
		Modality:     api.ModalityText,
		SystemPrompt: "ignored",
		TextInput: &api.TextInput{Messages: []api.ChatMessage{
			{Role: "system", Content: "already here"},
			{Role: "user", Content: "hi"},
		}},
	}
	got := server.TranslateInferRequest(req, "rid")
	if len(got.Messages) != 2 || got.Messages[0].Content != "already here" {
		t.Errorf("existing system message not respected: %+v", got.Messages)
	}
}
