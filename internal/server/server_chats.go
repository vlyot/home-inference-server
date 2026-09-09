package server

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/ngkaichong/home-inference-server/api"
)

// handleChatsCollection serves GET /v1/chats (list) and POST /v1/chats (create).
func (s *Server) handleChatsCollection(w http.ResponseWriter, r *http.Request) {
	if s.chatStore == nil {
		writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "chat persistence not enabled", "")
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, s.chatStore.List())
	case http.MethodPost:
		var body chatPutBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON body", "")
			return
		}
		s.saveConversation(w, strings.ReplaceAll(uuid.New().String(), "-", ""), body)
	default:
		writeError(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "method not allowed", "")
	}
}

// handleChatItem serves GET/PUT/DELETE /v1/chats/{id}.
func (s *Server) handleChatItem(w http.ResponseWriter, r *http.Request, id string) {
	if s.chatStore == nil {
		writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "chat persistence not enabled", "")
		return
	}
	if id == "" || strings.Contains(id, "/") {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "invalid conversation id", "")
		return
	}

	switch r.Method {
	case http.MethodGet:
		c, ok := s.chatStore.Get(id)
		if !ok {
			writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "conversation not found", "")
			return
		}
		writeJSON(w, http.StatusOK, c)
	case http.MethodPut:
		var body chatPutBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON body", "")
			return
		}
		s.saveConversation(w, id, body)
	case http.MethodDelete:
		if err := s.chatStore.Delete(id); err != nil {
			writeError(w, http.StatusInternalServerError, api.ErrCodeInternal, "delete failed", "")
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		writeError(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "method not allowed", "")
	}
}

type chatPutBody struct {
	Title    string            `json:"title"`
	Messages []api.ChatMessage `json:"messages"`
}

func (s *Server) saveConversation(w http.ResponseWriter, id string, body chatPutBody) {
	meta, err := s.chatStore.Save(api.Conversation{
		ID:       id,
		Title:    body.Title,
		Messages: body.Messages,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, api.ErrCodeInternal, "save failed", "")
		return
	}
	writeJSON(w, http.StatusOK, meta)
}

// defaultNCtx is the context window the server assumes when no model is loaded
// (matches llama-server's --ctx-size). Serving this avoids a cold model spawn
// just to answer /v1/tokenize or /v1/model/props (e.g. on chat-page load).
const defaultNCtx = 4096

// handleTokenize proxies POST /v1/tokenize to the loaded model's tokenizer.
// When no model is loaded it returns a whitespace-word estimate + the default
// n_ctx with model_loaded=false rather than starting a model.
func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	if s.tokenizer == nil {
		writeError(w, http.StatusNotImplemented, api.ErrCodeUnavailable, "tokenizer not available", "")
		return
	}
	var body api.TokenizeRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON body", "")
		return
	}

	n, err := s.tokenizer.Tokenize(r.Context(), body.Text)
	if err != nil {
		writeJSON(w, http.StatusOK, api.TokenizeResponse{
			Tokens: len(strings.Fields(body.Text)), NCtx: defaultNCtx, ModelLoaded: false,
		})
		return
	}
	nctx, err := s.tokenizer.NCtx(r.Context())
	if err != nil {
		nctx = defaultNCtx
	}
	writeJSON(w, http.StatusOK, api.TokenizeResponse{Tokens: n, NCtx: nctx, ModelLoaded: true})
}

// handleModelProps proxies GET /v1/model/props. When no model is loaded it
// returns the default n_ctx with model_loaded=false rather than starting a model.
func (s *Server) handleModelProps(w http.ResponseWriter, r *http.Request) {
	if s.tokenizer == nil {
		writeError(w, http.StatusNotImplemented, api.ErrCodeUnavailable, "model props not available", "")
		return
	}
	nctx, err := s.tokenizer.NCtx(r.Context())
	if err != nil {
		writeJSON(w, http.StatusOK, api.ModelPropsResponse{NCtx: defaultNCtx, ModelLoaded: false})
		return
	}
	writeJSON(w, http.StatusOK, api.ModelPropsResponse{NCtx: nctx, ModelLoaded: true})
}
