package hisclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
)

func TestInferDecodesResponseAndHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(api.HeaderAPIKey) != "k" {
			t.Errorf("missing api key header")
		}
		w.Header().Set(api.HeaderQualityDegraded, "true")
		_ = json.NewEncoder(w).Encode(api.InferResponse{Output: "hi", RequestID: "r1", ModelTier: "weak"})
	}))
	defer srv.Close()

	c := New(srv.URL, WithAPIKey("k"))
	res, err := c.Infer(context.Background(), api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: "x"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "hi" || !res.QualityDegraded {
		t.Fatalf("res = %+v", res)
	}
}

func TestInferMapsErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(api.ErrorResponse{Code: api.ErrCodeInvalidModality, Message: "nope"})
	}))
	defer srv.Close()

	c := New(srv.URL)
	_, err := c.Infer(context.Background(), api.InferRequest{Modality: "weird"})
	ae, ok := AsAPIError(err)
	if !ok {
		t.Fatalf("want APIError, got %v", err)
	}
	if ae.HTTPStatus != 400 || ae.Code != api.ErrCodeInvalidModality {
		t.Fatalf("ae = %+v", ae)
	}
}

func sse(w http.ResponseWriter, lines ...string) {
	f := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	for _, l := range lines {
		fmt.Fprintf(w, "data: %s\n\n", l)
		f.Flush()
	}
}

func TestInferStreamParsesDeltaReasoningAndDoneChunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mk := func(v api.StreamChunk) string { b, _ := json.Marshal(v); return string(b) }
		sse(w,
			mk(api.StreamChunk{Reasoning: "think..."}),
			mk(api.StreamChunk{Delta: "Hel"}),
			mk(api.StreamChunk{Delta: "lo"}),
			mk(api.StreamChunk{Done: true, TokensGenerated: 2, ModelTier: "weak"}),
		)
	}))
	defer srv.Close()

	var content, reasoning strings.Builder
	c := New(srv.URL)
	final, err := c.InferStream(context.Background(), api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: "x"}},
		func(ch api.StreamChunk) error {
			content.WriteString(ch.Delta)
			reasoning.WriteString(ch.Reasoning)
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if content.String() != "Hello" || reasoning.String() != "think..." {
		t.Fatalf("content=%q reasoning=%q", content.String(), reasoning.String())
	}
	if final == nil || !final.Done || final.TokensGenerated != 2 {
		t.Fatalf("final = %+v", final)
	}
}

func TestInferStreamHandlesDoneSentinelLine(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mk := func(v api.StreamChunk) string { b, _ := json.Marshal(v); return string(b) }
		sse(w,
			mk(api.StreamChunk{Delta: "hi"}),
			mk(api.StreamChunk{Done: true}),
			"[DONE]",
		)
	}))
	defer srv.Close()

	c := New(srv.URL)
	final, err := c.InferStream(context.Background(), api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: "x"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if final == nil || !final.Done {
		t.Fatalf("final = %+v", final)
	}
}

func TestInferStreamSurfacesErrorChunk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mk := func(v api.StreamChunk) string { b, _ := json.Marshal(v); return string(b) }
		sse(w, mk(api.StreamChunk{Done: true, Error: api.ErrCodeReasoningExhausted}))
	}))
	defer srv.Close()

	c := New(srv.URL)
	_, err := c.InferStream(context.Background(), api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: "x"}}, nil)
	ae, ok := AsAPIError(err)
	if !ok || ae.Code != api.ErrCodeReasoningExhausted {
		t.Fatalf("want reasoning_exhausted APIError, got %v", err)
	}
}

func TestInferStreamReportsDeferred202(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]any{"deferred": true, "result_url": "https://relay/x"})
	}))
	defer srv.Close()

	c := New(srv.URL)
	_, err := c.InferStream(context.Background(), api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: "x"}}, nil)
	ae, ok := AsAPIError(err)
	if !ok || ae.HTTPStatus != 202 || ae.Code != "deferred" {
		t.Fatalf("want deferred 202, got %v", err)
	}
}
