package hosted

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

const secretQuestion = "<question>\nWhat is Kai's secret project?\n</question>\n\nAnswer:"

var testRequest = api.InferRequest{
	SystemPrompt: "You answer questions about Kai.",
	TextInput:    &api.TextInput{Prompt: secretQuestion},
	MaxTokens:    400,
	Temperature:  0.2,
}

func testClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	c := New("sk-or-test", DefaultModel)
	c.url = srv.URL
	return c
}

func reply(content string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": content}}},
		})
	}
}

func TestAnswerSendsTheBuiltRequest(t *testing.T) {
	var got completionRequest
	var auth string
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		json.NewDecoder(r.Body).Decode(&got) //nolint:errcheck
		reply("Kai built GrabMaps [1].")(w, r)
	})
	if _, err := c.Answer(context.Background(), testRequest); err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer sk-or-test" {
		t.Fatalf("Authorization = %q", auth)
	}
	if got.Model != DefaultModel || got.MaxTokens != 400 || got.Temperature != 0.2 || got.Reasoning.Enabled {
		t.Fatalf("request = %+v", got)
	}
	want := []message{{"system", testRequest.SystemPrompt}, {"user", secretQuestion}}
	if len(got.Messages) != 2 || got.Messages[0] != want[0] || got.Messages[1] != want[1] {
		t.Fatalf("messages = %+v", got.Messages)
	}
}

func TestAnswerReturnsTheReply(t *testing.T) {
	out, err := testClient(t, reply("Kai built GrabMaps [1].")).Answer(context.Background(), testRequest)
	if err != nil || out != "Kai built GrabMaps [1]." {
		t.Fatalf("Answer = %q, %v", out, err)
	}
}

func TestAnswerErrors(t *testing.T) {
	for name, h := range map[string]http.HandlerFunc{
		"429 upstream": func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":{"code":429,"message":"`+secretQuestion+`"}}`, http.StatusTooManyRequests)
		},
		"error inside 200": func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(`{"error":{"code":502,"message":"provider failed"}}`)) //nolint:errcheck
		},
		"no choices":  func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"choices":[]}`)) }, //nolint:errcheck
		"blank reply": reply("  \n "),
		"not json":    func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("<html>")) }, //nolint:errcheck
	} {
		_, err := testClient(t, h).Answer(context.Background(), testRequest)
		if err == nil {
			t.Errorf("%s: no error", name)
			continue
		}
		if strings.Contains(err.Error(), "secret project") {
			t.Errorf("%s: error leaks the question: %v", name, err)
		}
	}
}

func TestAnswerStopsAtTheDeadline(t *testing.T) {
	release := make(chan struct{})
	c := testClient(t, func(w http.ResponseWriter, r *http.Request) { <-release })
	t.Cleanup(func() { close(release) }) // runs before the server's Close
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := c.Answer(ctx, testRequest)
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 2*time.Second {
		t.Fatalf("slow model = %v after %v", err, time.Since(start))
	}
}

func TestNewWithoutKeyIsNil(t *testing.T) {
	if New("", DefaultModel) != nil {
		t.Fatal("a client without a key must be nil")
	}
}
