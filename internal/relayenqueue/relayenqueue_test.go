package relayenqueue

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/queue"
)

func TestEnqueuePostsWithAPIKeyAndReturnsID(t *testing.T) {
	var gotKey, gotPath string
	var gotBody queue.EnqueueRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get(api.HeaderAPIKey)
		gotPath = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(queue.EnqueueResponse{ID: "job-123"})
	}))
	defer srv.Close()

	c := New(srv.URL, "secret-key")
	id, err := c.Enqueue(context.Background(), "corr-1", json.RawMessage(`{"modality":"text"}`))
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	if id != "job-123" {
		t.Fatalf("id = %q, want job-123", id)
	}
	if gotKey != "secret-key" {
		t.Fatalf("X-API-Key = %q", gotKey)
	}
	if gotPath != "/enqueue" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotBody.CorrelationID != "corr-1" || string(gotBody.Payload) != `{"modality":"text"}` {
		t.Fatalf("body = %+v", gotBody)
	}
	if want := srv.URL + "/result/job-123"; c.ResultURL("job-123") != want {
		t.Fatalf("ResultURL = %q, want %q", c.ResultURL("job-123"), want)
	}
}

func TestEnqueueNon2xxIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"code":"queue_full"}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "k")
	if _, err := c.Enqueue(context.Background(), "c", json.RawMessage(`{}`)); err == nil {
		t.Fatal("expected an error on 503")
	}
}

func TestEnqueueUnreachableIsError(t *testing.T) {
	c := New("http://127.0.0.1:1", "k") // nothing listening
	if _, err := c.Enqueue(context.Background(), "c", json.RawMessage(`{}`)); err == nil {
		t.Fatal("expected an error when the relay is unreachable")
	}
}
