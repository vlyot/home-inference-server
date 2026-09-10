package server_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
)

// fakeDescriber is a server.Describer double.
type fakeDescriber struct {
	desc string
	tier string
	err  error
	last []byte
}

func (f *fakeDescriber) Describe(_ context.Context, img []byte) (string, string, error) {
	f.last = img
	if f.err != nil {
		return "", "", f.err
	}
	return f.desc, f.tier, nil
}

func TestDescribe_Returns200WithDescription(t *testing.T) {
	h := newHarness(t, 0)
	fd := &fakeDescriber{desc: "a red square, solid fill", tier: "weak"}
	h.srv.SetDescriber(fd)

	imgB64 := base64.StdEncoding.EncodeToString([]byte("fake-png-bytes"))
	resp := post(t, h.ts.URL+api.PathDescribe, api.DescribeRequest{ImageBase64: imgB64})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var dr api.DescribeResponse
	json.NewDecoder(resp.Body).Decode(&dr)
	if dr.Description != "a red square, solid fill" || dr.ModelTier != "weak" {
		t.Errorf("got %+v; want description + tier", dr)
	}
	if string(fd.last) != "fake-png-bytes" {
		t.Errorf("describer got %q; want decoded image bytes", fd.last)
	}
}

func TestDescribe_NoDescriberReturns501(t *testing.T) {
	h := newHarness(t, 0) // no SetDescriber
	imgB64 := base64.StdEncoding.EncodeToString([]byte("x"))
	resp := post(t, h.ts.URL+api.PathDescribe, api.DescribeRequest{ImageBase64: imgB64})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("want 501, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeUnavailable {
		t.Errorf("code = %q; want unavailable", er.Code)
	}
}

func TestDescribe_MissingImageReturns400(t *testing.T) {
	h := newHarness(t, 0)
	h.srv.SetDescriber(&fakeDescriber{desc: "x", tier: "weak"})
	resp := post(t, h.ts.URL+api.PathDescribe, api.DescribeRequest{})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	var er api.ErrorResponse
	json.NewDecoder(resp.Body).Decode(&er)
	if er.Code != api.ErrCodeInvalidRequest {
		t.Errorf("code = %q; want invalid_request", er.Code)
	}
}

func TestDescribe_BadBase64Returns400(t *testing.T) {
	h := newHarness(t, 0)
	h.srv.SetDescriber(&fakeDescriber{desc: "x", tier: "weak"})
	resp := post(t, h.ts.URL+api.PathDescribe, api.DescribeRequest{ImageBase64: "!!! not base64 !!!"})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
}

func TestDescribe_BackendTimeoutMapsTo408(t *testing.T) {
	h := newHarness(t, 0)
	h.srv.SetDescriber(&fakeDescriber{err: &backend.BackendError{Code: api.ErrCodeTimeout, Message: "slow"}})
	imgB64 := base64.StdEncoding.EncodeToString([]byte("x"))
	resp := post(t, h.ts.URL+api.PathDescribe, api.DescribeRequest{ImageBase64: imgB64})
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusRequestTimeout {
		t.Fatalf("want 408, got %d", resp.StatusCode)
	}
}

func TestDescribe_WrongMethodReturns405(t *testing.T) {
	h := newHarness(t, 0)
	h.srv.SetDescriber(&fakeDescriber{desc: "x", tier: "weak"})
	req, _ := http.NewRequest(http.MethodGet, h.ts.URL+api.PathDescribe, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("want 405, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Allow") != http.MethodPost {
		t.Errorf("Allow = %q; want POST", resp.Header.Get("Allow"))
	}
}
