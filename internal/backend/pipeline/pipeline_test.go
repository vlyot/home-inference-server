package pipeline

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/ngkaichong/home-inference-server/backend"
)

// fakeStage records the requests it receives and returns a canned response or
// error. It also records call ordering against a shared log.
type fakeStage struct {
	mu          sync.Mutex
	name        string
	log         *[]string
	inferResp   backend.Response
	inferErr    error
	lastReq     backend.Request
	shutdownErr error
	stream      bool // if true, implements backend.Streamer
}

func (f *fakeStage) Modality() backend.ModalityKind { return backend.ModalityKindText }
func (f *fakeStage) Ready() bool                    { return true }

func (f *fakeStage) Infer(_ context.Context, req backend.Request) (backend.Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	*f.log = append(*f.log, f.name+".Infer")
	f.lastReq = req
	return f.inferResp, f.inferErr
}

func (f *fakeStage) Shutdown(_ context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	*f.log = append(*f.log, f.name+".Shutdown")
	return f.shutdownErr
}

// fakeStreamStage is a fakeStage that also implements backend.Streamer.
type fakeStreamStage struct {
	fakeStage
	deltas []string
}

func (f *fakeStreamStage) InferStream(_ context.Context, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	f.mu.Lock()
	*f.log = append(*f.log, f.name+".InferStream")
	f.lastReq = req
	f.mu.Unlock()
	if f.inferErr != nil {
		return backend.Response{}, f.inferErr
	}
	for _, d := range f.deltas {
		chunkFn(backend.ChunkContent, d)
	}
	return f.inferResp, nil
}

func testSpec() Spec {
	return Spec{
		Modality:             backend.ModalityKindVision,
		PerceptionPrompt:     "inventory this image",
		ReasonSystemPrompt:   "you cannot see images",
		DescriptionMaxTokens: 256,
	}
}

func TestInfer_RunsPerceiveThenReason(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "a red ball on a table"}}
	reason := &fakeStage{name: "reason", log: &log, inferResp: backend.Response{Output: "It's a ball."}}

	b := New(perceive, reason, testSpec())
	resp, err := b.Infer(context.Background(), backend.Request{Prompt: "what is this?", ImageData: []byte{1}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Output != "It's a ball." {
		t.Errorf("output = %q; want reason hop output", resp.Output)
	}
	// Reason hop must have seen the description and the original question.
	last := reason.lastReq
	if len(last.Messages) != 2 || last.Messages[0].Role != "system" {
		t.Fatalf("reason messages = %+v; want [system, user]", last.Messages)
	}
	user := last.Messages[1].Content
	if !contains(user, "a red ball on a table") || !contains(user, "what is this?") {
		t.Errorf("reason user turn = %q; want description + question", user)
	}
}

func TestInfer_ShutsDownPerceiveBeforeReason(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStage{name: "reason", log: &log, inferResp: backend.Response{Output: "answer"}}

	b := New(perceive, reason, testSpec())
	if _, err := b.Infer(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}}); err != nil {
		t.Fatal(err)
	}
	want := []string{"perceive.Infer", "perceive.Shutdown", "reason.Infer"}
	if !equal(log, want) {
		t.Errorf("call order = %v; want %v", log, want)
	}
}

func TestInfer_PerceiveErrorPropagates_ReasonNotCalled(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferErr: &backend.BackendError{Code: "timeout", Message: "slow"}}
	reason := &fakeStage{name: "reason", log: &log}

	b := New(perceive, reason, testSpec())
	_, err := b.Infer(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}})
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "timeout" {
		t.Fatalf("err = %v; want BackendError timeout", err)
	}
	for _, e := range log {
		if e == "reason.Infer" {
			t.Error("reason hop ran despite perceive failure")
		}
	}
}

func TestInfer_ReasonOverloadedFallsBackToDescription(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "the full description", TokensGenerated: 12, ModelTier: "weak"}}
	reason := &fakeStage{name: "reason", log: &log, inferErr: &backend.BackendError{Code: "overloaded", Message: "no tier fits"}}

	b := New(perceive, reason, testSpec())
	resp, err := b.Infer(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}})
	if err != nil {
		t.Fatalf("expected graceful degrade, got error: %v", err)
	}
	if resp.Output != "the full description" || !resp.QualityDegraded {
		t.Errorf("resp = %+v; want description output + QualityDegraded", resp)
	}
}

func TestInfer_ReasonDeferredErrorPropagates(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStage{name: "reason", log: &log, inferErr: &backend.BackendError{Code: "timeout", Message: "deadline"}}

	b := New(perceive, reason, testSpec())
	_, err := b.Infer(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}})
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "timeout" {
		t.Fatalf("err = %v; want propagated timeout", err)
	}
}

func TestInfer_ForwardsResponseFormatAndPriorityToReason(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStage{name: "reason", log: &log, inferResp: backend.Response{Output: "{}"}}

	b := New(perceive, reason, testSpec())
	rf := []byte(`{"type":"json_object"}`)
	_, err := b.Infer(context.Background(), backend.Request{
		Prompt:         "q",
		ImageData:      []byte{1},
		Priority:       "high",
		MinTier:        "mid",
		ResponseFormat: rf,
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(reason.lastReq.ResponseFormat) != string(rf) {
		t.Errorf("reason ResponseFormat = %q; want forwarded", reason.lastReq.ResponseFormat)
	}
	if reason.lastReq.Priority != "high" || reason.lastReq.MinTier != "mid" {
		t.Errorf("reason priority/min_tier = %q/%q; want high/mid", reason.lastReq.Priority, reason.lastReq.MinTier)
	}
	if perceive.lastReq.Priority != "high" {
		t.Errorf("perceive priority = %q; want high", perceive.lastReq.Priority)
	}
}

func TestInfer_PerceptionPromptUsedNotUserPrompt(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStage{name: "reason", log: &log, inferResp: backend.Response{Output: "a"}}

	b := New(perceive, reason, testSpec())
	_, _ = b.Infer(context.Background(), backend.Request{Prompt: "what colour is the car?", ImageData: []byte{1}})
	if perceive.lastReq.Prompt != "inventory this image" {
		t.Errorf("perceive prompt = %q; want the spec perception prompt", perceive.lastReq.Prompt)
	}
	if len(perceive.lastReq.ImageData) == 0 {
		t.Error("perceive request missing image data")
	}
}

func TestInferStream_StreamsReasonHopAfterBlockingPerception(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStreamStage{
		fakeStage: fakeStage{name: "reason", log: &log, inferResp: backend.Response{TokensGenerated: 3}},
		deltas:    []string{"one ", "two ", "three"},
	}

	b := New(perceive, reason, testSpec())
	var got []string
	_, err := b.InferStream(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}},
		func(_ backend.ChunkKind, d string) { got = append(got, d) })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !equal(got, []string{"one ", "two ", "three"}) {
		t.Errorf("streamed deltas = %v; want the reason hop's", got)
	}
	want := []string{"perceive.Infer", "perceive.Shutdown", "reason.InferStream"}
	if !equal(log, want) {
		t.Errorf("call order = %v; want %v", log, want)
	}
}

func TestInferStream_FallbackWhenReasonNotStreamer(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "desc"}}
	reason := &fakeStage{name: "reason", log: &log, inferResp: backend.Response{Output: "single answer"}}

	b := New(perceive, reason, testSpec())
	var got []string
	_, err := b.InferStream(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}},
		func(_ backend.ChunkKind, d string) { got = append(got, d) })
	if err != nil {
		t.Fatal(err)
	}
	if !equal(got, []string{"single answer"}) {
		t.Errorf("deltas = %v; want one delta from Infer fallback", got)
	}
}

func TestInferStream_ReasonOverloadedEmitsDescription(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "the description", ModelTier: "weak"}}
	reason := &fakeStreamStage{
		fakeStage: fakeStage{name: "reason", log: &log, inferErr: &backend.BackendError{Code: "overloaded"}},
	}

	b := New(perceive, reason, testSpec())
	var got []string
	resp, err := b.InferStream(context.Background(), backend.Request{Prompt: "q", ImageData: []byte{1}},
		func(_ backend.ChunkKind, d string) { got = append(got, d) })
	if err != nil {
		t.Fatalf("expected graceful degrade, got: %v", err)
	}
	if !equal(got, []string{"the description"}) || !resp.QualityDegraded {
		t.Errorf("got=%v degraded=%v; want description delta + degraded", got, resp.QualityDegraded)
	}
}

func TestShutdown_ShutsDownPerceiveNotReason(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log}
	reason := &fakeStage{name: "reason", log: &log}

	b := New(perceive, reason, testSpec())
	if err := b.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !equal(log, []string{"perceive.Shutdown"}) {
		t.Errorf("shutdown calls = %v; want only perceive.Shutdown", log)
	}
}

func TestModality_ReturnsSpecModality(t *testing.T) {
	b := New(&fakeStage{name: "p", log: &[]string{}}, &fakeStage{name: "r", log: &[]string{}}, testSpec())
	if b.Modality() != backend.ModalityKindVision {
		t.Errorf("modality = %q; want vision", b.Modality())
	}
}

func TestNew_DefaultsDescriptionMaxTokens(t *testing.T) {
	b := New(&fakeStage{name: "p", log: &[]string{}}, &fakeStage{name: "r", log: &[]string{}}, Spec{Modality: backend.ModalityKindVision})
	if b.spec.DescriptionMaxTokens != 512 {
		t.Errorf("DescriptionMaxTokens = %d; want 512 default", b.spec.DescriptionMaxTokens)
	}
}

func TestDescribe_RunsPerceiveOnlyAndEvicts(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "a red square", ModelTier: "weak"}}
	reason := &fakeStage{name: "reason", log: &log}

	b := New(perceive, reason, testSpec())
	desc, tier, err := b.Describe(context.Background(), []byte{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if desc != "a red square" || tier != "weak" {
		t.Errorf("got (%q, %q); want (a red square, weak)", desc, tier)
	}
	if !equal(log, []string{"perceive.Infer", "perceive.Shutdown"}) {
		t.Errorf("call log = %v; want perceive Infer then Shutdown, reason untouched", log)
	}
}

func TestDescribe_PerceiveErrorPropagates(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferErr: &backend.BackendError{Code: "timeout", Message: "slow"}}
	reason := &fakeStage{name: "reason", log: &log}

	b := New(perceive, reason, testSpec())
	desc, _, err := b.Describe(context.Background(), []byte{1})
	be := &backend.BackendError{}
	if !errors.As(err, &be) || be.Code != "timeout" {
		t.Fatalf("err = %v; want BackendError timeout", err)
	}
	if desc != "" {
		t.Errorf("desc = %q; want empty on error", desc)
	}
}

func TestDescribe_UsesPerceptionPromptAndMaxTokens(t *testing.T) {
	log := []string{}
	perceive := &fakeStage{name: "perceive", log: &log, inferResp: backend.Response{Output: "x"}}
	b := New(perceive, &fakeStage{name: "r", log: &log}, testSpec())
	if _, _, err := b.Describe(context.Background(), []byte{9}); err != nil {
		t.Fatal(err)
	}
	if perceive.lastReq.Prompt != testSpec().PerceptionPrompt {
		t.Errorf("perceive prompt = %q; want the spec perception prompt", perceive.lastReq.Prompt)
	}
	if perceive.lastReq.MaxTokens != 256 { // testSpec sets DescriptionMaxTokens: 256
		t.Errorf("perceive MaxTokens = %d; want 256", perceive.lastReq.MaxTokens)
	}
	if len(perceive.lastReq.ImageData) == 0 {
		t.Error("perceive request missing image data")
	}
}

// --- helpers ---

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
