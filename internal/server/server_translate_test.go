package server_test

import (
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/server"
)

// TestTranslateInferRequest_VisionSystemPromptNotConcatenated guards against a
// regression of the /v1/describe refusal bug: a vision request's
// system_prompt must flow through as backend.Request.SystemPrompt (a real
// chat-template system turn), not get string-concatenated into Prompt, which
// left llama-server's jinja template with no system-role message at all and
// made the small vision-language model prone to a generic "I cannot see
// images" refusal.
func TestTranslateInferRequest_VisionSystemPromptNotConcatenated(t *testing.T) {
	req := api.InferRequest{
		Modality:     api.ModalityVision,
		SystemPrompt: "You are a helpful assistant.",
		VisionInput:  &api.VisionInput{Prompt: "describe it", ImageBase64: ""},
	}
	got := server.TranslateInferRequest(req, "req-1")
	if got.Prompt != "describe it" {
		t.Errorf("Prompt = %q; want unmodified 'describe it' (system prompt must not be concatenated in)", got.Prompt)
	}
	if got.SystemPrompt != "You are a helpful assistant." {
		t.Errorf("SystemPrompt = %q; want 'You are a helpful assistant.'", got.SystemPrompt)
	}
}

func TestTranslateInferRequest_VisionNoSystemPromptLeavesFieldEmpty(t *testing.T) {
	req := api.InferRequest{
		Modality:    api.ModalityVision,
		VisionInput: &api.VisionInput{Prompt: "describe it"},
	}
	got := server.TranslateInferRequest(req, "req-1")
	if got.SystemPrompt != "" {
		t.Errorf("SystemPrompt = %q; want empty", got.SystemPrompt)
	}
}
