package testapp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/hisclient"
)

func jsonUnmarshal(s string, v any) error { return json.Unmarshal([]byte(s), v) }

// Checks returns every conformance check for the given capabilities.
func Checks(caps Caps) []Check {
	return []Check{
		// ── infer ────────────────────────────────────────────────────────────
		{"text prompt", "infer", func(ctx context.Context, c *hisclient.Client) error {
			r, err := c.Infer(ctx, textReq("Say hi."))
			if err != nil {
				return err
			}
			if r.Output == "" || r.RequestID == "" {
				return fmt.Errorf("empty output or request_id: %+v", r)
			}
			return nil
		}},
		{"chat messages multi-turn", "infer", func(ctx context.Context, c *hisclient.Client) error {
			r, err := c.Infer(ctx, api.InferRequest{
				Modality: api.ModalityText,
				TextInput: &api.TextInput{Messages: []api.ChatMessage{
					{Role: "user", Content: "My name is Sam."},
					{Role: "assistant", Content: "Nice to meet you, Sam."},
					{Role: "user", Content: "What is my name?"},
				}},
			})
			if err != nil {
				return err
			}
			if r.Output == "" {
				return fmt.Errorf("empty output for multi-turn chat")
			}
			return nil
		}},
		{"system_prompt accepted", "infer", func(ctx context.Context, c *hisclient.Client) error {
			req := textReq("Reply.")
			req.SystemPrompt = "You are terse."
			_, err := c.Infer(ctx, req)
			return err
		}},
		{"max_tokens + temperature accepted", "infer", func(ctx context.Context, c *hisclient.Client) error {
			req := textReq("Count to three.")
			req.MaxTokens = 16
			req.Temperature = 0.2
			_, err := c.Infer(ctx, req)
			return err
		}},
		{"preferred_tier pin", "infer", func(ctx context.Context, c *hisclient.Client) error {
			req := textReq("hi")
			req.PreferredTier = api.MinTierWeak
			r, err := c.Infer(ctx, req)
			if err != nil {
				return err
			}
			if r.ModelTier != api.MinTierWeak {
				return fmt.Errorf("model_tier = %q, want weak", r.ModelTier)
			}
			return nil
		}},
		{"min_tier floor", "infer", func(ctx context.Context, c *hisclient.Client) error {
			// Ask for strong as a floor while pinning weak — the floor must win
			// (overloaded) rather than silently serving weak.
			req := textReq("hi")
			req.PreferredTier = api.MinTierWeak
			req.MinTier = api.MinTierStrong
			_, err := c.Infer(ctx, req)
			return expectAPIError(err, api.ErrCodeOverloaded, 429)
		}},
		{"priority accepted", "infer", func(ctx context.Context, c *hisclient.Client) error {
			req := textReq("hi")
			req.Priority = api.PriorityHigh
			_, err := c.Infer(ctx, req)
			return err
		}},
		{"timeout_ms -> 408 timeout", "infer", func(ctx context.Context, c *hisclient.Client) error {
			if caps.StubBackend {
				return skip("stub backend has ~0 latency; timeout not observable")
			}
			req := textReq("Write a long essay about the ocean.")
			req.TimeoutMS = 1
			_, err := c.Infer(ctx, req)
			return expectAPIError(err, api.ErrCodeTimeout, 408)
		}},

		// ── streaming ────────────────────────────────────────────────────────
		{"stream delta + terminator", "streaming", func(ctx context.Context, c *hisclient.Client) error {
			var content strings.Builder
			var sawReasoning bool
			final, err := c.InferStream(ctx, textReq("Say hi in a few words."), func(ch api.StreamChunk) error {
				content.WriteString(ch.Delta)
				if ch.Reasoning != "" {
					sawReasoning = true
				}
				return nil
			})
			if err != nil {
				return err
			}
			if final == nil || !final.Done {
				return fmt.Errorf("no terminating done chunk")
			}
			if content.Len() == 0 && !sawReasoning {
				return fmt.Errorf("stream produced neither delta nor reasoning")
			}
			return nil
		}},

		// ── vision ───────────────────────────────────────────────────────────
		{"vision modality", "vision", func(ctx context.Context, c *hisclient.Client) error {
			// A 1x1 transparent PNG — enough for the perceive stage to run.
			png1x1 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
			req := api.InferRequest{Modality: api.ModalityVision, VisionInput: &api.VisionInput{
				Prompt: "Describe this.", ImageBase64: png1x1,
			}}
			_, err := c.Infer(ctx, req)
			// Both the stub backend and the real weak-tier (native vision)
			// backend return a 200 with output. Under extreme VRAM pressure a
			// non-vision-capable tier may be all that fits, which returns
			// overloaded rather than a wrong-tier answer — that path is not
			// exercised by this conformance check.
			return err
		}},

		// ── status / introspection ───────────────────────────────────────────
		{"GET /v1/status", "status", func(ctx context.Context, c *hisclient.Client) error {
			s, err := c.Status(ctx)
			if err != nil {
				return err
			}
			if _, ok := s["queue_depth"]; !ok {
				return fmt.Errorf("status missing queue_depth")
			}
			return nil
		}},
		{"GET /v1/status/pressure", "status", func(ctx context.Context, c *hisclient.Client) error {
			_, err := c.Pressure(ctx)
			return err
		}},
		{"GET /healthz", "status", func(ctx context.Context, c *hisclient.Client) error {
			h, err := c.Health(ctx)
			if err != nil {
				return err
			}
			if h.Status != "ok" {
				return fmt.Errorf("health status = %q", h.Status)
			}
			return nil
		}},
		{"GET /v1/logs", "status", func(ctx context.Context, c *hisclient.Client) error {
			out, err := c.Logs(ctx, "INFO", "", "1h", 20)
			if err != nil {
				return err
			}
			if _, ok := out["logs"]; !ok {
				return fmt.Errorf("logs response missing 'logs'")
			}
			return nil
		}},
		{"POST /v1/tokenize", "introspection", func(ctx context.Context, c *hisclient.Client) error {
			tr, err := c.Tokenize(ctx, "one two three")
			if err != nil {
				return err
			}
			if tr.NCtx <= 0 {
				return fmt.Errorf("n_ctx = %d", tr.NCtx)
			}
			return nil
		}},
		{"GET /v1/model/props", "introspection", func(ctx context.Context, c *hisclient.Client) error {
			mp, err := c.ModelProps(ctx)
			if err != nil {
				return err
			}
			if mp.NCtx <= 0 {
				return fmt.Errorf("n_ctx = %d", mp.NCtx)
			}
			return nil
		}},

		// ── chats lifecycle (creates ≤2, cleans up) ──────────────────────────
		{"chats CRUD lifecycle", "chats", func(ctx context.Context, c *hisclient.Client) error {
			meta, err := c.CreateChat(ctx, "conformance", []api.ChatMessage{{Role: "user", Content: "hi"}})
			if err != nil {
				return err
			}
			defer c.DeleteChat(context.Background(), meta.ID)

			list, err := c.ListChats(ctx)
			if err != nil {
				return err
			}
			found := false
			for _, m := range list {
				if m.ID == meta.ID {
					found = true
				}
			}
			if !found {
				return fmt.Errorf("created chat %s not in list", meta.ID)
			}

			got, err := c.GetChat(ctx, meta.ID)
			if err != nil {
				return err
			}
			if len(got.Messages) != 1 {
				return fmt.Errorf("get returned %d messages", len(got.Messages))
			}

			if _, err := c.UpdateChat(ctx, meta.ID, "conformance-2", append(got.Messages,
				api.ChatMessage{Role: "assistant", Content: "hello"})); err != nil {
				return err
			}
			if err := c.DeleteChat(ctx, meta.ID); err != nil {
				return err
			}
			if _, err := c.GetChat(ctx, meta.ID); err == nil {
				return fmt.Errorf("chat still present after delete")
			}
			return nil
		}},

		// ── error taxonomy ───────────────────────────────────────────────────
		{"malformed body -> 400 invalid_request", "errors", func(ctx context.Context, c *hisclient.Client) error {
			return postRawExpect(ctx, c, "not json", 400, api.ErrCodeInvalidRequest)
		}},
		{"missing modality -> 400 invalid_request", "errors", func(ctx context.Context, c *hisclient.Client) error {
			return postRawExpect(ctx, c, `{"text_input":{"prompt":"x"}}`, 400, api.ErrCodeInvalidRequest)
		}},
		{"unknown modality -> 400 invalid_modality", "errors", func(ctx context.Context, c *hisclient.Client) error {
			return postRawExpect(ctx, c, `{"modality":"audio","text_input":{"prompt":"x"}}`, 400, api.ErrCodeInvalidModality)
		}},
		{"text without text_input -> 400 invalid_request", "errors", func(ctx context.Context, c *hisclient.Client) error {
			return postRawExpect(ctx, c, `{"modality":"text"}`, 400, api.ErrCodeInvalidRequest)
		}},
		{"GET /v1/infer -> 405", "errors", func(ctx context.Context, c *hisclient.Client) error {
			code, _, err := c.RawGet(ctx, api.PathInfer)
			if err != nil {
				return err
			}
			if code != 405 {
				return fmt.Errorf("GET /v1/infer status = %d, want 405", code)
			}
			return nil
		}},

		// ── auth (only when a key is configured) ─────────────────────────────
		{"auth: missing key -> 401", "auth", func(ctx context.Context, c *hisclient.Client) error {
			if caps.APIKey == "" {
				return skip("server has no HIS_API_KEY set")
			}
			no := hisclient.New(c.BaseURL())
			_, err := no.Status(ctx)
			return expectAPIError(err, api.ErrCodeUnauthorized, 401)
		}},
		{"auth: correct key -> 200", "auth", func(ctx context.Context, c *hisclient.Client) error {
			if caps.APIKey == "" {
				return skip("server has no HIS_API_KEY set")
			}
			_, err := c.Status(ctx)
			return err
		}},

		// ── deferral (best effort — only checkable under real pressure) ───────
		{"deferred -> 202 with relay result_url", "deferred", func(ctx context.Context, c *hisclient.Client) error {
			code, body, err := c.RawPost(ctx, api.PathInfer, `{"modality":"text","priority":"normal","text_input":{"prompt":"hi"}}`)
			if err != nil {
				return err
			}
			if code != 202 {
				return skip("server is not under pressure (got " + itoa(code) + ")")
			}
			if !strings.Contains(body, "result_url") || !strings.Contains(body, "/result/") {
				return fmt.Errorf("202 body missing a relay result_url: %s", body)
			}
			return nil
		}},
	}
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

// postRawExpect POSTs a raw body to /v1/infer and asserts the HTTP status and
// error code.
func postRawExpect(ctx context.Context, c *hisclient.Client, body string, wantStatus int, wantCode string) error {
	code, raw, err := c.RawPost(ctx, api.PathInfer, body)
	if err != nil {
		return err
	}
	if code != wantStatus {
		return fmt.Errorf("status = %d, want %d (body: %s)", code, wantStatus, raw)
	}
	var er api.ErrorResponse
	_ = jsonUnmarshal(raw, &er)
	if er.Code != wantCode {
		return fmt.Errorf("code = %q, want %q", er.Code, wantCode)
	}
	return nil
}
