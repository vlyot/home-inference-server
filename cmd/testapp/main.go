// testapp is a small demo client for the Home Inference Server. It doubles as
// the acceptance check that a separately-built app can connect and use every
// documented endpoint (`testapp doctor`).
//
//	testapp doctor                 run the full conformance checklist
//	testapp chat "your prompt"     stream a single chat turn
//	testapp summarize <file>       one-shot summary with a system prompt
//	testapp status                 print the server status snapshot
//
// Flags: -addr (or HIS_TESTAPP_ADDR), -api-key (or HIS_API_KEY).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/hisclient"
	"github.com/ngkaichong/home-inference-server/internal/testapp"
)

func main() {
	addr := flag.String("addr", envOr("HIS_TESTAPP_ADDR", "http://localhost:8080"), "server base URL")
	apiKey := flag.String("api-key", os.Getenv("HIS_API_KEY"), "X-API-Key, if the server requires one")
	tier := flag.String("tier", "", "preferred_tier for chat/summarize (weak|mid|strong)")
	stub := flag.Bool("stub", os.Getenv("HIS_TESTAPP_STUB") == "1", "target is running --backend=stub (relaxes vision/timeout checks)")
	settle := flag.Duration("settle", 0, "pause between doctor checks; use ~1500ms against a real GPU server so a stream and a blocking request don't collide on the single llama-server")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		usage()
	}

	opts := []hisclient.Option{}
	if *apiKey != "" {
		opts = append(opts, hisclient.WithAPIKey(*apiKey))
	}
	c := hisclient.New(*addr, opts...)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var err error
	switch args[0] {
	case "doctor":
		err = doctor(ctx, c, testapp.Caps{APIKey: *apiKey, StubBackend: *stub, Settle: *settle})
	case "chat":
		if len(args) < 2 {
			usage()
		}
		err = chat(ctx, c, strings.Join(args[1:], " "), *tier)
	case "summarize":
		if len(args) < 2 {
			usage()
		}
		err = summarize(ctx, c, args[1], *tier)
	case "status":
		err = printJSON(c.Status(ctx))
	case "pressure":
		p, e := c.Pressure(ctx)
		err = printJSON(p, e)
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func doctor(ctx context.Context, c *hisclient.Client, caps testapp.Caps) error {
	dctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	h, err := c.Health(dctx)
	if err != nil {
		return fmt.Errorf("server not reachable at %s: %w", c.BaseURL(), err)
	}
	fmt.Printf("Home Inference Server — conformance report\n  target : %s\n  version: %s\n\n", c.BaseURL(), h.Version)

	res := testapp.Run(dctx, c, caps)
	res.Print(os.Stdout)
	if !res.Ok() {
		return fmt.Errorf("%d check(s) failed", len(res.Failed))
	}
	return nil
}

func chat(ctx context.Context, c *hisclient.Client, prompt, tier string) error {
	req := api.InferRequest{
		Modality:      api.ModalityText,
		PreferredTier: tier,
		TextInput:     &api.TextInput{Messages: []api.ChatMessage{{Role: "user", Content: prompt}}},
	}
	inReasoning := false
	final, err := c.InferStream(ctx, req, func(ch api.StreamChunk) error {
		if ch.Reasoning != "" {
			if !inReasoning {
				fmt.Print("\x1b[2m(thinking) ")
				inReasoning = true
			}
			fmt.Print(ch.Reasoning)
		}
		if ch.Delta != "" {
			if inReasoning {
				fmt.Print("\x1b[0m\n\n")
				inReasoning = false
			}
			fmt.Print(ch.Delta)
		}
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Println()
	if final != nil {
		fmt.Printf("\n[tier=%s tokens=%d %.1f tok/s]\n", final.ModelTier, final.TokensGenerated, final.TokensPerSec)
	}
	return nil
}

func summarize(ctx context.Context, c *hisclient.Client, path, tier string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	req := api.InferRequest{
		Modality:      api.ModalityText,
		PreferredTier: tier,
		SystemPrompt:  "Summarise the user's text in three sentences.",
		MaxTokens:     256,
		TextInput:     &api.TextInput{Prompt: string(data)},
	}
	r, err := c.Infer(ctx, req)
	if err != nil {
		return err
	}
	fmt.Println(r.Output)
	if r.QualityDegraded {
		fmt.Fprintln(os.Stderr, "(served on a lower tier than requested)")
	}
	return nil
}

func printJSON(v any, err error) error {
	if err != nil {
		return err
	}
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(b))
	return nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func usage() {
	fmt.Fprint(os.Stderr, `testapp — Home Inference Server demo client

  testapp doctor                 run the full conformance checklist
  testapp chat "your prompt"     stream one chat turn
  testapp summarize <file>       one-shot summary
  testapp status | pressure      print a server snapshot

flags: -addr <url>  -api-key <key>  -tier <weak|mid|strong>
`)
	os.Exit(2)
}
