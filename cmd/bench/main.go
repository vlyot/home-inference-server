// Package main is a benchmark harness that fires HTTP requests at the running
// inference server and reports cold-start latency, wall-clock throughput,
// p50/p95/p99 response times, tier distribution, and observed tok/sec. Run it
// at -concurrency 1/2/4/8 to trace the throughput curve.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

type result struct {
	dur       time.Duration
	tier      string
	tokPerSec float64
	err       error
}

type benchConfig struct {
	addr        string
	concurrency int
	requests    int
	priority    string
	minTier     string
	prompt      string
}

func main() {
	cfg := benchConfig{}
	flag.StringVar(&cfg.addr, "addr", "http://localhost:8080", "server base URL")
	flag.IntVar(&cfg.concurrency, "concurrency", 1, "number of concurrent workers")
	flag.IntVar(&cfg.requests, "requests", 20, "total requests to fire")
	flag.StringVar(&cfg.priority, "priority", "normal", "job priority: high|normal|low")
	flag.StringVar(&cfg.minTier, "min-tier", "", "minimum tier: strong|mid|weak (empty = no floor)")
	flag.StringVar(&cfg.prompt, "prompt", "Say hello in exactly one word.", "prompt text to send")
	flag.Parse()

	fmt.Printf("Bench: addr=%s concurrency=%d requests=%d priority=%s\n\n",
		cfg.addr, cfg.concurrency, cfg.requests, cfg.priority)

	results, wall, coldStart := runBench(cfg)
	printTable(results, wall, coldStart)
}

// runBench fires cfg.requests requests through cfg.concurrency closed-loop
// workers. It returns every result, the wall-clock span of the whole run (for
// throughput), and the latency of the genuinely-first completion (cold start,
// captured by arrival order — not the fastest request after sorting).
func runBench(cfg benchConfig) (results []result, wall time.Duration, coldStart time.Duration) {
	jobs := make(chan struct{}, cfg.requests)
	for i := 0; i < cfg.requests; i++ {
		jobs <- struct{}{}
	}
	close(jobs)

	results = make([]result, 0, cfg.requests)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var firstOnce sync.Once

	start := time.Now()
	for i := 0; i < cfg.concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range jobs {
				r := fireRequest(cfg)
				firstOnce.Do(func() { coldStart = r.dur })
				mu.Lock()
				results = append(results, r)
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	wall = time.Since(start)
	return results, wall, coldStart
}

func fireRequest(cfg benchConfig) result {
	// Use the chat-messages path: the plain /completion endpoint applies no chat
	// template, so instruction-tuned models (Gemma) return empty output there.
	body := api.InferRequest{
		Modality: api.ModalityText,
		TextInput: &api.TextInput{
			Messages: []api.ChatMessage{{Role: "user", Content: cfg.prompt}},
		},
		Priority:  cfg.priority,
		MinTier:   cfg.minTier,
		MaxTokens: 32,
	}
	payload, _ := json.Marshal(body)

	start := time.Now()
	resp, err := http.Post(cfg.addr+"/v1/infer", "application/json", bytes.NewReader(payload))
	dur := time.Since(start)

	if err != nil {
		return result{dur: dur, err: err}
	}
	defer resp.Body.Close()

	var ir api.InferResponse
	if decErr := json.NewDecoder(resp.Body).Decode(&ir); decErr != nil {
		return result{dur: dur, err: fmt.Errorf("decode: %w", decErr)}
	}
	if resp.StatusCode != http.StatusOK {
		return result{dur: dur, err: fmt.Errorf("status %d", resp.StatusCode)}
	}
	return result{
		dur:       dur,
		tier:      ir.ModelTier,
		tokPerSec: ir.TokensPerSec,
	}
}

func printTable(results []result, wall, coldStart time.Duration) {
	var ok []result
	var errCount int
	for _, r := range results {
		if r.err != nil {
			errCount++
			fmt.Fprintf(os.Stderr, "  error: %v\n", r.err)
		} else {
			ok = append(ok, r)
		}
	}

	if len(ok) == 0 {
		fmt.Println("No successful results.")
		return
	}

	durs := make([]float64, len(ok))
	tierCounts := map[string]int{}
	var tokSum float64
	var tokN int
	for i, r := range ok {
		durs[i] = float64(r.dur.Milliseconds())
		tierCounts[r.tier]++
		// Ignore the degenerate "1 token in ~0 ms" sample the server reports as
		// tokens_per_sec == 1e6 (an empty/instant completion) — it swamps the mean.
		if r.tokPerSec > 0 && r.tokPerSec < 1e6 {
			tokSum += r.tokPerSec
			tokN++
		}
	}
	sort.Float64s(durs)

	p := func(pct float64) float64 {
		idx := int(math.Ceil(pct/100*float64(len(durs)))) - 1
		if idx < 0 {
			idx = 0
		}
		return durs[idx]
	}

	reqPerSec := float64(len(ok)) / wall.Seconds()

	fmt.Printf("Results (%d ok, %d errors)\n", len(ok), errCount)
	fmt.Println("─────────────────────────────────────────")
	fmt.Printf("  wall time:          %6.2f s\n", wall.Seconds())
	fmt.Printf("  requests/sec:       %6.2f\n", reqPerSec)
	fmt.Printf("  true cold start:    %6.0f ms\n", float64(coldStart.Milliseconds()))
	fmt.Printf("  p50 latency:        %6.0f ms\n", p(50))
	fmt.Printf("  p95 latency:        %6.0f ms\n", p(95))
	fmt.Printf("  p99 latency:        %6.0f ms\n", p(99))
	fmt.Printf("  min / max:          %6.0f / %6.0f ms\n", durs[0], durs[len(durs)-1])
	if tokN > 0 {
		fmt.Printf("  avg tok/sec (req):  %6.1f\n", tokSum/float64(tokN))
		fmt.Printf("  aggregate tok/sec:  %6.1f\n", tokSum)
	}
	fmt.Println()
	fmt.Println("Tier distribution:")
	for tier, count := range tierCounts {
		pct := 100.0 * float64(count) / float64(len(ok))
		fmt.Printf("  %-10s %3d  (%.0f%%)\n", tier, count, pct)
	}
}
