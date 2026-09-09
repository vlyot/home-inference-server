// Package testapp is an integration harness that proves a separately-built app
// can connect to a running Home Inference Server and use every documented
// endpoint. It drives the server through the reference SDK (hisclient) and
// reports a pass/fail checklist.
//
// It is used two ways:
//   - `go run ./cmd/testapp doctor` against a live server (a human watches it);
//   - `go test ./internal/testapp` which starts a GPU-free stub server and runs
//     the same checks in CI.
package testapp

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/hisclient"
)

// Caps describes what the target server supports, so checks that don't apply are
// skipped rather than failed.
type Caps struct {
	// StubBackend is true when the server runs --backend=stub: vision returns a
	// fake 200 instead of 501, and tier/quality behaviour is simplified.
	StubBackend bool
	// APIKey, when set, is sent on every request and the auth check is run.
	APIKey string
	// Settle is a pause between checks. Against a real single-subprocess backend
	// (no llama-server --parallel) a stream and a blocking request that overlap
	// on the same subprocess can collide; a real single-app integrator does not
	// fire 25 tier-switching requests back to back. Default 0 (stub); set ~1.5s
	// for a live GPU server. -stub keeps it 0.
	Settle time.Duration
}

// Check is one named assertion.
type Check struct {
	Name  string
	Group string
	Run   func(ctx context.Context, c *hisclient.Client) error
}

// Result records the outcome of running all checks.
type Result struct {
	Passed  []string
	Failed  []struct{ Name, Err string }
	Skipped []struct{ Name, Why string }
}

// Ok reports whether every check that ran passed.
func (r *Result) Ok() bool { return len(r.Failed) == 0 }

// Run executes every check against c and returns the aggregate result.
func Run(ctx context.Context, c *hisclient.Client, caps Caps) *Result {
	res := &Result{}
	for i, ch := range Checks(caps) {
		if i > 0 && caps.Settle > 0 {
			select {
			case <-ctx.Done():
			case <-time.After(caps.Settle):
			}
		}
		label := ch.Group + " / " + ch.Name
		err := ch.Run(ctx, c)
		switch {
		case err == nil:
			res.Passed = append(res.Passed, label)
		case isSkip(err):
			res.Skipped = append(res.Skipped, struct{ Name, Why string }{label, unSkip(err)})
		default:
			res.Failed = append(res.Failed, struct{ Name, Err string }{label, err.Error()})
		}
	}
	return res
}

// Print writes a grouped checklist to w.
func (r *Result) Print(w io.Writer) {
	lines := map[string][]string{}
	order := []string{}
	add := func(group, line string) {
		if _, seen := lines[group]; !seen {
			order = append(order, group)
		}
		lines[group] = append(lines[group], line)
	}
	for _, p := range r.Passed {
		g, n := split(p)
		add(g, "  ✓ "+n)
	}
	for _, f := range r.Failed {
		g, n := split(f.Name)
		add(g, "  ✗ "+n+"  — "+f.Err)
	}
	for _, s := range r.Skipped {
		g, n := split(s.Name)
		add(g, "  ○ "+n+"  ("+s.Why+")")
	}
	sort.Strings(order)
	for _, g := range order {
		fmt.Fprintf(w, "%s\n", g)
		for _, l := range lines[g] {
			fmt.Fprintln(w, l)
		}
	}
	fmt.Fprintf(w, "\n%d passed, %d failed, %d skipped\n", len(r.Passed), len(r.Failed), len(r.Skipped))
}

func split(label string) (group, name string) {
	if i := strings.Index(label, " / "); i >= 0 {
		return label[:i], label[i+3:]
	}
	return "", label
}

// skipErr signals "this check does not apply", distinct from a failure.
type skipErr struct{ why string }

func (e skipErr) Error() string { return "skip: " + e.why }
func skip(why string) error     { return skipErr{why} }
func isSkip(err error) bool     { _, ok := err.(skipErr); return ok }
func unSkip(err error) string   { return err.(skipErr).why }

func expectAPIError(err error, wantCode string, wantStatus int) error {
	ae, ok := hisclient.AsAPIError(err)
	if !ok {
		return fmt.Errorf("want APIError %q/%d, got %v", wantCode, wantStatus, err)
	}
	if wantCode != "" && ae.Code != wantCode {
		return fmt.Errorf("code = %q, want %q", ae.Code, wantCode)
	}
	if wantStatus != 0 && ae.HTTPStatus != wantStatus {
		return fmt.Errorf("status = %d, want %d", ae.HTTPStatus, wantStatus)
	}
	return nil
}

func textReq(prompt string) api.InferRequest {
	return api.InferRequest{Modality: api.ModalityText, TextInput: &api.TextInput{Prompt: prompt}}
}

// waitFor polls fn until it returns true or the deadline passes.
func waitFor(ctx context.Context, d time.Duration, fn func() bool) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(150 * time.Millisecond):
		}
	}
	return fn()
}
