package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestPrintTableReportsThroughput(t *testing.T) {
	results := []result{
		{dur: 100 * time.Millisecond, tier: "strong", tokPerSec: 10},
		{dur: 200 * time.Millisecond, tier: "strong", tokPerSec: 20},
		{dur: 150 * time.Millisecond, tier: "mid", tokPerSec: 30},
		{dur: 120 * time.Millisecond, tier: "mid", tokPerSec: 40},
	}

	out := capture(t, func() {
		printTable(results, 2*time.Second, 180*time.Millisecond)
	})

	for _, want := range []string{
		"requests/sec:", "2.00",
		"aggregate tok/sec:", "100.0",
		"true cold start:", "180 ms",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n---\n%s", want, out)
		}
	}
}

// capture redirects stdout for the duration of fn and returns what was written.
func capture(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()
	fn()
	_ = w.Close()
	os.Stdout = orig
	return <-done
}
