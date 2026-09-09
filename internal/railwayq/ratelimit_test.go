package railwayq

import (
	"testing"
	"time"
)

func TestLimiterAllowsUpToBurst(t *testing.T) {
	l := NewLimiter(5) // burst 5

	for i := 0; i < 5; i++ {
		if !l.Allow("user:a") {
			t.Fatalf("call %d denied, expected allow", i+1)
		}
	}
	if l.Allow("user:a") {
		t.Fatal("6th call allowed, expected deny")
	}
}

func TestLimiterRefillsOverTime(t *testing.T) {
	// 60/min => 1 token per second; burst 60.
	l := NewLimiter(60)
	for i := 0; i < 60; i++ {
		l.Allow("user:b")
	}
	if l.Allow("user:b") {
		t.Fatal("expected bucket empty")
	}
	time.Sleep(1100 * time.Millisecond)
	if !l.Allow("user:b") {
		t.Fatal("expected one token to have refilled after ~1s")
	}
}

func TestLimiterPerIdentityIsolation(t *testing.T) {
	l := NewLimiter(2)
	l.Allow("user:x")
	l.Allow("user:x")
	if l.Allow("user:x") {
		t.Fatal("user:x should be exhausted")
	}
	if !l.Allow("user:y") {
		t.Fatal("user:y should be unaffected by user:x")
	}
}

func TestLimiterEvictsIdleIdentities(t *testing.T) {
	l := NewLimiter(1)
	l.Allow("user:stale")
	if len(l.seen) != 1 {
		t.Fatalf("expected 1 tracked identity, got %d", len(l.seen))
	}
	l.evictIdle(0) // everything older than "now" — i.e. all
	if len(l.seen) != 0 {
		t.Fatalf("expected map emptied, got %d", len(l.seen))
	}
}
