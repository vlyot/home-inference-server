package main

import "testing"

func TestRandomPasswordIsNonEmptyAndUnique(t *testing.T) {
	a, err := randomPassword()
	if err != nil {
		t.Fatalf("randomPassword: %v", err)
	}
	if len(a) < 24 {
		t.Fatalf("password too short: %d chars", len(a))
	}
	b, err := randomPassword()
	if err != nil {
		t.Fatalf("randomPassword: %v", err)
	}
	if a == b {
		t.Fatal("two calls produced the same password")
	}
}
