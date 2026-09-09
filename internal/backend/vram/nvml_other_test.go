//go:build !windows

package vram

import "testing"

func TestNVMLOtherReturnsUnlimited(t *testing.T) {
	p := NVMLProvider{}
	if got, err := p.AvailableMB(); err != nil || got != -1 {
		t.Fatalf("AvailableMB() = (%d, %v), want (-1, nil)", got, err)
	}
	if got, err := p.TotalMB(); err != nil || got != -1 {
		t.Fatalf("TotalMB() = (%d, %v), want (-1, nil)", got, err)
	}
}
