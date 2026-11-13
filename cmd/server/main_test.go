package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/internal/config"
	"github.com/ngkaichong/home-inference-server/types"
)

func none(string) bool { return false }

func TestStartupRosterErrorsWhenNoModelFilesInstalled(t *testing.T) {
	_, err := startupRoster(config.Config{ModelsDir: "/m"}, "llamacpp", none)
	if err == nil {
		t.Fatal("want an error with no model files")
	}
	absDir, _ := filepath.Abs("/m")
	for _, want := range []string{absDir, "gemma-4-e4b-q4_k_m.gguf", "gemma-4-e2b-q4_k_m.gguf", "lfm2-vl-3b-instruct-q4_k_m.gguf"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q missing %q", err, want)
		}
	}
}

func TestStartupRosterKeepsOnlyInstalledModels(t *testing.T) {
	strong := filepath.Join("/m", "gemma-4-e4b-q4_k_m.gguf")
	r, err := startupRoster(config.Config{ModelsDir: "/m"}, "llamacpp", func(p string) bool { return p == strong })
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 1 || r[0].TierLabel != types.TierStrong {
		t.Errorf("roster = %+v; want strong only", r)
	}
}

func TestStartupRosterStubBackendSkipsFileChecks(t *testing.T) {
	r, err := startupRoster(config.Config{ModelsDir: "/m"}, "stub", none)
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 3 {
		t.Errorf("stub roster has %d tiers; want all 3", len(r))
	}
}
