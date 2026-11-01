package railwayq

import (
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/hosted"
)

func TestLoadConfigDefaults(t *testing.T) {
	for _, k := range []string{
		"RESULT_TTL_SECONDS", "MAX_PENDING_DEPTH", "RATE_PER_MIN",
		"DAILY_QUOTA", "MAX_BODY_BYTES", "STACK_JWKS_URL", "STACK_PROJECT_ID",
	} {
		t.Setenv(k, "")
	}
	t.Setenv("QUEUE_API_KEY", "static-key")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.ResultTTL != time.Hour {
		t.Errorf("ResultTTL = %v, want 1h", cfg.ResultTTL)
	}
	if cfg.MaxPendingDepth != 500 {
		t.Errorf("MaxPendingDepth = %d, want 500", cfg.MaxPendingDepth)
	}
	if cfg.RatePerMin != 10 {
		t.Errorf("RatePerMin = %v, want 10", cfg.RatePerMin)
	}
	if cfg.DailyQuota != 200 {
		t.Errorf("DailyQuota = %d, want 200", cfg.DailyQuota)
	}
	if cfg.MaxBodyBytes != 64*1024 {
		t.Errorf("MaxBodyBytes = %d, want 65536", cfg.MaxBodyBytes)
	}
	if cfg.StaticAPIKey != "static-key" {
		t.Errorf("StaticAPIKey = %q", cfg.StaticAPIKey)
	}
}

func TestLoadConfigHosted(t *testing.T) {
	t.Setenv("QUEUE_API_KEY", "static-key")
	t.Setenv("OPENROUTER_API_KEY", "")
	t.Setenv("HOSTED_MODEL", "")
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.OpenRouterKey != "" || cfg.HostedModel != hosted.DefaultModel {
		t.Errorf("defaults: key %q model %q, want no key and %s", cfg.OpenRouterKey, cfg.HostedModel, hosted.DefaultModel)
	}

	t.Setenv("OPENROUTER_API_KEY", "sk-or-x")
	t.Setenv("HOSTED_MODEL", "google/gemma-4-31b-it:free")
	if cfg, _ = LoadConfig(); cfg.OpenRouterKey != "sk-or-x" || cfg.HostedModel != "google/gemma-4-31b-it:free" {
		t.Errorf("set: key %q model %q", cfg.OpenRouterKey, cfg.HostedModel)
	}
}

func TestLoadConfigErrorsWithNoAuth(t *testing.T) {
	t.Setenv("QUEUE_API_KEY", "")
	t.Setenv("STACK_JWKS_URL", "")
	t.Setenv("STACK_PROJECT_ID", "")
	if _, err := LoadConfig(); err == nil {
		t.Fatal("expected error when neither credential source is configured")
	}
}

func TestLoadConfigStackURLRequiresProjectID(t *testing.T) {
	t.Setenv("QUEUE_API_KEY", "")
	t.Setenv("STACK_JWKS_URL", "https://api.stack-auth.com/x/jwks.json")
	t.Setenv("STACK_PROJECT_ID", "")
	if _, err := LoadConfig(); err == nil {
		t.Fatal("expected error when STACK_JWKS_URL is set without STACK_PROJECT_ID")
	}
}

func TestLoadConfigAcceptsStackConfig(t *testing.T) {
	t.Setenv("QUEUE_API_KEY", "")
	t.Setenv("STACK_JWKS_URL", "https://api.stack-auth.com/x/jwks.json")
	t.Setenv("STACK_PROJECT_ID", "proj-abc")
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.StackProjectID != "proj-abc" {
		t.Errorf("StackProjectID = %q", cfg.StackProjectID)
	}
}

func TestNormalizeDSN(t *testing.T) {
	cases := map[string]string{
		"postgres://h/db":                     "postgres://h/db?sslmode=require",
		"postgres://h/db?x=1":                 "postgres://h/db?x=1&sslmode=require",
		"postgres://h/db?sslmode=disable":     "postgres://h/db?sslmode=disable",
		"postgres://h/db?a=1&sslmode=require": "postgres://h/db?a=1&sslmode=require",
	}
	for in, want := range cases {
		if got := NormalizeDSN(in); got != want {
			t.Errorf("NormalizeDSN(%q) = %q, want %q", in, got, want)
		}
	}
}
