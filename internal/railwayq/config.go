package railwayq

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"time"
)

// NormalizeDSN appends sslmode=require to a Postgres DSN that does not already
// specify an sslmode. Both cmd/railway and cmd/memberadd point at the same Neon
// database (which requires TLS), so they must agree — a DSN with an explicit
// sslmode (e.g. the Railway-set DATABASE_URL, or a local sslmode=disable for
// dev) is left untouched.
func NormalizeDSN(dsn string) string {
	if strings.Contains(dsn, "sslmode=") {
		return dsn
	}
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	return dsn + sep + "sslmode=require"
}

// Config holds the relay's runtime knobs, read from the environment.
type Config struct {
	ResultTTL       time.Duration
	MaxPendingDepth int
	RatePerMin      float64
	DailyQuota      int
	MaxBodyBytes    int64

	// StackJWKSURL and StackProjectID configure verification of Neon Auth
	// (Stack Auth) access tokens presented by web clients. Empty disables the
	// Bearer path (X-API-Key still works).
	StackJWKSURL   string
	StackProjectID string

	// StaticAPIKey is the shared key for native / CLI clients and the PC worker.
	StaticAPIKey string
}

// LoadConfig reads relay configuration from the environment, applying defaults.
// It errors only if the service would accept no credentials at all (neither a
// Stack Auth JWKS URL nor a static API key).
func LoadConfig() (Config, error) {
	cfg := Config{
		ResultTTL:       time.Duration(envInt("RESULT_TTL_SECONDS", 3600)) * time.Second,
		MaxPendingDepth: envInt("MAX_PENDING_DEPTH", 500),
		RatePerMin:      envFloat("RATE_PER_MIN", 10),
		DailyQuota:      envInt("DAILY_QUOTA", 200),
		MaxBodyBytes:    int64(envInt("MAX_BODY_BYTES", 64*1024)),
		StackJWKSURL:    os.Getenv("STACK_JWKS_URL"),
		StackProjectID:  os.Getenv("STACK_PROJECT_ID"),
		StaticAPIKey:    os.Getenv("QUEUE_API_KEY"),
	}

	if cfg.StackJWKSURL != "" && cfg.StackProjectID == "" {
		return Config{}, errors.New("railwayq: STACK_JWKS_URL set but STACK_PROJECT_ID missing")
	}
	if cfg.StackJWKSURL == "" && cfg.StaticAPIKey == "" {
		return Config{}, errors.New("railwayq: need STACK_JWKS_URL (+ STACK_PROJECT_ID) or QUEUE_API_KEY")
	}
	return cfg, nil
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}
