// Package config holds the per-device settings of the local inference server:
// one config.json, overridden field by field by the HIS_* / RAILWAY_* env vars
// the Windows autostart already sets.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/ngkaichong/home-inference-server/types"
)

// DefaultMaxParallel is the concurrent-inference degree: llama-server's
// --parallel N, the vram.Backend semaphore capacity, and the dispatcher's
// worker count. Sized for an 8 GB card where 2 KV caches fit alongside a
// partially-offloaded strong tier.
const DefaultMaxParallel = 2

// DefaultPort is the HTTP port when neither config nor PORT sets one.
const DefaultPort = 8080

// Config is one device's settings. Zero values mean "use the default"; Load
// fills them in.
type Config struct {
	Port      int    `json:"port,omitempty"`
	LocalOnly bool   `json:"local_only,omitempty"`
	BindAddr  string `json:"bind_addr,omitempty"`

	APIKey      string   `json:"api_key,omitempty"`
	CORSOrigins []string `json:"cors_origins,omitempty"`

	EnginePath string                  `json:"engine_path,omitempty"`
	ModelsDir  string                  `json:"models_dir,omitempty"`
	Models     []types.ModelDescriptor `json:"models,omitempty"`

	RailwayQueueURL string `json:"railway_queue_url,omitempty"`
	RailwayAPIKey   string `json:"railway_api_key,omitempty"`

	MaxParallel    int    `json:"max_parallel,omitempty"`
	CtxSize        int    `json:"ctx_size,omitempty"`
	InferTimeoutMS int    `json:"infer_timeout_ms,omitempty"`
	ChatsDir       string `json:"chats_dir,omitempty"`
}

// Locate picks the config file: --config, then HIS_CONFIG, then ./config.json
// if present (the PC's existing location), then the per-user config path. The
// returned file may not exist yet.
func Locate(flagPath string, env Env) string {
	if flagPath != "" {
		return flagPath
	}
	if p := env.Getenv("HIS_CONFIG"); p != "" {
		return p
	}
	if env.Exists("config.json") {
		return "config.json"
	}
	if p := UserConfigPath(env); p != "" {
		return p
	}
	return "config.json"
}

// Load reads path (a missing file is fine, a malformed one is an error),
// applies env-var overrides, fills defaults and validates. unknown lists
// top-level keys the file has that Config doesn't (keys starting with "_",
// like "_comment", are allowed).
func Load(path string, env Env) (cfg Config, unknown []string, err error) {
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if err := json.Unmarshal(data, &cfg); err != nil {
			return Config{}, nil, fmt.Errorf("%s: %w", path, err)
		}
		unknown = unknownKeys(data)
	case errors.Is(err, fs.ErrNotExist):
	default:
		return Config{}, nil, err
	}
	if err := cfg.applyEnv(env.Getenv); err != nil {
		return Config{}, nil, err
	}
	cfg.applyDefaults(env)
	if err := cfg.Validate(); err != nil {
		return Config{}, nil, fmt.Errorf("%s: %w", path, err)
	}
	return cfg, unknown, nil
}

func (c *Config) applyEnv(getenv func(string) string) error {
	for key, dst := range map[string]*string{
		"HIS_BIND_ADDR":         &c.BindAddr,
		"HIS_API_KEY":           &c.APIKey,
		"HIS_LLAMA_SERVER":      &c.EnginePath,
		"HIS_MODELS_DIR":        &c.ModelsDir,
		"CHATS_DIR":             &c.ChatsDir,
		"RAILWAY_QUEUE_URL":     &c.RailwayQueueURL,
		"RAILWAY_QUEUE_API_KEY": &c.RailwayAPIKey,
	} {
		if v := getenv(key); v != "" {
			*dst = v
		}
	}
	if v := getenv("HIS_CORS_ORIGINS"); v != "" {
		c.CORSOrigins = nil
		for _, o := range strings.Split(v, ",") {
			if o = strings.TrimSpace(o); o != "" {
				c.CORSOrigins = append(c.CORSOrigins, o)
			}
		}
	}
	for key, dst := range map[string]*int{
		"PORT":                 &c.Port,
		"HIS_MAX_PARALLEL":     &c.MaxParallel,
		"HIS_CTX_SIZE":         &c.CtxSize,
		"HIS_INFER_TIMEOUT_MS": &c.InferTimeoutMS,
	} {
		if v := getenv(key); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				return fmt.Errorf("env %s=%q is not a whole number", key, v)
			}
			*dst = n
		}
	}
	return nil
}

func (c *Config) applyDefaults(env Env) {
	if c.Port == 0 {
		c.Port = DefaultPort
	}
	if c.MaxParallel == 0 {
		c.MaxParallel = DefaultMaxParallel
	}
	if c.ModelsDir == "" {
		c.ModelsDir = "models"
	}
	if c.ChatsDir == "" {
		c.ChatsDir = DefaultChatsDir(env)
	}
	if c.EnginePath == "" {
		c.EnginePath = DefaultEnginePath(env)
	}
}

// Validate rejects settings the server can't run with.
func (c Config) Validate() error {
	switch {
	case c.Port < 1 || c.Port > 65535:
		return fmt.Errorf("port %d must be between 1 and 65535", c.Port)
	case c.MaxParallel < 1:
		return fmt.Errorf("max_parallel %d must be at least 1", c.MaxParallel)
	case c.CtxSize != 0 && c.CtxSize < 512:
		return fmt.Errorf("ctx_size %d must be at least 512", c.CtxSize)
	case c.InferTimeoutMS < 0:
		return fmt.Errorf("infer_timeout_ms %d must not be negative", c.InferTimeoutMS)
	case c.RailwayQueueURL != "" && c.RailwayAPIKey == "":
		return errors.New("railway_queue_url is set but railway_api_key is empty")
	}
	return nil
}

// ListenAddr is the HTTP bind address: bind_addr / HIS_BIND_ADDR when set,
// loopback-only when local_only, otherwise every interface.
func (c Config) ListenAddr() string {
	switch {
	case c.BindAddr != "":
		return c.BindAddr
	case c.LocalOnly:
		return fmt.Sprintf("127.0.0.1:%d", c.Port)
	default:
		return fmt.Sprintf(":%d", c.Port)
	}
}

// Write saves c as indented JSON, creating parent folders. The file is
// owner-only because it can hold the API and relay keys.
func Write(path string, c Config) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}

func unknownKeys(data []byte) []string {
	var raw map[string]json.RawMessage
	if json.Unmarshal(data, &raw) != nil {
		return nil
	}
	known := map[string]bool{}
	t := reflect.TypeOf(Config{})
	for i := 0; i < t.NumField(); i++ {
		known[strings.Split(t.Field(i).Tag.Get("json"), ",")[0]] = true
	}
	var out []string
	for k := range raw {
		if !known[k] && !strings.HasPrefix(k, "_") {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
