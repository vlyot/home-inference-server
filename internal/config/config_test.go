package config

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/ngkaichong/home-inference-server/types"
)

// fakeEnv is a darwin host with the given env vars, no llama-server on PATH and
// only the listed files present.
func fakeEnv(vars map[string]string, files ...string) Env {
	present := map[string]bool{}
	for _, f := range files {
		present[f] = true
	}
	return Env{
		GOOS:          "darwin",
		Getenv:        func(k string) string { return vars[k] },
		LookPath:      func(string) (string, error) { return "", errors.New("not found") },
		Exists:        func(p string) bool { return present[p] },
		UserConfigDir: func() (string, error) { return "/Users/me/Library/Application Support", nil },
	}
}

func writeFile(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadReturnsDefaultsWhenNoFileExists(t *testing.T) {
	cfg, unknown, err := Load(filepath.Join(t.TempDir(), "missing.json"), fakeEnv(nil, "/opt/homebrew/bin/llama-server"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Port != 8080 || cfg.MaxParallel != DefaultMaxParallel || cfg.ModelsDir != "models" || cfg.CtxSize != 0 {
		t.Errorf("defaults = %+v", cfg)
	}
	if cfg.EnginePath != "/opt/homebrew/bin/llama-server" {
		t.Errorf("EnginePath = %q", cfg.EnginePath)
	}
	if want := filepath.Join("/Users/me/Library/Application Support", AppDirName, "chats"); cfg.ChatsDir != want {
		t.Errorf("ChatsDir = %q; want %q", cfg.ChatsDir, want)
	}
	if len(unknown) != 0 {
		t.Errorf("unknown = %v", unknown)
	}
}

func TestLoadAppliesFileValues(t *testing.T) {
	p := writeFile(t, `{
		"port": 9000, "local_only": true, "api_key": "k", "cors_origins": ["http://a"],
		"engine_path": "/e", "models_dir": "/m", "max_parallel": 1, "ctx_size": 8192,
		"infer_timeout_ms": 5000, "chats_dir": "/c",
		"railway_queue_url": "https://r", "railway_api_key": "rk"
	}`)
	cfg, _, err := Load(p, fakeEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	want := Config{
		Port: 9000, LocalOnly: true, APIKey: "k", CORSOrigins: []string{"http://a"},
		EnginePath: "/e", ModelsDir: "/m", MaxParallel: 1, CtxSize: 8192,
		InferTimeoutMS: 5000, ChatsDir: "/c",
		RailwayQueueURL: "https://r", RailwayAPIKey: "rk",
	}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("cfg = %+v\nwant %+v", cfg, want)
	}
}

func TestLoadEnvVarsOverrideFile(t *testing.T) {
	p := writeFile(t, `{"port": 9000, "api_key": "file", "cors_origins": ["http://file"], "engine_path": "/file",
		"models_dir": "/file", "chats_dir": "/file", "max_parallel": 1, "ctx_size": 1024, "infer_timeout_ms": 1,
		"railway_queue_url": "https://file", "railway_api_key": "file", "bind_addr": "1.1.1.1:1"}`)
	cfg, _, err := Load(p, fakeEnv(map[string]string{
		"PORT": "9100", "HIS_API_KEY": "env", "HIS_CORS_ORIGINS": "http://a, http://b,", "HIS_LLAMA_SERVER": "/env",
		"HIS_MODELS_DIR": "/env", "CHATS_DIR": "/env", "HIS_MAX_PARALLEL": "3", "HIS_CTX_SIZE": "2048",
		"HIS_INFER_TIMEOUT_MS": "7", "RAILWAY_QUEUE_URL": "https://env", "RAILWAY_QUEUE_API_KEY": "env",
		"HIS_BIND_ADDR": "127.0.0.1:1",
	}))
	if err != nil {
		t.Fatal(err)
	}
	want := Config{
		Port: 9100, APIKey: "env", CORSOrigins: []string{"http://a", "http://b"}, EnginePath: "/env",
		ModelsDir: "/env", ChatsDir: "/env", MaxParallel: 3, CtxSize: 2048, InferTimeoutMS: 7,
		RailwayQueueURL: "https://env", RailwayAPIKey: "env", BindAddr: "127.0.0.1:1",
	}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("cfg = %+v\nwant %+v", cfg, want)
	}
}

func TestLoadReadsExistingPCConfigUnchanged(t *testing.T) {
	p := writeFile(t, `{
		"_comment": "Copy to config.json (git-ignored) OR set the equivalent env vars.",
		"railway_queue_url": "https://your-queue-api.up.railway.app",
		"railway_api_key": "secret"
	}`)
	cfg, unknown, err := Load(p, fakeEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.RailwayQueueURL != "https://your-queue-api.up.railway.app" || cfg.RailwayAPIKey != "secret" {
		t.Errorf("relay settings = %q / %q", cfg.RailwayQueueURL, cfg.RailwayAPIKey)
	}
	if len(unknown) != 0 {
		t.Errorf("_comment reported as unknown: %v", unknown)
	}
}

func TestLoadErrorsOnMalformedJSON(t *testing.T) {
	p := writeFile(t, `{"port": 8080,}`)
	if _, _, err := Load(p, fakeEnv(nil)); err == nil || !strings.Contains(err.Error(), p) {
		t.Errorf("err = %v; want a parse error naming %s", err, p)
	}
}

func TestLoadErrorsOnNonNumericEnv(t *testing.T) {
	_, _, err := Load(filepath.Join(t.TempDir(), "none.json"), fakeEnv(map[string]string{"HIS_MAX_PARALLEL": "two"}))
	if err == nil || !strings.Contains(err.Error(), "HIS_MAX_PARALLEL") {
		t.Errorf("err = %v; want it to name HIS_MAX_PARALLEL", err)
	}
}

func TestLoadReportsUnknownKeys(t *testing.T) {
	p := writeFile(t, `{"prot": 9000, "_note": "fine", "local_only": true}`)
	cfg, unknown, err := Load(p, fakeEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(unknown, []string{"prot"}) {
		t.Errorf("unknown = %v; want [prot]", unknown)
	}
	if cfg.Port != 8080 || !cfg.LocalOnly {
		t.Errorf("known keys not applied: %+v", cfg)
	}
}

func TestValidateRejectsBadSettings(t *testing.T) {
	ok := Config{Port: 8080, MaxParallel: 1}
	for name, mutate := range map[string]func(*Config){
		"port too high":         func(c *Config) { c.Port = 70000 },
		"negative max_parallel": func(c *Config) { c.MaxParallel = -1 },
		"tiny ctx_size":         func(c *Config) { c.CtxSize = 100 },
		"negative timeout":      func(c *Config) { c.InferTimeoutMS = -5 },
		"relay URL without key": func(c *Config) { c.RailwayQueueURL = "https://r" },
	} {
		c := ok
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("%s: Validate accepted %+v", name, c)
		}
	}
	if err := ok.Validate(); err != nil {
		t.Errorf("valid config rejected: %v", err)
	}
}

func TestListenAddr(t *testing.T) {
	for _, tc := range []struct {
		cfg  Config
		want string
	}{
		{Config{Port: 8080}, ":8080"},
		{Config{Port: 8080, LocalOnly: true}, "127.0.0.1:8080"},
		{Config{Port: 8080, LocalOnly: true, BindAddr: "0.0.0.0:9"}, "0.0.0.0:9"},
	} {
		if got := tc.cfg.ListenAddr(); got != tc.want {
			t.Errorf("%+v.ListenAddr() = %q; want %q", tc.cfg, got, tc.want)
		}
	}
}

func TestLocateFollowsFlagEnvCwdUserDirOrder(t *testing.T) {
	user := filepath.Join("/Users/me/Library/Application Support", AppDirName, "config.json")
	if got := Locate("/flag.json", fakeEnv(map[string]string{"HIS_CONFIG": "/env.json"}, "config.json")); got != "/flag.json" {
		t.Errorf("flag: got %q", got)
	}
	if got := Locate("", fakeEnv(map[string]string{"HIS_CONFIG": "/env.json"}, "config.json")); got != "/env.json" {
		t.Errorf("env: got %q", got)
	}
	if got := Locate("", fakeEnv(nil, "config.json")); got != "config.json" {
		t.Errorf("cwd: got %q", got)
	}
	if got := Locate("", fakeEnv(nil)); got != user {
		t.Errorf("user dir: got %q; want %q", got, user)
	}
}

func TestWriteRoundTripsThroughLoad(t *testing.T) {
	p := filepath.Join(t.TempDir(), "nested", "config.json")
	in := Config{Port: 8181, LocalOnly: true, EnginePath: "/e", ModelsDir: "/m", ChatsDir: "/c", MaxParallel: 2, CORSOrigins: []string{"http://x"}}
	if err := Write(p, in); err != nil {
		t.Fatal(err)
	}
	out, unknown, err := Load(p, fakeEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, in) || len(unknown) != 0 {
		t.Errorf("round trip: got %+v (unknown %v); want %+v", out, unknown, in)
	}
}

func TestDefaultEnginePath(t *testing.T) {
	win := fakeEnv(nil)
	win.GOOS = "windows"
	if got := DefaultEnginePath(win); got != WindowsEnginePath {
		t.Errorf("windows: %q", got)
	}

	onPath := fakeEnv(nil, "/opt/homebrew/bin/llama-server")
	onPath.LookPath = func(string) (string, error) { return "/custom/bin/llama-server", nil }
	if got := DefaultEnginePath(onPath); got != "/custom/bin/llama-server" {
		t.Errorf("darwin PATH first: %q", got)
	}
	if got := DefaultEnginePath(fakeEnv(nil, "/opt/homebrew/bin/llama-server")); got != "/opt/homebrew/bin/llama-server" {
		t.Errorf("darwin Homebrew: %q", got)
	}
	if got := DefaultEnginePath(fakeEnv(nil)); got != "" {
		t.Errorf("darwin none: %q; want empty", got)
	}
}

func TestDefaultChatsDirWindowsUsesLocalAppData(t *testing.T) {
	win := fakeEnv(map[string]string{"LOCALAPPDATA": "/AppData/Local"})
	win.GOOS = "windows"
	if got, want := DefaultChatsDir(win), filepath.Join("/AppData/Local", AppDirName, "chats"); got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

func TestRosterJoinsBuiltinFilenamesToModelsDir(t *testing.T) {
	r, err := Config{ModelsDir: "/m"}.Roster()
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 3 {
		t.Fatalf("len = %d; want 3", len(r))
	}
	if got, want := r[2].FilePath, filepath.Join("/m", "gemma-4-e4b-q4_k_m.gguf"); got != want {
		t.Errorf("strong FilePath = %q; want %q", got, want)
	}
	if got, want := r[0].MMProjPath, filepath.Join("/m", "mmproj-lfm2-vl-3b-instruct-q8_0.gguf"); got != want {
		t.Errorf("weak MMProjPath = %q; want %q", got, want)
	}
	if BuiltinModels()[2].FilePath != "gemma-4-e4b-q4_k_m.gguf" {
		t.Error("Roster mutated the built-in list")
	}
}

func TestRosterCustomModelsReplaceBuiltinAndSort(t *testing.T) {
	abs := filepath.Join(t.TempDir(), "big.gguf")
	cfg := Config{ModelsDir: "/m", Models: []types.ModelDescriptor{
		{TierLabel: types.TierStrong, Name: "big", FilePath: abs},
		{TierLabel: types.TierWeak, Name: "small", FilePath: "small.gguf"},
	}}
	r, err := cfg.Roster()
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 2 || r[0].Name != "small" || r[1].Name != "big" {
		t.Fatalf("roster = %+v; want [small big]", r)
	}
	if r[0].FilePath != filepath.Join("/m", "small.gguf") || r[1].FilePath != abs {
		t.Errorf("paths = %q, %q", r[0].FilePath, r[1].FilePath)
	}
	if r[0].Port != 8093 || r[1].Port != 8092 || r[0].Modality != "text" {
		t.Errorf("defaults not filled: %+v", r)
	}
}

func TestRosterRejectsInvalidModels(t *testing.T) {
	for name, models := range map[string][]types.ModelDescriptor{
		"duplicate tier": {
			{TierLabel: types.TierStrong, Name: "a", FilePath: "a"},
			{TierLabel: types.TierStrong, Name: "b", FilePath: "b"},
		},
		"duplicate port": {
			{TierLabel: types.TierStrong, Name: "a", FilePath: "a", Port: 9000},
			{TierLabel: types.TierMid, Name: "b", FilePath: "b", Port: 9000},
		},
		"unknown tier": {{TierLabel: "huge", Name: "a", FilePath: "a"}},
		"missing file": {{TierLabel: types.TierStrong, Name: "a"}},
	} {
		if _, err := (Config{Models: models}).Roster(); err == nil {
			t.Errorf("%s: Roster accepted %+v", name, models)
		}
	}
}

func TestFilterInstalledKeepsOnlyModelsPresent(t *testing.T) {
	r, _ := Config{ModelsDir: "/m"}.Roster()
	kept, notes := FilterInstalled(r, func(p string) bool { return p == filepath.Join("/m", "gemma-4-e4b-q4_k_m.gguf") })
	if len(kept) != 1 || kept[0].TierLabel != types.TierStrong {
		t.Fatalf("kept = %+v; want strong only", kept)
	}
	if len(notes) != 2 {
		t.Errorf("notes = %v; want one per skipped model", notes)
	}
}

func TestFilterInstalledDropsMMProjWhenProjectorMissing(t *testing.T) {
	r, _ := Config{ModelsDir: "/m"}.Roster()
	weak := filepath.Join("/m", "lfm2-vl-3b-instruct-q4_k_m.gguf")
	kept, notes := FilterInstalled(r[:1], func(p string) bool { return p == weak })
	if len(kept) != 1 || kept[0].HasVision() || kept[0].VisionKVFixedMB != 0 {
		t.Fatalf("kept = %+v; want weak as text-only", kept)
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "text-only") {
		t.Errorf("notes = %v", notes)
	}
}
