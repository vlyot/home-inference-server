package setup

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/ngkaichong/home-inference-server/internal/config"
)

type harness struct {
	work    string // absolute working dir; models live in work/models
	env     Env
	out     bytes.Buffer
	written map[string]string
	cmds    []string
}

// newHarness is a darwin host with an absolute temp working dir, the named
// files present in its models folder, and Homebrew's llama-server installed.
func newHarness(t *testing.T, modelFiles ...string) *harness {
	t.Helper()
	h := &harness{work: t.TempDir(), written: map[string]string{}}
	present := map[string]bool{"/opt/homebrew/bin/llama-server": true}
	modelsFS := fstest.MapFS{}
	for _, f := range modelFiles {
		present[h.model(f)] = true
		modelsFS[f] = &fstest.MapFile{}
	}
	h.env = Env{
		Env: config.Env{
			GOOS:          "darwin",
			Getenv:        func(string) string { return "" },
			LookPath:      func(string) (string, error) { return "", errors.New("not found") },
			Exists:        func(p string) bool { return present[p] },
			UserConfigDir: func() (string, error) { return "/Users/me/Library/Application Support", nil },
		},
		ConfigPath: filepath.Join(t.TempDir(), "config.json"),
		Getwd:      func() (string, error) { return h.work, nil },
		ReadDir: func(dir string) ([]os.DirEntry, error) {
			if dir != filepath.Join(h.work, "models") {
				return nil, fs.ErrNotExist
			}
			return fs.ReadDir(modelsFS, ".")
		},
		NewKey:     func() string { return "generated-key" },
		Executable: func() (string, error) { return "/usr/local/bin/his-server", nil },
		HomeDir:    func() (string, error) { return "/Users/me", nil },
		UID:        501,
		WriteFile: func(p string, data []byte) error {
			h.written[p] = string(data)
			return nil
		},
		RunCmd: func(name string, args ...string) error {
			h.cmds = append(h.cmds, name+" "+strings.Join(args, " "))
			return nil
		},
	}
	return h
}

func (h *harness) model(name string) string { return filepath.Join(h.work, "models", name) }

func (h *harness) run(t *testing.T, answers ...string) config.Config {
	t.Helper()
	if err := Run(strings.NewReader(strings.Join(answers, "\n")+"\n"), &h.out, h.env); err != nil {
		t.Fatalf("Run: %v\noutput:\n%s", err, h.out.String())
	}
	cfg, _, err := config.Load(h.env.ConfigPath, h.env.Env)
	if err != nil {
		t.Fatalf("load written config: %v", err)
	}
	return cfg
}

const strongFile = "gemma-4-e4b-q4_k_m.gguf"

func TestRunWritesLocalOnlyConfigWithFoundEngine(t *testing.T) {
	h := newHarness(t, strongFile)
	// engine, models dir, local only, relay, CORS, start at login
	cfg := h.run(t, "", "", "", "", "", "n")

	if !cfg.LocalOnly || cfg.APIKey != "" || cfg.RailwayQueueURL != "" {
		t.Errorf("cfg = %+v; want local-only, no key, no relay", cfg)
	}
	if cfg.EnginePath != "/opt/homebrew/bin/llama-server" {
		t.Errorf("EnginePath = %q", cfg.EnginePath)
	}
	if cfg.ModelsDir != filepath.Join(h.work, "models") {
		t.Errorf("ModelsDir = %q", cfg.ModelsDir)
	}
	if want := filepath.Join("/Users/me/Library/Application Support", config.AppDirName, "chats"); cfg.ChatsDir != want {
		t.Errorf("ChatsDir = %q; want %q", cfg.ChatsDir, want)
	}
	out := h.out.String()
	if !strings.Contains(out, "✓ strong") || !strings.Contains(out, "✗ mid") {
		t.Errorf("model checklist missing from output:\n%s", out)
	}
	if !strings.Contains(out, "--config") {
		t.Errorf("start command should pass --config for a non-default path:\n%s", out)
	}
}

func TestRunRecordsRelayURLAndKey(t *testing.T) {
	h := newHarness(t, strongFile)
	cfg := h.run(t, "", "", "", "y", "https://relay.example", "relay-key", "", "n")
	if cfg.RailwayQueueURL != "https://relay.example" || cfg.RailwayAPIKey != "relay-key" {
		t.Errorf("relay = %q / %q", cfg.RailwayQueueURL, cfg.RailwayAPIKey)
	}
}

func TestRunGeneratesAPIKeyWhenNotLocalOnly(t *testing.T) {
	h := newHarness(t, strongFile)
	cfg := h.run(t, "", "", "n", "", "http://localhost:5173, http://localhost:3000", "n")
	if cfg.LocalOnly || cfg.APIKey != "generated-key" {
		t.Errorf("cfg = %+v; want network access with generated key", cfg)
	}
	if !strings.Contains(h.out.String(), "X-API-Key: generated-key") {
		t.Error("generated key not shown to the user")
	}
	if len(cfg.CORSOrigins) != 2 || cfg.CORSOrigins[1] != "http://localhost:3000" {
		t.Errorf("CORSOrigins = %v", cfg.CORSOrigins)
	}
}

func TestRunListsUnrecognisedGGUFWithoutAddingIt(t *testing.T) {
	h := newHarness(t, strongFile, "my-custom.gguf")
	cfg := h.run(t, "", "", "", "", "", "n")
	if !strings.Contains(h.out.String(), "my-custom.gguf isn't a built-in model") {
		t.Errorf("custom model not reported:\n%s", h.out.String())
	}
	if len(cfg.Models) != 0 {
		t.Errorf("Models = %+v; want none written", cfg.Models)
	}
}

func TestRunPromptsForEnginePathWhenNotFound(t *testing.T) {
	h := newHarness(t, strongFile)
	exists := h.env.Exists
	h.env.Exists = func(p string) bool { return p != "/opt/homebrew/bin/llama-server" && exists(p) }

	cfg := h.run(t, "/custom/llama-server", "", "", "", "", "n")
	out := h.out.String()
	if !strings.Contains(out, "brew install llama.cpp") {
		t.Errorf("install hint missing:\n%s", out)
	}
	if cfg.EnginePath != "/custom/llama-server" {
		t.Errorf("EnginePath = %q", cfg.EnginePath)
	}
}

func TestRunDeclineOverwriteLeavesExistingConfig(t *testing.T) {
	h := newHarness(t)
	if err := os.WriteFile(h.env.ConfigPath, []byte(`{"port": 9999}`), 0o600); err != nil {
		t.Fatal(err)
	}
	exists := h.env.Exists
	h.env.Exists = func(p string) bool { return p == h.env.ConfigPath || exists(p) }

	if err := Run(strings.NewReader("n\n"), &h.out, h.env); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(h.env.ConfigPath); string(b) != `{"port": 9999}` {
		t.Errorf("config changed to %s", b)
	}
}

func TestRunInstallsLaunchAgent(t *testing.T) {
	h := newHarness(t, strongFile)
	h.run(t, "", "", "", "", "", "y")

	plistPath := filepath.Join("/Users/me", "Library", "LaunchAgents", LaunchAgentLabel+".plist")
	plist, ok := h.written[plistPath]
	if !ok {
		t.Fatalf("no plist written; wrote %v", h.written)
	}
	for _, want := range []string{"/usr/local/bin/his-server", h.env.ConfigPath, "<key>KeepAlive</key>", "<key>RunAtLoad</key>"} {
		if !strings.Contains(plist, want) {
			t.Errorf("plist missing %q", want)
		}
	}
	if len(h.cmds) != 2 || h.cmds[1] != "launchctl bootstrap gui/501 "+plistPath {
		t.Errorf("commands = %v; want bootout then bootstrap", h.cmds)
	}
}

func TestRunSkipsLaunchAgentUnderGoRun(t *testing.T) {
	h := newHarness(t, strongFile)
	h.env.Executable = func() (string, error) {
		return filepath.Join(os.TempDir(), "go-build123", "b001", "exe", "server"), nil
	}
	h.run(t, "", "", "", "", "", "y")
	if len(h.written) != 0 || len(h.cmds) != 0 {
		t.Errorf("installed a LaunchAgent for a go run binary: %v %v", h.written, h.cmds)
	}
	if out := h.out.String(); !strings.Contains(out, "go build -o his-server") || !strings.Contains(out, "go run ./cmd/server") {
		t.Errorf("missing go-run guidance:\n%s", out)
	}
}

func TestLaunchAgentPlistEscapesPaths(t *testing.T) {
	p := string(launchAgentPlist("/a&b/his-server", "/c<d>/config.json", "/l.log"))
	if !strings.Contains(p, "/a&amp;b/his-server") || !strings.Contains(p, "/c&lt;d&gt;/config.json") {
		t.Errorf("paths not XML-escaped:\n%s", p)
	}
}
