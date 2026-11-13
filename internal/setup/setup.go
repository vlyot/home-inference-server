// Package setup is the interactive first-run assistant behind
// `home-inference-server setup`: it finds llama-server and the model files,
// asks how the server should be reachable, and writes config.json.
package setup

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ngkaichong/home-inference-server/internal/config"
)

// LaunchAgentLabel names the macOS LaunchAgent that starts the server at login.
const LaunchAgentLabel = "com.home-inference-server"

// Env is the host setup reads and changes, injectable for tests.
type Env struct {
	config.Env
	ConfigPath string // where config.json is written
	Getwd      func() (string, error)
	ReadDir    func(string) ([]os.DirEntry, error)
	NewKey     func() string
	Executable func() (string, error)
	HomeDir    func() (string, error)
	UID        int
	WriteFile  func(path string, data []byte) error
	RunCmd     func(name string, args ...string) error
}

// OSEnv is the real host, writing config to configPath (the per-user config
// path when empty).
func OSEnv(configPath string) Env {
	env := Env{
		Env:        config.OSEnv(),
		ConfigPath: configPath,
		Getwd:      os.Getwd,
		ReadDir:    os.ReadDir,
		NewKey:     newKey,
		Executable: os.Executable,
		HomeDir:    os.UserHomeDir,
		UID:        os.Getuid(),
		WriteFile: func(path string, data []byte) error {
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return err
			}
			return os.WriteFile(path, data, 0o644)
		},
		RunCmd: runCmd,
	}
	if env.ConfigPath == "" {
		env.ConfigPath = config.UserConfigPath(env.Env)
	}
	return env
}

func runCmd(name string, args ...string) error {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func newKey() string {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		panic(err) // crypto/rand never fails on supported platforms
	}
	return hex.EncodeToString(b)
}

type session struct {
	in  *bufio.Scanner
	out io.Writer
	env Env
}

func (s *session) printf(format string, args ...any) { fmt.Fprintf(s.out, format, args...) }

// ask prints prompt and returns the trimmed answer, or def on an empty answer
// or end of input.
func (s *session) ask(prompt, def string) string {
	if def != "" {
		s.printf("%s [%s]: ", prompt, def)
	} else {
		s.printf("%s: ", prompt)
	}
	if !s.in.Scan() {
		s.printf("\n")
		return def
	}
	if a := strings.TrimSpace(s.in.Text()); a != "" {
		return a
	}
	return def
}

func (s *session) yesNo(prompt string, def bool) bool {
	hint := "y/N"
	if def {
		hint = "Y/n"
	}
	for {
		switch strings.ToLower(s.ask(prompt+" ["+hint+"]", "")) {
		case "":
			return def
		case "y", "yes":
			return true
		case "n", "no":
			return false
		}
		s.printf("Please answer y or n.\n")
	}
}

func (s *session) required(prompt string) string {
	for {
		if a := s.ask(prompt, ""); a != "" {
			return a
		}
		s.printf("This one is required.\n")
	}
}

// Run walks through setup on in/out and writes the config file.
func Run(in io.Reader, out io.Writer, env Env) error {
	s := &session{in: bufio.NewScanner(in), out: out, env: env}
	s.printf("Home Inference Server setup\n\nDetected: %s\nSettings file: %s\n\n", osName(env.GOOS), env.ConfigPath)

	if env.Exists(env.ConfigPath) && !s.yesNo("A settings file already exists. Replace it?", false) {
		s.printf("Left unchanged.\n")
		return nil
	}

	cfg := config.Config{
		EnginePath: s.engine(),
		ModelsDir:  s.models(),
		ChatsDir:   config.DefaultChatsDir(env.Env),
	}

	cfg.LocalOnly = s.yesNo("\nOnly this computer can use the server (local only)?", true)
	if !cfg.LocalOnly {
		cfg.APIKey = env.NewKey()
		s.printf("Other devices must send this header with every request:\n  X-API-Key: %s\n", cfg.APIKey)
	}

	if s.yesNo("\nConnect to a hosted relay (queues requests while this device is off)?", false) {
		cfg.RailwayQueueURL = s.required("Relay URL")
		cfg.RailwayAPIKey = s.required("Relay API key")
	}

	if origins := s.ask("\nWeb apps allowed to call the server from a browser (comma-separated, e.g. http://localhost:5173; Enter for none)", ""); origins != "" {
		for _, o := range strings.Split(origins, ",") {
			if o = strings.TrimSpace(o); o != "" {
				cfg.CORSOrigins = append(cfg.CORSOrigins, o)
			}
		}
	}

	if err := config.Write(env.ConfigPath, cfg); err != nil {
		return fmt.Errorf("write %s: %w", env.ConfigPath, err)
	}
	s.printf("\nSaved %s\n", env.ConfigPath)
	if env.ConfigPath != "config.json" && env.Exists("config.json") {
		s.printf("Note: the config.json in this folder takes priority when the server is started from here.\n")
	}

	exe, fromGoRun := s.serverCommand()
	if env.GOOS == "darwin" {
		if err := s.offerLaunchAgent(exe, fromGoRun); err != nil {
			return err
		}
	}
	start := exe
	if fromGoRun {
		start = "go run ./cmd/server"
	}
	if env.ConfigPath != config.UserConfigPath(env.Env) {
		start += " --config " + strconv.Quote(env.ConfigPath)
	}
	s.printf("\nStart the server with:\n  %s\n", start)
	return nil
}

func osName(goos string) string {
	switch goos {
	case "darwin":
		return "macOS"
	case "windows":
		return "Windows"
	}
	return goos
}

func (s *session) engine() string {
	if p := config.DefaultEnginePath(s.env.Env); p != "" && s.env.Exists(p) {
		s.printf("AI engine found: %s\n", p)
		return s.ask("llama-server path", p)
	}
	switch s.env.GOOS {
	case "darwin":
		s.printf("llama-server not found. Install it with: brew install llama.cpp\n")
	case "windows":
		s.printf("llama-server not found. Download a CUDA build from https://github.com/ggml-org/llama.cpp/releases to C:\\llama\\\n")
	default:
		s.printf("llama-server not found. Install llama.cpp and put llama-server on your PATH.\n")
	}
	p := s.ask("Path to llama-server (Enter to set later)", "")
	if p != "" && !s.env.Exists(p) {
		s.printf("Warning: %s doesn't exist yet; saved anyway.\n", p)
	}
	return p
}

func (s *session) models() string {
	def := "models"
	if wd, err := s.env.Getwd(); err == nil {
		def = filepath.Join(wd, "models")
	}
	dir := s.ask("\nModels folder", def)
	if !filepath.IsAbs(dir) {
		if wd, err := s.env.Getwd(); err == nil {
			dir = filepath.Join(wd, dir)
		}
	}

	known := map[string]bool{}
	found := 0
	for _, d := range config.BuiltinModels() {
		known[d.FilePath] = true
		mark, note := "✗", "  (not downloaded)"
		if s.env.Exists(filepath.Join(dir, d.FilePath)) {
			mark, note = "✓", ""
			found++
		}
		if d.MMProjPath != "" {
			known[d.MMProjPath] = true
			if note == "" && !s.env.Exists(filepath.Join(dir, d.MMProjPath)) {
				note = "  (text only: " + d.MMProjPath + " missing)"
			}
		}
		s.printf("  %s %-6s %s%s\n", mark, d.TierLabel, d.FilePath, note)
	}
	if entries, err := s.env.ReadDir(dir); err == nil {
		for _, e := range entries {
			if n := e.Name(); strings.HasSuffix(strings.ToLower(n), ".gguf") && !known[n] {
				s.printf("  ? %s isn't a built-in model; add it under \"models\" in the settings file to use it\n", n)
			}
		}
	}
	if found == 0 {
		s.printf("No built-in model found yet. Put at least one in %s before starting the server.\n", dir)
	}
	return dir
}

// serverCommand is the binary a LaunchAgent should run. fromGoRun is true when
// setup runs under `go run`, whose binary is deleted when it exits.
func (s *session) serverCommand() (exe string, fromGoRun bool) {
	exe, err := s.env.Executable()
	if err != nil {
		return "home-inference-server", false
	}
	return exe, strings.Contains(exe, string(filepath.Separator)+"go-build")
}

func (s *session) offerLaunchAgent(exe string, fromGoRun bool) error {
	if !s.yesNo("\nStart the server automatically at login (and restart it if it stops)?", true) {
		return nil
	}
	if fromGoRun {
		s.printf("Start at login needs a built program, not `go run`. Build one and rerun its setup:\n" +
			"  go build -o his-server ./cmd/server && ./his-server setup\n")
		return nil
	}
	home, err := s.env.HomeDir()
	if err != nil {
		return err
	}
	plistPath := filepath.Join(home, "Library", "LaunchAgents", LaunchAgentLabel+".plist")
	logPath := filepath.Join(home, "Library", "Logs", "home-inference-server.log")
	if err := s.env.WriteFile(plistPath, launchAgentPlist(exe, s.env.ConfigPath, logPath)); err != nil {
		return fmt.Errorf("write %s: %w", plistPath, err)
	}
	domain := "gui/" + strconv.Itoa(s.env.UID)
	_ = s.env.RunCmd("launchctl", "bootout", domain+"/"+LaunchAgentLabel) // not loaded yet is fine
	if err := s.env.RunCmd("launchctl", "bootstrap", domain, plistPath); err != nil {
		return fmt.Errorf("launchctl bootstrap %s: %w", plistPath, err)
	}
	s.printf("Started, and it will start again at every login.\n"+
		"  Logs: %s\n  Stop: launchctl bootout %s/%s\n", logPath, domain, LaunchAgentLabel)
	return nil
}

func launchAgentPlist(exe, configPath, logPath string) []byte {
	esc := func(v string) string {
		var b bytes.Buffer
		_ = xml.EscapeText(&b, []byte(v))
		return b.String()
	}
	return []byte(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + LaunchAgentLabel + `</string>
	<key>ProgramArguments</key>
	<array>
		<string>` + esc(exe) + `</string>
		<string>--config</string>
		<string>` + esc(configPath) + `</string>
	</array>
	<key>WorkingDirectory</key>
	<string>` + esc(filepath.Dir(configPath)) + `</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>StandardOutPath</key>
	<string>` + esc(logPath) + `</string>
	<key>StandardErrorPath</key>
	<string>` + esc(logPath) + `</string>
</dict>
</plist>
`)
}
