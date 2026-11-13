package config

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// AppDirName is the per-user folder (under the OS config dir, or
// %LOCALAPPDATA% for chats on Windows) holding this server's files.
const AppDirName = "home-inference-server"

// WindowsEnginePath is where the Windows PC has always kept llama-server.
const WindowsEnginePath = `C:\llama\llama-server.exe`

// Env is the slice of the host the config layer reads, injectable for tests.
type Env struct {
	GOOS          string
	Getenv        func(string) string
	LookPath      func(string) (string, error)
	Exists        func(string) bool // true for an existing regular file
	UserConfigDir func() (string, error)
}

// OSEnv is the real host.
func OSEnv() Env {
	return Env{
		GOOS:          runtime.GOOS,
		Getenv:        os.Getenv,
		LookPath:      exec.LookPath,
		Exists:        FileExists,
		UserConfigDir: os.UserConfigDir,
	}
}

// FileExists reports whether p is an existing regular file.
func FileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.Mode().IsRegular()
}

// DefaultEnginePath finds llama-server: the fixed C:\llama path on Windows;
// elsewhere PATH, then the Homebrew prefixes (Apple Silicon, then Intel).
// Returns "" when nothing is found.
func DefaultEnginePath(env Env) string {
	if env.GOOS == "windows" {
		return WindowsEnginePath
	}
	if p, err := env.LookPath("llama-server"); err == nil {
		return p
	}
	for _, p := range []string{"/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"} {
		if env.Exists(p) {
			return p
		}
	}
	return ""
}

// DefaultChatsDir keeps Windows on %LOCALAPPDATA% (where the PC's chats already
// live) and uses the OS config dir elsewhere, e.g.
// ~/Library/Application Support/home-inference-server/chats on macOS.
func DefaultChatsDir(env Env) string {
	if env.GOOS == "windows" {
		return filepath.Join(env.Getenv("LOCALAPPDATA"), AppDirName, "chats")
	}
	if dir, err := env.UserConfigDir(); err == nil {
		return filepath.Join(dir, AppDirName, "chats")
	}
	return filepath.Join(AppDirName, "chats")
}

// UserConfigPath is the per-user config.json location setup writes to, or ""
// when the OS has no config dir.
func UserConfigPath(env Env) string {
	dir, err := env.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, AppDirName, "config.json")
}
