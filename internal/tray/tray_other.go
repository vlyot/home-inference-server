//go:build !windows

// Package tray manages the system tray icon. The real implementation
// (tray_windows.go) uses getlantern/systray and needs CGo; on non-Windows
// builds this no-op version just blocks until ctx is cancelled so the server
// runs headless (used for Linux CI and the GPU-free contract test).
package tray

import "context"

// Run blocks until ctx is cancelled. No tray icon is shown on this platform.
func Run(ctx context.Context, _ string, _ <-chan string) {
	<-ctx.Done()
}
