//go:build windows

// Package tray manages the system tray icon using getlantern/systray.
// Requires CGo (MinGW on Windows).
// On non-Windows builds a no-op implementation is used (tray_other.go).
package tray

import (
	"context"
	"os/exec"

	"github.com/getlantern/systray"
)

// Run blocks until ctx is cancelled, showing a system tray icon with an
// "Open Dashboard" menu item and a "Quit" item that stops the tray only
// (server shutdown is driven by the OS signal handler in main).
// notifyCh carries alert strings (e.g. quality-degraded events) that are
// surfaced by updating the tray tooltip. Pass a nil channel to disable.
func Run(ctx context.Context, dashboardURL string, notifyCh <-chan string) {
	ready := make(chan struct{})

	go systray.Run(func() {
		systray.SetTitle("Home Inference Server")
		systray.SetTooltip("Home Inference Server")

		mOpen := systray.AddMenuItem("Open Dashboard", "Open the dashboard in your browser")
		systray.AddSeparator()
		mQuit := systray.AddMenuItem("Quit", "Quit the tray icon")

		close(ready)

		for {
			select {
			case <-mOpen.ClickedCh:
				openURL(dashboardURL)
			case <-mQuit.ClickedCh:
				systray.Quit()
				return
			case msg, ok := <-notifyCh:
				if ok {
					systray.SetTooltip(msg)
				}
			case <-ctx.Done():
				systray.Quit()
				return
			}
		}
	}, func() {})

	<-ready
	<-ctx.Done()
}

func openURL(url string) {
	// Use cmd /c start on Windows to open the default browser.
	_ = exec.Command("cmd", "/c", "start", url).Start()
}
