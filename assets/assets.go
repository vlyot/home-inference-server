// Package assets embeds the static web files for the dashboard.
package assets

import "embed"

//go:embed web
var FS embed.FS
