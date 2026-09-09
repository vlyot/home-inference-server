package railwayq

import (
	"context"
	"database/sql"
	"log/slog"
	"time"

	"github.com/ngkaichong/home-inference-server/logschema"
)

// StartTTLSweeper periodically deletes terminal jobs older than ttl. It runs
// until ctx is cancelled. The sweep interval is ttl capped at 10 minutes so a
// short TTL still gets swept promptly.
func StartTTLSweeper(ctx context.Context, db *sql.DB, ttl time.Duration) {
	interval := ttl
	if interval > 10*time.Minute {
		interval = 10 * time.Minute
	}
	if interval < time.Minute {
		interval = time.Minute
	}
	t := time.NewTicker(interval)
	defer t.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			n, err := PurgeExpired(ctx, db, ttl)
			if err != nil {
				slog.Warn("ttl sweep failed", "err", err)
				continue
			}
			if n > 0 {
				slog.Info("purged expired results",
					slog.String(logschema.FieldEvent, string(logschema.EventResultsPurged)),
					slog.Int64(logschema.FieldCount, n),
				)
			}
		}
	}
}
