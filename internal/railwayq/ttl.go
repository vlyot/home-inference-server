package railwayq

import (
	"context"
	"database/sql"
	"log/slog"
	"time"

	"github.com/ngkaichong/home-inference-server/logschema"
)

// StartTTLSweeper deletes terminal jobs older than ttl. It runs until ctx is
// cancelled.
//
// It only queries when a finished job has actually passed its TTL. Ack records
// each finish time in memory, so on an idle relay every tick is a no-op and the
// database is left alone to suspend. The tick interval is ttl capped at 10
// minutes (and at least 1 minute), so a purge runs at most one interval after
// it falls due.
//
// The relay can restart with finished rows still in the table, so it also
// checks once at startup, when Open's ping has already woken the database.
func StartTTLSweeper(ctx context.Context, db *sql.DB, ttl time.Duration) {
	interval := ttl
	if interval > 10*time.Minute {
		interval = 10 * time.Minute
	}
	if interval < time.Minute {
		interval = time.Minute
	}

	s := &ttlSweeper{
		ttl:     ttl,
		tracker: &finishedJobs,
		now:     time.Now,
		purge: func(ctx context.Context) (int64, error) {
			return PurgeExpired(ctx, db, ttl)
		},
		oldest: func(ctx context.Context) (*time.Time, error) {
			return OldestFinishedAt(ctx, db)
		},
	}
	s.checkAtStartup(ctx)

	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.sweepIfDue(ctx)
		}
	}
}

// ttlSweeper decides whether a purge is due. The database calls are funcs so
// tests can drive it without Postgres.
type ttlSweeper struct {
	ttl     time.Duration
	tracker *finishTracker
	now     func() time.Time
	purge   func(context.Context) (int64, error)
	oldest  func(context.Context) (*time.Time, error)
}

// checkAtStartup records the oldest finished row already in the table, then
// purges it at once if it has expired.
func (s *ttlSweeper) checkAtStartup(ctx context.Context) {
	oldest, err := s.oldest(ctx)
	if err != nil {
		slog.Warn("ttl sweep: startup check failed", "err", err)
		// Unknown, so treat it as due and let the first tick look again.
		s.tracker.record(s.now().Add(-s.ttl))
		return
	}
	if oldest != nil {
		s.tracker.record(*oldest)
	}
	s.sweepIfDue(ctx)
}

// sweepIfDue purges when the oldest known finished job has passed its TTL, and
// does nothing (no query) otherwise.
func (s *ttlSweeper) sweepIfDue(ctx context.Context) {
	oldest := s.tracker.peek()
	if oldest.IsZero() || s.now().Before(oldest.Add(s.ttl)) {
		return
	}
	s.tracker.take()

	n, err := s.purge(ctx)
	if err != nil {
		slog.Warn("ttl sweep failed", "err", err)
		s.tracker.record(oldest) // still due; retry next tick
		return
	}
	if n > 0 {
		slog.Info("purged expired results",
			slog.String(logschema.FieldEvent, string(logschema.EventResultsPurged)),
			slog.Int64(logschema.FieldCount, n),
		)
	}

	// Rows that finished within the TTL are still there. Find the oldest so
	// the next purge happens when it expires, not on every tick.
	next, err := s.oldest(ctx)
	if err != nil {
		slog.Warn("ttl sweep: oldest finished lookup failed", "err", err)
		s.tracker.record(s.now().Add(-s.ttl)) // unknown, so look again next tick
		return
	}
	if next != nil {
		s.tracker.record(*next)
	}
}
