package railwayq

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestJobSignal_NotifyWakesEveryWaiter(t *testing.T) {
	var s jobSignal
	a, b := s.wait(), s.wait()
	s.notify()
	for i, ch := range []<-chan struct{}{a, b} {
		select {
		case <-ch:
		default:
			t.Fatalf("waiter %d not woken", i)
		}
	}
	// A wait taken after the notify needs the next one.
	c := s.wait()
	select {
	case <-c:
		t.Fatal("new waiter woken by an earlier notify")
	default:
	}
	s.notify()
	<-c
}

func TestJobSignal_NotifyWithoutWaitersIsHarmless(t *testing.T) {
	var s jobSignal
	s.notify()
	s.notify()
	ch := s.wait()
	select {
	case <-ch:
		t.Fatal("wait returned a closed channel")
	default:
	}
}

func TestFinishTracker_KeepsEarliest(t *testing.T) {
	var f finishTracker
	t0 := time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)
	f.record(t0.Add(time.Minute))
	f.record(t0)
	f.record(t0.Add(2 * time.Minute))
	if got := f.peek(); !got.Equal(t0) {
		t.Fatalf("peek = %v, want %v", got, t0)
	}
	if got := f.take(); !got.Equal(t0) {
		t.Fatalf("take = %v, want %v", got, t0)
	}
	if !f.peek().IsZero() {
		t.Fatal("take did not clear the tracker")
	}
}

func TestPCPresence_OnlineForTwoLongPolls(t *testing.T) {
	var p pcPresence
	t0 := time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)
	if p.online(t0) {
		t.Fatal("online before any request")
	}
	p.seen(t0)
	p.seen(t0.Add(-time.Minute)) // a late, older mark doesn't move it back
	if !p.online(t0.Add(59 * time.Second)) {
		t.Fatal("offline 59 s after the last request")
	}
	if p.online(t0.Add(60 * time.Second)) {
		t.Fatal("online 60 s after the last request")
	}
}

// fakeSweeper drives a ttlSweeper with a settable clock and counts the
// database calls it would have made.
type fakeSweeper struct {
	*ttlSweeper
	clock       time.Time
	purges      int
	lookups     int
	remaining   *time.Time
	purgeErr    error
	lookupErr   error
	duringPurge func()
}

func newFakeSweeper(ttl time.Duration) *fakeSweeper {
	f := &fakeSweeper{clock: time.Date(2026, 10, 31, 12, 0, 0, 0, time.UTC)}
	f.ttlSweeper = &ttlSweeper{
		ttl:     ttl,
		tracker: &finishTracker{},
		now:     func() time.Time { return f.clock },
		purge: func(context.Context) (int64, error) {
			f.purges++
			if f.duringPurge != nil {
				f.duringPurge()
			}
			return 1, f.purgeErr
		},
		oldest: func(context.Context) (*time.Time, error) {
			f.lookups++
			return f.remaining, f.lookupErr
		},
	}
	return f
}

func TestTTLSweeper_IdleRelayNeverQueries(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	for i := 0; i < 50; i++ {
		f.clock = f.clock.Add(10 * time.Minute)
		f.sweepIfDue(context.Background())
	}
	if f.purges != 0 || f.lookups != 0 {
		t.Fatalf("idle sweeper made %d purges and %d lookups, want 0", f.purges, f.lookups)
	}
}

func TestTTLSweeper_PurgesOnlyOnceTheTTLHasPassed(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	finished := f.clock
	f.tracker.record(finished)

	f.clock = finished.Add(time.Hour - time.Second)
	f.sweepIfDue(context.Background())
	if f.purges != 0 {
		t.Fatalf("purged %d times before the TTL passed", f.purges)
	}

	f.clock = finished.Add(time.Hour)
	f.sweepIfDue(context.Background())
	if f.purges != 1 || f.lookups != 1 {
		t.Fatalf("at the TTL: %d purges, %d lookups, want 1 and 1", f.purges, f.lookups)
	}

	// Nothing left, so later ticks stay quiet.
	for i := 0; i < 10; i++ {
		f.clock = f.clock.Add(10 * time.Minute)
		f.sweepIfDue(context.Background())
	}
	if f.purges != 1 || f.lookups != 1 {
		t.Fatalf("after the purge: %d purges, %d lookups, want 1 and 1", f.purges, f.lookups)
	}
}

func TestTTLSweeper_NextPurgeFollowsTheOldestRemainingRow(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	f.tracker.record(f.clock)
	newer := f.clock.Add(40 * time.Minute)
	f.remaining = &newer

	f.clock = f.clock.Add(time.Hour)
	f.sweepIfDue(context.Background())
	if f.purges != 1 {
		t.Fatalf("purges = %d, want 1", f.purges)
	}
	f.remaining = nil

	f.clock = newer.Add(time.Hour - time.Minute)
	f.sweepIfDue(context.Background())
	if f.purges != 1 {
		t.Fatalf("purged before the remaining row expired (purges = %d)", f.purges)
	}
	f.clock = newer.Add(time.Hour)
	f.sweepIfDue(context.Background())
	if f.purges != 2 {
		t.Fatalf("purges = %d, want 2 once the remaining row expired", f.purges)
	}
}

func TestTTLSweeper_FailedPurgeIsRetried(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	f.tracker.record(f.clock)
	f.clock = f.clock.Add(time.Hour)
	f.purgeErr = errors.New("connection refused")
	f.sweepIfDue(context.Background())
	if f.lookups != 0 {
		t.Fatalf("looked up after a failed purge")
	}

	f.purgeErr = nil
	f.clock = f.clock.Add(10 * time.Minute)
	f.sweepIfDue(context.Background())
	if f.purges != 2 {
		t.Fatalf("purges = %d, want 2 (failure, then retry)", f.purges)
	}
}

func TestTTLSweeper_FailedLookupLooksAgainNextTick(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	f.tracker.record(f.clock)
	f.clock = f.clock.Add(time.Hour)
	f.lookupErr = errors.New("connection reset")
	f.sweepIfDue(context.Background())

	f.lookupErr = nil
	f.clock = f.clock.Add(10 * time.Minute)
	f.sweepIfDue(context.Background())
	if f.purges != 2 || f.lookups != 2 {
		t.Fatalf("got %d purges, %d lookups, want 2 and 2", f.purges, f.lookups)
	}
}

func TestTTLSweeper_JobFinishedDuringAPurgeIsNotLost(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	f.tracker.record(f.clock)
	f.clock = f.clock.Add(time.Hour)
	finishedMidPurge := f.clock
	f.duringPurge = func() { f.tracker.record(finishedMidPurge) }
	// The lookup ran before that row committed, so it reports nothing left.
	f.remaining = nil
	f.sweepIfDue(context.Background())

	if got := f.tracker.peek(); !got.Equal(finishedMidPurge) {
		t.Fatalf("tracker = %v, want the mid-purge finish %v", got, finishedMidPurge)
	}
}

func TestTTLSweeper_StartupPurgesRowsLeftBeforeARestart(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	old := f.clock.Add(-2 * time.Hour)
	f.remaining = &old
	f.checkAtStartup(context.Background())
	if f.purges != 1 {
		t.Fatalf("purges = %d, want 1 at startup", f.purges)
	}
}

func TestTTLSweeper_StartupWithRecentRowsWaitsForThem(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	recent := f.clock.Add(-10 * time.Minute)
	f.remaining = &recent
	f.checkAtStartup(context.Background())
	if f.purges != 0 {
		t.Fatalf("purged %d times at startup, want 0", f.purges)
	}
	if got := f.tracker.peek(); !got.Equal(recent) {
		t.Fatalf("tracker = %v, want %v", got, recent)
	}
}

func TestTTLSweeper_StartupWithEmptyTableStaysQuiet(t *testing.T) {
	f := newFakeSweeper(time.Hour)
	f.checkAtStartup(context.Background())
	for i := 0; i < 20; i++ {
		f.clock = f.clock.Add(10 * time.Minute)
		f.sweepIfDue(context.Background())
	}
	if f.purges != 0 || f.lookups != 1 {
		t.Fatalf("got %d purges, %d lookups, want 0 and 1 (the startup check)", f.purges, f.lookups)
	}
}
