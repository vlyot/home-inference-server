package railwayq

import (
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/queue"
)

// The relay's database is a Neon compute that suspends after a few idle
// minutes and wakes on the next query. The helpers below keep the relay from
// querying when it has nothing to do, so the compute can stay suspended. All
// are in-process, which is enough because the relay runs as one Railway
// replica; a second replica would need them moved into the database.

// jobSignal wakes Claim calls that are waiting for work when Enqueue adds a
// job, so a long-poll doesn't have to query the database every second to
// notice it. A job inserted some other way (by hand, or by another process)
// is still picked up: every Claim checks the table once when it starts, and
// the PC worker starts a new Claim at least every long-poll interval.
type jobSignal struct {
	mu sync.Mutex
	ch chan struct{}
}

// wait returns a channel that is closed by the next notify. Take it before
// checking the table, so a job enqueued between the check and the wait still
// wakes the caller.
func (s *jobSignal) wait() <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ch == nil {
		s.ch = make(chan struct{})
	}
	return s.ch
}

// notify wakes every caller currently holding a channel from wait.
func (s *jobSignal) notify() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ch != nil {
		close(s.ch)
		s.ch = nil
	}
}

// jobsAdded is notified by Enqueue and waited on by Claim.
var jobsAdded jobSignal

// finishTracker remembers the earliest finished_at that may still be in the
// jobs table, so the TTL sweeper only runs its DELETE once something has
// actually expired. Zero means no finished job is waiting to be purged.
type finishTracker struct {
	mu     sync.Mutex
	oldest time.Time
}

// record notes that a job finished at t.
func (f *finishTracker) record(t time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.oldest.IsZero() || t.Before(f.oldest) {
		f.oldest = t
	}
}

// peek returns the earliest recorded finish time, or zero.
func (f *finishTracker) peek() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.oldest
}

// take returns the earliest recorded finish time and clears it. Jobs that
// finish while the caller is purging are recorded again as usual.
func (f *finishTracker) take() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	t := f.oldest
	f.oldest = time.Time{}
	return t
}

// finishedJobs is recorded by Ack and read by the TTL sweeper.
var finishedJobs finishTracker

// pcPresence remembers when the PC worker was last heard from (a /claim poll
// or an /ack), so /status, /apps/status and /apps/ask can tell whether it is
// online without waking the database. A relay restart forgets it, so the PC
// reads as offline until its next poll, which follows within seconds because
// the restart drops its open long-poll.
type pcPresence struct {
	mu   sync.Mutex
	last time.Time
}

// seen records that the PC worker made a request at t.
func (p *pcPresence) seen(t time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if t.After(p.last) {
		p.last = t
	}
}

// online reports whether the PC worker was heard from within twice the
// long-poll interval, so an idle PC between polls still counts.
func (p *pcPresence) online(now time.Time) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return !p.last.IsZero() &&
		now.Sub(p.last) < 2*time.Duration(queue.LongPollTimeoutSeconds)*time.Second
}
