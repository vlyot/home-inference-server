package railwayq

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// Limiter holds a per-identity token bucket. Buckets are created lazily and
// swept when an identity goes idle so the map cannot grow without bound.
type Limiter struct {
	perMin float64
	mu     sync.Mutex
	seen   map[string]*entry
}

type entry struct {
	lim      *rate.Limiter
	lastSeen time.Time
}

// NewLimiter returns a Limiter allowing perMin requests per identity per minute,
// with a burst equal to perMin (rounded up, minimum 1).
func NewLimiter(perMin float64) *Limiter {
	return &Limiter{perMin: perMin, seen: make(map[string]*entry)}
}

// Allow reports whether the identity may perform one more request right now.
func (l *Limiter) Allow(identity string) bool {
	return l.AllowAt(identity, l.perMin)
}

// AllowAt is Allow with a per-identity rate, for app identities whose limit
// comes from their spec. A changed rate takes effect on the next call.
func (l *Limiter) AllowAt(identity string, perMin float64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	burst := int(perMin)
	if burst < 1 {
		burst = 1
	}
	e, ok := l.seen[identity]
	if !ok {
		e = &entry{lim: rate.NewLimiter(rate.Limit(perMin/60.0), burst)}
		l.seen[identity] = e
	} else if e.lim.Limit() != rate.Limit(perMin/60.0) {
		e.lim.SetLimit(rate.Limit(perMin / 60.0))
		e.lim.SetBurst(burst)
	}
	e.lastSeen = time.Now()
	return e.lim.Allow()
}

// evictIdle drops identities unseen for longer than older.
func (l *Limiter) evictIdle(older time.Duration) {
	cutoff := time.Now().Add(-older)
	l.mu.Lock()
	defer l.mu.Unlock()
	for id, e := range l.seen {
		if !e.lastSeen.After(cutoff) {
			delete(l.seen, id)
		}
	}
}

// StartEvictor runs evictIdle on a ticker until stop is closed.
func (l *Limiter) StartEvictor(stop <-chan struct{}) {
	t := time.NewTicker(10 * time.Minute)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			l.evictIdle(time.Hour)
		}
	}
}
