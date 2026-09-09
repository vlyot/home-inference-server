// Package logbuf provides an in-memory, process-local ring buffer of structured
// log records. It wraps a slog.Handler so every record that reaches stdout is
// also retained for the dashboard's /logs page. Nothing is persisted; the
// buffer empties on restart.
package logbuf

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/logschema"
)

// Record is one captured log line.
type Record struct {
	Time  time.Time         `json:"time"`
	Level string            `json:"level"` // "DEBUG" | "INFO" | "WARN" | "ERROR"
	Msg   string            `json:"msg"`
	Event string            `json:"event,omitempty"` // logschema "event" attr, if present
	Attrs map[string]string `json:"attrs,omitempty"` // all other attrs, stringified
}

// Buffer is a bounded, newest-first store of Records. Safe for concurrent use.
type Buffer struct {
	mu      sync.Mutex
	records []Record
	max     int
	maxAge  time.Duration
}

// NewBuffer returns a Buffer holding at most max records and dropping any
// record older than maxAge on insert.
func NewBuffer(max int, maxAge time.Duration) *Buffer {
	return &Buffer{max: max, maxAge: maxAge}
}

func (b *Buffer) add(r Record) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.records = append([]Record{r}, b.records...)
	if len(b.records) > b.max {
		b.records = b.records[:b.max]
	}
	if b.maxAge > 0 {
		cutoff := time.Now().Add(-b.maxAge)
		for len(b.records) > 0 && b.records[len(b.records)-1].Time.Before(cutoff) {
			b.records = b.records[:len(b.records)-1]
		}
	}
}

// Query returns matching records, newest-first, capped at limit.
// An empty level or event disables that filter; since == 0 disables the time
// filter.
func (b *Buffer) Query(level, event string, since time.Duration, limit int) []Record {
	b.mu.Lock()
	defer b.mu.Unlock()

	var cutoff time.Time
	if since > 0 {
		cutoff = time.Now().Add(-since)
	}
	level = strings.ToUpper(strings.TrimSpace(level))

	out := make([]Record, 0, min(limit, len(b.records)))
	for _, r := range b.records {
		if len(out) >= limit {
			break
		}
		if level != "" && r.Level != level {
			continue
		}
		if event != "" && r.Event != event {
			continue
		}
		if !cutoff.IsZero() && r.Time.Before(cutoff) {
			continue
		}
		out = append(out, r)
	}
	return out
}

// Handler wraps inner so records passing through are also stored in b.
func (b *Buffer) Handler(inner slog.Handler) slog.Handler {
	return &handler{buf: b, inner: inner}
}

type handler struct {
	buf   *Buffer
	inner slog.Handler
	attrs []slog.Attr
}

func (h *handler) Enabled(ctx context.Context, l slog.Level) bool {
	return h.inner.Enabled(ctx, l)
}

func (h *handler) Handle(ctx context.Context, r slog.Record) error {
	err := h.inner.Handle(ctx, r)

	rec := Record{
		Time:  r.Time,
		Level: r.Level.String(),
		Msg:   r.Message,
		Attrs: map[string]string{},
	}
	put := func(a slog.Attr) {
		if a.Key == logschema.FieldEvent {
			rec.Event = a.Value.String()
			return
		}
		rec.Attrs[a.Key] = fmt.Sprint(a.Value.Any())
	}
	for _, a := range h.attrs {
		put(a)
	}
	r.Attrs(func(a slog.Attr) bool {
		put(a)
		return true
	})
	if len(rec.Attrs) == 0 {
		rec.Attrs = nil
	}
	h.buf.add(rec)
	return err
}

func (h *handler) WithAttrs(attrs []slog.Attr) slog.Handler {
	merged := make([]slog.Attr, 0, len(h.attrs)+len(attrs))
	merged = append(merged, h.attrs...)
	merged = append(merged, attrs...)
	return &handler{buf: h.buf, inner: h.inner.WithAttrs(attrs), attrs: merged}
}

func (h *handler) WithGroup(name string) slog.Handler {
	return &handler{buf: h.buf, inner: h.inner.WithGroup(name), attrs: h.attrs}
}
