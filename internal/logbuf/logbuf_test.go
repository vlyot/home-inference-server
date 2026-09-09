package logbuf_test

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/logschema"
)

type spyHandler struct{ seen int }

func (s *spyHandler) Enabled(context.Context, slog.Level) bool  { return true }
func (s *spyHandler) Handle(context.Context, slog.Record) error { s.seen++; return nil }
func (s *spyHandler) WithAttrs([]slog.Attr) slog.Handler        { return s }
func (s *spyHandler) WithGroup(string) slog.Handler             { return s }

func newLogger(b *logbuf.Buffer, inner slog.Handler) *slog.Logger {
	if inner == nil {
		inner = &spyHandler{}
	}
	return slog.New(b.Handler(inner))
}

func TestBufferCapByCount(t *testing.T) {
	b := logbuf.NewBuffer(5000, 24*time.Hour)
	lg := newLogger(b, nil)
	for i := 0; i < 6000; i++ {
		lg.Info("msg", "i", i)
	}
	recs := b.Query("", "", 0, 100000)
	if len(recs) != 5000 {
		t.Fatalf("want 5000 records, got %d", len(recs))
	}
	if recs[0].Attrs["i"] != "5999" {
		t.Fatalf("want newest-first (i=5999), got i=%s", recs[0].Attrs["i"])
	}
}

func TestBufferPruneByAge(t *testing.T) {
	b := logbuf.NewBuffer(5000, 24*time.Hour)

	// Directly exercise age pruning: an old record followed by a fresh insert.
	inject(b, logbuf.Record{Time: time.Now().Add(-25 * time.Hour), Level: "INFO", Msg: "old"})

	newLogger(b, nil).Info("fresh")

	recs := b.Query("", "", 0, 100)
	for _, r := range recs {
		if r.Msg == "old" {
			t.Fatalf("stale record survived age prune")
		}
	}
	if len(recs) != 1 || recs[0].Msg != "fresh" {
		t.Fatalf("want only the fresh record, got %+v", recs)
	}
}

func TestQueryFilterByLevel(t *testing.T) {
	b := logbuf.NewBuffer(100, time.Hour)
	lg := newLogger(b, nil)
	lg.Info("i1")
	lg.Warn("w1")
	lg.Error("e1")

	recs := b.Query("ERROR", "", 0, 100)
	if len(recs) != 1 || recs[0].Level != "ERROR" {
		t.Fatalf("want 1 ERROR record, got %+v", recs)
	}
	// case-insensitive
	if got := b.Query("error", "", 0, 100); len(got) != 1 {
		t.Fatalf("level filter should be case-insensitive, got %d", len(got))
	}
}

func TestQueryFilterByEvent(t *testing.T) {
	b := logbuf.NewBuffer(100, time.Hour)
	lg := newLogger(b, nil)
	lg.Error("boom", slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)))
	lg.Info("loaded", slog.String(logschema.FieldEvent, string(logschema.EventModelLoaded)))

	recs := b.Query("", string(logschema.EventJobFailed), 0, 100)
	if len(recs) != 1 || recs[0].Event != string(logschema.EventJobFailed) {
		t.Fatalf("want 1 job.failed record, got %+v", recs)
	}
}

func TestQueryFilterBySince(t *testing.T) {
	b := logbuf.NewBuffer(100, 24*time.Hour)
	inject(b, logbuf.Record{Time: time.Now().Add(-2 * time.Hour), Level: "INFO", Msg: "old"})
	newLogger(b, nil).Info("new")

	recs := b.Query("", "", time.Hour, 100)
	if len(recs) != 1 || recs[0].Msg != "new" {
		t.Fatalf("want only records within 1h, got %+v", recs)
	}
}

func TestHandlerTeesToInner(t *testing.T) {
	spy := &spyHandler{}
	b := logbuf.NewBuffer(100, time.Hour)
	lg := slog.New(b.Handler(spy))
	lg.Info("a")
	lg.Warn("b")

	if spy.seen != 2 {
		t.Fatalf("inner handler saw %d records, want 2", spy.seen)
	}
	if len(b.Query("", "", 0, 100)) != 2 {
		t.Fatalf("buffer should also hold 2 records")
	}
}

func TestHandlerExtractsEventAttr(t *testing.T) {
	b := logbuf.NewBuffer(100, time.Hour)
	lg := newLogger(b, nil)
	lg.Error("job failed",
		slog.String(logschema.FieldEvent, "job.failed"),
		slog.String("error_code", "timeout"),
	)

	recs := b.Query("", "", 0, 100)
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d", len(recs))
	}
	r := recs[0]
	if r.Event != "job.failed" {
		t.Fatalf("event not lifted into Record.Event: %q", r.Event)
	}
	if _, dup := r.Attrs["event"]; dup {
		t.Fatalf("event should not be duplicated into Attrs")
	}
	if r.Attrs["error_code"] != "timeout" {
		t.Fatalf("other attrs should remain: %+v", r.Attrs)
	}
}

// inject appends a pre-built record via a fresh log call is not possible with a
// backdated time, so we route through a tiny handler that forwards one record.
func inject(b *logbuf.Buffer, rec logbuf.Record) {
	h := b.Handler(&spyHandler{})
	sr := slog.NewRecord(rec.Time, levelFromString(rec.Level), rec.Msg, 0)
	if rec.Event != "" {
		sr.AddAttrs(slog.String("event", rec.Event))
	}
	for k, v := range rec.Attrs {
		sr.AddAttrs(slog.String(k, v))
	}
	_ = h.Handle(context.Background(), sr)
}

func levelFromString(s string) slog.Level {
	switch s {
	case "DEBUG":
		return slog.LevelDebug
	case "WARN":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
