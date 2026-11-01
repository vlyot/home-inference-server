package railwayq

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/queue"
)

// TestAckRecordsFinishForTheSweeper checks the link the TTL sweeper relies on:
// a successful Ack records its finish time, and a no-op Ack does not.
func TestAckRecordsFinishForTheSweeper(t *testing.T) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set; skipping Postgres tests")
	}
	db, err := Open(dbURL)
	if err != nil {
		t.Fatalf("open test db: %v", err)
	}
	t.Cleanup(func() {
		db.Exec("DELETE FROM jobs") //nolint:errcheck
		db.Close()
	})
	db.Exec("DELETE FROM jobs") //nolint:errcheck
	ctx := context.Background()

	finishedJobs.take()
	defer finishedJobs.take()

	// Acking a job that isn't processing changes nothing, so records nothing.
	if err := Ack(ctx, db, queue.AckRequest{JobID: "missing", Status: queue.QueueJobDone}); err != nil {
		t.Fatalf("ack missing: %v", err)
	}
	if !finishedJobs.peek().IsZero() {
		t.Fatal("no-op ack recorded a finish")
	}

	if err := Enqueue(ctx, db, queue.QueueJob{ID: "j1", CorrelationID: "r1",
		Payload: []byte(`{"modality":"text"}`), MaxAttempts: 3}); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	job, err := Claim(ctx, db, "w", time.Second)
	if err != nil || job == nil {
		t.Fatalf("claim: %v, %v", job, err)
	}
	before := time.Now()
	if err := Ack(ctx, db, queue.AckRequest{JobID: "j1", Status: queue.QueueJobDone,
		ResultPayload: []byte(`{}`)}); err != nil {
		t.Fatalf("ack: %v", err)
	}
	got := finishedJobs.peek()
	if got.IsZero() || got.Before(before.Add(-time.Second)) {
		t.Fatalf("finish recorded as %v, want about %v", got, before)
	}
}
