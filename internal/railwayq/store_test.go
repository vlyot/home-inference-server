package railwayq_test

import (
	"context"
	"database/sql"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/railwayq"
	"github.com/ngkaichong/home-inference-server/queue"
)

func openTestDB(t *testing.T) *sql.DB {
	t.Helper()
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set; skipping Postgres tests")
	}
	db, err := railwayq.Open(dbURL)
	if err != nil {
		t.Fatalf("open test db: %v", err)
	}
	reset := func() {
		db.Exec("DELETE FROM jobs")            //nolint:errcheck
		db.Exec("DELETE FROM usage")           //nolint:errcheck
		db.Exec("DELETE FROM allowed_members") //nolint:errcheck
	}
	t.Cleanup(func() {
		reset()
		db.Close()
	})
	reset()
	return db
}

func newJob(id string) queue.QueueJob {
	return queue.QueueJob{
		ID:            id,
		CorrelationID: "req-" + id,
		Payload:       []byte(`{"modality":"text","text_input":{"prompt":"hello"}}`),
		MaxAttempts:   3,
	}
}

func newJobWithIdentity(id, identity string) queue.QueueJob {
	j := newJob(id)
	j.Identity = identity
	return j
}

func TestClaimReturnsNilWhenNoJobs(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	job, err := railwayq.Claim(ctx, db, "worker-1", 2*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if job != nil {
		t.Fatalf("expected nil job, got %+v", job)
	}
}

func TestClaimMarksJobProcessing(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJob("job-1")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	job, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second)
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if job == nil {
		t.Fatal("expected a job, got nil")
	}
	if job.Status != queue.QueueJobProcessing {
		t.Errorf("status = %q, want %q", job.Status, queue.QueueJobProcessing)
	}
	if job.ClaimedAt == nil {
		t.Error("claimed_at should be set")
	}
	if job.AttemptCount != 1 {
		t.Errorf("attempt_count = %d, want 1", job.AttemptCount)
	}
}

// TestPayloadRoundTripsAsObject is the regression test for the enqueue
// double-encoding bug: Enqueue used to json.Marshal(job.Payload), which
// base64-encodes a []byte into a JSON string, so the JSONB column held
// "eyJtb2RhbGl0eSI6..." instead of {"modality":...}. On claim the remote
// worker then failed with "cannot unmarshal string into api.InferRequest"
// and every offline job was dead on arrival.
func TestPayloadRoundTripsAsObject(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	original := []byte(`{"modality":"text","text_input":{"prompt":"round trip"},"priority":"low"}`)
	job := newJob("rt-1")
	job.Payload = original

	if err := railwayq.Enqueue(ctx, db, job); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	claimed, err := railwayq.Claim(ctx, db, "worker-rt", 5*time.Second)
	if err != nil || claimed == nil {
		t.Fatalf("claim: err=%v job=%v", err, claimed)
	}

	// The claimed payload must be a JSON object, not a JSON string.
	var probe map[string]any
	if err := json.Unmarshal(claimed.Payload, &probe); err != nil {
		t.Fatalf("claimed payload is not a JSON object (double-encoded?): %s", claimed.Payload)
	}
	if probe["modality"] != "text" {
		t.Errorf("modality = %v, want text; payload = %s", probe["modality"], claimed.Payload)
	}
}

func TestAckTransitionsToDone(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJob("job-2")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	job, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second)
	if err != nil || job == nil {
		t.Fatalf("claim: err=%v job=%v", err, job)
	}

	result := []byte(`{"output":"hi","tokens_generated":1}`)
	if err := railwayq.Ack(ctx, db, queue.AckRequest{
		JobID:         job.ID,
		Status:        queue.QueueJobDone,
		ResultPayload: result,
	}); err != nil {
		t.Fatalf("ack: %v", err)
	}

	// After ack, the job should no longer be claimable.
	job2, err := railwayq.Claim(ctx, db, "worker-1", time.Second)
	if err != nil {
		t.Fatalf("second claim: %v", err)
	}
	if job2 != nil {
		t.Errorf("expected no more jobs, got %+v", job2)
	}
}

func TestAckTransitionsToFailed(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJob("job-3")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	job, _ := railwayq.Claim(ctx, db, "worker-1", 5*time.Second)

	if err := railwayq.Ack(ctx, db, queue.AckRequest{
		JobID:     job.ID,
		Status:    queue.QueueJobFailed,
		ErrorCode: "internal_error",
	}); err != nil {
		t.Fatalf("ack: %v", err)
	}
}

func TestSweepExpiredReturnsJobToPending(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	// Insert a job that was already claimed 130 seconds ago (past visibility timeout).
	db.ExecContext(ctx, `
		INSERT INTO jobs (id, correlation_id, status, payload, claimed_at, attempt_count, max_attempts)
		VALUES ('stale-job', 'req-stale', 'processing', '{}', NOW() - INTERVAL '130 seconds', 1, 3)
	`) //nolint:errcheck

	// Claim triggers sweepExpired internally; the stale job should return to pending.
	job, err := railwayq.Claim(ctx, db, "worker-1", 3*time.Second)
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if job == nil {
		t.Fatal("expected swept job to be claimable, got nil")
	}
	if job.ID != "stale-job" {
		t.Errorf("expected stale-job, got %q", job.ID)
	}
}

func TestClaimSkipsPoisonPillJob(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	// Insert a job that has exhausted its max_attempts.
	db.ExecContext(ctx, `
		INSERT INTO jobs (id, correlation_id, status, payload, attempt_count, max_attempts)
		VALUES ('poison', 'req-poison', 'pending', '{}', 3, 3)
	`) //nolint:errcheck

	job, err := railwayq.Claim(ctx, db, "worker-1", 2*time.Second)
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if job != nil {
		t.Errorf("expected nil (poison pill skipped), got %+v", job)
	}
}

func TestEnqueueTagsIdentity(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJobWithIdentity("id-job", "user:alice")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	var got string
	if err := db.QueryRowContext(ctx, `SELECT identity FROM jobs WHERE id = 'id-job'`).Scan(&got); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if got != "user:alice" {
		t.Errorf("identity = %q, want user:alice", got)
	}
}

func TestGetResultOwnershipEnforced(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJobWithIdentity("owned", "user:alice")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	if _, err := railwayq.GetResult(ctx, db, "owned", "user:bob"); err != railwayq.ErrNotFound {
		t.Fatalf("foreign identity: want ErrNotFound, got %v", err)
	}
	res, err := railwayq.GetResult(ctx, db, "owned", "user:alice")
	if err != nil {
		t.Fatalf("owner GetResult: %v", err)
	}
	if res.Status != queue.QueueJobPending {
		t.Errorf("status = %q, want pending", res.Status)
	}
}

func TestGetResultUnknownIDIsNotFound(t *testing.T) {
	db := openTestDB(t)
	if _, err := railwayq.GetResult(context.Background(), db, "nope", "user:alice"); err != railwayq.ErrNotFound {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

func TestGetResultStatusProgression(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJobWithIdentity("prog", "user:alice")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	job, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second)
	if err != nil || job == nil {
		t.Fatalf("claim: err=%v job=%v", err, job)
	}
	res, _ := railwayq.GetResult(ctx, db, "prog", "user:alice")
	if res.Status != queue.QueueJobProcessing {
		t.Errorf("after claim: status = %q, want processing", res.Status)
	}

	if err := railwayq.Ack(ctx, db, queue.AckRequest{
		JobID:         "prog",
		Status:        queue.QueueJobDone,
		ResultPayload: []byte(`{"output":"hi"}`),
	}); err != nil {
		t.Fatalf("ack: %v", err)
	}
	res, _ = railwayq.GetResult(ctx, db, "prog", "user:alice")
	if res.Status != queue.QueueJobDone {
		t.Errorf("after ack: status = %q, want done", res.Status)
	}
	// Postgres JSONB normalises whitespace, so compare parsed values.
	var got map[string]string
	if err := json.Unmarshal(res.ResultPayload, &got); err != nil {
		t.Fatalf("result_payload not valid JSON: %s", res.ResultPayload)
	}
	if got["output"] != "hi" {
		t.Errorf("result_payload = %s, want output=hi", res.ResultPayload)
	}
}

func TestGetResultCarriesErrorCodeOnFailure(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if err := railwayq.Enqueue(ctx, db, newJobWithIdentity("failjob", "user:alice")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	if _, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second); err != nil {
		t.Fatalf("claim: %v", err)
	}
	if err := railwayq.Ack(ctx, db, queue.AckRequest{
		JobID:     "failjob",
		Status:    queue.QueueJobFailed,
		ErrorCode: "internal_error",
	}); err != nil {
		t.Fatalf("ack: %v", err)
	}
	res, _ := railwayq.GetResult(ctx, db, "failjob", "user:alice")
	if res.Status != queue.QueueJobFailed || res.ErrorCode != "internal_error" {
		t.Errorf("got status=%q code=%q", res.Status, res.ErrorCode)
	}
}

func TestCountPendingReflectsQueue(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	for _, id := range []string{"p1", "p2", "p3"} {
		if err := railwayq.Enqueue(ctx, db, newJob(id)); err != nil {
			t.Fatalf("enqueue %s: %v", id, err)
		}
	}
	n, err := railwayq.CountPending(ctx, db)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 3 {
		t.Errorf("CountPending = %d, want 3", n)
	}

	if _, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second); err != nil {
		t.Fatalf("claim: %v", err)
	}
	n, _ = railwayq.CountPending(ctx, db)
	if n != 2 {
		t.Errorf("after one claim CountPending = %d, want 2", n)
	}
}

func TestLastClaimAtTracksMostRecent(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	if last, err := railwayq.LastClaimAt(ctx, db); err != nil || last != nil {
		t.Fatalf("empty queue: last=%v err=%v", last, err)
	}

	if err := railwayq.Enqueue(ctx, db, newJob("lc1")); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	if _, err := railwayq.Claim(ctx, db, "worker-1", 5*time.Second); err != nil {
		t.Fatalf("claim: %v", err)
	}
	last, err := railwayq.LastClaimAt(ctx, db)
	if err != nil {
		t.Fatalf("LastClaimAt: %v", err)
	}
	if last == nil || time.Since(*last) > time.Minute {
		t.Errorf("LastClaimAt = %v, want ~now", last)
	}
}

func TestUsageUpsertIncrements(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	t.Cleanup(func() { db.Exec("DELETE FROM usage") }) //nolint:errcheck
	db.Exec("DELETE FROM usage")                       //nolint:errcheck

	if n, err := railwayq.UsageToday(ctx, db, "user:alice"); err != nil || n != 0 {
		t.Fatalf("initial usage: n=%d err=%v", n, err)
	}
	for i := 0; i < 3; i++ {
		if err := railwayq.IncrementUsage(ctx, db, "user:alice"); err != nil {
			t.Fatalf("increment %d: %v", i, err)
		}
	}
	n, err := railwayq.UsageToday(ctx, db, "user:alice")
	if err != nil {
		t.Fatalf("UsageToday: %v", err)
	}
	if n != 3 {
		t.Errorf("UsageToday = %d, want 3", n)
	}
	if other, _ := railwayq.UsageToday(ctx, db, "user:bob"); other != 0 {
		t.Errorf("bob usage = %d, want 0", other)
	}
}

func TestPurgeExpiredDeletesOldTerminalRows(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()

	db.ExecContext(ctx, `
		INSERT INTO jobs (id, correlation_id, status, payload, finished_at, attempt_count, max_attempts)
		VALUES
		 ('old-done',  'r1', 'done',   '{}', NOW() - INTERVAL '2 hours', 1, 3),
		 ('new-done',  'r2', 'done',   '{}', NOW(),                       1, 3),
		 ('old-pend',  'r3', 'pending','{}', NULL,                        0, 3)
	`) //nolint:errcheck

	n, err := railwayq.PurgeExpired(ctx, db, time.Hour)
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if n != 1 {
		t.Errorf("purged %d rows, want 1", n)
	}
	var remaining int
	db.QueryRowContext(ctx, `SELECT COUNT(*) FROM jobs`).Scan(&remaining) //nolint:errcheck
	if remaining != 2 {
		t.Errorf("remaining rows = %d, want 2 (new-done + old-pend)", remaining)
	}
}
