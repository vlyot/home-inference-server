package railwayq

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/ngkaichong/home-inference-server/queue"
)

// ErrNotFound is returned by GetResult when no job matches the id for the
// calling identity. A foreign job and an unknown id are deliberately
// indistinguishable.
var ErrNotFound = errors.New("railwayq: job not found")

// Enqueue inserts a new job with status=pending, tagged with job.Identity.
// job.Payload already holds raw JSON (the serialised api.InferRequest); it is
// written straight into the JSONB column. Marshaling it again would base64-encode
// the bytes into a JSON string and double-encode the payload.
func Enqueue(ctx context.Context, db *sql.DB, job queue.QueueJob) error {
	if !json.Valid(job.Payload) {
		return fmt.Errorf("payload is not valid JSON")
	}
	_, err := db.ExecContext(ctx, `
		INSERT INTO jobs (id, correlation_id, status, payload, max_attempts, identity)
		VALUES ($1, $2, 'pending', $3, $4, $5)`,
		job.ID, job.CorrelationID, string(job.Payload), job.MaxAttempts, job.Identity,
	)
	return err
}

// Claim selects the oldest pending job, marks it processing, and returns it.
// It polls every second up to longPollTimeout. Returns (nil, nil) if no job
// arrives within the timeout. Runs the visibility-timeout sweep before each poll.
func Claim(ctx context.Context, db *sql.DB, workerID string, longPollTimeout time.Duration) (*queue.QueueJob, error) {
	deadline := time.Now().Add(longPollTimeout)
	for {
		if err := sweepExpired(ctx, db); err != nil {
			return nil, fmt.Errorf("sweep expired: %w", err)
		}

		job, err := tryClaimOne(ctx, db)
		if err != nil {
			return nil, err
		}
		if job != nil {
			return job, nil
		}

		if time.Now().After(deadline) {
			return nil, nil
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

func tryClaimOne(ctx context.Context, db *sql.DB) (*queue.QueueJob, error) {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback() //nolint:errcheck

	row := tx.QueryRowContext(ctx, `
		SELECT id, correlation_id, status, payload, attempt_count, max_attempts, created_at, identity
		FROM jobs
		WHERE status = 'pending' AND attempt_count < max_attempts
		ORDER BY created_at ASC
		LIMIT 1
		FOR UPDATE SKIP LOCKED`,
	)

	var j queue.QueueJob
	var payload []byte
	err = row.Scan(&j.ID, &j.CorrelationID, &j.Status, &payload,
		&j.AttemptCount, &j.MaxAttempts, &j.CreatedAt, &j.Identity)
	if err == sql.ErrNoRows {
		tx.Rollback() //nolint:errcheck
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	j.Payload = payload

	now := time.Now()
	j.ClaimedAt = &now
	j.Status = queue.QueueJobProcessing
	j.AttemptCount++

	_, err = tx.ExecContext(ctx, `
		UPDATE jobs
		SET status = 'processing', claimed_at = $1, attempt_count = $2
		WHERE id = $3`,
		now, j.AttemptCount, j.ID,
	)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return &j, nil
}

// Ack updates a claimed job to done or failed with its result payload and
// (for failures) its error code. No-ops if the job is no longer processing
// (idempotent on retry).
func Ack(ctx context.Context, db *sql.DB, req queue.AckRequest) error {
	now := time.Now()
	// req.ResultPayload is already raw JSON (a serialised api.InferResponse or
	// api.ErrorResponse); write it straight into JSONB, do not re-marshal.
	var resultPayload any
	if len(req.ResultPayload) > 0 {
		if !json.Valid(req.ResultPayload) {
			return fmt.Errorf("result payload is not valid JSON")
		}
		resultPayload = string(req.ResultPayload)
	}
	var errCode any
	if req.ErrorCode != "" {
		errCode = req.ErrorCode
	}
	_, err := db.ExecContext(ctx, `
		UPDATE jobs
		SET status = $1, result_payload = $2, error_code = $3, finished_at = $4
		WHERE id = $5 AND status = 'processing'`,
		string(req.Status), resultPayload, errCode, now, req.JobID,
	)
	return err
}

// GetResult returns the current status and (when terminal) the result payload
// for a job, scoped to the calling identity. Returns ErrNotFound if no job
// matches the id for that identity.
func GetResult(ctx context.Context, db *sql.DB, id, identity string) (queue.ResultResponse, error) {
	row := db.QueryRowContext(ctx, `
		SELECT status, result_payload, error_code
		FROM jobs
		WHERE id = $1 AND identity = $2`,
		id, identity,
	)
	var (
		status  string
		payload []byte
		errCode sql.NullString
	)
	if err := row.Scan(&status, &payload, &errCode); err != nil {
		if err == sql.ErrNoRows {
			return queue.ResultResponse{}, ErrNotFound
		}
		return queue.ResultResponse{}, err
	}
	return queue.ResultResponse{
		Status:        queue.QueueJobStatus(status),
		ResultPayload: json.RawMessage(payload),
		ErrorCode:     errCode.String,
	}, nil
}

// CountPending returns the number of jobs waiting to be claimed.
func CountPending(ctx context.Context, db *sql.DB) (int, error) {
	var n int
	err := db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM jobs WHERE status = 'pending'`).Scan(&n)
	return n, err
}

// LastClaimAt returns the most recent time any job was claimed, or nil if none
// ever has been.
func LastClaimAt(ctx context.Context, db *sql.DB) (*time.Time, error) {
	var t sql.NullTime
	if err := db.QueryRowContext(ctx,
		`SELECT MAX(claimed_at) FROM jobs`).Scan(&t); err != nil {
		return nil, err
	}
	if !t.Valid {
		return nil, nil
	}
	return &t.Time, nil
}

// Heartbeat records that workerID polled /claim just now (whether or not it got
// a job). GET /status uses this to report pc_connected.
func Heartbeat(ctx context.Context, db *sql.DB, workerID string) error {
	_, err := db.ExecContext(ctx, `
		INSERT INTO worker_heartbeat (worker_id, last_poll_at)
		VALUES ($1, NOW())
		ON CONFLICT (worker_id) DO UPDATE SET last_poll_at = NOW()`,
		workerID,
	)
	return err
}

// LastHeartbeat returns the most recent worker poll time across all workers, or
// nil if none has ever polled.
func LastHeartbeat(ctx context.Context, db *sql.DB) (*time.Time, error) {
	var t sql.NullTime
	if err := db.QueryRowContext(ctx,
		`SELECT MAX(last_poll_at) FROM worker_heartbeat`).Scan(&t); err != nil {
		return nil, err
	}
	if !t.Valid {
		return nil, nil
	}
	return &t.Time, nil
}

// PurgeExpired deletes terminal (done/failed) jobs whose finished_at is older
// than ttl. Returns the number of rows removed.
func PurgeExpired(ctx context.Context, db *sql.DB, ttl time.Duration) (int64, error) {
	res, err := db.ExecContext(ctx, `
		DELETE FROM jobs
		WHERE status IN ('done', 'failed')
		  AND finished_at IS NOT NULL
		  AND finished_at < NOW() - make_interval(secs => $1)`,
		ttl.Seconds(),
	)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// IncrementUsage bumps the caller's job count for the current UTC day.
func IncrementUsage(ctx context.Context, db *sql.DB, identity string) error {
	_, err := db.ExecContext(ctx, `
		INSERT INTO usage (identity, day, count)
		VALUES ($1, CURRENT_DATE, 1)
		ON CONFLICT (identity, day) DO UPDATE SET count = usage.count + 1`,
		identity,
	)
	return err
}

// UsageToday returns how many jobs the caller has enqueued during the current
// UTC day.
func UsageToday(ctx context.Context, db *sql.DB, identity string) (int, error) {
	var n int
	err := db.QueryRowContext(ctx,
		`SELECT COALESCE(count, 0) FROM usage WHERE identity = $1 AND day = CURRENT_DATE`,
		identity,
	).Scan(&n)
	if err == sql.ErrNoRows {
		return 0, nil
	}
	return n, err
}

// IsAllowedMember reports whether a Stack Auth user id is on the invite list.
// This is the "no public signup" gate: Stack Auth may let anyone create an
// account, but only ids inserted into allowed_members can use the relay.
func IsAllowedMember(ctx context.Context, db *sql.DB, stackUserID string) (bool, error) {
	var one int
	err := db.QueryRowContext(ctx,
		`SELECT 1 FROM allowed_members WHERE stack_user_id = $1`, stackUserID,
	).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// sweepExpired returns stale processing jobs to pending after the visibility timeout.
func sweepExpired(ctx context.Context, db *sql.DB) error {
	_, err := db.ExecContext(ctx, `
		UPDATE jobs
		SET status = 'pending', claimed_at = NULL
		WHERE status = 'processing'
		  AND claimed_at < NOW() - INTERVAL '120 seconds'`,
	)
	return err
}
