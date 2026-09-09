package queue

import (
	"encoding/json"
	"time"
)

// QueueJobStatus is the visibility state of a job in the Railway Postgres queue.
type QueueJobStatus string

const (
	QueueJobPending    QueueJobStatus = "pending"
	QueueJobProcessing QueueJobStatus = "processing"
	QueueJobDone       QueueJobStatus = "done"
	QueueJobFailed     QueueJobStatus = "failed"
)

// VisibilityTimeoutSeconds is how long a claimed job stays hidden before
// returning to "pending" if not acknowledged. Sized to exceed worst-case
// single-batch inference time plus network round-trip margin.
const VisibilityTimeoutSeconds = 120

// LongPollTimeoutSeconds is how long the local server holds the HTTP
// long-poll connection open while waiting for a new job from Railway.
const LongPollTimeoutSeconds = 30

// QueueJob is the full row shape stored in and retrieved from the Railway
// Postgres jobs table. Payload stores the raw JSON of api.InferRequest so
// the queue schema doesn't need migration when InferRequest fields change.
type QueueJob struct {
	ID            string         `json:"id"             db:"id"`
	CorrelationID string         `json:"correlation_id" db:"correlation_id"`
	Status        QueueJobStatus `json:"status"         db:"status"`
	// Payload is the serialised api.InferRequest body, stored as JSONB. It is
	// json.RawMessage so it travels as a JSON object on the wire (not a
	// base64 string) — safe for a non-Go client of /claim.
	Payload json.RawMessage `json:"payload"        db:"payload"`
	// ResultPayload is the serialised api.InferResponse or api.ErrorResponse.
	// Null until the job transitions to done or failed.
	ResultPayload json.RawMessage `json:"result_payload,omitempty" db:"result_payload"`
	CreatedAt     time.Time       `json:"created_at"     db:"created_at"`
	// ClaimedAt is set when the local server claims the job (status → processing).
	ClaimedAt  *time.Time `json:"claimed_at,omitempty"  db:"claimed_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty" db:"finished_at"`
	// AttemptCount tracks how many times this job has been claimed.
	// Used to detect poison-pill jobs that repeatedly fail.
	AttemptCount int `json:"attempt_count" db:"attempt_count"`
	MaxAttempts  int `json:"max_attempts"  db:"max_attempts"`
	// Identity is the resolved caller identity that enqueued this job
	// ("user:<id>" for a JWT, "apikey" for the static key). Result retrieval
	// is scoped to this value.
	Identity string `json:"identity,omitempty" db:"identity"`
}

// EnqueueRequest is the body a friends-and-family app POSTs to /enqueue.
// The relay generates the job ID; the client never supplies one.
type EnqueueRequest struct {
	CorrelationID string          `json:"correlation_id"`
	Payload       json.RawMessage `json:"payload"`
	MaxAttempts   int             `json:"max_attempts,omitempty"`
}

// EnqueueResponse is the relay's reply to a successful EnqueueRequest.
type EnqueueResponse struct {
	ID string `json:"id"`
}

// ResultResponse is the body of GET /result/{id}. Status is one of
// pending | processing | done | failed.
type ResultResponse struct {
	Status        QueueJobStatus  `json:"status"`
	ResultPayload json.RawMessage `json:"result_payload,omitempty"`
	ErrorCode     string          `json:"error_code,omitempty"`
}

// QueueStatusResponse is the body of GET /status. PCConnected is true when the
// PC has long-polled within a small multiple of the long-poll timeout.
type QueueStatusResponse struct {
	PCConnected  bool       `json:"pc_connected"`
	LastClaimAt  *time.Time `json:"last_claim_at,omitempty"`
	PendingDepth int        `json:"pending_depth"`
}

// ClaimRequest is the body sent by the local server to claim the next pending job.
type ClaimRequest struct {
	// WorkerID identifies the local server instance (e.g. hostname + PID).
	WorkerID string `json:"worker_id"`
}

// ClaimResponse is Railway's reply to a ClaimRequest.
// Job is nil when no pending jobs exist (long-poll returned empty).
type ClaimResponse struct {
	Job *QueueJob `json:"job,omitempty"`
}

// AckRequest is sent by the local server after finishing a claimed job.
type AckRequest struct {
	JobID  string         `json:"job_id"`
	Status QueueJobStatus `json:"status"` // done or failed
	// ResultPayload is the serialised api.InferResponse / api.ErrorResponse,
	// carried as a JSON object (json.RawMessage), not a base64 string.
	ResultPayload json.RawMessage `json:"result_payload"`
	ErrorCode     string          `json:"error_code,omitempty"`
}
