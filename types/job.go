package types

import "time"

// JobStatus is the lifecycle state of a single inference job.
type JobStatus string

const (
	JobStatusQueued    JobStatus = "queued"
	JobStatusDone      JobStatus = "done"
	JobStatusFailed    JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
)

// JobSource distinguishes requests that arrived locally from those drained
// from the Railway offline queue.
type JobSource string

const (
	JobSourceLocal   JobSource = "local"
	JobSourceOffline JobSource = "offline_queue"
)

// JobEntry is an immutable record of one inference job, written once when a
// job completes or fails, surfaced by the /status endpoint and dashboard.
type JobEntry struct {
	RequestID     string    `json:"request_id"`
	CorrelationID string    `json:"correlation_id"`
	Source        JobSource `json:"source"`
	Status        JobStatus `json:"status"`
	Modality      string    `json:"modality"`
	// ModelTier is the tier that actually handled the job (may differ from
	// what was requested, due to fallback).
	ModelTier       string    `json:"model_tier"`
	Priority        string    `json:"priority,omitempty"`
	MinTier         string    `json:"min_tier,omitempty"`
	EnqueuedAt      time.Time `json:"enqueued_at"`
	StartedAt       time.Time `json:"started_at"`
	FinishedAt      time.Time `json:"finished_at"`
	DurationMS      int64     `json:"duration_ms"`
	QueueWaitMS     int64     `json:"queue_wait_ms"`
	InferenceMS     int64     `json:"inference_ms"`
	TokensGenerated int       `json:"tokens_generated"`
	Deferred        bool      `json:"deferred,omitempty"`
	// ErrorCode is non-empty for failed jobs; matches api.ErrCode* constants.
	ErrorCode string `json:"error_code,omitempty"`
}
