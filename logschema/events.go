package logschema

import "time"

// EventKind is the machine-readable event name carried in every log line.
// Do not rename values without a migration — the dashboard and tooling key on these.
type EventKind string

const (
	// Job lifecycle
	EventJobReceived   EventKind = "job.received"
	EventJobQueued     EventKind = "job.queued"
	EventJobDispatched EventKind = "job.dispatched"
	EventJobInferring  EventKind = "job.inferring"
	EventJobDone       EventKind = "job.done"
	EventJobFailed     EventKind = "job.failed"

	// Batch dispatch (concurrent worker pool)
	EventBatchDispatched EventKind = "batch.dispatched"

	// Model lifecycle
	EventModelLoading  EventKind = "model.loading"
	EventModelLoaded   EventKind = "model.loaded"
	EventModelEvicting EventKind = "model.evicting"
	EventModelEvicted  EventKind = "model.evicted"
	EventModelOOM      EventKind = "model.oom"
	EventModelFallback EventKind = "model.fallback"

	// VRAM
	EventVRAMCheck    EventKind = "vram.check"
	EventVRAMPressure EventKind = "vram.pressure"

	// Offline queue
	EventQueuePoll     EventKind = "queue.poll"
	EventQueueClaimed  EventKind = "queue.claimed"
	EventQueueAcked    EventKind = "queue.acked"
	EventQueueReturned EventKind = "queue.returned"

	// Server lifecycle
	EventServerStart   EventKind = "server.start"
	EventServerStop    EventKind = "server.stop"
	EventAdminDrain    EventKind = "admin.drain"
	EventAdminResume   EventKind = "admin.resume"
	EventAdminRestart  EventKind = "admin.restart"
	EventWorkerPaused  EventKind = "worker.paused"
	EventWorkerResumed EventKind = "worker.resumed"

	// Contention
	EventContention EventKind = "contention.detected"

	// Pressure routing
	EventQualityDegraded EventKind = "quality.degraded"
	EventSpeedFloor      EventKind = "speed.floor"
	EventJobDeferred     EventKind = "job.deferred"

	// Remote relay
	EventRateLimited   EventKind = "relay.rate_limited"
	EventQuotaExceeded EventKind = "relay.quota_exceeded"
	EventResultServed  EventKind = "relay.result_served"
	EventResultsPurged EventKind = "relay.results_purged"
)

// CorrelationIDPrefix is prepended to UUID v4 values for human-scannable log output.
// Format: "req-<uuid4-no-hyphens>". Example: "req-4f9a2c1b8e3d4a5f9b2c1b8e3d4a5f9b"
const CorrelationIDPrefix = "req-"

// slog attribute key strings. Use these constants instead of raw strings to
// keep the field registry in one place.
const (
	FieldEvent           = "event"
	FieldCorrelationID   = "correlation_id"
	FieldRequestID       = "request_id"
	FieldModality        = "modality"
	FieldModelTier       = "model_tier"
	FieldDurationMS      = "duration_ms"
	FieldTokensGenerated = "tokens_generated"
	FieldErrorCode       = "error_code"
	FieldQueueDepth      = "queue_depth"
	FieldBatchSize       = "batch_size"
	FieldVRAMAvailMB     = "vram_avail_mb"
	FieldVRAMRequiredMB  = "vram_required_mb"
	FieldCPUPct          = "cpu_pct"
	FieldTokensPerSec    = "tokens_per_sec"
	FieldDeferQueueDepth = "defer_queue_depth"
	FieldIdentity        = "identity"
	FieldJobID           = "job_id"
	FieldCount           = "count"
	FieldInFlight        = "in_flight"
	FieldMaxParallel     = "max_parallel"
)

// LogEvent is the authoritative field registry for structured log output.
// It is never instantiated at runtime — emit logs via log/slog using the
// Field* constants as attribute keys:
//
//	slog.Info("job done",
//	    slog.String(logschema.FieldEvent,         string(logschema.EventJobDone)),
//	    slog.String(logschema.FieldCorrelationID, entry.CorrelationID),
//	    slog.Int64(logschema.FieldDurationMS,     entry.DurationMS),
//	)
type LogEvent struct {
	Timestamp       time.Time `json:"time"`
	Level           string    `json:"level"`
	Event           EventKind `json:"event"`
	CorrelationID   string    `json:"correlation_id,omitempty"`
	RequestID       string    `json:"request_id,omitempty"`
	Modality        string    `json:"modality,omitempty"`
	ModelTier       string    `json:"model_tier,omitempty"`
	DurationMS      int64     `json:"duration_ms,omitempty"`
	TokensGenerated int       `json:"tokens_generated,omitempty"`
	ErrorCode       string    `json:"error_code,omitempty"`
	Message         string    `json:"msg"`
}
