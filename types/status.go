package types

import "time"

// LoadedModel describes the model currently resident in VRAM.
// Nil on ServerStatus means no model is loaded (idle, ~0 VRAM footprint).
type LoadedModel struct {
	Descriptor ModelDescriptor `json:"descriptor"`
	LoadedAt   time.Time       `json:"loaded_at"`
	// IdleSince is when the last job using this model finished.
	// Zero value means a job is currently active.
	IdleSince time.Time `json:"idle_since,omitempty"`
	// GPULayers is the --n-gpu-layers value in effect: -1 full GPU, 0 CPU-only,
	// or a partial count. TotalLayers is the model's transformer block count.
	GPULayers   int `json:"gpu_layers"`
	TotalLayers int `json:"total_layers"`
}

// ServerStatus is the complete snapshot returned by GET /v1/status.
type ServerStatus struct {
	Timestamp  time.Time `json:"timestamp"`
	QueueDepth int       `json:"queue_depth"`
	// ActiveBatchSize is the number of jobs in the batcher's current un-flushed
	// batch. 0 when idle. Distinct from InFlight (jobs actually executing).
	ActiveBatchSize int `json:"active_batch_size"`
	// InFlight is the number of requests currently executing inference in the
	// worker pool. MaxParallel is the pool/--parallel ceiling.
	InFlight    int          `json:"in_flight"`
	MaxParallel int          `json:"max_parallel"`
	LoadedModel *LoadedModel `json:"loaded_model,omitempty"`
	// AvailableVRAMMB is the free VRAM reported by NVML. -1 when NVML is unavailable (CPU-only).
	AvailableVRAMMB int64 `json:"available_vram_mb"`
	// SelfVRAMMB is the VRAM currently held by this server's own inference
	// subprocesses. 0 when idle or NVML is unavailable.
	SelfVRAMMB   int64   `json:"self_vram_mb"`
	CPUPct       float64 `json:"cpu_pct"`
	TokensPerSec float64 `json:"tokens_per_sec"`
	// DeferredCount is the number of requests in RecentJobs that were handed to
	// the hosted relay because the server was under pressure. (There is no local
	// defer queue — deferred jobs are durable on the relay and drain back through
	// the offline-queue worker.)
	DeferredCount int `json:"defer_queue_depth"`
	// RecentJobs is the N most recent completed/failed entries, newest-first.
	RecentJobs  []JobEntry `json:"recent_jobs"`
	UptimeSince time.Time  `json:"uptime_since"`
	Version     string     `json:"version"`
	// State is "ok" normally or "draining" after POST /v1/admin/drain.
	State string `json:"state"`
}
