package railwayq

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/queue"
)

func handleEnqueue(db *sql.DB, cfg Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, cfg.MaxBodyBytes)

		var req queue.EnqueueRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				writeErr(w, http.StatusRequestEntityTooLarge, api.ErrCodeInvalidRequest, "request body too large")
				return
			}
			writeErr(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON")
			return
		}
		if req.CorrelationID == "" || len(req.Payload) == 0 {
			writeErr(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "correlation_id and payload are required")
			return
		}

		pending, err := CountPending(r.Context(), db)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "queue check failed")
			return
		}
		if pending >= cfg.MaxPendingDepth {
			writeErr(w, http.StatusServiceUnavailable, api.ErrCodeQueueFull, "queue is full, try again later")
			return
		}

		identity := identityFrom(r.Context())
		maxAttempts := req.MaxAttempts
		if maxAttempts <= 0 {
			maxAttempts = 3
		}
		job := queue.QueueJob{
			ID:            strings.ReplaceAll(uuid.New().String(), "-", ""),
			CorrelationID: req.CorrelationID,
			Payload:       req.Payload,
			MaxAttempts:   maxAttempts,
			Identity:      identity,
		}
		if err := Enqueue(r.Context(), db, job); err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "enqueue failed")
			return
		}
		if err := IncrementUsage(r.Context(), db, identity); err != nil {
			// The job is already durably enqueued; a usage-counter miss is not
			// worth failing the request over.
			_ = err
		}
		writeJSON(w, http.StatusCreated, queue.EnqueueResponse{ID: job.ID})
	}
}

func handleClaim(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req queue.ClaimRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "malformed JSON", http.StatusBadRequest)
			return
		}

		timeout := time.Duration(queue.LongPollTimeoutSeconds)*time.Second + 2*time.Second
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()

		// Record the poll before the (possibly long) claim so /status reflects a
		// connected PC even across an idle stretch or a TTL purge of old rows.
		if req.WorkerID != "" {
			if err := Heartbeat(ctx, db, req.WorkerID); err != nil {
				slog.Warn("worker heartbeat failed", "worker_id", req.WorkerID, "err", err)
			}
		}

		job, err := Claim(ctx, db, req.WorkerID, time.Duration(queue.LongPollTimeoutSeconds)*time.Second)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			http.Error(w, "claim failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(queue.ClaimResponse{Job: job}) //nolint:errcheck
	}
}

func handleAck(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req queue.AckRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "malformed JSON", http.StatusBadRequest)
			return
		}
		if req.JobID == "" {
			http.Error(w, "job_id is required", http.StatusBadRequest)
			return
		}
		if req.Status != queue.QueueJobDone && req.Status != queue.QueueJobFailed {
			http.Error(w, "status must be done or failed", http.StatusBadRequest)
			return
		}
		if err := Ack(r.Context(), db, req); err != nil {
			http.Error(w, "ack failed", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}
}

func handleResult(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimPrefix(r.URL.Path, "/result/")
		if id == "" || strings.Contains(id, "/") {
			writeErr(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed job id")
			return
		}
		res, err := GetResult(r.Context(), db, id, identityFrom(r.Context()))
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				// Unknown id, another caller's job, or a completed job already
				// aged past RESULT_TTL_SECONDS — deliberately indistinguishable.
				writeErr(w, http.StatusNotFound, api.ErrCodeNotFound, "no such job (or its result has expired)")
				return
			}
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "result lookup failed")
			return
		}
		writeJSON(w, http.StatusOK, res)
	}
}

func handleStatus(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		lastClaim, err := LastClaimAt(r.Context(), db)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "status lookup failed")
			return
		}
		lastPoll, err := LastHeartbeat(r.Context(), db)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "status lookup failed")
			return
		}
		pending, err := CountPending(r.Context(), db)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "status lookup failed")
			return
		}
		// pc_connected: the PC has polled /claim (heartbeat) or claimed a job
		// within ~2× the long-poll interval — whichever is more recent.
		recent := lastPoll
		if lastClaim != nil && (recent == nil || lastClaim.After(*recent)) {
			recent = lastClaim
		}
		connected := recent != nil &&
			time.Since(*recent) < 2*time.Duration(queue.LongPollTimeoutSeconds)*time.Second
		writeJSON(w, http.StatusOK, queue.QueueStatusResponse{
			PCConnected:  connected,
			LastClaimAt:  lastClaim,
			PendingDepth: pending,
		})
	}
}
