package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/internal/server"
	"github.com/ngkaichong/home-inference-server/logschema"
	queuepkg "github.com/ngkaichong/home-inference-server/queue"
	"github.com/ngkaichong/home-inference-server/types"
)

const (
	backoffMin = time.Second
	backoffMax = 30 * time.Second

	// maxJobWait bounds how long the worker waits for a claimed job to finish
	// before acking it failed. Kept under the relay's 120 s visibility timeout so
	// a stuck job is surfaced as failed rather than silently re-pended and re-run.
	maxJobWait = 100 * time.Second

	// ackRetries is how many times a lost /ack POST is retried before giving up
	// (a lost ack on a done job otherwise causes a full re-run after the
	// visibility timeout).
	ackRetries = 3

	// pausePoll is how often the Run loop re-checks the pause flag while paused
	// (a fallback tick; Resume wakes it immediately via resumeCh).
	pausePoll = 2 * time.Second
)

// Client long-polls a Railway queue backend and feeds claimed jobs into the
// local inference pipeline.
type Client struct {
	baseURL  string
	apiKey   string
	workerID string
	q        *queue.Queue
	http     *http.Client

	// recordJob is called after each job completes so the local server can
	// include offline jobs in /v1/status history.
	recordJob func(types.JobEntry)

	// paused, when set (via Pause), stops the Run loop from claiming new relay
	// jobs. A job already in flight when Pause is called still finishes and acks.
	paused atomic.Bool
	// resumeCh wakes the Run loop out of its pause sleep the moment Resume is
	// called, so claiming restarts promptly rather than after pausePoll.
	resumeCh chan struct{}
}

// Pause stops the Run loop from claiming new relay jobs. An in-flight job is
// unaffected. Toggled by the admin drain endpoint.
func (c *Client) Pause() {
	if !c.paused.Swap(true) {
		slog.Info("offline worker paused",
			slog.String(logschema.FieldEvent, string(logschema.EventWorkerPaused)))
	}
}

// Resume re-enables relay-job claiming after Pause and wakes the loop.
func (c *Client) Resume() {
	if c.paused.Swap(false) {
		slog.Info("offline worker resumed",
			slog.String(logschema.FieldEvent, string(logschema.EventWorkerResumed)))
		select {
		case c.resumeCh <- struct{}{}:
		default:
		}
	}
}

// Paused reports whether relay-job claiming is currently paused.
func (c *Client) Paused() bool { return c.paused.Load() }

// New constructs a Client. recordJob may be nil (history tracking is skipped).
func New(baseURL, apiKey string, q *queue.Queue, recordJob func(types.JobEntry)) *Client {
	return &Client{
		baseURL:   strings.TrimRight(baseURL, "/"),
		apiKey:    apiKey,
		workerID:  WorkerID(),
		q:         q,
		http:      &http.Client{Timeout: 60 * time.Second},
		recordJob: recordJob,
		resumeCh:  make(chan struct{}, 1),
	}
}

// Run is the long-poll loop. It exits only when ctx is cancelled.
func (c *Client) Run(ctx context.Context) {
	backoff := backoffMin
	for {
		if ctx.Err() != nil {
			return
		}

		if c.paused.Load() {
			select {
			case <-ctx.Done():
				return
			case <-c.resumeCh:
			case <-time.After(pausePoll):
			}
			continue
		}

		job, err := c.claim(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			slog.Warn("railway claim error, backing off",
				slog.String(logschema.FieldEvent, string(logschema.EventQueuePoll)),
				"err", err,
				"backoff", backoff.String(),
			)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			backoff = min(backoff*2, backoffMax)
			continue
		}

		// Reset backoff on a successful round-trip (even if no job).
		backoff = backoffMin

		if job == nil {
			// Long-poll timeout — no jobs. Loop immediately.
			continue
		}

		slog.Info("railway job claimed",
			slog.String(logschema.FieldEvent, string(logschema.EventQueueClaimed)),
			slog.String(logschema.FieldCorrelationID, job.CorrelationID),
			"job_id", job.ID,
			"attempt", job.AttemptCount,
		)

		c.processJob(ctx, job)
	}
}

func (c *Client) claim(ctx context.Context) (*queuepkg.QueueJob, error) {
	var resp queuepkg.ClaimResponse
	if err := c.postJSON(ctx, "/claim", queuepkg.ClaimRequest{WorkerID: c.workerID}, &resp); err != nil {
		return nil, err
	}
	return resp.Job, nil
}

func (c *Client) processJob(ctx context.Context, job *queuepkg.QueueJob) {
	var inferReq api.InferRequest
	if err := json.Unmarshal(job.Payload, &inferReq); err != nil {
		slog.Error("failed to decode railway job payload",
			"job_id", job.ID,
			"err", err,
		)
		c.ackFailed(ctx, job.ID, job.CorrelationID, api.ErrCodeInvalidRequest, nil)
		return
	}

	requestID := strings.ReplaceAll(uuid.New().String(), "-", "")
	if inferReq.CorrelationID == "" {
		inferReq.CorrelationID = job.CorrelationID
	}

	// Bound the wait for this job. Prefer the caller's timeout_ms; otherwise the
	// worker cap, itself kept under the relay's visibility timeout.
	wait := maxJobWait
	if inferReq.TimeoutMS > 0 {
		if d := time.Duration(inferReq.TimeoutMS) * time.Millisecond; d < wait {
			wait = d
		}
	}
	jobCtx, cancelJob := context.WithTimeout(ctx, wait)
	defer cancelJob()

	resultCh := make(chan queue.Result, 1)
	enqueuedAt := time.Now()
	qJob := queue.Job{
		ID:            requestID,
		CorrelationID: inferReq.CorrelationID,
		Req:           server.TranslateInferRequest(inferReq, requestID),
		ResultCh:      resultCh,
	}

	if err := c.q.Enqueue(jobCtx, qJob); err != nil {
		if ctx.Err() != nil {
			// Server shutting down — leave job to visibility-timeout recovery.
			return
		}
		c.ackFailed(ctx, job.ID, inferReq.CorrelationID, api.ErrCodeOverloaded, nil)
		return
	}

	var result queue.Result
	select {
	case result = <-resultCh:
	case <-ctx.Done():
		// Shutdown mid-job — let visibility timeout return the job to pending.
		return
	case <-jobCtx.Done():
		// Job exceeded its deadline; ack it failed so the relay row does not sit
		// hidden until the visibility timeout and get re-run.
		c.ackFailed(ctx, job.ID, inferReq.CorrelationID, api.ErrCodeTimeout, nil)
		return
	}

	finishedAt := time.Now()
	durationMS := finishedAt.Sub(enqueuedAt).Milliseconds()

	if result.Err != nil {
		errCode := api.ErrCodeInternal
		if be, ok := result.Err.(*backend.BackendError); ok {
			errCode = be.Code
		}
		errResp, _ := json.Marshal(api.ErrorResponse{
			Code:          errCode,
			Message:       result.Err.Error(),
			CorrelationID: inferReq.CorrelationID,
		})
		c.ackFailed(ctx, job.ID, inferReq.CorrelationID, errCode, errResp)

		if c.recordJob != nil {
			c.recordJob(types.JobEntry{
				RequestID:     requestID,
				CorrelationID: inferReq.CorrelationID,
				Source:        types.JobSourceOffline,
				Status:        types.JobStatusFailed,
				Modality:      string(inferReq.Modality),
				Priority:      inferReq.Priority,
				MinTier:       inferReq.MinTier,
				EnqueuedAt:    enqueuedAt,
				StartedAt:     enqueuedAt,
				FinishedAt:    finishedAt,
				DurationMS:    durationMS,
				ErrorCode:     errCode,
			})
		}
		return
	}

	resultPayload, _ := json.Marshal(api.InferResponse{
		RequestID:       requestID,
		CorrelationID:   inferReq.CorrelationID,
		Modality:        inferReq.Modality,
		Output:          result.Output,
		Reasoning:       result.Reasoning,
		TokensGenerated: result.TokensGenerated,
		DurationMS:      durationMS,
		ModelTier:       result.ModelTier,
		FinishedAt:      finishedAt,
	})

	ackReq := queuepkg.AckRequest{
		JobID:         job.ID,
		Status:        queuepkg.QueueJobDone,
		ResultPayload: resultPayload,
	}
	if err := c.ackWithRetry(ctx, ackReq, inferReq.CorrelationID); err != nil {
		slog.Warn("railway ack failed after retries",
			slog.String(logschema.FieldEvent, string(logschema.EventQueueReturned)),
			slog.String(logschema.FieldCorrelationID, inferReq.CorrelationID),
			"job_id", job.ID,
			"err", err,
		)
	} else {
		slog.Info("railway job acked",
			slog.String(logschema.FieldEvent, string(logschema.EventQueueAcked)),
			slog.String(logschema.FieldCorrelationID, inferReq.CorrelationID),
			"job_id", job.ID,
		)
	}

	if c.recordJob != nil {
		c.recordJob(types.JobEntry{
			RequestID:       requestID,
			CorrelationID:   inferReq.CorrelationID,
			Source:          types.JobSourceOffline,
			Status:          types.JobStatusDone,
			Modality:        string(inferReq.Modality),
			ModelTier:       result.ModelTier,
			Priority:        inferReq.Priority,
			MinTier:         inferReq.MinTier,
			EnqueuedAt:      enqueuedAt,
			StartedAt:       enqueuedAt,
			FinishedAt:      finishedAt,
			DurationMS:      durationMS,
			TokensGenerated: result.TokensGenerated,
		})
	}
}

// ackWithRetry POSTs /ack, retrying a transport error or 5xx up to ackRetries
// times with the standard backoff. A lost ack on a done job otherwise leaves the
// relay row hidden until the visibility timeout, which re-runs the whole job.
func (c *Client) ackWithRetry(ctx context.Context, ackReq queuepkg.AckRequest, correlationID string) error {
	backoff := backoffMin
	var err error
	for attempt := 0; attempt < ackRetries; attempt++ {
		if err = c.postJSON(ctx, "/ack", ackReq, nil); err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return err
		}
		if attempt < ackRetries-1 {
			slog.Warn("railway ack attempt failed, retrying",
				slog.String(logschema.FieldCorrelationID, correlationID),
				"job_id", ackReq.JobID,
				"attempt", attempt+1,
				"err", err,
			)
			select {
			case <-ctx.Done():
				return err
			case <-time.After(backoff):
			}
			backoff = min(backoff*2, backoffMax)
		}
	}
	return err
}

func (c *Client) ackFailed(ctx context.Context, jobID, correlationID, errCode string, errRespBytes []byte) {
	if errRespBytes == nil {
		errRespBytes, _ = json.Marshal(api.ErrorResponse{
			Code:          errCode,
			CorrelationID: correlationID,
		})
	}
	ackReq := queuepkg.AckRequest{
		JobID:         jobID,
		Status:        queuepkg.QueueJobFailed,
		ResultPayload: errRespBytes,
		ErrorCode:     errCode,
	}
	if err := c.postJSON(ctx, "/ack", ackReq, nil); err != nil {
		slog.Warn("railway ack (failed job) error",
			slog.String(logschema.FieldEvent, string(logschema.EventQueueReturned)),
			"job_id", jobID,
			"err", err,
		)
	}
}

// postJSON marshals body as JSON, POSTs to baseURL+path with auth, and decodes
// the response into out (if non-nil). Returns an error on non-2xx status.
func (c *Client) postJSON(ctx context.Context, path string, body, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("railway %s returned %d", path, resp.StatusCode)
	}

	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}
	return nil
}
