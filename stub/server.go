// Package main provides a minimal HTTP server that serves mock data conforming
// to the API contract. Not part of the production binary — a dev aid for
// building clients against before the real server is running.
//
// Run with:  go run ./stub
// Listens on: http://localhost:8080
package main

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/types"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	mux := http.NewServeMux()
	mux.HandleFunc(api.PathHealth, handleHealth)
	mux.HandleFunc(api.PathStatus, handleStatus)
	mux.HandleFunc(api.PathInfer, handleInfer)

	slog.Info("stub server starting", "addr", ":8080")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		slog.Error("stub server failed", "err", err)
		os.Exit(1)
	}
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`)) //nolint:errcheck
}

func handleStatus(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().UTC()
	status := types.ServerStatus{
		Timestamp:       now,
		QueueDepth:      3,
		ActiveBatchSize: 2,
		LoadedModel: &types.LoadedModel{
			Descriptor: types.ModelDescriptor{
				TierLabel:      types.TierMid,
				Name:           "mistral-7b-q4-stub",
				RequiredVRAMMB: 4096,
				Modality:       "text",
			},
			LoadedAt:  now.Add(-5 * time.Minute),
			IdleSince: time.Time{},
		},
		AvailableVRAMMB: 3200,
		RecentJobs:      mockJobHistory(),
		UptimeSince:     now.Add(-1 * time.Hour),
		Version:         "stub-0.0.0",
	}
	writeJSON(w, http.StatusOK, status)
}

func handleInfer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, api.ErrorResponse{
			Code:    api.ErrCodeInvalidRequest,
			Message: "method not allowed",
		})
		return
	}

	var req api.InferRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, api.ErrorResponse{
			Code:    api.ErrCodeInvalidRequest,
			Message: "invalid JSON body",
		})
		return
	}

	writeJSON(w, http.StatusOK, api.InferResponse{
		RequestID:       "stub-req-0001",
		CorrelationID:   req.CorrelationID,
		Modality:        req.Modality,
		Output:          "[stub] inference runtime not yet connected",
		TokensGenerated: 12,
		DurationMS:      42,
		ModelTier:       string(types.TierMid),
		FinishedAt:      time.Now().UTC(),
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func mockJobHistory() []types.JobEntry {
	now := time.Now().UTC()
	return []types.JobEntry{
		{
			RequestID:       "stub-req-0001",
			CorrelationID:   "req-4f9a2c1b8e3d4a5f9b2c1b8e3d4a5f9b",
			Source:          types.JobSourceLocal,
			Status:          types.JobStatusDone,
			Modality:        "text",
			ModelTier:       string(types.TierMid),
			EnqueuedAt:      now.Add(-30 * time.Second),
			StartedAt:       now.Add(-28 * time.Second),
			FinishedAt:      now.Add(-20 * time.Second),
			DurationMS:      8000,
			TokensGenerated: 312,
		},
		{
			RequestID:     "stub-req-0002",
			CorrelationID: "req-9b2c1b8e3d4a5f9b4f9a2c1b8e3d4a5f",
			Source:        types.JobSourceOffline,
			Status:        types.JobStatusFailed,
			Modality:      "vision",
			ModelTier:     string(types.TierWeak),
			EnqueuedAt:    now.Add(-2 * time.Minute),
			StartedAt:     now.Add(-2*time.Minute + 2*time.Second),
			FinishedAt:    now.Add(-2*time.Minute + 5*time.Second),
			DurationMS:    3000,
			ErrorCode:     api.ErrCodeModelLoadFailed,
		},
	}
}
