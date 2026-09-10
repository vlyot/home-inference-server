package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

// handleDescribe runs POST /v1/describe: an image is passed through the vision
// perception model only (no reasoning hop) and its literal description returned.
// The chat UI calls this on image attach so it can fold the description into a
// text conversation. 501 when no describer is wired.
func (s *Server) handleDescribe(w http.ResponseWriter, r *http.Request) {
	if s.describer == nil {
		writeError(w, http.StatusNotImplemented, api.ErrCodeUnavailable, "vision describe not available", "")
		return
	}
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, api.ErrCodeDraining,
			"server is draining for maintenance; retry shortly", "")
		return
	}

	var body api.DescribeRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON body", "")
		return
	}
	if body.ImageBase64 == "" {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "image_base64 is required", "")
		return
	}
	img, err := base64.StdEncoding.DecodeString(body.ImageBase64)
	if err != nil {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "image_base64 is not valid base64", "")
		return
	}

	requestID := strings.ReplaceAll(uuid.New().String(), "-", "")
	correlationID := logschema.CorrelationIDPrefix + requestID
	ctx, cancel := context.WithTimeout(r.Context(), s.inferTO)
	defer cancel()

	startedAt := time.Now()
	desc, tier, err := s.describer.Describe(ctx, img)
	finishedAt := time.Now()

	if err != nil {
		var be *backend.BackendError
		code := api.ErrCodeInternal
		if errors.As(err, &be) {
			code = be.Code
		}
		slog.Info("describe failed",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, correlationID),
			slog.String(logschema.FieldErrorCode, code),
		)
		s.recordJob(types.JobEntry{
			RequestID:     requestID,
			CorrelationID: correlationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusFailed,
			Modality:      string(api.ModalityVision),
			EnqueuedAt:    startedAt,
			StartedAt:     startedAt,
			FinishedAt:    finishedAt,
			DurationMS:    finishedAt.Sub(startedAt).Milliseconds(),
			InferenceMS:   finishedAt.Sub(startedAt).Milliseconds(),
		})
		writeError(w, backendErrToHTTPStatus(code), code, err.Error(), correlationID)
		return
	}

	s.recordJob(types.JobEntry{
		RequestID:     requestID,
		CorrelationID: correlationID,
		Source:        types.JobSourceLocal,
		Status:        types.JobStatusDone,
		Modality:      string(api.ModalityVision),
		ModelTier:     tier,
		EnqueuedAt:    startedAt,
		StartedAt:     startedAt,
		FinishedAt:    finishedAt,
		DurationMS:    finishedAt.Sub(startedAt).Milliseconds(),
		InferenceMS:   finishedAt.Sub(startedAt).Milliseconds(),
	})
	writeJSON(w, http.StatusOK, api.DescribeResponse{Description: desc, ModelTier: tier})
}
