// Package pipeline implements a backend.Backend that composes two sub-backends:
// a "perceive" stage that turns a non-text input (an image, later audio) into a
// text description, and a "reason" stage — the ordinary text backend — that
// answers the user's request over that description.
//
// The two sub-backends are never resident at the same time: the perceive stage
// is explicitly Shutdown before the reason stage runs, so peak VRAM is whatever
// the reason hop needs and the single-model-resident invariant holds. Because
// the reason hop is just a normal text inference, tier cascade, speed floors,
// min_tier and deferral all apply to it unchanged.
//
// A future audio or OCR modality registers by constructing another
// pipeline.Backend with a different perceive stage and Spec — nothing in the
// core queue → batcher → dispatcher → router changes.
package pipeline

import (
	"context"
	"log/slog"
	"time"

	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/logschema"
)

// Stage is the perceive half of a pipeline. It is exactly a backend.Backend;
// the alias documents intent (it is run once, blocking, and then shut down).
type Stage = backend.Backend

// Spec configures a pipeline for one modality.
type Spec struct {
	// Modality this pipeline is registered under (e.g. backend.ModalityKindVision).
	Modality backend.ModalityKind
	// PerceptionPrompt is sent as the prompt to the perceive stage, along with
	// the request's non-text data. It is deliberately not the user's question.
	PerceptionPrompt string
	// ReasonSystemPrompt is prepended as a system turn to the reason hop.
	ReasonSystemPrompt string
	// DescriptionMaxTokens caps the perceive stage's output. 0 → 512.
	DescriptionMaxTokens int
}

// Backend orchestrates perceive → reason. It satisfies backend.Backend and
// backend.Streamer.
type Backend struct {
	perceive Stage
	reason   backend.Backend
	spec     Spec
}

// New builds a pipeline. perceive is owned by the pipeline (it is Shutdown by
// the pipeline); reason is borrowed (the caller owns its lifecycle — typically
// the shared text vram.Backend). Neither sub-backend is started here.
func New(perceive Stage, reason backend.Backend, spec Spec) *Backend {
	if spec.DescriptionMaxTokens <= 0 {
		spec.DescriptionMaxTokens = 512
	}
	return &Backend{perceive: perceive, reason: reason, spec: spec}
}

func (b *Backend) Modality() backend.ModalityKind { return b.spec.Modality }

// Ready reports the perceive stage's readiness; the reason stage (a vram.Backend)
// is always ready.
func (b *Backend) Ready() bool { return b.perceive.Ready() }

// perceiveThenEvict runs the perception hop and then tears the perceive
// subprocess down so it is never resident alongside the reason model. The
// eviction runs even if perception failed.
func (b *Backend) perceiveThenEvict(ctx context.Context, req backend.Request) (backend.Response, error) {
	defer func() {
		shCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := b.perceive.Shutdown(shCtx); err != nil {
			slog.Warn("pipeline: perceive stage did not shut down cleanly — VRAM may still be held",
				slog.String(logschema.FieldCorrelationID, req.CorrelationID),
				slog.Any("err", err),
			)
		}
	}()

	return b.perceive.Infer(ctx, backend.Request{
		CorrelationID: req.CorrelationID,
		RequestID:     req.RequestID,
		Modality:      b.spec.Modality,
		Prompt:        b.spec.PerceptionPrompt,
		ImageData:     req.ImageData,
		MaxTokens:     b.spec.DescriptionMaxTokens,
		Priority:      req.Priority,
	})
}

// reasonRequest builds the text request for the reason hop from the perception
// description and the caller's original request.
func (b *Backend) reasonRequest(req backend.Request, description string) backend.Request {
	return backend.Request{
		CorrelationID: req.CorrelationID,
		RequestID:     req.RequestID,
		Modality:      backend.ModalityKindText,
		Messages: []backend.Message{
			{Role: "system", Content: b.spec.ReasonSystemPrompt},
			{Role: "user", Content: "Image description:\n" + description + "\n\nRequest: " + req.Prompt},
		},
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		Priority:       req.Priority,
		MinTier:        req.MinTier,
		PreferredTier:  req.PreferredTier,
		ResponseFormat: req.ResponseFormat,
	}
}

// Infer runs perceive → evict → reason. If the reason hop cannot load any tier
// at all ("overloaded"), the perception description is returned as the answer
// with QualityDegraded set, rather than failing the request. Other reason-hop
// errors (timeout, deferred, draining) propagate unchanged.
func (b *Backend) Infer(ctx context.Context, req backend.Request) (backend.Response, error) {
	desc, err := b.perceiveThenEvict(ctx, req)
	if err != nil {
		return backend.Response{}, err
	}

	resp, err := b.reason.Infer(ctx, b.reasonRequest(req, desc.Output))
	if err != nil {
		if be, ok := err.(*backend.BackendError); ok && be.Code == "overloaded" {
			slog.Info("pipeline: reason hop overloaded — returning perception description degraded",
				slog.String(logschema.FieldEvent, string(logschema.EventQualityDegraded)),
				slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			)
			return backend.Response{
				Output:          desc.Output,
				TokensGenerated: desc.TokensGenerated,
				ModelTier:       desc.ModelTier,
				QualityDegraded: true,
			}, nil
		}
		return backend.Response{}, err
	}

	if desc.QualityDegraded {
		resp.QualityDegraded = true
	}
	return resp, nil
}

// InferStream runs the perception hop blocking, then streams the reason hop. If
// the reason backend is not a Streamer, its single response is emitted as one
// content delta. Implements backend.Streamer.
func (b *Backend) InferStream(ctx context.Context, req backend.Request, chunkFn func(kind backend.ChunkKind, delta string)) (backend.Response, error) {
	desc, err := b.perceiveThenEvict(ctx, req)
	if err != nil {
		return backend.Response{}, err
	}

	reasonReq := b.reasonRequest(req, desc.Output)

	streamer, ok := b.reason.(backend.Streamer)
	if !ok {
		resp, ierr := b.reason.Infer(ctx, reasonReq)
		if ierr != nil {
			if be, isBE := ierr.(*backend.BackendError); isBE && be.Code == "overloaded" {
				chunkFn(backend.ChunkContent, desc.Output)
				return backend.Response{Output: desc.Output, ModelTier: desc.ModelTier, QualityDegraded: true}, nil
			}
			return backend.Response{}, ierr
		}
		if resp.Output != "" {
			chunkFn(backend.ChunkContent, resp.Output)
		}
		if desc.QualityDegraded {
			resp.QualityDegraded = true
		}
		return resp, nil
	}

	resp, err := streamer.InferStream(ctx, reasonReq, chunkFn)
	if err != nil {
		if be, isBE := err.(*backend.BackendError); isBE && be.Code == "overloaded" {
			chunkFn(backend.ChunkContent, desc.Output)
			return backend.Response{Output: desc.Output, ModelTier: desc.ModelTier, QualityDegraded: true}, nil
		}
		return backend.Response{}, err
	}
	if desc.QualityDegraded {
		resp.QualityDegraded = true
	}
	return resp, nil
}

// Describe runs only the perception hop: it loads the perceive model, gets a
// literal description of the image, and evicts the model — the reason stage is
// never touched. Used by callers (the chat UI) that want to fold an image into
// a text conversation as a description turn rather than run a full vision
// inference per follow-up. A perceive-stage BackendError propagates unchanged.
func (b *Backend) Describe(ctx context.Context, imageData []byte) (description, modelTier string, err error) {
	resp, err := b.perceiveThenEvict(ctx, backend.Request{ImageData: imageData})
	if err != nil {
		return "", "", err
	}
	return resp.Output, resp.ModelTier, nil
}

// Shutdown shuts down the perceive stage. The reason stage is borrowed and its
// lifecycle is the caller's responsibility.
func (b *Backend) Shutdown(ctx context.Context) error {
	return b.perceive.Shutdown(ctx)
}
