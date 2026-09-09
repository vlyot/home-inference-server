// Package vision is a placeholder for the vision modality. Real multimodal
// inference (a llama.cpp mmproj model) is not wired up in this build; every
// request returns 501 not_implemented. The roster still lists a vision model to
// document the intent — see cmd/server's defaultRoster and the docs page.
package vision

import (
	"context"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
)

// Stub is the not-yet-implemented vision backend.
type Stub struct{}

// New returns the vision Stub.
func New() *Stub { return &Stub{} }

func (s *Stub) Modality() backend.ModalityKind { return backend.ModalityKindVision }

func (s *Stub) Infer(_ context.Context, _ backend.Request) (backend.Response, error) {
	return backend.Response{}, &backend.BackendError{
		Code:    api.ErrCodeNotImplemented,
		Message: "vision inference is not implemented in this build",
	}
}

// Ready is false: the modality is registered for routing but cannot serve.
func (s *Stub) Ready() bool { return false }

func (s *Stub) Shutdown(_ context.Context) error { return nil }
