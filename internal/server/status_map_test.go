package server

import (
	"net/http"
	"testing"

	"github.com/ngkaichong/home-inference-server/api"
)

func TestBackendErrToHTTPStatus(t *testing.T) {
	cases := map[string]int{
		api.ErrCodeInvalidRequest:      http.StatusBadRequest,
		api.ErrCodeInvalidModality:     http.StatusBadRequest,
		api.ErrCodeUnauthorized:        http.StatusUnauthorized,
		api.ErrCodeNotFound:            http.StatusNotFound,
		api.ErrCodeOverloaded:          http.StatusTooManyRequests,
		api.ErrCodeRateLimited:         http.StatusTooManyRequests,
		api.ErrCodeQuotaExceeded:       http.StatusTooManyRequests,
		api.ErrCodeTimeout:             http.StatusRequestTimeout,
		api.ErrCodeNotImplemented:      http.StatusNotImplemented,
		api.ErrCodeModelLoadFailed:     http.StatusServiceUnavailable,
		api.ErrCodeUnavailable:         http.StatusServiceUnavailable,
		api.ErrCodeQueueFull:           http.StatusServiceUnavailable,
		api.ErrCodeInternal:            http.StatusInternalServerError,
		api.ErrCodeReasoningExhausted:  http.StatusInternalServerError,
		"something_totally_unexpected": http.StatusInternalServerError,
	}
	for code, want := range cases {
		if got := backendErrToHTTPStatus(code); got != want {
			t.Errorf("%s -> %d, want %d", code, got, want)
		}
	}
}
