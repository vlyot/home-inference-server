package server

import (
	"log/slog"
	"net/http"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/logschema"
)

// state returns the lifecycle state string reported on /healthz and /v1/status.
func (s *Server) state() string {
	if s.draining.Load() {
		return api.StateDraining
	}
	return api.StateOK
}

// Draining reports whether the server is currently draining for maintenance.
func (s *Server) Draining() bool { return s.draining.Load() }

// setDraining toggles the drain flag, drives the offline-worker pause hook to
// match, and logs the transition with the current in-flight / queue-depth.
func (s *Server) setDraining(v bool) {
	if prev := s.draining.Swap(v); prev == v {
		return // no-op: already in the requested state
	}

	if s.pauseSink != nil {
		if v {
			s.pauseSink.Pause()
		} else {
			s.pauseSink.Resume()
		}
	}

	snap := s.statusFn()
	event, msg := logschema.EventAdminResume, "server resumed"
	if v {
		event, msg = logschema.EventAdminDrain, "server draining for maintenance"
	}
	slog.Info(msg,
		slog.String(logschema.FieldEvent, string(event)),
		slog.Int(logschema.FieldInFlight, snap.InFlight),
		slog.Int(logschema.FieldQueueDepth, snap.QueueDepth),
	)
}

// adminStateResponse snapshots the fields the /v1/admin/* endpoints return.
func (s *Server) adminStateResponse(state string) api.AdminStateResponse {
	snap := s.statusFn()
	return api.AdminStateResponse{State: state, InFlight: snap.InFlight, QueueDepth: snap.QueueDepth}
}

// handleAdmin serves the loopback-only /v1/admin/* control surface:
//
//	GET  /v1/admin          — state probe
//	POST /v1/admin/drain    — reject new inference (503 draining), pause the
//	                          offline worker; in-flight requests finish
//	POST /v1/admin/resume   — clear the drain, resume the offline worker
//	POST /v1/admin/restart  — drain, then signal cmd/server to run the shutdown
//	                          sequence and re-exec via the run.cmd loop
//
// A non-loopback caller gets 404 so the endpoints are invisible from the LAN.
func (s *Server) handleAdmin(w http.ResponseWriter, r *http.Request) {
	if !IsLoopback(r.RemoteAddr) {
		writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "not found", "")
		return
	}

	switch {
	case r.Method == http.MethodGet && r.URL.Path == api.PathAdmin:
		writeJSON(w, http.StatusOK, s.adminStateResponse(s.state()))

	case r.Method == http.MethodPost && r.URL.Path == api.PathAdminDrain:
		s.setDraining(true)
		writeJSON(w, http.StatusOK, s.adminStateResponse(api.StateDraining))

	case r.Method == http.MethodPost && r.URL.Path == api.PathAdminResume:
		s.setDraining(false)
		writeJSON(w, http.StatusOK, s.adminStateResponse(api.StateOK))

	case r.Method == http.MethodPost && r.URL.Path == api.PathAdminRestart:
		if s.restartCh == nil {
			writeError(w, http.StatusNotImplemented, api.ErrCodeNotImplemented,
				"restart is not wired in this build", "")
			return
		}
		s.setDraining(true)
		slog.Info("restart requested",
			slog.String(logschema.FieldEvent, string(logschema.EventAdminRestart)))
		select {
		case s.restartCh <- struct{}{}:
		default: // already signalled — a second restart request is a no-op
		}
		writeJSON(w, http.StatusOK, s.adminStateResponse("restarting"))

	case r.URL.Path == api.PathAdmin,
		r.URL.Path == api.PathAdminDrain,
		r.URL.Path == api.PathAdminResume,
		r.URL.Path == api.PathAdminRestart:
		// Known path, wrong method.
		allow := http.MethodPost
		if r.URL.Path == api.PathAdmin {
			allow = http.MethodGet
		}
		w.Header().Set("Allow", allow)
		writeError(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "method not allowed", "")

	default:
		writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "not found", "")
	}
}
