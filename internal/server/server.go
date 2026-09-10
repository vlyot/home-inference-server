package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/backend"
	"github.com/ngkaichong/home-inference-server/internal/logbuf"
	"github.com/ngkaichong/home-inference-server/internal/queue"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/types"
)

const jobHistoryMax = 200

// defaultInferTimeout bounds a single inference when the caller did not set
// timeout_ms. Sized to the cold-model-load ceiling (llama-server's own 120 s
// health-poll deadline) plus generation headroom. Overridable via
// HIS_INFER_TIMEOUT_MS; see cmd/server. Without this a hung llama-server would
// wedge the dispatch goroutine and fill the queue indefinitely.
const defaultInferTimeout = 10 * time.Minute

// CPUMetrics is satisfied by system.CPUMonitor.
type CPUMetrics interface {
	Pct() float64
}

// PressureSource is satisfied by any backend that exposes real-time tok/sec,
// the currently loaded tier, the VRAM its own subprocesses hold, and its
// concurrent-inference occupancy (e.g. vram.Backend).
type PressureSource interface {
	backend.Measurable
	LoadedTier() string
	SelfVRAMMB() int64
	InFlight() int
	MaxParallel() int
}

// VRAMTotaler optionally reports total GPU VRAM (for the self/other split).
type VRAMTotaler interface {
	TotalMB() (int64, error)
}

// VRAMSource is satisfied by anything that can report current free VRAM.
type VRAMSource interface {
	AvailableMB() (int64, error)
}

// Tokenizer is satisfied by the text llama-server backend; it powers the
// /v1/tokenize and /v1/model/props endpoints the chat UI uses for its
// context-limit banner.
type Tokenizer interface {
	Tokenize(ctx context.Context, text string) (int, error)
	NCtx(ctx context.Context) (int, error)
}

// LogSource is satisfied by *logbuf.Buffer; it backs the GET /v1/logs route.
type LogSource interface {
	Query(level, event string, since time.Duration, limit int) []logbuf.Record
}

// ChatStore is the subset of *chatstore.Store the server needs. Kept as an
// interface so tests can supply a lightweight fake.
type ChatStore interface {
	List() []api.ConversationMeta
	Get(id string) (api.Conversation, bool)
	Save(c api.Conversation) (api.ConversationMeta, error)
	Delete(id string) error
}

// RelayEnqueuer hands a request off to the durable hosted relay when the local
// server is under pressure. Satisfied by *relayenqueue.Client.
type RelayEnqueuer interface {
	Enqueue(ctx context.Context, correlationID string, payload json.RawMessage) (jobID string, err error)
	ResultURL(jobID string) string
}

// Server is the HTTP handler for the inference API.
type Server struct {
	q         *queue.Queue
	statusFn  func() types.ServerStatus
	cpu       CPUMetrics     // may be nil during tests
	pressure  PressureSource // may be nil during tests
	vramSrc   VRAMSource     // may be nil during tests; used for the deferral check
	relay     RelayEnqueuer  // may be nil; when nil the server never defers
	inferTO   time.Duration  // per-request inference deadline when timeout_ms is unset
	version   string
	startedAt time.Time

	mu      sync.Mutex
	history []types.JobEntry // ring buffer, newest-first

	backendsMu sync.RWMutex
	backends   map[backend.ModalityKind]backend.Backend // for streaming path; may be nil

	chatStore ChatStore // may be nil; when nil the /v1/chats* routes 404
	tokenizer Tokenizer // may be nil; when nil /v1/tokenize + /v1/model/props 501
	logs      LogSource // may be nil; when nil /v1/logs 501

	// draining, when set (POST /v1/admin/drain), makes handleInfer reject new
	// work with 503 draining. Read-only endpoints are unaffected.
	draining atomic.Bool
	// restartCh receives a signal from POST /v1/admin/restart; cmd/server selects
	// on it to run the shutdown sequence and re-exec. nil ⇒ /v1/admin/restart 501.
	restartCh chan<- struct{}
	// pauseSink, when set, is the offline-queue worker's pause hook, toggled
	// alongside draining. nil ⇒ drain still gates HTTP; the worker isn't paused.
	pauseSink PauseSink
}

// PauseSink is the offline-queue worker's pause/resume hook, toggled by the
// admin drain/resume endpoints. Satisfied directly by *remote.Client.
type PauseSink interface {
	Pause()
	Resume()
}

// New creates a Server. statusFn is called on GET /v1/status to snapshot
// dynamic state (queue depth, active batch size, loaded model) that the server
// itself doesn't own. version and startedAt are used by GET /healthz.
func New(q *queue.Queue, statusFn func() types.ServerStatus, version string, startedAt time.Time) *Server {
	return &Server{q: q, statusFn: statusFn, version: version, startedAt: startedAt, inferTO: defaultInferTimeout}
}

// SetBackends registers the backend map for the streaming dispatch path.
// Call before the server starts accepting requests.
func (s *Server) SetBackends(backends map[backend.ModalityKind]backend.Backend) {
	s.backendsMu.Lock()
	s.backends = backends
	s.backendsMu.Unlock()
}

// SetPressureSources wires optional runtime dependencies for the /v1/status/pressure
// endpoint. Both arguments may be nil (the endpoint still returns zero-value fields).
func (s *Server) SetPressureSources(cpu CPUMetrics, ps PressureSource) {
	s.cpu = cpu
	s.pressure = ps
}

// SetRelayEnqueuer wires the hosted-relay hand-off and the VRAM source used for
// the deferral decision. Both must be non-nil together; if either is nil the
// server never defers (it serves, cascades tiers, or returns 503 from the queue
// as normal). relay is nil when RAILWAY_QUEUE_URL is unset.
func (s *Server) SetRelayEnqueuer(relay RelayEnqueuer, vram VRAMSource) {
	s.relay = relay
	s.vramSrc = vram
}

// SetInferTimeout overrides the default per-request inference deadline applied
// when the caller does not set timeout_ms. A non-positive value keeps the
// package default.
func (s *Server) SetInferTimeout(d time.Duration) {
	if d > 0 {
		s.inferTO = d
	}
}

// SetChatStore wires conversation persistence for the /v1/chats* routes.
func (s *Server) SetChatStore(store ChatStore) { s.chatStore = store }

// SetTokenizer wires the tokenize/props proxy for the chat UI's context meter.
func (s *Server) SetTokenizer(t Tokenizer) { s.tokenizer = t }

// SetLogSource wires the in-memory log buffer for the /v1/logs route.
func (s *Server) SetLogSource(l LogSource) { s.logs = l }

// SetRestartChannel wires the buffered channel cmd/server selects on to run the
// shutdown-and-re-exec sequence for POST /v1/admin/restart. When unset, that
// endpoint returns 501.
func (s *Server) SetRestartChannel(ch chan<- struct{}) { s.restartCh = ch }

// SetPauseSink wires the offline-queue worker's pause hook so drain/resume also
// stop/start relay-job claiming. nil is fine (no relay configured).
func (s *Server) SetPauseSink(p PauseSink) { s.pauseSink = p }

// shouldDefer returns true when the system is under pressure and the request
// should be handed to the hosted relay rather than served: VRAM < 500 MB AND
// CPU > 90%. High-priority requests always bypass deferral. Requires a relay to
// be configured — without one there is nowhere durable to defer to.
func (s *Server) shouldDefer(priority string) bool {
	if s.relay == nil || s.vramSrc == nil {
		return false
	}
	if priority == "high" {
		return false
	}
	avail, err := s.vramSrc.AvailableMB()
	if err != nil || avail < 0 {
		return false
	}
	if avail >= 500 {
		return false
	}
	if s.cpu == nil || s.cpu.Pct() <= 90 {
		return false
	}
	return true
}

// methodGuarded maps a single-method path to the method it accepts, so a path
// hit with the wrong method returns 405 (with Allow) rather than 404.
var methodGuarded = map[string]string{
	api.PathInfer:          http.MethodPost,
	api.PathStatus:         http.MethodGet,
	api.PathStatusPressure: http.MethodGet,
	api.PathHealth:         http.MethodGet,
	api.PathTokenize:       http.MethodPost,
	api.PathModelProps:     http.MethodGet,
	api.PathLogs:           http.MethodGet,
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if want, ok := methodGuarded[r.URL.Path]; ok && r.Method != want {
		w.Header().Set("Allow", want)
		writeError(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "method not allowed", "")
		return
	}

	switch {
	case r.Method == http.MethodPost && r.URL.Path == api.PathInfer:
		s.handleInfer(w, r)
	case r.Method == http.MethodGet && r.URL.Path == api.PathStatus:
		s.handleStatus(w, r)
	case r.Method == http.MethodGet && r.URL.Path == api.PathStatusPressure:
		s.handlePressure(w, r)
	case r.Method == http.MethodGet && r.URL.Path == api.PathHealth:
		s.handleHealth(w, r)
	case r.Method == http.MethodPost && r.URL.Path == api.PathTokenize:
		s.handleTokenize(w, r)
	case r.Method == http.MethodGet && r.URL.Path == api.PathModelProps:
		s.handleModelProps(w, r)
	case r.Method == http.MethodGet && r.URL.Path == api.PathLogs:
		s.handleLogs(w, r)
	case r.URL.Path == api.PathChats:
		s.handleChatsCollection(w, r)
	case strings.HasPrefix(r.URL.Path, api.PathChats+"/"):
		s.handleChatItem(w, r, strings.TrimPrefix(r.URL.Path, api.PathChats+"/"))
	case r.URL.Path == api.PathAdmin || strings.HasPrefix(r.URL.Path, api.PathAdmin+"/"):
		s.handleAdmin(w, r)
	default:
		writeError(w, http.StatusNotFound, api.ErrCodeInvalidRequest, "not found", "")
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, api.HealthResponse{
		Status:  "ok",
		Version: s.version,
		UptimeS: int64(time.Since(s.startedAt).Seconds()),
		State:   s.state(),
	})
}

func (s *Server) handleInfer(w http.ResponseWriter, r *http.Request) {
	if s.draining.Load() {
		writeError(w, http.StatusServiceUnavailable, api.ErrCodeDraining,
			"server is draining for maintenance; retry shortly", "")
		return
	}

	var req api.InferRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON body", "")
		return
	}

	if req.CorrelationID == "" {
		req.CorrelationID = "req-" + strings.ReplaceAll(uuid.New().String(), "-", "")
	}

	slog.Info("job received",
		slog.String(logschema.FieldEvent, string(logschema.EventJobReceived)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldModality, string(req.Modality)),
	)

	switch req.Modality {
	case api.ModalityText:
		if req.TextInput == nil {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "text_input required for text modality", req.CorrelationID)
			return
		}
		if req.TextInput.Prompt == "" && len(req.TextInput.Messages) == 0 {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "text_input must have prompt or messages", req.CorrelationID)
			return
		}
	case api.ModalityVision:
		if req.VisionInput == nil {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "vision_input required for vision modality", req.CorrelationID)
			return
		}
		if req.VisionInput.Prompt == "" {
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "vision_input.prompt is required", req.CorrelationID)
			return
		}
		if req.VisionInput.ImageBase64 == "" {
			if req.VisionInput.ImageURL != "" {
				writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "image_url is not supported; send the image as image_base64", req.CorrelationID)
				return
			}
			writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "vision_input needs image_base64", req.CorrelationID)
			return
		}
	case "":
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "modality is required", req.CorrelationID)
		return
	default:
		writeError(w, http.StatusBadRequest, api.ErrCodeInvalidModality, "unknown modality", req.CorrelationID)
		return
	}

	requestID := strings.ReplaceAll(uuid.New().String(), "-", "")
	backendReq := TranslateInferRequest(req, requestID)

	// Always bound the request: the caller's timeout_ms when given, otherwise the
	// server default. A missing deadline lets a hung llama-server block the
	// dispatch goroutine forever and fill the queue.
	d := time.Duration(req.TimeoutMS) * time.Millisecond
	if d <= 0 {
		d = s.inferTO
	}
	ctx, cancel := context.WithTimeout(r.Context(), d)
	defer cancel()
	r = r.WithContext(ctx)

	// Branch to streaming path (bypasses queue/batcher) or blocking path.
	if req.Stream {
		s.handleInferStream(w, r, req, backendReq)
	} else {
		s.handleInferBlocking(w, r, req, backendReq)
	}
}

func (s *Server) handlePressure(w http.ResponseWriter, r *http.Request) {
	snap := api.PressureSnapshot{Timestamp: time.Now()}

	if s.cpu != nil {
		snap.CPUPct = s.cpu.Pct()
	}
	if s.pressure != nil {
		snap.TokPerSec = s.pressure.TokPerSec()
		snap.ActiveTier = s.pressure.LoadedTier()
		snap.SelfVRAMMB = s.pressure.SelfVRAMMB()
		snap.InFlight = s.pressure.InFlight()
		snap.MaxParallel = s.pressure.MaxParallel()
	}

	// Pull VRAM from the status function which already queries NVML.
	status := s.statusFn()
	snap.VRAMAvailMB = status.AvailableVRAMMB
	snap.OtherVRAMMB = s.otherVRAMMB(status.AvailableVRAMMB, snap.SelfVRAMMB)

	writeJSON(w, http.StatusOK, snap)
}

// otherVRAMMB is total VRAM minus free minus what our own subprocesses hold,
// i.e. what other applications are using. Returns 0 when total VRAM is unknown.
func (s *Server) otherVRAMMB(availMB, selfMB int64) int64 {
	totaler, ok := s.vramSrc.(VRAMTotaler)
	if !ok || availMB < 0 {
		return 0
	}
	total, err := totaler.TotalMB()
	if err != nil || total < 0 {
		return 0
	}
	other := total - availMB - selfMB
	if other < 0 {
		return 0
	}
	return other
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	snap := s.statusFn()

	if s.cpu != nil {
		snap.CPUPct = s.cpu.Pct()
	}
	if s.pressure != nil {
		snap.TokensPerSec = s.pressure.TokPerSec()
		snap.SelfVRAMMB = s.pressure.SelfVRAMMB()
	}
	s.mu.Lock()
	snap.RecentJobs = make([]types.JobEntry, len(s.history))
	copy(snap.RecentJobs, s.history)
	s.mu.Unlock()

	for _, j := range snap.RecentJobs {
		if j.Deferred {
			snap.DeferredCount++
		}
	}
	snap.State = s.state()

	writeJSON(w, http.StatusOK, snap)
}

// handleLogs serves the in-memory log buffer for the dashboard's /logs page.
// Query params: level, event (passthrough filters), since (1h/12h/24h, default
// and max 24h), limit (default 500, max 2000).
func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request) {
	if s.logs == nil {
		writeError(w, http.StatusNotImplemented, api.ErrCodeUnavailable, "log buffer not enabled", "")
		return
	}

	q := r.URL.Query()

	since := 24 * time.Hour
	if raw := q.Get("since"); raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			since = d
		}
	}
	if since > 24*time.Hour {
		since = 24 * time.Hour
	}

	limit := 500
	if raw := q.Get("limit"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	if limit > 2000 {
		limit = 2000
	}

	recs := s.logs.Query(q.Get("level"), q.Get("event"), since, limit)
	writeJSON(w, http.StatusOK, map[string]any{"logs": recs})
}

func (s *Server) recordJob(entry types.JobEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Prepend (newest-first); trim to cap.
	s.history = append([]types.JobEntry{entry}, s.history...)
	if len(s.history) > jobHistoryMax {
		s.history = s.history[:jobHistoryMax]
	}
}

// RecordJob is the public variant used by the remote client to record offline jobs.
func (s *Server) RecordJob(entry types.JobEntry) { s.recordJob(entry) }

// RecentJobs returns a snapshot of the job history (newest-first) for use in
// the status function provided to New.
func (s *Server) RecentJobs() []types.JobEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]types.JobEntry, len(s.history))
	copy(out, s.history)
	return out
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, message, correlationID string) {
	writeJSON(w, status, api.ErrorResponse{
		Code:          code,
		Message:       message,
		CorrelationID: correlationID,
	})
}

func backendErrToHTTPStatus(code string) int {
	switch code {
	case api.ErrCodeInvalidRequest, api.ErrCodeInvalidModality, api.ErrCodeInvalidGrammar:
		return http.StatusBadRequest
	case api.ErrCodeUnauthorized:
		return http.StatusUnauthorized
	case api.ErrCodeNotFound:
		return http.StatusNotFound
	case api.ErrCodeOverloaded, api.ErrCodeRateLimited, api.ErrCodeQuotaExceeded:
		return http.StatusTooManyRequests
	case api.ErrCodeTimeout:
		return http.StatusRequestTimeout
	case api.ErrCodeNotImplemented:
		return http.StatusNotImplemented
	case api.ErrCodeModelLoadFailed, api.ErrCodeUnavailable, api.ErrCodeQueueFull:
		return http.StatusServiceUnavailable
	default:
		// ErrCodeInternal and ErrCodeReasoningExhausted (stream-only) land here.
		return http.StatusInternalServerError
	}
}

// TranslateInferRequest converts an api.InferRequest into a backend.Request.
// Exported so the remote client can share the same translation logic without
// duplicating it and risking drift when new fields are added.
func TranslateInferRequest(req api.InferRequest, requestID string) backend.Request {
	prompt := ""
	var imageData []byte

	var messages []backend.Message

	if req.TextInput != nil {
		switch {
		case len(req.TextInput.Messages) > 0:
			messages = translateMessages(req.TextInput.Messages, req.SystemPrompt)
		case req.TextInput.Prompt != "":
			prompt = req.TextInput.Prompt
			if req.SystemPrompt != "" {
				prompt = req.SystemPrompt + "\n\n" + prompt
			}
		}
	}
	if req.VisionInput != nil {
		prompt = req.VisionInput.Prompt
		if req.VisionInput.ImageBase64 != "" {
			if decoded, err := base64.StdEncoding.DecodeString(req.VisionInput.ImageBase64); err == nil {
				imageData = decoded
			}
		}
		if req.SystemPrompt != "" {
			prompt = req.SystemPrompt + "\n\n" + prompt
		}
	}

	return backend.Request{
		CorrelationID:  req.CorrelationID,
		RequestID:      requestID,
		Modality:       backend.ModalityKind(req.Modality),
		Prompt:         prompt,
		Messages:       messages,
		ImageData:      imageData,
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		Priority:       req.Priority,
		MinTier:        req.MinTier,
		PreferredTier:  req.PreferredTier,
		Stream:         req.Stream,
		ResponseFormat: req.ResponseFormat,
	}
}

// translateMessages copies the API message list into the backend shape,
// prepending SystemPrompt as a system turn only if the caller didn't already
// supply one.
func translateMessages(msgs []api.ChatMessage, systemPrompt string) []backend.Message {
	hasSystem := false
	for _, m := range msgs {
		if m.Role == "system" {
			hasSystem = true
			break
		}
	}
	out := make([]backend.Message, 0, len(msgs)+1)
	if systemPrompt != "" && !hasSystem {
		out = append(out, backend.Message{Role: "system", Content: systemPrompt})
	}
	for _, m := range msgs {
		out = append(out, backend.Message{Role: m.Role, Content: m.Content})
	}
	return out
}

// deferToRelay hands a request that cannot be served now (VRAM/CPU pressure) to
// the durable hosted relay and writes the caller's response: 202 with a
// result_url to poll on success, or 503 overloaded if the relay is unreachable.
// It returns true when it has written the response and the caller must stop.
func (s *Server) deferToRelay(w http.ResponseWriter, r *http.Request, req api.InferRequest, backendReq backend.Request) bool {
	if !s.shouldDefer(req.Priority) {
		return false
	}

	payload, err := json.Marshal(req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, api.ErrCodeInternal, "could not serialise request for deferral", req.CorrelationID)
		return true
	}

	jobID, err := s.relay.Enqueue(r.Context(), req.CorrelationID, payload)
	if err != nil {
		slog.Warn("deferral hand-off failed",
			slog.String(logschema.FieldEvent, string(logschema.EventJobDeferred)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			"err", err,
		)
		writeError(w, http.StatusServiceUnavailable, api.ErrCodeOverloaded,
			"under pressure and the offline queue is unreachable — retry shortly", req.CorrelationID)
		return true
	}

	slog.Info("job deferred to relay",
		slog.String(logschema.FieldEvent, string(logschema.EventJobDeferred)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
		slog.String(logschema.FieldJobID, jobID),
	)
	s.recordJob(types.JobEntry{
		RequestID:     jobID,
		CorrelationID: req.CorrelationID,
		Source:        types.JobSourceOffline,
		Status:        types.JobStatusQueued,
		Modality:      string(req.Modality),
		Priority:      req.Priority,
		MinTier:       req.MinTier,
		EnqueuedAt:    time.Now(),
		Deferred:      true,
	})
	writeJSON(w, http.StatusAccepted, map[string]any{
		"deferred":   true,
		"request_id": jobID,
		"result_url": s.relay.ResultURL(jobID),
	})
	return true
}

// handleInferBlocking is the non-streaming fallback used by handleInferStream
// when the backend does not implement backend.Streamer or the client doesn't
// support flushing.
func (s *Server) handleInferBlocking(w http.ResponseWriter, r *http.Request, req api.InferRequest, backendReq backend.Request) {
	enqueuedAt := time.Now()

	if s.deferToRelay(w, r, req, backendReq) {
		return
	}

	resultCh := make(chan queue.Result, 1)
	job := queue.Job{
		ID:            backendReq.RequestID,
		CorrelationID: req.CorrelationID,
		EnqueuedAt:    enqueuedAt,
		Req:           backendReq,
		ResultCh:      resultCh,
	}

	if err := s.q.Enqueue(r.Context(), job); err != nil {
		writeError(w, http.StatusServiceUnavailable, api.ErrCodeOverloaded, "queue full or request cancelled", req.CorrelationID)
		return
	}

	slog.Info("job queued",
		slog.String(logschema.FieldEvent, string(logschema.EventJobQueued)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
		slog.Int(logschema.FieldQueueDepth, s.q.Depth()),
	)

	var result queue.Result
	select {
	case result = <-resultCh:
	case <-r.Context().Done():
		// Caller hung up (or hit its timeout_ms / the server default). A timeout
		// maps to 408; a plain disconnect to 503.
		now := time.Now()
		code, status := api.ErrCodeUnavailable, http.StatusServiceUnavailable
		if r.Context().Err() == context.DeadlineExceeded {
			code, status = api.ErrCodeTimeout, http.StatusRequestTimeout
		}
		writeError(w, status, code, "request cancelled or timed out", req.CorrelationID)
		s.recordJob(types.JobEntry{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusCancelled,
			Modality:      string(req.Modality),
			Priority:      req.Priority,
			MinTier:       req.MinTier,
			EnqueuedAt:    enqueuedAt,
			FinishedAt:    now,
			DurationMS:    now.Sub(enqueuedAt).Milliseconds(),
		})
		return
	}

	finishedAt := time.Now()
	durationMS := finishedAt.Sub(enqueuedAt).Milliseconds()
	startedAt := result.StartedAt
	if startedAt.IsZero() {
		startedAt = enqueuedAt
	}
	queueWaitMS := startedAt.Sub(enqueuedAt).Milliseconds()
	inferenceMS := finishedAt.Sub(startedAt).Milliseconds()

	if result.Err != nil {
		var be *backend.BackendError
		if errors.As(result.Err, &be) {
			writeError(w, backendErrToHTTPStatus(be.Code), be.Code, be.Message, req.CorrelationID)
		} else {
			writeError(w, http.StatusInternalServerError, api.ErrCodeInternal, result.Err.Error(), req.CorrelationID)
		}
		errCode := api.ErrCodeInternal
		if be != nil {
			errCode = be.Code
		}
		slog.Info("job failed",
			slog.String(logschema.FieldEvent, string(logschema.EventJobFailed)),
			slog.String(logschema.FieldCorrelationID, req.CorrelationID),
			slog.String(logschema.FieldRequestID, backendReq.RequestID),
			slog.Int64(logschema.FieldDurationMS, durationMS),
			slog.String(logschema.FieldErrorCode, errCode),
		)
		s.recordJob(types.JobEntry{
			RequestID:     backendReq.RequestID,
			CorrelationID: req.CorrelationID,
			Source:        types.JobSourceLocal,
			Status:        types.JobStatusFailed,
			Modality:      string(req.Modality),
			Priority:      req.Priority,
			MinTier:       req.MinTier,
			EnqueuedAt:    enqueuedAt,
			StartedAt:     startedAt,
			FinishedAt:    finishedAt,
			DurationMS:    durationMS,
			QueueWaitMS:   queueWaitMS,
			InferenceMS:   inferenceMS,
		})
		return
	}

	slog.Info("job done",
		slog.String(logschema.FieldEvent, string(logschema.EventJobDone)),
		slog.String(logschema.FieldCorrelationID, req.CorrelationID),
		slog.String(logschema.FieldRequestID, backendReq.RequestID),
		slog.Int64(logschema.FieldDurationMS, durationMS),
		slog.Int(logschema.FieldTokensGenerated, result.TokensGenerated),
		slog.String(logschema.FieldModelTier, result.ModelTier),
	)

	s.recordJob(types.JobEntry{
		RequestID:       backendReq.RequestID,
		CorrelationID:   req.CorrelationID,
		Source:          types.JobSourceLocal,
		Status:          types.JobStatusDone,
		Modality:        string(req.Modality),
		ModelTier:       result.ModelTier,
		Priority:        req.Priority,
		MinTier:         req.MinTier,
		EnqueuedAt:      enqueuedAt,
		StartedAt:       startedAt,
		FinishedAt:      finishedAt,
		DurationMS:      durationMS,
		QueueWaitMS:     queueWaitMS,
		InferenceMS:     inferenceMS,
		TokensGenerated: result.TokensGenerated,
	})

	if result.QualityDegraded {
		w.Header().Set(api.HeaderQualityDegraded, "true")
	}
	writeJSON(w, http.StatusOK, api.InferResponse{
		RequestID:       backendReq.RequestID,
		CorrelationID:   req.CorrelationID,
		Modality:        req.Modality,
		Output:          result.Output,
		Reasoning:       result.Reasoning,
		TokensGenerated: result.TokensGenerated,
		TokensPerSec:    result.TokPerSecSample,
		QueueWaitMS:     queueWaitMS,
		InferenceMS:     inferenceMS,
		DurationMS:      durationMS,
		ModelTier:       result.ModelTier,
		FinishedAt:      finishedAt,
	})
}
