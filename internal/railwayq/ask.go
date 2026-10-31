package railwayq

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"golang.org/x/time/rate"

	"github.com/ngkaichong/home-inference-server/api"
	"github.com/ngkaichong/home-inference-server/internal/apps"
	"github.com/ngkaichong/home-inference-server/logschema"
	"github.com/ngkaichong/home-inference-server/queue"
)

const (
	askMaxBodyBytes = 8 * 1024
	askPoll         = 500 * time.Millisecond
	// knowledgeWait bounds how long a question waits for an app's knowledge
	// file to load the first time anyone asks after a relay restart.
	knowledgeWait = 10 * time.Second
)

// AskRequest is the body of POST /apps/ask. It is everything a public client
// can send: the app's own key, the visitor's question and their earlier
// questions. Earlier answers are never accepted from the client.
type AskRequest struct {
	Key     string   `json:"key"`
	Input   string   `json:"input"`
	History []string `json:"history,omitempty"`
}

// askStore is the part of the database the ask handler needs.
type askStore interface {
	LookupKey(ctx context.Context, key string) (App, error)
	PCConnected(ctx context.Context) (bool, error)
	UsageToday(ctx context.Context, identity string) (int, error)
	// Enqueue stores the job and counts it against the identity's daily usage.
	Enqueue(ctx context.Context, job queue.QueueJob) error
	Result(ctx context.Context, id, identity string) (queue.ResultResponse, error)
	CancelPending(ctx context.Context, id, identity string) error
}

type dbAskStore struct{ db *sql.DB }

func (s dbAskStore) LookupKey(ctx context.Context, key string) (App, error) {
	return LookupKey(ctx, s.db, key)
}

func (s dbAskStore) PCConnected(ctx context.Context) (bool, error) { return pcConnected(ctx, s.db) }

func (s dbAskStore) UsageToday(ctx context.Context, identity string) (int, error) {
	return UsageToday(ctx, s.db, identity)
}

func (s dbAskStore) Enqueue(ctx context.Context, job queue.QueueJob) error {
	if err := Enqueue(ctx, s.db, job); err != nil {
		return err
	}
	return IncrementUsage(ctx, s.db, job.Identity)
}

func (s dbAskStore) Result(ctx context.Context, id, identity string) (queue.ResultResponse, error) {
	return GetResult(ctx, s.db, id, identity)
}

func (s dbAskStore) CancelPending(ctx context.Context, id, identity string) error {
	return CancelPending(ctx, s.db, id, identity)
}

// knowledgeFunc returns the index for an app's knowledge_url, or nil when
// none is loaded. *apps.Registry.Index satisfies it.
type knowledgeFunc func(ctx context.Context, url string) *apps.Index

// askVisitors rate-limits each app's visitors by hashed client address, per
// minute and per UTC day, in memory only. Entries from earlier days are
// swept as the map is used.
type askVisitors struct {
	mu        sync.Mutex
	seen      map[string]*askVisitor
	lastSweep time.Time
}

type askVisitor struct {
	lim   *rate.Limiter
	day   string
	count int
}

func newAskVisitors() *askVisitors {
	return &askVisitors{seen: map[string]*askVisitor{}, lastSweep: time.Now()}
}

// allow returns "" when the visitor may ask, or the error code to reply with.
func (v *askVisitors) allow(key string, l apps.Limits, now time.Time) string {
	day := now.UTC().Format(time.DateOnly)
	v.mu.Lock()
	defer v.mu.Unlock()

	if now.Sub(v.lastSweep) > 10*time.Minute {
		for k, e := range v.seen {
			if e.day != day {
				delete(v.seen, k)
			}
		}
		v.lastSweep = now
	}

	e, ok := v.seen[key]
	if !ok || e.day != day {
		e = &askVisitor{lim: rate.NewLimiter(rate.Limit(l.VisitorPerMin/60), l.VisitorBurst), day: day}
		v.seen[key] = e
	}
	if e.count >= l.VisitorPerDay {
		return api.ErrCodeQuotaExceeded
	}
	if !e.lim.AllowN(now, 1) {
		return api.ErrCodeRateLimited
	}
	e.count++
	return ""
}

// inFlight counts each app's questions waiting on the PC.
type inFlight struct {
	mu sync.Mutex
	n  map[string]int
}

func (f *inFlight) acquire(appID string, max int) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.n[appID] >= max {
		return false
	}
	f.n[appID]++
	return true
}

func (f *inFlight) release(appID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.n[appID]--
}

// visitorKey hashes the client address per app. Railway's edge sets
// X-Real-IP itself; RemoteAddr covers local runs.
func visitorKey(r *http.Request, salt, appID string) string {
	ip := strings.TrimSpace(r.Header.Get("X-Real-IP"))
	if ip == "" {
		ip, _, _ = net.SplitHostPort(r.RemoteAddr)
	}
	sum := sha256.Sum256([]byte(salt + ip))
	return appID + ":" + hex.EncodeToString(sum[:12])
}

// handleAsk serves POST /apps/ask, the one inference route a browser can
// call without a secret. The app's key picks its server-side spec, and the
// relay builds the whole prompt from that spec; the client only supplies a
// question. limiter is the relay's per-identity limiter, so an app's
// questions and its secret-key /enqueue calls share one per-minute budget.
func handleAsk(cfg Config, store askStore, limiter *Limiter, knowledge knowledgeFunc) http.HandlerFunc {
	visitors := newAskVisitors()
	running := &inFlight{n: map[string]int{}}

	return func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if r.Method == http.MethodOptions {
			// A preflight carries no body, so no key and no app to check the
			// origin against. Allowing it is harmless: the POST that follows
			// is origin-checked before any work is done.
			if origin != "" {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Add("Vary", "Origin")
			}
			w.Header().Set("Access-Control-Allow-Methods", "POST")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.Header().Set("Access-Control-Max-Age", "600")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST, OPTIONS")
			writeErr(w, http.StatusMethodNotAllowed, api.ErrCodeInvalidRequest, "use POST")
			return
		}

		var req AskRequest
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, askMaxBodyBytes))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			var maxErr *http.MaxBytesError
			if errors.As(err, &maxErr) {
				writeErr(w, http.StatusRequestEntityTooLarge, api.ErrCodeInvalidRequest, "request body too large")
				return
			}
			writeErr(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, "malformed JSON")
			return
		}

		if !IsAppKey(req.Key) {
			writeErr(w, http.StatusUnauthorized, api.ErrCodeUnauthorized, "unknown or revoked app key")
			return
		}
		app, err := store.LookupKey(r.Context(), req.Key)
		if errors.Is(err, ErrNotFound) {
			writeErr(w, http.StatusUnauthorized, api.ErrCodeUnauthorized, "unknown or revoked app key")
			return
		}
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "key check failed")
			return
		}
		spec := app.Spec
		identity := app.Identity()

		if origin != "" {
			if !slices.Contains(spec.Origins, origin) {
				writeErr(w, http.StatusForbidden, api.ErrCodeUnauthorized, "origin not allowed for this app")
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Add("Vary", "Origin")
		}

		if msg := validateAsk(&req, spec.Limits); msg != "" {
			writeErr(w, http.StatusBadRequest, api.ErrCodeInvalidRequest, msg)
			return
		}

		if code := visitors.allow(visitorKey(r, cfg.AppIPSalt, spec.ID), spec.Limits, time.Now()); code != "" {
			logAsk(logschema.EventAskLimited, identity, code)
			if code == api.ErrCodeQuotaExceeded {
				writeErr(w, http.StatusTooManyRequests, code, "daily question limit reached, try again tomorrow")
			} else {
				writeErr(w, http.StatusTooManyRequests, code, "too many questions, wait a moment")
			}
			return
		}

		var idx *apps.Index
		if spec.KnowledgeURL != "" {
			kctx, cancel := context.WithTimeout(r.Context(), knowledgeWait)
			idx = knowledge(kctx, spec.KnowledgeURL)
			cancel()
			if idx == nil {
				logAsk(logschema.EventAskAsleep, identity, "no_knowledge")
				writeErr(w, http.StatusServiceUnavailable, api.ErrCodeUnavailable, "the server is asleep")
				return
			}
		}
		connected, err := store.PCConnected(r.Context())
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "status lookup failed")
			return
		}
		if !connected {
			logAsk(logschema.EventAskAsleep, identity, "pc_offline")
			writeErr(w, http.StatusServiceUnavailable, api.ErrCodeUnavailable, "the server is asleep")
			return
		}

		if !limiter.AllowAt(identity, spec.Limits.PerMin) {
			logAsk(logschema.EventAskLimited, identity, "app_per_min")
			writeErr(w, http.StatusTooManyRequests, api.ErrCodeRateLimited, "this app is busy, try again shortly")
			return
		}
		if !running.acquire(spec.ID, spec.Limits.MaxConcurrent) {
			logAsk(logschema.EventAskLimited, identity, api.ErrCodeOverloaded)
			writeErr(w, http.StatusServiceUnavailable, api.ErrCodeOverloaded, "busy, try again shortly")
			return
		}
		defer running.release(spec.ID)

		used, err := store.UsageToday(r.Context(), identity)
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "usage check failed")
			return
		}
		if used >= spec.Limits.PerDay {
			logAsk(logschema.EventAskLimited, identity, "app_per_day")
			writeErr(w, http.StatusTooManyRequests, api.ErrCodeQuotaExceeded, "this app has answered all it can today, try again tomorrow")
			return
		}

		var sources []apps.Doc
		if idx != nil {
			sources = idx.Retrieve(req.Input, req.History, spec.IgnoreTerms)
		}
		payload, err := json.Marshal(apps.BuildRequest(spec, sources, req.Input, req.History))
		if err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "request build failed")
			return
		}
		id := strings.ReplaceAll(uuid.New().String(), "-", "")
		job := queue.QueueJob{ID: id, CorrelationID: spec.ID + "-" + id, Payload: payload, MaxAttempts: 1, Identity: identity}
		if err := store.Enqueue(r.Context(), job); err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "enqueue failed")
			return
		}

		start := time.Now()
		res, err := waitForResult(r.Context(), store, id, identity, time.Duration(spec.Limits.WaitSeconds)*time.Second)
		if err != nil {
			// Out of time or the visitor left: make sure an unclaimed job never runs.
			cancelErr := store.CancelPending(context.WithoutCancel(r.Context()), id, identity)
			if r.Context().Err() != nil {
				return
			}
			if cancelErr == nil {
				writeErr(w, http.StatusServiceUnavailable, api.ErrCodeOverloaded, "busy, try again shortly")
			} else {
				writeErr(w, http.StatusGatewayTimeout, api.ErrCodeTimeout, "the answer took too long")
			}
			return
		}
		if res.Status == queue.QueueJobFailed {
			logAsk(logschema.EventAskFailed, identity, res.ErrorCode)
			writeErr(w, http.StatusServiceUnavailable, api.ErrCodeOverloaded, "the model couldn't answer, try again shortly")
			return
		}

		var out api.InferResponse
		if err := json.Unmarshal(res.ResultPayload, &out); err != nil {
			writeErr(w, http.StatusInternalServerError, api.ErrCodeInternal, "unreadable result")
			return
		}
		answer := apps.Clean(out.Output, sources, spec)
		event := logschema.EventAskAnswered
		if answer.Refused {
			event = logschema.EventAskRefused
		}
		// Questions are never logged, only their length.
		slog.Info("app ask",
			slog.String(logschema.FieldEvent, string(event)),
			slog.String(logschema.FieldIdentity, identity),
			slog.Int(logschema.FieldCount, utf8.RuneCountInString(req.Input)),
			slog.Int64(logschema.FieldDurationMS, time.Since(start).Milliseconds()),
			slog.String(logschema.FieldModelTier, out.ModelTier),
		)
		writeJSON(w, http.StatusOK, answer)
	}
}

func validateAsk(req *AskRequest, l apps.Limits) string {
	req.Input = strings.TrimSpace(req.Input)
	if req.Input == "" {
		return "input is required"
	}
	if utf8.RuneCountInString(req.Input) > l.MaxInputChars {
		return "input is too long"
	}
	if len(req.History) > l.MaxHistory {
		return "too many earlier questions"
	}
	for _, h := range req.History {
		if utf8.RuneCountInString(h) > l.MaxHistoryChars {
			return "an earlier question is too long"
		}
	}
	return ""
}

func waitForResult(ctx context.Context, store askStore, id, identity string, wait time.Duration) (queue.ResultResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, wait)
	defer cancel()
	t := time.NewTicker(askPoll)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return queue.ResultResponse{}, ctx.Err()
		case <-t.C:
			res, err := store.Result(ctx, id, identity)
			if err != nil {
				if ctx.Err() != nil {
					return queue.ResultResponse{}, ctx.Err()
				}
				return queue.ResultResponse{}, err
			}
			if res.Status == queue.QueueJobDone || res.Status == queue.QueueJobFailed {
				return res, nil
			}
		}
	}
}

func logAsk(event logschema.EventKind, identity, reason string) {
	slog.Info("app ask",
		slog.String(logschema.FieldEvent, string(event)),
		slog.String(logschema.FieldIdentity, identity),
		slog.String(logschema.FieldErrorCode, reason),
	)
}
