// Package apps defines public client apps: the server-side spec that pins
// what an app's publishable key can make the model do, the knowledge loader
// and retrieval that ground its answers, the prompt frame and the answer
// cleaner. It has no HTTP-handler or database dependencies; the relay
// (internal/railwayq) wires it to keys, limits and the job queue. See
// docs/public-apps.md for the threat model.
package apps

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"

	"github.com/ngkaichong/home-inference-server/api"
)

// Key kinds. A publishable key ("pk_…") is safe to ship in a browser: it only
// reaches POST /apps/ask, where the relay builds the whole prompt. A secret
// key ("sk_…") is for an app's own backend: it reaches the raw /enqueue
// contract under the app's identity and limits.
const (
	KeyPublishable = "pk"
	KeySecret      = "sk"
)

// DefaultRefusal is used when an app grounds its answers in knowledge but
// does not set its own refusal sentence.
const DefaultRefusal = "I can't answer that from what I know."

// Spec is one app's server-side definition, stored as JSONB in the relay's
// apps table and pushed with cmd/appadd from a file kept in the app's repo.
type Spec struct {
	ID string `json:"id"`
	// Origins are the browser origins allowed to call /apps/ask, e.g.
	// "https://vlyot.github.io". Requests without an Origin (scripts,
	// backends) are not origin-checked; the pinned prompt and quotas are the
	// real defence.
	Origins []string `json:"origins"`
	// SystemPrompt is the app's own instructions: who the assistant is and
	// what it talks about. The relay wraps it in a fixed safety frame.
	SystemPrompt string `json:"system_prompt"`
	// KnowledgeURL, when set, is fetched every few minutes and ranked against
	// each question; the best documents become numbered sources.
	KnowledgeURL string `json:"knowledge_url,omitempty"`
	// RequireCitations replaces any answer without a valid [n] citation with
	// Refusal. Requires KnowledgeURL.
	RequireCitations bool   `json:"require_citations,omitempty"`
	Refusal          string `json:"refusal,omitempty"`
	// IgnoreTerms are dropped from questions before ranking — words every
	// document is about (the site owner's name), which say nothing about
	// relevance.
	IgnoreTerms []string `json:"ignore_terms,omitempty"`
	MinTier     string   `json:"min_tier,omitempty"`
	// PreferredTier, when set, pins the PC to exactly this tier instead of the
	// largest one that fits in VRAM. It can't sit below MinTier.
	PreferredTier   string `json:"preferred_tier,omitempty"`
	MaxOutputTokens int    `json:"max_output_tokens,omitempty"`
	Limits          Limits `json:"limits"`
}

// Limits bound what one app can consume. Zero values take the defaults in
// ApplyDefaults.
type Limits struct {
	MaxInputChars   int `json:"max_input_chars,omitempty"`
	MaxHistory      int `json:"max_history,omitempty"`
	MaxHistoryChars int `json:"max_history_chars,omitempty"`

	// Per visitor (hashed client IP) on /apps/ask.
	VisitorPerMin float64 `json:"visitor_per_min,omitempty"`
	VisitorBurst  int     `json:"visitor_burst,omitempty"`
	VisitorPerDay int     `json:"visitor_per_day,omitempty"`
	// HostedVisitorPerDay is the separate daily count for questions the
	// hosted model answers while the PC is offline; it defaults to
	// VisitorPerDay. Answers from the PC never use it up.
	HostedVisitorPerDay int `json:"hosted_visitor_per_day,omitempty"`

	// Whole app: shared by /apps/ask and the app's secret-key /enqueue calls.
	PerMin float64 `json:"per_min,omitempty"`
	PerDay int     `json:"per_day,omitempty"`

	// MaxConcurrent caps /apps/ask questions in flight at once.
	MaxConcurrent int `json:"max_concurrent,omitempty"`
	// WaitSeconds is how long /apps/ask waits for the PC before giving up and
	// cancelling the job.
	WaitSeconds int `json:"wait_seconds,omitempty"`
}

var idRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{1,39}$`)

// ParseSpec decodes a spec strictly (unknown fields are an error), fills
// defaults and validates it.
func ParseSpec(data []byte) (Spec, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var s Spec
	if err := dec.Decode(&s); err != nil {
		return Spec{}, fmt.Errorf("app spec: %w", err)
	}
	s.ApplyDefaults()
	if err := s.Validate(); err != nil {
		return Spec{}, err
	}
	return s, nil
}

// ApplyDefaults fills every zero-valued knob.
func (s *Spec) ApplyDefaults() {
	if s.MinTier == "" {
		// The weak tier forces the chat template, where Gemma starts
		// "thinking"; mid and strong answer the plain prompt directly.
		s.MinTier = api.MinTierMid
	}
	if s.MaxOutputTokens == 0 {
		s.MaxOutputTokens = 400
	}
	if s.KnowledgeURL != "" && s.Refusal == "" {
		s.Refusal = DefaultRefusal
	}
	l := &s.Limits
	setInt(&l.MaxInputChars, 400)
	setInt(&l.MaxHistory, 3)
	setInt(&l.MaxHistoryChars, 200)
	if l.VisitorPerMin == 0 {
		l.VisitorPerMin = 5
	}
	setInt(&l.VisitorBurst, 3)
	setInt(&l.VisitorPerDay, 30)
	setInt(&l.HostedVisitorPerDay, l.VisitorPerDay)
	if l.PerMin == 0 {
		l.PerMin = 20
	}
	setInt(&l.PerDay, 300)
	setInt(&l.MaxConcurrent, 2)
	setInt(&l.WaitSeconds, 35)
}

func setInt(p *int, def int) {
	if *p == 0 {
		*p = def
	}
}

// Validate rejects a spec the relay could not enforce safely.
func (s *Spec) Validate() error {
	if !idRe.MatchString(s.ID) {
		return errors.New("app spec: id must be 2-40 chars of a-z, 0-9 and -")
	}
	if len(s.Origins) == 0 {
		return errors.New("app spec: origins must list at least one origin")
	}
	for _, o := range s.Origins {
		u, err := url.Parse(o)
		if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.Path != "" {
			return fmt.Errorf("app spec: origin %q must be scheme://host with no path", o)
		}
	}
	if s.SystemPrompt == "" {
		return errors.New("app spec: system_prompt is required")
	}
	if s.KnowledgeURL != "" {
		if u, err := url.Parse(s.KnowledgeURL); err != nil || u.Scheme != "https" {
			return errors.New("app spec: knowledge_url must be an https URL")
		}
	}
	if s.RequireCitations && s.KnowledgeURL == "" {
		return errors.New("app spec: require_citations needs knowledge_url")
	}
	switch s.MinTier {
	case api.MinTierWeak, api.MinTierMid, api.MinTierStrong:
	default:
		return fmt.Errorf("app spec: unknown min_tier %q", s.MinTier)
	}
	if s.PreferredTier != "" {
		rank := map[string]int{api.MinTierWeak: 1, api.MinTierMid: 2, api.MinTierStrong: 3}
		if rank[s.PreferredTier] == 0 {
			return fmt.Errorf("app spec: unknown preferred_tier %q", s.PreferredTier)
		}
		if rank[s.PreferredTier] < rank[s.MinTier] {
			return errors.New("app spec: preferred_tier is below min_tier")
		}
	}
	l := s.Limits
	if s.MaxOutputTokens < 0 || l.MaxInputChars < 0 || l.MaxHistory < 0 || l.MaxHistoryChars < 0 ||
		l.VisitorPerMin < 0 || l.VisitorBurst < 0 || l.VisitorPerDay < 0 || l.HostedVisitorPerDay < 0 ||
		l.PerMin < 0 || l.PerDay < 0 || l.MaxConcurrent < 0 || l.WaitSeconds < 0 {
		return errors.New("app spec: limits must not be negative")
	}
	if l.WaitSeconds < 10 {
		return errors.New("app spec: wait_seconds must be at least 10")
	}
	return nil
}
