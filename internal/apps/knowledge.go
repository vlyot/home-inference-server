package apps

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// MaxKnowledgeBytes caps one knowledge file download.
const MaxKnowledgeBytes = 512 * 1024

// Knowledge is the generic shape every app's knowledge_url serves. Each site
// builds its own documents; the relay knows nothing about their meaning.
type Knowledge struct {
	Version   int        `json:"version"`
	Documents []Document `json:"documents"`
}

// Document is one citable source. Link is what a citation of it resolves to
// (a path or URL the client app renders). Always marks documents sent with
// every question, such as a site's profile.
type Document struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Link   string `json:"link,omitempty"`
	Text   string `json:"text"`
	Always bool   `json:"always,omitempty"`
}

func (k *Knowledge) validate() error {
	if k.Version != 1 {
		return fmt.Errorf("unsupported knowledge version %d", k.Version)
	}
	if len(k.Documents) == 0 {
		return fmt.Errorf("knowledge has no documents")
	}
	seen := map[string]bool{}
	for i, d := range k.Documents {
		if d.ID == "" || d.Title == "" || d.Text == "" {
			return fmt.Errorf("knowledge document %d needs id, title and text", i)
		}
		if seen[d.ID] {
			return fmt.Errorf("knowledge document id %q is repeated", d.ID)
		}
		seen[d.ID] = true
	}
	return nil
}

// Source holds the most recent valid knowledge from one URL, indexed for
// retrieval.
type Source struct {
	url    string
	client *http.Client
	// ready closes once the first fetch attempt has finished, good or bad.
	ready chan struct{}

	mu    sync.RWMutex
	index *Index
}

func NewSource(url string) *Source {
	return &Source{url: url, client: &http.Client{Timeout: 15 * time.Second}, ready: make(chan struct{})}
}

// Index returns the current index, or nil until a fetch has succeeded.
func (s *Source) Index() *Index {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.index
}

// Refresh fetches and validates the knowledge file. On any failure the
// previous index is kept.
func (s *Source) Refresh(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return err
	}
	res, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("knowledge fetch returned %d", res.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(res.Body, MaxKnowledgeBytes+1))
	if err != nil {
		return err
	}
	if len(body) > MaxKnowledgeBytes {
		return fmt.Errorf("knowledge is larger than %d bytes", MaxKnowledgeBytes)
	}
	var k Knowledge
	if err := json.Unmarshal(body, &k); err != nil {
		return fmt.Errorf("knowledge is not valid JSON: %w", err)
	}
	if err := k.validate(); err != nil {
		return err
	}
	idx := NewIndex(&k)
	s.mu.Lock()
	s.index = idx
	s.mu.Unlock()
	return nil
}

// Run refreshes immediately and then every interval until ctx is cancelled.
func (s *Source) Run(ctx context.Context, interval time.Duration) {
	refresh := func() {
		if err := s.Refresh(ctx); err != nil && ctx.Err() == nil {
			slog.Warn("app knowledge refresh failed", "url", s.url, "err", err)
		}
	}
	refresh()
	close(s.ready)
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			refresh()
		}
	}
}

// Registry starts one refreshing Source per knowledge URL the first time an
// app asks for it, so registering an app needs no relay restart.
type Registry struct {
	ctx      context.Context
	interval time.Duration

	mu      sync.Mutex
	sources map[string]*Source
}

// NewRegistry returns a Registry whose sources refresh every interval until
// ctx is cancelled.
func NewRegistry(ctx context.Context, interval time.Duration) *Registry {
	return &Registry{ctx: ctx, interval: interval, sources: map[string]*Source{}}
}

// Index returns the knowledge index for url. The first call for a URL waits
// (up to ctx) for its first fetch; nil means no valid knowledge is loaded.
func (r *Registry) Index(ctx context.Context, url string) *Index {
	r.mu.Lock()
	s, ok := r.sources[url]
	if !ok {
		s = NewSource(url)
		r.sources[url] = s
		go s.Run(r.ctx, r.interval)
	}
	r.mu.Unlock()

	select {
	case <-s.ready:
	case <-ctx.Done():
	}
	return s.Index()
}
