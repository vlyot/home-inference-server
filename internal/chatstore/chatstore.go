// Package chatstore persists chat conversations as one JSON file each in a
// local directory. It holds at most MaxConversations; creating one past the cap
// evicts the least-recently-updated. All state stays on the local machine.
package chatstore

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

// MaxConversations is the hard cap on stored conversations.
const MaxConversations = 5

// Store is a directory-backed set of conversations, safe for concurrent use.
type Store struct {
	dir string

	mu    sync.RWMutex
	convs map[string]api.Conversation
}

// Open loads every *.json conversation in dir into memory, creating dir if
// needed. Files that fail to parse are skipped with a warning, not fatal.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{dir: dir, convs: make(map[string]api.Conversation)}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			slog.Warn("chatstore: unreadable conversation file", "file", e.Name(), "err", err)
			continue
		}
		var c api.Conversation
		if err := json.Unmarshal(data, &c); err != nil || c.ID == "" {
			slog.Warn("chatstore: skipping corrupt conversation file", "file", e.Name())
			continue
		}
		s.convs[c.ID] = c
	}
	return s, nil
}

// List returns all conversations as metadata, newest-updated first.
func (s *Store) List() []api.ConversationMeta {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]api.ConversationMeta, 0, len(s.convs))
	for _, c := range s.convs {
		out = append(out, meta(c))
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})
	return out
}

// Get returns one conversation and whether it exists.
func (s *Store) Get(id string) (api.Conversation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c, ok := s.convs[id]
	return c, ok
}

// Save creates or replaces a conversation. A brand-new conversation that would
// push the count past MaxConversations evicts the least-recently-updated one.
// CreatedAt is set on first save; UpdatedAt is always bumped to now. An empty
// Title is derived from the first user message.
func (s *Store) Save(c api.Conversation) (api.ConversationMeta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	if existing, ok := s.convs[c.ID]; ok {
		c.CreatedAt = existing.CreatedAt
	} else {
		c.CreatedAt = now
		if len(s.convs) >= MaxConversations {
			s.evictOldestLocked()
		}
	}
	c.UpdatedAt = now
	if strings.TrimSpace(c.Title) == "" {
		c.Title = deriveTitle(c.Messages)
	}

	if err := s.writeFileLocked(c); err != nil {
		return api.ConversationMeta{}, err
	}
	s.convs[c.ID] = c
	return meta(c), nil
}

// Delete removes a conversation. Missing ids are not an error.
func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.convs[id]; !ok {
		return nil
	}
	delete(s.convs, id)
	err := os.Remove(filepath.Join(s.dir, id+".json"))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func (s *Store) evictOldestLocked() {
	var oldestID string
	var oldest time.Time
	for id, c := range s.convs {
		if oldestID == "" || c.UpdatedAt.Before(oldest) {
			oldestID, oldest = id, c.UpdatedAt
		}
	}
	if oldestID == "" {
		return
	}
	delete(s.convs, oldestID)
	if err := os.Remove(filepath.Join(s.dir, oldestID+".json")); err != nil && !os.IsNotExist(err) {
		slog.Warn("chatstore: failed to remove evicted conversation", "id", oldestID, "err", err)
	}
}

func (s *Store) writeFileLocked(c api.Conversation) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	final := filepath.Join(s.dir, c.ID+".json")
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

func meta(c api.Conversation) api.ConversationMeta {
	turns := 0
	for _, m := range c.Messages {
		if m.Role == "user" {
			turns++
		}
	}
	return api.ConversationMeta{
		ID:        c.ID,
		Title:     c.Title,
		CreatedAt: c.CreatedAt,
		UpdatedAt: c.UpdatedAt,
		TurnCount: turns,
	}
}

func deriveTitle(msgs []api.ChatMessage) string {
	for _, m := range msgs {
		if m.Role != "user" {
			continue
		}
		line := strings.TrimSpace(strings.SplitN(m.Content, "\n", 2)[0])
		if line == "" {
			continue
		}
		r := []rune(line)
		if len(r) > 48 {
			return string(r[:48]) + "…"
		}
		return line
	}
	return "New conversation"
}
