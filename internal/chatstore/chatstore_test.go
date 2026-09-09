package chatstore

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ngkaichong/home-inference-server/api"
)

func conv(id string, userMsg string) api.Conversation {
	return api.Conversation{
		ID:       id,
		Messages: []api.ChatMessage{{Role: "user", Content: userMsg}},
	}
}

func TestSaveAndGetRoundTrips(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	in := conv("abc", "hello there")
	if _, err := s.Save(in); err != nil {
		t.Fatal(err)
	}
	got, ok := s.Get("abc")
	if !ok {
		t.Fatal("Get returned not-found after Save")
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "hello there" {
		t.Errorf("messages not round-tripped: %+v", got.Messages)
	}
	if got.CreatedAt.IsZero() || got.UpdatedAt.IsZero() {
		t.Error("timestamps not set")
	}
}

func TestListSortedByUpdatedDesc(t *testing.T) {
	s, _ := Open(t.TempDir())
	for _, id := range []string{"a", "b", "c"} {
		if _, err := s.Save(conv(id, id)); err != nil {
			t.Fatal(err)
		}
		time.Sleep(2 * time.Millisecond)
	}
	list := s.List()
	if len(list) != 3 {
		t.Fatalf("list len = %d; want 3", len(list))
	}
	if list[0].ID != "c" || list[2].ID != "a" {
		t.Errorf("order = %s,%s,%s; want c,b,a", list[0].ID, list[1].ID, list[2].ID)
	}
}

func TestSaveDerivesTitleFromFirstUserMessage(t *testing.T) {
	s, _ := Open(t.TempDir())
	m, err := s.Save(api.Conversation{
		ID: "x",
		Messages: []api.ChatMessage{
			{Role: "system", Content: "you are helpful"},
			{Role: "user", Content: "What is the capital of France?\nsecond line"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Title != "What is the capital of France?" {
		t.Errorf("title = %q", m.Title)
	}
}

func TestCapEvictsOldest(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	ids := []string{"c1", "c2", "c3", "c4", "c5", "c6"}
	for _, id := range ids {
		if _, err := s.Save(conv(id, id)); err != nil {
			t.Fatal(err)
		}
		time.Sleep(2 * time.Millisecond)
	}
	if len(s.List()) != MaxConversations {
		t.Fatalf("stored %d; want %d", len(s.List()), MaxConversations)
	}
	if _, ok := s.Get("c1"); ok {
		t.Error("c1 should have been evicted")
	}
	if _, err := os.Stat(filepath.Join(dir, "c1.json")); !os.IsNotExist(err) {
		t.Error("c1.json file should be gone")
	}
	if _, ok := s.Get("c6"); !ok {
		t.Error("c6 should be present")
	}
}

func TestSaveIsAtomic(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	if _, err := s.Save(conv("id1", "hi")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "id1.json.tmp")); !os.IsNotExist(err) {
		t.Error(".tmp file left behind after Save")
	}
	data, err := os.ReadFile(filepath.Join(dir, "id1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Error("saved file is empty")
	}
}

func TestOpenSkipsCorruptFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "garbage.json"), []byte("not json{{{"), 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := Open(dir)
	if err != nil {
		t.Fatalf("Open failed on corrupt file: %v", err)
	}
	if len(s.List()) != 0 {
		t.Errorf("corrupt file was loaded: %d entries", len(s.List()))
	}
}

func TestDeleteRemovesFileAndMap(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	s.Save(conv("d1", "hi"))
	if err := s.Delete("d1"); err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Get("d1"); ok {
		t.Error("still in map after Delete")
	}
	if _, err := os.Stat(filepath.Join(dir, "d1.json")); !os.IsNotExist(err) {
		t.Error("file still present after Delete")
	}
}

func TestDeleteMissingIsNoError(t *testing.T) {
	s, _ := Open(t.TempDir())
	if err := s.Delete("nope"); err != nil {
		t.Errorf("Delete of missing id errored: %v", err)
	}
}

func TestOpenReloadsSavedConversations(t *testing.T) {
	dir := t.TempDir()
	s1, _ := Open(dir)
	s1.Save(conv("keep", "remember me"))

	s2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := s2.Get("keep")
	if !ok || got.Messages[0].Content != "remember me" {
		t.Error("conversation not reloaded from disk on second Open")
	}
}
