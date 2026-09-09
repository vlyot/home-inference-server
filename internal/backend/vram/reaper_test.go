package vram

import (
	"errors"
	"testing"
)

type fakePortOwner struct {
	byPort  map[int]int
	images  map[int]string
	killed  []int
	killErr error
}

func (f *fakePortOwner) pidOnPort(port int) int { return f.byPort[port] }
func (f *fakePortOwner) imageName(pid int) string {
	if s, ok := f.images[pid]; ok {
		return s
	}
	return ""
}
func (f *fakePortOwner) kill(pid int) error {
	f.killed = append(f.killed, pid)
	return f.killErr
}

func newReaperWithOwner(ports []int, tracked map[int]bool, fo portOwner) *Reaper {
	return &Reaper{ports: ports, owned: func() map[int]bool { return tracked }, os: fo}
}

func TestReaper_KillsStrayLlamaOnRosterPort(t *testing.T) {
	fo := &fakePortOwner{
		byPort: map[int]int{8092: 999},
		images: map[int]string{999: "llama-server.exe"},
	}
	r := newReaperWithOwner([]int{8090, 8091, 8092, 8093}, nil, fo)
	r.Sweep()
	if len(fo.killed) != 1 || fo.killed[0] != 999 {
		t.Errorf("killed = %v; want [999]", fo.killed)
	}
}

func TestReaper_SkipsTrackedSubprocess(t *testing.T) {
	fo := &fakePortOwner{
		byPort: map[int]int{8092: 555},
		images: map[int]string{555: "llama-server.exe"},
	}
	r := newReaperWithOwner([]int{8092}, map[int]bool{555: true}, fo)
	r.Sweep()
	if len(fo.killed) != 0 {
		t.Errorf("killed a tracked subprocess: %v", fo.killed)
	}
}

func TestReaper_SkipsNonLlamaProcess(t *testing.T) {
	fo := &fakePortOwner{
		byPort: map[int]int{8091: 42},
		images: map[int]string{42: "chrome.exe"},
	}
	r := newReaperWithOwner([]int{8091}, nil, fo)
	r.Sweep()
	if len(fo.killed) != 0 {
		t.Errorf("killed a non-llama process: %v", fo.killed)
	}
}

func TestReaper_IgnoresPortsOutsideRoster(t *testing.T) {
	fo := &fakePortOwner{
		byPort: map[int]int{9999: 7},
		images: map[int]string{7: "llama-server.exe"},
	}
	r := newReaperWithOwner([]int{8090, 8091, 8092, 8093}, nil, fo)
	r.Sweep()
	if len(fo.killed) != 0 {
		t.Errorf("killed a process outside the roster ports: %v", fo.killed)
	}
}

func TestReaper_NoListenerNoop(t *testing.T) {
	fo := &fakePortOwner{byPort: map[int]int{}}
	r := newReaperWithOwner([]int{8090, 8091}, nil, fo)
	r.Sweep()
	if len(fo.killed) != 0 {
		t.Errorf("killed something with no listener present: %v", fo.killed)
	}
}

func TestReaper_KillErrorDoesNotPanic(t *testing.T) {
	fo := &fakePortOwner{
		byPort:  map[int]int{8093: 111},
		images:  map[int]string{111: "llama-server.exe"},
		killErr: errors.New("access denied"),
	}
	r := newReaperWithOwner([]int{8093}, nil, fo)
	if n := r.Sweep(); n != 0 { // a kill that errored is not counted
		t.Errorf("Sweep() = %d; want 0 (kill errored)", n)
	}
	if len(fo.killed) != 1 {
		t.Errorf("kill not attempted: %v", fo.killed)
	}
}

func TestReaper_SweepReturnsKillCount(t *testing.T) {
	fo := &fakePortOwner{
		byPort: map[int]int{8092: 999, 8093: 998},
		images: map[int]string{999: "llama-server.exe", 998: "llama-server.exe"},
	}
	r := newReaperWithOwner([]int{8090, 8091, 8092, 8093}, nil, fo)
	if n := r.Sweep(); n != 2 {
		t.Errorf("Sweep() = %d; want 2", n)
	}
}
