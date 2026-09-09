package vram

import (
	"log/slog"

	"github.com/ngkaichong/home-inference-server/logschema"
)

// llamaImageName is the exact executable base name the reaper is willing to kill.
const llamaImageName = "llama-server.exe"

// portOwner abstracts the OS calls the reaper needs. Injected for tests.
type portOwner interface {
	// pidOnPort returns the PID listening on the given localhost port, or 0.
	pidOnPort(port int) int
	// imageName returns the lowercased executable base name for a PID, or "".
	imageName(pid int) string
	// kill terminates a PID.
	kill(pid int) error
}

// Reaper kills stray llama-server subprocesses left listening on one of our
// roster ports — for example after a previous server crash, or if an eviction
// failed to tear a subprocess down. It only ever targets a process that is
// (1) on a roster port, (2) not one we currently track, and (3) named exactly
// llama-server.exe.
type Reaper struct {
	ports []int
	owned func() map[int]bool
	os    portOwner
}

// NewReaper builds a Reaper for the given roster ports. owned returns the set of
// PIDs of subprocesses the server currently manages (and must not kill).
func NewReaper(ports []int, owned func() map[int]bool) *Reaper {
	return &Reaper{ports: ports, owned: owned, os: defaultPortOwner{}}
}

// Sweep checks every roster port once and kills any stray llama-server on it.
// It returns the number of processes it successfully killed (a kill that errored
// is not counted).
func (r *Reaper) Sweep() int {
	var tracked map[int]bool
	if r.owned != nil {
		tracked = r.owned()
	}
	killed := 0
	for _, port := range r.ports {
		pid := r.os.pidOnPort(port)
		if pid == 0 || tracked[pid] {
			continue
		}
		if r.os.imageName(pid) != llamaImageName {
			continue
		}
		slog.Warn("reaper: killing stray llama-server",
			slog.String(logschema.FieldEvent, string(logschema.EventModelEvicted)),
			slog.Int("pid", pid),
			slog.Int("port", port),
		)
		if err := r.os.kill(pid); err != nil {
			// A stray we can't kill (usually: started by a more-privileged
			// session). It keeps holding VRAM. The autostart task runs elevated
			// so this shouldn't happen in a real deploy; surface it loudly.
			slog.Error("reaper: could NOT kill a stray llama-server — it is still holding VRAM; kill it manually or run elevated",
				slog.Int("pid", pid),
				slog.Int("port", port),
				slog.Any("err", err),
			)
			continue
		}
		killed++
	}
	return killed
}
