//go:build !windows

package llamacpp

import "os/exec"

// assignToJobObject is a no-op on non-Windows platforms; a graceful SIGINT to
// the subprocess is sufficient there.
func assignToJobObject(_ *exec.Cmd) (func(), error) {
	return func() {}, nil
}
