//go:build windows

package llamacpp

import (
	"errors"
	"os/exec"
	"unsafe"

	"golang.org/x/sys/windows"
)

// terminatePID asks the OS to kill exactly pid via TerminateProcess. A pid that
// no longer exists yields OpenProcess ERROR_INVALID_PARAMETER; callers treat
// that (via errors.Is) as "already gone". No logging here — the caller decides.
func terminatePID(pid int) error {
	h, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(h)
	return windows.TerminateProcess(h, 1)
}

// pidGone reports whether err from terminatePID/OpenProcess means there is
// nothing left to kill: the PID is invalid, or the process has already exited
// (Windows briefly returns ERROR_ACCESS_DENIED for a just-terminated PID before
// the object is reclaimed).
func pidGone(err error) bool {
	return errors.Is(err, windows.ERROR_INVALID_PARAMETER) ||
		errors.Is(err, windows.ERROR_ACCESS_DENIED)
}

// assignToJobObject puts the started subprocess into a new job object whose
// kill-on-close limit guarantees the whole process tree dies when the job
// handle is closed — including if this server process is force-killed. The
// returned func closes the handle; call it when the subprocess is stopped.
func assignToJobObject(cmd *exec.Cmd) (func(), error) {
	if cmd.Process == nil {
		return func() {}, nil
	}

	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}

	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{
		BasicLimitInformation: windows.JOBOBJECT_BASIC_LIMIT_INFORMATION{
			LimitFlags: windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
		},
	}
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}

	h, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	defer windows.CloseHandle(h)

	if err := windows.AssignProcessToJobObject(job, h); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}

	return func() { windows.CloseHandle(job) }, nil
}
