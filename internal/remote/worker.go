package remote

import (
	"fmt"
	"os"
)

// WorkerID returns a stable identifier for this process instance.
// Format: "<hostname>-<pid>". Unique across machines and process restarts.
func WorkerID() string {
	host, _ := os.Hostname()
	return fmt.Sprintf("%s-%d", host, os.Getpid())
}
