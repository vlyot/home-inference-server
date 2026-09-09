package railwayq

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

const schema = `
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    correlation_id  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    payload         JSONB NOT NULL,
    result_payload  JSONB,
    error_code      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    attempt_count   INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 3,
    identity        TEXT NOT NULL DEFAULT ''
);
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_code TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS identity TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS jobs_identity_idx ON jobs(identity);

CREATE TABLE IF NOT EXISTS usage (
    identity  TEXT NOT NULL,
    day       DATE NOT NULL,
    count     INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (identity, day)
);

CREATE TABLE IF NOT EXISTS allowed_members (
    stack_user_id  TEXT PRIMARY KEY,
    email          TEXT NOT NULL,
    added_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- worker_heartbeat records the last time the PC worker polled /claim, even when
-- the long-poll returned no job. GET /status derives pc_connected from this, so
-- a connected-but-idle PC (or one whose old job rows were TTL-purged) still
-- reads as connected.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_id    TEXT PRIMARY KEY,
    last_poll_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
`

// Open opens a Postgres connection pool and applies the schema migration.
func Open(databaseURL string) (*sql.DB, error) {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return nil, fmt.Errorf("open postgres: %w", err)
	}
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	// The relay runs behind a connection pooler (Railway / Neon); keep the
	// per-instance pool small and recycle idle connections so we don't exhaust
	// the pooler's client-side limit.
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(5)
	db.SetConnMaxIdleTime(2 * time.Minute)
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	return db, nil
}
