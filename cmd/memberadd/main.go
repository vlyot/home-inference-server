// Command memberadd manages the relay's friends-and-family invite list.
// Neon Auth (Stack Auth) lets anyone create an account; only Stack Auth user
// ids present in allowed_members may use the relay.
//
//	memberadd -email you@example.com            # invite (must have signed in once)
//	memberadd -email you@example.com -remove    # revoke
//	memberadd -list
//
// DATABASE_URL must point at the same Neon database the relay uses (the one
// Neon Auth syncs users into as neon_auth.users_sync).
package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"

	"github.com/ngkaichong/home-inference-server/internal/railwayq"
)

const ensureTable = `
CREATE TABLE IF NOT EXISTS allowed_members (
    stack_user_id  TEXT PRIMARY KEY,
    email          TEXT NOT NULL,
    added_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);`

func main() {
	email := flag.String("email", "", "member email (must have signed in via Neon Auth at least once)")
	remove := flag.Bool("remove", false, "remove the member instead of adding")
	list := flag.Bool("list", false, "list current members")
	flag.Parse()

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		fmt.Fprintln(os.Stderr, "error: DATABASE_URL is required")
		os.Exit(1)
	}
	dbURL = railwayq.NormalizeDSN(dbURL)

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open db: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	if _, err := db.ExecContext(ctx, ensureTable); err != nil {
		fmt.Fprintf(os.Stderr, "ensure table: %v\n", err)
		os.Exit(1)
	}

	if *list {
		rows, err := db.QueryContext(ctx,
			`SELECT stack_user_id, email, added_at FROM allowed_members ORDER BY added_at`)
		if err != nil {
			fmt.Fprintf(os.Stderr, "list: %v\n", err)
			os.Exit(1)
		}
		defer rows.Close()
		for rows.Next() {
			var id, em string
			var at time.Time
			rows.Scan(&id, &em, &at) //nolint:errcheck
			fmt.Printf("%s  %s  %s\n", id, em, at.Format(time.RFC3339))
		}
		return
	}

	if *email == "" {
		fmt.Fprintln(os.Stderr, "error: -email is required")
		flag.Usage()
		os.Exit(2)
	}

	if *remove {
		res, err := db.ExecContext(ctx,
			`DELETE FROM allowed_members WHERE lower(email) = lower($1)`, *email)
		if err != nil {
			fmt.Fprintf(os.Stderr, "remove: %v\n", err)
			os.Exit(1)
		}
		n, _ := res.RowsAffected()
		fmt.Printf("removed %d row(s) for %s\n", n, *email)
		return
	}

	// Resolve the Stack Auth user id from the synced directory.
	var stackID string
	err = db.QueryRowContext(ctx,
		`SELECT id FROM neon_auth.users_sync WHERE lower(email) = lower($1) AND deleted_at IS NULL`,
		*email,
	).Scan(&stackID)
	if err == sql.ErrNoRows {
		fmt.Fprintf(os.Stderr,
			"no Neon Auth user for %s — they must sign in once before being invited\n", *email)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "lookup: %v\n", err)
		os.Exit(1)
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO allowed_members (stack_user_id, email) VALUES ($1, $2)
		ON CONFLICT (stack_user_id) DO UPDATE SET email = EXCLUDED.email`,
		stackID, *email,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "insert: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("invited %s (stack_user_id=%s)\n", *email, stackID)
}
