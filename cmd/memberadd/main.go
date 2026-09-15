// Command memberadd manages the relay's friends-and-family invite list. It
// is the only way an account comes into existence — no client app exposes
// its own sign-up screen. memberadd creates the Stack Auth account directly
// (via the server-side admin API) and adds it to allowed_members in one step.
//
//	memberadd -email you@example.com            # provision + invite
//	memberadd -email you@example.com -remove    # revoke
//	memberadd -list
//
// DATABASE_URL must point at the same Neon database the relay uses.
// STACK_PROJECT_ID and STACK_SECRET_SERVER_KEY authenticate the account
// creation call against Stack Auth's server API.
package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"flag"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"

	"github.com/ngkaichong/home-inference-server/internal/railwayq"
	"github.com/ngkaichong/home-inference-server/internal/stackauth"
)

const ensureTable = `
CREATE TABLE IF NOT EXISTS allowed_members (
    stack_user_id  TEXT PRIMARY KEY,
    email          TEXT NOT NULL,
    added_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE allowed_members ADD COLUMN IF NOT EXISTS username TEXT;
ALTER TABLE allowed_members ADD COLUMN IF NOT EXISTS display_name TEXT;`

func main() {
	email := flag.String("email", "", "member email (an account is created if one doesn't already exist)")
	username := flag.String("username", "", "member username (admin-set; blank leaves any existing value unchanged)")
	name := flag.String("name", "", "member display name (admin-set; blank leaves any existing value unchanged)")
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
			`SELECT stack_user_id, email, username, display_name, added_at FROM allowed_members ORDER BY added_at`)
		if err != nil {
			fmt.Fprintf(os.Stderr, "list: %v\n", err)
			os.Exit(1)
		}
		defer rows.Close()
		for rows.Next() {
			var id, em string
			var uname, dname sql.NullString
			var at time.Time
			rows.Scan(&id, &em, &uname, &dname, &at) //nolint:errcheck
			fmt.Printf("%s  %s  %s  %s  %s\n", id, em, uname.String, dname.String, at.Format(time.RFC3339))
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

	projectID := os.Getenv("STACK_PROJECT_ID")
	secretKey := os.Getenv("STACK_SECRET_SERVER_KEY")
	if projectID == "" || secretKey == "" {
		fmt.Fprintln(os.Stderr, "error: STACK_PROJECT_ID and STACK_SECRET_SERVER_KEY are required")
		os.Exit(1)
	}
	admin := stackauth.NewAdminClient(projectID, secretKey)

	stackID, err := admin.FindUserByEmail(ctx, *email)
	if err != nil {
		fmt.Fprintf(os.Stderr, "lookup: %v\n", err)
		os.Exit(1)
	}
	created := stackID == ""
	if created {
		tempPassword, err := randomPassword()
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate password: %v\n", err)
			os.Exit(1)
		}
		stackID, err = admin.CreateUser(ctx, *email, *name, tempPassword)
		if err != nil {
			fmt.Fprintf(os.Stderr, "create user: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("provisioned %s (stack_user_id=%s, temp password: %s)\n", *email, stackID, tempPassword)
		fmt.Println("Give this password to the person directly — it is not stored or emailed.")
	}

	_, err = db.ExecContext(ctx, `
		INSERT INTO allowed_members (stack_user_id, email, username, display_name)
		VALUES ($1, $2, NULLIF($3, ''), NULLIF($4, ''))
		ON CONFLICT (stack_user_id) DO UPDATE
			SET email = EXCLUDED.email,
			    username = COALESCE(EXCLUDED.username, allowed_members.username),
			    display_name = COALESCE(EXCLUDED.display_name, allowed_members.display_name)`,
		stackID, *email, *username, *name,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "insert: %v\n", err)
		os.Exit(1)
	}
	if !created {
		fmt.Printf("invited %s (stack_user_id=%s)\n", *email, stackID)
	}
}

// randomPassword returns a 24-byte cryptographically random password,
// base64url-encoded, for a freshly provisioned account. It is printed once
// and never stored — the person is expected to set their own on first use.
func randomPassword() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}
