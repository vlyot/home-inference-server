package railwayq

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/apps"
)

// keyPrefixLen is how much of a key is stored in clear for listing and
// revoking: "pk_" plus 9 random characters.
const keyPrefixLen = 12

// App is an enabled app resolved from one of its keys.
type App struct {
	Spec apps.Spec
	// Kind is the kind of key that resolved it: apps.KeyPublishable or
	// apps.KeySecret.
	Kind string
}

// Identity is the job/usage identity every request on the app's behalf
// runs under, whichever key made it.
func (a App) Identity() string { return "app:" + a.Spec.ID }

// IsAppKey reports whether a credential looks like an app key rather than
// the owner's static key.
func IsAppKey(key string) bool {
	return strings.HasPrefix(key, apps.KeyPublishable+"_") || strings.HasPrefix(key, apps.KeySecret+"_")
}

func hashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// LookupKey resolves an app key to its app. A key that is unknown, revoked or
// belongs to a disabled app returns ErrNotFound — deliberately
// indistinguishable.
func LookupKey(ctx context.Context, db *sql.DB, key string) (App, error) {
	var kind string
	var raw []byte
	err := db.QueryRowContext(ctx, `
		SELECT k.kind, a.spec
		FROM app_keys k JOIN apps a ON a.id = k.app_id
		WHERE k.key_hash = $1 AND k.revoked_at IS NULL AND a.enabled`,
		hashKey(key),
	).Scan(&kind, &raw)
	if err == sql.ErrNoRows {
		return App{}, ErrNotFound
	}
	if err != nil {
		return App{}, err
	}
	spec, err := apps.ParseSpec(raw)
	if err != nil {
		return App{}, fmt.Errorf("stored spec: %w", err)
	}
	return App{Spec: spec, Kind: kind}, nil
}

// UpsertApp stores a validated spec, creating the app (enabled) or replacing
// an existing app's spec. Enabled state is left as it was on update.
func UpsertApp(ctx context.Context, db *sql.DB, spec apps.Spec) error {
	raw, err := json.Marshal(spec)
	if err != nil {
		return err
	}
	_, err = db.ExecContext(ctx, `
		INSERT INTO apps (id, spec) VALUES ($1, $2)
		ON CONFLICT (id) DO UPDATE SET spec = EXCLUDED.spec, updated_at = NOW()`,
		spec.ID, string(raw),
	)
	return err
}

// SetAppEnabled switches every key of an app on or off at once.
func SetAppEnabled(ctx context.Context, db *sql.DB, id string, enabled bool) error {
	res, err := db.ExecContext(ctx, `UPDATE apps SET enabled = $2, updated_at = NOW() WHERE id = $1`, id, enabled)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// IssueKey creates a new key of kind for an app and returns it. Only its
// hash is stored, so this is the one time the key is ever visible.
func IssueKey(ctx context.Context, db *sql.DB, appID, kind string) (string, error) {
	if kind != apps.KeyPublishable && kind != apps.KeySecret {
		return "", fmt.Errorf("unknown key kind %q", kind)
	}
	buf := make([]byte, 24)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	key := kind + "_" + hex.EncodeToString(buf)
	_, err := db.ExecContext(ctx, `
		INSERT INTO app_keys (key_hash, app_id, kind, prefix) VALUES ($1, $2, $3, $4)`,
		hashKey(key), appID, kind, key[:keyPrefixLen],
	)
	if err != nil {
		return "", err
	}
	return key, nil
}

// RevokeKey revokes the key whose stored prefix matches.
func RevokeKey(ctx context.Context, db *sql.DB, prefix string) error {
	res, err := db.ExecContext(ctx,
		`UPDATE app_keys SET revoked_at = NOW() WHERE prefix = $1 AND revoked_at IS NULL`, prefix)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// AppKeyInfo is one key as listed by cmd/appadd.
type AppKeyInfo struct {
	Prefix    string
	Kind      string
	CreatedAt time.Time
	Revoked   bool
}

// AppInfo is one app as listed by cmd/appadd.
type AppInfo struct {
	ID      string
	Enabled bool
	Keys    []AppKeyInfo
}

// ListApps returns every app with its keys, apps ordered by id.
func ListApps(ctx context.Context, db *sql.DB) ([]AppInfo, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT a.id, a.enabled, k.prefix, k.kind, k.created_at, k.revoked_at IS NOT NULL
		FROM apps a LEFT JOIN app_keys k ON k.app_id = a.id
		ORDER BY a.id, k.created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []AppInfo
	for rows.Next() {
		var id string
		var enabled bool
		var prefix, kind sql.NullString
		var created sql.NullTime
		var revoked sql.NullBool
		if err := rows.Scan(&id, &enabled, &prefix, &kind, &created, &revoked); err != nil {
			return nil, err
		}
		if len(out) == 0 || out[len(out)-1].ID != id {
			out = append(out, AppInfo{ID: id, Enabled: enabled})
		}
		if prefix.Valid {
			last := &out[len(out)-1]
			last.Keys = append(last.Keys, AppKeyInfo{Prefix: prefix.String, Kind: kind.String, CreatedAt: created.Time, Revoked: revoked.Bool})
		}
	}
	return out, rows.Err()
}
