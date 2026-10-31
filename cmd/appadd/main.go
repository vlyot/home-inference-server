// Command appadd manages the relay's public client apps and their keys. An
// app's spec (origins, pinned prompt, knowledge URL, limits — see
// internal/apps.Spec) lives as a JSON file in the app's own repo and is
// pushed here; keys are printed once when issued and only their hash is
// stored.
//
//	appadd -spec his-app.json                 # create or update the app
//	appadd -spec his-app.json -new-key pk     # ... and issue a publishable key
//	appadd -app portfolio -new-key sk         # issue a secret key
//	appadd -revoke pk_1a2b3c4d5               # revoke one key by its listed prefix
//	appadd -app portfolio -disable            # switch every key off (-enable to undo)
//	appadd -list
//
// DATABASE_URL must point at the same Neon database the relay uses.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/ngkaichong/home-inference-server/internal/apps"
	"github.com/ngkaichong/home-inference-server/internal/railwayq"
)

func main() {
	specPath := flag.String("spec", "", "app spec JSON file to create or update")
	appID := flag.String("app", "", "app id (taken from -spec when given)")
	newKey := flag.String("new-key", "", `issue a key: "pk" (publishable, for browsers) or "sk" (secret, for a backend)`)
	revoke := flag.String("revoke", "", "revoke the key with this prefix (as shown by -list)")
	disable := flag.Bool("disable", false, "disable the app")
	enable := flag.Bool("enable", false, "re-enable the app")
	list := flag.Bool("list", false, "list apps and their keys")
	flag.Parse()

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		fail("DATABASE_URL is required")
	}
	db, err := railwayq.Open(railwayq.NormalizeDSN(dbURL))
	if err != nil {
		fail("open db: %v", err)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	switch {
	case *list:
		infos, err := railwayq.ListApps(ctx, db)
		if err != nil {
			fail("list: %v", err)
		}
		for _, a := range infos {
			state := "enabled"
			if !a.Enabled {
				state = "disabled"
			}
			fmt.Printf("%s  %s\n", a.ID, state)
			for _, k := range a.Keys {
				status := ""
				if k.Revoked {
					status = "  revoked"
				}
				fmt.Printf("  %s…  %s  %s%s\n", k.Prefix, k.Kind, k.CreatedAt.Format(time.RFC3339), status)
			}
		}
		return

	case *revoke != "":
		if err := railwayq.RevokeKey(ctx, db, *revoke); err != nil {
			fail("revoke %s: %v", *revoke, err)
		}
		fmt.Printf("revoked %s…\n", *revoke)
		return
	}

	if *specPath != "" {
		data, err := os.ReadFile(*specPath)
		if err != nil {
			fail("read spec: %v", err)
		}
		spec, err := apps.ParseSpec(data)
		if err != nil {
			fail("%v", err)
		}
		if err := railwayq.UpsertApp(ctx, db, spec); err != nil {
			fail("upsert: %v", err)
		}
		*appID = spec.ID
		fmt.Printf("saved app %s\n", spec.ID)
	}
	if *appID == "" {
		fmt.Fprintln(os.Stderr, "error: -spec or -app is required")
		flag.Usage()
		os.Exit(2)
	}

	if *disable || *enable {
		if err := railwayq.SetAppEnabled(ctx, db, *appID, *enable); err != nil {
			fail("set enabled: %v", err)
		}
		fmt.Printf("app %s enabled=%t\n", *appID, *enable)
	}

	if *newKey != "" {
		key, err := railwayq.IssueKey(ctx, db, *appID, *newKey)
		if err != nil {
			fail("issue key: %v", err)
		}
		fmt.Printf("new %s key for %s (shown once, not stored):\n%s\n", *newKey, *appID, key)
	}
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
