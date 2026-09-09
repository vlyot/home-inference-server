// Package stackauth verifies Neon Auth (Stack Auth) access tokens presented by
// friends-and-family web apps. Stack Auth signs its JWTs with ES256 and
// publishes the public keys as a JWKS; this package fetches and caches that key
// set, refreshes it on an unknown key id, and validates the token's signature,
// issuer, and expiry.
package stackauth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var (
	ErrNoToken = errors.New("stackauth: empty token")
	ErrKeyID   = errors.New("stackauth: token key id not in JWKS")
	ErrInvalid = errors.New("stackauth: token invalid")
	ErrIssuer  = errors.New("stackauth: unexpected issuer")
)

// Claims is the subset of a Stack Auth access token this service cares about.
type Claims struct {
	// Subject is the Stack Auth user id — the stable identity for a member.
	Subject string
	// Email is the user's primary email when present in the token.
	Email string
}

// Verifier fetches and caches a JWKS and validates tokens against it.
type Verifier struct {
	jwksURL   string
	projectID string
	http      *http.Client

	mu       sync.RWMutex
	keys     map[string]*ecdsa.PublicKey
	fetched  time.Time
	minRefre time.Duration
}

// New returns a Verifier for the given JWKS URL and Stack Auth project id. The
// project id is the issuer scope: Stack Auth tokens carry
// iss = https://api.stack-auth.com/api/v1/projects/<projectID>.
func New(jwksURL, projectID string) *Verifier {
	return &Verifier{
		jwksURL:   jwksURL,
		projectID: projectID,
		http:      &http.Client{Timeout: 10 * time.Second},
		keys:      map[string]*ecdsa.PublicKey{},
		minRefre:  time.Minute,
	}
}

// newWithKeys builds a Verifier with a fixed key set and no network access.
// Used by tests that mint their own ES256 tokens.
func newWithKeys(projectID string, keys map[string]*ecdsa.PublicKey) *Verifier {
	return &Verifier{
		projectID: projectID,
		http:      &http.Client{Timeout: time.Second},
		keys:      keys,
		fetched:   time.Now(),
		minRefre:  time.Hour,
	}
}

// Verify validates an access token and returns its claims.
func (v *Verifier) Verify(ctx context.Context, token string) (Claims, error) {
	if token == "" {
		return Claims{}, ErrNoToken
	}

	parsed, err := jwt.Parse(token, func(t *jwt.Token) (any, error) {
		if _, ok := t.Method.(*jwt.SigningMethodECDSA); !ok {
			return nil, fmt.Errorf("stackauth: unexpected signing method %v", t.Header["alg"])
		}
		kid, _ := t.Header["kid"].(string)
		key, err := v.keyForID(ctx, kid)
		if err != nil {
			return nil, err
		}
		return key, nil
	}, jwt.WithValidMethods([]string{"ES256"}))
	if err != nil {
		if errors.Is(err, ErrKeyID) {
			return Claims{}, ErrKeyID
		}
		return Claims{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}

	mapClaims, ok := parsed.Claims.(jwt.MapClaims)
	if !ok || !parsed.Valid {
		return Claims{}, ErrInvalid
	}

	wantIss := "https://api.stack-auth.com/api/v1/projects/" + v.projectID
	if iss, _ := mapClaims["iss"].(string); iss != wantIss {
		return Claims{}, ErrIssuer
	}

	sub, _ := mapClaims["sub"].(string)
	if sub == "" {
		return Claims{}, ErrInvalid
	}
	email, _ := mapClaims["primary_email"].(string)
	if email == "" {
		email, _ = mapClaims["email"].(string)
	}
	return Claims{Subject: sub, Email: email}, nil
}

// keyForID returns the cached key, fetching the JWKS once if the id is unknown
// (and not fetched within the last minute).
func (v *Verifier) keyForID(ctx context.Context, kid string) (*ecdsa.PublicKey, error) {
	v.mu.RLock()
	key, ok := v.keys[kid]
	stale := time.Since(v.fetched) > v.minRefre
	v.mu.RUnlock()
	if ok {
		return key, nil
	}
	if !stale {
		return nil, ErrKeyID
	}

	if err := v.refresh(ctx); err != nil {
		return nil, err
	}

	v.mu.RLock()
	defer v.mu.RUnlock()
	if key, ok := v.keys[kid]; ok {
		return key, nil
	}
	return nil, ErrKeyID
}

type jwksDoc struct {
	Keys []struct {
		Kty string `json:"kty"`
		Crv string `json:"crv"`
		Kid string `json:"kid"`
		X   string `json:"x"`
		Y   string `json:"y"`
	} `json:"keys"`
}

func (v *Verifier) refresh(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.jwksURL, nil)
	if err != nil {
		return err
	}
	resp, err := v.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return fmt.Errorf("stackauth: JWKS fetch %d: %s", resp.StatusCode, body)
	}

	var doc jwksDoc
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		return fmt.Errorf("stackauth: decode JWKS: %w", err)
	}

	keys := make(map[string]*ecdsa.PublicKey, len(doc.Keys))
	for _, k := range doc.Keys {
		if k.Kty != "EC" || k.Crv != "P-256" || k.Kid == "" {
			continue
		}
		xb, err := base64.RawURLEncoding.DecodeString(k.X)
		if err != nil {
			continue
		}
		yb, err := base64.RawURLEncoding.DecodeString(k.Y)
		if err != nil {
			continue
		}
		keys[k.Kid] = &ecdsa.PublicKey{
			Curve: elliptic.P256(),
			X:     new(big.Int).SetBytes(xb),
			Y:     new(big.Int).SetBytes(yb),
		}
	}
	if len(keys) == 0 {
		return errors.New("stackauth: JWKS had no usable P-256 keys")
	}

	v.mu.Lock()
	v.keys = keys
	v.fetched = time.Now()
	v.mu.Unlock()
	return nil
}
