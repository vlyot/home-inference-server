package stackauth

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// TestIssuer mints ES256 tokens against an in-memory keypair and hands back a
// Verifier wired to the matching public key. It exists so other packages'
// tests can exercise the Bearer path without reaching api.stack-auth.com.
type TestIssuer struct {
	projectID string
	kid       string
	priv      *ecdsa.PrivateKey
	Verifier  *Verifier
}

// NewTestIssuer creates a keypair for the given Stack Auth project id.
func NewTestIssuer(projectID string) *TestIssuer {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(err)
	}
	const kid = "test-kid"
	return &TestIssuer{
		projectID: projectID,
		kid:       kid,
		priv:      priv,
		Verifier:  newWithKeys(projectID, map[string]*ecdsa.PublicKey{kid: &priv.PublicKey}),
	}
}

// Token returns a signed access token for the given subject and email.
func (ti *TestIssuer) Token(subject, email string) string {
	return ti.token(subject, email, time.Now().Add(time.Hour))
}

// ExpiredToken returns a token whose exp is in the past.
func (ti *TestIssuer) ExpiredToken(subject string) string {
	return ti.token(subject, "", time.Now().Add(-time.Minute))
}

func (ti *TestIssuer) token(subject, email string, exp time.Time) string {
	claims := jwt.MapClaims{
		"iss": "https://api.stack-auth.com/api/v1/projects/" + ti.projectID,
		"sub": subject,
		"exp": exp.Unix(),
		"iat": time.Now().Unix(),
	}
	if email != "" {
		claims["primary_email"] = email
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodES256, claims)
	tok.Header["kid"] = ti.kid
	signed, err := tok.SignedString(ti.priv)
	if err != nil {
		panic(err)
	}
	return signed
}
