package stackauth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

const testProject = "proj-123"

func TestVerifyValidToken(t *testing.T) {
	ti := NewTestIssuer(testProject)
	claims, err := ti.Verifier.Verify(context.Background(), ti.Token("user-9", "u9@example.com"))
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if claims.Subject != "user-9" || claims.Email != "u9@example.com" {
		t.Fatalf("claims = %+v", claims)
	}
}

func TestVerifyRejectsExpired(t *testing.T) {
	ti := NewTestIssuer(testProject)
	if _, err := ti.Verifier.Verify(context.Background(), ti.ExpiredToken("user-9")); err == nil {
		t.Fatal("expired token accepted")
	}
}

func TestVerifyRejectsWrongIssuer(t *testing.T) {
	ti := NewTestIssuer(testProject)
	// A verifier scoped to a different project must reject the token.
	other := newWithKeys("proj-OTHER", ti.Verifier.keys)
	if _, err := other.Verify(context.Background(), ti.Token("user-9", "e")); err != ErrIssuer {
		t.Fatalf("want ErrIssuer, got %v", err)
	}
}

func TestVerifyRejectsUnknownKID(t *testing.T) {
	ti := NewTestIssuer(testProject)
	// Sign with a key the verifier doesn't have; minRefre is 1h so no refetch.
	priv, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	claims := jwt.MapClaims{
		"iss": "https://api.stack-auth.com/api/v1/projects/" + testProject,
		"sub": "user-9",
		"exp": time.Now().Add(time.Hour).Unix(),
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodES256, claims)
	tok.Header["kid"] = "stranger-kid"
	signed, _ := tok.SignedString(priv)
	if _, err := ti.Verifier.Verify(context.Background(), signed); err != ErrKeyID {
		t.Fatalf("want ErrKeyID, got %v", err)
	}
}

func TestVerifyRejectsAlgNone(t *testing.T) {
	ti := NewTestIssuer(testProject)
	claims := jwt.MapClaims{
		"iss": "https://api.stack-auth.com/api/v1/projects/" + testProject,
		"sub": "user-9",
		"exp": time.Now().Add(time.Hour).Unix(),
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodNone, claims)
	signed, _ := tok.SignedString(jwt.UnsafeAllowNoneSignatureType)
	if _, err := ti.Verifier.Verify(context.Background(), signed); err == nil {
		t.Fatal("alg=none token accepted")
	}
}

func TestVerifyEmptyToken(t *testing.T) {
	ti := NewTestIssuer(testProject)
	if _, err := ti.Verifier.Verify(context.Background(), ""); err != ErrNoToken {
		t.Fatalf("want ErrNoToken, got %v", err)
	}
}
