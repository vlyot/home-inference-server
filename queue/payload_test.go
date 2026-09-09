package queue

import (
	"encoding/json"
	"strings"
	"testing"
)

// json.RawMessage payloads must travel as a JSON object on the wire, not a
// base64-encoded string (which a []byte field would produce), so a non-Go
// client of /claim or /ack can read them.
func TestPayloadsMarshalAsJSONObjects(t *testing.T) {
	job := QueueJob{
		ID:      "j1",
		Payload: json.RawMessage(`{"modality":"text","text_input":{"prompt":"hi"}}`),
	}
	b, err := json.Marshal(job)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"payload":{"modality":"text"`) {
		t.Fatalf("payload was not embedded as an object: %s", b)
	}

	ack := AckRequest{JobID: "j1", Status: QueueJobDone, ResultPayload: json.RawMessage(`{"output":"ok"}`)}
	ab, err := json.Marshal(ack)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ab), `"result_payload":{"output":"ok"}`) {
		t.Fatalf("result_payload was not embedded as an object: %s", ab)
	}

	// And it round-trips.
	var back QueueJob
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(back.Payload, &m); err != nil {
		t.Fatalf("payload did not round-trip as an object: %v", err)
	}
}
