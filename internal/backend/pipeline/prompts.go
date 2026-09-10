package pipeline

// VisionPerceptionPrompt is what the perceive stage (a small VLM) is asked —
// not the user's question. It aims for an exhaustive, literal inventory so the
// reasoning stage has as much to work with as possible; the VLM never has to be
// the smart one.
const VisionPerceptionPrompt = `List every visible element in this image with ` +
	`literal detail: all objects and where they are, every piece of text and ` +
	`every number exactly as written, colours, spatial relationships, any ` +
	`people and what they are doing, and any other notable detail. Be ` +
	`exhaustive and factual. Do not summarise, interpret, or answer questions — ` +
	`only inventory what is present.`

// VisionReasonSystemPrompt is prepended as a system turn to the reasoning stage.
// It tells the (blind) text model that the "image" it is being asked about is
// actually the perception stage's description.
const VisionReasonSystemPrompt = `You cannot see images. The user provided an ` +
	`image; below is a detailed textual description of it produced by a vision ` +
	`model. Answer the user's request using only that description. If the ` +
	`description lacks a detail needed to answer, say so rather than guessing.`
