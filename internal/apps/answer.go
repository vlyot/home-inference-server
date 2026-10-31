package apps

import (
	"regexp"
	"strconv"
	"strings"
)

// Citation maps an [n] in the answer to the document it cites.
type Citation struct {
	N     int    `json:"n"`
	Title string `json:"title"`
	Link  string `json:"link,omitempty"`
}

// Answer is the body of a successful POST /apps/ask.
type Answer struct {
	Text      string     `json:"answer"`
	Citations []Citation `json:"citations"`
	Refused   bool       `json:"refused"`
}

var (
	thinkRe = regexp.MustCompile(`(?s)<think>.*?</think>`)
	// A plain completion can run on into a made-up next turn; cut it there.
	runOnRe    = regexp.MustCompile(`(?i)\n\s*(?:<question>|question:|earlier questions|answer:)`)
	tagRe      = regexp.MustCompile(`<[^>]*>`)
	mdLinkRe   = regexp.MustCompile(`\[([^\]]+)\]\([^)]*\)`)
	urlRe      = regexp.MustCompile(`(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+/\S*`)
	citeRe     = regexp.MustCompile(`(?i)\[(?:sources?\s*)?(\d+(?:\s*,\s*(?:sources?\s*)?\d+)*)\]`)
	headingRe  = regexp.MustCompile(`(?m)^\s*#+\s*`)
	bulletRe   = regexp.MustCompile(`(?m)^\s*[*•]\s+`)
	emphasisRe = regexp.MustCompile("\\*\\*|__|`")
	spacesRe   = regexp.MustCompile(`[ \t]+`)
	blankRe    = regexp.MustCompile(`\n{3,}`)
	spaceDotRe = regexp.MustCompile(`\s+([.,;:!?])`)
	digitsRe   = regexp.MustCompile(`\d+`)
)

// Clean turns raw model output into a safe answer. It strips markup and URLs
// and keeps only citations that point at a supplied source. When the spec
// requires citations, an answer left without one (the model's own refusal
// included) becomes spec.Refusal; an answer left empty always does.
func Clean(output string, sources []Doc, spec Spec) Answer {
	s := thinkRe.ReplaceAllString(output, "")
	if loc := runOnRe.FindStringIndex(s); loc != nil {
		s = s[:loc[0]]
	}
	s = mdLinkRe.ReplaceAllString(s, "$1")
	s = tagRe.ReplaceAllString(s, "")
	s = urlRe.ReplaceAllString(s, "")
	s = headingRe.ReplaceAllString(s, "")
	s = bulletRe.ReplaceAllString(s, "- ")
	s = emphasisRe.ReplaceAllString(s, "")

	seen := map[int]bool{}
	cites := []Citation{}
	s = citeRe.ReplaceAllStringFunc(s, func(m string) string {
		var kept strings.Builder
		for _, n := range digitsRe.FindAllString(m, -1) {
			i, err := strconv.Atoi(n)
			if err != nil || i < 1 || i > len(sources) {
				continue
			}
			kept.WriteString("[" + n + "]")
			if !seen[i] {
				seen[i] = true
				cites = append(cites, Citation{N: i, Title: sources[i-1].Title, Link: sources[i-1].Link})
			}
		}
		return kept.String()
	})

	s = spacesRe.ReplaceAllString(s, " ")
	s = spaceDotRe.ReplaceAllString(s, "$1")
	s = blankRe.ReplaceAllString(s, "\n\n")
	s = strings.TrimSpace(s)

	if s == "" || (spec.RequireCitations && len(cites) == 0) {
		return Answer{Text: refusal(spec), Citations: []Citation{}, Refused: true}
	}
	return Answer{Text: s, Citations: cites, Refused: spec.Refusal != "" && s == spec.Refusal}
}

func refusal(spec Spec) string {
	if spec.Refusal != "" {
		return spec.Refusal
	}
	return DefaultRefusal
}
