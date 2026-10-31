package apps

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

const (
	// SourceBudget caps the characters of source text sent per question, so the
	// prompt fits the 4096-token context with room for the answer.
	SourceBudget = 6000
	// MaxSources caps how many sources one question carries, always-documents
	// included.
	MaxSources = 8
	// maxDocChars trims long documents so one can't take the whole budget.
	maxDocChars   = 2000
	historyWeight = 0.5
)

// Doc is one indexed, citable source.
type Doc struct {
	Title  string
	Link   string
	Text   string
	always bool
	terms  map[string]int
	len    int
}

// Index is a BM25 index over one knowledge file.
type Index struct {
	docs   []*Doc
	df     map[string]int
	avgLen float64
}

func NewIndex(k *Knowledge) *Index {
	idx := &Index{df: map[string]int{}}
	total := 0
	for _, d := range k.Documents {
		doc := &Doc{Title: d.Title, Link: d.Link, Text: trimText(d.Text), always: d.Always, terms: map[string]int{}}
		// The title counts twice when scoring.
		for _, t := range tokenize(doc.Title+" "+doc.Title+" "+doc.Text, nil) {
			doc.terms[t]++
			doc.len++
		}
		for t := range doc.terms {
			idx.df[t]++
		}
		total += doc.len
		idx.docs = append(idx.docs, doc)
	}
	idx.avgLen = float64(total) / float64(len(idx.docs))
	return idx
}

func trimText(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxDocChars {
		return s
	}
	s = s[:maxDocChars]
	if cut := strings.LastIndexAny(s, ".\n"); cut > maxDocChars/2 {
		s = s[:cut+1]
	}
	return s
}

var stopwords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`a about an and any are as at be been but by can could did do does
		for from had has have he her him his how i if in into is it its me my of on or our she so tell
		than that the their them then there these they this to was we were what when where which who
		why will with would you your`) {
		stopwords[w] = true
	}
}

// tokenize lower-cases, splits on anything but letters, digits, + and #, drops
// stopwords and ignore, and folds a trailing plural "s" on longer words.
func tokenize(s string, ignore map[string]bool) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '+' || r == '#')
	})
	out := fields[:0]
	for _, f := range fields {
		if stopwords[f] || ignore[f] {
			continue
		}
		if len(f) > 4 && strings.HasSuffix(f, "s") && !strings.HasSuffix(f, "ss") {
			f = f[:len(f)-1]
		}
		out = append(out, f)
	}
	return out
}

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

func (idx *Index) score(d *Doc, query map[string]float64) float64 {
	n := float64(len(idx.docs))
	var s float64
	for t, w := range query {
		tf := float64(d.terms[t])
		if tf == 0 {
			continue
		}
		df := float64(idx.df[t])
		idf := math.Log(1 + (n-df+0.5)/(df+0.5))
		s += w * idf * tf * (bm25K1 + 1) / (tf + bm25K1*(1-bm25B+bm25B*float64(d.len)/idx.avgLen))
	}
	return s
}

// Retrieve returns the sources for one question: every always-document
// first, then the best-scoring others while they fit SourceBudget. Earlier
// questions count at half weight so follow-ups ("what stack did it use?")
// still find their subject. ignoreTerms are dropped from the query only;
// a term no query contains never affects a score.
func (idx *Index) Retrieve(question string, history, ignoreTerms []string) []Doc {
	ignore := map[string]bool{}
	for _, t := range ignoreTerms {
		ignore[strings.ToLower(t)] = true
	}
	query := map[string]float64{}
	for _, t := range tokenize(question, ignore) {
		query[t] += 1
	}
	for _, h := range history {
		for _, t := range tokenize(h, ignore) {
			query[t] += historyWeight
		}
	}

	type scored struct {
		doc   *Doc
		score float64
	}
	var out []Doc
	used := 0
	var ranked []scored
	for _, d := range idx.docs {
		if d.always {
			out = append(out, *d)
			used += len(d.Text)
		} else if s := idx.score(d, query); s > 0 {
			ranked = append(ranked, scored{d, s})
		}
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	for _, r := range ranked {
		if len(out) >= MaxSources {
			break
		}
		if used+len(r.doc.Text) > SourceBudget {
			continue
		}
		out = append(out, *r.doc)
		used += len(r.doc.Text)
	}
	return out
}
