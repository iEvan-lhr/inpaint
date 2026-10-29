package reconstruct

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

var hexColor = regexp.MustCompile(`^#[0-9a-fA-F]{6}$`)

func (t *Task) validateRecognition(r Recognition) (map[string]RecognizedRegion, error) {
	if r.AnalysisID != t.analysis.ID {
		return nil, errors.New("reconstruct: analysis_id does not match this task")
	}
	known := make(map[string]bool, len(t.analysis.Regions))
	for _, c := range t.analysis.Regions {
		known[c.ID] = true
	}
	answers := make(map[string]RecognizedRegion, len(r.Regions))
	total := 0
	for _, a := range r.Regions {
		if !known[a.RegionID] {
			return nil, fmt.Errorf("reconstruct: unknown region %q", a.RegionID)
		}
		if _, ok := answers[a.RegionID]; ok {
			return nil, fmt.Errorf("reconstruct: duplicate region %q", a.RegionID)
		}
		if a.Status != Text && a.Status != Graphic && a.Status != Unreadable {
			return nil, fmt.Errorf("reconstruct: invalid status for %s", a.RegionID)
		}
		if a.Color != "" && !hexColor.MatchString(a.Color) {
			return nil, fmt.Errorf("reconstruct: invalid #RRGGBB color for %s", a.RegionID)
		}
		switch a.Role {
		case "", "text", "heading", "paragraph", "list-item", "caption":
		default:
			return nil, fmt.Errorf("reconstruct: invalid role for %s", a.RegionID)
		}
		if a.Text != "" && len(a.Runs) > 0 {
			return nil, fmt.Errorf("reconstruct: %s must use text or runs, not both", a.RegionID)
		}
		if len(a.Runs) > 128 {
			return nil, fmt.Errorf("reconstruct: too many runs for %s", a.RegionID)
		}
		a.Runs = append([]Run(nil), a.Runs...)
		text := recognizedText(a)
		if !utf8.ValidString(text) || len(text) > 65536 {
			return nil, fmt.Errorf("reconstruct: invalid or oversized text for %s", a.RegionID)
		}
		total += len(text)
		if total > 4<<20 {
			return nil, errors.New("reconstruct: recognized text exceeds 4 MiB")
		}
		answers[a.RegionID] = a
	}
	seen := map[string]bool{}
	for _, id := range r.ReadingOrder {
		if !known[id] || seen[id] {
			return nil, fmt.Errorf("reconstruct: unknown or duplicate reading_order id %q", id)
		}
		seen[id] = true
	}
	return answers, nil
}

func recognizedText(a RecognizedRegion) string {
	if len(a.Runs) == 0 {
		return a.Text
	}
	var b strings.Builder
	for i, r := range a.Runs {
		if a.SplitColors && i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(r.Text)
	}
	return b.String()
}

// Reconstruct erases only readable, confirmed text from the original snapshot.
// Omitted, graphic, unreadable, blank, unsupported or conflicting regions remain
// in the background. Invalid IDs/statuses/styles fail before image processing.
// It runs no OCR or detection; the detector may be closed after Analyze returns.
func (t *Task) Reconstruct(ctx context.Context, recognition Recognition, options RenderOptions) (*Result, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	answers, err := t.validateRecognition(recognition)
	if err != nil {
		return nil, err
	}
	skipped := map[string]string{}
	for _, c := range t.analysis.Regions {
		a, ok := answers[c.ID]
		switch {
		case !ok:
			skipped[c.ID] = "missing"
		case a.Status != Text:
			skipped[c.ID] = string(a.Status)
		case strings.TrimSpace(recognizedText(a)) == "":
			skipped[c.ID] = "empty_text"
		case strings.ContainsAny(recognizedText(a), "\r\n"):
			skipped[c.ID] = "multiline_text"
		case !horizontal(c.Polygon):
			skipped[c.ID] = "unsupported_orientation"
		}
	}
	var background *image.RGBA
	var mask *image.Gray
	var lines []Line
	for {
		if err := validContext(ctx); err != nil {
			return nil, err
		}
		// A preserved region has priority. Conservatively retain intersecting
		// candidate boxes too, avoiding duplicate HTML over preserved ink.
		protected := append([]textinpaint.Region(nil), t.protected...)
		for _, c := range t.analysis.Regions {
			if skipped[c.ID] != "" {
				protected = append(protected, c.Region)
			}
		}
		changed := true
		for changed {
			changed = false
			for _, c := range t.analysis.Regions {
				if skipped[c.ID] != "" {
					continue
				}
				for _, p := range protected {
					if polygonBounds(c.Polygon).Overlaps(polygonBounds(p.Polygon)) {
						skipped[c.ID] = "overlaps_preserved_region"
						protected = append(protected, c.Region)
						changed = true
						break
					}
				}
			}
		}
		var selected []textinpaint.Region
		for _, c := range t.analysis.Regions {
			if skipped[c.ID] == "" {
				selected = append(selected, c.Region)
			}
		}
		mask, err = textinpaint.BuildMaskWithProtection(t.source, selected, protected, t.mask)
		if err != nil {
			return nil, err
		}
		if err = validContext(ctx); err != nil {
			return nil, err
		}
		background, err = textinpaint.Inpaint(t.source, mask, t.radius)
		if err != nil {
			return nil, err
		}
		lines = make([]Line, 0, len(selected))
		retry := false
		for _, c := range t.analysis.Regions {
			if skipped[c.ID] != "" {
				continue
			}
			a := answers[c.ID]
			spans := makeSpans(t.source, background, mask, c.Region, a)
			if len(spans) == 0 {
				skipped[c.ID] = "no_reliable_ink"
				retry = true
				continue
			}
			lines = append(lines, Line{RegionID: c.ID, Text: recognizedText(a), Role: a.Role, Region: c.Region, Spans: spans})
		}
		if !retry {
			break
		}
	}
	if err = validContext(ctx); err != nil {
		return nil, err
	}
	order := map[string]int{}
	for i, id := range recognition.ReadingOrder {
		order[id] = i
	}
	sort.SliceStable(lines, func(i, j int) bool {
		a, okA := order[lines[i].RegionID]
		b, okB := order[lines[j].RegionID]
		if okA != okB {
			return okA
		}
		return okA && a < b
	})
	page := Page{AnalysisID: t.analysis.ID, Width: t.analysis.Width, Height: t.analysis.Height, SourceSHA256: t.analysis.SourceSHA256, Lines: lines, Preserved: []Preserved{}, OriginalPNG: t.OriginalPNG()}
	for _, c := range t.analysis.Regions {
		if reason := skipped[c.ID]; reason != "" {
			page.Preserved = append(page.Preserved, Preserved{RegionID: c.ID, Reason: reason})
		}
	}
	page.BackgroundPNG, err = encodePNG(background)
	if err != nil {
		return nil, err
	}
	title := options.Title
	if title == "" {
		title = "图片重建"
	}
	doc := Document{Title: title, Pages: []Page{page}}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return nil, err
	}
	html, err := RenderHTML(doc)
	if err != nil {
		return nil, err
	}
	if err = validContext(ctx); err != nil {
		return nil, err
	}
	return &Result{Page: page, Background: background, Mask: mask, HTML: html, JSON: data}, nil
}

func horizontal(q textinpaint.Quad) bool {
	w := math.Hypot(q[1].X-q[0].X, q[1].Y-q[0].Y)
	h := math.Hypot(q[3].X-q[0].X, q[3].Y-q[0].Y)
	return w > 0 && h <= w*1.8 && math.Abs(q[1].Y-q[0].Y) <= math.Abs(q[1].X-q[0].X)*.2
}

// WriteDir writes fixed output filenames, replacing existing files in dir.
// The HTML is standalone; images and JSON are also saved for inspection.
func (r *Result) WriteDir(dir string) error {
	if r == nil || r.Mask == nil {
		return errors.New("reconstruct: nil result")
	}
	mask, err := encodePNG(r.Mask)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	for _, f := range []struct {
		name string
		data []byte
	}{
		{"index.html", r.HTML}, {"background.png", r.Page.BackgroundPNG}, {"original.png", r.Page.OriginalPNG}, {"mask.png", mask}, {"reconstruction.json", r.JSON},
	} {
		if err = os.WriteFile(filepath.Join(dir, f.name), f.data, 0644); err != nil {
			return err
		}
	}
	return nil
}
