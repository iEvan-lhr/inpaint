package reconstruct

import (
	"bytes"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"image/png"
	"strings"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

//go:embed page.html
var pageTemplate string

var renderer = template.Must(template.New("document").Parse(pageTemplate))

// RenderHTML combines one or more reconstructed pages into standalone HTML.
// No external assets, fonts or scripts are fetched. Text is inserted using
// textContent, never as raw HTML; JSON is encoded with Go's HTML-safe escaping.
func RenderHTML(doc Document) ([]byte, error) {
	if len(doc.Pages) == 0 || len(doc.Pages) > 256 {
		return nil, errors.New("reconstruct: document needs 1..256 pages")
	}
	if len(doc.Title) > 4096 {
		return nil, errors.New("reconstruct: title is too long")
	}
	if doc.Title == "" {
		doc.Title = "图片重建"
	}
	type viewPage struct {
		Page
		Number     int    `json:"number"`
		Original   string `json:"original"`
		Background string `json:"background"`
	}
	view := struct {
		Title string     `json:"title"`
		Pages []viewPage `json:"pages"`
	}{Title: doc.Title}
	for i, p := range doc.Pages {
		if p.Lines == nil {
			p.Lines = []Line{}
		}
		if p.Preserved == nil {
			p.Preserved = []Preserved{}
		}
		if p.Width <= 0 || p.Height <= 0 || p.Width > textinpaint.MaxImagePixels/p.Height {
			return nil, fmt.Errorf("reconstruct: page %d has invalid dimensions", i+1)
		}
		for _, data := range [][]byte{p.OriginalPNG, p.BackgroundPNG} {
			cfg, err := png.DecodeConfig(bytes.NewReader(data))
			if err != nil || cfg.Width != p.Width || cfg.Height != p.Height {
				return nil, fmt.Errorf("reconstruct: page %d has invalid PNG dimensions", i+1)
			}
		}
		for _, line := range p.Lines {
			for _, s := range line.Spans {
				b := s.Bounds
				if b.X < 0 || b.Y < 0 || b.Width <= 0 || b.Height <= 0 || b.X > p.Width-b.Width || b.Y > p.Height-b.Height || !hexColor.MatchString(s.Color) {
					return nil, fmt.Errorf("reconstruct: page %d has invalid span bounds/color", i+1)
				}
				var text strings.Builder
				for _, r := range s.Runs {
					text.WriteString(r.Text)
				}
				if strings.TrimSpace(text.String()) == "" || strings.ContainsAny(text.String(), "\r\n") {
					return nil, fmt.Errorf("reconstruct: page %d has an empty or multiline span", i+1)
				}
			}
		}
		view.Pages = append(view.Pages, viewPage{Page: p, Number: i + 1, Original: "data:image/png;base64," + base64.StdEncoding.EncodeToString(p.OriginalPNG), Background: "data:image/png;base64," + base64.StdEncoding.EncodeToString(p.BackgroundPNG)})
	}
	data, err := json.Marshal(view)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	// template.JS is safe only because data was produced above by encoding/json
	// with its default EscapeHTML behavior (including escaping </script>).
	err = renderer.Execute(&buf, struct {
		Title string
		Data  template.JS
	}{doc.Title, template.JS(data)})
	return buf.Bytes(), err
}
