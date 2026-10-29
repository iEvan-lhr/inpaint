package reconstruct

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"strings"
	"sync"
	"testing"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

type fixtureDetector struct {
	calls              int
	regions, protected []textinpaint.Region
}

func (f *fixtureDetector) DetectWithProtection(_ context.Context, im image.Image, _ *textinpaint.ProcessingOptions) (*textinpaint.DetectionResult, error) {
	f.calls++
	return &textinpaint.DetectionResult{Regions: f.regions, ProtectedRegions: f.protected, Width: im.Bounds().Dx(), Height: im.Bounds().Dy()}, nil
}
func quad(x0, y0, x1, y1 float64) textinpaint.Quad {
	return textinpaint.Quad{{X: x0, Y: y0}, {X: x1, Y: y0}, {X: x1, Y: y1}, {X: x0, Y: y1}}
}
func fixture(t *testing.T) (*Task, *fixtureDetector, *image.RGBA) {
	t.Helper()
	im := image.NewRGBA(image.Rect(100, 200, 260, 270))
	draw.Draw(im, im.Rect, image.NewUniform(color.White), image.Point{}, draw.Src)
	for _, r := range []image.Rectangle{image.Rect(124, 218, 128, 235), image.Rect(136, 218, 140, 235), image.Rect(124, 225, 140, 229), image.Rect(190, 218, 210, 238)} {
		draw.Draw(im, r, image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	f := &fixtureDetector{regions: []textinpaint.Region{{Polygon: quad(20, 12, 48, 41), Score: .9, TextHeight: 17}, {Polygon: quad(85, 12, 115, 42), Score: .8, TextHeight: 20}}}
	m := textinpaint.DefaultMaskOptions()
	m.PreserveGraphics = false
	task, err := Analyze(context.Background(), f, im, Options{Mask: &m})
	if err != nil {
		t.Fatal(err)
	}
	return task, f, im
}
func response(task *Task) Recognition {
	return Recognition{AnalysisID: task.Analysis().ID, Regions: []RecognizedRegion{{RegionID: "r000001", Status: Text, Text: "HI"}, {RegionID: "r000002", Status: Graphic}}}
}

func TestConfirmedTextOnlyAndSnapshotIsolation(t *testing.T) {
	task, detector, src := fixture(t)
	before := task.OriginalPNG()
	// Callers may reuse source buffers and edit the returned JSON metadata.
	draw.Draw(src, src.Rect, image.NewUniform(color.RGBA{R: 255, A: 255}), image.Point{}, draw.Src)
	a := task.Analysis()
	a.Regions[0].Polygon = quad(0, 0, 3, 3)
	a.Regions[0].ID = "changed"
	original := task.OriginalPNG()
	original[0] = 0
	r, err := task.Reconstruct(context.Background(), response(task), RenderOptions{Title: "Test"})
	if err != nil {
		t.Fatal(err)
	}
	if detector.calls != 1 {
		t.Fatal("reconstruction detected again")
	}
	if !bytes.Equal(before, task.OriginalPNG()) {
		t.Fatal("task snapshot was mutated")
	}
	if r.Background.RGBAAt(25, 22).R < 240 {
		t.Fatal("confirmed text was not removed")
	}
	if r.Background.RGBAAt(95, 22).R != 0 || r.Mask.GrayAt(95, 22).Y != 0 {
		t.Fatal("graphic was erased")
	}
	if len(r.Page.Lines) != 1 || r.Page.Lines[0].RegionID != "r000001" {
		t.Fatal("unexpected HTML text")
	}
	for y := 0; y < 70; y++ {
		for x := 0; x < 160; x++ {
			if r.Mask.GrayAt(x, y).Y == 0 && r.Background.RGBAAt(x, y) != task.source.RGBAAt(x, y) {
				t.Fatal("unmasked pixel changed")
			}
		}
	}
}

func TestMissingUnreadableAndEmptyTextPreserveOriginal(t *testing.T) {
	task, _, _ := fixture(t)
	for _, regions := range [][]RecognizedRegion{nil, {{RegionID: "r000001", Status: Unreadable}}, {{RegionID: "r000001", Status: Text, Text: "  "}}} {
		r, err := task.Reconstruct(context.Background(), Recognition{AnalysisID: task.Analysis().ID, Regions: regions}, RenderOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Page.Lines) != 0 || len(r.Page.Preserved) != 2 || !bytes.Equal(r.Background.Pix, task.source.Pix) {
			t.Fatal("unrecognized regions must remain intact")
		}
	}
}

func TestRecognitionValidation(t *testing.T) {
	task, _, _ := fixture(t)
	cases := []Recognition{
		{AnalysisID: "other", Regions: response(task).Regions},
		{AnalysisID: task.analysis.ID, Regions: []RecognizedRegion{{RegionID: "r999999", Status: Text, Text: "x"}}},
		{AnalysisID: task.analysis.ID, Regions: []RecognizedRegion{{RegionID: "r000001", Status: Text, Text: "a"}, {RegionID: "r000001", Status: Graphic}}},
		{AnalysisID: task.analysis.ID, Regions: []RecognizedRegion{{RegionID: "r000001", Status: "guess", Text: "a"}}},
		{AnalysisID: task.analysis.ID, Regions: []RecognizedRegion{{RegionID: "r000001", Status: Text, Text: "a", Color: "url(https://example.org)"}}},
		{AnalysisID: task.analysis.ID, Regions: []RecognizedRegion{{RegionID: "r000001", Status: Text, Text: "a", Runs: []Run{{Text: "b"}}}}},
		{AnalysisID: task.analysis.ID, ReadingOrder: []string{"r000001", "r000001"}},
	}
	for i, r := range cases {
		if _, err := task.Reconstruct(context.Background(), r, RenderOptions{}); err == nil {
			t.Fatalf("case %d accepted invalid response", i)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := task.Reconstruct(ctx, response(task), RenderOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatal("cancellation ignored")
	}
}

func TestProtectedOverlapAndDilatedMask(t *testing.T) {
	task, _, _ := fixture(t)
	task.protected = []textinpaint.Region{{Polygon: quad(38, 15, 52, 36)}}
	r, err := task.Reconstruct(context.Background(), response(task), RenderOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Page.Lines) != 0 || !bytes.Equal(r.Background.Pix, task.source.Pix) {
		t.Fatal("preserved overlap lost priority")
	}
	if r.Page.Preserved[0].Reason != "overlaps_preserved_region" {
		t.Fatal("missing preservation reason")
	}
	// The low-level API must also reserve pixels after mask dilation.
	m := textinpaint.DefaultMaskOptions()
	m.PreserveGraphics = false
	m.Dilation = 4
	mask, err := textinpaint.BuildMaskWithProtection(task.source, []textinpaint.Region{{Polygon: quad(20, 12, 48, 41)}}, []textinpaint.Region{{Polygon: quad(36, 15, 44, 40)}}, m)
	if err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 40; y++ {
		for x := 37; x < 44; x++ {
			if mask.GrayAt(x, y).Y != 0 {
				t.Fatal("dilation erased protected region")
			}
		}
	}
}

func TestCropsAndConcurrentReuse(t *testing.T) {
	task, _, _ := fixture(t)
	data, err := task.CropPNG("r000001", 4)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width != 37 || cfg.Height != 38 {
		t.Fatalf("unexpected crop: %+v %v", cfg, err)
	}
	if _, err = task.CropPNG("missing", 0); err == nil {
		t.Fatal("unknown crop id accepted")
	}
	var wg sync.WaitGroup
	errors := make(chan error, 3)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := task.Reconstruct(context.Background(), response(task), RenderOptions{})
			errors <- err
		}()
	}
	wg.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestHTMLTextEscapingAndMultiPage(t *testing.T) {
	task, _, _ := fixture(t)
	attack := `</script><script>alert("x")</script><img src=x onerror=alert(1)>`
	rq := response(task)
	rq.Regions[0].Text = attack
	r, err := task.Reconstruct(context.Background(), rq, RenderOptions{Title: attack})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(r.HTML, []byte(attack)) || strings.Count(string(r.HTML), "</script>") != 2 {
		t.Fatal("OCR text escaped the JSON script")
	}
	var doc Document
	if err = json.Unmarshal(r.JSON, &doc); err != nil || doc.Pages[0].Lines[0].Text != attack {
		t.Fatal("escaped text did not round-trip")
	}
	html, err := RenderHTML(Document{Title: "Two pages", Pages: []Page{r.Page, r.Page}})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(html, []byte(`"number":2`)) {
		t.Fatal("second page missing")
	}
	if err = r.WriteDir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
}

func TestUnreadableInkIsNotErased(t *testing.T) {
	task, _, _ := fixture(t)
	// A detected blank box can be hallucinated by an external model. No ink
	// means no overlay and no removal rather than an invented visible label.
	draw.Draw(task.source, task.source.Rect, image.NewUniform(color.White), image.Point{}, draw.Src)
	r, err := task.Reconstruct(context.Background(), response(task), RenderOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Page.Lines) != 0 || r.Page.Preserved[0].Reason != "no_reliable_ink" {
		t.Fatal("blank region gained hallucinated text")
	}
}
