package textinpaint

import (
	"context"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/iEvan-lhr/inpaint/internal/cv"
)

func rectangle(x1, y1, x2, y2 float64) Quad { return Quad{{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}} }
func whiteImage(r image.Rectangle) *image.RGBA {
	im := image.NewRGBA(r)
	draw.Draw(im, r, image.NewUniform(color.White), image.Point{}, draw.Src)
	return im
}

func TestMaskInpaintAndInputPreservation(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 100, 60))
	draw.Draw(im, image.Rect(30, 20, 39, 32), image.NewUniform(color.Black), image.Point{}, draw.Src)
	options := DefaultMaskOptions()
	options.PreserveGraphics = false
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(25, 15, 45, 37)}}, options)
	if err != nil {
		t.Fatal(err)
	}
	if mask.GrayAt(34, 25).Y == 0 || mask.GrayAt(0, 0).Y != 0 || mask.GrayAt(26, 16).Y != 0 {
		t.Fatal("mask must select foreground glyphs only")
	}
	out, err := Inpaint(im, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	if out.RGBAAt(34, 25).R < 240 {
		t.Fatal("uniform background was not reconstructed")
	}
	if im.RGBAAt(34, 25).R != 0 {
		t.Fatal("input was modified")
	}
	for y := 0; y < 60; y++ {
		for x := 0; x < 100; x++ {
			if mask.GrayAt(x, y).Y == 0 && out.RGBAAt(x, y) != im.RGBAAt(x, y) {
				t.Fatalf("unmasked pixel changed at %d,%d", x, y)
			}
		}
	}
}
func TestImageBoundsMaskStrideAndEmptyMask(t *testing.T) {
	src := whiteImage(image.Rect(100, 200, 120, 220))
	src.SetRGBA(105, 206, color.RGBA{R: 25, G: 75, B: 150, A: 255})
	parent := image.NewGray(image.Rect(0, 0, 100, 100))
	mask := parent.SubImage(image.Rect(25, 30, 45, 50)).(*image.Gray)
	out, err := Inpaint(src, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	if out.Bounds() != image.Rect(0, 0, 20, 20) || out.RGBAAt(5, 6) != src.RGBAAt(105, 206) {
		t.Fatal("nonzero origins/strides not normalized correctly")
	}
	if _, err = Inpaint(src, image.NewGray(image.Rect(0, 0, 2, 2)), 3); err == nil {
		t.Fatal("expected size validation")
	}
	full := image.NewGray(image.Rect(0, 0, 20, 20))
	for i := range full.Pix {
		full.Pix[i] = 255
	}
	if _, err = Inpaint(src, full, 3); err == nil {
		t.Fatal("fully masked images must be rejected")
	}
}
func TestColorMarkerProtection(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 100, 60))
	green := color.RGBA{177, 210, 60, 255}
	draw.Draw(im, image.Rect(15, 20, 31, 36), image.NewUniform(green), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(45, 20, 50, 35), image.NewUniform(color.Black), image.Point{}, draw.Src)
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(10, 15, 60, 40)}}, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	if mask.GrayAt(22, 28).Y != 0 || mask.GrayAt(47, 25).Y == 0 {
		t.Fatal("marker should be preserved while foreground text is selected")
	}
}
func TestPPOCRBGRNormalization(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 32, 32))
	draw.Draw(im, im.Rect, image.NewUniform(color.RGBA{R: 255, A: 255}), image.Point{}, draw.Src)
	data, err := cv.Prepare(im, im.Rect, 1, 32, 32)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(data[0]+1)) > 1e-6 || math.Abs(float64(data[1024]+1)) > 1e-6 || math.Abs(float64(data[2048]-1)) > 1e-6 {
		t.Fatal("model expects normalized BGR channels in NCHW order")
	}
}

func TestColoredLetterStrokesAreNotGraphics(t *testing.T) {
	for _, background := range []color.RGBA{{255, 255, 255, 255}, {55, 13, 61, 255}} {
		im := image.NewRGBA(image.Rect(0, 0, 360, 150))
		draw.Draw(im, im.Bounds(), image.NewUniform(background), image.Point{}, draw.Src)
		ink := color.RGBA{245, 175, 0, 255}
		if background.R == 255 {
			ink = color.RGBA{75, 31, 111, 255}
		}
		// A long horizontal stroke plus a stem must be erased together.
		draw.Draw(im, image.Rect(30, 30, 110, 34), image.NewUniform(ink), image.Point{}, draw.Src)
		draw.Draw(im, image.Rect(68, 30, 73, 110), image.NewUniform(ink), image.Point{}, draw.Src)
		// A Y has a triangular convex hull, but is not a solid warning triangle.
		for y := 0; y < 30; y++ {
			for dx := 0; dx < 4; dx++ {
				im.SetRGBA(160+y/2+dx, 40+y, ink)
				im.SetRGBA(190-y/2+dx, 40+y, ink)
			}
		}
		draw.Draw(im, image.Rect(175, 68, 180, 96), image.NewUniform(ink), image.Point{}, draw.Src)
		mask, err := BuildMask(im, []Region{{Polygon: rectangle(25, 25, 115, 115)}, {Polygon: rectangle(155, 35, 200, 100)}}, DefaultMaskOptions())
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < im.Rect.Dy(); y++ {
			for x := 0; x < im.Rect.Dx(); x++ {
				if im.RGBAAt(x, y) == ink && mask.GrayAt(x, y).Y == 0 {
					t.Fatalf("letter stroke protected at (%d,%d), background=%v", x, y, background)
				}
			}
		}
		out, err := Inpaint(im, mask, 3)
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < out.Rect.Dy(); y++ {
			for x := 0; x < out.Rect.Dx(); x++ {
				if out.RGBAAt(x, y) != background {
					t.Fatalf("uniform background has residual color at (%d,%d): %v", x, y, out.RGBAAt(x, y))
				}
			}
		}
	}
}

func TestLeadingIconAndIndependentText(t *testing.T) {
	icon := color.RGBA{128, 71, 175, 255}
	bg := color.RGBA{244, 236, 249, 255}
	im := image.NewRGBA(image.Rect(0, 0, 180, 70))
	draw.Draw(im, im.Bounds(), image.NewUniform(bg), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(20, 23, 44, 47), image.NewUniform(icon), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(23, 26, 41, 44), image.NewUniform(bg), image.Point{}, draw.Src)
	for _, x := range []int{64, 85, 106, 127} {
		draw.Draw(im, image.Rect(x, 24, x+10, 45), image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	// The model box starts inside the icon. Looking only inside the box would
	// erase its right edge. A separate partial detection must not defeat this.
	regions := []Region{{Polygon: rectangle(40, 20, 141, 50)}, {Polygon: rectangle(25, 27, 41, 39)}}
	mask, err := BuildMask(im, regions, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	out, err := Inpaint(im, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	for y := 23; y < 47; y++ {
		for x := 20; x < 44; x++ {
			if out.RGBAAt(x, y) != im.RGBAAt(x, y) {
				t.Fatalf("icon damaged at (%d,%d)", x, y)
			}
		}
	}
	if out.RGBAAt(68, 30) != bg {
		t.Fatal("text beside icon was not removed")
	}
	// A full-height independent text box provides evidence this is a colored
	// token (such as a numbered bullet), not an icon to preserve.
	regions[1].Polygon = rectangle(17, 20, 47, 50)
	mask, err = BuildMask(im, regions, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	if mask.GrayAt(42, 34).Y == 0 {
		t.Fatal("independently detected colored text was treated as an icon")
	}
}

func TestInpaintKeepsGradientAndUnmaskedPixels(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 180, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 180; x++ {
			v := uint8(x + 40)
			im.SetRGBA(x, y, color.RGBA{v, v, v, 255})
		}
	}
	mask := image.NewGray(im.Bounds())
	draw.Draw(mask, image.Rect(40, 46, 140, 52), image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(40, 46, 140, 52), image.NewUniform(color.Black), image.Point{}, draw.Src)
	out, err := Inpaint(im, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	if int(out.RGBAAt(125, 49).R)-int(out.RGBAAt(55, 49).R) < 50 {
		t.Fatal("varying background was replaced with a uniform color")
	}
	for y := 0; y < 100; y++ {
		for x := 0; x < 180; x++ {
			if mask.GrayAt(x, y).Y == 0 && out.RGBAAt(x, y) != im.RGBAAt(x, y) {
				t.Fatalf("unmasked pixel changed at (%d,%d)", x, y)
			}
		}
	}
}

func TestShortGrayGlyphIsNotAnIcon(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 210, 80))
	draw.Draw(im, image.Rect(22, 29, 38, 46), image.NewUniform(color.RGBA{145, 145, 147, 255}), image.Point{}, draw.Src)
	for _, x := range []int{46, 68, 90, 112, 134} {
		draw.Draw(im, image.Rect(x, 25, x+10, 47), image.NewUniform(color.RGBA{84, 83, 86, 255}), image.Point{}, draw.Src)
	}
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(18, 22, 150, 49)}}, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	if mask.GrayAt(28, 37).Y == 0 {
		t.Fatal("short antialiased gray character was protected as an icon")
	}
}

func TestDocumentRuleDoesNotBleedIntoRemovedLabel(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 300, 100))
	for _, y := range []int{59, 61} {
		draw.Draw(im, image.Rect(10, y, 280, y+1), image.NewUniform(color.RGBA{110, 110, 110, 255}), image.Point{}, draw.Src)
	}
	draw.Draw(im, image.Rect(10, 60, 280, 61), image.NewUniform(color.Black), image.Point{}, draw.Src)
	for _, x := range []int{30, 52, 74, 96, 118, 140} {
		draw.Draw(im, image.Rect(x, 32, x+10, 53), image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(24, 27, 156, 64)}}, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	out, err := Inpaint(im, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	for y := 25; y <= 63; y++ {
		for x := 20; x < 165; x++ {
			want := color.RGBA{255, 255, 255, 255}
			if y >= 59 && y <= 61 {
				want = im.RGBAAt(x, y)
			}
			if out.RGBAAt(x, y) != want {
				t.Fatalf("label residue or rule damage at (%d,%d): %v", x, y, out.RGBAAt(x, y))
			}
		}
	}
}
func TestQuadDeduplicationAndValidation(t *testing.T) {
	a, b := rectangle(0, 0, 10, 10), rectangle(5, 0, 15, 10)
	if got := intersectionArea(a, b); math.Abs(got-50) > 1e-6 {
		t.Fatalf("intersection=%v", got)
	}
	regions := deduplicate([]Region{{Polygon: rectangle(1, 1, 9, 9)}, {Polygon: a}, {Polygon: rectangle(20, 0, 30, 10)}})
	if len(regions) != 2 {
		t.Fatalf("kept %d regions", len(regions))
	}
	im := whiteImage(image.Rect(0, 0, 100, 60))
	for _, q := range []Quad{rectangle(-1, 0, 20, 10), rectangle(10, 10, 10, 20), {{0, 0}, {10, 10}, {10, 0}, {0, 10}}, {{math.NaN(), 0}, {10, 0}, {10, 10}, {0, 10}}} {
		if _, err := BuildMask(im, []Region{{Polygon: q}}, DefaultMaskOptions()); err == nil {
			t.Fatal("invalid quad accepted")
		}
	}
	if _, err := BuildMask(nil, nil, DefaultMaskOptions()); err == nil {
		t.Fatal("nil input accepted")
	}
}

func TestSmallTextFilterUsesGlyphsNotExpandedBox(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 240, 110))
	for _, x := range []int{30, 42, 54, 66} {
		draw.Draw(im, image.Rect(x, 29, x+5, 36), image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	for _, x := range []int{130, 150, 170, 190} {
		draw.Draw(im, image.Rect(x, 24, x+9, 44), image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	// Both detector boxes are 32 px tall; actual glyphs are 7 and 20 px.
	regions := []Region{{Polygon: rectangle(22, 18, 80, 50)}, {Polygon: rectangle(122, 18, 208, 50)}}
	kept, err := filterSmallText(im, regions, 12)
	if err != nil {
		t.Fatal(err)
	}
	if len(kept) != 1 || kept[0].Polygon != regions[1].Polygon || kept[0].TextHeight != 20 {
		t.Fatalf("wrong retained text: %+v", kept)
	}
	all, err := filterSmallText(im, regions, 0)
	if err != nil || len(all) != 2 || all[0].TextHeight != 7 {
		t.Fatalf("zero should disable filtering: %+v, %v", all, err)
	}
	mask, err := BuildMask(im, kept, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	out, err := Inpaint(im, mask, 3)
	if err != nil {
		t.Fatal(err)
	}
	if out.RGBAAt(32, 32) != im.RGBAAt(32, 32) || out.RGBAAt(133, 32) != (color.RGBA{255, 255, 255, 255}) {
		t.Fatal("small text must stay while readable text is removed")
	}
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), 257} {
		if validProcessingOptions(ProcessingOptions{MinTextHeight: value}) == nil {
			t.Fatalf("accepted threshold %v", value)
		}
	}
}

func TestVerticalTextFilterUsesCharacterWidth(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 100, 180))
	for _, y := range []int{25, 42, 59, 76} {
		draw.Draw(im, image.Rect(30, y, 37, y+8), image.NewUniform(color.Black), image.Point{}, draw.Src)
	}
	regions := []Region{{Polygon: rectangle(20, 18, 48, 96)}}
	kept, err := filterSmallText(im, regions, 12)
	if err != nil || len(kept) != 0 {
		t.Fatalf("vertical line length must not turn tiny characters into large text: %+v, %v", kept, err)
	}
}

func TestTinyTextInsetPreservesButtonArtwork(t *testing.T) {
	im := whiteImage(image.Rect(0, 0, 600, 300))
	draw.Draw(im, image.Rect(20, 30, 230, 210), image.NewUniform(color.Black), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(27, 37, 223, 203), image.NewUniform(color.RGBA{245, 250, 210, 255}), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(120, 105, 160, 145), image.NewUniform(color.RGBA{100, 180, 20, 255}), image.Point{}, draw.Src)
	regions := []Region{
		{Polygon: rectangle(40, 55, 90, 70), TextHeight: 7},
		{Polygon: rectangle(40, 80, 90, 95), TextHeight: 8},
		{Polygon: rectangle(118, 102, 163, 148), TextHeight: 30}, // false text detection on a button icon
		{Polygon: rectangle(300, 60, 500, 90), TextHeight: 20},
	}
	kept, ignored, err := selectDetectedText(im, regions, 12, true)
	if err != nil || len(kept) != 1 || kept[0].Polygon != regions[3].Polygon {
		t.Fatalf("framed UI must stay intact: %+v, %v", kept, err)
	}
	mask := image.NewGray(im.Rect)
	draw.Draw(mask, image.Rect(15, 25, 235, 215), image.NewUniform(color.White), image.Point{}, draw.Src)
	protectRegions(mask, ignored)
	if mask.GrayAt(140, 125).Y != 0 || mask.GrayAt(22, 80).Y != 0 || mask.GrayAt(16, 26).Y == 0 {
		t.Fatal("panel protection must include artwork and frame, within its bounds")
	}
	for _, tc := range []struct {
		minimum float64
		protect bool
		want    int
	}{{0, true, 4}, {12, false, 2}} {
		selected, _, err := selectDetectedText(im, regions, tc.minimum, tc.protect)
		if err != nil || len(selected) != tc.want {
			t.Fatalf("option override: %+v, %v", selected, err)
		}
	}
	regions[0].TextHeight = 20
	regions[1].TextHeight = 20
	selected, _, err := selectDetectedText(im, regions, 12, true)
	if err != nil || len(selected) != 4 {
		t.Fatal("a framed region dominated by readable text must not be treated as a tiny interface")
	}
}

func TestColoredBackgroundHolesAreNotMarkers(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 160, 100))
	bg := color.RGBA{177, 210, 60, 255}
	draw.Draw(im, im.Bounds(), image.NewUniform(bg), image.Point{}, draw.Src)
	// A triangular background hole enclosed by the strokes of a numeral.
	for y := 0; y < 24; y++ {
		for x := 0; x < 4; x++ {
			im.SetRGBA(54-y/2+x, 25+y, color.RGBA{30, 30, 30, 255})
		}
	}
	draw.Draw(im, image.Rect(52, 25, 56, 60), image.NewUniform(color.Black), image.Point{}, draw.Src)
	draw.Draw(im, image.Rect(42, 46, 63, 50), image.NewUniform(color.Black), image.Point{}, draw.Src)
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(36, 18, 70, 66)}}, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	if mask.GrayAt(54, 40).Y == 0 || mask.GrayAt(46, 48).Y == 0 {
		t.Fatal("a colored background hole protected numeral strokes")
	}
}

func TestGradientDoesNotReverseTextPolarity(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 220, 80))
	for y := 0; y < 80; y++ {
		for x := 0; x < 220; x++ {
			im.SetRGBA(x, y, color.RGBA{uint8(210 + x/8), uint8(220 + y/8), 250, 255})
		}
	}
	for _, x := range []int{30, 50, 70, 90, 110, 130, 150} {
		draw.Draw(im, image.Rect(x, 25, x+9, 46), image.NewUniform(color.RGBA{10, 20, 60, 255}), image.Point{}, draw.Src)
	}
	mask, err := BuildMask(im, []Region{{Polygon: rectangle(24, 19, 166, 52)}}, DefaultMaskOptions())
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []int{34, 54, 74, 94, 114, 134, 154} {
		if mask.GrayAt(x, 35).Y == 0 {
			t.Fatalf("dark glyph missed on gradient at %d", x)
		}
	}
}
func TestTileCoverage(t *testing.T) {
	for _, length := range []int{64, 1280, 1281, 2529, 10000} {
		starts := tileStarts(length, 1280)
		if starts[0] != 0 {
			t.Fatal("missing start")
		}
		if starts[len(starts)-1]+1280 < length {
			t.Fatal("missing end")
		}
		for i := 1; i < len(starts); i++ {
			if starts[i]-starts[i-1] >= 1280 {
				t.Fatal("tiles must overlap")
			}
		}
	}
}
func TestRuntimeLifecycleAndBlankImage(t *testing.T) {
	model := os.Getenv("TEXTINPAINT_MODEL")
	if model == "" {
		model = filepath.Join("models", "ch_PP-OCRv4_det_infer.onnx")
	}
	lib := os.Getenv("TEXTINPAINT_ORT")
	if lib == "" && runtime.GOOS == "windows" {
		lib = filepath.Join("native", "windows-amd64", "bin", "onnxruntime.dll")
	}
	if _, err := os.Stat(model); err != nil {
		t.Skip("set TEXTINPAINT_MODEL to run model integration tests")
	}
	if lib != "" {
		if _, err := os.Stat(lib); err != nil {
			t.Skip("set TEXTINPAINT_ORT to run model integration tests")
		}
	}
	cfg := Config{ModelPath: model, RuntimePath: lib}
	first, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	blank := whiteImage(image.Rect(0, 0, 128, 96))
	if _, err = first.Detect(context.Background(), blank); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed engine: %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = second.Detect(canceled, blank); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	result, err := second.Remove(context.Background(), blank)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Regions) != 0 {
		t.Fatalf("blank page has %d detections", len(result.Regions))
	}
	if result.Image.RGBAAt(64, 48) != (color.RGBA{255, 255, 255, 255}) {
		t.Fatal("empty-mask output changed")
	}
	if err = second.Close(); err != nil {
		t.Fatal(err)
	}
	if err = second.Close(); err != nil {
		t.Fatal("Close is not idempotent")
	}
	third, err := New(cfg)
	if err != nil {
		t.Fatalf("runtime did not reinitialize: %v", err)
	}
	third.Close()
}
