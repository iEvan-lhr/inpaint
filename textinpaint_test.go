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
