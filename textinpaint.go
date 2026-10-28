// Package textinpaint detects text with PP-OCRv4 and removes it with OpenCV.
// All processing is local. Image coordinates are relative to the top-left
// corner of the input, even for images whose Bounds().Min is nonzero.
package textinpaint

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"reflect"

	"github.com/iEvan-lhr/inpaint/internal/cv"
)

const MaxImagePixels = 64_000_000

var ErrClosed = errors.New("textinpaint: engine is closed")

type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}
type Quad [4]Point
type Region struct {
	Polygon Quad    `json:"polygon"`
	Score   float32 `json:"score"`
}

// DetectionOptions configure the PP-OCRv4 DB text detector. TileSize=0 disables
// regular detail tiles; DetectMargins independently enables enlarged top/bottom
// margin passes. Scores are model scores, not a guarantee of text.
type DetectionOptions struct {
	LongSide       int
	TileSize       int
	TileScale      float64
	PixelThreshold float32
	BoxThreshold   float32
	UnclipRatio    float32
	MaxCandidates  int
	Threads        int
	DetectMargins  bool
}

func DefaultDetectionOptions() DetectionOptions {
	return DetectionOptions{LongSide: 1920, TileSize: 1280, TileScale: 1.5, PixelThreshold: .3, BoxThreshold: .5, UnclipRatio: 1.6, MaxCandidates: 1000, Threads: 4, DetectMargins: true}
}

// MaskOptions control glyph extraction. PreserveGraphics uses contextual
// heuristics for rules, colored markers and leading icons; turn it off for
// decorative lettering that resembles graphics.
type MaskOptions struct {
	Padding           float64
	ContrastThreshold int
	Dilation          int
	PreserveGraphics  bool
}

func DefaultMaskOptions() MaskOptions {
	return MaskOptions{Padding: 2, ContrastThreshold: 13, Dilation: 2, PreserveGraphics: true}
}

// Config selects local model/runtime files. Nil option pointers select defaults.
// RuntimePath can be empty to use an already initialized ONNX Runtime or the
// platform library search path. InpaintRadius=0 selects the default radius 3.
type Config struct {
	ModelPath     string
	RuntimePath   string
	Detection     *DetectionOptions
	Mask          *MaskOptions
	InpaintRadius float64
}
type Timings struct {
	DetectionMS int64 `json:"detection_ms"`
	MaskMS      int64 `json:"mask_ms"`
	InpaintMS   int64 `json:"inpaint_ms"`
}
type Result struct {
	Image   *image.RGBA `json:"-"`
	Mask    *image.Gray `json:"-"`
	Regions []Region    `json:"regions"`
	Width   int         `json:"width"`
	Height  int         `json:"height"`
	Timings Timings     `json:"timings"`
}

func validImage(src image.Image) error {
	if src == nil || (reflect.ValueOf(src).Kind() == reflect.Ptr && reflect.ValueOf(src).IsNil()) {
		return errors.New("textinpaint: nil image")
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 || w > MaxImagePixels || h > MaxImagePixels || w > MaxImagePixels/h {
		return fmt.Errorf("textinpaint: image must contain 1..%d pixels", MaxImagePixels)
	}
	return nil
}
func normalized(src image.Image) (*image.RGBA, error) {
	if err := validImage(src); err != nil {
		return nil, err
	}
	out := image.NewRGBA(image.Rect(0, 0, src.Bounds().Dx(), src.Bounds().Dy()))
	draw.Draw(out, out.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(out, out.Bounds(), src, src.Bounds().Min, draw.Over)
	return out, nil
}
func validMaskOptions(o MaskOptions) error {
	if math.IsNaN(o.Padding) || math.IsInf(o.Padding, 0) || o.Padding < 0 || o.Padding > 128 || o.ContrastThreshold < 0 || o.ContrastThreshold > 255 || o.Dilation < 0 || o.Dilation > 16 {
		return errors.New("textinpaint: invalid mask options")
	}
	return nil
}
func flatRegions(regions []Region, w, h int) ([]float32, error) {
	if len(regions) > 100000 {
		return nil, errors.New("textinpaint: too many regions")
	}
	out := make([]float32, 0, len(regions)*8)
	for i, r := range regions {
		q := r.Polygon
		for _, p := range q {
			if math.IsNaN(p.X) || math.IsNaN(p.Y) || math.IsInf(p.X, 0) || math.IsInf(p.Y, 0) || p.X < 0 || p.Y < 0 || p.X > float64(w-1) || p.Y > float64(h-1) {
				return nil, fmt.Errorf("textinpaint: region %d has invalid or out-of-image coordinates", i)
			}
		}
		if !convex(q) || polygonArea(q[:]) < 1 {
			return nil, fmt.Errorf("textinpaint: region %d must be a nondegenerate convex quadrilateral in boundary order", i)
		}
		for _, p := range q {
			out = append(out, float32(p.X), float32(p.Y))
		}
	}
	return out, nil
}

// BuildMask builds a binary mask from automatic or caller-supplied regions.
// White selects pixels to remove; black keeps them. Input pixels are never changed.
func BuildMask(src image.Image, regions []Region, options MaskOptions) (*image.Gray, error) {
	if err := validMaskOptions(options); err != nil {
		return nil, err
	}
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	polygons, err := flatRegions(regions, im.Rect.Dx(), im.Rect.Dy())
	if err != nil {
		return nil, err
	}
	if len(regions) == 0 {
		return image.NewGray(im.Rect), nil
	}
	return cv.Mask(im, polygons, options.Padding, options.ContrastThreshold, options.Dilation, options.PreserveGraphics)
}

// Inpaint removes nonzero mask pixels using local background color when a
// component's boundary is uniform, and OpenCV Telea for other backgrounds.
// The mask must have the same dimensions as src. Transparent inputs are flattened
// onto white. The returned image has zero-based bounds and does not alias src.
func Inpaint(src image.Image, mask *image.Gray, radius float64) (*image.RGBA, error) {
	if math.IsNaN(radius) || math.IsInf(radius, 0) || radius <= 0 || radius > 100 {
		return nil, errors.New("textinpaint: radius must be finite and in (0,100]")
	}
	if mask == nil {
		return nil, errors.New("textinpaint: nil mask")
	}
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	if mask.Bounds().Size() != im.Rect.Size() {
		return nil, errors.New("textinpaint: mask and image dimensions differ")
	}
	compact := image.NewGray(im.Rect)
	draw.Draw(compact, compact.Rect, mask, mask.Rect.Min, draw.Src)
	all := true
	for _, v := range compact.Pix {
		if v == 0 {
			all = false
			break
		}
	}
	if all {
		return nil, errors.New("textinpaint: a fully masked image has no source background")
	}
	return cv.Inpaint(im, compact, radius)
}

// DrawRegions returns a preview with colored polygon outlines.
func DrawRegions(src image.Image, regions []Region) (*image.RGBA, error) {
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	polygons, err := flatRegions(regions, im.Rect.Dx(), im.Rect.Dy())
	if err != nil {
		return nil, err
	}
	return cv.Overlay(im, polygons)
}
