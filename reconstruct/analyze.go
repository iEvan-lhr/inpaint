package reconstruct

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"reflect"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

// Task owns an immutable image snapshot and the original detector coordinates.
// Keep it in your application's job store while OCR is running. It has no
// global registry or TTL and needs no Close. Its methods allow concurrent calls.
type Task struct {
	source    *image.RGBA
	original  []byte
	analysis  Analysis
	protected []textinpaint.Region
	mask      textinpaint.MaskOptions
	radius    float64
}

func validContext(ctx context.Context) error {
	if ctx == nil {
		return errors.New("reconstruct: nil context")
	}
	return ctx.Err()
}

func isNil(v any) bool {
	return v == nil || (reflect.ValueOf(v).Kind() == reflect.Ptr && reflect.ValueOf(v).IsNil())
}

// Analyze detects once without erasing any pixels. Neither later mutations of
// src nor edits to Analysis() can change the task's coordinates or background.
func Analyze(ctx context.Context, detector Detector, src image.Image, options Options) (*Task, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	if isNil(detector) || isNil(src) {
		return nil, errors.New("reconstruct: detector and image are required")
	}
	b := src.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 || b.Dx() > textinpaint.MaxImagePixels/b.Dy() {
		return nil, errors.New("reconstruct: invalid image size")
	}
	im := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(im, im.Rect, image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(im, im.Rect, src, b.Min, draw.Over)
	m := textinpaint.DefaultMaskOptions()
	if options.Mask != nil {
		m = *options.Mask
	}
	// Use the public mask validator even when the image has no detections.
	if _, err := textinpaint.BuildMask(im, nil, m); err != nil {
		return nil, err
	}
	radius := options.InpaintRadius
	if radius == 0 {
		radius = 3
	}
	if math.IsNaN(radius) || math.IsInf(radius, 0) || radius <= 0 || radius > 100 {
		return nil, errors.New("reconstruct: invalid inpaint radius")
	}
	var processing *textinpaint.ProcessingOptions
	if options.Processing != nil {
		value := *options.Processing
		if math.IsNaN(value.MinTextHeight) || math.IsInf(value.MinTextHeight, 0) || value.MinTextHeight < 0 || value.MinTextHeight > 256 {
			return nil, errors.New("reconstruct: minimum text height must be finite and in [0,256]")
		}
		processing = &value
	}
	// The detector receives its own image so a custom implementation cannot
	// mutate the task's canonical pixels after returning.
	detectionImage := image.NewRGBA(im.Rect)
	copy(detectionImage.Pix, im.Pix)
	found, err := detector.DetectWithProtection(ctx, detectionImage, processing)
	if err != nil {
		return nil, err
	}
	if found == nil || found.Width != b.Dx() || found.Height != b.Dy() {
		return nil, errors.New("reconstruct: detector dimensions mismatch")
	}
	regions := append([]textinpaint.Region(nil), found.Regions...)
	protected := append([]textinpaint.Region(nil), found.ProtectedRegions...)
	for _, r := range regions {
		if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) || math.IsNaN(float64(r.TextHeight)) || math.IsInf(float64(r.TextHeight), 0) || r.TextHeight < 0 {
			return nil, errors.New("reconstruct: detector returned invalid score or text height")
		}
	}
	// DrawRegions validates polygon coordinates without running inference again.
	if _, err := textinpaint.DrawRegions(im, append(append([]textinpaint.Region(nil), regions...), protected...)); err != nil {
		return nil, err
	}
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	original, err := encodePNG(im)
	if err != nil {
		return nil, err
	}
	var nonce [16]byte
	if _, err = rand.Read(nonce[:]); err != nil {
		return nil, err
	}
	hash := sha256.Sum256(original)
	analysis := Analysis{ID: hex.EncodeToString(nonce[:]), Width: b.Dx(), Height: b.Dy(), SourceSHA256: hex.EncodeToString(hash[:]), Regions: make([]Candidate, 0, len(regions))}
	for i, r := range regions {
		analysis.Regions = append(analysis.Regions, Candidate{ID: fmt.Sprintf("r%06d", i+1), Region: r})
	}
	return &Task{source: im, original: original, analysis: analysis, protected: protected, mask: m, radius: radius}, nil
}

// Analysis returns a copy. Do not treat edited copies as a way to alter geometry.
func (t *Task) Analysis() Analysis {
	a := t.analysis
	a.Regions = append([]Candidate{}, a.Regions...)
	return a
}

func (t *Task) OriginalPNG() []byte { return append([]byte(nil), t.original...) }

// CropPNG returns an axis-aligned PNG for one candidate, using Go coordinates.
// Padding is measured in original pixels, clipped to the image, and must be 0..256.
func (t *Task) CropPNG(regionID string, padding int) ([]byte, error) {
	if padding < 0 || padding > 256 {
		return nil, errors.New("reconstruct: crop padding must be 0..256")
	}
	for _, c := range t.analysis.Regions {
		if c.ID == regionID {
			r := polygonBounds(c.Polygon).Inset(-padding).Intersect(t.source.Rect)
			return encodePNG(t.source.SubImage(r))
		}
	}
	return nil, fmt.Errorf("reconstruct: unknown region %q", regionID)
}

func encodePNG(im image.Image) ([]byte, error) {
	var b bytes.Buffer
	if err := png.Encode(&b, im); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func polygonBounds(q textinpaint.Quad) image.Rectangle {
	x0, y0, x1, y1 := q[0].X, q[0].Y, q[0].X, q[0].Y
	for _, p := range q {
		x0 = math.Min(x0, p.X)
		y0 = math.Min(y0, p.Y)
		x1 = math.Max(x1, p.X)
		y1 = math.Max(y1, p.Y)
	}
	return image.Rect(int(math.Floor(x0)), int(math.Floor(y0)), int(math.Ceil(x1))+1, int(math.Ceil(y1))+1)
}
