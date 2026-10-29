package textinpaint

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/iEvan-lhr/inpaint/internal/cv"
	ort "github.com/yalue/onnxruntime_go"
)

var runtimeState struct {
	sync.Mutex
	refs  int
	owned bool
	path  string
}

func acquireRuntime(path string) error {
	runtimeState.Lock()
	defer runtimeState.Unlock()
	if path != "" {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return err
		}
		path = absolute
	}
	if runtimeState.refs > 0 {
		if path != "" && path != runtimeState.path {
			return errors.New("textinpaint: cannot select another ONNX Runtime while engines are open")
		}
		runtimeState.refs++
		return nil
	}
	if ort.IsInitialized() {
		if path != "" {
			return errors.New("textinpaint: ONNX Runtime was initialized externally; omit RuntimePath to reuse it")
		}
		runtimeState.owned = false
	} else {
		ort.SetSharedLibraryPath(path)
		if err := ort.InitializeEnvironment(); err != nil {
			return fmt.Errorf("initialize ONNX Runtime: %w", err)
		}
		runtimeState.owned = true
	}
	runtimeState.refs = 1
	runtimeState.path = path
	return nil
}
func releaseRuntime() error {
	runtimeState.Lock()
	defer runtimeState.Unlock()
	runtimeState.refs--
	if runtimeState.refs == 0 {
		owned := runtimeState.owned
		runtimeState.owned = false
		runtimeState.path = ""
		if owned {
			return ort.DestroyEnvironment()
		}
	}
	return nil
}

// Engine owns an ONNX inference session. Detect calls on a shared engine are
// serialized; Close waits for active inference and is idempotent. Context
// cancellation is checked between inference passes, not inside a native call.
// An Engine must not be copied after creation.
type Engine struct {
	mu        sync.Mutex
	session   *ort.DynamicAdvancedSession
	detection DetectionOptions
	mask      MaskOptions
	radius    float64
	closed    bool
}

// New loads a PP-OCRv4 mobile DB detection model with one float32 NCHW input
// and one float32 [1,1,H,W] probability output. It performs no downloads.
func New(config Config) (*Engine, error) {
	d, m := DefaultDetectionOptions(), DefaultMaskOptions()
	if config.Detection != nil {
		d = *config.Detection
	}
	if config.Mask != nil {
		m = *config.Mask
	}
	if err := validMaskOptions(m); err != nil {
		return nil, err
	}
	if err := validProcessingOptions(ProcessingOptions{MinTextHeight: d.MinTextHeight}); err != nil {
		return nil, err
	}
	if d.LongSide < 64 || d.LongSide > 4096 || d.TileSize < 0 || (d.TileSize > 0 && d.TileSize < 64) || d.TileSize > 4096 || math.IsNaN(d.TileScale) || math.IsInf(d.TileScale, 0) || d.TileScale <= 0 || d.TileScale > 4 || !(d.PixelThreshold > 0 && d.PixelThreshold < 1) || !(d.BoxThreshold > 0 && d.BoxThreshold < 1) || !(d.UnclipRatio > 0 && d.UnclipRatio <= 5) || d.MaxCandidates < 1 || d.MaxCandidates > 10000 || d.Threads < 1 || d.Threads > 128 {
		return nil, errors.New("textinpaint: invalid detection options; start with DefaultDetectionOptions()")
	}
	radius := config.InpaintRadius
	if radius == 0 {
		radius = 3
	}
	if math.IsNaN(radius) || math.IsInf(radius, 0) || radius <= 0 || radius > 100 {
		return nil, errors.New("textinpaint: invalid inpaint radius")
	}
	if config.ModelPath == "" {
		return nil, errors.New("textinpaint: ModelPath is required")
	}
	info, err := os.Stat(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}
	if info.IsDir() {
		return nil, errors.New("textinpaint: model path is a directory")
	}
	if err := acquireRuntime(config.RuntimePath); err != nil {
		return nil, err
	}
	success := false
	defer func() {
		if !success {
			_ = releaseRuntime()
		}
	}()
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()
	if err = options.SetIntraOpNumThreads(d.Threads); err != nil {
		return nil, err
	}
	if err = options.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	inputs, outputs, err := ort.GetInputOutputInfoWithOptions(config.ModelPath, options)
	if err != nil {
		return nil, fmt.Errorf("inspect detector model: %w", err)
	}
	if len(inputs) != 1 || len(outputs) != 1 || len(inputs[0].Dimensions) != 4 || len(outputs[0].Dimensions) != 4 || inputs[0].DataType != ort.TensorElementDataTypeFloat || outputs[0].DataType != ort.TensorElementDataTypeFloat {
		return nil, errors.New("textinpaint: expected a PP-OCR float32 NCHW detection model with one input and one output")
	}
	session, err := ort.NewDynamicAdvancedSession(config.ModelPath, []string{inputs[0].Name}, []string{outputs[0].Name}, options)
	if err != nil {
		return nil, fmt.Errorf("create detector session: %w", err)
	}
	success = true
	return &Engine{session: session, detection: d, mask: m, radius: radius}, nil
}

func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	err := e.session.Destroy()
	e.session = nil
	return errors.Join(err, releaseRuntime())
}

// Detect returns deduplicated quadrilaterals in input-pixel coordinates.
func (e *Engine) Detect(ctx context.Context, src image.Image) ([]Region, error) {
	return e.DetectWithOptions(ctx, src, ProcessingOptions{MinTextHeight: e.detection.MinTextHeight})
}

// DetectWithOptions overrides the small-text threshold for this call only.
// The shared engine's configuration is not modified.
func (e *Engine) DetectWithOptions(ctx context.Context, src image.Image, options ProcessingOptions) ([]Region, error) {
	if err := validProcessingOptions(options); err != nil {
		return nil, err
	}
	if ctx == nil {
		return nil, errors.New("textinpaint: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, ErrClosed
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	var regions []Region
	addPass := func(crop image.Rectangle, scale float64) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		found, err := e.detectPass(im, crop, scale)
		if err != nil {
			return err
		}
		regions = append(regions, found...)
		return ctx.Err()
	}
	if err = addPass(im.Rect, 1); err != nil {
		return nil, err
	}
	d := e.detection
	if d.TileSize > 0 && max(im.Rect.Dx(), im.Rect.Dy()) > d.TileSize {
		for _, top := range tileStarts(im.Rect.Dy(), d.TileSize) {
			for _, left := range tileStarts(im.Rect.Dx(), d.TileSize) {
				crop := image.Rect(left, top, min(left+d.TileSize, im.Rect.Dx()), min(top+d.TileSize, im.Rect.Dy()))
				if err = addPass(crop, d.TileScale); err != nil {
					return nil, err
				}
			}
		}
	}
	// Page numbers and tiny footers can disappear in a full-page resize.
	// Use narrow, enlarged margin crops with the same model and thresholds.
	if d.DetectMargins && im.Rect.Dy() >= 256 {
		width := min(640, im.Rect.Dx())
		height := min(256, max(64, im.Rect.Dy()/10))
		for _, top := range []int{0, im.Rect.Dy() - height} {
			for _, left := range tileStarts(im.Rect.Dx(), width) {
				if err = addPass(image.Rect(left, top, left+width, top+height), 2); err != nil {
					return nil, err
				}
			}
		}
	}
	filtered, err := filterSmallText(im, deduplicate(regions), options.MinTextHeight, e.mask.PreserveGraphics)
	if err != nil {
		return nil, err
	}
	return filtered, ctx.Err()
}

func filterSmallText(im *image.RGBA, regions []Region, minimum float64, preserveGraphics ...bool) ([]Region, error) {
	polygons, err := flatRegions(regions, im.Rect.Dx(), im.Rect.Dy())
	if err != nil {
		return nil, err
	}
	heights, err := cv.TextHeights(im, polygons)
	if err != nil {
		return nil, err
	}
	measured := make([]Region, 0, len(regions))
	for i, r := range regions {
		r.TextHeight = heights[i]
		measured = append(measured, r)
	}
	protect := len(preserveGraphics) == 0 || preserveGraphics[0]
	kept, _, err := selectDetectedText(im, measured, minimum, protect)
	return kept, err
}

// DetectionResult retains ignored polygons so a later selective removal can
// protect small text and embedded artwork without detecting the image again.
type DetectionResult struct {
	Regions          []Region `json:"regions"`
	ProtectedRegions []Region `json:"protected_regions"`
	Width            int      `json:"width"`
	Height           int      `json:"height"`
}

// DetectWithProtection detects once and returns both eligible and protected
// regions. Nil options use this engine's configured small-text threshold.
func (e *Engine) DetectWithProtection(ctx context.Context, src image.Image, options *ProcessingOptions) (*DetectionResult, error) {
	minimum := e.detection.MinTextHeight
	if options != nil {
		if err := validProcessingOptions(*options); err != nil {
			return nil, err
		}
		minimum = options.MinTextHeight
	}
	all, err := e.DetectWithOptions(ctx, src, ProcessingOptions{})
	if err != nil {
		return nil, err
	}
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	kept, ignored, err := selectDetectedText(im, all, minimum, e.mask.PreserveGraphics)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return &DetectionResult{Regions: kept, ProtectedRegions: ignored, Width: im.Rect.Dx(), Height: im.Rect.Dy()}, nil
}

// A framed inset dominated by tiny labels is a picture of an interface. Keeping
// the whole inset also protects button artwork falsely detected as large text.
func selectDetectedText(im *image.RGBA, regions []Region, minimum float64, preserveGraphics bool) ([]Region, []Region, error) {
	var protected []Region
	if minimum > 0 && preserveGraphics {
		panels, err := cv.GraphicPanels(im)
		if err != nil {
			return nil, nil, err
		}
		for _, panel := range panels {
			q := Quad{{float64(panel.Min.X), float64(panel.Min.Y)}, {float64(panel.Max.X - 1), float64(panel.Min.Y)}, {float64(panel.Max.X - 1), float64(panel.Max.Y - 1)}, {float64(panel.Min.X), float64(panel.Max.Y - 1)}}
			inside, small := 0., 0.
			for _, r := range regions {
				if intersectionArea(q, r.Polygon) < polygonArea(r.Polygon[:])*.8 {
					continue
				}
				// Weight a line by its approximate character count, so a few
				// circular callout numbers do not outweigh an entire tiny label.
				longest := 0.
				for k, p := range r.Polygon {
					next := r.Polygon[(k+1)%4]
					longest = math.Max(longest, math.Hypot(p.X-next.X, p.Y-next.Y))
				}
				weight := math.Max(1, longest/math.Max(1, float64(r.TextHeight)))
				inside += weight
				if float64(r.TextHeight) < minimum {
					small += weight
				}
			}
			if small < 2 || small < inside*.6 {
				continue
			}
			pad := panel.Inset(-3).Intersect(im.Rect)
			protected = append(protected, Region{Polygon: Quad{{float64(pad.Min.X), float64(pad.Min.Y)}, {float64(pad.Max.X - 1), float64(pad.Min.Y)}, {float64(pad.Max.X - 1), float64(pad.Max.Y - 1)}, {float64(pad.Min.X), float64(pad.Max.Y - 1)}}})
		}
	}
	kept, ignored := make([]Region, 0, len(regions)), append([]Region(nil), protected...)
	for _, r := range regions {
		skip := float64(r.TextHeight) < minimum
		for _, p := range protected {
			if intersectionArea(p.Polygon, r.Polygon) > polygonArea(r.Polygon[:])*.8 {
				skip = true
				break
			}
		}
		if skip {
			ignored = append(ignored, r)
		} else {
			kept = append(kept, r)
		}
	}
	return kept, ignored, nil
}

func (e *Engine) detectPass(im *image.RGBA, crop image.Rectangle, scale float64) ([]Region, error) {
	sw, sh := int(math.RoundToEven(float64(crop.Dx())*scale)), int(math.RoundToEven(float64(crop.Dy())*scale))
	ratio := math.Min(1, float64(e.detection.LongSide)/float64(max(sw, sh)))
	w := max(32, int(math.RoundToEven(float64(int(float64(sw)*ratio))/32))*32)
	h := max(32, int(math.RoundToEven(float64(int(float64(sh)*ratio))/32))*32)
	input, err := cv.Prepare(im, crop, scale, w, h)
	if err != nil {
		return nil, err
	}
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(h), int64(w)), input)
	if err != nil {
		return nil, err
	}
	defer tensor.Destroy()
	output := make([]ort.Value, 1)
	defer func() {
		for _, v := range output {
			if v != nil {
				_ = v.Destroy()
			}
		}
	}()
	if err = e.session.Run([]ort.Value{tensor}, output); err != nil {
		return nil, fmt.Errorf("run detector: %w", err)
	}
	probability, ok := output[0].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("textinpaint: detector returned a non-float32 output")
	}
	shape := probability.GetShape()
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] < 1 || shape[3] < 1 || shape[2] > 4096 || shape[3] > 4096 {
		return nil, fmt.Errorf("textinpaint: unexpected detector output shape %v", shape)
	}
	d := e.detection
	raw, err := cv.Boxes(probability.GetData(), int(shape[3]), int(shape[2]), sw, sh, d.PixelThreshold, d.BoxThreshold, d.UnclipRatio, d.MaxCandidates)
	if err != nil {
		return nil, err
	}
	out := make([]Region, 0, len(raw)/9)
	for i := 0; i < len(raw); i += 9 {
		var q Quad
		for k := 0; k < 4; k++ {
			q[k] = Point{X: math.Min(float64(im.Rect.Dx()-1), float64(raw[i+k*2])/scale+float64(crop.Min.X)), Y: math.Min(float64(im.Rect.Dy()-1), float64(raw[i+k*2+1])/scale+float64(crop.Min.Y))}
		}
		out = append(out, Region{Polygon: orderQuad(q), Score: raw[i+8]})
	}
	return out, nil
}

// Remove performs detection, mask generation and inpainting. Returned regions
// can be saved as JSON for review or subsequent HTML reconstruction.
func (e *Engine) Remove(ctx context.Context, src image.Image) (*Result, error) {
	return e.RemoveWithOptions(ctx, src, ProcessingOptions{MinTextHeight: e.detection.MinTextHeight})
}

// RemoveWithOptions applies the same filter as DetectWithOptions before masks
// are built, keeping ignored small text out of both the result JSON and mask.
func (e *Engine) RemoveWithOptions(ctx context.Context, src image.Image, options ProcessingOptions) (*Result, error) {
	if err := validProcessingOptions(options); err != nil {
		return nil, err
	}
	started := time.Now()
	all, err := e.DetectWithOptions(ctx, src, ProcessingOptions{})
	if err != nil {
		return nil, err
	}
	im, err := normalized(src)
	if err != nil {
		return nil, err
	}
	regions, ignored, err := selectDetectedText(im, all, options.MinTextHeight, e.mask.PreserveGraphics)
	if err != nil {
		return nil, err
	}
	detected := time.Now()
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	mask, err := BuildMask(src, regions, e.mask)
	if err != nil {
		return nil, err
	}
	// A neighbouring large text mask may expand into a small-text box. Reserve
	// ignored regions after expansion as well, keeping their original pixels.
	protectRegions(mask, ignored)
	masked := time.Now()
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	repaired, err := Inpaint(src, mask, e.radius)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return &Result{Image: repaired, Mask: mask, Regions: regions, Width: repaired.Rect.Dx(), Height: repaired.Rect.Dy(), Timings: Timings{DetectionMS: detected.Sub(started).Milliseconds(), MaskMS: masked.Sub(detected).Milliseconds(), InpaintMS: time.Since(masked).Milliseconds()}}, nil
}
