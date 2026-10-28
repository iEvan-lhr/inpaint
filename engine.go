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
	return deduplicate(regions), nil
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
	started := time.Now()
	regions, err := e.Detect(ctx, src)
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
