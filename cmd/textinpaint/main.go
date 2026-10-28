package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	input := flag.String("input", "", "input PNG or JPEG")
	model := flag.String("model", "models/ch_PP-OCRv4_det_infer.onnx", "local PP-OCRv4 detection model")
	runtime := flag.String("runtime", "", "ONNX Runtime shared library path")
	output := flag.String("out", "result", "output directory")
	tile := flag.Int("tile-size", 1280, "detail crop size; 0 disables detail crops")
	score := flag.Float64("box-threshold", .5, "minimum text box score")
	detectOnly := flag.Bool("detect-only", false, "only output polygons and their preview")
	preserve := flag.Bool("preserve-graphics", true, "preserve document rules and geometric markers")
	flag.Parse()
	if *input == "" {
		return fmt.Errorf("-input is required")
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	size, _, err := image.DecodeConfig(f)
	if err != nil {
		f.Close()
		return fmt.Errorf("read image header: %w", err)
	}
	if size.Width <= 0 || size.Height <= 0 || size.Height > textinpaint.MaxImagePixels || size.Width > textinpaint.MaxImagePixels/size.Height {
		f.Close()
		return fmt.Errorf("input image is too large or empty")
	}
	if _, err = f.Seek(0, 0); err != nil {
		f.Close()
		return err
	}
	src, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}
	d, m := textinpaint.DefaultDetectionOptions(), textinpaint.DefaultMaskOptions()
	d.TileSize = *tile
	d.BoxThreshold = float32(*score)
	m.PreserveGraphics = *preserve
	engine, err := textinpaint.New(textinpaint.Config{ModelPath: *model, RuntimePath: *runtime, Detection: &d, Mask: &m})
	if err != nil {
		return err
	}
	defer engine.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err = os.MkdirAll(*output, 0755); err != nil {
		return err
	}
	var result *textinpaint.Result
	if *detectOnly {
		started := time.Now()
		regions, e := engine.Detect(ctx, src)
		if e != nil {
			return e
		}
		result = &textinpaint.Result{Regions: regions, Width: src.Bounds().Dx(), Height: src.Bounds().Dy(), Timings: textinpaint.Timings{DetectionMS: time.Since(started).Milliseconds()}}
	} else {
		result, err = engine.Remove(ctx, src)
		if err != nil {
			return err
		}
		if err = writePNG(filepath.Join(*output, "no_text.png"), result.Image); err != nil {
			return err
		}
		if err = writePNG(filepath.Join(*output, "mask.png"), result.Mask); err != nil {
			return err
		}
	}
	overlay, err := textinpaint.DrawRegions(src, result.Regions)
	if err != nil {
		return err
	}
	if err = writePNG(filepath.Join(*output, "detections.png"), overlay); err != nil {
		return err
	}
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(*output, "regions.json"), data, 0644); err != nil {
		return err
	}
	fmt.Printf("regions=%d size=%dx%d detection=%dms mask=%dms inpaint=%dms output=%s\n", len(result.Regions), result.Width, result.Height, result.Timings.DetectionMS, result.Timings.MaskMS, result.Timings.InpaintMS, *output)
	return nil
}
func writePNG(path string, im image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err = png.Encode(f, im); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
