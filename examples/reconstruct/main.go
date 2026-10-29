// This caller owns the OCR endpoint. The library only handles image analysis,
// selective inpainting, and HTML rendering in the current Go process.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"time"

	textinpaint "github.com/iEvan-lhr/inpaint"
	"github.com/iEvan-lhr/inpaint/reconstruct"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	input := flag.String("input", "", "input PNG/JPEG")
	model := flag.String("model", os.Getenv("INPAINT_MODEL"), "detector model path")
	runtime := flag.String("runtime", os.Getenv("INPAINT_RUNTIME"), "ONNX Runtime path")
	endpoint := flag.String("ocr-url", "", "your OCR adapter URL; see README_MODULE.md")
	output := flag.String("out", "result-html", "output directory")
	title := flag.String("title", "图片重建", "HTML title")
	minimum := flag.Float64("min-text-height", 12, "0 includes tiny text")
	flag.Parse()
	if *input == "" || *endpoint == "" {
		return fmt.Errorf("-input and -ocr-url are required")
	}
	f, err := os.Open(*input)
	if err != nil {
		return err
	}
	size, _, err := image.DecodeConfig(f)
	if err != nil {
		f.Close()
		return err
	}
	if size.Width <= 0 || size.Height <= 0 || size.Width > textinpaint.MaxImagePixels/size.Height {
		f.Close()
		return fmt.Errorf("input image too large or empty")
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		f.Close()
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}
	engine, err := textinpaint.New(textinpaint.Config{ModelPath: *model, RuntimePath: *runtime})
	if err != nil {
		return err
	}
	defer engine.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	task, err := reconstruct.Analyze(ctx, engine, img, reconstruct.Options{Processing: &textinpaint.ProcessingOptions{MinTextHeight: *minimum}})
	if err != nil {
		return err
	}
	// []byte becomes base64 in JSON. The endpoint must perform OCR or call your
	// multimodal provider and return reconstruct.Recognition in the response.
	request := struct {
		Analysis reconstruct.Analysis `json:"analysis"`
		ImagePNG []byte               `json:"image_png_base64"`
	}{task.Analysis(), task.OriginalPNG()}
	body, err := json.Marshal(request)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, *endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := os.Getenv("OCR_API_KEY"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("OCR adapter returned HTTP %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, (5<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 5<<20 {
		return fmt.Errorf("OCR response exceeds 5 MiB")
	}
	var recognition reconstruct.Recognition
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(&recognition); err != nil {
		return fmt.Errorf("decode OCR response: %w", err)
	}
	if err = decoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("OCR response must contain exactly one JSON object")
	}
	result, err := task.Reconstruct(ctx, recognition, reconstruct.RenderOptions{Title: *title})
	if err != nil {
		return err
	}
	if err = result.WriteDir(*output); err != nil {
		return err
	}
	fmt.Printf("regions=%d html_lines=%d preserved=%d output=%s\n", len(task.Analysis().Regions), len(result.Page.Lines), len(result.Page.Preserved), *output)
	return nil
}
