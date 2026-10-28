package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	textinpaint "github.com/iEvan-lhr/inpaint"
	"github.com/iEvan-lhr/inpaint/httpapi"
)

func main() {
	if err := run(); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run() error {
	modelDefault, runtimeDefault := defaultFiles()
	listen := flag.String("listen", "0.0.0.0:8080", "HTTP listening address")
	model := flag.String("model", modelDefault, "PP-OCRv4 detection model")
	runtimePath := flag.String("runtime", runtimeDefault, "ONNX Runtime shared library")
	maxUpload := flag.Int64("max-upload-mb", 20, "maximum multipart request size in MiB")
	maxPixels := flag.Int("max-pixels", 16_000_000, "maximum decoded image pixels")
	concurrent := flag.Int("max-concurrent", 1, "simultaneous processing requests; excess requests receive 503")
	timeout := flag.Duration("timeout", 2*time.Minute, "processing deadline")
	tile := flag.Int("tile-size", 1280, "detail crop size; 0 disables detail crops")
	score := flag.Float64("box-threshold", .5, "minimum text box score")
	preserve := flag.Bool("preserve-graphics", true, "preserve document rules and geometric markers")
	margins := flag.Bool("detect-margins", true, "enlarge page margins to find tiny headers and page numbers")
	minimum := flag.Float64("min-text-height", textinpaint.DefaultDetectionOptions().MinTextHeight, "minimum estimated glyph height in original pixels; 0 includes tiny text")
	flag.Parse()
	if *maxUpload < 1 || *maxUpload > 1024 {
		return fmt.Errorf("-max-upload-mb must be in [1,1024]")
	}
	d, m := textinpaint.DefaultDetectionOptions(), textinpaint.DefaultMaskOptions()
	d.TileSize, d.BoxThreshold = *tile, float32(*score)
	d.DetectMargins = *margins
	d.MinTextHeight = *minimum
	m.PreserveGraphics = *preserve
	engine, err := textinpaint.New(textinpaint.Config{ModelPath: *model, RuntimePath: *runtimePath, Detection: &d, Mask: &m})
	if err != nil {
		return err
	}
	defer engine.Close()
	handler, err := httpapi.New(engine, httpapi.Options{MaxUploadBytes: *maxUpload << 20, MaxPixels: *maxPixels, MaxConcurrent: *concurrent, Timeout: *timeout, APIKey: os.Getenv("TEXTINPAINT_API_KEY")})
	if err != nil {
		return err
	}
	// New's zero timeout means the default, and the transport must use it too.
	processingTimeout := *timeout
	if processingTimeout == 0 {
		processingTimeout = httpapi.DefaultOptions().Timeout
	}
	server := &http.Server{
		Addr: *listen, Handler: handler, ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 30 * time.Second, WriteTimeout: processingTimeout + 30*time.Second,
		IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	listener, err := net.Listen("tcp", *listen)
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	serving := make(chan error, 1)
	go func() { serving <- server.Serve(listener) }()
	log.Printf("Textinpaint ready at %s; open / to test, GET /healthz, POST /v1/remove", listener.Addr())
	log.Printf("upload limit=%d MiB, pixel limit=%d, concurrency=%d, API key enabled=%t", *maxUpload, *maxPixels, *concurrent, os.Getenv("TEXTINPAINT_API_KEY") != "")
	select {
	case err = <-serving:
	case <-ctx.Done():
		stop()
		log.Print("Shutting down; waiting for active requests")
		shutdownContext, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if shutdownErr := server.Shutdown(shutdownContext); shutdownErr != nil {
			_ = server.Close()
			return shutdownErr
		}
		err = <-serving
	}
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func defaultFiles() (string, string) {
	model, runtimeLibrary := "models/ch_PP-OCRv4_det_infer.onnx", ""
	executable, err := os.Executable()
	if err != nil {
		return model, runtimeLibrary
	}
	bin := filepath.Dir(executable)
	bundledModel := filepath.Join(bin, "..", "models", "ch_PP-OCRv4_det_infer.onnx")
	if info, err := os.Stat(bundledModel); err == nil && !info.IsDir() {
		model = filepath.Clean(bundledModel)
	}
	if runtime.GOOS == "windows" {
		bundledRuntime := filepath.Join(bin, "onnxruntime.dll")
		if info, err := os.Stat(bundledRuntime); err == nil && !info.IsDir() {
			runtimeLibrary = bundledRuntime
		}
	}
	return model, runtimeLibrary
}
