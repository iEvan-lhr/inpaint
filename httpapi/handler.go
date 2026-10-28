// Package httpapi exposes text detection and removal over HTTP.
package httpapi

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/subtle"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"log/slog"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"strconv"
	"time"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

//go:embed index.html
var indexHTML []byte

// Processor is implemented by *textinpaint.Engine. The caller owns its lifetime
// and must close it only after the HTTP server has finished serving requests.
type Processor interface {
	Detect(context.Context, image.Image) ([]textinpaint.Region, error)
	Remove(context.Context, image.Image) (*textinpaint.Result, error)
}

// OptionsProcessor optionally supports a per-request small-text threshold.
// *textinpaint.Engine implements this without changing shared configuration.
type OptionsProcessor interface {
	DetectWithOptions(context.Context, image.Image, textinpaint.ProcessingOptions) ([]textinpaint.Region, error)
	RemoveWithOptions(context.Context, image.Image, textinpaint.ProcessingOptions) (*textinpaint.Result, error)
}

// Options configure resource limits. Zero values use the documented defaults.
// APIKey, when nonempty, requires Authorization: Bearer <key> on POST endpoints.
type Options struct {
	MaxUploadBytes int64
	MaxPixels      int
	MaxConcurrent  int
	Timeout        time.Duration
	APIKey         string
}

func DefaultOptions() Options {
	return Options{MaxUploadBytes: 20 << 20, MaxPixels: 16_000_000, MaxConcurrent: 1, Timeout: 2 * time.Minute}
}

type Handler struct {
	processor Processor
	options   Options
	slots     chan struct{}
	mux       *http.ServeMux
}

// New creates a handler with a browser test page, /healthz, /v1/remove and
// /v1/detect. Requests beyond MaxConcurrent receive 503 instead of queuing.
func New(processor Processor, options Options) (*Handler, error) {
	if processor == nil {
		return nil, errors.New("httpapi: a processor is required")
	}
	defaults := DefaultOptions()
	if options.MaxUploadBytes == 0 {
		options.MaxUploadBytes = defaults.MaxUploadBytes
	}
	if options.MaxPixels == 0 {
		options.MaxPixels = defaults.MaxPixels
	}
	if options.MaxConcurrent == 0 {
		options.MaxConcurrent = defaults.MaxConcurrent
	}
	if options.Timeout == 0 {
		options.Timeout = defaults.Timeout
	}
	if options.MaxUploadBytes < 1 || options.MaxUploadBytes > 1<<30 || options.MaxPixels < 1 || options.MaxPixels > textinpaint.MaxImagePixels || options.MaxConcurrent < 1 || options.MaxConcurrent > 32 || options.Timeout < 0 || options.Timeout > time.Hour {
		return nil, errors.New("httpapi: invalid resource limits")
	}
	h := &Handler{processor: processor, options: options, slots: make(chan struct{}, options.MaxConcurrent), mux: http.NewServeMux()}
	h.mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})
	h.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	h.mux.HandleFunc("POST /v1/remove", h.process)
	h.mux.HandleFunc("POST /v1/detect", h.process)
	return h, nil
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) process(w http.ResponseWriter, r *http.Request) {
	if h.options.APIKey != "" && subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+h.options.APIKey)) != 1 {
		w.Header().Set("WWW-Authenticate", "Bearer")
		writeError(w, http.StatusUnauthorized, "unauthorized", "请提供正确的 API Key")
		return
	}
	format := r.URL.Query().Get("format")
	detectOnly := r.URL.Path == "/v1/detect"
	if (detectOnly && format != "") || (!detectOnly && format != "" && format != "png" && format != "zip") {
		writeError(w, http.StatusBadRequest, "invalid_format", "去字接口的 format 仅支持 png 或 zip；检测接口无需 format")
		return
	}
	var processing *textinpaint.ProcessingOptions
	var configurable OptionsProcessor
	if values, present := r.URL.Query()["min_text_height"]; present {
		minimum, parseErr := strconv.ParseFloat(values[0], 64)
		if len(values) != 1 || parseErr != nil || math.IsNaN(minimum) || math.IsInf(minimum, 0) || minimum < 0 || minimum > 256 {
			writeError(w, http.StatusBadRequest, "invalid_min_text_height", "最小文字高度须为 0 到 256 的有限数字；0 表示不过滤小字")
			return
		}
		var ok bool
		configurable, ok = h.processor.(OptionsProcessor)
		if !ok {
			writeError(w, http.StatusBadRequest, "unsupported_options", "当前处理器不支持按请求设置文字高度")
			return
		}
		processing = &textinpaint.ProcessingOptions{MinTextHeight: minimum}
	}
	select {
	case h.slots <- struct{}{}:
		defer func() { <-h.slots }()
	default:
		w.Header().Set("Retry-After", "2")
		writeError(w, http.StatusServiceUnavailable, "busy", "服务正在处理图片，请稍后重试")
		return
	}
	started := time.Now()
	ctx, cancel := context.WithTimeout(r.Context(), h.options.Timeout)
	defer cancel()
	src, problem := h.readImage(w, r)
	if problem != nil {
		writeError(w, problem.status, problem.code, problem.message)
		return
	}
	if err := ctx.Err(); err != nil {
		h.processingError(w, err)
		return
	}
	var result *textinpaint.Result
	var err error
	if detectOnly {
		detectionStarted := time.Now()
		var regions []textinpaint.Region
		if processing != nil {
			regions, err = configurable.DetectWithOptions(ctx, src, *processing)
		} else {
			regions, err = h.processor.Detect(ctx, src)
		}
		result = &textinpaint.Result{Regions: regions, Width: src.Bounds().Dx(), Height: src.Bounds().Dy(), Timings: textinpaint.Timings{DetectionMS: time.Since(detectionStarted).Milliseconds()}}
	} else {
		if processing != nil {
			result, err = configurable.RemoveWithOptions(ctx, src, *processing)
		} else {
			result, err = h.processor.Remove(ctx, src)
		}
	}
	if err != nil {
		h.processingError(w, err)
		return
	}
	if err = ctx.Err(); err != nil {
		h.processingError(w, err)
		return
	}
	if result == nil {
		h.processingError(w, errors.New("processor returned no result"))
		return
	}
	if result.Regions == nil {
		result.Regions = []textinpaint.Region{}
	}
	var payload []byte
	contentType, disposition := "application/json; charset=utf-8", ""
	switch {
	case detectOnly:
		payload, err = json.Marshal(result)
	case format == "zip":
		payload, err = encodeBundle(src, result)
		contentType, disposition = "application/zip", `attachment; filename="textinpaint-result.zip"`
	default:
		var output bytes.Buffer
		if result.Image == nil {
			err = errors.New("processor returned no image")
		} else {
			err = png.Encode(&output, result.Image)
		}
		payload = output.Bytes()
		contentType, disposition = "image/png", `inline; filename="no_text.png"`
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		h.processingError(w, err)
		return
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	w.Header().Set("X-Region-Count", strconv.Itoa(len(result.Regions)))
	w.Header().Set("X-Processing-Ms", strconv.FormatInt(time.Since(started).Milliseconds(), 10))
	if disposition != "" {
		w.Header().Set("Content-Disposition", disposition)
	}
	_, _ = w.Write(payload)
}

type imageProblem struct {
	status  int
	code    string
	message string
}

func (h *Handler) readImage(w http.ResponseWriter, r *http.Request) (image.Image, *imageProblem) {
	tooLarge := &imageProblem{http.StatusRequestEntityTooLarge, "upload_too_large", fmt.Sprintf("上传请求不能超过 %d 字节", h.options.MaxUploadBytes)}
	if r.ContentLength > h.options.MaxUploadBytes {
		return nil, tooLarge
	}
	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" || params["boundary"] == "" {
		return nil, &imageProblem{http.StatusUnsupportedMediaType, "invalid_content_type", "请使用 multipart/form-data，图片字段名为 image"}
	}
	r.Body = http.MaxBytesReader(w, r.Body, h.options.MaxUploadBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var sizeError *http.MaxBytesError
		if errors.As(err, &sizeError) {
			return nil, tooLarge
		}
		return nil, &imageProblem{http.StatusBadRequest, "invalid_body", "无法读取上传内容"}
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	var data []byte
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, &imageProblem{http.StatusBadRequest, "invalid_multipart", "上传表单不完整"}
		}
		if part.FormName() != "image" || part.FileName() == "" || data != nil {
			_ = part.Close()
			return nil, &imageProblem{http.StatusBadRequest, "invalid_image_field", "请上传且仅上传一个 image 文件字段"}
		}
		data, err = io.ReadAll(part)
		_ = part.Close()
		if err != nil {
			return nil, &imageProblem{http.StatusBadRequest, "invalid_multipart", "图片上传不完整"}
		}
	}
	if len(data) == 0 {
		return nil, &imageProblem{http.StatusBadRequest, "missing_image", "请选择一张图片，上传字段名为 image"}
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || (format != "png" && format != "jpeg") {
		return nil, &imageProblem{http.StatusUnsupportedMediaType, "unsupported_image", "仅支持有效的 PNG 或 JPEG 图片"}
	}
	if config.Width <= 0 || config.Height <= 0 || config.Height > h.options.MaxPixels || config.Width > h.options.MaxPixels/config.Height {
		return nil, &imageProblem{http.StatusRequestEntityTooLarge, "too_many_pixels", fmt.Sprintf("图片不能超过 %d 像素", h.options.MaxPixels)}
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, &imageProblem{http.StatusBadRequest, "invalid_image", "图片内容损坏，无法解码"}
	}
	return src, nil
}

func encodeBundle(src image.Image, result *textinpaint.Result) ([]byte, error) {
	if result.Image == nil || result.Mask == nil {
		return nil, errors.New("processor returned an incomplete result")
	}
	overlay, err := textinpaint.DrawRegions(src, result.Regions)
	if err != nil {
		return nil, err
	}
	var output bytes.Buffer
	archive := zip.NewWriter(&output)
	for _, item := range []struct {
		name  string
		image image.Image
	}{{"no_text.png", result.Image}, {"mask.png", result.Mask}, {"detections.png", overlay}} {
		file, err := archive.CreateHeader(&zip.FileHeader{Name: item.name, Method: zip.Store})
		if err != nil {
			return nil, err
		}
		if err = png.Encode(file, item.image); err != nil {
			return nil, err
		}
	}
	metadata, err := archive.Create("regions.json")
	if err != nil {
		return nil, err
	}
	if err = json.NewEncoder(metadata).Encode(result); err != nil {
		return nil, err
	}
	if err = archive.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func (h *Handler) processingError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusGatewayTimeout, "processing_timeout", "图片处理超时，请缩小图片或增加服务超时设置")
	case errors.Is(err, context.Canceled):
		writeError(w, http.StatusRequestTimeout, "request_canceled", "请求已取消")
	default:
		slog.Error("image processing failed", "error", err)
		writeError(w, http.StatusInternalServerError, "processing_failed", "图片处理失败，请查看服务日志")
	}
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
