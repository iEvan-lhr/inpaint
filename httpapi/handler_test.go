package httpapi_test

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	textinpaint "github.com/iEvan-lhr/inpaint"
	"github.com/iEvan-lhr/inpaint/httpapi"
)

type processor struct {
	remove func(context.Context, image.Image) (*textinpaint.Result, error)
	calls  int
}

type optionsProcessor struct {
	processor
	thresholds []float64
}

func (p *optionsProcessor) DetectWithOptions(ctx context.Context, src image.Image, options textinpaint.ProcessingOptions) ([]textinpaint.Region, error) {
	p.thresholds = append(p.thresholds, options.MinTextHeight)
	return p.Detect(ctx, src)
}
func (p *optionsProcessor) RemoveWithOptions(ctx context.Context, src image.Image, options textinpaint.ProcessingOptions) (*textinpaint.Result, error) {
	p.thresholds = append(p.thresholds, options.MinTextHeight)
	return p.Remove(ctx, src)
}

func TestPerRequestMinimumTextHeight(t *testing.T) {
	p := &optionsProcessor{}
	h, err := httpapi.New(p, httpapi.Options{})
	if err != nil {
		t.Fatal(err)
	}
	data := imageBytes(t, "png", 20, 20)
	for _, path := range []string{"/v1/detect?min_text_height=18", "/v1/remove?min_text_height=0", "/v1/remove?format=zip&min_text_height=12"} {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, upload(t, path, data))
		if rr.Code != 200 {
			t.Fatalf("%s: %d %s", path, rr.Code, rr.Body.String())
		}
	}
	if len(p.thresholds) != 3 || p.thresholds[0] != 18 || p.thresholds[1] != 0 || p.thresholds[2] != 12 {
		t.Fatalf("options not forwarded: %v", p.thresholds)
	}
	for _, value := range []string{"", "-1", "257", "NaN", "Inf", "abc", "12&min_text_height=16"} {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, upload(t, "/v1/detect?min_text_height="+value, data))
		checkError(t, rr, 400, "invalid_min_text_height")
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, upload(t, "/v1/detect", data))
	if rr.Code != 200 || len(p.thresholds) != 3 {
		t.Fatal("per-request options leaked into the next request")
	}
	rr = httptest.NewRecorder()
	handler(t, &processor{}, httpapi.Options{}).ServeHTTP(rr, upload(t, "/v1/detect?min_text_height=12", data))
	checkError(t, rr, 400, "unsupported_options")
}

func (p *processor) Detect(ctx context.Context, src image.Image) ([]textinpaint.Region, error) {
	p.calls++
	return []textinpaint.Region{{Polygon: textinpaint.Quad{{X: 2, Y: 2}, {X: 8, Y: 2}, {X: 8, Y: 8}, {X: 2, Y: 8}}, Score: .9}}, nil
}

func (p *processor) Remove(ctx context.Context, src image.Image) (*textinpaint.Result, error) {
	if p.remove != nil {
		return p.remove(ctx, src)
	}
	regions, err := p.Detect(ctx, src)
	return &textinpaint.Result{Image: white(src.Bounds().Dx(), src.Bounds().Dy()), Mask: image.NewGray(src.Bounds()), Regions: regions, Width: src.Bounds().Dx(), Height: src.Bounds().Dy()}, err
}

func white(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(im, im.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	return im
}

func imageBytes(t *testing.T, format string, w, h int) []byte {
	t.Helper()
	var data bytes.Buffer
	var err error
	if format == "jpeg" {
		err = jpeg.Encode(&data, white(w, h), nil)
	} else {
		err = png.Encode(&data, white(w, h))
	}
	if err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func upload(t *testing.T, path string, files ...[]byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	for _, data := range files {
		file, err := form.CreateFormFile("image", "input.png")
		if err != nil {
			t.Fatal(err)
		}
		if _, err = file.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := form.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, path, &body)
	request.Header.Set("Content-Type", form.FormDataContentType())
	return request
}

func handler(t *testing.T, p *processor, options httpapi.Options) *httpapi.Handler {
	t.Helper()
	h, err := httpapi.New(p, options)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

func checkError(t *testing.T, rr *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if rr.Code != status {
		t.Fatalf("status=%d, want %d, body=%s", rr.Code, status, rr.Body.String())
	}
	var result struct {
		Error struct{ Code, Message string }
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &result); err != nil || result.Error.Code != code || result.Error.Message == "" {
		t.Fatalf("invalid error response: %s (%v)", rr.Body.String(), err)
	}
}

func TestPNGJPEGDetectionAndZIP(t *testing.T) {
	p := &processor{}
	h := handler(t, p, httpapi.Options{})
	for _, format := range []string{"png", "jpeg"} {
		t.Run(format, func(t *testing.T) {
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, upload(t, "/v1/remove", imageBytes(t, format, 16, 16)))
			if rr.Code != 200 || rr.Header().Get("Content-Type") != "image/png" || rr.Header().Get("X-Region-Count") != "1" || rr.Header().Get("X-Processing-Ms") == "" {
				t.Fatalf("response: %d %v %s", rr.Code, rr.Header(), rr.Body.String())
			}
			im, err := png.Decode(rr.Body)
			if err != nil || im.Bounds().Size() != image.Pt(16, 16) {
				t.Fatalf("invalid output PNG: %v", err)
			}
		})
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, upload(t, "/v1/detect", imageBytes(t, "png", 16, 16)))
	var metadata textinpaint.Result
	if err := json.Unmarshal(rr.Body.Bytes(), &metadata); rr.Code != 200 || err != nil || metadata.Width != 16 || len(metadata.Regions) != 1 {
		t.Fatalf("invalid detection response: %d %s (%v)", rr.Code, rr.Body.String(), err)
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, upload(t, "/v1/remove?format=zip", imageBytes(t, "png", 16, 16)))
	if rr.Code != 200 || rr.Header().Get("Content-Type") != "application/zip" {
		t.Fatalf("invalid ZIP response: %d %s", rr.Code, rr.Body.String())
	}
	archive, err := zip.NewReader(bytes.NewReader(rr.Body.Bytes()), int64(rr.Body.Len()))
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string]bool{"no_text.png": false, "mask.png": false, "detections.png": false, "regions.json": false}
	for _, file := range archive.File {
		if _, ok := expected[file.Name]; !ok {
			t.Fatalf("unexpected ZIP entry %s", file.Name)
		}
		reader, err := file.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(reader)
		reader.Close()
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(file.Name, ".png") {
			im, err := png.Decode(bytes.NewReader(data))
			if err != nil || im.Bounds().Size() != image.Pt(16, 16) {
				t.Fatalf("invalid bundled PNG %s: %v", file.Name, err)
			}
		} else if err := json.Unmarshal(data, &metadata); err != nil || len(metadata.Regions) != 1 {
			t.Fatalf("invalid bundled metadata: %v", err)
		}
		expected[file.Name] = true
	}
	for name, found := range expected {
		if !found {
			t.Errorf("ZIP missing %s", name)
		}
	}
}

func TestAuthenticationAndPublicPages(t *testing.T) {
	p := &processor{}
	h := handler(t, p, httpapi.Options{APIKey: "test-key"})
	for _, path := range []string{"/", "/healthz"} {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, httptest.NewRequest("GET", path, nil))
		if rr.Code != 200 || rr.Body.Len() == 0 {
			t.Fatalf("public page %s failed", path)
		}
	}
	for _, path := range []string{"/v1/remove", "/v1/detect"} {
		for _, key := range []string{"", "Bearer wrong", "Bearer test-key"} {
			r := upload(t, path, imageBytes(t, "png", 16, 16))
			r.Header.Set("Authorization", key)
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, r)
			if key != "Bearer test-key" {
				checkError(t, rr, 401, "unauthorized")
			} else if rr.Code != 200 {
				t.Fatalf("authorized request failed: %d %s", rr.Code, rr.Body.String())
			}
		}
	}
	if p.calls != 2 {
		t.Fatalf("unauthorized requests reached processor: %d", p.calls)
	}
}

func TestRejectInvalidUploadsBeforeProcessing(t *testing.T) {
	p := &processor{}
	h := handler(t, p, httpapi.Options{MaxUploadBytes: 1024, MaxPixels: 256})
	valid := imageBytes(t, "png", 16, 16)
	unknownLength := upload(t, "/v1/remove", bytes.Repeat([]byte("x"), 2000))
	unknownLength.ContentLength = -1
	cases := []struct {
		name   string
		req    *http.Request
		code   string
		status int
	}{
		{"content-type", httptest.NewRequest("POST", "/v1/remove", strings.NewReader("hello")), "invalid_content_type", 415},
		{"missing", upload(t, "/v1/remove"), "missing_image", 400},
		{"duplicate", upload(t, "/v1/remove", valid, valid), "invalid_image_field", 400},
		{"not-image", upload(t, "/v1/remove", []byte("hello")), "unsupported_image", 415},
		{"pixels", upload(t, "/v1/remove", imageBytes(t, "png", 32, 32)), "too_many_pixels", 413},
		{"size", upload(t, "/v1/remove", bytes.Repeat([]byte("x"), 2000)), "upload_too_large", 413},
		{"streamed-size", unknownLength, "upload_too_large", 413},
		{"format", upload(t, "/v1/remove?format=exe", valid), "invalid_format", 400},
		{"detect-format", upload(t, "/v1/detect?format=zip", valid), "invalid_format", 400},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, tc.req)
			checkError(t, rr, tc.status, tc.code)
		})
	}
	if p.calls != 0 {
		t.Fatalf("invalid requests reached processor: %d", p.calls)
	}
}

func TestCapacityIsReleasedAndHealthRemainsAvailable(t *testing.T) {
	entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	p := &processor{remove: func(ctx context.Context, src image.Image) (*textinpaint.Result, error) {
		once.Do(func() { close(entered) })
		select {
		case <-release:
			return &textinpaint.Result{Image: white(16, 16)}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}
	h := handler(t, p, httpapi.Options{Timeout: time.Second})
	first := httptest.NewRecorder()
	request := upload(t, "/v1/remove", imageBytes(t, "png", 16, 16))
	go func() { defer close(finished); h.ServeHTTP(first, request) }()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("processor did not start")
	}
	busy := httptest.NewRecorder()
	h.ServeHTTP(busy, upload(t, "/v1/remove", imageBytes(t, "png", 16, 16)))
	checkError(t, busy, 503, "busy")
	if busy.Header().Get("Retry-After") == "" {
		t.Fatal("missing retry hint")
	}
	health := httptest.NewRecorder()
	h.ServeHTTP(health, httptest.NewRequest("GET", "/healthz", nil))
	if health.Code != 200 {
		t.Fatal("health blocked by image processing")
	}
	close(release)
	<-finished
	if first.Code != 200 {
		t.Fatalf("first request failed: %s", first.Body.String())
	}
	next := httptest.NewRecorder()
	h.ServeHTTP(next, upload(t, "/v1/remove", imageBytes(t, "png", 16, 16)))
	if next.Code != 200 {
		t.Fatalf("capacity leaked: %s", next.Body.String())
	}
}

func TestDeadlineCancellationAndInternalErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		fail   func(context.Context, image.Image) (*textinpaint.Result, error)
		status int
		code   string
	}{
		{"deadline", func(ctx context.Context, _ image.Image) (*textinpaint.Result, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}, 504, "processing_timeout"},
		{"failure", func(context.Context, image.Image) (*textinpaint.Result, error) {
			return nil, errors.New("private-server-path")
		}, 500, "processing_failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := handler(t, &processor{remove: tc.fail}, httpapi.Options{Timeout: 20 * time.Millisecond})
			for i := 0; i < 2; i++ {
				rr := httptest.NewRecorder()
				h.ServeHTTP(rr, upload(t, "/v1/remove", imageBytes(t, "png", 16, 16)))
				checkError(t, rr, tc.status, tc.code)
				if strings.Contains(rr.Body.String(), "private-server-path") {
					t.Fatal("internal details exposed")
				}
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	p := &processor{}
	h := handler(t, p, httpapi.Options{})
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, upload(t, "/v1/remove", imageBytes(t, "png", 16, 16)).WithContext(ctx))
	checkError(t, rr, 408, "request_canceled")
	if p.calls != 0 {
		t.Fatal("canceled request reached processor")
	}
}
