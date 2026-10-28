// A standard-library-only HTTP client. It does not need OpenCV, ONNX or cgo.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	input := flag.String("input", "", "input PNG/JPEG file")
	output := flag.String("output", "no_text.png", "output PNG file")
	endpoint := flag.String("url", "http://127.0.0.1:8080/v1/remove", "textinpaint remove endpoint")
	flag.Parse()
	if *input == "" {
		return fmt.Errorf("-input is required")
	}
	file, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer file.Close()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("image", filepath.Base(*input))
	if err != nil {
		return err
	}
	if _, err = io.Copy(part, file); err != nil {
		return err
	}
	if err = form.Close(); err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, *endpoint, &body)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", form.FormDataContentType())
	if key := os.Getenv("TEXTINPAINT_API_KEY"); key != "" {
		request.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 3 * time.Minute}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("server returned HTTP %d: %s", response.StatusCode, message)
	}
	if response.Header.Get("Content-Type") != "image/png" {
		return fmt.Errorf("expected image/png, got %q", response.Header.Get("Content-Type"))
	}
	out, err := os.Create(*output)
	if err != nil {
		return err
	}
	if _, err = io.Copy(out, response.Body); err != nil {
		out.Close()
		return err
	}
	if err = out.Close(); err != nil {
		return err
	}
	fmt.Printf("saved %s; regions=%s; processing=%sms\n", *output, response.Header.Get("X-Region-Count"), response.Header.Get("X-Processing-Ms"))
	return nil
}
