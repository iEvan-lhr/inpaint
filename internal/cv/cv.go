// Package cv contains the small cgo bridge to OpenCV core, imgproc and photo.
package cv

/*
#cgo CXXFLAGS: -std=c++17
#cgo windows CXXFLAGS: -I${SRCDIR}/../../native/windows-amd64/include/opencv4
#cgo windows LDFLAGS: -L${SRCDIR}/../../native/windows-amd64/lib -lopencv_photo4120 -lopencv_imgproc4120 -lopencv_core4120 -lstdc++
#cgo linux darwin pkg-config: opencv4
#cgo linux LDFLAGS: -lstdc++
#cgo darwin LDFLAGS: -lc++
#include "bridge.h"
*/
import "C"
import (
	"fmt"
	"image"
	"unsafe"
)

func bytes(p []byte) *C.uchar {
	if len(p) == 0 {
		return nil
	}
	return (*C.uchar)(unsafe.Pointer(&p[0]))
}
func floats(p []float32) *C.float {
	if len(p) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&p[0]))
}
func errResult(code C.int, msg *[2048]C.char) error {
	if code != 0 {
		return fmt.Errorf("opencv: %s", C.GoString(&msg[0]))
	}
	return nil
}
func Prepare(im *image.RGBA, crop image.Rectangle, scale float64, w, h int) ([]float32, error) {
	out := make([]float32, 3*w*h)
	var msg [2048]C.char
	e := C.tp_prepare(bytes(im.Pix), C.int(im.Rect.Dx()), C.int(im.Rect.Dy()), C.int(im.Stride), C.int(crop.Min.X), C.int(crop.Min.Y), C.int(crop.Dx()), C.int(crop.Dy()), C.double(scale), C.int(w), C.int(h), floats(out), &msg[0], 2048)
	return out, errResult(e, &msg)
}
func Boxes(prob []float32, w, h, sw, sh int, threshold, score, unclip float32, maximum int) ([]float32, error) {
	out := make([]float32, maximum*9)
	var msg [2048]C.char
	var count C.int
	e := C.tp_boxes(floats(prob), C.int(w), C.int(h), C.int(sw), C.int(sh), C.float(threshold), C.float(score), C.float(unclip), C.int(maximum), floats(out), &count, &msg[0], 2048)
	if err := errResult(e, &msg); err != nil {
		return nil, err
	}
	return out[:int(count)*9], nil
}
func Mask(im *image.RGBA, polygons []float32, padding float64, contrast, dilation int, protect bool) (*image.Gray, error) {
	out := image.NewGray(im.Rect)
	var msg [2048]C.char
	var p C.int
	if protect {
		p = 1
	}
	e := C.tp_mask(bytes(im.Pix), C.int(im.Rect.Dx()), C.int(im.Rect.Dy()), C.int(im.Stride), floats(polygons), C.int(len(polygons)/8), C.float(padding), C.int(contrast), C.int(dilation), p, bytes(out.Pix), &msg[0], 2048)
	return out, errResult(e, &msg)
}
func Inpaint(im *image.RGBA, mask *image.Gray, radius float64) (*image.RGBA, error) {
	out := image.NewRGBA(im.Rect)
	var msg [2048]C.char
	e := C.tp_inpaint(bytes(im.Pix), C.int(im.Rect.Dx()), C.int(im.Rect.Dy()), C.int(im.Stride), bytes(mask.Pix), C.double(radius), bytes(out.Pix), &msg[0], 2048)
	return out, errResult(e, &msg)
}
func Overlay(im *image.RGBA, polygons []float32) (*image.RGBA, error) {
	out := image.NewRGBA(im.Rect)
	var msg [2048]C.char
	e := C.tp_overlay(bytes(im.Pix), C.int(im.Rect.Dx()), C.int(im.Rect.Dy()), C.int(im.Stride), floats(polygons), C.int(len(polygons)/8), bytes(out.Pix), &msg[0], 2048)
	return out, errResult(e, &msg)
}
