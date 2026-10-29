// Package reconstruct pairs Go text detections with caller-supplied OCR text,
// selectively removes confirmed text, and renders selectable HTML. It does not
// call a model service, start a server, or launch a CLI subprocess.
package reconstruct

import (
	"context"
	"image"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

// Detector is implemented by *textinpaint.Engine. Alternative implementations
// must return validated original-image polygons, including protected regions.
type Detector interface {
	DetectWithProtection(context.Context, image.Image, *textinpaint.ProcessingOptions) (*textinpaint.DetectionResult, error)
}

// Options are copied by Analyze; nil pointers select defaults. Processing=nil
// uses the detector's threshold. Rendering does not use or own the detector.
type Options struct {
	Processing    *textinpaint.ProcessingOptions
	Mask          *textinpaint.MaskOptions
	InpaintRadius float64 // 0 selects 3.
}

type Candidate struct {
	ID string `json:"id"`
	textinpaint.Region
}

// Analysis is safe to send to the external OCR caller. Its IDs only belong to
// this analysis. SourceSHA256 hashes the normalized PNG from OriginalPNG.
type Analysis struct {
	ID           string      `json:"analysis_id"`
	Width        int         `json:"width"`
	Height       int         `json:"height"`
	SourceSHA256 string      `json:"source_sha256"`
	Regions      []Candidate `json:"regions"`
}

type Status string

const (
	Text       Status = "text"
	Graphic    Status = "graphic"
	Unreadable Status = "unreadable"
)

type Run struct {
	Text string `json:"text"`
	Bold bool   `json:"bold,omitempty"`
}

// RecognizedRegion deliberately has no coordinates. Text and Runs are mutually
// exclusive. Color is an optional #RRGGBB override. SplitColors associates each
// run with a separate color block measured by Go, from left to right.
type RecognizedRegion struct {
	RegionID    string `json:"region_id"`
	Status      Status `json:"status"`
	Text        string `json:"text,omitempty"`
	Runs        []Run  `json:"runs,omitempty"`
	Bold        bool   `json:"bold,omitempty"`
	Role        string `json:"role,omitempty"` // text, heading, paragraph, list-item, caption
	Color       string `json:"color,omitempty"`
	SplitColors bool   `json:"split_colors,omitempty"`
}

type Recognition struct {
	AnalysisID   string             `json:"analysis_id"`
	Regions      []RecognizedRegion `json:"regions"`
	ReadingOrder []string           `json:"reading_order,omitempty"`
}

type RenderOptions struct {
	Title string
}

type Rect struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

type Span struct {
	Bounds Rect   `json:"ink_bounds"`
	Color  string `json:"color"`
	Runs   []Run  `json:"runs"`
}

type Line struct {
	RegionID string             `json:"region_id"`
	Text     string             `json:"text"`
	Role     string             `json:"role,omitempty"`
	Region   textinpaint.Region `json:"go_region"`
	Spans    []Span             `json:"spans"`
}

type Preserved struct {
	RegionID string `json:"region_id"`
	Reason   string `json:"reason"`
}

// Page contains metadata and encoded images. JSON excludes the image bytes;
// RenderHTML embeds them into a standalone document. Treat returned pages as
// immutable while rendering them concurrently.
type Page struct {
	AnalysisID    string      `json:"analysis_id"`
	Width         int         `json:"width"`
	Height        int         `json:"height"`
	SourceSHA256  string      `json:"source_sha256"`
	Lines         []Line      `json:"lines"`
	Preserved     []Preserved `json:"preserved"`
	OriginalPNG   []byte      `json:"-"`
	BackgroundPNG []byte      `json:"-"`
}

type Document struct {
	Title string `json:"title"`
	Pages []Page `json:"pages"`
}

type Result struct {
	Page       Page
	Background *image.RGBA
	Mask       *image.Gray
	HTML       []byte
	JSON       []byte // Document metadata; images are omitted.
}
