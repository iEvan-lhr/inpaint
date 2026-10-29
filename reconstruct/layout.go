package reconstruct

import (
	"fmt"
	"image"
	"image/color"
	"sort"
	"strings"

	textinpaint "github.com/iEvan-lhr/inpaint"
)

type pixel struct {
	X, Y  int
	Color color.NRGBA
}

func rgba(c color.Color) color.NRGBA { return color.NRGBAModel.Convert(c).(color.NRGBA) }
func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
func distance(a, b color.NRGBA) int {
	return max(abs(int(a.R)-int(b.R)), abs(int(a.G)-int(b.G)), abs(int(a.B)-int(b.B)))
}

func makeSpans(original, background image.Image, mask *image.Gray, region textinpaint.Region, a RecognizedRegion) []Span {
	pixels := inkPixels(original, background, mask, region)
	if len(pixels) == 0 {
		return nil
	}
	runs := append([]Run(nil), a.Runs...)
	if len(runs) == 0 {
		runs = []Run{{Text: a.Text, Bold: a.Bold}}
	}
	var spans []Span
	if a.SplitColors && len(runs) > 1 {
		groups := colorGroups(pixels, len(runs))
		if len(groups) != len(runs) {
			return nil
		}
		for _, g := range groups {
			if len(g) == 0 {
				return nil
			}
		}
		sort.Slice(groups, func(i, j int) bool { return bounds(groups[i]).X < bounds(groups[j]).X })
		for i, g := range groups {
			spans = append(spans, newSpan(g, []Run{runs[i]}))
		}
	} else if len(runs) == 1 && strings.HasPrefix(runs[0].Text, "• ") {
		bullet, body := splitBullet(pixels)
		if len(bullet) > 0 && len(body) > 0 {
			spans = []Span{newSpan(bullet, []Run{{Text: "•", Bold: runs[0].Bold}}), newSpan(body, []Run{{Text: strings.TrimPrefix(runs[0].Text, "• "), Bold: runs[0].Bold}})}
		} else {
			spans = []Span{newSpan(pixels, runs)}
		}
	} else {
		spans = []Span{newSpan(pixels, runs)}
	}
	if a.Color != "" {
		for i := range spans {
			spans[i].Color = a.Color
		}
	}
	return spans
}

func inkPixels(original, background image.Image, mask *image.Gray, r textinpaint.Region) []pixel {
	rect := polygonBounds(r.Polygon).Inset(-2).Intersect(original.Bounds())
	var pixels []pixel
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			a, b := rgba(original.At(x, y)), rgba(background.At(x, y))
			if mask.GrayAt(x, y).Y > 0 && distance(a, b) >= 30 {
				pixels = append(pixels, pixel{x, y, a})
			}
		}
	}
	return dominantRows(pixels)
}

// Keep the dominant ink band when detector rounding includes a neighboring row.
func dominantRows(p []pixel) []pixel {
	if len(p) == 0 {
		return p
	}
	r := bounds(p)
	rows := make([]int, r.Height)
	for _, v := range p {
		rows[v.Y-r.Y]++
	}
	bestStart, bestEnd, bestCount := 0, len(rows)-1, 0
	for start := 0; start < len(rows); {
		if rows[start] == 0 {
			start++
			continue
		}
		last, count := start, 0
		for y := start; y < len(rows) && y-last <= 3; y++ {
			if rows[y] > 0 {
				last = y
				count += rows[y]
			}
		}
		if count > bestCount {
			bestStart, bestEnd, bestCount = start, last, count
		}
		start = last + 1
	}
	var keep []pixel
	for _, v := range p {
		if v.Y >= r.Y+bestStart && v.Y <= r.Y+bestEnd {
			keep = append(keep, v)
		}
	}
	return keep
}

func bounds(p []pixel) Rect {
	x0, y0, x1, y1 := p[0].X, p[0].Y, p[0].X, p[0].Y
	for _, v := range p {
		x0 = min(x0, v.X)
		y0 = min(y0, v.Y)
		x1 = max(x1, v.X)
		y1 = max(y1, v.Y)
	}
	return Rect{x0, y0, x1 - x0 + 1, y1 - y0 + 1}
}

type colorCount struct {
	Color color.NRGBA
	Count int
}

func colors(p []pixel) []colorCount {
	m := map[color.NRGBA]int{}
	for _, v := range p {
		m[v.Color]++
	}
	var out []colorCount
	for c, n := range m {
		out = append(out, colorCount{c, n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count == out[j].Count {
			return rgbKey(out[i].Color) < rgbKey(out[j].Color)
		}
		return out[i].Count > out[j].Count
	})
	return out
}
func rgbKey(c color.NRGBA) int { return int(c.R)<<16 | int(c.G)<<8 | int(c.B) }
func colorGroups(p []pixel, n int) [][]pixel {
	var palette []color.NRGBA
	for _, c := range colors(p) {
		distinct := true
		for _, v := range palette {
			if distance(c.Color, v) < 45 {
				distinct = false
			}
		}
		if distinct {
			palette = append(palette, c.Color)
		}
		if len(palette) == n {
			break
		}
	}
	groups := make([][]pixel, len(palette))
	for _, v := range p {
		best, d := 0, 256
		for i, c := range palette {
			if q := distance(v.Color, c); q < d {
				best, d = i, q
			}
		}
		if d <= 45 {
			groups[best] = append(groups[best], v)
		}
	}
	return groups
}
func newSpan(p []pixel, runs []Run) Span {
	c := colors(p)[0].Color
	var core []pixel
	for _, v := range p {
		if distance(v.Color, c) <= 8 {
			core = append(core, v)
		}
	}
	if len(core) > 0 {
		p = core
	}
	return Span{Bounds: bounds(p), Color: fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B), Runs: append([]Run(nil), runs...)}
}
func splitBullet(p []pixel) ([]pixel, []pixel) {
	r := bounds(p)
	cols := make([]int, r.Width)
	for _, v := range p {
		cols[v.X-r.X]++
	}
	end := 0
	for end < len(cols) && cols[end] > 0 {
		end++
	}
	start := end
	for start < len(cols) && cols[start] == 0 {
		start++
	}
	if start == len(cols) || start <= end {
		return nil, nil
	}
	// Avoid splitting the first letter when the OCR text contains a bullet but
	// the original image does not have a compact separate marker.
	if end > max(3, r.Height/2) || start-end < 2 {
		return nil, nil
	}
	cut := r.X + (end+start)/2
	var a, b []pixel
	for _, v := range p {
		if v.X < cut {
			a = append(a, v)
		} else {
			b = append(b, v)
		}
	}
	return a, b
}
