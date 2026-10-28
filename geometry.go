package textinpaint

import (
	"math"
	"sort"
)

func cross(a, b, c Point) float64 { return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X) }
func signedArea(p []Point) float64 {
	v := 0.
	for i, a := range p {
		b := p[(i+1)%len(p)]
		v += a.X*b.Y - b.X*a.Y
	}
	return v / 2
}
func polygonArea(p []Point) float64 { return math.Abs(signedArea(p)) }
func convex(q Quad) bool {
	sign := 0.
	for i := 0; i < 4; i++ {
		v := cross(q[i], q[(i+1)%4], q[(i+2)%4])
		if math.Abs(v) < 1e-8 {
			return false
		}
		if sign*v < 0 {
			return false
		}
		sign = v
	}
	return true
}
func orderQuad(q Quad) Quad {
	center := Point{}
	for _, p := range q {
		center.X += p.X / 4
		center.Y += p.Y / 4
	}
	sort.Slice(q[:], func(i, j int) bool {
		return math.Atan2(q[i].Y-center.Y, q[i].X-center.X) < math.Atan2(q[j].Y-center.Y, q[j].X-center.X)
	})
	first := 0
	for i := 1; i < 4; i++ {
		if q[i].X+q[i].Y < q[first].X+q[first].Y {
			first = i
		}
	}
	out := Quad{}
	for i := 0; i < 4; i++ {
		out[i] = q[(first+i)%4]
	}
	return out
}
func intersectionArea(a, b Quad) float64 {
	poly := append([]Point(nil), a[:]...)
	direction := 1.
	if signedArea(b[:]) < 0 {
		direction = -1
	}
	for i := 0; i < 4 && len(poly) > 0; i++ {
		edgeA, edgeB := b[i], b[(i+1)%4]
		input := poly
		poly = nil
		prev := input[len(input)-1]
		prevD := direction * cross(edgeA, edgeB, prev)
		for _, cur := range input {
			curD := direction * cross(edgeA, edgeB, cur)
			if (curD >= -1e-7) != (prevD >= -1e-7) {
				t := prevD / (prevD - curD)
				poly = append(poly, Point{prev.X + t*(cur.X-prev.X), prev.Y + t*(cur.Y-prev.Y)})
			}
			if curD >= -1e-7 {
				poly = append(poly, cur)
			}
			prev, prevD = cur, curD
		}
	}
	return polygonArea(poly)
}
func deduplicate(regions []Region) []Region {
	sort.SliceStable(regions, func(i, j int) bool { return polygonArea(regions[i].Polygon[:]) > polygonArea(regions[j].Polygon[:]) })
	kept := make([]Region, 0, len(regions))
	for _, r := range regions {
		area := polygonArea(r.Polygon[:])
		if area < 8 || !convex(r.Polygon) {
			continue
		}
		duplicate := false
		for _, k := range kept {
			if intersectionArea(r.Polygon, k.Polygon)/area > .65 {
				duplicate = true
				break
			}
		}
		if !duplicate {
			kept = append(kept, r)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool {
		a, b := kept[i].Polygon[0], kept[j].Polygon[0]
		if a.Y == b.Y {
			return a.X < b.X
		}
		return a.Y < b.Y
	})
	return kept
}
func tileStarts(length, size int) []int {
	if length <= size {
		return []int{0}
	}
	n := int(math.Ceil(float64(length-size)/(float64(size)*.8))) + 1
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, int(math.Round(float64(i*(length-size))/float64(n-1))))
	}
	return out
}
