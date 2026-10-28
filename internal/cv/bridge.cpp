#include "bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace {
template<class F> int guarded(F fn, char* error, int capacity) {
    try { fn(); return 0; }
    catch (const std::exception& e) { std::snprintf(error, capacity, "%s", e.what()); }
    catch (...) { std::snprintf(error, capacity, "unknown OpenCV exception"); }
    return -1;
}
cv::Mat rgba(const unsigned char* data, int w, int h, int stride) {
    return cv::Mat(h, w, CV_8UC4, const_cast<unsigned char*>(data), stride);
}
std::vector<cv::Point2f> quad(const float* p) {
    std::vector<cv::Point2f> out;
    for (int i = 0; i < 4; ++i) out.emplace_back(p[2*i], p[2*i+1]);
    return out;
}
unsigned colorKey(const cv::Vec3b& p) {
    return (unsigned(p[0])<<16)|(unsigned(p[1])<<8)|p[2];
}
cv::Vec3b dominantColor(const cv::Mat& image, const cv::Mat& selection) {
    std::unordered_map<unsigned,int> histogram;
    unsigned best=0; int bestCount=0;
    for (int y=0; y<image.rows; ++y) for (int x=0; x<image.cols; ++x) {
        if (!selection.at<unsigned char>(y,x)) continue;
        unsigned key=colorKey(image.at<cv::Vec3b>(y,x));
        int count=++histogram[key];
        if (count>bestCount || (count==bestCount && key<best)) { best=key; bestCount=count; }
    }
    return cv::Vec3b((best>>16)&255,(best>>8)&255,best&255);
}
int colorDistance(cv::Vec3b a, cv::Vec3b b) {
    return std::max({std::abs(int(a[0])-b[0]),std::abs(int(a[1])-b[1]),std::abs(int(a[2])-b[2])});
}

// Fit a local color plane to the visible boundary. Robust fitting excludes a
// few antialiased remnants or a nearby rule, but rejects textured boundaries.
// Only masked pixels are written; photographs still fall back to Telea.
bool fillSmoothBackground(const cv::Mat& source, const cv::Mat& component,
    const cv::Mat& ring, cv::Mat destination) {
    std::vector<cv::Vec3d> positions, colors;
    for (int y=0; y<ring.rows; ++y) for (int x=0; x<ring.cols; ++x) {
        if (!ring.at<unsigned char>(y,x)) continue;
        positions.emplace_back(1.,double(x)/std::max(1,ring.cols-1),double(y)/std::max(1,ring.rows-1));
        auto c=source.at<cv::Vec3b>(y,x); colors.emplace_back(c[0],c[1],c[2]);
    }
    if (positions.size()<24) return false;
    std::vector<double> residuals(positions.size(),0.);
    cv::Matx33d coefficients;
    double cutoff=256.;
    for (int iteration=0; iteration<4; ++iteration) {
        cv::Matx33d normal=cv::Matx33d::zeros(), values=cv::Matx33d::zeros();
        for (size_t i=0; i<positions.size(); ++i) {
            if (residuals[i]>cutoff) continue;
            auto p=positions[i], c=colors[i];
            normal+=p*p.t(); values+=p*c.t();
        }
        if (!cv::solve(normal,values,coefficients,cv::DECOMP_SVD)) return false;
        for (size_t i=0; i<positions.size(); ++i) {
            auto difference=coefficients.t()*positions[i]-colors[i];
            residuals[i]=std::max({std::abs(difference[0]),std::abs(difference[1]),std::abs(difference[2])});
        }
        auto ordered=residuals;
        std::nth_element(ordered.begin(),ordered.begin()+ordered.size()/2,ordered.end());
        cutoff=std::max(3.,ordered[ordered.size()/2]*2.5);
    }
    size_t consistent=0;
    for (auto residual:residuals) if (residual<=4.) ++consistent;
    if (double(consistent)/positions.size()<.85) return false;
    for (int y=0; y<component.rows; ++y) for (int x=0; x<component.cols; ++x) {
        if (!component.at<unsigned char>(y,x)) continue;
        auto prediction=coefficients.t()*cv::Vec3d(1.,double(x)/std::max(1,ring.cols-1),double(y)/std::max(1,ring.rows-1));
        destination.at<cv::Vec3b>(y,x)=cv::Vec3b(cv::saturate_cast<unsigned char>(prediction[0]),
            cv::saturate_cast<unsigned char>(prediction[1]),cv::saturate_cast<unsigned char>(prediction[2]));
    }
    return true;
}

// Learn compact leading icons from their spacing and their color relative to
// the following text. The wider context also covers icons clipped by OCR boxes.
// Apply all protections after collecting all regions, independent of box order.
void protectLeadingIcons(const cv::Mat& bgr, const float* polygons, int count, cv::Mat& protection) {
    for (int i=0; i<count; ++i) {
        auto rotated=cv::minAreaRect(quad(polygons+i*8));
        cv::Rect text=cv::boundingRect(quad(polygons+i*8)) & cv::Rect(0,0,bgr.cols,bgr.rows);
        double height=std::min(rotated.size.width,rotated.size.height);
        if (height<12 || text.width<height*2.2 || text.height>height*1.4) continue;
        cv::Mat entire(text.size(),CV_8U,cv::Scalar(255));
        auto bg=dominantColor(bgr(text),entire);
        cv::Rect area(text.x-cvRound(4.5*height),text.y-cvRound(height*1.5),
            text.width+cvRound(4.5*height)+2,text.height+cvRound(height*3));
        area &= cv::Rect(0,0,bgr.cols,bgr.rows);
        cv::Mat patch=bgr(area), ink=cv::Mat::zeros(area.size(),CV_8U);
        std::vector<cv::Point> runs;
        for (int x=0; x<patch.cols; ++x) {
            bool occupied=false;
            for (int y=0; y<patch.rows; ++y) {
                if (colorDistance(patch.at<cv::Vec3b>(y,x),bg)>40) {
                    ink.at<unsigned char>(y,x)=255; occupied=true;
                }
            }
            if (!occupied) continue;
            int gap=std::max(2,cvRound(height*.1));
            if (runs.empty() || x-runs.back().y>gap) runs.emplace_back(x,x);
            else runs.back().y=x;
        }
        for (size_t k=0; k+1<runs.size(); ++k) {
            auto run=runs[k];
            int width=run.y-run.x+1, gap=runs[k+1].x-run.y-1;
            if (run.x+area.x>text.x+height || width<height*.25 || width>height*3 || gap<height*.28) continue;
            if (runs.back().y-runs[k+1].x<height*1.2) continue;
            cv::Rect candidate(run.x,0,width,patch.rows);
            std::vector<cv::Point> pixels; cv::findNonZero(ink(candidate),pixels);
            if (pixels.empty()) continue;
            auto bounds=cv::boundingRect(pixels);
            if (bounds.height<height*.65 || bounds.width<double(bounds.height)*.4) continue;
            if (std::abs(area.y+bounds.y+bounds.height*.5-text.y-text.height*.5)>height*.85) continue;
            auto iconColor=dominantColor(patch(candidate),ink(candidate));
            // Neutral characters (4, ON, kana) can have different antialiasing
            // from their neighbours; this alone is not evidence of an icon.
            if (std::max({iconColor[0],iconColor[1],iconColor[2]})-
                std::min({iconColor[0],iconColor[1],iconColor[2]})<40) continue;
            cv::Rect rest=text & area;
            rest.x-=area.x; rest.y-=area.y;
            auto textColor=dominantColor(patch(rest),ink(rest));
            if (colorDistance(iconColor,textColor)<32) continue;
            cv::Rect actual(area.x+run.x+bounds.x,area.y+bounds.y,bounds.width,bounds.height);
            bool independentText=false;
            for (int other=0; other<count; ++other) {
                if (other==i) continue;
                auto evidence=cv::boundingRect(quad(polygons+other*8));
                double overlap=double((actual & evidence).area())/actual.area();
                if (evidence.width>evidence.height*1.8 && overlap>.25) { independentText=true; break; }
                if (overlap<.6) continue;
                // Respect full-height detections of numbered bullets and
                // neighbouring labels. False icon detections commonly cover
                // only their internal strokes, not the complete icon.
                if ((evidence.height>=actual.height*.85 && actual.height<height*1.25) || evidence.width>evidence.height*1.8) {
                    independentText=true; break;
                }
            }
            if (independentText) continue;
            cv::Rect icon(area.x+run.x+bounds.x-2,area.y+bounds.y-2,bounds.width+4,bounds.height+4);
            icon &= cv::Rect(0,0,bgr.cols,bgr.rows);
            protection(icon).setTo(255);
        }
    }
}

cv::Mat graphicsProtection(const cv::Mat& bgr, const float* polygons, int count) {
    cv::Mat protectedPixels = cv::Mat::zeros(bgr.size(), CV_8U), gray, dark;
    cv::Mat protectedLines = cv::Mat::zeros(bgr.size(), CV_8U);
    cv::Mat textHeight=cv::Mat::zeros(bgr.size(),CV_32F);
    for (int i=0; i<count; ++i) {
        auto q=quad(polygons+i*8); auto rect=cv::minAreaRect(q);
        std::vector<cv::Point> points; for (auto p:q) points.emplace_back(cvRound(p.x),cvRound(p.y));
        cv::fillConvexPoly(textHeight,points,std::min(rect.size.width,rect.size.height));
    }
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::compare(gray, 185, dark, cv::CMP_LT);
    int length = std::max(48, std::min(bgr.rows, bgr.cols) / 30);
    for (auto shape : {cv::Size(length, 1), cv::Size(1, length)}) {
        cv::Mat lines, labels, stats, centers;
        cv::morphologyEx(dark, lines, cv::MORPH_OPEN, cv::Mat::ones(shape, CV_8U),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0));
        int n = cv::connectedComponentsWithStats(lines, labels, stats, centers);
        for (int i = 1; i < n; ++i) {
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (std::min(w,h)>5 || std::max(w,h)<length) continue;
            int x=stats.at<int>(i,cv::CC_STAT_LEFT), y=stats.at<int>(i,cv::CC_STAT_TOP);
            for (int yy=y; yy<y+h; ++yy) for (int xx=x; xx<x+w; ++xx) {
                // Long letter strokes are not document rules. Compare with
                // local text height rather than a fixed pixel threshold.
                if (labels.at<int>(yy,xx)==i && std::max(w,h)>=std::max(64.f,4*textHeight.at<float>(yy,xx)))
                    protectedLines.at<unsigned char>(yy,xx)=255;
            }
        }
    }
    // Preserve the antialiased edge and a narrow guard band around rules.
    // Otherwise mask expansion lets Telea spread the rule into removed text.
    cv::dilate(protectedLines,protectedPixels,cv::Mat::ones(3,3,CV_8U));
    cv::Mat hsv, color, labels, stats, centers;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 101, 111), cv::Scalar(180, 255, 255), color);
    int n = cv::connectedComponentsWithStats(color, labels, stats, centers);
    for (int i = 1; i < n; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        double aspect = double(w) / h;
        double fill=double(area)/(w*h);
        bool bar=std::min(w,h)<=24 && std::max(w,h)>=64 && std::max(aspect,1/aspect)>=6 && fill>.85;
        if (bar) {
            cv::Rect full(x-2,y-2,w+4,h+4);
            full &= cv::Rect(0,0,bgr.cols,bgr.rows);
            protectedPixels(full).setTo(255);
            continue;
        }
        if (w < 8 || h < 8 || w > 80 || h > 80 || aspect < .65 || aspect > 1.55) continue;
        cv::Rect r(x, y, w, h);
        cv::Mat component = labels(r) == i;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;
        auto largest = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b){ return cv::contourArea(a) < cv::contourArea(b); });
        std::vector<cv::Point> hull, corners;
        cv::convexHull(*largest, hull);
        cv::approxPolyDP(hull, corners, .06 * cv::arcLength(hull, true), true);
        double hue = cv::mean(hsv(r), component)[0];
        bool square = fill >= .85 && aspect >= .8 && aspect <= 1.25;
        double hullArea=std::max(1.,cv::contourArea(hull));
        bool triangle = hue >= 18 && hue <= 40 && corners.size() == 3 && fill > .25 && area/hullArea>.8;
        if (!square && !triangle) continue;
        // A colored hole inside a glyph (4, A, kana) has the same color as the
        // surrounding background. It must not be mistaken for a marker.
        cv::Rect surrounding(x-4,y-4,w+8,h+8);
        surrounding &= cv::Rect(0,0,bgr.cols,bgr.rows);
        cv::Mat rim(surrounding.size(),CV_8U,cv::Scalar(255));
        rim(cv::Rect(r.tl()-surrounding.tl(),r.size())).setTo(0);
        if (cv::countNonZero(rim) && colorDistance(dominantColor(bgr(r),component),dominantColor(bgr(surrounding),rim))<25) continue;
        for (auto& p : hull) p += cv::Point(x, y);
        cv::Mat marker = cv::Mat::zeros(bgr.size(), CV_8U);
        cv::fillConvexPoly(marker, hull, 255);
        if (triangle) cv::dilate(marker, marker, cv::Mat::ones(5, 5, CV_8U));
        protectedPixels |= marker;
    }
    protectLeadingIcons(bgr,polygons,count,protectedPixels);
    return protectedPixels;
}

cv::Mat textStrokes(const cv::Mat& bgr, cv::Rect bounds, const cv::Mat& region,
    double height, int contrast) {
    int size=std::clamp(cvRound(height*.7),3,81)|1;
    cv::Rect context(bounds.x-size,bounds.y-size,bounds.width+2*size,bounds.height+2*size);
    context &= cv::Rect(0,0,bgr.cols,bgr.rows);
    cv::Mat gray,background,response;
    cv::cvtColor(bgr(context),gray,cv::COLOR_BGR2GRAY);
    cv::Rect local(bounds.tl()-context.tl(),bounds.size());
    // Exact-color modes are unreliable on gradients: a repeated ink color can
    // outnumber every individual background shade. Use intensity quantiles to
    // decide whether the minority strokes are darker or lighter instead.
    int histogram[256]={0},total=0;
    for (int y=0; y<bounds.height; ++y) for (int x=0; x<bounds.width; ++x) {
        if (!region.at<unsigned char>(y,x)) continue;
        ++histogram[gray.at<unsigned char>(y+local.y,x+local.x)]; ++total;
    }
    auto quantile=[&](double fraction) {
        int accumulated=0;
        for (int v=0; v<256; ++v) { accumulated+=histogram[v]; if (accumulated>=total*fraction) return v; }
        return 255;
    };
    int median=quantile(.5);
    bool darkText=median-quantile(.05)>=quantile(.95)-median;
    auto kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(size,size));
    cv::morphologyEx(gray,background,darkText?cv::MORPH_CLOSE:cv::MORPH_OPEN,kernel);
    if (darkText) cv::subtract(background,gray,response);
    else cv::subtract(gray,background,response);
    cv::compare(response,contrast,response,cv::CMP_GT);
    cv::Mat inside=cv::Mat::zeros(context.size(),CV_8U);
    region.copyTo(inside(local));
    cv::Mat labels,stats,centers;
    int n=cv::connectedComponentsWithStats(response,labels,stats,centers);
    std::vector<int> selected(n,0);
    for (int y=local.y; y<local.y+local.height; ++y) for (int x=local.x; x<local.x+local.width; ++x)
        if (inside.at<unsigned char>(y,x)) ++selected[labels.at<int>(y,x)];
    std::vector<bool> keep(n,false);
    for (int i=1; i<n; ++i) {
        // Reject a device frame or other edge continuing well outside the box.
        int shortest=std::min(stats.at<int>(i,cv::CC_STAT_WIDTH),stats.at<int>(i,cv::CC_STAT_HEIGHT));
        int longest=std::max(stats.at<int>(i,cv::CC_STAT_WIDTH),stats.at<int>(i,cv::CC_STAT_HEIGHT));
        bool outsideRule=longest>shortest*8 && longest>height*3 && selected[i]<stats.at<int>(i,cv::CC_STAT_AREA)*.3;
        keep[i]=selected[i]>0 && !outsideRule;
    }
    cv::Mat strokes=cv::Mat::zeros(bounds.size(),CV_8U);
    for (int y=0; y<bounds.height; ++y) for (int x=0; x<bounds.width; ++x)
        if (region.at<unsigned char>(y,x) && keep[labels.at<int>(y+local.y,x+local.x)]) strokes.at<unsigned char>(y,x)=255;
    return strokes;
}
}

extern "C" int tp_prepare(const unsigned char* pixels, int w, int h, int stride,
    int x, int y, int cw, int ch, double scale, int dw, int dh, float* output, char* error, int cap) {
    return guarded([&] {
        cv::Mat bgr, scaled, resized;
        cv::cvtColor(rgba(pixels,w,h,stride)(cv::Rect(x,y,cw,ch)), bgr, cv::COLOR_RGBA2BGR);
        if (scale != 1.) cv::resize(bgr, scaled, cv::Size(), scale, scale, cv::INTER_CUBIC);
        else scaled = bgr;
        cv::resize(scaled, resized, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);
        for (int yy=0; yy<dh; ++yy) for (int xx=0; xx<dw; ++xx) {
            auto p = resized.at<cv::Vec3b>(yy,xx);
            for (int c=0; c<3; ++c) output[c*dw*dh+yy*dw+xx] = (p[c]*(1.f/255.f)-.5f)/.5f;
        }
    }, error, cap);
}

extern "C" int tp_boxes(const float* probability, int w, int h, int sourceW, int sourceH,
    float threshold, float boxThreshold, float unclip, int maximum, float* output, int* count, char* error, int cap) {
    return guarded([&] {
        *count = 0;
        cv::Mat map(h,w,CV_32F,const_cast<float*>(probability)), binary;
        cv::compare(map, threshold, binary, cv::CMP_GT);
        cv::dilate(binary,binary,cv::Mat::ones(2,2,CV_8U));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary,contours,cv::RETR_LIST,cv::CHAIN_APPROX_SIMPLE);
        for (size_t i=0; i<contours.size() && i<size_t(maximum); ++i) {
            auto rect = cv::minAreaRect(contours[i]);
            if (std::min(rect.size.width, rect.size.height) < 3) continue;
            cv::Point2f corners[4]; rect.points(corners);
            std::vector<cv::Point> ints;
            for (auto& p : corners) ints.emplace_back(int(p.x),int(p.y));
            cv::Rect bounds = cv::boundingRect(ints) & cv::Rect(0,0,w,h);
            if (bounds.empty()) continue;
            for (auto& p : ints) p -= bounds.tl();
            cv::Mat scoreMask = cv::Mat::zeros(bounds.size(),CV_8U);
            cv::fillConvexPoly(scoreMask,ints,1);
            float score = float(cv::mean(map(bounds),scoreMask)[0]);
            if (score < boxThreshold) continue;
            // DB rectangle expansion: area * unclip / perimeter. The minimum
            // rectangle of an offset rectangle expands by this distance on each side.
            float distance = rect.size.area()*unclip/(2*(rect.size.width+rect.size.height));
            rect.size.width += 2*distance; rect.size.height += 2*distance;
            if (std::min(rect.size.width,rect.size.height) < 5) continue;
            rect.points(corners);
            int at = (*count)++ * 9;
            for (int k=0;k<4;++k) {
                output[at+2*k] = std::clamp(float(std::round(corners[k].x/w*sourceW)),0.f,float(sourceW-1));
                output[at+2*k+1] = std::clamp(float(std::round(corners[k].y/h*sourceH)),0.f,float(sourceH-1));
            }
            output[at+8] = score;
        }
    },error,cap);
}

extern "C" int tp_graphic_panels(const unsigned char* pixels,int w,int h,int stride,
    int* output,int maximum,int* count,char* error,int cap) {
    return guarded([&] {
        *count=0;
        cv::Mat gray,dark;
        cv::cvtColor(rgba(pixels,w,h,stride),gray,cv::COLOR_RGBA2GRAY);
        cv::compare(gray,100,dark,cv::CMP_LT);
        cv::Mat originalDark=dark.clone();
        int gap=std::clamp(std::min(w,h)/100,3,21)|1;
        cv::morphologyEx(dark,dark,cv::MORPH_CLOSE,cv::Mat::ones(gap,gap,CV_8U));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dark,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour:contours) {
            auto r=cv::boundingRect(contour);
            if (r.width<96 || r.height<64 || r.width>r.height*6 || r.height>r.width*4 || r.area()>double(w)*h*.6) continue;
            // A closed frame has a near-rectangular outer contour and a mostly
            // non-dark interior. Filled panels and ordinary text fail this.
            if (cv::contourArea(contour)<r.area()*.82 || cv::countNonZero(originalDark(r))>r.area()*.4) continue;
            int band=std::max(3,std::min(r.width,r.height)/12);
            auto occupancy=[&](cv::Rect edge) {return double(cv::countNonZero(dark(edge)))/edge.area();};
            int sides=0;
            for (auto edge:{cv::Rect(r.x,r.y,r.width,band),cv::Rect(r.x,r.y+r.height-band,r.width,band),
                cv::Rect(r.x,r.y,band,r.height),cv::Rect(r.x+r.width-band,r.y,band,r.height)})
                if (occupancy(edge)>=.18) ++sides;
            // A callout arrow attached to one edge can widen the contour box.
            if (sides<3) continue;
            if (*count>=maximum) break;
            int at=(*count)++*4;
            output[at]=r.x;output[at+1]=r.y;output[at+2]=r.width;output[at+3]=r.height;
        }
    },error,cap);
}

extern "C" int tp_text_heights(const unsigned char* pixels,int w,int h,int stride,
    const float* polygons,int count,float* output,char* error,int cap) {
    return guarded([&] {
        cv::Mat bgr; cv::cvtColor(rgba(pixels,w,h,stride),bgr,cv::COLOR_RGBA2BGR);
        for (int i=0; i<count; ++i) {
            auto q=quad(polygons+i*8); auto rect=cv::minAreaRect(q);
            double height=std::min(rect.size.width,rect.size.height);
            auto bounds=cv::boundingRect(q)&cv::Rect(0,0,w,h);
            output[i]=0;
            if (bounds.empty()) continue;
            std::vector<cv::Point> poly;
            for (auto p:q) poly.emplace_back(cvRound(p.x)-bounds.x,cvRound(p.y)-bounds.y);
            cv::Mat region=cv::Mat::zeros(bounds.size(),CV_8U);
            cv::fillConvexPoly(region,poly,255);
            auto ink=textStrokes(bgr,bounds,region,height,13);
            cv::Mat labels,stats,centers;
            int n=cv::connectedComponentsWithStats(ink,labels,stats,centers);
            double angle=(rect.angle+(rect.size.width>=rect.size.height?0:90))*CV_PI/180;
            double nx=-std::sin(angle),ny=std::cos(angle);
            std::vector<double> low(n,1e30),high(n,-1e30);
            for (int y=0; y<ink.rows; ++y) for (int x=0; x<ink.cols; ++x) {
                int label=labels.at<int>(y,x); if (!label) continue;
                double t=nx*x+ny*y; low[label]=std::min(low[label],t); high[label]=std::max(high[label],t);
            }
            std::vector<std::pair<double,int>> sizes; int total=0;
            for (int j=1; j<n; ++j) {
                int area=stats.at<int>(j,cv::CC_STAT_AREA);
                int longest=std::max(stats.at<int>(j,cv::CC_STAT_WIDTH),stats.at<int>(j,cv::CC_STAT_HEIGHT));
                int shortest=std::min(stats.at<int>(j,cv::CC_STAT_WIDTH),stats.at<int>(j,cv::CC_STAT_HEIGHT));
                if (area<3 || longest>height*3 || longest>shortest*4) continue;
                sizes.emplace_back(high[j]-low[j]+1,area); total+=area;
            }
            std::sort(sizes.begin(),sizes.end()); int accumulated=0;
            for (auto size:sizes) {
                accumulated+=size.second;
                if (accumulated>=total*.6) { output[i]=float(size.first); break; }
            }
        }
    },error,cap);
}

extern "C" int tp_mask(const unsigned char* pixels, int w, int h, int stride,
    const float* polygons, int count, float padding, int contrast, int dilation, int protect, unsigned char* output, char* error, int cap) {
    return guarded([&] {
        cv::Mat bgr;
        cv::cvtColor(rgba(pixels,w,h,stride),bgr,cv::COLOR_RGBA2BGR);
        cv::Mat mask(h,w,CV_8U,output); mask.setTo(0);
        cv::Mat protectedPixels = protect ? graphicsProtection(bgr,polygons,count) : cv::Mat::zeros(bgr.size(),CV_8U);
        for (int i=0; i<count; ++i) {
            auto rect = cv::minAreaRect(quad(polygons+i*8));
            rect.size.width += 2*padding; rect.size.height += 2*padding;
            cv::Point2f corners[4]; rect.points(corners);
            std::vector<cv::Point> poly;
            for (auto p : corners) poly.emplace_back(std::clamp(cvRound(p.x),0,w-1),std::clamp(cvRound(p.y),0,h-1));
            cv::Rect bounds = cv::boundingRect(poly) & cv::Rect(0,0,w,h);
            if (bounds.empty()) continue;
            for (auto& p : poly) p -= bounds.tl();
            cv::Mat region = cv::Mat::zeros(bounds.size(),CV_8U);
            cv::fillConvexPoly(region,poly,255);
            // Direct foreground separation retains thin and connected strokes
            // on flat backgrounds. Use local morphology only when the box
            // contains several background colors; never fill the whole box.
            auto bg=dominantColor(bgr(bounds),region);
            cv::Mat glyphs=cv::Mat::zeros(bounds.size(),CV_8U);
            int flat=0,total=0;
            for (int y=0; y<bounds.height; ++y) for (int x=0; x<bounds.width; ++x) {
                if (!region.at<unsigned char>(y,x)) continue;
                int distance=colorDistance(bgr.at<cv::Vec3b>(bounds.y+y,bounds.x+x),bg);
                ++total; if (distance<=18) ++flat;
                if (distance>contrast) glyphs.at<unsigned char>(y,x)=255;
            }
            if (!total || double(flat)/total<.65)
                glyphs=textStrokes(bgr,bounds,region,std::min(rect.size.width,rect.size.height),contrast);
            mask(bounds) |= glyphs;
        }
        // Remove protected ink before expansion, otherwise the expansion alone
        // leaves a halo outside a preserved rule or icon.
        mask.setTo(0,protectedPixels);
        if (dilation>0) cv::dilate(mask,mask,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)),cv::Point(-1,-1),dilation);
        mask.setTo(0,protectedPixels);
    },error,cap);
}

extern "C" int tp_inpaint(const unsigned char* pixels,int w,int h,int stride,const unsigned char* mask,double radius,unsigned char* output,char* error,int cap) {
    return guarded([&] {
        cv::Mat bgr,repaired;
        cv::cvtColor(rgba(pixels,w,h,stride),bgr,cv::COLOR_RGBA2BGR);
        cv::Mat selection(h,w,CV_8U,const_cast<unsigned char*>(mask));
        cv::Mat dst(h,w,CV_8UC4,output);
        if (cv::countNonZero(selection)==0) rgba(pixels,w,h,stride).copyTo(dst);
        else {
            // Telea introduces color drift even on a perfectly uniform slide.
            // Fill a connected mask component directly only when its unmasked
            // boundary is confidently uniform. All other components use Telea.
            repaired=bgr.clone();
            cv::Mat remaining=selection.clone(), labels, stats, centers;
            int n=cv::connectedComponentsWithStats(selection,labels,stats,centers);
            for (int i=1; i<n; ++i) {
                cv::Rect bounds(stats.at<int>(i,cv::CC_STAT_LEFT)-4,stats.at<int>(i,cv::CC_STAT_TOP)-4,
                    stats.at<int>(i,cv::CC_STAT_WIDTH)+8,stats.at<int>(i,cv::CC_STAT_HEIGHT)+8);
                bounds &= cv::Rect(0,0,w,h);
                cv::Mat component=labels(bounds)==i, ring;
                cv::dilate(component,ring,cv::Mat::ones(5,5,CV_8U));
                ring.setTo(0,selection(bounds));
                int samples=cv::countNonZero(ring);
                if (samples<16) continue;
                auto bg=dominantColor(bgr(bounds),ring);
                int matching=0;
                for (int y=0; y<bounds.height; ++y) for (int x=0; x<bounds.width; ++x)
                    if (ring.at<unsigned char>(y,x) && colorDistance(bgr.at<cv::Vec3b>(bounds.y+y,bounds.x+x),bg)<=3) ++matching;
                if (double(matching)/samples>=.98) {
                    repaired(bounds).setTo(cv::Scalar(bg[0],bg[1],bg[2]),component);
                } else if (!fillSmoothBackground(bgr(bounds),component,ring,repaired(bounds))) continue;
                remaining(bounds).setTo(0,component);
            }
            if (cv::countNonZero(remaining)) {
                cv::Mat textured;
                cv::inpaint(repaired,remaining,textured,radius,cv::INPAINT_TELEA);
                repaired=textured;
            }
            cv::cvtColor(repaired,dst,cv::COLOR_BGR2RGBA);
        }
    },error,cap);
}

extern "C" int tp_overlay(const unsigned char* pixels,int w,int h,int stride,const float* polygons,int count,unsigned char* output,char* error,int cap) {
    return guarded([&] {
        cv::Mat dst(h,w,CV_8UC4,output); rgba(pixels,w,h,stride).copyTo(dst);
        for (int i=0;i<count;++i) {
            std::vector<cv::Point> points;
            for (auto p:quad(polygons+i*8)) points.emplace_back(cvRound(p.x),cvRound(p.y));
            cv::polylines(dst,points,true,cv::Scalar(255,30,90,255),2,cv::LINE_AA);
        }
    },error,cap);
}
