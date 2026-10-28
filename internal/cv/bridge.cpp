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
        cv::Rect area(text.x-cvRound(2*height),text.y-2,text.width+cvRound(2*height)+2,text.height+4);
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
            if (run.x+area.x>text.x+height || width<height*.25 || width>height*1.5 || gap<height*.28) continue;
            if (runs.back().y-runs[k+1].x<height*1.2) continue;
            cv::Rect candidate(run.x,0,width,patch.rows);
            std::vector<cv::Point> pixels; cv::findNonZero(ink(candidate),pixels);
            if (pixels.empty()) continue;
            auto bounds=cv::boundingRect(pixels);
            if (bounds.height<height*.65 || bounds.width<double(bounds.height)*.4) continue;
            cv::Rect rest(runs[k+1].x,0,patch.cols-runs[k+1].x,patch.rows);
            auto iconColor=dominantColor(patch(candidate),ink(candidate));
            auto textColor=dominantColor(patch(rest),ink(rest));
            if (colorDistance(iconColor,textColor)<32) continue;
            cv::Rect actual(area.x+run.x+bounds.x,area.y+bounds.y,bounds.width,bounds.height);
            bool independentText=false;
            for (int other=0; other<count; ++other) {
                if (other==i) continue;
                auto evidence=cv::boundingRect(quad(polygons+other*8));
                if (double((actual & evidence).area())/actual.area()<.6) continue;
                // Respect full-height detections of numbered bullets and
                // neighbouring labels. False icon detections commonly cover
                // only their internal strokes, not the complete icon.
                if (evidence.height>=actual.height*.85 || evidence.width>evidence.height*1.8) {
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
    cv::dilate(protectedLines,protectedPixels,cv::Mat::ones(7,7,CV_8U));
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
        for (auto& p : hull) p += cv::Point(x, y);
        cv::Mat marker = cv::Mat::zeros(bgr.size(), CV_8U);
        cv::fillConvexPoly(marker, hull, 255);
        if (triangle) cv::dilate(marker, marker, cv::Mat::ones(5, 5, CV_8U));
        protectedPixels |= marker;
    }
    protectLeadingIcons(bgr,polygons,count,protectedPixels);
    return protectedPixels;
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
            cv::Mat patch = bgr(bounds);
            std::unordered_map<unsigned,int> histogram;
            int total=0, bestCount=0; unsigned best=0;
            for (int yy=0; yy<patch.rows; ++yy) for (int xx=0; xx<patch.cols; ++xx) {
                if (!region.at<unsigned char>(yy,xx)) continue;
                auto p=patch.at<cv::Vec3b>(yy,xx);
                unsigned key=(unsigned(p[0])<<16)|(unsigned(p[1])<<8)|p[2];
                int c=++histogram[key]; ++total;
                if (c>bestCount || (c==bestCount && key<best)) {best=key;bestCount=c;}
            }
            if (!total) continue;
            int bg[3]={int((best>>16)&255),int((best>>8)&255),int(best&255)}, flat=0;
            cv::Mat glyphs = cv::Mat::zeros(bounds.size(),CV_8U);
            for (int yy=0; yy<patch.rows; ++yy) for (int xx=0; xx<patch.cols; ++xx) {
                if (!region.at<unsigned char>(yy,xx)) continue;
                auto p=patch.at<cv::Vec3b>(yy,xx);
                int d=std::max({std::abs(int(p[0])-bg[0]),std::abs(int(p[1])-bg[1]),std::abs(int(p[2])-bg[2])});
                if (d<=18) ++flat;
                if (d>contrast) glyphs.at<unsigned char>(yy,xx)=255;
            }
            if (double(flat)/total < .5) glyphs=region;
            mask(bounds) |= glyphs;
        }
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
                if (double(matching)/samples<.98) continue;
                repaired(bounds).setTo(cv::Scalar(bg[0],bg[1],bg[2]),component);
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
