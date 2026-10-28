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
cv::Mat graphicsProtection(const cv::Mat& bgr) {
    cv::Mat protectedPixels = cv::Mat::zeros(bgr.size(), CV_8U), gray, dark;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::compare(gray, 185, dark, cv::CMP_LT);
    int length = std::max(48, std::min(bgr.rows, bgr.cols) / 30);
    for (auto shape : {cv::Size(length, 1), cv::Size(1, length)}) {
        cv::Mat lines, labels, stats, centers;
        cv::morphologyEx(dark, lines, cv::MORPH_OPEN, cv::Mat::ones(shape, CV_8U));
        int n = cv::connectedComponentsWithStats(lines, labels, stats, centers);
        for (int i = 1; i < n; ++i) {
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (std::min(w, h) <= 5 && std::max(w, h) >= length)
                protectedPixels.setTo(255, labels == i);
        }
    }
    cv::Mat hsv, color, labels, stats, centers;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 101, 111), cv::Scalar(180, 255, 255), color);
    int n = cv::connectedComponentsWithStats(color, labels, stats, centers);
    for (int i = 1; i < n; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        double aspect = double(w) / h;
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
        double hue = cv::mean(hsv(r), component)[0], fill = double(area) / (w*h);
        bool square = fill >= .85 && aspect >= .8 && aspect <= 1.25;
        bool triangle = hue >= 18 && hue <= 40 && corners.size() == 3 && fill > .25;
        if (!square && !triangle) continue;
        for (auto& p : hull) p += cv::Point(x, y);
        cv::Mat marker = cv::Mat::zeros(bgr.size(), CV_8U);
        cv::fillConvexPoly(marker, hull, 255);
        if (triangle) cv::dilate(marker, marker, cv::Mat::ones(5, 5, CV_8U));
        protectedPixels |= marker;
    }
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
        cv::Mat protectedPixels = protect ? graphicsProtection(bgr) : cv::Mat::zeros(bgr.size(),CV_8U);
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
            if (protect) {
                int length=std::max(40,cvRound(std::min(rect.size.width,rect.size.height)*4));
                cv::Mat horizontal, vertical;
                cv::morphologyEx(glyphs,horizontal,cv::MORPH_OPEN,cv::Mat::ones(1,length,CV_8U));
                cv::morphologyEx(glyphs,vertical,cv::MORPH_OPEN,cv::Mat::ones(length,1,CV_8U));
                protectedPixels(bounds) |= horizontal | vertical;
            }
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
            cv::inpaint(bgr,selection,repaired,radius,cv::INPAINT_TELEA);
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
