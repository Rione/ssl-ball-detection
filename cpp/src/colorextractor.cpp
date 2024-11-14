#include "colorextractor.h"
#include <opencv2/opencv.hpp>

ColorExtractor::ColorExtractor(int mode, int method) : mode(mode), method(method){}

std::vector<std::vector<cv::Point>> ColorExtractor::extractOneColor(const cv::Mat &frame, cv::Mat &hsv){
    cv::Mat mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, lowColor, highColor, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, mode, method);
    return contours;
}
