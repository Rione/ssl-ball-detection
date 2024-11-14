#include "centroidcalculator.h"
#include <opencv2/opencv.hpp>

CentroidCalculator::CentroidCalculator(double threshold):threshold(threshold){}

std::pair<cv::Point, std::vector<cv::Point>> CentroidCalculator::calculate(const cv::Size &frameSize, 
    const std::vector<std::vector<cv::Point>> &contours){
    int h = frameSize.height;
    int w = frameSize.width;

    double maxArea = 0;
    int maxIdx = -1;
    for(size_t i = 0; i < contours.size(); i++){
        double area = cv::contourArea(contours[i]);
        if(area > maxArea){
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }

    if(maxIdx == -1 || maxArea / (h * w) < threshold){
        return {cv::Point(-1, -1), {}};
    }

    cv::Moments m = cv::moments(contours[maxIdx]);
    cv::Point centroid(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

    std::vector<cv::Point> hull;
    cv::convexHull(contours[maxIdx], hull);

    return {centroid, hull};
}
