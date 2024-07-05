#include <detectorangeball.h>
#include <iostream>

DetectOrangeBall::detectOrangeBall(double threshold, const cv::Scalar &lowColor, const cv::Scalar &highColor)
    : threshold(threshold), lowColor(lowColor), highColor(highColor){}

cv::Point detectOrangeBall::calculateCentroid(const cv::Mat &frame){
    int h, w = frame.rows, frame.cols;

    //RGB=>HSV
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    //make a mask 
    cv::Mat mask;
    cv::inRange(hsv, lowColor, highColor, mask);

    //extract orange
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    //calculate the largest orange area
    std::vector<double> areas(contours.size());
    for(size_t i = 0; i < contours.size(); ++i){
        areas[i] = cv::contourArea(contours[i]);
    }

    //if the orange area is too small
    if(areas.empty() || *std::max_element(areas.begin(), areas.end()) / (h * w) < threshold){
        std::cout << "The area is too small" << std::endl;

        return cv::Point(-1, -1);
    } 
    else{
        //calculate the centroid of the area
        auto maxIt = std::max_element(areas.begin(), areas.end());
        int maxIdx = std::distance(areas.begin(), maxIt);
        cv::Moments result = cv::moments(contours[maxIdx]);
        int x = static_cast<int>(result.m10 / result.m00);
        int y = static_cast<int>(result.m01 / result.m00);

        return cv::Point(x, y);
    }
}