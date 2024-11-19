#include "imagedisplayer.h"

ImageDisplayer::ImageDisplayer(cv::Scalar color, int radius, const std::string &windowName)
    : color(color), radius(radius), windowName(windowName) {}

void ImageDisplayer::indicateCentroid(cv::Mat &frame, const cv::Point &center, const std::vector<cv::Point> &circle){
    if(center.x >= 0 && center.y >= 0){
        cv::drawContours(frame, std::vector<std::vector<cv::Point>>{circle}, -1, color, 2);
        cv::circle(frame, center, radius, color, -1);
    }
    cv::imshow(windowName, frame);
}