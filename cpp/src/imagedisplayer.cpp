#include "imagedisplayer.h"
#include <opencv2/opencv.hpp>

ImageDisplayer::ImageDisplayer(cv::Scalar color, int radius, const std::string &windowName)
    : color(color), radius(radius), windowName(windowName){}

void ImageDisplayer::indicateCentroid(cv::Mat &frame, const cv::Point &pos, const std::vector<cv::Point> &hull){
    if(pos != cv::Point(-1, -1)){
        cv::drawContours(frame, std::vector<std::vector<cv::Point>>{hull}, -1, color, 2);
        cv::circle(frame, pos, radius, color, -1);
    }
    cv::imshow(windowName, frame);
}