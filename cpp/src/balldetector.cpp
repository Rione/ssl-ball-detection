#include "balldetector.h"

BallDetector::BallDetector(int mode, int method, double threshold)
    : mode(mode), method(method), centroidCalculator(CentroidCalculator(threshold)) {}

std::pair<cv::Point, std::vector<cv::Point>> BallDetector::detect(const cv::Mat &frame) {
    cv::Mat hsv, mask;
    imageProcessor.extractColors(frame, hsv, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, mode, method);

    return centroidCalculator.calculateAll(hsv.size(), contours);
}