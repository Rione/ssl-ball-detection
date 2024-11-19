#include "imageprocessor.h"

ImageProcessor::ImageProcessor(int d, double sigmaColor, double sigmaSpace, 
                                cv::MorphShapes shape, cv::Size size, int operation, 
                                cv::Scalar lowColor, cv::Scalar highColor)
    : d(d), sigmaColor(sigmaColor), sigmaSpace(sigmaSpace), 
    shape(shape), size(size), operation(operation), 
    lowColor(lowColor), highColor(highColor) {}

cv::Mat ImageProcessor::filterFrame(const cv::Mat &frame){
    cv::Mat filteredFrame;
    cv::bilateralFilter(frame, filteredFrame, d, sigmaColor, sigmaSpace);
    return filteredFrame;
}

cv::Mat ImageProcessor::applyMorphologicalTransformations(const cv::Mat &frame){
    cv::Mat transformed;
    cv::Mat kernel = cv::getStructuringElement(shape, size);
    cv::morphologyEx(frame, transformed, operation, kernel);
    return transformed;
}

void ImageProcessor::extractColors(const cv::Mat &frame, cv::Mat &hsv, cv::Mat &masked){
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    hsv = applyMorphologicalTransformations(hsv);
    cv::inRange(hsv, lowColor, highColor, masked);
}