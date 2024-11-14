#include "imageprocessor.h"

ImageProcessor::ImageProcessor(cv::MorphShapes shape, cv::Size size, int operation)
    : shape(shape), size(size), operation(operation){}

cv::Mat ImageProcessor::applyMorphology(const cv::Mat &frame){
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(shape, size);
    cv::morphologyEx(frame, result, operation, kernel);
    return result;
}