#ifndef BALLDETECTOR_H
#define BALLDETECTOR_H

#include "imageprocessor.h"
#include "centroidcalculator.h"

class BallDetector {
public:
    BallDetector(int mode = cv::RETR_EXTERNAL, int method = cv::CHAIN_APPROX_SIMPLE, double threshold = 5e-3);

    std::pair<cv::Point, std::vector<cv::Point>> detect(const cv::Mat &frame);

private:
    ImageProcessor imageProcessor;
    CentroidCalculator centroidCalculator;
    int mode;
    int method;
};

#endif
