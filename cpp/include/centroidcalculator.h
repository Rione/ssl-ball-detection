#ifndef CENTROIDCALCULATOR_H
#define CENTROIDCALCULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class CentroidCalculator{
    public:
        CentroidCalculator(double threshold=0.005);
        std::pair<cv::Point, std::vector<cv::Point>> calculate(const cv::Size &frameSize, 
            const std::vector<std::vector<cv::Point>> &contours);

    private:
        double threshold;
};

#endif // CENTROIDCALCULATOR_H
