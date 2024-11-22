#ifndef CENTROIDCALCULATOR_H
#define CENTROIDCALCULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class CentroidCalculator{
    public:
        CentroidCalculator(double threshold=5e-3, int points=16);

        std::pair<cv::Point, std::vector<cv::Point>> calculateAll(const cv::Size& frameSize, 
            const std::vector<std::vector<cv::Point>>& contours);

    private:
        double threshold;
        int points;
        const std::vector<cv::Point>& findLargestContour(const std::vector<std::vector<cv::Point>>& contours);
        std::pair<cv::Point, float> calculateEnclosingCircle(const std::vector<cv::Point>& contour);
        std::vector<cv::Point> generateCircleContour(const cv::Point2f& center, float radius);
};

#endif // CENTROIDCALCULATOR_H
