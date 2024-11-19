#include "centroidcalculator.h"

#include "CentroidCalculator.h"

CentroidCalculator::CentroidCalculator(double threshold, int points)
    : threshold(threshold), points(points) {}

const std::vector<cv::Point> &CentroidCalculator::findLargestContour(const std::vector<std::vector<cv::Point>> &contours){
    return *std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
}

std::pair<cv::Point, cv::Point2f> CentroidCalculator::calculateCentroidAndCircle(const std::vector<cv::Point> &contour){
    cv::Moments moments = cv::moments(contour);
    cv::Point centroid(static_cast<int>(moments.m10 / moments.m00),
                        static_cast<int>(moments.m01 / moments.m00));

    cv::Point2f circleCenter;
    float radius;
    cv::minEnclosingCircle(contour, circleCenter, radius);

    return {centroid, circleCenter};
}

std::vector<cv::Point> CentroidCalculator::generateCircleContour(const cv::Point2f &center, float radius){
    std::vector<cv::Point> circleContour;
    for(int i = 0; i < points; ++i){
        double angle = 2.0 * CV_PI * i / points;
        circleContour.emplace_back(
            cv::Point(center.x + radius * std::cos(angle),
                      center.y + radius * std::sin(angle)));
    }
    return circleContour;
}

std::pair<cv::Point, std::vector<cv::Point>> CentroidCalculator::calculateAll(const cv::Size& frameSize, 
                                                                            const std::vector<std::vector<cv::Point>> &contours){
    if(contours.empty()){
        return {cv::Point(-1, -1), {}};
    }

    const auto &largestContour = findLargestContour(contours);

    double maxArea = cv::contourArea(largestContour);
    double frameArea = frameSize.width * frameSize.height;

    if(maxArea / frameArea < threshold){
        return {cv::Point(-1, -1), {}};
    }

    auto [centroid, circleCenter] = calculateCentroidAndCircle(largestContour);

    float radius;
    cv::minEnclosingCircle(largestContour, circleCenter, radius);

    std::vector<cv::Point> circleContour = generateCircleContour(circleCenter, radius);

    return {centroid, circleContour};
}