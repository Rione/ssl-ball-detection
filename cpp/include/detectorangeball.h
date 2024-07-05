#ifndef DETECTORANGEBALL_H
#define DETECTORANGEBALL_H

#include <opencv2/opencv.hpp>
#include <iostream>

class DetectOrangeBall
{
    public:
        DetectOrangeBall(
            cv::Scalar &lowColor = cv::Scalar(0, 150, 150), 
            cv::Scalar &highColor = cv::Scalar(15, 255, 255), 
            double threshold = 0.005
        );
        cv::Point calculateCentroid(const cv::Mat &frame);

    private:
        cv::Scalar lowColor;
        cv::Scalar highColor;
        double threshold;     
};

#endif //DETECTORANGEBALL_H