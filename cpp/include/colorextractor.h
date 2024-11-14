#ifndef COLOREXTRACTOR_H
#define COLOREXTRACTOR_H

#include "colordetectorbase.h"
#include <vector>

class ColorExtractor : public ColorDetectorBase{
    public:
        ColorExtractor(int mode=cv::RETR_EXTERNAL, int method=cv::CHAIN_APPROX_SIMPLE);
        std::vector<std::vector<cv::Point>> extractOneColor(const cv::Mat &frame, cv::Mat &hsv);

    private:
        int mode;
        int method;
};

#endif // COLOREXTRACTOR_H
