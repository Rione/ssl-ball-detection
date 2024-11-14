#ifndef COLORDETECTORBASE_H
#define COLORDETECTORBASE_H

#include <opencv2/opencv.hpp>

class ColorDetectorBase{
protected:
    static const cv::Scalar lowColor;
    static const cv::Scalar highColor;
};

#endif // COLORDETECTORBASE_H
