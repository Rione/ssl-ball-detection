#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>

class ImageProcessor{
    public:
        ImageProcessor(cv::MorphShapes shape=cv::MORPH_RECT, 
                    cv::Size size=cv::Size(3, 3), 
                    int operation=cv::MORPH_OPEN);
        cv::Mat applyMorphology(const cv::Mat &frame);

    private:
        cv::MorphShapes shape;
        cv::Size size;
        int operation;

};

#endif // IMAGEPROCESSOR_H