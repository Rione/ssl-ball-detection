#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessor{
    public:
        ImageProcessor(int d=5, double sigmaColor=75, double sigmaSpace=75, 
                        cv::MorphShapes shape=cv::MORPH_RECT, 
                        cv::Size size=cv::Size(3, 3), 
                        int operation=cv::MORPH_OPEN, 
                        cv::Scalar lowColor=cv::Scalar(5, 80, 60),
                        cv::Scalar highColor=cv::Scalar(25, 255, 255));
        
        cv:: Mat filterFrame(const cv::Mat &frame);
        cv::Mat applyMorphologicalTransformations(const cv::Mat &frame);
        void extractColors(const cv::Mat &frame, cv::Mat &hsv, cv::Mat &mask);

    private:
        int d;
        double sigmaColor;
        double sigmaSpace;
        cv::MorphShapes shape;
        cv::Size size;
        int operation;
        cv::Scalar lowColor;
        cv::Scalar highColor;
};

#endif // IMAGEPROCESSOR_H