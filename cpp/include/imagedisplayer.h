#ifndef IMAGEDISPLAYER_H
#define IMAGEDISPLAYER_H

#include <opencv2/opencv.hpp>
#include <vector>

class ImageDisplayer{
    public:
        ImageDisplayer(cv::Scalar color=cv::Scalar(255, 0, 0), 
                        int radius=5, 
                        const std::string& windowName="Frame");

        void indicateCentroid(cv::Mat& frame, const cv::Point& center, const std::vector<cv::Point>& circle);

    private:
        cv::Scalar color;
        int radius;             
        std::string windowName;
};

#endif // IMAGEDISPLAYER_H
