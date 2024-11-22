#ifndef VIDEOCAPTURE_H
#define VIDEOCAPTURE_H

#include <opencv2/opencv.hpp>

class VideoCapture{
    public:
        VideoCapture(int device=0, int fps=30, int bufferSize=4);
        bool setProperties();
        bool read(cv::Mat& frame);
        void release();
    
    private:
        cv::VideoCapture cap;
        int fps;
        int bufferSize;
};

#endif // VIDEOCAPTURE_H