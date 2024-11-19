#include "videocapture.h"

VideoCapture::VideoCapture(int device, int fps, int bufferSize)
    :cap(device), fps(fps), bufferSize(bufferSize) {}

bool VideoCapture::setProperties(){
    bool success = true;
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, bufferSize);
    return success;
}

bool VideoCapture::read(cv::Mat &frame){
    return cap.read(frame);
}

void VideoCapture::release(){
    cap.release();
}