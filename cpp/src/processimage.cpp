#include <proccessimage.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

processImage::processImage(const std::string &frame):frame(frame){}

void processImage::applyClane(){
    //read the image
    cv::Mat img = cv::imread(frame);

    if(img.empty()){
        std::cerr << "Error:Couldnot open the image" << std::endl;
        return;
    }

    //RGB=>YUV
    cv::Mat img_yuv;
    cv::cvtColor(img, img_yuv, cv::COLOR_BGR2YUV);

    //make clahe object
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    //apply clahe to Y channel
    std::vector<cv::Mat> yuv_channels;
    cv::split(img_yuv, yuv_channels); 
    clahe->apply(yuv_channels[0], yuv_channels[0]);
    cv::merge(yuv_channels, img_yuv);

    //YUV=>RGB
    cv::cvtColor(img_yuv, img, cv::COLOR_YUV2BGR);

    return img;
}