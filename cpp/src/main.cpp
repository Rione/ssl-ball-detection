#include "imageprocessor.h"
#include "colorextractor.h"
#include "centroidcalculator.h"
#include "imagedisplayer.h"
#include "videocapture.h"
#include <raspiVideoCapture.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int main(){
    ImageProcessor imageProcessor;
    ColorExtractor colorExtractor;
    CentroidCalculator centroidCalculator;
    ImageDisplayer imageDisplayer;
    VideoCapture videoCapture;

    cv::Mat frame;
    //videoCapture.setProperty();
    while(true){
        videoCapture.read(frame);
        if(frame.empty()){
            printf("Error: Cannot load the image\n");
            break;
        }

        frame = imageProcessor.applyMorphology(frame);
        cv::Mat hsv;
        std::vector<std::vector<cv::Point>> contours = colorExtractor.extractOneColor(frame, hsv);
        auto [pos, hull] = centroidCalculator.calculate(frame.size(), contours);
        std::cout << "Centroid of the ball: " << pos << std::endl;
        imageDisplayer.indicateCentroid(frame, pos, hull);

        char key = (char)cv::waitKey(1);
        if(key == 'q'){
            break;
        }
    }

    videoCapture.release();
    cv::destroyAllWindows();

    return 0;
}
