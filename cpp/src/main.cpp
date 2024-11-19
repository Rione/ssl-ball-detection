#include "balldetector.h"
#include "imagedisplayer.h"
#include "videocapture.h"
#include <raspiVideoCapture.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int main(){
    BallDetector ballDetector;
    ImageDisplayer imageDisplayer;
    VideoCapture videoCapture;

    if(!videoCapture.setProperties()){
        printf("Error: Unable to set video capture properties\n");
        return -1;
    }

    while(true){
        cv::Mat frame;
        videoCapture.read(frame);
        if(!videoCapture.read(frame)){
            printf("Error: Unable to load the image\n");
            break;
        }

        auto [center, ballContour] = ballDetector.detect(frame);
        std::cout << "Centroid of the ball: " << center << std::endl;
        imageDisplayer.indicateCentroid(frame, center, ballContour);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = (end - start);
        std::cout << "Time: " << duration.count() << std::endl;

        char key = (char)cv::waitKey(1);
        if(key == 'q'){
            break;
        }
    }
q   videoCapture.release();
    cv::destroyAllWindows();

    return 0;
}

