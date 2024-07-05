#include <opencv2/opencv.hpp>
#include <raspiVideoCapture.h>
//#include <proccessimage.h>
#include <detectorangeball.h>

int main(){
    DetectOrangeBall detectOrangeBall(threshold, lowColor, highColor);
    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        std::cerr << "Could not open camera" << std::endl;
        return -1;
    }

    while(true){
        cv::Mat frame;
        cap >> frame;

        if(frame.empty()){
            std::cerr << "Captured empty frame" << std::endl;
            break;
        }

        cv::Point centroid = detectOrangeBall.calculateCentroid(frame);

        if(centroid.x != -1 && centroid.y != -1){
            std::cout << "Centroid of the ball: (" << centroid.x << ", " << centroid.y << ")" << std::endl;
            cv::circle(frame, centroid, 5, cv::Scalar(0, 255, 0), -1);
        }else{
            std::cout << "No orange ball detected" << std::endl;
        }

        cv::imshow("Frame", frame);

        if(cv::waitKey(27) >= 0) 
            break;
    }

    return 0;
}
