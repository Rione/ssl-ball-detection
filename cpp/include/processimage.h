#ifndef PROCESSIMAGE_H
#define PROCESSIMAGE_H

#include <opencv2/opencv.hpp>
#include <string>

class ProcessImage
{
    public:
    processImage(const std::string &frame):frame(frame){}
        void applyClahe();
    
    private:
        std::string frame;

};

#endif // PROCESSIMAGE_H