#ifndef PROCCESSIMAGE_H
#define PROCCESSIMAGE_H

#include <opencv2/opencv.hpp>
#include <string>

class ProccessImage
{
    public:
    proccessImage(const std::string &frame):frame(frame){}
        void applyClahe();
    
    private:
        std::string frame;

};

#endif // PROCESSIMAGE_H