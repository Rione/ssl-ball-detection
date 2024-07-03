import cv2
import numpy as np
from libcamera import Transform

class proccessImage:
    def applyClahe(frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) 
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8)) 
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0]) 
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) 

        return img

class detectBall:
    def extractSpecificColor(frame, lowColor, highColor):
        h, w, c = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowColor, highColor)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return hsv, contours
    
    def calculateCentroid(frame, contours, areaRatioThreshold):
        h, w, c = frame.shape
        areas = np.array(list(map(cv2.contourArea, contours)))

        if len(areas) == 0 or np.max(areas) / (h * w) < areaRatioThreshold:
            print("the area is too small")
            
            return None
        else:
            maxIdx = np.argmax(areas)
            result = cv2.moments(contours[maxIdx])
            x = int(result["m10"] / result["m00"])
            y = int(result["m01"] / result["m00"])

            return (x, y)

class display:
    def indicateCentroid(frame, pos):
        centroid = cv2.circle(frame, (pos[0], pos[1]), 10, (255, 0, 0), -1)
        cv2.imshow("Frame", centroid)

def main():
    lowColor = np.array([5, 150, 150])
    highColor = np.array([15, 255, 255])
    areaRatioThreshold = 0.005

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        pos = detectBall.calculateCentroid(frame, areaRatioThreshold, lowColor, highColor)
        display.indicateCentroid(frame, pos)

        print("centroid of ball:")
        print(pos)

        key = cv2.waitKey(1)

        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
