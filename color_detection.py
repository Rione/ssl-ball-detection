import cv2
import numpy as np
from libcamera import Transform

class proccessImage:
    def applyClahe(frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) 
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8)) 
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0]) 
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) 

        return rgb

class detectBall:
    lowColor = np.array([5, 150, 150])
    highColor = np.array([15, 255, 255])
    threshold = 0.005

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
        if pos is not None:
            centroid = cv2.circle(frame, (pos[0], pos[1]), 10, (255, 0, 0), -1)
            cv2.imshow("Frame", centroid)
        else:
            cv2.imshow("Frame", frame)
        
def main():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        #frame = cv2.imread("ball_sample.png")
        hsv, contours = detectBall.extractSpecificColor(frame, detectBall.lowColor, detectBall.highColor)
        pos = detectBall.calculateCentroid(hsv, contours, detectBall.threshold)
        print("centroid of ball:")
        print(pos)
        """
        display.indicateCentroid(frame, pos)

        key = cv2.waitKey(1)
        if key == 27:
            break
        """

    camera.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
