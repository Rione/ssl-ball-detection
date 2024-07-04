import cv2
import numpy as np
from libcamera import Transform

"""
class ProcessImage:
    def __init__(self, clipLimit = 2.0, tileGridSize = (8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def applyClahe(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV) 
        clahe = cv2.createCLAHE(clipLimit = self.clipLimit, tileGridSize = self.tileGridSize) 
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0]) 
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) 
        return rgb
"""

class DetectOrangeBall:
    def __init__(self, lowColor = np.array([0, 150, 150]), highColor = np.array([15, 255, 255]), threshold = 0.005):
        self.lowColor = lowColor
        self.highColor = highColor
        self.threshold = threshold

    def extractSpecificColor(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowColor, self.highColor)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return hsv, contours
    
    def calculateCentroid(self, frame, contours):
        h, w, _ = frame.shape
        areas = np.array(list(map(cv2.contourArea, contours)))

        if len(areas) == 0 or np.max(areas) / (h * w) < self.threshold:
            print("The area is too small")
            return None
        else:
            maxIdx = np.argmax(areas)
            result = cv2.moments(contours[maxIdx])
            x = int(result["m10"] / result["m00"])
            y = int(result["m01"] / result["m00"])
            return (x, y)

"""
class Display:
    def __init__(self, color = (255, 0, 0), radius = 5, windowName = "Frame"):
        self.color = color
        self.radius = radius
        self.windowName = windowName

    def indicateCentroid(self, frame, pos):
        if pos is not None:
            centroid = cv2.circle(frame, (pos[0], pos[1]), self.radius, self.color, -1)
            cv2.imshow(self.windowName, centroid)
        else:
            cv2.imshow(self.windowName, frame)
"""

def main():
    #processImage = ProcessImage()
    detectOrangeBall = DetectOrangeBall()
    #display = Display()

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Cannot load the image")
            break

        #frame = cv2.imread("ball_sample.png")
        #frame = processImage.applyClahe(frame)
        hsv, contours = detectOrangeBall.extractSpecificColor(frame)
        pos = detectOrangeBall.calculateCentroid(hsv, contours)
        print("centroid of ball:", pos)        
        #display.indicateCentroid(frame, pos)

        key = cv2.waitKey(1)
        if key == 27:
            break
        

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
