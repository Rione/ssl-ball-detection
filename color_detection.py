import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from libcamera import Transform

"""
class ProcessImage:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def applyClahe(self, frame):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return rgb
"""

class DetectOrangeBall:
    def __init__(self, lowColor, highColor, threshold):
        self.lowColor = lowColor
        self.highColor = highColor
        self.threshold = threshold

    def extractSpecificColor(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowColor, self.highColor)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return hsv, contours

    def interpolateContours(self, contour):
        contour = contour.reshape(-1, 2)
        x, y = contour[:, 0], contour[:, 1]
        tck, u = splprep([x, y], s=0)
        uNew = np.linspace(u.min(), u.max(), 1000)
        xNew, yNew = splev(uNew, tck, der=0)

        interpolatedContour = np.vstack((xNew, yNew)).T.astype(np.int32).reshape(-1, 1, 2)
        return interpolatedContour

    def calculateCentroid(self, frameShape, contours):
        h, w = frameShape[:2]
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])

        if len(areas) == 0 or np.max(areas) / (h * w) < self.threshold:
            return None, None
        else:
            maxIdx = np.argmax(areas)
            contour = contours[maxIdx]

            hull = cv2.convexHull(contour)
            hull = self.interpolateContours(hull)
            area = cv2.contourArea(hull)

            if area / (h * w) >= self.threshold:
                result = cv2.moments(hull)
                x = int(result["m10"] / result["m00"])
                y = int(result["m01"] / result["m00"])
                return (x, y), hull
            else:
                return None, None

class Display:
    def __init__(self, color=(255, 0, 0), radius=5, windowName="Frame"):
        self.color = color
        self.radius = radius
        self.windowName = windowName

    def indicateCentroid(self, frame, pos, hull):
        if pos is not None:
            centroid = cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            centroid = cv2.circle(frame, (pos[0], pos[1]), self.radius, self.color, -1)
            cv2.imshow(self.windowName, centroid)
        else:
            cv2.imshow(self.windowName, frame)

def main():
    #processImage = ProcessImage()
    detectOrangeBall = DetectOrangeBall(np.array([5, 100, 80]), np.array([20, 255, 255]), 0.005)
    display = Display()

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Cannot load the image")
            break

        #frame = cv2.imread("ball_sample2.jpg")
        # frame = processImage.applyClahe(frame)
        hsv, contours = detectOrangeBall.extractSpecificColor(frame)
        pos, hull = detectOrangeBall.calculateCentroid(hsv.shape, contours)
        print("centroid of the ball:", pos)
        display.indicateCentroid(frame, pos, hull)

        key = cv2.waitKey(1)
        if key == 27:
            break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
