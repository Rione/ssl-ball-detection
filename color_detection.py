import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from libcamera import Transform


class ImageProcessor:
    def __init__(self, shape=cv2.MORPH_RECT, size=(3, 3), operation=cv2.MORPH_OPEN):
        self.shape = shape
        self.size = size
        self.operation = operation

    def applyMorphogy(self, frame):
        kernel = cv2.getStructuringElement(self.shape, self.size)
        opening = cv2.morphologyEx(frame, self.operation, kernel)
        return opening


class ColorDetectorBase:
    def __init__(self, lowColor, highColor):
        self.lowColor = lowColor
        self.highColor = highColor


class ColorExtracter(ColorDetectorBase):
    def extractOneColor(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lowColor, self.highColor)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return hsv, contours


class ContourProcessor:
    def interpolateContours(contour):
        contour = contour.reshape(-1, 2)
        x, y = contour[:, 0], contour[:, 1]
        tck, u = splprep([x, y], s=0)
        uNew = np.linspace(u.min(), u.max(), 100)
        xNew, yNew = splev(uNew, tck, der=0)
        interpolatedContour = np.vstack((xNew, yNew)).T.astype(np.int32).reshape(-1, 1, 2)
        return interpolatedContour


class BallDetector(ColorDetectorBase):
    def __init__(self, lowColor=np.array([5, 100, 80]), 
                    highColor=np.array([20, 255, 255]), threshold=0.005):
        super().__init__(lowColor, highColor)
        self.centroidCalculator = CentroidCalculator(threshold)
        self.colorExtractor = ColorExtracter(lowColor, highColor)

    def detect(self, frame):
        hsv, contours = self.colorExtractor.extractOneColor(frame)
        pos, hull = self.centroidCalculator.calculate(hsv.shape, contours)
        return pos, hull


class CentroidCalculator:
    def __init__(self, threshold=0.005):
        self.threshold = threshold

    def interpolateContours(self, contour):
        contour = contour.reshape(-1, 2)
        x, y = contour[:, 0], contour[:, 1]
        tck, u = splprep([x, y], s=0)
        uNew = np.linspace(u.min(), u.max(), 50)
        xNew, yNew = splev(uNew, tck, der=0)
        interpolatedContour = np.vstack((xNew, yNew)).T.astype(np.int32).reshape(-1, 1, 2)
        return interpolatedContour

    def calculate(self, frameShape, contours):
        h, w = frameShape[:2]
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])

        if len(areas) == 0 or np.max(areas) / (h * w) < self.threshold:
            return None, None
        
        maxIdx = np.argmax(areas)
        contour = contours[maxIdx]
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approxContour = cv2.approxPolyDP(contour, epsilon, True)
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


class ImageDisplayer:
    def __init__(self, color=(255, 0, 0), radius=5, windowName="Frame"):
        self.color = color
        self.radius = radius
        self.windowName = windowName

    def indicateCentroid(self, frame, pos, hull):
        if pos is not None:
            contours = cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            centroid = cv2.circle(frame, (pos[0], pos[1]), self.radius, self.color, -1)
            cv2.imshow(self.windowName, centroid)
        else:
            cv2.imshow(self.windowName, frame)

def main():
    processImage = ImageProcessor()
    ballDetector = BallDetector()
    imageDisplayer = ImageDisplayer()

    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Cannot load the image")
            break

        #frame = cv2.imread("ball_sample2.jpg")
        #frame = processImage.applyMorphogy(frame)
        pos, hull = ballDetector.detect(frame)
        print("centroid of the ball:", pos)
        imageDisplayer.indicateCentroid(frame, pos, hull)

        key = cv2.waitKey(1)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
