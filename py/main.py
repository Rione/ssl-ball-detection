import cv2
import numpy as np
from libcamera import Transform


class ImageProcessor:
    def __init__(self, d=5, sigmaColor=75, sigmaSpace=75, 
                shape=cv2.MORPH_RECT, size=(3, 3), operation=cv2.MORPH_OPEN,
                lowColor=np.array([1, 120, 120]), highColor=np.array([25, 255, 255])):
        self._d = d
        self._sigmaColor = sigmaColor
        self._sigmaSpace = sigmaSpace
        self._shape = shape
        self._size = size
        self._operation = operation
        self._lowColor = lowColor
        self._highColor = highColor

    def extractColors(self, frame):
        filtered = self._filterFrame(frame)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lowColor, self._highColor)
        mask = self._applyMorphologicalTransformations(mask)
        return mask

    def _filterFrame(self, frame):
        return cv2.bilateralFilter(frame, self._d, self._sigmaColor, self._sigmaSpace)

    def _applyMorphologicalTransformations(self, mask):
        kernel = cv2.getStructuringElement(self._shape, self._size)
        return cv2.morphologyEx(mask, self._operation, kernel)


class BallDetector:
    def __init__(self, radius=150):
        self.imageProcessor = ImageProcessor()
        self._radius = radius
        self._previousCenter = None

    def detect(self, frame):
        roi, offset, vertices = self._focus(frame, self._previousCenter)
        mask = self.imageProcessor.extractColors(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            maxContour = max(contours, key=cv2.contourArea)
            if self._isCircular(maxContour):
                (x, y), radius = cv2.minEnclosingCircle(maxContour)
                center = (int(x) + offset[0], int(y) + offset[1])
                circleContour = self._createCircleContour(x, y, radius, offset)
                self._previousCenter = center
                return center, circleContour, vertices
        return None, None, None

    def _isCircular(self, contour, circularityThreshold=0.4):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        return circularity > circularityThreshold

    def _focus(self, frame, center):
        height, width = frame.shape[:2]
        if center is not None:
            x, y = center
            xMin, yMin = max(0, x - self._radius), max(0, y - self._radius)
            xMax, yMax = min(width, x + self._radius), min(height, y + self._radius)
        else:
            xMin, yMin, xMax, yMax = 0, 0, width, height

        roi = frame[yMin:yMax, xMin:xMax]
        offset = (xMin, yMin)
        vertices = (xMin, yMin, xMax, yMax)
        return roi, offset, vertices

    def _createCircleContour(self, x, y, radius, offset):
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        circleContour = np.column_stack((
            int(x) + radius * np.cos(angles) + offset[0],
            int(y) + radius * np.sin(angles) + offset[1]
        )).astype(np.int32).reshape(-1, 1, 2)
        return circleContour

class Visualizer:
    def __init__(self, radius=5, windowName="Frame"):
        self._radius = radius
        self._windowName = windowName

    def draw(self, frame, center, circle, vertices):
        self._drawRectangle(frame, vertices)
        self._drawCircle(frame, circle)
        self._drawCenter(frame, center)
        cv2.imshow(self._windowName, frame)

    def _drawRectangle(self, frame, vertices):
        if vertices:
            cv2.rectangle(frame, (vertices[0], vertices[1]), (vertices[2], vertices[3]), (0, 0, 255), 1)

    def _drawCircle(self, frame, circle):
        if circle is not None:
            cv2.drawContours(frame, [circle], -1, (255, 0, 0), 1)

    def _drawCenter(self, frame, center):
        if center is not None:
            cv2.circle(frame, center, self._radius, (255, 0, 0), -1)


class VideoCapture:
    def __init__(self, device=0, fps=60, bufferSize=4):
        self.cap = cv2.VideoCapture(device)
        self._fps = fps
        self._bufferSize = bufferSize
        
    def setProperties(self):
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._bufferSize)
    
    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def main():
    ballDetector = BallDetector()
    visualizer = Visualizer()
    videoCapture = VideoCapture()

    videoCapture.setProperties()
    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Error: Failed to load the image")
            break
        
        center, circleContour, vertices = ballDetector.detect(frame)
        print("Center of the ball:", center)
        visualizer.draw(frame, center, circleContour, vertices)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()