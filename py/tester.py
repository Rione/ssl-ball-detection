import base64
import json
import time

import cv2
import numpy as np
import serial
from robot_communicator import RobotCommunicator

class ImageProcessor:
    def __init__(
        self,
        minThreshold=np.array([1, 120, 100]),
        maxThreshold=np.array([15, 255, 255]),
        ksize=(5, 5),
        sigmaX=0,
        shape=cv2.MORPH_RECT,
        size=(3, 3),
        operation=cv2.MORPH_OPEN,
    ):
        self._minThreshold = minThreshold
        self._maxThreshold = maxThreshold
        self._ksize = ksize
        self._sigmaX = sigmaX
        self._shape = shape
        self._size = size
        self._operation = operation

    def extractColors(self, frame):
        filtered = self._filterFrame(frame)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        shadowMask = self._detectShadows(hsv)
        hsv = self._equalizeHist(hsv)
        mask = cv2.inRange(hsv, self._minThreshold, self._maxThreshold)
        mask = cv2.bitwise_and(mask, mask, mask=shadowMask)
        mask = self._applyMorphologicalTransformations(mask)
        return mask

    def _filterFrame(self, frame):
        return cv2.GaussianBlur(frame, self._ksize, self._sigmaX)

    def _detectShadows(self, hsv):
        v = hsv[:, :, 2]
        shadowMask = cv2.inRange(v, 0, 50)
        shadowMask = cv2.bitwise_not(shadowMask)
        return shadowMask

    def _equalizeHist(self, hsv):
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        return cv2.merge((h, s, v))

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
                diameter = int(radius * 2)
                distance = 320 * 70 / diameter  # Assuming a fixed distance calculation
                return center, circleContour, vertices, distance
        return None, None, None, None

    def _isCircular(self, contour, circularityThreshold=0.8):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return False
        circularity = (4 * np.pi * area) / (perimeter**2)
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
        circleContour = (
            np.column_stack(
                (
                    int(x) + radius * np.cos(angles) + offset[0],
                    int(y) + radius * np.sin(angles) + offset[1],
                )
            )
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )
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
            cv2.rectangle(
                frame,
                (vertices[0], vertices[1]),
                (vertices[2], vertices[3]),
                (0, 0, 255),
                1,
            )

    def _drawCircle(self, frame, circle):
        if circle is not None:
            cv2.drawContours(frame, [circle], -1, (255, 0, 0), 1)

    def _drawCenter(self, frame, center):
        if center is not None:
            cv2.circle(frame, center, self._radius, (255, 0, 0), -1)


class VideoCapture:
    def __init__(self, device=0, fps=30, bufferSize=4):
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


class Encoder:
    @staticmethod
    def encode_data(frame, center=None, distance=None):
        # 画像データをbase64エンコード
        frame_bytes = base64.b64encode(frame.tobytes()).decode("utf-8")

        # 座標データをbase64エンコード
        if center:
            x = base64.b64encode(str(center[0]).encode()).decode("utf-8")
            y = base64.b64encode(str(distance).encode()).decode("utf-8")
        else:
            x = base64.b64encode(b"None").decode("utf-8")
            y = base64.b64encode(b"None").decode("utf-8")

        # JSON形式でデータを返す
        return json.dumps({"frame": frame_bytes, "x": x, "y": y})


def main():
    ballDetector = BallDetector()
    visualizer = Visualizer()
    videoCapture = VideoCapture()

    videoCapture.setProperties()

    while True:
        start_time = time.time()

        ret, frame = videoCapture.read()
        if not ret:
            print("Failed to load the image")
            break

        center, circleContour, vertices, distance = ballDetector.detect(frame)
        visualizer.draw(frame, center, circleContour, vertices)

        encoded_data = Encoder.encode_data(frame, center, distance)
        encoded_json = json.loads(encoded_data)

        print("X coordinate:", encoded_json["x"])
        print("Y coordinate:", encoded_json["y"])
        print("Frame length:", len(encoded_json["frame"]))

        process_time = (time.time() - start_time) * 1000
        print(f"処理時間: {process_time:.1f} ms")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
