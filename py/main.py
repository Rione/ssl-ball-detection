import cv2
import numpy as np
import base64
import json
import socket
import os


def loadThresholds(file_path="thresholds.txt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        file_path,  
        os.path.join(base_dir, file_path),  
    ]
    
    thresholds = {}
    
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"reading the file: {path}")
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('//'):
                        continue
                        
                    key, value = [part.strip() for part in line.split(':', 1)]
                    
                    if key in ["minThreshold", "maxThreshold"]:
                        # 配列値を処理
                        values = [int(v.strip()) for v in value.split(',')]
                        thresholds[key] = np.array(values)
                    else:
                        # 数値を処理
                        thresholds[key] = float(value)
            
            print("reading completed")
            break
    else:
        print(f"Warning: configure file {file_path} not found.")
    
    return thresholds

class ImageProcessor:
    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        
        self._minThreshold = settings.get("minThreshold", np.array([1, 120, 100]))
        self._maxThreshold = settings.get("maxThreshold", np.array([15, 255, 255]))
        self._ksize = (5, 5)
        self._sigmaX = 0
        self._shape = cv2.MORPH_RECT
        self._size = (3, 3)
        self._operation = cv2.MORPH_OPEN

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
    def __init__(self, settings=None):
        if settings is None:
            settings = {}
            
        self.imageProcessor = ImageProcessor(settings)
        self._radius = settings.get("ballDetectRadius", 150)
        self._circularityThreshold = settings.get("circularityThreshold", 0.2)
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

    def _isCircular(self, contour, circularityThreshold=None):
        if circularityThreshold is None:
            circularityThreshold = self._circularityThreshold
            
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return False
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return circularity > circularityThreshold

    def _focus(self, frame, center):
        height, width = frame.shape[:2]
        if center is not None:
            x, y = center
            xMin, yMin = max(0, x - self._radius), max(0, y - self._radius)
            xMax, yMax = min(width, x + self._radius), min(height, y + self._radius)
        else:
            xMin, yMin, xMax, yMax = 0, 0, width, height

        xMin, yMin, xMax, yMax = int(xMin), int(yMin), int(xMax), int(yMax)

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
    def __init__(self, device=0, settings=None):
        if settings is None:
            settings = {}
            
        self.cap = cv2.VideoCapture(device)
        self._fps = settings.get("fps", 30)
        self._bufferSize = settings.get("bufferSize", 4)
        
    def setProperties(self):
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._bufferSize)
    
    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

class Encoder:
    @staticmethod
    def encodeData(frame, center=None):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            raise ValueError("Failed to encode image")

        frame_bytes = base64.b64encode(frame.tobytes()).decode('utf-8')
        
        if center:
            x = center[0]
            y = center[1]
        else:
            x = None
            y = None
        
        return json.dumps({
            'frame': frame_bytes,
            'x': x,
            'y': y
        })    
class UDPClient:
    def __init__(self, host='172.16.0.14', port=31133):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send(self, data):
        try:
            self.socket.sendto(str(data).encode(), (self.host, self.port))
            return True
        except Exception as e:
            print(f"error: {e}")
            return False
    
    def close(self):
        if self.socket:
            self.socket.close()


def main():
    settings = loadThresholds()
    ballDetector = BallDetector(settings)  
    visualizer = Visualizer()
    videoCapture = VideoCapture(0, settings)
    udpClient = UDPClient() 
    
    try:
        while True:
            ret, frame = videoCapture.read()
            if not ret:
                print("Error: Failed to load the image")
                break
            
            center, circleContour, vertices = ballDetector.detect(frame)
            print("Center of the ball:", center)
            encodedData = Encoder.encodeData(frame, center)
            encodedJson = json.loads(encodedData)
            udpClient.send(encodedJson)                        
            #visualizer.draw(frame, center, circleContour, vertices)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        videoCapture.release()
        udpClient.close()  
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()