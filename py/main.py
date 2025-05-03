import time
import cv2
import numpy as np
import base64
import json
import socket
import os
from picamera2 import Picamera2


def loadSettingsFromJson(file_path="thresholds.json"):
    """Loads settings from a JSON file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_try = [
        file_path,
        os.path.join(base_dir, file_path),
    ]

    settings = {}  # Use 'settings' instead of 'thresholds' for clarity

    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Reading JSON settings file: {path}")
            try:
                with open(path, "r") as f:
                    settings = json.load(f)  # Load the entire JSON structure

                # --- Convert specific lists to numpy arrays ---
                # Check if keys exist before attempting conversion
                if "minThreshold" in settings and isinstance(
                    settings["minThreshold"], list
                ):
                    settings["minThreshold"] = np.array(
                        settings["minThreshold"], dtype=np.uint8
                    )  # Use uint8 for HSV
                if "maxThreshold" in settings and isinstance(
                    settings["maxThreshold"], list
                ):
                    settings["maxThreshold"] = np.array(
                        settings["maxThreshold"], dtype=np.uint8
                    )  # Use uint8 for HSV

                # Ensure numeric types for others if needed (json loads them as int/float)
                if "ballDetectRadius" in settings:
                    settings["ballDetectRadius"] = int(
                        settings["ballDetectRadius"]
                    )  # Ensure int if needed
                if "circularityThreshold" in settings:
                    settings["circularityThreshold"] = float(
                        settings["circularityThreshold"]
                    )

                print("JSON settings loaded successfully.")
                break  # Exit loop once file is found and loaded
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {path}: {e}")
                settings = {}  # Reset settings if JSON is invalid
                break  # Stop processing if the found file is invalid
            except Exception as e:
                print(f"An unexpected error occurred while reading {path}: {e}")
                settings = {}
                break
    else:
        # This block runs if the loop completes without finding the file
        print(
            f"Warning: Settings file '{file_path}' not found in checked locations. Using default values."
        )
        # Default values will be handled by the classes' __init__ methods

    return settings


class ImageProcessor:
    def __init__(self, settings=None):
        if settings is None:
            settings = {}

        # Use .get() to provide defaults if keys are missing from JSON or file wasn't found
        self._minThreshold = settings.get(
            "minThreshold", np.array([1, 120, 100], dtype=np.uint8)
        )
        self._maxThreshold = settings.get(
            "maxThreshold", np.array([15, 255, 255], dtype=np.uint8)
        )
        self._ksize = tuple(
            settings.get("gaussianKernelSize", (5, 5))
        )  # Example: allow configuring ksize via JSON
        self._sigmaX = settings.get("gaussianSigmaX", 0)
        self._shape = cv2.MORPH_RECT  # Could also be made configurable
        self._size = tuple(
            settings.get("morphKernelSize", (3, 3))
        )  # Example: allow configuring morph size
        self._operation = cv2.MORPH_OPEN  # Could also be made configurable

    def extractColors(self, frame):
        filtered = self._filterFrame(frame)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        # Shadow detection might need adjustments depending on lighting
        # shadowMask = self._detectShadows(hsv) # Consider if shadow detection is always needed
        hsv = self._equalizeHist(hsv)
        mask = cv2.inRange(hsv, self._minThreshold, self._maxThreshold)
        # mask = cv2.bitwise_and(mask, mask, mask=shadowMask) # Apply shadow mask if used
        mask = self._applyMorphologicalTransformations(mask)
        return mask

    def _filterFrame(self, frame):
        return cv2.GaussianBlur(frame, self._ksize, self._sigmaX)

    def _detectShadows(self, hsv):
        # Shadow detection thresholds might need tuning
        v = hsv[:, :, 2]
        # Using a fixed threshold (50) might not be robust
        _, shadowMask = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY)
        # shadowMask = cv2.inRange(v, 0, 50) # Original approach
        # shadowMask = cv2.bitwise_not(shadowMask) # Invert if needed
        return shadowMask

    def _equalizeHist(self, hsv):
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_equalized = clahe.apply(v)
        return cv2.merge((h, s, v_equalized))

    def _applyMorphologicalTransformations(self, mask):
        kernel = cv2.getStructuringElement(self._shape, self._size)
        # Opening removes small noise (erosion followed by dilation)
        return cv2.morphologyEx(mask, self._operation, kernel)


class BallDetector:
    def __init__(self, settings=None):
        if settings is None:
            settings = {}

        self.imageProcessor = ImageProcessor(settings)
        # Use .get() for robustness against missing keys
        self._radius = int(settings.get("ballDetectRadius", 150))  # Ensure int
        self._circularityThreshold = float(
            settings.get("circularityThreshold", 0.2)
        )  # Ensure float
        self._minContourArea = float(
            settings.get("minContourArea", 100)
        )  # Example: add min area threshold
        self._previousCenter = None

    def detect(self, frame):
        roi, offset, vertices = self._focus(frame, self._previousCenter)
        if roi.size == 0:  # Check if ROI is valid
            print("Warning: ROI is empty.")
            self._previousCenter = None  # Reset focus if ROI becomes invalid
            return None, None, None

        mask = self.imageProcessor.extractColors(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > self._minContourArea
        ]

        if not valid_contours:
            # Optional: Gradually expand ROI if nothing found?
            # self._previousCenter = None # Or keep last known position?
            return None, None, vertices  # Return vertices even if no ball found

        bestContour = max(valid_contours, key=cv2.contourArea)

        if self._isCircular(bestContour):
            (x, y), radius = cv2.minEnclosingCircle(bestContour)
            # Adjust center coordinates relative to the original frame
            center = (int(x + offset[0]), int(y + offset[1]))
            # Create a circle contour for visualization in the original frame coordinates
            circleContour = self._createCircleContour(
                x + offset[0], y + offset[1], radius
            )
            self._previousCenter = center
            # Return the absolute center, the absolute circle contour, and the ROI vertices
            return center, circleContour, vertices

        # If the largest contour isn't circular enough
        self._previousCenter = None  # Reset focus if no suitable ball found
        return None, None, vertices  # Return vertices even if no ball found

    def _isCircular(self, contour):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0 or area == 0:
            return False
        circularity = (4 * np.pi * area) / (perimeter**2)
        return circularity > self._circularityThreshold

    def _focus(self, frame, center):
        height, width = frame.shape[:2]
        radius = self._radius  # Use the instance variable

        if center is not None:
            cx, cy = center
            # Define ROI boundaries, ensuring they are within frame limits
            xMin = max(0, cx - radius)
            yMin = max(0, cy - radius)
            xMax = min(width, cx + radius)
            yMax = min(height, cy + radius)
        else:
            # If no previous center, use the whole frame
            xMin, yMin, xMax, yMax = 0, 0, width, height

        # Ensure coordinates are integers for slicing
        xMin, yMin, xMax, yMax = int(xMin), int(yMin), int(xMax), int(yMax)

        # Check if the calculated ROI is valid (has non-zero dimensions)
        if yMin >= yMax or xMin >= xMax:
            print(
                f"Warning: Invalid ROI calculated: ({xMin},{yMin}) to ({xMax},{yMax}). Using full frame."
            )
            xMin, yMin, xMax, yMax = 0, 0, width, height  # Fallback to full frame

        roi = frame[yMin:yMax, xMin:xMax]
        offset = (
            xMin,
            yMin,
        )  # Top-left corner of the ROI in original frame coordinates
        vertices = (xMin, yMin, xMax, yMax)  # ROI boundaries
        return roi, offset, vertices

    def _createCircleContour(self, centerX, centerY, radius, num_points=36):
        """Creates a circle contour in absolute frame coordinates."""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        # Calculate points directly in absolute coordinates
        contour_points = np.array(
            [
                [centerX + radius * np.cos(ang), centerY + radius * np.sin(ang)]
                for ang in angles
            ],
            dtype=np.int32,
        )
        # Reshape for drawContours: needs shape (num_points, 1, 2)
        return contour_points.reshape((-1, 1, 2))


class Visualizer:
    def __init__(self, radius=5, windowName="Frame"):
        self._radius = radius  # Radius for drawing the center point
        self._windowName = windowName
        # cv2.namedWindow(self._windowName, cv2.WINDOW_NORMAL) # Make window resizable

    def draw(self, frame, center, circleContour, vertices):
        # Draw ROI rectangle (optional)
        if vertices:
            cv2.rectangle(
                frame,
                (vertices[0], vertices[1]),
                (vertices[2], vertices[3]),
                (0, 0, 255),
                1,
            )  # Red ROI box

        # Draw detected circle contour
        if circleContour is not None:
            cv2.drawContours(
                frame, [circleContour], -1, (255, 0, 0), 2
            )  # Blue circle outline

        # Draw center point
        if center is not None:
            cv2.circle(
                frame, center, self._radius, (0, 255, 0), -1
            )  # Green center filled

        # cv2.imshow(self._windowName, frame)
        return frame

    def destroy(self):
        pass
        # cv2.destroyWindow(self._windowName)


class VideoCapture:
    def __init__(self, device=0, settings=None):
        if settings is None:
            settings = {}

        self.cap = Picamera2()
        config = self.cap.create_preview_configuration({"format": "RGB888"})
        self.cap.configure(config)
        self.cap.start()
        # if not self.cap.isOpened():
        #    raise IOError(f"Cannot open video capture device {device}")

        # Get settings or use defaults
        self._fps = int(settings.get("fps", 30))
        self._bufferSize = int(settings.get("bufferSize", 4))
        self._width = int(
            settings.get("frameWidth", 640)
        )  # Example: Add configurable width
        self._height = int(
            settings.get("frameHeight", 480)
        )  # Example: Add configurable height

    def setProperties(self):
        """Applies configured properties to the video capture device."""
        print(
            f"Attempting to set camera properties: Resolution={self._width}x{self._height}, FPS={self._fps}, BufferSize={self._bufferSize}"
        )
        # Setting properties might not work on all cameras/backends
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self._bufferSize)

        # Verify settings
        # actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # print(f"Actual camera properties: Resolution={actual_width}x{actual_height}, FPS={actual_fps}")
        pass

    def read(self):
        # print("aa")
        return (True, self.cap.capture_array())

    def release(self):
        # if self.cap.isOpened():
        #    self.cap.release()
        self.cap.close()
        print("Video capture released.")


class Encoder:
    @staticmethod
    def encodeData(frame, center=None, quality=90):
        """Encodes frame to JPEG bytes, Base64, and bundles with center coords in JSON."""
        # Ensure frame is not empty
        if frame is None or frame.size == 0:
            print("Error: Cannot encode empty frame.")
            return None

        # Encode frame to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded_image = cv2.imencode(".jpg", frame, encode_param)

        if not result:
            print("Error: Failed to encode image to JPEG.")
            return None

        # Convert JPEG bytes to Base64 string
        frame_bytes_b64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

        # Prepare coordinates
        x_coord = center[0] if center is not None else None
        y_coord = center[1] if center is not None else None

        # Create data dictionary
        data = {"frame": frame_bytes_b64, "x": x_coord, "y": y_coord}

        # Serialize dictionary to JSON string
        try:
            return json.dumps(data)
        except TypeError as e:
            print(f"Error serializing data to JSON: {e}")
            return None


class UDPClient:
    def __init__(self, host="127.0.0.1", port=31133):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set a timeout for socket operations (optional, but good practice)
        # self.socket.settimeout(1.0)
        print(f"UDP Client initialized for {self.host}:{self.port}")

    def send(self, data_json_string):
        """Sends data string over UDP."""
        if not data_json_string:
            print("UDP Send Error: No data to send.")
            return False
        try:
            # Encode the JSON string to bytes for sending
            self.socket.sendto(data_json_string.encode("utf-8"), (self.host, self.port))
            # print(f"UDP data sent to {self.host}:{self.port} ({len(data_json_string)} bytes)") # Verbose logging
            return True
        except socket.error as e:
            print(f"UDP Send Error: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during UDP send: {e}")
            return False

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None  # Prevent further use
            print("UDP Client socket closed.")


def main():
    # --- Load settings from JSON ---
    settings = loadSettingsFromJson()  # Tries to load "thresholds.json"

    try:
        # --- Initialize components with settings ---
        ballDetector = BallDetector(settings)
        visualizer = Visualizer()
        videoCapture = VideoCapture(
            0, settings
        )  # Pass settings for FPS, buffer, resolution
        videoCapture.setProperties()  # Apply camera settings
        udpClient = UDPClient(
            host=settings.get(
                "udpHost", "172.16.0.14"
            ),  # Allow configuring UDP target via JSON
            port=int(settings.get("udpPort", 31133)),
        )

        output_width = int(
            settings.get("outputFrameWidth", 160)
        )  # Configurable output size
        output_height = int(
            settings.get("outputFrameHeight", 96)
        )  # Configurable output size
        jpeg_quality = int(settings.get("jpegQuality", 90))  # Configurable quality

        while True:
            ret, frame = videoCapture.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame from camera.")
                time.sleep(0.5)  # Avoid busy-looping if camera fails
                continue  # Skip processing for this iteration

            # Detect the ball
            center, circleContour, vertices = ballDetector.detect(
                frame.copy()
            )  # Work on a copy for detection

            frame = visualizer.draw(frame, center, circleContour, vertices)

            # Prepare frame for encoding (resize)
            frame_resized = cv2.resize(
                frame, (output_width, output_height), interpolation=cv2.INTER_AREA
            )

            # Encode data (resized frame and center coordinates)
            encoded_json_string = Encoder.encodeData(
                frame_resized, center, quality=jpeg_quality
            )

            # Send data via UDP if encoding was successful
            if encoded_json_string:
                udpClient.send(encoded_json_string)

            # Visualization (optional - use original frame for better quality)
            # visualizer.draw(frame, center, circleContour, vertices) # Draw on the original frame

            # # Exit condition
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Exit requested.")
            #     break

    except IOError as e:
        print(f"Initialization Error: {e}")
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred in main loop: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
    finally:
        # --- Cleanup resources ---
        print("Cleaning up resources...")
        if "videoCapture" in locals() and videoCapture:
            videoCapture.release()
        if "udpClient" in locals() and udpClient:
            udpClient.close()
        if "visualizer" in locals() and visualizer:
            visualizer.destroy()  # Explicitly destroy window if using the class method
        # cv2.destroyAllWindows() # Alternative cleanup if not using Visualizer.destroy()
        print("Cleanup complete.")


if __name__ == "__main__":
    # Optional: Add command-line argument parsing for config file path
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", default="thresholds.json", help="Path to the JSON settings file")
    # args = parser.parse_args()
    # main(config_path=args.config) # Pass path to main if using args

    main()
