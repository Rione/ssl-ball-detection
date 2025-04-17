import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from picamera2 import Picamera2

MIN = 0
MAX = 255


class GUI:
    def __init__(self, root):
        self.defaultValues = {
            "minH": MIN,
            "maxH": MAX,
            "minS": MIN,
            "maxS": MAX,
            "minV": MIN,
            "maxV": MAX,
        }
        self.hsvValues = {
            key: tk.IntVar(value=value) for key, value in self.defaultValues.items()
        }

        self.previousValues = self.defaultValues.copy()
        self.running = True
        self.root = root
        self.createWidgets()

    def createWidgets(self):
        for key in self.hsvValues.keys():
            frame = ttk.Frame(self.root)
            frame.pack(fill="x", padx=5, pady=5)

            label = ttk.Label(frame, text=key, width=8)
            label.pack(side="left")

            slider = ttk.Scale(
                frame,
                from_=MIN,
                to=MAX,
                orient="horizontal",
                variable=self.hsvValues[key],
                length=300,
                command=lambda val, k=key: self.updateSliderValue(k, int(float(val))),
            )
            slider.pack(side="left", fill="x", expand=True, padx=10)

            entry = ttk.Entry(
                frame,
                textvariable=self.hsvValues[key],
                validate="focusout",
                validatecommand=(self.root.register(lambda value, k=key: self.validateEntry(value, k)), "%P"),
                invalidcommand=(self.root.register(lambda k=key: self.restorePreviousValue(k)),),
                width=5,
            )
            entry.pack(side="right")

        buttonFrame = ttk.Frame(self.root)
        buttonFrame.pack(pady=10)

        quitButton = ttk.Button(buttonFrame, text="Quit", command=self.quit)
        quitButton.pack(side="left", padx=10)

        resetButton = ttk.Button(buttonFrame, text="Reset", command=self.resetValues)
        resetButton.pack(side="left", padx=10)

    def updateSliderValue(self, key, value):
        self.hsvValues[key].set(value)
        self.previousValues[key] = value

    def validateEntry(self, value, key):
        if value.isdigit() and MIN <= int(value) <= MAX:
            self.previousValues[key] = int(value)
            return True
        return False

    def restorePreviousValue(self, key):
        self.hsvValues[key].set(self.previousValues[key])

    def resetValues(self):
        for key, value in self.defaultValues.items():
            self.hsvValues[key].set(value)
            self.previousValues[key] = value

    def getBounds(self):
        lower = [
            self.hsvValues["minH"].get(),
            self.hsvValues["minS"].get(),
            self.hsvValues["minV"].get(),
        ]
        upper = [
            self.hsvValues["maxH"].get(),
            self.hsvValues["maxS"].get(),
            self.hsvValues["maxV"].get(),
        ]
        return lower, upper

    def quit(self):
        self.running = False


def updateCamera(gui, camera):
    if not gui.running:
        camera.stop()
        cv2.destroyAllWindows()
        gui.root.quit()
        return

    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # フレームサイズを縮小
    frame = cv2.resize(frame, (640, 360))
    
    lowerBound, upperBound = gui.getBounds()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lowerBound), np.array(upperBound))
    
    # ウィンドウ設定を追加
    cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Masked", 640, 360)
    cv2.imshow("Masked", mask)

    gui.root.after(10, updateCamera, gui, camera)


def main():
    root = tk.Tk()
    root.title("HSV Controller")
    root.geometry("500x400")

    gui = GUI(root)
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"size": (640, 360)}  # カメラのキャプチャサイズも調整
    ))
    camera.start()

    updateCamera(gui, camera)

    root.mainloop()


if __name__ == "__main__":
    main()
