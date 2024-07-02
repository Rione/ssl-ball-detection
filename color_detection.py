import cv2
import numpy as np
from libcamera import Transform

# オレンジ色のHSV範囲
lowColor = np.array([5, 150, 150])
highColor = np.array([15, 255, 255])

# 抽出するオレンジ色の塊のしきい値
areaRatioThreshold = 0.005
"""
def clahe(img_name):

    img_yuv = cv2.cvtColor(img_name, cv2.COLOR_BGR2YUV) # RGB => YUV(YCbCr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # claheオブジェクトを生成
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0]) # 輝度にのみヒストグラム平坦化
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # YUV => RGB

    return img
"""

def calculateCentroid(frame, areaRatioThreshold, lowColor, highColor):
    h, w, c = frame.shape

    # hsv色空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 色を抽出する
    mask = cv2.inRange(hsv, lowColor, highColor)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 面積を計算
    areas = np.array(list(map(cv2.contourArea, contours)))

    if len(areas) == 0 or np.max(areas) / (h * w) < areaRatioThreshold:
        # 見つからなかったらNoneを返す
        print("the area is too small")

        return None
    else:
        # 面積が最大の塊の重心を計算し返す
        max_idx = np.argmax(areas)
        result = cv2.moments(contours[max_idx])
        x = int(result["m10"] / result["m00"])
        y = int(result["m01"] / result["m00"])

        return (x, y)

def main():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()

        # 位置を抽出
        pos = calculateCentroid(frame, areaRatioThreshold, lowColor, highColor)

        cnt = cv2.circle(frame, pos, 10, (255, 0, 0), -1)

        #if pos is not None:
        cv2.circle(frame, pos, 10, (255, 0, 0), -1)

        print("centroid of ball:")
        print(pos)
        cv2.imshow("Frame", cnt)

        key = cv2.waitKey(1)

        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
