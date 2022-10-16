import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

class ObjectDetector:
    def __init__(self):
        # Create mask for orange color
        self.low = np.array([13, 126, 119])
        self.high = np.array([132, 194, 255])

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks with color ranges
        mask = cv2.inRange(hsv_img, self.low, self.high)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        box = (0, 0, 0, 0)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box = (x, y, x + w, y + h)
            break

        return box