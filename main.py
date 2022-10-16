from object_detector import ObjectDetector
from kalmanfilter import  KalmanFilter
import cv2

apple = ObjectDetector()
kf = KalmanFilter()
cap = cv2.VideoCapture("./apple_tracking_video2.mp4")
predicted = 0, 0
out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1280, 720))
while True:
    ret, frame = cap.read()

    x, y, x2, y2 = apple.detect(frame)
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)

    if (x+y+x2+y2 > 100):
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 4)
    else:
        cx, cy = predicted[0], predicted[1]

    predicted = kf.predict(cx, cy)
    cv2.circle(frame, (predicted[0], predicted[1]), 5, (0, 0, 255), 4)

    # cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
    flipped_img = cv2.flip(frame, 1)
    cv2.circle(flipped_img, (20, 20), 5, (255, 0, 0), 4)
    cv2.putText(flipped_img, '  : real position', (30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.circle(flipped_img, (20, 50), 5, (0, 0, 255), 4)
    cv2.putText(flipped_img, '  : predicted position', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    out.write(flipped_img)

    cv2.imshow("Frame", flipped_img)
    key = cv2.waitKey(150)
    if key == 27:
        break
out.release()
