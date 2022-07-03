import cv2
import numpy as np
import imutils

# HISTOGRAM OF ORIENTED GRADIENT DETECTOR
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# IMAGE READING AND PRE-PROCESSING
frame = cv2.imread("beatles.jpg")
width = frame.shape[1]
max_width = 750
if width > max_width:
    frame = imutils.resize(frame, width=max_width)

# PEDESTRIAN DETECTOR
pedestrians, weights = HOG.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
pedestrians = np.array([[x, y, x + w, y + h] for (x, y, w, h) in pedestrians])

# BOX DRAWING
count = 1
for x, y, w, h in pedestrians:
    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y - 20), (w, y), (0, 255, 0), -1)
    cv2.putText(frame, f'Beatle{count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    count += 1

cv2.putText(frame, f'Total: {count - 1}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# RESULT SHOWING
cv2.imshow("PEDESTRIAN DETECTION", frame)
cv2.waitKey(0)
