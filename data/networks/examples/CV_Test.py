import numpy as np
import cv2 as cv 

cap = cv.Videocapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, fram = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    gray = cv.cvtColor(frame, cv.Color_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


