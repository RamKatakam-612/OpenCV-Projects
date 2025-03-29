import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
cap=cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=HandDetector(detectionCon=0.8)
while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)
    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break