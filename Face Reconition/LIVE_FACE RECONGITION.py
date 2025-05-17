import os
import cv2 as cv
import numpy as np

harr=cv.CascadeClassifier("/home/ramkarthik/opencv python/Face Detection/harrcascade_frontalface_default.xml")
people=[]
for i in os.listdir("/home/ramkarthik/opencv python/Faces/train"):
    people.append(i)
print(people)
#created model with predefined data
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('/home/ramkarthik/opencv python/Face Reconition/face_trained_model.yml')
vid=cv.VideoCapture(0)
while True:
    isTrue,img=vid.read()
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    face_rect=harr.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face_rect:
        face_roi=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(face_roi)
        print(f'Lable={people[label]} ,index={label} with a confidence of {confidence}')
        cv.putText(img, str(people[label]), (100,100), cv.FONT_HERSHEY_TRIPLEX, 2.0, (0,255,0),thickness=3)
        cv.rectangle(img, (x,y),(x+w,y+h), (0,255,0),thickness=2)
    cv.imshow("live video",img)
    if(cv.waitKey(1) & 0xFF==ord('d')):
        break
vid.release()
cv.destroyAllWindows()