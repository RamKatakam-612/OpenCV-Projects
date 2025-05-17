import os
import cv2 as cv
import numpy as np
people=[]
for i in os.listdir("/home/ramkarthik/opencv python/Faces/train"):
    people.append(i)
print(people)
DIR="/home/ramkarthik/opencv python/Faces/train"
#adding harrcasecade to detect the face and store in the images
harr=cv.CascadeClassifier("/home/ramkarthik/opencv python/Face Detection/harrcascade_frontalface_default.xml")
#Creation Of Train Set
features=[]
labels=[] 
def create_train():
    for person in people:
        
        path=os.path.join(DIR,person)
        label=people.index(person)
        
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            
            Img=cv.imread(img_path)
            gray=cv.cvtColor(Img,cv.COLOR_BGR2GRAY)
            
            face_rect=harr.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            
            for(x,y,w,h) in face_rect:
                face_roi=gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)
create_train()
print("***---------------Trained Data is read to cook---------------***")
#Now Features and the label(data need to be trained under the built face recognizer model)are created 
#we need to fit this data in the built facerecognizer model           

#print(f'Length of features = {len(features)}')
#print(f'Length of labels = {len(labels)}')            

#need to conver it into the numpy arrays
features=np.array(features,dtype="object")
labels=np.array(labels)

#built face recognition model

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Train the data in this model
face_recognizer.train(features,labels)

print("***---------------Model Trained successfully---------------***")

#Saving this model as file which will help to use any where portability

face_recognizer.save('face_trained_model.yml')

print("***---------------Trained Model Saved---------------***")

#Saving data to be trained 

np.save('features.npy',features)
np.save('labels.npy',labels)

print("***---------------Trained Data Saved---------------***")