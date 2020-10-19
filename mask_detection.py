import cv2
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)
covid = cv2.imread('covid.png')
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('D:\Downloads\mask_model')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    gray = cv2.filter2D(gray,-1,kernel)
    faces = face.detectMultiScale(gray,1.3,5)
    glob_img = []
    for face_example in faces:
        (x,y,width,height) = face_example
        img = gray[y-20:y+height+20,x-20:x+width+20]
        try:
            img = cv2.resize(img,(100,100),cv2.INTER_AREA)
        except Exception:
            continue
        img = img/255.0
        img = np.reshape(img,(1,100,100,1))
        label = model.predict(img) > 0.5
        if not label:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0))
            cv2.putText(frame,'PUT ON MASK',(x,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2,cv2.LINE_AA)


    cv2.imshow('Frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break