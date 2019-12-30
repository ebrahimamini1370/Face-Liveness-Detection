import cv2
from PIL import Image
from utils import *

# initializing parameters
number_points = 8
radius = 2
font_path = 'simhei.ttf'
face_haar_file = 'haarcascade_frontalface_default.xml'

#load the classifier for real-time accomplishment
model = load_svm_clf()
face_cascade = cv2.CascadeClassifier(face_haar_file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h, x:x+w]
        face2 = face.copy()
        face_image = Image.fromarray(face2)
        result = predict_liveness_real_time(face_image, model, number_points, radius)
        location = (x, y)
        frame = draw_ch_zn(frame,str(result[0]),font_path,location)
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'): # exit on ESC or Q
        break

cv2.destroyWindow("Webcam")
cap.release()
