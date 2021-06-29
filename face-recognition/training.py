import numpy as np
import cv2
import os

import faceRecognition as fr

print(fr)

test_img = cv2.imread(r'')

faces_detected, gray_img = fr.faceDetection(test_img)
print("Face Detected", faces_detected)

# training here

faces,faceID=fr.labels_for_training_data(r'')
face_recognizer=fr.train_Classifier(faces,faceID)
face_recognizer.save(r'')

name={0: "Mark Zuckerberg", 1: "Jennifer Aniston"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray=gray_img[y:y+w,x:x+h]
    label, confidence=face_recognizer.predict(roi_gray)
    print("Confidence :", confidence)
    print("label :", label)
    fr.draw_rect(test_img,face)
    predict_name = name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(800,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
