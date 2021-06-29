import numpy as np
import cv2
import os

def faceDetection(input_img):
    gray_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    face_haar=cv2.CascadeClassifier(r'YOUR DATA LOCATION + face-recognition\data\haarcascade_frontalface_alt.xml')
    faces=face_haar.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=3)
    return faces, gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filesnames in os.walk(directory):
        for filesname in filesnames:
            if filesname.startswith("."):
                print("skipping system File")
                continue
            id=os.path.basename(path)
            img_path = os.path.join(path,filesname)
            print("img_path", img_path)
            print("id", id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("not loaded properly")
                continue

            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h) = faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID

def train_Classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# Drawing a Rectangle on the Face Function
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)


# Putting text on images
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 6)


