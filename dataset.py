import cv2
import os

##initialization the alogrithm
haar_file ="haarcascade_frontalface_default.xml"
dataset ="Datasets"
sub_data="Elon"

#file create
path = os.path.join(dataset,sub_data) #Datasets/ragul
if not os.path.isdir(path):   
    os.mkdir(path)              
    
(width,height)=(130,100)
#loading the alogrithm
face_cascade = cv2.CascadeClassifier(haar_file)   
#cam initialization
webcam = cv2.VideoCapture(0)   # cam ini

count=1

while count<101:
    print(count)
    # cam reading
    _,img=webcam.read()
    # color  to converting grayscaleimg
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # face detection
    faces = face_cascade.detectMultiScale(grayimg,1.3,4) 
    for (x,y,w,h) in faces:
        # create rectangle in face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # img crop
        face=grayimg[y:y+h,x:x+w]
        # face resize
        face_resize = cv2.resize(face,(width,height))
        # image save
        cv2.imwrite("%s/%s.png"%(path,count),face_resize)
        cv2.imshow("image",img)
    count +=1
