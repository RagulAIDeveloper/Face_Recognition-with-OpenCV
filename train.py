import cv2
import os
import numpy

##initialization the alogrithm
haar_file ="haarcascade_frontalface_default.xml"
dataset = "Datasets"

#image preprocessing
(images,labels,names,id) =([],[],{},0)
for (subdirs,dirs,file) in os.walk(dataset):
    for subdirs in dirs:
        names[id] = subdirs
        subjectpath = os.path.join(dataset,subdirs)
        for filename in os.listdir(subjectpath):
            path = subjectpath +"/"+filename
            #print(path)
            #print(filename)
            label=id
            #append image to images
            images.append(cv2.imread(path,0))
            #append id to labels
            labels.append(int(label))
        id+=1

# convert to array
(images,labels) = [numpy.array(lis) for lis in (images,labels)] 

#model training
model = cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
# loading algorithm
face_cascade= cv2.CascadeClassifier(haar_file)
#camera initializtion
cam =cv2.VideoCapture(0) 

count =0

while True:
    # reading camera
    _,img = cam.read()
    # image convert to grayscale img 
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # getting face coordinates
    faces = face_cascade.detectMultiScale(gray_img,1.3,9)

    for (x,y,w,h) in faces:
        #crop the img
        face = gray_img[y:y+h,x:x+w]
        #image resized
        face_resize = cv2.resize(face,(130,100))
        
        #predict the face
        prediction = model.predict(face_resize) #return the two value  image and confidence level
        #draw rectangle
        cv2.rectangle(img,(x,y),(x+w ,y+h),(0,255,0),3)
        if prediction[1]<800: #if confidence less than 800 level
            cv2.putText(img,"%s - %.0f"% (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(51,255,255))
            print(names[prediction[0]])
            count=0
        else:
            count +=1
            cv2.putText(img,"unKnown",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(51,255,255))
            if count>100:
                print("unknown person")
                cv2.imwrite("input.jpg",img)
                count=0
    cv2.imshow("OpenCV",img)
    key = cv2.waitKey(27)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()



                        
