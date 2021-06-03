import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False


while rval:
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30,30),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("preview", frame)
    
    key = cv2.waitKey(20) # keep open until 'ESC' is pressed
    if key == 27:
        break


vc.release()
cv2.destroyWindow("preview")