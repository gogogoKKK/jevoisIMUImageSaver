import numpy as np
import cv2

cap = cv2.VideoCapture(2)
nimg = 31

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break;

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print("img size: ", gray.shape)
    # print(gray[0,:100])

    # Display the resulting frame
    cv2.imshow('frame',gray)
    key = cv2.waitKey(1)
    if  key & 0xFF == ord(' '):
        imgname = "imgs/%06d.png"%(nimg)
        cv2.imwrite(imgname, gray)
        print("save image: ", imgname)
        nimg+=1
    if key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
