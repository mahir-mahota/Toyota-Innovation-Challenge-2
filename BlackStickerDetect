import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    copy = img.copy()
        # "scorching the image" so its more visible for sticker
    img = cv2.addWeighted(img, 7, img, 0, 0)
    img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Do some image processing here. As an example, do point detection
    img = cv2.medianBlur(img,9)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    firstLower = np.array([0])
    firstUpper = np.array([220])
    img_mask = cv2.inRange(img_gry,firstLower,firstUpper)

    img_edge = cv2.Canny(img_mask,900,1000)

    ret, thresh = cv2.threshold(img_edge, 200, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = 0
    boxh = 0
    boxy = 0
    for i in contours:
      #M = cv2.moments(i)
      #if M['m00'] != 0:
      #    cx = int(M['m10']/M['m00'])
      #    cy = int(M['m01']/M['m00'])
        (x,y,w,h) = cv2.boundingRect(i)
        boxh = h
        boxy = y
        if(rects < 2 and w > 20 and h > 50):
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 255), 2)
            rects +=1
            img_gry2 = img_gry[boxy:(boxh + boxy), x:(w+x)]
            secondLower = np.array([0])
            secondUpper = np.array([20])
            second_mask = cv2.inRange(img_gry2,secondLower,secondUpper)
            img_edge2 = cv2.Canny(second_mask,900,1000)
            ret, thresh = cv2.threshold(img_edge, 200, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in contours:
                (x,y,w,h) = cv2.boundingRect(i)
                copymask = img_mask[y:y+h, x:x+w]
                ratio = cv2.countNonZero(copymask)/copymask.size
                if(w/h < 1.1 and w/h > 0.9):
                    #cv2.drawContours(frame, [i], -1, (255, 255, 255), 2)
                    centerx = int(x+w/2)
                    centery = int(y -h/2)
                    r,g,b = copy[centery,centerx]
                    if(ratio > 0.3 and w > 12 and h > 12 and w*h < 950):
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 0), 2)
                        cv2.putText(frame, "empty", (x - 50, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif (ratio < 0.8 and ratio > 0.3 and w > 15 and h > 15 and w*h >= 790):
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "Placed", (x - 50, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                elif(w/h < 0.8  and w/h > 0.2 and ratio > 0.5 and ratio < 0.8 and w > 10 and h > 10 and w < 20 and h < 45):#side hole special case
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 0), 2)
                    cv2.putText(frame, "side hole", (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                elif(w/h < 0.8  and w/h > 0.2 and ratio > 0.3 and ratio < 0.9 and w > 10 and h > 20 and w < 20 and h < 45):#side hole special case
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "side hole", (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                elif(w*h > 1000 and w*h < 1500 and w/h < 0.95  and w/h > 0.2 and ratio > 0.5 and ratio < 0.8 and w > 10 and h > 10):
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                    cv2.putText(frame, "misplaced", (x - 20, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
