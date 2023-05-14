import cv2
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "image.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()






img = cv2.imread('image.png',cv2.IMREAD_COLOR)
copy = img.copy()
#Do some image processing here. As an example, do point detection

# "scorching the image" so its more visible for sticker
img = cv2.addWeighted(img, 3, img, 0, 0)
img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Do some image processing here. As an example, do point detection
img = cv2.medianBlur(img,5)
GreenLower = np.array([0])
GreenUpper = np.array([80])
img_mask = cv2.inRange(img_gry,GreenLower,GreenUpper)

img_edge = cv2.Canny(img_mask,100,200)
img_edgeline = cv2.Canny(img_mask,50,100)


ret, thresh = cv2.threshold(img_edge, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for i in contours:
  M = cv2.moments(i)
  if M['m00'] != 0:
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      (x,y,w,h) = cv2.boundingRect(i)
      copymask = img_mask[y:y+h, x:x+w]
      ratio = cv2.countNonZero(copymask)/copymask.size
      if(w/h < 1.2 and w/h > 0.8):
        #cv2.drawContours(img, [i], -1, (0, 255, 0), 2)
        centerx = int(x+w/2)
        centery = int(y -h/2)
        r,g,b = copy[centery,centerx]
        if(ratio > 0.5 and w > 12 and h > 12):
          cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 255), 2)
          cv2.putText(img, "empty", (cx - 50, cy - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif (ratio < 0.9 and ratio > 0.3 and w > 20 and h > 20):
          cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
          cv2.putText(img, "Placed", (cx - 50, cy - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
      elif(w/h < 0.8  and w/h > 0.2 and ratio > 0.5 and ratio < 0.8 and w > 10 and h > 10 and w < 20 and h < 45):#side hole special case
          cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 255), 2)
          cv2.putText(img, "side hole", (x - 20, y - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
      elif(w/h < 0.8  and w/h > 0.2 and ratio > 0.5 and ratio < 0.8 and w > 10 and h > 10):
          cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
          cv2.putText(img, "misplaced", (x - 20, y - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
      
          
#plt.figure(figsize=(20,10))
#plt.subplot(131),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('edges',color='c')
#plt.subplot(132),plt.imshow(cv2.cvtColor(img_mask,cv2.COLOR_BGR2RGB)),plt.title('mask')
#plt.subplot(133),plt.imshow(cv2.cvtColor(img_edgeline,cv2.COLOR_BGR2RGB)),plt.title('line')
plt.figure(figsize=(200,150))
plt.subplot(131),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('img')
plt.show()
