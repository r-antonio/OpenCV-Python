import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lower_red = np.array([165,100,100])
upper_red = np.array([185,255,255])

lower_cyan = np.array([81,80,80])
upper_cyan = np.array([101,255,255])

mask_right = cv2.inRange(hsv, lower_red, upper_red)
mask_left = cv2.inRange(hsv, lower_cyan, upper_cyan)

res_right = cv2.bitwise_and(frame,frame, mask = mask_right)
res_left = cv2.bitwise_and(frame,frame, mask = mask_left)

cv2.imshow('resRight', res_right)
cv2.imshow('resLeft', res_left)
median_right = cv2.medianBlur(res_right,15)
median_left = cv2.medianBlur(res_left,15)

dst = cv2.add(median_right, median_left)
gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
im1, contours, hier = cv2.findContours(gray_dst, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

faceCascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

print ("Encontradas {0} caras!".format(len(faces)))

for (x,y,w,h) in faces:
	cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 0), 2)

cv2.imshow('suma', dst)
cv2.imshow('frame', frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
