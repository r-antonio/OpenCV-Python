import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def skin_extraction(img, roi):
	hist_skin = cv2.calcHist([img],[0, 1],None,[12, 12],[0,180,0,256])
	
	cv2.normalize(hist_skin, hist_skin, 0, 255, cv2.NORM_MINMAX)
	dst = cv2.calcBackProject([roi], [0,1], hist_skin, [0,180,0,256],1)
	
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	cv2.filter2D(dst,-1,disc, dst)
	
	ret,thresh = cv2.threshold(dst,254,255,cv2.THRESH_BINARY)
	thresh = cv2.merge((thresh,thresh,thresh))
	
	return cv2.bitwise_and(roi, thresh)

def get_possible_hands(gray):
	_, contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
	contours = sorted(contours, key = cv2.contourArea ,reverse=True)
	return contours[:2]
	
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))

if len(faces) > 0:
	faces = sorted(faces, key = lambda face: face[2]*face[3], reverse = True)

	x_face,y_face,w_face,h_face = faces[0]

	w_roi = w_face * 5
	h_roi = h_face * 3
	x_roi = x_face - int(2*w_face)
	y_roi = y_face
	
	cv2.rectangle(frame, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,0,255), 2)
	cv2.rectangle(frame, (x_roi,y_roi), (x_roi+w_roi,y_roi+h_roi), (0,255,255), 2)
	
	crop_face = hsv[y_face:y_face+h_face, x_face:x_face+w_face]
	
	removal = hsv.copy()
	removal[y_face:y_face+int(h_face*1.2), x_face:x_face+w_face] = 0
	
	roi = removal[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
	
	res = skin_extraction(crop_face, roi)
	
	#kernel = np.ones((5,5),np.uint8)
	#res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
	#res = cv2.erode(res,kernel,iterations = 1)
	#res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
	#res = cv2.dilate(res,kernel,iterations = 1)
	
	h,s,v = cv2.split(res)
	hands = get_possible_hands(v)
	
	res =cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
	
	cv2.drawContours(res, hands, -1, (0,255,0), 2)
	
	cv2.imshow('Result',res)
else:
	print("No fueron encontradas caras")
cv2.imshow('frame',frame)

cv2.waitKey(0)