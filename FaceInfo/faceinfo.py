import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
	removal[y_face:y_face+h_face, x_face:x_face+w_face] = 0
	
	roi = removal[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
	
	mask = np.zeros(crop_face.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (int(w_face/2),int(h_face/2),int(w_face/4),int(h_face/4))
	cv2.grabCut(crop_face,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	crop_face = crop_face*mask2[:,:,np.newaxis]
	
	cv2.imshow('Face',crop_face)
	cv2.waitKey(0)
	
	hist_skin = cv2.calcHist([crop_face],[0, 1],None,[180, 256],[0,180,0,256])
	
	cv2.normalize(hist_skin, hist_skin, 0, 255, cv2.NORM_MINMAX)
	dst = cv2.calcBackProject([roi], [0,1], hist_skin, [0,180,0,256],1)
	
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	cv2.filter2D(dst,-1,disc, dst)
	
	ret,thresh = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)
	thresh = cv2.merge((thresh,thresh,thresh))
	
	res = cv2.bitwise_and(roi, thresh)
	
	#res = np.vstack((roi, thresh, res))
	
	res =cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
	cv2.imshow('Result',res)
else:
	print("No fueron encontradas caras")
cv2.imshow('frame',frame)

cv2.waitKey(0)