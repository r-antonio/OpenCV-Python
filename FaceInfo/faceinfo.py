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
	maxArea = 0
	maxAreaFace = -1
	index = 0
	for (x,y,w,h) in faces:
		area = w*h
		if area > maxArea:
			maxArea = area
			maxAreaFaceIdx = index
		index += 1
		
	x_face,y_face,w_face,h_face = faces[maxAreaFaceIdx]

	w_roi = w_face * 5
	h_roi = h_face * 3
	x_roi = x_face - int(2*w_face)
	y_roi = y_face
	
	crop_face = gray[y_face:y_face+h_face, x_face:x_face+w_face]
	crop_roi = gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
	
	removal = frame.copy()
	removal[y_face:y_face+h_face, x_face:x_face+w_face] = 0
	
	
	
	hist = cv2.calcHist([crop_face],[0],None,[256],[0,256])
	plt.figure()
	plt.title("Grayscale Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	plt.plot(hist)
	plt.xlim([0, 256])
	plt.show()
	
	cv2.rectangle(frame, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,0,255), 2)
	cv2.rectangle(frame, (x_roi,y_roi), (x_roi+w_roi,y_roi+h_roi), (0,255,255), 2)
else:
	print("No fueron encontradas caras")
cv2.imshow('frame',frame)

cv2.waitKey(0)