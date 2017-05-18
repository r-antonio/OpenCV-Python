import argparse
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to the root directory of uncompressed buffy dataset")
ap.add_argument("-d", "--debug", required=False,
	help="flag to enable debug mode", default=False, nargs='?', const=True, type=bool)
args = vars(ap.parse_args())

root_dir = args['path']
debug = args['debug']
imgs_path = root_dir+'/images'
sticks_path = root_dir+'/data'
max_distance = 6

def read_sticks_data(file_path):
	annotations = []
	with open(file_path,'r') as f:
		lines = []
		for line in f:
			lines.append(line)
			if len(lines) > 6:
				annotations.append([l.split() for l in lines[1:]])
				lines = []
	return np.array(annotations, dtype='float')

def detect_face_point(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faceCascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	if len(faces) == 0:
		return -1,-1
	x,y,w,h = faces[0]
	return (x+int(w/2),y+int(h/2))

def get_head_midpoint_from_sticks(sticks):
	x1,y1,x2,y2 = sticks[5]
	return (int((x1+x2)/2), int((y1+y2)/2))
	
def normalize(x):
	norm = np.linalg.norm(x)
	if norm == 0:
		return x
	return x/norm
	
stick_data = np.array([read_sticks_data(sticks_path+'/'+stick_file) for stick_file in os.listdir(sticks_path)])
total = np.sum(sf.shape[0] for sf in stick_data)
distances = []
cant = 0
for i,dir in enumerate(os.listdir(imgs_path)):
	for j,img in enumerate(os.listdir(imgs_path+'/'+dir)):
		frame = cv2.imread(imgs_path+'/'+dir+'/'+img)
		face_point = detect_face_point(frame)
		head_point = get_head_midpoint_from_sticks(stick_data[i][j])
		dist = math.sqrt(distance.euclidean(face_point,head_point))
		distances.append(dist)
		
		if debug:
			print(imgs_path+'/'+dir+'/'+img)
			cv2.circle(frame,face_point,2,(0,255,0))
			cv2.circle(frame,head_point,2,(0,255,255))
			print(dist)
			cv2.imshow(img,frame)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		cant+=1
		if cant%20==0:
			print('{0:.2f}% procesado'.format(cant*100/total))

distances = np.array(distances)
plt.hist(distances, bins='auto')
plt.title('Histogram of distances')
plt.show()

normalized = np.array([int(i<max_distance) for i in distances])
print('Accuracy:',np.sum(normalized)/normalized.size)