import os
import re
import cv2

if not os.path.exists('koller/'):
	os.makedirs('koller/')
with open('annotations.txt','r') as f:
	i = 0
	for line in f.readlines():
		l = line.split()
		img = cv2.imread(l[0])
		out = 'koller/'+str(i)+"-"+l[1]+".jpg"
		cv2.imwrite(out, img)
		i += 1
