import argparse
import cv2
import numpy as np

def filter_red(hsv, frame):
	lower_red = np.array([165,100,100])
	upper_red = np.array([185,255,255])
	mask_red = cv2.inRange(hsv, lower_red, upper_red)
	res_red = cv2.bitwise_and(frame,frame, mask = mask_red)
	return (mask_red, res_red)
	
def filter_cyan(hsv, frame):
	lower_cyan = np.array([81,80,80])
	upper_cyan = np.array([101,255,255])
	mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
	res_cyan = cv2.bitwise_and(frame,frame, mask = mask_cyan)
	return (mask_cyan, res_cyan)
	
def filter_red_cyan(hsv, frame):
	mask_cyan, res_cyan = filter_cyan(hsv, frame)
	mask_red, res_red = filter_red(hsv, frame)
	mask_colors = cv2.add(mask_cyan, mask_red)
	res_colors = cv2.add(res_cyan, res_red)
	return (mask_colors, res_colors)
	
def find_faces(gray, frame):
	faceCascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

	print ("Encontradas {0} caras!".format(len(faces)))

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 0), 2)
	return faces
	
def hands_bb(gray, frame):
	im1, contours, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	return im1,contours,hier
	
def hands_image(gray, frame):
	thresh, im_bw = cv2.threshold(gray_dst, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	manos = cv2.bitwise_and(frame, frame, mask = im_bw)
	manos_hsv = cv2.cvtColor(manos, cv2.COLOR_BGR2HSV)
	return filter_red_cyan(manos_hsv, manos)
		
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Se aplican filtros para buscar los guantes
mask_red, res_red = filter_red(hsv, frame)
mask_cyan, res_cyan = filter_cyan(hsv, frame)

# Se realiza un blur para descartar pixeles aislados
median_right = cv2.medianBlur(res_red,15)
median_left = cv2.medianBlur(res_cyan,15)

dst = cv2.add(median_right, median_left)
gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# Se genera imagen de las manos
mask_manos, res_manos = hands_image(gray_dst, frame)

cv2.imshow('Mascara manos', mask_manos)
cv2.imshow('Resultado manos', res_manos)

# Se recuadran manos en la imagen
hands_bb(gray_dst, frame)

find_faces(gray, frame)

cv2.imshow('frame', frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
