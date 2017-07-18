import os
import re
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_dir", required=True,
	help="path to the root directory of image dataset")
args = vars(ap.parse_args())

images_dir = os.path.join(args['images_dir'],'')

if not os.path.exists('tmp/'):
	os.makedirs('tmp/')

png_images = [images_dir+f for f in os.listdir(images_dir) if re.search('png|PNG', f)]
i = 0
for f in png_images:
	img = cv2.imread(f)
	out = 'tmp/'+str(i)+'-'+f.split(os.sep)[-1].split("_")[0]+".jpg"
	#img = cv2.resize(img,(64,64))
	cv2.imwrite(out, img)
	i += 1
