import os
import re
import argparse
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import sklearn
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_dir", required=True,
	help="path to the root directory of image dataset")
args = vars(ap.parse_args())

model_dir = 'imagenet'
images_dir = os.path.join(args['images_dir'],'')

#Crea imagenes temporales en formato jpg por compatibilidad con inceptionV3
def create_tmp_images():
	png_images = [images_dir+f for f in os.listdir(images_dir) if re.search('png|PNG', f)]

	for f in png_images:
		img = cv2.imread(f)
		out = f.split(".")[0]+".jpg"
		cv2.imwrite(out, img)

#Carga la red pre-entrenada inceptionV3
def create_graph():
	with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

#Extrae los features que calcula inception V3 hasta la capa pool3:0
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	labels = []

	create_graph()

	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			if (ind%100 == 0):
				print('Processing %s...' % (image))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)

			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
			
			#Se guarda el label a partir del nombre de la imagen
			labels.append(re.split('_\d+',image.split(os.sep)[1])[0])

	return features, labels

create_tmp_images()

#Se buscan las imagenes jpg temporales
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
	
features,labels = extract_features(list_images)

for f in list_images:
	os.remove(f)

#Se guardan los features extraidos y los labels en archivos
pickle.dump(features, open('features.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))
print("Features and labels dumped...")