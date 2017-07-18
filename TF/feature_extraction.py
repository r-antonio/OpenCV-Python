import os
import re
import argparse
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import sklearn
import json
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_dir", required=True,
	help="path to the root directory of image dataset")
args = vars(ap.parse_args())

model_dir = 'imagenet'
images_dir = os.path.join(args['images_dir'],'')

#Carga la red pre-entrenada inceptionV3
def create_graph():
	with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

create_graph()

#Extrae los features que calcula inception V3 hasta la capa pool3:0
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((len(list_images),nb_features))
	labels = []

	lbls = {}
	i = 0

	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
		for ind, image in enumerate(list_images):
			if (ind%100 == 0):
				print('Processing image {0}/{1}...'.format(ind,len(list_images)))
			if not gfile.Exists(image):
				tf.logging.fatal('File does not exist %s', image)

			image_data = gfile.FastGFile(image, 'rb').read()
			predictions = sess.run(next_to_last_tensor,{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
			
			#Se guarda el label a partir del nombre de la imagen
			lbl = image.split('-')[1].split('.')[0]	
			labels.append(lbl)
			if not lbl in lbls:
				lbls[lbl] = i
				i += 1			
	with open('labels.json','w') as f:
		json.dump(lbls,f)

	return features, labels

#Se buscan las imagenes jpg temporales
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
	
features,labels = extract_features(list_images)

#Se guardan los features extraidos y los labels en archivos
pickle.dump(features, open('features.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))
print("Features and labels dumped...")
