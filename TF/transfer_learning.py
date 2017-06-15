import os
import re
import argparse
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images_dir", required=True,
	help="path to the root directory of image dataset")
args = vars(ap.parse_args())

model_dir = 'imagenet'
images_dir = os.path.join(args['images_dir'],'')

def create_tmp_images():
	png_images = [images_dir+f for f in os.listdir(images_dir) if re.search('png|PNG', f)]

	for f in png_images:
		img = cv2.imread(f)
		out = f.split(".")[0]+".jpg"
		cv2.imwrite(out, img)
	
def create_graph():
	with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')
		
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
			labels.append(re.split('_\d+',image.split('/')[1])[0])

	return features, labels

create_tmp_images()
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
	
features,labels = extract_features(list_images)
#pickle.dump(features, open('features', 'wb'))
#pickle.dump(labels, open('labels', 'wb'))

#features = pickle.load(open('features'))
#labels = pickle.load(open('labels'))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))

for f in list_images:
	os.remove(f)