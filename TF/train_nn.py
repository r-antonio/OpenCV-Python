import argparse
import pickle
from sklearn import model_selection
import numpy as np
from tflearn.data_utils import to_categorical,shuffle 
import tflearn
import json
from Dataset import Dataset

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required=True,
	help="path to the features pickle file")
ap.add_argument("-l", "--labels", required=True,
	help="path to the features pickle file")
args = vars(ap.parse_args())

features = pickle.load(open(args['features'],'rb'))
labels = pickle.load(open(args['labels'],'rb'))

with open('labels.json','r') as f:
	lbl_json = json.load(f)
labels = [lbl_json[l] for l in labels]

d=Dataset('test',features,labels)
d.remove_classes_with_few_examples(8)
n_classes = d.classes()
def get_model(n_classes):
	# Building deep neural network
	input_layer = tflearn.input_data(shape=[None, 2048])
	net = tflearn.fully_connected(input_layer, 512, activation='elu')
	softmax = tflearn.fully_connected(net, n_classes, activation='softmax')

	# Regression using Adam with learning rate decay
	adam = tflearn.Adam(learning_rate=0.001)
	acc = tflearn.metrics.Accuracy()
	#acc = tflearn.metrics.Top_k(k=5)
	net = tflearn.regression(softmax, optimizer=adam, metric=acc, loss='categorical_crossentropy')

	# Training
	model = tflearn.DNN(net, tensorboard_verbose=0)
	return model

def train():
	train_d,test_d = d.split_stratified()
	train_d.y_one_hot = to_categorical(train_d.y,n_classes)
	test_d.y_one_hot = to_categorical(test_d.y,n_classes)

	model=get_model(n_classes)	

	model.fit(train_d.x, train_d.y_one_hot, n_epoch=20, show_metric=True)

	train_predicted= np.argmax(model.predict(train_d.x),axis=1)
	train_accuracy=np.mean(train_predicted==train_d.y)
	test_predicted= np.argmax(model.predict(test_d.x),axis=1)
	test_accuracy=np.mean(test_predicted==test_d.y)
	print("Train accuracy: %f" % train_accuracy)
	print("Test accuracy: %f" % test_accuracy)
	train_acc.append(train_accuracy)
	test_acc.append(test_accuracy)

train_acc = []
test_acc = []
for _ in range(30):
	train()
print("Train mean accuracy: %f" % np.mean(train_acc))
print("Test mean accuracy: %f" % np.mean(test_acc))

