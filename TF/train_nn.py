import argparse
import pickle
from sklearn import model_selection
import numpy as np
import tflearn

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required=True,
	help="path to the features pickle file")
ap.add_argument("-l", "--labels", required=True,
	help="path to the features pickle file")
args = vars(ap.parse_args())

features = pickle.load(open(args['features'],'rb'))
labels = pickle.load(open(args['labels'],'rb'))

features = pickle.load(open('features.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))
labels = np.array(labels, dtype='|S4')
labels = labels.astype(np.int)
labels = labels-1
labels = np.eye(16)[labels]

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 2048])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, 16, activation='softmax')

# Regression using SGD with learning rate decay
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
acc = tflearn.metrics.Accuracy()
net = tflearn.regression(softmax, optimizer=sgd, metric=acc, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train, y_train, n_epoch=20, validation_set=(X_test, y_test), show_metric=True, run_id="dense_model")