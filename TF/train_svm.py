import argparse
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features", required=True,
	help="path to the features pickle file")
ap.add_argument("-l", "--labels", required=True,
	help="path to the features pickle file")
args = vars(ap.parse_args())

features = pickle.load(open(args['features'],'rb'))
labels = pickle.load(open(args['labels'],'rb'))

train_scores = np.zeros(10)
test_scores = np.zeros(10)

for i in range(10):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

	clf = LinearSVC(C=0.003, loss='squared_hinge', penalty='l2',multi_class='ovr')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_train_pred = clf.predict(X_train)
	
	train_scores[i] = accuracy_score(y_train,y_train_pred)
	test_scores[i] = accuracy_score(y_test,y_pred)
	print("\nIteration {0}".format(i+1))
	print("Training accuracy: {0:0.1f}%".format(train_scores[i]*100))
	print("Accuracy: {0:0.1f}%".format(test_scores[i]*100))
print("\nTraining mean score: {0:0.2f}%".format(np.mean(train_scores)*100))
print("Tests mean score: {0:0.2f}%".format(np.mean(test_scores)*100))