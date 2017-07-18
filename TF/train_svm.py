import argparse
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import pickle
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

train_scores = []
test_scores = []

for i in range(30):
	#X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
	train_d,test_d = d.split_stratified()

	clf = LinearSVC(C=1, loss='squared_hinge', penalty='l2',multi_class='ovr')
	clf.fit(train_d.x, train_d.y)
	y_pred = clf.predict(test_d.x)
	y_train_pred = clf.predict(train_d.x)
	
	train_scores.append(accuracy_score(train_d.y,y_train_pred))
	test_scores.append(accuracy_score(test_d.y,y_pred))
	print("\nIteration {0}".format(i+1))
	print("Training accuracy: {0:0.1f}%".format(train_scores[i]*100))
	print("Accuracy: {0:0.1f}%".format(test_scores[i]*100))
print("\nTraining mean score: {0:0.2f}%".format(np.mean(train_scores)*100))
print("Tests mean score: {0:0.2f}%".format(np.mean(test_scores)*100))
