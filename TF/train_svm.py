import argparse
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

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
print("Training accuracy: {0:0.1f}%".format(accuracy_score(y_train,y_train_pred)*100))
print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))