from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import pickle

features = pickle.load(open('features.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))