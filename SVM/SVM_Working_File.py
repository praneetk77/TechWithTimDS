import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

classes=['malignant','benign']

clf = svm.SVC(kernel="linear",C=2)
knn = KNeighborsClassifier(n_neighbors=24)
clf.fit(x_train,y_train)
knn.fit(x_train,y_train)

y_prediction_clf = clf.predict(x_test)
y_prediction_knn = knn.predict(x_test)

acc_clf = metrics.accuracy_score(y_test,y_prediction_clf)
acc_knn = metrics.accuracy_score(y_test, y_prediction_knn)
print(acc_clf, acc_knn)

