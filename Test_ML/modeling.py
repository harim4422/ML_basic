from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class model:

    def __init__(self,X_train,X_test,y_train,y_test):
        import time
        import numpy as np
        self.X = X_train
        self.X_test = X_test
        self.y = y_train
        self.y_test = y_test

        self.sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42, loss='log')
        self.sgd_time = time.time()
        self.sgd_clf.fit(self.X, self.y)
        self.sgd_time = time.time() - self.sgd_time


        self.knn_clf = KNeighborsClassifier(n_neighbors=2)
        self.knn_time = time.time()
        self.knn_clf.fit(self.X, self.y)
        self.knn_time = time.time() - self.knn_time


        self.tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        self.tree_time = time.time()
        self.tree_clf.fit(self.X, self.y)
        self.tree_time = time.time() - self.tree_time


        self.svm_clf = SVC(gamma='auto', C=2, random_state=42, probability=True)
        self.svc_time = time.time()
        self.svm_clf.fit(self.X, self.y)
        self.svc_time = time.time() - self.svc_time

    def getScore(self,model):
        from sklearn.metrics import accuracy_score
        y_score = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_score)
        return accuracy

    def printScore(self):
        print('SGD classifier Accuracy : {}'.format(self.getScore(self.sgd_clf)))
        print('KNN classifier Accuracy : {}'.format(self.getScore(self.knn_clf)))
        print('Decision Tree classifier Accuracy : {}'.format(self.getScore(self.tree_clf)))
        print('SVM classifier Accuracy : {}'.format(self.getScore(self.svm_clf)))
        print()

    def letTest(self):
        print('<Test data>')
        print('input data : ', self.X_test[13])
        print('output data : ', self.y_test[13])
        print('<Prediction>')
        print('SGD Prediction : ',self.sgd_clf.predict([self.X_test[13]]))
        print('KNN Prediction : ',self.knn_clf.predict([self.X_test[13]]))
        print('Decision Tree Prediction : ',self.tree_clf.predict([self.X_test[13]]))
        print('SVC Prediction : ',self.svm_clf.predict([self.X_test[13]]))
        print()

    def getTime(self):
        print('<Training Time>')
        print('SGD Classifier : {}'.format(self.sgd_time))
        print('knn Classifier : {}'.format(self.knn_time))
        print('tree Classifier : {}'.format(self.tree_time))
        print('SVC Classifier : {}'.format(self.svc_time))
        print()

def printAll(X,y):
    from pre_processing import split
    X_train, X_test, y_train, y_test = split(X, y)
    p_model = model(X_train, X_test, y_train, y_test)
    p_model.printScore()
    p_model.letTest()
    p_model.getTime()
    print('=====================================================================')
    print()