from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
import numpy as np
#dataset
import pandas as pd
wine = pd.read_csv('./datasets/wine/wine.csv')
from pre_processing import pre
X,y =pre(wine)

# #original data
# from modeling import printAll
# print('<< model : Original data >>')
# printAll(X,y)
#
# #Reduce features using PCA
# from PCA_trans import PCA_print
#
# #n_components = 2
# PCA_print(X,y,2)
# #n_components = 4
# PCA_print(X,y,4)
# #n_components = 6
# PCA_print(X,y,6)

def grid_kernelPCA(model,n):
    clf = Pipeline([
        ('kpca', KernelPCA(n_components=n)),
        ('reg',model)
    ])

    param = [{
        'kpca__gamma' : np.linspace(0.03,0.05,10),
        'kpca__kernel' : ['rbf','sigmoid']
    }]
    grid_search = GridSearchCV(clf, param_grid=param, cv=5)
    grid_search.fit(X,y)
    return grid_search
def getResult_grid(grid):
    cvres = grid.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)
    print('##########################################################')
    print('Best hyperparameter: {}'.format(grid.best_params_))

#n_components = 6
print('<SGD>')
grid_sgd_6 = grid_kernelPCA(SGDClassifier(max_iter=5, tol=-np.infty, random_state=42, loss='log'),6)
getResult_grid(grid_sgd_6)

print('<KNN>')
grid_knn_6 = grid_kernelPCA(KNeighborsClassifier(n_neighbors=2),6)
getResult_grid(grid_knn_6)

print('<Decision Tree>')
grid_tree_6 = grid_kernelPCA(DecisionTreeClassifier(max_depth=2, random_state=42),6)
getResult_grid(grid_tree_6)

print('<SVC>')
grid_svc_6 = grid_kernelPCA(SVC(gamma='auto', C=2, random_state=42, probability=True),6)
getResult_grid(grid_svc_6)