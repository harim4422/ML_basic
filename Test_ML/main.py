#dataset
import pandas as pd
wine = pd.read_csv('./datasets/wine/wine.csv')
from pre_processing import pre
X,y =pre(wine)

#original data
from modeling import printAll
print('<< model : Original data >>')
printAll(X,y)

#Reduce features using PCA
from sklearn.decomposition import PCA
pca_2 = PCA(n_components=2)
pca_3 = PCA(n_components=3)
pca_5 = PCA(n_components=5)
X_2 = pca_2.fit_transform(X)
X_3 = pca_3.fit_transform(X)
X_5 = pca_5.fit_transform(X)

#feature 2
print('<< model : PCA - feature 2 >>')
printAll(X_2,y)
#feature 3
print('<< model : PCA - feature 3 >>')
printAll(X_3,y)
#feature 5
print('<< model : PCA - feature 5 >>')
printAll(X_5,y)