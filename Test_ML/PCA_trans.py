from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
import time


class PCA_preprocessing:
    def __init__(self ,X, n):
        self.n=n
        self.X=X
        # PCA
        self.pca = PCA(n_components=n)
        self.PCA_time = time.time()
        self.X_PCA = self.pca.fit_transform(X)
        self.PCA_time = time.time() - self.PCA_time

        # IncrementalPCA
        self.n_batches = 2
        self.inc_pca = IncrementalPCA(n_components=n)
        self.IPCA_time = time.time()
        for X_batch in np.array_split(X, self.n_batches):
            self.inc_pca.partial_fit(X_batch)
        self.X_IPCA = self.inc_pca.transform(X)
        self.IPCA_time = time.time() - self.IPCA_time

        # Randomized PCA
        self.rnd_pca = PCA(n_components=n, svd_solver='randomized')
        self.RPCA_time = time.time()
        self.X_RPCA = self.rnd_pca.fit_transform(X)
        self.RPCA_time = time.time() - self.RPCA_time

    def getTime(self):
        print('PCA fit_transform time : ', self.PCA_time)
        print('IPCA fit_transform time : ', self.IPCA_time)
        print('IPCA fit_transform time : ', self.RPCA_time)

    def getX(self):
        return self.X_PCA , self.X_IPCA , self.X_RPCA

def PCA_print(X,y,n):
    pca = PCA_preprocessing(X, n)
    X_PCA, X_IPCA, X_RPCA = pca.getX()
    from modeling import printAll
    print('<PCA n_components = {}>'.format(n))
    printAll(X_PCA,y)
    print('<IPCA n_components = {}>'.format(n))
    printAll(X_IPCA, y)
    print('<RPCA n_components = {}>'.format(n))
    printAll(X_RPCA, y)

