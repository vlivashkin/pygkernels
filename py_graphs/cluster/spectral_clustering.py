import numpy as np
import sklearn.cluster
from sklearn.base import ClusterMixin, BaseEstimator


class SpectralClustering(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        X = self._max_ort(K)
        X = self._sign_flip(X)
        cls = sklearn.cluster.KMeans(n_clusters=self.n_clusters)
        prd = cls.fit_predict(X)
        return prd

    def _max_ort(self, M):
        val, vec = np.linalg.eig(M)
        ind = np.argpartition(val, -self.n_clusters)[-self.n_clusters:]
        X = np.asmatrix(vec[:, ind])
        return X

    def _sign_flip(self, X):
        max_pos = np.argmax(np.abs(np.asarray(X)), axis=0)
        sgns = np.sign(np.asarray(X)[max_pos, range(X.shape[1])])
        S = np.asmatrix(np.diag(sgns))
        return np.asmatrix(X) * S


