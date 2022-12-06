from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from pygkernels.cluster.base import KernelEstimator


class SpectralClustering_rubanov(KernelEstimator):
    name = "SpectralClustering_rubanov"

    def __init__(self, n_clusters, n_init=10, random_state=None):
        super().__init__(n_clusters)
        self.n_init = n_init
        self.random_state = random_state

    def _max_ort(self, M):
        val, vec = np.linalg.eig(M)
        ind = np.argpartition(val, -self.n_clusters)[-self.n_clusters :]
        return vec[:, ind]

    def _sign_flip(self, X):
        max_pos = np.argmax(np.abs(X), axis=0)
        sgns = np.sign(X[max_pos, range(X.shape[1])])
        S = np.diag(sgns)
        return X.dot(S)

    def predict(self, K, A: Optional[np.array] = None):
        X = self._max_ort(K)
        X = self._sign_flip(X)
        cls = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
        prd = cls.fit_predict(X)
        return prd
