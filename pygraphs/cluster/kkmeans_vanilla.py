import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, deprecated

from pygraphs.graphs import Datasets


# Francois Fouss, Marco Saerens, Masashi Shimbo
# Algorithms and Models for Network Data and Link Analysis
# Algorithm 7.2
@deprecated()
class KKMeansVanilla(BaseEstimator, ClusterMixin):
    name = 'VanillaKernelKMeans'

    def __init__(self, n_clusters=3, max_iter=100, random_state=0):
        self.m = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        n = K.shape[0]
        U = np.zeros((n, self.m))

        # initialization
        rs = check_random_state(self.random_state)
        q_idx = rs.randint(0, n, size=(self.m,))
        h = np.zeros((self.m, n))
        for i in range(self.m):
            h[i][q_idx[i]] = 1
        e = np.eye(n)
        nn = np.zeros((self.m,))

        for i in range(100):
            U = np.zeros((n, self.m))
            for i in range(0, n):
                ka = np.argmin([(h[k] - e[i])[None].dot(K).dot((h[k] - e[i])[None].T) for k in range(0, self.m)])
                U[i][ka] = 1
            for k in range(0, self.m):
                nn[k] = np.sum([U[i][k] for i in range(0, n)])
                h[k] = U[:, k] / nn[k]

        return np.argmax(U, axis=1)


if __name__ == '__main__':
    graph, info = Datasets().news_2cl_1
    X, y = graph[0]
    print(y)

    km = KKMeansVanilla(n_clusters=2, max_iter=100, random_state=42)
    print(km.fit_predict(X))
    print(km.predict(X))
