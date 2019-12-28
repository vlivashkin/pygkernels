from abc import ABC, abstractmethod

import numpy as np

from pygraphs.cluster import _kmeans_numpy, _kmeans_pytorch
from pygraphs.cluster.base import KernelEstimator


class KMeans_Fouss(KernelEstimator, ABC):
    def __init__(self, n_clusters, n_init=10, max_rerun=100, max_iter=100, init='all', random_state=None,
                 backend='pytorch'):
        super().__init__(n_clusters)
        self.n_init = n_init
        self.max_rerun = max_rerun
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.eps = 10 ** -10
        self.backend = backend

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def _init_h(self, K: np.array):
        n = K.shape[0]
        e = np.eye(n, dtype=np.float64)

        q_idx = np.arange(n)
        np.random.shuffle(q_idx)

        h = np.zeros((self.n_clusters, n), dtype=np.float64)
        if self.init == 'one':  # init: choose one node for each cluster
            for i in range(self.n_clusters):
                h[i, q_idx[i]] = 1.
        elif self.init == 'all':  # init: choose (almost) all nodes to clusters
            nodes_per_cluster = n // self.n_clusters
            for i in range(self.n_clusters):
                for j in range(i * nodes_per_cluster, (i + 1) * nodes_per_cluster):
                    h[i, q_idx[j]] = 1. / nodes_per_cluster
        elif self.init == 'k-means++':
            first_centroid = np.random.randint(n)
            h[0, first_centroid] = 1
            for c_idx in range(1, self.n_clusters):
                min_distances = [np.min([_kmeans_numpy._hKh(h[k], e[i], K) for k in range(c_idx)]) for i in range(n)]
                min_distances = np.power(min_distances, 2)
                # next_centroid = np.argmax(min_distances)

                if np.sum(min_distances) > 0:
                    next_centroid = np.random.choice(range(n), p=min_distances / np.sum(min_distances))
                else:  # no way to make all different centroids; let's choose random one just for rerun
                    next_centroid = np.random.choice(range(n))

                h[c_idx, next_centroid] = 1
        else:
            raise NotImplementedError()
        return h

    def _predict_successful_once(self, K: np.array):
        for i in range(self.max_rerun):
            K = K.astype(np.float64)
            labels, inertia, success = self._predict_once(K)
            if success:
                return labels, inertia
        # print('reruns exceeded, take last result')
        return labels, inertia

    @abstractmethod
    def _predict_once(self, K: np.array):
        pass

    def predict(self, K):
        np.random.seed(self.random_state)
        best_labels, best_inertia = [], float('+inf')
        for i in range(self.n_init):
            labels, inertia = self._predict_successful_once(K)
            if inertia < best_inertia:
                best_labels = labels

        return best_labels


class KKMeans_vanilla(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.2: Simple kernel k-means clustering of nodes
    """

    name = 'KKMeans_vanilla'

    def _predict_once(self, K: np.array):
        h_init = self._init_h(K)

        if self.backend == 'numpy':
            return _kmeans_numpy._vanilla_predict(K, h_init, self.max_iter)
        elif self.backend == 'pytorch':
            return _kmeans_pytorch._vanilla_predict(K, h_init, self.max_iter)


class KKMeans_iterative(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.3: Simple iterative kernel k-means clustering of nodes
    """

    name = 'KKMeans'

    def _init_h_U_l_nn(self, K: np.array):
        n = K.shape[0]
        e = np.eye(n, dtype=np.float64)

        for i in range(self.max_iter):  # check all clusters used
            h = self._init_h(K)
            U, l = np.zeros((n, self.n_clusters), dtype=np.float64), np.zeros((n,), dtype=np.uint8)
            for i in range(n):
                k_star = np.array([_kmeans_numpy._hKh(h[k], e[i], K) for k in range(self.n_clusters)]).argmin()
                U[i][k_star] = 1
                l[i] = k_star
            nn = np.sum(U, axis=0, keepdims=True)
            if np.any(nn == 0):  # bad start, rerun
                continue
            h = (U / nn).T
            return h, U, l, nn[0], True
        return h, U, l, nn[0], False

    def _predict_once(self, K: np.array):
        h, U, l, nn, success_init = self._init_h_U_l_nn(K)
        if success_init:
            if self.backend == 'numpy':
                return _kmeans_numpy._iterative_predict(K, h, U, l, nn, self.max_iter, self.eps)
            elif self.backend == 'pytorch':
                return _kmeans_pytorch._iterative_predict(K, h, U, l, nn, self.max_iter, self.eps)
        else:  # no way to initialize properly
            labels = l.copy()
            inertia = _kmeans_numpy._inertia(h, K, labels)
            return labels, inertia, ~np.isnan(inertia)
