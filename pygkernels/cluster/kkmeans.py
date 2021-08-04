import traceback
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pygkernels.cluster import _kkmeans_pytorch as _backend
from pygkernels.cluster.base import KernelEstimator


class KMeans_Fouss(KernelEstimator, ABC):
    EPS = 10 ** -10
    INIT_NAMES = ['one', 'all', 'k-means++']

    def __init__(self, n_clusters, n_init=10, max_rerun=5, max_iter=100, init='k-means++', init_measure='modularity',
                 random_state=42, device=None):
        super().__init__(n_clusters, random_state=random_state, device=device)

        self.n_init = n_init
        self.max_rerun = max_rerun
        self.max_iter = max_iter
        self.init = init
        self.init_measure = init_measure

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def _init_simple(self, K, init):
        n = K.shape[0]
        q_idx = np.arange(n)
        np.random.shuffle(q_idx)

        h = np.zeros((self.n_clusters, n), dtype=np.float64)
        if init == 'one':  # one: choose one node for each cluster
            for i in range(self.n_clusters):
                h[i, q_idx[i]] = 1.
        elif init == 'all':  # all: choose (almost) all nodes to clusters
            nodes_per_cluster = n // self.n_clusters
            for i in range(self.n_clusters):
                for j in range(i * nodes_per_cluster, (i + 1) * nodes_per_cluster):
                    h[i, q_idx[j]] = 1. / nodes_per_cluster
        else:
            raise NotImplementedError()
        return h

    def _init_h(self, K: np.array, init: str):
        if init in ['one', 'all']:
            h = self._init_simple(K, init=init)
        elif init == 'k-means++':
            h = _backend.kmeanspp(K, self.n_clusters, device=self.device)
        else:
            raise NotImplementedError()
        return h

    def _choose_measure_to_detect_best_trial(self, inertia, modularity):
        if self.init_measure == 'inertia':
            quality = -inertia
        elif self.init_measure == 'modularity':
            quality = modularity
        else:
            raise NotImplementedError(f'wrong init_measure: {self.init_measure}')
        return quality

    def _predict_successful_once(self, K: np.array, init_idx: int, init: str, A: Optional[np.array] = None):
        np.random.seed(self.random_state + init_idx)
        labels, inertia, modularity = None, np.nan, np.nan
        for _ in range(self.max_rerun):
            try:
                K = K.astype(np.float64)
                labels, inertia, modularity, success = self._predict_once(K, init, A=A)
                if success:
                    quality = self._choose_measure_to_detect_best_trial(inertia, modularity)
                    return labels, quality, inertia, modularity
            except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
                print(f'trial: {e}')
                traceback.print_exc()
                # pass
        # case if all reruns were unsuccessful
        print(f'all reruns were unsuccessful')
        quality = self._choose_measure_to_detect_best_trial(inertia, modularity)
        return labels, quality, inertia, modularity

    @abstractmethod
    def _predict_once(self, K: np.array, init: str, A: Optional[np.array] = None):
        pass

    def predict(self, K, explicit=False, A: Optional[np.array] = None):
        if A is not None:
            A = A.astype(np.float32)

        inits, best_labels, best_quality = [], None, np.inf
        init_names = self.INIT_NAMES if self.init == 'any' else [self.init]
        for init in init_names:
            results = [self._predict_successful_once(K, i, init, A=A) for i in range(self.n_init)]
            for labels, quality, inertia, modularity in results:
                if explicit:
                    inits.append({
                        'labels': labels,
                        'inertia': inertia,
                        'modularity': modularity,
                        'init': init
                    })
                else:
                    if quality > best_quality or best_labels is None:
                        best_quality, best_labels = quality, labels
        return inits if explicit else best_labels


class KKMeans(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.2: Simple kernel k-means clustering of nodes
    """

    name = 'KKMeans'

    def _predict_once(self, K: np.array, init: str, A: Optional[np.array] = None):
        h_init = self._init_h(K, init)
        labels, inertia, modularity, is_ok = _backend.predict(K, h_init, self.max_iter, A, device=self.device)
        return labels, inertia, modularity, is_ok


class KKMeans_iterative(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.3: Simple iterative kernel k-means clustering of nodes
    """

    name = 'KKMeans_iterative'

    def _predict_once(self, K: np.array, init: str, A: Optional[np.array] = None):
        h_init = self._init_h(K, init)
        labels, inertia, modularity, is_ok = _backend.iterative_predict(K, h_init, self.max_iter, self.EPS, A,
                                                                        device=self.device)
        return labels, inertia, modularity, is_ok
