import sys

import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils import deprecated


class Cluster:
    def __init__(self, nodes, all_nodes_count):
        self.nodes = nodes
        self.n = len(nodes)
        self.h = np.zeros((all_nodes_count, 1))
        inverse_n = 1.0 / self.n
        for node in nodes:
            self.h[node, 0] = inverse_n
        self.ΔJ = {}

    def getΔJ(self, K, Cl):
        return self.ΔJ[Cl] if Cl in self.ΔJ else self.calcΔJ(K, Cl)

    # ΔJ = (n_k * n_l)/(n_k + n_l) * (h_k - h_l)^T * K * (h_k - h_l)
    def calcΔJ(self, K, Cl):
        hkhl = np.array(self.h - Cl.h).reshape((-1, 1))
        hkhlT = hkhl.T
        currentΔJ = (self.n * Cl.n) * hkhlT.dot(K).dot(hkhl)[0][0] / (self.n + Cl.n)
        self.ΔJ[Cl] = currentΔJ
        return currentΔJ


@deprecated()
class KWard(ClusterMixin, BaseEstimator):
    name = 'Ward'

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        clusters = [Cluster([i], K.shape[0]) for i in range(K.shape[0])]
        for i in range(K.shape[0] - self.n_clusters):
            self._iteration(K, clusters)

        result = np.zeros((K.shape[0],), dtype=np.int)
        for cluster_idx, cluster in enumerate(clusters):
            for node in cluster.nodes:
                result[node] = cluster_idx
        return result

    def _iteration(self, K, clusters):
        minCk, minCl, minΔJ = None, None, sys.float_info.max
        for Ck_idx, Ck in enumerate(clusters):
            for Cl_idx, Cl in enumerate(clusters[Ck_idx + 1:]):
                currentΔJ = Ck.getΔJ(K, Cl)
                if currentΔJ < minΔJ:
                    minCk, minCl, minΔJ = Ck, Cl, currentΔJ
        self._merge(K, clusters, minCk, minCl)

    def _merge(self, K, clusters, Ck, Cl):
        union = Ck.nodes + Cl.nodes
        clusters.remove(Ck)
        clusters.remove(Cl)
        clusters.append(Cluster(union, K.shape[0]))
