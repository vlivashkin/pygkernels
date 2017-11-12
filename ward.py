import numpy as np
import sys


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


class Ward:
    def predict(self, K: np.matrixlib.defmatrix.matrix, clusters_count):
        clusters = [Cluster([i], K.shape[0]) for i in range(K.shape[0])]
        for i in range(K.shape[0] - clusters_count):
            self.iteration(K, clusters)

        result = np.zeros((K.shape[0], ))
        for cluster_idx, cluster in enumerate(clusters):
            for node in cluster.nodes:
                result[node] = cluster_idx
        return result

    def iteration(self, K,  clusters):
        minCk, minCl, minΔJ = None, None, sys.float_info.max
        for Ck_idx, Ck in enumerate(clusters):
            for Cl_idx, Cl in enumerate(clusters[Ck_idx + 1:]):
                currentΔJ = Ck.getΔJ(K, Cl)
                if currentΔJ < minΔJ:
                    minCk, minCl, minΔJ = Ck, Cl, currentΔJ
        self.merge(K, clusters, minCk, minCl)

    def merge(self, K, clusters, Ck,  Cl):
        union = Ck.nodes + Cl.nodes
        clusters.remove(Ck)
        clusters.remove(Cl)
        clusters.append(Cluster(union, K.shape[0]))
