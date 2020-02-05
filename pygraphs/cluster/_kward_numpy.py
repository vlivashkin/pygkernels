import numpy as np


class _KWardCluster:
    def __init__(self, id, nodes, all_nodes_count):
        self.id = id
        self.nodes = nodes
        self.n = len(nodes)
        self.h = np.zeros((all_nodes_count,), dtype=np.float32)
        self.h[nodes] = 1.0 / self.n


def _calcΔJ(Ck: _KWardCluster, Cl: _KWardCluster, K):
    """
    ΔJ = (n_k * n_l)/(n_k + n_l) * (h_k - h_l)^T * K * (h_k - h_l)
    """
    hkhl = (Ck.h - Cl.h)[:, None]
    currentΔJ = (Ck.n * Cl.n) / (Ck.n + Cl.n) * hkhl.T.dot(K).dot(hkhl)
    return currentΔJ


def predict(K, n_clusters, device=None):
    n_nodes = K.shape[0]
    C = [_KWardCluster(i, [i], n_nodes) for i in range(n_nodes)]
    cacheΔJ = {}
    for i in range(n_nodes - n_clusters):
        np.random.shuffle(C)
        # find pair of clusters to merge
        minΔJ_k, minΔJ_l, minΔJ_val = None, None, np.inf
        for k in range(1, len(C)):
            for l in range(k + 1, len(C)):
                key = (C[k].id, C[l].id)
                if key not in cacheΔJ:
                    cacheΔJ[key] = _calcΔJ(C[k], C[l], K).item()
                ΔJ = cacheΔJ[key]
                if ΔJ < minΔJ_val:
                    minΔJ_k, minΔJ_l, minΔJ_val = k, l, ΔJ

        # merge clusters
        union = C[minΔJ_k].nodes + C[minΔJ_l].nodes
        C.append(_KWardCluster(C[-1].id + 1, union, n_nodes))
        del C[minΔJ_l]
        del C[minΔJ_k]

    result = np.zeros((n_nodes,), dtype=np.uint8)
    for cluster_idx, cluster in enumerate(C):
        for node in cluster.nodes:
            result[node] = cluster_idx
    return result
