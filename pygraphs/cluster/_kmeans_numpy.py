import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _hKh(hk: np.array, ei: np.array, K: np.array):
    hk_ei = np.expand_dims((hk - ei), axis=0)
    return hk_ei.dot(K).dot(hk_ei.T)[0, 0]


@jit(nopython=True, cache=True)
def _inertia(h: np.array, K: np.array, labels: np.array):
    n = K.shape[0]
    e = np.eye(n, dtype=np.float64)
    return np.array([_hKh(h[labels[i]], e[i], K) for i in range(0, n)]).sum()


@jit(nopython=True, cache=True)
def _vanilla_predict(K: np.array, h: np.array, max_iter: int):
    n_clusters, n = h.shape
    e = np.eye(n, dtype=np.float64)

    labels, success = np.zeros((n,), dtype=np.uint8), True
    for _ in range(max_iter):
        # fix h, update U
        U, l = np.zeros((n, n_clusters), dtype=np.float64), np.zeros((n,), dtype=np.uint8)
        for i in range(n):
            k_star, min_inertia = -1, np.inf
            for k in range(n_clusters):
                inertia = _hKh(h[k], e[i], K)
                if inertia < min_inertia:
                    k_star, min_inertia = k, inertia
            U[i, k_star] = 1
            l[i] = k_star

        # early stop
        if np.all(labels == l):  # nothing changed
            break
        labels = l

        # fix U, update h
        nn = np.expand_dims(np.sum(U, axis=0), axis=0)
        if np.any(nn == 0):  # empty cluster! exit with success=False
            success = False
            break
        h = (U / nn).T

    inertia = _inertia(h, K, labels)
    return labels, inertia, success


@jit(nopython=True, cache=True)
def _iterative_predict(K: np.array, h: np.array, U: np.array, l: np.array, nn: np.array, max_iter: int, eps: float):
    n_clusters, n = h.shape
    e = np.eye(n, dtype=np.float64)

    labels = l.copy()
    for _ in range(max_iter):
        node_order = np.array(list(range(n)))
        np.random.shuffle(node_order)
        for i in node_order:  # for each node
            k_star, minΔJ = -1, np.inf
            for k in range(n_clusters):
                ΔJ1 = nn[k] / (nn[k] + 1 + eps) * _hKh(h[k], e[i], K)
                ΔJ2 = nn[l[i]] / (nn[l[i]] - 1 + eps) * _hKh(h[l[i]], e[i], K)
                ΔJ = ΔJ1 - ΔJ2
                if ΔJ < minΔJ:
                    minΔJ, k_star = ΔJ, k
            if minΔJ < 0 and l[i] != k_star:
                if nn[l[i]] == 1:  # it will cause empty cluster! exit with success=False
                    inertia = _inertia(h, K, labels)
                    return labels, inertia, False
                h[l[i]] = 1. / (nn[l[i]] - 1 + eps) * (nn[l[i]] * h[l[i]] - e[i])
                U[i, l[i]] = 0
                nn[l[i]] -= 1
                h[k_star] = 1. / (nn[k_star] + 1 + eps) * (nn[k_star] * h[k_star] + e[i])
                U[i, k_star] = 1
                nn[k_star] += 1
                l[i] = k_star

        # early stop
        if np.all(labels == l):  # nothing changed
            break
        labels = l.copy()

    inertia = _inertia(h, K, labels)
    return labels, inertia, ~np.isnan(inertia)
