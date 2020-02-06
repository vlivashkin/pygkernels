import numpy as np


def _hKh(hk: np.array, ei: np.array, K: np.array):
    hk_ei = hk - ei
    return np.einsum('i,ij,j->', hk_ei, K, hk_ei)


def _inertia(h: np.array, e, K: np.array, labels: np.array):
    h_e = h[labels] - e
    return np.einsum('ij,jk,ki->', h_e, K, h_e)


def kmeanspp(K, n_clusters, device=None):
    n = K.shape[0]
    e = np.eye(n, dtype=np.float64)
    h = np.zeros((n_clusters, n), dtype=np.float64)

    first_centroid = np.random.randint(n)
    h[0, first_centroid] = 1
    for c_idx in range(1, n_clusters):
        h_e = h[:, None] - e[None]  # [k, n, n]
        min_distances = np.min(np.einsum('kni,ij,knj->kn', h_e, K, h_e), axis=0)
        min_distances = np.power(min_distances, 2)
        # next_centroid = np.argmax(min_distances)

        if np.sum(min_distances) > 0:
            next_centroid = np.random.choice(range(n), p=min_distances / np.sum(min_distances))
        else:  # no way to make all different centroids; let's choose random one just for rerun
            next_centroid = np.random.choice(range(n))
        h[c_idx, next_centroid] = 1
    return h


def vanilla_predict(K: np.array, h: np.array, max_iter: int, device=None):
    n_clusters, n = h.shape
    e = np.eye(n, dtype=np.float64)

    labels, success = np.zeros((n,), dtype=np.uint8), True
    for iter in range(max_iter):
        h_e = h[:, None] - e[None]  # [k, n, n]
        l = np.argmin(np.einsum('kni,ij,knj->kn', h_e, K, h_e), axis=0)
        if np.all(labels == l):  # early stop
            break
        labels = l

        U = np.zeros((n, n_clusters), dtype=np.float64)
        U[range(n), labels] = 1
        nn = U.sum(axis=0, keepdims=True)
        if np.any(nn == 0):  # empty cluster! exit with success=False
            success = False
            break
        h = (U / nn).T

    inertia = _inertia(h, e, K, labels)
    return labels, inertia, success


def iterative_predict(K: np.array, h: np.array, max_iter: int, eps: float):
    n_clusters, n = h.shape
    e = np.eye(n, dtype=np.float64)

    # initialization
    h_e = h[:, None] - e[None]  # [k, n, n]
    l = np.argmin(np.einsum('kni,ij,knj->kn', h_e, K, h_e), axis=0)

    U = np.zeros((n, n_clusters), dtype=np.float64)
    U[range(n), l] = 1
    nn = U.sum(axis=0, keepdims=True)
    if np.any(nn == 0):  # bad start, rerun
        inertia = _inertia(h, e, K, l)
        return l, inertia, False
    h = (U / nn).T

    # iterative steps
    labels = l.copy()
    for _ in range(max_iter):
        node_order = np.array(list(range(n)))
        np.random.shuffle(node_order)
        for i in node_order:  # for each node
            h_ei = h - e[i][None]
            ΔJ1 = nn / (nn + 1 + eps) * np.einsum('ki,ij,kj->k', h_ei, K, h_ei)
            k_star = np.argmin(ΔJ1)
            ΔJ2 = nn[l[i]] / (nn[l[i]] - 1 + eps) * _hKh(h[l[i]], e[i], K)
            minΔJ = ΔJ1[k_star] - ΔJ2
            if minΔJ < 0 and l[i] != k_star:
                if nn[l[i]] == 1:  # it will cause empty cluster! exit with success=False
                    inertia = _inertia(h, e, K, labels)
                    return labels, inertia, False
                h[l[i]] = 1. / (nn[l[i]] - 1 + eps) * (nn[l[i]] * h[l[i]] - e[i])
                h[k_star] = 1. / (nn[k_star] + 1 + eps) * (nn[k_star] * h[k_star] + e[i])
                U[i, l[i]], U[i, k_star] = 0, 1
                nn[l[i]], nn[k_star] = nn[l[i]] - 1, nn[k_star] + 1
                l[i] = k_star

        # early stop
        if np.all(labels == l):  # nothing changed
            break
        labels = l.copy()

    inertia = _inertia(h, e, K, labels)
    return labels, inertia, ~np.isnan(inertia)
