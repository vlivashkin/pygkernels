import numpy as np
import torch


def torch_func(func):
    def wrapper(*args, device=1, **kwargs):
        with torch.no_grad():
            args = [torch.from_numpy(x).float().to(device)
                    if type(x) == np.ndarray and x.dtype in [np.float32, np.float64] else x
                    for x in args]
            results = func(*args, **kwargs)
            results = [x.numpy() if type(x) == torch.tensor else x for x in results]
        return results

    return wrapper


def _hKh(hk, ei, K):
    hk_ei = (hk - ei).unsqueeze(dim=0)
    return hk_ei.mm(K).mm(hk_ei.transpose(0, 1))[0, 0]


def _inertia(h, e, K, labels):
    n = K.shape[0]
    return np.array([_hKh(h[labels[i]], e[i], K) for i in range(0, n)]).sum()


@torch_func
def _vanilla_predict(K, h, max_iter: int, device=1):
    n_clusters, n = h.shape
    e = torch.eye(n, dtype=torch.float32).to(device)

    labels, success = np.zeros((n,), dtype=np.uint8), True
    for _ in range(max_iter):
        # fix h, update U
        U = torch.zeros((n, n_clusters), dtype=torch.float32).to(device)
        l = np.zeros((n,), dtype=np.uint8)
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
        nn = torch.unsqueeze(torch.sum(U, dim=0), dim=0)
        if torch.any(nn == 0):  # empty cluster! exit with success=False
            success = False
            break
        h = (U / nn).transpose(0, 1)

    inertia = _inertia(h, e, K, labels)
    return labels, inertia, success


@torch_func
def _iterative_predict(K, h, U, l, nn, max_iter: int, eps: float, device=1):
    n_clusters, n = h.shape
    e = torch.eye(n, dtype=torch.float32).to(device)

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
                    inertia = _inertia(h, e, K, labels)
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

    inertia = _inertia(h, e, K, labels)
    return labels, inertia, ~np.isnan(inertia)
