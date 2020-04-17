import numpy as np
import torch

from pygraphs.cluster.base import torch_func


def _hKh(hk, ei, K):
    hk_ei = hk - ei
    return torch.einsum('i,ij,j->', [hk_ei, K, hk_ei])


def _inertia(h, e, K, labels):
    h_e = h.gather(0, labels[None]) - e
    return torch.einsum('ki,ij,kj->', [h_e, K, h_e])


def _modularity(A, labels):
    """
    Simplified version only for undirected graphs
    """
    n_edges = torch.sum(A)
    degrees = torch.sum(A, dim=1, keepdim=True)

    Q_items = A + torch.diagonal(A) - degrees.mm(degrees.transpose(1, 0)) / n_edges
    Q = torch.Tensor([0])
    for class_name in range(torch.max(labels).item() + 1):
        mask = labels == class_name
        Q += torch.sum(Q_items[mask][:, mask])
    return Q / n_edges


@torch_func
def kmeanspp(K, n_clusters, device):
    """
    k-means++ initialization for k-means
    The method will work only if all the distances is finite
    """
    assert torch.all(~torch.isnan(K))

    n = K.shape[0]
    e = torch.eye(n, dtype=torch.float32).to(device)
    h = torch.zeros((n_clusters, n), dtype=torch.float32).to(device)

    first_centroid = np.random.randint(n)
    h[0, first_centroid] = 1
    for c_idx in range(1, n_clusters):
        h_e = h.unsqueeze(1) - e.unsqueeze(0)  # [k, n, n]
        min_distances, _ = torch.min(torch.einsum('kni,ij,knj->kn', [h_e, K, h_e]), dim=0)
        min_distances.pow_(2)
        if torch.sum(min_distances) > 0:
            next_centroid = np.random.choice(range(n), p=(min_distances / min_distances.sum()).cpu().numpy())
        else:  # no way to make all different centroids; let's choose random one just for rerun
            next_centroid = np.random.choice(range(n))
        h[c_idx, next_centroid] = 1
    return h


@torch_func
def predict(K, h, max_iter: int, A, device):
    n_clusters, n = h.shape
    e = torch.eye(n, dtype=torch.float32).to(device)

    labels, success = torch.zeros((n,), dtype=torch.int64).to(device), True
    for _ in range(max_iter):
        h_e = h.unsqueeze(1) - e.unsqueeze(0)  # [k, n, n]
        l = torch.einsum('kni,ij,knj->kn', [h_e, K, h_e]).argmin(dim=0)
        if torch.all(labels == l):  # early stop
            break
        labels = l

        U = torch.zeros((n, n_clusters), dtype=torch.float32).to(device)
        U[range(n), labels] = 1
        nn = U.sum(dim=0, keepdim=True)
        if torch.any(nn == 0):  # empty cluster! exit with success=False
            success = False
            break
        h = (U / nn).transpose(0, 1)

    inertia = _inertia(h, e, K, labels)
    modularity = _modularity(A, labels) if A is not None else None
    return labels, inertia, modularity, success


@torch_func
def iterative_predict(K, h, max_iter: int, eps: float, A, device):
    n_clusters, n = h.shape
    e = torch.eye(n, dtype=torch.float32).to(device)

    # initialization
    h_e = h.unsqueeze(1) - e.unsqueeze(0)  # [k, n, n]
    l = torch.einsum('kni,ij,knj->kn', [h_e, K, h_e]).argmin(dim=0)

    U = torch.zeros((n, n_clusters), dtype=torch.float32).to(device)
    U[range(n), l] = 1
    nn = U.sum(dim=0, keepdim=True)
    if torch.any(nn == 0):  # bad start, rerun
        inertia = _inertia(h, e, K, l)
        return l, inertia, False
    h = (U / nn).transpose(0, 1)
    nn = nn.squeeze()

    # iterative steps
    labels = l.clone()
    for _ in range(max_iter):
        node_order = np.array(list(range(n)))
        np.random.shuffle(node_order)
        for i in node_order:  # for each node
            h_ei = h - e[i][None]
            ΔJ1 = nn / (nn + 1 + eps) * torch.einsum('ki,ij,kj->k', [h_ei, K, h_ei])
            ΔJ1, k_star = ΔJ1.min(dim=0)
            ΔJ2 = nn[l[i]] / (nn[l[i]] - 1 + eps) * _hKh(h[l[i]], e[i], K)
            minΔJ = ΔJ1 - ΔJ2
            if minΔJ < 0 and l[i] != k_star:
                if nn[l[i]] == 1:  # it will cause empty cluster! exit with success=False
                    inertia = _inertia(h, e, K, labels)
                    modularity = _modularity(A, labels) if A is not None else None
                    return labels, inertia, modularity, False
                h[l[i]] = 1. / (nn[l[i]] - 1 + eps) * (nn[l[i]] * h[l[i]] - e[i])
                h[k_star] = 1. / (nn[k_star] + 1 + eps) * (nn[k_star] * h[k_star] + e[i])
                U[i, l[i]], U[i, k_star] = 0, 1
                nn[l[i]], nn[k_star] = nn[l[i]] - 1, nn[k_star] + 1
                l[i] = k_star

        if torch.all(labels == l):  # early stop
            break
        labels = l.clone()

    inertia = _inertia(h, e, K, labels)
    modularity = _modularity(A, labels) if A is not None else None
    return labels, inertia, modularity, ~np.isnan(inertia)
