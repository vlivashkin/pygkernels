import random
from itertools import combinations

import numpy as np
import torch

from pygkernels.cluster.base import torch_func


def calc_cache_batch(Ck_n, Cl_n, Ck_h, Cl_h, K, device):
    """
    dJ = (n_k * n_l)/(n_k + n_l) * (h_k - h_l)^T * K * (h_k - h_l)
    """
    Ck_n = torch.from_numpy(np.array(Ck_n)).to(device)
    Cl_n = torch.from_numpy(np.array(Cl_n)).to(device)
    Ck_h, Cl_h = torch.stack(Ck_h, dim=0), torch.stack(Cl_h, dim=0)

    hkhl = Ck_h - Cl_h
    dJ = (Ck_n * Cl_n) * torch.einsum("ki,ij,kj->k", hkhl, K, hkhl) / (Ck_n + Cl_n)
    dJ = dJ.cpu().numpy()

    return dJ


def _calc_cache_batched(Ck_n_all, Cl_n_all, Ck_h_all, Cl_h_all, K, device, batch_size=100000):
    num_pages = len(Ck_n_all) // batch_size + 1

    if num_pages > 1:
        dJ_all = []
        for p in range(num_pages):
            slce = slice(p * batch_size, (p + 1) * batch_size)
            c2_Ck_n, c2_Cl_n, c2_Ck_h, c2_Cl_h = Ck_n_all[slce], Cl_n_all[slce], Ck_h_all[slce], Cl_h_all[slce]
            c2_dJ = calc_cache_batch(c2_Ck_n, c2_Cl_n, c2_Ck_h, c2_Cl_h, K, device)
            dJ_all.append(c2_dJ)
        dJ_all = np.concatenate(dJ_all, axis=0)
    else:
        dJ_all = calc_cache_batch(Ck_n_all, Cl_n_all, Ck_h_all, Cl_h_all, K, device)

    return dJ_all


class _KWardCluster:
    def __init__(self, id, nodes, all_nodes_count, device):
        self.id = id
        self.nodes = nodes
        self.n = len(nodes)
        self.h = torch.zeros((all_nodes_count,), dtype=torch.float32).to(device)
        self.h[nodes] = 1.0 / self.n


@torch_func
def predict(K, n_clusters, device):
    n_nodes = K.shape[0]
    C = dict((i, _KWardCluster(i, [i], n_nodes, device)) for i in range(n_nodes))
    last_taken_id = n_nodes - 1

    # initialize cache
    cache_dJ = {}
    c2_key, c2_Ck_n, c2_Cl_n, c2_Ck_h, c2_Cl_h = [], [], [], [], []
    for Ck, Cl in combinations(C.values(), 2):
        c2_key.append((Ck.id, Cl.id))
        c2_Ck_n.append(Ck.n)
        c2_Cl_n.append(Cl.n)
        c2_Ck_h.append(Ck.h)
        c2_Cl_h.append(Cl.h)
    c2_dJ = _calc_cache_batched(c2_Ck_n, c2_Cl_n, c2_Ck_h, c2_Cl_h, K, device)
    for key, c2_dJ_item in zip(c2_key, c2_dJ):
        cache_dJ[key] = c2_dJ_item
    del c2_Cl_n, c2_Cl_h  # it should help to free cuda memory of taken by deleted clusters

    for iter in range(n_nodes - n_clusters):
        if iter > 0:
            # update cache: add to cache all pairs not in cache
            c2_key, c2_Ck_n, c2_Ck_h = [], [], []
            for Ck in C.values():
                if Ck.id != last_taken_id:
                    c2_key.append((Ck.id, last_taken_id))
                    c2_Ck_n.append(Ck.n)
                    c2_Ck_h.append(Ck.h)
            Cl = C[last_taken_id]
            c2_dJ = _calc_cache_batched(c2_Ck_n, [Cl.n], c2_Ck_h, [Cl.h], K, device)
            for key, c2_dJ_item in zip(c2_key, c2_dJ):
                cache_dJ[key] = c2_dJ_item

        # find pair of clusters to merge
        minJ_k, minJ_l, minJ_val = None, None, np.inf
        all_ids = list(C.keys())
        random.shuffle(all_ids)
        for k, l in combinations(all_ids, 2):
            dJ = cache_dJ[(k, l) if k < l else (l, k)]
            if dJ < minJ_val:
                minJ_k, minJ_l, minJ_val = k, l, dJ

        # merge clusters
        union = C[minJ_k].nodes + C[minJ_l].nodes
        del C[minJ_k]
        del C[minJ_l]
        new_id = last_taken_id + 1
        C[new_id] = _KWardCluster(last_taken_id + 1, union, n_nodes, device)
        last_taken_id = new_id

        # check that nothing lost
        # node_idx = defaultdict(lambda: 0)
        # for Ci in C:
        #     for node in Ci.nodes:
        #         node_idx[node] += 1
        # for i in range(n_nodes):
        #     assert node_idx[i] == 1

    result = np.zeros((n_nodes,), dtype=np.uint8)
    for cluster_idx, cluster in enumerate(C.values()):
        for node in cluster.nodes:
            result[node] = cluster_idx

    return result
