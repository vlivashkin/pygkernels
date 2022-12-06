import unittest
from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from pygkernels import score
from pygkernels.cluster import (
    KWard,
    SpectralClustering_rubanov,
    KMeans_sklearn,
    Ward_sklearn,
    KKMeans,
    KKMeans_iterative,
)
from pygkernels.cluster import _kkmeans_pytorch
from pygkernels.data import Datasets, LFRGenerator
from pygkernels.measure import logComm_H


class TestEstimators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators = [
            # KKMeans,
            KKMeans,
            KKMeans_iterative,
            # KKMeans_kernlab,
            KWard,
            SpectralClustering_rubanov,
            # SpectralClustering_sklearn,
            # SpectralClustering_kernlab,
            KMeans_sklearn,
            Ward_sklearn,
        ]

    def test_estimators_news_3cl(self):
        (A, gt), info = Datasets()["football"]
        K = logComm_H(A).get_K(0.5)

        for estimator in tqdm(self.estimators):
            km = estimator(n_clusters=info["k"])
            km.predict(K, A=A)

    @staticmethod
    def _calc_nx_modularity(G: nx.Graph, labels: Dict):
        """
        NetworkX modularity; it's slow
        """
        communities = defaultdict(list)
        for name, label in labels.items():
            communities[label].append(name)
        communities = list(communities.values())
        return nx.community.modularity(G, communities)

    def test_modularity_datasets(self):
        (A, gt), _ = Datasets().polbooks
        G = nx.from_numpy_array(A)
        nx.set_node_attributes(G, dict(enumerate(gt)), "gt")

        modularity_nx = self._calc_nx_modularity(G, nx.get_node_attributes(G, "gt"))

        mod_ours_numpy1 = score.modularity(A, gt)
        mod_ours_numpy2 = score.modularity(A, gt)
        mod_ours_torch = _kkmeans_pytorch._modularity(torch.from_numpy(A).float(), torch.from_numpy(gt).int())
        self.assertTrue(np.isclose(modularity_nx, mod_ours_numpy1, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, mod_ours_numpy2, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, mod_ours_torch, atol=0.0001))

    def test_modularity_LFR(self):
        A, gt = LFRGenerator(200, 5, 5, 0.7, min_degree=5).generate_graph()
        A, gt = np.array(A), np.array(gt)
        np.fill_diagonal(A, 0)

        G = nx.from_numpy_array(A)
        nx.set_node_attributes(G, dict(enumerate(gt)), "gt")
        modularity_nx = self._calc_nx_modularity(G, nx.get_node_attributes(G, "gt"))

        mod_ours_numpy1 = score.modularity(A, gt)
        mod_ours_numpy2 = score.modularity2(A, gt)
        mod_ours_torch = _kkmeans_pytorch._modularity(torch.from_numpy(A).float(), torch.from_numpy(gt).int())
        self.assertTrue(np.isclose(modularity_nx, mod_ours_numpy1, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, mod_ours_numpy2, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, mod_ours_torch, atol=0.0001))
