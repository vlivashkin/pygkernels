import unittest
from collections import defaultdict
from typing import Dict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from pygkernels import score
from pygkernels.cluster import KWard, SpectralClustering_rubanov, KMeans_sklearn, Ward_sklearn, \
    KKMeans, KKMeans_iterative
from pygkernels.cluster import _kkmeans_pytorch
from pygkernels.data import Datasets
from pygkernels.measure import logComm_H


class TestEstimators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()
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

    @unittest.skip
    def test_estimators_news_2cl(self):
        graphs, Gs, info = Datasets().news_2cl_1
        (A, gt), G = graphs[0], Gs[0]
        K = logComm_H(A).get_K(0.5)

        for estimator in tqdm(self.estimators):
            km = estimator(n_clusters=2)
            km.predict(K, G=G)

    @unittest.skip
    def test_estimators_news_3cl(self):
        graphs, Gs, info = Datasets().news_3cl_1
        (A, gt), G = graphs[0], Gs[0]
        K = logComm_H(A).get_K(0.5)

        for estimator in tqdm(self.estimators):
            km = estimator(n_clusters=3)
            km.predict(K, G=G)

    @staticmethod
    def _calc_modularity_slow(G: nx.Graph, labels: Dict):
        communities = defaultdict(list)
        for name, label in labels.items():
            communities[label].append(name)
        communities = list(communities.values())
        return nx.community.modularity(G, communities)

    def test_modularity(self):
        graphs, Gs, _ = Datasets().polbooks
        (A, y_true), G = graphs[0], Gs[0]
        modularity_nx = self._calc_modularity_slow(G, nx.get_node_attributes(G, 'gt'))

        class_mapping = dict([(class_name, idx) for idx, class_name in enumerate(set(y_true))])
        y_true_clean = np.array([class_mapping[item] for item in y_true])
        mod_ours_numpy = score.modularity(A, y_true_clean)
        mod_ours_torch = _kkmeans_pytorch._modularity(torch.from_numpy(A).float(), torch.from_numpy(y_true_clean).int())
        self.assertTrue(np.isclose(modularity_nx, mod_ours_numpy, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, mod_ours_torch, atol=0.0001))
