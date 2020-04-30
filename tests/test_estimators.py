import unittest
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from pygkernels import score
from pygkernels.cluster import KWard, SpectralClustering_rubanov, KMeans_sklearn, Ward_sklearn, \
    KKMeans, KKMeans_iterative
from pygkernels.cluster import _kmeans_pytorch
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

    def _calc_modularity_slow(self, G: nx.Graph, labels):
        communities = defaultdict(list)
        for idx, label in enumerate(labels):
            communities[label].append(idx)
        communities = list(communities.values())
        return nx.community.modularity(G, communities)

    def test_modularity(self):
        graphs, Gs, info = Datasets().polbooks
        (A, y_true), G = graphs[0], Gs[0]
        y_true_better = np.zeros((len(y_true),))
        mapping = dict([(class_name, i) for i, class_name in enumerate(set(y_true))])
        for i, item in enumerate(y_true):
            y_true_better[i] = mapping[item]

        modularity_nx = self._calc_modularity_slow(G, y_true_better)
        ours_nx = score.modularity(A, y_true_better)
        ours2_nx = _kmeans_pytorch._modularity(torch.from_numpy(A).float(), torch.from_numpy(y_true_better).int())
        self.assertTrue(np.isclose(modularity_nx, ours_nx, atol=0.0001))
        self.assertTrue(np.isclose(modularity_nx, ours2_nx, atol=0.0001))
