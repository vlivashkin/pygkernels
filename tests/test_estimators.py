import unittest

from tqdm import tqdm

from pygraphs.cluster import KWard, SpectralClustering_rubanov, KMeans_sklearn, Ward_sklearn, \
    KKMeans, KKMeans_iterative
from pygraphs.graphs import Datasets
from pygraphs.measure import logComm_H


@unittest.skip
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

    def test_estimators_news_2cl(self):
        graphs, Gs, info = Datasets().news_2cl_1
        (A, gt), G = graphs[0], Gs[0]
        K = logComm_H(A).get_K(0.5)

        for estimator in tqdm(self.estimators):
            km = estimator(n_clusters=2)
            km.predict(K, G=G)

    def test_estimators_news_3cl(self):
        graphs, Gs, info = Datasets().news_3cl_1
        (A, gt), G = graphs[0], Gs[0]
        K = logComm_H(A).get_K(0.5)

        for estimator in tqdm(self.estimators):
            km = estimator(n_clusters=3)
            km.predict(K, G=G)
