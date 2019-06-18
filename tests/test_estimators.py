import unittest

from pygraphs.cluster import KKMeans, KKMeans_vanilla, KWard, SpectralClustering_rubanov, KMeans_sklearn, Ward_sklearn
from pygraphs.graphs import Datasets
from pygraphs.measure import logComm_K


class TestEstimators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()
        self.estimators = [
            KKMeans,
            KKMeans_vanilla,
            KWard,
            SpectralClustering_rubanov,
            KMeans_sklearn,
            Ward_sklearn,
            # KKMeansKernlab,
            # SpectralClusteringKernlab
        ]

    def test_estimators_news_2cl(self):
        graph, info = Datasets().news_2cl_1
        A, gt = graph[0]
        K = logComm_K(A).get_K(0.5)

        for estimator in self.estimators:
            km = estimator(n_clusters=2)
            km.fit_predict(K)
            km.predict(K)

    def test_estimators_news_3cl(self):
        graph, info = Datasets().news_3cl_1
        A, gt = graph[0]
        K = logComm_K(A).get_K(0.5)

        for estimator in self.estimators:
            km = estimator(n_clusters=3)
            km.fit_predict(K)
            km.predict(K)