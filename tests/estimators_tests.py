import unittest

from graphs import sample
from cluster import KernelKMeans, Ward, SpectralClustering


class EstimatorsTests(unittest.TestCase):
    def test_simple_Ward(self):
        y_pred = Ward(2).predict(sample.diploma_matrix)
        print(y_pred)

    def test_all_estimators(self):
        K = sample.diploma_matrix

        y_pred_kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0).predict(K)
        y_pred_ward = Ward(n_clusters=2).predict(K)
        y_pred_spectral = SpectralClustering(n_clusters=2).predict(K)
        print('KMeans:', y_pred_kmeans)
        print('Ward:', y_pred_ward)
        print('Spectral Clustering:', y_pred_spectral)

    if __name__ == '__main__':
        unittest.main()