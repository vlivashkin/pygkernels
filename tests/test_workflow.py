import logging
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from pygkernels import util
from pygkernels.cluster import KKMeans, SpectralClustering_rubanov
from pygkernels.cluster.kward import KWard
from pygkernels.data import Samples, Datasets
from pygkernels.measure import kernels


class TestEstimators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()

    def test_simple_Ward(self):
        y_pred = KWard(2).predict(Samples.diploma_matrix)
        logging.info(y_pred)

    def test_all_estimators(self):
        K = Samples.diploma_matrix  # this is not kernel but who cares

        y_pred_kmeans = KKMeans(n_clusters=2, device="cpu", init_measure="inertia").fit_predict(K)
        y_pred_ward = KWard(n_clusters=2).fit_predict(K)
        y_pred_spectral = SpectralClustering_rubanov(n_clusters=2).fit_predict(K)
        logging.info("KMeans: {}".format(y_pred_kmeans))
        logging.info("Ward: {}".format(y_pred_ward))
        logging.info("Spectral Clustering: {}".format(y_pred_spectral))


class TestWorkflow(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_ward_clustering(self):
        (A, gt), info = Datasets().polbooks
        flat_param = 0.5
        for kernel_class in kernels:
            kernel = kernel_class(A)
            param = list(kernel.scaler.scale_list([flat_param]))[0]
            D = kernel.get_K(param)
            y_pred = KWard(n_clusters=info["k"]).predict(D)
            assert len(y_pred) == len(gt)
            assert not np.isnan(y_pred).any()

            score = adjusted_rand_score(gt, y_pred)
            logging.info("{}\t{}\t{}".format(kernel_class.name, flat_param, score))


if __name__ == "__main__":
    unittest.main()
