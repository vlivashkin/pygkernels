import logging
import operator
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

        y_pred_kmeans = KKMeans(n_clusters=2, device='cpu', init_measure='inertia').fit_predict(K)
        y_pred_ward = KWard(n_clusters=2).fit_predict(K)
        y_pred_spectral = SpectralClustering_rubanov(n_clusters=2).fit_predict(K)
        logging.info('KMeans: {}'.format(y_pred_kmeans))
        logging.info('Ward: {}'.format(y_pred_ward))
        logging.info('Spectral Clustering: {}'.format(y_pred_spectral))


class TestWorkflow(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()

    def test_ward_clustering(self):
        graphs, _, info = self.datasets.polbooks
        for measure in kernels:
            measureparamdict = {}
            mean = []
            for edges, nodes in graphs:
                measure_o = measure(edges)
                param = list(measure_o.scaler.scale_list([0.5]))[0]
                D = measure_o.get_K(param)
                y_pred = KWard(len(list(set(graphs[0][1])))).predict(D)
                ari = adjusted_rand_score(nodes, y_pred)
                mean.append(ari)
            mean = [m for m in mean if m is not None and m == m]
            score = np.array(mean).mean()
            if score is not None and score == score:
                measureparamdict[0.5] = score
            maxparam = max(measureparamdict.items(), key=operator.itemgetter(1))[0]
            logging.info("{}\t{}\t{}".format(measure.name, maxparam, measureparamdict[maxparam]))


if __name__ == "__main__":
    unittest.main()
