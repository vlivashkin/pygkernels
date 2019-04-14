import logging
import operator
import unittest

from sklearn.metrics import adjusted_rand_score

from pygraphs import util
from pygraphs.cluster import KernelKMeans, SpectralClustering
from pygraphs.cluster.ward import Ward
from pygraphs.graphs import Samples, Datasets
from pygraphs.measure import kernels
from pygraphs.measure.shortcuts import *


class TestEstimators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()

    def test_simple_Ward(self):
        y_pred = Ward(2).predict(Samples.diploma_matrix)
        logging.info(y_pred)

    def test_all_estimators(self):
        K = Samples.diploma_matrix  # this is not kernel but who cares

        y_pred_kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0).fit_predict(K)
        y_pred_ward = Ward(n_clusters=2).fit_predict(K)
        y_pred_spectral = SpectralClustering(n_clusters=2).fit_predict(K)
        logging.info('KMeans: {}'.format(y_pred_kmeans))
        logging.info('Ward: {}'.format(y_pred_ward))
        logging.info('Spectral Clustering: {}'.format(y_pred_spectral))


class TestWorkflow(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()

    def test_ward_clustering(self):
        graphs, info = self.datasets.polbooks
        for measure in kernels:
            measureparamdict = {}
            mean = []
            for edges, nodes in graphs:
                measure_o = measure(edges)
                param = list(measure_o.scaler.scale_list([0.5]))[0]
                D = measure_o.get_K(param)
                y_pred = Ward(len(list(set(graphs[0][1])))).predict(D)
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
