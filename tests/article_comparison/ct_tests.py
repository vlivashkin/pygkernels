import logging
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

import util
from cluster import KernelKMeans
from graphs.dataset import news
from measure.kernel import logComm_H
from measure.shortcuts import resistance_kernel
from scorer import rand_index


# Luh Yen: Graph Nodes Clustering based on the Commute-Time Kernel
# https://pdfs.semanticscholar.org/1206/63cc9644efbd4a4f92f6dc3d83b78e11791f.pdf


class Table1Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.etalon = {
            'news_2cl_1': (97.5, 0.95, 97.8, 0.96, 91.8, 0.85, 94.5, 0.90),
            'news_2cl_2': (90.6, 0.83, 91.5, 0.84, 81.5, 0.70, 93.0, 0.87),
            'news_2cl_3': (95.5, 0.91, 96.0, 0.92, 94.8, 0.90, 95.7, 0.92),
            'news_3cl_1': (93.9, 0.91, 94.5, 0.92, 89.2, 0.85, 92.7, 0.90),
            'news_3cl_2': (93.6, 0.91, 93.5, 0.91, 86.7, 0.82, 92.0, 0.89),
            'news_3cl_3': (93.9, 0.91, 92.8, 0.90, 87.4, 0.83, 81.7, 0.78),
            'news_5cl_1': (83.0, 0.80, 85.4, 0.83, 80.4, 0.79, 76.7, 0.78),
            'news_5cl_2': (74.8, 0.77, 78.4, 0.79, 64.4, 0.69, 67.7, 0.72),
            'news_5cl_3': (76.4, 0.75, 80.1, 0.79, 64.9, 0.69, 64.0, 0.72),
        }

    def _newsgroup_results(self, func, name, scorer, idx, atol):
        results = []
        for graphs, info in news:
            A, labels_true = graphs[0]
            K = func(A)
            labels_pred = KernelKMeans(n_clusters=info['k'], max_iter=5000, random_state=8).fit_predict(K)
            test_nmi = scorer(labels_true, labels_pred)

            true_nmi = self.etalon[info['name']][idx]
            diff = np.abs(test_nmi - true_nmi)

            # logging results for report
            logging.info('measure\tgraph\ttest nmi\ttrue nmi\tdiff')
            logging.info('{}\t{}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(name, info['name'], test_nmi, true_nmi, diff))

            results.append({
                'measure_name': name,
                'graph_name': info['name'],
                'test_nmi': test_nmi,
                'true_nmi': true_nmi,
                'diff': diff
            })

        for result in results:
            self.assertTrue(np.isclose(result['test_nmi'], result['true_nmi'], atol=atol),
                            "{}, {}: {:0.3f} != {:0.3f}, diff:{:0.3f}".format(
                                result['graph_name'], result['measure_name'], result['test_nmi'],
                                result['true_nmi'], result['diff']))

    def test_CT_100RI(self):
        self._newsgroup_results(lambda x: resistance_kernel(x), 'CT 100RI', lambda x, y: 100 * rand_index(x, y), 0, 10)

    def test_CT_ARI(self):
        self._newsgroup_results(lambda x: resistance_kernel(x), 'CT ARI', adjusted_rand_score, 1, 0.01)

    def test_Comm(self):
        self._newsgroup_results(lambda A: logComm_H(A).get_K(0.01), 'Comm', adjusted_rand_score, 1, 0.01)