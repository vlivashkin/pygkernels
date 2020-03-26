"""
Luh Yen: Graph Nodes Clustering based on the Commute-Time Kernel
https://pdfs.semanticscholar.org/1206/63cc9644efbd4a4f92f6dc3d83b78e11791f.pdf
"""

import logging
import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from pygraphs import util
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.graphs import Datasets
from pygraphs.measure import *
from pygraphs.scorer import rand_index


# Luh Yen: Graph Nodes Clustering based on the Commute-Time Kernel
# https://www.isys.ucl.ac.be/staff/marco/Publications/2007_GraphNodesClustering.pdf
class Table1Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.etalon = {  # SCT RI, SCT ARI
            'news_2cl_1': (.975, .95),
            'news_2cl_2': (.906, .83),
            'news_2cl_3': (.955, .91),
            'news_3cl_1': (.939, .91),
            'news_3cl_2': (.936, .91),
            'news_3cl_3': (.939, .91),
            'news_5cl_1': (.830, .80),
            'news_5cl_2': (.748, .77),
            'news_5cl_3': (.764, .75)
        }
        self.datasets = Datasets()

    def _newsgroup_results(self, name, scorer_func, result_idx, atol=0.1):
        results = []
        for graphs, Gs, info in [
            self.datasets['news_2cl_1'], self.datasets['news_2cl_2'], self.datasets['news_2cl_3'],
            # self.datasets['news_3cl_1'], self.datasets['news_3cl_2'], self.datasets['news_3cl_3'],
            # self.datasets['news_5cl_1'], self.datasets['news_5cl_2'], self.datasets['news_5cl_3']
        ]:
            (A, labels_true), G = graphs[0], Gs[0]
            K = SCT_H(A).get_K(22)
            true_nmi = self.etalon[info['name']][result_idx]

            labels_pred = KKMeans(n_clusters=info['k'], n_init=30, device='cpu').predict(K, G=G)
            test_nmi = scorer_func(labels_true, labels_pred)
            diff = np.abs(test_nmi - true_nmi)

            logging.info('measure\tgraph\ttest nmi\ttrue nmi\tdiff')
            logging.info(f'{name}\t{info["name"]}\t{test_nmi:.3f}\t{true_nmi:.3f}\t{diff:.3f}')

            results.append({
                'measure_name': name,
                'graph_name': info['name'],
                'test_nmi': test_nmi,
                'true_nmi': true_nmi,
                'diff': diff
            })

        for result in results:
            self.assertTrue(result['test_nmi'] + atol > result['true_nmi'],
                            f'{result["graph_name"]}, {result["measure_name"]}: '
                            f'ours:{result["test_nmi"]:.3f} != gt:{result["true_nmi"]:.3f}, diff={result["diff"]:.3f}')

    def test_SCT_RI(self):
        self._newsgroup_results('SCT RI', rand_index, 0)

    @unittest.skip
    def test_SCT_ARI(self):
        self._newsgroup_results('SCT ARI', adjusted_rand_score, 1)


if __name__ == "__main__":
    unittest.main()
