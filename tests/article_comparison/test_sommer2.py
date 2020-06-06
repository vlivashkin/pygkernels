"""
Sommer: Modularity-Driven Kernel k-means for Community Detection
no known link to paper
"""

import logging
import unittest
from abc import ABC, abstractmethod
from functools import partial

from sklearn.metrics import normalized_mutual_info_score

from pygkernels import util
from pygkernels.cluster import KKMeans
from pygkernels.data import Datasets
from pygkernels.measure import SCT_H, SCCT_H, FE_K, RSP_K, SPCT_H, Comm_H, For_H, Heat_H, Katz_H, logComm_H, logFor_H, \
    logHeat_H, logKatz_H


class TestTable1(ABC):
    """Table 1 with optimal parameters from PREVIOUS PAPER"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.etalon = {  # SCT CCT FE RSP SP Com For Heat Walk lCom lFor lHeat lWalk LVN
            'football': (0.908, 0.889, 0.884, 0.884, 0.812, 0.545, 0.293,
                         0.492, 0.749, 0.598, 0.424, 0.583, 0.249, 0.698),
            'lfr1': (0.976, 0.990, 0.981, 0.983, 0.890, 0.052, 0.027,
                     0.051, 0.003, 0.475, 0.500, 0.511, 0.485, 0.962),
            'lfr2': (1.000, 1.000, 1.000, 1.000, 0.986, 0.147, 0.164,
                     0.266, 0.694, 0.481, 0.478, 0.449, 0.682, 0.823),
            'lfr3': (1.000, 0.997, 1.000, 1.000, 0.989, 0.087, 0.130,
                     0.159, 0.502, 0.520, 0.379, 0.376, 0.729, 0.827),
            'news_2cl1': (0.608, 0.638, 0.584, 0.581, 0.434, 0.018, 0.032,
                           0.032, 0.249, 0.291, 0.243, 0.253, 0.462, 0.573),
            'news_2cl2': (0.356, 0.396, 0.359, 0.346, 0.296, 0.008, 0.027,
                           0.100, 0.394, 0.584, 0.123, 0.207, 0.233, 0.432),
            'news_2cl3': (0.579, 0.584, 0.590, 0.598, 0.616, 0.333, 0.053,
                           0.117, 0.357, 0.682, 0.553, 0.611, 0.496, 0.586),
            'news_3cl1': (0.697, 0.706, 0.702, 0.696, 0.660, 0.096, 0.032,
                           0.125, 0.245, 0.489, 0.471, 0.478, 0.748, 0.699),
            'news_3cl2': (0.656, 0.689, 0.711, 0.706, 0.572, 0.068, 0.035,
                           0.040, 0.294, 0.345, 0.343, 0.281, 0.572, 0.661),
            'news_3cl3': (0.642, 0.605, 0.681, 0.678, 0.720, 0.067, 0.031,
                           0.033, 0.380, 0.295, 0.255, 0.248, 0.423, 0.673),
            'news_5cl1': (0.641, 0.643, 0.648, 0.651, 0.614, 0.087, 0.019,
                           0.032, 0.284, 0.259, 0.273, 0.291, 0.504, 0.684),
            'news_5cl2': (0.616, 0.630, 0.643, 0.633, 0.596, 0.185, 0.026,
                           0.024, 0.291, 0.345, 0.214, 0.234, 0.374, 0.634),
            'news_5cl3': (0.573, 0.624, 0.615, 0.571, 0.480, 0.034, 0.021,
                           0.021, 0.369, 0.300, 0.337, 0.281, 0.383, 0.572),
            'polblogs': (0.556, 0.556, 0.567, 0.568, 0.549, 0.423, 0.322,
                         0.641, 0.298, 0.511, 0.550, 0.527, 0.577, 0.597),
            'karate': (0.832, 0.724, 0.838, 0.832, 1.000, 0.262, 0.395,
                       1.000, 0.875, 1.000, 0.866, 1.000, 0.697, 0.982)
        }
        self.datasets = Datasets()

    @abstractmethod
    def dataset_results(self, measure_class, best_param, etalon_idx):
        pass

    def _dataset_results(self, measure_class, best_param, etalon_idx, estimator_class, n_init_inertia=10):
        results = []
        for graphs, Gs, info in [
            self.datasets['football'], self.datasets['karate'],
            self.datasets['news_2cl1'], self.datasets['news_2cl2'], self.datasets['news_2cl3'],
            # self.datasets['news_3cl1'], self.datasets['news_3cl2'], self.datasets['news_3cl3'],
            # self.datasets['news_5cl1'], self.datasets['news_5cl2'], self.datasets['news_5cl3']
        ]:
            (A, labels_true), G = graphs[0], Gs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)

            kkmeans = estimator_class(n_clusters=info['k'], n_init=n_init_inertia, random_state=5432)
            labels_pred = kkmeans.predict(K, A=A)
            test_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')

            true_nmi = self.etalon[info['name']][etalon_idx]
            diff = true_nmi - test_nmi

            logging.info(f'{measure.name}\t{true_nmi:0.3f}\t{test_nmi:0.3f}\t{diff:0.3f}')

            results.append({
                'measure_name': measure.name,
                'graph_name': info['name'],
                'test_nmi': test_nmi,
                'true_nmi': true_nmi,
                'diff': diff
            })

        for result in results:
            self.assertTrue(result['test_nmi'] + .11 > result['true_nmi'],
                            f'{result["graph_name"]}, {result["measure_name"]}: '
                            f'ours:{result["test_nmi"]:.4f} != gt:{result["true_nmi"]:.4f}, diff={result["diff"]:.4f}')

    def test_SCT(self):
        self.dataset_results(SCT_H, 22, 0)

    def test_CCT(self):
        self.dataset_results(SCCT_H, 26, 1)

    def test_FE(self):
        self.dataset_results(FE_K, 0.1, 2)

    def test_RSP(self):
        self.dataset_results(RSP_K, 0.03, 3)

    def test_SP(self):
        self.dataset_results(SPCT_H, 1, 4)

    @unittest.skip
    def test_Comm(self):
        self.dataset_results(Comm_H, -1, 5)

    @unittest.skip
    def test_For(self):
        self.dataset_results(For_H, -1, 6)

    @unittest.skip
    def test_Heat(self):
        self.dataset_results(Heat_H, -1, 7)

    @unittest.skip
    def test_Katz(self):
        self.dataset_results(Katz_H, -1, 8)

    @unittest.skip
    def test_logComm(self):
        self.dataset_results(logComm_H, -1, 9)

    def test_logFor(self):
        self.dataset_results(logFor_H, 1.0, 10)

    @unittest.skip
    def test_logHeat(self):
        self.dataset_results(logHeat_H, -1, 11)

    @unittest.skip
    def test_logKatz(self):
        self.dataset_results(logKatz_H, -1, 12)


class TestTable3_KKMeans(TestTable1, unittest.TestCase):
    def dataset_results(self, measure_class, best_param, etalon_idx):
        estimator = partial(KKMeans, device='cpu')
        return self._dataset_results(measure_class, best_param, etalon_idx, estimator)


if __name__ == "__main__":
    unittest.main()
