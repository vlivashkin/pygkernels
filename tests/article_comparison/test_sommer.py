"""
Sommer: Comparison of Graph Node Distances on Clustering Tasks
no known link to paper
"""

import logging
import unittest
from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import normalized_mutual_info_score

from pygraphs import util
from pygraphs.cluster import KKMeans
from pygraphs.graphs import Datasets
from pygraphs.measure import SCCT_H, FE_K, logFor_H, RSP_K, SCT_H, SPCT_H


class TestTable3(ABC):
    """Table 3 with optimal parameters from Table 2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.etalon = {  # CCT, FE, logFor, RSP, SCT, SP
            'football': (.7928, .9061, .9028, .9092, .8115, .8575),
            'karate': (1., 1., 1., 1., 1., 1.),
            'polblogs': (.5525, .5813, .5811, .5815, .5757, .5605),
            'news_2cl_1': (.7944, .8050, .8381, .7966, .8174, .6540),
            'news_2cl_2': (.5819, .5909, .5844, .5797, .5523, .5159),
            'news_2cl_3': (.7577, .8107, .7482, .7962, .7857, .8592),
            'news_3cl_1': (.7785, .7810, .7530, .7810, .7730, .7426),
            'news_3cl_2': (.7616, .7968, .7585, .7761, .7282, .6246),
            'news_3cl_3': (.7455, .7707, .7487, .7300, .7627, .7203),
            'news_5cl_1': (.6701, .6922, .6143, .7078, .6658, .6815),
            'news_5cl_2': (.6177, .6401, .5977, .6243, .6154, .5970),
            'news_5cl_3': (.6269, .6065, .5729, .5750, .5712, .4801)
        }
        self.datasets = Datasets()

    @abstractmethod
    def dataset_results(self, measure_class, best_param, etalon_idx):
        pass

    def _dataset_results(self, measure_class, best_param, etalon_idx, estimator_class, n_init_inertia=3,
                         n_init_nmi=1, parallel=False, start_random_seed=5003):
        results = []
        for graphs, Gs, info in [
            self.datasets['football'], self.datasets['karate'],
            self.datasets['news_2cl_1'], self.datasets['news_2cl_2'], self.datasets['news_2cl_3'],
            # self.datasets['news_3cl_1'], self.datasets['news_3cl_2'], self.datasets['news_3cl_3'],
            # self.datasets['news_5cl_1'], self.datasets['news_5cl_2'], self.datasets['news_5cl_3']
        ]:
            A, labels_true = graphs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)

            if parallel:
                def whole_kmeans_run(i_run):
                    kkmeans = estimator_class(n_clusters=info['k'], n_init=n_init_inertia, random_state=i_run,
                                              device=i_run % 2)
                    labels_pred = kkmeans.predict(K, A=A)
                    assert (len(labels_true) == len(labels_pred))
                    item_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
                    return item_nmi

                init_nmi = Parallel(n_jobs=6)(delayed(whole_kmeans_run)(i) for i in range(n_init_nmi))
            else:
                init_nmi = []
                for i_run in range(n_init_nmi):
                    kkmeans = estimator_class(n_clusters=info['k'], n_init=n_init_inertia,
                                              random_state=start_random_seed + i_run, device='cpu')
                    labels_pred = kkmeans.predict(K, A=A)
                    assert (len(labels_true) == len(labels_pred))
                    item_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
                    init_nmi.append(item_nmi)

            test_nmi_mean = np.mean(init_nmi)
            test_nmi_median = np.median(init_nmi)
            test_nmi_std = np.std(init_nmi)

            true_nmi = self.etalon[info['name']][etalon_idx]
            diff = true_nmi - test_nmi_mean

            logging.info(f'{measure.name}\t{true_nmi:0.3f}\t{test_nmi_mean:0.3f}\t{diff:0.3f}\t'
                         f'{test_nmi_median:0.3f}\t{test_nmi_std:0.3f}\t{info["name"]}')

            results.append({
                'measure_name': measure.name,
                'graph_name': info['name'],
                'test_nmi': test_nmi_mean,
                'true_nmi': true_nmi,
                'diff': diff
            })

        for result in results:
            self.assertTrue(result['test_nmi'] + .11 > result['true_nmi'],
                            f'{result["graph_name"]}, {result["measure_name"]}. '
                            f'ours:{result["test_nmi"]:.4f} != gt:{result["true_nmi"]:.4f}, diff:{result["diff"]:.4f}')

    def test_CCT(self):
        self.dataset_results(SCCT_H, 26, 0)

    def test_FE(self):
        self.dataset_results(FE_K, 0.1, 1)

    def test_logForH(self):
        self.dataset_results(logFor_H, 1.0, 2)

    def test_RSP(self):
        self.dataset_results(RSP_K, 0.03, 3)

    def test_SCT(self):
        self.dataset_results(SCT_H, 22, 4)

    def test_SP(self):
        self.dataset_results(SPCT_H, 1, 5)


class TestTable3_KKMeans_vanilla_kmpp_pytorch(TestTable3, unittest.TestCase):
    def dataset_results(self, measure_class, best_param, etalon_idx):
        estimator = partial(KKMeans, init='k-means++')
        self._dataset_results(measure_class, best_param, etalon_idx, estimator, parallel=False, start_random_seed=5014)


if __name__ == "__main__":
    unittest.main()
