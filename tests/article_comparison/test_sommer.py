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

from pygkernels import util
from pygkernels.cluster import KKMeans
from pygkernels.data import Datasets
from pygkernels.measure import SCCT_H, FE_K, logFor_H, RSP_K, SCT_H, SPCT_H


class TestTable3(ABC):
    """Table 3 with optimal parameters from Table 2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.etalon = {  # CCT, FE, logFor, RSP, SCT, SP
            "football": (0.7928, 0.9061, 0.9028, 0.9092, 0.8115, 0.8575),
            "karate": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "polblogs": (0.5525, 0.5813, 0.5811, 0.5815, 0.5757, 0.5605),
            "news_2cl1": (0.7944, 0.8050, 0.8381, 0.7966, 0.8174, 0.6540),
            "news_2cl2": (0.5819, 0.5909, 0.5844, 0.5797, 0.5523, 0.5159),
            "news_2cl3": (0.7577, 0.8107, 0.7482, 0.7962, 0.7857, 0.8592),
            "news_3cl1": (0.7785, 0.7810, 0.7530, 0.7810, 0.7730, 0.7426),
            "news_3cl2": (0.7616, 0.7968, 0.7585, 0.7761, 0.7282, 0.6246),
            "news_3cl3": (0.7455, 0.7707, 0.7487, 0.7300, 0.7627, 0.7203),
            "news_5cl1": (0.6701, 0.6922, 0.6143, 0.7078, 0.6658, 0.6815),
            "news_5cl2": (0.6177, 0.6401, 0.5977, 0.6243, 0.6154, 0.5970),
            "news_5cl3": (0.6269, 0.6065, 0.5729, 0.5750, 0.5712, 0.4801),
        }
        self.datasets = Datasets()

    @abstractmethod
    def dataset_results(self, measure_class, best_param, etalon_idx):
        pass

    def _dataset_results(
        self,
        measure_class,
        best_param,
        etalon_idx,
        estimator_class,
        n_init_inertia=3,
        n_init_nmi=1,
        parallel=False,
        start_random_seed=5003,
    ):
        results = []
        for (A, gt), info in [
            self.datasets["football"],  # self.datasets['karate'],
            self.datasets["news_2cl1"],
            self.datasets["news_2cl2"],
            self.datasets["news_2cl3"],
            # self.datasets['news_3cl1'], self.datasets['news_3cl2'], self.datasets['news_3cl3'],
            # self.datasets['news_5cl1'], self.datasets['news_5cl2'], self.datasets['news_5cl3']
        ]:
            measure = measure_class(A)
            K = measure.get_K(best_param)

            if parallel:

                def whole_kmeans_run(i_run):
                    kkmeans = estimator_class(
                        n_clusters=info["k"], n_init=n_init_inertia, random_state=i_run, device=i_run % 2
                    )
                    y_pred = kkmeans.predict(K, A=A)
                    assert len(gt) == len(y_pred)
                    item_nmi = normalized_mutual_info_score(gt, y_pred, average_method="geometric")
                    return item_nmi

                init_nmi = Parallel(n_jobs=6)(delayed(whole_kmeans_run)(i) for i in range(n_init_nmi))
            else:
                init_nmi = []
                for i_run in range(n_init_nmi):
                    kkmeans = estimator_class(
                        n_clusters=info["k"],
                        n_init=n_init_inertia,
                        random_state=start_random_seed + i_run,
                        device="cpu",
                    )
                    y_pred = kkmeans.predict(K, A=A)
                    assert len(gt) == len(y_pred)
                    item_nmi = normalized_mutual_info_score(gt, y_pred, average_method="geometric")
                    init_nmi.append(item_nmi)

            test_nmi_mean = np.mean(init_nmi)
            test_nmi_median = np.median(init_nmi)
            test_nmi_std = np.std(init_nmi)

            true_nmi = self.etalon[info["name"]][etalon_idx]
            diff = true_nmi - test_nmi_mean

            logging.info(
                f"{measure.name}\t{true_nmi:0.3f}\t{test_nmi_mean:0.3f}\t{diff:0.3f}\t"
                f'{test_nmi_median:0.3f}\t{test_nmi_std:0.3f}\t{info["name"]}'
            )

            results.append(
                {
                    "measure_name": measure.name,
                    "graph_name": info["name"],
                    "test_nmi": test_nmi_mean,
                    "true_nmi": true_nmi,
                    "diff": diff,
                }
            )

        for result in results:
            self.assertTrue(
                result["test_nmi"] + 0.11 > result["true_nmi"],
                f'{result["graph_name"]}, {result["measure_name"]}. '
                f'ours:{result["test_nmi"]:.4f} != gt:{result["true_nmi"]:.4f}, diff:{result["diff"]:.4f}',
            )

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


class TestTable3_KKMeans(TestTable3, unittest.TestCase):
    def dataset_results(self, measure_class, best_param, etalon_idx):
        estimator = partial(KKMeans)
        self._dataset_results(measure_class, best_param, etalon_idx, estimator, parallel=False, start_random_seed=5014)


if __name__ == "__main__":
    unittest.main()
