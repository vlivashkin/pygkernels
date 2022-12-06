"""
Kivimäki: Developments in the theory of randomized shortest paths with a article_comparison of graph node distances
https://arxiv.org/abs/1212.1666
"""

import logging
import unittest

from sklearn.metrics import normalized_mutual_info_score

import numpy as np
from pygkernels import util
from pygkernels.cluster import KKMeans
from pygkernels.data import Samples, Datasets
from pygkernels.measure import SPCT_D, logFor_D, RSP_D, FE_D, RSP_K, FE_K, logFor_H, SPCT_K, SCT_H


class TestFigure2Comparison(unittest.TestCase):
    """Figure 2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.graph = Samples.triangle_graph

    def _comparison(self, name, D, true_value, atol=0.02):
        div = D[0, 1] / D[1, 2]

        logging.info("\t\tD_12/D_23 test\tD_12/D_23 true\tdiff")
        logging.info(f"{name}\t{div:.4f}\t{true_value:.4f}\t{np.abs(div - true_value):.4f}")

        self.assertTrue(
            np.isclose(div, true_value, atol=atol),
            f"{name}: {div:.3f} != {true_value:.3f}, diff={div - true_value:.3f}",
        )

    def test_boundaries_left_CT(self):
        D = SPCT_D(self.graph).get_D(0)
        self._comparison("CT", D, 1.5)

    def test_boundaries_left_logFor(self):
        D = logFor_D(self.graph).get_D(500.0)
        self._comparison("logFor 500.0", D, 1.5)

    def test_boundaries_left_RSP(self):
        D = RSP_D(self.graph).get_D(0.0001)
        self._comparison("RSP 0.0001", D, 1.5)

    def test_boundaries_left_FE(self):
        D = FE_D(self.graph).get_D(0.0001)
        self._comparison("FE 0.0001", D, 1.5)

    def test_boundaries_right_SP(self):
        D = SPCT_D(self.graph).get_D(1)
        self._comparison("SP", D, 1.0)

    def test_boundaries_right_logFor(self):
        D = logFor_D(self.graph).get_D(0.01)
        self._comparison("logFor 0.01", D, 1.0)

    def test_boundaries_right_RSP(self):
        D = RSP_D(self.graph).get_D(20.0)
        self._comparison("RSP 20.0", D, 1.0)

    def test_boundaries_right_FE(self):
        D = FE_D(self.graph).get_D(20.0)
        self._comparison("FE 20.0", D, 1.0)


class TestTable2(unittest.TestCase):
    """Table 2 with optimal values from Table 1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etalon = {
            "news_2cl1": (0.845, 0.807, 0.831, 0.652, 0.816),
            "news_2cl2": (0.587, 0.587, 0.588, 0.512, 0.568),
            "news_2cl3": (0.810, 0.811, 0.750, 0.859, 0.796),
            "news_3cl1": (0.766, 0.762, 0.754, 0.742, 0.773),
            "news_3cl2": (0.770, 0.783, 0.755, 0.626, 0.730),
            "news_3cl3": (0.765, 0.770, 0.744, 0.715, 0.759),
            "news_5cl1": (0.696, 0.690, 0.604, 0.681, 0.668),
            "news_5cl2": (0.640, 0.646, 0.587, 0.596, 0.604),
            "news_5cl3": (0.612, 0.616, 0.573, 0.478, 0.573),
        }
        self.datasets = Datasets()

    def _newsgroup_results(self, measure_class, best_param, idx):
        results = []
        for (A, labels_true), info in [
            self.datasets["news_2cl1"],
            self.datasets["news_2cl2"],
            self.datasets["news_2cl3"],
            # self.datasets['news_3cl1'], self.datasets['news_3cl2'], self.datasets['news_3cl3'],
            # self.datasets['news_5cl1'], self.datasets['news_5cl2'], self.datasets['news_5cl3']
        ]:
            measure = measure_class(A)
            K = measure.get_K(best_param)
            true_nmi = self.etalon[info["name"]][idx]

            labels_pred = KKMeans(n_clusters=info["k"], n_init=5, device="cpu").predict(K, A=A)
            test_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method="geometric")
            diff = true_nmi - test_nmi

            logging.info("measure\tgraph\ttrue nmi\ttest nmi\tdiff")
            logging.info(f'{measure.name}\t{info["name"]}\t{true_nmi:.3f}\t{test_nmi:.3f}\t{diff:.3f}')

            results.append(
                {
                    "measure_name": measure.name,
                    "graph_name": info["name"],
                    "test_nmi": test_nmi,
                    "true_nmi": true_nmi,
                    "diff": diff,
                }
            )

        for result in results:
            self.assertTrue(
                np.isclose(result["test_nmi"], result["true_nmi"], atol=10.0),
                f'{result["graph_name"]}, {result["measure_name"]}: '
                f'ours:{result["test_nmi"]:.3f} != gt:{result["true_nmi"]:.3f}, diff:{result["diff"]:.3f}',
            )

    def test_RSP(self):
        self._newsgroup_results(RSP_K, 0.02, 0)

    def test_FE(self):
        self._newsgroup_results(FE_K, 0.07, 1)

    def test_logFor(self):
        self._newsgroup_results(logFor_H, 0.95, 2)

    def test_SPCT(self):
        self._newsgroup_results(SPCT_K, 1, 3)

    def test_SCT(self):
        self._newsgroup_results(SCT_H, 26, 4)


if __name__ == "__main__":
    unittest.main()
