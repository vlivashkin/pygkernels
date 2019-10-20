"""
Kivimäki: Developments in the theory of randomized shortest paths with a article_comparison of graph node distances
https://arxiv.org/abs/1212.1666
"""

import logging
import unittest

from sklearn.metrics import normalized_mutual_info_score

from pygraphs import util
from pygraphs.cluster import KKMeans
from pygraphs.graphs import Samples, Datasets
from pygraphs.measure import *
from pygraphs.measure.shortcuts import *


class TestFigure2Comparison(unittest.TestCase):
    """Figure 2"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.graph = Samples.triangle_graph

    def _comparison(self, name, D, true_value, atol=0.02):
        div = D[0, 1] / D[1, 2]

        # logging results for report
        logging.info('\t\tD_12/D_23 test\tD_12/D_23 true\tdiff')
        logging.info('{}\t{:0.4f}\t{:0.4f}\t{:0.4f}'.format(name, div, true_value, np.abs(div - true_value)))

        self.assertTrue(np.isclose(div, true_value, atol=atol),
                        "{}: {:0.3f} != {:0.3f}, diff={:0.3f}".format(name, div, true_value, div - true_value))

    def test_boundaries_left_CT(self):
        D = SPCT_D(self.graph).get_D(0)
        self._comparison('CT', D, 1.5)

    def test_boundaries_left_logFor(self):
        D = logFor_D(self.graph).get_D(500.0)
        self._comparison('logFor 500.0', D, 1.5)

    def test_boundaries_left_RSP(self):
        D = RSP_D(self.graph).get_D(0.0001)
        self._comparison('RSP 0.0001', D, 1.5)

    def test_boundaries_left_FE(self):
        D = FE_D(self.graph).get_D(0.0001)
        self._comparison('FE 0.0001', D, 1.5)

    def test_boundaries_right_SP(self):
        D = SPCT_D(self.graph).get_D(1)
        self._comparison('SP', D, 1.0)

    def test_boundaries_right_logFor(self):
        D = logFor_D(self.graph).get_D(0.01)
        self._comparison('logFor 0.01', D, 1.0)

    def test_boundaries_right_RSP(self):
        D = RSP_D(self.graph).get_D(20.0)
        self._comparison('RSP 20.0', D, 1.0)

    def test_boundaries_right_FE(self):
        D = FE_D(self.graph).get_D(20.0)
        self._comparison('FE 20.0', D, 1.0)


class TestTable2(unittest.TestCase):
    """Table 2 with optimal values from Table 1"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etalon = {
            'news_2cl_1': (84.5, 80.7, 83.1, 65.2, 81.6),
            'news_2cl_2': (58.7, 58.7, 58.8, 51.2, 56.8),
            'news_2cl_3': (81.0, 81.1, 75.0, 85.9, 79.6),
            'news_3cl_1': (76.6, 76.2, 75.4, 74.2, 77.3),
            'news_3cl_2': (77.0, 78.3, 75.5, 62.6, 73.0),
            'news_3cl_3': (76.5, 77.0, 74.4, 71.5, 75.9),
            'news_5cl_1': (69.6, 69.0, 60.4, 68.1, 66.8),
            'news_5cl_2': (64.0, 64.6, 58.7, 59.6, 60.4),
            'news_5cl_3': (61.2, 61.6, 57.3, 47.8, 57.3),
        }
        self.datasets = Datasets()

    def _newsgroup_results(self, measure_class, best_param, idx):
        results = []
        for graphs, info in [
            self.datasets['news_2cl_1'], self.datasets['news_2cl_2'], self.datasets['news_2cl_3'],
            self.datasets['news_3cl_1'], self.datasets['news_3cl_2'], self.datasets['news_3cl_3'],
            self.datasets['news_5cl_1'], self.datasets['news_5cl_2'], self.datasets['news_5cl_3']
        ]:
            A, labels_true = graphs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)

            # labels_pred = KKMeans(n_clusters=info['k'], init_choose_objective='nmi', init_choose_strategy='max').fit_predict(K, y=labels_true)

            n_init = 20
            init_nmi = []
            for _ in range(n_init):
                labels_pred = KKMeans(n_clusters=info['k'], n_init=1).fit_predict(K)
                test_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
                init_nmi.append(test_nmi)
            test_nmi = np.mean(init_nmi)

            true_nmi = self.etalon[info['name']][idx] / 100
            diff = true_nmi - test_nmi

            # logging results for report
            # logging.info('measure\tgraph\ttrue nmi\ttest nmi\tdiff')
            logging.info(
                '{}\t{}\t{:0.3f}\t{:0.3f}\t{:0.3f}'.format(measure.name, info['name'], true_nmi, test_nmi, diff))

            results.append({
                'measure_name': measure.name,
                'graph_name': info['name'],
                'test_nmi': test_nmi,
                'true_nmi': true_nmi,
                'diff': diff
            })

        for result in results:
            self.assertTrue(np.isclose(result['test_nmi'], result['true_nmi'], atol=10.),
                            "{}, {}: {:0.3f} != {:0.3f}, diff:{:0.3f}".format(
                                result['graph_name'], result['measure_name'], result['test_nmi'],
                                result['true_nmi'], result['diff']))

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
