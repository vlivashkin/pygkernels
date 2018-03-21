import unittest

from sklearn.metrics import normalized_mutual_info_score

from cluster import KernelKMeans
from graphs import sample
from graphs.dataset import *
from measure.shortcuts import *
from measure import kernel, distance


# Kivimäki: Developments in the theory of randomized shortest paths with a article_comparison of graph node distances
# https://arxiv.org/abs/1212.1666

# Figure 2
class Figure2ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.triangle_graph

    def _comparison(self, name, D, true_value, atol=0.02):
        div = D[0, 1] / D[1, 2]
        self.assertTrue(np.isclose(div, true_value, atol=atol),
                        "{}: {:0.3f} != {:0.3f}, diff={:0.3f}".format(name, div, true_value, div - true_value))

    def test_boundaries_left_CT(self):
        D = distance.SPCT(self.graph).get_D(1)
        self._comparison('CT', D, 1.5)

    def test_boundaries_left_logFor(self):
        D = distance.logFor(self.graph).get_D(500.0)
        self._comparison('logFor', D, 1.5)

    def test_boundaries_left_RSP(self):
        D = distance.RSP(self.graph).get_D(0.0001)
        self._comparison('RSP', D, 1.5)

    def test_boundaries_left_FE(self):
        D = distance.FE(self.graph).get_D(0.0001)
        self._comparison('FE', D, 1.5)

    def test_boundaries_right_SP(self):
        D = distance.SPCT(self.graph).get_D(0)
        self._comparison('SP', D, 1.0)

    def test_boundaries_right_logFor(self):
        D = distance.logFor(self.graph).get_D(0.01)
        self._comparison('logFor', D, 1.0)

    def test_boundaries_right_RSP(self):
        D = distance.RSP(self.graph).get_D(20.0)
        self._comparison('RSP', D, 1.0)

    def test_boundaries_right_FE(self):
        D = distance.FE(self.graph).get_D(20.0)
        self._comparison('FE', D, 1.0)

    if __name__ == '__main__':
        unittest.main()


# Table 2 with optimal values from Table 1
class Table2Tests(unittest.TestCase):
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

    def _newsgroup_results(self, measure_class, best_param, idx):
        for graphs, info in news:
            A, labels_true = graphs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)
            labels_pred = KernelKMeans(n_clusters=info['k'], max_iter=5000, random_state=8).fit_predict(K)
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            self.assertTrue(np.isclose(nmi * 100, self.etalon[info['name']][idx], atol=6.),
                            "{}, {}: {:0.3f} != {:0.3f}, diff:{:0.3f}".format(
                                info['name'], measure.name, nmi * 100, self.etalon[info['name']][idx],
                                np.abs(nmi * 100 - self.etalon[info['name']][idx])))
            print('{} success'.format(info['name']))

    def test_RSP(self):
        self._newsgroup_results(kernel.RSP_K, 0.02, 0)

    def test_FE(self):
        self._newsgroup_results(kernel.FE_K, 0.07, 1)

    def test_logFor(self):
        self._newsgroup_results(kernel.logFor_H, 0.95, 2)

    def test_SPCT(self):
        self._newsgroup_results(kernel.SPCT_K, 1, 3)

    def test_SCT(self):
        self._newsgroup_results(kernel.SCT_H, 26, 4)

    if __name__ == '__main__':
        unittest.main()
