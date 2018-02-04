import unittest

from sklearn.metrics import normalized_mutual_info_score

from cluster import KernelKMeans
from graphs.dataset import *
from measure.kernel import *


# Kivimäki: Developments in the theory of randomized shortest paths with a comparison of graph node distances
# https://arxiv.org/abs/1212.1666
class Article1Tests(unittest.TestCase):
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
            self.assertTrue(np.isclose(nmi * 100, self.etalon[info['name']][idx], atol=10.),
                            "{}, {}: {:0.3f} != {:0.3f}, diff:{:0.3f}".format(
                                info['name'], measure.name, nmi * 100, self.etalon[info['name']][idx],
                                np.abs(nmi * 100 - self.etalon[info['name']][idx])))
            print('{} success'.format(info['name']))

    def test_RSP(self):
        self._newsgroup_results(RSP, 0.02, 0)  # 8.6

    def test_FE(self):
        self._newsgroup_results(FE, 0.07, 1)  # 13.5

    def test_logFor(self):
        self._newsgroup_results(logFor_H, 0.95, 2)  # 0.26

    def test_SPCT(self):
        self._newsgroup_results(SPCT_H, 1, 3)  # 0.51

    def test_SCT(self):
        self._newsgroup_results(SCT_H, 26, 4)  # 1.92

    if __name__ == '__main__':
        unittest.main()


# Sommer: Comparison of Graph Node Distances on Clustering Tasks
class Article2Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etalon = {
            'football': (0.7928, 0.9061, 0.9028, 0.9092, 0.8115, 0.8575),
            'news_2cl_1': (0.7944, 0.8050, 0.8381, 0.7966, 0.8174, 0.6540),
            'news_2cl_2': (0.5819, 0.5909, 0.5844, 0.5797, 0.5523, 0.5159),
            'news_2cl_3': (0.7577, 0.8107, 0.7482, 0.7962, 0.7857, 0.8592),
            'news_3cl_1': (0.7785, 0.7810, 0.7530, 0.7810, 0.7730, 0.7426),
            'news_3cl_2': (0.7616, 0.7968, 0.7585, 0.7761, 0.7282, 0.6246),
            'news_3cl_3': (0.7455, 0.7707, 0.7487, 0.7300, 0.7627, 0.7203),
            'news_5cl_1': (0.6701, 0.6922, 0.6143, 0.7078, 0.6658, 0.6815),
            'news_5cl_2': (0.6177, 0.6401, 0.5977, 0.6243, 0.6154, 0.5970),
            'news_5cl_3': (0.6269, 0.6065, 0.5729, 0.5750, 0.5712, 0.4801),
            'polblogs': (0.5525, 0.5813, 0.5811, 0.5815, 0.5757, 0.5605),
            # 'zachary': (1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000)
        }

    def _dataset_results(self, measure_class, best_param, idx):
        for graphs, info in [  # football,
            news_2cl_1, news_2cl_2, news_2cl_3, news_3cl_1, news_3cl_2, news_3cl_3,
            # polblogs,
            # zachary
        ]:
            A, labels_true = graphs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)
            labels_pred = KernelKMeans(n_clusters=info['k'], max_iter=5000, random_state=8).fit_predict(K)
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            self.assertTrue(np.isclose(nmi, self.etalon[info['name']][idx], atol=0.1),
                            "{}, {}: Test {:0.3f} != True {:0.3f}, diff:{:0.3f}".format(
                                info['name'], measure.name, nmi, self.etalon[info['name']][idx],
                                np.abs(nmi - self.etalon[info['name']][idx])))
            print('{} success'.format(info['name']))

    def test_CCT(self):
        self._dataset_results(SCCT_H, 26, 0)

    def test_FE(self):
        self._dataset_results(FE, 0.1, 1)

    def test_logFor(self):
        self._dataset_results(logFor_H, 1, 2)

    def test_RSP(self):
        self._dataset_results(RSP, 0.03, 3)

    def test_SCT(self):
        self._dataset_results(SCT_H, 22, 4)

    def test_SP(self):
        self._dataset_results(SP_K, 0, 5)

    if __name__ == '__main__':
        unittest.main()
