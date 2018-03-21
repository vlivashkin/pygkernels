import unittest

from sklearn.metrics import normalized_mutual_info_score

from cluster import KernelKMeans
from graphs.dataset import *
from measure.kernel import *


# Sommer: Comparison of Graph Node Distances on Clustering Tasks

# Table 3 with optimal parameters from Table 2
class Table3Tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etalon = {  # CCT, FE, logFor, RSP, SCT, SP
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
            'zachary': (1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000)
        }

    def _dataset_results(self, measure_class, best_param, idx):
        for graphs, info in [news_2cl_1, news_2cl_2, news_2cl_3, news_3cl_1, news_3cl_2, news_3cl_3,
                             zachary, football  #, polblogs
                             ]:
            A, labels_true = graphs[0]
            measure = measure_class(A)
            K = measure.get_K(best_param)
            labels_pred = KernelKMeans(n_clusters=info['k'], max_iter=5000, random_state=42).fit_predict(K)
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            self.assertTrue(np.isclose(nmi, self.etalon[info['name']][idx], atol=0.08),
                            "{}, {}: Test {:0.4f} != True {:0.4f}, diff:{:0.4f}".format(
                                info['name'], measure.name, nmi, self.etalon[info['name']][idx],
                                np.abs(nmi - self.etalon[info['name']][idx])))
            print('{} success'.format(info['name']))

    def test_CCT(self):
        self._dataset_results(SCCT_H, 26, 0)

    def test_FE(self):
        self._dataset_results(FE_K, 0.1, 1)

    def test_logFor(self):
        self._dataset_results(logFor_H, 1, 2)

    def test_RSP(self):
        self._dataset_results(RSP_K, 0.03, 3)

    def test_SCT(self):
        self._dataset_results(SCT_H, 22, 4)

    def test_SP(self):
        self._dataset_results(SP_K, 0, 5)

    if __name__ == '__main__':
        unittest.main()
