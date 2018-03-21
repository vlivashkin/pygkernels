import unittest

from sklearn.metrics import normalized_mutual_info_score

from cluster import KernelKMeans
from graphs import dataset
from graphs import sample
from measure.distance import *
from measure.shortcuts import *


# This is important:
# lambda = 0 -> CT
# lambda = 1 -> SP


class spctTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_CT = lambda A: SPCT(A).get_D(0)
        self.get_SP = lambda A: SPCT(A).get_D(1)

    def test_SPCT_order(self):
        A = sample.diploma_matrix
        SP, CT = D_SP(A), 2. * H_to_D(H_R(A))
        self.assertTrue(np.allclose(SP, self.get_SP(A)))
        self.assertTrue(np.allclose(CT, self.get_CT(A)))

    def test_tree_SPCT_inequality(self):
        A = sample.diploma_matrix
        self.assertFalse(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_chain_SPCT_equality(self):
        A = sample.chain_graph
        self.assertTrue(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_big_chain_SPCT_equality(self):
        A = sample.big_chain
        self.assertTrue(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_full_graph_SPCT_equality(self):
        A = sample.full_graph
        self.assertTrue(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_tree_SPCT_equality(self):
        A = sample.tree_matrix
        self.assertTrue(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_weighted_graph_SP(self):
        A = sample.weighted
        SP = sample.weighted_sp
        self.assertTrue(np.allclose(self.get_SP(A), SP))

    def test_compare_clustering_quality(self):
        graphs, info = dataset.news_2cl_1
        A, y_true = graphs[0]

        def test_quality(K):
            kmeans = KernelKMeans(n_clusters=info['k'])
            y_test = kmeans.fit_predict(K)
            return normalized_mutual_info_score(y_true, y_test)

        # without normalization
        K = D_to_K(D_SP(A))
        quality1 = test_quality(K)
        print('without normalization', '\t', quality1)

        # normalization of distance
        K = D_to_K(normalize(D_SP(A)))
        quality2 = test_quality(K)
        print('normalization of distance', '\t', quality2)

        # normalization of kernel
        K = D_to_K(normalize(D_SP(A)))
        quality3 = test_quality(K)
        print('normalization of kernel', '\t', quality3)

        # both
        K = normalize(D_to_K(normalize(D_SP(A))))
        quality4 = test_quality(K)
        print('both', '\t', quality4)

        # distance
        K = D_SP(A)
        quality4 = test_quality(K)
        print('distance', '\t', quality4)

        # another measure (FE)
        K = D_to_K(FE(A).get_D(0.01))
        quality5 = test_quality(K)
        print('another measure (FE)', '\t', quality5)
