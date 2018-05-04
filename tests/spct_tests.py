import unittest

from sklearn.metrics import adjusted_rand_score

import util
from cluster import KernelKMeans
from graphs import sample, dataset
from measure.distance import *
from measure.shortcuts import *


# This is important:
# lambda = 0 -> CT
# lambda = 1 -> SP
class spctTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.get_CT = lambda A: SPCT(A).get_D(0)
        self.get_SP = lambda A: SPCT(A).get_D(1)

    def test_SPCT_order(self):
        A = sample.diploma_matrix
        SP, CT = sp_distance(A), 2. * H_to_D(resistance_kernel(A))
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
        sp_normed = normalize(self.get_SP(A))
        ct_normed = normalize(self.get_CT(A))
        self.assertTrue(np.allclose(sp_normed, ct_normed))

    def test_tree_SPCT_equality(self):
        A = sample.tree_matrix
        self.assertTrue(np.allclose(self.get_SP(A), self.get_CT(A)))

    def test_weighted_graph_SP(self):
        A = sample.weighted
        true_SP = sample.weighted_sp
        self.assertTrue(np.allclose(self.get_SP(A), true_SP))

    # def test_compare_clustering_quality(self):
    #     graphs, info = dataset.news_2cl_1
    #     A, y_true = graphs[0]
    #
    #     def test_quality(K):
    #         kmeans = KernelKMeans(n_clusters=info['k'])
    #         y_test = kmeans.fit_predict(K)
    #         return normalized_mutual_info_score(y_true, y_test)
    #
    #     # without normalization
    #     K = D_to_K(sp_distance(A))
    #     quality1 = test_quality(K)
    #     logging.info('without normalization', '\t', quality1)
    #
    #     # normalization of distance
    #     K = D_to_K(normalize(sp_distance(A)))
    #     quality2 = test_quality(K)
    #     logging.info('normalization of distance', '\t', quality2)
    #
    #     # normalization of kernel
    #     K = D_to_K(normalize(sp_distance(A)))
    #     quality3 = test_quality(K)
    #     logging.info('normalization of kernel', '\t', quality3)
    #
    #     # both
    #     K = normalize(D_to_K(normalize(sp_distance(A))))
    #     quality4 = test_quality(K)
    #     logging.info('both', '\t', quality4)
    #
    #     # distance
    #     K = sp_distance(A)
    #     quality4 = test_quality(K)
    #     logging.info('distance', '\t', quality4)
    #
    #     # another measure (FE)
    #     K = D_to_K(FE(A).get_D(0.01))
    #     quality5 = test_quality(K)
    #     logging.info('another measure (FE)', '\t', quality5)


class Figure2ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()

        graph, info = dataset.news_2cl_1
        self.graph, self.y_true = graph[0]

    def call_and_print(self, name, K):
        ari = adjusted_rand_score(KernelKMeans(2).fit_predict(K), self.y_true)
        print('{}\t{:0.3f}'.format(name, ari))

    def test_CT(self):
        K_CT = kernel.SPCT_H(self.graph).get_K(0)
        self.call_and_print('CT\t', K_CT)
        K_logFor = kernel.logFor_H(self.graph).get_K(500.0)
        self.call_and_print('logFor\t500', K_logFor)
        K_RSP = kernel.RSP_K(self.graph).get_K(0.0001)
        self.call_and_print('RSP\t0.0001', K_RSP)
        K_FE = kernel.FE_K(self.graph).get_K(0.0001)
        self.call_and_print('FE\t0.0001', K_FE)

    def test_SP(self):
        K_SP = sp_kernel(self.graph)
        self.call_and_print('SP\t', K_SP)
        K_logFor = kernel.logFor_H(self.graph).get_K(0.001)
        self.call_and_print('logFor\t0.001', K_logFor)
        K_RSP = kernel.RSP_K(self.graph).get_K(19.0)
        self.call_and_print('RSP\t19', K_RSP)
        K_FE = kernel.FE_K(self.graph).get_K(19.0)
        self.call_and_print('FE\t19', K_FE)
