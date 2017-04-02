import unittest

from measure.distance import Walk, logFor, SPCT
from measure.shortcuts import *
from tests import sample_graphs


class MeasureBoundariesTests(unittest.TestCase):
    def test_chain_SPCT_equality(self):
        SP = SPCT().getD(sample_graphs.chain_graph, 0)
        CT = SPCT().getD(sample_graphs.chain_graph, 1)
        self.assertTrue(sample_graphs.equal_arrays_strict(SP, CT))

    def test_big_chain_SPCT_equality(self):
        big_chain = np.zeros((100, 100))
        for i in range(100):
            if i + 1 < 100:
                big_chain[i][i + 1] = 1
            if i - 1 >= 0:
                big_chain[i][i - 1] = 1
        SP = SPCT().getD(big_chain, 0)
        CT = SPCT().getD(big_chain, 1)
        self.assertTrue(sample_graphs.equal_arrays_strict(SP, CT))

    def test_full_graph_SPCT_equality(self):
        SP = SPCT().getD(sample_graphs.full_graph, 0)
        CT = SPCT().getD(sample_graphs.full_graph, 1)
        self.assertTrue(sample_graphs.equal_arrays_strict(SP, CT))

    def test_full_graph_SP_logFor_Walk_equality(self):
        parameter = 0.00001
        D_SP = normalize(SPCT().getD(sample_graphs.chain_graph, 0))
        D_logFor = normalize(logFor().getD(sample_graphs.chain_graph, parameter))
        D_Walk = normalize(Walk().getD(sample_graphs.chain_graph, parameter))
        self.assertTrue(sample_graphs.equal_arrays_non_strict(D_SP, D_logFor))
        self.assertTrue(sample_graphs.equal_arrays_non_strict(D_logFor, D_Walk))
        self.assertTrue(sample_graphs.equal_arrays_non_strict(D_Walk, D_SP))

    if __name__ == '__main__':
        unittest.main()
