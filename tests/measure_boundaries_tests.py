import unittest

from measure.distance import Walk, logFor, SPCT
from measure.shortcuts import *
from graphs import sample


class MeasureBoundariesTests(unittest.TestCase):
    def test_SPCT_order(self):
        SP = normalize(D_SP(sample.diploma_matrix))
        CT = normalize(HtoD(H_R(sample.diploma_matrix)))
        SPCT_SP = normalize(SPCT(sample.diploma_matrix).getD(0))
        SPCT_CT = normalize(SPCT(sample.diploma_matrix).getD(1))
        self.assertTrue(np.allclose(SP, SPCT_SP))
        self.assertTrue(np.allclose(CT, SPCT_CT))

    def test_tree_SPCT_inequality(self):
        SP = SPCT(sample.diploma_matrix).getD(0)
        CT = SPCT(sample.diploma_matrix).getD(1)
        self.assertFalse(np.allclose(SP, CT))

    def test_chain_SPCT_equality(self):
        SP = SPCT(sample.chain_graph).getD(0)
        CT = SPCT(sample.chain_graph).getD(1)
        self.assertTrue(np.allclose(SP, CT))

    def test_big_chain_SPCT_equality(self):
        SP = SPCT(sample.big_chain).getD(0)
        CT = SPCT(sample.big_chain).getD(1)
        self.assertTrue(np.allclose(SP, CT))

    def test_full_graph_SPCT_equality(self):
        SP = SPCT(sample.full_graph).getD(0)
        CT = SPCT(sample.full_graph).getD(1)
        self.assertTrue(np.allclose(SP, CT))

    def test_full_graph_SP_logFor_Walk_equality(self):
        parameter = 0.00001
        # DSP = normalize(D_SP(sample.chain_graph))
        DlogFor = normalize(logFor(sample.chain_graph).getD(parameter))
        DWalk = normalize(Walk(sample.chain_graph).getD(parameter))
        # self.assertTrue(np.allclose(DSP, DlogFor))
        # self.assertTrue(np.allclose(DWalk, DSP))
        self.assertTrue(np.allclose(DlogFor, DWalk, atol=0.01))

    if __name__ == '__main__':
        unittest.main()
