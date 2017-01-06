import unittest

from measure.distance import Walk, logFor, For, pWalk, Comm, SPCT, FE
from measure.kernel import For_H
from measure.scale import AlphaToT
from measure.shortcuts import *
from tests import sample_graphs


class TestStringMethods(unittest.TestCase):
    def test_chain_SP(self):
        D = D_SP(sample_graphs.chain_graph)
        D = D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 1.000), "distances not equal: 1.000 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 1.000), "distances not equal: 1.000 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 2.000), "distances not equal: 2.000 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 3.000), "distances not equal: 3.000 != {}".format(D[0, 3]))

    def test_chain_R(self):
        D = HtoD(H_R(sample_graphs.chain_graph))
        D = D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 1.000), "distances not equal: 1.000 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 1.000), "distances not equal: 1.000 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 2.000), "distances not equal: 2.000 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 3.000), "distances not equal: 3.000 != {}".format(D[0, 3]))

    def test_chain_pWalk(self):
        parameter = AlphaToT().calc(sample_graphs.chain_graph, 4.5)
        D = pWalk().getD(sample_graphs.chain_graph, parameter)
        D = 1.025 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 1.025), "distances not equal: 1.025 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 0.950), "distances not equal: 0.950 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 1.541), "distances not equal: 1.541 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 1.466), "distances not equal: 1.466 != {}".format(D[0, 3]))

        parameter = AlphaToT().calc(sample_graphs.chain_graph, 1.0)
        D = pWalk().getD(sample_graphs.chain_graph, parameter)
        D = 0.988 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 0.988), "distances not equal: 0.988 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 1.025), "distances not equal: 1.025 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 1.379), "distances not equal: 1.379 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 1.416), "distances not equal: 1.416 != {}".format(D[0, 3]))

    def test_chain_Walk(self):
        parameter = AlphaToT.calc(sample_graphs.chain_graph, 1.0)
        D = Walk().getD(sample_graphs.chain_graph, parameter)
        D = 1.025 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 1.025), "distances not equal: 1.025 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 0.950), "distances not equal: 0.950 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 1.975), "distances not equal: 1.975 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 3.000), "distances not equal: 3.000 != {}".format(D[0, 3]))

    def test_chain_For(self):
        D = For().getD(sample_graphs.chain_graph, 1.0)
        D = 1.026 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 1.026), "distances not equal: 1.026 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 0.947), "distances not equal: 0.947 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 1.500), "distances not equal: 1.500 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 1.895), "distances not equal: 1.895 != {}".format(D[0, 3]))

    def test_chain_logFor(self):
        D = logFor().getD(sample_graphs.chain_graph, 2.0)
        D = 0.959 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 0.959), "distances not equal: 0.959 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 1.081), "distances not equal: 1.081 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 2.040), "distances not equal: 2.040 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 2.999), "distances not equal: 2.999 != {}".format(D[0, 3]))

    def test_chain_Comm(self):
        D = Comm().getD(sample_graphs.chain_graph, 1.0)
        D = 0.964 * D / D[0, 1]
        self.assertTrue(sample_graphs.equal_double(D[0, 1], 0.964), "distances not equal: 0.964 != {}".format(D[0, 1]))
        self.assertTrue(sample_graphs.equal_double(D[1, 2], 1.072), "distances not equal: 1.072 != {}".format(D[1, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 2], 1.492), "distances not equal: 1.492 != {}".format(D[0, 2]))
        self.assertTrue(sample_graphs.equal_double(D[0, 3], 1.564), "distances not equal: 1.564 != {}".format(D[0, 3]))

    def test_triangle_SPCT(self):
        D = SPCT().getD(sample_graphs.triangle_graph, 0)
        self.assertTrue(sample_graphs.equal_double(D[0, 1] / D[1, 2], 1.0),
                        "SP distance attitude not equal 1.0: {}".format(D[0, 1] / D[1, 2]))
        D = SPCT().getD(sample_graphs.triangle_graph, 1)
        self.assertTrue(sample_graphs.equal_double(D[0, 1] / D[1, 2], 1.5),
                        "CT distance attitude not equal 1.5: {}".format(D[0, 1] / D[1, 2]))

    def test_triangle_logFor(self):
        D = logFor().getD(sample_graphs.triangle_graph, 0.01)
        self.assertTrue(sample_graphs.equal_double_non_strict(D[0, 1] / D[1, 2], 1.0),
                        "Logarithmic Forest distance attitude not equal 1.0: {}".format(D[0, 1] / D[1, 2]))
        D = logFor().getD(sample_graphs.triangle_graph, 500.0)
        self.assertTrue(sample_graphs.equal_double_non_strict(D[0, 1] / D[1, 2], 1.5),
                        "Logarithmic Forest distance attitude not equal 1.5: {}".format(D[0, 1] / D[1, 2]))

    def test_triangle_FE(self):
        D = FE().getD(sample_graphs.triangle_graph, 0.0001)
        self.assertTrue(sample_graphs.equal_double_non_strict(D[0, 1] / D[1, 2], 1.5),
                        "Free Energy distance attitude not equal 1.5: {}".format(D[0, 1] / D[1, 2]))
        D = FE().getD(sample_graphs.triangle_graph, 30.0)
        self.assertTrue(sample_graphs.equal_double_non_strict(D[0, 1] / D[1, 2], 1.0),
                        "Free Energy distance attitude not equal 1.0: {}".format(D[0, 1] / D[1, 2]))

    def test_tree_SPCT_equality(self):
        SP = SPCT().getD(sample_graphs.tree_matrix, 0)
        CT = SPCT().getD(sample_graphs.tree_matrix, 1)
        self.assertTrue(sample_graphs.equal_arrays_strict(SP, CT))

    def test_chain_For_H(self):
        for_chain0 = For_H().getK(sample_graphs.chain_graph, 0)
        for_chain0_etalon = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.assertTrue(sample_graphs.equal_arrays_strict(for_chain0, for_chain0_etalon))

        for_chain05 = For_H().getK(sample_graphs.chain_graph, 0.5)
        for_chain05_etalon = np.array([
            [0.73214286, 0.19642857, 0.05357143, 0.01785714],
            [0.19642857, 0.58928571, 0.16071429, 0.05357143],
            [0.05357143, 0.16071429, 0.58928571, 0.19642857],
            [0.01785714, 0.05357143, 0.19642857, 0.73214286]
        ])
        self.assertTrue(sample_graphs.equal_arrays_strict(for_chain05, for_chain05_etalon))

    def test_triangle_For_H(self):
        for_chain02 = For_H().getK(sample_graphs.triangle_graph, 0.2)
        for_chain02_etalon = np.array([
            [0.85185185, 0.11111111, 0.01851852, 0.01851852],
            [0.11111111, 0.66666667, 0.11111111, 0.11111111],
            [0.01851852, 0.11111111, 0.74768519, 0.12268519],
            [0.01851852, 0.11111111, 0.12268519, 0.74768519]
        ])
        self.assertTrue(sample_graphs.equal_arrays_strict(for_chain02, for_chain02_etalon))

    if __name__ == '__main__':
        unittest.main()
