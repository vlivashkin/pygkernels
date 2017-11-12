import unittest

from graphs import sample
from measure.distance import *
from measure.kernel import For_H
from measure.scaler import AlphaToT
from measure.shortcuts import *


# Chebotarev: Studying new classes of graph metrics
# https://arxiv.org/abs/1305.7514
class Article1ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.chain_graph

    def _comparison(self, D, true_values, atol=0.04):
        D *= true_values[0] / D[0, 1]
        for d, t in zip([D[0, 1], D[1, 2], D[0, 2], D[0, 3]][:len(true_values)], true_values):
            self.assertTrue(np.isclose(d, t, atol=atol), "{} != {}, diff={}".format(d, t, np.abs(d - t)))

    def test_chain_SP(self):
        D = D_SP(self.graph)
        self._comparison(D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_R(self):
        D = HtoD(H_R(self.graph))
        self._comparison(D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_Walk(self):
        parameter = list(AlphaToT(self.graph).scale([1.0]))[0]
        D = Walk(self.graph).getD(parameter)
        self._comparison(D, [1.025, 0.950, 1.975])

    def test_chain_logFor(self):
        D = logFor(self.graph).getD(2.0)
        self._comparison(D, [0.959, 1.081, 2.040])

    def test_chain_For(self):
        D = For(self.graph).getD(1.0)
        self._comparison(D, [1.026, 0.947, 1.500, 1.895])

    def test_chain_SqResistance(self):
        D = np.sqrt(HtoD(H_R(self.graph)))
        self._comparison(D, [1.000, 1.000, 1.414, 1.732])

    def test_chain_Comm(self):
        D = Comm(self.graph).getD(1.0)
        self._comparison(D, [0.964, 1.072, 1.492, 1.564])

    def test_chain_pWalk(self):
        parameter = AlphaToT(self.graph).scale([4.5]).__next__
        D = pWalk(self.graph).getD(parameter)
        self._comparison(D, [1.025, 0.950, 1.541, 1.466])

        parameter = AlphaToT(self.graph).scale([1.0]).__next__
        D = pWalk(self.graph).getD(parameter)
        self._comparison(D, [0.988, 1.025, 1.379, 1.416])

    if __name__ == '__main__':
        unittest.main()


# Kivimäki: Developments in the theory of randomized shortest paths with a comparison of graph node distances
# https://arxiv.org/abs/1212.1666
class Article2Comparison(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.graph = self.graph

    def _comparison(self, name, D, true_value, atol=0.02):
        div = D[0, 1] / D[1, 2]
        self.assertTrue(np.isclose(div, true_value, atol=atol),
                        "{}: {} != {}, diff={}".format(name, div, true_value, div - true_value))

    def test_boundaries_10(self):
        D = SPCT(self.graph).getD(0)
        self._comparison('SP', D, 1.0)
        D = logFor(self.graph).getD(0.01)
        self._comparison('logFor', D, 1.0)
        D = RSP(self.graph).getD(20.0)
        self._comparison('RSP', D, 1.0)
        D = FE(self.graph).getD(30.0)
        self._comparison('FE', D, 1.0)

    def test_boundaries_15(self):
        D = SPCT(self.graph).getD(1)
        self._comparison('CT', D, 1.5)
        D = logFor(self.graph).getD(500.0)
        # self._comparison('logFor', D, 1.5)
        D = RSP(self.graph).getD(0.0001)
        self._comparison('RSP', D, 1.5)
        D = FE(self.graph).getD(0.0001)
        self._comparison('FE', D, 1.5)

    if __name__ == '__main__':
        unittest.main()


class MeasureSamplesTests(unittest.TestCase):
    def test_tree_SPCT_equality(self):
        SP = SPCT(sample.tree_matrix).getD(0)
        CT = SPCT(sample.tree_matrix).getD(1)
        self.assertTrue(np.allclose(SP, CT))

    def test_chain_For_H(self):
        for_chain0 = For_H(sample.triangle_graph).getK(0)
        for_chain0_etalon = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        self.assertTrue(np.allclose(for_chain0, for_chain0_etalon))

        for_chain05 = For_H(sample.triangle_graph).getK(0.5)
        for_chain05_etalon = np.array([
            [0.73214286, 0.19642857, 0.05357143, 0.01785714],
            [0.19642857, 0.58928571, 0.16071429, 0.05357143],
            [0.05357143, 0.16071429, 0.58928571, 0.19642857],
            [0.01785714, 0.05357143, 0.19642857, 0.73214286]
        ], dtype=np.float64)
        self.assertTrue(np.allclose(for_chain05, for_chain05_etalon))

    def test_triangle_For_H(self):
        for_chain02 = For_H(sample.triangle_graph).getK(0.2)
        for_chain02_etalon = np.array([
            [0.85185185, 0.11111111, 0.01851852, 0.01851852],
            [0.11111111, 0.66666667, 0.11111111, 0.11111111],
            [0.01851852, 0.11111111, 0.74768519, 0.12268519],
            [0.01851852, 0.11111111, 0.12268519, 0.74768519]
        ], dtype=np.float64)
        self.assertTrue(np.allclose(for_chain02, for_chain02_etalon))

    if __name__ == '__main__':
        unittest.main()
