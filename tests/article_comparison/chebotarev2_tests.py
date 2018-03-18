import unittest

from graphs import sample
from measure.distance import *
from measure.scaler import AlphaToT, Rho, Linear
from measure.shortcuts import *


# Chebotarev: The Walk Distances in Graphs
# https://arxiv.org/abs/1103.2059

class Table1ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.chain_graph

    def _comparison(self, D, true_values, atol=0.1):
        print("True\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(*true_values))
        print("Test\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(D[0, 1] / D[1, 2], (D[0, 1] + D[1, 2]) / D[0, 2], D[0, 3] / D[0, 2]))
        for d, t in zip([D[0, 1] / D[1, 2], (D[0, 1] + D[1, 2]) / D[0, 2], D[0, 3] / D[0, 2]], true_values):
            self.assertTrue(np.isclose(d, t, atol=atol),
                            "Test {:0.3f} != True {:0.3f}, diff={:0.3f}".format(d, t, np.abs(d - t)))

    def test_chain_SP(self):
        D = D_SP(self.graph)
        self._comparison(D, [1., 1., 1.5])

    def test_chain_R(self):
        D = H_to_D(H_R(self.graph))
        self._comparison(D, [1., 1., 1.5])

    def test_chain_Walk(self):
        parameter = AlphaToT(self.graph).scale_one(1.0)
        D = Walk(self.graph).get_D(parameter)
        D = np.power(D, 2)
        self._comparison(D, [1.08, 1., 1.52])

    def test_chain_logFor(self):
        parameter = Linear(self.graph).scale_one(2.0)
        D = logFor(self.graph).get_D(parameter)
        D = np.power(D, 2)
        self._comparison(D, [0.89, 1., 1.47])

    def test_chain_For(self):
        parameter = Linear(self.graph).scale_one(1.0)
        D = For(self.graph).get_D(parameter)
        D = np.power(D, 2)
        self._comparison(D, [1.08, 1.32, 1.26])

    def test_chain_pWalk_45(self):
        parameter = AlphaToT(self.graph).scale_one(4.5)
        D = pWalk(self.graph).get_D(parameter)
        D = np.power(D, 2)
        self._comparison(D, [1.08, 1.28, 0.95])

    def test_chain_pWalk_1(self):
        parameter = AlphaToT(self.graph).scale_one(1.0)
        D = pWalk(self.graph).get_D(parameter)
        D = np.power(D, 2)
        self._comparison(D, [0.96, 1.46, 1.03])

    if __name__ == '__main__':
        unittest.main()
