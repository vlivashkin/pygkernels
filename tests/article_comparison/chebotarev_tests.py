import unittest

from graphs import sample
from measure.distance import *
from measure.scaler import AlphaToT, Linear
from measure.shortcuts import *


# Chebotarev: Studying new classes of graph metrics
# https://arxiv.org/abs/1305.7514

class Figure1ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.chain_graph

    def _comparison(self, name, D, true_values, atol=0.1):
        D *= 3. / (D[0, 1] + D[1, 2] + D[2, 3])

        # logging results for report
        print('{}\tD_12\tD_23\tD_13\tD_14'.format(name))
        print("True\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(*true_values))
        print("Test\t{:0.3f}\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(D[0, 1], D[1, 2], D[0, 2], D[0, 3]))

        for d, t in zip([D[0, 1], D[1, 2], D[0, 2], D[0, 3]][:len(true_values)], true_values):
            self.assertTrue(np.isclose(d, t, atol=atol),
                            "Test {:0.3f} != True {:0.3f}, diff={:0.3f}".format(d, t, np.abs(d - t)))

    def test_chain_SP(self):
        D = sp_distance(self.graph)
        self._comparison('SP', D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_R(self):
        D = H_to_D(resistance_kernel(self.graph))
        self._comparison('R', D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_Walk(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = Walk(self.graph).get_D(parameter)
        self._comparison('Walk', D, [1.025, 0.950, 1.975, 3.000])

    def test_chain_logFor(self):
        parameter = Linear(self.graph).scale(2.0)
        D = logFor(self.graph).get_D(parameter)
        self._comparison('logFor', D, [0.959, 1.081, 2.040, 3.000])

    def test_chain_For(self):
        parameter = Linear(self.graph).scale(1.0)
        D = For(self.graph).get_D(parameter)
        self._comparison('For', D, [1.026, 0.947, 1.500, 1.895])

    def test_chain_SqResistance(self):
        D = np.sqrt(H_to_D(resistance_kernel(self.graph)))
        self._comparison('SqResistance', D, [1.000, 1.000, 1.414, 1.732])

    def test_chain_Comm(self):
        D = Comm(self.graph).get_D(1.0)
        self._comparison('Comm', D, [0.964, 1.072, 1.492, 1.564])

    def test_chain_pWalk_45(self):
        parameter = AlphaToT(self.graph).scale(4.5)
        D = pWalk(self.graph).get_D(parameter)
        self._comparison('pWalk 4.5', D, [1.025, 0.950, 1.541, 1.466])

    def test_chain_pWalk_1(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = pWalk(self.graph).get_D(parameter)
        self._comparison('pWalk 1.0', D, [0.988, 1.025, 1.379, 1.416])


# Chebotarev: The Walk Distances in Graphs
# https://arxiv.org/abs/1103.2059

class Table1ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.chain_graph

    def _comparison(self, name, D, true_values, atol=0.1):

        # logging results for report
        print('{}\tD_12/D_23\t(D_12+D_23)/D_13\tD_14/D_12'.format(name))
        print("True\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(*true_values))
        print("Test\t{:0.3f}\t{:0.3f}\t{:0.3f}".format(D[0, 1] / D[1, 2], (D[0, 1] + D[1, 2]) / D[0, 2],
                                                       D[0, 3] / D[0, 2]))

        for d, t in zip([D[0, 1] / D[1, 2], (D[0, 1] + D[1, 2]) / D[0, 2], D[0, 3] / D[0, 2]], true_values):
            self.assertTrue(np.isclose(d, t, atol=atol),
                            "Test {:0.3f} != True {:0.3f}, diff={:0.3f}".format(d, t, np.abs(d - t)))

    def test_chain_SP(self):
        D = sp_distance(self.graph)
        self._comparison('SP', D, [1., 1., 1.5])

    def test_chain_R(self):
        D = H_to_D(resistance_kernel(self.graph))
        self._comparison('R', D, [1., 1., 1.5])

    def test_chain_Walk(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = Walk(self.graph).get_D(parameter)
        self._comparison('Walk', D, [1.08, 1., 1.52])

    def test_chain_logFor(self):
        parameter = Linear(self.graph).scale(2.0)
        D = logFor(self.graph).get_D(parameter)
        self._comparison('logFor', D, [0.89, 1., 1.47])

    def test_chain_For(self):
        parameter = Linear(self.graph).scale(1.0)
        D = For(self.graph).get_D(parameter)
        self._comparison('For', D, [1.08, 1.32, 1.26])

    def test_chain_pWalk_45(self):
        parameter = AlphaToT(self.graph).scale(4.5)
        D = pWalk(self.graph).get_D(parameter)
        self._comparison('pWalk 4.5', D, [1.08, 1.28, 0.95])

    def test_chain_pWalk_1(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = pWalk(self.graph).get_D(parameter)
        self._comparison('pWalk 1.0', D, [0.96, 1.46, 1.03])
