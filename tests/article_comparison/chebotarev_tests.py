import unittest

from graphs import sample
from measure.distance import *
from measure.scaler import AlphaToT
from measure.shortcuts import *


# Chebotarev: Studying new classes of graph metrics
# https://arxiv.org/abs/1305.7514

class Figure1ComparisonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.chain_graph

    def _comparison(self, D, true_values, atol=0.04):
        print("True: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(*true_values))
        print("Test: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(D[0, 1], D[1, 2], D[0, 2], D[0, 3]))
        D *= 3. / (D[0, 1] + D[1, 2] + D[2, 3])
        print("Test: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(D[0, 1], D[1, 2], D[0, 2], D[0, 3]))
        for d, t in zip([D[0, 1], D[1, 2], D[0, 2], D[0, 3]][:len(true_values)], true_values):
            self.assertTrue(np.isclose(d, t, atol=atol),
                            "Test {:0.3f} != True {:0.3f}, diff={:0.3f}".format(d, t, np.abs(d - t)))

    def test_chain_SP(self):
        D = D_SP(self.graph)
        self._comparison(D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_R(self):
        D = H_to_D(H_R(self.graph))
        self._comparison(D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_Walk(self):
        parameter = list(AlphaToT(self.graph).scale([1.0]))[0]
        D = Walk(self.graph).get_D(parameter)
        self._comparison(D, [1.025, 0.950, 1.975, 3.000])

    def test_chain_logFor(self):
        D = logFor(self.graph).get_D(2.0)
        self._comparison(D, [0.959, 1.081, 2.040, 3.000])

    def test_chain_For(self):
        D = For(self.graph).get_D(1.0)
        self._comparison(D, [1.026, 0.947, 1.500, 1.895])

    def test_chain_SqResistance(self):
        D = np.sqrt(H_to_D(H_R(self.graph)))
        self._comparison(D, [1.000, 1.000, 1.414, 1.732])

    def test_chain_Comm(self):
        D = Comm(self.graph).get_D(1.0)
        self._comparison(D, [0.964, 1.072, 1.492, 1.564])

    def test_chain_pWalk(self):
        parameter = AlphaToT(self.graph).scale([4.5]).__next__()
        D = pWalk(self.graph).get_D(parameter)
        self._comparison(D, [1.025, 0.950, 1.541, 1.466])

        parameter = AlphaToT(self.graph).scale([1.0]).__next__()
        D = pWalk(self.graph).get_D(parameter)
        self._comparison(D, [0.988, 1.025, 1.379, 1.416])

    if __name__ == '__main__':
        unittest.main()
