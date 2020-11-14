import logging
import unittest

import numpy as np
from pygkernels import util
from pygkernels.data import Samples
from pygkernels.measure import SP_D, CT_D, logKatz_D, For_D, Comm_D, Katz_D, logFor_D
from pygkernels.measure.scaler import AlphaToT, Linear


class TestFigure1Comparison(unittest.TestCase):
    """
    Chebotarev: Studying new classes of graph metrics
    https://arxiv.org/abs/1305.7514
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = Samples.chain_graph
        util.configure_logging()

    def _comparison(self, name, D, gt, atol=0.001):
        D *= 3. / (D[0, 1] + D[1, 2] + D[2, 3])

        logging.info(f'{name}\tD_12\tD_23\tD_13\tD_14')
        logging.info(f'True\t{gt[0]:.5f}\t{gt[1]:.5f}\t{gt[2]:.5f}\t{gt[3]:.5f}')
        logging.info(f'Test\t{D[0, 1]:.5f}\t{D[1, 2]:.5f}\t{D[0, 2]:.5f}\t{D[0, 3]:.5f}')

        for d, t in zip([D[0, 1], D[1, 2], D[0, 2], D[0, 3]][:len(gt)], gt):
            self.assertTrue(np.isclose(d, t, atol=atol), f'ours:{d:0.3f} != gt:{t:0.3f}, diff={np.abs(d - t):0.3f}')

    def test_chain_SP(self):
        D = SP_D(self.graph).get_D(-1)
        self._comparison('SP', D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_R(self):
        D = CT_D(self.graph).get_D(-1)
        self._comparison('R', D, [1.000, 1.000, 2.000, 3.000])

    def test_chain_logKatz(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = logKatz_D(self.graph).get_D(parameter)
        self._comparison('Walk', D, [1.025, 0.950, 1.975, 3.000])

    def test_chain_logFor(self):
        parameter = Linear(self.graph).scale(2.0)
        D = logFor_D(self.graph).get_D(parameter)
        self._comparison('logFor', D, [0.959, 1.081, 2.040, 3.000])

    def test_chain_For(self):
        parameter = Linear(self.graph).scale(1.0)
        D = For_D(self.graph).get_D(parameter)
        self._comparison('For', D, [1.026, 0.947, 1.500, 1.895])

    def test_chain_SqResistance(self):
        D = np.sqrt(CT_D(self.graph).get_D(-1))
        self._comparison('SqResistance', D, [1.000, 1.000, 1.414, 1.732])

    def test_chain_Comm(self):
        D = Comm_D(self.graph).get_D(1.0)
        self._comparison('Comm', D, [0.964, 1.072, 1.492, 1.564])

    def test_chain_pWalk_45(self):
        parameter = AlphaToT(self.graph).scale(4.5)
        D = Katz_D(self.graph).get_D(parameter)
        self._comparison('pWalk 4.5', D, [1.025, 0.950, 1.541, 1.466])

    def test_chain_pWalk_1(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = Katz_D(self.graph).get_D(parameter)
        self._comparison('pWalk 1.0', D, [0.988, 1.025, 1.379, 1.416])


class TestTable1Comparison(unittest.TestCase):
    """
    Chebotarev: The Walk Distances in Graphs
    https://arxiv.org/abs/1103.2059
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = Samples.chain_graph
        util.configure_logging()

    def _comparison(self, name, D, gt, atol=0.01):
        # logging results for report

        r1 = D[0, 1] / D[1, 2]
        r2 = (D[0, 1] + D[1, 2]) / D[0, 2]
        r3 = D[0, 3] / D[0, 2]

        logging.info(f'{name}\tD_12/D_23\t(D_12+D_23)/D_13\tD_14/D_12')
        logging.info(f'True\t{gt[0]:.4f}\t{gt[1]:.4f}\t{gt[2]:.4f}')
        logging.info(f'Test\t{r1:.4f}\t{r2:.4f}\t{r3:.4f}')

        for d, t in zip([r1, r2, r3], gt):
            self.assertTrue(np.isclose(d, t, atol=atol), f'Test {d:.2f} != True {t:.2f}, diff={np.abs(d - t):.2f}')

    def test_chain_SP(self):
        D = SP_D(self.graph).get_D(-1)
        self._comparison('SP', D, [1., 1., 1.5])

    def test_chain_R(self):
        D = CT_D(self.graph).get_D(-1)
        self._comparison('R', D, [1., 1., 1.5])

    def test_chain_Walk(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = logKatz_D(self.graph).get_D(parameter)
        self._comparison('Walk', D, [1.08, 1., 1.52])

    def test_chain_logFor(self):
        parameter = Linear(self.graph).scale(2.0)
        D = logFor_D(self.graph).get_D(parameter)
        self._comparison('logFor', D, [0.89, 1., 1.47])

    def test_chain_For(self):
        parameter = Linear(self.graph).scale(1.0)
        D = For_D(self.graph).get_D(parameter)
        self._comparison('For', D, [1.08, 1.32, 1.26])

    def test_chain_pWalk_45(self):
        parameter = AlphaToT(self.graph).scale(4.5)
        D = Katz_D(self.graph).get_D(parameter)
        self._comparison('pWalk 4.5', D, [1.08, 1.28, 0.95])

    def test_chain_pWalk_1(self):
        parameter = AlphaToT(self.graph).scale(1.0)
        D = Katz_D(self.graph).get_D(parameter)
        self._comparison('pWalk 1.0', D, [0.96, 1.46, 1.03])


if __name__ == "__main__":
    unittest.main()
