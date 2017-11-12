import unittest

from graphs import sample
from measure.kernel import *
from measure.kernel_new import *


class NewMeasuresEqualutyTests(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.graph = sample.diploma_matrix

    def katz(self):
        walk = Walk_H(self.graph)
        katz = Katz(self.graph)
        for param in scaler.Rho(self.graph).scale(np.linspace(0, 1, 50)):
            self.assertTrue(np.allclose(walk.getK(param), katz.getK(param)))

    def estrada(self):
        comm = Comm_H(self.graph)
        estrada = Estrada(self.graph)
        for param in scaler.Rho(self.graph).scale(np.linspace(0, 1, 50)):
            self.assertTrue(np.allclose(comm.getK(param), estrada.getK(param)))

    def heat(self):
        heat = Heat_H(self.graph)
        heat_new = Heat(self.graph)
        for param in scaler.Rho(self.graph).scale(np.linspace(0, 1, 50)):
            self.assertTrue(np.allclose(heat.getK(param), heat_new.getK(param)))

    def regularized_laplacian(self):
        forest = For_H(self.graph)
        reg_laplacian = RegularizedLaplacian(self.graph)
        for param in scaler.Rho(self.graph).scale(np.linspace(0, 1, 50)):
            self.assertTrue(np.allclose(forest.getK(param), reg_laplacian.getK(param)))
