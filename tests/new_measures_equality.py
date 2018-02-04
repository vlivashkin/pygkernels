import unittest

from graphs import sample
from measure.kernel import *
from measure.kernel_new import *


class NewMeasuresEqualutyTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.diploma_matrix

    def test_katz(self):
        walk = Walk_H(self.graph)
        katz = Katz(self.graph)
        for param in scaler.Rho(self.graph).scale(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(walk.get_K(param).ravel(), katz.get_K(param).ravel(), atol=0.0001))

    def test_estrada(self):
        comm = logComm_H(self.graph)
        estrada = Estrada(self.graph)
        for param in scaler.Fraction().scale(np.linspace(0.1, 0.7, 50)):
            print(param)
            self.assertTrue(np.allclose(comm.get_K(param).ravel(), estrada.get_K(param).ravel(), atol=0.0001))

    def test_heat(self):
        heat = logHeat_H(self.graph)
        heat_new = Heat(self.graph)
        for param in scaler.Fraction().scale(np.linspace(0.1, 0.7, 50)):
            print(param)
            self.assertTrue(np.allclose(heat.get_K(param).ravel(), heat_new.get_K(param).ravel(), atol=0.0001))

    def test_regularized_laplacian(self):
        forest = logFor_H(self.graph)
        reg_laplacian = RegularizedLaplacian(self.graph)
        for param in scaler.Fraction().scale(np.linspace(0.1, 0.9, 50)):
            print(param)
            self.assertTrue(np.allclose(forest.get_K(param).ravel(), reg_laplacian.get_K(param).ravel(), atol=0.0001))

    if __name__ == '__main__':
        unittest.main()
