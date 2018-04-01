import unittest

from graphs import sample
from measure.kernel import logHeat_H, logFor_H, logComm_H, Walk_H
from measure.kernel_new import *
from measure.shortcuts import *


# Konstantin Avrachenkov, Pavel Chebotarev, Dmytro Rubanov: Kernels on Graphs as Proximity Measures
# https://hal.inria.fr/hal-01647915/document


# class BalancedModel(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         generator = StochasticBlockModelGraphGenerator()
#         self.graphs, _ = generator.generate_graphs(100, 200, 2, 0.1, 0.02)
#
#     def _compare(self, measure, y_need):
#         results = defaultdict(lambda: [])
#         count, passes = 0, 0
#         for A, y_true in self.graphs:
#             mg = measure(A)
#             for param in mg.scaler.scale(np.linspace(0, 1, 50)):
#                 try:
#                     count += 1
#                     K = mg.get_K(param)
#                     y_pred = KernelKMeans(2).fit_predict(K)
#                     results[param].append(max_accuracy(y_true, y_pred))
#                     passes += 1
#                 except:
#                     pass
#         print('Passes: {}/{}'.format(passes, count))
#         y_final = 1. - np.max([np.average(x) for x in results.values()])
#         print('Min error: {:0.3f}'.format(y_final))
#         self.assertTrue(np.isclose(y_final, y_need, atol=0.002),
#                         "Test {:0.3f} != True {:0.3f}, diff={:0.3f}".format(y_final, y_need, np.abs(y_final - y_need)))
#
#     def test_katz(self):
#         self._compare(Katz, 0.0072)
#
#     def test_communicability(self):
#         self._compare(Estrada, 0.0084)
#
#     def test_heat(self):
#         self._compare(Heat_H, 0.0064)
#
#     def test_normalizedHeat(self):
#         self._compare(NormalizedHeat, 0.0066)
#
#     def test_regularizedLaplacian(self):
#         self._compare(RegularizedLaplacian, 0.0072)
#
#     def test_personalizedPageRank(self):
#         self._compare(PersonalizedPageRank, 0.0073)
#
#     def test_modifiedPageRank(self):
#         self._compare(ModifiedPersonalizedPageRank, 0.0072)
#
#     def test_heatPageRank(self):
#         self._compare(HeatPersonalizedPageRank, 0.0074)


class NewMeasuresEqualutyTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = sample.diploma_matrix

    def test_katz(self):
        walk = Walk_H(self.graph)
        katz = Katz(self.graph)
        for param in scaler.Rho(self.graph).scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(walk.get_K(param).ravel(), katz.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_estrada(self):
        comm = logComm_H(self.graph)
        estrada = Estrada(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(comm.get_K(param).ravel(), estrada.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_heat(self):
        heat = logHeat_H(self.graph)
        heat_new = Heat(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(heat.get_K(param).ravel(), heat_new.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_regularized_laplacian(self):
        forest = logFor_H(self.graph)
        reg_laplacian = RegularizedLaplacian(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(forest.get_K(param).ravel(), reg_laplacian.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    if __name__ == '__main__':
        unittest.main()
