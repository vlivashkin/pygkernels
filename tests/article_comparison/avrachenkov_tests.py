import unittest
from collections import defaultdict

from cluster import KernelKMeans
from graphs import sample
from graphs.generator import StochasticBlockModel
from measure.kernel import logHeat_H, logFor_H, logComm_H, Walk_H
from measure.kernel_new import *
from measure.shortcuts import *
from scorer import max_accuracy


# Konstantin Avrachenkov, Pavel Chebotarev, Dmytro Rubanov: Kernels on Graphs as Proximity Measures
# https://hal.inria.fr/hal-01647915/document

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


class Competition(unittest.TestCase):
    def __init__(self, atol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.atol = atol
        self.graphs = None

    def _search_best_param(self, measure):
        results = defaultdict(lambda: [])
        count, passes = 0, 0
        for A, y_true in self.graphs:
            mg = measure(A)
            for param in mg.scaler.scale(np.linspace(0, 1, 50)):
                try:
                    count += 1
                    K = mg.get_K(param)
                    y_pred = KernelKMeans(2).fit_predict(K)
                    results[param].append(max_accuracy(y_true, y_pred))
                    passes += 1
                except:
                    pass
        pairs = [(key, np.average(value)) for key, value in results.items()]
        best_param, best_quality = pairs[np.argmax([x[1] for x in pairs])]

        # logging results for report
        print('{}; Passes: {}/{}; Min error: {:0.4f}, Best param: {:0.4f}'.format(
            measure.name, passes, count, 1 - best_quality, best_param))

        return best_param, best_quality

    def _use_best_param(self, measure, param):
        results = []
        count, passes = 0, 0
        for A, y_true in self.graphs:
            try:
                count += 1
                K = measure(A).get_K(param)
                y_pred = KernelKMeans(2).fit_predict(K)
                results.append(max_accuracy(y_true, y_pred))
                passes += 1
            except:
                pass
        quality = np.average(results)

        # logging results for report
        print('{}; Passes: {}/{}; Using param {:0.4f}, error: {:0.4f}'.format(
            measure.name, passes, count, param, 1 - quality))

        return quality

    def _compare(self, measure, error_true, param=None):
        if param is not None:
            best_quality = self._use_best_param(measure, param)
        else:
            _, best_quality = self._search_best_param(measure)
        error_test = 1. - best_quality
        diff = np.abs(error_test - error_true)

        # logging results for report
        print('{}; Min error: {:0.4f}, true={:0.4f}, diff={:0.4f}'.format(measure.name, error_test, error_true, diff))

        self.assertTrue(np.isclose(error_test, error_true, atol=self.atol),
                        "Test {:0.4f} != True {:0.4f}, diff={:0.4f}".format(error_test, error_true, diff))


class BalancedModel(Competition):
    def __init__(self, *args, **kwargs):
        super().__init__(0.004, *args, **kwargs)  # error bars in paper: 0.002
        self.graphs, _ = StochasticBlockModel(200, 2, 0.1, 0.02).generate_graphs(100)

    def test_katz(self):
        self._compare(Katz, 0.0072, 0.0017)

    def test_communicability(self):
        self._compare(Estrada, 0.0084, 0.0833)

    def test_heat(self):
        self._compare(Heat, 0.0064, 0.0326)

    def test_normalizedHeat(self):
        self._compare(NormalizedHeat, 0.0066, 0.4074)

    def test_regularizedLaplacian(self):
        self._compare(RegularizedLaplacian, 0.0072, 0.0326)

    def test_personalizedPageRank(self):
        self._compare(PersonalizedPageRank, 0.0073, 0.0104)

    def test_modifiedPageRank(self):
        self._compare(ModifiedPersonalizedPageRank, 0.0072, 0.0568)

    def test_heatPageRank(self):
        self._compare(HeatPersonalizedPageRank, 0.0074, 0.1621)


# class UnbalancedModel(Competition):
#     def __init__(self, *args, **kwargs):
#         super().__init__(0.012, *args, **kwargs)  # error bars in paper: 0.006
#         self.graphs, _ = StochasticBlockModel(200, 2, 0.1, 0.02, [50, 150]).generate_graphs(100)  # 1000 graphs in paper
#
#     def test_katz(self):
#         self._compare(Katz, 0.012, 0.0068)
#
#     def test_communicability(self):
#         self._compare(Estrada, 0.011)
#
#     def test_heat(self):
#         self._compare(Heat, 0.026, 0.0104)
#
#     def test_normalizedHeat(self):
#         self._compare(NormalizedHeat, 0.009)
#
#     def test_regularizedLaplacian(self):
#         self._compare(RegularizedLaplacian, 0.0026)
#
#     def test_personalizedPageRank(self):
#         self._compare(PersonalizedPageRank, 0.0021)
#
#     def test_modifiedPageRank(self):
#         self._compare(ModifiedPersonalizedPageRank, 0.0022)
#
#     def test_heatPageRank(self):
#         self._compare(HeatPersonalizedPageRank, 0.0021)
