import unittest

from graphs import sample
from measure.distance import *
from measure.kernel import For_H
from measure.shortcuts import *


class ShortcutsTests(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.A = np.array([
            [1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        self.D = np.array([
            [3, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.L = np.array([
            [2, -1, 0, 0, -1, 0],
            [-1, 3, -1, 0, -1, 0],
            [0, -1, 2, -1, 0, 0],
            [0, 0, -1, 3, -1, -1],
            [-1, -1, 0, -1, 3, 0],
            [0, 0, 0, -1, 0, 1]
        ])

    def test_get_D(self):
        D = get_D(self.A)
        self.assertTrue(np.array_equal(D, self.D))

    def test_get_L(self):
        L = get_L(self.A)
        self.assertTrue(np.array_equal(L, self.L))


class MeasureCommonTests(unittest.TestCase):
    def test_chain_all_distances_more_than_zero(self):
        start, end, n_params = 0.1, 0.6, 30
        for distance in Distance.get_all():
            distance = distance(sample.chain_graph)
            for idx, param in enumerate(distance.scaler.scale_list(np.linspace(start, end, n_params))):
                D = distance.get_D(param)
                for i in range(D.shape[0]):
                    for j in range(D.shape[1]):
                        self.assertTrue(D[i][j] >= 0,
                                        "{} < 0 at {}({}) {}/{}".format(D[i][j], distance.name, param, idx, n_params))

    def test_chain_all_distances_symmetry_matrix(self):
        start, end, n_params = 0.1, 0.6, 30
        for distance in Distance.get_all():
            distance = distance(sample.chain_graph)
            for param in distance.scaler.scale_list(np.linspace(start, end, n_params)):
                D = distance.get_D(param)
                for i in range(D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        self.assertTrue(np.allclose(D[i][j], D[j][i]),
                                        "{:0.3f} != {:0.3f} at {}, {}".format(D[i][j], D[j][i], i, j))

    def test_chain_all_distances_main_diagonal_zero(self):
        start, end, n_params = 0.0001, 0.5, 30
        for distance in Distance.get_all():
            distance = distance(sample.chain_graph)
            for param in np.linspace(start, end, n_params):
                D = distance.get_D(param)
                for i in range(D.shape[0]):
                    self.assertTrue(D[i][i] == 0)

    def test_full_graph_SP_logFor_Walk_equality(self):
        param = 0.00001
        DSP = normalize(sp_distance(sample.chain_graph))
        DlogFor = normalize(logFor(sample.chain_graph).get_D(param))
        DWalk = normalize(Walk(sample.chain_graph).get_D(param))
        self.assertTrue(np.allclose(DSP, DlogFor, atol=0.01))
        self.assertTrue(np.allclose(DWalk, DSP, atol=0.01))
        self.assertTrue(np.allclose(DlogFor, DWalk, atol=0.01))
