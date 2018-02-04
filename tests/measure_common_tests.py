import unittest

from measure.distance import Distance
from measure.shortcuts import *
from graphs import sample


class MeasureCommonTests(unittest.TestCase):
    def test_chain_all_distances_more_than_zero(self):
        start, end, n_params = 0.1, 0.6, 30
        for distance in Distance.get_all():
            distance = distance(sample.chain_graph)
            for idx, param in enumerate(distance.scaler.scale(np.linspace(start, end, n_params))):
                D = distance.get_D(param)
                for i in range(D.shape[0]):
                    for j in range(D.shape[1]):
                        self.assertTrue(D[i][j] >= 0,
                                        "{} < 0 at {}({}) {}/{}".format(D[i][j], distance.name, param, idx, n_params))

    def test_chain_all_distances_symmetry_matrix(self):
        start, end, n_params = 0.1, 0.6, 30
        for distance in Distance.get_all():
            distance = distance(sample.chain_graph)
            for param in distance.scaler.scale(np.linspace(start, end, n_params)):
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

    if __name__ == '__main__':
        unittest.main()
