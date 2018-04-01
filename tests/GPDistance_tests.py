import unittest

import numpy as np

from graphs import dataset
from measure.distance import RSP, FE, GPD_FE, GPD_RSP
from measure.shortcuts import get_L, commute_distance, H_to_D, resistance_kernel


# GPDistance
# https://github.com/jmmcd/GPDistance/blob/master/python/RandomWalks/graph_distances.py

class GPDistance_tests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test1 = {
            'A': np.array([[0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 1],
                           [0, 1, 1, 0]]),
            'RSP_true': np.array([[0.00000, np.inf, np.inf, np.inf],
                                  [np.inf, 0.00000, 2.14516, 1.07258],
                                  [np.inf, 2.14516, 0.00000, 1.07258],
                                  [np.inf, 1.07258, 1.07258, 0.00000]]),
            'FE_true': np.array([[0.00000, np.inf, np.inf, np.inf],
                                 [np.inf, 0.00000, 2.62308, 1.31154],
                                 [np.inf, 2.62308, 0.00000, 1.31154],
                                 [np.inf, 1.31154, 1.31154, 0.00000]])
        }

        self.test2 = {
            'A': np.array([[.5, .25, .25],
                           [.25, .0, .75],
                           [.25, .75, .0]]),
            'RSP_true': np.array([[0.00000, 4.34699, 4.34699],
                                  [4.34699, 0.00000, 1.33429],
                                  [4.34699, 1.33429, 0.00000]]),
            'FE_true': np.array([[0.00000, 5.15091, 5.15091],
                                 [5.15091, 0.00000, 1.62088],
                                 [5.15091, 1.62088, 0.00000]])
        }

    def test_L(self):
        A = np.array([[0, 1, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1],
                      [1, 1, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0]])
        L_true = np.array([[2., -1., 0., 0., -1., 0.],
                           [-1., 3., -1., 0., -1., 0.],
                           [0., -1., 2., -1., 0., 0.],
                           [0., 0., -1., 3., -1., -1.],
                           [-1., -1., 0., -1., 3., 0.],
                           [0., 0., 0., -1., 0., 1.]])

        L_test = get_L(A)
        self.assertTrue(np.array_equal(L_test, L_true))

    def test_compare_CT_and_Resistance(self):
        graphs, info = dataset.news_2cl_1
        A, y_true = graphs[0]
        D_CT = commute_distance(A)
        D_R = H_to_D(resistance_kernel(A))
        D_CT /= np.average(D_CT)
        D_R /= np.average(D_R)
        self.assertTrue(np.allclose(D_CT, D_R))

    def test_1_RSP_vanilla(self):
        A, D_RSP_true = self.test1['A'], self.test1['RSP_true']
        D_RSP_test = RSP(A).get_D(1)
        self.assertTrue(np.allclose(D_RSP_test, D_RSP_true, atol=0.00001, equal_nan=True))

    def test_1_FE_vanilla(self):
        A, D_FE_true = self.test1['A'], self.test1['FE_true']
        D_FE_test = FE(A).get_D(1)
        self.assertTrue(np.allclose(D_FE_test, D_FE_true, atol=0.00001, equal_nan=True))

    def test_1_RSP_GPD(self):
        A, D_RSP_true = self.test1['A'], self.test1['RSP_true']
        D_RSP_test = GPD_RSP(A).get_D(1)
        self.assertTrue(np.allclose(D_RSP_test, D_RSP_true, atol=0.00001, equal_nan=True))

    def test_1_FE_GPD(self):
        A, D_FE_true = self.test1['A'], self.test1['FE_true']
        D_FE_test = GPD_FE(A).get_D(1)
        self.assertTrue(np.allclose(D_FE_test, D_FE_true, atol=0.00001, equal_nan=True))

    def test_2_RSP_vanilla(self):
        A, D_RSP_true = self.test2['A'], self.test2['RSP_true']
        D_RSP_test = RSP(A).get_D(1)
        self.assertTrue(np.allclose(D_RSP_test, D_RSP_true, atol=0.00001))

    def test_2_FE_vanilla(self):
        A, D_FE_true = self.test2['A'], self.test2['FE_true']
        D_FE_test = FE(A).get_D(1)
        self.assertTrue(np.allclose(D_FE_test, D_FE_true, atol=0.00001))

    def test_2_RSP_GPD(self):
        A, D_RSP_true = self.test2['A'], self.test2['RSP_true']
        D_RSP_test = GPD_RSP(A).get_D(1)
        self.assertTrue(np.allclose(D_RSP_test, D_RSP_true, atol=0.00001))

    def test_2_FE_GPD(self):
        A, D_FE_true = self.test2['A'], self.test2['FE_true']
        D_FE_test = GPD_FE(A).get_D(1)
        self.assertTrue(np.allclose(D_FE_test, D_FE_true, atol=0.00001))

    def test_compare_RSP_versions(self):
        graphs, info = dataset.news_2cl_1
        A, y_true = graphs[0]
        my_RSP = RSP(A).get_D(1)
        gpd_RSP = GPD_RSP(A).get_D(1)

        print('max abs my', np.max([x for x in np.abs(my_RSP).ravel() if not np.isnan(x)]))
        print('max abs gpd', np.max([x for x in np.abs(gpd_RSP).ravel() if not np.isnan(x)]))

        self.assertTrue(np.allclose(my_RSP, gpd_RSP, atol=100., equal_nan=True),
                        np.max([x for x in np.abs(my_RSP - gpd_RSP).ravel() if not np.isnan(x)]))

    def test_compare_FE_versions(self):
        graphs, info = dataset.news_2cl_1
        A, y_true = graphs[0]
        my_FE = FE(A).get_D(1)
        gpd_FE = GPD_FE(A).get_D(1)

        print('max abs my', np.max([x for x in np.abs(my_FE).ravel() if not np.isnan(x)]))
        print('max abs gpd', np.max([x for x in np.abs(gpd_FE).ravel() if not np.isnan(x)]))

        self.assertTrue(np.allclose(my_FE, gpd_FE, atol=100., equal_nan=True),
                        np.max([x for x in np.abs(my_FE - gpd_FE).ravel() if not np.isnan(x)]))
