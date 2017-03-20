import unittest

import numpy as np

from measure import shortcuts


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
        D = shortcuts.getD(self.A)
        self.assertTrue(np.array_equal(D, self.D))

    def test_get_L(self):
        L = shortcuts.getL(self.A)
        self.assertTrue(np.array_equal(L, self.L))
