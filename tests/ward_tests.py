import unittest

from tests import sample_graphs
from ward import Ward


class ShortcutsTests(unittest.TestCase):
    def test_get_D(self):
        y_pred = Ward().predict(sample_graphs.diploma_matrix, 2)
        print(y_pred)