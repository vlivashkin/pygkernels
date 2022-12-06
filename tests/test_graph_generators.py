import unittest
from collections import Counter

import numpy as np

from pygkernels.data import StochasticBlockModel
from tests.article_comparison._rubanov_sbm_model import RubanovStochasticBlockModel


class TestGraphGenerators(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_nodes, self.n_classes = 100, 2
        self.p_in, self.p_out = 0.3, 0.1

    def _check_shapes(self, graphs, n_nodes, n_classes):
        print("X[0].shape", graphs[0][0].shape)
        print("Counter(y[0])", Counter(graphs[0][1]))

        self.assertTrue(all([X.shape == (n_nodes, n_nodes) for X, _ in graphs]))
        for _, y in graphs:
            if list(Counter(y).values()) != [n_nodes // n_classes] * n_classes:
                print(list(Counter(y).values()), [n_nodes // n_classes] * n_classes)
        self.assertTrue(all([list(Counter(y).values()) == [n_nodes // n_classes] * n_classes for _, y in graphs]))

    def _check_probabilities(self, graphs, p_in, p_out):
        whole_in_stat, whole_out_stat = [], []
        for X, y in graphs:
            in_stat, out_stat = [], []
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[1]):
                    stat = in_stat if y[i] == y[j] else out_stat
                    stat.append(X[i][j])
            in_mean, out_mean = np.mean(in_stat), np.mean(out_stat)
            whole_in_stat.append(in_mean)
            whole_out_stat.append(out_mean)
        whole_in_stat, whole_out_stat = np.mean(whole_in_stat), np.mean(whole_out_stat)

        print("In:", whole_in_stat, p_in)
        print("Out:", whole_out_stat, p_out)

        self.assertTrue(np.isclose(whole_in_stat, p_in, atol=0.001))
        self.assertTrue(np.isclose(whole_out_stat, p_out, atol=0.001))

    def test_stochasticblockmodel_pinpout(self):
        model = StochasticBlockModel(self.n_nodes, 2, p_in=self.p_in, p_out=self.p_out)
        graphs, _ = model.generate_graphs(1000)
        self._check_shapes(graphs, self.n_nodes, self.n_classes)
        self._check_probabilities(graphs, self.p_in, self.p_out)

    def test_stochasticblockmodel_pmatrix(self):
        model = StochasticBlockModel(
            self.n_nodes, 2, probability_matrix=np.array([[self.p_in, self.p_out], [self.p_out, self.p_in]])
        )
        graphs, _ = model.generate_graphs(1000)
        self._check_shapes(graphs, self.n_nodes, self.n_classes)
        self._check_probabilities(graphs, self.p_in, self.p_out)

    def test_stochasticblockmodel_clustersizes(self):
        model = StochasticBlockModel(
            self.n_nodes, 2, cluster_sizes=[self.n_nodes // 2, self.n_nodes // 2], p_in=self.p_in, p_out=self.p_out
        )
        graphs, _ = model.generate_graphs(1000)
        self._check_shapes(graphs, self.n_nodes, self.n_classes)
        self._check_probabilities(graphs, self.p_in, self.p_out)

    def test_rubanovsmodel(self):
        model = RubanovStochasticBlockModel(
            np.array([50, 50]), np.array([[self.p_in, self.p_out], [self.p_out, self.p_in]])
        )
        graphs, _ = model.generate_graphs(1000)
        self._check_shapes(graphs, self.n_nodes, self.n_classes)
        self._check_probabilities(graphs, self.p_in, self.p_out)


if __name__ == "__main__":
    unittest.main()
