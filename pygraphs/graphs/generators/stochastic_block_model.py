import logging
from itertools import product

import numpy as np
from sklearn.utils import deprecated


@deprecated()
class StochasticBlockModel:
    def __init__(self, n_nodes, n_classes, p_in, p_out, cluster_sizes=None, probabilities=None):
        self.n = n_nodes
        self.k = n_classes
        self.p_in = p_in
        self.p_out = p_out

        if cluster_sizes is not None:
            if len(cluster_sizes) == n_classes and np.sum(cluster_sizes) == n_nodes:
                self.cluster_sizes = cluster_sizes
            else:
                raise ValueError()
        else:
            cluster_size = self.n // self.k
            self.cluster_sizes = [cluster_size] * self.k
            self.n = cluster_size * self.k

        if probabilities is not None:
            self.probabilities = probabilities
        else:
            self.probabilities = np.zeros((self.k, self.k))
            for i, j in product(range(self.k), range(self.k)):
                self.probabilities[i][j] = self.p_in if i == j else self.p_out

    def generate_graph(self):
        nodes = []
        for i in range(self.k):
            nodes.extend([i] * self.cluster_sizes[i])

        edges = np.zeros((self.n, self.n))

        random_matrix = [[[] for _ in range(self.k)] for _ in range(self.k)]
        for i in range(self.k):
            for j in range(i, self.k):
                a = np.random.choice([0, 1], edges.shape, p=[1 - self.probabilities[i][j], self.probabilities[i][j]])
                random_matrix[i][j] = a
                random_matrix[j][i] = a

        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                edges[i][j] = random_matrix[nodes[i]][nodes[j]][i][j]

        return edges, nodes

    def generate_graphs(self, n_graphs):
        logging.info('StochasticBlockModel: count={}, n={}, k={}, p_in={}, p_out={}, cluster_sizes={}'.format(
            n_graphs, self.n, self.k, self.p_in, self.p_out, self.cluster_sizes))

        info = {
            'name': 'count:{}, n:{}, k:{}, p_in:{}, p_out:{}'.format(n_graphs, self.n, self.k, self.p_in, self.p_out),
            'n_graphs': n_graphs,
            'n': self.n,
            'k': self.k,
            'p_in': self.p_in,
            'p_out': self.p_out
        }
        graphs = [self.generate_graph() for _ in range(n_graphs)]
        return graphs, info


if __name__ == "__main__":
    model = StochasticBlockModel(100, 2, 0.3, 0.1)
    model.generate_graph()
