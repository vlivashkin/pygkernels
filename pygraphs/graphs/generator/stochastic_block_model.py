import logging

import numpy as np


class StochasticBlockModel:
    def __init__(self, n_nodes, n_classes, p_in, p_out, cluster_sizes=None):
        self.n = n_nodes
        self.k = n_classes
        self.p_in = p_in
        self.p_out = p_out

        if cluster_sizes is None:
            self.cluster_sizes = self.n // self.k

    def generate_graph(self):
        nodes = []
        for i in range(self.k):
            nodes.extend([i] * self.cluster_sizes[i])

        edges = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p = self.p_in if nodes[i] == nodes[j] else self.p_out
                k = np.random.choice([0, 1], p=[1 - p, p])
                if k:
                    edges[i][j] = 1
                    edges[j][i] = 1

        return edges, nodes

    def generate_graphs(self, n_graphs):
        logging.info('StochasticBlockModel: count={}, n={}, k={}, p_in={}, p_out={}'.format(
            n_graphs, self.n, self.k, self.p_in, self.p_out))

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
