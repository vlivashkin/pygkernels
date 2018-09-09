import logging

import numpy as np


class StochasticBlockModel2:
    def __init__(self, n_nodes, n_classes, p_in, p_out):
        self.n = n_nodes
        self.k = n_classes
        self.p_in = p_in
        self.p_out = p_out

    def generate_graph(self):
        nodes = []
        for i in range(self.k):
            nodes.extend([i] * (self.n // self.k))

        edges = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1c, self.n):
                p = self.p_in if nodes[i] == nodes[j] else self.p_out
                k = np.random.choice([0, 1], p=[1 - p, p])
                if k:
                    edges[i][j] = 1
                    edges[j][i] = 1

        return edges, nodes

    def generate_graphs(self, n_graphs):
        logging.info('StochasticBlockModel2: count={}, n={}, k={}, p_in={}, p_out={}'.format(
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
    model = StochasticBlockModel2(100, 2, 0.3, 0.1)
    model.generate_graph()
