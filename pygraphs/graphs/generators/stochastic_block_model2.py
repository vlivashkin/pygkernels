import logging

import numpy as np


class StochasticBlockModel2:
    def __init__(self, n, k, p_in, p_out):
        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out

    def generate_graph(self):
        nodes = []
        for i in range(self.k):
            nodes.extend([i] * (self.n // self.k))

        edges = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                is_in = nodes[i] == nodes[j]
                k = np.random.choice([0, 1], 1, p=[0.7, 0.3]) if is_in else np.random.choice([0, 1], 1, p=[0.9, 0.1])
                if (is_in and k) or (not is_in and k):
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
