import logging

import networkx as nx
import numpy as np


class RubanovModel():
    def __init__(self, sizes, probs):
        self.sizes = sizes
        self.sums = np.cumsum(self.sizes)
        self.probs = probs

    def _get_graph(self, sizes, probs):
        n_comms = len(sizes)
        sums = np.cumsum(sizes)
        ranges = [range(sums[0])] + [range(sums[k - 1], sums[k]) for k in range(1, n_comms)]
        g = nx.Graph()
        g.add_nodes_from(range(sums[-1]))
        for i in range(n_comms):
            for x in ranges[i]:
                for y in range(x + 1, sums[i]):
                    k = np.random.rand()
                    if k < probs[i, i]:
                        g.add_edge(x, y)
            for j in range(i + 1, n_comms):
                for x in ranges[i]:
                    for y in ranges[j]:
                        k = np.random.rand()
                        if k < probs[i, j]:
                            g.add_edge(x, y)
        return g

    def generate_graph(self):
        while True:
            g = self._get_graph(self.sizes, self.probs)
            if nx.is_connected(g):
                return np.array(np.array(nx.adjacency_matrix(g).todense()))
            logging.info("Not connected")

    def _generate_y_true(self):
        y_true = []
        for idx, cls_size in enumerate(self.sizes):
            y_true.extend([idx] * cls_size)
        return np.array(y_true)

    def generate_graphs(self, n_graphs):
        graphs = []
        y_true = self._generate_y_true()
        for _ in range(n_graphs):
            graphs.append((self.generate_graph(), y_true))

        return graphs, {}
