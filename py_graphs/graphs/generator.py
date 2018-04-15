import networkx as nx
import numpy as np


class StochasticBlockModel:
    def __init__(self, n, k, p_in, p_out, cluster_sizes=None):
        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out

        if cluster_sizes is not None:
            if len(cluster_sizes) == k and np.sum(cluster_sizes) == n:
                self.cluster_sizes = cluster_sizes
            else:
                raise ValueError()
        else:
            cluster_size = self.n // self.k
            self.cluster_sizes = [cluster_size] * self.k
            self.n = cluster_size * self.k

    def generate_graph(self):
        nodes = []
        for i in range(self.k):
            nodes.extend([i] * self.cluster_sizes[i])

        edges = np.zeros((self.n, self.n))
        random_pin = np.random.choice([0, 1], edges.shape, p=[1 - self.p_in, self.p_in])
        random_pout = np.random.choice([0, 1], edges.shape, p=[1 - self.p_out, self.p_out])
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                edges[i][j] = random_pin[i][j] if nodes[i] == nodes[j] else random_pout[i][j]

        return edges, nodes

    def generate_graphs(self, n_graphs):
        print('StochasticBlockModel: count={}, n={}, k={}, p_in={}, p_out={}, cluster_sizes={}'.format(
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
            print("Not connected")

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
