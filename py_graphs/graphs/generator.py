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

        print('Parameters for generation:\nn={}, k={}, p_in={}, p_out={}, cluster_sizes={}'.format(
            self.n, self.k, self.p_in, self.p_out, self.cluster_sizes))

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

    def generate_graphs(self, count):
        info = {
            'name': 'count:{}, n:{}, k:{}, p_in:{}, p_out:{}'.format(count, self.n, self.k, self.p_in, self.p_out),
            'count': count,
            'n': self.n,
            'k': self.k,
            'p_in': self.p_in,
            'p_out': self.p_out
        }
        graphs = [self.generate_graph() for _ in range(count)]
        return graphs, info
