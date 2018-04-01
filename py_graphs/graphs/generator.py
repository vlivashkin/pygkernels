import numpy as np


class StochasticBlockModel:
    def generate_graph(self, n, k, p_in, p_out):
        cluster_size = n // k
        edges = np.zeros((cluster_size * k, cluster_size * k))
        nodes = np.random.choice(range(k), edges.shape[1])
        random_pin = np.random.choice([0, 1], edges.shape, p=[1 - p_in, p_in])
        random_pout = np.random.choice([0, 1], edges.shape, p=[1 - p_out, p_out])

        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                edges[i][j] = random_pin[i][j] if nodes[i] == nodes[j] else random_pout[i][j]

        return edges, nodes

    def generate_graphs(self, count, n, k, p_in, p_out):
        info = {
            'name': 'count:{}, n:{}, k:{}, p_in:{}, p_out:{}'.format(count, n, k, p_in, p_out),
            'count': count,
            'n': n,
            'k': k,
            'p_in': p_in,
            'p_out': p_out
        }
        graphs = [self.generate_graph(n, k, p_in, p_out) for _ in range(count)]
        return graphs, info
