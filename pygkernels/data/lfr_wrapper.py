import logging
from collections import defaultdict, Counter

import networkx as nx
import numpy as np
from community import community_louvain
from networkx.generators.community import LFR_benchmark_graph
from scipy.optimize import curve_fit

from pygkernels.data.graph_generator import GraphGenerator


def linear(x, a, b):
    return x * a + b


def estimate_mu(graph, partition):
    Eout = 0
    Gsize = graph.size()
    for n1, n2 in graph.edges():  # links:
        if partition[n1] != partition[n2]:
            Eout += 1
    return float(Eout) / Gsize


class LFRGenerator(GraphGenerator):
    def __init__(self, n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community):
        self.n = n
        self.tau1 = tau1
        self.tau2 = tau2
        self.mu = mu
        self.average_degree = average_degree
        self.max_degree = max_degree
        self.min_community = min_community
        self.max_community = max_community
        logging.info(f'LFR params: {" ".join(self.generate_info())}')

    @classmethod
    def from_graph(cls, G) -> 'LFRGenerator':
        n = G.number_of_nodes()  # estimate N

        # estimate average_degree and max_degree
        M = G.number_of_edges()
        degree_sequence = [d for n, d in G.degree()]  # nx.degree(G).values()
        max_degree = max(degree_sequence)
        average_degree = 2 * float(M) / n

        # estimate mu
        partition = community_louvain.best_partition(G)
        mu = estimate_mu(G, partition)

        # estimate min_community and max_community
        communities = defaultdict(lambda: 0)
        for i in partition.values():
            communities[i] += 1
        sizes = communities.values()
        min_community = min(sizes)
        max_community = max(sizes)

        # estimate tau1
        degree_count = sorted(Counter(degree_sequence).items(), key=lambda x: x[0], reverse=True)
        deg, cnt = zip(*degree_count)
        cnt = np.cumsum(cnt)
        params, _ = curve_fit(linear, np.log(deg), np.log(cnt))
        tau1 = -1 * params[0] + 1

        # estimate t2
        size_count = sorted(Counter(sizes).items(), key=lambda x: x[0], reverse=True)
        size, cnt = zip(*size_count)
        cnt = np.cumsum(cnt)
        params, _ = curve_fit(linear, np.log(size), np.log(cnt), maxfev=10000)
        tau2 = -1 * params[0] + 1

        return LFRGenerator(n, tau1, tau2, mu, average_degree, max_degree, min_community, max_community)

    def generate_info(self, n_graphs=None):
        info = {
            'name': f'LFR',
            'n': self.n,
            'tau1': self.tau1,
            'tau2': self.tau2,
            'mu': self.mu,
            'average_degree': self.average_degree,
            'max_degree': self.max_degree,
            'min_community': self.min_community,
            'max_community': self.max_community
        }
        if n_graphs is not None:
            info['n_graphs'] = n_graphs
        return info

    def generate_graph(self, seed=None):
        G = LFR_benchmark_graph(self.n, self.tau1, self.tau2, self.mu, average_degree=self.average_degree,
                                max_degree=self.max_degree, min_community=self.min_community,
                                max_community=self.max_community, seed=seed)
        nodes_order, partition = zip(*nx.get_node_attributes(G, 'community').items())
        A = nx.adjacency_matrix(G, nodelist=nodes_order)
        return A, partition


if __name__ == '__main__':
    from pygkernels.data import Datasets

    (A, partition), info = Datasets()['news_2cl1_0.1']
    gen = LFRGenerator.from_adj_matrix(A, partition)
    print(gen.generate_info())
    A, partition = gen.generate_graph()
    print(A.shape)
    gen = LFRGenerator.from_adj_matrix(A, partition)
    print(gen.generate_info())
