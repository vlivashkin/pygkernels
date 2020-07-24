from collections import Counter

import numpy as np
from networkx.generators.community import LFR_benchmark_graph
from scipy.optimize import curve_fit

from pygkernels.data.graph_generator import GraphGenerator
from pygkernels.data.utils import nx2np


def estimate_mu(graph, partition):
    n_out_edges = 0
    n_nodes = graph.size()
    for n1, n2 in graph.edges():  # links:
        if partition[n1] != partition[n2]:
            n_out_edges += 1
    return n_out_edges / n_nodes


def power_law(values):
    values_count = sorted(Counter(values).items(), key=lambda x: x[0], reverse=True)
    deg, cnt = zip(*values_count)
    cnt = np.cumsum(cnt)
    params, _ = curve_fit(lambda x, a, b: x * a + b, np.log(deg), np.log(cnt))
    return -1 * params[0] + 1


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
        print(f'LFR params: ' + ', '.join(
            [f"{k}: " + (f"{v:.3f}" if type(v) in (float, np.float64) else str(v)) for k, v in
             self.generate_info().items()]))

    @classmethod
    def params_from_graph(cls, G, k) -> 'LFRGenerator':
        # Graph-based parameters
        n = G.number_of_nodes()
        average_degree = 2 * G.number_of_edges() / n
        node_degrees = [d for n, d in G.degree()]
        max_degree = np.max(node_degrees)
        tau1 = power_law(node_degrees)

        # Community-based parameters
        A, partition = nx2np(G)
        # partition = KKMeans(n_clusters=k).predict(CCT_H(A).get_K(None), A=A)
        mu = estimate_mu(G, partition)
        community_sizes = list(Counter(partition).values())
        min_community = np.min(community_sizes)
        max_community = np.max(community_sizes)
        tau2 = power_law(community_sizes)

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
                                max_community=self.max_community, seed=seed, tol=1e-5, max_iters=100000)
        return nx2np(G)
