from collections import Counter

import numpy as np
import powerlaw
from networkx.algorithms.communities import LFR_benchmark_graph

from pygkernels.data.graph_generator import GraphGenerator
from pygkernels.data.utils import nx2np


def estimate_mu(graph, partition):
    n_out_edges = 0
    n_nodes = graph.size()
    for n1, n2 in graph.edges():  # links:
        if partition[n1] != partition[n2]:
            n_out_edges += 1
    return n_out_edges / n_nodes


def power_law(values, maxval=200):
    tau = powerlaw.Fit(values, verbose=False).alpha
    if tau > maxval or np.isnan(tau):
        tau = maxval
    return tau


class LFRGenerator(GraphGenerator):
    def __init__(self, n, tau1, tau2, mu, average_degree=None, min_degree=None, max_degree=None, min_community=None,
                 max_community=None, name='LFR'):
        self.name = name
        self.n = n
        self.k = None
        self.tau1 = tau1
        self.tau2 = tau2
        self.mu = mu
        self.average_degree = average_degree
        self.min_degree = min_degree  # min_degree
        self.max_degree = max_degree
        self.min_community = min_community
        self.max_community = max_community
        # print(f'LFR params: ' + ', '.join([f"{k}: " + (f"{v:.3f}" if type(v) in (float, np.float64) else str(v))
        #                                    for k, v in self.generate_info().items()]))

    @classmethod
    def params_from_graph(cls, G, name='LFR') -> 'LFRGenerator':
        # Graph-based parameters
        n = G.number_of_nodes()
        average_degree = 2 * G.number_of_edges() / n
        node_degrees = [d for n, d in G.degree()]
        min_degree = np.min(node_degrees)
        max_degree = np.max(node_degrees)
        tau1 = power_law(node_degrees, 17)

        # Community-based parameters
        A, partition = nx2np(G)
        # partition = KKMeans(n_clusters=k).predict(CCT_H(A).get_K(None), A=A)
        mu = min(max(estimate_mu(G, partition), 0.05), 0.30)
        community_sizes = list(Counter(partition).values())
        min_community = np.min(community_sizes)
        max_community = np.max(community_sizes)
        tau2 = power_law(community_sizes, 200)

        return LFRGenerator(n, tau1, tau2, mu, average_degree, min_degree, max_degree, min_community, max_community,
                            name=name)

    def generate_info(self, n_graphs=None):
        info = {
            'name': self.name,
            'n': self.n,
            'k': self.k,
            'tau1': self.tau1,
            'tau2': self.tau2,
            'mu': self.mu,
            'average_degree': self.average_degree,
            'min_degree': self.min_degree,
            'max_degree': self.max_degree,
            'min_community': self.min_community,
            'max_community': self.max_community
        }
        if n_graphs is not None:
            info['n_graphs'] = n_graphs
        return info

    def generate_graph(self, seed=None):
        G = LFR_benchmark_graph(self.n, self.tau1, self.tau2, self.mu, average_degree=self.average_degree,
                                min_degree=self.min_degree, max_degree=self.max_degree,
                                min_community=self.min_community, max_community=self.max_community, tol=1.0e-3,
                                max_iters=500, seed=seed)
        A, partition = nx2np(G)
        self.k = len(set(partition))
        return A, partition
