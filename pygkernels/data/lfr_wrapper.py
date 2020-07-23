import logging
from collections import defaultdict, Counter

import numpy as np
from networkx.generators.community import LFR_benchmark_graph
from scipy.optimize import curve_fit

from pygkernels.cluster import KKMeans
from pygkernels.data.graph_generator import GraphGenerator
from pygkernels.data.utils import nx2np
from pygkernels.measure.kernel import CCT_H


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
    def params_from_graph(cls, G, k) -> 'LFRGenerator':
        # Graph-based parameters
        n = G.number_of_nodes()
        degree_sequence = [d for n, d in G.degree()]
        average_degree = 2 * G.number_of_edges() / n
        max_degree = np.max(degree_sequence)

        # estimate tau1 (power law degree distribution)
        degree_count = sorted(Counter(degree_sequence).items(), key=lambda x: x[0], reverse=True)
        deg, cnt = zip(*degree_count)
        cnt = np.cumsum(cnt)
        params, _ = curve_fit(linear, np.log(deg), np.log(cnt))
        tau1 = -1 * params[0] + 1

        # Community-based parameters
        A, _ = nx2np(G)
        partition = KKMeans(n_clusters=2).predict(CCT_H(A).get_K(None), A=A)
        mu = estimate_mu(G, partition)

        communities = defaultdict(lambda: 0)
        for i in partition:
            communities[i] += 1
        sizes = list(communities.values())
        min_community = np.min(sizes)
        max_community = np.max(sizes)

        # estimate tau2
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
        return nx2np(G)
