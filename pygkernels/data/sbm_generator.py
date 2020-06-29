import logging

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class StochasticBlockModel:
    def __init__(self, n_nodes, n_classes, cluster_sizes=None, balance=None, p_in=None, p_out=None,
                 probability_matrix=None):
        self.n = n_nodes
        self.k = n_classes

        assert (p_in is None) ^ (probability_matrix is None), 'provide either (p_in, p_out) or probability_matrix'
        if p_in is not None and p_out is not None:
            self.p_in = p_in
            self.p_out = p_out
            self.probability_matrix_mode = False
            probability_matrix_str = f'p_in={self.p_in}, p_out={self.p_out}'
        else:  # probability_matrix is not None
            assert probability_matrix.shape[0] == n_classes and probability_matrix.shape[1] == n_classes
            self.probability_matrix = probability_matrix
            self.probability_matrix_mode = True
            probability_matrix_str = 'probability matrix'

        assert not (cluster_sizes is not None and balance is not None), 'can\'t fill both cluster_sizes and balance'
        if balance is not None:
            softmax = lambda x, beta: np.exp(beta * x) / np.sum(np.exp(beta * x), axis=0)
            self.cluster_sizes = ([1] * self.k + (self.n - self.k) * softmax(np.arange(self.k)[::-1], beta=balance))\
                .astype(np.int)
            cluster_sizes_str = f'balance={balance:.2f}'
        elif cluster_sizes is not None:
            assert len(cluster_sizes) == n_classes
            self.cluster_sizes = cluster_sizes
            cluster_sizes_str = f'cluster_sizes={cluster_sizes}'
        else:
            self.cluster_sizes = [self.n // self.k] * self.k
            cluster_sizes_str = f'equal classes'
        self.cluster_sizes[0] += self.n - np.sum(self.cluster_sizes)

        name_parts = ['n={}'.format(self.n), 'k={}'.format(self.k), cluster_sizes_str, probability_matrix_str]
        self.name = ', '.join(name_parts)

    def generate_graph(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        partition = []
        for i in range(self.k):
            partition.extend([i] * self.cluster_sizes[i])

        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.probability_matrix_mode:
                    p = self.probability_matrix[partition[i]][partition[j]]
                else:
                    p = self.p_in if partition[i] == partition[j] else self.p_out
                k = np.random.choice([0, 1], p=[1 - p, p])
                if k:
                    A[i][j] = 1
                    A[j][i] = 1

        return A, partition

    def generate_connected_graph(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A, partition = self.generate_graph()
        while not nx.is_connected(nx.from_numpy_matrix(A)):
            A, partition = self.generate_graph()
        return A, partition

    def generate_graphs(self, n_graphs, is_connected=True, verbose=False, n_jobs=6):
        logging.info('StochasticBlockModel: count={}, {}'.format(n_graphs, self.name))

        info = {
            'name': 'count:{}, {}'.format(n_graphs, self.name),
            'n_graphs': n_graphs,
            'n': self.n,
            'k': self.k,
            'cluster_sizes': self.cluster_sizes,
            'probability_matrix_mode': self.probability_matrix_mode
        }
        if not self.probability_matrix_mode:
            info.update({
                'p_in': self.p_in,
                'p_out': self.p_out
            })
        if verbose:
            print(self.name)
            graphs_range = tqdm(range(n_graphs), desc=f'SBM')
        else:
            graphs_range = range(n_graphs)
        if is_connected:  # may take time, parallel will be handy
            graphs = Parallel(n_jobs=n_jobs)(delayed(self.generate_connected_graph)(idx) for idx in graphs_range)
        else:
            graphs = [self.generate_graph(idx) for idx in graphs_range]
        return graphs, info
