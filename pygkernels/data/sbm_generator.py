import logging

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class StochasticBlockModel:
    def __init__(self, n_nodes, n_classes, cluster_sizes=None, p_in=None, p_out=None, probability_matrix=None):
        self.n = n_nodes
        self.k = n_classes

        if p_in is not None and p_out is not None and probability_matrix is None:
            self.p_in = p_in
            self.p_out = p_out
            self.probability_matrix_mode = False
        elif p_in is None and p_out is None and probability_matrix is not None:
            assert probability_matrix.shape[0] == n_classes and probability_matrix.shape[1] == n_classes
            self.probability_matrix = probability_matrix
            self.probability_matrix_mode = True
        else:
            raise ValueError('provide either (p_in, p_out) or probability_matrix')

        if cluster_sizes is not None:
            assert len(cluster_sizes) == n_classes
            self.cluster_sizes = cluster_sizes
            self.cluster_sizes_mode = False
        else:
            self.cluster_sizes = [self.n // self.k] * self.k
            self.cluster_sizes_mode = True

        name_parts = ['n={}'.format(self.n), 'k={}'.format(self.k)]
        if self.cluster_sizes_mode:
            name_parts.append('cluster_sizes mode')
        if self.probability_matrix_mode:
            name_parts.append('probability_matrix mode')
        else:
            name_parts.append(f'p_in={self.p_in}, p_out={self.p_out}')
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
            'cluster_sizes_mode': self.cluster_sizes_mode,
            'probability_matrix_mode': self.probability_matrix_mode
        }
        if not self.probability_matrix_mode:
            info.update({
                'p_in': self.p_in,
                'p_out': self.p_out
            })
        graphs_range = tqdm(range(n_graphs), desc=f'SBM {self.n}, {self.k}') if verbose else range(n_graphs)
        if is_connected:  # may take time, parallel will be handy
            graphs = Parallel(n_jobs=n_jobs)(delayed(self.generate_connected_graph)(idx) for idx in graphs_range)
        else:
            graphs = [self.generate_graph(idx) for idx in graphs_range]
        return graphs, info
