import logging

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class GraphGenerator:
    @classmethod
    def from_adj_matrix(cls, A, partition):
        G = nx.from_numpy_matrix(A)
        nx.set_node_attributes(G, partition, 'community')
        return cls.from_graph(G)

    @classmethod
    def from_graph(cls, G) -> 'GraphGenerator':
        raise NotImplementedError()

    def generate_graph(self, seed=None) -> (np.array, np.array):
        raise NotImplementedError()

    def generate_info(self, n_graphs) -> dict:
        raise NotImplementedError()

    def generate_connected_graph(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A, partition = self.generate_graph()
        while not nx.is_connected(nx.from_numpy_matrix(A)):
            A, partition = self.generate_graph()
        return A, partition

    def generate_graphs(self, n_graphs, is_connected=True, verbose=False, n_jobs=6):
        logging.info(f'{self.__class__.__name__}: count={n_graphs}')

        info = self.generate_info(n_graphs)
        graphs_range = range(n_graphs)
        if verbose:
            print(self.name)
            graphs_range = tqdm(graphs_range, desc=f'SBM')
        if is_connected:  # may take time, parallel will be handy
            graphs = Parallel(n_jobs=n_jobs)(delayed(self.generate_connected_graph)(idx) for idx in graphs_range)
        else:
            graphs = [self.generate_graph(idx) for idx in graphs_range]
        return graphs, info
