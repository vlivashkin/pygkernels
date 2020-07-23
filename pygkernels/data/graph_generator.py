import logging

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pygkernels.data.utils import np2nx


class GraphGenerator:
    @classmethod
    def params_from_adj_matrix(cls, A, partition, k):
        return cls.params_from_graph(np2nx(A, partition), k)

    @classmethod
    def params_from_graph(cls, G, k) -> 'GraphGenerator':
        raise NotImplementedError()

    def generate_graph(self, seed=None) -> (np.ndarray, np.ndarray):
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
            graphs_range = tqdm(graphs_range, desc=f'{self.__class__.__name__}')
        if is_connected:  # may take time, parallel will be handy
            graphs = Parallel(n_jobs=n_jobs)(delayed(self.generate_connected_graph)(idx) for idx in graphs_range)
        else:
            graphs = [self.generate_graph(idx) for idx in graphs_range]
        return graphs, info
