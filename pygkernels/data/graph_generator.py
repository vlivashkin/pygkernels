import logging

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pygkernels.data.utils import np2nx


class GraphGenerator:
    @classmethod
    def params_from_adj_matrix(cls, A, partition, name=None):
        return cls.params_from_graph(np2nx(A, partition), name=name)

    @classmethod
    def params_from_graph(cls, G, name=None) -> 'GraphGenerator':
        raise NotImplementedError()

    def generate_graph(self, seed=None) -> (np.ndarray, np.ndarray):
        raise NotImplementedError()

    def generate_info(self, n_graphs) -> dict:
        raise NotImplementedError()

    def generate_connected_graph(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        A, partition = self.generate_graph()
        G = nx.from_numpy_matrix(A)
        while not nx.is_connected(G):
            components = list(nx.connected_components(G))
            print(f'not connected! {len(components)} components')
            G.add_edge(np.random.choice(list(components[0])), np.random.choice(list(components[1])))
        return A, partition

    def generate_graphs(self, n_graphs, is_connected=True, verbose=False, n_jobs=1):
        logging.info(f'{self.__class__.__name__}: count={n_graphs}')

        graphs_range = range(n_graphs)
        if verbose:
            graphs_range = tqdm(graphs_range, desc=f'{self.__class__.__name__}')
        generate_method = self.generate_connected_graph if is_connected else self.generate_graph
        graphs = Parallel(n_jobs=n_jobs)(delayed(generate_method)(idx) for idx in graphs_range)
        info = self.generate_info(n_graphs)
        return graphs, info
