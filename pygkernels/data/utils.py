import networkx as nx
import numpy as np


def nx2np(G: nx.Graph):
    nodes_order, partition = zip(*nx.get_node_attributes(G, 'community').items())
    A = nx.adjacency_matrix(G, nodelist=nodes_order).toarray()
    return A, partition


def np2nx(A: np.ndarray, partition: np.ndarray):
    G = nx.from_numpy_matrix(A)
    nx.set_node_attributes(G, partition, 'community')
    return G
