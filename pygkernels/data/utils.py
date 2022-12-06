import networkx as nx
import numpy as np


def nx2np(G: nx.Graph):
    nodes_order, partition = zip(*nx.get_node_attributes(G, "community").items())
    A = nx.adjacency_matrix(G, nodelist=nodes_order).toarray()

    if type(partition[0]) == set:
        # crutch for LFR generator
        clusters = set([frozenset(x) for x in partition])
        partition = [0] * A.shape[0]
        for cluster_num, cluster in enumerate(clusters):
            for item in cluster:
                partition[item] = cluster_num

    return A, list(partition)


def np2nx(A: np.ndarray, partition: np.ndarray):
    G = nx.from_numpy_matrix(A)
    nx.set_node_attributes(G, dict(enumerate(partition)), "community")
    return G
