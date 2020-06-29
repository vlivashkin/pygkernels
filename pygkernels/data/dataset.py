import os
from typing import List

import networkx as nx
import numpy as np


class Datasets:
    """
    Example datasets.
    The class uses https://github.com/vlivashkin/community-graphs, mounted as submodule to ./community-graphs
    """

    def __init__(self, datasets_root=None):
        if datasets_root is None:
            folder_of_this_src_file = os.path.dirname(os.path.abspath(__file__))
            self.datasets_root = f'{folder_of_this_src_file}/community-graphs/gml_connected_subgraphs'
        else:
            self.datasets_root = datasets_root

        self._lazy_datasets = {
            'cora_AI': lambda: self._read_gml('cora_subset/Artificial_Intelligence.gml'),
            'cora_AI_ML': lambda: self._read_gml('cora_subset/Artificial_Intelligence__Machine_Learning.gml'),
            'cora_DS_AT': lambda: self._read_gml('cora_subset/Data_Structures__Algorithms_and_Theory.gml'),
            'cora_DB': lambda: self._read_gml('cora_subset/Databases.gml'),
            'cora_EC': lambda: self._read_gml('cora_subset/Encryption_and_Compression.gml'),
            'cora_HA': lambda: self._read_gml('cora_subset/Hardware_and_Architecture.gml'),
            'cora_HCI': lambda: self._read_gml('cora_subset/Human_Computer_Interaction.gml'),
            'cora_IR': lambda: self._read_gml('cora_subset/Information_Retrieval.gml'),
            'cora_Net': lambda: self._read_gml('cora_subset/Networking.gml'),
            'cora_OS': lambda: self._read_gml('cora_subset/Operating_Systems.gml'),
            'cora_Prog': lambda: self._read_gml('cora_subset/Programming.gml'),
            'dolphins': lambda: self._read_gml('dolphins.gml'),
            'eu-core': lambda: self._read_gml('eu-core.gml'),
            'eurosis': lambda: self._read_gml('eurosis.gml'),
            'football': lambda: self._read_gml('football.gml'),
            'karate': lambda: self._read_gml('karate.gml'),
            'news_2cl1': lambda: self._read_gml('newsgroup/news_2cl1.gml'),
            'news_2cl2': lambda: self._read_gml('newsgroup/news_2cl2.gml'),
            'news_2cl3': lambda: self._read_gml('newsgroup/news_2cl3.gml'),
            'news_3cl1': lambda: self._read_gml('newsgroup/news_3cl1.gml'),
            'news_3cl2': lambda: self._read_gml('newsgroup/news_3cl2.gml'),
            'news_3cl3': lambda: self._read_gml('newsgroup/news_3cl3.gml'),
            'news_5cl1': lambda: self._read_gml('newsgroup/news_5cl1.gml'),
            'news_5cl2': lambda: self._read_gml('newsgroup/news_5cl2.gml'),
            'news_5cl3': lambda: self._read_gml('newsgroup/news_5cl3.gml'),
            'news_2cl1_0.1': lambda: self._read_gml('newsgroup_0.1/news_2cl1_0.1.gml'),
            'news_2cl2_0.1': lambda: self._read_gml('newsgroup_0.1/news_2cl2_0.1.gml'),
            'news_2cl3_0.1': lambda: self._read_gml('newsgroup_0.1/news_2cl3_0.1.gml'),
            'news_3cl1_0.1': lambda: self._read_gml('newsgroup_0.1/news_3cl1_0.1.gml'),
            'news_3cl2_0.1': lambda: self._read_gml('newsgroup_0.1/news_3cl2_0.1.gml'),
            'news_3cl3_0.1': lambda: self._read_gml('newsgroup_0.1/news_3cl3_0.1.gml'),
            'news_5cl1_0.1': lambda: self._read_gml('newsgroup_0.1/news_5cl1_0.1.gml'),
            'news_5cl2_0.1': lambda: self._read_gml('newsgroup_0.1/news_5cl2_0.1.gml'),
            'news_5cl3_0.1': lambda: self._read_gml('newsgroup_0.1/news_5cl3_0.1.gml'),
            'polblogs': lambda: self._read_gml('polblogs.gml'),
            'polbooks': lambda: self._read_gml('polbooks.gml'),
            'sp_school_day_1': lambda: self._read_gml('sp_school/sp_school_day_1.gml'),
            'sp_school_day_2': lambda: self._read_gml('sp_school/sp_school_day_2.gml'),
        }

        self._loaded_datasets = {}

    @staticmethod
    def simplify_partition(partition: List):
        class_mapping = dict([(y, x) for x, y in enumerate(set(partition))])
        return np.array([class_mapping[x] for x in partition], dtype=np.uint8)

    def _read_gml(self, rel_path):
        fpath = f'{self.datasets_root}/{rel_path}'
        G = nx.read_gml(fpath)
        nodes_order, partition = zip(*nx.get_node_attributes(G, 'gt').items())
        A = np.array(nx.adjacency_matrix(G, nodelist=nodes_order).todense())
        partition = self.simplify_partition(partition)

        n, k = len(partition), len(set(partition))
        unbalance = None  # TODO: unbalance
        potential_edges_in = np.sum([x * (x - 1) / 2 for x in [len(partition[partition == ki]) for ki in range(k)]])
        potential_edges_out = n * (n - 1) / 2 - potential_edges_in
        actual_edges_in = np.sum([A[i, j] for i in range(n) for j in range(i + 1, n) if partition[i] == partition[j]])
        actual_edges_out = np.sum([A[i, j] for i in range(n) for j in range(i + 1, n)]) - actual_edges_in
        p_in = actual_edges_in / potential_edges_in
        p_out = actual_edges_out / potential_edges_out

        info = {
            'name': os.path.splitext(os.path.basename(fpath))[0],
            'n': len(partition),
            'k': len(set(partition)),
            'unbalance': unbalance,
            'p_in': p_in,
            'p_out': p_out
        }
        return (A, partition), info

    def __getitem__(self, item):
        if item not in self._loaded_datasets:
            print(f'Item {item} not in cache; reload')
            self._loaded_datasets[item] = self._lazy_datasets[item]()
        return self._loaded_datasets[item]

    def __getattr__(self, name):
        return self[name]
