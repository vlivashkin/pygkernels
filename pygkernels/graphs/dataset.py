import os

import networkx as nx
import numpy as np


class Datasets:
    """
    Example datasets.
    Class uses https://github.com/vlivashkin/community-graphs, mounted as submodule to ./graphs
    """

    def __init__(self, datasets_root=None):
        if datasets_root is None:
            folder_of_this_src_file = os.path.dirname(os.path.abspath(__file__))
            self.datasets_root = f'{folder_of_this_src_file}/datasets/gml_graphs'
        else:
            self.datasets_root = datasets_root

        self._lazy_datasets = {
            'cora_AI': lambda: self._load_gml('cora_subset/Artificial_Intelligence.gml'),
            'cora_AI_ML': lambda: self._load_gml('cora_subset/Artificial_Intelligence__Machine_Learning.gml'),
            'cora_DS_AT': lambda: self._load_gml('cora_subset/Data_Structures__Algorithms_and_Theory.gml'),
            'cora_DB': lambda: self._load_gml('cora_subset/Databases.gml'),
            'cora_EC': lambda: self._load_gml('cora_subset/Encryption_and_Compression.gml'),
            'cora_HA': lambda: self._load_gml('cora_subset/Hardware_and_Architecture.gml'),
            'cora_HCI': lambda: self._load_gml('cora_subset/Human_Computer_Interaction.gml'),
            'cora_IR': lambda: self._load_gml('cora_subset/Information_Retrieval.gml'),
            'cora_Net': lambda: self._load_gml('cora_subset/Networking.gml'),
            'cora_OS': lambda: self._load_gml('cora_subset/Operating_Systems.gml'),
            'cora_Prog': lambda: self._load_gml('cora_subset/Programming.gml'),
            'dolphins': lambda: self._load_gml('dolphins.gml'),
            'eu-core': lambda: self._load_gml('eu-core.gml'),
            'eurosis': lambda: self._load_gml('eu-eurosis.gml'),
            'football': lambda: self._load_gml('football.gml'),
            'karate': lambda: self._load_gml('karate.gml'),
            'news_2cl_1': lambda: self._load_gml('newsgroup/news_2cl_1.gml'),
            'news_2cl_2': lambda: self._load_gml('newsgroup/news_2cl_2.gml'),
            'news_2cl_3': lambda: self._load_gml('newsgroup/news_2cl_3.gml'),
            'news_3cl_1': lambda: self._load_gml('newsgroup/news_3cl_1.gml'),
            'news_3cl_2': lambda: self._load_gml('newsgroup/news_3cl_2.gml'),
            'news_3cl_3': lambda: self._load_gml('newsgroup/news_3cl_3.gml'),
            'news_5cl_1': lambda: self._load_gml('newsgroup/news_5cl_1.gml'),
            'news_5cl_2': lambda: self._load_gml('newsgroup/news_5cl_2.gml'),
            'news_5cl_3': lambda: self._load_gml('newsgroup/news_5cl_3.gml'),
            'polblogs': lambda: self._load_gml('polblogs.gml'),
            'polbooks': lambda: self._load_gml('polbooks.gml'),
            'sp_school_day_1': lambda: self._load_gml('sp_school/sp_school_day_1.gml'),
            'sp_school_day_2': lambda: self._load_gml('sp_school/sp_school_day_2.gml'),
        }

        self._loaded_datasets = {}

    def _load_gml(self, fpath):
        G = nx.read_gml(fpath)
        nodes_order, partition = list(nx.get_node_attributes(G, 'gt').items())
        edges = np.array(nx.adjacency_matrix(G, nodelist=nodes_order).todense())
        info = {
            'name': os.path.splitext(os.path.basename(fpath))[0],
            'count': 1,
            'n': len(self.partition),
            'k': len(set(self.partition)),
            'p_in': None,
            'p_out': None
        }
        return [(edges, partition)], [G], info

    @property
    def cora_subsets(self):
        cora_names = ['cora_AI', 'cora_AI_ML', 'cora_DS_AT', 'cora_DB', 'cora_EC', 'cora_HA',
                      'cora_HCI', 'cora_IR', 'cora_Net', 'cora_OS', 'cora_Prog']
        return [self[x] for x in cora_names]

    @property
    def newsgroup_subsets(self):
        newsgroup_names = ['news_2cl_1', 'news_2cl_2', 'news_2cl_3',
                           'news_3cl_1', 'news_3cl_2', 'news_3cl_3',
                           'news_5cl_1', 'news_5cl_2', 'news_5cl_3']
        return [self[x] for x in newsgroup_names]

    @property
    def all(self):
        return [self[x] for x in self._lazy_datasets.keys()]

    def __getitem__(self, item):
        if item not in self._loaded_datasets:
            self._loaded_datasets[item] = self._lazy_datasets[item]()
        return self._loaded_datasets[item]

    def __getattr__(self, name):
        return self[name]
