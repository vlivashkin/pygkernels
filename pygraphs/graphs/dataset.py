import os
from os.path import join as pj

import numpy as np
from collections import namedtuple
Car = namedtuple('Car' , 'color mileage')

ROOT_PATH = pj(os.path.dirname(os.path.abspath(__file__)), 'datasets')


class ImportedGraphBuilder:
    def __init__(self):
        self.name = None
        self.nodes_mapping = []
        self.nodes_class = []
        self.edges = None

    def set_name(self, name):
        self.name = name
        return self

    def import_nodes(self, filename, sep='\t', startline=0, name_col_idx=0, class_col_idx=1):
        with open(filename) as f:
            for line_idx, line in enumerate(f):
                if len(line) > 0 and line_idx >= startline:
                    line_content = (line[:-2] if line.endswith(';\n') else line[:-1]).split(sep)
                    self.nodes_mapping.append(str(line_idx) if name_col_idx == 'idx' else line_content[name_col_idx])
                    self.nodes_class.append(line_content[class_col_idx])
        self.edges = np.zeros((len(self.nodes_class), len(self.nodes_class)))
        return self

    def import_nodes_and_edges(self, filename):
        stage = -1  # 0 if vertices, 1 if edges
        with open(filename) as f:
            for line in f:
                if len(line) > 0:
                    if line.startswith('*'):
                        stage += 1
                        if stage == 0:
                            self.nodes_class = []
                        elif stage == 1:
                            self.edges = np.zeros((len(self.nodes_class), len(self.nodes_class)))
                        continue
                    elif stage == 0:
                        self.nodes_class.append(int(line[:-1].split(' ')[1][1:-1]))
                    elif stage == 1:
                        v1, v2 = line[:-1].strip().split(' ')
                        self.edges[int(v1) - 1, int(v2) - 1] = 1
                        self.edges[int(v2) - 1, int(v1) - 1] = 1
        return self

    def import_edges(self, filename, sep='\t', startline=0, node1_col_idx=0, node2_col_idx=1):
        with open(filename) as f:
            for line_idx, line in enumerate(f):
                if len(line) > 0 and line_idx >= startline:
                    line_content = (line[:-2] if line.endswith(';\n') else line[:-1]).split(sep)
                    v1 = self.nodes_mapping.index(line_content[node1_col_idx])
                    v2 = self.nodes_mapping.index(line_content[node2_col_idx])
                    self.edges[int(v1), int(v2)] = 1
                    self.edges[int(v2), int(v1)] = 1
        return self

    def import_adjacency_matrix(self, filename):
        with open(filename) as f:
            for i, line in enumerate(f):
                if len(line) > 0:
                    for j, value in enumerate(line.split(',')):
                        self.edges[i, j] = value
        return self

    def build(self):
        if self.name is None or self.nodes_class is None or self.edges is None:
            raise NotImplementedError()
        info = {
            'name': self.name,
            'count': 1,
            'n': len(self.nodes_class),
            'k': len(list(set(self.nodes_class))),
            'p_in': None,
            'p_out': None
        }
        return [(self.edges, self.nodes_class)], info


class Datasets:
    _lazy_datasets = {
        'citeseer': lambda: Datasets.load_webkb('citeseer', 'citeseer/citeseer.nodes', 'citeseer/citeseer.edges'),
        'cora': lambda: Datasets.load_webkb('cora', 'cora/cora.nodes', 'cora/cora.edges'),
        'football': lambda: Datasets.load_polbooks_or_football('football', 'football_nodes.csv',
                                                               'football_edges.csv'),
        'news_2cl_1': lambda: Datasets.load_newsgroup('news_2cl_1', 'newsgroup/news_2cl_1_classeo.csv',
                                                      'newsgroup/news_2cl_1_Docr.csv'),
        'news_2cl_2': lambda: Datasets.load_newsgroup('news_2cl_2', 'newsgroup/news_2cl_2_classeo.csv',
                                                      'newsgroup/news_2cl_2_Docr.csv'),
        'news_2cl_3': lambda: Datasets.load_newsgroup('news_2cl_3', 'newsgroup/news_2cl_3_classeo.csv',
                                                      'newsgroup/news_2cl_3_Docr.csv'),
        'news_3cl_1': lambda: Datasets.load_newsgroup('news_3cl_1', 'newsgroup/news_3cl_1_classeo.csv',
                                                      'newsgroup/news_3cl_1_Docr.csv'),
        'news_3cl_2': lambda: Datasets.load_newsgroup('news_3cl_2', 'newsgroup/news_3cl_2_classeo.csv',
                                                      'newsgroup/news_3cl_2_Docr.csv'),
        'news_3cl_3': lambda: Datasets.load_newsgroup('news_3cl_3', 'newsgroup/news_3cl_3_classeo.csv',
                                                      'newsgroup/news_3cl_3_Docr.csv'),
        'news_5cl_1': lambda: Datasets.load_newsgroup('news_5cl_1', 'newsgroup/news_5cl_1_classeo.csv',
                                                      'newsgroup/news_5cl_1_Docr.csv'),
        'news_5cl_2': lambda: Datasets.load_newsgroup('news_5cl_2', 'newsgroup/news_5cl_2_classeo.csv',
                                                      'newsgroup/news_5cl_2_Docr.csv'),
        'news_5cl_3': lambda: Datasets.load_newsgroup('news_5cl_3', 'newsgroup/news_5cl_3_classeo.csv',
                                                      'newsgroup/news_5cl_3_Docr.csv'),
        'polbooks': lambda: Datasets.load_polbooks_or_football('polbooks', 'polbooks_nodes.csv',
                                                               'polbooks_edges.csv'),
        'polblogs': lambda: Datasets.load_polblogs_or_zachary('polblogs', 'polblogs.net'),
        'webkb_cornel': lambda: Datasets.load_webkb('webkb_cornell', 'webKB/cornell/webkb-cornell.nodes',
                                                    'webKB/cornell/webkb-cornell.edges'),
        'webkb_texas': lambda: Datasets.load_webkb('webkb_texas', 'webKB/texas/webkb-texas.nodes',
                                                   'webKB/texas/webkb-texas.edges'),
        'webkb_washington': lambda: Datasets.load_webkb('webkb_washington', 'webKB/washington/webkb-washington.nodes',
                                                        'webKB/washington/webkb-washington.edges'),
        'webkb_wisconsin': lambda: Datasets.load_webkb('webkb_wisconsin', 'webKB/wisconsin/webkb-wisconsin.nodes',
                                                       'webKB/wisconsin/webkb-wisconsin.edges'),
        'zachary': lambda: Datasets.load_polblogs_or_zachary('zachary', 'zachary.net')
    }

    _loaded_datasets = {}

    @property
    def newsgroup(self):
        newsgroup_names = ['news_2cl_1', 'news_2cl_2', 'news_2cl_3',
                           'news_3cl_1', 'news_3cl_2', 'news_3cl_3',
                           'news_5cl_1', 'news_5cl_2', 'news_5cl_3']
        return [self[x] for x in newsgroup_names]

    @property
    def webkb(self):
        webkb_names = ['webkb_cornel', 'webkb_texas', 'webkb_washington', 'webkb_wisconsin']
        return [self[x] for x in webkb_names]

    @property
    def all(self):
        return [self[x] for x in self._lazy_datasets.keys()]

    @staticmethod
    def load_polbooks_or_football(name, nodes_path, edges_path):
        return ImportedGraphBuilder() \
            .set_name(name) \
            .import_nodes(pj(ROOT_PATH, nodes_path), name_col_idx='idx', class_col_idx=2) \
            .import_edges(pj(ROOT_PATH, edges_path)) \
            .build()

    @staticmethod
    def load_polblogs_or_zachary(name, graph_path):
        return ImportedGraphBuilder() \
            .set_name(name) \
            .import_nodes_and_edges(pj(ROOT_PATH, graph_path)) \
            .build()

    @staticmethod
    def load_newsgroup(name, nodes_path, edges_path):
        return ImportedGraphBuilder() \
            .set_name(name) \
            .import_nodes(pj(ROOT_PATH, nodes_path), name_col_idx='idx', class_col_idx=0) \
            .import_adjacency_matrix(pj(ROOT_PATH, edges_path)) \
            .build()

    @staticmethod
    def load_webkb(name, nodes_path, edges_path):
        return ImportedGraphBuilder() \
            .set_name(name) \
            .import_nodes(pj(ROOT_PATH, nodes_path), startline=2, name_col_idx=0, class_col_idx=-1) \
            .import_edges(pj(ROOT_PATH, edges_path), startline=3, node1_col_idx=1, node2_col_idx=3) \
            .build()

    def __getitem__(self, item):
        if item not in self._loaded_datasets:
            self._loaded_datasets[item] = self._lazy_datasets[item]()
        return self._loaded_datasets[item]

    def __getattr__(self, name):
        return self[name]
