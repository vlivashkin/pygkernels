import os

import numpy as np

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


class ImportedGraphBuilder:
    def __init__(self):
        self.name = None
        self.nodes = None
        self.edges = None

    def set_name(self, name):
        self.name = name
        return self

    def import_nodes_id_name_class(self, filename):
        with open(filename) as f:
            labels = [line[:-2].split('\t')[2] for line in f if len(line) > 0]
        classes = list(set(labels))
        self.nodes = [classes.index(label) for label in labels]
        self.edges = np.zeros((len(self.nodes), len(self.nodes)))
        return self

    def import_nodes_class(self, filename):
        with open(filename) as f:
            self.nodes = [int(line[:-1]) for line in f if len(line) > 0]
        self.edges = np.zeros((len(self.nodes), len(self.nodes)))
        return self

    def import_nodes_and_edges(self, filename):
        stage = -1  # 0 if vertices, 1 if edges
        with open(filename) as f:
            for line in f:
                if len(line) > 0:
                    if line.startswith('*'):
                        stage += 1
                        if stage == 0:
                            self.nodes = []
                        elif stage == 1:
                            self.edges = np.zeros((len(self.nodes), len(self.nodes)))
                        continue
                    elif stage == 0:
                        self.nodes.append(int(line[:-1].split(' ')[1][1:-1]))
                    elif stage == 1:
                        v1, v2 = line[:-1].strip().split(' ')
                        self.edges[int(v1) - 1, int(v2) - 1] = 1
                        self.edges[int(v2) - 1, int(v1) - 1] = 1
        return self

    def import_edges(self, filename):
        with open(filename) as f:
            for line in f:
                if len(line) > 0:
                    v1, v2 = line[:-2].split('\t')
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
        if self.name is None or self.nodes is None or self.edges is None:
            raise NotImplementedError()
        info = {
            'name': self.name,
            'count': 1,
            'n': len(self.nodes),
            'k': len(list(set(self.nodes))),
            'p_in': None,
            'p_out': None
        }
        return [(self.edges, self.nodes)], info


def load_polbooks_or_football(name, nodes, edges):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_id_name_class(os.path.join(ROOT_PATH, nodes)) \
        .import_edges(os.path.join(ROOT_PATH, edges)) \
        .build()


def load_polblogs_or_zachary(name, nodes):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_and_edges(os.path.join(ROOT_PATH, nodes)) \
        .build()


def load_newsgroup_graph(name, nodes, edges):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_class(os.path.join(ROOT_PATH, nodes)) \
        .import_adjacency_matrix(os.path.join(ROOT_PATH, edges)) \
        .build()


football = load_polbooks_or_football('football', 'football_nodes.csv', 'football_edges.csv')
polbooks = load_polbooks_or_football('polbooks', 'polbooks_nodes.csv', 'polbooks_edges.csv')
polblogs = load_polblogs_or_zachary('polblogs', 'polblogs.net')
zachary = load_polblogs_or_zachary('zachary', 'zachary.net')
news_2cl_1 = load_newsgroup_graph('news_2cl_1', 'newsgroup/news_2cl_1_classeo.csv', 'newsgroup/news_2cl_1_Docr.csv')
news_2cl_2 = load_newsgroup_graph('news_2cl_2', 'newsgroup/news_2cl_2_classeo.csv', 'newsgroup/news_2cl_2_Docr.csv')
news_2cl_3 = load_newsgroup_graph('news_2cl_3', 'newsgroup/news_2cl_3_classeo.csv', 'newsgroup/news_2cl_3_Docr.csv')
news_3cl_1 = load_newsgroup_graph('news_3cl_1', 'newsgroup/news_3cl_1_classeo.csv', 'newsgroup/news_3cl_1_Docr.csv')
news_3cl_2 = load_newsgroup_graph('news_3cl_2', 'newsgroup/news_3cl_2_classeo.csv', 'newsgroup/news_3cl_2_Docr.csv')
news_3cl_3 = load_newsgroup_graph('news_3cl_3', 'newsgroup/news_3cl_3_classeo.csv', 'newsgroup/news_3cl_3_Docr.csv')
news_5cl_1 = load_newsgroup_graph('news_5cl_1', 'newsgroup/news_5cl_1_classeo.csv', 'newsgroup/news_5cl_1_Docr.csv')
news_5cl_2 = load_newsgroup_graph('news_5cl_2', 'newsgroup/news_5cl_2_classeo.csv', 'newsgroup/news_5cl_2_Docr.csv')
news_5cl_3 = load_newsgroup_graph('news_5cl_3', 'newsgroup/news_5cl_3_classeo.csv', 'newsgroup/news_5cl_3_Docr.csv')
news = [
    news_2cl_1, news_2cl_2, news_2cl_3,
    news_3cl_1, news_3cl_2, news_3cl_3,
    # news_5cl_1, news_5cl_2, news_5cl_3
]
