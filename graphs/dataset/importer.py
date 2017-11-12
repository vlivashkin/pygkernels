import numpy as np


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
            'count': 1,
            'n': len(self.nodes),
            'k': len(list(set(self.nodes))),
            'p_in': None,
            'p_out': None
        }
        return [(self.edges, self.nodes)], info
