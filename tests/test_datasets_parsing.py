import unittest
from collections import Counter

import numpy as np

from pygraphs.graphs import Datasets


class TestDatasetsParsing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = Datasets()

    def test_load_every_dataset(self):
        for name in self.datasets._lazy_datasets.keys():
            _, info = self.datasets[name]
            print(f'{info["name"]} OK! {info}')

    def test_load_all_datasets(self):
        _ = self.datasets.all

    def test_altsoph_football(self):
        old_football_graph, _ = self.datasets._load_polbooks_or_football(
            'football', '_old/football_nodes.csv', '_old/football_edges.csv')
        new_football_graph, _ = self.datasets.football

        (old_edges, old_nodes), (new_edges, new_nodes) = old_football_graph[0], new_football_graph[0]
        print(f'Nodes count:\n\told {len(old_nodes)},\n\tnew {len(new_nodes)}')
        print(f'Edges shape:\n\told {old_edges.shape},\n\tnew {new_edges.shape}')
        assert len(old_nodes) == len(new_nodes)
        assert old_edges.shape == new_edges.shape

        old_classes = (np.array(old_nodes, dtype=np.uint8) + 1)
        new_classes = np.array(new_nodes, dtype=np.uint8)
        old_distr, new_distr = Counter(old_classes), Counter(new_classes)
        print(f'Equal classes:\n\told {old_distr},\n\tnew {new_distr}')
        assert old_distr == new_distr

    def test_altsoph_karate(self):
        old_karate_graph, _ = self.datasets._load_polblogs_or_zachary('karate', '_old/zachary.net')
        new_karate_graph, _ = self.datasets.karate

        (old_edges, old_nodes), (new_edges, new_nodes) = old_karate_graph[0], new_karate_graph[0]
        print(f'Nodes count: old {len(old_nodes)}, new {len(new_nodes)}')
        print(f'Edges shape: old {old_edges.shape}, new {new_edges.shape}')
        assert len(old_nodes) == len(new_nodes)
        assert old_edges.shape == new_edges.shape

        old_classes = (np.array(old_nodes, dtype=np.uint8))
        new_classes = np.array(new_nodes, dtype=np.uint8)
        eq_classes = np.sum([1 if o == n else 0 for o, n in zip(old_classes, new_classes)])
        neq_classes = np.sum([1 if o != n else 0 for o, n in zip(old_classes, new_classes)])
        print(f'Equal classes: {eq_classes}, non-equal: {neq_classes}')
        assert np.allclose(np.array(old_nodes, dtype=np.uint8), np.array(new_nodes, dtype=np.uint8))
        assert np.allclose(old_edges, new_edges)

    def test_altsoph_polblogs(self):
        old_polblogs_graph, _ = self.datasets._load_polblogs_or_zachary('polblogs', '_old/polblogs.net')
        new_polblogs_graph, _ = self.datasets.polblogs

        (old_edges, old_nodes), (new_edges, new_nodes) = old_polblogs_graph[0], new_polblogs_graph[0]
        print(f'Nodes count: old {len(old_nodes)}, new {len(new_nodes)}')
        print(f'Edges shape: old {old_edges.shape}, new {new_edges.shape}')
        assert len(old_nodes) == len(new_nodes)
        assert old_edges.shape == new_edges.shape

    def test_altsoph_polbooks(self):
        old_polbooks_graph, _ = self.datasets._load_polbooks_or_football(
            'polbooks', '_old/polbooks_nodes.csv', '_old/polbooks_edges.csv')
        new_polbooks_graph, _ = self.datasets.polbooks

        def map_labels(nodes):
            result = []
            for node in nodes:
                if node == 'n':
                    result.append(0)
                elif node == 'l':
                    result.append(1)
                elif node == 'c':
                    result.append(2)
                else:
                    result.append(node)
            return result

        (old_edges, old_nodes), (new_edges, new_nodes) = old_polbooks_graph[0], new_polbooks_graph[0]
        print(f'Nodes count: old {len(old_nodes)}, new {len(new_nodes)}')
        print(f'Edges shape: old {old_edges.shape}, new {new_edges.shape}')
        assert len(old_nodes) == len(new_nodes)
        assert old_edges.shape == new_edges.shape

        old_classes = np.array(map_labels(old_nodes), dtype=np.uint8)
        new_classes = np.array(new_nodes, dtype=np.uint8)
        old_distr, new_distr = Counter(old_classes), Counter(new_classes)
        print(f'Equal classes:\n\told {old_distr},\n\tnew {new_distr}')
        assert old_distr == new_distr
