import json
import logging
import os

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class Datasets:
    """
    Example datasets.
    The class uses https://github.com/vlivashkin/community-graphs, mounted as submodule to /submodules/community-graphs
    """

    def __init__(self, paths, datasets_root=None):
        self.datasets_root = datasets_root
        if self.datasets_root is None:
            self.datasets_root = f"{CURRENT_FOLDER}/../../submodules/community-graphs/gml_connected_subgraphs"

        stat_path = f"{self.datasets_root}/stat.json"
        if os.path.exists(stat_path):
            with open(stat_path, "r") as f:
                self._meta = json.load(f)
        else:
            log.info("No stat.json file")

        self._paths = paths
        self._loaded_datasets = {}

    def _read_gml(self, rel_path):
        fpath = f"{self.datasets_root}/{rel_path}"
        G = nx.read_gml(fpath)
        A = np.array(nx.adjacency_matrix(G).todense())
        return A

    def __getitem__(self, name):
        if name not in self._loaded_datasets:
            print(f"Dataset {name} not in cache; reload")
            self._loaded_datasets[name] = self._read_gml(self._paths[name])
        return self._loaded_datasets[name]

    def __getattr__(self, name):
        return self[name]
