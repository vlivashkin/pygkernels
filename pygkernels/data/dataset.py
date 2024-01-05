import json
import os
from typing import List

import networkx as nx
import numpy as np

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

class Datasets:
    """
    Example datasets.
    The class uses https://github.com/vlivashkin/community-graphs, mounted as submodule to /submodules/community-graphs
    """

    def __init__(self, datasets_root=None):
        if datasets_root is None:
            self.datasets_root = f"{CURRENT_FOLDER}/../../submodules/community-graphs/gml_connected_subgraphs"
        else:
            self.datasets_root = datasets_root

        with open(f"{self.datasets_root}/stat.json", "r") as f:
            self._meta = json.load(f)

        self._paths = {
            "cora_AI": "cora_subset/Artificial_Intelligence.gml",
            "cora_AI_ML": "cora_subset/Artificial_Intelligence__Machine_Learning.gml",
            "cora_DS_AT": "cora_subset/Data_Structures__Algorithms_and_Theory.gml",
            "cora_DB": "cora_subset/Databases.gml",
            "cora_EC": "cora_subset/Encryption_and_Compression.gml",
            "cora_HA": "cora_subset/Hardware_and_Architecture.gml",
            "cora_HCI": "cora_subset/Human_Computer_Interaction.gml",
            "cora_IR": "cora_subset/Information_Retrieval.gml",
            "cora_Net": "cora_subset/Networking.gml",
            "cora_OS": "cora_subset/Operating_Systems.gml",
            "cora_Prog": "cora_subset/Programming.gml",
            "dolphins": "dolphins.gml",
            "eu-core": "eu-core.gml",
            "eurosis": "eurosis.gml",
            "football": "football.gml",
            "karate": "karate.gml",
            "news_2cl1": "newsgroup/news_2cl1.gml",
            "news_2cl2": "newsgroup/news_2cl2.gml",
            "news_2cl3": "newsgroup/news_2cl3.gml",
            "news_3cl1": "newsgroup/news_3cl1.gml",
            "news_3cl2": "newsgroup/news_3cl2.gml",
            "news_3cl3": "newsgroup/news_3cl3.gml",
            "news_5cl1": "newsgroup/news_5cl1.gml",
            "news_5cl2": "newsgroup/news_5cl2.gml",
            "news_5cl3": "newsgroup/news_5cl3.gml",
            "news_2cl1_0.1": "newsgroup_0.1/news_2cl1_0.1.gml",
            "news_2cl2_0.1": "newsgroup_0.1/news_2cl2_0.1.gml",
            "news_2cl3_0.1": "newsgroup_0.1/news_2cl3_0.1.gml",
            "news_3cl1_0.1": "newsgroup_0.1/news_3cl1_0.1.gml",
            "news_3cl2_0.1": "newsgroup_0.1/news_3cl2_0.1.gml",
            "news_3cl3_0.1": "newsgroup_0.1/news_3cl3_0.1.gml",
            "news_5cl1_0.1": "newsgroup_0.1/news_5cl1_0.1.gml",
            "news_5cl2_0.1": "newsgroup_0.1/news_5cl2_0.1.gml",
            "news_5cl3_0.1": "newsgroup_0.1/news_5cl3_0.1.gml",
            "polblogs": "polblogs.gml",
            "polbooks": "polbooks.gml",
            "sp_school_day_1": "sp_school/sp_school_day_1.gml",
            "sp_school_day_2": "sp_school/sp_school_day_2.gml",
        }

        self._loaded_datasets = {}

    @staticmethod
    def simplify_partition(partition: List):
        class_mapping = dict([(y, x) for x, y in enumerate(set(partition))])
        return np.array([class_mapping[x] for x in partition], dtype=np.uint8)

    def _read_gml(self, name, rel_path):
        fpath = f"{self.datasets_root}/{rel_path}"
        G = nx.read_gml(fpath)
        nodes_order, partition = zip(*nx.get_node_attributes(G, "gt").items())
        A = np.array(nx.adjacency_matrix(G, nodelist=nodes_order).todense())
        partition = self.simplify_partition(partition)

        meta = self._meta[os.path.splitext(os.path.basename(rel_path))[0]]
        n, k = meta["n_nodes"], meta["n_classes"]
        S, P = np.array(meta["cluster_sizes"]), np.array(meta["edge_probs"])
        p_in = np.sum([(S[i] * S[i] / 2) * P[i, i] for i in range(k)]) / np.sum([(S[i] * S[i] / 2) for i in range(k)])
        p_out = np.sum([[S[i] * S[j] * P[i, j] for i in range(k) for j in range(i + 1, k)]]) / np.sum(
            [[S[i] * S[j] for i in range(k) for j in range(i + 1, k)]]
        )

        info = {
            "name": os.path.splitext(os.path.basename(fpath))[0],
            "n": n,
            "k": k,
            "p_in": p_in,
            "p_out": p_out,
            "unbalance": None,
            "S": S,
            "P": P,
        }
        return (A, partition), info

    def __getitem__(self, name):
        if name not in self._loaded_datasets:
            print(f"Dataset {name} not in cache; reload")
            self._loaded_datasets[name] = self._read_gml(name, self._paths[name])
        return self._loaded_datasets[name]

    def __getattr__(self, name):
        return self[name]
