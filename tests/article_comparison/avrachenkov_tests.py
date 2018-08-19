import json
import logging
import os
import unittest
from collections import defaultdict
from os.path import join as pj

import networkx as nx
import networkx.readwrite.json_graph as jg
from tqdm import tqdm

from pygraphs import util
from pygraphs.cluster import SpectralClustering
from pygraphs.graphs import sample
from pygraphs.graphs.generators import RubanovModel
from pygraphs.measure import *
from pygraphs.measure import scaler
from pygraphs.measure.shortcuts import *
from pygraphs.scorer import FC


# Konstantin Avrachenkov, Pavel Chebotarev, Dmytro Rubanov: Kernels on Graphs as Proximity Measures
# https://hal.inria.fr/hal-01647915/document

class NewMeasuresEqualutyTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.graph = sample.diploma_matrix

    def test_katz(self):
        walk = Walk_H(self.graph)
        katz = Katz_R(self.graph)
        for param in scaler.Rho(self.graph).scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(walk.get_K(param).ravel(), katz.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_estrada(self):
        comm = logComm_H(self.graph)
        estrada = Estrada_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(comm.get_K(param).ravel(), estrada.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_heat(self):
        heat = logHeat_H(self.graph)
        heat_new = Heat_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(heat.get_K(param).ravel(), heat_new.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))

    def test_regularized_laplacian(self):
        forest = logFor_H(self.graph)
        reg_laplacian = RegularizedLaplacian_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(forest.get_K(param).ravel(), reg_laplacian.get_K(param).ravel(), atol=0.0001),
                            'error in param={:0.3f}'.format(param))


class Competition(unittest.TestCase):
    def __init__(self, atol, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()

        self.atol = atol
        self.n_params = 200

    def _calc_score(self, measure, params=None):
        results = defaultdict(lambda: [])
        count, passes = 0, 0
        for graph_idx, (A, y_true) in tqdm(enumerate(self.graphs), total=len(self.graphs), desc=measure.name):
            mg = measure(A)
            if params is None:
                params = mg.scaler.scale(np.linspace(0, 1, self.n_params))
            for param in params:
                try:
                    count += 1
                    K = mg.get_K(param)
                    y_pred = SpectralClustering(2).fit_predict(K)
                    results[graph_idx].append(FC(y_true, y_pred))
                    passes += 1
                except:
                    pass

        mins = [min(x) for x in results.values()]
        quality = np.mean(mins)

        # logging results for report
        logging.info('{}; Passes: {}/{}; Min error: {:0.4f}'.format(measure.name, passes, count, 1 - quality))

        return quality

    def _compare(self, measure, params, error_true):
        error_test = self._calc_score(measure, params)
        diff = np.abs(error_test - error_true)

        # logging results for report
        logging.info(
            '{}; Min error: {:0.4f}, true={:0.4f}, diff={:0.4f}'.format(measure.name, error_test, error_true, diff))

        self.assertTrue(np.isclose(error_test, error_true, atol=self.atol),
                        "Test {:0.4f} != True {:0.4f}, diff={:0.4f}".format(error_test, error_true, diff))


@unittest.skip
class BalancedModel(Competition):
    def __init__(self, *args, **kwargs):
        super().__init__(0.002, *args, **kwargs)  # error bars in paper: 0.002
        util.configure_logging()

    @classmethod
    def setUpClass(cls):
        sizes = np.array([100, 100])
        probs = np.array([[0.1, 0.02],
                          [0.02, 0.1]])

        folder = os.path.dirname(os.path.abspath(__file__))

        with open(pj(folder, "Grahs_g100_100x100.json"), "r") as fp:
            DATA = json.load(fp)

        R_COMMS = cls.real_comms(sizes)

        GS = [jg.node_link_graph(d) for d in DATA["GS"]]
        GS = [(np.array(np.array(nx.adjacency_matrix(g).todense())), R_COMMS) for g in GS]

        cls.graphs = GS

    @staticmethod
    def real_comms(sizes):
        return np.array(sum(([i] * size for i, size in enumerate(sizes)), []))

    def test_katz(self):
        self._compare(Katz_R, None, 0.0072)

    def test_communicability(self):
        self._compare(Estrada_R, np.linspace(0, 0.3, 101)[1:-1], 0.0084)

    def test_heat(self):
        self._compare(Heat_R, np.linspace(0, 1.5, 101)[1:-1], 0.0064)

    def test_normalizedHeat(self):
        self._compare(NormalizedHeat_R, np.linspace(0, 20, 101)[1:-1], 0.0066)

    def test_regularizedLaplacian(self):
        self._compare(RegularizedLaplacian_R, np.linspace(0, 20, 101)[1:-1], 0.0072)

    def test_personalizedPageRank(self):
        self._compare(PPageRank_R, np.linspace(0, 1, 101)[1:-1], 0.0073)

    def test_modifiedPageRank(self):
        self._compare(ModifiedPPageRank_R, np.linspace(0, 1, 101)[1:-1], 0.0072)

    def test_heatPageRank(self):
        self._compare(HeatPPageRank_R, np.linspace(0, 20, 101)[1:-1], 0.0074)


@unittest.skip
class UnbalancedModel(Competition):
    def __init__(self, *args, **kwargs):
        super().__init__(0.006, *args, **kwargs)  # error bars in paper: 0.006

    @classmethod
    def setUpClass(cls):
        sizes = np.array([50, 150])
        probs = np.array([[0.1, 0.02],
                          [0.02, 0.1]])
        cls.graphs, _ = RubanovModel(sizes, probs).generate_graphs(1000)  # 1000 graphs in paper

    def test_katz(self):
        self._compare(Katz_R, None, 0.012)

    def test_communicability(self):
        self._compare(Estrada_R, np.linspace(0, 0.3, 101)[1:-1], 0.011)

    def test_heat(self):
        self._compare(Heat_R, np.linspace(0, 1.5, 101)[1:-1], 0.0104)

    def test_normalizedHeat(self):
        self._compare(NormalizedHeat_R, np.linspace(0, 20, 101)[1:-1], 0.009)

    def test_regularizedLaplacian(self):
        self._compare(RegularizedLaplacian_R, np.linspace(0, 20, 101)[1:-1], 0.0026)

    def test_personalizedPageRank(self):
        self._compare(PPageRank_R, np.linspace(0, 1, 101)[1:-1], 0.0021)

    def test_modifiedPageRank(self):
        self._compare(ModifiedPPageRank_R, np.linspace(0, 1, 101)[1:-1], 0.0022)

    def test_heatPageRank(self):
        self._compare(HeatPPageRank_R, np.linspace(0, 20, 101)[1:-1], 0.0021)
