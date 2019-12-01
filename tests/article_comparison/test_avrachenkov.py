"""
Konstantin Avrachenkov, Pavel Chebotarev, Dmytro Rubanov: Kernels on Graphs as Proximity Measures
https://hal.inria.fr/hal-01647915/document
"""

import json
import logging
import os
import unittest
from abc import ABC
from collections import defaultdict
from os.path import join as pj

import networkx as nx
import networkx.readwrite.json_graph as jg
from joblib import Parallel, delayed
from tqdm import tqdm

from pygraphs import util
from pygraphs.cluster import SpectralClustering_rubanov
from pygraphs.graphs import Samples, RubanovModel
from pygraphs.measure import *
from pygraphs.measure import scaler
from pygraphs.measure.shortcuts import *
from pygraphs.scorer import FC


class TestNewMeasuresEqualuty(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.graph = Samples.diploma_matrix

    def test_katz(self):
        walk = Walk_H(self.graph)
        katz = Katz_R(self.graph)
        for param in scaler.Rho(self.graph).scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(walk.get_K(param).ravel(), katz.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_estrada(self):
        comm = logComm_H(self.graph)
        estrada = Estrada_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(comm.get_K(param).ravel(), estrada.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_heat(self):
        heat = logHeat_H(self.graph)
        heat_rubanov = Heat_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(heat.get_K(param).ravel(), heat_rubanov.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_regularized_laplacian(self):
        forest = logFor_H(self.graph)
        reg_laplacian = RegularizedLaplacian_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.9, 50)):
            self.assertTrue(np.allclose(forest.get_K(param).ravel(), reg_laplacian.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_logPPR(self):
        logppr = logPPR_H(self.graph)
        ppr_rubanov = logPPR_R(self.graph)
        for param in scaler.Linear().scale_list(np.linspace(0.0, 1.0, 50)):
            self.assertTrue(np.allclose(logppr.get_K(param).ravel(), ppr_rubanov.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_logModifPPR(self):
        logppr = logModifPPR_H(self.graph)
        ppr_rubanov = logModifPPR_R(self.graph)
        for param in scaler.Linear().scale_list(np.linspace(0.0, 0.9, 50)):
            self.assertTrue(np.allclose(logppr.get_K(param).ravel(), ppr_rubanov.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')

    def test_logHeatPPR(self):
        logppr = logHeatPPR_H(self.graph)
        ppr_rubanov = logHeatPPR_R(self.graph)
        for param in scaler.Fraction().scale_list(np.linspace(0.1, 0.7, 50)):
            self.assertTrue(np.allclose(logppr.get_K(param).ravel(), ppr_rubanov.get_K(param).ravel(), atol=0.0001),
                            f'error in param={param:0.3f}')


class TestCompetition(unittest.TestCase, ABC):
    def __init__(self, atol, *args, **kwargs):
        super().__init__(*args, **kwargs)
        util.configure_logging()
        self.atol = atol

    def _calc_score(self, measure, params):
        results = dict()
        for param in tqdm(params, total=len(params), desc=measure.name):
            # param_results = []
            # for graph_idx, (A, y_true) in enumerate(self.graphs):
            #     mg = measure(A)
            #     try:
            #         K = mg.get_K(param)
            #         y_pred = SpectralClustering_rubanov(2).fit_predict(K)
            #         param_results.append(FC(y_true, y_pred))
            #     except:
            #         print(f'Exception at param {param}, graph #{graph_idx}')

            def whole_kmeans_run(A, y_true):
                mg = measure(A)
                try:
                    K = mg.get_K(param)
                    y_pred = SpectralClustering_rubanov(2).fit_predict(K)
                    return FC(y_true, y_pred)
                except:
                    return np.nan

            param_results = Parallel(n_jobs=-1)(delayed(whole_kmeans_run)(A, y_true) for (A, y_true) in self.graphs)
            results[param] = np.nanmean(param_results)

        results = dict([(a, b) for (a, b) in results.items() if ~np.isnan(b)])
        results_idx = np.argmin(list(results.values()))
        best_param, best_error = list(results.keys())[results_idx], list(results.values())[results_idx],

        # logging results for report
        logging.info(f'{measure.name}; Best param: {best_param:0.4f}; Min error: {best_error:0.4f}')

        return best_error

    def _compare(self, measure, params, error_true):
        error_test = self._calc_score(measure, params)
        diff = error_test - error_true

        # logging results for report
        logging.info(f'{measure.name}; Min error: {error_test:0.4f}, true={error_true:0.4f}, diff={diff:0.4f}')
        self.assertTrue(np.isclose(error_test, error_true, atol=self.atol),
                        f'Test {error_test:0.4f} != True {error_true:0.4f}, diff={diff:0.4f}')


# @unittest.skip
class BalancedModel(TestCompetition):
    """Fig. 1"""
    def __init__(self, *args, **kwargs):
        super().__init__(0.002, *args, **kwargs)  # error bars in paper: 0.002
        util.configure_logging()

    @classmethod
    def setUpClass(cls):
        sizes = np.array([100, 100])
        folder = os.path.dirname(os.path.abspath(__file__))
        with open(pj(folder, "sample_graphs/Graphs_g100_100x100.json"), "r") as fp:
            DATA = json.load(fp)
        R_COMMS = cls.real_comms(sizes)
        GS = [jg.node_link_graph(d) for d in DATA["GS"]]
        GS = [(np.array(np.array(nx.adjacency_matrix(g).todense())), R_COMMS) for g in GS]
        cls.graphs = GS

    @staticmethod
    def real_comms(sizes):
        return np.array(sum(([i] * size for i, size in enumerate(sizes)), []))

    def test_Walk(self):
        self._compare(Katz_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.0072)

    def test_logComm(self):
        self._compare(Estrada_R, params=np.linspace(0, 0.3, 101)[1:-1], error_true=0.0084)

    def test_logHeat(self):
        self._compare(Heat_R, params=np.linspace(0, 1.5, 101)[1:-1], error_true=0.0064)

    def test_logNHeat(self):
        self._compare(NormalizedHeat_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.0066)

    def test_logFor(self):
        self._compare(RegularizedLaplacian_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.0072)

    def test_logPPR(self):
        self._compare(logPPR_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.0073)

    def test_logModifPPR(self):
        self._compare(logModifPPR_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.0072)

    def test_logHeatPPR(self):
        self._compare(logHeatPPR_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.0074)


@unittest.skip
class TestUnbalancedModel(TestCompetition):
    """Fig. 2"""
    def __init__(self, *args, **kwargs):
        super().__init__(0.006, *args, **kwargs)  # error bars in paper: 0.006

    @classmethod
    def setUpClass(cls):
        sizes = np.array([50, 150])
        probs = np.array([[0.1, 0.02],
                          [0.02, 0.1]])
        GS, _ = RubanovModel(sizes, probs).generate_graphs(1000)  # 1000 graphs in paper
        cls.graphs = GS

    def test_Walk(self):
        self._compare(Katz_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.012)

    def test_logComm(self):
        self._compare(Estrada_R, params=np.linspace(0, 0.3, 101)[1:-1], error_true=0.011)

    def test_logHeat(self):
        self._compare(Heat_R, params=np.linspace(0, 1.5, 101)[1:-1], error_true=0.0104)

    def test_logNHeat(self):
        self._compare(NormalizedHeat_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.009)

    def test_logFor(self):
        self._compare(RegularizedLaplacian_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.0026)

    def test_logPPR(self):
        self._compare(logPPR_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.0021)

    def test_logModifPPR(self):
        self._compare(logModifPPR_R, params=np.linspace(0, 1, 101)[1:-1], error_true=0.0022)

    def test_logHeatPPR(self):
        self._compare(logHeatPPR_R, params=np.linspace(0, 20, 101)[1:-1], error_true=0.0021)


if __name__ == "__main__":
    unittest.main()
