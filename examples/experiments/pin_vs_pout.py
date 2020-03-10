import sys
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

sys.path.append('../..')
from pygraphs.util import load_or_calc_and_save
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.scenario import ParallelByGraphs, d3_colors

# Experiment 1
# Calc a 2d field p_in vs p_out for several n and k(balanced classes)
# n = 50, 100, 150, 200  k = 2, 3, 4, 5, 10  p_{in}, p_{out} \ in (0, 1] \text{with step}=0.05
# mean( or median) by 100 graphs


kernel_colors = dict([(k.name, i) for i, k in enumerate(kernels)])


class CellMeasureResults:
    def __init__(self, name, params, ari, error):
        self.measure_name = name
        self.params = params
        self.ari = ari
        self.error = error

    def mari(self, method='max'):
        if len(self.ari) == 0:
            return 0
        elif method == 'max':
            return np.nanmax(self.ari)
        elif method == 'mean':
            return np.nanmean(self.ari)
        elif method == 'median':
            return np.nanmedian(self.ari)


class PlotCellResults:
    def __init__(self):
        self.measure_results = {}

    def best_measure(self):
        measure_results_list = [(measure_name, measure_result.mari(method='max')) \
                                for measure_name, measure_result in self.measure_results.items()]
        measure_results_list = sorted(measure_results_list, key=lambda x: x[1], reverse=True)
        return measure_results_list[0]  # tuple (name, ari)


class PlotResults:
    def __init__(self, name, n_pin, n_pout):
        self.name = name
        self.results = [[PlotCellResults() for _ in range(n_pout)] for _ in range(n_pin)]

    def best_measure_map(self, kernel_colors):
        measure_map = np.full((len(self.results), len(self.results[0])), np.nan)
        ari_map = np.full((len(self.results), len(self.results[0])), np.nan)
        for i in range(len(self.results)):
            for j in range(len(self.results[0])):
                name, ari = self.results[i][j].best_measure()
                measure_map[i, j] = kernel_colors[name]
                ari_map[i, j] = ari
        return measure_map, ari_map


def calc_one_pixel(n, k, p_in, p_out, estimator, n_graphs, n_params, n_jobs, n_gpu):
    @load_or_calc_and_save(f'./cache/pin_vs_pout/{p_in:.2f}_{p_out:.2f}.pkl')
    def _calc(n_graphs=100, n_params=21, n_jobs=6):
        graphs, _ = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
        classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False, ignore_errors=True)

        cell_results = PlotCellResults()
        for kernel in tqdm(list(kernels), desc=f'{p_in}, {p_out}'):
            params, ari, error = classic_plot.perform(estimator, kernel, graphs, k, n_jobs=n_jobs, n_gpu=n_gpu)
            cell_results.measure_results[kernel.name] = CellMeasureResults(kernel.name, params, ari, error)
        return cell_results

    return _calc(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)


def calc(n, k, estimator, p_ins, p_outs, pin_pout_step, experiment_name):
    n_graphs = 50
    n_params = 31
    n_jobs = 12
    n_gpu = 4

    picture_results = PlotResults(experiment_name, p_ins.shape[0], p_outs.shape[0])
    for p_in, p_out in tqdm(list(product(p_ins, p_outs)), desc=experiment_name):  # one pixel
        p_in_idx, p_out_idx = int(p_in / pin_pout_step), int(p_out / pin_pout_step)
        cell_results = calc_one_pixel(n, k, p_in, p_out, estimator, n_graphs, n_params, n_jobs, n_gpu)
        picture_results.results[p_in_idx][p_out_idx] = cell_results
    return picture_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pin_offset', type=float)
    parser.add_argument('pout_offset', type=float)
    args = parser.parse_args()

    pin_offset, pout_offset = args.pin_offset, args.pout_offset

    ns, ks, estimators, pin_pout_step = [100], [2], [KKMeans], 0.1
    p_ins, p_outs = np.arange(0.0 + pin_offset, 1.0001, pin_pout_step), np.arange(0.0 + pout_offset, 1.0001, pin_pout_step)

    for n, k, estimator in product(ns, ks, estimators):  # one picture
        experiment_name = f'{n}_{k}_{estimator.name}'
        picture_results = calc(n, k, estimator, p_ins, p_outs, pin_pout_step, experiment_name)

        for cell_x, content_x in enumerate(picture_results.results):
            for cell_y, content_xy in enumerate(content_x):
                for measure_name, content_xy_measure in content_xy.measure_results.items():
                    if len(content_xy_measure.ari) == 0:
                        content_xy_measure.ari = [0]
