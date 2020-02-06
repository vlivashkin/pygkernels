import json
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from pygraphs.util import load_or_calc_and_save

sys.path.append('../..')
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.scenario import ParallelByGraphs

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


def calc(n, k, estimator, p_ins, p_outs, pin_pout_step, experiment_name):
    @load_or_calc_and_save(f'./cache/pin_vs_pout_{n}_{k}.pkl')
    def _calc(n_graphs=10, n_params=21, n_jobs=6):
        classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False)

        all_graphs = {}
        for p_in, p_out in product(p_ins, p_outs):  # one pixel
            graphs, info = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
            all_graphs[(p_in, p_out)] = graphs

        picture_results = PlotResults(experiment_name, p_ins.shape[0], p_outs.shape[0])
        for p_in, p_out, kernel in tqdm(list(product(p_ins, p_outs, kernels)), desc=experiment_name):  # one pixel
            p_in_idx, p_out_idx = int(p_in / pin_pout_step), int(p_out / pin_pout_step)
            graphs = all_graphs[(p_in, p_out)]
            try:
                params, ari, error = classic_plot.perform(estimator, kernel, graphs, k, n_jobs=n_jobs)
            except KeyboardInterrupt:
                exit(1)
            except:
                print(f'{p_in}, {p_out}, {kernel.name} -- error')
                params, ari, error = [], [], []
            cell_results = picture_results.results[p_in_idx][p_out_idx]
            cell_results.measure_results[kernel.name] = CellMeasureResults(kernel.name, params, ari, error)
        return picture_results

    return _calc(n_graphs=10, n_params=31, n_jobs=6)


def draw(picture_results, p_ins, p_outs, img_path):
    colors, ari = picture_results.best_measure_map(kernel_colors)

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    ax[0].imshow(colors)
    ax[0].set_xticks(range(p_ins.shape[0]))
    ax[0].set_yticks(range(p_outs.shape[0]))
    ax[0].set_xticklabels([f'{x:.1f}' for x in p_ins])
    ax[0].set_yticklabels([f'{x:.1f}' for x in p_outs])
    ax[0].set_xlabel('$p_{in}$')
    ax[0].set_ylabel('$p_{out}$')

    ax[1].imshow(ari, vmin=0, vmax=1)
    ax[1].set_xticks(range(p_ins.shape[0]))
    ax[1].set_yticks(range(p_outs.shape[0]))
    ax[1].set_xticklabels([f'{x:.1f}' for x in p_ins])
    ax[1].set_yticklabels([f'{x:.1f}' for x in p_outs])
    ax[1].set_xlabel('$p_{in}$')
    ax[1].set_ylabel('$p_{out}$')

    plt.savefig(img_path, bbox_inches='tight')


def save_to_json(picture_results, p_ins, p_outs, pin_pout_step, experiment_name, filename):
    jjson = {}
    for p_in, p_out in tqdm(list(product(p_ins, p_outs)), desc=experiment_name):  # one pixel
        p_in_idx, p_out_idx = int(p_in / pin_pout_step), int(p_out / pin_pout_step)
        cell_results = picture_results.results[p_in_idx][p_out_idx]

        jcell = {}
        for measure_result in cell_results.measure_results.values():
            jcell[measure_result.measure_name] = {
                'params': measure_result.params.tolist() if type(measure_result.params) != list else [],
                'ari': measure_result.ari.tolist() if type(measure_result.ari) != list else [],
                'error': measure_result.error.tolist() if type(measure_result.error) != list else [],
            }
        jjson[f'{p_in}, {p_out}'] = jcell

    with open(filename, 'w') as f:
        json.dump(jjson, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    ns, ks, estimators, pin_pout_step = [100], [2], [KKMeans], 0.1
    p_ins, p_outs = np.arange(0.0, 1.0001, pin_pout_step), np.arange(0.0, 1.0001, pin_pout_step)

    for n, k, estimator in product(ns, ks, estimators):  # one picture
        experiment_name = f'{n}_{k}_{estimator.name}'
        picture_results = calc(n, k, estimator, p_ins, p_outs, pin_pout_step, experiment_name)

        for cell_x, content_x in enumerate(picture_results.results):
            for cell_y, content_xy in enumerate(content_x):
                for measure_name, content_xy_measure in content_xy.measure_results.items():
                    if len(content_xy_measure.ari) == 0:
                        content_xy_measure.ari = [0]

        draw(picture_results, p_ins, p_outs, './results/pin_pout-100_2.png')
        save_to_json(picture_results, p_ins, p_outs, pin_pout_step, experiment_name, './results/pin_pout-100_2.json')
