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
    n_jobs = 6
    n_gpu = 4

    picture_results = PlotResults(experiment_name, p_ins.shape[0], p_outs.shape[0])
    for p_in, p_out in tqdm(list(product(p_ins, p_outs)), desc=experiment_name):  # one pixel
        p_in_idx, p_out_idx = int(p_in / pin_pout_step), int(p_out / pin_pout_step)
        cell_results = calc_one_pixel(n, k, p_in, p_out, estimator, n_graphs, n_params, n_jobs, n_gpu)
        picture_results.results[p_in_idx][p_out_idx] = cell_results
    return picture_results


def draw(plot_results: PlotResults, p_ins, p_outs,
         plot1_name='./results/pin_vs_pout-top.png',
         plot2_name='./results/pin_vs_pout-legend.png',
         plot3_name='./results/pin_vs_pout-maxari.png',
         plot4_name='./results/pin_vs_pout-rating.png'):
    kernel_name_to_id = dict([(k.name, i) for i, k in enumerate(kernels)])
    kernel_id_to_name = list(kernel_name_to_id.keys())
    print(kernel_name_to_id, kernel_id_to_name)

    side_len = p_ins.shape[0]
    best_measure, best_ari = np.full((side_len, side_len), np.nan), np.full((side_len, side_len), np.nan)
    measure_counter = defaultdict(lambda: 0)
    global_ratings = defaultdict(lambda: 0)
    global_ratings_upper = defaultdict(lambda: 0)
    global_ratings_lower = defaultdict(lambda: 0)

    measure_ratings = np.full((side_len, side_len, len(kernel_name_to_id)), np.nan)
    for pin_idx in range(side_len):
        for pout_idx in range(side_len):
            cell_results = plot_results.results[pin_idx][pout_idx]

            measure_best_ari = []
            for measure_name, measure_results in cell_results.measure_results.items():
                if len(measure_results.ari) > 0:
                    best_ari_of_measure = measure_results.mari()
                    if ~np.isnan(best_ari_of_measure):
                        measure_best_ari.append((measure_name, best_ari_of_measure))
            measure_best_ari = sorted(measure_best_ari, key=lambda x: x[1], reverse=True)

            for current_name, current_ari in measure_best_ari:
                better_than, worse_than = 0, 0
                for measure_name, measure_ari in measure_best_ari:
                    if current_ari > measure_ari:
                        better_than += 1
                    if current_ari < measure_ari:
                        worse_than += 1
                measure_ratings[pin_idx, pout_idx, kernel_name_to_id[current_name]] = worse_than + 1
                if pin_idx != pout_idx:
                    global_ratings[current_name] += better_than - worse_than
                if pin_idx > pout_idx:
                    global_ratings_upper[current_name] += better_than - worse_than
                if pin_idx < pout_idx:
                    global_ratings_lower[current_name] += better_than - worse_than

            if np.sum(np.array([x[1] for x in measure_best_ari]) == 1) < 2:
                best_measure[pin_idx, pout_idx] = kernel_name_to_id[measure_best_ari[0][0]]
                best_ari[pin_idx, pout_idx] = measure_best_ari[0][1]
                measure_counter[measure_best_ari[0][0]] += 1

    best_measure_color = np.zeros((side_len, side_len, 3), dtype=np.uint8)
    for i in range(side_len):
        for j in range(side_len):
            if not np.isnan(best_measure[i][j]):
                color = d3_colors[kernel_id_to_name[int(best_measure[i][j])]]
                color = tuple(int(color[1:][i:i + 2], 16) for i in (0, 2, 4))
                best_measure_color[i][j] = np.array(color)

    p_ins = [f'{x:.2}' for x in np.arange(0, 1.0001, 0.1)]
    p_outs = [f'{x:.2}' for x in np.arange(0, 1.0001, 0.1)]

    colors_legend, names_legend = [], []
    for measure_name in list(kernel_name_to_id.keys()):
        color = d3_colors[measure_name]
        color = tuple(int(color[1:][i:i + 2], 16) for i in (0, 2, 4))
        name = f'{measure_name} ({measure_counter[measure_name]})'
        colors_legend.append(color)
        names_legend.append(name)
    colors_legend, names_legend = np.array(colors_legend), np.array(names_legend)

    # plot 1
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.imshow(best_measure_color[::-1])
    ax.set_yticks(range(0, len(p_ins), 1))  # * 2, 2
    ax.set_xticks(range(0, len(p_outs), 1))
    ax.set_yticklabels(p_ins[::-1])
    ax.set_xticklabels(p_outs)
    ax.set_ylabel('$p_{in}$')
    ax.set_xlabel('$p_{out}$')

    for i in range(side_len):
        for j in range(side_len):
            if not np.isnan(best_measure[i][j]):
                ax.text(j - 0.29, side_len - i - 0.75, str(int(best_measure[i][j])).rjust(2))
    plt.savefig(plot1_name, bbox_inches='tight')

    # plot 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(colors_legend[:, None, :])
    ax.set_yticks(range(len(names_legend)))
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.yaxis.tick_right()
    ax.set_yticklabels(names_legend)

    for i in range(len(names_legend)):
        ax.text(-0.4, i + 0.3, str(i).rjust(2))

    plt.savefig(plot2_name, bbox_inches='tight')

    # plot 3
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    a = ax.imshow(best_ari[::-1], vmin=0, vmax=1)
    ax.set_yticks(range(0, len(p_ins), 1))  # * 2, 2
    ax.set_xticks(range(0, len(p_outs), 1))
    ax.set_yticklabels(p_ins[::-1])
    ax.set_xticklabels(p_outs)
    ax.set_ylabel('$p_{in}$')
    ax.set_xlabel('$p_{out}$')
    plt.colorbar(a)

    plt.savefig(plot3_name, bbox_inches='tight')

    # plot 4
    fig, ax = plt.subplots(4, 6, figsize=(18, 12), sharex=True, sharey=True)
    for measure_name, measure_idx in kernel_name_to_id.items():
        axi = ax[measure_idx // 6][measure_idx % 6]
        axi.imshow(measure_ratings[::-1, :, measure_idx], cmap='viridis_r')

        axi.set_yticks(range(0, len(p_ins), 1))  # * 2, 2
        axi.set_xticks(range(0, len(p_outs), 1))
        axi.set_yticklabels(p_ins[::-1])
        axi.set_xticklabels(p_outs)
        axi.set_ylabel('$p_{in}$')
        axi.set_xlabel('$p_{out}$')

        axi.set_title(measure_name)
        axi.plot(range(side_len)[::-1], range(side_len), color='black')

    plt.savefig(plot4_name, bbox_inches='tight')


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

        draw(picture_results, p_ins, p_outs)
