import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt

from _classic_plot_kkmeans import classic_plots_kkmeans
from _classic_plot_kward import classic_plots_kward
from pygraphs.measure import kernels
from pygraphs.scenario import plot_results


def _plot_log_results(results, img_path):
    fig, ax = plt.subplots(2, 8, figsize=(15, 6))
    for idx_i, estimator in enumerate(['KKMeans', 'KWard']):
        ax[idx_i][0].set_ylabel(f'{estimator}, f1')
        for idx_j, (name1, name2, xlim, ylim) in enumerate([
            ['pWalk', 'Walk', (0, 1), (0, 1)],
            ['For', 'logFor', (0, 1), (0, 1)],
            ['Comm', 'logComm', (0, 0.83), (0, 1)],
            ['Heat', 'logHeat', (0, 0.83), (0, 1)],
            ['NHeat''logNHeat', (0, 0.83), (0, 1)],
            ['PPR', 'logPPR', (0, 1), (0, 1)],
            ['ModifPPR', 'logModifPPR', (0, 1), (0, 1)],
            ['HeatPPR', 'logHeatPPR', (0, 1), (0, 1)]
        ]):
            toplot = [
                (name1, *results[estimator][name1]),
                (name2, *results[estimator][name2]),
            ]
            plot_results(ax[idx_i][idx_j], toplot, xlim, ylim)
    plt.savefig(img_path)


def _plot_log_results4(results, img_path):
    fig, ax = plt.subplots(2, 4, figsize=(7, 5), sharey=True)
    plt.subplots_adjust(hspace=.3, wspace=.2)
    for idx_j, (name1, name2, xlim, ylim) in enumerate([
        ['pWalk', 'Walk', (0, 1), (0, 1.05)],
        ['For', 'logFor', (0, 1), (0, 1.05)],
        ['Comm', 'logComm', (0, 0.6), (0, 1.05)],
        ['Heat', 'logHeat', (0, 0.83), (0, 1.05)],
        ['NHeat', 'logNHeat', (0, 1), (0, 1.05)],
        ['PPR', 'logPPR', (0, 1), (0, 1.05)],
        ['ModifPPR', 'logModifPPR', (0, 1), (0, 1.05)],
        ['HeatPPR', 'logHeatPPR', (0, 1), (0, 1.05)]
    ]):
        toplot = [
            ("KKMeans, plain", *results['KKMeans'][name1]),
            ("KKMeans, log", *results['KKMeans'][name2]),
            ("KWard, plain", *results['KWard'][name1]),
            ("KWard, log", *results['KWard'][name2]),
        ]
        plot_results(ax[idx_j // 4][idx_j % 4], toplot, xlim, ylim, nolegend=True)
        ax[idx_j // 4][idx_j % 4].set_title(name1[1])
        if idx_j == 0 or idx_j == 4:
            ax[idx_j // 4][idx_j % 4].set_ylabel(f'ARI')
        if idx_j == 3:
            ax[idx_j // 4][idx_j % 4].legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(img_path)


def calc_part2(n_graphs=200, n_jobs=6):
    # classic_plots: [column][kernel_name][init][feature]
    plots_kkmeans = classic_plots_kkmeans(n_graphs=n_graphs, n_jobs=n_jobs)
    plots_kward = classic_plots_kward(n_graphs=n_graphs, n_jobs=n_jobs)

    init = 'k-means++'
    columns = [
        (100, 2, 0.2, 0.05),
        (102, 3, 0.3, 0.1),
        (200, 2, 0.3, 0.1)
    ]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for column, kernel in product(columns, kernels):
        results[column]['KKMeans'][kernel.name] = plots_kkmeans[column][kernel.name][init]
        results[column]['KWard'][kernel.name] = plots_kward[column][kernel.name]

    _plot_log_results(results[(100, 2, 0.2, 0.05)], 'results/2_11_1.png')
    _plot_log_results4(results[(100, 2, 0.2, 0.05)], 'results/2_11_2.png')

    _plot_log_results(results[(102, 3, 0.3, 0.1)], 'results/2_21_1.png')
    _plot_log_results4(results[(102, 3, 0.3, 0.1)], 'results/2_21_2.png')

    _plot_log_results(results[(200, 2, 0.3, 0.1)], 'results/2_31_1.png')
    _plot_log_results4(results[(200, 2, 0.3, 0.1)], 'results/2_31_2.png')


if __name__ == '__main__':
    calc_part2(n_graphs=1)
