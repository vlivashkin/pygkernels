import sys
import warnings
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.append('../..')
from _generated_kkmeans import generated_kkmeans_any
from _generated_kward import generated_kward
from pygkernels.measure import kernels
from pygkernels.scenario import plot_results


def _plot_log_results(results, img_path):
    print(f'_plot_log_results: {img_path}')

    fig, ax = plt.subplots(2, 8, figsize=(15, 6), sharex=True, sharey=True)
    for idx_i, estimator in enumerate(['KKMeans', 'KWard']):
        ax[idx_i][0].set_ylabel(f'{estimator}, ARI')
        for idx_j, (name_plain, name_log, xlim, ylim) in enumerate([
            ('pWalk', 'Walk', (0, 1), (0, 1.05)),
            ('For', 'logFor', (0, 1), (0, 1.05)),
            ('Comm', 'logComm', (0, 1), (0, 1.05)),
            ('Heat', 'logHeat', (0, 1), (0, 1.05)),
            ('NHeat', 'logNHeat', (0, 1), (0, 1.05)),
            ('PPR', 'logPPR', (0, 1), (0, 1.05)),
            ('ModifPPR', 'logModifPPR', (0, 1), (0, 1.05)),
            ('HeatPPR', 'logHeatPPR', (0, 1), (0, 1.05))
        ]):
            toplot = [
                ('plain', *[results[estimator][name_plain][x] for x in ['x', 'y', 'error']]),
                ('log', *[results[estimator][name_log][x] for x in ['x', 'y', 'error']]),
            ]
            plot_results(ax[idx_i][idx_j], toplot, xlim, ylim, nolegend=True)
            if idx_i == 0:
                ax[0][idx_j].set_title(name_plain)
            if idx_i == 0 and idx_j == 7:
                ax[0][7].legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(img_path, bbox_inches='tight')


def _plot_log_results4(results, img_path):
    print(f'_plot_log_results4: {img_path}')

    fig, ax = plt.subplots(2, 4, figsize=(7, 5), sharex=True, sharey=True)
    # plt.subplots_adjust(hspace=.3, wspace=.2)
    for idx_j, (name_plain, name_log, xlim, ylim) in enumerate([
        ('pWalk', 'Walk', (0, 1), (0, 1.05)),
        ('For', 'logFor', (0, 1), (0, 1.05)),
        ('Comm', 'logComm', (0, 1), (0, 1.05)),
        ('Heat', 'logHeat', (0, 1), (0, 1.05)),
        ('NHeat', 'logNHeat', (0, 1), (0, 1.05)),
        ('PPR', 'logPPR', (0, 1), (0, 1.05)),
        ('ModifPPR', 'logModifPPR', (0, 1), (0, 1.05)),
        ('HeatPPR', 'logHeatPPR', (0, 1), (0, 1.05))
    ]):
        toplot = [
            ("KKMeans, plain", *[results['KKMeans'][name_plain][x] for x in ['x', 'y', 'error']]),
            ("KKMeans, log", *[results['KKMeans'][name_log][x] for x in ['x', 'y', 'error']]),
            ("KWard, plain", *[results['KWard'][name_plain][x] for x in ['x', 'y', 'error']]),
            ("KWard, log", *[results['KWard'][name_log][x] for x in ['x', 'y', 'error']]),
        ]
        plot_results(ax[idx_j // 4][idx_j % 4], toplot, xlim, ylim, nolegend=True)
        ax[idx_j // 4][idx_j % 4].set_title(name_plain)
        if idx_j == 0 or idx_j == 4:
            ax[idx_j // 4][idx_j % 4].set_ylabel('ARI')
        if idx_j == 3:
            ax[idx_j // 4][idx_j % 4].legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(img_path, bbox_inches='tight')


def calc_part2(n_graphs=200, n_jobs=6):
    # classic_plots: [column][kernel_name][init][feature]
    cache_kkmeans = generated_kkmeans_any(n_graphs=n_graphs, n_jobs=n_jobs)
    cache_kward = generated_kward(n_graphs=n_graphs, n_jobs=n_jobs)

    init = 'any'
    columns = [
        (100, 2, 0.2, 0.05),
        (102, 3, 0.3, 0.1),
        (200, 2, 0.3, 0.1)
    ]

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for column, kernel in product(columns, kernels):
        results[column]['KKMeans'][kernel.name] = cache_kkmeans[column][kernel.name][init]
        results[column]['KWard'][kernel.name] = cache_kward[column][kernel.name]

    _plot_log_results(results[(100, 2, 0.2, 0.05)], 'results/p2-g100_2_02_005-1.png')
    _plot_log_results4(results[(100, 2, 0.2, 0.05)], 'results/p2-g100_2_02_005-2.png')

    _plot_log_results(results[(102, 3, 0.3, 0.1)], 'results/p2-g102_3_03_01-1.png')
    _plot_log_results4(results[(102, 3, 0.3, 0.1)], 'results/p2-g102_3_03_01-2.png')

    _plot_log_results(results[(200, 2, 0.3, 0.1)], 'results/p2-g200_2_03_01-1.png')
    _plot_log_results4(results[(200, 2, 0.3, 0.1)], 'results/p2-g200_2_03_01-2.png')


if __name__ == '__main__':
    calc_part2(n_graphs=1)
