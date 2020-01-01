import os
import sys
import warnings
from itertools import product

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from pygraphs.graphs import StochasticBlockModel
from pygraphs.measure import *
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.scenario import ParallelByGraphs, plot_results
from pygraphs.util import load_or_calc_and_save


def _calc(graphs, name=None, n_jobs=-1):
    estimators = [KKMeans]
    measures = [pWalk_H, Walk_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, NHeat_H, logNHeat_H,
                PPR_H, logPPR_H, ModifPPR_H, logModifPPR_H, HeatPPR_H, logHeatPPR_H]

    classic_plot = ParallelByGraphs(adjusted_rand_score, np.linspace(0, 1, 101), progressbar=True)
    results = defaultdict(list)
    for estimator, measure in tqdm(list(product(estimators, measures)), desc=name):
        results[f"{estimator.name}_{measure.name}"] = classic_plot.perform(estimator, measure, graphs, 2, n_jobs)
    return results


def _plot_log_results(results, img_path):
    fig, ax = plt.subplots(2, 8, figsize=(15, 6))
    for idx_i, estimator in enumerate([KKMeans]):
        ax[idx_i][0].set_ylabel(f'{estimator.name}, f1')
        for idx_j, (name1, name2, xlim, ylim) in enumerate([
            [(f'{estimator.name}_pWalk', 'pWalk'),
             (f'{estimator.name}_Walk', 'Walk'), (0, 1), (0, 1)],
            [(f'{estimator.name}_For', 'For'),
             (f'{estimator.name}_logFor', 'logFor'), (0, 1), (0, 1)],
            [(f'{estimator.name}_Comm', 'Comm'),
             (f'{estimator.name}_logComm', 'logComm'), (0, 0.83), (0, 1)],
            [(f'{estimator.name}_Heat', 'Heat'),
             (f'{estimator.name}_logHeat', 'logHeat'), (0, 0.83), (0, 1)],
            [(f'{estimator.name}_NHeat', 'NHeat'),
             (f'{estimator.name}_logNHeat', 'logNHeat'), (0, 0.83), (0, 1)],
            [(f'{estimator.name}_PPR', 'PPR'),
             (f'{estimator.name}_logPPR', 'logPPR'), (0, 1), (0, 1)],
            [(f'{estimator.name}_ModifPPR', 'ModifPPR'),
             (f'{estimator.name}_logModifPPR', 'logModifPPR'), (0, 1), (0, 1)],
            [(f'{estimator.name}_HeatPPR', 'HeatPPR'),
             (f'{estimator.name}_logHeatPPR', 'logHeatPPR'), (0, 1), (0, 1)]
        ]):
            toplot = [
                (name1[1], *results[name1[0]]),
                (name2[1], *results[name2[0]]),
            ]
            plot_results(ax[idx_i][idx_j], toplot, xlim, ylim)
    plt.savefig(img_path)


def _plot_log_results4(results, img_path):
    fig, ax = plt.subplots(2, 4, figsize=(7, 5), sharey=True)
    plt.subplots_adjust(hspace=.3, wspace=.2)
    for idx_j, (name1, name2, name3, name4, xlim, ylim) in enumerate([
        [(f'KKMeans_pWalk', 'pWalk'), (f'KKMeans_Walk', 'Walk'),
         (f'KWard_pWalk', 'pWalk'), (f'KWard_Walk', 'Walk'), (0, 1), (0, 1.05)],
        [(f'KKMeans_For', 'For'), (f'KKMeans_logFor', 'logFor'),
         (f'KWard_For', 'For'), (f'KWard_logFor', 'logFor'), (0, 1), (0, 1.05)],
        [(f'KKMeans_Comm', 'Comm'), (f'KKMeans_logComm', 'logComm'),
         (f'KWard_Comm', 'Comm'), (f'KWard_logComm', 'logComm'), (0, 0.6), (0, 1.05)],
        [(f'KKMeans_Heat', 'Heat'), (f'KKMeans_logHeat', 'logHeat'),
         (f'KWard_Heat', 'Heat'), (f'KWard_logHeat', 'logHeat'), (0, 0.83), (0, 1.05)],
        [(f'KKMeans_NHeat', 'NHeat'), (f'KKMeans_logNHeat', 'logNHeat'),
         (f'KWard_NHeat', 'NHeat'), (f'KWard_logNHeat', 'logNHeat'), (0, 1), (0, 1.05)],
        [(f'KKMeans_PPR', 'PPR'), (f'KKMeans_logPPR', 'logPPR'),
         (f'KWard_PPR', 'PPR'), (f'KWard_logPPR', 'logPPR'), (0, 1), (0, 1.05)],
        [(f'KKMeans_ModifPPR', 'ModifPPR'), (f'KKMeans_logModifPPR', 'logModifPPR'),
         (f'KWard_ModifPPR', 'ModifPPR'), (f'KWard_logModifPPR', 'logModifPPR'), (0, 1), (0, 1.05)],
        [(f'KKMeans_HeatPPR', 'HeatPPR'), (f'KKMeans_logHeatPPR', 'logHeatPPR'),
         (f'KWard_HeatPPR', 'HeatPPR'), (f'KWard_logHeatPPR', 'logHeatPPR'), (0, 1), (0, 1.05)]
    ]):
        toplot = [
            ("KKMeans, plain", *results[name1[0]]),
            ("KKMeans, log", *results[name2[0]]),
            ("KWard, plain", *results[name3[0]]),
            ("KWard, log", *results[name4[0]]),
        ]
        plot_results(ax[idx_j // 4][idx_j % 4], toplot, xlim, ylim, nolegend=True)
        ax[idx_j // 4][idx_j % 4].set_title(name1[1])
        if idx_j == 0 or idx_j == 4:
            ax[idx_j // 4][idx_j % 4].set_ylabel(f'ARI')
        if idx_j == 3:
            ax[idx_j // 4][idx_j % 4].legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.savefig(img_path)


@load_or_calc_and_save('cache/2_11_ari.pkl')
def calc2_11(n_graphs=200, n_jobs=-1):
    """**Fig. 1** Logarithmic vs. plain measures for G(100,(2)0.2,0.05)"""
    graphs, info = StochasticBlockModel(100, 2, p_in=0.2, p_out=0.05).generate_graphs(n_graphs)
    return _calc(graphs, name='2_11', n_jobs=n_jobs)


@load_or_calc_and_save('cache/2_21_ari.pkl')
def calc2_21(n_graphs=200, n_jobs=-1):
    """**Fig. 2** Logarithmic vs. plain measures for G(100,(3)0.3,0.1)"""
    graphs, info = StochasticBlockModel(102, 3, p_in=0.3, p_out=0.1).generate_graphs(n_graphs)
    return _calc(graphs, name='2_21', n_jobs=n_jobs)


@load_or_calc_and_save('cache/2_31_ari.pkl')
def calc2_31(n_graphs=200, n_jobs=-1):
    """**Fig. 3** Logarithmic vs. plain measures for G(200,(2)0.3,0.1)"""
    graphs, info = StochasticBlockModel(200, 2, p_in=0.3, p_out=0.1).generate_graphs(n_graphs)
    return _calc(graphs, name='2_31', n_jobs=n_jobs)


def calc_part2(n_graphs=200, n_jobs=6):
    results = calc2_11(n_graphs, n_jobs)
    _plot_log_results(results, 'results/2_11_1.png')
    # _plot_log_results4(results, 'results/2_11_2.png')

    results = calc2_21(n_graphs, n_jobs)
    _plot_log_results(results, 'results/2_21_1.png')
    # _plot_log_results4(results, 'results/2_21_2.png')

    results = calc2_31(n_graphs, n_jobs)
    _plot_log_results(results, 'results/2_31_1.png')
    # _plot_log_results4(results, 'results/2_31_2.png')


if __name__ == '__main__':
    calc_part2(n_graphs=1)
