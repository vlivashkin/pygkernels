import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import *
from pygraphs.scenario import RejectCurve, d3_colors
from _classic_plot_kkmeans import classic_plots_kkmeans

distances_kernels_pairs = [
    ('pWalk', pWalk_H, pWalk_D),
    ('Walk', Walk_H, Walk_D),
    ('For', For_H, For_D),
    ('logFor', logFor_H, logFor_D),
    ('Comm', Comm_H, Comm_D),
    ('logComm', logComm_H, logComm_D),
    ('Heat', Heat_H, Heat_D),
    ('logHeat', logHeat_H, logHeat_D),
    ('NHeat', NHeat_H, NHeat_D),
    ('logNHeat', logNHeat_H, logNHeat_D),
    ('SCT', SCT_H, SCT_D),
    ('SCCT', SCCT_H, SCCT_D),
    ('RSP', RSP_K, RSP_D),
    ('FE', FE_K, FE_D),
    ('logPPR', PPR_H, PPR_D),
    ('logPPR', logPPR_H, logPPR_D),
    ('ModifPPR', ModifPPR_H, ModifPPR_D),
    ('logModifPPR', logModifPPR_H, logModifPPR_D),
    ('HeatPPR', HeatPPR_H, HeatPPR_D),
    ('logHeatPPR', logHeatPPR_H, logHeatPPR_D),
    ('SP-CT', SPCT_H, SPCT_D),
    ('SP', SP_K, SP_D),
    ('CT', CT_H, CT_D)
]

all_names = [x[0] for x in distances_kernels_pairs]
all_measures = [x[1] for x in distances_kernels_pairs]
all_distances = [x[2] for x in distances_kernels_pairs]


def _draw_one_by_one(results_rc, out_name):
    print(f'_draw_one_by_one: out_name={out_name}')

    fig, ax = plt.subplots(int((len(all_measures) + 1) / 2), 6, figsize=(20, 30), sharex=True, sharey=True)
    for column_idx, column in enumerate(results_rc.keys()):
        for measure_name_idx, measure_name in enumerate(all_names):
            axi = ax[measure_name_idx // 2][column_idx + 3 * (measure_name_idx % 2)]
            for graph_idx, (tpr, fpr) in enumerate(results_rc[column][measure_name]):
                axi.plot(tpr, fpr, color='black', alpha=0.1)
            axi.set_title("G({}, ({}){}, {}), {}".format(*column, measure_name))
            axi.set_xlim(0, 1)
            axi.set_ylim(0, 1)
    plt.savefig(out_name, bbox_inches='tight')


def _draw_g100_2_03_01(results_rc, out_name):
    print(f'_draw_g100_2_03_01: out_name={out_name}')

    fig, ax = plt.subplots(5, 5, figsize=(16, 16), sharex=True, sharey=True)
    column = list(results_rc.keys())[1]
    for measure_name_idx, measure_name in enumerate(all_names):
        axi = ax[measure_name_idx // 5][measure_name_idx % 5]
        for graph_idx, (tpr, fpr) in enumerate(results_rc[column][measure_name]):
            axi.plot(tpr, fpr, color='black', alpha=0.1)
        axi.set_title("G({}, ({}){}, {}), {}".format(*column, measure_name))
        axi.set_xlim(0, 1)
        axi.set_ylim(0, 1)
    plt.savefig(out_name, bbox_inches='tight')


def _draw_all(results_rc, out_name):
    print(f'_draw_all: out_name={out_name}')

    fig, axi = plt.subplots(1, figsize=(5, 4))
    for measure_name_idx, measure_name in enumerate(all_names):
        tpr_all = defaultdict(list)
        for graph_idx, (tpr, fpr) in enumerate(results_rc[(100, 2, 0.3, 0.10)][measure_name]):
            tprg = defaultdict(list)
            for ti, fi in zip(fpr, tpr):
                tprg[np.floor(fi * 100)].append(ti)
            for bucket, fis in tprg.items():
                tpr_all[bucket].append(np.mean(fis))
        for bucket, fis in tpr_all.items():
            tpr_all[bucket] = np.mean(fis)

        axi.plot(np.array(list(tpr_all.keys()), dtype=np.float) / 100, list(tpr_all.values()),
                 label=measure_name, color=d3_colors[measure_name])
    axi.set_xlabel("nodes from different classes")
    axi.set_ylabel("nodes from the same class")

    box = axi.get_position()
    axi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name, bbox_inches='tight')


def _draw_4best(results_rc, out_name):
    print(f'_draw_4best: out_name={out_name}')

    fig, axi = plt.subplots(1, figsize=(5, 4))
    for measure_name_idx, measure_name in enumerate(all_names):
        tpr_all = defaultdict(list)
        for graph_idx, (tpr, fpr) in enumerate(results_rc[(100, 2, 0.3, 0.10)][measure_name]):
            tprg = defaultdict(list)
            for ti, fi in zip(fpr, tpr):
                tprg[np.floor(fi * 100)].append(ti)
            for bucket, fis in tprg.items():
                tpr_all[bucket].append(np.mean(fis))
        for bucket, fis in tpr_all.items():
            tpr_all[bucket] = np.mean(fis)

        if measure_name not in ['logComm', 'logFor', 'logHeat', 'SCCT']:
            continue

        axi.plot(np.array(list(tpr_all.keys()), dtype=np.float) / 100, list(tpr_all.values()),
                 label=measure_name, color=d3_colors[measure_name])
    axi.set_xlabel("nodes from different classes")
    axi.set_ylabel("nodes from the same class")

    box = axi.get_position()
    axi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name, bbox_inches='tight')


def calc_part4(n_graphs=100):
    cache_kkmeans, init = classic_plots_kkmeans(), 'k-means++'
    result_params = defaultdict(lambda: defaultdict(dict))  # choose k-means++ init
    for column in cache_kkmeans.keys():
        for kernel_name in cache_kkmeans[column].keys():
            result_params[column][kernel_name] = cache_kkmeans[column][kernel_name][init]['best_param']

    columns = [  # n_nodes, n_classes, p_in, p_out
        (100, 2, 0.3, 0.05),
        (100, 2, 0.3, 0.1),
        (100, 2, 0.3, 0.15),
    ]

    rc = RejectCurve(columns, all_distances, StochasticBlockModel, result_params)
    results_rc = rc.perform(n_graphs=n_graphs)

    _draw_one_by_one(results_rc, 'results/p4-one_by_one.png')  # draw all rc
    _draw_g100_2_03_01(results_rc, 'results/p4-g100_2_0.3_0.1.png')  # draw (100, 2, 0.3, 0.1) for all measures
    _draw_all(results_rc, 'results/p4-all.png')  # draw all in one pic
    _draw_4best(results_rc, 'results/p4-4best.png')  # draw 4 best measures


if __name__ == '__main__':
    calc_part4(n_graphs=10)
