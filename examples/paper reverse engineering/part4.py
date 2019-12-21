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
from pygraphs.cluster.kkmeans import KKMeans_iterative as KKMeans
from pygraphs.scenario import RejectCurve, d3_colors

distances_kernels_pairs = [
    (pWalk_H, pWalk_D),
    (Walk_H, Walk_D),
    (For_H, For_D),
    (logFor_H, logFor_D),
    (Comm_H, Comm_D),
    (logComm_H, logComm_D),
    (Heat_H, Heat_D),
    (logHeat_H, logHeat_D),
    (SCT_H, SCT_D),
    (SCCT_H, SCCT_D),
    (RSP_K, RSP_D),
    (FE_K, FE_D),
    (logPPR_H, logPPR_D),
    (ModifPPR_H, ModifPPR_D),
    (logModifPPR_H, logModifPPR_D),
    (HeatPPR_H, HeatPPR_D),
    (logHeatPPR_H, logHeatPPR_D),
    (SPCT_H, SPCT_D),
    (SP_K, SP_D),
    (CT_H, CT_D)
]

all_measures = [x[0] for x in distances_kernels_pairs]
all_distances = [x[1] for x in distances_kernels_pairs]

result_params = {
    (100, 2, 0.3, 0.05): {
        "CT H": 0.00,
        "SP K": 0.00,
        "Comm H": 0.42,
        "logComm H": 0.46,
        "Heat H": 0.70,
        "Walk H": 0.82,
        "logHeat H": 0.70,
        "SCT H": 0.46,
        "logFor H": 0.72,
        "RSP K": 0.98,
        "For H": 0.96,
        "FE K": 0.96,
        "SCCT H": 0.98,
        "pWalk H": 0.86,
        "SP-CT H": 0.00
    },
    (100, 2, 0.3, 0.1): {
        "CT H": 0.00,
        "SP K": 0.00,
        "Comm H": 0.36,
        "logComm H": 0.54,
        "Heat H": 0.74,
        "Walk H": 0.76,
        "logHeat H": 0.46,
        "SCT H": 0.50,
        "logFor H": 0.40,
        "RSP K": 0.98,
        "For H": 0.98,
        "FE K": 0.92,
        "SCCT H": 0.74,
        "pWalk H": 0.80,
        "SP-CT H": 0.04
    },
    (100, 2, 0.3, 0.15): {
        "CT H": 0.00,
        "SP K": 0.00,
        "Comm H": 0.24,
        "logComm H": 0.64,
        "Heat H": 0.82,
        "Walk H": 0.76,
        "logHeat H": 0.18,
        "SCT H": 0.48,
        "logFor H": 0.28,
        "RSP K": 0.98,
        "For H": 0.44,
        "FE K": 0.76,
        "SCCT H": 0.44,
        "pWalk H": 0.86,
        "SP-CT H": 0.36
    }
}


def _draw_one_by_one(results_rc, out_name='results/4_one_by_one.png'):
    fig, ax = plt.subplots(len(all_measures), 3, figsize=(12, 50), sharex=True, sharey=True)
    for column_idx, column in enumerate(results_rc.keys()):
        for measure_name_idx, measure in enumerate(all_measures):
            measure_name = measure.name[:-2]
            axi = ax[measure_name_idx][column_idx]
            for graph_idx, (tpr, fpr) in enumerate(results_rc[column][measure_name]):
                axi.plot(tpr, fpr, color='black', alpha=0.1)
            axi.set_title("G({}, ({}){}, {}), {}".format(*column, measure_name))
            axi.set_xlim(0, 1)
            axi.set_ylim(0, 1)
    plt.savefig(out_name)


def _draw_pout01(results_rc, out_name='4_100(2)_0.3_0.1.png'):
    fig, ax = plt.subplots(5, 4, figsize=(16, 16), sharex=True, sharey=True)
    column = list(results_rc.keys())[1]
    for measure_name_idx, measure in enumerate(all_measures):
        measure_name = measure.name[:-2]
        axi = ax[measure_name_idx // 4][measure_name_idx % 4]
        for graph_idx, (tpr, fpr) in enumerate(results_rc[column][measure_name]):
            axi.plot(tpr, fpr, color='black', alpha=0.1)
        axi.set_title("G({}, ({}){}, {}), {}".format(*column, measure_name))
        axi.set_xlim(0, 1)
        axi.set_ylim(0, 1)
    plt.savefig(out_name)


def _draw_all(results_rc, out_name='results/4_all.png'):
    fig, axi = plt.subplots(1, figsize=(5, 4))
    for measure_name_idx, measure in enumerate(all_measures):
        measure_name = measure.name[:-2]
        tpr_all = defaultdict(list)
        for graph_idx, (tpr, fpr) in enumerate(results_rc[(100, 2, 0.3, 0.10)][measure_name]):
            tprg = defaultdict(list)
            for ti, fi in zip(fpr, tpr):
                tprg[np.floor(fi * 100)].append(ti)
            for bucket, fis in tprg.items():
                tpr_all[bucket].append(np.mean(fis))
        for bucket, fis in tpr_all.items():
            tpr_all[bucket] = np.mean(fis)

        axi.plot(np.array(list(tpr_all.keys()), dtype=np.float) / 100, tpr_all.values(),
                 label=measure_name, color=d3_colors[measure_name])
    axi.set_xlabel("nodes from different classes")
    axi.set_ylabel("nodes from the same class")

    box = axi.get_position()
    axi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name)


def _draw_4best(results_rc, out_name='results/4_4best.png'):
    fig, axi = plt.subplots(1, figsize=(5, 4))
    for measure_name_idx, measure in enumerate(all_measures):
        measure_name = measure.name[:-2]
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

        axi.plot(np.array(list(tpr_all.keys()), dtype=np.float) / 100, tpr_all.values(),
                 label=measure_name, color=d3_colors[measure_name])
    axi.set_xlabel("nodes from different classes")
    axi.set_ylabel("nodes from the same class")

    box = axi.get_position()
    axi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name)


def _calc_rc(recalc=False):
    columns = [
        # n_nodes, n_classes, p_in, p_out
        (100, 2, 0.3, 0.05),
        (100, 2, 0.3, 0.1),
        (100, 2, 0.3, 0.15),
    ]
    rc = RejectCurve(columns, all_distances, StochasticBlockModel)
    if recalc:
        rc.calc_best_params(all_measures, KKMeans, 100)
    else:
        rc.set_best_params(result_params)
    return rc.perform(100)


def calc_part4():
    results_rc = _calc_rc(recalc=True)
    _draw_one_by_one(results_rc)  # draw all rc
    _draw_pout01(results_rc)  # draw (100, 2, 0.3, 0.1) for all measures
    _draw_all(results_rc)  # draw all in one pic
    _draw_4best(results_rc)  # draw 4 best measures


if __name__ == '__main__':
    calc_part4()
