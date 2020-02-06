import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append('../..')
from _generated_kkmeans import generated_kkmeans_any
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import *
from pygraphs.scenario import RejectCurve, d3_colors
from pygraphs.util import load_or_calc_and_save

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
    ('PPR', PPR_H, PPR_D),
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


@load_or_calc_and_save('cache/p4_rc.pkl')
def _calc(n_graphs=100, n_params=None, n_jobs=None):
    cache_kkmeans, init = generated_kkmeans_any(), 'k-means++'
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
    return results_rc


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
        axi.set_title(measure_name)
        axi.set_xlim(0, 1)
        axi.set_ylim(0, 1)
    plt.savefig(out_name, bbox_inches='tight')


def _draw_avg(tpr_all, out_name, allowed_measures=None):
    print(f'_draw_avg: out_name={out_name}, allowed_measures={allowed_measures}')

    fig, axi = plt.subplots(1, figsize=(5, 4))
    for measure_name in all_names:
        if allowed_measures is not None and measure_name not in allowed_measures:
            continue
        tpr_measure = tpr_all[measure_name]
        axi.plot(np.array(list(tpr_measure.keys()), dtype=np.float) / 100, list(tpr_measure.values()),
                 label=measure_name, color=d3_colors[measure_name])
    axi.set_xlabel("nodes from different classes")
    axi.set_ylabel("nodes from the same class")

    box = axi.get_position()
    axi.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name, bbox_inches='tight')


def _print_avg_results(tpr_all, filename):
    with open(filename, 'w') as f:
        for measure_name in all_names:
            tpr_measure = tpr_all[measure_name]
            auc = np.mean(list(tpr_measure.values()))
            f.write(f'{measure_name}\t{auc:0.4f}\n')


def calc_part4(n_graphs=100):
    results_rc = _calc(n_graphs=n_graphs, n_params=None, n_jobs=None)

    _draw_one_by_one(results_rc, './results/p4-one_by_one.png')  # draw all rc
    _draw_g100_2_03_01(results_rc, './results/p4-g100_2_0.3_0.1.png')  # draw (100, 2, 0.3, 0.1) for all measures

    tpr_all = defaultdict(dict)
    for measure_name in all_names:
        measure_results = results_rc[(100, 2, 0.3, 0.10)][measure_name]

        tpr_measure = []
        for graph_idx, (tpr, fpr) in enumerate(measure_results):
            tpr_graph = defaultdict(list)
            for ti, fi in zip(fpr, tpr):
                tpr_graph[np.floor(fi * 100)].append(ti)
            tpr_graph = dict([(bucket, np.mean(values)) for bucket, values in tpr_graph.items()])
            tpr_measure.append(tpr_graph)
        for bucket in range(101):
            tpr_all[measure_name][bucket] = np.mean([x[bucket] for x in tpr_measure])

    _draw_avg(tpr_all, './results/p4-avg.png')  # draw all in one pic
    _draw_avg(tpr_all, './results/p4-avg-4best.png', ['SCCT', 'HeatPPR', 'logNHeat', 'logComm'])  # draw 4 best measures
    _print_avg_results(tpr_all, './results/p4-auc.tsv')


if __name__ == '__main__':
    calc_part4(n_graphs=100)
