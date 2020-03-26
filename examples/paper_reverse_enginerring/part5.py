import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.cluster.kkmeans import KKMeans as KKMeans
from pygraphs.scenario import ParallelByGraphs, d3_colors
from pygraphs.util import load_or_calc_and_save, ddict2dict


@load_or_calc_and_save('./cache/p5_2class.pkl')
def _calc5_vary1(n_graphs=200, n_params=101, n_jobs=6):
    results = defaultdict(lambda: defaultdict(list))
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False)
    for first_class in tqdm(sorted([1, 2, 3, 7] + list(range(0, 51, 5)))):
        graphs, info = StochasticBlockModel(100, 2, p_in=0.3, p_out=0.1, cluster_sizes=[first_class, 100 - first_class]) \
            .generate_graphs(n_graphs)
        for measure_class in tqdm(kernels, desc=str(first_class)):
            x, y, error = classic_plot.perform(KKMeans, measure_class, graphs, 2, n_jobs=n_jobs)
            _, best_y = sorted(zip(x, y), key=lambda x: -x[1])[0]
            mean_y = np.mean(y)
            results[measure_class.name][first_class] = (best_y, mean_y)
    return ddict2dict(results)


@load_or_calc_and_save('./cache/p5_six.pkl')
def _calc5_6cl(n_graphs=100, n_params=101, n_jobs=6):
    cluster_sizes = [65, 35, 25, 13, 8, 4]
    probability_matrix = np.array([
        [0.30, 0.20, 0.10, 0.15, 0.07, 0.25],
        [0.20, 0.24, 0.08, 0.13, 0.05, 0.17],
        [0.10, 0.08, 0.16, 0.09, 0.04, 0.12],
        [0.15, 0.13, 0.09, 0.20, 0.02, 0.14],
        [0.07, 0.05, 0.04, 0.02, 0.12, 0.04],
        [0.25, 0.17, 0.12, 0.14, 0.04, 0.40]
    ])
    graphs, info = StochasticBlockModel(150, 6, cluster_sizes=cluster_sizes, probability_matrix=probability_matrix) \
        .generate_graphs(n_graphs)

    results = {}
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=True)
    for measure_class in kernels:
        x, y, error = classic_plot.perform(KKMeans, measure_class, graphs, 2, n_jobs=n_jobs)
        results[measure_class.name] = (x, y, error)
    return results


def _draw5_vary1(results, out_name, allowed_measures=None):
    print(f'_draw5_vary1: out_name={out_name}, allowed_measures={allowed_measures}')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    for measure_class in kernels:
        measure_name = measure_class.name
        if allowed_measures is not None and measure_name not in allowed_measures:
            continue
        measure_result = results[measure_name]
        x, y = zip(*sorted([(x, y[0]) for x, y in measure_result.items()], key=lambda x: x[0]))
        ax.plot(x, y, label=measure_name, color=d3_colors[measure_name])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel('Nodes in the first class')
    ax.set_ylabel('ARI')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(out_name, bbox_inches='tight', dpi=400)
    plt.close(fig)


def _draw5_6cl(results, out_name):
    print(f'_draw5_6cl: out_name={out_name}')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    for measure_class in kernels:
        measure_name = measure_class.name
        x, y, error = results[measure_name]
        ax.plot(range(len(y)), sorted(y, reverse=True), label=measure_name, color=d3_colors[measure_name])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel('param, sorted desc')
    ax.set_ylabel('ARI')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.4)

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(out_name, bbox_inches='tight', dpi=400)
    plt.close(fig)


def calc_part5(n_graphs=200, n_params=101, n_jobs=6):
    # 5.1 Vary first class
    results = _calc5_vary1(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)
    _draw5_vary1(results, out_name='./results/p5-vary1.png')
    _draw5_vary1(results, out_name='./results/p5-vary1-4best.png',
                 allowed_measures=['logComm', 'logNHeat', 'SCCT', 'HeatPPR'])

    # 5.2 6 clusters
    results = _calc5_6cl(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)
    _draw5_6cl(results, out_name='./results/p5-6cl.png')


if __name__ == '__main__':
    calc_part5()
