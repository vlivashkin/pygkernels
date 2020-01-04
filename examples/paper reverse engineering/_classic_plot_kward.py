import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.cluster import KWard
from pygraphs.scenario import ParallelByGraphs
from pygraphs.util import load_or_calc_and_save, ddict2dict


def _calc_best_params(column, n_graphs=100, n_params=31, n_jobs=-1):
    """
    Find classic plot, best params and 95% percentile
    """

    n_nodes, n_classes, p_in, p_out = column
    graphs, _ = StochasticBlockModel(n_nodes, n_classes, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False, verbose=True)

    best_params = defaultdict(dict)
    for measure_class in tqdm(kernels, desc=str(column)):
        x, y, error = classic_plot.perform(KWard, measure_class, graphs, n_classes, n_jobs=n_jobs)
        best_idx = np.argmax(y)
        percentile_idx = list(y).index(np.percentile(y, 90, interpolation='lower'))

        print(f'{column} {measure_class.name}')
        print(f'\tbest param: {x[best_idx]:0.2f}\tari: {y[best_idx]:0.2f}')
        print(f'\tperc param: {x[percentile_idx]:0.2f}\tari: {y[percentile_idx]:0.2f}')

        best_params[measure_class.name] = {
            'x': x, 'y': y, 'error': error,
            'best_param': x[best_idx], 'best_ari': y[best_idx],
            'percentile_param': x[percentile_idx], 'percentile_ari': y[percentile_idx],
        }
    return ddict2dict(best_params)


@load_or_calc_and_save(f'cache/kward_100_2_02_005.pkl')
def _kward_100_2_02_005(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 2, 0.2, 0.05), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_100_2_03_005.pkl')
def _kward_100_2_03_005(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 2, 0.3, 0.05), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_100_2_03_010.pkl')
def _kward_100_2_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 2, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_100_2_03_015.pkl')
def _kward_100_2_03_015(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 2, 0.3, 0.15), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_102_3_03_010.pkl')
def _kward_102_3_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((102, 3, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_100_4_03_010.pkl')
def _kward_100_4_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 4, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_100_4_03_015.pkl')
def _kward_100_4_03_015(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((100, 4, 0.3, 0.15), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_200_2_03_005.pkl')
def _kward_200_2_03_005(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((200, 2, 0.3, 0.05), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_200_2_03_010.pkl')
def _kward_200_2_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((200, 2, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_200_2_03_015.pkl')
def _kward_200_2_03_015(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((200, 2, 0.3, 0.15), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_201_3_03_010.pkl')
def _kward_201_3_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((201, 3, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_200_4_03_010.pkl')
def _kward_200_4_03_010(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((200, 4, 0.3, 0.1), n_graphs, n_params, n_jobs)


@load_or_calc_and_save(f'cache/kward_200_4_03_015.pkl')
def _kward_200_4_03_015(n_graphs=100, n_params=31, n_jobs=-1):
    return _calc_best_params((200, 4, 0.3, 0.15), n_graphs, n_params, n_jobs)


def classic_plots_kward(n_graphs=200, n_params=101, n_jobs=1):
    return {
        (100, 2, 0.2, 0.05): _kward_100_2_02_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (100, 2, 0.3, 0.05): _kward_100_2_03_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (100, 2, 0.3, 0.10): _kward_100_2_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (100, 2, 0.3, 0.15): _kward_100_2_03_015(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (102, 3, 0.3, 0.10): _kward_102_3_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (100, 4, 0.3, 0.10): _kward_100_4_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (100, 4, 0.3, 0.15): _kward_100_4_03_015(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (200, 2, 0.3, 0.05): _kward_200_2_03_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (200, 2, 0.3, 0.10): _kward_200_2_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (200, 2, 0.3, 0.15): _kward_200_2_03_015(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        # (201, 3, 0.3, 0.10): _kward_201_3_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        # (200, 4, 0.3, 0.10): _kward_200_4_03_010(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        # (200, 4, 0.3, 0.15): _kward_200_4_03_015(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)
        (201, 3, 0.3, 0.10): _kward_200_2_03_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (200, 4, 0.3, 0.10): _kward_200_2_03_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs),
        (200, 4, 0.3, 0.15): _kward_200_2_03_005(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)
    }


if __name__ == '__main__':
    classic_plots_kward()
