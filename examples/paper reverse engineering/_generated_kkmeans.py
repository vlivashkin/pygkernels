import os
import sys
import warnings
from functools import partial

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score

from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.scenario import ParallelByGraphs
from pygraphs.util import load_or_calc_and_save


def _calc_best_params(column, init, n_graphs=100, n_params=31, n_jobs=-1):
    """
    Find classic plot, best params and 95% percentile
    """

    n_nodes, n_classes, p_in, p_out = column
    graphs, _ = StochasticBlockModel(n_nodes, n_classes, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False, verbose=True)

    best_params = {}
    for measure_class in tqdm(kernels, desc=str(column)):
        estimator = partial(KKMeans, init=init)
        x, y, error = classic_plot.perform(estimator, measure_class, graphs, n_classes, n_jobs=n_jobs)
        best_idx = np.argmax(y)
        percentile_idx = list(y).index(np.percentile(y, 90, interpolation='lower'))

        print(f'{column} {measure_class.name} {init}')
        print(f'\tbest param: {x[best_idx]:0.2f}\tari: {y[best_idx]:0.2f}')
        print(f'\tperc param: {x[percentile_idx]:0.2f}\tari: {y[percentile_idx]:0.2f}')

        best_params[measure_class.name] = {
            'x': x, 'y': y, 'error': error,
            'best_param': x[best_idx], 'best_ari': y[best_idx],
            'percentile_param': x[percentile_idx], 'percentile_ari': y[percentile_idx],
        }
    return best_params


def _column(column, init, n_graphs=100, n_params=31, n_jobs=-1):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.1f}_{p_out:.2f}'

    @load_or_calc_and_save(f'cache/generated_kkmeans/{column_str}_{init}.pkl')
    def _calc(n_graphs=100, n_params=31, n_jobs=-1):
        return _calc_best_params(column, init, n_graphs, n_params, n_jobs)

    return _calc(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)


def generated_kkmeans(n_graphs=200, n_params=101, n_jobs=1):
    columns = [
        (100, 2, 0.2, 0.05),
        (100, 2, 0.3, 0.05),
        (100, 2, 0.3, 0.10),
        (100, 2, 0.3, 0.15),
        (102, 3, 0.3, 0.10),
        (100, 4, 0.3, 0.10),
        (100, 4, 0.3, 0.15),
        (200, 2, 0.3, 0.05),
        (200, 2, 0.3, 0.10),
        (200, 2, 0.3, 0.15),
        (201, 3, 0.3, 0.10),
        (200, 4, 0.3, 0.10),
        (200, 4, 0.3, 0.15)
    ]
    init = 'random'
    params = {'n_graphs': n_graphs, 'n_params': n_params, 'n_jobs': n_jobs}
    return dict([(column, _column(column, init, **params)) for column in columns])


if __name__ == '__main__':
    generated_kkmeans()
