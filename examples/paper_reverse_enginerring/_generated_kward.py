import sys
import warnings
from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from pygkernels.cluster import KWard
from pygkernels.graphs.generator import StochasticBlockModel
from pygkernels.measure import kernels
from pygkernels.scenario import ParallelByGraphs
from pygkernels.util import load_or_calc_and_save, ddict2dict


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


def _column(column, n_graphs=100, n_params=31, n_jobs=-1, root='./cache/generated_kward'):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.1f}_{p_out:.2f}'

    @load_or_calc_and_save(f'{root}/{column_str}.pkl')
    def _calc(n_graphs=100, n_params=31, n_jobs=-1):
        return _calc_best_params(column, n_graphs, n_params, n_jobs)

    return _calc(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)


def _print_params_kward(results, features, filename, transform_params=False, append=False):
    print(f'_print_params_kward: features={features}')

    columns = list(results.keys())
    features = features if type(features) == list else [features]

    with open(filename, 'a' if append else 'w') as f:
        if append:
            f.write('\n')

        f.write('\t' + (''.join(['\t'] * len(features))).join([str(x) for x in columns]) + '\n')
        f.write('\t' + '\t'.join(features * len(columns)) + '\n')

        for kernel in kernels:
            kernel_object = kernel(np.eye(100))
            values = [kernel.name]
            for column, feature in product(columns, features):
                v = results[column][kernel.name][feature]
                values += [f'{kernel_object.scaler.scale(v):.2e}'] if transform_params else [f'{v:.2f}']
            f.write('\t'.join(values) + '\n')


def generated_kward(n_graphs=200, n_params=101, n_jobs=1):
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
    params = {'n_graphs': n_graphs, 'n_params': n_params, 'n_jobs': n_jobs}
    cache_kward = dict([(column, _column(column, **params)) for column in columns])
    return cache_kward


if __name__ == '__main__':
    cache_kward = generated_kward()

    print('SAVE PARAMS KWARD')
    best_fn, p90_fn = 'results/p0-KWard-params_best.tsv', 'results/p0-KWard-params_90p.tsv'
    _print_params_kward(cache_kward, 'best_param', filename=best_fn)
    _print_params_kward(cache_kward, 'best_param', filename=best_fn, transform_params=True, append=True)
    _print_params_kward(cache_kward, 'best_ari', filename=best_fn, append=True)
    _print_params_kward(cache_kward, 'percentile_param', filename=p90_fn)
    _print_params_kward(cache_kward, 'percentile_param', filename=p90_fn, transform_params=True, append=True)
    _print_params_kward(cache_kward, 'percentile_ari', filename=p90_fn, append=True)
