import sys
import warnings
from functools import partial
from itertools import product

import numpy as np
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
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


def _print_params_kkmeans(results, inits, features, filename, append=False):
    print(f'_print_params_kkmeans: inits={inits}, features={features}')

    columns = list(results.keys())
    inits = inits if type(inits) == list else [inits]
    features = features if type(features) == list else [features]

    with open(filename, 'a' if append else 'w') as f:
        f.write('column\t' + (''.join(['\t'] * len(inits) * len(features))).join([str(x) for x in columns]) + '\n')
        f.write('init\t' + (''.join(['\t'] * len(features))).join(inits * len(columns)) + '\n')
        f.write('\t' + '\t'.join(features * len(columns) * len(inits)) + '\n')

        for kernel in kernels:
            # TODO: temp crutch for typo
            # values = [kernel.name] + [f'{results[column][kernel.name][init][feature]:.2f}'
            #                           for column, init, feature in product(columns, inits, features)]
            values = [kernel.name]
            for column, init, feature in product(columns, inits, features):
                keys = results[column][kernel.name].keys()  # [init]
                if feature not in keys:  # check typo
                    if feature == 'percentile_param':
                        feature = 'precentile_param'
                    elif feature == 'percentile_ari':
                        feature = 'precentile_ari'
                values += [f'{results[column][kernel.name][feature]:.2f}']  # [init]
            f.write('\t'.join(values) + '\n')


# def generated_kkmeans(n_graphs=200, n_params=101, n_jobs=1):
#     columns = [
#         (100, 2, 0.2, 0.05),
#         (100, 2, 0.3, 0.05),
#         (100, 2, 0.3, 0.10),
#         (100, 2, 0.3, 0.15),
#         (102, 3, 0.3, 0.10),
#         (100, 4, 0.3, 0.10),
#         (100, 4, 0.3, 0.15),
#         (200, 2, 0.3, 0.05),
#         (200, 2, 0.3, 0.10),
#         (200, 2, 0.3, 0.15),
#         (201, 3, 0.3, 0.10),
#         (200, 4, 0.3, 0.10),
#         (200, 4, 0.3, 0.15)
#     ]
#     init = ['one', 'all', 'k-means++']
#     params = {'n_graphs': n_graphs, 'n_params': n_params, 'n_jobs': n_jobs}
#     return dict([(column, _column(column, init, **params)) for column in columns])


def generated_kkmeans_any(n_graphs=200, n_params=101, n_jobs=1):
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
    init = 'any'
    params = {'n_graphs': n_graphs, 'n_params': n_params, 'n_jobs': n_jobs}
    cache_kkmeans = dict([(column, _column(column, init, **params)) for column in columns])

    print('SAVE PARAMS KKMEANS')
    # _print_params_kkmeans(results, ['one', 'all', 'k-means++'], ['best_param', 'best_ari'])
    _print_params_kkmeans(cache_kkmeans, init, 'best_ari',
                          filename='results/p3-KKMeans-params_best.tsv')
    _print_params_kkmeans(cache_kkmeans, init, 'percentile_ari',
                          filename='results/p3-KKMeans-params_90p.tsv')

    return cache_kkmeans


if __name__ == '__main__':
    # generated_kkmeans()
    generated_kkmeans_any()
