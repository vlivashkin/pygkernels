import sys
import warnings
from collections import defaultdict
from itertools import product

from scipy import stats
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from _generated_kward import generated_kward
from pygraphs.cluster import KWard
from pygraphs.graphs.generator import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.score import copeland


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


def _print_params_kward(results, features, filename):
    print(f'_print_params_kward: features={features}')

    columns = list(results.keys())
    features = features if type(features) == list else [features]

    with open(filename, 'w') as f:
        f.write('\t' + (''.join(['\t'] * len(features))).join([str(x) for x in columns]) + '\n')
        f.write('\t' + '\t'.join(features * len(columns)) + '\n')

        for kernel in kernels:
            values = [kernel.name] + [f'{results[column][kernel.name][feature]:.2f}'
                                      for column, feature in product(columns, features)]
            f.write('\t'.join(values) + '\n')


def _calc_competitions(estimator, best_params, n_graphs=600):
    """
    Calc competition for given params
    :return dict[column][kernel] -> score
    """

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

    results = defaultdict(lambda: defaultdict(lambda: 0))
    for column in tqdm(columns):
        n_nodes, n_classes, p_in, p_out = column
        graphs, info = StochasticBlockModel(n_nodes, n_classes, p_in=p_in, p_out=p_out).generate_graphs(n_graphs * 2)
        success = 0
        for edges, nodes in tqdm(graphs, desc=str(column), total=n_graphs):
            try:
                single_competition_best = {}
                for kernel_class in kernels:
                    best_param = best_params[column][kernel_class.name]
                    kernel = kernel_class(edges)
                    param = kernel.scaler.scale(best_param)
                    K = kernel.get_K(param)
                    y_pred = estimator(n_classes).fit_predict(K)
                    ari = adjusted_rand_score(nodes, y_pred)
                    single_competition_best[kernel_class.name] = ari
                single_competition_score = copeland(single_competition_best.items())
                for kernel_name, delta in single_competition_score.items():
                    results[column][kernel_name] += delta
                    results['sum'][kernel_name] += delta
                success += 1
            except Exception or FloatingPointError as e:
                print(e)
            if success == n_graphs:
                break
    return results


def calc_competitions(estimator, classic_plots, n_graphs_inference):
    best_feature, percentile_feature = 'best_param', 'percentile_param'
    columns = list(classic_plots.keys())

    best_params, percentile_params = defaultdict(dict), defaultdict(dict)
    for column, kernel in product(columns, kernels):
        best_params[column][kernel.name] = classic_plots[column][kernel.name][best_feature]
        percentile_params[column][kernel.name] = classic_plots[column][kernel.name][
            # TODO: temp crutch for typo
            percentile_feature if percentile_feature in classic_plots[column][kernel.name] else 'precentile_param'
        ]

    # best
    print(f'calc_competitions, best: estimator={estimator.name}')
    results = _calc_competitions(estimator, best_params, n_graphs=n_graphs_inference)
    _print_results(results, f'./results/p3-{estimator.name}-competitions_best.tsv')

    # percentile
    print(f'calc_competitions, percentile: estimator={estimator.name}')
    results = _calc_competitions(estimator, percentile_params, n_graphs=n_graphs_inference)
    _print_results(results, f'./results/p3-{estimator.name}-competitions_percentile.tsv')


def _print_results(results, filename):
    columns_order = [
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
        (200, 4, 0.3, 0.15),
        'sum'
    ]

    mr_transposed = {}
    for column in columns_order:
        scores = [results[column][x.name] for x in kernels]
        ranks = stats.rankdata([-results[column][x.name] for x in kernels], 'min')
        mr_transposed[column] = dict(zip(kernels, zip(scores, ranks)))

    with open(filename, 'w') as f:
        f.write('\t' + '\t\t'.join([str(x) for x in columns_order]) + '\n')
        f.write('\t' + '\t'.join(['score', 'rank'] * len(columns_order)) + '\n')
        for idx, kernel in enumerate(kernels):
            f.write(kernel.name + '\t' + '\t'.join(['\t'.join([str(x) for x in mr_transposed[col_name][kernel]])
                                                    for col_name in columns_order]) + '\n')


def calc_part3(n_graphs_train=100, n_graphs_inference=500, n_jobs=1):
    # classic_plots: [column][kernel_name][init][feature]
    cache_kward = generated_kward(n_graphs=n_graphs_train, n_jobs=n_jobs)

    print('CALC COMPETITIONS KWARD')
    calc_competitions(KWard, cache_kward, n_graphs_inference)


if __name__ == '__main__':
    calc_part3(n_graphs_inference=500)
