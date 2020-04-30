import sys
import warnings
from collections import defaultdict
from functools import partial
from itertools import product

from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from pygkernels.util import load_or_calc_and_save, ddict2dict

warnings.filterwarnings("ignore")
sys.path.append('../..')
from _generated_kkmeans import generated_kkmeans_any
from pygkernels.cluster import KKMeans as KKMeans
from pygkernels.graphs.generator import StochasticBlockModel
from pygkernels.measure import kernels
from pygkernels.score import copeland


def _single_competition(best_params_column, edges, nodes, estimator):
    single_competition_best = {}
    for kernel_class in kernels:
        best_param = best_params_column[kernel_class.name]
        kernel = kernel_class(edges)
        param = kernel.scaler.scale(best_param)
        K = kernel.get_K(param)
        y_pred = estimator().fit_predict(K)
        ari = adjusted_rand_score(nodes, y_pred)
        single_competition_best[kernel_class.name] = ari
    single_competition_score = copeland(single_competition_best.items())
    return ddict2dict(single_competition_score)


def _column(column, init, estimator, best_params, n_graphs=100, n_jobs=-1,
            root='./cache/competitions_kkmeans', best_or_percentile='best'):
    n_nodes, n_clusters, p_in, p_out = column
    column_str = f'{n_nodes}_{n_clusters}_{p_in:.1f}_{p_out:.2f}'

    @load_or_calc_and_save(f'{root}/{column_str}-{init}-{best_or_percentile}.pkl')
    def _calc(n_graphs=100, n_params=None, n_jobs=-1):
        graphs, info = StochasticBlockModel(n_nodes, n_clusters, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
        single_competition_scores = Parallel(n_jobs=n_jobs)(delayed(_single_competition)(
            best_params[column], edges, nodes,
            partial(estimator, n_clusters=n_clusters, init=init, device=graph_idx % 2)
        ) for graph_idx, (edges, nodes) in enumerate(tqdm(graphs, desc=str(column), total=n_graphs)))
        return single_competition_scores

    return _calc(n_graphs=n_graphs, n_params=None, n_jobs=n_jobs)


def _calc_competitions(estimator, best_params, n_graphs=600, n_jobs=6, best_or_percentile='best'):
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
        single_competition_scores = _column(column, 'any', estimator, best_params, n_graphs=n_graphs, n_jobs=n_jobs,
                                            best_or_percentile=best_or_percentile)
        for single_competition_score in single_competition_scores:
            for kernel_name, delta in single_competition_score.items():
                results[column][kernel_name] += delta
                results['sum'][kernel_name] += delta
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
    print(f'calc_competitions, best: estimator={estimator(None).name}')
    results = _calc_competitions(estimator, best_params, n_graphs=n_graphs_inference, best_or_percentile='best')
    _print_results(results, f'./results/p3-{estimator(None).name}-competitions_best.tsv')

    # percentile
    print(f'calc_competitions, percentile: estimator={estimator(None).name}')
    results = _calc_competitions(estimator, percentile_params, n_graphs=n_graphs_inference,
                                 best_or_percentile='percentile')
    _print_results(results, f'./results/p3-{estimator(None).name}-competitions_percentile.tsv')


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
    cache_kkmeans = generated_kkmeans_any(n_graphs=n_graphs_train, n_jobs=n_jobs)

    print('CALC COMPETITIONS KKMEANS')
    kkmeans_plots = defaultdict(lambda: defaultdict(dict))  # choose k-means++ init
    for column in cache_kkmeans.keys():
        for kernel_name in cache_kkmeans[column].keys():
            kkmeans_plots[column][kernel_name] = cache_kkmeans[column][kernel_name]  # [init]
    calc_competitions(KKMeans, kkmeans_plots, n_graphs_inference)


if __name__ == '__main__':
    calc_part3(n_graphs_inference=500, n_jobs=6)
