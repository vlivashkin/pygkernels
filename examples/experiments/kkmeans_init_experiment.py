import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

sys.path.append('../..')
from pygraphs.cluster import KKMeans
from pygraphs.graphs import StochasticBlockModel
from pygraphs.measure import kernels
from pygraphs.util import load_or_calc_and_save

CACHE_ROOT = '/media/illusionww/68949C3149F4E819/pygraphs/kkmeans_init_experiments2'
columns = [
    (100, 2, 0.2, 0.05),
    (100, 2, 0.3, 0.05),
    (100, 2, 0.3, 0.1),
    (100, 2, 0.3, 0.15),
    (102, 3, 0.3, 0.1),
    (100, 4, 0.3, 0.1),
    (100, 4, 0.3, 0.15),
    (200, 2, 0.3, 0.05),
    (200, 2, 0.3, 0.1),
    (200, 2, 0.3, 0.15),
    (201, 3, 0.3, 0.1),
    (200, 4, 0.3, 0.1),
    (200, 4, 0.3, 0.15)
]


def generate_graphs(column, n_graphs, root=f'{CACHE_ROOT}/graphs'):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.1f}_{p_out:.2f}'

    @load_or_calc_and_save(f'{root}/{column_str}_{n_graphs}_graphs.pkl')
    def _calc(n_graphs=n_graphs, n_params=None, n_jobs=None):
        graphs, _ = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
        return graphs

    return _calc(n_graphs=n_graphs, n_params=None, n_jobs=None)


def perform_graph(graph, kernel_class, estimator, n_params=51):
    edges, y_true = graph
    kernel = kernel_class(edges)

    results = {}
    for param_flat in np.linspace(0, 1, n_params):
        param_results = []
        try:
            param = kernel.scaler.scale(param_flat)
            K = kernel.get_K(param)
            inits = estimator.predict_explicit(K)
            for init in inits:
                y_pred = init['labels']
                param_results.append({
                    'labels': y_pred,
                    'inertia': init['inertia'],
                    'init': init['init'],
                    'score_ari': adjusted_rand_score(y_true, y_pred),
                    'score_nmi': normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
                })
        except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError:
            pass
        results[param_flat] = param_results
    return results


def perform_kernel(column, graphs, kernel_class, n_params=51, n_jobs=6, n_gpu=2,
                   root=f'{CACHE_ROOT}/by_column_and_kernel'):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.1f}_{p_out:.2f}'

    @load_or_calc_and_save(f'{root}/{column_str}_{kernel_class.name}_results.pkl')
    def _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs):
        return Parallel(n_jobs=n_jobs)(delayed(perform_graph)(
            graph, kernel_class, KKMeans(k, device=graph_idx % n_gpu, random_state=2000 + graph_idx, n_init=10),
            n_params=n_params
        ) for graph_idx, graph in enumerate(graphs))

    return _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs)


def perform_column(column, graphs):
    n, k, p_in, p_out = column
    column_str = f'{n}_{k}_{p_in:.1f}_{p_out:.2f}'
    for kernel_class in tqdm(kernels, desc=column_str):
        perform_kernel(column, graphs, kernel_class)


def perform(n_graphs=100):
    for column in columns:
        graphs = generate_graphs(column, n_graphs)
        perform_column(column, graphs)


if __name__ == '__main__':
    perform()
