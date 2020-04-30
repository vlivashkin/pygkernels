import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

sys.path.append('../..')
from pygkernels.cluster import KKMeans
from pygkernels.graphs import Datasets
from pygkernels.measure import kernels
from pygkernels.util import load_or_calc_and_save
import networkx as nx

CACHE_ROOT = '/media/illusionww/68949C3149F4E819/phd/pygkernels/kkmeans_init_datasets_modularity'
dataset_names = [
    'dolphins',
    'football',
    'karate',
    'polbooks',
    'news_2cl_1',
    'news_2cl_2',
    'news_2cl_3',
    'news_3cl_1',
    'news_3cl_2',
    'news_3cl_3',
    'news_5cl_1',
    'news_5cl_2',
    'news_5cl_3',
]


def perform_param(param_flat, graph, kernel_class, estimator):
    A, y_true, G = graph
    kernel = kernel_class(A)

    param_results = []
    try:
        param = kernel.scaler.scale(param_flat)
        K = kernel.get_K(param)
        inits = estimator.predict(K, explicit=True, G=G)
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

    return param_flat, param_results


def perform_graph(graph, kernel_class, k, n_params=51, n_jobs=6, n_gpu=2):
    params = np.linspace(0, 1, n_params)
    graph = graph[0], graph[1], nx.from_numpy_matrix(graph[0])

    results = dict(Parallel(n_jobs=n_jobs)(delayed(perform_param)(
        param_flat, graph, kernel_class,
        KKMeans(k, device=param_idx % n_gpu, random_state=2000 + param_idx, n_init=10)
    ) for param_idx, param_flat in enumerate(params)))

    return results


def perform_kernel(dataset_name, graphs, kernel_class, k, n_params=51, n_jobs=6,
                   root=f'{CACHE_ROOT}/by_column_and_kernel'):
    @load_or_calc_and_save(f'{root}/{dataset_name}_{kernel_class.name}_results.pkl')
    def _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs):
        results = []
        for graph_idx, graph in enumerate(graphs):
            result = perform_graph(graph, kernel_class, k, n_params=n_params)
            results.append(result)
        return results

    return _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs)


def perform_column(dataset_name, graphs, k):
    for kernel_class in tqdm(kernels, desc=dataset_name):
        perform_kernel(dataset_name, graphs, kernel_class, k)


def perform():
    for dataset_name in dataset_names:
        graphs, info = Datasets()[dataset_name]
        perform_column(dataset_name, graphs, info['k'])


if __name__ == '__main__':
    perform()
