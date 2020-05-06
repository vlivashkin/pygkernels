import argparse
import sys
from functools import partial
from typing import Type

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from pygkernels.score import sns1

sys.path.append('../../..')
from pygkernels.cluster import KKMeans
from pygkernels.data import Datasets
from pygkernels.measure import kernels, Kernel
from pygkernels.util import load_or_calc_and_save

CACHE_ROOT = '/media/illusionww/68949C3149F4E819/phd/pygkernels/kkmeans_init_datasets_modularity'
dataset_names = [
    #     'cora_DB',
    #     'cora_EC',
    #     'cora_HA',
    #     'cora_HCI',
    #     'cora_IR',
    #     'cora_Net',
    'dolphins',
    # 'eu-core',
    # 'eurosis',
    'football',
    'karate',
    'news_2cl_1',
    'news_2cl_2',
    'news_2cl_3',
    'news_3cl_1',
    'news_3cl_2',
    'news_3cl_3',
    'news_5cl_1',
    'news_5cl_2',
    'news_5cl_3',
    'polblogs',
    'polbooks',
    'sp_school_day_1',
    'sp_school_day_2'
]


def perform_graph(dataset_name, graph, kernel_class: Type[Kernel], k, root=f'{CACHE_ROOT}/by_column_and_kernel'):
    A, y_true = graph
    kernel: Kernel = kernel_class(A)

    def perform_param(estimator, param_flat):
        param_results = []
        try:
            param = kernel.scaler.scale(param_flat)
            K = kernel.get_K(param)
            inits = estimator.predict(K, explicit=True, A=A)
            for init in inits:
                y_pred = init['labels']
                param_results.append({
                    'labels': y_pred,
                    'inertia': init['inertia'],
                    'modularity': init['modularity'],
                    'init': init['init'],
                    'score_ari': adjusted_rand_score(y_true, y_pred),
                    'score_nmi': normalized_mutual_info_score(y_true, y_pred, average_method='geometric'),
                    'score_sns1': sns1(y_true, y_pred)
                })
        except Exception or ValueError or FloatingPointError or np.linalg.LinAlgError as e:
            print(f'{kernel.name}, p={param_flat}: {e}')
        return param_flat, param_results

    @load_or_calc_and_save(f'{root}/{dataset_name}_{kernel_class.name}_results.pkl', ignore_if_exist=True)
    def _calc(n_graphs=None, n_params=None, n_jobs=None):
        kmeans = partial(KKMeans, n_clusters=k, init='any', n_init=N_INITS, init_measure='modularity')
        results = Parallel(n_jobs=N_JOBS)(
            delayed(perform_param)(kmeans(device=param_flat % N_GPU, random_state=2000 + param_flat), param_flat)
            for param_flat in np.linspace(0, 1, N_PARAMS))
        return dict(results)

    return {
        'results': _calc(n_graphs=None, n_params=None, n_jobs=None),
        'y_true': y_true
    }


def perform():
    for dataset_name in dataset_names:
        graphs, _, info = Datasets()[dataset_name]
        for kernel_class in tqdm(kernels, desc=dataset_name):
            perform_graph(dataset_name, graphs[0], kernel_class, info['k'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=6, required=False)
    parser.add_argument('--n_gpu', type=int, default=2, required=False)
    parser.add_argument('--n_inits', type=int, default=30, required=False)
    parser.add_argument('--n_params', type=int, default=51, required=False)

    args = parser.parse_args()
    print(args)

    N_JOBS = args.n_jobs
    N_GPU = args.n_gpu
    N_INITS = args.n_inits
    N_PARAMS = args.n_params
    perform()
