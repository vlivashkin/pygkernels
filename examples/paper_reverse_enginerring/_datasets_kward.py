import logging
import sys
import warnings
from functools import partial

from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from pygraphs.cluster import KWard
from example_graphs.dataset import Datasets
from pygraphs.measure import kernels
from pygraphs.scenario import ParallelByGraphs
from pygraphs.util import load_or_calc_and_save, configure_logging

configure_logging()
logger = logging.getLogger()


def _calc_one_measure(n_params, estimator_class, measure_class, graphs, info):
    print(f'Start {measure_class}')
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False, verbose=True, ignore_errors=True)
    x, y, error = classic_plot.perform(estimator_class, measure_class, graphs, info['k'], n_jobs=1)
    return measure_class.name, {
        'x': x, 'y': y, 'error': error
    }


def _calc_one_measure_with_cache(n_params, estimator_class, measure_class, graphs, info, root='./cache/datasets_kward/temp',
                                 dataset_name=""):
    @load_or_calc_and_save(f'{root}/{dataset_name}_{measure_class.name}.pkl')
    def _calc(n_graphs=None, n_params=101, n_jobs=-1):
        return _calc_one_measure(n_params, estimator_class, measure_class, graphs, info)

    return _calc(n_graphs=None, n_params=n_params, n_jobs=1)


def _calc_best_params(dataset_name, n_params, n_jobs, backend):
    graphs, info = Datasets()[dataset_name]
    print(info)

    dataset_results = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_calc_one_measure_with_cache)(
        n_params, partial(KWard, backend=backend, device=idx % 2), measure_class, graphs, info,
        dataset_name=dataset_name
    ) for idx, measure_class in enumerate(tqdm(kernels, desc=info['name'])))
    dataset_results = dict(dataset_results)

    print(f'COMPLETED {info["name"]}')
    return info['name'], dataset_results


def _dataset(dataset_name, n_params=101, n_jobs=-1, root='./cache/datasets_kward', backend=''):
    @load_or_calc_and_save(f'{root}/{dataset_name}.pkl')
    def _calc(n_graphs=None, n_params=101, n_jobs=-1):
        return _calc_best_params(dataset_name, n_params, n_jobs=n_jobs, backend=backend)

    return _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs)


def datasets_kward(backend, n_params=101, n_jobs=-1):
    datasets = [
        'dolphins',
        'polbooks',
        'football',
        'football_old',
        'karate',
        'news_2cl_1',
        # 'news_2cl_1_numpy',
        'news_2cl_2',
        'news_2cl_3',
        'news_3cl_1',
        'news_3cl_2',
        'news_3cl_3',
        'news_5cl_1',
        'news_5cl_2',
        'news_5cl_3',
        # 'eu-core',
    ]
    return dict([(dataset, _dataset(dataset, n_params, n_jobs, backend=backend)) for dataset in datasets])


if __name__ == '__main__':
    datasets_kward(backend='pytorch', n_jobs=4)
