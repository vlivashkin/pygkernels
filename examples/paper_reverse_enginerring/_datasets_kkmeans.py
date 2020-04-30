import logging
import sys
import warnings
from functools import partial

from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append('../..')
from pygkernels.cluster import KKMeans as KKMeans
from example_graphs.dataset import Datasets
from pygkernels.measure import kernels
from pygkernels.scenario import ParallelByGraphs
from pygkernels.util import load_or_calc_and_save, configure_logging

configure_logging()
logger = logging.getLogger()


def _calc_best_params(dataset_name, init, n_params, n_jobs):
    dataset_results = {}
    graphs, info = Datasets()[dataset_name]
    print(info)

    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False)
    for measure_class in tqdm(kernels, desc=info['name']):
        x, y, error = classic_plot.perform(partial(KKMeans, init=init), measure_class, graphs * 10, info['k'],
                                           n_jobs=n_jobs)
        dataset_results[measure_class.name] = {
            'x': x, 'y': y, 'error': error
        }
    print(f'COMPLETED {info["name"]}')
    return info['name'], dataset_results


def _dataset(dataset_name, init, n_params=101, n_jobs=-1, root='./cache/datasets_kkmeans'):
    @load_or_calc_and_save(f'{root}/{dataset_name}-{init}.pkl')
    def _calc(n_graphs=None, n_params=101, n_jobs=-1):
        return _calc_best_params(dataset_name, init, n_params, n_jobs=n_jobs)

    return _calc(n_graphs=None, n_params=n_params, n_jobs=n_jobs)


def datasets_kkmeans_any(n_params=51, n_jobs=6):
    datasets = [
        'dolphins',
        'polbooks',
        'football',
        'football_old',
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
        'eu-core',
    ]
    init = 'any'
    return dict([(_dataset(dataset, init, n_params, n_jobs)) for dataset in datasets])


if __name__ == '__main__':
    datasets_kkmeans_any()
