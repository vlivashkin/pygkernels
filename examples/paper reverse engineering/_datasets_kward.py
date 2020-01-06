import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

from tqdm import tqdm
import logging
from sklearn.metrics import adjusted_rand_score

from pygraphs.graphs.dataset import Datasets
from pygraphs.measure import kernels
from pygraphs.cluster import KWard
from pygraphs.scenario import ParallelByGraphs
from pygraphs.util import load_or_calc_and_save, configure_logging

configure_logging()
logger = logging.getLogger()


def _calc_best_params(dataset, n_params, n_jobs):
    dataset_results = {}
    graphs, info = dataset
    print(info)

    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False)
    for measure_class in tqdm(kernels, desc=info['name']):
        x, y, error = classic_plot.perform(KWard, measure_class, graphs * 10, info['k'], n_jobs=n_jobs)
        dataset_results[measure_class.name] = {
            'x': x, 'y': y, 'error': error
        }
    print(f'COMPLETED {info["name"]}')
    return info['name'], dataset_results


@load_or_calc_and_save('cache/datasets_kward/dolphins.pkl')
def _dataset_dolphins(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['dolphins'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/eucore.pkl')
def _dataset_eucore(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['eu-core'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/football.pkl')
def _dataset_football(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['football'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/football_old.pkl')
def _dataset_football_old(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['football_old'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/karate.pkl')
def _dataset_karate(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['karate'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_2cl_1.pkl')
def _dataset_news_2cl_1(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_2cl_2.pkl')
def _dataset_news_2cl_2(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_2cl_3.pkl')
def _dataset_news_2cl_3(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_3cl_1.pkl')
def _dataset_news_3cl_1(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_3cl_2.pkl')
def _dataset_news_3cl_2(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_3cl_3.pkl')
def _dataset_news_3cl_3(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_5cl_1.pkl')
def _dataset_news_5cl_1(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_5cl_2.pkl')
def _dataset_news_5cl_2(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/news_5cl_3.pkl')
def _dataset_news_5cl_3(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['news_2cl_1'], n_params, n_jobs=n_jobs)


@load_or_calc_and_save('cache/datasets_kward/polbooks.pkl')
def _dataset_polbooks(n_graphs=None, n_params=101, n_jobs=-1):
    return _calc_best_params(Datasets()['polbooks'], n_params, n_jobs=n_jobs)


def datasets_kward(n_params=101, n_jobs=6):
    params = {'n_graphs': None, 'n_params': n_params, 'n_jobs': n_jobs}
    return {
        'dolphins': _dataset_dolphins(**params),
        'eu-core': _dataset_eucore(**params),
        'football': _dataset_football(**params),
        'football_old': _dataset_football_old(**params),
        'karate': _dataset_karate(**params),
        'news_2cl_1': _dataset_news_2cl_1(**params),
        'news_2cl_2': _dataset_news_2cl_2(**params),
        'news_2cl_3': _dataset_news_2cl_3(**params),
        'news_3cl_1': _dataset_news_3cl_1(**params),
        'news_3cl_2': _dataset_news_3cl_2(**params),
        'news_3cl_3': _dataset_news_3cl_3(**params),
        'news_5cl_1': _dataset_news_5cl_1(**params),
        'news_5cl_2': _dataset_news_5cl_2(**params),
        'news_5cl_3': _dataset_news_5cl_3(**params),
        'polbooks': _dataset_polbooks(**params)
    }


if __name__ == '__main__':
    datasets_kward()
