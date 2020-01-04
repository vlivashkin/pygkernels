import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed

from pygraphs.graphs.dataset import Datasets
from pygraphs.measure import kernels
from pygraphs.cluster import KKMeans_vanilla as KKMeans
from pygraphs.scenario import ParallelByGraphs
from pygraphs.util import load_or_calc_and_save, configure_logging

import logging

configure_logging()
logger = logging.getLogger()


@load_or_calc_and_save('cache/kkmeans_datasets.pkl')
def datasets_kkmeans(n_graphs=None, n_params=101, n_jobs=6):
    classic_plot = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=False)

    def perform(classic_plot, dataset):
        dataset_results = {}
        graphs, info = dataset
        print(info)
        for measure_class in tqdm(kernels, desc=info['name']):
            x, y, error = classic_plot.perform(KKMeans, measure_class, graphs, info['k'], n_jobs=1)
            dataset_results[measure_class.name] = (x, y)
        print(f'COMPLETED {info["name"]}')
        return info['name'], dataset_results

    return dict(Parallel(n_jobs=n_jobs)(delayed(perform)(classic_plot, dataset) for dataset in Datasets().all))


if __name__ == '__main__':
    datasets_kkmeans()
