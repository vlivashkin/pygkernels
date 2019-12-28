import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from joblib import Parallel, delayed

from pygraphs.graphs.dataset import Datasets
from pygraphs.measure import kernels
from pygraphs.cluster import KKMeans_iterative as KKMeans
from pygraphs.scenario import ParallelByGraphs, d3_colors
from pygraphs.util import load_or_calc_and_save, configure_logging

import logging

configure_logging()
logger = logging.getLogger()

all_datasets = Datasets().all


@load_or_calc_and_save('cache/6_1b_6.pkl')
def _calc(n_graphs=0, n_jobs=6):
    classic_plot = ParallelByGraphs(adjusted_rand_score, np.linspace(0, 1, 51), progressbar=False)

    def perform(classic_plot, dataset):
        dataset_results = {}
        graphs, info = dataset
        print(info)
        for measure_class in tqdm(kernels, desc=info['name']):
            x, y, error = classic_plot.perform(KKMeans, measure_class, graphs, info['k'], n_jobs=1)
            dataset_results[measure_class.name] = (x, y)
        print(f'COMPLETED {info["name"]}')
        return info['name'], dataset_results

    return dict(Parallel(n_jobs=n_jobs)(delayed(perform)(classic_plot, dataset) for dataset in all_datasets))


def calc_part6(n_jobs=6):
    results = _calc(n_graphs=0, n_jobs=n_jobs)

    for dataset_name, measure_results in results.items():
        print(dataset_name)
        for measure_name in measure_results.keys():
            x, y = measure_results[measure_name]
            plt.plot(range(len(y)), sorted(y, reverse=True), color=d3_colors[measure_name])
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        plt.show()

    print('', end="\t")
    for dataset in all_datasets:
        dataset_name = dataset[1]['name']
        print(dataset_name, end="\t")
    print()
    for kernel in kernels:
        kernel_name = kernel.name
        print(kernel_name, end="\t")
        for dataset in all_datasets:
            dataset_name = dataset[1]['name']
            try:
                measure_results = np.max(results[dataset_name][kernel_name][1])
            except:
                measure_results = '-'
            print('{}\t'.format(measure_results), end=" ")
        print()


if __name__ == '__main__':
    calc_part6()
