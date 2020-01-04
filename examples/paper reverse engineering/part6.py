import os
import sys
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt
import logging

from pygraphs.graphs.dataset import Datasets
from pygraphs.measure import kernels
from pygraphs.scenario import d3_colors
from pygraphs.util import configure_logging
from _datasets_kkmeans import datasets_kkmeans

configure_logging()
logger = logging.getLogger()


def calc_part6(n_jobs=6):
    results = datasets_kkmeans(n_graphs=0, n_jobs=n_jobs)

    datasets = Datasets().all

    for dataset_name, measure_results in results.items():
        print(dataset_name)
        for measure_name in measure_results.keys():
            x, y = measure_results[measure_name]
            plt.plot(range(len(y)), sorted(y, reverse=True), color=d3_colors[measure_name])
        plt.xlim(0, 50)
        plt.ylim(0, 1)
        plt.show()

    print('', end="\t")
    for dataset in datasets:
        dataset_name = dataset[1]['name']
        print(dataset_name, end="\t")
    print()
    for kernel in kernels:
        kernel_name = kernel.name
        print(kernel_name, end="\t")
        for dataset in datasets:
            dataset_name = dataset[1]['name']
            try:
                measure_results = np.max(results[dataset_name][kernel_name][1])
            except:
                measure_results = '-'
            print('{}\t'.format(measure_results), end=" ")
        print()


if __name__ == '__main__':
    calc_part6()
