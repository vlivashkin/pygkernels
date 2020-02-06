import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
sys.path.append('../..')
from _datasets_kward import datasets_kward
from pygraphs.measure import kernels
from pygraphs.scenario import d3_colors
from pygraphs.util import configure_logging

configure_logging()
logger = logging.getLogger()

dataset_names = [
    'dolphins',
    '',
    # 'eu-core',
    'football',
    'karate',
    'polbooks',
    '',
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
]


def _plot_sorted(results, img_path):
    print(f'_plot_sorted: filename={img_path}')

    fig, ax = plt.subplots(5, 3, figsize=(10, 12), sharex=True, sharey=True)
    for idx, dataset_name in enumerate(dataset_names):
        if dataset_name == '':
            continue
        dataset_results = results[dataset_name]
        for measure_name, measure_results in dataset_results[1].items():
            x, y = measure_results['x'], measure_results['y']
            ax[idx // 3][idx % 3].plot(range(len(y)), sorted(y, reverse=True), color=d3_colors[measure_name])
        ax[idx // 3][idx % 3].set_xlim(0, 101)
        ax[idx // 3][idx % 3].set_ylim(0, 1)
        if idx // 3 == 4:
            ax[idx // 3][idx % 3].set_xlabel('param, sorted desc')
        if idx % 3 == 0:
            ax[idx // 3][idx % 3].set_ylabel('ARI')
        ax[idx // 3][idx % 3].set_title(dataset_name)
    plt.savefig(img_path, bbox_inches='tight')


def _print_results(results, filename):
    print(f'_print_results: filename={filename}')

    with open(filename, 'w') as f:
        f.write('\t' + '\t'.join(dataset_names) + '\n')
        # f.write('\t' + '\t'.join(['best_param', 'best_ari'] * len(dataset_names)) + '\n')
        for kernel in kernels:
            f.write(kernel.name + '\t')
            for dataset_name in dataset_names:
                if dataset_name == '':
                    continue
                measure_results = results[dataset_name][1][kernel.name]
                x, y = measure_results['x'], measure_results['y']
                best_idx = np.argmax(y)
                best_param, best_ari = x[best_idx], y[best_idx]
                f.write(f'{best_ari:.2f}\t')
            f.write('\n')


def calc_part6(n_jobs=6):
    results = datasets_kward(n_params=101, n_jobs=n_jobs, backend='pytorch')

    _plot_sorted(results, './results/p6-KWard-sorted.png')
    _print_results(results, './results/p6-KWard-results.tsv')


if __name__ == '__main__':
    calc_part6()
