import logging
from collections import defaultdict
from itertools import combinations
from random import shuffle

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm_notebook as tqdm

from pygraphs.util import ddict2dict

d3_category20 = [
    '#1f77b4',
    '#aec7e8',
    '#ff7f0e',
    '#ffbb78',
    '#2ca02c',
    '#98df8a',
    '#d62728',
    '#ff9896',
    '#9467bd',
    '#c5b0d5',
    '#8c564b',
    '#c49c94',
    '#e377c2',
    '#f7b6d2',
    '#7f7f7f',
    '#c7c7c7',
    '#bcbd22',
    '#dbdb8d',
    '#17becf',
    '#9edae5',
    '#cccccc'
]


def d3():
    idx = 0
    while True:
        yield d3_category20[idx], d3_category20[idx + 1]
        idx = (idx + 2) % 20


d3_colors = {
    'pWalk': d3_category20[0],
    'Walk': d3_category20[1],
    'For': d3_category20[2],
    'logFor': d3_category20[3],
    'Comm': d3_category20[4],
    'logComm': d3_category20[5],
    'Heat': d3_category20[6],
    'logHeat': d3_category20[7],
    'NHeat': d3_category20[8],
    'logNHeat': d3_category20[9],
    'SCT': d3_category20[10],
    'SCCT': d3_category20[11],
    'RSP': d3_category20[12],
    'FE': d3_category20[13],
    'PPR': d3_category20[14],
    'logPPR': d3_category20[15],
    'ModifPPR': d3_category20[16],
    'logModifPPR': d3_category20[17],
    'HeatPPR': d3_category20[18],
    'logHeatPPR': d3_category20[19],
    'SP-CT': d3_category20[20]
}


class ParallelByGraphs:
    """
    High-level class for calculate quality vs. param graphs
    """

    def __init__(self, scorer, params_flat, progressbar=False, verbose=False):
        self.scorer = scorer
        self.params_flat = params_flat
        self.progressbar = progressbar
        self.verbose = verbose

    def _calc_graph(self, graph, kernel_class, clf, graph_idx):
        edges, nodes = graph
        kernel = kernel_class(edges)
        graph_results = {}
        for param_flat in self.params_flat:
            param = -1
            try:
                param = kernel.scaler.scale(param_flat)
                K = kernel.get_K(param)
                y_pred = clf.fit_predict(K)
                ari = self.scorer(nodes, y_pred)
                graph_results[param_flat] = ari
            except Exception or FloatingPointError as e:
                if self.verbose:
                    logging.error("{}, {:.2f}, graph {}: {}".format(kernel_class.name, param, graph_idx, e))
        return graph_results

    def perform(self, estimator_class, kernel_class, graphs, n_class, n_jobs=1):
        clf = estimator_class(n_class)

        raw_param_dict = defaultdict(list)
        if self.progressbar:
            graphs = tqdm(graphs, desc=kernel_class.name)
        if n_jobs == 1:  # not parallel
            # logging.info('n_jobs == 1, run NOT in parallel')
            for graph_idx, graph in enumerate(graphs):
                graph_results = self._calc_graph(graph, kernel_class, clf, graph_idx)
                for param_flat, ari in graph_results.items():
                    raw_param_dict[param_flat].append(ari)
        else:
            # logging.info(f'n_jobs == {n_jobs}, run in parallel!')
            all_graph_results = Parallel(n_jobs=n_jobs)(delayed(self._calc_graph)(graph, kernel_class, clf, graph_idx)
                                                        for graph_idx, graph in enumerate(graphs))
            for graph_results in all_graph_results:
                for param_flat, ari in graph_results.items():
                    raw_param_dict[param_flat].append(ari)

        param_dict = {}
        for param, values in raw_param_dict.items():
            # print(len(values), 0.5 * len(graphs))
            if len(values) >= 0.5 * len(graphs):
                param_dict[param] = np.nanmean(values), np.nanstd(values)
        # print('param_dict', param_dict)
        if len(param_dict) > 0:
            x, y, error = zip(*[(x, y[0], y[1]) for x, y in sorted(param_dict.items(), key=lambda x: x[0])])
        else:
            x, y, error = [], [], []
        return np.array(x), np.array(y), np.array(error)


def plot_ax(ax, name, x, y, error, color1, color2):
    ax.plot(x, y, color=color1, label=name)
    low_error = y - error
    low_error[low_error < 0] = 0
    high_error = y + error
    high_error[high_error > 1] = 1
    ax.fill_between(x, low_error, high_error,
                    alpha=0.2, edgecolor=color1, facecolor=color2,
                    linewidth=1, antialiased=True)


def plot_results(ax, toplot, xlim=(0, 1), ylim=(-0.01, 1.01), nolegend=False):
    for (name, x, y, error), (color1, color2) in zip(toplot, d3()):
        plot_ax(ax, name, x, y, error, color1, color2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not nolegend:
        ax.legend()


class RejectCurve:
    """
    High-level class for calculation i.e. reject curves: tpr vs. fpr
    """

    def __init__(self, columns: list, distances: list, generator_class, n_graphs):
        self.columns = columns
        self.distances = distances
        self.generator_class = generator_class

        self._best_params = None

    def set_best_params(self, best_params):
        sorted_distance_names = sorted([x.name for x in self.distances])
        assert sorted(list(best_params.keys())) == sorted(self.columns)
        for distances_in_column in [x.keys() for x in best_params.values()]:
            assert sorted(distances_in_column) == sorted_distance_names
        self._best_params = best_params

    def calc_best_params(self, kernels: list, estimator_class, n_graphs, n_jobs=-1):
        print("calc data to find best params...")
        results = defaultdict(lambda: defaultdict(lambda: 0))
        for column in tqdm(self.columns):
            n_nodes, n_classes, p_in, p_out = column
            graphs, info = self.generator_class(n_nodes, n_classes, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
            classic_plot = ParallelByGraphs(adjusted_rand_score, np.linspace(0, 1, 51), progressbar=True)
            for kernel_class in tqdm(kernels, desc=str(column)):
                results[column][kernel_class.name] = classic_plot.perform(estimator_class, kernel_class, graphs,
                                                                          n_classes, n_jobs=n_jobs)

        print("find best params...")
        best_params = defaultdict(lambda: defaultdict(lambda: 0))
        for column, kernels_results in results.items():
            for kernel_name, kernel_results in kernels_results.items():
                x, y, error = kernel_results
                best_idx = np.argmax(y)
                print('{}\t{}\t{:0.2f} ({:0.2f})'.format(column, kernel_name.ljust(8, ' '), x[best_idx], y[best_idx]))
                best_params[column][kernel_name[:-2]] = x[best_idx]

        self._best_params = ddict2dict(best_params)
        return results

    @staticmethod
    def _reject_curve(K, y_true, need_shuffle):
        pairs = [(K[a, b], y_true[a] == y_true[b])
                 for a, b in combinations(range(K.shape[0]), 2) if a != b and not np.isnan(K[a, b])]
        if need_shuffle:
            shuffle(pairs)
        pairs = sorted(pairs, key=lambda x: x[0])
        tpr, fpr = [0], [0]
        for _, same_class in pairs:
            if same_class:
                increment = 1, 0
            else:
                increment = 0, 1
            tpr.append(tpr[-1] + increment[0])
            fpr.append(fpr[-1] + increment[1])
        return np.array(fpr, dtype=np.float) / fpr[-1], np.array(tpr, dtype=np.float) / tpr[-1]

    def perform(self, n_graphs, need_shuffle=True):
        results = defaultdict(lambda: defaultdict(lambda: list()))
        for column in tqdm(self._best_params.keys()):
            n_nodes, n_classes, p_in, p_out = column
            graphs, info = self.generator_class(n_nodes, n_classes, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
            for edges, nodes in graphs:
                for distance_class in self.distances:
                    param_flat = self._best_params[column][distance_class.name]
                    distance = distance_class(edges)
                    best_param = distance.scaler.scale(param_flat)
                    D = distance.get_D(best_param)
                    tpr, fpr = self._reject_curve(D, nodes, need_shuffle=need_shuffle)
                    results[column][distance_class.name].append((tpr, fpr))
        return results
