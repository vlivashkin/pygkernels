import logging
from collections import defaultdict
from itertools import product, combinations
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
    '#9edae5'
]


def d3():
    idx = 0
    while True:
        yield d3_category20[idx], d3_category20[idx + 1]
        idx = (idx + 2) % 20


d3_right_order = []
for i in range(6):
    d3_right_order.extend([d3_category20[2 * i + 1], d3_category20[2 * i]])
d3_right_order.append(d3_category20[12])

measures_right_order = [
    'pWalk H',
    'Walk H',
    'For H',
    'logFor H',
    'Comm H',
    'logComm H',
    'Heat H',
    'logHeat H',
    'SCT H',
    'SCCT H',
    'RSP K',
    'FE K',
    'SP-CT H',
    'SP K',
    'CT H'
]


class ParallelByGraphs:
    """
    High-level class for calculate quality vs. param graphs
    """

    def __init__(self, scorer, params_flat, progressbar=False, verbose=False):
        self.scorer = scorer
        self.params_flat = params_flat
        self.progressbar = progressbar
        self.verbose = verbose

    def perform(self, clf_class, kernel_class, graphs, n_class, n_jobs=1):
        clf = clf_class(n_class)

        raw_param_dict = defaultdict(list)
        if self.progressbar:
            graphs = tqdm(graphs, desc=kernel_class.name)
        if n_jobs == 1:  # not parallel
            for graph_idx, graph in enumerate(graphs):
                graph_results = self.calc_graph(graph, kernel_class, clf, graph_idx)
                for param_flat, ari in graph_results.items():
                    raw_param_dict[param_flat].append(ari)
        else:
            all_graph_results = Parallel(n_jobs=n_jobs)(
                delayed(self.calc_graph)(graph, kernel_class, clf, graph_idx) for graph_idx, graph in enumerate(graphs))
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

    def calc_graph(self, graph, kernel_class, clf, graph_idx):
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


def plot_ax(ax, name, x, y, error, color1, color2):
    ax.plot(x, y, color=color1, label=name)
    ax.fill_between(x, y - error, y + error,
                    alpha=0.2, edgecolor=color1, facecolor=color2,
                    linewidth=1, antialiased=True)


def plot_results(ax, toplot, xlim=(0, 1), ylim=(-0.01, 1.01)):
    for (name, x, y, error), (color1, color2) in zip(toplot, d3()):
        plot_ax(ax, name, x, y, error, color1, color2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend()


class RejectCurve:
    """
    High-level class for calculation i.e. reject curves: tpr vs. fpr
    """

    def __init__(self, n_nodes: list, n_classes: list, p_in: list, p_out: list, kernels: list, distances: list,
                 generator_class, estimator_class):
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out
        self.kernels = kernels
        self.distances = distances
        self.generator_class = generator_class
        self.estimator_class = estimator_class

        self.best_params = None

    def calc(self, n_graphs, n_jobs=-1):
        print("calc data to find best params...")
        results = defaultdict(lambda: defaultdict(lambda: 0))
        for column in tqdm(list(product(self.n_nodes, self.n_classes, self.p_in, self.p_out))):
            n_nodes, n_classes, p_in, p_out = column
            graphs, info = self.generator_class(n_nodes, n_classes, p_in, p_out).generate_graphs(n_graphs)
            classic_plot = ParallelByGraphs(adjusted_rand_score, np.linspace(0, 1, 51), progressbar=True)
            for kernel_class in tqdm(self.kernels, desc=str(column)):
                results[column][kernel_class.name] = classic_plot.perform(self.estimator_class, kernel_class, graphs,
                                                                          n_classes, n_jobs=n_jobs)

        print("find best params...")
        best_params = defaultdict(lambda: defaultdict(lambda: 0))
        for column, measures in results.items():
            for measure_name, measure_results in measures.items():
                x, y, error = measure_results
                best_idx = np.argmax(y)
                print('{}\t{}\t{:0.2f} ({:0.2f})'.format(column, measure_name.ljust(8, ' '), x[best_idx], y[best_idx]))
                best_params[column][measure_name] = x[best_idx]

        self.best_params = ddict2dict(best_params)
        return results

    def _reject_curve(self, K, y_true):
        pairs = [(K[a, b], y_true[a] == y_true[b])
                 for a, b in combinations(range(K.shape[0]), 2) if a != b and not np.isnan(K[a, b])]
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

    def perform(self, n_graphs):
        results = defaultdict(lambda: defaultdict(lambda: list()))
        for column in tqdm(self.best_params.keys()):
            n_nodes, n_classes, p_in, p_out = column
            graphs, info = self.generator_class(n_nodes, n_classes, p_in, p_out).generate_graphs(n_graphs)
            for edges, nodes in graphs:
                for distance_class in self.distances:
                    try:
                        param_flat = self.best_params[column][distance_class.name + ' H']
                    except:
                        param_flat = self.best_params[column][distance_class.name + ' K']
                    kernel = distance_class(edges)
                    best_param = kernel.scaler.scale(param_flat)
                    K = kernel.get_D(best_param)
                    tpr, fpr = self._reject_curve(K, nodes)
                    results[column][distance_class.name].append((tpr, fpr))
        return results
