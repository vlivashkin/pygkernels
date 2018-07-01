import logging
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm

from py_graphs.colors import d3


def linspace(start, end, count):
    grid = list(np.linspace(start, end, count))
    step = (end - start) / (count - 1)
    grid.extend([0.1 * step, 0.5 * step, end - 0.1 * step, end - 0.5 * step])
    return sorted(grid)


class ParallelByGraphs:
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
            print(len(graphs), len(values))
            if len(values) >= 0.5 * len(graphs):
                param_dict[param] = np.nanmean(values), np.nanstd(values)
        x, y, error = zip(*[(x, y[0], y[1]) for x, y in sorted(param_dict.items(), key=lambda x: x[0])])
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
                y_pred = clf.predict(K)
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


def plot_results(ax, toplot):
    for (name, x, y, error), (color1, color2) in zip(toplot, d3()):
        plot_ax(ax, name, x, y, error, color1, color2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
