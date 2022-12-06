import logging
from collections import defaultdict
from functools import partial
from itertools import combinations
from random import shuffle

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from pygkernels.util import ddict2dict

d3_category20 = [
    "#aec7e8",
    "#1f77b4",  #  0  1
    "#ffbb78",
    "#ff7f0e",  #  2  3
    "#98df8a",
    "#2ca02c",  #  4  5
    "#ff9896",
    "#d62728",  #  6  7
    "#c5b0d5",
    "#9467bd",  #  8  9
    "#9edae5",
    "#17becf",  # 10 11
    "#f7b6d2",
    "#e377c2",  # 12 13
    "#c7c7c7",
    "#7f7f7f",  # 14 15
    "#dbdb8d",
    "#bcbd22",  # 16 17
    "#c49c94",
    "#8c564b",  # 18 19
    "#7eeaba",
    "#4cb787",  # 20 21
    "#dd00dd",
    "#aa00aa",  # 22 23
    "#555555",
    "#cccccc",  # 24 25
]


def d3():
    idx = 0
    while True:
        yield d3_category20[idx], d3_category20[idx + 1]
        idx = (idx + 2) % 20


d3_colors = {
    "Katz": d3_category20[0],
    "logKatz": d3_category20[1],
    "For": d3_category20[2],
    "logFor": d3_category20[3],
    "Comm": d3_category20[4],
    "logComm": d3_category20[5],
    "Heat": d3_category20[6],
    "logHeat": d3_category20[7],
    "NHeat": d3_category20[8],
    "logNHeat": d3_category20[9],
    "SCT": d3_category20[10],
    "SCCT": d3_category20[11],
    "RSP": d3_category20[12],
    "FE": d3_category20[13],
    "PPR": d3_category20[14],
    "logPPR": d3_category20[15],
    "ModifPPR": d3_category20[16],
    "logModifPPR": d3_category20[17],
    "HeatPR": d3_category20[18],
    "logHeatPR": d3_category20[19],
    "DF": d3_category20[20],
    "logDF": d3_category20[21],
    "Abs": d3_category20[22],
    "logAbs": d3_category20[23],
    "SP-CT": d3_category20[24],
    "several": d3_category20[25],
}


class ParallelByGraphs:
    """
    High-level class for calculate "quality vs. param" plots
    """

    def __init__(self, scorer, params_flat, progressbar=False, verbose=False, ignore_errors=False):
        self.scorer = scorer
        self.params_flat = (
            params_flat
            if type(params_flat) == list or type(params_flat) == np.array
            else np.linspace(0, 1, params_flat)
        )
        self.progressbar = progressbar
        self.verbose = verbose
        self.ignore_errors = ignore_errors

    def _calc_param(self, param_flat, kernel, estimator, y_true):
        param = kernel.scaler.scale(param_flat)
        K = kernel.get_K(param)
        y_pred = estimator.fit_predict(K)
        score = self.scorer(y_true, y_pred)
        return score

    def secure_run(self, func, error_prefix):
        if self.ignore_errors:
            try:
                return func()
            except Exception or FloatingPointError or np.linalg.LinAlgError as e:
                if self.verbose:
                    logging.error(f"{error_prefix}: {e}")
                return None
        else:
            return func()

    def _calc_graph(self, graph, kernel_class, estimator, graph_idx, single_graph=False):
        edges, y_true = graph
        graph_results = {}

        kernel = self.secure_run(partial(kernel_class, edges), f"{kernel_class.name}, graph {graph_idx}")
        if kernel is None:
            return graph_results

        params = self.params_flat
        if single_graph and self.progressbar:
            params = tqdm(params, desc=kernel_class.name)
        for param_flat in params:
            score = self.secure_run(
                partial(self._calc_param, param_flat, kernel, estimator, y_true),
                f"{kernel_class.name}, graph {graph_idx}",
            )
            if score is not None:
                graph_results[param_flat] = score
        return graph_results

    def perform(self, estimator_class, kernel_class, graphs, n_classes, n_jobs=1, n_gpu=2):
        raw_param_dict = defaultdict(list)
        if len(graphs) == 1:  # single graph scenario
            graph_results = self._calc_graph(
                graphs[0], kernel_class, estimator_class(n_classes, random_state=2000), 0, single_graph=True
            )
            for param_flat, ari in graph_results.items():
                raw_param_dict[param_flat].append(ari)
        elif n_jobs > 1:  # parallel
            if self.progressbar:
                graphs = tqdm(graphs, desc=kernel_class.name)
            all_graph_results = Parallel(n_jobs=n_jobs)(
                delayed(self._calc_graph)(
                    graph,
                    kernel_class,
                    estimator_class(
                        n_classes, device=graph_idx % n_gpu if n_gpu > 0 else "cpu", random_state=2000 + graph_idx
                    ),
                    graph_idx,
                )
                for graph_idx, graph in enumerate(graphs)
            )
            for graph_results in all_graph_results:
                for param_flat, ari in graph_results.items():
                    raw_param_dict[param_flat].append(ari)
        else:
            if self.progressbar:
                graphs = tqdm(graphs, desc=kernel_class.name)
            for graph_idx, graph in enumerate(graphs):
                graph_results = self._calc_graph(
                    graph, kernel_class, estimator_class(n_classes, random_state=2000 + graph_idx), graph_idx
                )
                for param_flat, ari in graph_results.items():
                    raw_param_dict[param_flat].append(ari)

        param_dict = {}
        for param, values in raw_param_dict.items():
            # logging.info(f'{param:.2f}: {len(values)}, {0.5 * len(graphs)}')
            if len(values) >= 0.5 * len(graphs):
                param_dict[param] = np.nanmean(values), np.nanstd(values)
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
    ax.fill_between(
        x, low_error, high_error, alpha=0.2, edgecolor=color1, facecolor=color2, linewidth=1, antialiased=True
    )


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

    def __init__(self, columns: list, distances: list, generator_class, best_params):
        self.columns = columns
        self.distances = distances
        self.generator_class = generator_class

        assert all([column in list(best_params.keys()) for column in self.columns])
        self._best_params = defaultdict(dict)
        for column in self.columns:
            for distance in self.distances:
                self._best_params[column][distance.name] = best_params[column][distance.name]
        self._best_params = ddict2dict(self._best_params)

    @staticmethod
    def _reject_curve(K, y_true, need_shuffle):
        pairs = [
            (K[a, b], y_true[a] == y_true[b])
            for a, b in combinations(range(K.shape[0]), 2)
            if a != b and not np.isnan(K[a, b])
        ]
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
        return ddict2dict(results)
