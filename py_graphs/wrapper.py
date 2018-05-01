from collections import defaultdict

import numpy as np
from numpy.linalg import LinAlgError
from tqdm import tqdm_notebook as tqdm


def linspace(start, end, count):
    grid = list(np.linspace(start, end, count))
    step = (end - start) / (count - 1)
    grid.extend([0.1 * step, 0.5 * step, end - 0.1 * step, end - 0.5 * step])
    return sorted(grid)


class ParallelByGraphs:
    def __init__(self, measures, clf, score_function, graphs):
        self.measure_graphs = defaultdict(lambda: [])
        for measure in tqdm(measures, desc='prepare'):
            for (A, y_true) in graphs:
                mg = measure(A)
                self.measure_graphs[mg.name].append((mg, y_true))
        self.measure_graphs = dict(self.measure_graphs)
        self.n_graphs = len(graphs)
        self.clf = clf
        self.score_function = score_function

    def parallel_by_graphs(self, params, n_jobs=1):
        measure_results = defaultdict(lambda: {})
        for measure_name in tqdm(self.measure_graphs.keys(), desc='n_measure'):
            for param in tqdm(params, desc=measure_name):
                result = self.avg_by_graphs(measure_name, param, n_jobs)
                if result is not None:
                    measure_results[measure_name][param] = result
                    # logging.info("{} {:0.2f} {:0.2f} {:0.2f}".format(measure_name, param, result[0], result[1]))
        return dict(measure_results)

    def avg_by_graphs(self, measure_name, param, n_jobs):
        # out = Parallel(n_jobs=n_jobs)(
        #     delayed(self.calc_for_graph)(measure_name, idx, param) for idx in range(self.n_graphs))
        out = [self.calc_for_graph(measure_name, idx, param) for idx in range(self.n_graphs)]

        out = list(filter(lambda item: item is not None, out))
        return (np.average(out), np.std(out)) if len(out) > 0.5 * self.n_graphs else None

    def calc_for_graph(self, measure_name, graph_idx, param):
        mg, y_true = self.measure_graphs[measure_name][graph_idx]
        try:
            K = mg.get_K(param)
            y_pred = self.clf.fit_predict(K)
            return self.score_function(y_true, y_pred)
        except (ValueError, LinAlgError, FloatingPointError):
            return None
