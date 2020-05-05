from sklearn.metrics import adjusted_rand_score

from pygkernels.cluster import KKMeans
from pygkernels.data import StochasticBlockModel
from pygkernels.measure import logComm_H
from pygkernels.scenario import ParallelByGraphs

n_graphs, n, k, p_in, p_out = 100, 100, 2, 0.3, 0.1  # params for G(n (k)p_in, p_out) graph generator
n_params = 30  # grid search through n_params in [0, 1] space

graphs, _ = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
gridsearch_results = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=True, ignore_errors=True)

params, ari, error = gridsearch_results.perform(KKMeans, logComm_H, graphs, k, n_jobs=-1, n_gpu=2)
