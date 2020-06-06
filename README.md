[![Build Status](https://travis-ci.com/vlivashkin/pygkernels.svg?branch=master)](https://travis-ci.com/vlivashkin/pygkernels)
[![codecov](https://codecov.io/gh/vlivashkin/pygkernels/branch/master/graph/badge.svg)](https://codecov.io/gh/vlivashkin/pygkernels)
# pygkernels &mdash; Kernels on Graphs for Python

Framework for clustering graph nodes using various similarity/dissimilarity measures.

#### List of measures:
* Adjacency matrix based kernels:
  * **Katz**: Katz kernel (a.k.a. Walk, Von Neumann diffusion kernel)
  * **Comm**: Communicability kernel (a.k.a. Exponential diffusion kernel)
  * **DFS**: Double Factorial similarity
* Laplacian based kernels:
  * **For**: Forest kernel (a.k.a. Regularized Laplacian kernel)
  * **Heat**: Heat kernel (a.k.a. Laplacian exponential diffusion kernel)
  * **NHeat**: Normalized Heat kernel
  * **Abs**: Absorption kernel
* Markov matrix based kernels and measures:
  * **PPR**: Personalized PageRank
  * **MPPR**: Modified Personalized PageRank
  * **HPR**: PageRank heat similarity measure
  * **RSP**: Randomized Shortest Path distance
  * **FE**: Free Energy distance
* Sigmoid Commute Time:
  * **SCT**: Sigmoid Commute Time
  * **SCCT**: Sigmoid Corrected Commute Time
* **SP-CT**: Shortest Path, Commute Time, and their linear combination
* logarithmic version of every measure

Every measure is presented as dissimilarity (distance) and similarity (kernel/proximity) measure. All of them can be used in any classification/clustering/community detection algorithm which uses kernel trick (e.g. kernel k-means).

#### List of clustering algoritms:
* Kernel k-means (GPU support, powered with pytorch)
* Spectral clustering
* Ward clustering
* Wrappers for kernel k-means from kernlab, sklearn k-means

#### Graph generators:
* Stochastic Block Model

#### Graph datsets:
https://github.com/vlivashkin/community-graphs


## Usage

#### Simple clustering:
```python
import networkx as nx
from pygkernels.cluster import KKMeans
from pygkernels.measure import logComm_H

G = nx.read_gml('news_2cl1.gml')
A = nx.adjacency_matrix(G).todense()

estimator = KKMeans(n_clusters=2)
K = logComm_H(A).get_K(param=0.1)
partition = estimator.predict(K, A=A)
```

#### Grid search measure parameters:
```python
import numpy as np
from sklearn.metrics import adjusted_rand_score
from pygkernels.cluster import KKMeans
from pygkernels.data import StochasticBlockModel
from pygkernels.measure import logComm_H
from pygkernels.scenario import ParallelByGraphs

n_graphs, n, k, p_in, p_out = 100, 100, 2, 0.3, 0.1  # params for G(n (k)p_in, p_out) graph generator
n_params = 30  # grid search through n_params in [0, 1] space

graphs, _ = StochasticBlockModel(n, k, p_in=p_in, p_out=p_out).generate_graphs(n_graphs)
gridsearch = ParallelByGraphs(adjusted_rand_score, n_params, progressbar=True, ignore_errors=True)

params, ari, error = gridsearch.perform(KKMeans, logComm_H, graphs, k, n_jobs=-1, n_gpu=2)
best_param = params[np.argmax(ari)]
```

All examples are located in [./examples](./examples).
More usage approaches are in a separate repo https://github.com/vlivashkin/pygkernels-experiments

## Citation

```
@inproceedings{ivashkin2016logarithmic,
  title={Do logarithmic proximity measures outperform plain ones in graph clustering?},
  author={Ivashkin, Vladimir and Chebotarev, Pavel},
  booktitle={International Conference on Network Analysis},
  pages={87--105},
  year={2016},
  organization={Springer}
}
```