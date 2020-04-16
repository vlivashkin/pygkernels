[![Build Status](https://travis-ci.com/vlivashkin/pygraphs.svg?branch=master)](https://travis-ci.com/vlivashkin/pygraphs)
[![codecov](https://codecov.io/gh/vlivashkin/pygraphs/branch/master/graph/badge.svg)](https://codecov.io/gh/vlivashkin/pygraphs)
# pygkernels &mdash; Kernels on graphs for Python

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

Every measure is presented as dissimilarity (distance) and similarity (kernel/proximity measure). All of them can be used in any classification/clustering/community detection algorithm which uses kernel trick (e.g. kernel k-means).

#### List of clustering algoritms:
* Kernel k-means
* Spectral clustering
* Ward clustering
* Wrappers for kernel k-means from kernlab, sklearn k-means

#### Graph generators:
* Stochastic Block Model

#### Graph datsets:
* Dolphins
* EU-core
* Football
* Zachary karate
* Newsgroup (9 subsets)
* Polbooks


## Usage

#### Simple clustering:
```.python
import networkx as nx
from pygraphs.cluster import KKMeans
from pygraphs.graphs import Datasets
from pygraphs.measure import logComm_H

_, Gs, _ = Datasets().news_2cl_1  # example graph
G: nx.Graph = Gs[0]
A = nx.adjacency_matrix(G).todense()

estimator = KKMeans(n_clusters=2)
K = logComm_H(A).get_K(param=0.1)
y_pred = estimator.predict(K, G=G)
```


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