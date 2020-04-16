[![Build Status](https://travis-ci.com/vlivashkin/pygraphs.svg?branch=master)](https://travis-ci.com/vlivashkin/pygraphs)
[![codecov](https://codecov.io/gh/vlivashkin/pygraphs/branch/master/graph/badge.svg)](https://codecov.io/gh/vlivashkin/pygraphs)
# pygraphs

Framework for clustering graphs using various distances/proximity measures.

List of distances/proximity measures:
* **SP-CT**: Shortest Path and Commute Time (and linear combination)
* **Katz**: Katz (a.k.a. Walk, Von Neumann diffusion kernel)
* **For**: Forest (a.k.a. Regularized Laplacian kernel)
* **Comm**: Communicability (a.k.a. Exponential diffusion kernel)
* **Heat**: Heat kernel (a.k.a. Laplacian exponential diffusion kernel)
* **NHeat**: Normalized Heat
* **SCT**: Sigmoid Commute Time
* **SCCT**: Sigmoid Corrected Commute Time
* **RSP**: Randomized Shortest Path
* **FE**: Free Energy
* **PPR**: Personalized PageRank
* **ModifPPR**: Modified Personalized PageRank
* **HeatPPR**: Heat Personalized PageRank
* logarithmic versions of all these measures

List of clustering algoritms:
* Kernel k-means
* Spectral clustering
* Ward
* Wrappers for kernel k-means from kernlab, sklearn k-means

List of graph generators:
* Stochastic Block Model

List of graph samples:
* Dolphins
* EU-core
* Football
* Zachary karate
* Newsgroup (9 subsets)
* Polbooks


## Usage

Simple clustering:
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