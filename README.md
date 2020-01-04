[![Build Status](https://travis-ci.com/vlivashkin/pygraphs.svg?branch=master)](https://travis-ci.com/vlivashkin/pygraphs)
[![codecov](https://codecov.io/gh/vlivashkin/pygraphs/branch/master/graph/badge.svg)](https://codecov.io/gh/vlivashkin/pygraphs)
# pygraphs

Framework for clustering graphs using various distances/proximity measures.

List of distances/proximity measures:
* **SP-CT**: Shortest Path and Commute Time (and linear combination)
* **pWalk**: plain Walk (a.k.a. Von Neumann diffusion kernel)
* **For**: Forest (a.k.a. Regularized Laplacian kernel)
* **Comm**: Communicability (a.k.a. Exponential diffusion kernel)
* **Heat**: Heat kernel (a.k.a. Laplacian exponential diffusion kernel)
* **NHeat**: Normalized Heat
* **Walk**, **logFor**, **logComm**, **logHeat**, **logNHeat**: logarithmic versions of pWalk, For, Comm, Heat, NHeat
* **SCT**: Sigmoid Commute Time
* **SCCT**: Sigmoid Corrected Commute Time
* **RSP**: Randomized Shortest Path
* **FE**: Free Energy
* **PPR**: Personalized PageRank
* **ModifPPR**: Modified Personalized PageRank
* **HeatPPR**: Heat Personalized PageRank
* **logPPR**, **logModifPPR**, **logHeatPPR**: logarithmic versions of PPR, ModifPPR, HeatPPR

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
* Newsgroup
* Polbooks


Publications
------------

If you wish to use and cite this work, please cite this earlier paper
which used many of the same concepts and methods (a newer publication
is in preparation):

> Ivashkin and Chebotarev, "Do logarithmic proximity measures outperform plain ones in graph clustering?" International Conference on Network Analysis, Springer, 2016.
