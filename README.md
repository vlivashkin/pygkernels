# py_graphs

Framework for clustering graphs using various distances/proximity measures.

List of distances/proximity measures:
* Shortest Path
* Commute Time
* plain Walk (Von Neumann diffusion kernel)
* Forest(Regularized Laplacian kernel)
* Communicability (Exponential diffusion kernel)
* Heat kernel (Laplacian exponential diffusion kernel)
* Walk (logarithmic), logarithmic Forest, logarithmic Communicability, logarithmic Heat
* Sigmoid Commute Time
* Sigmoid Corrected Commute Time
* Randomized Shortest Path
* Free Energy
* NormalizedHeat
* RegularizedLaplacian
* PersonalizedPageRank
* ModifiedPersonalizedPageRank
* HeatPersonalizedPageRank

List of clustering algoritms:
* Kernel k-means
* Spectral clustering
* Ward

List of graph generators:
* Stochastic Block Model

List of graph samples:
* Football
* Polbooks
* Polblogs
* Zachary
* Newsgroup


Publications
------------

If you wish to use and cite this work, please cite this earlier paper
which used many of the same concepts and methods (a newer publication
is in preparation):

Ivashkin and Chebotarev, "Do logarithmic proximity measures outperform plain ones in graph clustering?." International Conference on Network Analysis, Springer, 2016.
