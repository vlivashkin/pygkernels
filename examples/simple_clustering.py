import networkx as nx

from pygkernels.cluster import KKMeans
from pygkernels.data import Datasets
from pygkernels.measure import logComm_H

_, Gs, _ = Datasets().news_2cl_1  # example graph
G: nx.Graph = Gs[0]
A = nx.adjacency_matrix(G).todense()

estimator = KKMeans(n_clusters=2)
K = logComm_H(A).get_K(param=0.1)
y_pred = estimator.predict(K, G=G)
