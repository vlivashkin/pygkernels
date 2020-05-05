import networkx as nx

from pygkernels.cluster import KKMeans
from pygkernels.measure import logComm_H

G = nx.read_gml('news_2cl_1.gml')
A = nx.adjacency_matrix(G).todense()

estimator = KKMeans(n_clusters=2)
K = logComm_H(A).get_K(param=0.1)
partition = estimator.predict(K, A=A)
