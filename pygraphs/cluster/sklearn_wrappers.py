from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster.k_means_ import k_means


class KernelKMeansSklearn(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        _, prediction, _ = k_means(K, n_clusters=self.n_clusters, precompute_distances=True, random_state=0)
        return prediction


class KernelWardSklearn(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, A):
        self.n_clusters = n_clusters
        self.A = A

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        prediction = AgglomerativeClustering(n_clusters=self.n_clusters, connectivity=self.A, linkage='ward') \
            .fit_predict(K)
        return prediction
