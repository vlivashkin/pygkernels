from sklearn.cluster import AgglomerativeClustering, k_means

from pygraphs.cluster.base import KernelEstimator, REstimatorWrapper


class KMeans_sklearn(KernelEstimator):
    name = 'KMeans_sklearn'

    def predict(self, K):
        _, prediction, _ = k_means(K, n_clusters=self.n_clusters, precompute_distances=True, random_state=0)
        return prediction


class Ward_sklearn(KernelEstimator):
    name = 'Ward_sklearn'

    def predict(self, K):
        prediction = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward').fit_predict(K)
        return prediction


class KKMeans_kernlab(REstimatorWrapper):
    name = 'KernelKMeans_kernlab'

    def predict(self, K):
        return self._predict(K, 'kkmeans.r')


class SpectralClustering_kernlab(REstimatorWrapper):
    name = 'SpectralClustering_kernlab'

    def predict(self, K):
        return self._predict(K, 'spectral_clustering.r')
