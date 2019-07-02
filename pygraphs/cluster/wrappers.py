import numpy as np
from sklearn.cluster import AgglomerativeClustering, k_means, SpectralClustering
from sklearn.utils import deprecated

from pygraphs.cluster.base import KernelEstimator, REstimatorWrapper


@deprecated("This is not a kernel method!")
class KMeans_sklearn(KernelEstimator):
    name = 'KMeans_sklearn'

    def predict(self, K):
        _, pred, _ = k_means(K, n_clusters=self.n_clusters, precompute_distances=True, random_state=0)
        return pred


@deprecated("This is not a kernel method!")
class Ward_sklearn(KernelEstimator):
    name = 'Ward_sklearn'

    def predict(self, K):
        return AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward').fit_predict(K)


class SpectralClustering_sklearn(KernelEstimator):
    name = 'SpectralClustering_sklearn'

    def predict(self, K):
        return SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed').fit_predict(K - np.nanmin(K))


class KKMeans_kernlab(REstimatorWrapper):
    name = 'KernelKMeans_kernlab'

    def predict(self, K):
        return self._predict(K, 'kkmeans.r')


class SpectralClustering_kernlab(REstimatorWrapper):
    name = 'SpectralClustering_kernlab_-min'

    def predict(self, K):
        return self._predict(K - np.nanmin(K), 'spectral_clustering.r')


class SpectralClustering_kernlab_100(REstimatorWrapper):
    name = 'SpectralClustering_kernlab_+100'

    def predict(self, K):
        return self._predict(K + 100, 'spectral_clustering.r')
