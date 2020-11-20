from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, k_means, SpectralClustering
from sklearn.utils import deprecated

from pygkernels.cluster.base import KernelEstimator, REstimatorWrapper


# @deprecated("This is not a kernel method!")
class KMeans_sklearn(KernelEstimator):
    name = 'KMeans_sklearn'

    def __init__(self, n_clusters, init='k-means++', algorithm='auto', n_init=10, random_state=None):
        super().__init__(n_clusters)
        self.init = init
        self.algorithm = algorithm
        self.n_init = n_init
        self.random_state = random_state

    def predict(self, K, A: Optional[np.array] = None):
        _, pred, _ = k_means(K, n_clusters=self.n_clusters, init=self.init, algorithm=self.algorithm,
                             n_init=self.n_init, random_state=self.random_state)
        return pred


@deprecated("This is not a kernel method!")
class Ward_sklearn(KernelEstimator):
    name = 'Ward_sklearn'

    def predict(self, K, A: Optional[np.array] = None):
        return AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward').fit_predict(K)


class SpectralClustering_sklearn(KernelEstimator):
    name = 'SpectralClustering_sklearn'

    def predict(self, K, A: Optional[np.array] = None):
        return SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed').fit_predict(K - np.nanmin(K))


class KKMeans_kernlab(REstimatorWrapper):
    name = 'KernelKMeans_kernlab'

    def predict(self, K, A: Optional[np.array] = None):
        return self._predict(K, 'kkmeans.r')


class SpectralClustering_kernlab(REstimatorWrapper):
    name = 'SpectralClustering_kernlab_-min'

    def predict(self, K, A: Optional[np.array] = None):
        return self._predict(K - np.nanmin(K), 'spectral_clustering.r')


class SpectralClustering_kernlab_100(REstimatorWrapper):
    name = 'SpectralClustering_kernlab_+100'

    def predict(self, K, A: Optional[np.array] = None):
        return self._predict(K + 100, 'spectral_clustering.r')
