from pygraphs.cluster import _kward_numpy, _kward_pytorch
from pygraphs.cluster.base import KernelEstimator


class KWard(KernelEstimator):
    name = 'KWard'

    def __init__(self, n_clusters, backend='pytorch', device=None, random_state=None):
        super().__init__(n_clusters, device=device, random_state=random_state)

        if backend == 'numpy':
            self.backend = _kward_numpy
        elif backend == 'pytorch':
            self.backend = _kward_pytorch

    def predict(self, K):
        return self.backend.predict(K, n_clusters=self.n_clusters, device=self.device)
