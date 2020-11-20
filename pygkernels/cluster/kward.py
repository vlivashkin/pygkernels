from typing import Optional

import numpy as np

from pygkernels.cluster import _kward_pytorch
from pygkernels.cluster.base import KernelEstimator


class KWard(KernelEstimator):
    name = 'KWard'

    def __init__(self, n_clusters, device=None, random_state=None):
        super().__init__(n_clusters, device=device, random_state=random_state)

    def predict(self, K, A: Optional[np.array] = None):
        return _kward_pytorch.predict(K, n_clusters=self.n_clusters, device=self.device)
