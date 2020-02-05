import os
import subprocess
import uuid
from abc import abstractmethod, ABC
from os.path import join as pj

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin


class KernelEstimator(BaseEstimator, ClusterMixin, ABC):
    def __init__(self, n_clusters, device=None, random_state=None):
        self.n_clusters = n_clusters
        self.device = device
        self.random_state = random_state

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    @abstractmethod
    def predict(self, K):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


class REstimatorWrapper(KernelEstimator, ABC):
    RSCRIPT_ROOT_PATH = pj(os.path.dirname(os.path.abspath(__file__)), 'r')

    def _predict(self, K, script_name):
        temp_name = f"{uuid.uuid4()}.csv"
        np.savetxt(temp_name, K, delimiter=",")
        try:
            subprocess.check_output(
                ["Rscript", "--vanilla", pj(REstimatorWrapper.RSCRIPT_ROOT_PATH, script_name),
                 temp_name, str(self.n_clusters)], timeout=60)
            result = list(pd.read_csv(temp_name + '_result.csv')['x'])
        finally:
            os.remove(temp_name)
            os.remove(temp_name + '_result.csv')
        return result
