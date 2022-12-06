import os
import subprocess
import uuid
from abc import abstractmethod, ABC
from os.path import join as pj
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClusterMixin


class KernelEstimator(BaseEstimator, ClusterMixin, ABC):
    def __init__(self, n_clusters, random_state=None, device=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
            print(f"Auto chosen device: {self.device}")
        else:
            self.device = device

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    @abstractmethod
    def predict(self, K, A: Optional[np.array] = None):
        pass

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


class REstimatorWrapper(KernelEstimator, ABC):
    RSCRIPT_ROOT_PATH = pj(os.path.dirname(os.path.abspath(__file__)), "r")

    def _predict(self, K, script_name):
        temp_name = f"{uuid.uuid4()}.csv"
        np.savetxt(temp_name, K, delimiter=",")
        try:
            subprocess.check_output(
                [
                    "Rscript",
                    "--vanilla",
                    pj(REstimatorWrapper.RSCRIPT_ROOT_PATH, script_name),
                    temp_name,
                    str(self.n_clusters),
                ],
                timeout=60,
            )
            result = list(pd.read_csv(temp_name + "_result.csv")["x"])
        finally:
            os.remove(temp_name)
            os.remove(temp_name + "_result.csv")
        return result


def torch_func(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            args = [
                torch.from_numpy(x).float().to(kwargs["device"])
                if type(x) in [np.ndarray, np.memmap] and x.dtype in [np.float32, np.float64]
                else x
                for x in args
            ]
            results = func(*args, **kwargs)
            if type(results) == tuple:
                results = tuple(x.cpu().numpy() if type(x) == torch.Tensor else x for x in results)
            else:
                results = results.cpu().numpy() if type(results) == torch.Tensor else results
        return results

    return wrapper
