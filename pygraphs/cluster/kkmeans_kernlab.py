import os
import subprocess
import uuid
from os.path import join as pj

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin

from pygraphs.graphs import Datasets


class KKMeansKernlab(BaseEstimator, ClusterMixin):
    name = 'KernelKMeansKernlab'

    RSCRIPT_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        temp_name = f"{uuid.uuid4()}.csv"
        np.savetxt(temp_name, K, delimiter=",")
        subprocess.check_output(["Rscript", "--vanilla", pj(KKMeansKernlab.RSCRIPT_ROOT_PATH, "kmeans_2_clusters.r"),
                                 temp_name, str(self.n_clusters)])
        result = list(pd.read_csv(temp_name + '_result.csv')['x'])
        os.remove(temp_name)
        os.remove(temp_name + '_result.csv')
        return result


if __name__ == '__main__':
    graph, info = Datasets().news_2cl_1
    X, y = graph[0]

    km = KKMeansKernlab(n_clusters=2)
    print(km.fit_predict(X))
    print(km.predict(X))
