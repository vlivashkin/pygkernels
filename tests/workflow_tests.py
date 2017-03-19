import operator
import unittest

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from graphs import datasets
from measure.kernel import Kernel
from measure.shortcuts import *


class WorkflowTests(unittest.TestCase):
    def test_ward_clustering(self):
        graphs, info = datasets.football
        for measure in Kernel.get_all_H_plus_RSP_FE():
            measureparamdict = {}
            for param in [0.1]:
                mean = []
                for edges, nodes in graphs:
                    D = measure.getK(edges, measure.scale().calc(edges, param))
                    ward = AgglomerativeClustering(n_clusters=len(list(set(graphs[0][1]))))
                    y_pred = ward.fit_predict(D)
                    ari = adjusted_rand_score(nodes, y_pred)
                    mean.append(ari)
                mean = [m for m in mean if m is not None and m == m]
                score = np.array(mean).mean()
                if score is not None and score == score:
                    measureparamdict[param] = score
            maxparam = max(measureparamdict.items(), key=operator.itemgetter(1))[0]
            print("{}\t{}\t{}".format(measure.name, maxparam, measureparamdict[maxparam]))
