import operator
import unittest

from sklearn.metrics import adjusted_rand_score

from cluster import KernelKMeans, SpectralClustering
from cluster.ward import Ward
from graphs import dataset
from graphs import sample
from measure.kernel import *
from measure.kernel_new import KernelNew
from measure.shortcuts import *


class EstimatorsTests(unittest.TestCase):
    def test_simple_Ward(self):
        y_pred = Ward(2).predict(sample.diploma_matrix)
        print(y_pred)

    def test_all_estimators(self):
        K = sample.diploma_matrix  # this is not kernel but who cares

        y_pred_kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0).fit_predict(K)
        y_pred_ward = Ward(n_clusters=2).fit_predict(K)
        y_pred_spectral = SpectralClustering(n_clusters=2).fit_predict(K)
        print('KMeans:', y_pred_kmeans)
        print('Ward:', y_pred_ward)
        print('Spectral Clustering:', y_pred_spectral)

    if __name__ == '__main__':
        unittest.main()


class WorkflowTests(unittest.TestCase):
    def test_ward_clustering(self):
        graphs, info = dataset.polbooks
        for measure in Kernel.get_all_H():
            measureparamdict = {}
            mean = []
            for edges, nodes in graphs:
                measure_o = measure(edges)
                param = list(measure_o.scaler.scale_list([0.5]))[0]
                D = measure_o.get_K(param)
                y_pred = Ward(len(list(set(graphs[0][1])))).predict(D)
                ari = adjusted_rand_score(nodes, y_pred)
                mean.append(ari)
            mean = [m for m in mean if m is not None and m == m]
            score = np.array(mean).mean()
            if score is not None and score == score:
                measureparamdict[0.5] = score
            maxparam = max(measureparamdict.items(), key=operator.itemgetter(1))[0]
            print("{}\t{}\t{}".format(measure_o.name, maxparam, measureparamdict[maxparam]))

    def test_ward_clustering_new_kernels(self):
        graphs, info = dataset.polbooks
        for measure in KernelNew.get_all_new():
            measureparamdict = {}
            mean = []
            for edges, nodes in graphs:
                measure_o = measure(edges)
                param = list(measure_o.scaler.scale_list([0.5]))[0]
                D = measure_o.get_K(param)
                if D is not None:
                    y_pred = Ward(len(list(set(graphs[0][1])))).predict(D, )
                    ari = adjusted_rand_score(nodes, y_pred)
                    mean.append(ari)
                else:
                    mean.append(None)
            mean = [m for m in mean if m is not None and m == m]
            score = np.array(mean).mean()
            if score is not None and score == score:
                measureparamdict[0.5] = score
            maxparam = max(measureparamdict.items(), key=operator.itemgetter(1))[0]
            print("{}\t{}\t{}".format(measure_o.name, maxparam, measureparamdict[maxparam]))
