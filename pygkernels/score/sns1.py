import networkx as nx
import numpy as np


# https://github.com/MartijnGosgens/validation_indices


def fraction(numerator, denominator):
    if denominator != 0:
        return numerator / denominator
    if numerator == 0:
        return 0
    if numerator > 0:
        return float("inf")
    return -float("inf")


class Contingency(dict):
    def __init__(self, A, B):
        self.A = Clustering.from_anything(A)
        self.B = Clustering.from_anything(B)
        self.n = len(self.A)
        self.meet = Clustering.meet(self.A, self.B)
        self.sizesA = [len(Ai) for Ai in self.A.partition()]
        self.sizesB = [len(Bj) for Bj in self.B.partition()]
        super().__init__({key: len(p) for key, p in self.meet.items()})


class Score:
    isdistance = False

    @classmethod
    def score(cls, A, B):
        pass


class Clustering(list):
    def __init__(self, clustering_list):
        super().__init__(clustering_list)
        self.clusters = {}
        for i, c in enumerate(clustering_list):
            if not c in self.clusters:
                self.clusters[c] = set()
            self.clusters[c] = self.clusters[c].union({i})

    # Override
    def __setitem__(self, key, value):
        self.clusters[self[key]] -= {key}
        if not value in self.clusters:
            self.clusters[value] = set()
        self.clusters[value] = self.clusters[value].union({key})
        if len(self.clusters[self[key]]) == 0:
            del self.clusters[self[key]]
        super(Clustering, self).__setitem__(key, value)

    # Override
    def copy(self):
        return Clustering(super().copy())

    def labels(self):
        return set(self)

    def append(self, value):
        self.clusters[value] = self.clusters[value].union({len(self)})
        super(Clustering, self).append(value)

    def swap(self, i, j):
        self[i], self[j] = self[j], self[i]

    def merge(self, c1, c2):
        for i, c in enumerate(self):
            if c == c2:
                self[i] = c1
        del self.clusters[c2]

    def newlabel(self):
        labels = self.clusters.keys()
        label = len(labels)
        while label in labels:
            label = (label + 1) % len(labels)
        return label

    def splitoff(self, newset):
        label = self.newlabel()
        for i in newset:
            self[i] = label

    def intra_pairs(self):
        return sum([int(size * (size - 1) / 2) for size in self.sizes()])

    def partition(self):
        return list(self.clusters.values())

    def sizes(self):
        return [len(cluster) for cluster in self.clusters.values()]

    @staticmethod
    def meet(A, B):
        return {(i, j): Ai.intersection(Bj) for i, Ai in enumerate(A.partition()) for j, Bj in enumerate(B.partition())}

    # Operator overload of A*B will return the meet of the clusterings.
    def __mul__(self, other):
        return Clustering.from_partition(Clustering.meet(self, other).values())

    @staticmethod
    def from_sizes(sizes):
        return Clustering(sum([[c] * size for c, size in enumerate(sizes)], []))

    @staticmethod
    def from_partition(partition):
        # Assume that partition contains all integers in [0..n-1]
        n = sum([len(p) for p in partition])
        clustering = list(range(n))
        for c, p in enumerate(partition):
            for i in p:
                clustering[i] = c
        return Clustering(clustering)

    @staticmethod
    def from_anything(A):
        # Check if not already a clustering.
        if type(A) == Clustering:
            return A
        # If its a dict, we assume its just a labeled partition.
        if isinstance(A, dict):
            return Clustering.from_partition(A.values())
        # See whether it is iterable.
        if hasattr(A, "__iter__"):
            A = list(A)
            # If the first item is an integer, we assume it's a list
            # of clusterlabels so that we can call the constructor.
            if type(A[0]) == int:
                return Clustering(A)
            elif type(A[0]) in {set, list}:
                # If the first item is a set or list, we consider it a partition.
                return Clustering.from_partition(A)
        print("Clustering.FromAnything was unable to cast {}".format(A))

    @staticmethod
    def balanced_sizes(n, k):
        smallSize = int(n / k)
        n_larger = n - k * smallSize
        return [smallSize + 1] * n_larger + [smallSize] * (k - n_larger)

    @staticmethod
    def BalancedClustering(n, k):
        return Clustering.from_sizes(Clustering.balanced_sizes(n, k))

    def random_same_sizes(self, rand=None):
        if rand == None:
            rand = np.random
        c = list(self).copy()
        rand.shuffle(c)
        return Clustering(c)

    @staticmethod
    def uniform_random(n, k, rand=None):
        if rand == None:
            rand = np.random
        return Clustering(rand.randint(k, size=n))


class PairCounts(dict):
    def __init__(self, N00, N01, N10, N11):
        self.N00 = N00
        self.N01 = N01
        self.N10 = N10
        self.N11 = N11
        super().__init__({"N00": N00, "N01": N01, "N10": N10, "N11": N11})
        self.N = sum(self.values())
        self.mA = N11 + N10
        self.mB = N11 + N01

    @staticmethod
    def from_clusterings(A, B):
        A = Clustering.from_anything(A)
        B = Clustering.from_anything(B)

        mA = A.intra_pairs()
        mB = B.intra_pairs()
        N11 = (A * B).intra_pairs()
        N10 = mA - N11
        N01 = mB - N11
        n = len(A)
        N = int(n * (n - 1) / 2)
        N00 = N - N11 - N10 - N01
        return PairCounts(N00, N01, N10, N11)

    def adjacent_counts(self):
        all_directions = {
            "add disagreeing": PairCounts(
                N11=self.N11,
                N10=self.N10,
                N01=self.N01 + 1,
                N00=self.N00 - 1,
            ),
            "rem disagreeing": PairCounts(
                N11=self.N11,
                N10=self.N10,
                N01=self.N01 - 1,
                N00=self.N00 + 1,
            ),
            "add agreeing": PairCounts(
                N11=self.N11 + 1,
                N10=self.N10 - 1,
                N01=self.N01,
                N00=self.N00,
            ),
            "rem agreeing": PairCounts(
                N11=self.N11 - 1,
                N10=self.N10 + 1,
                N01=self.N01,
                N00=self.N00,
            ),
        }
        # Remove invalid
        return {action: pc for action, pc in all_directions.items() if -1 not in pc.values()}

    @staticmethod
    def from_graphs(A, B):
        mA, mB = (len(G.edges) for G in (A, B))
        N11 = len(set(A.edges).intersection(set(B.edges)))
        N = int(len(A.nodes) * (len(A.nodes) - 1) / 2)
        return PairCounts(N11=N11, N10=mA - N11, N01=mB - N11, N00=N - mA - mB + N11)

    @staticmethod
    def from_graph_and_clustering(G, C):
        G = nx.Graph(G)
        C = Clustering.from_anything(C)
        mA = len(G.edges)
        mB = C.intra_pairs()
        N11 = sum([1 for v, w in G.edges if C[v] == C[w]])
        N = int(len(C) * (len(C) - 1) / 2)
        return PairCounts(N11=N11, N10=mA - N11, N01=mB - N11, N00=N - mA - mB + N11)

    @staticmethod
    def from_clustering_and_graph(C, G):
        return PairCounts.from_graph_and_clustering(G, C).interchangeAB()

    @staticmethod
    def from_anything(anything):
        if type(anything) == PairCounts:
            return anything
        if type(anything) == Contingency:
            sizes2intra = lambda sizes: int((0.5 * sizes * (sizes - 1)).sum())
            N11 = sizes2intra(np.array(list(anything.values())))
            mA, mB = (sizes2intra(np.array(sizes)) for sizes in [anything.sizesA, anything.sizesB])
            N = int(anything.n * (anything.n - 1) / 2)
            return PairCounts(N11=N11, N10=mA - N11, N01=mB - N11, N00=N - mA - mB + N11)
        if type(anything) == dict:
            return PairCounts(**anything)
        if type(anything) == tuple:
            if type(anything[0]) == Clustering and type(anything[1]) == Clustering:
                return PairCounts.from_clusterings(*anything)
            if type(anything[0]) == Clustering and not type(anything[1]) == Clustering:
                return PairCounts.from_clustering_and_graph(*anything)
            if not type(anything[0]) == Clustering and type(anything[1]) == Clustering:
                return PairCounts.from_graph_and_clustering(*anything)
            else:
                return PairCounts.from_graphs(*anything)

    def interchangeAB(self):
        return PairCounts(N00=self.N00, N01=self.N10, N10=self.N01, N11=self.N11)

    def invertA(self):
        return PairCounts(N00=self.N10, N01=self.N11, N10=self.N00, N11=self.N01)

    def invertB(self):
        return PairCounts(N00=self.N01, N01=self.N00, N10=self.N11, N11=self.N10)

    def invertAB(self):
        return PairCounts(N00=self.N11, N01=self.N10, N10=self.N01, N11=self.N00)


class PairCountingScore(Score):
    def paircounting_score(**kwargs):
        pass

    @classmethod
    def score(cls, A, B, **kwargs):
        A = Clustering.from_anything(A)
        B = Clustering.from_anything(B)
        return cls.score_comparison(PairCounts.from_clusterings(A, B), **kwargs)

    @classmethod
    def score_comparison(cls, pc, **kwargs):
        if type(pc) != PairCounts:
            pc = PairCounts.from_anything(pc)
        return cls.paircounting_score(**pc, N=pc.N, paircounts=pc, **kwargs)


class SokalAndSneath1(PairCountingScore):
    @staticmethod
    def paircounting_score(N00, N01, N10, N11, **kwargs):
        return (
            fraction(N11, N11 + N10) + fraction(N11, N11 + N01) + fraction(N00, N00 + N10) + fraction(N00, N00 + N01)
        ) / 4


def preddict2list(y):
    result = []
    classes, last_class = {}, -1
    for key in sorted(y.keys()):
        class_ = y[key]
        if class_ not in classes:
            classes[class_] = last_class + 1
            last_class += 1
        result.append(classes[class_])
    return result


def sns1(y_true, y_pred):
    return SokalAndSneath1().score(y_true, y_pred.tolist())
