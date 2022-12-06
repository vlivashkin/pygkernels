from networkx.algorithms.community import LFR_benchmark_graph

from pygkernels.data import Datasets, LFRGenerator
from pygkernels.data.utils import nx2np


def use_direct_nx():
    print("use_direct_nx")
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
    A, partition = nx2np(G)
    print("success!")
    gen = LFRGenerator.params_from_adj_matrix(A, partition)
    print(gen.generate_info())


def generate_by_fixed_params():
    print("generate_by_fixed_params")
    n = 250
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    A, partition = LFRGenerator(n, tau1, tau2, mu, average_degree=5, min_community=20).generate_connected_graph(seed=10)
    print("success!")
    gen = LFRGenerator.params_from_adj_matrix(A, partition)
    print(gen.generate_info())


def generate_like():
    print("generate_like")
    (A, partition), info = Datasets()["news_2cl1_0.1"]
    gen = LFRGenerator.params_from_adj_matrix(A, partition)
    print(gen.generate_info())
    A, partition = gen.generate_graph()
    gen = LFRGenerator.params_from_adj_matrix(A, partition)
    print(gen.generate_info())


if __name__ == "__main__":
    use_direct_nx()
    generate_by_fixed_params()
    generate_like()
