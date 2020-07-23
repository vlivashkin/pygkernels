from pygkernels.data import Datasets, LFRGenerator

(A, partition), info = Datasets()['news_2cl1_0.1']
gen = LFRGenerator.params_from_adj_matrix(A, partition, info['k'])
print(gen.generate_info())
A, partition = gen.generate_graph()
gen = LFRGenerator.params_from_adj_matrix(A, partition, info['k'])
print(gen.generate_info())
