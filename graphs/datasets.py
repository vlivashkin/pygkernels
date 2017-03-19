import os

from graphs.importer import ImportedGraphBuilder

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')


def load_polbooks_or_football(name, nodes, edges):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_id_name_class(os.path.join(ROOT_PATH, nodes)) \
        .import_edges(os.path.join(ROOT_PATH, edges)) \
        .build()


def load_polblogs_or_zachary(name, nodes):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_and_edges(os.path.join(ROOT_PATH, nodes)) \
        .build()


def load_newsgroup_graph(name, nodes, edges):
    return ImportedGraphBuilder() \
        .set_name(name) \
        .import_nodes_class(os.path.join(ROOT_PATH, nodes)) \
        .import_adjacency_matrix(os.path.join(ROOT_PATH, edges)) \
        .build()


football = load_polbooks_or_football('football', 'football_nodes.csv', 'football_edges.csv')
polbooks = load_polbooks_or_football('polbooks', 'polbooks_nodes.csv', 'polbooks_edges.csv')
polblogs = load_polblogs_or_zachary('polblogs', 'polblogs.net')
zachary = load_polblogs_or_zachary('zachary', 'zachary.net')
news_2cl_1 = load_newsgroup_graph('news_2cl_1', 'newsgroup/news_2cl_1_classeo.csv', 'newsgroup/news_2cl_1_Docr.csv')
news_2cl_2 = load_newsgroup_graph('news_2cl_2', 'newsgroup/news_2cl_2_classeo.csv', 'newsgroup/news_2cl_2_Docr.csv')
news_2cl_3 = load_newsgroup_graph('news_2cl_3', 'newsgroup/news_2cl_3_classeo.csv', 'newsgroup/news_2cl_3_Docr.csv')
news_3cl_1 = load_newsgroup_graph('news_3cl_1', 'newsgroup/news_3cl_1_classeo.csv', 'newsgroup/news_3cl_1_Docr.csv')
news_3cl_2 = load_newsgroup_graph('news_3cl_2', 'newsgroup/news_3cl_2_classeo.csv', 'newsgroup/news_3cl_2_Docr.csv')
news_3cl_3 = load_newsgroup_graph('news_3cl_3', 'newsgroup/news_3cl_3_classeo.csv', 'newsgroup/news_3cl_3_Docr.csv')
news_5cl_1 = load_newsgroup_graph('news_5cl_1', 'newsgroup/news_5cl_1_classeo.csv', 'newsgroup/news_5cl_1_Docr.csv')
news_5cl_2 = load_newsgroup_graph('news_5cl_2', 'newsgroup/news_5cl_2_classeo.csv', 'newsgroup/news_5cl_2_Docr.csv')
news_5cl_3 = load_newsgroup_graph('news_5cl_3', 'newsgroup/news_5cl_3_classeo.csv', 'newsgroup/news_5cl_3_Docr.csv')
