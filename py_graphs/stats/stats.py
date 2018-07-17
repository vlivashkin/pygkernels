from statistics import mean, stdev
from scipy.stats import friedmanchisquare, rankdata, norm
import matplotlib.pyplot as plt
import Orange


def get_stats(results):
    """
    :param results: list of dicts (key - kernel name, value - performance metric), one dict per experiment with one dataset
    :return: dict of mean ranks and dict of stdevs
    """
    if len(results) == 0:
        raise ValueError('Dict of results is empty')

    kernels_count = len(results[0])

    mean_ranks = {}
    stdevs = {}

    ranks = {}
    for result in results:
        if len(result) != kernels_count:
            raise ValueError('There are different amount of kernels in some experiments')
        ranked_data = __get_ranked(result)
        for kernel, rank in ranked_data.items():
            if not kernel in ranks:
                ranks[kernel] = []
            ranks[kernel].append(rank)

    for kernel, kernel_ranks in ranks.items():
        mean_ranks[kernel] = mean(kernel_ranks)
        stdevs[kernel] = stdev(kernel_ranks)

    return mean_ranks, stdevs


def draw_stats(results, **kwargs):
    """

    :param results: list of dicts (key - kernel name, value - performance metric), one dict per experiment with one dataset
    :param kwargs:
            title : title of graph
    """
    title = kwargs.get('title', 'Results')
    mean_ranks, stdevs = get_stats(results)
    kernels = list(results[0].keys())
    plt.errorbar(mean_ranks.values(), kernels, None, stdevs.values(), linestyle='None', marker='o')
    plt.title(title)
    plt.show()


def draw_cd(results, **kwargs):
    """
    :param results: list of dicts (key - kernel name, value - performance metric), one dict per experiment with one dataset
    :param kwargs:
            title : title of graph
    """
    title = kwargs.get('title', 'Results')
    mean_ranks, stdevs = get_stats(results)
    kernels = list(results[0].keys())
    cd = Orange.evaluation.compute_CD(mean_ranks.values(), len(results))
    Orange.evaluation.graph_ranks(mean_ranks.values(), kernels, cd=cd, width=6, textspace=1.5)
    plt.title(title)
    plt.show()


def friedman_test(results):
    """
    :param results: list of dicts (key - kernel name, value - performance metric), one dict per experiment with one dataset
    :return:
        statistic : float
            the test statistic, correcting for ties
        pvalue : float
            the associated p-value assuming that the test statistic has a chi
            squared distribution
    """
    processed_results = {}
    for result in results:
        for kernel, kernel_performance in result.items():
            if not kernel in processed_results:
                processed_results[kernel] = []
            processed_results[kernel].append(kernel_performance)
    return friedmanchisquare(*list(processed_results.values()))


def __get_ranked(results):
    """

    :param results: dict (key - kernel name, value - performance metric)
    :return: dict (key - kernel name, value -  rank)
    """
    rank_info = ((1 + len(results)) - rankdata(list(results.values())))
    to_return = {}
    kernels = list(results.keys())
    for i in range(0, len(results)):
        kernel = kernels[i]
        to_return[kernel] = rank_info[i]
    return to_return
