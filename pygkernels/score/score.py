import itertools
from collections import defaultdict

import numpy as np
from scipy import stats


def max_accuracy(y_true, y_pred):
    names_true, names_pred, max_result = list(set(y_true)), list(set(y_pred)), 0
    for perm in itertools.permutations(names_pred):
        acc = np.average([1. if names_true.index(ti) == perm.index(pi) else 0. for ti, pi in zip(y_true, y_pred)])
        if acc > max_result:
            max_result = acc
    return max_result


def rand_index(y_true, y_pred):
    good, all = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_pred)):
            if (y_true[i] == y_true[j]) == (y_pred[i] == y_pred[j]):
                good += 1
            all += 1
    return good / all


def triplet_measure(y_true, D_pred):
    good, all = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            for k in range(j + 1, len(y_true)):
                items_by_class = defaultdict(list)
                for item in [i, j, k]:
                    items_by_class[y_true[item]].append(item)
                if len(items_by_class.keys()) == 2:  # seems to two items in one class
                    key1, key2 = items_by_class.keys()
                    if len(items_by_class[key1]) == 2:
                        same_class_pair, another_class_item = items_by_class[key1], items_by_class[key2][0]
                    else:
                        same_class_pair, another_class_item = items_by_class[key2], items_by_class[key1][0]
                    # check d(s1, s2) < d(s1, a) and d(s1, s2) < d(s2, a)
                    d_s1_s2 = D_pred[same_class_pair[0], same_class_pair[1]]
                    d_s1_a = D_pred[same_class_pair[0], another_class_item]
                    d_s2_a = D_pred[same_class_pair[1], another_class_item]
                    if d_s1_s2 < d_s1_a:
                        good += 1
                    if d_s1_s2 < d_s2_a:
                        good += 1
                    all += 2
    return good / all


def ranking(measure1_ari, measure2_ari):
    assert measure1_ari.shape == measure2_ari.shape
    n = measure1_ari.shape[0]

    # 1. генерируем ранги
    measure1_rank = stats.rankdata(-measure1_ari)
    measure2_rank = stats.rankdata(-measure2_ari)

    # 2. Для каждой пары мер считаем сумму квадратов разностей
    sum_sq_delta = np.sum(np.power(measure1_rank - measure2_rank, 2))

    # 3. По формуле Спирмена считаем элементы матрицы корреляций
    return 1 - (6 * sum_sq_delta) / ((n - 1) * n * (n + 1))


def copeland(results):
    scores = defaultdict(lambda: 0)
    for a, b in list(itertools.combinations(results, 2)):

        if a[1] > b[1]:
            scores[a[0]] += 1
            scores[b[0]] -= 1
        elif a[1] < b[1]:
            scores[a[0]] -= 1
            scores[b[0]] += 1
    return scores


def _getMs(comms1, comms2):
    if len(comms1) != len(comms2):
        raise ValueError
    l = len(comms1)
    m1 = max(comms1) + 1
    m2 = max(comms2) + 1
    M = [[sum(1 for v in range(l) if comms1[v] == i and comms2[v] == j) for j in range(m2)] for i in range(m1)]
    return np.array(M)


def _getMatch(M, perm):
    return sum(M[i, j] if i < M.shape[0] and j < M.shape[1] else 0 for i, j in enumerate(perm))


def FC(comms1, comms2):
    l = len(comms1)
    M = _getMs(comms1, comms2)
    return 1 - 1 / l * max(_getMatch(M, perm) for perm in itertools.permutations(range(max(M.shape))))

def modularity(A: np.array, partition):
    """
    Simplified version only for undirected graphs
    """
    n_edges = np.sum(A)
    degrees = np.sum(A, axis=1, keepdims=True)

    Q_items = A + np.diagonal(A) - degrees.dot(degrees.T) / n_edges
    Q = 0
    for class_name in set(partition):
        mask = np.array(partition) == class_name
        Q += np.sum(Q_items[mask][:, mask])
    return Q / n_edges