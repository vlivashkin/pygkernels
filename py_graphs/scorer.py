import itertools

import numpy as np


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
