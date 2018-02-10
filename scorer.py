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
