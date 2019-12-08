import numpy as np
from sklearn.utils import deprecated
from numba import jit


@deprecated()
def normalize(dm):
    return dm / dm.std() if dm.std() != 0 else dm


@jit
def get_D(A):
    return np.diag(np.sum(A, axis=0))


@jit
def get_D_1(A):
    """
    D_1 = D^{-1}
    """
    return np.diag(1. / np.sum(A, axis=0))


@jit
def get_L(A):
    return get_D(A) - A


@jit
def get_normalized_L(A):
    D = get_D(A)
    L = get_L(A)
    D_12 = np.linalg.inv(np.sqrt(D))
    return D_12.dot(L).dot(D_12)


def H0_to_H(H0):
    """
    H = element - wise log(H0)
    """
    H = np.log(H0)
    H[np.isnan(H)] = 0
    return H


def H_to_D(H):
    """
    D = (h * 1^T + 1 * h^T - H - H ^ T) / 2
    """
    size = H.shape[0]
    h = np.diagonal(H).reshape(-1, 1)
    i = np.ones((size, 1))
    return 0.5 * ((h.dot(i.transpose()) + i.dot(h.transpose())) - H - H.transpose())


@jit
def D_to_K(D):
    """
    K = -1 / 2 HΔH
    H = I - E/n
    """
    size = D.shape[0]
    I, E = np.eye(size), np.ones((size, size))
    H = I - (E / size)
    K = -0.5 * H.dot(D).dot(H)
    return K
