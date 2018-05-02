import functools
import warnings

import networkx as nx
import numpy as np
from scipy.sparse.csgraph._shortest_path import shortest_path

from pykernels.graph import ShortestPath


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
def normalize(dm):
    return dm / dm.std() if dm.std() != 0 else dm


def get_D(A):
    return np.diag(np.sum(A, axis=0))


def get_D_1(A):
    """
    D_1 = D^{-1}
    """
    return np.diag(1. / np.sum(A, axis=0))


def get_L(A):
    return get_D(A) - A


@deprecated
def sp_distance(A):
    return shortest_path(A, directed=False)


def sp_kernel(A):
    return D_to_K(sp_distance(A))
    # return ShortestPath().gram(A)


def CT(A):
    """
    Commute time kernel function.
    Ref: Fouss (2007)
    """
    G = nx.from_numpy_matrix(A)
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).toarray().astype('float')
    K = np.linalg.pinv(L)
    return K

def resistance_kernel(A):
    """
    H = (L + J)^{-1}
    """
    size = A.shape[0]
    L = get_L(A)
    J = np.ones((size, size)) / size
    return np.linalg.pinv(L + J)

def resistance_kernel2(A):
    """
    H = (I + L)^{-1}
    """
    size = A.shape[0]
    I = np.eye(size)
    L = get_L(A)
    return np.linalg.pinv(I + L)


def commute_distance(A):
    """Original code copyright (C) Ulrike Von Luxburg, Python
        implementation by James McDermott."""

    size = A.shape[0]
    L = get_L(A)

    Linv = np.linalg.inv(L + np.ones(L.shape) / size) - np.ones(L.shape) / size

    Linv_diag = np.diag(Linv).reshape((size, 1))
    Rexact = Linv_diag * np.ones((1, size)) + np.ones((size, 1)) * Linv_diag.T - 2 * Linv

    # convert from a resistance distance to a commute time distance
    vol = np.sum(A)
    Rexact *= vol

    return Rexact


def H0_to_H(H0):
    """
    H = element - wise log(H0)
    """
    return np.log(H0)


def H_to_D(H):
    """
    D = (h * 1^T + 1 * h^T - H - H ^ T) / 2
    """
    size = H.shape[0]
    h = np.diagonal(H).reshape(-1, 1)
    i = np.ones((size, 1))
    return 0.5 * ((h.dot(i.transpose()) + i.dot(h.transpose())) - H - H.transpose())


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


def H_CCT(A):
    """
    H = I - E / n
    M = D^{-1/2}(A - dd^T/vol(G))D^{-1/2},
        d is a vector of the diagonal elements of D,
        vol(G) is the volume of the graph
    K_CCT = HD^{-1/2}M(I - M)^{-1}MD^{-1/2}H
    """
    size = A.shape[0]
    I = np.eye(size)
    d = np.sum(A, axis=0).reshape((-1, 1))
    D05 = np.diag(np.power(d, -0.5)[:, 0])
    H = np.eye(size) - np.ones((size, size)) / size
    volG = np.sum(A)
    M = D05.dot(A - d.dot(d.transpose()) / volG).dot(D05)
    return H.dot(D05).dot(M).dot(np.linalg.pinv(I - M)).dot(M).dot(D05).dot(H)
