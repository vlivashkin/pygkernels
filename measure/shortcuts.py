import numpy as np
from scipy.sparse.csgraph import johnson


def normalize(dm: np.ndarray):
    return dm / dm.std() if dm.std() != 0 else dm


def getD(A: np.ndarray):
    return np.diag(np.sum(A, axis=0))


def getL(A: np.ndarray):
    return getD(A) - A


def H0toH(H0: np.ndarray):
    """
    H = element - wise log(H0)
    """
    return np.log(H0)


def HtoD(H: np.ndarray):
    """
    D = (h * 1^T + 1 * h^T - H - H ^ T) / 2
    """
    size = H.shape[0]
    h = np.diagonal(H).reshape(-1, 1)
    i = np.ones((size, 1))
    return 0.5 * ((h.dot(i.transpose()) + i.dot(h.transpose())) - H - H.transpose())


def DtoK(D: np.ndarray):
    """
    K = -1 / 2 HΔH
    """
    size = D.shape[0]
    H = np.eye(size) - (np.ones((size, size)) / size)
    return -0.5 * np.dot(H, D).dot(H)


def D_SP(A: np.ndarray):
    """
    Johnson's Algorithm
    """
    return johnson(A, directed=False)


def H_R(A: np.ndarray):
    """
    H = (L + J)^{-1}
    """
    size = A.shape[0]
    L = getL(A)
    J = np.ones((size, size)) / size
    return np.linalg.pinv(L + J)


def H_CCT(A: np.ndarray):
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
