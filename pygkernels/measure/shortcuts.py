import numpy as np
from sklearn.utils import deprecated


@deprecated()
def normalize(dm):
    return dm / dm.std() if dm.std() != 0 else dm


def get_D(A):
    """
    Degree matrix
    """
    return np.diag(np.sum(A, axis=0))


def get_L(A):
    """
    Ordinary (or combinatorial) Laplacian matrix.
    L = D - A
    """
    return get_D(A) - A


def get_normalized_L(A):
    """
    Normalized Laplacian matrix.
    L = D^{-1/2}*L*D^{-1/2}
    """
    D = get_D(A)
    L = get_L(A)
    D_12 = np.linalg.inv(np.sqrt(D))
    return D_12.dot(L).dot(D_12)


def get_P(A):
    """
    Markov matrix.
    P = D^{-1}*A
    """
    D = get_D(A)
    return np.linalg.inv(D).dot(A)


def ewlog(K):
    """
    logK = element-wise log(K)
    """
    mask = K <= 0
    K[mask] = 1
    logK = np.log(K)
    logK[mask] = -np.inf
    return logK


def K_to_D(K):
    """
    D = (k * 1^T + 1 * k^T - K - K^T) / 2
    k = diag(K)
    """
    size = K.shape[0]
    k = np.diagonal(K).reshape(-1, 1)
    i = np.ones((size, 1))
    return 0.5 * ((k.dot(i.transpose()) + i.dot(k.transpose())) - K - K.transpose())


def D_to_K(D):
    """
    K = -1/2 H*D*H
    H = I - E/n
    """
    size = D.shape[0]
    I, E = np.eye(size), np.ones((size, size))
    H = I - (E / size)
    K = -0.5 * H.dot(D).dot(H)
    return K
