from abc import ABC

import networkx as nx
import numpy as np
from scipy.linalg import expm

import pygraphs.measure.shortcuts as h
from pygraphs.measure import scaler
from pygraphs.measure.scaler import Scaler


class Kernel(ABC):
    EPS = 10 ** -10
    name, _default_scaler = None, None
    _parent_distance_class, _parent_kernel_class = None, None

    def __init__(self, A: np.ndarray):
        assert not (self._parent_distance_class and self._parent_kernel_class)
        if self._parent_distance_class:
            self._parent_kernel = None
            self._parent_distance = self._parent_distance_class(A)
            self._default_scaler = self._parent_distance._default_scaler
        elif self._parent_kernel_class:
            self._parent_kernel = self._parent_kernel_class(A)
            self._parent_distance = None
            self._default_scaler = self._parent_kernel._default_scaler
        self.scaler: Scaler = self._default_scaler(A)
        self.A = A

    def get_K(self, param):
        if self._parent_distance:  # use D -> K transform
            D = self._parent_distance.get_D(param)
            return h.D_to_K(D)
        elif self._parent_kernel:  # use element-wise log transform
            H0 = self._parent_kernel.get_K(param)
            return h.ewlog(H0)
        else:
            raise NotImplementedError()


class CT_H(Kernel):
    name, _default_scaler = 'CT', scaler.Linear

    def CT(self):
        """
        Commute time kernel function.
        Ref: Fouss (2007)
        """
        G = nx.from_numpy_matrix(self.A)
        L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes)).toarray().astype('float')
        K = np.linalg.pinv(L)
        return K

    def resistance_kernel(self):
        """
        H = (L + J)^{-1}
        """
        size = self.A.shape[0]
        L = h.get_L(self.A)
        J = np.ones((size, size)) / size
        return np.linalg.pinv(L + J)

    def resistance_kernel2(self):
        """
        H = (I + L)^{-1}
        """
        size = self.A.shape[0]
        I = np.eye(size)
        L = h.get_L(self.A)
        return np.linalg.pinv(I + L)

    def get_K(self, param):
        return self.resistance_kernel()


class Katz_H(Kernel):
    name, _default_scaler = 'Katz', scaler.Rho

    def get_K(self, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * self.A)


class For_H(Kernel):
    name, _default_scaler = 'For', scaler.Fraction

    def get_K(self, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.inv(np.eye(size) + t * h.get_L(self.A))


class Comm_H(Kernel):
    name, _default_scaler = 'Comm', scaler.Fraction

    def get_K(self, t):
        """
        H0 = exp(tA)
        """
        return expm(t * self.A)  # if t < 30 else None


class Heat_H(Kernel):
    name, _default_scaler = 'Heat', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.L = h.get_L(self.A)

    def get_K(self, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * self.L)


class NHeat_H(Kernel):
    name, _default_scaler = 'NHeat', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.nL = h.get_normalized_L(A)

    def get_K(self, t):
        """
        H0 = exp(-t*nL)
        """
        return expm(-t * self.nL)


class SCT_H(Kernel):
    name, _default_scaler = 'SCT', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.K_CT = np.linalg.pinv(h.get_L(self.A))
        self.sigma = self.K_CT.std()
        self.Kds = self.K_CT / (self.sigma + self.EPS)

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class SCCT_H(Kernel):
    name, _default_scaler = 'SCCT', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.K_CCT = self.H_CCT(A)
        self.sigma = self.K_CCT.std()
        self.Kds = self.K_CCT / self.sigma

    def H_CCT(self, A: np.ndarray):
        """
        H = I - E / n
        M = D^{-1/2}(A - dd^T/vol(G))D^{-1/2},
            d is a vector of the diagonal elements of D,
            vol(G) is the volume of the graph (sum of all elements of A)
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

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class PPR_H(Kernel):
    name, _default_scaler = 'PPR', scaler.Linear

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = h.get_P(A)

    def get_K(self, alpha):
        """
        H = (I - αP)^{-1}
        """
        return np.linalg.inv(self.I - alpha * self.P)


class ModifPPR_H(Kernel):
    name, _default_scaler = 'ModifPPR', scaler.Linear

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.D = h.get_D(A)

    def get_K(self, alpha):
        """
        H = (I - αP)^{-1}*D^{-1} = (D - αA)^{-1}
        """
        return np.linalg.inv(self.D - alpha * self.A)


class HeatPPR_H(Kernel):
    name, _default_scaler = 'HeatPPR', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = h.get_P(A)

    def get_K(self, t):
        """
        H = expm(-t(I - P))
        """
        return expm(-t * (self.I - self.P))


class DF_H(Kernel):
    name, _default_scaler = 'DF', scaler.Fraction

    def __init__(self, A: np.ndarray, n_iter=30):
        super().__init__(A)
        self.n_iter = n_iter
        self.dfac = self.calc_double_factorial(n_iter)

    @staticmethod
    def calc_double_factorial(max_k):
        mem = np.zeros((max_k + 1,))
        mem[0], mem[1] = 1, 1
        for i in range(2, max_k + 1):
            mem[i] = mem[i - 2] * i
        return mem

    def get_K(self, t):
        tA = t * self.A
        K, tA_k = np.eye(tA.shape[0]), np.eye(tA.shape[0])
        for i in range(1, self.n_iter):
            tA_k = tA_k.dot(tA)
            K += tA_k / self.dfac[i]
        return K


class Abs_H(Kernel):
    name, _default_scaler = 'Abs', scaler.Fraction

    def __init__(self, A: np.ndarray):
        super().__init__(A)
        self.L = h.get_L(A)

    def get_K(self, t):
        return np.linalg.pinv(t * self.A + self.L)
