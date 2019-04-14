from abc import ABC

import networkx as nx
import numpy as np
from scipy.linalg import expm

from pygraphs.measure import scaler
from pygraphs.measure.shortcuts import get_D, get_L, get_normalized_L, D_to_K, H0_to_H


class Kernel(ABC):
    name, default_scaler = None, None
    parent_distance_class, parent_kernel_class = None, None

    def __init__(self, A):
        self.scaler = self.default_scaler(A)

        assert not self.parent_distance_class or not self.parent_kernel_class
        self.parent_distance = self.parent_distance_class(A) if self.parent_distance_class else None
        self.parent_kernel = self.parent_kernel_class(A) if self.parent_kernel_class else None

        self.A = A

    def get_K(self, param):
        if self.parent_distance:  # use D -> K transform
            D = self.parent_distance.get_D(param)
            return D_to_K(D)
        elif self.parent_kernel:  # use H0 -> H transform
            H0 = self.parent_kernel.get_K(param)
            return H0_to_H(H0)
        else:
            raise NotImplementedError()


class CT_H(Kernel):
    name, default_scaler = 'CT', scaler.Linear

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
        L = get_L(self.A)
        J = np.ones((size, size)) / size
        return np.linalg.pinv(L + J)

    def resistance_kernel2(self):
        """
        H = (I + L)^{-1}
        """
        size = self.A.shape[0]
        I = np.eye(size)
        L = get_L(self.A)
        return np.linalg.pinv(I + L)

    def get_K(self, param):
        return self.resistance_kernel()


class pWalk_H(Kernel):
    name, default_scaler = 'pWalk', scaler.Rho

    def get_K(self, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * self.A)


class For_H(Kernel):
    name, default_scaler = 'For', scaler.Fraction

    def get_K(self, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) + t * get_L(self.A))


class Comm_H(Kernel):
    name, default_scaler = 'Comm', scaler.Fraction

    def get_K(self, t):
        """
        H0 = exp(tA)
        """
        return expm(t * self.A) if t < 30 else None


class Heat_H(Kernel):
    name, default_scaler = 'Heat', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.L = get_L(self.A)

    def get_K(self, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * self.L)


class NHeat_H(Kernel):
    name, default_scaler = 'NHeat', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.nL = get_normalized_L(A)

    def get_K(self, t):
        """
        H0 = exp(-t*nL)
        """
        return expm(-t * self.nL)


class SCT_H(Kernel):
    name, default_scaler = 'SCT', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.K_CT = np.linalg.pinv(get_L(self.A))
        self.sigma = self.K_CT.std()
        self.Kds = self.K_CT / self.sigma

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class SCCT_H(Kernel):
    name, default_scaler = 'SCCT', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.K_CCT = self.H_CCT(A)
        self.sigma = self.K_CCT.std()
        self.Kds = self.K_CCT / self.sigma

    def H_CCT(self, A):
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

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class PPR_H(Kernel):
    name, default_scaler = 'PPR', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = np.linalg.inv(get_D(A)).dot(A)

    def get_K(self, alpha):
        return np.linalg.inv(self.I - alpha * self.P)


class ModifPPR_H(Kernel):
    name, default_scaler = 'ModifPPR', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.D = get_D(A)

    def get_K(self, alpha):
        return np.linalg.inv(self.D - alpha * self.A)


class HeatPPR_H(Kernel):
    name, default_scaler = 'HeatPPR', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.I = np.eye(A.shape[0])
        self.P = np.linalg.inv(get_D(A)).dot(A)

    def get_K(self, t):
        return expm(-t * (self.I - self.P))
