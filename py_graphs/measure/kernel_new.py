import numpy as np
import numpy.matlib

from . import kernel
from . import scaler
from .kernel import Kernel
from .shortcuts import get_D, get_L


# Avrachenkov: Kernels on Graphs as Proximity Measures
# Implementation from Dmytro Rubanov

class KernelNew(Kernel):
    name, default_scaler = None, None

    @staticmethod
    def mat_exp(M, n=15):
        A = np.matlib.eye(M.shape[0])
        R = A.copy()
        for i in range(1, n):
            A *= M / i
            R += A
        return R


class Katz(KernelNew):
    name, default_scaler = 'NEW Katz', scaler.Rho

    def __init__(self, A):
        super().__init__(A)
        self.rad = self._get_radius(self.A)

    @staticmethod
    def _get_radius(M):
        val, _ = np.linalg.eig(M)
        return np.max(np.abs(val))

    def get_K(self, t):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - t * self.A)
        return np.array(np.log(K))


class Estrada(KernelNew):
    name, default_scaler = 'NEW Estrada', scaler.Fraction

    def get_K(self, t):
        K = KernelNew.mat_exp(t * self.A)
        return np.array(np.log(K))


class Heat(KernelNew):  # this is logHeat, actually
    name, default_scaler = 'NEW Heat', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.L = np.matlib.array(get_L(A))

    def get_K(self, t):
        K = KernelNew.mat_exp(- t * self.L, n=50)
        if np.any(K < 0):
            # print(t, "K < 0")
            return None
        return np.array(np.log(K))


class NormalizedHeat(KernelNew):
    name, default_scaler = 'NEW NormalizedHeat', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        L = D - A
        D_12 = np.linalg.inv(np.sqrt(D))
        self.Ll = D_12.dot(L).dot(D_12)

    def get_K(self, t):
        K = KernelNew.mat_exp(-t * self.Ll, n=50)
        if np.any(K < 0):
            # print(t, "K < 0")
            return None
        return np.array(np.log(K))


class RegularizedLaplacian(KernelNew):
    name, default_scaler = 'NEW RegularizedLaplacian', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        self.L = D - A

    def get_K(self, beta):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) + beta * self.L)
        if np.any(K < 0):
            # print(beta, "K < 0")
            return None
        return np.array(np.log(K))


class PersonalizedPageRank(KernelNew):
    name, default_scaler = 'NEW PersonalizedPageRank', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        self.P = np.linalg.inv(D).dot(A)

    def get_K(self, alpha):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - alpha * self.P)
        if np.any(K < 0):
            # print(alpha, "K < 0")
            return None
        return np.array(np.log(K))


class ModifiedPersonalizedPageRank(KernelNew):
    name, default_scaler = 'NEW ModifiedPersonalizedPageRank', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.D = get_D(A)

    def get_K(self, alpha):
        K = np.linalg.inv(self.D - alpha * self.A)
        if np.any(K < 0):
            # print(alpha, "K < 0")
            return None
        return np.array(np.log(K))


class HeatPersonalizedPageRank(KernelNew):
    name, default_scaler = 'NEW HeatPersonalizedPageRank', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.D = get_D(A)
        self.P = np.linalg.inv(self.D).dot(A)

    def get_K(self, t):
        K = KernelNew.mat_exp(- t * (np.matlib.eye(self.A.shape[0]) - self.P))
        if np.any(K < 0):
            # print(t, "K < 0")
            return None
        return np.array(np.log(K))


NEW_kernels = [
    Katz, Estrada, Heat,
    NormalizedHeat,
    RegularizedLaplacian,
    PersonalizedPageRank,
    ModifiedPersonalizedPageRank,
    HeatPersonalizedPageRank
]

ALL_kernels = kernel.H_kernels_plus_RSP_FE + NEW_kernels
