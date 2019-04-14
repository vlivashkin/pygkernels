from abc import ABC

import numpy as np
import numpy.matlib
from sklearn.utils import deprecated

from . import scaler
from .kernel import Kernel
from .shortcuts import get_D, get_L


# Avrachenkov: Kernels on Graphs as Proximity Measures
# Implementation by Dmytro Rubanov

class _KernelNew(Kernel, ABC):
    name, default_scaler = None, None

    @staticmethod
    def mat_exp(M, n=15):
        A = np.matlib.eye(M.shape[0])
        R = A.copy()
        for i in range(1, n):
            A *= M / i
            R += A
        return R


@deprecated()
class Katz_R(_KernelNew):
    name, default_scaler = 'Katz R', scaler.Rho

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


@deprecated()
class Estrada_R(_KernelNew):
    name, default_scaler = 'Estrada R', scaler.Fraction

    def get_K(self, t):
        K = _KernelNew.mat_exp(t * self.A)
        return np.array(np.log(K))


@deprecated()
class Heat_R(_KernelNew):  # this is logHeat, actually
    name, default_scaler = 'Heat R', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.L = np.matlib.array(get_L(A))

    def get_K(self, t):
        K = _KernelNew.mat_exp(- t * self.L, n=50)
        if np.any(K < 0):
            # logging.info(t, "K < 0")
            return None
        return np.array(np.log(K))


@deprecated()
class NormalizedHeat_R(_KernelNew):
    name, default_scaler = 'logNHeat R', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        L = D - A
        D_12 = np.linalg.inv(np.sqrt(D))
        self.Ll = D_12.dot(L).dot(D_12)

    def get_K(self, t):
        K = _KernelNew.mat_exp(-t * self.Ll, n=50)
        if np.any(K < 0):
            # logging.info(t, "K < 0")
            return None
        return np.array(np.log(K))


@deprecated()
class RegularizedLaplacian_R(_KernelNew):
    name, default_scaler = 'RegularizedLaplacian R', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        self.L = D - A

    def get_K(self, beta):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) + beta * self.L)
        if np.any(K < 0):
            # logging.info(beta, "K < 0")
            return None
        return np.array(np.log(K))


@deprecated()
class logPPR_R(_KernelNew):
    name, default_scaler = 'logPPR R', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        D = get_D(A)
        self.P = np.linalg.inv(D).dot(A)

    def get_K(self, alpha):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - alpha * self.P)
        if np.any(K < 0):
            # logging.info(alpha, "K < 0")
            return None
        return np.array(np.log(K))


@deprecated()
class logModifPPR_R(_KernelNew):
    name, default_scaler = 'logModifPPR R', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.D = get_D(A)

    def get_K(self, alpha):
        K = np.linalg.inv(self.D - alpha * self.A)
        if np.any(K < 0):
            # logging.info(alpha, "K < 0")
            return None
        return np.array(np.log(K))


@deprecated()
class logHeatPPR_R(_KernelNew):
    name, default_scaler = 'logHeatPPR R', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.D = get_D(A)
        self.P = np.linalg.inv(self.D).dot(A)

    def get_K(self, t):
        K = _KernelNew.mat_exp(- t * (np.matlib.eye(self.A.shape[0]) - self.P))
        if np.any(K < 0):
            # logging.info(t, "K < 0")
            return None
        return np.array(np.log(K))
