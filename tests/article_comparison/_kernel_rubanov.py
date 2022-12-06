from abc import ABC

import numpy as np
import numpy.matlib
from numpy import warnings

import pygkernels.measure.shortcuts as h
from pygkernels.measure import scaler
from pygkernels.measure.kernel import Kernel


# Avrachenkov: Kernels on Graphs as Proximity Measures
# Implementation by Dmytro Rubanov


class _KernelR(Kernel, ABC):
    name, _default_scaler = None, None

    @staticmethod
    def mat_exp(M, n=15):
        A = np.matlib.eye(M.shape[0])
        R = A.copy()
        for i in range(1, n):
            A *= M / i
            R += A
        return R


class Katz_R(_KernelR):
    name, _default_scaler = "Katz R", scaler.Rho

    def __init__(self, A):
        super().__init__(A)
        self.rad = self._get_radius(self.A)

    @staticmethod
    def _get_radius(M):
        val, _ = np.linalg.eig(M)
        return np.max(np.abs(val))

    def get_K(self, t):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - t * self.A)
            return np.log(np.array(K))


class Estrada_R(_KernelR):
    name, _default_scaler = "Estrada R", scaler.Fraction

    def get_K(self, t):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = _KernelR.mat_exp(t * self.A)
            return np.log(np.array(K))


class Heat_R(_KernelR):  # this is logHeat, actually
    name, _default_scaler = "Heat R", scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.L = np.matlib.array(h.get_L(A))

    def get_K(self, t):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = _KernelR.mat_exp(-t * self.L, n=50)
            if np.any(K < 0):
                # logging.info(t, "K < 0")
                return None
            return np.log(np.array(K))


class NormalizedHeat_R(_KernelR):
    name, _default_scaler = "logNHeat R", scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = h.get_D(A)
        L = D - A
        D_12 = np.linalg.inv(np.sqrt(D))
        self.Ll = D_12.dot(L).dot(D_12)

    def get_K(self, t):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = _KernelR.mat_exp(-t * self.Ll, n=50)
            if np.any(K < 0):
                return None
            K[K == 0] = self.EPS
            return np.log(np.array(K))


class RegularizedLaplacian_R(_KernelR):
    name, _default_scaler = "RegularizedLaplacian R", scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        D = h.get_D(A)
        self.L = D - A

    def get_K(self, beta):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) + beta * self.L)
            if np.any(K < 0):
                return None
            K[K == 0] = self.EPS
            return np.log(np.array(K))


class logPPR_R(_KernelR):
    name, _default_scaler = "logPPR R", scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        D = h.get_D(A)
        self.P = np.linalg.inv(D).dot(A)

    def get_K(self, alpha):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - alpha * self.P)
            if np.any(K < 0):
                return None
            K[K == 0] = self.EPS
            return np.log(np.array(K))


class logModifPPR_R(_KernelR):
    name, _default_scaler = "logModifPPR R", scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.D = h.get_D(A)

    def get_K(self, alpha):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = np.linalg.inv(self.D - alpha * self.A)
            if np.any(K < 0):
                return None
            K[K == 0] = self.EPS
            return np.log(np.array(K))


class logHeatPR_R(_KernelR):
    name, _default_scaler = "logHeatPR R", scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.D = h.get_D(A)
        self.P = np.linalg.inv(self.D).dot(A)

    def get_K(self, t):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            K = _KernelR.mat_exp(-t * (np.matlib.eye(self.A.shape[0]) - self.P))
            if np.any(K < 0):
                return None
            K[K == 0] = self.EPS
            return np.log(np.array(K))
