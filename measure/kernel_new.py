import numpy as np
import numpy.matlib

from measure import scaler
from measure.kernel import Kernel, getD


class KernelNew(Kernel):
    @staticmethod
    def get_all_new():
        return [
            Katz, Estrada, Heat,
            NormalizedHeat,
            RegularizedLaplacian,
            PersonalizedPageRank,
            ModifiedPersonalizedPageRank,
            HeatPersonalizedPageRank
        ]

    @staticmethod
    def get_ALL():
        return Kernel.get_all_H_plus_RSP_FE() + KernelNew.get_all_new()

    @staticmethod
    def mat_exp(M, n=15):
        A = np.matlib.eye(M.shape[0])
        R = A.copy()
        for i in range(1, n):
            A *= M / i
            R += A
        return R


class Katz(KernelNew):
    def __init__(self, A: np.ndarray):
        super().__init__('NEW Katz', scaler.Rho, A)
        self.rad = self._get_radius(self.A)

    def _get_radius(self, M: np.ndarray):
        val, vec = np.linalg.eig(M)
        return np.max(np.abs(val))

    def getK(self, t):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - t * self.A)
        return np.log(K)


class Estrada(KernelNew):  # !
    def __init__(self, A: np.ndarray):
        super().__init__('NEW Estrada', scaler.Fraction, A)

    def getK(self, t):
        K = KernelNew.mat_exp(t * self.A)
        return np.log(K)


class Heat(KernelNew):  # !
    def __init__(self, A: np.ndarray):
        super().__init__('NEW Heat', scaler.Fraction, A)
        D = getD(A)
        self.L = D - A

    def getK(self, t):
        K = KernelNew.mat_exp(- t * self.L, n=50)
        if np.any(K < 0):
            print(t, "K < 0")
            return None
        return np.log(K)


class NormalizedHeat(KernelNew):  # !
    def __init__(self, A: np.ndarray):
        super().__init__('NEW NormalizedHeat', scaler.Fraction, A)
        D = getD(A)
        L = D - A
        D_12 = np.linalg.inv(np.sqrt(D))
        self.Ll = D_12.dot(L).dot(D_12)

    def getK(self, t):
        K = KernelNew.mat_exp(-t * self.Ll, n=50)
        if np.any(K < 0):
            print(t, "K < 0")
            return None
        return np.log(K)


class RegularizedLaplacian(KernelNew):
    def __init__(self, A: np.ndarray):
        super().__init__('NEW RegularizedLaplacian', scaler.Fraction, A)
        D = getD(A)
        self.L = D - A

    def getK(self, beta):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) + beta * self.L)
        if np.any(K < 0):
            print(beta, "K < 0")
            return None
        return np.log(K)


class PersonalizedPageRank(KernelNew):
    def __init__(self, A: np.ndarray):
        super().__init__('NEW PersonalizedPageRank', scaler.Linear, A)
        D = getD(A)
        self.P = np.linalg.inv(D).dot(A)

    def getK(self, alpha):
        K = np.linalg.inv(np.matlib.eye(self.A.shape[0]) - alpha * self.P)
        if np.any(K < 0):
            print(alpha, "K < 0")
            return None
        return np.log(K)


class ModifiedPersonalizedPageRank(KernelNew):
    def __init__(self, A: np.ndarray):
        super().__init__('NEW ModifiedPersonalizedPageRank', scaler.Linear, A)
        self.D = getD(A)

    def getK(self, alpha):
        K = np.linalg.inv(self.D - alpha * self.A)
        if np.any(K < 0):
            print(alpha, "K < 0")
            return None
        return np.log(K)


class HeatPersonalizedPageRank(KernelNew):  # !
    def __init__(self, A: np.ndarray):
        super().__init__('NEW HeatPersonalizedPareRank', scaler.Fraction, A)
        self.D = getD(A)
        self.P = np.linalg.inv(self.D).dot(A)

    def getK(self, t):
        K = KernelNew.mat_exp(- t * (np.matlib.eye(self.A.shape[0]) - self.P))
        if np.any(K < 0):
            print(t, "K < 0")
            return None
        return np.log(K)
