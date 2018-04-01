from scipy.linalg import expm

from . import distance, scaler
from .shortcuts import *


class Kernel:
    name, default_scaler = None, None

    def __init__(self, A, parent_distance=None):
        self.scaler = self.default_scaler(A)
        self.A = A
        self.parent_distance = parent_distance(A) if parent_distance is not None else None

    def get_K(self, param):
        D = self.parent_distance.get_D(param)
        return D_to_K(D)


class pWalk_H(Kernel):
    name, default_scaler = 'pWalk H', scaler.Rho

    def get_K(self, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * self.A)


class Walk_H(Kernel):
    name, default_scaler = 'Walk H', scaler.Rho

    def __init__(self, A):
        super().__init__(A)
        self.parent_distance = pWalk_H(self.A)

    def get_K(self, t):
        return H0_to_H(self.parent_distance.get_K(t))


class For_H(Kernel):
    name, default_scaler = 'For H', scaler.Fraction

    def get_K(self, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) + t * get_L(self.A))


class logFor_H(Kernel):
    name, default_scaler = 'logFor H', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.parent_distance = For_H(self.A)

    def get_K(self, t):
        return H0_to_H(self.parent_distance.get_K(t))


class Comm_H(Kernel):
    name, default_scaler = 'Comm H', scaler.Fraction

    def get_K(self, t):
        """
        H0 = exp(tA)
        """
        return expm(t * self.A) if t < 30 else None


class logComm_H(Kernel):
    name, default_scaler = 'logComm H', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.parent_distance = Comm_H(self.A)

    def get_K(self, t):
        return H0_to_H(self.parent_distance.get_K(t))


class Heat_H(Kernel):
    name, default_scaler = 'Heat H', scaler.Fraction

    def get_K(self, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * get_L(self.A))


class logHeat_H(Kernel):
    name, default_scaler = 'logHeat H', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.parent_distance = Heat_H(self.A)

    def get_K(self, t):
        return H0_to_H(self.parent_distance.get_K(t))


class SCT_H(Kernel):
    name, default_scaler = 'SCT H', scaler.Fraction

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
    name, default_scaler = 'SCCT H', scaler.Fraction

    def __init__(self, A):
        super().__init__(A)
        self.K_CCT = H_CCT(A)
        self.sigma = self.K_CCT.std()
        self.Kds = self.K_CCT / self.sigma

    def get_K(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class SPCT_H(Kernel):
    name, default_scaler = 'SP-CT H', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.H_SP = D_to_K(sp_distance(A))
        self.H_CT = 2 * resistance_kernel(A)

    def get_K(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.H_SP + (1. - lmbda) * self.H_CT

    def get_SP(self):
        return self.get_K(1)

    def get_CT(self):
        return self.get_K(0)


class pWalk_K(Kernel):
    name, default_scaler = 'pWalk K', scaler.Rho

    def __init__(self, A):
        super().__init__(A, distance.pWalk)


class Walk_K(Kernel):
    name, default_scaler = 'Walk K', scaler.Rho

    def __init__(self, A):
        super().__init__(A, distance.Walk)


class For_K(Kernel):
    name, default_scaler = 'For K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.For)


class logFor_K(Kernel):
    name, default_scaler = 'logFor K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.logFor)


class Comm_K(Kernel):
    name, default_scaler = 'Comm K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.Comm)


class logComm_K(Kernel):
    name, default_scaler = 'logComm K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.logComm)


class Heat_K(Kernel):
    name, default_scaler = 'Heat K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.Heat)


class logHeat_K(Kernel):
    name, default_scaler = 'logHeat K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.logHeat)


class SCT_K(Kernel):
    name, default_scaler = 'SCT K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.SCT)


class SCCT_K(Kernel):
    name, default_scaler = 'SCCT K', scaler.Fraction

    def __init__(self, A):
        super().__init__(A, distance.SCCT)


class RSP_vanilla_K(Kernel):
    name, default_scaler = 'RSP vanilla K', scaler.FractionReversed

    def __init__(self, A):
        super().__init__(A, distance.RSP_vanilla)


class FE_vanilla_K(Kernel):
    name, default_scaler = 'FE vanilla K', scaler.FractionReversed

    def __init__(self, A):
        super().__init__(A, distance.FE_vanilla)


class RSP_K(Kernel):
    name, default_scaler = 'RSP K', scaler.FractionReversed

    def __init__(self, A):
        super().__init__(A, distance.RSP)


class FE_K(Kernel):
    name, default_scaler = 'FE K', scaler.FractionReversed

    def __init__(self, A):
        super().__init__(A, distance.FE)


class SPCT_K(Kernel):
    name, default_scaler = 'SP-CT K', scaler.Linear

    def __init__(self, A):
        super().__init__(A, distance.SPCT)

    def get_SP(self):
        return self.get_K(1)

    def get_CT(self):
        return self.get_K(0)


H_kernels = [pWalk_H, Walk_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, SCT_H, SCCT_H, SPCT_H]
H_kernels_plus_RSP_FE = H_kernels + [RSP_K, FE_K]
K_kernels = [pWalk_K, Walk_K, For_K, logFor_K, Comm_K, logComm_K, Heat_K, logHeat_K, SCT_K, SCCT_K, RSP_K, FE_K, SPCT_K]
