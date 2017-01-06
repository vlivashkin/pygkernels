import statistics

from scipy.linalg import expm, expm2, expm3
from sklearn.preprocessing import normalize

from measure import distance
from measure import scale
from measure.shortcuts import *


class Kernel:
    def __init__(self, name, scale, parent_distance):
        self.name = name
        self.scale = scale
        self.parent_distance = parent_distance

    def getK(self, A: np.ndarray, param):
        D = self.parent_distance.getD(A, param)
        return DtoK(D)

    @staticmethod
    def get_all_H():
        return [pWalk_H(), Walk_H(), For_H(), logFor_H(), Comm_H(), logComm_H(),
                Heat_H(), logHeat_H(), SCT_H(), SCCT_H()]

    @staticmethod
    def get_all_H_plus_RSP_FE():
        return [pWalk_H(), Walk_H(), For_H(), logFor_H(), Comm_H(), logComm_H(),
                Heat_H(), logHeat_H(), SCT_H(), SCCT_H(), RSP_K(), FE_K()]

    @staticmethod
    def get_all_K():
        return [pWalk_K(), Walk_K(), For_K(), logFor_K(), Comm_K(), logComm_K(),
                Heat_K(), logHeat_K(), SCT_K(), SCCT_K(), RSP_K(), FE_K()]

class pWalk_H(Kernel):
    def __init__(self):
        super().__init__('pWalk H', scale.Rho, None)

    def getK(self, A: np.ndarray, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * A)


class Walk_H(Kernel):
    def __init__(self):
        super().__init__('Walk H', scale.Rho, None)

    def getK(self, A: np.ndarray, t):
        H0 = pWalk_H().getK(A, t)
        return H0toH(H0)


class For_H(Kernel):
    def __init__(self):
        super().__init__('For H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = A.shape[0]
        return np.linalg.pinv(np.eye(size) + t * getL(A))


class logFor_H(Kernel):
    def __init__(self):
        super().__init__('logFor H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        H0 = For_H().getK(A, t)
        return H0toH(H0)


class Comm_H(Kernel):
    def __init__(self):
        super().__init__('Comm H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        """
        H0 = exp(tA)
        """
        return expm(t * A)


class logComm_H(Kernel):
    def __init__(self):
        super().__init__('logComm H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        H0 = Comm_H().getK(A, t)
        return H0toH(H0)


class Heat_H(Kernel):
    def __init__(self):
        super().__init__('Heat H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * getL(A))


class logHeat_H(Kernel):
    def __init__(self):
        super().__init__('logHeat H', scale.Fraction, None)

    def getK(self, A: np.ndarray, t):
        H0 = Heat_H().getK(A, t)
        return H0toH(H0)


class SCT_H(Kernel):
    def __init__(self):
        super().__init__('SCT H', scale.Fraction, None)

    def getK(self, A: np.ndarray, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        K_CT = np.linalg.pinv(getL(A))
        sigma = statistics.stdev(K_CT)
        return 1 / (1 + expm(-alpha * K_CT / sigma))


class SCCT_H(Kernel):
    def __init__(self):
        super().__init__('SCCT H', scale.Fraction, None)

    def getK(self, A: np.ndarray, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        K_CCT = H_CCT(A)
        sigma = statistics.stdev(K_CCT)
        return 1 / (1 + expm(-alpha * K_CCT / sigma))


class SPCT_H(Kernel):
    def __init__(self):
        super().__init__('SP-CT H', scale.Linear, None)

    def getK(self, A: np.ndarray, lambda_):
        Hs = normalize(DtoK(D_SP(A)))
        Hc = normalize(H_R(A))
        return (1 - lambda_) * Hs + lambda_ * Hc


class pWalk_K(Kernel):
    def __init__(self):
        super().__init__('pWalk K', scale.Rho, distance.pWalk)


class Walk_K(Kernel):
    def __init__(self):
        super().__init__('Walk K', scale.Rho, distance.Walk)


class For_K(Kernel):
    def __init__(self):
        super().__init__('For K', scale.Fraction, distance.For)


class logFor_K(Kernel):
    def __init__(self):
        super().__init__('logFor K', scale.Fraction, distance.logFor)


class Comm_K(Kernel):
    def __init__(self):
        super().__init__('Comm K', scale.Fraction, distance.Comm)


class logComm_K(Kernel):
    def __init__(self):
        super().__init__('logComm K', scale.Fraction, distance.logComm)


class Heat_K(Kernel):
    def __init__(self):
        super().__init__('Heat K', scale.Fraction, distance.Heat)


class logHeat_K(Kernel):
    def __init__(self):
        super().__init__('logHeat K', scale.Fraction, distance.logHeat)


class SCT_K(Kernel):
    def __init__(self):
        super().__init__('SCT K', scale.Fraction, distance.SCT)


class SCCT_K(Kernel):
    def __init__(self):
        super().__init__('SCCT K', scale.Fraction, distance.SCCT)


class RSP_K(Kernel):
    def __init__(self):
        super().__init__('RSP K', scale.FractionReversed, distance.RSP)


class FE_K(Kernel):
    def __init__(self):
        super().__init__('FE K', scale.FractionReversed, distance.FE)


class SPCT_K(Kernel):
    def __init__(self):
        super().__init__('SP-CT K', scale.Linear, distance.SPCT)
