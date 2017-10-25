from scipy.linalg import expm
from sklearn.preprocessing import normalize

from measure import distance, scaler
from measure.shortcuts import *


class Kernel:
    def __init__(self, name, scaler, A: np.ndarray, parent_distance=None):
        self.name = name
        self.scaler = scaler(A)
        self.A = A
        self.parent_distance = parent_distance(A) if parent_distance is not None else None

    def getK(self, param):
        D = self.parent_distance.getD(param)
        return DtoK(D)

    def grid_search(self, params=np.linspace(0, 1, 55)):
        results = np.array((params.shape[0], ))
        for idx, param in enumerate(self.scaler.scale(params)):
            results[idx] = self.getK(param)
        return results

    @staticmethod
    def get_all_H():
        return [pWalk_H, Walk_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, SCT_H, SCCT_H, SPCT_H]

    @staticmethod
    def get_all_H_plus_RSP_FE():
        return Kernel.get_all_H() + [RSP_K, FE_K]

    @staticmethod
    def get_all_K():
        return [pWalk_K, Walk_K, For_K, logFor_K, Comm_K, logComm_K,
                Heat_K, logHeat_K, SCT_K, SCCT_K, RSP_K, FE_K, SPCT_K]


class pWalk_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('pWalk H', scaler.Rho, A)

    def getK(self, t):
        """
        H0 = (I - tA)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) - t * self.A)


class Walk_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Walk H', scaler.Rho, A)
        self.parent_distance = pWalk_H(self.A)

    def getK(self, t):
        return H0toH(self.parent_distance.getK(t))


class For_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('For H', scaler.Fraction, A)

    def getK(self, t):
        """
        H0 = (I + tL)^{-1}
        """
        size = self.A.shape[0]
        return np.linalg.pinv(np.eye(size) + t * getL(self.A))


class logFor_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logFor H', scaler.Fraction, A)
        self.parent_distance = For_H(self.A)

    def getK(self, t):
        return H0toH(self.parent_distance.getK(t))


class Comm_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Comm H', scaler.Fraction, A)

    def getK(self, t):
        """
        H0 = exp(tA)
        """
        return expm(t * self.A) if t < 30 else None


class logComm_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logComm H', scaler.Fraction, A)
        self.parent_distance = Comm_H(self.A)

    def getK(self, t):
        return H0toH(self.parent_distance.getK(t))


class Heat_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Heat H', scaler.Fraction, A)

    def getK(self, t):
        """
        H0 = exp(-tL)
        """
        return expm(-t * getL(self.A))


class logHeat_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logHeat H', scaler.Fraction, A)
        self.parent_distance = Heat_H(self.A)

    def getK(self, t):
        return H0toH(self.parent_distance.getK(t))


class SCT_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SCT H', scaler.Fraction, A)
        self.K_CT = np.linalg.pinv(getL(self.A))
        self.sigma = self.K_CT.std()
        self.Kds = self.K_CT / self.sigma

    def getK(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class SCCT_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SCCT H', scaler.Fraction, A)
        self.K_CCT = H_CCT(A)
        self.sigma = self.K_CCT.std()
        self.Kds = self.K_CCT / self.sigma

    def getK(self, alpha):
        """
        H = 1/(1 + exp(-αL+/σ))
        """
        return 1. / (1. + np.exp(-alpha * self.Kds))


class SPCT_H(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SP-CT H', scaler.Linear, A)
        self.Hs = normalize(DtoK(D_SP(A)))
        self.Hc = normalize(H_R(A))

    def getK(self, lmbda):
        return (1 - lmbda) * self.Hs + lmbda * self.Hc


class pWalk_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('pWalk K', scaler.Rho, A, distance.pWalk)


class Walk_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Walk K', scaler.Rho, A, distance.Walk)


class For_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('For K', scaler.Fraction, A, distance.For)


class logFor_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logFor K', scaler.Fraction, A, distance.logFor)


class Comm_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Comm K', scaler.Fraction, A, distance.Comm)


class logComm_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logComm K', scaler.Fraction, A, distance.logComm)


class Heat_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('Heat K', scaler.Fraction, A, distance.Heat)


class logHeat_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('logHeat K', scaler.Fraction, A, distance.logHeat)


class SCT_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SCT K', scaler.Fraction, A, distance.SCT)


class SCCT_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SCCT K', scaler.Fraction, A, distance.SCCT)


class RSP_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('RSP K', scaler.FractionReversed, A, distance.RSP)


class FE_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('FE K', scaler.FractionReversed, A, distance.FE)


class SPCT_K(Kernel):
    def __init__(self, A: np.ndarray):
        super().__init__('SP-CT K', scaler.Linear, A, distance.SPCT)
