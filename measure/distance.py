from measure import kernel
from measure import scaler
from measure.shortcuts import *


class Distance:
    def __init__(self, name, scaler, A: np.ndarray, parent_kernel=None, power=1.):
        self.name = name
        self.scaler = scaler(A)
        self.A = A
        self.parent_kernel = parent_kernel(A) if parent_kernel is not None else None
        self.power = power

    def getD(self, param):
        H = self.parent_kernel.getK(param)
        D = HtoD(H)
        return np.power(D, self.power) if self.power != 1 else D

    def grid_search(self, params=np.linspace(0, 1, 55)):
        results = np.array((params.shape[0], ))
        for idx, param in enumerate(self.scaler.scale(params)):
            results[idx] = self.getD(param)
        return results

    @staticmethod
    def get_all():
        return [pWalk, Walk, For, logFor, Comm, logComm, Heat, logHeat, SCT, SCCT, RSP, FE, SPCT]


class pWalk(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('pWalk', scaler.Rho, A, kernel.pWalk_H, .5)


class Walk(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('Walk', scaler.Rho, A, kernel.Walk_H, .5)


class For(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('For', scaler.Fraction, A, kernel.For_H, .5)


class logFor(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('logFor', scaler.Fraction, A, kernel.logFor_H, .5)


class Comm(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('Comm', scaler.Fraction, A, kernel.Comm_H, .5)


class logComm(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('logComm', scaler.Fraction, A, kernel.logComm_H, .5)


class Heat(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('Heat', scaler.Fraction, A, kernel.Heat_H, .5)


class logHeat(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('logHeat', scaler.Fraction, A, kernel.logHeat_H, .5)


class SCT(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('SCT', scaler.Fraction, A, kernel.SCT_H, .5)


class SCCT(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('SCCT', scaler.Fraction, A, kernel.SCCT_H, .5)


class RSP(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('RSP', scaler.FractionReversed, A)
        self.size = A.shape[0]
        self.Pref = np.linalg.pinv(getD(A)).dot(A)
        self.C = johnson(A, directed=False)

    def getD(self, beta):
        """
        P^{ref} = D^{-1}*A, D = Diag(A*e)
        W = P^{ref} ◦ exp(-βC); ◦ is element-wise *
        Z = (I - W)^{-1}
        S = (Z(C ◦ W)Z)÷Z; ÷ is element-wise /
        C_ = S - e(d_S)^T; d_S = diag(S)
        Δ_RSP = (C_ + C_^T)/2
        """
        W = self.Pref * np.exp(-beta * self.C)
        Z = np.linalg.pinv(np.eye(self.size) - W)
        S = (Z.dot(self.C * W).dot(Z)) / Z
        C_ = S - np.ones((self.size, 1)).dot(np.diag(S).reshape((-1, 1)).transpose())
        Δ_RSP = 0.5 * (C_ + C_.transpose())
        return Δ_RSP - np.diag(np.diag(Δ_RSP))


class FE(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('FE', scaler.FractionReversed, A)
        self.size = A.shape[0]
        self.Pref = np.linalg.pinv(getD(A)).dot(A)
        self.C = johnson(A, directed=False)

    def getD(self, beta):
        """
        P^{ref} = D^{-1}*A, D = Diag(A*e)
        W = P^{ref} (element-wise)* exp(-βC)
        Z = (I - W)^{-1}
        Z^h = Z * D_h^{-1}, D_h = Diag(Z)
        Φ = -1/β * log(Z^h)
        Δ_FE = (Φ + Φ^T)/2
        """
        W = self.Pref * np.exp(-beta * self.C)
        Z = np.linalg.pinv(np.eye(self.size) - W)
        Dh = np.diag(np.diag(Z))
        Zh = Z.dot(np.linalg.pinv(Dh))
        Φ = np.log(Zh) / -beta
        Δ_FE = 0.5 * (Φ + Φ.transpose())
        return Δ_FE - np.diag(np.diag(Δ_FE))


class SPCT(Distance):
    def __init__(self, A: np.ndarray):
        super().__init__('SP-CT', scaler.Linear, A)
        self.Ds = normalize(D_SP(A))
        self.Dr = normalize(HtoD(H_R(A)))

    def getD(self, lmbda):
        return (1. - lmbda) * self.Ds + lmbda * self.Dr
