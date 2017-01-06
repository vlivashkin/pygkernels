import scipy as sc
from measure import kernel
from measure import scale
from measure.shortcuts import *


class Distance:
    def __init__(self, name, scale, parent_kernel, power=1):
        self.name = name
        self.scale = scale
        self.parent_kernel = parent_kernel
        self.power = power

    def getD(self, A: np.ndarray, param):
        H = self.parent_kernel.getK(A, param)
        D = HtoD(H)
        return np.power(D, self.power) if self.power != 1 else D


class pWalk(Distance):
    def __init__(self):
        super().__init__('pWalk', scale.Rho, kernel.pWalk_H(), 1)


class Walk(Distance):
    def __init__(self):
        super().__init__('Walk', scale.Rho, kernel.Walk_H(), 1)


class For(Distance):
    def __init__(self):
        super().__init__('For', scale.Fraction, kernel.For_H(), 1)


class logFor(Distance):
    def __init__(self):
        super().__init__('logFor', scale.Fraction, kernel.logFor_H(), 1)


class Comm(Distance):
    def __init__(self):
        super().__init__('Comm', scale.Fraction, kernel.Comm_H(), 1)


class logComm(Distance):
    def __init__(self):
        super().__init__('logComm', scale.Fraction, kernel.logComm_H(), 1)


class Heat(Distance):
    def __init__(self):
        super().__init__('Heat', scale.Fraction, kernel.Heat_H(), 1)


class logHeat(Distance):
    def __init__(self):
        super().__init__('logHeat', scale.Fraction, kernel.logHeat_H(), 1)


class SCT(Distance):
    def __init__(self):
        super().__init__('SCT', scale.Fraction, kernel.SCT_H(), 1)


class SCCT(Distance):
    def __init__(self):
        super().__init__('SCCT', scale.Fraction, kernel.SCCT_H(), 1)


class RSP(Distance):
    def __init__(self):
        super().__init__('RSP', scale.FractionReversed, None, 1)

    def getD(self, A: np.ndarray, beta):
        """
        W = P^{ref} ◦ exp(-βC); ◦ is element-wise *
        Z = (I - W)^{-1}
        S = (Z(C ◦ W)Z)÷Z; ÷ is element-wise /
        C_ = S - e(d_S)^T; d_S = diag(S)
        Δ_RSP = (C_ + C_^T)/2
        """
        size = A.shape[0]
        Pref = getPref(A)
        C = johnson(A, directed=False)
        W = Pref * np.exp(-beta * C)
        Z = np.linalg.pinv(np.eye(size) - W)
        S = (Z.dot(C * W).dot(Z)) / Z
        C_ = S - np.ones((size, 1)).dot(np.diag(S).transpose())
        Δ_RSP = 0.5 * (C_ + C_.transpose())
        return Δ_RSP - np.diag(np.diag(Δ_RSP))


class FE(Distance):
    def __init__(self):
        super().__init__('FE', scale.FractionReversed, None, 1)

    def getD(self, A, beta):
        """
        P^{ref} = D^{-1}*A, D = Diag(A*e)
        W = P^{ref} (element-wise)* exp(-βC)
        Z = (I - W)^{-1}
        Z^h = Z * D_h^{-1}, D_h = Diag(Z)
        Φ = -1/β * log(Z^h)
        Δ_FE = (Φ + Φ^T)/2
        """
        size = A.shape[0]
        Pref = getPref(A)
        C = johnson(A, directed=False)
        W = Pref * np.exp(-beta * C)
        Z = np.linalg.pinv(np.eye(size) - W)
        Dh = np.diag(np.diag(Z))
        Zh = Z.dot(np.linalg.pinv(Dh))
        Φ = np.log(Zh) / -beta
        Δ_FE = 0.5 * (Φ + Φ.transpose())
        return Δ_FE - np.diag(np.diag(Δ_FE))


class SPCT(Distance):
    def __init__(self):
        super().__init__('SP-CT', scale.Linear, None, 1)

    def getD(self, A, lambda_):
        Ds = normalize(D_SP(A))
        Dr = normalize(HtoD(H_R(A)))
        return (1 - lambda_) * Ds + lambda_ * Dr
