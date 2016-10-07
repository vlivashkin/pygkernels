import numpy as np

from measure import kernel
from measure import scale
from measure.shortcuts import Shortcuts


class Distance:
    def __init__(self, name, scale, parent_kernel, take_sqrt):
        self.name = name
        self.scale = scale
        self.parent_kernel = parent_kernel
        self.take_sqrt = take_sqrt

    def getD(self, A, param):
        H = self.parent_kernel.getK(A, param)
        D = Shortcuts.HtoD(H)
        return np.sqrt(D) if self.take_sqrt else D


class pWalk(Distance):
    def __init__(self):
        super().__init__('pWalk', scale.Rho, kernel.pWalk_H(), False)


class Walk(Distance):
    def __init__(self):
        super().__init__('Walk', scale.Rho, kernel.Walk_H(), False)


class For(Distance):
    def __init__(self):
        super().__init__('For', scale.Fraction, kernel.For_H(), False)


class logFor(Distance):
    def __init__(self):
        super().__init__('logFor', scale.Fraction, kernel.logFor_H(), False)


class Comm(Distance):
    def __init__(self):
        super().__init__('Comm', scale.Fraction, kernel.Comm_H(), True)


class logComm(Distance):
    def __init__(self):
        super().__init__('logComm', scale.Fraction, kernel.logComm_H(), True)


class Heat(Distance):
    def __init__(self):
        super().__init__('Heat', scale.Fraction, kernel.Heat_H(), True)


class logHeat(Distance):
    def __init__(self):
        super().__init__('logHeat', scale.Fraction, kernel.logHeat_H(), True)


class SCT(Distance):
    def __init__(self):
        super().__init__('SCT', scale.Fraction, kernel.SCT_H(), False)


class SCCT(Distance):
    def __init__(self):
        super().__init__('SCCT', scale.Fraction, kernel.SCCT_H(), False)


class RSP(Distance):
    def __init__(self):
        super().__init__('RSP', scale.FractionReversed, None, False)

    def getD(self, A, beta):
        pass


class FE(Distance):
    def __init__(self):
        super().__init__('FE', scale.FractionReversed, None, False)

    def getD(self, A, beta):
        pass


class SPCT(Distance):
    def __init__(self):
        super().__init__('SP-CT', scale.Linear, None, False)

    def getD(self, A, lambda_):
        Ds = Shortcuts.normalize(Shortcuts.getD_SP(A))
        Dr = Shortcuts.normalize(Shortcuts.HtoD(Shortcuts.getH_R(A)))
        return (1 - lambda_) * Ds + lambda_ * Dr
