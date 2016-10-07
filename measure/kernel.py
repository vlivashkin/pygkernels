from measure import distance
from measure import scale
from measure.shortcuts import Shortcuts


class Kernel:
    def __init__(self, name, scale, parent_distance):
        self.name = name
        self.scale = scale
        self.parent_distance = parent_distance

    def getK(self, A, param):
        D = self.parent_distance.getD(A, param)
        return Shortcuts.DtoK(D)


class pWalk_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Rho, None)


class Walk_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Rho, None)


class For_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class logFor_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class Comm_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class logComm_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class Heat_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class logHeat_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class SCT_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class SCCT_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, None)


class SPCT_H(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Linear, None)


class pWalk_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Rho, distance.pWalk)


class Walk_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Rho, distance.Walk)


class For_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.For)


class logFor_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.logFor)


class Comm_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.Comm)


class logComm_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.logComm)


class Heat_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.Heat)


class logHeat_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.logHeat)


class SCT_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.SCT)


class SCCT_K(Kernel):
    def __init__(self):
        super().__init__('pWalk_H', scale.Fraction, distance.SCCT)


class RSP_K(Kernel):
    def __init__(self):
        super().__init__('RSP K', scale.FractionReversed, distance.RSP)


class FE_K(Kernel):
    def __init__(self):
        super().__init__('FE K', scale.FractionReversed, distance.FE)


class SPCT_K(Kernel):
    def __init__(self):
        super().__init__('SP-CT K', scale.Linear, distance.SPCT)
