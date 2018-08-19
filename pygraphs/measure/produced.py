from pygraphs.measure import scaler, kernel, distance
from pygraphs.measure.distance import Distance
from pygraphs.measure.kernel import Kernel
from pygraphs.measure.shortcuts import H_to_D


# H KERNELS

class SPCT_H(Kernel):
    name, default_scaler = 'SP-CT H', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.H_SP = SP_K(A).get_K(-1)
        self.H_CT = 2 * kernel.CT_H(A).get_K(-1)

    def get_K(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.H_SP + (1. - lmbda) * self.H_CT


class Walk_H(Kernel):
    name, default_scaler, parent_kernel_class = 'Walk H', scaler.Rho, kernel.pWalk_H


class logFor_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logFor H', scaler.Fraction, kernel.For_H


class logComm_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logComm H', scaler.Fraction, kernel.Comm_H


class logHeat_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logHeat H', scaler.Fraction, kernel.Heat_H


# DISTANCES

class SPCT(Distance):
    name, default_scaler = 'SP-CT', scaler.Linear

    def __init__(self, A):
        super().__init__(A)

        self.D_SP = distance.SP(A).get_D(-1)
        self.D_CT = 2 * H_to_D(kernel.CT_H(A).get_K(-1))

    def get_D(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.D_SP + (1. - lmbda) * self.D_CT


class pWalk(Distance):
    name, default_scaler, parent_kernel_class = 'pWalk', scaler.Rho, kernel.pWalk_H


class Walk(Distance):
    name, default_scaler, parent_kernel_class = 'Walk', scaler.Rho, Walk_H


class For(Distance):
    name, default_scaler, parent_kernel_class = 'For', scaler.Fraction, kernel.For_H


class logFor(Distance):
    name, default_scaler, parent_kernel_class = 'logFor', scaler.Fraction, logFor_H


class Comm(Distance):
    name, default_scaler, parent_kernel_class, power = 'Comm', scaler.Fraction, kernel.Comm_H, .5


class logComm(Distance):
    name, default_scaler, parent_kernel_class, power = 'logComm', scaler.Fraction, logComm_H, .5


class Heat(Distance):
    name, default_scaler, parent_kernel_class = 'Heat', scaler.Fraction, kernel.Heat_H


class logHeat(Distance):
    name, default_scaler, parent_kernel_class = 'logHeat', scaler.Fraction, logHeat_H


class SCT(Distance):
    name, default_scaler, parent_kernel_class = 'SCT', scaler.Fraction, kernel.SCT_H


class SCCT(Distance):
    name, default_scaler, parent_kernel_class = 'SCCT', scaler.Fraction, kernel.SCCT_H


# K KERNELS

class SP_K(Kernel):
    name, default_scaler, parent_distance_class = 'SP K', scaler.Linear, distance.SP


class CT_K(Kernel):
    name, default_scaler, parent_distance_class = 'CT K', scaler.Linear, distance.SP


class pWalk_K(Kernel):
    name, default_scaler, parent_distance_class = 'pWalk K', scaler.Rho, pWalk


class Walk_K(Kernel):
    name, default_scaler, parent_distance_class = 'Walk K', scaler.Rho, Walk


class For_K(Kernel):
    name, default_scaler, parent_distance_class = 'For K', scaler.Fraction, For


class logFor_K(Kernel):
    name, default_scaler, parent_distance_class = 'logFor K', scaler.Fraction, logFor


class Comm_K(Kernel):
    name, default_scaler, parent_distance_class = 'Comm K', scaler.Fraction, Comm


class logComm_K(Kernel):
    name, default_scaler, parent_distance_class = 'logComm K', scaler.Fraction, logComm


class Heat_K(Kernel):
    name, default_scaler, parent_distance_class = 'Heat K', scaler.Fraction, Heat


class logHeat_K(Kernel):
    name, default_scaler, parent_distance_class = 'logHeat K', scaler.Fraction, logHeat


class SCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SCT K', scaler.Fraction, SCT


class SCCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SCCT K', scaler.Fraction, SCCT


class RSP_vanilla_K(Kernel):
    name, default_scaler, parent_distance_class = 'RSP vanilla K', scaler.FractionReversed, distance.RSP_vanilla


class FE_vanilla_K(Kernel):
    name, default_scaler, parent_distance_class = 'FE vanilla K', scaler.FractionReversed, distance.FE_vanilla


class RSP_K(Kernel):
    name, default_scaler, parent_distance_class = 'RSP K', scaler.FractionReversed, distance.RSP


class FE_K(Kernel):
    name, default_scaler, parent_distance_class = 'FE K', scaler.FractionReversed, distance.FE


class SPCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SP-CT K', scaler.Linear, SPCT
