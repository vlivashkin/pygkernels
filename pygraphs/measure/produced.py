from sklearn.utils import deprecated

from pygraphs.measure import scaler, kernel, distance
from pygraphs.measure.distance import Distance
from pygraphs.measure.kernel import Kernel
from pygraphs.measure.shortcuts import H_to_D


# H KERNELS

class SPCT_H(Kernel):
    name, default_scaler = 'SP-CT', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.H_SP = SP_K(A).get_K(-1)
        self.H_CT = 2 * kernel.CT_H(A).get_K(-1)

    def get_K(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.H_SP + (1. - lmbda) * self.H_CT


class Walk_H(Kernel):
    name, default_scaler, parent_kernel_class = 'Walk', scaler.Rho, kernel.pWalk_H


class logFor_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logFor', scaler.Fraction, kernel.For_H


class logComm_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logComm', scaler.Fraction, kernel.Comm_H


class logHeat_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logHeat', scaler.Fraction, kernel.Heat_H


class logNHeat_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logNHeat', scaler.Fraction, kernel.NHeat_H


class logPPR_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logPPR', scaler.Linear, kernel.PPR_H


class logModifPPR_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logModifPPR', scaler.Linear, kernel.ModifPPR_H


class logHeatPPR_H(Kernel):
    name, default_scaler, parent_kernel_class = 'logHeatPPR', scaler.Linear, kernel.HeatPPR_H


# DISTANCES
class SPCT_D(Distance):
    name, default_scaler = 'SP-CT', scaler.Linear

    def __init__(self, A):
        super().__init__(A)

        self.D_SP = distance.SP_D(A).get_D(-1)
        self.D_CT = 2 * H_to_D(kernel.CT_H(A).get_K(-1))

    def get_D(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.D_SP + (1. - lmbda) * self.D_CT


class pWalk_D(Distance):
    name, default_scaler, parent_kernel_class = 'pWalk', scaler.Rho, kernel.pWalk_H


class Walk_D(Distance):
    name, default_scaler, parent_kernel_class = 'Walk', scaler.Rho, Walk_H


class For_D(Distance):
    name, default_scaler, parent_kernel_class = 'For', scaler.Fraction, kernel.For_H


class logFor_D(Distance):
    name, default_scaler, parent_kernel_class = 'logFor', scaler.Fraction, logFor_H


class Comm_D(Distance):
    name, default_scaler, parent_kernel_class, power = 'Comm', scaler.Fraction, kernel.Comm_H, .5


class logComm_D(Distance):
    name, default_scaler, parent_kernel_class, power = 'logComm', scaler.Fraction, logComm_H, .5


class Heat_D(Distance):
    name, default_scaler, parent_kernel_class = 'Heat', scaler.Fraction, kernel.Heat_H


class logHeat_D(Distance):
    name, default_scaler, parent_kernel_class = 'logHeat', scaler.Fraction, logHeat_H


class NHeat_D(Distance):
    name, default_scaler, parent_kernel_class = 'NHeat', scaler.Fraction, kernel.NHeat_H


class logNHeat_D(Distance):
    name, default_scaler, parent_kernel_class = 'logNHeat', scaler.Fraction, logNHeat_H


class SCT_D(Distance):
    name, default_scaler, parent_kernel_class = 'SCT', scaler.Fraction, kernel.SCT_H


class SCCT_D(Distance):
    name, default_scaler, parent_kernel_class = 'SCCT', scaler.Fraction, kernel.SCCT_H


class PPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'PPR', scaler.Linear, kernel.PPR_H


class logPPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'logPPR', scaler.Linear, logPPR_H


class ModifPPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'ModifPPR', scaler.Linear, kernel.ModifPPR_H


class logModifPPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'logModifPPR', scaler.Linear, logModifPPR_H


class HeatPPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'HeatPPR', scaler.Linear, kernel.HeatPPR_H


class logHeatPPR_D(Distance):
    name, default_scaler, parent_kernel_class = 'logHeatPPR', scaler.Linear, logHeatPPR_H


# K KERNELS

class SP_K(Kernel):
    name, default_scaler, parent_distance_class = 'SP K', scaler.Linear, distance.SP_D


class CT_K(Kernel):
    name, default_scaler, parent_distance_class = 'CT K', scaler.Linear, distance.SP_D


class pWalk_K(Kernel):
    name, default_scaler, parent_distance_class = 'pWalk K', scaler.Rho, pWalk_D


class Walk_K(Kernel):
    name, default_scaler, parent_distance_class = 'Walk K', scaler.Rho, Walk_D


class For_K(Kernel):
    name, default_scaler, parent_distance_class = 'For K', scaler.Fraction, For_D


class logFor_K(Kernel):
    name, default_scaler, parent_distance_class = 'logFor K', scaler.Fraction, logFor_D


class Comm_K(Kernel):
    name, default_scaler, parent_distance_class = 'Comm K', scaler.Fraction, Comm_D


class logComm_K(Kernel):
    name, default_scaler, parent_distance_class = 'logComm K', scaler.Fraction, logComm_D


class Heat_K(Kernel):
    name, default_scaler, parent_distance_class = 'Heat K', scaler.Fraction, Heat_D


class logHeat_K(Kernel):
    name, default_scaler, parent_distance_class = 'logHeat K', scaler.Fraction, logHeat_D


class NHeat_K(Kernel):
    name, default_scaler, parent_distance_class = 'NHeat K', scaler.Fraction, NHeat_D


class logNHeat_K(Kernel):
    name, default_scaler, parent_distance_class = 'logNHeat K', scaler.Fraction, logNHeat_D


class SCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SCT K', scaler.Fraction, SCT_D


class SCCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SCCT K', scaler.Fraction, SCCT_D


@deprecated()
class RSP_vanilla_K(Kernel):
    name, default_scaler, parent_distance_class = 'RSP vanilla K', scaler.FractionReversed, distance.RSP_vanilla_D


@deprecated()
class FE_vanilla_K(Kernel):
    name, default_scaler, parent_distance_class = 'FE vanilla K', scaler.FractionReversed, distance.FE_vanilla_D


class RSP_K(Kernel):
    name, default_scaler, parent_distance_class = 'RSP', scaler.FractionReversed, distance.RSP_D


class FE_K(Kernel):
    name, default_scaler, parent_distance_class = 'FE', scaler.FractionReversed, distance.FE_D


class SPCT_K(Kernel):
    name, default_scaler, parent_distance_class = 'SP-CT K', scaler.Linear, SPCT_D


class PPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'PPR', scaler.Linear, PPR_D


class logPPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'logPPR', scaler.Linear, logPPR_D


class ModifPPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'ModifPPR', scaler.Linear, ModifPPR_D


class logModifPPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'logModifPPR', scaler.Linear, logModifPPR_D


class HeatPPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'HeatPPR', scaler.Linear, HeatPPR_D


class logHeatPPR_K(Kernel):
    name, default_scaler, parent_distance_class = 'logHeatPPR', scaler.Linear, logHeatPPR_D
