from sklearn.utils import deprecated

from pygraphs.measure import scaler, kernel, distance
from pygraphs.measure.distance import Distance, SP_D, CT_D
from pygraphs.measure.kernel import Kernel, CT_H


# H KERNELS (log kernels derived from kernels)
class SPCT_H(Kernel):
    name, _default_scaler = 'SP-CT', scaler.Linear

    def __init__(self, A):
        super().__init__(A)
        self.H_SP = SP_K(A).get_K(-1)
        self.H_CT = 2 * CT_H(A).get_K(-1)

    def get_K(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.H_SP + (1. - lmbda) * self.H_CT


class logKatz_H(Kernel):
    name, _parent_kernel_class = 'Katz', kernel.Katz_H


class logFor_H(Kernel):
    name, _parent_kernel_class = 'logFor', kernel.For_H


class logComm_H(Kernel):
    name, _parent_kernel_class = 'logComm', kernel.Comm_H


class logHeat_H(Kernel):
    name, _parent_kernel_class = 'logHeat', kernel.Heat_H


class logNHeat_H(Kernel):
    name, _parent_kernel_class = 'logNHeat', kernel.NHeat_H


class logPPR_H(Kernel):
    name, _parent_kernel_class = 'logPPR', kernel.PPR_H


class logModifPPR_H(Kernel):
    name, _parent_kernel_class = 'logModifPPR', kernel.ModifPPR_H


class logHeatPPR_H(Kernel):
    name, _parent_kernel_class = 'logHeatPPR', kernel.HeatPPR_H


class logDF_H(Kernel):
    name, _parent_kernel_class = 'logDF', kernel.DF_H


class logAbs_H(Kernel):
    name, _parent_kernel_class = 'logAbs', kernel.Abs_H


# DISTANCES (distances derived from kernels)
class SPCT_D(Distance):
    name, _default_scaler = 'SP-CT', scaler.Linear

    def __init__(self, A):
        super().__init__(A)

        self.D_SP = SP_D(A).get_D(-1)
        self.D_CT = 2 * CT_D(A).get_D(-1)

    def get_D(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.D_SP + (1. - lmbda) * self.D_CT


class Katz_D(Distance):
    name, _parent_kernel_class = 'pWalk', kernel.Katz_H


class logKatz_D(Distance):
    name, _parent_kernel_class = 'Walk', logKatz_H


class For_D(Distance):
    name, _parent_kernel_class = 'For', kernel.For_H


class logFor_D(Distance):
    name, _parent_kernel_class = 'logFor', logFor_H


class Comm_D(Distance):
    name, _parent_kernel_class, power = 'Comm', kernel.Comm_H, .5


class logComm_D(Distance):
    name, _parent_kernel_class, power = 'logComm', logComm_H, .5


class Heat_D(Distance):
    name, _parent_kernel_class = 'Heat', kernel.Heat_H


class logHeat_D(Distance):
    name, _parent_kernel_class = 'logHeat', logHeat_H


class NHeat_D(Distance):
    name, _parent_kernel_class = 'NHeat', kernel.NHeat_H


class logNHeat_D(Distance):
    name, _parent_kernel_class = 'logNHeat', logNHeat_H


class SCT_D(Distance):
    name, _parent_kernel_class = 'SCT', kernel.SCT_H


class SCCT_D(Distance):
    name, _parent_kernel_class = 'SCCT', kernel.SCCT_H


class PPR_D(Distance):
    name, _parent_kernel_class = 'PPR', kernel.PPR_H


class logPPR_D(Distance):
    name, _parent_kernel_class = 'logPPR', logPPR_H


class ModifPPR_D(Distance):
    name, _parent_kernel_class = 'ModifPPR', kernel.ModifPPR_H


class logModifPPR_D(Distance):
    name, _parent_kernel_class = 'logModifPPR', logModifPPR_H


class HeatPPR_D(Distance):
    name, _parent_kernel_class = 'HeatPPR', kernel.HeatPPR_H


class logHeatPPR_D(Distance):
    name, _parent_kernel_class = 'logHeatPPR', logHeatPPR_H


class DF_D(Distance):
    name, _parent_kernel_class = 'DF', kernel.DF_H


class logDF_D(Distance):
    name, _parent_kernel_class = 'logDF', logDF_H


class Abs_D(Distance):
    name, _parent_kernel_class = 'Abs', kernel.Abs_H


class logAbs_D(Distance):
    name, _parent_kernel_class = 'Abs', logAbs_H


# K KERNELS (kernels derived from distances)
class SP_K(Kernel):
    name, _parent_distance_class = 'SP K', distance.SP_D


class SPCT_K(Kernel):
    name, _parent_distance_class = 'SP-CT K', SPCT_D


@deprecated()
class RSP_vanilla_K(Kernel):
    name, _parent_distance_class = 'RSP vanilla K', distance.RSP_vanilla_D


@deprecated()
class FE_vanilla_K(Kernel):
    name, _parent_distance_class = 'FE vanilla K', distance.FE_vanilla_D


class RSP_K(Kernel):
    name, _parent_distance_class = 'RSP', distance.RSP_D


class FE_K(Kernel):
    name, _parent_distance_class = 'FE', distance.FE_D
