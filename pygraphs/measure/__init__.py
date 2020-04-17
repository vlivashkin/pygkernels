from typing import List, Type

from .distance import SP_D, CT_D, RSP_D, FE_D, Distance, RSP_vanilla_D, FE_vanilla_D
from .kernel import CT_H, Katz_H, For_H, Comm_H, Heat_H, NHeat_H, SCT_H, SCCT_H, PPR_H, ModifPPR_H, HeatPPR_H, Kernel, \
    DF_H, Abs_H
from .produced import SPCT_D, Katz_D, logKatz_D, For_D, logFor_D, Comm_D, logComm_D, Heat_D, logHeat_D, NHeat_D, \
    logNHeat_D, SCT_D, SCCT_D, PPR_D, logPPR_D, ModifPPR_D, logModifPPR_D, HeatPPR_D, logHeatPPR_D, SPCT_H, logKatz_H, \
    logFor_H, logComm_H, logHeat_H, logNHeat_H, logPPR_H, logModifPPR_H, logHeatPPR_H, SP_K, RSP_K, FE_K, \
    RSP_vanilla_K, FE_vanilla_K, SPCT_K, logDF_D, DF_D, logDF_H, Abs_D, logAbs_D, logAbs_H

__all__ = [
    # Distances
    "SP_D", "CT_D", "SPCT_D",
    "Katz_D", "logKatz_D",
    "For_D", "logFor_D",
    "Comm_D", "logComm_D",
    "Heat_D", "logHeat_D",
    "NHeat_D", "logNHeat_D",
    "SCT_D", "SCCT_D",
    "RSP_vanilla_D", "RSP_D",
    "FE_vanilla_D", "FE_D",
    "PPR_D", "logPPR_D",
    "ModifPPR_D", "logModifPPR_D",
    "HeatPPR_D", "logHeatPPR_D",
    "DF_D", "logDF_D",
    "Abs_D", "logAbs_D",

    # H kernels (original kernels or produced from kernels)
    "CT_H", "SPCT_H",
    "Katz_H", "logKatz_H",
    "For_H", "logFor_H",
    "Comm_H", "logComm_H",
    "Heat_H", "logHeat_H",
    "NHeat_H", "logNHeat_H",
    "SCT_H", "SCCT_H",
    "PPR_H", "logPPR_H",
    "ModifPPR_H", "logModifPPR_H",
    "HeatPPR_H", "logHeatPPR_H",
    "DF_H", "logDF_H",
    "Abs_H", "logAbs_H",

    # K kernels (kernels derived from distances)
    "SP_K", "SPCT_K",
    "RSP_vanilla_K", "RSP_K",
    "FE_vanilla_K", "FE_K",

    # Lists
    "distances",
    "kernels"
]

distances: List[Type[Distance]] = [Katz_D, logKatz_D, For_D, logFor_D, Comm_D, logComm_D, Heat_D, logHeat_D, NHeat_D,
                                   logNHeat_D, SCT_D, SCCT_D, RSP_D, FE_D, PPR_D, logPPR_D, ModifPPR_D, logModifPPR_D,
                                   HeatPPR_D, logHeatPPR_D, DF_D, logDF_D, Abs_D, logAbs_D, SPCT_D]
kernels: List[Type[Kernel]] = [Katz_H, logKatz_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, NHeat_H,
                               logNHeat_H, SCT_H, SCCT_H, RSP_K, FE_K, PPR_H, logPPR_H, ModifPPR_H, logModifPPR_H,
                               HeatPPR_H, logHeatPPR_H, DF_H, logDF_H, Abs_H, logAbs_H, SPCT_H]
