from typing import List, Type

from .distance import SP_D, CT_D, RSP_D, FE_D, Distance
from .kernel import CT_H, pWalk_H, For_H, Comm_H, Heat_H, NHeat_H, SCT_H, SCCT_H, PPR_H, ModifPPR_H, HeatPPR_H, Kernel
from .produced import SPCT_D, pWalk_D, Walk_D, For_D, logFor_D, Comm_D, logComm_D, Heat_D, logHeat_D, NHeat_D, \
    logNHeat_D, SCT_D, SCCT_D, PPR_D, logPPR_D, ModifPPR_D, logModifPPR_D, HeatPPR_D, logHeatPPR_D, SPCT_H, Walk_H, \
    logFor_H, logComm_H, logHeat_H, logNHeat_H, logPPR_H, logModifPPR_H, logHeatPPR_H, SP_K, CT_K, SPCT_K, pWalk_K, \
    Walk_K, For_K, logFor_K, Comm_K, logComm_K, Heat_K, logHeat_K, NHeat_K, logNHeat_K, SCT_K, SCCT_K, RSP_K, FE_K, \
    PPR_K, logPPR_K, ModifPPR_K, logModifPPR_K, HeatPPR_K, logHeatPPR_K

__all__ = [
    # Distances
    "SP_D",
    "CT_D",
    "SPCT_D",
    "pWalk_D",
    "Walk_D",
    "For_D",
    "logFor_D",
    "Comm_D",
    "logComm_D",
    "Heat_D",
    "logHeat_D",
    "NHeat_D",
    "logNHeat_D",
    "SCT_D",
    "SCCT_D",
    "RSP_D",
    "FE_D",
    "PPR_D",
    "logPPR_D",
    "ModifPPR_D",
    "logModifPPR_D",
    "HeatPPR_D",
    "logHeatPPR_D",

    # H kernels
    "CT_H",
    "SPCT_H",
    "pWalk_H",
    "Walk_H",
    "For_H",
    "logFor_H",
    "Comm_H",
    "logComm_H",
    "Heat_H",
    "logHeat_H",
    "NHeat_H",
    "logNHeat_H",
    "SCT_H",
    "SCCT_H",
    "PPR_H",
    "logPPR_H",
    "ModifPPR_H",
    "logModifPPR_H",
    "HeatPPR_H",
    "logHeatPPR_H",

    # K kernels
    "SP_K",
    "CT_K",
    "SPCT_K",
    "pWalk_K",
    "Walk_K",
    "For_K",
    "logFor_K",
    "Comm_K",
    "logComm_K",
    "Heat_K",
    "logHeat_K",
    "NHeat_K",
    "logNHeat_K",
    "SCT_K",
    "SCCT_K",
    "RSP_K",
    "FE_K",
    "PPR_K",
    "logPPR_K",
    "ModifPPR_K",
    "logModifPPR_K",
    "HeatPPR_K",
    "logHeatPPR_K",

    # Lists
    "distances",
    "kernels"
]

distances: List[Type[Distance]] = [pWalk_D, Walk_D, For_D, logFor_D, Comm_D, logComm_D, Heat_D, logHeat_D, NHeat_D,
                                   logNHeat_D, SCT_D, SCCT_D, RSP_D, FE_D, PPR_D, logPPR_D, ModifPPR_D, logModifPPR_D,
                                   HeatPPR_D, logHeatPPR_D, SPCT_D]
kernels: List[Type[Kernel]] = [pWalk_H, Walk_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, NHeat_H,
                               logNHeat_H, SCT_H, SCCT_H, RSP_K, FE_K, PPR_H, logPPR_H, ModifPPR_H, logModifPPR_H,
                               HeatPPR_H, logHeatPPR_H, SPCT_H]
