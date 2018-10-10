from .distance import *
from .kernel import *
from .kernel_rubanov import *
from .produced import *

__all__ = [
    # distances
    "SP",
    "CT",
    "SPCT",
    "pWalk",
    "Walk",
    "For",
    "logFor",
    "Comm",
    "logComm",
    "Heat",
    "logHeat",
    "SCT",
    "SCCT",
    "RSP",
    "FE",

    # h kernels
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
    "SCT_H",
    "SCCT_H",

    # k kernels
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
    "SCT_K",
    "SCCT_K",
    "RSP_K",
    "FE_K",

    # Rubanov's kernels
    "Katz_R",
    "Estrada_R",
    "Heat_R",
    "NormalizedHeat_R",
    "RegularizedLaplacian_R",
    "PPageRank_R",
    "ModifiedPPageRank_R",
    "HeatPPageRank_R",

    # Lists
    "distances",
    "H_kernels",
    "H_kernels_plus_RSP_FE",
    "K_kernels",
    "R_kernels",
    "ALL_kernels"
]

distances = [pWalk, Walk, For, logFor, Comm, logComm, Heat, logHeat, SCT, SCCT, RSP, FE, SPCT, SP, CT]
H_kernels = [pWalk_H, Walk_H, For_H, logFor_H, Comm_H, logComm_H, Heat_H, logHeat_H, SCT_H, SCCT_H, SPCT_H]
H_kernels_plus_RSP_FE = H_kernels[:-1] + [RSP_K, FE_K, SPCT_H]
H_kernels_plus_RSP_FE_SP_CT = H_kernels_plus_RSP_FE + [SP_K, CT_H]
K_kernels = [pWalk_K, Walk_K, For_K, logFor_K, Comm_K, logComm_K, Heat_K, logHeat_K, SCT_K, SCCT_K, RSP_K, FE_K, SPCT_K]
R_kernels = [Katz_R, Estrada_R, Heat_R, NormalizedHeat_R, RegularizedLaplacian_R, PPageRank_R, ModifiedPPageRank_R,
             HeatPPageRank_R]
ALL_kernels = H_kernels_plus_RSP_FE + R_kernels
