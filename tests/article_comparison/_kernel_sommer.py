import numpy as np

from pygraphs.measure import Kernel, scaler, For_H
import pygraphs.measure.shortcuts as h


class logFor_S(Kernel):
    name, default_scaler = 'logFor_S', scaler.FractionReversed

    def get_K(self, alpha):
        size = self.A.shape[0]

        K_RL = For_H(self.A).get_K(alpha)
        if alpha != 1:
            S = (alpha - 1) * np.log(K_RL) / np.log(alpha)
        else:
            S = np.log(K_RL)
        e = np.ones((size, 1))
        ds = np.diagonal(S)[:, None]
        Δ_LF = ds * e.T + e * ds.T - 2 * S
        return h.D_to_K(Δ_LF)
