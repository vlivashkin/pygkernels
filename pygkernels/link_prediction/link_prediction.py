import numpy as np


class LP_AUC:
    def auc(self, A: np.array):
        """
        σ is the number of independent comparisons,
        σ1 is the number of times a missing link has a higher score than a non-existent link,
        σ2 is the number of times a missing link and a nonexistent link have the same score
        """

        E = G.edges

        σ = None
        σ1 = None
        σ2 = None

        AUC = (σ1 + 0.5 * σ2) / σ

        return AUC

    def split(self, E: np.array, beta: float = 0.05):
        pass

    def __call__(self, A: np.array, beta: float = 0.05):
        E = []
        for i in range(1, A.shape[0]):
            for j in range(i, A.shape[1]):
                if A[i, j] == 1:
                    E.append((i, j))
        E = np.array(E)
        Et, Ep = self.split(E, beta)

