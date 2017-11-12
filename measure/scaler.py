import numpy as np


class Scaler:
    def __init__(self, A: np.matrixlib.defmatrix.matrix = None):
        self.A = A

    def scale(self, ts):
        for t in ts:
            yield t


class Linear(Scaler):  # SP-CT
    pass


class AlphaToT(Scaler):  # α > 0 -> 0 < t < α^{-1}
    def __init__(self, A: np.matrixlib.defmatrix.matrix = None):
        super().__init__(A)
        cfm = np.linalg.eigvals(self.A)
        self.rho = np.max(cfm)

    def scale(self, alphas):
        for alpha in alphas:
            yield 1 / (1 / alpha + self.rho)


class Rho(Scaler):  # pWalk, Walk
    def __init__(self, A: np.matrixlib.defmatrix.matrix = None):
        super().__init__(A)
        cfm = np.linalg.eigvals(self.A)
        self.rho = np.max(cfm)

    def scale(self, ts):
        for t in ts:
            yield t / self.rho


class Fraction(Scaler):  # Forest, logForest, Comm, logComm, Heat, logHeat, SCT, SCCT
    def scale(self, ts):
        for t in ts:
            yield 0.5 * t / (1.0 - t)


class FractionReversed(Scaler):  # RSP, FE
    def scale(self, betas):
        for beta in betas:
            yield (1.0 - beta) / beta
