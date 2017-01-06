import numpy as np


class Scale:
    @staticmethod
    def calc(A, param):
        pass


class Linear(Scale):  # SP-CT
    @staticmethod
    def calc(A, t):
        return t


class AlphaToT(Scale):  # α > 0 -> 0 < t < α^{-1}
    @staticmethod
    def calc(A, alpha):
        cfm = np.linalg.eigvals(A)
        rho = np.max(cfm)
        return 1 / (1 / alpha + rho)


class Rho(Scale):  # pWalk, Walk
    @staticmethod
    def calc(A, t):
        cfm = np.linalg.eigvals(A)
        rho = np.max(cfm)
        return t / rho


class Fraction(Scale):  # Forest, logForest, Comm, logComm, Heat, logHeat, SCT, SCCT
    @staticmethod
    def calc(A, t):
        return 0.5 * t / (1.0 - t)


class FractionReversed(Scale):  # RSP, FE
    @staticmethod
    def calc(A, beta):
        return (1.0 - beta) / beta
