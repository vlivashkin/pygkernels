from . import kernel
from . import scaler
from .shortcuts import *


class Distance:
    def __init__(self, name, scaler, A, parent_kernel=None, power=1.):
        self.name = name
        self.scaler = scaler(A)
        self.A = A
        self.parent_kernel = parent_kernel(A) if parent_kernel is not None else None
        self.power = power

    def get_D(self, param):
        H = self.parent_kernel.get_K(param)
        D = H_to_D(H)
        return np.power(D, self.power) if self.power != 1 else D

    def grid_search(self, params=np.linspace(0, 1, 55)):
        results = np.array((params.shape[0],))
        for idx, param in enumerate(self.scaler.scale(params)):
            results[idx] = self.get_D(param)
        return results

    @staticmethod
    def get_all():
        return [pWalk, Walk, For, logFor, Comm, logComm, Heat, logHeat, SCT, SCCT, RSP, FE, SPCT]


class pWalk(Distance):
    def __init__(self, A):
        super().__init__('pWalk', scaler.Rho, A, kernel.pWalk_H, 1.)


class Walk(Distance):
    def __init__(self, A):
        super().__init__('Walk', scaler.Rho, A, kernel.Walk_H, 1.)


class For(Distance):
    def __init__(self, A):
        super().__init__('For', scaler.Fraction, A, kernel.For_H, 1.)


class logFor(Distance):
    def __init__(self, A):
        super().__init__('logFor', scaler.Fraction, A, kernel.logFor_H, 1.)


class Comm(Distance):
    def __init__(self, A):
        super().__init__('Comm', scaler.Fraction, A, kernel.Comm_H, .5)


class logComm(Distance):
    def __init__(self, A):
        super().__init__('logComm', scaler.Fraction, A, kernel.logComm_H, .5)


class Heat(Distance):
    def __init__(self, A):
        super().__init__('Heat', scaler.Fraction, A, kernel.Heat_H, 1.)


class logHeat(Distance):
    def __init__(self, A):
        super().__init__('logHeat', scaler.Fraction, A, kernel.logHeat_H, 1.)


class SCT(Distance):
    def __init__(self, A):
        super().__init__('SCT', scaler.Fraction, A, kernel.SCT_H, 1.)


class SCCT(Distance):
    def __init__(self, A):
        super().__init__('SCCT', scaler.Fraction, A, kernel.SCCT_H, 1.)


class RSP_like(Distance):
    def __init__(self, name, A, C=None):
        """
        P^{ref} = D^{-1}*A, D = Diag(A*e)
        """
        super().__init__(name, scaler.FractionReversed, A)
        self.size = A.shape[0]
        self.e = np.ones((self.size, 1))
        self.I = np.eye(self.size)
        self.Pref = get_D_1(A).dot(A)

        # self.C = johnson(A, directed=False)
        eps = 0.00000001
        max = np.finfo('d').max

        if C is None:
            self.C = A.copy()
            self.C[A >= eps] = 1.0 / A[A >= eps]
            self.C[A < eps] = max
        else:
            self.C = C

    def WZ(self, beta):
        W = self.Pref * np.exp(-beta * self.C)
        Z = np.linalg.pinv(self.I - W)
        return W, Z


class RSP(RSP_like):
    def __init__(self, A):
        super().__init__('RSP', A)

    def get_D(self, beta):
        """
        W = P^{ref} ◦ exp(-βC); ◦ is element-wise *
        Z = (I - W)^{-1}
        S = (Z(C ◦ W)Z)÷Z; ÷ is element-wise /
        C_ = S - e(d_S)^T; d_S = diag(S)
        Δ_RSP = (C_ + C_^T)/2
        """
        W, Z = self.WZ(beta)
        S = (Z.dot(self.C * W).dot(Z)) / Z
        RSP = S - np.ones((self.size, 1)).dot(np.diag(S).reshape((-1, 1)).transpose())
        D_RSP = 0.5 * (RSP + RSP.transpose())
        np.fill_diagonal(D_RSP, 0.0)
        return D_RSP


class FE(RSP_like):
    def __init__(self, A):
        super().__init__('FE', A)

    def get_D(self, beta):
        """
        W = P^{ref} (element-wise)* exp(-βC)
        Z = (I - W)^{-1}
        Z^h = Z * D_h^{-1}, D_h = Diag(Z)
        Φ = -1/β * log(Z^h)
        Δ_FE = (Φ + Φ^T)/2
        """
        W, Z = self.WZ(beta)
        Dh = np.diag(np.diag(Z))
        Zh = Z.dot(np.linalg.pinv(Dh))
        FE = np.log(Zh) / -beta
        D_FE = 0.5 * (FE + FE.transpose())
        np.fill_diagonal(D_FE, 0.0)
        return D_FE


class old_RSP(RSP):
    def __init__(self, A):
        super().__init__(A)
        self.name = 'old RSP'
        self.C = shortest_path(A, directed=False)


class old_FE(FE):
    def __init__(self, A):
        super().__init__(A)
        self.name = 'old FE'
        self.C = shortest_path(A, directed=False)


# From https://github.com/jmmcd/GPDistance
class GPD_RSP_like(Distance):
    def __init__(self, name, A):
        super().__init__(name, scaler.FractionReversed, A)

        max = np.finfo('d').max
        eps = 0.00000001

        # If A is integer-valued, and beta is floating-point, can get an
        # error in the matrix inversion, so convert A to float here. I
        # can't explain why beta being floating-point is related to the
        # problem. Anyway, this also converts in case it was a matrix, or
        # was sparse.
        A = np.array(A, dtype=np.float)

        A[A < eps] = 0.0
        self.n, m = A.shape
        if self.n != m:
            raise ValueError("The input matrix A must be square")

        self.C = A.copy()
        self.C[A >= eps] = 1.0 / A[A >= eps]
        self.C[A < eps] = max

        self.onesT = np.ones((self.n, 1))
        self.I = np.eye(self.n)

        # Computation of Pref, the reference transition probability matrix
        tmp = A.copy()
        s = np.sum(tmp, 1)
        s[s == 0] = 1  # avoid zero-division
        self.Pref = tmp / (s * self.onesT).T

    def WZ(self, beta):
        # Computation of the W and Z matrices
        W = np.exp(-beta * self.C) * self.Pref

        # compute Z
        Z = np.linalg.inv(self.I - W)
        return W, Z


class GPD_RSP(GPD_RSP_like):
    def __init__(self, A):
        super().__init__('RSP 2', A)

    def get_D(self, beta):
        W, Z = self.WZ(beta)

        # Computation of Z*(C.*W)*Z avoiding zero-division errors:
        numerator = np.dot(np.dot(Z, (self.C * W)), Z)
        D_nonabs = np.zeros((self.n, self.n))

        indx = (numerator > 0) & (Z > 0)
        D_nonabs[indx] = numerator[indx] / Z[indx]
        D_nonabs[~indx] = np.infty
        # D_nonabs above actually gives the expected costs of non-hitting paths
        # from i to j.

        # Expected costs of hitting paths -- avoid a possible inf-inf
        # which can arise with isolated nodes and would give a NaN -- we
        # prefer to have inf in that case.
        C_RSP = np.zeros((self.n, self.n))
        diag_D = np.dot(self.onesT, np.diag(D_nonabs).reshape((1, self.n)))
        indx = ~np.isinf(diag_D)
        C_RSP[indx] = D_nonabs[indx] - diag_D[indx]
        C_RSP[~indx] = np.infty

        # symmetrization
        D_RSP = 0.5 * (C_RSP + C_RSP.T)

        # Just in case, set diagonals to zero:
        np.fill_diagonal(D_RSP, 0.0)

        return D_RSP


class GPD_FE(GPD_RSP_like):
    def __init__(self, A):
        super().__init__('FE 2', A)

    def get_D(self, beta):
        W, Z = self.WZ(beta)

        # Free energies and symmetrization:
        Dh_1 = np.diag(1.0 / np.diag(Z))
        Zh = np.dot(Z, Dh_1)

        # If there any 0 values in Zh (because of isolated nodes), taking
        # log will raise a divide-by-zero error -- ignore it
        np.seterr(divide='ignore')
        FE = -np.log(Zh) / beta
        np.seterr(divide='raise')
        D_FE = 0.5 * (FE + FE.T)

        # Just in case, set diagonals to zero:
        np.fill_diagonal(D_FE, 0.0)

        return D_FE


class SPCT(Distance):
    def __init__(self, A):
        super().__init__('SP-CT', scaler.Linear, A)
        self.D_SP = sp_distance(A)
        self.D_CT = 2. * H_to_D(resistance_kernel(A))

    def get_D(self, lmbda):
        # when lambda = 0 this is CT, when lambda = 1 this is SP
        return lmbda * self.D_SP + (1. - lmbda) * self.D_CT
