from abc import ABC

from scipy.sparse.csgraph import shortest_path

from pygraphs.measure import scaler
from pygraphs.measure.shortcuts import *


class Distance(ABC):
    name, default_scaler, power = None, None, None
    parent_kernel_class = None

    def __init__(self, A):
        self.scaler = self.default_scaler(A)
        self.parent_kernel = self.parent_kernel_class(A) if self.parent_kernel_class else None
        self.A = A

    def get_D(self, param):
        H = self.parent_kernel.get_K(param)
        D = H_to_D(H)
        return np.power(D, self.power) if self.power else D

    def grid_search(self, params=np.linspace(0, 1, 55)):
        results = np.array((params.shape[0],))
        for idx, param in enumerate(self.scaler.scale_list(params)):
            results[idx] = self.get_D(param)
        return results


class SP(Distance):
    name, default_scaler = 'SP', scaler.Linear

    def get_D(self, param):
        with np.errstate(divide='ignore', invalid='ignore'):
            A = np.divide(1., self.A, where=lambda x: x != 0)
        return np.array(shortest_path(A, directed=False), dtype=np.float64)


class CT(Distance):
    name, default_scaler = 'CT', scaler.Linear

    def commute_distance(self):
        """
        Original code copyright (C) Ulrike Von Luxburg, Python implementation by James McDermott.
        """
        size = self.A.shape[0]
        L = get_L(self.A)

        Linv = np.linalg.inv(L + np.ones(L.shape) / size) - np.ones(L.shape) / size

        Linv_diag = np.diag(Linv).reshape((size, 1))
        Rexact = Linv_diag * np.ones((1, size)) + np.ones((size, 1)) * Linv_diag.T - 2 * Linv

        # convert from a resistance distance to a commute time distance
        vol = np.sum(self.A)
        Rexact *= vol

        return Rexact

    def get_D(self, param):
        return self.commute_distance()


class RSP_vanilla_like(Distance, ABC):
    def __init__(self, A, C=None):
        """
        P^{ref} = D^{-1}*A, D = Diag(A*e)
        """
        super().__init__(A)

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


@deprecated()
class RSP_vanilla(RSP_vanilla_like):
    name, default_scaler = 'RSP vanilla', scaler.FractionReversed

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


@deprecated()
class FE_vanilla(RSP_vanilla_like):
    name, default_scaler = 'FE vanilla', scaler.FractionReversed

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


# From https://github.com/jmmcd/GPDistance
class RSP_like(Distance, ABC):
    def __init__(self, A):
        super().__init__(A)

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


class RSP(RSP_like):
    name, default_scaler = 'RSP', scaler.FractionReversed

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


class FE(RSP_like):
    name, default_scaler = 'FE', scaler.FractionReversed

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
