from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cluster.k_means_ import _labels_inertia, _check_sample_weight
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import FLOAT_DTYPES

from pygraphs.cluster.base import KernelEstimator


# Francois Fouss, Marco Saerens, Masashi Shimbo
# Algorithms and Models for Network Data and Link Analysis
# Algorithm 7.2
class KKMeans_vanilla(KernelEstimator):
    name = 'KernelKMeans_vanilla'

    def __init__(self, n_clusters, max_iter=100, random_state=0):
        super().__init__(n_clusters)
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def predict(self, K):
        n = K.shape[0]
        U = np.zeros((n, self.n_clusters))

        # initialization
        rs = check_random_state(self.random_state)
        q_idx = rs.randint(0, n, size=(self.n_clusters,))
        h = np.zeros((self.n_clusters, n))
        for i in range(self.n_clusters):
            h[i][q_idx[i]] = 1
        e = np.eye(n)
        nn = np.zeros((self.n_clusters,))

        for i in range(100):
            U = np.zeros((n, self.n_clusters))
            for i in range(0, n):
                ka = np.argmin(
                    [(h[k] - e[i])[None].dot(K).dot((h[k] - e[i])[None].T) for k in range(0, self.n_clusters)])
                U[i][ka] = 1
            for k in range(0, self.n_clusters):
                nn[k] = np.sum([U[i][k] for i in range(0, n)])
                h[k] = U[:, k] / nn[k]

        return np.argmax(U, axis=1)


class KKMeans(KernelEstimator):
    """Kernel K-means clustering
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    Author: Mathieu Blondel <mathieu@mblondel.org>,
            Ishank Gulati <gulati.ishank@gmail.com>
    License: BSD 3 clause

    Parameters
    ----------
    n_clusters : int, optional (default=3)
        The number of clusters to be formed as well as number of centroids
        to be generated.
    max_iter : int, optional (default=300)
        Maximum number of iterations of algorithm for a single run.
    n_init : int, optional (default=10)
        Number of random initializations.
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    kernel : string, optional (default='linear')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'linear' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    coef0 : float, optional (default=1.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    kernel_params : dict, optional
        Additional kernel parameters that are only referenced when the
        kernel is a callable.
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data for probability estimation.
    verbose : int, optional (default=0)
        Verbosity mode.
    Attributes
    ----------
    sample_weight_ : array-like, shape=(n_samples,)
    labels_ : shape=(n_samples,)
        Labels of each training sample.
    within_distances_ : array, shape=(n_clusters,)
        Distance update.
    X_fit_ : array-like, shape=(n_samples, n_features)
        Data used in clustering.
    n_iter_ : Iteration in which algorithm converged
    Examples
    --------
    >>> from sklearn.cluster import KernelKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> ker_kmeans = KernelKMeans(n_clusters=2, random_state=0,
    ...                           kernel='rbf').fit(X)
    >>> ker_kmeans.labels_
    array([0, 1, 1, 0, 0, 1])
    >>> ker_kmeans.predict(np.array([[0, 0], [4, 4]]))
    array([1, 0])
    References
    ----------
    * Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis
      Kernel k-means, Spectral Clustering and Normalized Cuts
      http://www.cs.utexas.edu/users/inderjit/public_papers/kdd_spectral_kernelkmeans.pdf
    """

    name = 'KernelKMeans'

    def __init__(self, n_clusters, max_iter=300, tol=1e-3, n_init=10,
                 kernel='precomputed', gamma='auto', degree=3, coef0=1.0,
                 kernel_params=None, random_state=None, verbose=0, sklearn_inertia=True):
        super().__init__(n_clusters)
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

        self.sklearn_inertia = sklearn_inertia

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                        warn_on_dtype=True)
        n_samples, n_features = X.shape
        expected_n_features = self.X_fit_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))
        return X

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            if self.gamma == 'auto':
                params = {
                    "degree": self.degree,
                    "coef0": self.coef0
                }
            else:
                params = {
                    "gamma": self.gamma,
                    "degree": self.degree,
                    "coef0": self.coef0
                }
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
        sample_weight : array-like, shape=(n_samples,)
        """
        X = self._check_fit_data(X)
        n_samples = X.shape[0]

        if self.n_init <= 0:
            raise ValueError("Invalid number of initializations."
                             " n_init=%d must be bigger than zero."
                             % self.n_init)

        if self.max_iter <= 0:
            raise ValueError("Invalid iteration bound."
                             " max_iter=%d must be greater than zero."
                             % self.max_iter)

        best_labels, best_inertia, best_distances = None, None, None

        K = self._get_kernel(X)

        rs = check_random_state(self.random_state)

        sw = sample_weight if sample_weight is not None else np.ones(n_samples)
        self.sample_weight_ = sw
        for i in range(self.n_init):
            self.labels_ = rs.randint(self.n_clusters, size=n_samples)

            dist = np.zeros((n_samples, self.n_clusters))
            self.within_distances_ = np.zeros(self.n_clusters)

            for it in range(self.max_iter):
                dist.fill(0)
                self._compute_dist(K, dist, self.within_distances_,
                                   update_within=True)
                labels_old = self.labels_
                self.labels_ = dist.argmin(axis=1)

                # Compute the number of samples whose cluster did not change
                # since last iteration.
                n_same = np.sum((self.labels_ - labels_old) == 0).astype(np.float)
                if 1 - (n_same / n_samples) < self.tol:
                    if self.verbose:
                        print("Converged at iteration", it + 1)
                    break

            if 1 - (n_same / n_samples) > 0:
                dist.fill(0)
                self._compute_dist(K, dist, self.within_distances_,
                                   update_within=False)
                self.labels_ = dist.argmin(axis=1)

            if self.sklearn_inertia:
                # Keep only the best cluster centers across independent inits on
                # the common validation set
                sample_weight = _check_sample_weight(X, None)
                x_squared_norms = row_norms(X, squared=True)
                _, inertia = _labels_inertia(X, sample_weight, x_squared_norms, dist.transpose((1, 0)))
            else:
                # Computing inertia to choose the best initialization
                inertia = np.sum([d[l] ** 2 for d, l in zip(dist, self.labels_)])

            if self.verbose:
                print("Initialization %2d, inertia %.3f, ari %.3f" % (i, inertia))

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                n_iter = it + 1
                best_labels = self.labels_.copy()
                best_distances = self.within_distances_.copy()

        self.labels_ = best_labels.copy()
        self.within_distances_ = best_distances.copy()
        self.n_iter_ = n_iter
        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick.
        Parameters
        ----------
        K : Kernel matrix
        dist : array-like, shape=(n_samples, n_clusters)
            List with all elements initialized to zero.
        within_distances : array, shape=(n_clusters,)
            Distance update.
        update_within : bool
            To update within_distances or not.
        """
        sw = np.array(self.sample_weight_)
        for j in range(self.n_clusters):
            mask = self.labels_ == j

            # If cluster is empty, assign random labels and re-run
            if np.sum(mask) == 0:
                rs = check_random_state(self.random_state)
                n_samples = len(self.labels_)
                self.labels_ = rs.randint(self.n_clusters, size=n_samples)
                self.within_distances_.fill(0)
                break

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            New data to predict.
        Returns
        -------
        labels : array, shape = (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = self._check_test_data(X)
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)
