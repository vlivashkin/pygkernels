from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_kernels
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import FLOAT_DTYPES

from pygraphs.cluster.base import KernelEstimator


class KMeans_Fouss(KernelEstimator, ABC):
    def __init__(self, n_clusters, n_init=10, max_rerun=10, max_iter=100, random_state=0):
        super().__init__(n_clusters)
        self.n_init = n_init
        self.max_rerun = max_rerun
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, K, y=None, sample_weight=None):
        self.labels_ = self.predict(K)
        return self

    def _init_h(self, K: np.array, rs: np.random.RandomState):
        n = K.shape[0]

        # initialization: choose one node for each cluster
        q_idx = np.arange(n)
        rs.shuffle(q_idx)
        q_idx = q_idx[:self.n_clusters]
        assert len(list(set(q_idx))) == self.n_clusters

        # initialization: h
        h = np.zeros((self.n_clusters, n))
        for i in range(self.n_clusters):
            h[i][q_idx[i]] = 1

        return h

    def _predict_successful_once(self, K: np.array, rs: np.random.RandomState):
        for i in range(self.max_rerun):
            labels, inertia, success = self._predict_once(K, rs)
            if success:
                return labels, inertia
        return labels, inertia

    @abstractmethod
    def _predict_once(self, K: np.array, rs: np.random.RandomState):
        pass

    def predict(self, K):
        rs = check_random_state(self.random_state)
        best_labels, best_inertia = [], float('+inf')
        for i in range(self.n_init):
            labels, inertia = self._predict_successful_once(K, rs)
            if inertia < best_inertia:
                best_labels = labels

        return best_labels


class KKMeans_vanilla(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.2: Simple kernel k-means clustering of nodes
    """

    name = 'KernelKMeans_vanilla'

    def _predict_once(self, K: np.array, rs: np.random.RandomState):
        n = K.shape[0]
        e = np.eye(n)

        U = np.zeros((n, self.n_clusters))
        nn = np.ones((self.n_clusters,))
        h = self._init_h(K, rs)

        labels, inertia = [0] * n, float('+inf')
        for _ in range(self.max_iter):
            U[:] = 0

            # fix h, update U
            for i in range(0, n):
                k_star = np.argmin([(h[k] - e[i])[None].dot(K).dot((h[k] - e[i])[None].T)
                                    for k in range(0, self.n_clusters)])
                U[i][k_star] = 1

            # fix U, update h
            for k in range(0, self.n_clusters):
                nn[k] = np.sum([U[i][k] for i in range(0, n)])
                if nn[k] == 0:  # empty cluster! exit with success=False
                    return labels, inertia, False
                h[k] = U[:, k] / nn[k]

            # early stop
            if np.all(labels == np.argmax(U, axis=1)):  # nothing changed
                break

            labels = np.argmax(U, axis=1)
            inertia = np.sum([(h[labels[i]] - e[i])[None].dot(K).dot((h[labels[i]] - e[i])[None].T)
                              for i in range(0, n)])

        return labels, inertia, True


class KKMeans_iterative(KMeans_Fouss):
    """Kernel K-means clustering
    Reference
    ---------
    Francois Fouss, Marco Saerens, Masashi Shimbo
    Algorithms and Models for Network Data and Link Analysis
    Algorithm 7.3: Simple iterative kernel k-means clustering of nodes
    """

    name = 'KernelKMeans_iterative'

    def _init_l_U_nn_h(self, n, K, e, rs):
        nn = np.zeros((self.n_clusters,), dtype=np.uint8)
        while np.any(nn == 0):  # check all clusters used
            l = np.zeros((n,), dtype=np.uint8)
            U = np.zeros((n, self.n_clusters), dtype=np.uint8)
            nn = np.zeros((self.n_clusters,), dtype=np.uint8)
            h = self._init_h(K, rs)

            for i in range(0, n):
                k_star = np.argmin([(h[k] - e[i])[None].dot(K).dot((h[k] - e[i])[None].T)
                                    for k in range(0, self.n_clusters)])
                l[i] = k_star; U[i][k_star] = 1
            for k in range(0, self.n_clusters):
                nn[k] = np.sum([U[i][k] for i in range(0, n)])
                h[k] = U[:, k] / nn[k]

        return l, U, nn, h

    def _predict_successful_once(self, K: np.array, rs: np.random.RandomState):
        for i in range(self.max_rerun):
            labels, inertia, success = self._predict_once(K, rs)
            if success:
                return labels, inertia
        print('reruns exceeded, take last result')
        return labels, inertia

    def _predict_once(self, K: np.array, rs: np.random.RandomState):
        n = K.shape[0]
        e = np.eye(n)
        eps = 10 ** -16

        l, U, nn, h = self._init_l_U_nn_h(n, K, e, rs)

        labels, inertia = [0] * n, float('+inf')
        for _ in range(100):
            for i in range(n):  # for each node
                ΔJ = np.zeros((self.n_clusters,))
                for k in range(self.n_clusters):
                    ΔJ[k] = nn[k] / (nn[k] + 1) * (h[k] - e[i])[None].dot(K).dot((h[k] - e[i])[None].T) - \
                            nn[l[i]] / (nn[l[i]] - 1 + eps) * (h[l[i]] - e[i])[None].dot(K).dot((h[l[i]] - e[i])[None].T)
                k_star = np.argmin(ΔJ)
                if ΔJ[k_star] < 0:
                    h[l[i]] = 1. / (nn[l[i]] - 1 + eps) * (nn[l[i]] * h[l[i]] - e[i])
                    h[k_star] = 1. / (nn[k_star] + 1) * (nn[k_star] * h[k_star] + e[i])
                    nn[k_star] += 1; nn[l[i]] -= 1
                    if nn[l[i]] == 0:  # empty cluster! exit with success=False
                        return labels, inertia, False
                    U[i, l[i]] = 0; U[i, k_star] = 1
                    l[i] = k_star

            # early stop
            if np.all(labels == np.argmax(U, axis=1)):  # nothing changed
                break

            labels = np.argmax(U, axis=1)
            inertia = np.sum([(h[labels[i]] - e[i])[None].dot(K).dot((h[labels[i]] - e[i])[None].T)
                              for i in range(0, n)])

        return labels, inertia, True


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
    init_choose_strategy : str, optional (default='median')
        Strategy of choosing best initialization: 'min', 'median', 'max'
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

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_init=10, init_choose_objective='inertia',
                 init_choose_strategy='max', dist_compensation_strategy='+40', kernel='precomputed', gamma='auto',
                 degree=3, coef0=1.0, kernel_params=None, random_state=None, verbose=0):
        super().__init__(n_clusters)
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.init_choose_objective = init_choose_objective
        self.init_choose_strategy = init_choose_strategy
        self.dist_compensation_strategy = dist_compensation_strategy
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

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
            raise ValueError(f"Invalid number of initializations. n_init={self.n_init} must be bigger than zero.")

        if self.max_iter <= 0:
            raise ValueError(f"Invalid iteration bound. max_iter={self.max_iter} must be greater than zero.")

        self.sample_weight_ = sample_weight if sample_weight is not None else np.ones(n_samples)
        K = self._get_kernel(X)
        rs = check_random_state(self.random_state)

        results = []
        for i in range(self.n_init):
            dist = np.zeros((n_samples, self.n_clusters))
            self.labels_ = rs.randint(self.n_clusters, size=n_samples)
            self.within_distances_ = np.zeros(self.n_clusters)

            for it in range(self.max_iter):
                dist.fill(0)
                self._compute_dist(K, dist, update_within=True)
                labels_old = self.labels_
                self.labels_ = dist.argmin(axis=1)

                # Compute the number of samples whose cluster did not change
                # since last iteration.
                n_same = np.sum((self.labels_ - labels_old) == 0).astype(np.float)
                if 1 - (n_same / n_samples) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {it + 1}, tol={1 - (n_same / n_samples)}")
                    break

            if it == self.max_iter - 1:
                print('n iter exceeded')

            # Computing inertia to choose the best initialization
            if self.dist_compensation_strategy == '+40':
                dist += 40
            elif self.dist_compensation_strategy == '<0->0':
                dist[dist < 0] = 0
            elif self.dist_compensation_strategy == '-min':
                dist -= np.min(dist)
            else:
                raise NotImplemented()
            assert np.all(dist >= 0)

            inertia = np.sum([d[l] ** 2 for d, l in zip(dist, self.labels_)])

            if self.verbose:
                labels_pred = self.labels_
                labels_true = y
                test_nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='geometric')
                print("Initialization %2d, inertia %.3f, nmi %.3f" % (i, inertia, test_nmi))

            result = {
                'inertia': inertia,
                'n_iter': it + 1,
                'labels': self.labels_.copy(),
                'distances': self.within_distances_.copy()
            }
            if y is not None:
                result['nmi'] = normalized_mutual_info_score(y, self.labels_, average_method='geometric')
                result['ari'] = adjusted_rand_score(y, self.labels_)
            results.append(result)

        results.sort(key=lambda x: x[self.init_choose_objective])
        if self.init_choose_strategy == 'max':
            result = results[-1]
        elif self.init_choose_strategy == 'min':
            result = results[0]
        elif self.init_choose_strategy == 'median':
            result = results[len(results) // 2]
        else:
            raise NotImplemented()

        self.labels_ = result['labels']
        self.within_distances_ = result['distances']
        self.n_iter_ = result['n_iter']
        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, update_within):
        """Compute a n_samples x n_clusters distance matrix using the kernel trick.
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
                self.within_distances_[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += self.within_distances_[j]

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
        self._compute_dist(K, dist, update_within=False)
        return dist.argmin(axis=1)
