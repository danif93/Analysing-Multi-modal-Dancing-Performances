"""Sparse inverse covariance selection via ADMM.

More information can be found in the paper linked at:
http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
"""
from __future__ import division

import warnings

import numpy as np
from six.moves import range
from sklearn.covariance import GraphLasso, empirical_covariance
from sklearn.utils.extmath import fast_logdet
from sklearn.utils.validation import check_array

from group_lasso_utils.regain.norm import l1_od_norm
from group_lasso_utils.regain.prox import prox_logdet, soft_thresholding_sign
from group_lasso_utils.regain.update_rules import update_rho
from group_lasso_utils.regain.utils import convergence


def logl(emp_cov, precision):
    """Gaussian log-likelihood without constant term."""
    return fast_logdet(precision) - np.sum(emp_cov * precision)


def objective(S, X, Z, alpha):
    """Graph lasso objective."""
    return - logl(S, X) + alpha * l1_od_norm(Z)


def graph_lasso(
        emp_cov, alpha=0.01, rho=1, over_relax=1, max_iter=100, verbose=False,
        tol=1e-4, rtol=1e-4, return_history=False, return_n_iter=True,
        update_rho_options=None, compute_objective=True, mode='admm'):
    """Graph lasso solver via ADMM.

    Solves the following problem via ADMM:
        minimize  trace(S*X) - log det X + alpha ||X||_{od,1}

    where S is the empirical covariance of the data
    matrix D (training observations by features).

    Parameters
    ----------
    emp_cov : array-like
        Empirical covariance matrix.
    alpha : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    over_relax : float, optional
        Over-relaxation parameter (typically between 1.0 and 1.8).
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.
    return_n_iter : bool, optional
        Return the number of iteration before convergence.
    verbose : bool, default False
        Print info at each iteration.

    Returns
    -------
    X : numpy.array, 2-dimensional
        Solution to the problem.
    S : np.array, 2 dimensional
        Empirical covariance matrix.
    n_iter : int
        If return_n_iter, returns the number of iterations before convergence.
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    Z = np.zeros_like(emp_cov)
    U = np.zeros_like(emp_cov)
    Z_old = np.zeros_like(Z)

    checks = []
    for iteration_ in range(max_iter):
        # x-update
        A = Z - U
        A += A.T
        A /= 2.
        K = prox_logdet(emp_cov - rho * A, lamda=1. / rho)

        # z-update with relaxation
        K_hat = over_relax * K - (1 - over_relax) * Z
        Z = soft_thresholding_sign(K_hat + U, lamda=alpha / rho)

        # update residuals
        U += K_hat - Z

        # diagnostics, reporting, termination checks
        obj = objective(emp_cov, K, Z, alpha) if compute_objective else np.nan
        rnorm = np.linalg.norm(K - Z, 'fro')
        snorm = rho * np.linalg.norm(Z - Z_old, 'fro')
        check = convergence(
            obj=obj, rnorm=rnorm, snorm=snorm,
            e_pri=np.sqrt(K.size) * tol + rtol * max(
                np.linalg.norm(K, 'fro'), np.linalg.norm(Z, 'fro')),
            e_dual=np.sqrt(K.size) * tol + rtol * rho * np.linalg.norm(U)
        )

        Z_old = Z.copy()
        if verbose:
            print("obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                  "eps_pri: %.4f, eps_dual: %.4f" % check)

        checks.append(check)
        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        rho_new = update_rho(rho, rnorm, snorm, iteration=iteration_,
                             **(update_rho_options or {}))
        # scaled dual variables should be also rescaled
        U *= rho / rho_new
        rho = rho_new
    else:
        warnings.warn("Objective did not converge.")

    return_list = [Z, emp_cov]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_)
    return return_list


class GraphLasso(GraphLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        The regularization parameter: the higher alpha, the more
        regularization, the sparser the inverse covariance.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

    over_relax : positive float, deafult 1
        Over-relaxation parameter (typically between 1.0 and 1.8).

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    rtol : positive float, default 1e-4
        Relative tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    update_rho_options : dict, default None
        Options for the update of rho. See `update_rho` function for details.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    mode : {'admm'}, default 'admm'
        Minimisation algorithm. At the moment, only 'admm' is available,
        so this is ignored.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    n_iter_ : int
        Number of iterations run.

    """

    def __init__(self, alpha=0.01, rho=1., over_relax=1., max_iter=100,
                 mode='admm', tol=1e-4, rtol=1e-4, verbose=False,
                 assume_centered=False, update_rho_options=None,
                 compute_objective=True):
        super(GraphLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter, verbose=verbose,
            assume_centered=assume_centered, mode=mode)
        self.rho = rho
        self.rtol = rtol
        self.over_relax = over_relax
        self.update_rho_options = update_rho_options or {}
        self.compute_objective = compute_objective

    def fit(self, X, y=None):
        """Fits the GraphLasso model to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data from which to compute the covariance estimate
        y : (ignored)

        """
        # Covariance does not make sense for a single feature
        X = check_array(X, ensure_min_features=2, ensure_min_samples=2,
                        estimator=self)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        emp_cov = empirical_covariance(X, assume_centered=self.assume_centered)
        self.precision_, self.covariance_, self.n_iter_ = graph_lasso(
            emp_cov, alpha=self.alpha, tol=self.tol, rtol=self.rtol,
            max_iter=self.max_iter, over_relax=self.over_relax, rho=self.rho,
            verbose=self.verbose, return_n_iter=True, return_history=False,
            mode=self.mode, update_rho_options=self.update_rho_options,
            compute_objective=self.compute_objective)
        return self
