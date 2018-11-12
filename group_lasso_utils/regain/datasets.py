"""Dataset generation module."""
from __future__ import division
import numpy as np

from sklearn.datasets.base import Bunch


def normalize_matrix(x):
    """Normalize a matrix so to have 1 on the diagonal, in-place."""
    d = np.diag(x).reshape(1, x.shape[0])
    d = 1. / np.sqrt(d)
    x *= d
    x *= d.T


def is_pos_def(x, tol=1e-15):
    """Check if x is positive definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs > 0)


def is_pos_semidef(x, tol=1e-15):
    """Check if x is positive semi-definite."""
    eigs = np.linalg.eigvalsh(x)
    eigs[np.abs(eigs) < tol] = 0
    return np.all(eigs >= 0)


def generate_dataset(n_samples=100, n_dim_obs=100, n_dim_lat=10, T=10,
                     mode="evolving", **kwargs):
    """Generate a synthetic dataset.

    Parameters
    ----------
    n_samples: int,
        number of samples to generate
    n_dim_obs: int,
        number of observed variables of the graph
    n_dim_lat: int,
        number of latent variables of the graph
    T: int,
        number of times
    mode: string,
        "evolving": generate a dataset with evolving observed and latent
                    variables that have a small Frobenious norm between two
                    close time points
        "fixed": generate a dataset with evolving observed and fixed latent
                    variables that have a small Frobenious norm between two
                    close time points
        "l1": generate a dataset with evolving observed and latent variables
              that differs for a small l1 norm
        "l1l2": generate a dataset with evolving observed variables that
                differs for a small l1 norm and evolving latent variables
                that differs for a small l2 norm
        "sin": generate a dataset with fixed latent variables and evolving
                observed variables that are generated from sin functions.
    *kwargs: other arguments related to each specific data generation mode

    """
    modes = dict(
        evolving=generate_dataset_with_evolving_L,
        fixed=generate_dataset_with_fixed_L,
        fixedl1=make_l1_fixed_ell,
        fede=generate_dataset_fede,
        l1=generate_dataset_L1,
        l1l2=generate_dataset_L1L2,
        sin=generate_dataset_sin_cos,
        sklearn=make_sparse_low_rank,
        fixed_sparsity=make_fixed_sparsity,
        ma=make_ma_xue_zou, mak=make_ma_xue_zou_rand_k,
        norm=make_evolving, l1l1=generate_dataset_l1l1)
    func = modes.get(mode, None)
    if func is None:
        raise ValueError("Unknown mode %s. "
                         "Choices are: %s" % (mode, modes.keys()))

    n_dim_obs = int(n_dim_obs)
    n_dim_lat = int(n_dim_lat)
    n_samples = int(n_samples)

    thetas, thetas_obs, ells = func(n_dim_obs, n_dim_lat, T, **kwargs)
    sigmas = map(np.linalg.inv, thetas_obs)
    # map(normalize_matrix, sigmas)  # in place

    data_list = np.array([np.random.multivariate_normal(
        np.zeros(n_dim_obs), sigma, size=n_samples) for sigma in sigmas])
    return Bunch(data=data_list, thetas=np.array(thetas),
                 thetas_observed=np.array(thetas_obs),
                 ells=np.array(ells))


def make_ell(n_dim_obs=100, n_dim_lat=10):
    """Doc."""
    K_HO = np.zeros((n_dim_lat, n_dim_obs))
    for i in range(n_dim_lat):
        percentage = int(n_dim_obs * 0.8)
        indices = np.random.randint(0, high=n_dim_obs, size=percentage)
        K_HO[i, indices] = np.random.rand(percentage) * 0.12

    K_HO /= np.sum(K_HO, axis=1)[:, None] / 2
    L = K_HO.T.dot(K_HO)
    assert(is_pos_semidef(L))
    assert np.linalg.matrix_rank(L) == n_dim_lat
    # from sklearn.datasets import make_low_rank_matrix
    # L = make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)
    # L = (L + L.T) / 2.
    # print L
    return L, K_HO


def generate_starting_matrices(n_dim_obs=100, n_dim_lat=10, degree=2):
    """Doc."""
    L, K_HO = make_ell(n_dim_obs, n_dim_lat)
    theta = np.eye(n_dim_obs)
    for i in range(n_dim_obs):
        possible_idx = list(set(range(n_dim_obs)) - (
            set(np.nonzero(theta[i, :])[0]) |
            set(np.where(np.count_nonzero(theta, axis=1) > degree)[0])))
        if not possible_idx:
            continue
        indices = np.random.choice(
            possible_idx, degree - (np.count_nonzero(theta[i, :]) - 1))
        theta[i, indices] = theta[indices, i] = .5 / degree

    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))
    return theta, theta_observed, L, K_HO


def generate_starting_matrices_normalized(n_dim_obs=100, n_dim_lat=10, degree=2):
    """Doc."""
    L, K_HO = make_ell(n_dim_obs, n_dim_lat)
    theta = np.zeros((n_dim_obs, n_dim_obs))
    for i in range(n_dim_obs):
        possible_idx = list(set(range(n_dim_obs)) - (
            set(np.nonzero(theta[i, :])[0]) |
            set(np.where(np.count_nonzero(theta, axis=1) > degree)[0])))
        if not possible_idx:
            continue
        indices = np.random.choice(
            possible_idx, degree - (np.count_nonzero(theta[i, :]) - 1))
        theta[i, indices] = theta[indices, i] = 1. / degree

    theta.flat[::n_dim_obs+1] = np.sum(theta, axis=1) + 0.002

    assert(is_pos_def(theta))
    theta_observed = theta - L
    assert(is_pos_def(theta_observed))
    return theta, theta_observed, L, K_HO


def update_theta(
        theta_old, n_dim_obs, degree, epsilon, keep_sparsity=False,
        indices=None):
    addition = np.zeros_like(theta_old)
    for i in range(n_dim_obs):
        if keep_sparsity:
            ii = indices[i]
        else:
            ii = np.random.randint(0, n_dim_obs, size=degree)
        addition[i, ii] = np.random.randn(len(ii))
    addition[np.triu_indices(n_dim_obs)[::-1]] = \
        addition[np.triu_indices(n_dim_obs)]
    addition *= epsilon / np.linalg.norm(addition)
    np.fill_diagonal(addition, 0)
    theta = theta_old + addition
    theta[np.abs(theta) < 2 * epsilon / n_dim_obs] = 0
    return theta


def perturb_theta_l1(theta_init, no, n_dim_obs):
    theta = theta_init.copy()
    rows = np.zeros(no)
    cols = np.zeros(no)
    while (np.any(rows == cols)):
        rows = np.random.randint(0, n_dim_obs, no)
        cols = np.random.randint(0, n_dim_obs, no)
    for r, c in zip(rows, cols):
        theta[r, c] = np.random.choice([0.12, 0, 0]) if theta[r, c] == 0 else .06  # np.random.rand(1) * .35
        theta[c, r] = theta[r, c]
    assert(is_pos_def(theta))
    return theta


def generate_dataset_l1l1(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate matrices according to a l1-l1 model."""
    degree = kwargs.get('degree', 2)
    start_matrix_function = generate_starting_matrices_normalized \
        if kwargs.get('normalize_starting_matrices', True) else \
        generate_starting_matrices
    no = int(np.ceil(n_dim_obs / 20)) if kwargs.get('proportional', False) else 1

    theta, theta_observed, L, K_HO = start_matrix_function(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1, T):
        theta = perturb_theta_l1(thetas[-1], no, n_dim_obs)

        K_HO = K_HOs[-1].copy()
        picks = np.random.permutation(K_HO.size)[:no]
        K_HO = K_HO.ravel()
        for p in picks:
            K_HO[p] = np.random.choice([0.12, 0, 0]) if K_HO[p] == 0 else 0
        K_HO = np.reshape(K_HO, (n_dim_lat, n_dim_obs))
        L = K_HO.T.dot(K_HO)

        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_L1L2(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """DESCRIZIONE, PRIMA O POI."""
    degree = kwargs.get('degree', 2)
    epsilon = kwargs.get('epsilon', 1e-2)
    no = int(np.ceil(n_dim_obs / 20)) if kwargs.get('proportional', False) else 1

    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1, T):
        theta = perturb_theta_l1(thetas[-1], no, n_dim_obs)

        L, K_HO = update_ell_l2(K_HOs[-1], epsilon, n_dim_obs)
        K_HOs.append(K_HO)

        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_L1(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Yuan (2012) model."""
    # number of links for each node
    degree = kwargs.get('degree', 2)

    # if proportional, we change more than 1 links at the time
    # in particular, n_observed / 20 links
    no = int(np.ceil(n_dim_obs / 20)) if kwargs.get('proportional', False) else 1

    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    for i in range(1, T):
        theta = perturb_theta_l1(thetas[-1], no, n_dim_obs)

        K_HO = K_HOs[-1].copy()
        c = np.random.randint(0, n_dim_obs, 1)
        r = np.random.randint(0, n_dim_lat, 1)
        K_HO[r, c] = 0.12 if K_HO[r, c] == 0 else 0
        # K_HO[c,r] = K_HO[r,c]

        L = K_HO.T.dot(K_HO)
        assert np.linalg.matrix_rank(L) == n_dim_lat
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def update_ell_l2(K_HO_old, epsilon, n_dim_obs):
    K_HO = K_HO_old.copy()
    addition = np.random.rand(*K_HO.shape)
    addition *= epsilon / np.linalg.norm(addition)
    K_HO += addition
    K_HO /= np.sum(K_HO, axis=1)[:, None] / 2.
    # K_HO *= 0.12
    K_HO[np.abs(K_HO) < epsilon / n_dim_obs] = 0
    return K_HO.T.dot(K_HO), K_HO


def generate_dataset_with_evolving_L(
        n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate dataset with evolving L."""
    degree = kwargs.get('degree', 2)
    epsilon = kwargs.get('epsilon', 1e-2)
    keep_sparsity = kwargs.get('keep_sparsity', False)

    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    idx = [np.nonzero(row)[0] for row in theta] if keep_sparsity else None
    for i in range(1, T):
        theta = update_theta(thetas[-1], n_dim_obs, degree, epsilon,
                             keep_sparsity=keep_sparsity, indices=idx)

        # for j in range(n_dim_obs):
        #     indices = list(np.where(theta[j,:]!=0)[0])
        #     indices.remove(j)
        #     if(len(indices)>degree):
        #         choice = np.random.choice(indices, len(indices)-degree)
        #         theta[j,choice] = 0
        #         theta[choice,j] = 0

        assert(is_pos_def(theta))

        L, K_HO = update_ell_l2(K_HOs[-1], epsilon, n_dim_obs)
        K_HOs.append(K_HO)

        assert(np.linalg.matrix_rank(L) == n_dim_lat)
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))
        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def make_evolving(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate dataset with evolving L."""
    degree = kwargs.get('degree', 2)
    epsilon = kwargs.get('epsilon', 1e-2)
    keep_sparsity = kwargs.get('keep_sparsity', False)

    theta, theta_observed, L, K_HO = generate_starting_matrices_normalized(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    ells = [L]
    K_HOs = [K_HO]

    idx = [np.nonzero(row)[0] for row in theta] if keep_sparsity else None
    for i in range(1, T):
        theta = update_theta(thetas[-1], n_dim_obs, degree, epsilon,
                             keep_sparsity=keep_sparsity, indices=idx)

        # for j in range(n_dim_obs):
        #     indices = list(np.where(theta[j,:]!=0)[0])
        #     indices.remove(j)
        #     if(len(indices)>degree):
        #         choice = np.random.choice(indices, len(indices)-degree)
        #         theta[j,choice] = 0
        #         theta[choice,j] = 0

        assert(is_pos_def(theta))

        L, K_HO = update_ell_l2(K_HOs[-1], epsilon, n_dim_obs)
        K_HOs.append(K_HO)

        assert(np.linalg.matrix_rank(L) == n_dim_lat)
        assert(is_pos_semidef(L))
        assert(is_pos_def(theta - L))
        thetas.append(theta)
        thetas_obs.append(theta - L)
        ells.append(L)
        K_HOs.append(K_HO)

    return thetas, thetas_obs, ells


def generate_dataset_with_fixed_L(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate precisions with a fixed L matrix."""
    degree = kwargs.get('degree', 2)
    epsilon = kwargs.get('epsilon', 1e-2)
    keep_sparsity = kwargs.get('keep_sparsity', False)

    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    idx = [np.nonzero(row)[0] for row in theta] if keep_sparsity else None
    for i in range(1, T):
        theta = update_theta(thetas[-1], n_dim_obs, degree, epsilon,
                             keep_sparsity=keep_sparsity, indices=idx)

        # for j in range(n_dim_obs):
        #     indices = list(np.where(theta[j, :] != 0)[0])
        #     indices.remove(j)
        #     if len(indices) > degree:
        #         choice = np.random.choice(indices, len(indices) - degree)
        #         theta[choice, j] = theta[j, choice] = 0

        assert(is_pos_def(theta))
        assert(is_pos_def(theta - L))
        thetas.append(theta)
        thetas_obs.append(theta - L)

    return thetas, thetas_obs, np.array([L] * T)


def make_fixed_sparsity(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate precisions with a fixed L matrix and sparsity."""
    degree = kwargs.get('degree', 2)
    epsilon = kwargs.get('epsilon', 1e-2)
    t = kwargs.get('changing_time', T / 2.)
    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    theta = np.abs(theta)
    theta_observed = theta - L
    nonzeros = np.nonzero(theta-np.diag(theta))

    thetas = [theta]
    thetas_obs = [theta_observed]

    for i in range(1, T):
        theta = thetas[-1].copy()
        if t < T:
            theta[nonzeros] += np.random.randn(nonzeros[0].size) * 0.1
        else:
            theta[nonzeros] -= np.random.randn(nonzeros[0].size) * 0.1
        theta = np.abs(theta)
        theta = (theta + theta.T) / 2.
        print(theta)
        theta[theta < epsilon] = 0
        theta.flat[::n_dim_obs + 1] = np.sum(np.abs(theta), axis=1) \
                                    + np.sum(np.abs(L), axis=1) + .1
        nonzeros = np.nonzero(theta-np.diag(theta))

        theta_observed = theta - L
        print(theta_observed)
        assert is_pos_def(theta_observed)
        thetas.append(theta)
        thetas_obs.append(theta_observed)

    return thetas, thetas_obs, np.array([L] * T)


def make_l1_fixed_ell(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Generate precisions with a fixed L matrix."""
    degree = kwargs.get('degree', 2)
    no = int(np.ceil(n_dim_obs / 20)) if kwargs.get('proportional', False) else 1

    theta, theta_observed, L, K_HO = generate_starting_matrices(
        n_dim_obs, n_dim_lat, degree)

    thetas = [theta]
    thetas_obs = [theta_observed]
    for i in range(1, T):
        theta = perturb_theta_l1(thetas[-1], no, n_dim_obs)
        assert(is_pos_def(theta - L))

        thetas.append(theta)
        thetas_obs.append(theta - L)

    return thetas, thetas_obs, np.array([L] * T)


def generate_dataset_sin_cos(n_dim_obs=100, n_dim_lat=10, T=10, **kwargs):
    """Aggiungi descrizione."""
    degree = kwargs.get('degree', 2)
    eps = kwargs.get('epsilon', 1e-2)
    L, K_HO = make_ell(n_dim_obs, n_dim_lat, degree)

    phase = np.random.randn(n_dim_obs, n_dim_obs) * np.pi
    phase[np.triu_indices(n_dim_obs)[::-1]] = phase[np.triu_indices(n_dim_obs)]

    clip = np.zeros((n_dim_obs, n_dim_obs))
    picks = np.random.permutation(len(np.triu_indices(n_dim_obs, 1)[0]))
    dim = int(len(np.triu_indices(n_dim_obs, 1)[0]) * degree)
    picks = picks[:dim]
    clip1 = clip[np.triu_indices(n_dim_obs, 1)].ravel()
    clip1[picks] = 1
    clip[np.triu_indices(n_dim_obs, 1)[::-1]] = clip1
    clip[np.triu_indices(n_dim_obs, 1)] = clip1

    thetas = np.array([np.eye(n_dim_obs) for i in range(T)])

    x = np.linspace(0, 2*np.pi, T)
    for i in range(T):
        for r in range(n_dim_obs):
            for c in range(n_dim_obs):
                if r == c:
                    continue
                if clip[r, c]:
                    thetas[i, r, c] = np.sin((x[i]+phase[r, c])/T**2)
                else:
                    thetas[i, r, c] = np.sin((x[i]+phase[r, c]))
        thetas[i][clip == 1] = np.clip(thetas[i][clip == 1], 0, 1)
        thetas[i][np.abs(thetas[i]) < eps] = 0

        assert(is_pos_def(thetas[i]))
        theta_observed = thetas[i] - L
        assert(is_pos_def(theta_observed))
        thetas_obs = [theta_observed]

    return thetas, thetas_obs, np.array([L]*T)


def generate_dataset_fede(
        n_dim_obs=3, n_dim_lat=2, T=10, epsilon=1e-3, n_samples=50, **kwargs):
    """Generate dataset (new version)."""
    b = np.random.rand(1, n_dim_obs)
    es, Q = np.linalg.eigh(b.T.dot(b))  # Q random

    b = np.random.rand(1, n_dim_obs)
    es, R = np.linalg.eigh(b.T.dot(b))  # R random

    start_sigma = np.random.rand(n_dim_obs) + 1
    start_lamda = np.zeros(n_dim_obs)
    idx = np.random.randint(n_dim_obs, size=n_dim_lat)
    start_lamda[idx] = np.random.rand(n_dim_lat)

    Ks = []
    Ls = []
    Kobs = []

    for i in range(T):
        K = np.linalg.multi_dot((Q, np.diag(start_sigma), Q.T))
        L = np.linalg.multi_dot((R, np.diag(start_lamda), R.T))

        K[np.abs(K) < epsilon] = 0  # enforce sparsity on K

        # assert is_pos_def(K - L)
        # assert is_pos_semidef(L)

        start_sigma += 1 + np.random.rand(n_dim_obs)
        start_lamda[idx] += np.random.rand(n_dim_lat) * 2 - 1
        start_lamda = np.maximum(start_lamda, 0)

        Ks.append(K)
        Ls.append(L)
        Kobs.append(K - L)

    return Ks, Kobs, Ls


def make_sparse_low_rank(
        n_dim_obs=3, n_dim_lat=2, T=10, epsilon=1e-3, n_samples=50, **kwargs):
    """Generate dataset (new new version)."""
    from sklearn.datasets import make_sparse_spd_matrix, make_low_rank_matrix

    K = make_sparse_spd_matrix(n_dim_obs)
    L = make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)

    Ks = [K]
    Ls = [L]
    Kobs = [K - L]

    for i in range(1, T):
        K = K + make_sparse_spd_matrix(n_dim_obs)
        L = L + make_low_rank_matrix(n_dim_obs, n_dim_obs, effective_rank=n_dim_lat)

        # assert is_pos_def(K - L)
        # assert is_pos_semidef(L)

        Ks.append(K)
        Ls.append(L)
        Kobs.append(K - L)

    return Ks, Kobs, Ls


def make_ma_xue_zou(n_dim_obs=12, n_latent=3, T=1, epsilon=1e-3, sparsity=0.1):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    # p = n_dim_obs + n_latent  # int(n_dim_obs * 0.05)
    p = n_dim_obs + int(n_dim_obs * 0.05)
    po = n_dim_obs
    ph = p - n_dim_obs
    W = np.zeros((p, p))
    non_zeros = int(round(p*p*sparsity))
    picks = np.random.permutation(p*p)[:non_zeros]
    W = W.ravel(order='F')
    W[picks] = np.random.randn(non_zeros)
    W = np.reshape(W, (p, p), order="F")

    C = W.T.dot(W)
    C[:po, po:] += 0.5 * np.random.randn(po, ph)
    C = (C + C.T) / 2.

    C = np.clip(C - np.diag(np.diag(C)), -1, 1)
    eig, Q = np.linalg.eigh(C)
    K = C + max(-1.2 * np.min(eig), 0.001) * np.eye(p)
    K_O = K[:po, :po]
    K_OH = K[:po, po:]
    K_HO = K[po:, :po]
    K_H = K[po:, po:]

    # L = np.divide(K_OH, K_H.dot(K_HO))
    assert np.allclose(K_OH, K_HO.T)
    L = np.linalg.multi_dot((K_OH, np.linalg.inv(K_H), K_HO))
    K_O_tilde = K_O - L
    assert is_pos_def(K_O_tilde)
    assert is_pos_semidef(K_H)
    assert np.linalg.matrix_rank(L) == ph
    # print(ph)

    N = 5 * po
    print("Note that, with this method, the n_samples should be %d" % N)
    return [K_O] * T, [K_O_tilde] * T, [L] * T


def make_ma_xue_zou_rand_k(
        n_dim_obs=12, n_latent=3, T=1, epsilon=1e-3, sparsity=0.1):
    """Generate the dataset as in Ma, Xue, Zou (2012)."""
    # p = n_dim_obs + n_latent  # int(n_dim_obs * 0.05)
    p = n_dim_obs + int(n_dim_obs * 0.05)
    po = n_dim_obs
    ph = p - n_dim_obs
    nnzr = int(sparsity * (np.triu_indices(p, 1)[0].size))

    # Generate A, the original inverse covariance, with random sparsity pattern...
    A = np.eye(p)
    idx = np.vstack(np.triu_indices(p, 1))
    idx = idx[:, np.random.choice(idx.shape[1], nnzr, replace=False)]
    idx = (idx[0], idx[1])
    A[idx] = np.sign(np.random.rand(nnzr) - .5)
    A[np.triu_indices(p, 1)[::-1]] = A[np.triu_indices(p, 1)]

    # A is the gound truth inverse covariance matrix
    K = A.dot(A.T) + 1e-6 * np.eye(p)
    K = A
    K_O = K[:po, :po]
    K_OH = K[:po, po:]
    K_HO = K[po:, :po]
    K_H = K[po:, po:]
    L = np.linalg.multi_dot((K_OH, np.linalg.inv(K_H), K_HO))
    K_O_tilde = K_O - L

    N = 5 * po
    print("Note that, with this method, the n_samples should be %d" % N)
    return [K_O] * T, [K_O_tilde] * T, [L] * T
