import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from itertools import product

from filters.kalman_filter import *
from filters.extended_kalman_filter import *


def gauss_hermite_oned(num_pts : int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D Gauss-Hermite quadrature for expectation under a standard normal.

    Computes nodes and weights such that, for U ~ N(0,1),

        E[f(U)] ≈ sum_i w_i * f(u_i)

    The nodes u_i and weights w_i are obtained via the Golub–Welsch
    algorithm applied to the Jacobi matrix associated with
    probabilists' Hermite polynomials (orthogonal w.r.t. N(0,1)).

    Returns
    -------
    pts : (num_pts,) ndarray
        Quadrature nodes (sigma points) in 1D.
    wts : (num_pts,) ndarray
        Corresponding weights, summing to 1.
    """
    
    # Jacobi matrix for probabilists' Hermite polynomials.
    # Its eigenvalues give the quadrature nodes, and the
    # squared first components of eigenvectors give the weights.
    A = np.zeros((num_pts, num_pts))
    
    # Fill symmetric tridiagonal Jacobi matrix
    for ii in range(num_pts):
        #print("ii ", ii, ii==0, ii==(order-1))
        row = ii+1
        if ii == 0:
            A[ii, ii+1] = np.sqrt(row)
            A[ii+1, ii] = np.sqrt(row)
        elif ii == (num_pts-1):
            A[ii-1, ii] = np.sqrt(ii)
        else:
            A[ii, ii+1] = np.sqrt(row)
            A[ii+1, ii] = np.sqrt(row)
    
    # Eigen-decomposition: nodes = eigenvalues
    # weights = square of first component of eigenvectors
    pts, evec = np.linalg.eig(A)
    devec = np.dot(evec.T, evec)
    wts = evec[0,:]**2

    return pts, wts

def gauss_hermite_nd(dim : int = 2, num_pts: int = 3):
    """
    Tensor-product Gauss-Hermite quadrature for a multivariate
    standard normal distribution N(0, I_d).

    Approximates expectations of the form

        E[f(U)],   U ~ N(0, I_d)

    by forming the Cartesian product of 1D Gauss-Hermite rules
    across each dimension.

    Returns
    -------
    points : (num_pts**dim, dim) ndarray
        Quadrature points in R^dim.
    weights : (num_pts**dim,) ndarray
        Corresponding weights, summing to 1.
    """
    
    # 1D Gauss–Hermite nodes and weights for N(0,1)
    pts_1d, wts_1d = gauss_hermite_oned(num_pts) 
    
    # Cartesian product of indices: one index per dimension
    # Each index tuple selects one 1D node per coordinate
    index_sets = list(product(range(num_pts), repeat=dim))

    
    n_total = num_pts ** dim
    points = np.zeros((n_total, dim))
    weights = np.zeros(n_total)
    
    # Build multidimensional points and tensor-product weights
    for i, idx in enumerate(index_sets):
        points[i, :] = pts_1d[list(idx)]
        weights[i] = np.prod(wts_1d[list(idx)])
    
    return points, weights

def gaussian_affine_transform(
    points  :   np.ndarray,
    mean    :   np.ndarray,
    cov     :   np.ndarray,
    method  :   str = "chol"
) -> np.ndarray :
    """
    Map quadrature points from a standard normal distribution
    to a general Gaussian N(mean, cov).

    Implements the affine transformation

        X = mean + L * U,   where U ~ N(0, I)
        and L L^T = cov

    Parameters
    ----------
    points : (N, d) ndarray
        Quadrature points for N(0, I_d).
    mean : (d,) ndarray
        Target Gaussian mean.
    cov : (d, d) ndarray
        Target Gaussian covariance.
    method : {'chol', 'svd'}
        Matrix factorization used to compute sqrt(cov).

    Returns
    -------
    (N, d) ndarray
        Quadrature points distributed according to N(mean, cov).
    """
    
    if method == "chol":
        L = np.linalg.cholesky(cov)
    elif method == "svd":
        U, S, _ = np.linalg.svd(cov)
        L = U @ np.diag(np.sqrt(S))
    else:
        raise ValueError("method must be 'chol' or 'svd'")
    
    return mean + points @ L.T


def gaussian_expectation(
    func        :   Callable[[np.ndarray], np.ndarray],
    gaussian    :   Gaussian,
    num_pts     :   int = 3
):
    """
    Approximate E[f(X)] where X ~ N(mean, cov)
    """
    
    # Dimensionality of the random variable
    dim = gaussian.mean.shape[0]
    
    # 1. Quadrature points and weights for N(0, I_d)
    u_pts, wts = gauss_hermite_nd(dim, num_pts)
    
    # 2. Map points to N(mean, cov)
    x_pts = gaussian_affine_transform(u_pts, 
                                      gaussian.mean, 
                                      gaussian.cov, 
                                      method="chol")
    
    # 3. Weighted sum to approximate the expectation
    val = 0.0
    for i in range(len(wts)):
        val += wts[i] * func(x_pts[i])
    
    return val


@dataclass
class GaussHermiteKalmanFilterModel:
    """
    Nonlinear Gaussian model assumed by the GHKF.

    Expectations are computed using Gauss-Hermite quadrature
    instead of linearization.
    """

    Phi: Callable[[np.ndarray], np.ndarray]
    """State transition function x_{k+1} = Phi(x_k)"""

    h: Callable[[np.ndarray], np.ndarray]
    """Measurement function y_k = h(x_k)"""

    Q: np.ndarray
    """Process noise covariance"""

    R: np.ndarray
    """Measurement noise covariance"""

    num_pts: int = 3
    """Number of Gauss-Hermite points per dimension"""

def ghkf_prediction_step(
    model   :   GaussHermiteKalmanFilterModel,
    X_prior :   Gaussian
) -> Gaussian:
    """
    Gauss-Hermite Kalman filter prediction step.

    Computes:
        m_pred = E[Phi(X)]
        C_pred = Cov[Phi(X)] + Q
    using Gauss-Hermite quadrature.
    """
    
    
    # 1. Mean prediction
    m_pred = gaussian_expectation(
        func=model.Phi, 
        gaussian=X_prior,
        num_pts=model.num_pts
    )

    # 2. Covariance prediction
    def centered_outer(x):
        dx = model.Phi(x) - m_pred
        return np.outer(dx, dx)
    
    C_pred = gaussian_expectation(
        func=centered_outer,
        gaussian=X_prior,
        num_pts=model.num_pts
    ) + model.Q

    return Gaussian(m_pred, C_pred)


def ghkf_update_step(
    y_measurement   :   np.ndarray,
    model           :   GaussHermiteKalmanFilterModel,
    X_predicted     :   Gaussian
) ->Gaussian:
    """
    Gauss-Hermite Kalman Filter update step.

    Computes:
        mu = E[h(X_k)]
        U  = Cov[X_k, h(X_k)]
        S  = Cov[h(X_k)] + R
        m_k = m_k^- + U S^{-1} (y_k - mu)
        C_k = C_k^- - U S^{-1} U^T
    """
    
    # Predicted statistics
    m_pred = X_predicted.mean
    C_pred = X_predicted.cov
    
    # 1. Predicted measurement mean
    mu = gaussian_expectation(
        func=model.h,
        gaussian=X_predicted,
        num_pts=model.num_pts
    )
    
    # 2. Cross covariance U
    def cross_outer(x):
        dx = x - m_pred
        dy = model.h(x) - mu
        return np.outer(dx, dy)

    U = gaussian_expectation(
        func=cross_outer,
        gaussian=X_predicted,
        num_pts=model.num_pts
    )
    
    # 3. Innovation covariance S
    def innovation_outer(x):
        dy = model.h(x) - mu
        return np.outer(dy, dy)

    S = gaussian_expectation(
        func=innovation_outer,
        gaussian=X_predicted,
        num_pts=model.num_pts
    ) + model.R
    
    # 4. Measurement innovation
    innovation = y_measurement - mu

    m_post = m_pred + U @ np.linalg.solve(S, innovation)
    C_post = C_pred - U @ np.linalg.solve(S, U.T)

    return Gaussian(m_post, C_post)

def run_gauss_hermite_kalman_filter(
    y_Observations : Observations,
    model          : GaussHermiteKalmanFilterModel,
    X0             : Gaussian
) -> KFTracker:
    """
    Run a Gauss-Hermite Kalman Filter over a full time horizon.
    """

    # Dimensions
    N = y_Observations.times.shape[0]
    n = X0.mean.shape[0]

    # Allocate storage
    means_store = np.zeros((N, n))
    covs_store  = np.zeros((N, n, n))
    stds_store  = np.zeros((N, n))

    # Initialize with prior
    means_store[0] = X0.mean
    covs_store[0]  = X0.cov
    stds_store[0]  = np.sqrt(np.diag(X0.cov))

    # Current filtering state
    X_curr = Gaussian(X0.mean.copy(), X0.cov.copy())
    obs_counter = 0

    # Main GHKF loop
    for k in range(1, N):
        # Prediction X_k state
        X_pred = ghkf_prediction_step(model, X_curr)

        # Update if measurement available
        if obs_counter < len(y_Observations.obs_ind) and k == y_Observations.obs_ind[obs_counter]:
            y_k = y_Observations.obs[obs_counter]
            
            # update
            X_curr = ghkf_update_step(y_k, model, X_pred)
            obs_counter += 1
            
        # else we are just propagating the uncertainty forward
        else:
            X_curr = X_pred

        means_store[k] =  np.copy(X_curr.mean)
        covs_store[k]  = np.copy(X_curr.cov)
        stds_store[k]  = np.sqrt(np.diag(covs_store[k, :, :]))

    return KFTracker(means_store, covs_store, stds_store)