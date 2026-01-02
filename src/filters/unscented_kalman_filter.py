import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from dataclasses import dataclass

from filters.kalman_filter import *
from filters.extended_kalman_filter import *
from filters.gauss_hermite_kalman_filter import *



@dataclass
class UnscentedKalmanFilterModel:
    """
    Nonlinear Gaussian model assumed by the UKF.

    Expectations are approximated using the Unscented Transform.
    """

    Phi: Callable[[np.ndarray], np.ndarray]
    """State transition function x_{k+1} = Phi(x_k)"""

    h: Callable[[np.ndarray], np.ndarray]
    """Measurement function y_k = h(x_k)"""

    Q: np.ndarray
    """Process noise covariance"""

    R: np.ndarray
    """Measurement noise covariance"""

    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

def unscented_sigma_points(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
    method: str = "chol"
):
    """
    Generate sigma points and weights for X ~ N(mean, cov).
    """
    dim = mean.shape[0]

    lam = alpha**2 * (dim + kappa) - dim
    gamma = np.sqrt(dim + lam)

    # Matrix square root
    if method == "chol":
        L = np.linalg.cholesky(cov)
    elif method == "svd":
        U, S, _ = np.linalg.svd(cov)
        L = U @ np.diag(np.sqrt(S))
    else:
        raise ValueError("method must be 'chol' or 'svd'")

    # Sigma points
    points = np.zeros((2 * dim + 1, dim))
    points[0] = mean

    for i in range(dim):
        points[i + 1]       = mean + gamma * L[:, i]
        points[i + 1 + dim] = mean - gamma * L[:, i]

    # Weights
    Wm = np.full(2 * dim + 1, 1.0 / (2 * (dim + lam)))
    Wc = np.full(2 * dim + 1, 1.0 / (2 * (dim + lam)))

    Wm[0] = lam / (dim + lam)
    Wc[0] = lam / (dim + lam) + (1 - alpha**2 + beta)

    return points, Wm, Wc

def ukf_prediction_step(
    model: UnscentedKalmanFilterModel,
    X_prior: Gaussian
) -> Gaussian:
    """
    UKF prediction step.

    Computes:
        m_k^- = E[Phi(X)]
        C_k^- = Cov[Phi(X)] + Q
    """
    pts, Wm, Wc = unscented_sigma_points(
        X_prior.mean,
        X_prior.cov,
        model.alpha,
        model.beta,
        model.kappa
    )

    # Propagate sigma points through dynamics
    Y = np.array([model.Phi(p) for p in pts])

    # Predicted mean
    m_pred = np.sum(Wm[:, None] * Y, axis=0)

    # Predicted covariance
    C_pred = np.zeros((m_pred.shape[0], m_pred.shape[0]))
    for i in range(len(Wc)):
        dy = Y[i] - m_pred
        C_pred += Wc[i] * np.outer(dy, dy)

    C_pred += model.Q

    return Gaussian(m_pred, C_pred)

def ukf_update_step(
    y_measurement: np.ndarray,
    model: UnscentedKalmanFilterModel,
    X_predicted: Gaussian
) -> Gaussian:
    """
    UKF measurement update step.

    Computes:
        mu = E[h(X)]
        U  = Cov[X, h(X)]
        S  = Cov[h(X)] + R
        m_k = m_k^- + U S^{-1} (y_k - mu)
        C_k = C_k^- - U S^{-1} U^T
    """
    m_pred = X_predicted.mean
    C_pred = X_predicted.cov

    pts, Wm, Wc = unscented_sigma_points(
        m_pred,
        C_pred,
        model.alpha,
        model.beta,
        model.kappa
    )

    # Propagate sigma points through measurement model
    Z = np.array([model.h(p) for p in pts])

    # Measurement mean
    mu = np.sum(Wm[:, None] * Z, axis=0)

    # Innovation covariance S
    S = np.zeros((mu.shape[0], mu.shape[0]))
    for i in range(len(Wc)):
        dz = Z[i] - mu
        S += Wc[i] * np.outer(dz, dz)
    S += model.R

    # Cross covariance U
    U = np.zeros((m_pred.shape[0], mu.shape[0]))
    for i in range(len(Wc)):
        dx = pts[i] - m_pred
        dz = Z[i] - mu
        U += Wc[i] * np.outer(dx, dz)

    # Kalman update
    innovation = y_measurement - mu
    m_post = m_pred + U @ np.linalg.solve(S, innovation)
    C_post = C_pred - U @ np.linalg.solve(S, U.T)

    return Gaussian(m_post, C_post)


def run_unscented_kalman_filter(
    y_Observations: Observations,
    model: UnscentedKalmanFilterModel,
    X0: Gaussian
) -> KFTracker:
    """
    Run an Unscented Kalman Filter over a full time horizon.
    """
    
    # Dimensions
    N = y_Observations.times.shape[0]
    n = X0.mean.shape[0]

    # Allocate storage
    means_store = np.zeros((N, n))
    covs_store  = np.zeros((N, n, n))
    stds_store  = np.zeros((N, n))

    # Initialize with prior
    means_store[0] = np.copy(X0.mean)
    covs_store[0] = np.copy(X0.cov)
    stds_store[0] = np.sqrt(np.diag(covs_store[0, :, :]))

    # Current filtering state
    X_curr = Gaussian(X0.mean.copy(), X0.cov.copy())
    obs_counter = 0

    # Main UKF loop
    for k in range(1, N):
        # Prediction
        X_pred = ukf_prediction_step(model, X_curr)

        # Update if measurement exists
        if obs_counter < len(y_Observations.obs_ind) and k == y_Observations.obs_ind[obs_counter]:
            y_k = y_Observations.obs[obs_counter]
            X_curr = ukf_update_step(y_k, model, X_pred)
            obs_counter += 1
        else:
            X_curr = X_pred

        means_store[k] =  np.copy(X_curr.mean)
        covs_store[k]  = np.copy(X_curr.cov)
        stds_store[k]  = np.sqrt(np.diag(covs_store[k, :, :]))

    return KFTracker(means_store, covs_store, stds_store)