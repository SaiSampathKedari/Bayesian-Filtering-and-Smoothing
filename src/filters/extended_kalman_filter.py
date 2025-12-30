import numpy as np
from dataclasses import dataclass
from typing import Callable
from filters.kalman_filter import *



@dataclass
class ExtendedKalmanFilterModel:
    """
    Nonlinear Gaussian model assumed by the EKF. 
    
    Encodes Nonlinear Dynamics and measurement functions,
    along with their Jacobians.
    """
    
    Phi: Callable[[np.ndarray], np.ndarray]
    """State transition function: x_{k+1} = Phi(x_k)"""
    
    h: Callable[[np.ndarray], np.ndarray]
    """Measurement function h(x)"""
    
    jacobian_Phi: Callable[[np.ndarray], np.ndarray]
    """Jacobian of Phi evaluated at x"""

    jacobian_h: Callable[[np.ndarray], np.ndarray]
    """Jacobian of h evaluated at x"""
    
    Q: np.ndarray
    """Process noise co-variance matrix"""
    
    R: np.ndarray
    """Measurement noise co-varinace matrix"""

def ekf_prediction_step(
    model   :   ExtendedKalmanFilterModel,
    X_prior :   Gaussian
) -> Gaussian:
    """
    Extended Kalman filter prediction step
    
    Propagates the state belief through the Taylor
    linearization of the assumed non-linear dynamics:
        X_k = Phi(X_{k-1}) + eta_k, eta_k ~ N(0, Q)
    
    Paramters
    ----------
    model   :   ExtendedkalmanFilterModel
        Filter's assumed nonlinear dynamics, its jacobian and
        Process noise
    X_prior :   Gaussian
        State belief at previous time step.
        
    Returns
    ---------
    Gaussian
        Predicted state belief.
    """
    
    # Prior statistics
    m_prev = X_prior.mean
    C_prev = X_prior.cov
    
    # Mean propagation
    m_pred = model.Phi(m_prev)
    
    # Jacobian of dynamics evaluated at prior mean
    A_k = model.jacobian_Phi(m_prev)
    
    # Covariance propagation
    C_pred = A_k @ C_prev @ A_k.T + model.Q
    
    return Gaussian(m_pred, C_pred)


def ekf_update_step(
    y_measurement   :   np.ndarray,
    model           :   ExtendedKalmanFilterModel,
    X_predicted     :   Gaussian
)-> Gaussian:
    """
    Extended Kalman filter measurement update step.

    Incorporates a new measurement:
        y_k = h(x_k) + v_k,  v_k ~ N(0, R)

    Parameters
    ----------
    y_measurement : np.ndarray
        Measurement vector.
    model : ExtendedKalmanFilterModel
        Filter's assumed observation model and noise.
    X_predicted : Gaussian
        Predicted state belief before update.

    Returns
    -------
    Gaussian
        Updated (posterior) state belief.
    """
    
    # Predicted statistics
    m_pred = X_predicted.mean
    C_pred = X_predicted.cov
    
    # Jacobian of measurement model evaluated at predicted mean
    H_k = model.jacobian_h(m_pred)
    
    # Predicted measurement mean
    mu = model.h(m_pred)
    
    # Innovation statistics
    U = C_pred @ H_k.T
    S = H_k @ C_pred @ H_k.T + model.R
    
    # Measurement innovation
    innovation = y_measurement - mu
    
    # Posterior update
    m_post = m_pred + U @ np.linalg.solve(S, innovation)
    C_post = C_pred - U @ np.linalg.solve(S, U.T)
    
    return Gaussian(m_post, C_post)



def run_extended_kalman_filter(
    y_Observations  :   Observations,
    model           :   ExtendedKalmanFilterModel,
    X0              :   Gaussian
) -> KFTracker:

    # Dimensions
    N = y_Observations.times.shape[0]   # number of time steps
    n = X0.mean.shape[0]                # state dimension
    # m = y_Observations.obs.shape[1]     # dimension of the measurement
    
    
    # Allocate storage
    means_store = np.zeros((N, n))
    covs_store  = np.zeros((N, n, n))
    stds_store  = np.zeros((N, n))
    
    # Initialize with prior
    means_store[0] = np.copy(X0.mean)
    covs_store[0] = np.copy(X0.cov)
    stds_store[0] = np.sqrt(np.diag(covs_store[0, :, :]))
    
    # Current filtering state
    X_curr = Gaussian(np.copy(X0.mean), np.copy(X0.cov))
    obs_counter = 0
    
    # Main EKF loop
    for k in range(1, N):
        # predition X_k state
        X_pred = ekf_prediction_step(model, X_curr)
        
        # We have an observation so an update must occur
        if obs_counter < len(y_Observations.obs_ind)  and k == y_Observations.obs_ind[obs_counter]:
            y_k = y_Observations.obs[obs_counter]
            
            # update 
            X_update = ekf_update_step(y_k, model, X_pred)
            
            obs_counter += 1
            
            X_curr = X_update
        
        # else we are just propagating the uncertainty forward
        else:
            X_curr = X_pred
        
        # Store results
        means_store[k] = np.copy(X_curr.mean)
        covs_store[k]  = np.copy(X_curr.cov)
        stds_store[k]  = np.sqrt(np.diag(covs_store[k, :, :]))

    return KFTracker(means_store, covs_store, stds_store)