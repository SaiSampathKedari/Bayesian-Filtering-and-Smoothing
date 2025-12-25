import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class RegressionDataSet:
    """
    Dataset for 1D linear regression.

    Contains observed data (t, y) and ground-truth information
    used only for visualization or validation.
    """
    
    t       :   np.ndarray          # (n, ) independent variable
    y       :   np.ndarray          # (n, ) observations
    t_plot  :   np.ndarray          # (m, ) plotting grid
    y_true  :   np.ndarray          # (m, ) true model values (simulation only)
    

def generate_RegressionDataSet(
    num_data    :   int,
    max_time    :   float = 1.0,
    theta_1     :   float = 1.0,
    theta_2     :   float = 0.8,
    noise_std   :   float = 1e-1) -> RegressionDataSet:
    
    """
    Generate synthetic linear regression data.

    Data are generated according to
        y_k = theta_1 + theta_2 * t_k + noise,
    where noise is zero-mean Gaussian with standard deviation `noise_std`.

    Returns a dataset containing the noisy observations and the true
    underlying model evaluated on a dense time grid for plotting.
    """
    
    t = np.sort(np.random.rand(num_data) * max_time)
    y = theta_1 + theta_2 * t + noise_std*np.random.randn(num_data)
    
    t_plot = np.linspace(0.0, max_time, 1000)
    y_true = theta_1 + theta_2 * t_plot
    
    return RegressionDataSet(t=t, y=y, t_plot=t_plot, y_true=y_true)


def batch_linear_gaussian_update(
    y       : np.ndarray,
    H       : np.ndarray,
    mu_0    : np.ndarray,
    C_0     : np.ndarray,
    noise_std: float,  
):
    """
    Batch posterior for linear-Gaussian model using innovation (Kalman) form.

    Model
    -----
    y = H theta + eta
    eta ~ N(0, noise_std^2 I_n)
    theta ~ N(mu_0, C_0)
    """
    
    n = y.shape[0]
    
    # innovation Co-variance
    S = H @ C_0 @ H.T + noise_std**2 * np.eye(n)
    
    # Kalman Gain
    K = C_0 @ H.T @ np.linalg.solve(S, np.eye(n))
    
    # Posterior Mean
    mu_post = mu_0 + K @ ( y - H @ mu_0)
    
    # Posterior CoVariance
    C_post = C_0 - K @ H @ C_0
    
    return mu_post, C_post

def get_predictive_mean_covariance(
    H           :   np.ndarray,
    mu_theta    :   np.ndarray,
    C_theta     :   np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    m = H.shape[0]
    
    predictive_mean = H @ mu_theta              # shape (m,)
    predictive_covariance = H @ C_theta @ H.T   # shape (m,m)
    
    return predictive_mean, predictive_covariance