import numpy as np
from typing import Tuple, Callable
from filters.kalman_filter import *

def pendulum_dynamics(
    x   :   np.ndarray,
    dt  :   float,
    g   :   float = 9.81
) -> np.ndarray:
    """
    Discrete-time nonlinear pendulum dynamics.

    State:
        x = [angle, angular_velocity]

    Dynamics:
        x1_{k+1} = x1_k + x2_k * dt
        x2_{k+1} = x2_k - g * sin(x1_k) * dt
    """
    
    x1, x2 = x
    
    x1_next = x1 + x2 * dt
    x2_next = x2 - g * np.sin(x1) * dt
    
    return np.array([x1_next, x2_next])



def pendulum_jacobian_dynamics(
    x   :   np.ndarray,
    dt  :   float,
    g   :   float=9.81
) -> np.ndarray:
    """
    Jacobian of pendulum dynamics with respect to the state.

    A_k = d Phi(x) / d x evaluated at x
    """
    
    x1 = x[0]
    
    A_k = np.array([
        [1                      ,   dt], 
        [-g * np.cos(x1) * dt   ,   1 ]])
    
    return A_k



def pendulum_measurement(
    x   :   np.ndarray
) -> np.ndarray:
    """
    Measurement model.

    Only the angle is observed through a nonlinear sensor:
        y = sin(x1)
    """
    
    return np.array([np.sin(x[0])])



def pendulum_jacobian_measurement(
    x   :   np.ndarray
) -> np.ndarray:
    """
    Jacobian of measurement model with respect to the state.

    H_k = d h(x) / d x evaluated at x
    """
    x1 = x[0]
    
    return np.array([[np.cos(x1), 0.0]])



def pendulum_process_noise_cov(
    dt  :   float,
    qc  :   float
) -> np.ndarray:
    """
    Discrete-time process noise covariance for the pendulum.

    Derived from continuous-time white acceleration noise:

        Q = [[ qc * dt^3 / 3,  qc * dt^2 / 2 ],
             [ qc * dt^2 / 2,  qc * dt       ]]
    """
    
    return np.array([
        [qc * dt**3 / 3.0, qc * dt**2 / 2.0],
        [qc * dt**2 / 2.0, qc * dt]
    ])
    

@dataclass
class TrueNonlinearSystem:
    """
    Ground-truth nonlinear system used ONLY for data generation.

    Dynamics are deterministic.
    Measurement noise is added.
    """

    Phi: Callable[[np.ndarray], np.ndarray]
    """True nonlinear state transition: x_{k+1} = Phi(x_k)"""

    h: Callable[[np.ndarray], np.ndarray]
    """True nonlinear measurement model: y_k = h(x_k)"""

    measurement_noise_cov: np.ndarray
    """Sensor noise covariance (R) used ONLY for data generation"""

def make_true_pendulum_system(
    dt: float,
    R: np.ndarray,
    g: float = 9.81
) -> TrueNonlinearSystem:
    """
    Ground-truth pendulum system for data generation.
    """

    return TrueNonlinearSystem(
        Phi=lambda x: pendulum_dynamics(x, dt, g),
        h=pendulum_measurement,
        measurement_noise_cov=R
    )

    
def propagate_true_nonlinear_dynamics(
    x_k: np.ndarray,
    system: TrueNonlinearSystem
) -> np.ndarray:
    """
    Deterministic nonlinear ground-truth propagation.
    """

    return system.Phi(x_k)

def observe_nonlinear(
    x_k: np.ndarray,
    system: TrueNonlinearSystem,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate noisy nonlinear measurement.
    """

    v_k = rng.multivariate_normal(
        mean=np.zeros(system.measurement_noise_cov.shape[0]),
        cov=system.measurement_noise_cov
    )

    return system.h(x_k) + v_k

def generate_nonlinear_data(
    times: np.ndarray,
    x0: np.ndarray,
    obs_ind: np.ndarray,
    system: TrueNonlinearSystem,
    rng: np.random.Generator
) -> tuple[Observations, Observations]:
    """
    Generate ground-truth nonlinear states and noisy measurements.
    """

    N = len(times)
    n = x0.shape[0]
    m = system.measurement_noise_cov.shape[0]

    # Allocate storage
    x_true = np.zeros((N, n))
    y_obs = np.zeros((len(obs_ind), m))

    # Initial condition
    x_true[0] = x0
    
    obs_counter = 0

    # Simulate dynamics
    for k in range(1, N):
        x_true[k] = propagate_true_nonlinear_dynamics(x_true[k-1], system)

        if obs_counter < len(obs_ind) and k == obs_ind[obs_counter]:
            y_obs[obs_counter] = observe_nonlinear(x_true[k], system, rng)
            obs_counter += 1

    truth_states = Observations(
        times=times,
        obs_ind=np.arange(N),
        obs=x_true,
        names=["angle", "angular_velocity"]
    )

    measurements = Observations(
        times=times,
        obs_ind=obs_ind,
        obs=y_obs,
        names=["sin(angle)"]
    )

    return truth_states, measurements
