import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


@dataclass
class Gaussian:
    mean: np.ndarray
    cov: np.ndarray

@dataclass
class KFTracker:
    means: np.ndarray
    covs: np.ndarray
    stds: np.ndarray

@dataclass
class Observations:
    times: np.ndarray
    obs_ind: np.ndarray
    obs: np.ndarray
    names: list[str]

@dataclass
class TrueLinearSystem:
    """
    Ground-truth linear system used to generate state trajectories
    and synthetic sensor measurements.

    The state evolution is deterministic:
        x_{k+1} = A x_k

    Measurement noise is added to simulate sensors:
        y_k = H x_k + v_k
    """

    A: np.ndarray
    """True state transition matrix."""
    H: np.ndarray
    """True observation (measurement) matrix."""
    measurement_noise_cov: np.ndarray
    """Sensor noise covariance used only for data generation."""


@dataclass
class KalmanFilterModel:
    """
    Linear-Gaussian model assumed by the Kalman filter.

    Encodes the estimator's belief about system dynamics
    and uncertainty for prediction and update.
    """

    A: np.ndarray
    """Assumed state transition matrix."""
    H: np.ndarray
    """Assumed observation matrix."""
    Q: np.ndarray
    """Process noise covariance (model uncertainty)."""
    R: np.ndarray
    """Measurement noise covariance (sensor uncertainty)."""
    
    
def propagate_true_dynamics(
    x_k     :   np.ndarray,
    system   :   TrueLinearSystem
) -> np.ndarray:
    """
    Propagate the true system state forward one time step.

    Implements deterministic ground-truth dynamics:
        x_{k+1} = A x_k

    Parameters
    ----------
    x_k : np.ndarray
        True state at time k.
    system : TrueLinearSystem
        Ground-truth system dynamics.

    Returns
    -------
    np.ndarray
        True state at time k+1.
    """
    
    return system.A @ x_k

def observe(
    x_k     :   np.ndarray,
    system   :   TrueLinearSystem,
    rng     :   np.random.Generator
)->np.ndarray :
    """
    Generate a noisy measurement from the true system state.

    Measurement model:
        y_k = H x_k + v_k

    Parameters
    ----------
    x_k : np.ndarray
        True system state at measurement time.
    system : TrueLinearSystem
        Ground-truth observation model and sensor noise.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Noisy measurement vector.
    """
    
    v_k = rng.multivariate_normal(
        mean=np.zeros(system.measurement_noise_cov.shape[0]),
        cov=system.measurement_noise_cov
    )
    
    return system.H @ x_k + v_k

def generate_data(
    times   : np.ndarray,
    x0      : np.ndarray,
    obs_ind : np.ndarray,
    system   : TrueLinearSystem,
    rng:    np.random.Generator,
) -> tuple[Observations, Observations]:
    """
    Simulate ground-truth state trajectories and noisy measurements.

    The true system evolves deterministically at every time step:
        x_{k+1} = A x_k

    Measurements are generated only at indices `obs_ind` using:
        y_k = H x_k + measurement noise

    Parameters
    ----------
    times : np.ndarray
        Discrete simulation time grid.
    x0 : np.ndarray
        Initial true state.
    obs_ind : np.ndarray
        Time indices where measurements are available.
    system : TrueLinearSystem
        Ground-truth dynamics and sensor model.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Observations
        Ground-truth state trajectory.
    Observations
        Noisy measurements at observation times.
    """

    N = len(times)
    n = x0.shape[0]
    m = system.H.shape[0]

    # Allocate storage
    x_true = np.zeros((N, n))
    y_obs = np.zeros((len(obs_ind), m))

    # Initial condition
    x_true[0] = x0
    
    obs_counter = 0

    # Simulate dynamics
    for k in range(1, N):
        x_true[k] = propagate_true_dynamics(x_true[k-1], system)

        if obs_counter < len(obs_ind) and k == obs_ind[obs_counter]:
            y_obs[obs_counter] = observe(x_true[k], system, rng)
            obs_counter += 1

    truth_states = Observations(
        times=times,
        obs_ind=np.arange(N),
        obs=x_true,
        names=[f"x{i}" for i in range(n)],
    )

    measurements = Observations(
        times=times,
        obs_ind=obs_ind,
        obs=y_obs,
        names=[f"y{i}" for i in range(m)],
    )

    return truth_states, measurements


def linear_prediction_step(
    model   :   KalmanFilterModel,
    X_prior       :   Gaussian
    ) -> Gaussian:
    """
    Kalman filter prediction step.

    Propagates the state belief through the assumed dynamics:
        x_k = A x_{k-1} + w_k,  w_k ~ N(0, Q)

    Parameters
    ----------
    model : KalmanFilterModel
        Filter's assumed dynamics and process noise.
    X_prior : Gaussian
        State belief at previous time step.

    Returns
    -------
    Gaussian
        Predicted state belief.
    """
    
    prediction_mean = model.A @ X_prior.mean
    prediction_cov  = model.A @ X_prior.cov @ model.A.T + model.Q
    
    return Gaussian(prediction_mean, prediction_cov)

def linear_update_step(
    y_measurement   :   np.ndarray,
    model           :   KalmanFilterModel,
    X_predicted     :   Gaussian
) -> Gaussian:
    """
    Kalman filter measurement update step.

    Incorporates a new measurement:
        y_k = H x_k + v_k,  v_k ~ N(0, R)

    Parameters
    ----------
    y_measurement : np.ndarray
        Measurement vector.
    model : KalmanFilterModel
        Filter's assumed observation model and noise.
    X_predicted : Gaussian
        Predicted state belief before update.

    Returns
    -------
    Gaussian
        Updated (posterior) state belief.
    """
    
    U = X_predicted.cov @ model.H.T
    S = model.H @ X_predicted.cov @ model.H.T + model.R
    mu = model.H @ X_predicted.mean
    
    update_mean = X_predicted.mean + U @ np.linalg.solve(S, y_measurement - mu)
    update_cov = X_predicted.cov - U @ np.linalg.solve(S, U.T)
    
    return Gaussian(update_mean, update_cov)


def run_kalman_filter(
    y_Observations  :   Observations,
    model           :   KalmanFilterModel,
    X0              :   Gaussian
) -> KFTracker :
    """
    Run a linear Kalman filter over a full time horizon.

    Prediction is performed at every time step.
    Measurement updates occur only at observation indices.

    Parameters
    ----------
    y_Observations : Observations
        Measurement data and observation times.
    model : KalmanFilterModel
        Assumed system model for filtering.
    X0 : Gaussian
        Initial state belief.

    Returns
    -------
    KFTracker
        Filtered means, covariances, and standard deviations over time.
    """
    
    N = y_Observations.times.shape[0]
    n = X0.mean.shape[0]                # dimension of the state
    m = y_Observations.obs.shape[1]     # dimension of the measurement
    
    # storage
    means_store = np.zeros((N, n))
    covs_store = np.zeros((N, n, n))
    stds_store = np.zeros((N, n))
    
    # store Initial State
    means_store[0] = np.copy(X0.mean)
    covs_store[0] = np.copy(X0.cov)
    stds_store[0] = np.sqrt(np.diag(covs_store[0, :, :]))
    
    #Loop over all time steps
    X_previous = Gaussian(np.copy(X0.mean), np.copy(X0.cov))
    obs_counter = 0
    for k in range(1, N):
        # predict X_k state
        X_prediction = linear_prediction_step(model, X_previous)
        
        # We have an observation so an update must occur
        if obs_counter < len(y_Observations.obs_ind) and k == y_Observations.obs_ind[obs_counter]:
            y_k = y_Observations.obs[obs_counter]
            X_update = linear_update_step(y_k, model, X_prediction)
            obs_counter +=1
            
            X_previous = X_update
        
        # else we are just propagating the uncertanity forward
        else:
            X_previous = X_prediction
            
        # store Initial State
        means_store[k] = np.copy(X_previous.mean)
        covs_store[k] = np.copy(X_previous.cov)
        stds_store[k] = np.sqrt(np.diag(covs_store[k, :, :]))
        
    return KFTracker(means_store, covs_store, stds_store)


def plot_data_and_truth(
    truth: Observations,
    measurements: Observations,
    kf: Optional[KFTracker] = None,
    title: str | None = "State Estimation via Kalman Filtering",
    save_path: str | None = None,
):
    """
    Plot ground truth, measurements, and optional Kalman filter estimates.

    Visual conventions:
    - Ground truth: solid lines
    - Measurements: scatter points
    - Filter estimates: dashed lines
    - Uncertainty: (+-)2*std shaded regions

    Parameters
    ----------
    truth : Observations
        Ground-truth state trajectory defined at all time steps.
    measurements : Observations
        Measurement data defined only at observation indices.
    kf : KFTracker, optional
        Kalman filter state estimates.
    save_path : str, optional
        Path to save the figure.
    title : str, optional
        Figure title.
    """

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Consistent color cycle per state
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # -------------------------
    # Ground truth
    # -------------------------
    for i in range(truth.obs.shape[1]):
        ax.plot(
            truth.times,
            truth.obs[:, i],
            color=colors[i],
            linewidth=2.5,
            label=f"True {truth.names[i]}"
        )

    # -------------------------
    # Measurements
    # -------------------------
    for i in range(measurements.obs.shape[1]):
        ax.scatter(
            truth.times[measurements.obs_ind],
            measurements.obs[:, i],
            color=colors[i],
            alpha=0.45,
            s=28,
            marker='o',
            edgecolor='none',
            label=f"Measured {measurements.names[i]}"
        )

    # -------------------------
    # Kalman filter estimates
    # -------------------------
    if kf is not None:
        for i in range(kf.means.shape[1]):
            ax.plot(
                truth.times,
                kf.means[:, i],
                linestyle='--',
                linewidth=2.0,
                color=colors[i],
                label=f"KF estimate {truth.names[i]}"
            )

            ax.fill_between(
                truth.times,
                kf.means[:, i] - 2 * kf.stds[:, i],
                kf.means[:, i] + 2 * kf.stds[:, i],
                color=colors[i],
                alpha=0.18,
                linewidth=0
            )

    # -------------------------
    # Styling
    # -------------------------
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("State value", fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)

    ax.grid(True, which="both", linestyle="--", alpha=0.25)
    ax.legend(
        loc="best",
        frameon=False,
        fontsize=10,
        ncol=2
    )

    fig.tight_layout()

    # -------------------------
    # Save or return
    # -------------------------
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
