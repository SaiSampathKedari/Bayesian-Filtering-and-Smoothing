import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import matplotlib.pyplot as plt

from filters.kalman_filter import *

@dataclass
class ParticleSet:
    """
    Empirical distribution on trajectory space.
        P(X_n | Y_n) = \sum w_i * \delta_{X_i} 
        
    Each particle represents an entire state trajectory.
    This object approximates the full smoothing distribution.
    """
    
    particles   :   np.ndarray
    """
    Array of shape (N, T, d).

    particles[i, k] = x_k^{(i)}
    is the state at time k of the i-th particle trajectory.
    """
    
    weights     :   np.ndarray
    """
    Array of shape (N,).

    Importance weights associated with each trajectory.
    One weight per trajectory.
    """
    
    def normalize(self) -> None:
        """
        Normalize importance weights in-place.

        If weights are degenerate or invalid, reset to uniform.
        """
        s = np.sum(self.weights)
        if (not np.isfinite(s)) or (s <= 0.0):
            self.weights = np.full(self.weights.shape[0], 1.0/self.weights.shape[0])
        else:
            self.weights /= s
    
    def ess(self) -> float:
        self.normalize()
        return 1.0/np.sum(self.weights ** 2)
    
    def marginal_particles(self) -> np.ndarray:
        """
        Extract marginal filtering particles.
        Returns:
            x_T^{(i)} for all trajectories.

        Shape:
            (N, d)
        """
        return self.particles[:, -1, :]        


@dataclass 
class ParticleFilterModel:
    """"
    Nonlinear state-space model assumed by a Particle Filter
    implemented as Sequential Importance Sampling (SIS) with resampling.
    
    The filter operates on trajectory space:
        X_n = (x_0, ..., x_n)
    
    Structural parity with KF / EKF / UKF:
      - Phi : deterministic state transition map
      - h   : deterministic measurement map
    
    SIS-specific components:
        - proposal_sampler  : samples x_n ~ pi(x_n | X_{n-1})
        - proposal_logpdf   : evaluate log pi(x_n | X_{n-1})
        - transition_logpdf : evaluate log p(x_n | X_{n-1})
        - measurement_logpdf: evaluate log p(y_n | x_n)
    """
    
    Phi: Callable[[np.ndarray], np.ndarray]
    """State trasition function " X_{k+1} = Phi(X_k) (no noise injected here)."""
    
    h:  Callable[[np.ndarray], np.ndarray]
    """Measurement function h(x_k) (no noise injected here)."""
    
    proposal_sampler: Callable[
        [Optional[np.ndarray], np.random.Generator], np.ndarray]
    
    proposal_logpdf: Callable[[np.ndarray, Optional[np.ndarray]], float]
    """
    Evaluate proposal log-density.
        log pi(x_0)
        log pi(x_n | X_{n-1})
        
    Inputs:
      - x_n       : (d,)
      - traj_prev : (t, d) or None
    """
   
    transition_logpdf: Callable[[np.ndarray, np.ndarray], float]
    """
    Transition density:
    
        log p(x_k | x_{k-1})
    """
   
    measurement_logpdf: Callable[[np.ndarray, np.ndarray], float]
    """
    Measurement likelihood:
        log p(y_k | x_k)
    """ 
    
    ess_threshold_ratio: float = 0.5
    """
    Trigger resampling when:
        ESS < ess_threshold_ratio * N
    """
    
    state_dim: int 
    
    
def pf_initialization(
    model           : ParticleFilterModel,
    num_particles   : int,
    rng             : np.random.Generator) -> ParticleSet:
    
    """
    Initialize the particle system.

    Samples N independent initial states:
        x_0^{(i)} ~ π(x_0)

    Each particle is a trajectory of length 1.
    All weights are initialized uniformly.
    """
    
    d = model.state_dim # dimension of the state
    particles = np.zeros((num_particles, 1, d))
    weights = np.full(num_particles, 1.0/num_particles)
    
    for i in range(num_particles):
        # proposal_sampler must accept traj_prev=None
        x0 = model.proposal_sampler(None, rng)
        particles[i, 0, :] =  x0
        
    return ParticleSet( 
            particles= particles,
            weights=weights)

def pf_prediction_step(
    model           :   ParticleFilterModel,
    particles_prev  :   ParticleSet,
    rng             :   np.random.Generator
) -> ParticleSet:
    """
    Prediction step of the Particle Filter.

    For each particle trajectory:
        x_n^{(i)} ~ pi(x_n | X_{n-1}^{(i)})

    The new state is appended to the trajectory.
    Importance weights are unchanged.
    """
    
    N, T, d = particles_prev.particles.shape
    
    X_prev = particles_prev.particles
    w_prev = particles_prev.weights
    
    particles_new = np.zeros((N, T+1, d))
    
    for i in range(N):
        traj_prev = X_prev[i]       # shape (T, d)
        x_new = model.proposal_sampler(traj_prev, rng)
        
        particles_new[i, :T, :] = np.copy(traj_prev)
        particles_new[i, T, :] = x_new
        
    return ParticleSet(
        particles=particles_new, 
        weights=w_prev.copy()
    )

def pf_update_step(
    y_measurement   :   np.ndarray,
    model           :   ParticleFilterModel,
    particles  :   ParticleSet
) -> ParticleSet:
    """
    Weight update step of the Particle Filter (SIS).
    Updates importance weights
    The particle trajectories themselves are unchanged.
    """
    X = particles.particles           # (N, T, d)
    w_pred = particles.weights             # (N,)
    
    N, T, d = X.shape
    assert T >= 2, "Update requires at least two time steps"
    
    log_w = np.log(w_pred + 1e-300)
    
    for i in range(N):
        traj    =   X[i]        # (T, d)
        x_prev  =   traj[-2]         # (d, )
        x_curr  =   traj[-1]         # (d, )
        
        # Log likelihood term
        log_like = model.measurement_logpdf(y_measurement, x_curr)
        
        # Log transition density
        log_trans = model.transition_logpdf(x_curr, x_prev)
        
        # Log proposal density
        log_prop = model.proposal_logpdf(x_curr, traj[:-1])
        
        # Incremental log-weight
        log_w[i] += log_like + log_trans - log_prop
        
    # Stabilize and normalize
    log_w -= np.max(log_w)
    w_new = np.exp(log_w)
    
    updated_particles = ParticleSet(
        particles = X,
        weights   = w_new
    )
    updated_particles.normalize()
    
    return updated_particles