import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

import matplotlib.pyplot as plt

from filters.kalman_filter import *


@dataclass
class ParticleSet:
    """
    Container representing an empirical distribution over state trajectories.

    Each particle corresponds to one complete state trajectory through time,
    and each weight represents the importance of that trajectory given all
    observations processed so far.

    This object stores the full particle approximation of the smoothing
    distribution and is the particle-based analogue of the Gaussian belief
    used in Kalman filtering.
    """
    
    particles   : np.ndarray
    """
    Array of shape (N, T, d).

    particles[i] contains the full trajectory of the i-th particle.
    particles[i, k] is the state at time index k for that trajectory.

    N is the number of particles.
    T is the number of time steps currently stored.
    d is the dimension of the state space.
    """
    
    weights     : np.ndarray
    """
    Array of shape (N,).

    Importance weights associated with each particle trajectory.
    Each weight reflects how well that trajectory explains the observed data.

    The weights are assumed to be non-negative and are normalized when needed.
    """
    
    def normalize(self):
        """
        Normalize the particle weights in place.

        If the total weight is invalid or numerically degenerate,
        the weights are reset to a uniform distribution.

        This method is called automatically by functions that rely
        on properly normalized weights.
        """
        
        s = np.sum(self.weights)
        
        if (not np.isfinite(s)) or (s <= 0.0):
            self.weights = np.full(self.weights.shape[0], 1.0/self.weights.shape[0])
        else:
            self.weights = self.weights/s
    
    def ess(self) -> float:
        """
        Compute the effective sample size of the particle set.

        The effective sample size measures how many particles are
        meaningfully contributing to the approximation.

        A small value indicates severe weight degeneracy and is
        typically used as a trigger for resampling.
        """
        self.normalize()
        return 1.0/np.sum( self.weights**2)
    
@dataclass
class ParticleFilterModel:
    """
    Model container defining the assumptions of a particle filter.

    This object specifies the state-space model and all probability
    components required to perform Sequential Importance Sampling
    with resampling on trajectory space.

    It plays the same structural role as KalmanFilterModel,
    ExtendedKalmanFilterModel, and UnscentedKalmanFilterModel,
    but without any Gaussian assumptions.
    """
    
    Phi: Callable[[np.ndarray], np.ndarray]
    """State trasition function " X_{k+1} = Phi(X_k) (no noise injected here)."""
    
    h:  Callable[[np.ndarray], np.ndarray]
    """Measurement function h(x_k) (no noise injected here)."""
    
    proposal_sampler: Callable[[Optional[np.ndarray], np.random.Generator], np.ndarray]
    """
    Proposal distribution sampler.

    Samples the next state given a particle's past trajectory.

    For initialization, the input trajectory is None and the sampler
    must generate an initial state.

    For prediction, the input is the full previous trajectory and the
    sampler generates the next state conditioned on it.
    """
    
    proposal_logpdf: Callable[[np.ndarray, Optional[np.ndarray]], float]
    """
    Log-density of the proposal distribution.

    Evaluates the log probability of a sampled state under the proposal.

    For initialization, the trajectory argument is None.
    For prediction, the trajectory contains all past states.
    """
    
    transition_logpdf: Callable[[np.ndarray, np.ndarray], float]
    """
    Log-density of the state transition model.

    Evaluates the log probability of transitioning from the previous
    state to the current state under the assumed system dynamics.

    This term corrects the proposal distribution during importance
    weight updates.
    """
    
    measurement_logpdf: Callable[[np.ndarray, np.ndarray], float]
    """
    Log-likelihood of a measurement given the current state.

    Evaluates how well a particle's current state explains an observed
    measurement.

    This term is responsible for incorporating sensor information into
    the particle weights.
    """
    
    ess_threshold_ratio: float = 0.5
    """
    Threshold for resampling based on effective sample size.

    Resampling is triggered when the effective sample size falls below
    ess_threshold_ratio times the number of particles.
    """
    
def pf_initialization_step(
    model           :   ParticleFilterModel,
    num_particles   :   int, 
    rng             :   np.random.Generator   
) -> ParticleSet:
    """
    Initialize the particle system at time k = 0.

    Draws independent samples from the proposal distribution to form
    the initial set of particle trajectories.

    Each particle represents a trajectory of length one:
        X_0^{(i)} = (x_0^{(i)})

    All particles are assigned uniform importance weights.
    """
    
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    
    # Sample once to infer state dimension and validate sampler output
    x0_0 = model.proposal_sampler(None, rng)
    if x0_0.ndim != 1:
        raise ValueError(f"proposal_sampler(None, rng) must return (d,), got shape {x0_0.shape}")
    
    d = x0_0.shape[0] # dimension of the state
    
    # Allocate storage for trajectories and weights
    particles = np.empty((num_particles, 1, d), dtype=float)
    weights = np.full(num_particles, 1.0/num_particles)
    
    # Store first sampled particle
    particles[0, 0, :] = x0_0
    
    # Sample remaining initial particles
    for i in range(1, num_particles):
        x0 = model.proposal_sampler(None, rng)
        particles[i, 0, :] = x0
    
    return ParticleSet(particles=particles, weights=weights)

def pf_prediction_step(
    model           :   ParticleFilterModel,
    particleset_prev:   ParticleSet,
    rng             :   np.random.Generator
) -> ParticleSet:
    """
    Particle filter prediction step.

    Extends each particle trajectory by one time step by sampling
    from the proposal distribution:
        x_k^{(i)} ~ pi(x_k | X_{k-1}^{(i)})

    The resulting particle set represents trajectories:
        X_k^{(i)} = (X_{k-1}^{(i)}, x_k^{(i)})

    Importance weights are propagated unchanged.
    """
    
    particles_prev = particleset_prev.particles
    weights_prev   = particleset_prev.weights
    
    N, T, d = particles_prev.shape
    
    # Allocate storage for extended trajectories
    particles_curr = np.empty((N, T+1, d), dtype=float)
    
    for i in range(N):
        traj_prev = particles_prev[i, :, :]     # (T, d)
        x_new = model.proposal_sampler(traj_prev, rng)
        
        # Copy past trajectory and append new state
        particles_curr[i, :T, :] = np.copy(traj_prev)
        particles_curr[i, T, :] = x_new
    
    return ParticleSet(particles=particles_curr, weights=weights_prev.copy())

def pf_update_step(
    y_observation   :   np.ndarray,
    model           :   ParticleFilterModel,
    particleSet_curr:   ParticleSet
) ->ParticleSet:
    """
    Particle filter update step (Sequential Importance Sampling).
    Updates particle importance weights 
    Particle trajectories are not modified.
    """
     
    particleSet_curr.normalize()
    
    particles_curr = particleSet_curr.particles
    weights_curr   = particleSet_curr.weights
    
    N, T, d = particles_curr.shape
    
    if T < 2:
        raise ValueError("pf_update_step required T>=2")
    
    # Work in log-space for numerical stability
    log_w = np.log(weights_curr + 1e-300)
    
    for i in range(N):
        traj_curr = particles_curr[i]
        
        x_prev = traj_curr[-2] # x_{n-1}
        x_curr = traj_curr[-1] # x_n
        
        # Log likelihood term 
        measurement_log_likelihood  = model.measurement_logpdf(y_observation, x_curr)
        
        # Log transition term
        transition_log_likelihood   = model.transition_logpdf(x_curr, x_prev)
        
        # Log Proposal term
        log_prop = model.proposal_logpdf(x_curr, traj_curr[:-1])
        
        #Incremental log-weight
        log_w[i] += (measurement_log_likelihood + transition_log_likelihood - log_prop)
        
    # Stable normalization . Handle degenerace ( all -inf) safely
    a = np.max(log_w)
    if not np.isfinite(a):
        w_new = np.full(N, 1.0/N)
    else:
        w_new = np.exp(log_w - a)
    
    updated_particleSet = ParticleSet(particles=particles_curr, weights=w_new)
    updated_particleSet.normalize()
    
    return updated_particleSet

def pf_resample_step(
    particleSet_curr   :   ParticleSet,
    rng         :   np.random.Generator
) -> ParticleSet:
    """
    Resampling step of the particle filter.

    Performs systematic resampling on trajectory space to combat
    weight degeneracy. Entire trajectories are resampled according
    to their importance weights.

    After resampling:
      - The number of particles remains unchanged.
      - All resampled trajectories are assigned uniform weights.
      - The empirical distribution is preserved in expectation.
    """
    
    # Ensure weights form a valid probability distribution
    particleSet_curr.normalize()
    
    X = particleSet_curr.particles  # (N, T, d)
    w = particleSet_curr.weights    # (N,)
    N = X.shape[0]
    
    # Systematic resampling grid
    # One random offset shared across all particles
    positions = (rng.random() + np.arange(N)) / N
    
    # Cumulative distribution function of weights
    cdf = np.cumsum(w)
    
    # Numerical safety: ensure final CDF value is exactly 1
    # Prevents out-of-bounds indexing due to floating point error
    cdf[-1] = 1.0

    # Indices of selected trajectories
    idx = np.zeros(N, dtype=int)
    
    
    i = 0   # index over resampling positions
    j = 0   # index over CDF bins
    while i < N:
        if positions[i] < cdf[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    # Return resampled particle set with uniform weights
    return ParticleSet(
        particles=X[idx].copy(),
        weights=np.full(N, 1.0 / N)
    )


def run_particle_filter(
    y_observations  :   Observations,
    model           :   ParticleFilterModel, 
    num_particles   :   int,
    rng             :   np.random.Generator
)->ParticleSet:
    """
    Run a particle filter over a full time horizon.

    The algorithm operates on trajectory space and consists of:
      - initialization
      - repeated prediction (proposal sampling)
      - optional weight updates when measurements are available
      - adaptive resampling based on effective sample size

    The returned ParticleSet represents an empirical approximation
    of the smoothing distribution at the final time step.
    """
    
    # Total number of discrete time steps
    T_total = y_observations.times.shape[0]
    
    # 1) Initialize particle trajectories at time k = 0
    particleSet_curr = pf_initialization_step(model=model,
                                             num_particles=num_particles,
                                             rng=rng)
    
    obs_counter = 0     # index over available measurements
    
    # Main particle filter loop
    for k in range(1, T_total):
        
        # 2) Prediction: extend each trajectory by one state
        particleSet_curr = pf_prediction_step(model=model,
                                             particleset_prev=particleSet_curr,
                                             rng=rng)
        
        # 3) Update weights if a measurement is available at this time
        if obs_counter < len(y_observations.obs_ind) and k == y_observations.obs_ind[obs_counter]:
            y_k = y_observations.obs[obs_counter]
            
            particleSet_curr = pf_update_step(y_observation=y_k,
                                              model=model,
                                              particleSet_curr=particleSet_curr)
            obs_counter += 1
        
        # 4) Resample if particle degeneracy is detected
        if particleSet_curr.ess() < model.ess_threshold_ratio * num_particles:
            particleSet_curr = pf_resample_step(particleSet_curr=particleSet_curr, rng=rng) 
    
    return particleSet_curr