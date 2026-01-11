import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

from filters.kalman_filter import *




# ============================================================
# Particle container
# ============================================================
@dataclass
class ParticleSet:
    """
    Empirical approximation of filtering and smoothing distribution.

    Stores:
    particles[i, k] : state of particle i at time k
    weights[i, k]   : importance weight of particle i at time k
    """
    
    particles   : np.ndarray
    """
    Shape: (N, T, d)

    particles[i]      = full trajectory of particle i
    particles[i, k]   = state at time k for particle i

    N = number of particles
    T = trajectory length (time steps stored)
    d = state dimension
    """
    
    weights     : np.ndarray
    """
    Shape: (N, T)

    Importance weights associated with each particle(trajectory).
    Weights need only be proportional; normalization is handled internally.
    """
    
    def normalize(self, time_index: int):
        """
        Normalize weights at a specific time index.

        If weights are invalid or numerically degenerate,
        reset to a uniform distribution.
        """
        
        w = self.weights[:, time_index]
        s = np.sum(w)
        
        if (not np.isfinite(s)) or (s <= 0.0):
            self.weights[:, time_index] = 1.0 / w.shape[0]
        else:
            self.weights[:, time_index] = w / s
    
    def effective_sample_size(self, time_index: int) -> float:
        """
        Compute Effective Sample Size (ESS).

        ESS ≈ number of particles meaningfully contributing
        to the approximation. Used to detect weight degeneracy.
        """
        
        self.normalize(time_index)
        w = self.weights[:, time_index]
        return 1.0 / np.sum(w**2)


    # --------------------------------------------------
    # Filtering distribution
    # --------------------------------------------------
    def filtering_distribution(self, time_index: int):
        """
        Empirical filtering distribution at time_index.

        Returns:
            X_k : (N, d) particle states
            w_k : (N,) normalized weights
        """
        if time_index < 0 or time_index >= self.weights.shape[1]:
            raise IndexError("time_index out of bounds")

        self.normalize(time_index)

        X_k = self.particles[:, time_index, :]
        w_k = self.weights[:, time_index]

        return X_k, w_k
    
    
    
    # --------------------------------------------------
    # Point estimates
    # --------------------------------------------------
    def mean(self, time_index: int) -> np.ndarray:
        """
        Weighted mean of filtering distribution at time_index.
        """
        X_k, w_k = self.filtering_distribution(time_index)
        return np.sum(w_k[:, None] * X_k, axis=0)


    def covariance(self, time_index: int) -> np.ndarray:
        """
        Weighted covariance of filtering distribution at time_index.
        """
        X_k, w_k = self.filtering_distribution(time_index)
        mean = np.sum(w_k[:, None] * X_k, axis=0)

        diff = X_k - mean
        C = np.zeros((diff.shape[1], diff.shape[1]))
        for i in range(diff.shape[0]):
            C += w_k[i] * np.outer(diff[i], diff[i])

        return C
    
    def map_particle(self, time_index: int):
        """
        Return state of the maximum-weight particle at time_index.
        """
        if time_index < 0 or time_index >= self.weights.shape[1]:
            raise IndexError("time_index out of bounds")

        self.normalize(time_index)
        i = np.argmax(self.weights[:, time_index])
        return self.particles[i, time_index]




# ============================================================
# Abstract particle filter model interface
# ============================================================
class ParticleFilterModel(ABC):
    """
    Abstract interface for a particle filter variant.

    A concrete PF implementation must define:
      - how particles are initialized
      - how trajectories are propagated (proposal)
      - how importance weights are updated

    Resampling logic is intentionally kept external and shared.
    """
    
    # Resampling is triggered when ESS < ratio * N
    ess_threshold_ratio: float = 0.5
    
    @abstractmethod
    def initialize_particles(
        self,
        num_particles: int,
        rng: np.random.Generator
    ) -> ParticleSet:
        """Sample initial particle trajectories."""
        raise NotImplementedError
    
    @abstractmethod
    def propagate_particles(
        self,
        particle_set: ParticleSet,
        rng: np.random.Generator,
        y_observation: Optional[np.ndarray] = None
    ) -> ParticleSet:
        """
        Extend each trajectory by one time step.

        The proposal distribution may optionally depend on
        the current measurement (optimal proposals).
        """
        raise NotImplementedError
    
    @abstractmethod
    def reweight_particles(
        self,
        y_observation: np.ndarray,
        particle_set: ParticleSet
    ) -> ParticleSet:
        """
        Update importance weights using the measurement likelihood
        or its approximation.
        """
        raise NotImplementedError
    



# ============================================================
# Resampling (shared across all PF variants)
# ============================================================
def pf_resample_step(
    particle_set    :   ParticleSet,
    time_index      :   int,
    rng             :   np.random.Generator
) -> ParticleSet:
    """
    Perform systematic resampling on trajectory space.

    Entire trajectories are resampled according to their weights.
    After resampling, weights at time_index are reset to uniform..
    """
    
    # Ensure weights form a valid probability distribution
    particle_set.normalize(time_index)
    
    X = particle_set.particles  # (N, T, d)
    w = particle_set.weights[:, time_index]    # (N,)
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

    # Resample trajectories
    particles_resampled = X[idx].copy()

    # Copy full weight history, reset current column
    weights_resampled = particle_set.weights[idx].copy()
    weights_resampled[:, time_index] = 1.0 / N
    
    # Return resampled particle set with uniform weights
    return ParticleSet(particles_resampled, weights_resampled)



# ============================================================
# Generic particle filter runner
# ============================================================
def run_particle_filter(
    y_observations  :   Observations,
    model           :   ParticleFilterModel,
    num_particles   :   int,
    rng             :   np.random.Generator
)->ParticleSet:
    """
    Generic particle filter driver.

    Executes Sequential Importance Sampling with Resampling (SISR)
    on trajectory space:

        1) Initialize particles
        2) For each time step:
            - propagate trajectories
            - reweight if a measurement is available
            - resample if ESS drops below threshold

    Returns
    -------
    ParticleSet
        Empirical approximation of the smoothing distribution
        at the final time step.
    """

    
    # Total number of discrete time steps
    num_steps = y_observations.times.shape[0]
    
    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    particle_set = model.initialize_particles(num_particles=num_particles,
                                             rng=rng)
    
    obs_counter = 0     # index into observation list
    
    
    # --------------------------------------------------
    # Main filtering loop
    # --------------------------------------------------
    for k in range(1, num_steps):
        y_k = None
        have_measurement = (obs_counter < len(y_observations.obs_ind)) and (k == y_observations.obs_ind[obs_counter])
        if have_measurement:
            y_k = y_observations.obs[obs_counter]
        
        # Propagation (proposal sampling)
        particle_set = model.propagate_particles(
                            particle_set=particle_set,
                            rng=rng,
                            y_observation=y_k
                            )
        
        # Weight update
        if have_measurement:
            particle_set = model.reweight_particles(y_observation=y_k,
                                              particle_set=particle_set)
            obs_counter += 1
        
        # Resample if degeneracy is detected
        if particle_set.effective_sample_size(k) < model.ess_threshold_ratio * num_particles:
            particle_set = pf_resample_step(particle_set=particle_set, time_index=k, rng=rng) 
    
    return particle_set