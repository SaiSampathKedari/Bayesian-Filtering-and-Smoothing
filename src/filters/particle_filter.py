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
    Empirical approximation of a smoothing distribution.

    Stores a collection of weighted state trajectories:
        { (X_{0:k}^{(i)}, w_k^{(i)}) }_{i=1}^N

    This is the particle-based analogue of a Gaussian belief
    used in Kalman filtering, but defined on trajectory space.
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
    Shape: (N,)

    Importance weights associated with each particle(trajectory).
    Weights need only be proportional; normalization is handled internally.
    """
    
    def normalize(self):
        """
        Normalize weights in-place.

        If weights are invalid or numerically degenerate,
        reset to a uniform distribution.
        """
        
        s = np.sum(self.weights)
        
        if (not np.isfinite(s)) or (s <= 0.0):
            self.weights = np.full(self.weights.shape[0], 1.0/self.weights.shape[0])
        else:
            self.weights = self.weights/s
    
    def effective_sample_size(self) -> float:
        """
        Compute Effective Sample Size (ESS).

        ESS ≈ number of particles meaningfully contributing
        to the approximation. Used to detect weight degeneracy.
        """
        
        self.normalize()
        return 1.0/np.sum( self.weights**2)





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
    rng             :   np.random.Generator
) -> ParticleSet:
    """
    Perform systematic resampling on trajectory space.

    Entire trajectories are resampled according to their weights.
    After resampling, all weights are reset to uniform.
    """
    
    # Ensure weights form a valid probability distribution
    particle_set.normalize()
    
    X = particle_set.particles  # (N, T, d)
    w = particle_set.weights    # (N,)
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
        if particle_set.effective_sample_size() < model.ess_threshold_ratio * num_particles:
            particle_set = pf_resample_step(particle_set=particle_set, rng=rng) 
    
    return particle_set