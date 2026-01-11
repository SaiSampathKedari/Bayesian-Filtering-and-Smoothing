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
    
    def ess(self) -> float:
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
    particle_set_prev   :   ParticleSet,
    rng         :   np.random.Generator
) -> ParticleSet:
    """
    Perform systematic resampling on trajectory space.

    Entire trajectories are resampled according to their weights.
    After resampling, all weights are reset to uniform.
    """
    
    # Ensure weights form a valid probability distribution
    particle_set_prev.normalize()
    
    X = particle_set_prev.particles  # (N, T, d)
    w = particle_set_prev.weights    # (N,)
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
    
    Generic PF runner. Works for any model implementing ParticleFilterModel.
    """
    
    # Total number of discrete time steps
    T_total = y_observations.times.shape[0]
    
    # --------------------------------------------------
    # 1) Initialization
    # --------------------------------------------------
    particle_set = model.pf_initialize_step(num_particles=num_particles,
                                             rng=rng)
    
    obs_counter = 0     # index over available measurements
    
    
    # --------------------------------------------------
    # 2) Main filtering loop
    # --------------------------------------------------
    for k in range(1, T_total):
        y_k = None
        have_measurement = (obs_counter < len(y_observations.obs_ind)) and (k == y_observations.obs_ind[obs_counter])
        if have_measurement:
            y_k = y_observations.obs[obs_counter]
        
        # 3) propagate: extend each trajectory by one state (proposal may use y_k)
        particle_set = model.pf_propagate_step(
                            particle_set_prev=particle_set,
                            rng=rng,
                            y=y_k
                            )
        
        # 3) Update weights if a measurement is available at this time
        if have_measurement:
            particle_set = model.pf_reweight_step(y_observation=y_k,
                                              particle_set_curr=particle_set)
            obs_counter += 1
        
        # 4) Resample if particle degeneracy is detected
        if particle_set.ess() < model.ess_threshold_ratio * num_particles:
            particle_set = pf_resample_step(particle_set_prev=particle_set, rng=rng) 
    
    return particle_set