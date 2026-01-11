import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

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


class ParticleFilterModel(ABC):
    """
    Abstract interface for a particle filter variant.

    Any PF variant must define how to:
      - initialize particles
      - propagate (predict)
      - incorporate a measurement (update)

    Resampling is kept outside to stay identical across variants.
    """
    
    ess_threshold_ratio: float = 0.5
    
    @abstractmethod
    def pf_initialize_step(
        self,
        num_particles: int,
        rng: np.random.Generator
    ) -> ParticleSet:
        raise NotImplementedError
    
    @abstractmethod
    def pf_propagate_step(
        self,
        particle_set_prev: ParticleSet,
        rng: np.random.Generator,
        y: Optional[np.ndarray] = None
    ) -> ParticleSet:
        raise NotImplementedError
    
    @abstractmethod
    def pf_reweight_step(
        self,
        y_observation: np.ndarray,
        particle_set_curr: ParticleSet
    ) -> ParticleSet:
        raise NotImplementedError
    

def pf_resample_step(
    particle_set_prev   :   ParticleSet,
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