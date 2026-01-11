import numpy as np
from filters.particle_filter import *

from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class BootStrapParticleFilterModel(ParticleFilterModel):
    """
    Bootstrap Particle Filter model.

    This model implements Sequential Importance Sampling with resampling
    using the transition density as the proposal distribution.

    Proposal:
        x_k ~ p(x_k | x_{k-1})

    Weight update:
        w_k = w_{k-1} * p(y_k | x_k)

    No proposal correction terms are required.
    """

    transition_sampler: Callable[[Optional[np.ndarray], np.random.Generator], np.ndarray]
    """
    State transition sampler.

    - If input state is None, samples from the initial prior p(x_0).
    - Otherwise, samples x_k ~ p(x_k | x_{k-1}).
    """
    
    measurement_logpdf: Callable[[np.ndarray, np.ndarray], float]
    """
    Measurement log-likelihood.

    Computes log p(y_k | x_k).
    """
    
    
    
    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def pf_initialize_step(
        self,
        num_particles   :   int,
        rng             :   np.random.Generator
    ) -> ParticleSet:
        """
        Initialize particle trajectories at time k = 0.

        Each particle is a trajectory of length one.
        All particles are assigned uniform weights.
        """
        
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        
        # Sample Once to infer state dimention and validate sampler output
        x0_0 = self.transition_sampler(None, rng)
        if x0_0.ndim != 1:
            raise ValueError(f"transition_sampler(None, rng) must returen (d,), got shape {x0_0.shape}")

        d = x0_0.shape[0]
        
        # Allocate storage for trajectories and weights
        particles = np.empty((num_particles, 1, d))
        weights = np.full(num_particles, 1.0/num_particles)
        
        # Store first sampled particle
        particles[0, 0, :] = np.copy(x0_0)
        
        # Sample remaining initial particles
        for i in range(1, num_particles):
            x0 = self.transition_sampler(None, rng)
            particles[i, 0, :] = np.copy(x0)
            
        return ParticleSet(particles=particles, weights=weights)
    
    
    
    # ------------------------------------------------------------
    # Propagate
    # ------------------------------------------------------------
    def pf_propagate_step(
        self,
        particle_set_prev:   ParticleSet,
        rng             :   np.random.Generator,
        y: Optional[np.ndarray] = None
    ) -> ParticleSet:
        """
        Propagate particles forward using the transition model.

        For each particle:
            x_k ~ p(x_k | x_{k-1})

        Trajectories are extended by one state.
        Weights are propagated unchanged.
        """
        
        particles_prev = particle_set_prev.particles
        weights_prev   = particle_set_prev.weights
        
        N, T, d = particles_prev.shape
        
        # Allocate storage for extended trajectories
        particles_curr = np.empty((N, T+1, d), dtype=float)
        
        for i in range(N):
            traj_prev = particles_prev[i, :, :]             # (T, d)
            x_prev = traj_prev[-1]                          # x_{n-1}
            x_new = self.transition_sampler(x_prev, rng)    # x_n
            if x_new.ndim != 1:
                raise ValueError("transition_sampler must return a 1D state vector")

            # Copy past trajectory and append new state
            particles_curr[i, :T, :] = np.copy(traj_prev)
            particles_curr[i, T, :] = x_new
        
        return ParticleSet(particles=particles_curr, weights=weights_prev.copy())
    
    
    
    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------
    def pf_reweight_step(
        self,
        y_observation   :   np.ndarray,
        particle_set_curr:   ParticleSet
    ) ->ParticleSet:
        """
        Update particle weights using the measurement likelihood.

        Weight update:
            w_k ∝ w_{k-1} · p(y_k | x_k)
        """

        particle_set_curr.normalize()
        
        particles_curr = particle_set_curr.particles
        weights_curr   = particle_set_curr.weights
        
        N, T, d = particles_curr.shape
        
        if T < 2:
            raise ValueError("pf_update_step required T>=2")
        
        # Work in log-space for numerical stability
        log_w = np.log(weights_curr + 1e-300)
        
        for i in range(N):
            traj_curr = particles_curr[i]
            
            x_curr = traj_curr[-1] # x_n
            
            # Log likelihood term 
            measurement_log_likelihood  = self.measurement_logpdf(y_observation, x_curr)
                        
            #Incremental log-weight
            log_w[i] += measurement_log_likelihood
            
        # Stable normalization . Handle degenerace ( all -inf) safely
        a = np.max(log_w)
        if not np.isfinite(a):
            w_new = np.full(N, 1.0/N)
        else:
            w_new = np.exp(log_w - a)
        
        updated_particleSet = ParticleSet(particles=particles_curr, weights=w_new)
        updated_particleSet.normalize()
        
        return updated_particleSet