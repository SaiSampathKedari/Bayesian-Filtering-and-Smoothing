import numpy as np
from filters.particle_filter import *

from filters.kalman_filter import *
from filters.extended_kalman_filter import *

from dataclasses import dataclass
from typing import Optional, Callable


def mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Stable log N(x | mean, cov) via Cholesky.
    """
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)

    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, x - mean)
    quad = float(z.T @ z)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    d = x.shape[0]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class EKFParticleFilterModel(ParticleFilterModel):
    
    """
    EKF-based Particle Filter using one-step EKF Gaussian proposal.
    """
    
    ekf_model   :   ExtendedKalmanFilterModel
    X0          :   Gaussian # Initial Proposal of Particles
    
    ekf_update_one_step = staticmethod(ekf_update_step)
    
    @staticmethod
    def ekf_prediction_one_step(
        model   :   ExtendedKalmanFilterModel,
        X_prior :   Gaussian    
    )-> Gaussian:
        
        # Prior statistics
        m_prev = X_prior.mean
        C_prev = X_prior.cov
        
        # Mean propagation
        m_pred = model.h(m_prev)
        
        # Jacobian of dynamics evaluated at prior mean
        H_k = model.jacobian_h(m_prev)
        
        # Covariance propagation
        C_pred = H_k @ C_prev @ H_k.T + model.R
        
        return Gaussian(m_pred, C_pred)
    
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
        
        d = self.X0.mean.shape[0] #dimension of state
        
        # Allocate storage for trajectories and weights
        particles = np.empty((num_particles, 1, d), dtype=float)
        weights = np.full(num_particles, 1.0/num_particles)
        
        # vectorize this loop later
        for i in range(num_particles):
            particles[i, 0] = rng.multivariate_normal(self.X0.mean, self.X0.cov)
        
        return ParticleSet( particles=particles, weights=weights)
    
    def pf_propagate_step(
        self,
        particle_set_prev   :   ParticleSet,
        rng                 :   np.random.Generator,
        y: Optional[np.ndarray] = None
    ) -> ParticleSet:
        
        particles_prev  =   particle_set_prev.particles
        weights_prev    =   particle_set_prev.weights
        
        N, T, d = particles_prev.shape
        
        # Allocate storage for extended trajectories
        particles_curr = np.empty((N, T+1, d), dtype=float)
        
        for i in range(N):
            traj_prev   = particles_prev[i, :, :]       # (T, d)
            x_prev      = traj_prev[-1]                 # x_{n-1}
            
            # 1) Constructing X_n | x_{n-1} ~ N( Phi(x_{n-1}), Q)
            mean_prior = self.ekf_model.Phi(x_prev)
            cov_prior = self.ekf_model.Q
            X_prior = Gaussian(mean=mean_prior, cov=cov_prior)
            
            x_new = np.zeros((d))
            # If no measurement, fall back to bootstrap-style propagation:
            # x_n ~ p(X_n|x_{n-1}) approximated as N(Phi(x_{n-1}), Q)
            if y is None:
                x_new = rng.multivariate_normal(X_prior.mean, X_prior.cov)
            
            # x_n ~ p(x_n|x_{n-1}, y_n)
            else:
                X_posterior = self.ekf_update_one_step(y_measurement=y,
                                              model=self.ekf_model,
                                              X_predicted=X_prior)

                x_new = rng.multivariate_normal(X_posterior.mean, X_posterior.cov)
            
            # Copy past trajectory and append new state
            particles_curr[i, :T] = traj_prev.copy()
            particles_curr[i, T] = x_new
            
        return ParticleSet(particles=particles_curr, weights=weights_prev.copy())


    def pf_reweight_step(
        self,
        y_observation: np.ndarray,
        particle_set_curr: ParticleSet
    ) -> ParticleSet:
        
        particles_curr = particle_set_curr.particles
        weights_curr   = particle_set_curr.weights
        
        N, T, d = particles_curr.shape
        
        if T < 2:
            raise ValueError("pf_update_step required T>=2")
        
        # Work in log-space for numerical stability
        log_w = np.log(weights_curr + 1e-300)
        
        for i in range(N):
            traj_curr = particles_curr[i]
            
            x_prev = traj_curr[-2] # x_{n-1}
            
            # 1) Constructing X_n | x_{n-1} ~ N( Phi(x_{n-1}), Q)
            mean_prior = self.ekf_model.Phi(x_prev)
            cov_prior = self.ekf_model.Q
            X_prior = Gaussian(mean=mean_prior, cov=cov_prior)
            
            # find P(Y_n | x_{n-1}^{(i)})
            Y_n : Gaussian = self.ekf_prediction_one_step(model=self.ekf_model,
                                               X_prior = X_prior)
            
            # Log likelihood term
            measurement_log_likelihood = mvn_logpdf(x=y_observation,
                                                    mean=Y_n.mean,
                                                    cov=Y_n.cov)
            
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