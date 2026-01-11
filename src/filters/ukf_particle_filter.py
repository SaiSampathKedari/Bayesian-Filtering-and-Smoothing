import numpy as np

from typing import Callable, Optional
from dataclasses import dataclass

from filters.kalman_filter import *
from filters.extended_kalman_filter import *
from filters.gauss_hermite_kalman_filter import *
from filters.unscented_kalman_filter import *
from filters.particle_filter import *
from filters.ekf_particle_filter import *
from utils.gaussian_utils import *


@dataclass
class UKFParticleFilterModel(ParticleFilterModel):
    """
    """
    
    ukf_model   :   UnscentedKalmanFilterModel
    """
    Nonlinear model assumed by the UKF.
    Provides Phi, h, Q, and R.
    """
    
    initial_prior   :   Gaussian
    """
    Initial prior distribution p(x_0).
    Used only to sample initial particles.
    """
    
    # Reuse UKF measurement update directly
    ukf_update_one_step  = staticmethod(ukf_update_step)
    
    
    # ------------------------------------------------------------
    # Measurement prediction: p(y_n | x_{n-1}^{(i)})
    # ------------------------------------------------------------
    @staticmethod
    def ukf_prediction_one_step(
        model   :   UnscentedKalmanFilterModel,
        X_prior :   Gaussian
    ) -> Gaussian:
         
        pts, Wm, Wc = unscented_sigma_points(
            X_prior.mean,
            X_prior.cov,
            model.alpha,
            model.beta,
            model.kappa
        )
        
        # Propagate sigma points through measurement model
        Y = np.array([model.h(p) for p in pts])
        
        # Predicted mean
        m_pred = np.sum(Wm[:, None] * Y, axis=0)
        
        # Predicted covariance
        C_pred = np.zeros((m_pred.shape[0], m_pred.shape[0]))
        for i in range(len(Wc)):
            dy = Y[i] - m_pred
            C_pred += Wc[i] * np.outer(dy, dy)
        
        C_pred += model.R
        
        return Gaussian(m_pred, C_pred)
    
    
    
    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def initialize_particles(
        self,
        num_particles   :   int,
        rng             :   np.random.Generator
    ) -> ParticleSet:
        """
        Initialize particle trajectories at time n = 0.

        Each particle is sampled independently from the initial prior:
            x_0^{(i)} ~ p(x_0)

        All particles start with equal weights.
        """
        
        if num_particles <= 0:
            raise ValueError("num_particles must be positive")
        
        d = self.initial_prior.mean.shape[0] #dimension of state
        
        # Allocate storage for trajectories and weights
        particles = np.empty((num_particles, 1, d), dtype=float)
        weights   = np.full((num_particles, 1), 1.0 / num_particles)   # (N,1)
        
        # vectorize this loop later
        for i in range(num_particles):
            particles[i, 0] = rng.multivariate_normal(self.initial_prior.mean, self.initial_prior.cov)
        
        return ParticleSet( particles=particles, weights=weights)
    
    
    
    # ------------------------------------------------------------
    # Propagation (proposal sampling)
    # ------------------------------------------------------------
    def propagate_particles(
        self,
        particle_set   :   ParticleSet,
        rng                 :   np.random.Generator,
        y_observation: Optional[np.ndarray] = None
    ) -> ParticleSet:
        """
        Extend each particle trajectory by one state.

        If no measurement is available:
            x_n^{(i)} ~ p(x_n | x_{n-1}^{(i)})
                      ≈ N( Phi(x_{n-1}^{(i)}), Q )

        If measurement y_n is available:
            x_n^{(i)} ~ p(x_n | x_{n-1}^{(i)}, y_n)
                      ≈ uKF posterior Gaussian

        Importance weights are NOT updated here.
        """
        
        
        prev_particles  =   particle_set.particles
        prev_weights    =   particle_set.weights
        
        N, T, d = prev_particles.shape
        
        # Allocate storage for extended trajectories and weights
        curr_particles = np.empty((N, T+1, d), dtype=float)
        curr_weights   = np.empty((N, T + 1), dtype=float)
        
        # copy full history and carry-forward last weights into new time column
        curr_weights[:, :T] = prev_weights.copy()
        curr_weights[:, T]  = prev_weights[:, T - 1].copy()
        
        for i in range(N):
            traj_prev   = prev_particles[i, :, :]       # (T, d)
            x_prev      = traj_prev[-1]                 # x_{n-1}
            
            # 1) # Prior for  X_n | x_{n-1} ~ N( Phi(x_{n-1}), Q)
            mean_prior = self.ukf_model.Phi(x_prev)
            cov_prior = self.ukf_model.Q
            X_prior = Gaussian(mean=mean_prior, cov=cov_prior)
            
            x_new = np.zeros((d))
            # If no measurement, fall back to bootstrap-style propagation:
            # x_n ~ p(X_n|x_{n-1}) approximated as N(Phi(x_{n-1}), Q)
            if y_observation is None:
                x_new = rng.multivariate_normal(X_prior.mean, X_prior.cov)
            
            # x_n ~ p(x_n|x_{n-1}, y_n)
            else:
                X_posterior = self.ukf_update_one_step(y_measurement=y_observation,
                                              model=self.ukf_model,
                                              X_predicted=X_prior)

                x_new = rng.multivariate_normal(X_posterior.mean, X_posterior.cov)
            
            # Copy past trajectory and append new state
            curr_particles[i, :T] = traj_prev.copy()
            curr_particles[i, T] = x_new
            
        return ParticleSet(particles=curr_particles, weights=curr_weights)
    
    
    
    
    
    # ------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------
    def reweight_particles(
        self,
        y_observation: np.ndarray,
        particle_set: ParticleSet
    ) -> ParticleSet:
        """
        Update importance weights using:

            w_n^{(i)} ∝ w_{n-1}^{(i)} · p(y_n | x_{n-1}^{(i)})

        where p(y_n | x_{n-1}^{(i)}) is approximated using
        UKF-based Gaussian prediction.
        """
        
        curr_particles = particle_set.particles
        curr_weights   = particle_set.weights
        
        N, T, d = curr_particles.shape
        k = T - 1
        
        if T < 2:
            raise ValueError("pf_update_step required T>=2")
        
        # normalize previous column before log
        particle_set.normalize(k-1)
        
        # Work in log-space for numerical stability
        log_w = np.log(curr_weights[:, k-1] + 1e-300)
        
        for i in range(N):
            traj_curr = curr_particles[i]
            
            x_prev = traj_curr[-2] # x_{n-1}
            
            # 1) Constructing X_n | x_{n-1} ~ N( Phi(x_{n-1}), Q)
            mean_prior = self.ukf_model.Phi(x_prev)
            cov_prior = self.ukf_model.Q
            X_prior = Gaussian(mean=mean_prior, cov=cov_prior)
            
            # find P(Y_n | x_{n-1}^{(i)})
            Y_n : Gaussian = self.ukf_prediction_one_step(model=self.ukf_model,
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
        
        curr_weights[:, k] = w_new.copy()
        
        updated_particle_set = ParticleSet(particles=curr_particles, weights=curr_weights)
        updated_particle_set.normalize(k)
        
        return updated_particle_set