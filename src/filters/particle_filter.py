import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import matplotlib.pyplot as plt

from filters.kalman_filter import *

@dataclass
class ParticleSet:
    """
    Empirical distribution:
        P(X) = \sum w_i * \delta_{x_i} 
        
    particles : (N, n) array
    weights   : (N,) array, normalized
    """
    
    particles   :   np.ndarray  # (N, n)
    weights     :   np.ndarray  # (N,)
    
    def normalize(self) -> None:
        s = np.sum(self.weights)
        if (not np.isfinite(s)) or (s <= 0.0):
            self.weights = np.full(self.weights.shape[0], 1.0/self.weights.shape[0])
        
        else:
            self.weights = self.weights/s
    
    def ess(self) -> float:
        self.normalize()
        return 1.0/np.sum(self.weights ** 2)
    
    def moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted mean and covariance of particle cloud.
        """
        self.normalize()
        w = self.weights
        X = self.particles
        m = np.sum(X * w[:, None], axis=0)
        d = X - m[None, :]
        C = (d.T * w) @ d
        return m, C            


@dataclass
class PFTracker:
    means: np.ndarray
    covs: np.ndarray
    stds: np.ndarray
    ess: np.ndarray
    
