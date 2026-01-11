import numpy as np

# ============================================================
# Utility: Multivariate Gaussian log-pdf
# ============================================================
def mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Log-density of a multivariate Gaussian N(mean, cov).

    Implemented using Cholesky decomposition for numerical stability.
    Used for evaluating p(y_n | x_{n-1}^{(i)}) in the EKF-based PF.
    """
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)

    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, x - mean)
    quad = float(z.T @ z)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    d = x.shape[0]
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)