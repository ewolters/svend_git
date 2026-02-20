"""
Bayesian conjugate posterior routines for the Normal-Inverse-Gamma linear model.

All routines operate in **precision form** internally (Λn = Vn⁻¹).
Covariance is computed only when needed for reporting.

Math:
    y = Xβ + ε,  ε ~ N(0, σ²I)
    Prior: β|σ² ~ N(μ₀, σ²·Λ₀⁻¹),  σ² ~ InvGamma(α₀, β₀)
    Posterior: β|σ²,y ~ N(μn, σ²·Λn⁻¹),  σ²|y ~ InvGamma(αn, βn)

    where  Λn = Λ₀ + X'X
           μn = Λn⁻¹(Λ₀μ₀ + X'y)
           αn = α₀ + n/2
           βn = β₀ + ½(y'y + μ₀'Λ₀μ₀ − μn'Λnμn)

    Marginal: c'β | y ~ t(2αn, c'μn, √(βn/αn · c'Λn⁻¹c))
"""

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.special import gammaln


def bayesian_linear_posterior(X, y, mu0=None, Lambda0=None, alpha0=2.0, beta0=None):
    """
    Compute the Normal-Inverse-Gamma posterior for a linear model.

    Parameters
    ----------
    X : ndarray (n, p)
        Design matrix (should include intercept column if desired).
    y : ndarray (n,)
        Response vector.
    mu0 : ndarray (p,), optional
        Prior mean for β.  Default: zeros.
    Lambda0 : ndarray (p, p), optional
        Prior precision matrix.  Default: Zellner g-prior with g = max(n, p).
    alpha0 : float
        Prior shape for σ².  Default: 2.0 (finite variance).
    beta0 : float, optional
        Prior scale for σ².  Default: alpha0 * s² (centered on sample variance).

    Returns
    -------
    mu_n : ndarray (p,)
        Posterior mean of β.
    Lambda_n : ndarray (p, p)
        Posterior precision matrix.
    L_n : ndarray (p, p)
        Lower Cholesky factor of Lambda_n.
    alpha_n : float
        Posterior shape.
    beta_n : float
        Posterior scale.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape

    XtX = X.T @ X

    # Ridge stabilization scaled to trace
    trace_val = np.trace(XtX)
    lam = 1e-8 * trace_val / p if p > 0 and trace_val > 0 else 1e-8
    XtX_reg = XtX + lam * np.eye(p)

    # Default prior: Zellner g-prior, g = max(n, p)
    if Lambda0 is None:
        g = max(n, p)
        Lambda0 = XtX_reg / g
    else:
        Lambda0 = np.asarray(Lambda0, dtype=np.float64)

    if mu0 is None:
        mu0 = np.zeros(p)
    else:
        mu0 = np.asarray(mu0, dtype=np.float64).ravel()

    if beta0 is None:
        s2 = np.var(y, ddof=1) if n > 1 else 1.0
        beta0 = max(alpha0 * s2, 1e-10)

    # Posterior precision
    Lambda_n = Lambda0 + XtX

    # Cholesky decomposition for numerical stability
    # Add small ridge for collinear data
    try:
        L_n = np.linalg.cholesky(Lambda_n)
    except np.linalg.LinAlgError:
        Lambda_n += np.eye(Lambda_n.shape[0]) * 1e-6
        L_n = np.linalg.cholesky(Lambda_n)

    # Posterior mean via Cholesky solve
    Xty = X.T @ y
    rhs = Lambda0 @ mu0 + Xty
    mu_n = cho_solve((L_n, True), rhs)

    # Posterior shape and scale
    alpha_n = alpha0 + n / 2.0
    quad_prior = float(mu0 @ Lambda0 @ mu0)
    quad_post = float(mu_n @ Lambda_n @ mu_n)
    beta_n = beta0 + 0.5 * (float(y @ y) + quad_prior - quad_post)
    beta_n = max(beta_n, 1e-10)

    return mu_n, Lambda_n, L_n, alpha_n, beta_n


def contrast_posterior(c, mu_n, Lambda_n, L_n, alpha_n, beta_n):
    """
    Marginal posterior of a linear contrast c'β.

    c'β | y ~ Student-t(df=2αn, loc=c'μn, scale=√(βn/αn · c'Λn⁻¹c))

    Parameters
    ----------
    c : ndarray (p,)
        Contrast vector.
    mu_n, Lambda_n, L_n, alpha_n, beta_n :
        Posterior from bayesian_linear_posterior().

    Returns
    -------
    loc : float
        Posterior mean of c'β.
    scale : float
        Scale parameter of the Student-t.
    df : float
        Degrees of freedom (2·αn).
    """
    c = np.asarray(c, dtype=np.float64).ravel()
    loc = float(c @ mu_n)
    # Λn⁻¹ c via Cholesky
    v = cho_solve((L_n, True), c)
    scale = float(np.sqrt(max(beta_n / alpha_n * (c @ v), 1e-20)))
    df = 2.0 * alpha_n
    return loc, scale, df


def predictive_posterior(x_new, mu_n, Lambda_n, L_n, alpha_n, beta_n):
    """
    Posterior predictive distribution for a new observation.

    y*|x*,y ~ Student-t(df=2αn, loc=x*'μn, scale=√(βn/αn · (1 + x*'Λn⁻¹x*)))

    Parameters
    ----------
    x_new : ndarray (p,)
        Predictor vector for the new point (same coding as design matrix).
    mu_n, Lambda_n, L_n, alpha_n, beta_n :
        Posterior from bayesian_linear_posterior().

    Returns
    -------
    loc : float
        Predictive mean.
    scale : float
        Predictive scale.
    df : float
        Degrees of freedom.
    """
    x_new = np.asarray(x_new, dtype=np.float64).ravel()
    loc = float(x_new @ mu_n)
    v = cho_solve((L_n, True), x_new)
    pred_var_scale = float(beta_n / alpha_n * (1.0 + x_new @ v))
    scale = float(np.sqrt(max(pred_var_scale, 1e-20)))
    df = 2.0 * alpha_n
    return loc, scale, df


def marginal_log_likelihood(Lambda0, Lambda_n, L_0, L_n, alpha0, alpha_n,
                            beta0, beta_n, n):
    """
    Log marginal likelihood ln p(y|M) for model comparison.

    Uses Cholesky log-determinants for numerical stability.

    Parameters
    ----------
    Lambda0 : ndarray (p, p)
        Prior precision matrix.
    Lambda_n : ndarray (p, p)
        Posterior precision matrix.
    L_0 : ndarray (p, p)
        Lower Cholesky factor of Lambda0.
    L_n : ndarray (p, p)
        Lower Cholesky factor of Lambda_n.
    alpha0, alpha_n : float
        Prior and posterior shape parameters.
    beta0, beta_n : float
        Prior and posterior scale parameters.
    n : int
        Number of observations.

    Returns
    -------
    float
        Log marginal likelihood.
    """
    # Log-determinants via Cholesky diagonals
    log_det_Lambda0 = 2.0 * np.sum(np.log(np.diag(L_0)))
    log_det_Lambda_n = 2.0 * np.sum(np.log(np.diag(L_n)))

    return (
        -n / 2.0 * np.log(2.0 * np.pi)
        + 0.5 * log_det_Lambda0
        - 0.5 * log_det_Lambda_n
        + alpha0 * np.log(max(beta0, 1e-300))
        - alpha_n * np.log(max(beta_n, 1e-300))
        + gammaln(alpha_n)
        - gammaln(alpha0)
    )
