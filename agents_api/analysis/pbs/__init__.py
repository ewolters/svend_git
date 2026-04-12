"""
Process Belief System (PBS) — Core Engine.

Maintains a probabilistic belief state for process characteristics using
Normal-Gamma conjugate priors.  Every chart reads from this state.

Sections (matching specification):
 1. NormalGammaPosterior — core belief state with O(1) updates
 2. BeliefChart — BOCPD shift probability (Adams & MacKay 2007)
 3. UncertaintyFusedChart — per-point credible intervals fusing gage error
 4. EvidenceAccumulationChart — anytime-valid e-values
 5. PredictiveChart — Bayesian linear trend with prediction fan
 6. AdaptiveControlLimits — posterior predictive limits that narrow over time
 7. BayesianCpk — posterior distribution of Cpk via ancestral sampling
 8. CpkTrajectory — rolling Cpk with trend projection
 9. MultiStreamHealth — log-linear fusion of health streams
10. ProcessNarrative — deterministic template-based summary
11. ProbabilisticAlarms — decision-theoretic alert thresholds
12. ChartGenealogy — prior inheritance from parent processes

Dependencies: numpy, scipy.  No MCMC.  No LLM calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from ..common import _narrative

logger = logging.getLogger(__name__)

__all__ = [
    "NormalGammaPosterior",
    "BeliefChart",
    "EDetector",
    "UncertaintyFusion",
    "EvidenceAccumulation",
    "PredictiveChart",
    "AdaptiveControlLimits",
    "BayesianCpk",
    "CpkTrajectory",
    "MultiStreamHealth",
    "ProcessNarrative",
    "ProbabilisticAlarms",
    "ChartGenealogy",
    "run_pbs",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. CORE BELIEF STATE — Normal-Gamma Posterior
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NormalGammaPosterior:
    """
    Conjugate Normal-Gamma posterior for (μ, τ=1/σ²).

    Prior:  μ|τ ~ N(μ₀, 1/(κ₀·τ)),  τ ~ Gamma(α₀, β₀)
    """

    mu: float = 0.0
    kappa: float = 0.01
    alpha: float = 0.01
    beta: float = 0.01

    # Caps for numerical stability
    _KAPPA_MAX = 1e8
    _BETA_MIN = 1e-15
    _BETA_MAX = 1e15

    def update(self, x: np.ndarray) -> NormalGammaPosterior:
        """Update posterior with observations x (array).  Returns self."""
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        if n == 0:
            return self

        x_bar = float(np.mean(x))
        s2 = float(np.var(x))  # divide by n, per spec §1.2

        kappa_new = min(self.kappa + n, self._KAPPA_MAX)
        mu_new = (self.kappa * self.mu + n * x_bar) / kappa_new
        alpha_new = self.alpha + n / 2.0
        beta_new = self.beta + n * s2 / 2.0 + self.kappa * n * (x_bar - self.mu) ** 2 / (2.0 * kappa_new)
        beta_new = np.clip(beta_new, self._BETA_MIN, self._BETA_MAX)

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = float(beta_new)
        return self

    def update_single(self, x: float) -> NormalGammaPosterior:
        """Update with a single observation.  O(1)."""
        kappa_new = min(self.kappa + 1, self._KAPPA_MAX)
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + self.kappa * (x - self.mu) ** 2 / (2.0 * kappa_new)
        beta_new = np.clip(beta_new, self._BETA_MIN, self._BETA_MAX)

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = float(beta_new)
        return self

    @property
    def variance_estimate(self) -> float:
        """Point estimate E[σ²] = β/(α-1) for α>1, else β/α."""
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return self.beta / max(self.alpha, 0.01)

    @property
    def std_estimate(self) -> float:
        return math.sqrt(self.variance_estimate)

    def marginal_mu(self) -> tuple[float, float, float]:
        """Marginal posterior of μ: Student-t(2α, μ, β/(α·κ)).
        Returns (nu, loc, scale)."""
        nu = 2 * self.alpha
        loc = self.mu
        scale = math.sqrt(self.beta / (self.alpha * self.kappa))
        return nu, loc, scale

    def predictive(self) -> tuple[float, float, float]:
        """Posterior predictive for x_new: Student-t(2α, μ, β(κ+1)/(α·κ)).
        Returns (nu, loc, scale)."""
        nu = 2 * self.alpha
        loc = self.mu
        scale = math.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return nu, loc, scale

    def predictive_logpdf(self, x: float) -> float:
        """Log-pdf of x under posterior predictive."""
        nu, loc, scale = self.predictive()
        return float(sp_stats.t.logpdf(x, df=nu, loc=loc, scale=scale))

    def copy(self) -> NormalGammaPosterior:
        return NormalGammaPosterior(mu=self.mu, kappa=self.kappa, alpha=self.alpha, beta=self.beta)

    def discount(self, factor: float) -> NormalGammaPosterior:
        """Discount certainty by factor ∈ (0,1] for prior inheritance."""
        return NormalGammaPosterior(
            mu=self.mu,
            kappa=self.kappa * factor,
            alpha=max(self.alpha * factor, 0.5),
            beta=self.beta * factor,
        )


# ═══════════════════════════════════════════════════════════════════════════
# INVESTIGATION TIMELINE — single source of truth for cross-panel sync
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ChangePointEvent:
    obs: int  # observation index
    shift_prob: float  # P(shift) at detection
    e_value: float  # evidence accumulation ratio at detection
    evidence_level: str  # 'none'|'notable'|'strong'|'decisive'
    robustness: int = 0  # how many λ values detect this CP (out of grid size)
    robustness_total: int = 5  # grid size
    confirmation_obs: int = -1  # per-CP: obs where P(shift) >= 0.95
    near_known_transition: bool = False  # within ±3 obs of a metadata transition


@dataclass
class TaguchiLossResult:
    """Expected Taguchi quality loss per unit from a Normal-Gamma posterior.

    L(y) = k·(y - target)².  Under the posterior:
      E[L] = k·[ (μ - target)² + σ²_μ + E[σ²] ]
    where σ²_μ = β / (κ·(α-1)) captures posterior mean uncertainty,
    and E[σ²] = β / (α-1) is the expected process variance.
    """

    expected_loss: float  # E[k·(Y - target)²] per unit
    bias_loss: float  # k·(μ_post - target)² portion
    variance_loss: float  # k·E[σ²] portion
    uncertainty_loss: float  # k·σ²_μ portion (posterior uncertainty on mean)
    bias_fraction: float  # bias_loss / expected_loss
    variance_fraction: float  # variance_loss / expected_loss
    k: float  # loss coefficient used
    target: float  # target value used


def compute_taguchi_loss(posterior: NormalGammaPosterior, target: float, k: float = 1.0) -> TaguchiLossResult:
    """Compute expected Taguchi loss from a Normal-Gamma posterior.

    Analytical: no sampling needed. Same posterior used for Cpk.
    """
    # E[σ²] from inverse-gamma marginal
    if posterior.alpha > 1:
        e_var = posterior.beta / (posterior.alpha - 1)
    else:
        e_var = posterior.beta / max(posterior.alpha, 0.01)

    # Posterior uncertainty on the mean: Var[μ] = β / (κ·(α-1))
    if posterior.alpha > 1:
        var_mu = posterior.beta / (posterior.kappa * (posterior.alpha - 1))
    else:
        var_mu = posterior.beta / (posterior.kappa * max(posterior.alpha, 0.01))

    bias_sq = (posterior.mu - target) ** 2
    bias_loss = k * bias_sq
    variance_loss = k * e_var
    uncertainty_loss = k * var_mu
    total = bias_loss + variance_loss + uncertainty_loss

    if total > 0:
        bias_frac = bias_loss / total
        var_frac = variance_loss / total
    else:
        bias_frac = 0.0
        var_frac = 0.0

    return TaguchiLossResult(
        expected_loss=total,
        bias_loss=bias_loss,
        variance_loss=variance_loss,
        uncertainty_loss=uncertainty_loss,
        bias_fraction=bias_frac,
        variance_fraction=var_frac,
        k=k,
        target=target,
    )


@dataclass
class RegimeStats:
    label: str  # "Regime 1", "Regime 2", etc.
    start: int
    end: int
    n: int
    mean: float
    std: float
    cpk: object = None  # Optional[BayesianCpkResult] — None if no specs
    ci_narrowing_30: float = 0.0  # % CI narrows with 30 more obs
    taguchi: object = None  # Optional[TaguchiLossResult]


@dataclass
class KnownTransition:
    obs: int  # observation index where transition occurs
    column: str  # 'material_lot', 'operator', 'machine'
    from_value: str
    to_value: str


@dataclass
class LotCapability:
    lot_id: str
    start: int
    end: int
    n: int
    mean: float
    std: float
    cpk: object = None  # Optional[BayesianCpkResult]
    ci_narrowing_30: float = 0.0
    within_lot_cps: list = None  # BOCPD changepoints within this lot (not at boundary)
    taguchi: object = None  # Optional[TaguchiLossResult]


@dataclass
class InvestigationTimeline:
    changepoints: list  # List[ChangePointEvent]
    regimes: list  # List[RegimeStats]
    ed_peak_obs: int = -1  # observation of E-Detector max log(N)
    ed_peak_log_N: float = 0.0
    ed_first_alarm_obs: int = -1  # first E-Detector alarm, -1 if none
    ed_threshold: float = 3.0  # log(1/alpha) alarm threshold
    best_lambda: float = 0.0  # MAP λ from empirical Bayes grid
    lambda_log_evidences: dict = None  # {λ: log_marginal_likelihood}
    known_transitions: list = None  # List[KnownTransition]
    lot_capabilities: list = None  # List[LotCapability]


# ═══════════════════════════════════════════════════════════════════════════
# 2. BELIEF CHART — Bayesian Online Changepoint Detection
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BeliefChartPoint:
    t: int
    observation: float
    shift_probability: float
    most_likely_run_length: int
    current_regime_mean: float
    current_regime_std: float
    reference_mean: float
    alert_level: str  # 'nominal' | 'watch' | 'alert' | 'alarm'
    observation_weight: float = 1.0  # DPD weight (1.0 = standard, <1 = downweighted outlier)


class BeliefChart:
    """
    BOCPD (Adams & MacKay 2007) with Normal-Gamma sufficient statistics.

    When beta_robustness > 0, uses Dm-BOCD (Altamirano, Briol & Knoblauch,
    ICML 2023) — density power divergence scoring for outlier robustness.
    beta=0 recovers standard BOCPD.
    """

    def __init__(
        self,
        hazard_lambda: float = 200.0,
        prior: NormalGammaPosterior | None = None,
        max_run_lengths: int = 200,
        thresholds: dict[str, float] | None = None,
        beta_robustness: float = 0.0,
        max_neff: int | None = 50,
    ):
        self.hazard_lambda = hazard_lambda
        self.K = max_run_lengths
        self.prior = prior or NormalGammaPosterior()
        self.thresholds = thresholds or {"watch": 0.50, "alert": 0.80, "alarm": 0.95}
        self.beta_robustness = float(beta_robustness)
        self.max_neff = max_neff  # cap effective sample size per run length

        # Run length distribution: log P(r_t | x_{1:t})
        # Start: P(r_0 = 0) = 1
        self._log_R = np.array([0.0])  # log probabilities

        # Sufficient statistics per run length
        # Each entry: (mu, kappa, alpha, beta)
        self._suff = [(self.prior.mu, self.prior.kappa, self.prior.alpha, self.prior.beta)]

        self._reference_mean = self.prior.mu
        self._t = 0
        self.points: list[BeliefChartPoint] = []
        self.log_marginal_likelihood = 0.0  # Σ log p(x_t | x_{1:t-1}, λ)

    def _hazard(self, r: int) -> float:
        return 1.0 / self.hazard_lambda

    def process(self, x: float) -> BeliefChartPoint:
        """Process one observation, return belief chart point."""
        self._t += 1
        beta_r = self.beta_robustness

        n_r = len(self._log_R)

        # 1. Evaluate predictive probability under each run length
        log_pred = np.empty(n_r)
        dpd_weights = np.ones(n_r)  # DPD weights per run length
        for i in range(n_r):
            mu_i, kappa_i, alpha_i, beta_i = self._suff[i]
            nu = 2 * alpha_i
            loc = mu_i
            sigma2_pred = beta_i * (kappa_i + 1) / (alpha_i * kappa_i)
            scale = math.sqrt(sigma2_pred)
            log_pred[i] = sp_stats.t.logpdf(x, df=nu, loc=loc, scale=scale)

            # Compute DPD weight for robust update (§7.3 of spec)
            if beta_r > 0:
                z = (x - mu_i) ** 2 / sigma2_pred
                dpd_weights[i] = math.exp(-beta_r * z / 2.0)

        # 2. Growth probabilities: log P(r_t = r+1, x_{1:t})
        H = np.array([self._hazard(r) for r in range(n_r)])
        log_growth = self._log_R + log_pred + np.log(1.0 - H)

        # 3. Changepoint probability: log P(r_t = 0, x_{1:t})
        log_cp_terms = self._log_R + log_pred + np.log(H)
        log_cp = _logsumexp(log_cp_terms)

        # 4. Combine
        new_log_R = np.empty(n_r + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # 5. Normalize
        log_evidence = _logsumexp(new_log_R)
        new_log_R -= log_evidence
        self.log_marginal_likelihood += log_evidence

        # 6. Update sufficient statistics
        new_suff = []
        # r=0: reset to prior
        new_suff.append((self.prior.mu, self.prior.kappa, self.prior.alpha, self.prior.beta))
        # r>0: update each run's sufficient statistics
        # When beta_robustness > 0, use weighted updates (Dm-BOCD §8)
        for i in range(n_r):
            mu_i, kappa_i, alpha_i, beta_i = self._suff[i]
            w = dpd_weights[i]  # 1.0 when beta_robustness == 0
            kappa_new = kappa_i + w
            mu_new = (kappa_i * mu_i + w * x) / kappa_new
            alpha_new = alpha_i + w / 2.0
            beta_new = beta_i + w * kappa_i * (x - mu_i) ** 2 / (2.0 * kappa_new)
            # Windowed sufficient stats: cap kappa to prevent long regimes
            # from becoming immovable.  Decay alpha/beta proportionally to
            # preserve posterior mean of sigma^2 while widening uncertainty.
            if self.max_neff and kappa_new > self.max_neff:
                decay = self.max_neff / kappa_new
                kappa_new = float(self.max_neff)
                alpha_new *= decay
                beta_new *= decay
            new_suff.append((mu_new, min(kappa_new, 1e8), alpha_new, max(beta_new, 1e-15)))

        # 7. Truncate to top K run lengths
        if len(new_log_R) > self.K:
            top_k = np.argsort(new_log_R)[-self.K :]
            top_k.sort()
            new_log_R = new_log_R[top_k]
            new_suff = [new_suff[i] for i in top_k]
            # Renormalize
            new_log_R -= _logsumexp(new_log_R)

        self._log_R = new_log_R
        self._suff = new_suff

        # 8. Compute shift probability
        shift_prob = 1.0 - math.exp(new_log_R[-1]) if len(new_log_R) > 1 else 0.0
        shift_prob = np.clip(shift_prob, 0.0, 1.0)

        # 9. Most likely run length
        ml_idx = int(np.argmax(new_log_R))
        ml_run = ml_idx

        # Current regime parameters (MAP run length)
        mu_r, kappa_r, alpha_r, beta_r_param = self._suff[ml_idx]
        regime_std = math.sqrt(beta_r_param / max(alpha_r - 1, 0.5))

        # Observation weight: mean DPD weight across top run lengths
        obs_weight = float(dpd_weights[ml_idx]) if ml_idx < len(dpd_weights) else 1.0

        # Alert level
        if shift_prob >= self.thresholds["alarm"]:
            alert = "alarm"
        elif shift_prob >= self.thresholds["alert"]:
            alert = "alert"
        elif shift_prob >= self.thresholds["watch"]:
            alert = "watch"
        else:
            alert = "nominal"

        pt = BeliefChartPoint(
            t=self._t,
            observation=x,
            shift_probability=float(shift_prob),
            most_likely_run_length=ml_run,
            current_regime_mean=float(mu_r),
            current_regime_std=float(regime_std),
            reference_mean=float(self._reference_mean),
            alert_level=alert,
            observation_weight=obs_weight,
        )
        self.points.append(pt)
        return pt

    def reset_reference(self):
        """Reset reference to current regime (after acknowledged change)."""
        if self._suff:
            ml_idx = int(np.argmax(self._log_R))
            self._reference_mean = self._suff[ml_idx][0]


# ═══════════════════════════════════════════════════════════════════════════
# 2b. E-DETECTOR — Distribution-Free CUSUM Changepoint Detection
#     (Shin, Ramdas & Rinaldo 2024)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EDetectorPoint:
    t: int
    observation: float
    log_N_upper: float  # CUSUM statistic for upward shift
    log_N_lower: float  # CUSUM statistic for downward shift
    log_N_combined: float  # max(upper, lower) — used for alarm
    alarm: bool
    alarm_direction: str | None  # 'upper', 'lower', or None
    lambda_upper: float  # adaptive betting param (positive side)
    lambda_lower: float  # adaptive betting param (negative side)
    estimated_mean: float  # running mean estimate


class EDetector:
    """
    CUSUM-style e-detector for nonparametric sequential changepoint detection.

    Uses sub-Gaussian GROW e-values with adaptive lambda (ONS).
    Two-sided: runs parallel upper/lower detectors, alarms when either
    crosses threshold. ARL guarantee: E[tau] >= 1/(2*alpha) under any
    pre-change distribution satisfying sub-Gaussian bound.
    """

    def __init__(self, mu_0: float, bounds: tuple[float, float], alpha: float = 0.05):
        self.mu_0 = mu_0
        self.a, self.b = bounds
        self.alpha = alpha
        self.threshold = math.log(1.0 / alpha)  # per-side threshold
        self.range_sq = (self.b - self.a) ** 2 / 8.0  # sub-Gaussian proxy
        self.lambda_max = 1.0 / max(self.b - self.a, 1e-10)

        # State — two-sided CUSUM
        self._log_N_upper = 0.0
        self._log_N_lower = 0.0
        self._lambda_upper = 0.0
        self._lambda_lower = 0.0
        self._sum_x = 0.0
        self._count = 0
        self._t = 0
        self.points: list[EDetectorPoint] = []

    def process(self, x: float) -> EDetectorPoint:
        """Process one observation, return e-detector point."""
        self._t += 1
        self._count += 1
        self._sum_x += x
        x_bar = self._sum_x / self._count

        # Clamp observation to bounds for sub-Gaussian validity
        x_c = max(self.a, min(self.b, x))

        # Upper detector: testing for upward shift (lambda > 0)
        log_e_upper = self._lambda_upper * (x_c - self.mu_0) - self._lambda_upper**2 * self.range_sq
        self._log_N_upper = max(0.0, self._log_N_upper) + log_e_upper

        # Lower detector: testing for downward shift (lambda < 0)
        log_e_lower = self._lambda_lower * (x_c - self.mu_0) - self._lambda_lower**2 * self.range_sq
        self._log_N_lower = max(0.0, self._log_N_lower) + log_e_lower

        # Combined statistic
        log_N_combined = max(self._log_N_upper, self._log_N_lower)

        # Alarm check
        alarm = False
        alarm_dir = None
        if self._log_N_upper >= self.threshold:
            alarm = True
            alarm_dir = "upper"
        if self._log_N_lower >= self.threshold:
            alarm = True
            alarm_dir = "lower" if not alarm_dir else alarm_dir

        # Adapt lambda via ONS — optimal bet for estimated shift
        # Upper: lambda* = (x_bar - mu_0) / (2 * range_sq), clipped positive
        lam_opt_upper = (x_bar - self.mu_0) / (2.0 * self.range_sq + 1e-15)
        self._lambda_upper = max(0.0, min(lam_opt_upper, self.lambda_max))

        # Lower: lambda* = (x_bar - mu_0) / (2 * range_sq), clipped negative
        lam_opt_lower = (x_bar - self.mu_0) / (2.0 * self.range_sq + 1e-15)
        self._lambda_lower = min(0.0, max(lam_opt_lower, -self.lambda_max))

        pt = EDetectorPoint(
            t=self._t,
            observation=x,
            log_N_upper=float(self._log_N_upper),
            log_N_lower=float(self._log_N_lower),
            log_N_combined=float(log_N_combined),
            alarm=alarm,
            alarm_direction=alarm_dir,
            lambda_upper=float(self._lambda_upper),
            lambda_lower=float(self._lambda_lower),
            estimated_mean=float(x_bar),
        )
        self.points.append(pt)
        return pt


# ═══════════════════════════════════════════════════════════════════════════
# 3. UNCERTAINTY-FUSED CHART
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class UncertaintyFusedPoint:
    t: int
    observed_value: float
    fused_mean: float
    fused_std: float
    ci_lower: float
    ci_upper: float
    gage_health_pct: float
    gage_health_status: str  # 'excellent' | 'acceptable' | 'attention'
    process_std_estimate: float
    gage_std_estimate: float


class UncertaintyFusion:
    """Fuse process variation with measurement uncertainty."""

    def __init__(self, gage_alpha: float = 0.01, gage_beta: float = 0.01):
        self.gage_alpha = gage_alpha
        self.gage_beta = gage_beta

    @property
    def gage_variance(self) -> float:
        if self.gage_alpha > 1:
            return self.gage_beta / (self.gage_alpha - 1)
        return self.gage_beta / max(self.gage_alpha, 0.01)

    def update_gage(self, repeat_measurements: np.ndarray):
        """Update gage variance estimate from repeated measurements on same part."""
        y = np.asarray(repeat_measurements, dtype=float)
        k = len(y)
        if k < 2:
            return
        y_bar = np.mean(y)
        ss = float(np.sum((y - y_bar) ** 2))
        self.gage_alpha += k / 2.0
        self.gage_beta += ss / 2.0

    def fuse_point(self, x_obs: float, process_mu: float, process_var: float) -> UncertaintyFusedPoint:
        """Compute fused estimate of true value given observation."""
        gage_var = self.gage_variance
        gage_std = math.sqrt(gage_var)

        if gage_var < 1e-15 or process_var < 1e-15:
            fused_mean = x_obs
            fused_var = max(gage_var, process_var, 1e-15)
        else:
            fused_var = 1.0 / (1.0 / process_var + 1.0 / gage_var)
            fused_mean = fused_var * (x_obs / gage_var + process_mu / process_var)

        fused_std = math.sqrt(fused_var)

        # 5th/95th percentile
        ci_lower = fused_mean - 1.645 * fused_std
        ci_upper = fused_mean + 1.645 * fused_std

        # %GRR
        total_var = process_var + gage_var
        grr_pct = (gage_var / total_var * 100) if total_var > 0 else 0

        if grr_pct < 10:
            status = "excellent"
        elif grr_pct < 30:
            status = "acceptable"
        else:
            status = "attention"

        return UncertaintyFusedPoint(
            t=0,
            observed_value=x_obs,
            fused_mean=fused_mean,
            fused_std=fused_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            gage_health_pct=grr_pct,
            gage_health_status=status,
            process_std_estimate=math.sqrt(process_var),
            gage_std_estimate=gage_std,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVIDENCE ACCUMULATION CHART
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvidenceChartPoint:
    t: int
    observation: float
    e_value_individual: float
    e_value_accumulated: float
    log_e_accumulated: float
    evidence_level: str  # 'none' | 'notable' | 'strong' | 'decisive'


class EvidenceAccumulation:
    """
    Anytime-valid e-value accumulation using Normal mixture alternative.
    Uses FIXED reference parameters from calibration phase (Grünwald 2024).
    """

    def __init__(self, mu_0: float, sigma_ref: float, sigma_mix_ratio: float = 1.0):
        self.mu_0 = mu_0
        self.sigma2 = sigma_ref**2
        self.sigma2_mix = self.sigma2 * sigma_mix_ratio
        self.log_E = 0.0  # accumulated log e-value
        self._t = 0
        self.points: list[EvidenceChartPoint] = []

    def process(self, x: float) -> EvidenceChartPoint:
        """Compute e-value for observation x against H₀: μ = μ₀, σ = σ_ref."""
        self._t += 1

        # Gaussian mixture e-value (Grünwald 2024)
        # e_t = sqrt(σ²/(σ²+σ²_mix)) · exp(σ²_mix·(x−μ₀)² / (2σ²(σ²+σ²_mix)))
        ratio = self.sigma2 / (self.sigma2 + self.sigma2_mix)
        exponent = self.sigma2_mix * (x - self.mu_0) ** 2 / (2.0 * self.sigma2 * (self.sigma2 + self.sigma2_mix))
        log_e_i = 0.5 * math.log(max(ratio, 1e-300)) + exponent

        # Accumulate in log space (§4.4)
        self.log_E += log_e_i

        # Cap display at 10000:1 (anti-pattern #10)
        log_E_display = min(self.log_E, math.log(10000))
        E_display = math.exp(log_E_display)
        e_i = math.exp(min(log_e_i, math.log(10000)))

        if E_display >= 100:
            level = "decisive"
        elif E_display >= 20:
            level = "strong"
        elif E_display >= 5:
            level = "notable"
        else:
            level = "none"

        pt = EvidenceChartPoint(
            t=self._t,
            observation=x,
            e_value_individual=e_i,
            e_value_accumulated=E_display,
            log_e_accumulated=float(self.log_E),
            evidence_level=level,
        )
        self.points.append(pt)
        return pt

    def reset(self, new_mu_0: float):
        """Reset after acknowledged changepoint."""
        self.mu_0 = new_mu_0
        self.log_E = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. PREDICTIVE CHART
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PredictionPoint:
    horizon: int
    mean: float
    ci90_lower: float
    ci90_upper: float
    ci50_lower: float
    ci50_upper: float


@dataclass
class PredictiveChartOutput:
    current_mean: float
    current_slope: float
    slope_credible_interval: tuple[float, float]
    slope_probability_positive: float
    slope_probability_negative: float
    prediction_fan: list[PredictionPoint]
    prob_exceed_spec_10: float
    prob_exceed_spec_25: float
    estimated_obs_to_exceedance: int | None


class PredictiveChart:
    """Bayesian linear trend model on a rolling window."""

    def __init__(self, window: int = 20):
        self.window = window

    def compute(
        self,
        observations: np.ndarray,
        USL: float | None = None,
        LSL: float | None = None,
        horizon: int = 25,
    ) -> PredictiveChartOutput:
        data = np.asarray(observations, dtype=float)
        w = min(len(data), self.window)
        y = data[-w:]
        n = len(y)
        t_vals = np.arange(len(data) - w, len(data), dtype=float)

        # Design matrix
        X = np.column_stack([np.ones(n), t_vals])

        # Weak prior
        m0 = np.array([np.mean(y), 0.0])
        Lambda0 = np.diag([0.01, 0.01])
        alpha0 = 0.01
        beta0 = 0.01

        # Posterior (§5.3)
        Lambda_n = Lambda0 + X.T @ X
        m_n = np.linalg.solve(Lambda_n, Lambda0 @ m0 + X.T @ y)
        alpha_n = alpha0 + n / 2.0
        beta_n = beta0 + 0.5 * (y @ y + m0 @ Lambda0 @ m0 - m_n @ Lambda_n @ m_n)
        beta_n = max(beta_n, 1e-15)

        nu = 2 * alpha_n
        t_current = float(len(data) - 1)

        # Slope posterior: marginal of b
        # b ~ Student-t(nu, m_n[1], (beta_n/alpha_n) * Lambda_n_inv[1,1])
        Lambda_n_inv = np.linalg.inv(Lambda_n)  # OK here, 2x2 is fine
        slope_scale = math.sqrt(beta_n / alpha_n * Lambda_n_inv[1, 1])
        p_slope_pos = float(1.0 - sp_stats.t.cdf(0, df=nu, loc=m_n[1], scale=slope_scale))
        p_slope_neg = float(sp_stats.t.cdf(0, df=nu, loc=m_n[1], scale=slope_scale))
        slope_ci = (
            float(sp_stats.t.ppf(0.05, df=nu, loc=m_n[1], scale=slope_scale)),
            float(sp_stats.t.ppf(0.95, df=nu, loc=m_n[1], scale=slope_scale)),
        )

        # Prediction fan (§5.4)
        fan = []
        for h in range(1, horizon + 1):
            x_star = np.array([1.0, t_current + h])
            pred_mean = float(x_star @ m_n)
            pred_scale_sq = (beta_n / alpha_n) * (1.0 + x_star @ np.linalg.solve(Lambda_n, x_star))
            pred_scale = math.sqrt(max(pred_scale_sq, 1e-15))

            fan.append(
                PredictionPoint(
                    horizon=h,
                    mean=pred_mean,
                    ci90_lower=float(sp_stats.t.ppf(0.05, df=nu, loc=pred_mean, scale=pred_scale)),
                    ci90_upper=float(sp_stats.t.ppf(0.95, df=nu, loc=pred_mean, scale=pred_scale)),
                    ci50_lower=float(sp_stats.t.ppf(0.25, df=nu, loc=pred_mean, scale=pred_scale)),
                    ci50_upper=float(sp_stats.t.ppf(0.75, df=nu, loc=pred_mean, scale=pred_scale)),
                )
            )

        # Spec exceedance probability (§5.5)
        p_exceed_10 = 0.0
        p_exceed_25 = 0.0
        est_exceedance = None

        if USL is not None or LSL is not None:
            p_exceed_10 = self._prob_exceed(fan[:10], USL, LSL, nu, alpha_n, beta_n, Lambda_n, m_n, t_current)
            p_exceed_25 = self._prob_exceed(fan, USL, LSL, nu, alpha_n, beta_n, Lambda_n, m_n, t_current)
            # Estimate first horizon where P(exceed) > 50%
            p_all_ok = 1.0
            for h_idx, pt in enumerate(fan):
                x_s = np.array([1.0, t_current + pt.horizon])
                ps_sq = (beta_n / alpha_n) * (1.0 + x_s @ np.linalg.solve(Lambda_n, x_s))
                ps = math.sqrt(max(ps_sq, 1e-15))
                p_ok = 1.0
                if USL is not None:
                    p_ok -= 1 - sp_stats.t.cdf(USL, df=nu, loc=pt.mean, scale=ps)
                if LSL is not None:
                    p_ok -= sp_stats.t.cdf(LSL, df=nu, loc=pt.mean, scale=ps)
                p_all_ok *= max(p_ok, 0)
                if 1 - p_all_ok > 0.5 and est_exceedance is None:
                    est_exceedance = pt.horizon

        return PredictiveChartOutput(
            current_mean=float(m_n[0] + m_n[1] * t_current),
            current_slope=float(m_n[1]),
            slope_credible_interval=slope_ci,
            slope_probability_positive=p_slope_pos,
            slope_probability_negative=p_slope_neg,
            prediction_fan=fan,
            prob_exceed_spec_10=p_exceed_10,
            prob_exceed_spec_25=p_exceed_25,
            estimated_obs_to_exceedance=est_exceedance,
        )

    @staticmethod
    def _prob_exceed(fan, USL, LSL, nu, alpha_n, beta_n, Lambda_n, m_n, t_current) -> float:
        p_all_ok = 1.0
        for pt in fan:
            x_star = np.array([1.0, t_current + pt.horizon])
            ps_sq = (beta_n / alpha_n) * (1.0 + x_star @ np.linalg.solve(Lambda_n, x_star))
            ps = math.sqrt(max(ps_sq, 1e-15))
            p_ok = 1.0
            if USL is not None:
                p_ok -= 1 - sp_stats.t.cdf(USL, df=nu, loc=pt.mean, scale=ps)
            if LSL is not None:
                p_ok -= sp_stats.t.cdf(LSL, df=nu, loc=pt.mean, scale=ps)
            p_all_ok *= max(p_ok, 0)
        return 1.0 - p_all_ok


# ═══════════════════════════════════════════════════════════════════════════
# 6. ADAPTIVE CONTROL LIMITS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptiveLimitsPoint:
    t: int
    observation: float
    cl: float  # center line
    ucl: float  # upper control limit
    lcl: float  # lower control limit
    n_obs: int  # observations seen so far


class AdaptiveControlLimits:
    """Control limits from posterior predictive that narrow over time."""

    def __init__(self, gamma: float = 0.9973):
        """gamma: coverage probability (0.9973 ≈ ±3σ equivalent)."""
        self.gamma = gamma

    def compute_limits(self, posterior: NormalGammaPosterior, t: int, x: float) -> AdaptiveLimitsPoint:
        nu, loc, scale = posterior.predictive()

        lcl = float(sp_stats.t.ppf((1 - self.gamma) / 2, df=nu, loc=loc, scale=scale))
        ucl = float(sp_stats.t.ppf((1 + self.gamma) / 2, df=nu, loc=loc, scale=scale))

        return AdaptiveLimitsPoint(
            t=t,
            observation=x,
            cl=loc,
            ucl=ucl,
            lcl=lcl,
            n_obs=int(posterior.kappa - 0.01 + 1),
        )

    @staticmethod
    def check_consistency(bay_ucl, bay_lcl, cls_ucl, cls_lcl, n):
        """Warn if Bayesian limits diverge from classical by >10% at n≥50."""
        if n >= 50:
            bay_w = bay_ucl - bay_lcl
            cls_w = cls_ucl - cls_lcl
            if cls_w > 0 and abs(bay_w - cls_w) / cls_w > 0.10:
                logger.warning(
                    "Bayesian limits diverge from classical by "
                    f"{abs(bay_w - cls_w) / cls_w:.1%} at n={n}. "
                    "Check prior specification."
                )


# ═══════════════════════════════════════════════════════════════════════════
# 7. BAYESIAN Cpk
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BayesianCpkResult:
    cpk_point_estimate: float
    cpk_credible_interval: tuple[float, float]
    cpk_probability_above_1: float
    cpk_probability_above_133: float
    cpk_probability_above_167: float
    shift_estimate: float
    shift_credible_interval: tuple[float, float]
    classical_cpk: float
    n_observations: int
    prior_source: str


class BayesianCpk:
    """Posterior distribution of Cpk via ancestral sampling from Normal-Gamma."""

    def __init__(self, USL: float, LSL: float, sigma_shift: float = 1.5, n_samples: int = 10000):
        self.USL = USL
        self.LSL = LSL
        self.sigma_shift = sigma_shift
        self.n_samples = n_samples

    def compute(
        self,
        posterior: NormalGammaPosterior,
        n_obs: int,
        prior_source: str = "uninformative",
        seed: int = 42,
    ) -> BayesianCpkResult:
        rng = np.random.RandomState(seed)

        # Ancestral sampling: τ ~ Gamma(α, 1/β), μ|τ ~ N(μ_n, 1/(κ_n·τ))
        tau_samples = rng.gamma(posterior.alpha, 1.0 / posterior.beta, size=self.n_samples)
        sigma_mu = 1.0 / np.sqrt(posterior.kappa * tau_samples)
        mu_samples = rng.normal(posterior.mu, sigma_mu)
        sigma_samples = 1.0 / np.sqrt(tau_samples)

        # Short-term Cpk (§7.2)
        cpu = (self.USL - mu_samples) / (3 * sigma_samples)
        cpl = (mu_samples - self.LSL) / (3 * sigma_samples)
        cpk_samples = np.minimum(cpu, cpl)

        # Process-specific shift (§7.3)
        shift_samples = np.abs(rng.normal(0, self.sigma_shift, size=self.n_samples))

        # Long-term Cpk with process-specific shift
        cpk_samples - shift_samples / (3 * sigma_samples)

        # Classical Cpk (point estimate)
        sigma_est = posterior.std_estimate
        mu_est = posterior.mu
        classical = min((self.USL - mu_est) / (3 * sigma_est), (mu_est - self.LSL) / (3 * sigma_est))

        return BayesianCpkResult(
            cpk_point_estimate=float(np.median(cpk_samples)),
            cpk_credible_interval=(
                float(np.percentile(cpk_samples, 5)),
                float(np.percentile(cpk_samples, 95)),
            ),
            cpk_probability_above_1=float(np.mean(cpk_samples > 1.0)),
            cpk_probability_above_133=float(np.mean(cpk_samples > 1.33)),
            cpk_probability_above_167=float(np.mean(cpk_samples > 1.67)),
            shift_estimate=float(np.median(shift_samples)),
            shift_credible_interval=(
                float(np.percentile(shift_samples, 5)),
                float(np.percentile(shift_samples, 95)),
            ),
            classical_cpk=float(classical),
            n_observations=n_obs,
            prior_source=prior_source,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Cpk TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CpkTrajectoryPoint:
    t: int
    cpk_median: float
    cpk_ci_lower: float
    cpk_ci_upper: float
    n_observations: int


@dataclass
class CpkTrajectoryOutput:
    trajectory: list[CpkTrajectoryPoint]
    current_cpk: BayesianCpkResult
    trend_slope: float
    trend_slope_ci: tuple[float, float]
    prob_cpk_declining: float
    estimated_obs_to_threshold: int | None
    threshold: float


class CpkTrajectory:
    """Rolling Bayesian Cpk with trend projection."""

    def __init__(self, USL: float, LSL: float, window: int | None = None, threshold: float = 1.33):
        self.USL = USL
        self.LSL = LSL
        self.window = window
        self.threshold = threshold

    def compute(
        self,
        observations: np.ndarray,
        prior: NormalGammaPosterior,
        n_cpk_samples: int = 5000,
        seed: int = 42,
    ) -> CpkTrajectoryOutput:
        data = np.asarray(observations, dtype=float)
        n_total = len(data)
        np.random.RandomState(seed)
        trajectory = []

        cpk_engine = BayesianCpk(self.USL, self.LSL, n_samples=n_cpk_samples)

        # Compute Cpk at each step (or subsampled for large datasets)
        step = max(1, n_total // 100)  # at most 100 points
        for t in range(step - 1, n_total, step):
            if self.window and t >= self.window:
                chunk = data[t - self.window : t + 1]
            else:
                chunk = data[: t + 1]

            # Build posterior for this chunk
            post = prior.copy()
            post.update(chunk)

            cpk_result = cpk_engine.compute(post, len(chunk), seed=seed + t)
            trajectory.append(
                CpkTrajectoryPoint(
                    t=t,
                    cpk_median=cpk_result.cpk_point_estimate,
                    cpk_ci_lower=cpk_result.cpk_credible_interval[0],
                    cpk_ci_upper=cpk_result.cpk_credible_interval[1],
                    n_observations=len(chunk),
                )
            )

        # Current Cpk (full data)
        post_full = prior.copy()
        post_full.update(data)
        current = cpk_engine.compute(post_full, n_total, seed=seed)

        # Trend on Cpk trajectory (§8.3)
        if len(trajectory) >= 3:
            t_vals = np.array([p.t for p in trajectory], dtype=float)
            cpk_vals = np.array([p.cpk_median for p in trajectory])
            PredictiveChart(window=len(trajectory))
            # Fit trend to Cpk values
            n_t = len(t_vals)
            X = np.column_stack([np.ones(n_t), t_vals])
            m0 = np.array([np.mean(cpk_vals), 0.0])
            L0 = np.diag([0.01, 0.01])
            a0, b0 = 0.01, 0.01
            Ln = L0 + X.T @ X
            mn = np.linalg.solve(Ln, L0 @ m0 + X.T @ cpk_vals)
            an = a0 + n_t / 2.0
            bn = b0 + 0.5 * (cpk_vals @ cpk_vals + m0 @ L0 @ m0 - mn @ Ln @ mn)
            bn = max(bn, 1e-15)
            nu_t = 2 * an
            Ln_inv = np.linalg.inv(Ln)
            slope_scale = math.sqrt(bn / an * Ln_inv[1, 1])
            prob_declining = float(sp_stats.t.cdf(0, df=nu_t, loc=mn[1], scale=slope_scale))
            slope_ci = (
                float(sp_stats.t.ppf(0.05, df=nu_t, loc=mn[1], scale=slope_scale)),
                float(sp_stats.t.ppf(0.95, df=nu_t, loc=mn[1], scale=slope_scale)),
            )
            # Estimate obs to threshold crossing
            est_cross = None
            if mn[1] < 0:  # declining
                t_last = trajectory[-1].t
                t_star = (self.threshold - mn[0]) / mn[1]
                if t_star > t_last:
                    est_cross = int(t_star - t_last)
        else:
            prob_declining = 0.5
            slope_ci = (0.0, 0.0)
            mn = np.array([current.cpk_point_estimate, 0.0])
            est_cross = None

        return CpkTrajectoryOutput(
            trajectory=trajectory,
            current_cpk=current,
            trend_slope=float(mn[1]),
            trend_slope_ci=slope_ci,
            prob_cpk_declining=prob_declining,
            estimated_obs_to_threshold=est_cross,
            threshold=self.threshold,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 9. MULTI-STREAM HEALTH CHART
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_HEALTH_WEIGHTS = {
    "spc": 0.35,
    "cpk": 0.25,
    "gage": 0.15,
    "trend": 0.15,
    "material": 0.05,
    "env": 0.05,
}


@dataclass
class HealthDecomposition:
    overall_health: float
    stream_contributions: dict[str, float]
    stream_weights: dict[str, float]
    primary_driver: str
    driver_impact: float


class MultiStreamHealth:
    """Log-linear opinion pool for process health fusion."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or dict(DEFAULT_HEALTH_WEIGHTS)

    def fuse(
        self,
        streams: dict[str, float],
        previous_streams: dict[str, float] | None = None,
    ) -> HealthDecomposition:
        """Fuse health probabilities via log-linear pooling (§9.3)."""
        # Filter to available streams, redistribute weights
        available = {k: v for k, v in streams.items() if k in self.weights}
        if not available:
            return HealthDecomposition(
                overall_health=0.5,
                stream_contributions={},
                stream_weights={},
                primary_driver="none",
                driver_impact=0.0,
            )

        total_w = sum(self.weights[k] for k in available)
        norm_weights = {k: self.weights[k] / total_w for k in available}

        assert abs(sum(norm_weights.values()) - 1.0) < 1e-6

        log_health = sum(norm_weights[k] * math.log(max(available[k], 0.01)) for k in available)
        overall = math.exp(log_health)

        # Find primary driver (§9.5)
        driver = "none"
        driver_impact = 0.0
        if previous_streams:
            impacts = {}
            for key in available:
                if key in previous_streams:
                    cf = dict(available)
                    cf[key] = previous_streams[key]
                    cf_log = sum(norm_weights[k] * math.log(max(cf[k], 0.01)) for k in available)
                    impacts[key] = math.exp(cf_log) - overall
            if impacts:
                driver = max(impacts, key=impacts.get)
                driver_impact = impacts[driver]
        else:
            # Without previous, find lowest contributor
            driver = min(available, key=available.get)
            driver_impact = 0.0

        return HealthDecomposition(
            overall_health=float(overall),
            stream_contributions=dict(available),
            stream_weights=norm_weights,
            primary_driver=driver,
            driver_impact=float(driver_impact),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 10. PROCESS NARRATIVE ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class ProcessNarrative:
    """Deterministic template-based process narrative. NO LLM calls."""

    @staticmethod
    def generate(
        belief_pt: BeliefChartPoint | None,
        evidence_pt: EvidenceChartPoint | None,
        predictive: PredictiveChartOutput | None,
        cpk: BayesianCpkResult | None,
        health: HealthDecomposition | None,
        uncertainty_pt: UncertaintyFusedPoint | None = None,
        beta_robustness: float = 0.0,
        belief_points: list | None = None,
        timeline: InvestigationTimeline | None = None,
        USL: float | None = None,
        LSL: float | None = None,
        taguchi_k: float = 0.0,
        taguchi_target: float | None = None,
        y: np.ndarray | None = None,
        prior: NormalGammaPosterior | None = None,
    ) -> str:
        segments = []

        # ── Timeline-aware narrative (replaces simple state description) ──
        has_shifts = timeline and timeline.changepoints
        if has_shifts:
            cps = timeline.changepoints
            regimes = timeline.regimes

            # Shift location(s) with confirmation arc and robustness
            def _robustness_label(cp_ev):
                if cp_ev.robustness_total <= 1:
                    return ""
                if cp_ev.robustness >= cp_ev.robustness_total:
                    return " (robust — detected at all \u03bb values)"
                elif cp_ev.robustness >= cp_ev.robustness_total - 1:
                    return f" (robust — {cp_ev.robustness}/{cp_ev.robustness_total} \u03bb values)"
                elif cp_ev.robustness <= 2:
                    return (
                        f" (uncertain — {cp_ev.robustness}/{cp_ev.robustness_total} \u03bb values, sensitive to prior)"
                    )
                else:
                    return f" ({cp_ev.robustness}/{cp_ev.robustness_total} \u03bb values)"

            if len(cps) == 1:
                cp = cps[0]
                rob_lbl = _robustness_label(cp)
                if cp.confirmation_obs >= 0 and cp.confirmation_obs > cp.obs:
                    segments.append(
                        f"Shift first detected at observation {cp.obs} "
                        f"(P = {cp.shift_prob:.0%}), confirmed to "
                        f"P \u2265 95% by obs ~{cp.confirmation_obs}"
                        f"{rob_lbl}."
                    )
                else:
                    segments.append(f"Process shifted at observation {cp.obs} (P = {cp.shift_prob:.0%}){rob_lbl}.")
            else:
                cp_descs = []
                for c in cps:
                    rob_lbl = _robustness_label(c)
                    confirm_str = ""
                    if c.confirmation_obs >= 0 and c.confirmation_obs > c.obs:
                        confirm_str = f", confirmed obs ~{c.confirmation_obs}"
                    cp_descs.append(f"obs {c.obs} (P={c.shift_prob:.0%}{confirm_str}){rob_lbl}")
                segments.append(
                    f"Process shifted at {', '.join(cp_descs[:-1])} "
                    f"and {cp_descs[-1]}. "
                    f"{len(regimes)} regimes detected."
                )

            # Per-regime capability
            for r in regimes:
                cpk_r = r.cpk
                if cpk_r:
                    ci = cpk_r.cpk_credible_interval
                    regime_desc = (
                        f"{r.label} (obs {r.start + 1}\u2013{r.end}, n={r.n}): "
                        f"Cpk {cpk_r.cpk_point_estimate:.2f} "
                        f"[{ci[0]:.2f}, {ci[1]:.2f}],"
                        f" P(>1.33)={cpk_r.cpk_probability_above_133:.0%}"
                    )
                    if r.taguchi:
                        tl = r.taguchi
                        regime_desc += (
                            f", E[L]=${tl.expected_loss:.3f}/unit"
                            f" (bias: {tl.bias_fraction:.0%},"
                            f" var: {tl.variance_fraction:.0%})"
                        )
                    if r.n < 30:
                        regime_desc += (
                            f" \u2014 posterior wide, confidence limited. "
                            f"{30} additional observations would narrow CI "
                            f"by approximately {r.ci_narrowing_30:.0f}%"
                        )
                    segments.append(regime_desc + ".")
                else:
                    segments.append(
                        f"{r.label} (obs {r.start + 1}\u2013{r.end}, n={r.n}): \u03bc={r.mean:.4f}, \u03c3={r.std:.4f}."
                    )

            # Mean displacement direction
            if len(regimes) >= 2:
                r_prev, r_curr = regimes[-2], regimes[-1]
                delta = r_curr.mean - r_prev.mean
                direction = ""
                if USL is not None and delta > 0:
                    direction = " toward USL"
                elif LSL is not None and delta < 0:
                    direction = " toward LSL"
                segments.append(f"Mean displaced {delta:+.4f}{direction}.")

            # Evidence corroboration
            last_ev = cps[-1].e_value
            level = cps[-1].evidence_level
            if last_ev > 1:
                segments.append(f"Evidence strength: {last_ev:.0f}:1 ({level}).")
            if timeline.ed_peak_obs >= 0:
                if timeline.ed_peak_log_N >= timeline.ed_threshold:
                    segments.append(
                        f"E-Detector corroboration: alarm at "
                        f"obs {timeline.ed_peak_obs} "
                        f"(log(N) = {timeline.ed_peak_log_N:.1f})."
                    )
                else:
                    segments.append(
                        f"E-Detector: no independent alarm "
                        f"(peak log(N) = {timeline.ed_peak_log_N:.1f}, "
                        f"below threshold {timeline.ed_threshold:.1f}). "
                        f"Shift real but small."
                    )

        elif timeline and timeline.lot_capabilities:
            # No BOCPD changepoints but lot transitions present
            n_lots = len(timeline.lot_capabilities)
            segments.append(f"{n_lots} material lots detected. No unexpected within-lot shifts found by BOCPD.")
            # Per-lot capability summary
            for lc in timeline.lot_capabilities:
                cpk_r = lc.cpk
                if cpk_r:
                    ci = cpk_r.cpk_credible_interval
                    lot_desc = (
                        f"{lc.lot_id} (n={lc.n}): "
                        f"Cpk {cpk_r.cpk_point_estimate:.2f} "
                        f"[{ci[0]:.2f}, {ci[1]:.2f}],"
                        f" P(>1.33)={cpk_r.cpk_probability_above_133:.0%}"
                    )
                    if lc.taguchi:
                        tl = lc.taguchi
                        lot_desc += (
                            f", E[L]=${tl.expected_loss:.3f}/unit"
                            f" (bias: {tl.bias_fraction:.0%},"
                            f" var: {tl.variance_fraction:.0%})"
                        )
                    segments.append(lot_desc + ".")
                    # Within-lot shift: show pre/post loss if Taguchi is active
                    if lc.within_lot_cps and lc.taguchi and taguchi_k > 0 and taguchi_target is not None:
                        for wcp in lc.within_lot_cps:
                            pre_data = y[lc.start : wcp]
                            post_data = y[wcp : lc.end]
                            if len(pre_data) >= 2 and len(post_data) >= 2:
                                pre_post = prior.copy()
                                pre_post.update(pre_data)
                                post_post = prior.copy()
                                post_post.update(post_data)
                                pre_loss = compute_taguchi_loss(pre_post, taguchi_target, taguchi_k)
                                post_loss = compute_taguchi_loss(post_post, taguchi_target, taguchi_k)
                                segments.append(
                                    f"  \u26a0 Within-lot shift at obs {wcp}"
                                    f" \u2014 pre-shift ${pre_loss.expected_loss:.3f},"
                                    f" post-shift ${post_loss.expected_loss:.3f}/unit"
                                )

        elif belief_pt:
            # No shift — original behavior
            sp = belief_pt.shift_probability
            if sp < 0.20:
                segments.append(f"Process is stable. Shift probability: {sp:.0%}.")
            elif sp < 0.50:
                segments.append(f"Process shows early signs of change. Shift probability: {sp:.0%}.")
            elif sp < 0.80:
                segments.append(
                    f"Process is likely shifting. "
                    f"P(shift): {sp:.0%}. "
                    f"Estimated new mean: {belief_pt.current_regime_mean:.3f} "
                    f"(was {belief_pt.reference_mean:.3f})."
                )
            else:
                mag = abs(belief_pt.current_regime_mean - belief_pt.reference_mean)
                segments.append(
                    f"Process has shifted. P(shift): {sp:.0%}. "
                    f"New mean: {belief_pt.current_regime_mean:.3f} "
                    f"(ref: {belief_pt.reference_mean:.3f}). "
                    f"Magnitude: {mag:.3f}."
                )

        # 2. Measurement confidence
        if uncertainty_pt and uncertainty_pt.gage_health_status == "attention":
            segments.append(
                f"Measurement system degraded "
                f"({uncertainty_pt.gage_health_pct:.0f}% GRR). "
                f"True-value uncertainty: +/-{uncertainty_pt.fused_std:.3f}."
            )

        # 3. Evidence (only if not already covered by timeline)
        if not has_shifts and evidence_pt:
            ev = evidence_pt.e_value_accumulated
            if ev > 20:
                segments.append(f"Strong evidence against in-control ({ev:.0f}:1).")
            elif ev > 5:
                segments.append(f"Moderate evidence of change ({ev:.0f}:1).")

        # 4. Trajectory
        if predictive and predictive.prob_exceed_spec_10 > 0.10:
            segments.append(
                f"Trajectory: {predictive.prob_exceed_spec_10:.0%} probability of spec exceedance within 10 obs."
            )

        # 4b. Cross-component coherence check
        belief_stable = belief_pt and belief_pt.shift_probability < 0.20
        pred_risky = predictive and predictive.prob_exceed_spec_10 > 0.30
        ev_risky = evidence_pt and evidence_pt.e_value_accumulated > 20
        if belief_stable and (pred_risky or ev_risky):
            segments.append(
                "Caution: Predictive model and/or evidence accumulation "
                "indicate risk that the shift detector has not confirmed. "
                "Investigate the trend."
            )

        # 5. Capability (only if not already covered by timeline)
        if not has_shifts and cpk:
            segments.append(
                f"Bayesian Cpk: {cpk.cpk_point_estimate:.2f} "
                f"[{cpk.cpk_credible_interval[0]:.2f}, "
                f"{cpk.cpk_credible_interval[1]:.2f}]. "
                f"P(Cpk>1.33): {cpk.cpk_probability_above_133:.0%}."
            )

        # 6. Health
        if health:
            segments.append(f"Overall health: {health.overall_health:.0%}. Primary factor: {health.primary_driver}.")

        # 7. Robustness report (§9.2)
        if beta_robustness > 0:
            w_3sigma = math.exp(-beta_robustness * 9.0 / 2.0)
            segments.append(f"Robustness: beta = {beta_robustness:.2f} (outlier weight at 3 sigma: {w_3sigma:.1%}).")
            if belief_points:
                dw = [p for p in belief_points if p.observation_weight < 0.5]
                if dw:
                    obs_list = ", ".join(str(p.t) for p in dw[:5])
                    segments.append(
                        f"{len(dw)} obs downweighted as potential outliers "
                        f"(obs {obs_list}"
                        f"{', ...' if len(dw) > 5 else ''})."
                    )

        return " ".join(segments) if segments else "Insufficient data."


# ═══════════════════════════════════════════════════════════════════════════
# 11. PROBABILISTIC ALARMS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AlarmDecision:
    recommend_action: str  # 'continue' | 'investigate'
    shift_probability: float
    threshold: float
    expected_cost_ignore: float
    expected_cost_investigate: float
    cost_parameters: dict[str, float]


class ProbabilisticAlarms:
    """Decision-theoretic alarm framework (§11)."""

    def __init__(
        self,
        c_miss: float = 10.0,
        c_false_alarm: float = 1.0,
        c_investigate: float = 2.0,
    ):
        self.c_miss = c_miss
        self.c_fa = c_false_alarm
        self.c_inv = c_investigate

    @property
    def threshold(self) -> float:
        """P(shifted) threshold for recommending investigation."""
        return (self.c_fa + self.c_inv) / (self.c_miss + self.c_fa)

    def decide(self, shift_prob: float) -> AlarmDecision:
        ec_ignore = shift_prob * self.c_miss
        ec_investigate = (1 - shift_prob) * self.c_fa + self.c_inv

        if shift_prob > self.threshold:
            action = "investigate"
        else:
            action = "continue"

        return AlarmDecision(
            recommend_action=action,
            shift_probability=shift_prob,
            threshold=self.threshold,
            expected_cost_ignore=ec_ignore,
            expected_cost_investigate=ec_investigate,
            cost_parameters={
                "c_miss": self.c_miss,
                "c_false_alarm": self.c_fa,
                "c_investigate": self.c_inv,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# 12. CHART GENEALOGY
# ═══════════════════════════════════════════════════════════════════════════


class ChartGenealogy:
    """Prior inheritance from parent processes."""

    @staticmethod
    def inherit_prior(parent: NormalGammaPosterior, transfer_factor: float) -> NormalGammaPosterior:
        """Discount parent posterior for child's prior (§12.3)."""
        return parent.discount(transfer_factor)

    @staticmethod
    def multi_parent_prior(
        parents: list[tuple[NormalGammaPosterior, float, float]],
    ) -> NormalGammaPosterior:
        """Combine multiple parent posteriors (§12.4).
        parents: [(posterior, transfer_factor, relevance_weight), ...]
        """
        total_w = sum(w for _, _, w in parents)
        mu_0 = sum(w * p.mu for p, _, w in parents) / total_w
        kappa_0 = sum(w * tf * p.kappa for p, tf, w in parents) / total_w
        alpha_0 = max(sum(w * tf * p.alpha for p, tf, w in parents) / total_w, 0.5)
        beta_0 = sum(w * tf * p.beta for p, tf, w in parents) / total_w

        return NormalGammaPosterior(mu=mu_0, kappa=kappa_0, alpha=alpha_0, beta=beta_0)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def _logsumexp(log_arr: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    if len(log_arr) == 0:
        return -np.inf
    m = np.max(log_arr)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(log_arr - m))))


# ═══════════════════════════════════════════════════════════════════════════
# DSW INTEGRATION — run_pbs() dispatcher
# ═══════════════════════════════════════════════════════════════════════════


def run_pbs(df, analysis_id, config):
    """
    Dispatcher for Process Belief System analyses.

    analysis_id:
        'pbs_full'        — Full PBS analysis (all charts)
        'pbs_belief'      — Belief Chart only (shift probability)
        'pbs_edetector'   — E-Detector (distribution-free changepoint)
        'pbs_evidence'    — Evidence Accumulation only
        'pbs_predictive'  — Predictive Chart only
        'pbs_adaptive'    — Adaptive Control Limits only
        'pbs_cpk'         — Bayesian Cpk only
        'pbs_cpk_traj'    — Cpk Trajectory
        'pbs_health'      — Multi-Stream Health
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    col = config.get("column", "")
    if not col or col not in df.columns:
        result["summary"] = "Error: Select a measurement column."
        return result

    # Coerce non-numeric values (empty strings, text) to NaN before filtering
    import pandas as pd

    df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_mask = df[col].notna()
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    y = df_valid[col].values.astype(float)
    n = len(y)
    if n < 5:
        result["summary"] = "Error: Need at least 5 observations."
        return result

    def _safe_float(val):
        """Convert to float, returning None for empty/invalid values."""
        if val is None or val == "" or val == "null":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    USL = _safe_float(config.get("USL"))
    LSL = _safe_float(config.get("LSL"))
    target = _safe_float(config.get("target"))

    hazard_lambda_raw = config.get("hazard_lambda")
    hazard_lambda_val = _safe_float(hazard_lambda_raw)
    if hazard_lambda_val is not None and hazard_lambda_raw != "auto":
        hazard_lambda = hazard_lambda_val
    else:
        hazard_lambda = max(20.0, min(200.0, n / 4.0))
    beta_robustness = _safe_float(config.get("beta_robustness")) or 0.0

    # Extract metadata columns (lot, operator, machine) aligned to valid rows
    metadata = {}
    for mc in ["material_lot", "operator", "machine"]:
        if mc in df_valid.columns:
            vals = df_valid[mc].tolist()
            if len(set(vals)) > 1:  # only include if transitions exist
                metadata[mc] = vals

    # Calibration phase — first observations set informative prior
    n_cal = min(50, max(10, n // 5))
    cal = y[:n_cal]
    sigma_cal = float(np.std(cal, ddof=1)) if n_cal > 1 else 0.01

    # Build prior — center on target/spec/data
    if target is not None:
        mu_0 = target
    elif USL is not None and LSL is not None:
        mu_0 = (USL + LSL) / 2.0
    else:
        mu_0 = float(np.mean(cal))

    # Informative prior: kappa=1 (1 pseudo-obs for mean), alpha=2,
    # beta matched to calibration variance.  Predictive scale ≈ sigma_cal.
    prior = NormalGammaPosterior(mu=mu_0, kappa=1.0, alpha=2.0, beta=max(sigma_cal**2 * 2.0, 1e-10))

    if analysis_id == "pbs_full":
        return _run_full_pbs(
            y,
            prior,
            USL,
            LSL,
            target,
            hazard_lambda,
            config,
            sigma_cal,
            beta_robustness,
            metadata,
        )
    elif analysis_id == "pbs_belief":
        return _run_belief_only(y, prior, hazard_lambda, config, beta_robustness)
    elif analysis_id == "pbs_edetector":
        return _run_edetector_only(y, mu_0, USL, LSL, sigma_cal, config)
    elif analysis_id == "pbs_evidence":
        return _run_evidence_only(y, prior, mu_0, config, sigma_cal)
    elif analysis_id == "pbs_predictive":
        return _run_predictive_only(y, USL, LSL, config)
    elif analysis_id == "pbs_adaptive":
        return _run_adaptive_only(y, prior, config)
    elif analysis_id == "pbs_cpk":
        return _run_cpk_only(y, prior, USL, LSL, config)
    elif analysis_id == "pbs_cpk_traj":
        return _run_cpk_traj_only(y, prior, USL, LSL, config)
    elif analysis_id == "pbs_health":
        return _run_health_only(y, prior, USL, LSL, mu_0, hazard_lambda, config, beta_robustness)
    else:
        result["summary"] = f"Error: Unknown PBS analysis: {analysis_id}"
        return result


def _run_full_pbs(
    y,
    prior,
    USL,
    LSL,
    target,
    hazard_lambda,
    config,
    sigma_ref,
    beta_robustness=0.0,
    metadata=None,
):
    """Full PBS analysis — all charts."""
    result = {"plots": [], "summary": "", "guide_observation": ""}
    n = len(y)

    # 1. Run BOCPD — empirical Bayes over λ grid
    #    Put a uniform prior on λ and select MAP segmentation.
    #    Flag changepoints that appear across multiple λ as robust.
    lambda_grid = [20, 50, 100, 200, 500]
    bc_runs = {}
    for lam in lambda_grid:
        bc_i = BeliefChart(hazard_lambda=lam, prior=prior.copy(), beta_robustness=beta_robustness)
        for x in y:
            bc_i.process(x)
        bc_runs[lam] = bc_i

    # Select MAP λ (highest marginal log-likelihood)
    best_lam = max(lambda_grid, key=lambda lam: bc_runs[lam].log_marginal_likelihood)
    bc = bc_runs[best_lam]
    hazard_lambda = best_lam  # update for downstream summary

    # Robustness: extract changepoints per λ for cross-validation
    def _extract_cps(bc_obj, min_gap=5):
        """Detect changepoints via MAP run length drops.
        When the most likely run length drops to ≤2 from a value ≥5,
        a new regime has started at that observation.
        """
        cps = []
        last = -min_gap
        pts = bc_obj.points
        for t in range(2, n):
            prev_rl = pts[t - 1].most_likely_run_length
            curr_rl = pts[t].most_likely_run_length
            if prev_rl >= 5 and curr_rl <= 2 and t - last >= min_gap:
                cps.append(t)
                last = t
        return cps

    all_cps_by_lambda = {lam: _extract_cps(bc_runs[lam]) for lam in lambda_grid}
    # Count how many λ values detect each changepoint (within ±3 obs)
    cp_robustness = {}  # obs -> count of λ values that see it
    for cp in all_cps_by_lambda[best_lam]:
        count = 0
        for lam in lambda_grid:
            if any(abs(c - cp) <= 3 for c in all_cps_by_lambda[lam]):
                count += 1
        cp_robustness[cp] = count

    # 1b. Regime merging — collapse adjacent BOCPD regimes with similar means
    def _merge_similar_regimes(cps_in, data, merge_threshold=1.0, protected=None):
        """Remove CPs where adjacent regimes have similar means.
        CPs in `protected` (within ±3 obs) are never merged.
        """
        if not cps_in:
            return []
        keep = []
        bounds = [0] + list(cps_in) + [len(data)]
        for j, cp in enumerate(cps_in):
            # Never merge CPs near known transitions
            if protected and any(abs(cp - p) <= 3 for p in protected):
                keep.append(cp)
                continue
            left = data[bounds[j] : cp]
            right = data[cp : bounds[j + 2]]
            if len(left) < 2 or len(right) < 2:
                keep.append(cp)
                continue
            n_l, n_r = len(left), len(right)
            var_l = float(np.var(left, ddof=1))
            var_r = float(np.var(right, ddof=1))
            pooled_std = math.sqrt((var_l * (n_l - 1) + var_r * (n_r - 1)) / (n_l + n_r - 2))
            if pooled_std <= 0:
                keep.append(cp)
                continue
            if abs(float(np.mean(left)) - float(np.mean(right))) >= merge_threshold * pooled_std:
                keep.append(cp)
            # else: means too similar — drop this CP (merge regimes)
        return keep

    # 1c. Detect known transitions from metadata columns (before merge,
    #     so we can protect lot-boundary CPs from being merged away)
    known_transitions = []
    if metadata:
        for col_name, values in metadata.items():
            for i in range(1, n):
                if values[i] != values[i - 1]:
                    known_transitions.append(
                        KnownTransition(
                            obs=i,
                            column=col_name,
                            from_value=str(values[i - 1]),
                            to_value=str(values[i]),
                        )
                    )
    lot_transitions = [kt for kt in known_transitions if kt.column == "material_lot"]

    map_cps = list(all_cps_by_lambda[best_lam])
    # Protect CPs that are near known lot transitions from being merged
    protected_obs = [kt.obs for kt in lot_transitions]
    map_cps = _merge_similar_regimes(map_cps, y, merge_threshold=1.0, protected=protected_obs)

    # 2. Cumulative posterior
    post = prior.copy()
    post.update(y)

    # 3. Evidence accumulation (fixed reference from calibration)
    mu_0 = prior.mu
    ea = EvidenceAccumulation(mu_0=mu_0, sigma_ref=sigma_ref)
    for x in y:
        ea.process(x)

    # 3b. E-Detector (distribution-free companion to Belief Chart)
    if USL is not None and LSL is not None:
        ed_bounds = (float(LSL), float(USL))
    else:
        ed_bounds = (
            mu_0 - 4.0 * max(sigma_ref, 1e-6),
            mu_0 + 4.0 * max(sigma_ref, 1e-6),
        )
    ed_alpha = float(config.get("edetector_alpha", 0.05))
    ed = EDetector(mu_0=mu_0, bounds=ed_bounds, alpha=ed_alpha)
    for x in y:
        ed.process(x)

    # 4. Adaptive control limits
    acl = AdaptiveControlLimits()
    acl_post = prior.copy()
    acl_points = []
    for t, x in enumerate(y):
        acl_post.update_single(x)
        acl_points.append(acl.compute_limits(acl_post, t, x))

    # 5. Predictive chart
    pred = None
    if n >= 10:
        pc = PredictiveChart(window=min(20, n))
        pred = pc.compute(y, USL=USL, LSL=LSL, horizon=min(25, n))

    # 6. Bayesian Cpk
    cpk_result = None
    if USL is not None and LSL is not None:
        cpk_engine = BayesianCpk(USL, LSL)
        cpk_result = cpk_engine.compute(post, n)

    # 7. Health — fuse BOCPD, E-Detector, Cpk, trend (Anti-Pattern 7)
    h_spc = 1 - bc.points[-1].shift_probability if bc.points else 0.5
    h_cpk_raw = cpk_result.cpk_probability_above_133 if cpk_result else 0.5
    cpk_n_eff = n
    cpk_maturity = min(1.0, cpk_n_eff / 30.0)
    h_cpk = h_cpk_raw * cpk_maturity
    h_trend = 0.5
    if pred:
        h_trend = 1.0 - pred.prob_exceed_spec_10
    # E-Detector health: 1.0 if monitoring, 0.0 if alarm,
    # linear scale between threshold/2 and threshold
    ed_last = ed.points[-1] if ed.points else None
    if ed_last and ed_last.alarm:
        h_edet = 0.0
    elif ed_last:
        h_edet = max(0.0, 1.0 - ed_last.log_N_combined / ed.threshold)
    else:
        h_edet = 0.5

    streams = {"spc": h_spc, "edetector": h_edet, "cpk": h_cpk, "trend": h_trend}
    msh = MultiStreamHealth()
    health = msh.fuse(streams)

    # 7b. Investigation Timeline — synchronized changepoint data
    cp_indices = map_cps  # post-merge BOCPD changepoints

    # E-Detector peak (does NOT auto-reset)
    ed_peak_idx = max(range(n), key=lambda i: ed.points[i].log_N_combined) if n > 0 else 0
    ed_peak_val = ed.points[ed_peak_idx].log_N_combined if ed.points else 0.0
    ed_first_alarm = next((p.t - 1 for p in ed.points if p.alarm), -1)

    # Taguchi loss config — extracted once, used in both regime and lot loops
    # Default k from spec limits: k = 1/Δ₀² where Δ₀ = (USL-LSL)/2
    # Loss = 1.0 means "as bad as sitting on the spec limit"
    taguchi_k_raw = config.get("taguchi_k")
    if taguchi_k_raw not in (None, "", 0, "0"):
        taguchi_k = float(taguchi_k_raw)
    elif USL is not None and LSL is not None:
        delta0 = (USL - LSL) / 2.0
        taguchi_k = 1.0 / (delta0**2) if delta0 > 0 else 0.0
    else:
        taguchi_k = 0.0
    taguchi_target = target
    if taguchi_target is None and USL is not None and LSL is not None:
        taguchi_target = (USL + LSL) / 2.0

    # Per-regime stats + Cpk (BOCPD regimes)
    boundaries = [0] + cp_indices + [n]
    regimes = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        seg_data = y[seg_start:seg_end]
        seg_n = len(seg_data)
        seg_mean = float(np.mean(seg_data)) if seg_n > 0 else 0.0
        seg_std = float(np.std(seg_data, ddof=1)) if seg_n > 1 else 0.0
        seg_cpk = None
        seg_taguchi = None
        seg_post = None
        if seg_n >= 2:
            seg_post = prior.copy()
            seg_post.update(seg_data)
            if USL is not None and LSL is not None:
                seg_cpk = BayesianCpk(USL, LSL).compute(seg_post, seg_n, seed=42 + i)
        ci_narrow = 0.0
        if seg_n >= 2:
            seg_kappa = prior.kappa + seg_n
            ci_narrow = (1.0 - math.sqrt(seg_kappa / (seg_kappa + 30))) * 100
        if seg_post is not None and taguchi_k > 0 and taguchi_target is not None:
            seg_taguchi = compute_taguchi_loss(seg_post, taguchi_target, taguchi_k)
        regimes.append(
            RegimeStats(
                label=f"Regime {i + 1}",
                start=seg_start,
                end=seg_end,
                n=seg_n,
                mean=seg_mean,
                std=seg_std,
                cpk=seg_cpk,
                ci_narrowing_30=ci_narrow,
                taguchi=seg_taguchi,
            )
        )

    # Build CP events with per-CP confirmation and near-known-transition flag
    n_grid = len(lambda_grid)
    cp_events = []
    for cp in cp_indices:
        ev_at = ea.points[cp].e_value_accumulated if cp < len(ea.points) else 1.0
        ev_level = ea.points[cp].evidence_level if cp < len(ea.points) else "none"
        sp_at = bc.points[cp].shift_probability
        rob = cp_robustness.get(cp, 1)
        # Per-CP confirmation: scan forward from this CP
        cp_confirm = -1
        for t in range(cp, n):
            if bc.points[t].shift_probability >= 0.95:
                cp_confirm = t
                break
        # Near a known lot transition?
        near_kt = any(abs(cp - kt.obs) <= 3 for kt in lot_transitions)
        cp_events.append(
            ChangePointEvent(
                obs=cp,
                shift_prob=sp_at,
                e_value=ev_at,
                evidence_level=ev_level,
                robustness=rob,
                robustness_total=n_grid,
                confirmation_obs=cp_confirm,
                near_known_transition=near_kt,
            )
        )

    # Per-lot capability (metadata-driven segmentation)
    lot_capabilities = []
    if lot_transitions:
        lot_bounds = [0] + [kt.obs for kt in lot_transitions] + [n]
        lot_ids = []
        if metadata and "material_lot" in metadata:
            for i in range(len(lot_bounds) - 1):
                lot_ids.append(metadata["material_lot"][lot_bounds[i]])
        for i in range(len(lot_bounds) - 1):
            seg_start, seg_end = lot_bounds[i], lot_bounds[i + 1]
            seg_data = y[seg_start:seg_end]
            seg_n = len(seg_data)
            seg_mean = float(np.mean(seg_data)) if seg_n > 0 else 0.0
            seg_std = float(np.std(seg_data, ddof=1)) if seg_n > 1 else 0.0
            seg_cpk = None
            seg_taguchi = None
            seg_post = None
            if seg_n >= 2:
                seg_post = prior.copy()
                seg_post.update(seg_data)
                if USL is not None and LSL is not None:
                    seg_cpk = BayesianCpk(USL, LSL).compute(seg_post, seg_n, seed=200 + i)
            ci_narrow = 0.0
            if seg_n >= 2:
                seg_kappa = prior.kappa + seg_n
                ci_narrow = (1.0 - math.sqrt(seg_kappa / (seg_kappa + 30))) * 100
            # Taguchi loss — same posterior, different projection
            if seg_post is not None and taguchi_k > 0 and taguchi_target is not None:
                seg_taguchi = compute_taguchi_loss(seg_post, taguchi_target, taguchi_k)
            # Within-lot BOCPD CPs (not at a lot boundary)
            within_cps = [
                cp
                for cp in cp_indices
                if seg_start < cp < seg_end and not any(abs(cp - kt.obs) <= 3 for kt in lot_transitions)
            ]
            lot_capabilities.append(
                LotCapability(
                    lot_id=lot_ids[i] if lot_ids else f"Segment {i + 1}",
                    start=seg_start,
                    end=seg_end,
                    n=seg_n,
                    mean=seg_mean,
                    std=seg_std,
                    cpk=seg_cpk,
                    ci_narrowing_30=ci_narrow,
                    within_lot_cps=within_cps,
                    taguchi=seg_taguchi,
                )
            )

    lambda_log_evs = {lam: bc_runs[lam].log_marginal_likelihood for lam in lambda_grid}
    timeline = InvestigationTimeline(
        changepoints=cp_events,
        regimes=regimes,
        ed_peak_obs=ed_peak_idx,
        ed_peak_log_N=ed_peak_val,
        ed_first_alarm_obs=ed_first_alarm,
        ed_threshold=ed.threshold,
        best_lambda=float(best_lam),
        lambda_log_evidences=lambda_log_evs,
        known_transitions=known_transitions,
        lot_capabilities=lot_capabilities if lot_capabilities else None,
    )

    # 7c. Re-scope Predictive + Cpk to current regime when shift detected
    #     Use the later of: last BOCPD CP or last known lot transition
    last_anchor = 0
    if cp_indices:
        last_anchor = cp_indices[-1]
    if lot_transitions:
        last_anchor = max(last_anchor, lot_transitions[-1].obs)
    if last_anchor > 0:
        current_regime_data = y[last_anchor:]
        # Predictive: anchor from post-shift data only
        if len(current_regime_data) >= 10:
            pc_r = PredictiveChart(window=min(20, len(current_regime_data)))
            pred = pc_r.compute(
                current_regime_data,
                USL=USL,
                LSL=LSL,
                horizon=min(25, len(current_regime_data)),
            )
        else:
            pred = None
        # Cpk: posterior from current regime only
        if USL is not None and LSL is not None and len(current_regime_data) >= 2:
            post_current = prior.copy()
            post_current.update(current_regime_data)
            cpk_result = BayesianCpk(USL, LSL).compute(post_current, len(current_regime_data))
            # Update health Cpk component — discount by posterior maturity
            h_cpk_raw = cpk_result.cpk_probability_above_133
            cpk_n_eff = len(current_regime_data)
            cpk_maturity = min(1.0, cpk_n_eff / 30.0)
            h_cpk = h_cpk_raw * cpk_maturity
            streams["cpk"] = h_cpk
            health = msh.fuse(streams)

    # 8. Alarm
    alarm_engine = ProbabilisticAlarms(
        c_miss=float(config.get("c_miss", 10)),
        c_false_alarm=float(config.get("c_fa", 1)),
        c_investigate=float(config.get("c_inv", 2)),
    )
    alarm_dec = alarm_engine.decide(bc.points[-1].shift_probability if bc.points else 0.0)

    # 9. Narrative
    narrative = ProcessNarrative.generate(
        belief_pt=bc.points[-1] if bc.points else None,
        evidence_pt=ea.points[-1] if ea.points else None,
        predictive=pred,
        cpk=cpk_result,
        health=health,
        beta_robustness=beta_robustness,
        belief_points=bc.points,
        timeline=timeline,
        USL=USL,
        LSL=LSL,
        taguchi_k=taguchi_k,
        taguchi_target=taguchi_target,
        y=y,
        prior=prior,
    )

    # ── Build plots ──
    ts = list(range(n))

    # Synchronized changepoint shapes + annotations for all obs-axis plots
    # Red dashed = BOCPD-detected shift.  Gold dotted = known metadata transition.
    cp_shapes = []
    cp_annots = []
    for cpe in timeline.changepoints:
        cp_shapes.append(
            {
                "type": "line",
                "x0": cpe.obs,
                "x1": cpe.obs,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "width": 1.5, "dash": "dash"},
            }
        )
        cp_annots.append(
            {
                "x": cpe.obs,
                "y": 1.02,
                "yref": "paper",
                "text": f"CP {cpe.obs}",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 10},
                "xanchor": "center",
                "yanchor": "bottom",
            }
        )

    # Known transition lines (gold dotted)
    kt_shapes = []
    kt_annots = []
    for kt in lot_transitions:
        kt_shapes.append(
            {
                "type": "line",
                "x0": kt.obs,
                "x1": kt.obs,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#c9a227", "width": 1, "dash": "dot"},
            }
        )
        # Short label: strip common prefix for readability
        short_label = kt.to_value
        if short_label.startswith("LOT-2026-"):
            short_label = "L" + short_label[-3:]
        elif len(short_label) > 10:
            short_label = short_label[-6:]
        kt_annots.append(
            {
                "x": kt.obs,
                "y": 1.06,
                "yref": "paper",
                "text": short_label,
                "showarrow": False,
                "font": {"color": "#c9a227", "size": 9},
                "xanchor": "center",
                "yanchor": "bottom",
            }
        )

    def _with_cp(layout):
        """Merge changepoint + known transition shapes into a plot layout."""
        shapes = layout.get("shapes", [])
        annots = layout.get("annotations", [])
        if cp_shapes:
            shapes = shapes + cp_shapes
            annots = annots + cp_annots
        if kt_shapes:
            shapes = shapes + kt_shapes
            annots = annots + kt_annots
        layout["shapes"] = shapes
        layout["annotations"] = annots
        return layout

    # Plot 1: Belief Chart (shift probability)
    shift_probs = [p.shift_probability for p in bc.points]
    result["plots"].append(
        {
            "title": "Belief Chart — P(Process Shifted)",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": shift_probs,
                    "mode": "lines",
                    "name": "P(shift)",
                    "line": {"color": "#d94a4a", "width": 2},
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [0.5, 0.5],
                    "mode": "lines",
                    "name": "Watch (50%)",
                    "line": {"color": "#d4a24a", "dash": "dot", "width": 1},
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [0.95, 0.95],
                    "mode": "lines",
                    "name": "Alarm (95%)",
                    "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                },
            ],
            "layout": _with_cp(
                {
                    "template": "plotly_dark",
                    "height": 250,
                    "yaxis": {"title": "P(shifted)", "range": [0, 1.05]},
                    "xaxis": {"title": "Observation"},
                }
            ),
            "group": "Belief",
        }
    )

    # Plot 2: Evidence Accumulation (log scale)
    log_es = [p.log_e_accumulated for p in ea.points]
    result["plots"].append(
        {
            "title": "Evidence Accumulation — E-value (log scale)",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": log_es,
                    "mode": "lines",
                    "name": "log(E)",
                    "line": {"color": "#6ab7d4", "width": 2},
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [math.log(20), math.log(20)],
                    "mode": "lines",
                    "name": "Strong (20:1)",
                    "line": {"color": "#4a9f6e", "dash": "dash", "width": 1},
                },
            ],
            "layout": _with_cp(
                {
                    "template": "plotly_dark",
                    "height": 250,
                    "yaxis": {"title": "log(E-value)"},
                    "xaxis": {"title": "Observation"},
                }
            ),
            "group": "Belief",
        }
    )

    # Plot 2b: E-Detector (distribution-free changepoint)
    ed_log_Ns = [p.log_N_combined for p in ed.points]
    ed_traces = [
        {
            "type": "scatter",
            "x": ts,
            "y": ed_log_Ns,
            "mode": "lines",
            "name": "log(N)",
            "line": {"color": "#4a9f6e", "width": 2},
        },
        {
            "type": "scatter",
            "x": [0, n - 1],
            "y": [ed.threshold, ed.threshold],
            "mode": "lines",
            "name": f"1/alpha = {1 / ed_alpha:.0f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
        },
        {
            "type": "scatter",
            "x": [0, n - 1],
            "y": [0, 0],
            "mode": "lines",
            "name": "Reset",
            "line": {"color": "#666", "dash": "dot", "width": 1},
        },
    ]
    ed_alarm_pts = [p for p in ed.points if p.alarm]
    if ed_alarm_pts:
        ed_traces.append(
            {
                "type": "scatter",
                "x": [p.t - 1 for p in ed_alarm_pts],
                "y": [p.log_N_combined for p in ed_alarm_pts],
                "mode": "markers",
                "name": "Alarm",
                "marker": {"color": "#d94a4a", "symbol": "diamond", "size": 8},
            }
        )
    # E-Detector peak marker
    if timeline.ed_peak_obs >= 0 and ed_peak_val > 0:
        ed_traces.append(
            {
                "type": "scatter",
                "x": [timeline.ed_peak_obs],
                "y": [ed_peak_val],
                "mode": "markers+text",
                "name": "Peak",
                "marker": {"color": "#e8c547", "symbol": "star", "size": 10},
                "text": [f"Peak log(N)={ed_peak_val:.1f}"],
                "textposition": "top center",
                "textfont": {"color": "#e8c547", "size": 10},
            }
        )
    ed_layout = _with_cp(
        {
            "template": "plotly_dark",
            "height": 250,
            "yaxis": {"title": "log(N)"},
            "xaxis": {"title": "Observation"},
        }
    )
    # First alarm vertical line (threshold crossing)
    if timeline.ed_first_alarm_obs >= 0:
        ed_layout.setdefault("shapes", []).append(
            {
                "type": "line",
                "x0": timeline.ed_first_alarm_obs,
                "x1": timeline.ed_first_alarm_obs,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e8c547", "width": 1, "dash": "dot"},
            }
        )
    result["plots"].append(
        {
            "title": "E-Detector — Distribution-Free Changepoint",
            "data": ed_traces,
            "layout": ed_layout,
            "group": "Belief",
        }
    )

    # Plot 3: Adaptive Control Limits
    obs_vals = [p.observation for p in acl_points]
    cls = [p.cl for p in acl_points]
    ucls = [p.ucl for p in acl_points]
    lcls = [p.lcl for p in acl_points]
    result["plots"].append(
        {
            "title": "Adaptive Control Limits",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": obs_vals,
                    "mode": "markers",
                    "name": "Observations",
                    "marker": {"color": "#4a9f6e", "size": 4},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": cls,
                    "mode": "lines",
                    "name": "CL",
                    "line": {"color": "#d4a24a", "width": 1.5},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": ucls,
                    "mode": "lines",
                    "name": "UCL",
                    "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": lcls,
                    "mode": "lines",
                    "name": "LCL",
                    "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                },
            ],
            "layout": _with_cp(
                {
                    "template": "plotly_dark",
                    "height": 280,
                    "yaxis": {"title": config.get("column", "Value")},
                    "xaxis": {"title": "Observation"},
                }
            ),
            "group": "Control",
        }
    )

    # Plot 4: Predictive Fan (if available)
    if pred and pred.prediction_fan:
        fan_h = [p.horizon for p in pred.prediction_fan]
        fan_mean = [p.mean for p in pred.prediction_fan]
        fan_90l = [p.ci90_lower for p in pred.prediction_fan]
        fan_90u = [p.ci90_upper for p in pred.prediction_fan]
        fan_50l = [p.ci50_lower for p in pred.prediction_fan]
        fan_50u = [p.ci50_upper for p in pred.prediction_fan]

        fan_x = [n - 1 + h for h in fan_h]
        pred_traces = [
            {
                "type": "scatter",
                "x": ts[-20:],
                "y": y[-20:].tolist(),
                "mode": "lines+markers",
                "name": "Recent data",
                "line": {"color": "#4a9f6e", "width": 1},
                "marker": {"size": 3},
            },
            {
                "type": "scatter",
                "x": fan_x,
                "y": fan_mean,
                "mode": "lines",
                "name": "Predicted mean",
                "line": {"color": "#d4a24a", "width": 2},
            },
            {
                "type": "scatter",
                "x": fan_x + fan_x[::-1],
                "y": fan_90u + fan_90l[::-1],
                "fill": "toself",
                "fillcolor": "rgba(212,162,74,0.15)",
                "line": {"color": "transparent"},
                "name": "90% CI",
            },
            {
                "type": "scatter",
                "x": fan_x + fan_x[::-1],
                "y": fan_50u + fan_50l[::-1],
                "fill": "toself",
                "fillcolor": "rgba(212,162,74,0.30)",
                "line": {"color": "transparent"},
                "name": "50% CI",
            },
        ]
        if USL is not None:
            pred_traces.append(
                {
                    "type": "scatter",
                    "x": [fan_x[0], fan_x[-1]],
                    "y": [USL, USL],
                    "mode": "lines",
                    "name": "USL",
                    "line": {"color": "#d94a4a", "dash": "dot", "width": 1},
                }
            )
        if LSL is not None:
            pred_traces.append(
                {
                    "type": "scatter",
                    "x": [fan_x[0], fan_x[-1]],
                    "y": [LSL, LSL],
                    "mode": "lines",
                    "name": "LSL",
                    "line": {"color": "#d94a4a", "dash": "dot", "width": 1},
                }
            )
        result["plots"].append(
            {
                "title": "Predictive Chart — Forward Projection",
                "data": pred_traces,
                "layout": {
                    "template": "plotly_dark",
                    "height": 280,
                    "yaxis": {"title": config.get("column", "Value")},
                    "xaxis": {"title": "Observation"},
                },
                "group": "Prediction",
            }
        )

    # Plot 5: Bayesian Cpk posterior (if available)
    if cpk_result:
        # Use current-regime posterior when shift or lot transition detected
        cpk_post = post
        if last_anchor > 0:
            cpk_post = prior.copy()
            cpk_post.update(y[last_anchor:])
            n - last_anchor
        rng = np.random.RandomState(42)
        tau_s = rng.gamma(cpk_post.alpha, 1.0 / cpk_post.beta, size=10000)
        sig_mu = 1.0 / np.sqrt(cpk_post.kappa * tau_s)
        mu_s = rng.normal(cpk_post.mu, sig_mu)
        sigma_s = 1.0 / np.sqrt(tau_s)
        cpu_s = (USL - mu_s) / (3 * sigma_s)
        cpl_s = (mu_s - LSL) / (3 * sigma_s)
        cpk_s = np.minimum(cpu_s, cpl_s)
        cpk_s = cpk_s[cpk_s > -2]  # trim extreme negatives

        hist_v, bin_e = np.histogram(cpk_s, bins=50, density=True)
        bin_c = (bin_e[:-1] + bin_e[1:]) / 2

        cpk_title = "Bayesian Cpk Posterior"
        if last_anchor > 0:
            anchor_n = n - last_anchor
            # Identify current lot if available
            if lot_capabilities:
                cpk_title = (
                    f"Bayesian Cpk Posterior — {lot_capabilities[-1].lot_id} (obs {last_anchor + 1}–{n}, n={anchor_n})"
                )
            elif timeline.regimes:
                cr = timeline.regimes[-1]
                cpk_title = f"Bayesian Cpk Posterior — {cr.label} (obs {cr.start + 1}–{cr.end}, n={cr.n})"

        result["plots"].append(
            {
                "title": cpk_title,
                "data": [
                    {
                        "type": "bar",
                        "x": bin_c.tolist(),
                        "y": hist_v.tolist(),
                        "marker": {
                            "color": "rgba(74,159,110,0.5)",
                            "line": {"color": "#4a9f6e", "width": 0.5},
                        },
                        "name": "Posterior",
                    },
                    {
                        "type": "scatter",
                        "x": [1.33, 1.33],
                        "y": [0, max(hist_v) * 1.1],
                        "mode": "lines",
                        "name": "Cpk = 1.33",
                        "line": {"color": "#d4a24a", "dash": "dash", "width": 1.5},
                    },
                    {
                        "type": "scatter",
                        "x": [1.0, 1.0],
                        "y": [0, max(hist_v) * 1.1],
                        "mode": "lines",
                        "name": "Cpk = 1.0",
                        "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
                    },
                ],
                "layout": {
                    "template": "plotly_dark",
                    "height": 250,
                    "xaxis": {"title": "Cpk"},
                    "yaxis": {"title": "Posterior density"},
                },
                "group": "Capability",
            }
        )

    # Plot 5b: Taguchi Loss per Regime (stacked bar — bias vs variance)
    regimes_with_taguchi = [r for r in timeline.regimes if r.taguchi]
    if regimes_with_taguchi:
        labels = [f"{r.label}\n(obs {r.start + 1}–{r.end})" for r in regimes_with_taguchi]
        bias_vals = [r.taguchi.bias_loss for r in regimes_with_taguchi]
        var_vals = [r.taguchi.variance_loss for r in regimes_with_taguchi]
        unc_vals = [r.taguchi.uncertainty_loss for r in regimes_with_taguchi]
        # Hover text with decomposition
        hover_bias = [
            f"Bias: ${b:.4f}/unit ({r.taguchi.bias_fraction:.0%})" for b, r in zip(bias_vals, regimes_with_taguchi)
        ]
        hover_var = [
            f"Variance: ${v:.4f}/unit ({r.taguchi.variance_fraction:.0%})"
            for v, r in zip(var_vals, regimes_with_taguchi)
        ]
        result["plots"].append(
            {
                "title": f"Taguchi Loss per Regime (k={taguchi_k:.4g}, target={taguchi_target})",
                "data": [
                    {
                        "type": "bar",
                        "y": labels,
                        "x": bias_vals,
                        "orientation": "h",
                        "name": "Bias loss",
                        "marker": {"color": "rgba(217,74,74,0.7)"},
                        "text": hover_bias,
                        "hoverinfo": "text",
                    },
                    {
                        "type": "bar",
                        "y": labels,
                        "x": var_vals,
                        "orientation": "h",
                        "name": "Variance loss",
                        "marker": {"color": "rgba(74,159,180,0.7)"},
                        "text": hover_var,
                        "hoverinfo": "text",
                    },
                    {
                        "type": "bar",
                        "y": labels,
                        "x": unc_vals,
                        "orientation": "h",
                        "name": "Mean uncertainty",
                        "marker": {"color": "rgba(180,180,180,0.3)"},
                        "hoverinfo": "skip",
                    },
                ],
                "layout": {
                    "template": "plotly_dark",
                    "height": max(180, 60 * len(regimes_with_taguchi)),
                    "barmode": "stack",
                    "xaxis": {"title": "Expected loss ($/unit)"},
                    "yaxis": {"autorange": "reversed"},
                    "legend": {"orientation": "h", "y": -0.25},
                },
                "group": "Capability",
            }
        )

    # Plot 6: Health gauge
    result["plots"].append(
        {
            "title": "Process Health",
            "data": [
                {
                    "type": "indicator",
                    "mode": "gauge+number",
                    "value": health.overall_health * 100,
                    "gauge": {
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4a9f6e"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(217,74,74,0.3)"},
                            {"range": [50, 75], "color": "rgba(212,162,74,0.3)"},
                            {"range": [75, 100], "color": "rgba(74,159,110,0.3)"},
                        ],
                    },
                }
            ],
            "layout": {"template": "plotly_dark", "height": 200},
            "group": "Health",
        }
    )

    # ── Summary ──
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>PROCESS BELIEF SYSTEM<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(
        f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    "
        f"<<COLOR:highlight>>Hazard λ:<</COLOR>> {hazard_lambda:.0f} "
        f"(MAP from empirical Bayes grid)"
    )
    lines.append(f"<<COLOR:highlight>>Prior:<</COLOR>> Weakly informative (μ₀={prior.mu:.2f})")
    if USL is not None and LSL is not None:
        lines.append(f"<<COLOR:highlight>>Spec:<</COLOR>> [{LSL}, {USL}]")

    # λ grid evidence comparison
    if timeline.lambda_log_evidences:
        lines.append("\n<<COLOR:accent>>── \u03bb Selection (Empirical Bayes) ──<</COLOR>>")
        sorted_lams = sorted(timeline.lambda_log_evidences.items(), key=lambda x: x[1], reverse=True)
        best_ll = sorted_lams[0][1]
        for lam, ll in sorted_lams:
            delta = ll - best_ll
            marker = " <<COLOR:success>>◀ MAP<</COLOR>>" if lam == timeline.best_lambda else ""
            lines.append(f"  λ = {lam:<5.0f}  log p(y|λ) = {ll:>10.1f}  Δ = {delta:>7.1f}{marker}")

    lines.append("\n<<COLOR:accent>>── Narrative ──<</COLOR>>")
    lines.append(narrative)

    # Known Transitions (metadata-driven)
    if timeline.known_transitions:
        lot_kts = [kt for kt in timeline.known_transitions if kt.column == "material_lot"]
        if lot_kts:
            lines.append("\n<<COLOR:accent>>── Known Transitions ──<</COLOR>>")
            for kt in lot_kts:
                lines.append(f"  <<COLOR:highlight>>Obs {kt.obs}:<</COLOR>> Lot {kt.from_value} \u2192 {kt.to_value}")
        # Operator/machine transitions (if any)
        other_kts = [kt for kt in timeline.known_transitions if kt.column != "material_lot"]
        for kt in other_kts:
            lines.append(
                f"  <<COLOR:highlight>>Obs {kt.obs}:<</COLOR>> {kt.column}: {kt.from_value} \u2192 {kt.to_value}"
            )

    # Per-Lot Capability (metadata-driven segmentation)
    if timeline.lot_capabilities:
        lines.append("\n<<COLOR:accent>>── Per-Lot Capability ──<</COLOR>>")
        for lc in timeline.lot_capabilities:
            cpk_r = lc.cpk
            if cpk_r:
                ci = cpk_r.cpk_credible_interval
                p133 = cpk_r.cpk_probability_above_133
                if p133 >= 0.80:
                    color = "success"
                elif p133 >= 0.50:
                    color = "highlight"
                else:
                    color = "error"
                lines.append(
                    f"  <<COLOR:{color}>>{lc.lot_id} (obs {lc.start + 1}\u2013{lc.end}, "
                    f"n={lc.n}):<</COLOR>> Cpk = {cpk_r.cpk_point_estimate:.2f} "
                    f"[{ci[0]:.2f}, {ci[1]:.2f}], "
                    f"P(Cpk>1.33) = {p133:.0%}"
                )
                if lc.n < 30:
                    lines.append(
                        f"    <<COLOR:warning>>\u26a0 Posterior wide \u2014 {lc.n} "
                        f"observations. 30 more would narrow CI "
                        f"by ~{lc.ci_narrowing_30:.0f}%.<</COLOR>>"
                    )
            else:
                lines.append(
                    f"  {lc.lot_id} (obs {lc.start + 1}\u2013{lc.end}, n={lc.n}): "
                    f"\u03bc={lc.mean:.4f}, \u03c3={lc.std:.4f}"
                )
            # Within-lot BOCPD shifts
            if lc.within_lot_cps:
                for wcp in lc.within_lot_cps:
                    lines.append(
                        f"    <<COLOR:warning>>\u26a0 Within-lot shift at obs {wcp} \u2014 unknown cause<</COLOR>>"
                    )

    # Investigation Timeline (BOCPD + known transitions combined)
    if timeline.changepoints or timeline.known_transitions:
        lines.append("\n<<COLOR:accent>>── Investigation Timeline ──<</COLOR>>")

        # Merge BOCPD CPs and known transitions into chronological order
        events = []
        for cpe in timeline.changepoints:
            rob_tag = ""
            if cpe.robustness_total > 1:
                if cpe.robustness >= cpe.robustness_total:
                    rob_tag = " [robust]"
                elif cpe.robustness <= 2:
                    rob_tag = " [uncertain]"
                else:
                    rob_tag = f" [{cpe.robustness}/{cpe.robustness_total}]"
            at_boundary = " (at lot boundary)" if cpe.near_known_transition else ""
            events.append(
                (
                    cpe.obs,
                    "bocpd",
                    f"BOCPD shift detected (P = {cpe.shift_prob:.0%}){rob_tag}{at_boundary}",
                )
            )
            if cpe.confirmation_obs >= 0 and cpe.confirmation_obs > cpe.obs:
                events.append((cpe.confirmation_obs, "confirm", "Shift confirmed (P \u2265 95%)"))
            if cpe.e_value > 5:
                events.append(
                    (
                        cpe.obs,
                        "evidence",
                        f'E-value reached "{cpe.evidence_level}" ({cpe.e_value:.0f}:1)',
                    )
                )
        if timeline.known_transitions:
            for kt in timeline.known_transitions:
                if kt.column == "material_lot":
                    events.append(
                        (
                            kt.obs,
                            "known",
                            f"Lot transition {kt.from_value} \u2192 {kt.to_value} [known]",
                        )
                    )
        if timeline.ed_peak_obs >= 0 and timeline.ed_peak_log_N > 0:
            if timeline.ed_peak_log_N >= timeline.ed_threshold:
                events.append(
                    (
                        timeline.ed_peak_obs,
                        "edetector",
                        f"E-Detector alarm (log(N) = {timeline.ed_peak_log_N:.1f})",
                    )
                )
            else:
                events.append(
                    (
                        timeline.ed_peak_obs,
                        "edetector",
                        f"E-Detector peak "
                        f"(log(N) = {timeline.ed_peak_log_N:.1f}, "
                        f"below threshold {timeline.ed_threshold:.1f} "
                        f"\u2014 no alarm)",
                    )
                )

        events.sort(key=lambda e: (e[0], e[1]))
        for obs, etype, desc in events:
            if etype == "known":
                color = "highlight"
            elif etype == "confirm":
                color = "success"
            elif etype == "edetector":
                color = "highlight" if "no alarm" in desc else "success"
            else:
                color = "warning"
            lines.append(f"  <<COLOR:{color}>>Obs {obs}:<</COLOR>> {desc}")

    # Per-Regime Capability (BOCPD regimes — only when CPs detected and no lot data)
    if timeline.changepoints and not timeline.lot_capabilities:
        lines.append("\n<<COLOR:accent>>── Per-Regime Capability ──<</COLOR>>")
        for r in timeline.regimes:
            cpk_r = r.cpk
            if cpk_r:
                ci = cpk_r.cpk_credible_interval
                p133 = cpk_r.cpk_probability_above_133
                if p133 >= 0.80:
                    color = "success"
                elif p133 >= 0.50:
                    color = "highlight"
                else:
                    color = "error"
                lines.append(
                    f"  <<COLOR:{color}>>{r.label} (obs {r.start + 1}\u2013{r.end}, "
                    f"n={r.n}):<</COLOR>> Cpk = {cpk_r.cpk_point_estimate:.2f} "
                    f"[{ci[0]:.2f}, {ci[1]:.2f}], "
                    f"P(Cpk>1.33) = {p133:.0%}"
                )
                if r.n < 30:
                    lines.append(
                        f"    <<COLOR:warning>>\u26a0 Posterior wide \u2014 {r.n} "
                        f"observations. 30 more would narrow CI "
                        f"by ~{r.ci_narrowing_30:.0f}%.<</COLOR>>"
                    )
            else:
                lines.append(
                    f"  {r.label} (obs {r.start + 1}\u2013{r.end}, n={r.n}): \u03bc={r.mean:.4f}, \u03c3={r.std:.4f}"
                )

    lines.append("\n<<COLOR:accent>>── Belief Chart ──<</COLOR>>")
    last_bc = bc.points[-1] if bc.points else None
    if last_bc:
        lines.append(f"  P(shifted): {last_bc.shift_probability:.1%}  [{last_bc.alert_level.upper()}]")
        lines.append(f"  Current regime mean: {last_bc.current_regime_mean:.4f}")
        lines.append(f"  Run length: {last_bc.most_likely_run_length}")

    lines.append("\n<<COLOR:accent>>── E-Detector ──<</COLOR>>")
    if ed_last:
        ed_status = "ALARM" if ed_last.alarm else "MONITORING"
        ed_ratio = math.exp(min(ed_last.log_N_combined, 500))
        lines.append(f"  Status: {ed_status}  log(N) = {ed_last.log_N_combined:.1f}")
        lines.append(f"  Evidence: {ed_ratio:.0f}:1 against in-control")
        lines.append(f"  Guarantee: ARL >= {1 / ed_alpha:.0f} (alpha={ed_alpha})")

    lines.append("\n<<COLOR:accent>>── Evidence ──<</COLOR>>")
    last_ev = ea.points[-1] if ea.points else None
    if last_ev:
        lines.append(f"  E-value: {last_ev.e_value_accumulated:.1f}:1  [{last_ev.evidence_level.upper()}]")

    lines.append("\n<<COLOR:accent>>── Adaptive Limits ──<</COLOR>>")
    last_acl = acl_points[-1] if acl_points else None
    if last_acl:
        lines.append(f"  CL = {last_acl.cl:.4f}  LCL = {last_acl.lcl:.4f}  UCL = {last_acl.ucl:.4f}")

    if pred:
        lines.append("\n<<COLOR:accent>>── Predictive ──<</COLOR>>")
        lines.append(
            f"  Slope: {pred.current_slope:.5f}  "
            f"[{pred.slope_credible_interval[0]:.5f}, "
            f"{pred.slope_credible_interval[1]:.5f}]"
        )
        lines.append(f"  P(slope > 0): {pred.slope_probability_positive:.0%}")
        if USL is not None or LSL is not None:
            lines.append(f"  P(exceed spec, 10 obs): {pred.prob_exceed_spec_10:.1%}")
            lines.append(f"  P(exceed spec, 25 obs): {pred.prob_exceed_spec_25:.1%}")

    if cpk_result:
        lines.append("\n<<COLOR:accent>>── Bayesian Cpk ──<</COLOR>>")
        ci_w = cpk_result.cpk_credible_interval[1] - cpk_result.cpk_credible_interval[0]
        lines.append(
            f"  Cpk: {cpk_result.cpk_point_estimate:.2f}  "
            f"[{cpk_result.cpk_credible_interval[0]:.2f}, "
            f"{cpk_result.cpk_credible_interval[1]:.2f}]"
        )
        lines.append(f"  P(Cpk > 1.0): {cpk_result.cpk_probability_above_1:.0%}")
        lines.append(f"  P(Cpk > 1.33): {cpk_result.cpk_probability_above_133:.0%}")
        lines.append(f"  Classical Cpk: {cpk_result.classical_cpk:.2f}")
        lines.append(f"  n_eff: {cpk_n_eff}  Posterior precision: \u00b1{ci_w / 2:.2f} (95% CI width = {ci_w:.2f})")
        if cpk_maturity < 1.0:
            lines.append(
                f"  \u26a0 Health discount: maturity = {cpk_maturity:.0%} "
                f"(n={cpk_n_eff} < 30, health Cpk = "
                f"{h_cpk_raw:.0%} \u00d7 {cpk_maturity:.0%} = {h_cpk:.0%})"
            )

    lines.append("\n<<COLOR:accent>>── Decision ──<</COLOR>>")
    lines.append(f"  Alarm threshold: P = {alarm_dec.threshold:.0%}")
    lines.append(
        f"  Recommendation: <<COLOR:{'warning' if alarm_dec.recommend_action == 'investigate' else 'good'}>>"
        f"{alarm_dec.recommend_action.upper()}<</COLOR>>"
    )

    lines.append("\n<<COLOR:accent>>── Health ──<</COLOR>>")
    lines.append(f"  Overall: {health.overall_health:.0%}")
    for k, v in health.stream_contributions.items():
        suffix = ""
        if k == "cpk" and cpk_maturity < 1.0 and cpk_result:
            ci_w = cpk_result.cpk_credible_interval[1] - cpk_result.cpk_credible_interval[0]
            suffix = f" \u26a0 discounted from {h_cpk_raw:.0%} \u2014 n={cpk_n_eff}, CI width {ci_w:.2f}"
        lines.append(f"    {k}: {v:.0%} (weight {health.stream_weights.get(k, 0):.0%}){suffix}")

    if beta_robustness > 0:
        w_3sigma = math.exp(-beta_robustness * 9.0 / 2.0)
        lines.append("\n<<COLOR:accent>>── Robustness ──<</COLOR>>")
        lines.append(f"  beta = {beta_robustness:.2f}  (outlier weight at 3 sigma: {w_3sigma:.1%})")
        downweighted = [(p.t, p.observation_weight) for p in bc.points if p.observation_weight < 0.5]
        if downweighted:
            lines.append(f"  {len(downweighted)} obs downweighted (<50%):")
            for t, w in downweighted[:10]:
                lines.append(f"    Obs {t}: w={w:.2f}")
            if len(downweighted) > 10:
                lines.append(f"    ... and {len(downweighted) - 10} more")
        else:
            lines.append("  No observations downweighted — data clean.")

    result["summary"] = "\n".join(lines)

    result["narrative"] = _narrative(
        "Process Belief System",
        narrative,
        chart_guidance="Charts are grouped by tab: Belief (shift detection), Control (adaptive limits), Prediction (forward projection), Capability (Bayesian Cpk), Health (multi-stream fusion).",
    )

    result["education"] = {
        "title": "Understanding the Process Belief System",
        "content": "<dl>"
        "<dt>What is PBS?</dt>"
        "<dd>The Process Belief System fuses five complementary monitoring streams into a single coherent picture of process health. "
        "Unlike traditional SPC which uses fixed rules, PBS continuously updates a probabilistic belief state as each observation arrives.</dd>"
        "<dt>Belief Chart — P(shifted)</dt>"
        "<dd>Uses Bayesian Online Changepoint Detection (BOCPD) to estimate the probability your process has shifted. "
        "Below 20% = stable. Above 95% = alarm. The system runs multiple sensitivity settings (\u03bb values) to ensure detected shifts are robust.</dd>"
        "<dt>E-values &amp; E-Detector</dt>"
        "<dd>E-values are <em>anytime-valid</em> evidence measures — unlike p-values, they can be checked continuously without inflating false alarm rates. "
        "An E-value of 50:1 means the data is 50\u00d7 more likely under 'changed' than 'stable.' "
        "The E-Detector provides a distribution-free companion that works even for non-normal data.</dd>"
        "<dt>Adaptive Control Limits</dt>"
        "<dd>Bayesian control limits that start wide (reflecting uncertainty) and narrow as data accumulates. "
        "They converge toward traditional \u00b13\u03c3 limits as the posterior gains precision.</dd>"
        "<dt>Bayesian Cpk</dt>"
        "<dd>Instead of a single Cpk number, PBS gives you the full posterior distribution of Cpk. "
        "P(Cpk > 1.33) is the key metric: it tells you the probability your process is truly capable, accounting for estimation uncertainty.</dd>"
        "<dt>Health Score</dt>"
        "<dd>A 0\u2013100% composite that fuses shift detection, capability, trend, and E-Detector streams. "
        "Below 50% = unhealthy. 50\u201375% = at risk. Above 75% = healthy. The primary driver identifies which stream is pulling health down most.</dd>"
        "</dl>",
    }

    result["statistics"] = {
        "test": "pbs_full",
        "n": n,
        "hazard_lambda": hazard_lambda,
        "lambda_grid": {lam: bc_runs[lam].log_marginal_likelihood for lam in lambda_grid},
        "shift_probability": last_bc.shift_probability if last_bc else 0,
        "n_changepoints": len(timeline.changepoints),
        "e_value": last_ev.e_value_accumulated if last_ev else 1,
        "edetector_log_N": ed_last.log_N_combined if ed_last else 0,
        "edetector_alarm": ed_last.alarm if ed_last else False,
        "health": health.overall_health,
        "alarm_action": alarm_dec.recommend_action,
        "narrative": narrative,
    }
    if cpk_result:
        result["statistics"]["cpk_median"] = cpk_result.cpk_point_estimate
        result["statistics"]["cpk_ci"] = list(cpk_result.cpk_credible_interval)
        result["statistics"]["p_cpk_above_133"] = cpk_result.cpk_probability_above_133

    # Per-regime Taguchi loss (only when taguchi_k > 0)
    if taguchi_k > 0:
        regime_losses = []
        for r in timeline.regimes:
            entry = {
                "label": r.label,
                "start": r.start,
                "end": r.end,
                "n": r.n,
            }
            if r.cpk:
                entry["cpk"] = r.cpk.cpk_point_estimate
                entry["p_above_133"] = r.cpk.cpk_probability_above_133
            if r.taguchi:
                entry["expected_loss"] = r.taguchi.expected_loss
                entry["bias_fraction"] = r.taguchi.bias_fraction
                entry["variance_fraction"] = r.taguchi.variance_fraction
            regime_losses.append(entry)
        result["statistics"]["regime_losses"] = regime_losses

        if timeline.lot_capabilities:
            lot_losses = []
            for lc in timeline.lot_capabilities:
                entry = {
                    "lot_id": lc.lot_id,
                    "start": lc.start,
                    "end": lc.end,
                    "n": lc.n,
                }
                if lc.cpk:
                    entry["cpk"] = lc.cpk.cpk_point_estimate
                    entry["p_above_133"] = lc.cpk.cpk_probability_above_133
                if lc.taguchi:
                    entry["expected_loss"] = lc.taguchi.expected_loss
                    entry["bias_fraction"] = lc.taguchi.bias_fraction
                    entry["variance_fraction"] = lc.taguchi.variance_fraction
                if lc.within_lot_cps:
                    entry["within_lot_shifts"] = lc.within_lot_cps
                lot_losses.append(entry)
            result["statistics"]["lot_losses"] = lot_losses

    result["guide_observation"] = narrative
    return result


# ── Individual analysis runners ──


def _run_belief_only(y, prior, hazard_lambda, config, beta_robustness=0.0):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    bc = BeliefChart(hazard_lambda=hazard_lambda, prior=prior.copy(), beta_robustness=beta_robustness)
    for x in y:
        bc.process(x)

    ts = list(range(len(y)))
    sps = [p.shift_probability for p in bc.points]
    result["plots"].append(
        {
            "title": "Belief Chart — P(Process Shifted)",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": sps,
                    "mode": "lines",
                    "name": "P(shift)",
                    "line": {"color": "#d94a4a", "width": 2},
                },
                {
                    "type": "scatter",
                    "x": [0, len(y) - 1],
                    "y": [0.5, 0.5],
                    "mode": "lines",
                    "name": "Watch",
                    "line": {"color": "#d4a24a", "dash": "dot", "width": 1},
                },
                {
                    "type": "scatter",
                    "x": [0, len(y) - 1],
                    "y": [0.95, 0.95],
                    "mode": "lines",
                    "name": "Alarm",
                    "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "yaxis": {"title": "P(shifted)", "range": [0, 1.05]},
                "xaxis": {"title": "Observation"},
            },
        }
    )

    last = bc.points[-1]
    result["summary"] = (
        f"Belief Chart: P(shifted) = {last.shift_probability:.1%} "
        f"[{last.alert_level.upper()}]. "
        f"Regime mean: {last.current_regime_mean:.4f}. "
        f"Run length: {last.most_likely_run_length}."
    )
    result["statistics"] = {
        "test": "pbs_belief",
        "shift_probability": last.shift_probability,
        "alert_level": last.alert_level,
        "regime_mean": last.current_regime_mean,
    }
    sp = last.shift_probability
    if sp < 0.20:
        _bv = "Process is stable"
        _bb = f"Shift probability is {sp:.0%} \u2014 no evidence of a process change."
        _bn = "Continue monitoring. No action needed."
    elif sp < 0.50:
        _bv = "Early signs of change"
        _bb = f"Shift probability is {sp:.0%}. The detector is picking up some evidence but hasn't confirmed a shift."
        _bn = "Watch closely. If probability continues rising, investigate."
    elif sp < 0.80:
        _bv = f"Process is likely shifting \u2014 P = {sp:.0%}"
        _bb = f"Regime mean has moved to {last.current_regime_mean:.4f}. Evidence is building but not yet decisive."
        _bn = "Begin investigation. Check recent process changes, material lots, or operator shifts."
    else:
        _bv = f"Process has shifted \u2014 P = {sp:.0%}"
        _bb = f"New regime mean: {last.current_regime_mean:.4f}. Run length: {last.most_likely_run_length} observations since last shift."
        _bn = "Investigate immediately. Identify assignable cause and determine if corrective action is needed."
    result["narrative"] = _narrative(
        _bv,
        _bb,
        next_steps=_bn,
        chart_guidance="The y-axis shows P(shifted) \u2014 the probability that the process mean has changed. Below the gold dotted line (50%) is normal. Above the red dashed line (95%) is an alarm.",
    )
    result["education"] = {
        "title": "Understanding the Belief Chart",
        "content": "<dl>"
        "<dt>What is this?</dt>"
        "<dd>The Belief Chart uses Bayesian Online Changepoint Detection (BOCPD) to continuously estimate the probability that your process has shifted. "
        "Unlike traditional control charts that use fixed rules, this accumulates evidence and reports a probability.</dd>"
        "<dt>P(shifted)</dt>"
        "<dd>The probability that current data comes from a different distribution than before. "
        "Below 20% = stable. 20\u201350% = watch. 50\u201380% = likely shifting. Above 95% = alarm.</dd>"
        "<dt>Run length</dt>"
        "<dd>How many observations since the last detected regime change. A sudden drop in run length signals a new regime has started.</dd>"
        "<dt>What's good?</dt>"
        "<dd>A flat line near zero means your process is stable and predictable. Spikes that return to zero are transient \u2014 "
        "the detector considered a shift but the evidence didn't hold.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_edetector_only(y, mu_0, USL, LSL, sigma_cal, config):
    """E-Detector chart — distribution-free CUSUM changepoint detection."""
    result = {"plots": [], "summary": "", "guide_observation": ""}
    alpha = float(config.get("edetector_alpha", 0.05))

    # Bounds: prefer spec limits, fall back to calibration-based (Anti-Pattern 4)
    if USL is not None and LSL is not None:
        a, b = float(LSL), float(USL)
    else:
        # Tighter bounds from calibration data: X̄ ± 4s
        a = mu_0 - 4.0 * max(sigma_cal, 1e-6)
        b = mu_0 + 4.0 * max(sigma_cal, 1e-6)

    ed = EDetector(mu_0=mu_0, bounds=(a, b), alpha=alpha)
    for x in y:
        ed.process(x)

    ts = list(range(len(y)))
    log_Ns = [p.log_N_combined for p in ed.points]
    threshold = ed.threshold

    # Color segments: green below threshold, red above
    ["#d94a4a" if p.alarm else "#4a9f6e" for p in ed.points]

    # Find first alarm points for markers
    alarm_ts = [p.t - 1 for p in ed.points if p.alarm]
    alarm_vals = [p.log_N_combined for p in ed.points if p.alarm]

    traces = [
        {
            "type": "scatter",
            "x": ts,
            "y": log_Ns,
            "mode": "lines",
            "name": "log(N)",
            "line": {"color": "#4a9f6e", "width": 2},
        },
        # Threshold line
        {
            "type": "scatter",
            "x": [0, len(y) - 1],
            "y": [threshold, threshold],
            "mode": "lines",
            "name": f"1/alpha = {1 / alpha:.0f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
        },
        # Zero (reset) line
        {
            "type": "scatter",
            "x": [0, len(y) - 1],
            "y": [0, 0],
            "mode": "lines",
            "name": "Reset",
            "line": {"color": "#666", "dash": "dot", "width": 1},
        },
    ]
    if alarm_ts:
        traces.append(
            {
                "type": "scatter",
                "x": alarm_ts,
                "y": alarm_vals,
                "mode": "markers",
                "name": "Alarm",
                "marker": {"color": "#d94a4a", "symbol": "diamond", "size": 10},
            }
        )

    result["plots"].append(
        {
            "title": "E-Detector — Distribution-Free Changepoint",
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "yaxis": {"title": "log(N)"},
                "xaxis": {"title": "Observation"},
            },
        }
    )

    last = ed.points[-1]
    status = "ALARM" if last.alarm else "MONITORING"
    evidence_ratio = math.exp(min(last.log_N_combined, 500))

    result["summary"] = (
        f"E-Detector status: {status}. "
        f"Log-evidence: {last.log_N_combined:.1f} "
        f"(threshold: {threshold:.1f}). "
        f"Evidence ratio: {evidence_ratio:.0f}:1 against in-control. "
        f"Method: Distribution-free CUSUM e-detector. "
        f"Guarantee: ARL >= {1 / alpha:.0f} under any pre-change distribution."
    )
    result["statistics"] = {
        "test": "pbs_edetector",
        "status": status.lower(),
        "log_N": last.log_N_combined,
        "threshold": threshold,
        "evidence_ratio": evidence_ratio,
        "alpha": alpha,
        "n_alarms": len(alarm_ts),
    }
    if status == "ALARM":
        _ev = f"E-Detector alarm \u2014 {evidence_ratio:.0f}:1 evidence"
        _eb = f"The distribution-free detector has crossed the alarm threshold. This detection is guaranteed to have a false alarm rate \u2264 {alpha:.0%} regardless of the data distribution."
        _en = "Investigate the source of change. This alarm is valid even for non-normal data."
    else:
        _ev = "E-Detector monitoring \u2014 no alarm"
        _eb = f"Log-evidence: {last.log_N_combined:.1f} (threshold: {threshold:.1f}). The detector has not accumulated enough evidence to declare a change."
        _en = "Continue monitoring. The detector will alarm if cumulative evidence exceeds the threshold."
    result["narrative"] = _narrative(
        _ev,
        _eb,
        next_steps=_en,
        chart_guidance="The green line shows cumulative log-evidence against in-control. When it crosses the red dashed threshold, the detector alarms. Diamond markers show alarm points.",
    )
    result["education"] = {
        "title": "Understanding the E-Detector",
        "content": "<dl>"
        "<dt>What is this?</dt>"
        "<dd>The E-Detector is a distribution-free changepoint detector (Shin, Ramdas &amp; Rinaldo 2024). "
        "Unlike traditional tests that assume normality, this works for <em>any</em> data distribution \u2014 skewed, heavy-tailed, or otherwise.</dd>"
        "<dt>log(N)</dt>"
        "<dd>The cumulative evidence statistic. Think of it as a running score: each observation adds or subtracts evidence. "
        "When the score crosses the threshold, there's enough evidence to declare a change.</dd>"
        "<dt>The guarantee</dt>"
        "<dd>ARL \u2265 1/\u03b1 means: on average, you'll wait at least 1/\u03b1 observations before a false alarm. "
        f"With \u03b1 = {alpha}, that's at least {1 / alpha:.0f} observations \u2014 and this holds regardless of your data's shape.</dd>"
        "<dt>When to use this vs. Belief Chart</dt>"
        "<dd>Use E-Detector when you can't assume normality or want a formal false-alarm guarantee. "
        "Use Belief Chart when you want richer information (regime means, run lengths, shift probability).</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_evidence_only(y, prior, mu_0, config, sigma_ref):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    ea = EvidenceAccumulation(mu_0=mu_0, sigma_ref=sigma_ref)
    for x in y:
        ea.process(x)

    ts = list(range(len(y)))
    log_es = [p.log_e_accumulated for p in ea.points]
    result["plots"].append(
        {
            "title": "Evidence Accumulation — E-value (log scale)",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": log_es,
                    "mode": "lines",
                    "name": "log(E)",
                    "line": {"color": "#6ab7d4", "width": 2},
                },
                {
                    "type": "scatter",
                    "x": [0, len(y) - 1],
                    "y": [math.log(20), math.log(20)],
                    "mode": "lines",
                    "name": "Strong",
                    "line": {"color": "#4a9f6e", "dash": "dash", "width": 1},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "yaxis": {"title": "log(E-value)"},
                "xaxis": {"title": "Observation"},
            },
        }
    )
    last = ea.points[-1]
    result["summary"] = (
        f"Evidence: {last.e_value_accumulated:.1f}:1 against in-control [{last.evidence_level.upper()}]."
    )
    result["statistics"] = {
        "test": "pbs_evidence",
        "e_value": last.e_value_accumulated,
        "evidence_level": last.evidence_level,
    }
    ev = last.e_value_accumulated
    if ev < 3:
        _vv = "No evidence of change"
        _vb = f"E-value: {ev:.1f}:1 \u2014 not enough evidence to distinguish from normal variation."
        _vn = "Continue collecting data. E-values grow as evidence accumulates."
    elif ev < 20:
        _vv = f"Notable evidence \u2014 {ev:.1f}:1"
        _vb = "Some evidence against the in-control hypothesis, but not yet strong enough for a confident conclusion."
        _vn = "Continue monitoring. Evidence is building but not yet actionable."
    elif ev < 100:
        _vv = f"Strong evidence \u2014 {ev:.1f}:1 against in-control"
        _vb = "The data provides strong evidence that the process has changed from its reference state."
        _vn = "Investigate the process change. This level of evidence is rarely due to chance."
    else:
        _vv = f"Decisive evidence \u2014 {ev:.0f}:1 against in-control"
        _vb = f"Overwhelming evidence of a process change. The chance of this arising by random variation is less than 1 in {ev:.0f}."
        _vn = "Investigate immediately. The evidence is conclusive."
    result["narrative"] = _narrative(
        _vv,
        _vb,
        next_steps=_vn,
        chart_guidance="The y-axis shows log(E-value). The green dashed line marks 'Strong' evidence (20:1, log \u2248 3.0). Values above this line indicate a real process change.",
    )
    result["education"] = {
        "title": "Understanding E-Values",
        "content": "<dl>"
        "<dt>What is an E-value?</dt>"
        "<dd>An E-value measures evidence against a hypothesis \u2014 here, the hypothesis that your process is still in control. "
        "An E-value of 50 means the data is 50 times more likely under 'process changed' than under 'process stable.'</dd>"
        "<dt>Why not p-values?</dt>"
        "<dd>P-values can't be monitored continuously \u2014 if you keep checking, you'll eventually get a false alarm. "
        "E-values are <em>anytime-valid</em>: you can check as often as you want without inflating your error rate. This is critical for ongoing process monitoring.</dd>"
        "<dt>The log scale</dt>"
        "<dd>The chart shows log(E) because E-values can grow very large. log(E) \u2248 3 means E \u2248 20:1 (strong). "
        "log(E) \u2248 4.6 means E \u2248 100:1 (decisive). Below 0 = evidence favors in-control.</dd>"
        "<dt>Evidence levels</dt>"
        "<dd>None (&lt;3:1) \u2192 Notable (3\u201320:1) \u2192 Strong (20\u2013100:1) \u2192 Decisive (&gt;100:1). "
        "These thresholds are calibrated to match scientific standards of evidence.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_predictive_only(y, USL, LSL, config):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    pc = PredictiveChart(window=min(20, len(y)))
    pred = pc.compute(y, USL=USL, LSL=LSL)
    n = len(y)

    fan_h = [p.horizon for p in pred.prediction_fan]
    fan_x = [n - 1 + h for h in fan_h]
    fan_mean = [p.mean for p in pred.prediction_fan]
    fan_90l = [p.ci90_lower for p in pred.prediction_fan]
    fan_90u = [p.ci90_upper for p in pred.prediction_fan]

    traces = [
        {
            "type": "scatter",
            "x": list(range(max(0, n - 20), n)),
            "y": y[-20:].tolist(),
            "mode": "lines+markers",
            "name": "Recent data",
            "line": {"color": "#4a9f6e"},
            "marker": {"size": 3},
        },
        {
            "type": "scatter",
            "x": fan_x,
            "y": fan_mean,
            "mode": "lines",
            "name": "Predicted",
            "line": {"color": "#d4a24a", "width": 2},
        },
        {
            "type": "scatter",
            "x": fan_x + fan_x[::-1],
            "y": fan_90u + fan_90l[::-1],
            "fill": "toself",
            "fillcolor": "rgba(212,162,74,0.15)",
            "line": {"color": "transparent"},
            "name": "90% CI",
        },
    ]
    result["plots"].append(
        {
            "title": "Predictive Chart",
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": config.get("column", "Value")},
            },
        }
    )
    result["summary"] = (
        f"Slope: {pred.current_slope:.5f} "
        f"[{pred.slope_credible_interval[0]:.5f}, "
        f"{pred.slope_credible_interval[1]:.5f}]. "
        f"P(slope > 0): {pred.slope_probability_positive:.0%}."
    )
    if USL is not None or LSL is not None:
        result["summary"] += (
            f" P(exceed spec, 10 obs): {pred.prob_exceed_spec_10:.1%}."
            f" P(exceed spec, 25 obs): {pred.prob_exceed_spec_25:.1%}."
        )
    result["statistics"] = {
        "test": "pbs_predictive",
        "slope": pred.current_slope,
        "prob_exceed_10": pred.prob_exceed_spec_10,
        "prob_exceed_25": pred.prob_exceed_spec_25,
    }
    _slope = pred.current_slope
    _p_pos = pred.slope_probability_positive
    _p10 = pred.prob_exceed_spec_10
    if abs(_slope) < 1e-6:
        _pv = "Trend is flat \u2014 process is stable"
        _pb = f"Slope: {_slope:.5f} per observation. No meaningful drift detected."
    elif _p_pos > 0.9:
        _pv = f"Upward trend detected \u2014 slope = {_slope:.5f}"
        _pb = f"P(slope > 0) = {_p_pos:.0%}. Process is drifting upward."
    elif _p_pos < 0.1:
        _pv = f"Downward trend detected \u2014 slope = {_slope:.5f}"
        _pb = f"P(slope > 0) = {_p_pos:.0%}. Process is drifting downward."
    else:
        _pv = f"Uncertain trend \u2014 slope = {_slope:.5f}"
        _pb = f"P(slope > 0) = {_p_pos:.0%}. Direction not yet clear."
    if (USL is not None or LSL is not None) and _p10 > 0.05:
        _pb += f" Spec exceedance risk: {_p10:.0%} within 10 observations."
        _pn = (
            "Monitor closely \u2014 trend may push process out of spec."
            if _p10 > 0.20
            else "Risk is moderate. Continue monitoring."
        )
    else:
        _pn = (
            "No immediate spec exceedance risk."
            if (USL or LSL)
            else "Add spec limits (USL/LSL) to assess exceedance risk."
        )
    result["narrative"] = _narrative(
        _pv,
        _pb,
        next_steps=_pn,
        chart_guidance="The gold line is the predicted mean. The shaded fan shows the 90% credible interval \u2014 future observations should fall within this band 90% of the time.",
    )
    result["education"] = {
        "title": "Understanding the Predictive Chart",
        "content": "<dl>"
        "<dt>What is this?</dt>"
        "<dd>The Predictive Chart fits a Bayesian linear trend to recent data and projects it forward. "
        "The widening fan reflects increasing uncertainty the further you predict.</dd>"
        "<dt>Slope</dt>"
        "<dd>The estimated rate of change per observation. The credible interval shows plausible values for the true slope, "
        "accounting for estimation uncertainty. P(slope > 0) tells you how confident we are the trend is upward.</dd>"
        "<dt>Spec exceedance probability</dt>"
        "<dd>If you provided spec limits, this is the probability that future observations will fall outside spec. "
        "It accounts for both the trend direction <em>and</em> the random scatter around the trend. "
        "Under 5% = negligible risk. 5\u201320% = watch. Above 20% = take action.</dd>"
        "<dt>Credible intervals vs confidence intervals</dt>"
        "<dd>The shaded fan is a <em>credible interval</em> \u2014 there's a 90% probability the next value falls inside it. "
        "This is the Bayesian interpretation, which is what most people intuitively expect from an interval estimate.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_adaptive_only(y, prior, config):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    acl = AdaptiveControlLimits()
    post = prior.copy()
    points = []
    for t, x in enumerate(y):
        post.update_single(x)
        points.append(acl.compute_limits(post, t, x))

    ts = list(range(len(y)))
    result["plots"].append(
        {
            "title": "Adaptive Control Limits",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": [p.observation for p in points],
                    "mode": "markers",
                    "name": "Obs",
                    "marker": {"color": "#4a9f6e", "size": 4},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": [p.cl for p in points],
                    "mode": "lines",
                    "name": "CL",
                    "line": {"color": "#d4a24a", "width": 1.5},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": [p.ucl for p in points],
                    "mode": "lines",
                    "name": "UCL",
                    "line": {"color": "#d94a4a", "dash": "dash"},
                },
                {
                    "type": "scatter",
                    "x": ts,
                    "y": [p.lcl for p in points],
                    "mode": "lines",
                    "name": "LCL",
                    "line": {"color": "#d94a4a", "dash": "dash"},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": config.get("column", "Value")},
            },
        }
    )
    last = points[-1]
    result["summary"] = f"CL = {last.cl:.4f}, LCL = {last.lcl:.4f}, UCL = {last.ucl:.4f} (n = {last.n_obs})."
    result["statistics"] = {
        "test": "pbs_adaptive",
        "cl": last.cl,
        "ucl": last.ucl,
        "lcl": last.lcl,
    }
    _width = last.ucl - last.lcl
    result["narrative"] = _narrative(
        f"Adaptive limits: CL = {last.cl:.4f}",
        f"UCL = {last.ucl:.4f}, LCL = {last.lcl:.4f} (width = {_width:.4f}). "
        f"Based on {last.n_obs} observations. Limits narrow as posterior precision increases.",
        next_steps="These limits adapt to your data. Early observations show wider limits reflecting prior uncertainty; "
        "limits converge toward traditional \u00b13\u03c3 as more data arrives.",
        chart_guidance="The dashed red lines are the adaptive control limits. They narrow over time as the Bayesian posterior gains precision. "
        "Points outside the limits suggest a process change.",
    )
    result["education"] = {
        "title": "Understanding Adaptive Control Limits",
        "content": "<dl>"
        "<dt>What is this?</dt>"
        "<dd>Adaptive control limits are Bayesian prediction intervals that account for your current uncertainty about the process. "
        "They start wide (when you have little data) and narrow as evidence accumulates.</dd>"
        "<dt>Why do limits narrow?</dt>"
        "<dd>With each observation, the Bayesian posterior becomes more precise about the true process mean and variance. "
        "Wider limits early on prevent false alarms when you're still learning the process.</dd>"
        "<dt>How is this different from Shewhart?</dt>"
        "<dd>Traditional Shewhart charts use fixed limits (\u03bc \u00b1 3\u03c3) that don't change. Adaptive limits reflect actual uncertainty \u2014 "
        "they're wider when you're unsure and narrower when you're confident. They converge to Shewhart limits asymptotically.</dd>"
        "<dt>When to use</dt>"
        "<dd>Adaptive limits are best for new processes, short production runs, or after a known process change when you need to quickly re-establish limits.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_cpk_only(y, prior, USL, LSL, config):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    if USL is None or LSL is None:
        result["summary"] = "Error: Need both USL and LSL for Cpk."
        return result

    post = prior.copy()
    post.update(y)
    engine = BayesianCpk(USL, LSL)
    cpk = engine.compute(post, len(y))

    # Generate histogram
    rng = np.random.RandomState(42)
    tau_s = rng.gamma(post.alpha, 1.0 / post.beta, size=10000)
    sig_mu = 1.0 / np.sqrt(post.kappa * tau_s)
    mu_s = rng.normal(post.mu, sig_mu)
    sigma_s = 1.0 / np.sqrt(tau_s)
    cpk_s = np.minimum((USL - mu_s) / (3 * sigma_s), (mu_s - LSL) / (3 * sigma_s))
    cpk_s = cpk_s[(cpk_s > -2) & (cpk_s < 5)]

    hv, be = np.histogram(cpk_s, bins=50, density=True)
    bc = (be[:-1] + be[1:]) / 2
    result["plots"].append(
        {
            "title": "Bayesian Cpk Posterior",
            "data": [
                {
                    "type": "bar",
                    "x": bc.tolist(),
                    "y": hv.tolist(),
                    "marker": {"color": "rgba(74,159,110,0.5)"},
                    "name": "Posterior",
                },
                {
                    "type": "scatter",
                    "x": [1.33, 1.33],
                    "y": [0, max(hv) * 1.1],
                    "mode": "lines",
                    "name": "1.33",
                    "line": {"color": "#d4a24a", "dash": "dash"},
                },
                {
                    "type": "scatter",
                    "x": [1.0, 1.0],
                    "y": [0, max(hv) * 1.1],
                    "mode": "lines",
                    "name": "1.0",
                    "line": {"color": "#d94a4a", "dash": "dash"},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Cpk"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    result["summary"] = (
        f"Bayesian Cpk: {cpk.cpk_point_estimate:.2f} "
        f"[{cpk.cpk_credible_interval[0]:.2f}, "
        f"{cpk.cpk_credible_interval[1]:.2f}]. "
        f"P(Cpk>1.0): {cpk.cpk_probability_above_1:.0%}. "
        f"P(Cpk>1.33): {cpk.cpk_probability_above_133:.0%}. "
        f"Classical: {cpk.classical_cpk:.2f}."
    )
    result["statistics"] = {
        "test": "pbs_cpk",
        "cpk_median": cpk.cpk_point_estimate,
        "cpk_ci": list(cpk.cpk_credible_interval),
        "p_above_133": cpk.cpk_probability_above_133,
        "classical": cpk.classical_cpk,
    }
    _cpk_v = cpk.cpk_point_estimate
    _p133 = cpk.cpk_probability_above_133
    if _cpk_v >= 1.33 and _p133 >= 0.80:
        _cv = f"Process is capable \u2014 Cpk = {_cpk_v:.2f}"
        _cb = f"P(Cpk > 1.33) = {_p133:.0%}. High confidence the process meets the 4-sigma standard."
        _cn = "Process is performing well. Monitor for stability."
    elif _cpk_v >= 1.0:
        _cv = f"Marginally capable \u2014 Cpk = {_cpk_v:.2f}"
        _cb = f"P(Cpk > 1.33) = {_p133:.0%}. Process meets minimum but not the 4-sigma target."
        _cn = "Investigate sources of variation. Centering the process or reducing spread could push Cpk above 1.33."
    else:
        _cv = f"Not capable \u2014 Cpk = {_cpk_v:.2f}"
        _cb = f"P(Cpk > 1.33) = {_p133:.0%}. Process spread exceeds specification tolerance."
        _cn = "Reduce variation or widen spec limits. Identify dominant sources of variation."
    _cb += f" Classical Cpk: {cpk.classical_cpk:.2f}."
    result["narrative"] = _narrative(
        _cv,
        _cb,
        next_steps=_cn,
        chart_guidance="The histogram shows the posterior distribution of Cpk. The gold dashed line is the 1.33 target. "
        "The red dashed line is the 1.0 minimum. More density to the right of 1.33 = more confidence in capability.",
    )
    result["education"] = {
        "title": "Understanding Bayesian Cpk",
        "content": "<dl>"
        "<dt>What is Cpk?</dt>"
        "<dd>Cpk measures how well your process fits within specification limits. "
        "Cpk \u2265 1.33 means the process spread uses at most 75% of the spec tolerance (4-sigma standard). "
        "Cpk \u2265 1.0 is the minimum acceptable (3-sigma). Below 1.0, defects are expected.</dd>"
        "<dt>Why Bayesian?</dt>"
        "<dd>Classical Cpk gives you one number but no uncertainty. Bayesian Cpk gives you the <em>full distribution</em> \u2014 "
        "how confident you should be in that number. With 20 observations, a Cpk of 1.5 might really be anywhere from 0.9 to 2.1. "
        "With 200 observations, the uncertainty shrinks dramatically.</dd>"
        "<dt>P(Cpk > 1.33)</dt>"
        "<dd>This is the key metric: the probability that your process is truly capable. "
        "80%+ = confident. 50\u201380% = uncertain. Below 50% = more likely not capable.</dd>"
        "<dt>Credible interval width</dt>"
        "<dd>A wide interval means you need more data. A narrow interval means the estimate is reliable. "
        "Collecting 30+ observations typically gives a useful posterior.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_cpk_traj_only(y, prior, USL, LSL, config):
    result = {"plots": [], "summary": "", "guide_observation": ""}
    if USL is None or LSL is None:
        result["summary"] = "Error: Need both USL and LSL."
        return result

    engine = CpkTrajectory(USL, LSL)
    out = engine.compute(y, prior)

    ts = [p.t for p in out.trajectory]
    meds = [p.cpk_median for p in out.trajectory]
    lows = [p.cpk_ci_lower for p in out.trajectory]
    highs = [p.cpk_ci_upper for p in out.trajectory]

    result["plots"].append(
        {
            "title": "Cpk Trajectory",
            "data": [
                {
                    "type": "scatter",
                    "x": ts,
                    "y": meds,
                    "mode": "lines+markers",
                    "name": "Cpk median",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "marker": {"size": 3},
                },
                {
                    "type": "scatter",
                    "x": ts + ts[::-1],
                    "y": highs + lows[::-1],
                    "fill": "toself",
                    "fillcolor": "rgba(74,159,110,0.2)",
                    "line": {"color": "transparent"},
                    "name": "90% CI",
                },
                {
                    "type": "scatter",
                    "x": [ts[0], ts[-1]],
                    "y": [1.33, 1.33],
                    "mode": "lines",
                    "name": "Threshold",
                    "line": {"color": "#d4a24a", "dash": "dash"},
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Cpk"},
            },
        }
    )

    result["summary"] = (
        f"Cpk trend slope: {out.trend_slope:.5f} "
        f"[{out.trend_slope_ci[0]:.5f}, {out.trend_slope_ci[1]:.5f}]. "
        f"P(declining): {out.prob_cpk_declining:.0%}."
    )
    if out.estimated_obs_to_threshold is not None:
        result["summary"] += f" Est. {out.estimated_obs_to_threshold} obs to Cpk < {out.threshold}."
    result["statistics"] = {
        "test": "pbs_cpk_traj",
        "trend_slope": out.trend_slope,
        "prob_declining": out.prob_cpk_declining,
        "est_obs_to_threshold": out.estimated_obs_to_threshold,
    }
    _pd = out.prob_cpk_declining
    _sl = out.trend_slope
    if _pd < 0.30:
        _tv = f"Cpk is stable or improving \u2014 P(declining) = {_pd:.0%}"
        _tb = f"Trend slope: {_sl:.5f} per observation. No evidence of capability degradation."
        _tn = "Process capability is holding. Continue routine monitoring."
    elif _pd < 0.70:
        _tv = f"Cpk trend is uncertain \u2014 P(declining) = {_pd:.0%}"
        _tb = f"Trend slope: {_sl:.5f}. Not enough evidence to confirm whether capability is improving or declining."
        _tn = "Collect more data to resolve the trend direction."
    else:
        _tv = f"Cpk is declining \u2014 P(declining) = {_pd:.0%}"
        _tb = f"Trend slope: {_sl:.5f} per observation."
        if out.estimated_obs_to_threshold is not None:
            _tb += f" At this rate, Cpk will cross below {out.threshold} in approximately {out.estimated_obs_to_threshold} observations."
        _tn = "Investigate sources of increasing variation or mean drift. Act before capability falls below threshold."
    result["narrative"] = _narrative(
        _tv,
        _tb,
        next_steps=_tn,
        chart_guidance="The green line tracks Cpk over time with 90% credible bands. The gold dashed line is the 1.33 target. "
        "A downward slope means capability is deteriorating.",
    )
    result["education"] = {
        "title": "Understanding the Cpk Trajectory",
        "content": "<dl>"
        "<dt>What is this?</dt>"
        "<dd>The Cpk Trajectory tracks how your process capability evolves over time. A rolling Bayesian Cpk is computed "
        "at each observation and a linear trend is fitted to detect improvement or degradation.</dd>"
        "<dt>P(declining)</dt>"
        "<dd>The probability that the true trend slope is negative (Cpk getting worse). "
        "Below 30% = likely stable or improving. 30\u201370% = uncertain. Above 70% = likely declining.</dd>"
        "<dt>Time to threshold</dt>"
        "<dd>If the trend is declining, this estimates how many observations until Cpk crosses below the target (default 1.33). "
        "This assumes the linear trend continues \u2014 intervention can change the trajectory.</dd>"
        "<dt>The credible band</dt>"
        "<dd>The shaded region shows 90% credible interval for rolling Cpk. Early observations have wider bands "
        "because the posterior is still learning. Narrowing bands indicate increasing confidence.</dd>"
        "</dl>",
    }
    result["guide_observation"] = result["summary"]
    return result


def _run_health_only(y, prior, USL, LSL, mu_0, hazard_lambda, config, beta_robustness=0.0):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Compute all streams
    bc = BeliefChart(hazard_lambda=hazard_lambda, prior=prior.copy(), beta_robustness=beta_robustness)
    for x in y:
        bc.process(x)
    h_spc = 1 - bc.points[-1].shift_probability

    h_cpk = 0.5
    if USL is not None and LSL is not None:
        post = prior.copy()
        post.update(y)
        cpk_e = BayesianCpk(USL, LSL)
        cpk_r = cpk_e.compute(post, len(y))
        h_cpk = cpk_r.cpk_probability_above_133

    pc = PredictiveChart(window=min(20, len(y)))
    pred = pc.compute(y, USL=USL, LSL=LSL)
    h_trend = 1.0 - pred.prob_exceed_spec_10

    streams = {"spc": h_spc, "cpk": h_cpk, "trend": h_trend}
    msh = MultiStreamHealth()
    health = msh.fuse(streams)

    # Bar chart of stream contributions
    labels = list(health.stream_contributions.keys())
    vals = [health.stream_contributions[k] for k in labels]
    colors = ["#4a9f6e" if v > 0.7 else "#d4a24a" if v > 0.4 else "#d94a4a" for v in vals]

    result["plots"].append(
        {
            "title": f"Process Health: {health.overall_health:.0%}",
            "data": [
                {
                    "type": "bar",
                    "x": labels,
                    "y": vals,
                    "marker": {"color": colors},
                    "name": "Stream health",
                },
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 250,
                "yaxis": {"title": "Health", "range": [0, 1.05]},
            },
        }
    )

    result["summary"] = (
        f"Overall health: {health.overall_health:.0%}. "
        f"Primary driver: {health.primary_driver}. "
        + " | ".join(f"{k}: {v:.0%}" for k, v in health.stream_contributions.items())
    )
    result["statistics"] = {
        "test": "pbs_health",
        "overall": health.overall_health,
        "streams": health.stream_contributions,
        "primary_driver": health.primary_driver,
    }

    # --- narrative ---
    oh = health.overall_health
    driver = health.primary_driver
    stream_str = ", ".join(f"{k} {v:.0%}" for k, v in health.stream_contributions.items())
    if oh >= 0.75:
        verdict_word = "Healthy"
        body = (
            f"Overall health score is <strong>{oh:.0%}</strong> — the process "
            f"is performing well across monitored streams ({stream_str}). "
            f"Primary driver: <em>{driver}</em>."
        )
    elif oh >= 0.50:
        verdict_word = "At Risk"
        body = (
            f"Overall health score is <strong>{oh:.0%}</strong> — some streams "
            f"show degradation ({stream_str}). Primary concern: <em>{driver}</em>. "
            f"Investigate the weakest stream to prevent further decline."
        )
    else:
        verdict_word = "Unhealthy"
        body = (
            f"Overall health score is <strong>{oh:.0%}</strong> — significant "
            f"issues detected ({stream_str}). Primary driver: <em>{driver}</em>. "
            f"Immediate investigation recommended."
        )
    result["narrative"] = _narrative(
        f"Process Health: {verdict_word} — {oh:.0%}",
        body,
        chart_guidance="The bar chart shows each stream's health contribution. Green (>70%) is healthy, amber (40-70%) needs attention, red (<40%) needs action.",
        next_steps=(
            "Monitor — process is healthy."
            if oh >= 0.75
            else (
                f"Investigate the <em>{driver}</em> stream."
                if oh >= 0.50
                else f"Prioritise root-cause analysis on <em>{driver}</em>; consider running individual PBS analyses for detail."
            )
        ),
    )

    # --- education ---
    result["education"] = {
        "title": "Understanding Process Health Score",
        "content": (
            "<dl>"
            "<dt>What is the Health Score?</dt>"
            "<dd>A single 0–100% metric that fuses multiple monitoring streams into one "
            "view of process condition. It uses log-linear fusion — each stream contributes "
            "proportionally, and a single failing stream pulls the score down quickly.</dd>"
            "<dt>What streams feed the score?</dt>"
            "<dd><strong>SPC</strong> — stability from Belief Chart (shift probability). "
            "<strong>Cpk</strong> — capability relative to specs (P(Cpk &gt; 1.33)). "
            "<strong>Trend</strong> — forward-looking risk from Predictive Chart "
            "(probability of exceeding spec in 10 observations).</dd>"
            "<dt>What does 'Primary Driver' mean?</dt>"
            "<dd>The stream contributing most to the current score — i.e., the area "
            "having the biggest impact (positive or negative) on overall health.</dd>"
            "<dt>How to interpret</dt>"
            "<dd><strong>≥ 75%</strong>: Healthy — process is stable, capable, and "
            "not trending toward trouble. <strong>50–75%</strong>: At risk — one or more "
            "streams are degrading; investigate the primary driver. <strong>&lt; 50%</strong>: "
            "Unhealthy — significant issues; run individual PBS analyses for detail.</dd>"
            "</dl>"
        ),
    }

    result["guide_observation"] = result["summary"]
    return result
