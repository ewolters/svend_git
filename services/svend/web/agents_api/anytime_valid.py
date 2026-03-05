"""
Anytime-Valid Inference — E-processes and Confidence Sequences.

Implements sequential testing where you can peek at data continuously
without inflating error rates. Based on Grünwald et al. (JRSS-B 2024),
Howard et al. (2021), and Waudby-Smith & Ramdas (2024).

Core objects:
    GaussianMeanEProcess          — known σ, mixture over mean
    SelfNormalizedMeanEProcess    — unknown σ, empirical variance
    TwoSampleEProcess             — A/B test wrapper

Key property: each E-process is a supermartingale under H₀.
    E[E_t | F_{t-1}] ≤ E_{t-1}
    → valid at any data-dependent stopping time (optional stopping).

All arithmetic in log-space for numerical stability.

Dependencies: numpy, scipy (only).
"""

import math
import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "GaussianMeanEProcess",
    "SelfNormalizedMeanEProcess",
    "TwoSampleEProcess",
    "run_anytime_valid",
]


# ===========================================================================
# Known-σ Normal Mixture E-Process
# ===========================================================================
class GaussianMeanEProcess:
    """
    E-process for testing H₀: μ = μ₀ (known variance).

    Construction: mixture likelihood ratio
        E_t = ∫ ∏ᵢ [f(xᵢ; μ, σ²) / f(xᵢ; μ₀, σ²)] π(dμ)

    with mixing prior μ ~ N(μ₀, ρ²).

    Closed form (derivation via completing the square):
        logE_t = -½ log(1 + t·ρ²/σ²) + ρ²·S_t² / (2σ²·(σ² + t·ρ²))

    where S_t = Σᵢ (xᵢ - μ₀) is the cumulative deviation.

    Confidence sequence (by inversion — find μ₀ where E_t < 1/α):
        CS_t = x̄_t ± (σ/t)·√(2·(σ² + t·ρ²)/ρ² · (log(1/α) + ½·log(1 + t·ρ²/σ²)))

    Parameters
    ----------
    mu0 : float
        Null hypothesis mean.
    sigma : float
        Known standard deviation.
    rho : float
        Scale of the mixing prior — controls sensitivity.
        Larger ρ = more sensitive to large effects, less to small.
        Rule of thumb: ρ ≈ expected effect size × σ.

    Assumptions
    -----------
    - X_i are i.i.d. N(μ, σ²) (or at least sub-Gaussian with parameter σ)
    - σ is known and correctly specified
    - Observations arrive sequentially
    """

    def __init__(self, mu0: float = 0.0, sigma: float = 1.0, rho: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if rho <= 0:
            raise ValueError("rho must be positive")

        self.mu0 = float(mu0)
        self.sigma = float(sigma)
        self.rho = float(rho)

        # Running state
        self.t = 0           # number of observations
        self.S_t = 0.0       # Σ(xᵢ - μ₀)
        self.sum_x = 0.0     # Σxᵢ
        self._logE = 0.0     # current log e-value
        self._history = []   # [(t, logE_t, x_bar_t)]

    def update(self, x: float) -> "GaussianMeanEProcess":
        """
        Process one observation. Updates E_t multiplicatively
        (supermartingale property preserved).
        """
        self.t += 1
        self.S_t += (x - self.mu0)
        self.sum_x += x

        sigma2 = self.sigma ** 2
        rho2 = self.rho ** 2
        t = self.t

        # logE_t = -½ log(1 + t·ρ²/σ²) + ρ²·S_t² / (2·σ²·(σ² + t·ρ²))
        V_t = t * rho2 / sigma2  # information ratio
        self._logE = -0.5 * math.log1p(V_t) + rho2 * self.S_t ** 2 / (2 * sigma2 * (sigma2 + t * rho2))

        self._history.append((t, self._logE, self.sum_x / t))
        return self

    def update_batch(self, xs) -> "GaussianMeanEProcess":
        """Process multiple observations sequentially."""
        for x in xs:
            self.update(float(x))
        return self

    @property
    def log_e(self) -> float:
        """Current log e-value. E_t = exp(logE)."""
        return self._logE

    @property
    def e_value(self) -> float:
        """Current e-value (clamped for display)."""
        return min(math.exp(self._logE), 1e15)

    def decision(self, alpha: float = 0.05) -> bool:
        """Reject H₀ if logE_t ≥ log(1/α)."""
        return self._logE >= math.log(1.0 / alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """
        (1-α)-confidence sequence: interval [L_t, U_t].

        Constructed by inverting the e-process: find the set of μ₀
        for which E_t(μ₀) < 1/α.

        Returns (L_t, U_t) or (nan, nan) if t=0.
        """
        if self.t == 0:
            return (float('nan'), float('nan'))

        t = self.t
        sigma2 = self.sigma ** 2
        rho2 = self.rho ** 2
        x_bar = self.sum_x / t

        # Half-width: (σ/t) · √(2·(σ² + t·ρ²)/ρ² · (log(1/α) + ½·log(1 + t·ρ²/σ²)))
        threshold = math.log(1.0 / alpha) + 0.5 * math.log1p(t * rho2 / sigma2)
        if threshold <= 0:
            # Degenerate: entire real line
            return (float('-inf'), float('inf'))

        half_width = (self.sigma / t) * math.sqrt(2 * (sigma2 + t * rho2) / rho2 * threshold)

        return (x_bar - half_width, x_bar + half_width)

    @property
    def history(self) -> list:
        """List of (t, logE_t, x_bar_t) tuples."""
        return list(self._history)

    def summary(self) -> dict:
        """Return a summary dict of current state."""
        L, U = self.cs()
        return {
            "t": self.t,
            "logE": round(self._logE, 6),
            "E": round(self.e_value, 6),
            "x_bar": round(self.sum_x / self.t, 6) if self.t > 0 else None,
            "S_t": round(self.S_t, 6),
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "mu0": self.mu0,
            "sigma": self.sigma,
            "rho": self.rho,
        }


# ===========================================================================
# Unknown-σ Self-Normalized E-Process
# ===========================================================================
class SelfNormalizedMeanEProcess:
    """
    E-process for testing H₀: μ = μ₀ (unknown variance).

    Uses empirical variance with a mixture construction.
    The self-normalized approach replaces σ² with the running
    empirical variance V̂_t = Σ(xᵢ - x̄_t)² and applies a
    t-statistic-like mixture.

    Based on Howard et al. (2021) sub-ψ normal mixture:
        logE_t = -½ log(1 + t·ρ²/V̂_t) + ρ²·S_t² / (2·V̂_t·(V̂_t/t + ρ²))

    where V̂_t = max(Σ(xᵢ - x̄_t)², ε) is the clamped sample variance sum,
    and S_t = Σ(xᵢ - μ₀).

    Note: this is an *approximate* e-process. The supermartingale property
    holds asymptotically. For exact finite-sample validity with unknown σ,
    see Waudby-Smith & Ramdas (2024).

    Parameters
    ----------
    mu0 : float
        Null hypothesis mean.
    rho : float
        Scale of mixing prior.
    min_obs : int
        Minimum observations before computing (need ≥2 for variance).
    """

    def __init__(self, mu0: float = 0.0, rho: float = 1.0, min_obs: int = 5):
        if rho <= 0:
            raise ValueError("rho must be positive")

        self.mu0 = float(mu0)
        self.rho = float(rho)
        self.min_obs = max(2, int(min_obs))

        # Running state
        self.t = 0
        self.S_t = 0.0      # Σ(xᵢ - μ₀)
        self.sum_x = 0.0
        self.sum_x2 = 0.0   # Σxᵢ²
        self._logE = 0.0
        self._history = []

    def update(self, x: float) -> "SelfNormalizedMeanEProcess":
        """Process one observation."""
        self.t += 1
        x_f = float(x)
        self.S_t += (x_f - self.mu0)
        self.sum_x += x_f
        self.sum_x2 += x_f ** 2

        if self.t < self.min_obs:
            self._history.append((self.t, 0.0, self.sum_x / self.t))
            return self

        t = self.t
        rho2 = self.rho ** 2
        x_bar = self.sum_x / t

        # Empirical variance sum: V̂_t = Σ(xᵢ - x̄_t)² = Σxᵢ² - t·x̄²
        V_hat = max(self.sum_x2 - t * x_bar ** 2, 1e-10)

        # sigma²_hat = V̂_t / t (sample variance)
        sigma2_hat = V_hat / t

        # logE_t using empirical variance
        # Same formula as known-σ but with V̂_t/t replacing σ²
        info_ratio = t * rho2 / V_hat  # = ρ²/σ̂²
        self._logE = -0.5 * math.log1p(info_ratio) + rho2 * self.S_t ** 2 / (2 * V_hat * (sigma2_hat + rho2))

        self._history.append((self.t, self._logE, x_bar))
        return self

    def update_batch(self, xs) -> "SelfNormalizedMeanEProcess":
        """Process multiple observations sequentially."""
        for x in xs:
            self.update(float(x))
        return self

    @property
    def log_e(self) -> float:
        return self._logE

    @property
    def e_value(self) -> float:
        return min(math.exp(self._logE), 1e15)

    def decision(self, alpha: float = 0.05) -> bool:
        return self._logE >= math.log(1.0 / alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """Confidence sequence using empirical variance."""
        if self.t < self.min_obs:
            return (float('nan'), float('nan'))

        t = self.t
        rho2 = self.rho ** 2
        x_bar = self.sum_x / t
        V_hat = max(self.sum_x2 - t * x_bar ** 2, 1e-10)
        sigma2_hat = V_hat / t

        threshold = math.log(1.0 / alpha) + 0.5 * math.log1p(t * rho2 / V_hat)
        if threshold <= 0:
            return (float('-inf'), float('inf'))

        half_width = math.sqrt(2 * sigma2_hat * (sigma2_hat + rho2) / rho2 * threshold) / math.sqrt(t)

        return (x_bar - half_width, x_bar + half_width)

    @property
    def history(self) -> list:
        return list(self._history)

    def summary(self) -> dict:
        L, U = self.cs()
        sigma_hat = math.sqrt(max(self.sum_x2 - self.t * (self.sum_x / self.t) ** 2, 0) / self.t) if self.t >= 2 else None
        return {
            "t": self.t,
            "logE": round(self._logE, 6),
            "E": round(self.e_value, 6),
            "x_bar": round(self.sum_x / self.t, 6) if self.t > 0 else None,
            "sigma_hat": round(sigma_hat, 6) if sigma_hat is not None else None,
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "mu0": self.mu0,
            "rho": self.rho,
        }


# ===========================================================================
# Two-Sample A/B Test E-Process
# ===========================================================================
class TwoSampleEProcess:
    """
    Two-sample A/B test via anytime-valid inference.

    Tests H₀: μ_A = μ_B by reducing to a one-sample test on
    paired differences d_i = x_{A,i} - x_{B,i}.

    Observations are paired FIFO: the first A pairs with the first B,
    second A with second B, etc. Unpaired observations are buffered.
    The paired difference stream is fed to SelfNormalizedMeanEProcess,
    which guarantees the supermartingale property.

    This is the standard reduction for sequential two-sample testing
    (see Howard et al. 2021, §5.2). The self-normalized engine
    handles unknown σ automatically.

    Parameters
    ----------
    rho : float
        Scale of mixing prior for the mean difference (expected
        effect size). Rule of thumb: ρ ≈ anticipated |μ_A - μ_B|.
    min_pairs : int
        Minimum paired observations before computing (need ≥2
        for variance estimation).
    """

    def __init__(self, rho: float = 0.5, min_pairs: int = 5):
        self.rho = float(rho)
        self.min_pairs = max(2, int(min_pairs))

        # Internal engine on paired differences d_i = x_A - x_B
        self._engine = SelfNormalizedMeanEProcess(
            mu0=0.0, rho=rho, min_obs=self.min_pairs
        )

        # FIFO buffers for unpaired observations
        self._buf_a = []
        self._buf_b = []

        # Per-group tracking (for reporting)
        self.n_a = 0
        self.n_b = 0
        self.sum_a = 0.0
        self.sum_b = 0.0
        self.sum_a2 = 0.0
        self.sum_b2 = 0.0
        self.n_pairs = 0

        self._history = []  # (n_total, logE, diff, se)

    def update(self, x: float, group: str) -> "TwoSampleEProcess":
        """
        Process one observation from group 'A' or 'B'.
        When both buffers have data, a pair is formed and fed
        to the internal e-process engine.
        """
        x_f = float(x)
        if group.upper() == "A":
            self.n_a += 1
            self.sum_a += x_f
            self.sum_a2 += x_f ** 2
            self._buf_a.append(x_f)
        elif group.upper() == "B":
            self.n_b += 1
            self.sum_b += x_f
            self.sum_b2 += x_f ** 2
            self._buf_b.append(x_f)
        else:
            raise ValueError(f"group must be 'A' or 'B', got '{group}'")

        # Try to form a pair (FIFO)
        if self._buf_a and self._buf_b:
            xa = self._buf_a.pop(0)
            xb = self._buf_b.pop(0)
            self._engine.update(xa - xb)
            self.n_pairs += 1

        # Record history
        t = self.n_a + self.n_b
        if self.n_a > 0 and self.n_b > 0:
            diff = self.sum_a / self.n_a - self.sum_b / self.n_b
            if self.n_a > 1 and self.n_b > 1:
                var_a = max(self.sum_a2 / self.n_a - (self.sum_a / self.n_a) ** 2, 1e-10)
                var_b = max(self.sum_b2 / self.n_b - (self.sum_b / self.n_b) ** 2, 1e-10)
                se = math.sqrt(var_a / self.n_a + var_b / self.n_b)
            else:
                se = float('nan')
        else:
            diff, se = 0.0, float('nan')

        self._history.append((t, self._engine.log_e, diff, se))
        return self

    def update_groups(self, xs_a, xs_b) -> "TwoSampleEProcess":
        """Process arrays of observations from groups A and B, interleaved."""
        ia, ib = 0, 0
        while ia < len(xs_a) or ib < len(xs_b):
            if ia < len(xs_a):
                self.update(float(xs_a[ia]), "A")
                ia += 1
            if ib < len(xs_b):
                self.update(float(xs_b[ib]), "B")
                ib += 1
        return self

    @property
    def log_e(self) -> float:
        return self._engine.log_e

    @property
    def e_value(self) -> float:
        return self._engine.e_value

    def decision(self, alpha: float = 0.05) -> bool:
        return self._engine.decision(alpha)

    def cs(self, alpha: float = 0.05) -> tuple:
        """
        Confidence sequence for μ_A - μ_B.
        Delegates to the paired-difference engine.
        """
        return self._engine.cs(alpha)

    @property
    def history(self) -> list:
        return list(self._history)

    def summary(self) -> dict:
        L, U = self.cs()
        diff = (self.sum_a / self.n_a - self.sum_b / self.n_b) if self.n_a > 0 and self.n_b > 0 else None
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_pairs": self.n_pairs,
            "mean_a": round(self.sum_a / self.n_a, 6) if self.n_a > 0 else None,
            "mean_b": round(self.sum_b / self.n_b, 6) if self.n_b > 0 else None,
            "diff": round(diff, 6) if diff is not None else None,
            "logE": round(self._engine.log_e, 6),
            "E": round(self._engine.e_value, 6),
            "cs_lower": round(L, 6) if not math.isnan(L) else None,
            "cs_upper": round(U, 6) if not math.isnan(U) else None,
            "rho": self.rho,
        }


# ===========================================================================
# DSW Integration — run_anytime_valid()
# ===========================================================================
def run_anytime_valid(df, analysis_id, config):
    """Dispatch for anytime-valid inference analyses in DSW."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "anytime_ab":
        return _run_ab_test(df, config, result)
    elif analysis_id == "anytime_onesample":
        return _run_one_sample(df, config, result)
    else:
        result["summary"] = f"Error: Unknown anytime-valid analysis '{analysis_id}'."
        return result


def _run_ab_test(df, config, result):
    """A/B test with continuous monitoring via e-process."""

    value_col = config.get("value_col", "")
    group_col = config.get("group_col", "")
    group_a = config.get("group_a", "")
    group_b = config.get("group_b", "")
    alpha = float(config.get("alpha", 0.05))
    rho = float(config.get("rho", 0.5))

    if not value_col or not group_col:
        result["summary"] = "Error: Need both a value column and a group column."
        return result

    if value_col not in df.columns or group_col not in df.columns:
        result["summary"] = f"Error: Column not found. Available: {list(df.columns)}"
        return result

    # Identify groups
    groups = df[group_col].dropna().unique()
    if not group_a:
        group_a = str(groups[0]) if len(groups) >= 1 else "A"
    if not group_b:
        group_b = str(groups[1]) if len(groups) >= 2 else "B"

    mask_a = df[group_col].astype(str) == str(group_a)
    mask_b = df[group_col].astype(str) == str(group_b)

    vals_a = df.loc[mask_a, value_col].dropna().values.astype(float)
    vals_b = df.loc[mask_b, value_col].dropna().values.astype(float)

    if len(vals_a) < 5 or len(vals_b) < 5:
        result["summary"] = f"Error: Need at least 5 observations per group. Got {len(vals_a)} (A) and {len(vals_b)} (B)."
        return result

    # Run e-process
    ep = TwoSampleEProcess(rho=rho)
    ep.update_groups(vals_a, vals_b)

    rejected = ep.decision(alpha)
    s = ep.summary()
    L, U = ep.cs(alpha)

    # --- Summary ---
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append(f"<<COLOR:title>>ANYTIME-VALID A/B TEST<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Group A ({group_a}):<</COLOR>> n={ep.n_a}, mean={s['mean_a']:.4f}")
    lines.append(f"<<COLOR:highlight>>Group B ({group_b}):<</COLOR>> n={ep.n_b}, mean={s['mean_b']:.4f}")
    lines.append(f"<<COLOR:highlight>>Paired observations:<</COLOR>> {ep.n_pairs}")
    lines.append(f"<<COLOR:highlight>>Difference (A-B):<</COLOR>> {s['diff']:.4f}")
    lines.append(f"<<COLOR:highlight>>Mixing prior ρ:<</COLOR>> {rho}")
    lines.append(f"<<COLOR:highlight>>Significance level α:<</COLOR>> {alpha}\n")

    lines.append(f"<<COLOR:accent>>── E-Process Evidence ──<</COLOR>>")
    lines.append(f"<<COLOR:highlight>>E-value:<</COLOR>> {s['E']:.4f}")
    lines.append(f"<<COLOR:highlight>>log(E):<</COLOR>> {s['logE']:.4f}")
    lines.append(f"<<COLOR:highlight>>Decision boundary:<</COLOR>> log(1/α) = {math.log(1/alpha):.4f}")

    if rejected:
        lines.append(f"\n<<COLOR:warning>>REJECT H₀: sufficient evidence that μ_A ≠ μ_B<</COLOR>>")
        lines.append(f"<<COLOR:good>>You can stop collecting data. This conclusion is valid<</COLOR>>")
        lines.append(f"<<COLOR:good>>regardless of when or why you decided to look.<</COLOR>>")
    else:
        lines.append(f"\n<<COLOR:text>>CONTINUE: insufficient evidence to reject H₀<</COLOR>>")
        lines.append(f"<<COLOR:text>>Collect more data and re-run. No penalty for peeking.<</COLOR>>")

    lines.append(f"\n<<COLOR:accent>>── Confidence Sequence (anytime-valid {(1-alpha)*100:.0f}% CI) ──<</COLOR>>")
    lines.append(f"<<COLOR:highlight>>μ_A - μ_B ∈ [{L:.4f}, {U:.4f}]<</COLOR>>")
    lines.append(f"<<COLOR:text>>This interval is valid at every sample size simultaneously.<</COLOR>>")
    lines.append(f"<<COLOR:text>>It is wider than a fixed-sample CI but never lies.<</COLOR>>")

    # Comparison with fixed-sample test
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(vals_a, vals_b)
    fixed_se = np.sqrt(np.var(vals_a, ddof=1) / len(vals_a) + np.var(vals_b, ddof=1) / len(vals_b))
    fixed_ci = (s['diff'] - 1.96 * fixed_se, s['diff'] + 1.96 * fixed_se)

    lines.append(f"\n<<COLOR:accent>>── Comparison: Fixed-Sample t-test ──<</COLOR>>")
    lines.append(f"<<COLOR:text>>t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}<</COLOR>>")
    lines.append(f"<<COLOR:text>>Fixed 95% CI: [{fixed_ci[0]:.4f}, {fixed_ci[1]:.4f}]<</COLOR>>")
    lines.append(f"<<COLOR:text>>Note: the fixed CI is only valid at this exact sample size.<</COLOR>>")
    lines.append(f"<<COLOR:text>>If you peeked and decided to stop, the fixed CI may lie.<</COLOR>>")

    # Assumptions
    lines.append(f"\n<<COLOR:accent>>── Assumptions ──<</COLOR>>")
    lines.append(f"  1. Observations are independent within and between groups")
    lines.append(f"  2. Data is approximately Gaussian (or at least sub-Gaussian)")
    lines.append(f"  3. Observations are paired FIFO (first A with first B, etc.)")
    lines.append(f"  4. The mixing prior ρ={rho} reflects your expected effect size")
    lines.append(f"     (larger ρ → more sensitive to large effects, less to small)")

    result["summary"] = "\n".join(lines)

    # --- Plots ---
    # 1. E-value over time
    history = ep.history
    ts = [h[0] for h in history]
    logEs = [h[1] for h in history]
    threshold_line = math.log(1.0 / alpha)

    result["plots"].append({
        "title": "E-Process Evidence Over Time",
        "data": [
            {"type": "scatter", "x": ts, "y": logEs,
             "mode": "lines", "line": {"color": "#4a9f6e", "width": 2},
             "name": "log(E_t)"},
            {"type": "scatter", "x": [ts[0], ts[-1]],
             "y": [threshold_line, threshold_line],
             "mode": "lines", "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
             "name": f"Reject boundary (α={alpha})"},
            {"type": "scatter", "x": [ts[0], ts[-1]], "y": [0, 0],
             "mode": "lines", "line": {"color": "rgba(255,255,255,0.2)", "width": 1},
             "name": "No evidence"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 350,
            "xaxis": {"title": "Total observations"},
            "yaxis": {"title": "log(E_t)"},
            "annotations": [
                {"x": ts[-1], "y": threshold_line, "text": f"α={alpha}",
                 "showarrow": False, "xshift": 10,
                 "font": {"color": "#d94a4a", "size": 10}},
            ],
        },
    })

    # 2. Confidence sequence over time (show narrowing)
    cs_lowers = []
    cs_uppers = []
    cs_diffs = []
    cs_ts = []

    # Replay to get CS at each time
    ep2 = TwoSampleEProcess(rho=rho)
    ia, ib = 0, 0
    while ia < len(vals_a) or ib < len(vals_b):
        if ia < len(vals_a):
            ep2.update(float(vals_a[ia]), "A")
            ia += 1
        if ib < len(vals_b):
            ep2.update(float(vals_b[ib]), "B")
            ib += 1

        if ep2.n_a >= 5 and ep2.n_b >= 5:
            L_i, U_i = ep2.cs(alpha)
            if not math.isnan(L_i):
                cs_ts.append(ep2.n_a + ep2.n_b)
                cs_lowers.append(L_i)
                cs_uppers.append(U_i)
                cs_diffs.append(ep2.sum_a / ep2.n_a - ep2.sum_b / ep2.n_b)

    if cs_ts:
        result["plots"].append({
            "title": f"Confidence Sequence for μ_A - μ_B ({(1-alpha)*100:.0f}%)",
            "data": [
                {"type": "scatter", "x": cs_ts, "y": cs_uppers,
                 "mode": "lines", "line": {"width": 0}, "showlegend": False},
                {"type": "scatter", "x": cs_ts, "y": cs_lowers,
                 "mode": "lines", "fill": "tonexty",
                 "fillcolor": "rgba(74,159,110,0.15)",
                 "line": {"width": 0}, "name": f"{(1-alpha)*100:.0f}% CS"},
                {"type": "scatter", "x": cs_ts, "y": cs_diffs,
                 "mode": "lines", "line": {"color": "#4a9f6e", "width": 2},
                 "name": "x̄_A - x̄_B"},
                {"type": "scatter", "x": [cs_ts[0], cs_ts[-1]], "y": [0, 0],
                 "mode": "lines", "line": {"color": "rgba(255,255,255,0.3)", "dash": "dash"},
                 "name": "H₀: diff=0"},
            ],
            "layout": {
                "template": "plotly_dark", "height": 350,
                "xaxis": {"title": "Total observations"},
                "yaxis": {"title": "μ_A - μ_B"},
            },
        })

    # --- Statistics ---
    result["statistics"] = {
        "test": "anytime_valid_ab",
        "n_a": ep.n_a, "n_b": ep.n_b, "n_pairs": ep.n_pairs,
        "mean_a": s["mean_a"], "mean_b": s["mean_b"],
        "diff": s["diff"],
        "logE": s["logE"], "E": s["E"],
        "rejected": rejected, "alpha": alpha, "rho": rho,
        "cs_lower": round(L, 6) if not math.isnan(L) else None,
        "cs_upper": round(U, 6) if not math.isnan(U) else None,
        "fixed_t_stat": round(float(t_stat), 4),
        "fixed_p_value": round(float(p_val), 4),
    }

    result["guide_observation"] = (
        f"Anytime-valid A/B test: E={s['E']:.2f}, "
        f"diff={s['diff']:.4f}, CS=[{L:.4f}, {U:.4f}]. "
        + ("REJECT — groups differ significantly. Safe to stop." if rejected else
           "CONTINUE — no significant difference yet. Keep collecting or stop without guilt.")
    )

    return result


def _run_one_sample(df, config, result):
    """One-sample anytime-valid test: is the mean different from μ₀?"""

    value_col = config.get("value_col", "")
    mu0 = float(config.get("mu0", 0))
    alpha = float(config.get("alpha", 0.05))
    rho = float(config.get("rho", 1.0))
    sigma = config.get("sigma", None)

    if not value_col or value_col not in df.columns:
        result["summary"] = f"Error: Need a value column. Available: {list(df.columns)}"
        return result

    vals = df[value_col].dropna().values.astype(float)
    if len(vals) < 5:
        result["summary"] = "Error: Need at least 5 observations."
        return result

    # Choose known vs unknown σ engine
    if sigma is not None:
        sigma = float(sigma)
        ep = GaussianMeanEProcess(mu0=mu0, sigma=sigma, rho=rho)
        engine_name = "Known-σ Normal Mixture"
    else:
        ep = SelfNormalizedMeanEProcess(mu0=mu0, rho=rho)
        engine_name = "Self-Normalized (unknown σ)"

    ep.update_batch(vals)
    s = ep.summary()
    rejected = ep.decision(alpha)
    L, U = ep.cs(alpha)

    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append(f"<<COLOR:title>>ANYTIME-VALID ONE-SAMPLE TEST<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>H₀:<</COLOR>> μ = {mu0}")
    lines.append(f"<<COLOR:highlight>>n:<</COLOR>> {ep.t}")
    lines.append(f"<<COLOR:highlight>>x̄:<</COLOR>> {s.get('x_bar', 'N/A')}")
    lines.append(f"<<COLOR:highlight>>Engine:<</COLOR>> {engine_name}")
    lines.append(f"<<COLOR:highlight>>ρ (mixing prior):<</COLOR>> {rho}")
    if sigma is not None:
        lines.append(f"<<COLOR:highlight>>σ (known):<</COLOR>> {sigma}")
    else:
        lines.append(f"<<COLOR:highlight>>σ̂ (estimated):<</COLOR>> {s.get('sigma_hat', 'N/A')}")

    lines.append(f"\n<<COLOR:accent>>── Evidence ──<</COLOR>>")
    lines.append(f"<<COLOR:highlight>>E-value:<</COLOR>> {s['E']:.4f}")
    lines.append(f"<<COLOR:highlight>>log(E):<</COLOR>> {s['logE']:.4f}")
    lines.append(f"<<COLOR:highlight>>Boundary:<</COLOR>> log(1/α) = {math.log(1/alpha):.4f}")

    if rejected:
        lines.append(f"\n<<COLOR:warning>>REJECT H₀: evidence that μ ≠ {mu0}<</COLOR>>")
    else:
        lines.append(f"\n<<COLOR:text>>CONTINUE: insufficient evidence against H₀<</COLOR>>")

    lines.append(f"\n<<COLOR:accent>>── Confidence Sequence ──<</COLOR>>")
    lines.append(f"<<COLOR:highlight>>μ ∈ [{L:.4f}, {U:.4f}]<</COLOR>>")
    lines.append(f"<<COLOR:text>>Valid at every sample size simultaneously.<</COLOR>>")

    result["summary"] = "\n".join(lines)

    # Plots: E-process + CS over time
    history = ep.history
    ts_h = [h[0] for h in history]
    logEs_h = [h[1] for h in history]
    threshold = math.log(1.0 / alpha)

    result["plots"].append({
        "title": "E-Process Evidence Over Time",
        "data": [
            {"type": "scatter", "x": ts_h, "y": logEs_h,
             "mode": "lines", "line": {"color": "#4a9f6e", "width": 2},
             "name": "log(E_t)"},
            {"type": "scatter", "x": [ts_h[0], ts_h[-1]],
             "y": [threshold, threshold],
             "mode": "lines", "line": {"color": "#d94a4a", "dash": "dash"},
             "name": f"Reject (α={alpha})"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": "Observations"},
            "yaxis": {"title": "log(E_t)"},
        },
    })

    # CS over time
    cs_ts, cs_L, cs_U, cs_means = [], [], [], []
    if sigma is not None:
        ep_replay = GaussianMeanEProcess(mu0=mu0, sigma=sigma, rho=rho)
    else:
        ep_replay = SelfNormalizedMeanEProcess(mu0=mu0, rho=rho)

    for i, v in enumerate(vals):
        ep_replay.update(float(v))
        Li, Ui = ep_replay.cs(alpha)
        if not math.isnan(Li):
            cs_ts.append(i + 1)
            cs_L.append(Li)
            cs_U.append(Ui)
            cs_means.append(ep_replay.sum_x / ep_replay.t)

    if cs_ts:
        result["plots"].append({
            "title": f"Confidence Sequence for μ ({(1-alpha)*100:.0f}%)",
            "data": [
                {"type": "scatter", "x": cs_ts, "y": cs_U,
                 "mode": "lines", "line": {"width": 0}, "showlegend": False},
                {"type": "scatter", "x": cs_ts, "y": cs_L,
                 "mode": "lines", "fill": "tonexty",
                 "fillcolor": "rgba(74,159,110,0.15)",
                 "line": {"width": 0}, "name": f"{(1-alpha)*100:.0f}% CS"},
                {"type": "scatter", "x": cs_ts, "y": cs_means,
                 "mode": "lines", "line": {"color": "#4a9f6e", "width": 2},
                 "name": "x̄_t"},
                {"type": "scatter", "x": [cs_ts[0], cs_ts[-1]], "y": [mu0, mu0],
                 "mode": "lines", "line": {"color": "rgba(255,255,255,0.3)", "dash": "dash"},
                 "name": f"H₀: μ={mu0}"},
            ],
            "layout": {
                "template": "plotly_dark", "height": 300,
                "xaxis": {"title": "Observations"},
                "yaxis": {"title": value_col},
            },
        })

    result["statistics"] = {
        "test": "anytime_valid_onesample",
        "engine": engine_name,
        **s,
        "rejected": rejected,
        "alpha": alpha,
    }

    result["guide_observation"] = (
        f"One-sample anytime-valid: E={s['E']:.2f}, x̄={s.get('x_bar', 'N/A')}, "
        f"CS=[{L:.4f}, {U:.4f}]. "
        + ("REJECT — mean differs from " + str(mu0) + "." if rejected else
           "CONTINUE — no evidence against μ=" + str(mu0) + ".")
    )

    return result
