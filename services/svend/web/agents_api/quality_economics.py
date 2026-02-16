"""
Decision-Theoretic Quality Economics.

Bayesian decision theory meets Taguchi loss functions for optimal quality
decisions under uncertainty.  Instead of "is the process in control?",
answers "what is the expected cost of each possible action?"

Modules
-------
1.  TaguchiLoss        — quadratic loss functions (NIB, STB, LTB, asymmetric)
2.  ProcessDecision    — Bayesian optimal SPC action (continue/investigate/adjust)
3.  AcceptanceDecision — economic lot sentencing (accept/reject/screen)
4.  CostOfQuality      — CoQ breakdown with Bayesian uncertainty
5.  run_quality_econ() — DSW integration dispatcher

Dependencies: numpy only.
"""

import math
import numpy as np

__all__ = [
    "TaguchiLoss",
    "ProcessDecision",
    "AcceptanceDecision",
    "CostOfQuality",
    "run_quality_econ",
]


# ===========================================================================
# 1. Taguchi Loss Functions
# ===========================================================================
class TaguchiLoss:
    """
    Quadratic loss L(y) = k (y - T)^2 for nominal-is-best (NIB).

    Variants:
        NIB:  L = k (y - T)^2           k = A_0 / delta_0^2
        STB:  L = k y^2                 (target = 0)
        LTB:  L = k / y^2              (target = ∞)
        ASYM: L = k1 (y-T)^2 [y<T],  k2 (y-T)^2 [y≥T]

    Parameters
    ----------
    loss_type : str
        "nib", "stb", "ltb", or "asymmetric"
    target : float
        Target value (NIB/ASYM).  Ignored for STB/LTB.
    delta0 : float
        Customer tolerance (functional limit from target).
    cost_at_limit : float
        Loss incurred when y = T ± delta0 (e.g. warranty cost A_0).
    k_low, k_high : float, optional
        For asymmetric — separate k below / above target.
    """

    def __init__(self, loss_type="nib", target=0.0, delta0=1.0,
                 cost_at_limit=100.0, k_low=None, k_high=None):
        self.loss_type = loss_type.lower()
        self.target = target
        self.delta0 = delta0
        self.cost_at_limit = cost_at_limit

        if self.loss_type == "asymmetric":
            if k_low is None or k_high is None:
                raise ValueError("asymmetric requires k_low and k_high")
            self.k_low = k_low
            self.k_high = k_high
        else:
            self.k = cost_at_limit / (delta0 ** 2)

    def loss(self, y):
        """Point loss L(y)."""
        y = np.asarray(y, dtype=float)
        if self.loss_type == "nib":
            return self.k * (y - self.target) ** 2
        elif self.loss_type == "stb":
            return self.k * y ** 2
        elif self.loss_type == "ltb":
            return np.where(np.abs(y) < 1e-12, np.inf,
                            self.k / y ** 2)
        elif self.loss_type == "asymmetric":
            d = y - self.target
            return np.where(d < 0,
                            self.k_low * d ** 2,
                            self.k_high * d ** 2)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def expected_loss(self, mu, sigma):
        """
        Expected loss E[L(Y)] when Y ~ N(mu, sigma^2).

        NIB:  E[L] = k [sigma^2 + (mu - T)^2]
        STB:  E[L] = k [sigma^2 + mu^2]
        ASYM: Computed via split Gaussian integral.
        LTB:  Approximation for mu >> sigma.
        """
        if self.loss_type == "nib":
            return self.k * (sigma ** 2 + (mu - self.target) ** 2)
        elif self.loss_type == "stb":
            return self.k * (sigma ** 2 + mu ** 2)
        elif self.loss_type == "ltb":
            # E[1/Y^2] ≈ 1/mu^2 + 3*sigma^2/mu^4 for mu >> sigma
            if abs(mu) < 1e-8:
                return float("inf")
            return self.k * (1.0 / mu ** 2 + 3 * sigma ** 2 / mu ** 4)
        elif self.loss_type == "asymmetric":
            return self._expected_loss_asym(mu, sigma)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _expected_loss_asym(self, mu, sigma):
        """E[L] for asymmetric quadratic via split normal integral."""
        from scipy.stats import norm
        T = self.target
        z = (T - mu) / sigma if sigma > 1e-12 else float("inf")
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)

        # E[(Y-T)^2 | Y<T] * P(Y<T) + E[(Y-T)^2 | Y≥T] * P(Y≥T)
        # E[(Y-T)^2] = sigma^2 + (mu-T)^2
        # Split: integral_{-inf}^{T} (y-T)^2 f(y) dy
        var = sigma ** 2
        bias2 = (mu - T) ** 2

        # Lower part:  P(Y<T) * [var_trunc_low + bias_trunc_low^2]
        # For Gaussian, E[(Y-T)^2 * 1(Y<T)] = var*Phi(z) + (mu-T)^2*Phi(z)
        #   - sigma*(mu-T)*phi(z) — correction term
        # Actually, use direct integration result:
        # E[(Y-T)^2 1(Y<T)] = sigma^2 Phi(z) + (mu-T)^2 Phi(z)
        #                      - sigma (mu-T) phi(z)
        # (derived via completing the square)
        # But sign needs care. Let d = mu - T.
        d = mu - T
        E_low = var * Phi_z + d ** 2 * Phi_z - sigma * d * phi_z
        E_high = var * (1 - Phi_z) + d ** 2 * (1 - Phi_z) + sigma * d * phi_z

        return self.k_low * E_low + self.k_high * E_high

    def to_dict(self):
        d = {
            "type": self.loss_type,
            "target": self.target,
            "delta0": self.delta0,
            "cost_at_limit": self.cost_at_limit,
        }
        if self.loss_type == "asymmetric":
            d["k_low"] = self.k_low
            d["k_high"] = self.k_high
        else:
            d["k"] = self.k
        return d


# ===========================================================================
# 2. Process Decision — Bayesian Optimal SPC Action
# ===========================================================================
class ProcessDecision:
    """
    Bayesian decision for SPC: given posterior belief about process state,
    find the action minimizing expected cost.

    States:
        θ=0  process in control (IC)
        θ=1  process out of control (OOC)

    Actions:
        a=0  Continue (do nothing)
        a=1  Investigate (search for assignable cause)
        a=2  Adjust (reset process to target)

    Loss matrix L(a, θ):
        L(continue, IC)    = 0          (correct non-action)
        L(continue, OOC)   = C_miss     (cost per unit of missed defect)
        L(investigate, IC)  = C_fa       (false alarm investigation cost)
        L(investigate, OOC) = C_inv      (investigation cost, less than C_miss)
        L(adjust, IC)       = C_over     (unnecessary adjustment cost)
        L(adjust, OOC)      = C_adj      (adjustment cost — resets process)
    """

    def __init__(self, c_miss=500.0, c_fa=100.0, c_inv=80.0,
                 c_over=120.0, c_adj=150.0):
        self.c_miss = c_miss
        self.c_fa = c_fa
        self.c_inv = c_inv
        self.c_over = c_over
        self.c_adj = c_adj

        # Loss matrix: rows = actions, cols = states (IC, OOC)
        self.L = np.array([
            [0.0, c_miss],       # continue
            [c_fa, c_inv],       # investigate
            [c_over, c_adj],     # adjust
        ])
        self.action_names = ["Continue", "Investigate", "Adjust"]

    def optimal_action(self, p_ooc):
        """
        Given P(OOC|data), return optimal action and expected costs.

        Parameters
        ----------
        p_ooc : float
            Posterior probability that process is out of control.

        Returns
        -------
        dict with keys: action, action_name, expected_costs, p_ooc,
                        thresholds
        """
        p_ic = 1.0 - p_ooc
        expected = self.L @ np.array([p_ic, p_ooc])

        best = int(np.argmin(expected))

        # Decision boundaries (p_ooc thresholds where optimal action changes)
        # Continue vs Investigate: 0*(1-p) + c_miss*p = c_fa*(1-p) + c_inv*p
        #   → p = c_fa / (c_fa + c_miss - c_inv)
        denom_ci = self.c_fa + self.c_miss - self.c_inv
        thresh_ci = self.c_fa / denom_ci if denom_ci > 0 else 0.5

        # Continue vs Adjust: c_miss*p = c_over*(1-p) + c_adj*p
        #   → p = c_over / (c_over + c_miss - c_adj)
        denom_ca = self.c_over + self.c_miss - self.c_adj
        thresh_ca = self.c_over / denom_ca if denom_ca > 0 else 0.5

        # Investigate vs Adjust: c_fa*(1-p) + c_inv*p = c_over*(1-p) + c_adj*p
        #   → p = (c_over - c_fa) / (c_over - c_fa + c_inv - c_adj)
        denom_ia = (self.c_over - self.c_fa) + (self.c_inv - self.c_adj)
        thresh_ia = (self.c_over - self.c_fa) / denom_ia if abs(denom_ia) > 1e-10 else 0.5

        return {
            "action": best,
            "action_name": self.action_names[best],
            "expected_costs": {
                name: float(ec)
                for name, ec in zip(self.action_names, expected)
            },
            "p_ooc": float(p_ooc),
            "thresholds": {
                "continue_vs_investigate": float(thresh_ci),
                "continue_vs_adjust": float(thresh_ca),
                "investigate_vs_adjust": float(thresh_ia),
            },
            "cost_savings": float(np.max(expected) - np.min(expected)),
        }

    def sweep(self, n_points=200):
        """Expected cost curves over P(OOC) ∈ [0, 1]."""
        ps = np.linspace(0, 1, n_points)
        costs = np.array([self.L @ np.array([1 - p, p]) for p in ps])
        optimal = np.argmin(costs, axis=1)
        return {
            "p_ooc": ps.tolist(),
            "continue": costs[:, 0].tolist(),
            "investigate": costs[:, 1].tolist(),
            "adjust": costs[:, 2].tolist(),
            "optimal_action": optimal.tolist(),
        }


# ===========================================================================
# 3. Acceptance Decision — Economic Lot Sentencing
# ===========================================================================
class AcceptanceDecision:
    """
    Economic lot sentencing under uncertainty.

    Given posterior belief about lot defect rate p, compute expected cost
    of Accept / Reject / 100% Screen.

    Costs:
        C_accept(p)  = N * p * C_ext     (defectives reach customer)
        C_reject     = C_rej              (scrap/return entire lot)
        C_screen     = N * C_insp + N * p * C_int  (inspect all, catch internals)
    """

    def __init__(self, lot_size=1000, c_external=50.0, c_internal=5.0,
                 c_inspection=0.5, c_reject_lot=200.0):
        self.N = lot_size
        self.c_ext = c_external
        self.c_int = c_internal
        self.c_insp = c_inspection
        self.c_rej = c_reject_lot

    def expected_costs(self, p_defect):
        """
        Expected cost for each action given defect rate p.

        Parameters
        ----------
        p_defect : float or array
            Posterior expected defect rate(s).
        """
        p = np.asarray(p_defect, dtype=float)
        N = self.N

        cost_accept = N * p * self.c_ext
        cost_reject = np.full_like(p, self.c_rej)
        cost_screen = N * self.c_insp + N * p * self.c_int

        return {
            "accept": cost_accept,
            "reject": cost_reject,
            "screen": cost_screen,
        }

    def optimal_action(self, p_defect):
        """Optimal lot decision given posterior defect rate."""
        costs = self.expected_costs(p_defect)
        cost_arr = np.array([
            float(np.mean(costs["accept"])),
            float(np.mean(costs["reject"])),
            float(np.mean(costs["screen"])),
        ])
        names = ["Accept", "Reject", "100% Screen"]
        best = int(np.argmin(cost_arr))

        # Breakeven points
        # Accept = Reject:  N*p*c_ext = c_rej  →  p = c_rej/(N*c_ext)
        p_accept_reject = self.c_rej / (self.N * self.c_ext) if self.c_ext > 0 else 1.0
        # Accept = Screen:  N*p*c_ext = N*c_insp + N*p*c_int
        #   → p = c_insp / (c_ext - c_int)  [if c_ext > c_int]
        denom = self.c_ext - self.c_int
        p_accept_screen = self.c_insp / denom if denom > 0 else 1.0

        return {
            "action": best,
            "action_name": names[best],
            "expected_costs": {
                "Accept": float(cost_arr[0]),
                "Reject": float(cost_arr[1]),
                "100% Screen": float(cost_arr[2]),
            },
            "p_defect": float(np.mean(p_defect)),
            "breakeven": {
                "accept_vs_reject": float(min(p_accept_reject, 1.0)),
                "accept_vs_screen": float(min(p_accept_screen, 1.0)),
            },
            "cost_savings": float(np.max(cost_arr) - np.min(cost_arr)),
        }

    def sweep(self, n_points=200):
        """Cost curves over p ∈ [0, max_p]."""
        max_p = min(1.0, 3 * self.c_rej / (self.N * self.c_ext)
                    if self.c_ext > 0 else 0.1)
        max_p = max(max_p, 0.02)
        ps = np.linspace(0, max_p, n_points)
        costs = self.expected_costs(ps)
        return {
            "p_defect": ps.tolist(),
            "accept": costs["accept"].tolist(),
            "reject": costs["reject"].tolist(),
            "screen": costs["screen"].tolist(),
        }


# ===========================================================================
# 4. Cost of Quality (CoQ)
# ===========================================================================
class CostOfQuality:
    """
    PAF model: Prevention + Appraisal + (Internal + External) Failure.

    Given cost data, computes ratios, optimal prevention investment,
    and total quality cost with uncertainty bands.
    """

    def __init__(self, prevention=0.0, appraisal=0.0,
                 internal_failure=0.0, external_failure=0.0,
                 revenue=1.0):
        self.prevention = prevention
        self.appraisal = appraisal
        self.internal_failure = internal_failure
        self.external_failure = external_failure
        self.revenue = max(revenue, 1.0)

    @property
    def total(self):
        return (self.prevention + self.appraisal +
                self.internal_failure + self.external_failure)

    @property
    def conformance_cost(self):
        """Cost of conformance (prevention + appraisal)."""
        return self.prevention + self.appraisal

    @property
    def nonconformance_cost(self):
        """Cost of nonconformance (internal + external failure)."""
        return self.internal_failure + self.external_failure

    def summary(self):
        t = self.total
        rev = self.revenue
        return {
            "prevention": self.prevention,
            "appraisal": self.appraisal,
            "internal_failure": self.internal_failure,
            "external_failure": self.external_failure,
            "total_coq": t,
            "conformance": self.conformance_cost,
            "nonconformance": self.nonconformance_cost,
            "coq_pct_revenue": t / rev * 100 if rev > 0 else 0,
            "conformance_ratio": (self.conformance_cost / t * 100
                                  if t > 0 else 0),
            "nonconformance_ratio": (self.nonconformance_cost / t * 100
                                     if t > 0 else 0),
            "prevention_pct": self.prevention / t * 100 if t > 0 else 0,
            "appraisal_pct": self.appraisal / t * 100 if t > 0 else 0,
            "int_failure_pct": (self.internal_failure / t * 100
                                if t > 0 else 0),
            "ext_failure_pct": (self.external_failure / t * 100
                                if t > 0 else 0),
        }

    def optimal_prevention_model(self, n_points=50):
        """
        Simple economic model: as prevention spending increases,
        failure costs decrease (exponential decay).

        Returns cost curves for plotting.
        """
        base_failure = self.nonconformance_cost
        base_prevention = max(self.prevention, 1.0)
        base_appraisal = self.appraisal

        # Model: failure(p) = base_failure * exp(-λ * (p/base_prevention - 1))
        # λ calibrated so that doubling prevention halves failure
        lam = math.log(2)

        prev_range = np.linspace(0.1 * base_prevention,
                                 5.0 * base_prevention, n_points)
        failure_curve = base_failure * np.exp(
            -lam * (prev_range / base_prevention - 1))
        appraisal_curve = np.full(n_points, base_appraisal)
        total_curve = prev_range + appraisal_curve + failure_curve

        opt_idx = int(np.argmin(total_curve))

        return {
            "prevention": prev_range.tolist(),
            "failure": failure_curve.tolist(),
            "appraisal": appraisal_curve.tolist(),
            "total": total_curve.tolist(),
            "optimal_prevention": float(prev_range[opt_idx]),
            "optimal_total": float(total_curve[opt_idx]),
            "current_prevention": float(base_prevention),
            "current_total": float(self.total),
        }


# ===========================================================================
# 5. DSW Integration
# ===========================================================================
def run_quality_econ(df, analysis_id, config):
    """
    Dispatcher for Decision-Theoretic Quality Economics.

    analysis_id values:
        "taguchi_loss"       — Taguchi loss function analysis
        "process_decision"   — Bayesian optimal SPC action
        "lot_sentencing"     — Economic acceptance decision
        "cost_of_quality"    — PAF cost breakdown
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "taguchi_loss":
        result = _run_taguchi(df, config)
    elif analysis_id == "process_decision":
        result = _run_process_decision(df, config)
    elif analysis_id == "lot_sentencing":
        result = _run_lot_sentencing(df, config)
    elif analysis_id == "cost_of_quality":
        result = _run_coq(df, config)
    else:
        result["summary"] = f"Error: Unknown quality economics analysis: {analysis_id}"

    return result


def _run_taguchi(df, config):
    """Taguchi loss function on data column."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    col = config.get("column", "")
    loss_type = config.get("loss_type", "nib")
    target = float(config.get("target", 0))
    delta0 = float(config.get("delta0", 1))
    cost_at_limit = float(config.get("cost_at_limit", 100))

    if not col or col not in df.columns:
        result["summary"] = "Error: Select a measurement column."
        return result

    y = df[col].dropna().values.astype(float)
    if len(y) < 2:
        result["summary"] = "Error: Need at least 2 observations."
        return result

    loss_fn = TaguchiLoss(loss_type=loss_type, target=target,
                          delta0=delta0, cost_at_limit=cost_at_limit)
    mu, sigma = float(np.mean(y)), float(np.std(y, ddof=1))
    losses = loss_fn.loss(y)
    expected = loss_fn.expected_loss(mu, sigma)
    total_loss = float(np.sum(losses))
    avg_loss = float(np.mean(losses))

    # Summary
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>TAGUCHI LOSS FUNCTION ANALYSIS<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Column:<</COLOR>> {col}  "
                 f"(n = {len(y)})")
    lines.append(f"<<COLOR:highlight>>Loss type:<</COLOR>> "
                 f"{loss_type.upper()}")
    lines.append(f"<<COLOR:highlight>>Target:<</COLOR>> {target}  "
                 f"Δ₀ = {delta0}  A₀ = {cost_at_limit}")
    lines.append(f"<<COLOR:highlight>>k:<</COLOR>> "
                 f"{cost_at_limit / delta0**2:.4f}")

    lines.append(f"\n<<COLOR:accent>>── Process State ──<</COLOR>>")
    lines.append(f"  Mean (μ):              {mu:.4f}")
    lines.append(f"  Std Dev (σ):           {sigma:.4f}")
    lines.append(f"  Bias (μ − T):          {mu - target:.4f}")

    lines.append(f"\n<<COLOR:accent>>── Loss Summary ──<</COLOR>>")
    lines.append(f"  Expected loss E[L]:    <<COLOR:warning>>"
                 f"${expected:.2f}<</COLOR>> per unit")
    lines.append(f"  Average observed loss: ${avg_loss:.2f} per unit")
    lines.append(f"  Total observed loss:   ${total_loss:.2f}  "
                 f"(n = {len(y)} units)")

    # Decomposition: E[L] = k*sigma^2 + k*(mu-T)^2
    if loss_type in ("nib", "stb"):
        k = loss_fn.k
        var_component = k * sigma ** 2
        bias_component = k * (mu - target) ** 2
        pct_var = var_component / expected * 100 if expected > 0 else 0
        pct_bias = bias_component / expected * 100 if expected > 0 else 0
        lines.append(f"\n<<COLOR:accent>>── Loss Decomposition ──<</COLOR>>")
        lines.append(f"  Variance component:    ${var_component:.2f}  "
                     f"({pct_var:.0f}%)")
        lines.append(f"  Bias component:        ${bias_component:.2f}  "
                     f"({pct_bias:.0f}%)")
        if pct_bias > 60:
            lines.append(f"  <<COLOR:warning>>→ Dominant loss from bias — "
                         f"center the process on target<</COLOR>>")
        elif pct_var > 60:
            lines.append(f"  <<COLOR:warning>>→ Dominant loss from variance "
                         f"— reduce process variability<</COLOR>>")

    # What-if: if we center the process
    expected_centered = loss_fn.expected_loss(target, sigma)
    savings_center = expected - expected_centered
    if savings_center > 0.01:
        lines.append(f"\n<<COLOR:accent>>── What-If ──<</COLOR>>")
        lines.append(f"  If centered on target: ${expected_centered:.2f} "
                     f"per unit")
        lines.append(f"  <<COLOR:good>>Savings from centering: "
                     f"${savings_center:.2f} per unit<</COLOR>>")

    # What-if: halve sigma
    expected_half_sigma = loss_fn.expected_loss(mu, sigma / 2)
    savings_sigma = expected - expected_half_sigma
    if savings_sigma > 0.01:
        lines.append(f"  If σ halved:           ${expected_half_sigma:.2f} "
                     f"per unit")
        lines.append(f"  <<COLOR:good>>Savings from σ reduction: "
                     f"${savings_sigma:.2f} per unit<</COLOR>>")

    result["summary"] = "\n".join(lines)

    # Plot 1: Loss function curve + data
    y_sorted = np.sort(y)
    y_range = np.linspace(min(y.min(), target - 2 * delta0),
                          max(y.max(), target + 2 * delta0), 200)
    loss_curve = loss_fn.loss(y_range)

    result["plots"].append({
        "title": "Taguchi Loss Function",
        "data": [
            {"type": "scatter", "x": y_range.tolist(),
             "y": loss_curve.tolist(),
             "mode": "lines", "name": "L(y)",
             "line": {"color": "#d94a4a", "width": 2}},
            {"type": "scatter", "x": y.tolist(),
             "y": losses.tolist(),
             "mode": "markers", "name": "Observations",
             "marker": {"color": "#4a9f6e", "size": 4, "opacity": 0.5}},
            {"type": "scatter", "x": [target], "y": [0],
             "mode": "markers", "name": "Target",
             "marker": {"color": "#d4a24a", "size": 10, "symbol": "diamond"}},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": col},
            "yaxis": {"title": "Loss ($)"},
        },
    })

    # Plot 2: Loss histogram
    hist_vals, bin_edges = np.histogram(losses, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    result["plots"].append({
        "title": "Distribution of Per-Unit Loss",
        "data": [
            {"type": "bar", "x": bin_centers.tolist(),
             "y": hist_vals.tolist(),
             "marker": {"color": "rgba(217,74,74,0.5)",
                        "line": {"color": "#d94a4a", "width": 0.5}},
             "name": "Loss density"},
            {"type": "scatter",
             "x": [expected, expected],
             "y": [0, max(hist_vals) * 1.1],
             "mode": "lines",
             "line": {"color": "#d4a24a", "dash": "dash", "width": 2},
             "name": f"E[L] = ${expected:.2f}"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 250,
            "xaxis": {"title": "Loss per unit ($)"},
            "yaxis": {"title": "Density"},
        },
    })

    result["statistics"] = {
        "test": "taguchi_loss",
        "loss_type": loss_type,
        "target": target,
        "k": loss_fn.k if loss_type != "asymmetric" else None,
        "mu": mu,
        "sigma": sigma,
        "expected_loss": round(expected, 4),
        "avg_observed_loss": round(avg_loss, 4),
        "total_loss": round(total_loss, 2),
        "n": len(y),
    }

    result["guide_observation"] = (
        f"Taguchi {loss_type.upper()} loss: E[L] = ${expected:.2f}/unit "
        f"(μ={mu:.3f}, σ={sigma:.3f}, T={target}). "
        f"Total loss = ${total_loss:.0f} over {len(y)} units."
    )

    return result


def _run_process_decision(df, config):
    """Bayesian optimal SPC action."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    p_ooc = float(config.get("p_ooc", 0.5))
    c_miss = float(config.get("c_miss", 500))
    c_fa = float(config.get("c_fa", 100))
    c_inv = float(config.get("c_inv", 80))
    c_over = float(config.get("c_over", 120))
    c_adj = float(config.get("c_adj", 150))

    pd_obj = ProcessDecision(c_miss=c_miss, c_fa=c_fa, c_inv=c_inv,
                             c_over=c_over, c_adj=c_adj)
    dec = pd_obj.optimal_action(p_ooc)
    sweep = pd_obj.sweep()

    # Summary
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>BAYESIAN PROCESS DECISION<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>P(Out of Control | data):<</COLOR>> "
                 f"{p_ooc:.1%}")

    color = {"Continue": "good", "Investigate": "highlight",
             "Adjust": "warning"}
    c = color.get(dec["action_name"], "highlight")
    lines.append(f"\n<<COLOR:{c}>>OPTIMAL ACTION: "
                 f"{dec['action_name'].upper()}<</COLOR>>")
    lines.append(f"  Expected cost: ${dec['expected_costs'][dec['action_name']]:.2f}")
    lines.append(f"  Savings vs worst: ${dec['cost_savings']:.2f}")

    lines.append(f"\n<<COLOR:accent>>── Expected Cost of Each Action ──<</COLOR>>")
    for name, cost in dec["expected_costs"].items():
        marker = " ← optimal" if name == dec["action_name"] else ""
        lines.append(f"  {name:<14}  ${cost:>10.2f}{marker}")

    lines.append(f"\n<<COLOR:accent>>── Decision Boundaries ──<</COLOR>>")
    t = dec["thresholds"]
    lines.append(f"  Continue → Investigate at P(OOC) = "
                 f"{t['continue_vs_investigate']:.1%}")
    lines.append(f"  Continue → Adjust at P(OOC) = "
                 f"{t['continue_vs_adjust']:.1%}")

    lines.append(f"\n<<COLOR:accent>>── Cost Parameters ──<</COLOR>>")
    lines.append(f"  C_miss (miss defect):       ${c_miss:.0f}")
    lines.append(f"  C_fa (false alarm):          ${c_fa:.0f}")
    lines.append(f"  C_inv (investigate, found):  ${c_inv:.0f}")
    lines.append(f"  C_over (unnecessary adjust): ${c_over:.0f}")
    lines.append(f"  C_adj (adjustment cost):     ${c_adj:.0f}")

    result["summary"] = "\n".join(lines)

    # Plot: Decision boundary curves
    colors = {"continue": "#4a9f6e", "investigate": "#d4a24a",
              "adjust": "#d94a4a"}
    traces = []
    for key in ["continue", "investigate", "adjust"]:
        traces.append({
            "type": "scatter",
            "x": sweep["p_ooc"],
            "y": sweep[key],
            "mode": "lines",
            "name": key.capitalize(),
            "line": {"color": colors[key], "width": 2},
        })

    # Current position marker
    traces.append({
        "type": "scatter",
        "x": [p_ooc],
        "y": [dec["expected_costs"][dec["action_name"]]],
        "mode": "markers",
        "name": f"Current: {dec['action_name']}",
        "marker": {"color": colors.get(dec["action_name"].lower(),
                                        "#ffffff"),
                    "size": 12, "symbol": "star"},
    })

    result["plots"].append({
        "title": "Expected Cost vs P(Out of Control)",
        "data": traces,
        "layout": {
            "template": "plotly_dark", "height": 350,
            "xaxis": {"title": "P(OOC | data)", "tickformat": ".0%"},
            "yaxis": {"title": "Expected Cost ($)"},
        },
    })

    result["statistics"] = {
        "test": "process_decision",
        "optimal_action": dec["action_name"],
        "p_ooc": p_ooc,
        "expected_costs": dec["expected_costs"],
        "thresholds": dec["thresholds"],
        "cost_savings": dec["cost_savings"],
    }

    result["guide_observation"] = (
        f"Bayesian process decision: at P(OOC)={p_ooc:.1%}, "
        f"optimal action is {dec['action_name']} "
        f"(${dec['expected_costs'][dec['action_name']]:.0f} expected). "
        f"Saves ${dec['cost_savings']:.0f} vs worst alternative."
    )

    return result


def _run_lot_sentencing(df, config):
    """Economic lot sentencing."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    p_defect = float(config.get("p_defect", 0.01))
    lot_size = int(config.get("lot_size", 1000))
    c_ext = float(config.get("c_external", 50))
    c_int = float(config.get("c_internal", 5))
    c_insp = float(config.get("c_inspection", 0.5))
    c_rej = float(config.get("c_reject_lot", 200))

    ad = AcceptanceDecision(lot_size=lot_size, c_external=c_ext,
                            c_internal=c_int, c_inspection=c_insp,
                            c_reject_lot=c_rej)
    dec = ad.optimal_action(p_defect)
    sweep = ad.sweep()

    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>ECONOMIC LOT SENTENCING<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Posterior defect rate:<</COLOR>> "
                 f"{p_defect:.2%}")
    lines.append(f"<<COLOR:highlight>>Lot size:<</COLOR>> {lot_size}")

    color = {"Accept": "good", "Reject": "warning",
             "100% Screen": "highlight"}
    c = color.get(dec["action_name"], "highlight")
    lines.append(f"\n<<COLOR:{c}>>OPTIMAL ACTION: "
                 f"{dec['action_name'].upper()}<</COLOR>>")
    lines.append(f"  Expected cost: "
                 f"${dec['expected_costs'][dec['action_name']]:.2f}")
    lines.append(f"  Savings vs worst: ${dec['cost_savings']:.2f}")

    lines.append(f"\n<<COLOR:accent>>── Expected Cost of Each Action ──<</COLOR>>")
    for name, cost in dec["expected_costs"].items():
        marker = " ← optimal" if name == dec["action_name"] else ""
        lines.append(f"  {name:<14}  ${cost:>10.2f}{marker}")

    lines.append(f"\n<<COLOR:accent>>── Breakeven Points ──<</COLOR>>")
    be = dec["breakeven"]
    lines.append(f"  Accept = Reject at p = {be['accept_vs_reject']:.4%}")
    lines.append(f"  Accept = Screen at p = {be['accept_vs_screen']:.4%}")

    lines.append(f"\n<<COLOR:accent>>── Cost Parameters ──<</COLOR>>")
    lines.append(f"  External failure:  ${c_ext:.2f} / defective unit")
    lines.append(f"  Internal failure:  ${c_int:.2f} / defective unit")
    lines.append(f"  Inspection:        ${c_insp:.2f} / unit inspected")
    lines.append(f"  Lot rejection:     ${c_rej:.2f} / lot")

    result["summary"] = "\n".join(lines)

    # Plot: Cost curves
    traces = [
        {"type": "scatter", "x": sweep["p_defect"],
         "y": sweep["accept"], "mode": "lines",
         "name": "Accept",
         "line": {"color": "#4a9f6e", "width": 2}},
        {"type": "scatter", "x": sweep["p_defect"],
         "y": sweep["reject"], "mode": "lines",
         "name": "Reject",
         "line": {"color": "#d94a4a", "width": 2}},
        {"type": "scatter", "x": sweep["p_defect"],
         "y": sweep["screen"], "mode": "lines",
         "name": "100% Screen",
         "line": {"color": "#d4a24a", "width": 2}},
        {"type": "scatter",
         "x": [p_defect],
         "y": [dec["expected_costs"][dec["action_name"]]],
         "mode": "markers",
         "name": f"Current: {dec['action_name']}",
         "marker": {"size": 12, "symbol": "star",
                    "color": "#ffffff"}},
    ]

    result["plots"].append({
        "title": "Lot Sentencing: Cost vs Defect Rate",
        "data": traces,
        "layout": {
            "template": "plotly_dark", "height": 350,
            "xaxis": {"title": "Defect Rate (p)",
                      "tickformat": ".1%"},
            "yaxis": {"title": "Expected Cost ($)"},
        },
    })

    result["statistics"] = {
        "test": "lot_sentencing",
        "optimal_action": dec["action_name"],
        "p_defect": p_defect,
        "lot_size": lot_size,
        "expected_costs": dec["expected_costs"],
        "breakeven": dec["breakeven"],
        "cost_savings": dec["cost_savings"],
    }

    result["guide_observation"] = (
        f"Economic lot sentencing: at p={p_defect:.2%} defect rate, "
        f"optimal action is {dec['action_name']} "
        f"(${dec['expected_costs'][dec['action_name']]:.0f}). "
        f"Saves ${dec['cost_savings']:.0f} vs worst alternative."
    )

    return result


def _run_coq(df, config):
    """Cost of Quality breakdown."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    prevention = float(config.get("prevention", 0))
    appraisal = float(config.get("appraisal", 0))
    int_failure = float(config.get("internal_failure", 0))
    ext_failure = float(config.get("external_failure", 0))
    revenue = float(config.get("revenue", 1))

    if prevention + appraisal + int_failure + ext_failure <= 0:
        result["summary"] = "Error: Enter at least one cost value."
        return result

    coq = CostOfQuality(prevention, appraisal, int_failure, ext_failure,
                        revenue)
    s = coq.summary()
    opt = coq.optimal_prevention_model()

    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>COST OF QUALITY (PAF MODEL)<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:accent>>── Cost Breakdown ──<</COLOR>>")
    lines.append(f"  Prevention:         ${prevention:>12,.0f}  "
                 f"({s['prevention_pct']:.1f}%)")
    lines.append(f"  Appraisal:          ${appraisal:>12,.0f}  "
                 f"({s['appraisal_pct']:.1f}%)")
    lines.append(f"  Internal Failure:   ${int_failure:>12,.0f}  "
                 f"({s['int_failure_pct']:.1f}%)")
    lines.append(f"  External Failure:   ${ext_failure:>12,.0f}  "
                 f"({s['ext_failure_pct']:.1f}%)")
    lines.append(f"  {'─' * 45}")
    lines.append(f"  <<COLOR:warning>>Total CoQ:          "
                 f"${s['total_coq']:>12,.0f}<</COLOR>>")
    lines.append(f"  <<COLOR:highlight>>CoQ / Revenue:      "
                 f"{s['coq_pct_revenue']:.1f}%<</COLOR>>")

    lines.append(f"\n<<COLOR:accent>>── Conformance vs Nonconformance ──<</COLOR>>")
    lines.append(f"  Conformance (P+A):     ${s['conformance']:>12,.0f}  "
                 f"({s['conformance_ratio']:.0f}%)")
    lines.append(f"  Nonconformance (F):    ${s['nonconformance']:>12,.0f}  "
                 f"({s['nonconformance_ratio']:.0f}%)")

    # Benchmark guidance
    coq_pct = s["coq_pct_revenue"]
    if coq_pct > 25:
        grade = "Crisis"
    elif coq_pct > 15:
        grade = "High"
    elif coq_pct > 5:
        grade = "Typical"
    else:
        grade = "World-class"

    lines.append(f"\n<<COLOR:accent>>── Assessment ──<</COLOR>>")
    lines.append(f"  Grade: <<COLOR:highlight>>{grade}<</COLOR>> "
                 f"(CoQ = {coq_pct:.1f}% of revenue)")
    lines.append(f"  Benchmark: world-class < 5%, "
                 f"typical 5-15%, high 15-25%")

    if s["nonconformance_ratio"] > 70:
        lines.append(f"  <<COLOR:warning>>→ Failure costs dominate — "
                     f"invest more in prevention<</COLOR>>")
    elif s["conformance_ratio"] > 80:
        lines.append(f"  <<COLOR:good>>→ Strong prevention focus — "
                     f"verify ROI of appraisal spend<</COLOR>>")

    lines.append(f"\n<<COLOR:accent>>── Optimal Prevention Model ──<</COLOR>>")
    lines.append(f"  Current prevention:  ${opt['current_prevention']:>12,.0f}")
    lines.append(f"  Optimal prevention:  ${opt['optimal_prevention']:>12,.0f}")
    lines.append(f"  Current total CoQ:   ${opt['current_total']:>12,.0f}")
    lines.append(f"  Optimal total CoQ:   ${opt['optimal_total']:>12,.0f}")
    saving = opt["current_total"] - opt["optimal_total"]
    if saving > 0:
        lines.append(f"  <<COLOR:good>>Potential savings:    "
                     f"${saving:>12,.0f}<</COLOR>>")

    result["summary"] = "\n".join(lines)

    # Plot 1: Pie chart
    result["plots"].append({
        "title": "Cost of Quality Breakdown",
        "data": [
            {"type": "pie",
             "labels": ["Prevention", "Appraisal",
                        "Internal Failure", "External Failure"],
             "values": [prevention, appraisal, int_failure, ext_failure],
             "marker": {"colors": ["#4a9f6e", "#6ab7d4",
                                   "#d4a24a", "#d94a4a"]},
             "hole": 0.4,
             "textinfo": "label+percent"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
        },
    })

    # Plot 2: Prevention optimization curve
    result["plots"].append({
        "title": "Prevention Investment vs Total CoQ",
        "data": [
            {"type": "scatter", "x": opt["prevention"],
             "y": opt["failure"], "mode": "lines",
             "name": "Failure Cost",
             "line": {"color": "#d94a4a", "width": 2}},
            {"type": "scatter", "x": opt["prevention"],
             "y": opt["total"], "mode": "lines",
             "name": "Total CoQ",
             "line": {"color": "#d4a24a", "width": 2.5}},
            {"type": "scatter",
             "x": [opt["optimal_prevention"]],
             "y": [opt["optimal_total"]],
             "mode": "markers", "name": "Optimum",
             "marker": {"color": "#4a9f6e", "size": 12,
                        "symbol": "star"}},
            {"type": "scatter",
             "x": [opt["current_prevention"]],
             "y": [opt["current_total"]],
             "mode": "markers", "name": "Current",
             "marker": {"color": "#d94a4a", "size": 10,
                        "symbol": "diamond"}},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": "Prevention Spending ($)"},
            "yaxis": {"title": "Cost ($)"},
        },
    })

    result["statistics"] = {
        "test": "cost_of_quality",
        "total_coq": round(s["total_coq"], 2),
        "coq_pct_revenue": round(coq_pct, 2),
        "grade": grade,
        "conformance_ratio": round(s["conformance_ratio"], 1),
        "nonconformance_ratio": round(s["nonconformance_ratio"], 1),
        "optimal_prevention": round(opt["optimal_prevention"], 2),
        "potential_savings": round(max(saving, 0), 2),
    }

    result["guide_observation"] = (
        f"Cost of Quality: ${s['total_coq']:,.0f} total "
        f"({coq_pct:.1f}% of revenue, grade: {grade}). "
        f"Conformance/nonconformance split: "
        f"{s['conformance_ratio']:.0f}%/{s['nonconformance_ratio']:.0f}%. "
        f"Optimal prevention at ${opt['optimal_prevention']:,.0f}."
    )

    return result
