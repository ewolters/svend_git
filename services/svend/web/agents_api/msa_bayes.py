"""
Bayesian Measurement System Analysis (Gage R&R).

Replaces point-estimate ANOVA with a Gibbs sampler for the random
effects model, giving posterior distributions for every variance
component and probability-driven verdicts.

Model:
    y_ijk = μ + a_i + b_j + c_ij + ε_ijk

    a_i  ~ N(0, σ²_P)    part effect
    b_j  ~ N(0, σ²_O)    operator effect
    c_ij ~ N(0, σ²_PO)   interaction
    ε_ijk ~ N(0, σ²_E)   equipment (repeatability)

Priors:
    σ²_* ~ InvGamma(α₀, β₀)
    Vague: α₀ = β₀ = 0.01
    Historical: α₀ = ν/2, β₀ = ν·s²/2 (from prior study)

Derived metrics (posterior distributions):
    σ²_GRR = σ²_O + σ²_PO + σ²_E
    %GRR = σ_GRR / σ_Total × 100  (study variation)
    NDC = ⌊1.41 × σ_P / σ_GRR⌋

Dependencies: numpy only.
"""

import math
import numpy as np

__all__ = ["BayesianGageRR", "run_bayes_msa"]


# ===========================================================================
# Gibbs sampler for random-effects Gage R&R
# ===========================================================================
class BayesianGageRR:
    """
    Bayesian Gage R&R via Gibbs sampling on the crossed random effects model.

    Parameters
    ----------
    parts : array-like of str/int
        Part identifier for each observation.
    operators : array-like of str/int
        Operator identifier for each observation.
    measurements : array-like of float
        Measured values.
    prior : dict, optional
        Hyperparameters from a previous study for sequential updating.
        Keys: alpha_P, beta_P, alpha_O, beta_O, alpha_PO, beta_PO,
              alpha_E, beta_E.
    tolerance : float, optional
        USL - LSL for %tolerance computation.
    """

    def __init__(self, parts, operators, measurements,
                 prior=None, tolerance=None):
        parts = np.array(parts)
        operators = np.array(operators)
        measurements = np.array(measurements, dtype=float)

        # Encode factors to integer indices
        self.part_labels = np.unique(parts)
        self.op_labels = np.unique(operators)
        self.n_p = len(self.part_labels)
        self.n_o = len(self.op_labels)

        part_map = {v: i for i, v in enumerate(self.part_labels)}
        op_map = {v: i for i, v in enumerate(self.op_labels)}
        self._pi = np.array([part_map[p] for p in parts])
        self._oj = np.array([op_map[o] for o in operators])
        self._y = measurements
        self.N = len(measurements)
        self.tolerance = tolerance

        # Detect replicates
        counts = {}
        for i, j in zip(self._pi, self._oj):
            counts[(i, j)] = counts.get((i, j), 0) + 1
        self.n_r = max(counts.values()) if counts else 1

        # Priors: InvGamma(α, β) for each variance component
        default_a, default_b = 0.01, 0.01
        pr = prior or {}
        self._prior = {
            "P":  (pr.get("alpha_P", default_a), pr.get("beta_P", default_b)),
            "O":  (pr.get("alpha_O", default_a), pr.get("beta_O", default_b)),
            "PO": (pr.get("alpha_PO", default_a), pr.get("beta_PO", default_b)),
            "E":  (pr.get("alpha_E", default_a), pr.get("beta_E", default_b)),
        }
        self._has_prior = prior is not None

        # Results (filled by fit())
        self.samples = None
        self.fitted = False

    def fit(self, n_iter=2000, burn_in=500, thin=2, seed=None):
        """Run Gibbs sampler."""
        if seed is not None:
            np.random.seed(seed)

        n_p, n_o = self.n_p, self.n_o
        y = self._y
        pi = self._pi
        oj = self._oj
        N = self.N

        # Initialize from data
        mu = np.mean(y)
        a = np.zeros(n_p)       # part effects
        b = np.zeros(n_o)       # operator effects
        c = np.zeros((n_p, n_o))  # interaction
        sig2_P = np.var(y) * 0.5
        sig2_O = np.var(y) * 0.1
        sig2_PO = np.var(y) * 0.05
        sig2_E = np.var(y) * 0.2

        # Storage
        n_keep = (n_iter - burn_in) // thin
        samples = {
            "sig2_P": np.zeros(n_keep),
            "sig2_O": np.zeros(n_keep),
            "sig2_PO": np.zeros(n_keep),
            "sig2_E": np.zeros(n_keep),
            "mu": np.zeros(n_keep),
        }
        idx = 0

        for it in range(n_iter):
            # Residuals without each component
            resid_full = y - mu - a[pi] - b[oj] - c[pi, oj]

            # --- Sample μ ---
            r_mu = y - a[pi] - b[oj] - c[pi, oj]
            v_mu = sig2_E / N
            m_mu = np.mean(r_mu)
            mu = np.random.normal(m_mu, np.sqrt(v_mu))

            # --- Sample a_i (part effects) ---
            for i in range(n_p):
                mask = pi == i
                n_i = np.sum(mask)
                if n_i == 0:
                    continue
                r_i = y[mask] - mu - b[oj[mask]] - c[i, oj[mask]]
                v_ai = 1.0 / (n_i / sig2_E + 1.0 / sig2_P)
                m_ai = v_ai * (np.sum(r_i) / sig2_E)
                a[i] = np.random.normal(m_ai, np.sqrt(v_ai))

            # --- Sample b_j (operator effects) ---
            for j in range(n_o):
                mask = oj == j
                n_j = np.sum(mask)
                if n_j == 0:
                    continue
                r_j = y[mask] - mu - a[pi[mask]] - c[pi[mask], j]
                v_bj = 1.0 / (n_j / sig2_E + 1.0 / sig2_O)
                m_bj = v_bj * (np.sum(r_j) / sig2_E)
                b[j] = np.random.normal(m_bj, np.sqrt(v_bj))

            # --- Sample c_ij (interaction effects) ---
            for i in range(n_p):
                for j in range(n_o):
                    mask = (pi == i) & (oj == j)
                    n_ij = np.sum(mask)
                    if n_ij == 0:
                        continue
                    r_ij = y[mask] - mu - a[i] - b[j]
                    v_cij = 1.0 / (n_ij / sig2_E + 1.0 / sig2_PO)
                    m_cij = v_cij * (np.sum(r_ij) / sig2_E)
                    c[i, j] = np.random.normal(m_cij, np.sqrt(v_cij))

            # --- Sample σ²_E ---
            resid = y - mu - a[pi] - b[oj] - c[pi, oj]
            ss_e = np.sum(resid ** 2)
            a_post = self._prior["E"][0] + N / 2.0
            b_post = self._prior["E"][1] + ss_e / 2.0
            sig2_E = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

            # --- Sample σ²_P ---
            ss_p = np.sum(a ** 2)
            a_post = self._prior["P"][0] + n_p / 2.0
            b_post = self._prior["P"][1] + ss_p / 2.0
            sig2_P = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

            # --- Sample σ²_O ---
            ss_o = np.sum(b ** 2)
            a_post = self._prior["O"][0] + n_o / 2.0
            b_post = self._prior["O"][1] + ss_o / 2.0
            sig2_O = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

            # --- Sample σ²_PO ---
            ss_po = np.sum(c ** 2)
            a_post = self._prior["PO"][0] + (n_p * n_o) / 2.0
            b_post = self._prior["PO"][1] + ss_po / 2.0
            sig2_PO = 1.0 / np.random.gamma(a_post, 1.0 / b_post)

            # Store thinned post-burn-in samples
            if it >= burn_in and (it - burn_in) % thin == 0 and idx < n_keep:
                samples["sig2_P"][idx] = sig2_P
                samples["sig2_O"][idx] = sig2_O
                samples["sig2_PO"][idx] = sig2_PO
                samples["sig2_E"][idx] = sig2_E
                samples["mu"][idx] = mu
                idx += 1

        self.samples = samples
        self.fitted = True

        # Compute derived quantities
        self._compute_derived()
        return self

    def _compute_derived(self):
        """Compute %GRR, NDC, etc. from posterior samples."""
        s = self.samples
        self.sig2_grr = s["sig2_O"] + s["sig2_PO"] + s["sig2_E"]
        self.sig2_total = s["sig2_P"] + self.sig2_grr
        self.pct_grr = (np.sqrt(self.sig2_grr)
                        / np.sqrt(self.sig2_total) * 100)
        self.pct_repeat = (np.sqrt(s["sig2_E"])
                           / np.sqrt(self.sig2_total) * 100)
        self.pct_reprod = (np.sqrt(s["sig2_O"] + s["sig2_PO"])
                           / np.sqrt(self.sig2_total) * 100)
        self.pct_part = (np.sqrt(s["sig2_P"])
                         / np.sqrt(self.sig2_total) * 100)
        self.ndc_samples = np.floor(
            1.41 * np.sqrt(s["sig2_P"] / self.sig2_grr)
        ).astype(int)

    def summary(self):
        """Posterior summary statistics."""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        def _stat(arr):
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "std": float(np.std(arr)),
            }

        s = self.samples
        return {
            "var_part": _stat(s["sig2_P"]),
            "var_operator": _stat(s["sig2_O"]),
            "var_interaction": _stat(s["sig2_PO"]),
            "var_repeatability": _stat(s["sig2_E"]),
            "var_grr": _stat(self.sig2_grr),
            "var_total": _stat(self.sig2_total),
            "pct_grr": _stat(self.pct_grr),
            "pct_repeatability": _stat(self.pct_repeat),
            "pct_reproducibility": _stat(self.pct_reprod),
            "pct_part": _stat(self.pct_part),
            "ndc": _stat(self.ndc_samples),
            "p_grr_lt_10": float(np.mean(self.pct_grr < 10)),
            "p_grr_lt_30": float(np.mean(self.pct_grr < 30)),
            "p_ndc_ge_5": float(np.mean(self.ndc_samples >= 5)),
            "mu": _stat(s["mu"]),
            "n_samples": len(s["sig2_P"]),
        }

    def prior_for_next_study(self):
        """
        Export posterior as InvGamma hyperparameters for sequential updating.

        For each variance σ², the posterior InvGamma(α_post, β_post) becomes
        the prior for the next study.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        def _fit_ig(samples):
            m = np.mean(samples)
            v = np.var(samples)
            if v < 1e-15:
                return 2.01, m * 1.01
            alpha = m ** 2 / v + 2
            beta = m * (alpha - 1)
            return float(alpha), float(beta)

        s = self.samples
        aP, bP = _fit_ig(s["sig2_P"])
        aO, bO = _fit_ig(s["sig2_O"])
        aPO, bPO = _fit_ig(s["sig2_PO"])
        aE, bE = _fit_ig(s["sig2_E"])

        return {
            "alpha_P": aP, "beta_P": bP,
            "alpha_O": aO, "beta_O": bO,
            "alpha_PO": aPO, "beta_PO": bPO,
            "alpha_E": aE, "beta_E": bE,
        }


# ===========================================================================
# DSW Integration
# ===========================================================================
def run_bayes_msa(df, analysis_id, config):
    """Dispatch for Bayesian MSA in DSW."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    part_col = config.get("part", "")
    op_col = config.get("operator", "")
    meas_col = config.get("measurement", "")
    tolerance = config.get("tolerance")
    if tolerance:
        tolerance = float(tolerance)
    prior_json = config.get("prior")  # optional dict from previous study

    if not part_col or not meas_col:
        result["summary"] = "Error: Need part and measurement columns."
        return result

    for col in [part_col, op_col, meas_col]:
        if col and col not in df.columns:
            result["summary"] = f"Error: Column '{col}' not found."
            return result

    parts = df[part_col].values
    measurements = df[meas_col].dropna().values
    if op_col:
        operators = df[op_col].values
    else:
        operators = np.array(["Op1"] * len(parts))

    if len(measurements) < 6:
        result["summary"] = "Error: Need at least 6 observations."
        return result

    # Run Gibbs sampler
    prior = prior_json if isinstance(prior_json, dict) else None
    model = BayesianGageRR(parts, operators, measurements,
                           prior=prior, tolerance=tolerance)
    model.fit(n_iter=2000, burn_in=500, thin=2, seed=42)
    s = model.summary()

    # Verdict
    p10 = s["p_grr_lt_10"]
    p30 = s["p_grr_lt_30"]
    if p10 > 0.9:
        verdict = "Acceptable"
        verdict_color = "good"
    elif p30 > 0.9:
        verdict = "Marginal"
        verdict_color = "highlight"
    else:
        verdict = "Unacceptable"
        verdict_color = "warning"

    # --- Summary ---
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>BAYESIAN GAGE R&R<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Design:<</COLOR>> "
                 f"{model.n_p} parts × {model.n_o} operators × "
                 f"{model.n_r} replicates = {model.N} obs")
    lines.append(f"<<COLOR:highlight>>Prior:<</COLOR>> "
                 f"{'Historical (from previous study)' if model._has_prior else 'Weakly informative (vague)'}")
    lines.append(f"<<COLOR:highlight>>MCMC:<</COLOR>> "
                 f"{s['n_samples']} posterior samples "
                 f"(2000 iterations, 500 burn-in, thin 2)")

    lines.append(f"\n<<COLOR:accent>>── Verdict ──<</COLOR>>")
    lines.append(f"<<COLOR:{verdict_color}>>{verdict}: "
                 f"%GRR = {s['pct_grr']['mean']:.1f}% "
                 f"[{s['pct_grr']['ci_low']:.1f}%, "
                 f"{s['pct_grr']['ci_high']:.1f}%]<</COLOR>>")
    lines.append(f"<<COLOR:highlight>>P(%GRR < 10%):<</COLOR>> "
                 f"{p10:.1%}")
    lines.append(f"<<COLOR:highlight>>P(%GRR < 30%):<</COLOR>> "
                 f"{p30:.1%}")
    lines.append(f"<<COLOR:highlight>>NDC:<</COLOR>> "
                 f"{s['ndc']['mean']:.1f} "
                 f"[{s['ndc']['ci_low']:.0f}, {s['ndc']['ci_high']:.0f}] "
                 f"  P(NDC ≥ 5) = {s['p_ndc_ge_5']:.1%}")

    lines.append(f"\n<<COLOR:accent>>── Variance Components "
                 f"(posterior mean [95% CI]) ──<</COLOR>>")
    fmt = lambda k, label: (
        f"  {label:<22} "
        f"{s[k]['mean']:>10.4f}  "
        f"[{s[k]['ci_low']:.4f}, {s[k]['ci_high']:.4f}]"
    )
    lines.append(fmt("var_part", "Part-to-Part (σ²_P)"))
    lines.append(fmt("var_operator", "Operator (σ²_O)"))
    lines.append(fmt("var_interaction", "Interaction (σ²_PO)"))
    lines.append(fmt("var_repeatability", "Repeatability (σ²_E)"))
    lines.append(fmt("var_grr", "GRR (σ²_GRR)"))
    lines.append(fmt("var_total", "Total (σ²_Total)"))

    lines.append(f"\n<<COLOR:accent>>── % Study Variation ──<</COLOR>>")
    fmt2 = lambda k, label: (
        f"  {label:<22} "
        f"{s[k]['mean']:>8.1f}%  "
        f"[{s[k]['ci_low']:.1f}%, {s[k]['ci_high']:.1f}%]"
    )
    lines.append(fmt2("pct_grr", "GRR"))
    lines.append(fmt2("pct_repeatability", "Repeatability"))
    lines.append(fmt2("pct_reproducibility", "Reproducibility"))
    lines.append(fmt2("pct_part", "Part-to-Part"))

    if tolerance:
        grr_tol = 6 * np.sqrt(s["var_grr"]["mean"]) / tolerance * 100
        lines.append(f"\n<<COLOR:highlight>>%Tolerance (6σ_GRR/tol):"
                     f"<</COLOR>> {grr_tol:.1f}%")

    # Sequential updating info
    next_prior = model.prior_for_next_study()
    lines.append(f"\n<<COLOR:accent>>── Sequential Updating ──<</COLOR>>")
    lines.append("  This posterior can serve as the prior for your next "
                 "gage study.")
    lines.append("  Each successive study sharpens the estimates "
                 "without discarding prior knowledge.")

    lines.append(f"\n<<COLOR:accent>>── Assumptions ──<</COLOR>>")
    lines.append("  1. Balanced crossed design "
                 "(each operator measures each part)")
    lines.append("  2. Random effects are Gaussian")
    lines.append("  3. Variance components are positive "
                 "(enforced by InvGamma prior)")
    lines.append("  4. No systematic measurement drift within the study")

    result["summary"] = "\n".join(lines)

    # --- Plots ---
    # 1. %GRR posterior with thresholds
    pct_grr_arr = model.pct_grr
    hist_vals, bin_edges = np.histogram(pct_grr_arr, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    result["plots"].append({
        "title": "%GRR Posterior Distribution",
        "data": [
            {"type": "bar", "x": bin_centers.tolist(),
             "y": hist_vals.tolist(),
             "marker": {"color": "rgba(74,159,110,0.5)",
                        "line": {"color": "#4a9f6e", "width": 0.5}},
             "name": "Posterior density"},
            {"type": "scatter", "x": [10, 10],
             "y": [0, max(hist_vals) * 1.1],
             "mode": "lines",
             "line": {"color": "#4a9f6e", "dash": "dash", "width": 1.5},
             "name": "10% (acceptable)"},
            {"type": "scatter", "x": [30, 30],
             "y": [0, max(hist_vals) * 1.1],
             "mode": "lines",
             "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
             "name": "30% (unacceptable)"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": "% Study Variation (GRR)"},
            "yaxis": {"title": "Posterior density"},
            "annotations": [
                {"x": 10, "y": max(hist_vals) * 1.05,
                 "text": f"P(<10%)={p10:.0%}", "showarrow": False,
                 "font": {"color": "#4a9f6e", "size": 10}},
                {"x": 30, "y": max(hist_vals) * 1.05,
                 "text": f"P(<30%)={p30:.0%}", "showarrow": False,
                 "font": {"color": "#d94a4a", "size": 10}},
            ],
        },
    })

    # 2. Variance components comparison (posterior box plots)
    comp_names = ["Part", "Operator", "Interaction", "Repeatability"]
    comp_keys = ["sig2_P", "sig2_O", "sig2_PO", "sig2_E"]
    box_traces = []
    for name, key in zip(comp_names, comp_keys):
        box_traces.append({
            "type": "box", "y": model.samples[key].tolist(),
            "name": name, "boxmean": True,
            "marker": {"color": "rgba(74,159,110,0.5)"},
        })

    result["plots"].append({
        "title": "Posterior Distributions — Variance Components",
        "data": box_traces,
        "layout": {
            "template": "plotly_dark", "height": 300,
            "yaxis": {"title": "Variance (σ²)"},
        },
    })

    # 3. NDC posterior
    ndc_arr = model.ndc_samples
    ndc_vals, ndc_counts = np.unique(ndc_arr, return_counts=True)
    ndc_probs = ndc_counts / len(ndc_arr)

    result["plots"].append({
        "title": "NDC Posterior (Number of Distinct Categories)",
        "data": [
            {"type": "bar", "x": ndc_vals.tolist(),
             "y": ndc_probs.tolist(),
             "marker": {"color": ["#4a9f6e" if v >= 5 else "#d94a4a"
                                   for v in ndc_vals]},
             "name": "P(NDC=k)"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 250,
            "xaxis": {"title": "NDC", "dtick": 1},
            "yaxis": {"title": "Probability"},
            "annotations": [
                {"x": 5, "y": max(ndc_probs) * 1.05,
                 "text": "≥5 required", "showarrow": False,
                 "font": {"color": "#4a9f6e", "size": 10}},
            ],
        },
    })

    # 4. % Study Variation breakdown (stacked)
    labels = ["%Repeatability", "%Reproducibility", "%Part-to-Part"]
    means = [s["pct_repeatability"]["mean"],
             s["pct_reproducibility"]["mean"],
             s["pct_part"]["mean"]]
    colors = ["#d4a24a", "#6ab7d4", "#4a9f6e"]

    result["plots"].append({
        "title": "% Study Variation Breakdown",
        "data": [
            {"type": "bar", "x": [label], "y": [val],
             "name": label, "marker": {"color": col}}
            for label, val, col in zip(labels, means, colors)
        ],
        "layout": {
            "template": "plotly_dark", "height": 200,
            "barmode": "stack",
            "xaxis": {"showticklabels": False},
            "yaxis": {"title": "% Study Variation"},
        },
    })

    # --- Statistics ---
    result["statistics"] = {
        "test": "bayesian_gage_rr",
        "n_parts": model.n_p,
        "n_operators": model.n_o,
        "n_replicates": model.n_r,
        "n_total": model.N,
        "verdict": verdict,
        "pct_grr_mean": round(s["pct_grr"]["mean"], 2),
        "pct_grr_ci": [round(s["pct_grr"]["ci_low"], 2),
                       round(s["pct_grr"]["ci_high"], 2)],
        "p_grr_lt_10": round(p10, 4),
        "p_grr_lt_30": round(p30, 4),
        "ndc_mean": round(s["ndc"]["mean"], 1),
        "p_ndc_ge_5": round(s["p_ndc_ge_5"], 4),
        "has_prior": model._has_prior,
        "tolerance": tolerance,
        "posterior_summary": s,
        "prior_for_next": next_prior,
    }

    result["guide_observation"] = (
        f"Bayesian Gage R&R: {verdict}. "
        f"%GRR = {s['pct_grr']['mean']:.1f}% "
        f"[{s['pct_grr']['ci_low']:.1f}%, {s['pct_grr']['ci_high']:.1f}%]. "
        f"P(%GRR<10%) = {p10:.0%}, P(%GRR<30%) = {p30:.0%}. "
        f"NDC = {s['ndc']['mean']:.0f}."
    )

    return result
