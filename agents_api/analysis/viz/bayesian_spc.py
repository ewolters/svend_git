"""Bayesian SPC visualization suite — extracted from viz monolith.

Handlers: capability, changepoint, control, acceptance.
NIG helpers: _nig_posterior_update, _nig_sample, _cpk_from_params.

CR: 3c0d0e53
"""

import numpy as np

from ..common import SVEND_COLORS


def _nig_posterior_update(data, mu0, nu0, alpha0, beta0):
    """Normal-Inverse-Gamma conjugate posterior update."""
    n = len(data)
    x_bar = np.mean(data)
    nu_n = nu0 + n
    mu_n = (nu0 * mu0 + n * x_bar) / nu_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * np.sum((data - x_bar) ** 2) + (n * nu0 * (x_bar - mu0) ** 2) / (2.0 * nu_n)
    return mu_n, nu_n, alpha_n, beta_n


def _nig_sample(mu_n, nu_n, alpha_n, beta_n, n_samples=10000):
    """Draw (mu, sigma) samples from NIG posterior."""
    from scipy.stats import invgamma

    rng = np.random.default_rng(42)
    sigma2_samples = invgamma.rvs(a=alpha_n, scale=beta_n, size=n_samples, random_state=rng)
    mu_samples = rng.normal(loc=mu_n, scale=np.sqrt(sigma2_samples / nu_n))
    sigma_samples = np.sqrt(sigma2_samples)
    return mu_samples, sigma_samples


def _cpk_from_params(mu, sigma, usl=None, lsl=None):
    """Vectorized Cpk from arrays of mu and sigma. Supports one-sided specs."""
    if usl is not None and lsl is not None:
        cpu = (usl - mu) / (3.0 * sigma)
        cpl = (mu - lsl) / (3.0 * sigma)
        return np.minimum(cpu, cpl)
    elif usl is not None:
        return (usl - mu) / (3.0 * sigma)
    elif lsl is not None:
        return (mu - lsl) / (3.0 * sigma)
    else:
        return np.zeros_like(mu)


def run_bayes_spc_capability(df, config):
    result = {"plots": [], "summary": ""}

    # Bayesian Capability Analysis — eliminates the 1.5σ assumption
    from scipy.stats import t as tdist

    col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
    data = df[col].dropna().values.astype(float)
    usl_raw = config.get("usl")
    lsl_raw = config.get("lsl")
    usl = float(usl_raw) if usl_raw not in (None, "", "null") else None
    lsl = float(lsl_raw) if lsl_raw not in (None, "", "null") else None
    if usl is None and lsl is None:
        result["summary"] = "<<COLOR:error>>At least one spec limit (USL or LSL) is required<</COLOR>>"
        return result
    if usl is not None and lsl is not None and usl <= lsl:
        result["summary"] = "<<COLOR:error>>USL must be greater than LSL<</COLOR>>"
        return result
    target = config.get("target")
    if target not in (None, "", "null"):
        target = float(target)
    elif usl is not None and lsl is not None:
        target = (usl + lsl) / 2.0
    elif usl is not None:
        target = usl
    else:
        target = lsl
    spec_label = f"USL={usl}" if lsl is None else (f"LSL={lsl}" if usl is None else f"USL={usl}, LSL={lsl}")
    n_mc = int(config.get("n_mc", 10000))
    prior_type = config.get("prior_type", "weakly_informative")

    n = len(data)
    x_bar = float(np.mean(data))
    s = float(np.std(data, ddof=1)) if n > 1 else 0.01

    # Set prior
    if prior_type == "informative":
        pp = config.get("prior_params", {})
        mu0 = float(pp.get("mu0", x_bar))
        nu0 = float(pp.get("nu0", 5))
        alpha0 = float(pp.get("alpha0", 3))
        beta0 = float(pp.get("beta0", s**2 * 2))
    elif prior_type == "historical":
        pp = config.get("prior_params", {})
        hist_mean = float(pp.get("hist_mean", x_bar))
        hist_std = float(pp.get("hist_std", s))
        hist_n = int(pp.get("hist_n", 30))
        mu0, nu0, alpha0, beta0 = (
            hist_mean,
            float(hist_n),
            hist_n / 2.0,
            hist_n / 2.0 * hist_std**2,
        )
    else:  # weakly_informative — α₀=2 for finite σ² mean, β₀ centered on sample variance
        s2 = float(np.var(data, ddof=1)) if n > 1 else 1.0
        mu0, nu0, alpha0, beta0 = x_bar, 1.0, 2.0, max(s2, 1e-10)

    # Posterior update
    mu_n, nu_n, alpha_n, beta_n = _nig_posterior_update(data, mu0, nu0, alpha0, beta0)

    # Monte Carlo sampling
    mu_samples, sigma_samples = _nig_sample(mu_n, nu_n, alpha_n, beta_n, n_mc)
    cpk_samples = _cpk_from_params(mu_samples, sigma_samples, usl, lsl)

    cpk_median = float(np.median(cpk_samples))
    cpk_ci = (
        float(np.percentile(cpk_samples, 2.5)),
        float(np.percentile(cpk_samples, 97.5)),
    )
    p_gt_1 = float(np.mean(cpk_samples > 1.0))
    p_gt_133 = float(np.mean(cpk_samples > 1.33))
    p_gt_167 = float(np.mean(cpk_samples > 1.67))
    p_gt_2 = float(np.mean(cpk_samples > 2.0))

    # Frequentist Cpk point estimate
    if s > 0:
        if usl is not None and lsl is not None:
            cpk_freq = float(min((usl - x_bar) / (3 * s), (x_bar - lsl) / (3 * s)))
        elif usl is not None:
            cpk_freq = float((usl - x_bar) / (3 * s))
        else:
            cpk_freq = float((x_bar - lsl) / (3 * s))
    else:
        cpk_freq = 0.0

    # ── Additional capability indices ──
    # Cp / Pp — potential capability (ignores centering)
    cp_median = cp_ci = cp_freq = pp_median = pp_freq = None
    if usl is not None and lsl is not None and s > 0:
        cp_samples = (usl - lsl) / (6.0 * sigma_samples)
        cp_median = float(np.median(cp_samples))
        cp_ci = (
            float(np.percentile(cp_samples, 2.5)),
            float(np.percentile(cp_samples, 97.5)),
        )
        cp_freq = float((usl - lsl) / (6 * s))
        pp_median, pp_freq = cp_median, cp_freq  # identical for individual data

    # Ppk = Cpk for individual data (no within-subgroup estimator)
    ppk_median, ppk_freq = cpk_median, cpk_freq

    # Cpm — Taguchi (penalizes off-target)
    cpm_median = cpm_ci = cpm_freq = None
    if target is not None and usl is not None and lsl is not None and s > 0:
        cpm_denom = 6.0 * np.sqrt(sigma_samples**2 + (mu_samples - target) ** 2)
        cpm_samples = (usl - lsl) / cpm_denom
        cpm_median = float(np.median(cpm_samples))
        cpm_ci = (
            float(np.percentile(cpm_samples, 2.5)),
            float(np.percentile(cpm_samples, 97.5)),
        )
        cpm_freq = float((usl - lsl) / (6 * np.sqrt(s**2 + (x_bar - target) ** 2)))

    # Centering (k) — 0 = perfectly centered, 1 = mean at spec limit
    k_centering = None
    if usl is not None and lsl is not None:
        midpoint = (usl + lsl) / 2.0
        half_tol = (usl - lsl) / 2.0
        k_centering = abs(x_bar - midpoint) / half_tol if half_tol > 0 else 0.0

    # Posterior predictive via Monte Carlo (robust — no parameterization traps)
    rng_pp = np.random.default_rng(123)
    x_pred = rng_pp.normal(loc=mu_samples, scale=sigma_samples)
    oos_mask = np.zeros(len(x_pred), dtype=bool)
    if lsl is not None:
        oos_mask |= x_pred < lsl
    if usl is not None:
        oos_mask |= x_pred > usl
    p_oos = float(np.mean(oos_mask))

    # Student-t curve for display only
    df_t = 2 * alpha_n
    loc_t = mu_n
    scale_t = float(np.sqrt(beta_n * (nu_n + 1) / (alpha_n * nu_n)))
    pp_dist = tdist(df=df_t, loc=loc_t, scale=scale_t)
    dpmo = p_oos * 1e6

    # Sigma level + yield (from Bayesian DPMO)
    from scipy.stats import norm as normdist

    if p_oos > 0 and p_oos < 1:
        z_bench = float(normdist.ppf(1 - p_oos))
        sigma_level = z_bench + 1.5
    else:
        z_bench = 6.0 if p_oos == 0 else 0.0
        sigma_level = z_bench + 1.5
    yield_pct = (1.0 - p_oos) * 100.0

    # σ posterior sanity check
    sigma_99 = float(np.percentile(sigma_samples, 99))
    sigma_iqr = float(np.percentile(sigma_samples, 75) - np.percentile(sigma_samples, 25))
    sigma_warning = ""
    if sigma_iqr > 0 and sigma_99 > 5 * sigma_iqr + float(np.median(sigma_samples)):
        sigma_warning = "Data may be non-normal, from mixed processes, or contain outliers. Consider transformations or a mixture model."

    # Verdict (probability-driven, not point-estimate)
    if p_gt_133 >= 0.95:
        verdict_color, verdict = "success", "CAPABLE \u2014 P(Cpk > 1.33) \u2265 95%"
    elif p_gt_133 >= 0.80:
        verdict_color, verdict = (
            "highlight",
            "MARGINAL \u2014 P(Cpk > 1.33) between 80\u201395%",
        )
    else:
        verdict_color, verdict = "error", "NOT CAPABLE \u2014 P(Cpk > 1.33) < 80%"

    # ── Summary ──
    ci_w = cpk_ci[1] - cpk_ci[0]
    summary = f"<<COLOR:accent>>{'=' * 60}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN PROCESS CAPABILITY<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 60}<</COLOR>>\n\n"
    summary += (
        f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    "
        f"<<COLOR:highlight>>Spec:<</COLOR>> {spec_label}    "
        f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n\n"
    )

    # Capability indices table
    summary += f"<<COLOR:accent>>{'_' * 40}<</COLOR>>\n"
    summary += "<<COLOR:title>>Capability Indices<</COLOR>>\n"
    summary += f"  {'':18s} {'Bayesian':>10s}   {'95% CI':>18s}   {'Frequentist':>12s}\n"
    summary += (
        f"  {'Cpk:':18s} "
        f"<<COLOR:highlight>>{cpk_median:>10.4f}<</COLOR>>   "
        f"[{cpk_ci[0]:.4f}, {cpk_ci[1]:.4f}]   "
        f"{cpk_freq:>12.4f}\n"
    )
    summary += f"  {'Ppk:':18s} {ppk_median:>10.4f}   [{cpk_ci[0]:.4f}, {cpk_ci[1]:.4f}]   {ppk_freq:>12.4f}\n"
    if cp_median is not None:
        summary += f"  {'Cp:':18s} {cp_median:>10.4f}   [{cp_ci[0]:.4f}, {cp_ci[1]:.4f}]   {cp_freq:>12.4f}\n"
        summary += f"  {'Pp:':18s} {pp_median:>10.4f}   [{cp_ci[0]:.4f}, {cp_ci[1]:.4f}]   {pp_freq:>12.4f}\n"
    if cpm_median is not None:
        summary += (
            f"  {'Cpm (Taguchi):':18s} {cpm_median:>10.4f}   [{cpm_ci[0]:.4f}, {cpm_ci[1]:.4f}]   {cpm_freq:>12.4f}\n"
        )
    summary += "\n  <<COLOR:text>>Cpk = Ppk (individual data \u2014 no subgroup structure).<</COLOR>>\n"
    if k_centering is not None and cp_median is not None and k_centering > 0.05:
        summary += f"  <<COLOR:text>>Cp > Cpk: process is off-center by k = {k_centering:.0%}.<</COLOR>>\n"

    # Probability table
    summary += f"\n<<COLOR:accent>>{'_' * 40}<</COLOR>>\n"
    summary += "<<COLOR:title>>Probability Table<</COLOR>>\n"
    summary += f"  P(Cpk > 1.00) = <<COLOR:{'success' if p_gt_1 > 0.9 else 'error'}>>{p_gt_1:.1%}<</COLOR>>\n"
    summary += f"  P(Cpk > 1.33) = <<COLOR:{'success' if p_gt_133 > 0.9 else 'error'}>>{p_gt_133:.1%}<</COLOR>>\n"
    summary += f"  P(Cpk > 1.67) = <<COLOR:{'success' if p_gt_167 > 0.9 else 'text'}>>{p_gt_167:.1%}<</COLOR>>\n"
    summary += f"  P(Cpk > 2.00) = {p_gt_2:.1%}\n"

    # Expected performance
    summary += f"\n<<COLOR:accent>>{'_' * 40}<</COLOR>>\n"
    summary += "<<COLOR:title>>Expected Performance<</COLOR>>\n"
    summary += f"  P(out of spec): {p_oos:.6f}    DPMO: <<COLOR:highlight>>{dpmo:.0f}<</COLOR>>\n"
    summary += f"  Yield: <<COLOR:highlight>>{yield_pct:.4f}%<</COLOR>>    Sigma level: {sigma_level:.1f}\u03c3\n"
    summary += "  (No 1.5\u03c3 shift assumption \u2014 uncertainty is first-class)\n"

    # Verdict
    summary += f"\n<<COLOR:{verdict_color}>>{verdict}<</COLOR>>\n"

    if sigma_warning:
        summary += f"\n<<COLOR:error>>Warning: {sigma_warning}<</COLOR>>\n"

    # ── Narrative ──
    summary += f"\n<<COLOR:accent>>{'_' * 40}<</COLOR>>\n"
    summary += "<<COLOR:title>>Narrative<</COLOR>>\n"
    narr_parts = []

    # Centering
    if k_centering is not None:
        if k_centering <= 0.05:
            narr_parts.append(f"Process is well-centered (k = {k_centering:.1%}).")
        else:
            closer_to = "USL" if x_bar > (usl + lsl) / 2.0 else "LSL"
            narr_parts.append(f"Process runs {k_centering:.0%} off-center toward {closer_to}.")
            if cp_median is not None and cp_median > cpk_median + 0.1:
                narr_parts.append(
                    f"Recentering to target would improve Cpk from {cpk_median:.2f} to {cp_median:.2f} (= Cp)."
                )

    # Cpm insight
    if cpm_median is not None and cpm_median < cpk_median - 0.05:
        offset = abs(x_bar - target)
        narr_parts.append(
            f"Taguchi Cpm ({cpm_median:.2f}) < Cpk ({cpk_median:.2f}) \u2014 "
            f"the {offset:.4f} offset from target increases quality loss."
        )

    # Posterior maturity
    if ci_w < 0.3:
        narr_parts.append(f"Posterior is well-converged (95% CI width = {ci_w:.2f}).")
    elif ci_w < 0.8:
        narr_parts.append(
            f"Moderate posterior uncertainty (95% CI width = {ci_w:.2f}). More data would tighten the estimate."
        )
    else:
        narr_parts.append(
            f"Posterior is wide (95% CI width = {ci_w:.2f}) \u2014 "
            f"collect more data before making capability decisions."
        )

    # Bayesian vs frequentist
    cpk_diff = abs(cpk_freq - cpk_median)
    if cpk_diff < 0.05:
        narr_parts.append(
            f"Bayesian ({cpk_median:.3f}) and frequentist ({cpk_freq:.3f}) agree \u2014 posterior is data-driven."
        )
    else:
        narr_parts.append(
            f"Bayesian ({cpk_median:.3f}) differs from frequentist ({cpk_freq:.3f}) "
            f"by {cpk_diff:.3f} \u2014 prior influence visible, more data will converge them."
        )

    # Practical DPMO
    if dpmo > 0:
        defects_per_1k = dpmo / 1000.0
        narr_parts.append(
            f"At {dpmo:.0f} DPMO, expect ~{defects_per_1k:.1f} defects per 1,000 parts "
            f"({yield_pct:.4f}% yield, {sigma_level:.1f}\u03c3)."
        )
    else:
        narr_parts.append(f"Zero defects predicted (yield {yield_pct:.4f}%, >{sigma_level:.1f}\u03c3).")

    for part in narr_parts:
        summary += f"  {part}\n"

    result["summary"] = summary
    result["statistics"] = {
        "cpk_median": cpk_median,
        "cpk_ci_lower": cpk_ci[0],
        "cpk_ci_upper": cpk_ci[1],
        "cpk_frequentist": cpk_freq,
        "ppk_median": ppk_median,
        "ppk_frequentist": ppk_freq,
        "cp_median": cp_median,
        "cp_ci_lower": cp_ci[0] if cp_ci else None,
        "cp_ci_upper": cp_ci[1] if cp_ci else None,
        "cp_frequentist": cp_freq,
        "cpm_median": cpm_median,
        "cpm_ci_lower": cpm_ci[0] if cpm_ci else None,
        "cpm_ci_upper": cpm_ci[1] if cpm_ci else None,
        "cpm_frequentist": cpm_freq,
        "centering_k": k_centering,
        "p_cpk_gt_1": p_gt_1,
        "p_cpk_gt_133": p_gt_133,
        "p_cpk_gt_167": p_gt_167,
        "p_cpk_gt_2": p_gt_2,
        "dpmo": dpmo,
        "p_out_of_spec": p_oos,
        "yield_pct": yield_pct,
        "sigma_level": sigma_level,
        "z_bench": z_bench,
    }

    # Plot 1: Posterior Cpk histogram
    cpk_hist_vals, cpk_hist_edges = np.histogram(cpk_samples, bins=80)
    cpk_hist_centers = (cpk_hist_edges[:-1] + cpk_hist_edges[1:]) / 2
    ci_mask = (cpk_hist_centers >= cpk_ci[0]) & (cpk_hist_centers <= cpk_ci[1])
    cpk_ymax = int(max(cpk_hist_vals))
    result["plots"].append(
        {
            "title": "Posterior Distribution of Cpk",
            "data": [
                {
                    "type": "bar",
                    "x": cpk_hist_centers.tolist(),
                    "y": cpk_hist_vals.tolist(),
                    "marker": {"color": ["rgba(74,159,110,0.7)" if m else "rgba(74,159,110,0.2)" for m in ci_mask]},
                    "name": "Posterior",
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": [1.0, 1.0],
                    "y": [0, cpk_ymax],
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    "name": "Cpk = 1.0",
                },
                {
                    "type": "scatter",
                    "x": [1.33, 1.33],
                    "y": [0, cpk_ymax],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": "Cpk = 1.33",
                },
                {
                    "type": "scatter",
                    "x": [cpk_freq, cpk_freq],
                    "y": [0, cpk_ymax],
                    "mode": "lines",
                    "line": {"color": "#5b9bd5", "width": 2},
                    "name": "Frequentist",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Cpk"},
                "yaxis": {"title": "Count"},
                "legend": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                },
                "annotations": [
                    {
                        "x": cpk_median,
                        "y": cpk_ymax * 0.9,
                        "text": f"Median: {cpk_median:.3f}",
                        "showarrow": True,
                        "arrowhead": 2,
                        "font": {"color": "#4a9f6e"},
                    }
                ],
            },
        }
    )

    # Plot 2: Posterior predictive vs spec limits
    lo_bound = (lsl - 3 * s) if lsl is not None else (data.min() - 3 * s)
    hi_bound = (usl + 3 * s) if usl is not None else (data.max() + 3 * s)
    x_range = np.linspace(min(lo_bound, data.min()), max(hi_bound, data.max()), 300)
    pp_pdf = pp_dist.pdf(x_range)
    from scipy.stats import norm

    norm_pdf = norm.pdf(x_range, loc=x_bar, scale=s) if s > 0 else np.zeros_like(x_range)
    data_hist_vals, data_hist_edges = np.histogram(data, bins=40, density=True)
    data_hist_centers = (data_hist_edges[:-1] + data_hist_edges[1:]) / 2
    peak_y = max(max(pp_pdf), max(norm_pdf)) if len(pp_pdf) > 0 else 1
    pred_traces = [
        {
            "type": "bar",
            "x": data_hist_centers.tolist(),
            "y": data_hist_vals.tolist(),
            "marker": {"color": "rgba(74,159,110,0.3)"},
            "name": "Data",
            "showlegend": True,
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": pp_pdf.tolist(),
            "mode": "lines",
            "line": {"color": "#e89547", "width": 2},
            "name": "Predictive (Student-t)",
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": norm_pdf.tolist(),
            "mode": "lines",
            "line": {"color": "#5b9bd5", "dash": "dash", "width": 1.5},
            "name": "Normal fit",
        },
    ]
    if lsl is not None:
        pred_traces.append(
            {
                "type": "scatter",
                "x": [lsl, lsl],
                "y": [0, peak_y],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dot", "width": 2},
                "name": "LSL",
            }
        )
    if usl is not None:
        pred_traces.append(
            {
                "type": "scatter",
                "x": [usl, usl],
                "y": [0, peak_y],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dot", "width": 2},
                "name": "USL",
            }
        )
    ann_x = (usl + lsl) / 2 if (usl is not None and lsl is not None) else (usl if usl is not None else lsl)
    result["plots"].append(
        {
            "title": "Posterior Predictive vs Spec Limits",
            "data": pred_traces,
            "layout": {
                "height": 340,
                "xaxis": {"title": col},
                "legend": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                },
                "annotations": [
                    {
                        "x": ann_x,
                        "y": max(pp_pdf) * 0.95,
                        "text": f"DPMO: {dpmo:.0f}",
                        "showarrow": False,
                        "font": {"color": "#e89547", "size": 13},
                    }
                ],
            },
        }
    )

    # Plot 3: P(Cpk > threshold) curve
    thresholds = np.linspace(0.5, 3.0, 100)
    p_above = [float(np.mean(cpk_samples > t)) for t in thresholds]
    result["plots"].append(
        {
            "title": "P(Cpk > Threshold)",
            "data": [
                {
                    "type": "scatter",
                    "x": thresholds.tolist(),
                    "y": p_above,
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2.5},
                    "name": "P(Cpk > threshold)",
                },
                {
                    "type": "scatter",
                    "x": [0.5, 3.0],
                    "y": [0.95, 0.95],
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "95% confidence",
                },
            ],
            "layout": {
                "height": 280,
                "xaxis": {"title": "Threshold"},
                "yaxis": {"title": "Probability", "range": [0, 1.05]},
                "legend": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                },
            },
        }
    )

    # Plot 4: Data histogram with predictive overlay
    overlay_traces = [
        {
            "type": "histogram",
            "x": data.tolist(),
            "nbinsx": 40,
            "histnorm": "probability density",
            "marker": {
                "color": "rgba(74,159,110,0.4)",
                "line": {"color": "#4a9f6e", "width": 1},
            },
            "name": "Data",
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": pp_pdf.tolist(),
            "mode": "lines",
            "line": {"color": "#e89547", "width": 2.5},
            "name": "Posterior Predictive",
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": norm_pdf.tolist(),
            "mode": "lines",
            "line": {"color": "#5b9bd5", "dash": "dash", "width": 1.5},
            "name": "Normal Fit",
        },
    ]
    if lsl is not None:
        overlay_traces.append(
            {
                "type": "scatter",
                "x": [lsl, lsl],
                "y": [0, peak_y],
                "mode": "lines",
                "line": {"color": "#e85747", "width": 2},
                "name": "LSL",
                "showlegend": False,
            }
        )
    if usl is not None:
        overlay_traces.append(
            {
                "type": "scatter",
                "x": [usl, usl],
                "y": [0, peak_y],
                "mode": "lines",
                "line": {"color": "#e85747", "width": 2},
                "name": "USL",
                "showlegend": False,
            }
        )
    result["plots"].append(
        {
            "title": "Data vs Predictive Distribution",
            "data": overlay_traces,
            "layout": {
                "height": 320,
                "xaxis": {"title": col},
                "yaxis": {"title": "Density"},
                "legend": {
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                },
            },
        }
    )

    cpm_str = f", Cpm={cpm_median:.3f}" if cpm_median is not None else ""
    result["guide_observation"] = (
        f"Bayesian capability: Cpk median {cpk_median:.3f} "
        f"[{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}], "
        f"P(Cpk>1.33)={p_gt_133:.1%}, DPMO={dpmo:.0f}, "
        f"yield={yield_pct:.4f}%, {sigma_level:.1f}\u03c3{cpm_str}"
    )

    return result


def run_bayes_spc_changepoint(df, config):
    result = {"plots": [], "summary": ""}

    # Bayesian Online Change Point Detection (Adams & MacKay 2007)
    from scipy.special import gammaln, logsumexp

    col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
    data = df[col].dropna().values.astype(float)
    hazard_rate = float(config.get("hazard_rate", 0.01))
    min_seg = int(config.get("min_segment_length", 5))

    n = len(data)
    max_rl = min(500, n)  # truncate run-length window for performance

    # NIG sufficient statistics per run length
    log_H = np.log(hazard_rate)
    log_1mH = np.log(1 - hazard_rate)

    # Priors from calibration phase (first ~50 obs, not all data)
    n_cal = min(50, max(10, n // 5))
    cal_data = data[:n_cal]
    s2_cal = float(np.var(cal_data, ddof=1)) if n_cal > 1 else 1.0
    mu0_cp = float(np.mean(cal_data))
    nu0_cp = 1.0  # kappa: 1 pseudo-observation for mean
    alpha0_cp = 2.0  # shape: mildly informative
    beta0_cp = max(s2_cal * alpha0_cp, 1e-10)  # rate matched to calibration variance

    # Forward pass
    # run_length_probs[t] = log P(r_t = r | x_{1:t})
    log_R = -np.inf * np.ones((n + 1, max_rl + 1))
    log_R[0, 0] = 0.0  # P(r_0 = 0) = 1

    # Track sufficient stats: sum_x, sum_x2, count per run length
    ss_n = np.zeros(max_rl + 1)
    ss_sum = np.zeros(max_rl + 1)
    ss_sum2 = np.zeros(max_rl + 1)

    cp_prob = np.zeros(n)  # P(r_t = 0) — instantaneous changepoint
    shift_prob = np.zeros(n)  # 1 - P(r=t+1) — has ANY change occurred?

    for t in range(n):
        x = data[t]

        # Predictive probability for each run length (Student-t)
        # NIG predictive: t-distribution with updated params
        rl_range = min(t + 1, max_rl)
        log_pred = np.full(rl_range + 1, -np.inf)

        for r in range(rl_range + 1):
            nn = ss_n[r]
            if nn == 0:
                # Prior predictive
                nu_r = nu0_cp
                alpha_r = alpha0_cp
                mu_r = mu0_cp
                beta_r = beta0_cp
            else:
                xbar_r = ss_sum[r] / nn
                nu_r = nu0_cp + nn
                mu_r = (nu0_cp * mu0_cp + nn * xbar_r) / nu_r
                alpha_r = alpha0_cp + nn / 2.0
                beta_r = (
                    beta0_cp
                    + 0.5 * (ss_sum2[r] - nn * xbar_r**2)
                    + (nn * nu0_cp * (xbar_r - mu0_cp) ** 2) / (2.0 * nu_r)
                )
                beta_r = max(beta_r, 1e-10)

            # Student-t log pdf
            df_r = 2 * alpha_r
            scale_r = np.sqrt(beta_r * (nu_r + 1) / (alpha_r * nu_r))
            scale_r = max(scale_r, 1e-10)
            z = (x - mu_r) / scale_r
            log_pred[r] = (
                gammaln((df_r + 1) / 2)
                - gammaln(df_r / 2)
                - 0.5 * np.log(df_r * np.pi)
                - np.log(scale_r)
                - ((df_r + 1) / 2) * np.log(1 + z**2 / df_r)
            )

        # Growth: P(r_{t+1} = r+1) ∝ P(r_t = r) * pred(x) * (1-H)
        log_growth = np.full(max_rl + 1, -np.inf)
        for r in range(rl_range + 1):
            if r < max_rl and log_R[t, r] > -1e300:
                log_growth[r + 1] = log_R[t, r] + log_pred[r] + log_1mH

        # Changepoint: P(r_{t+1} = 0) = sum P(r_t = r) * pred(x) * H
        log_cp_terms = []
        for r in range(rl_range + 1):
            if log_R[t, r] > -1e300:
                log_cp_terms.append(log_R[t, r] + log_pred[r] + log_H)
        log_cp = logsumexp(log_cp_terms) if log_cp_terms else -np.inf

        # Combine and normalize
        log_R[t + 1, 0] = log_cp
        for r in range(1, max_rl + 1):
            log_R[t + 1, r] = log_growth[r]

        # Normalize
        log_evidence = logsumexp(log_R[t + 1, : max_rl + 1])
        log_R[t + 1, : max_rl + 1] -= log_evidence

        cp_prob[t] = np.exp(log_R[t + 1, 0])

        # Shift probability: 1 - P(original run continues from t=0)
        if t + 1 <= max_rl:
            shift_prob[t] = float(np.clip(1.0 - np.exp(log_R[t + 1, t + 1]), 0.0, 1.0))
        else:
            shift_prob[t] = 1.0

        # Update sufficient stats
        new_ss_n = np.zeros(max_rl + 1)
        new_ss_sum = np.zeros(max_rl + 1)
        new_ss_sum2 = np.zeros(max_rl + 1)
        for r in range(1, min(t + 2, max_rl + 1)):
            new_ss_n[r] = ss_n[r - 1] + 1
            new_ss_sum[r] = ss_sum[r - 1] + x
            new_ss_sum2[r] = ss_sum2[r - 1] + x**2
        new_ss_n[0] = 0
        new_ss_sum[0] = 0
        new_ss_sum2[0] = 0
        ss_n, ss_sum, ss_sum2 = new_ss_n, new_ss_sum, new_ss_sum2

    # Detect changepoints using shift probability (not instantaneous cp_prob)
    changepoints = []
    for t in range(min_seg, n - min_seg):
        if shift_prob[t] > 0.5 and (t == 0 or shift_prob[t - 1] <= 0.5):
            # Rising edge: shift just detected
            if not changepoints or (t - changepoints[-1]) >= min_seg:
                changepoints.append(t)

    # Segment statistics
    boundaries = [0] + changepoints + [n]
    segments = []
    for i in range(len(boundaries) - 1):
        seg_data = data[boundaries[i] : boundaries[i + 1]]
        segments.append(
            {
                "start": int(boundaries[i]),
                "end": int(boundaries[i + 1]),
                "mean": float(np.mean(seg_data)),
                "std": float(np.std(seg_data, ddof=1)) if len(seg_data) > 1 else 0,
                "n": len(seg_data),
            }
        )

    # ── Per-regime Bayesian Cpk (activates when specs provided) ──
    usl_raw = config.get("usl")
    lsl_raw = config.get("lsl")
    _usl = float(usl_raw) if usl_raw not in (None, "", "null") else None
    _lsl = float(lsl_raw) if lsl_raw not in (None, "", "null") else None
    has_specs = _usl is not None or _lsl is not None
    regime_cpk_samples = {}  # idx -> cpk_samples array for plotting

    if has_specs and len(segments) >= 1:
        n_mc_regime = 5000
        for idx, seg in enumerate(segments):
            seg_data = data[seg["start"] : seg["end"]]
            seg_n = len(seg_data)
            if seg_n < 2:
                continue
            # Weakly-informative NIG prior (matches bayes_spc_capability)
            s2_seg = float(np.var(seg_data, ddof=1))
            mu0_s, nu0_s, alpha0_s, beta0_s = (
                float(np.mean(seg_data)),
                1.0,
                2.0,
                max(s2_seg, 1e-10),
            )
            mu_n, nu_n, alpha_n, beta_n = _nig_posterior_update(seg_data, mu0_s, nu0_s, alpha0_s, beta0_s)
            mu_samp, sigma_samp = _nig_sample(mu_n, nu_n, alpha_n, beta_n, n_mc_regime)
            cpk_samp = _cpk_from_params(mu_samp, sigma_samp, _usl, _lsl)
            regime_cpk_samples[idx] = cpk_samp

            cpk_med = float(np.median(cpk_samp))
            cpk_lo = float(np.percentile(cpk_samp, 2.5))
            cpk_hi = float(np.percentile(cpk_samp, 97.5))
            p_gt_133 = float(np.mean(cpk_samp > 1.33))
            # P(next obs out of spec)
            x_pred = np.random.default_rng(42 + idx).normal(loc=mu_samp, scale=sigma_samp)
            oos = np.zeros(len(x_pred), dtype=bool)
            if _usl is not None:
                oos |= x_pred > _usl
            if _lsl is not None:
                oos |= x_pred < _lsl
            p_oos = float(np.mean(oos))
            n_eff = float(nu_n)

            # Value-of-information: CI width scales as 1/sqrt(n_eff)
            ci_width = cpk_hi - cpk_lo
            ci_width_plus30 = ci_width * np.sqrt(n_eff / (n_eff + 30))
            narrowing_30 = (1.0 - ci_width_plus30 / ci_width) * 100 if ci_width > 0 else 0

            seg["cpk_median"] = cpk_med
            seg["cpk_ci"] = [cpk_lo, cpk_hi]
            seg["p_cpk_gt_133"] = p_gt_133
            seg["p_oos"] = p_oos
            seg["n_eff"] = n_eff
            seg["ci_narrowing_at_plus_30"] = float(narrowing_30)

    # Summary
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION (BOCPD)<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += (
        f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    <<COLOR:highlight>>Hazard rate:<</COLOR>> {hazard_rate}\n\n"
    )

    if changepoints:
        summary += f"<<COLOR:success>>Detected {len(changepoints)} change point(s):<</COLOR>>\n\n"
        for i, cp in enumerate(changepoints):
            seg_before = segments[i]
            seg_after = segments[i + 1]
            summary += (
                f"  <<COLOR:highlight>>Change {i + 1}:<</COLOR>> observation {cp}, P(shifted) = {shift_prob[cp]:.3f}\n"
            )
            summary += f"    Before: μ = {seg_before['mean']:.4f}, σ = {seg_before['std']:.4f} (n={seg_before['n']})\n"
            summary += f"    After:  μ = {seg_after['mean']:.4f}, σ = {seg_after['std']:.4f} (n={seg_after['n']})\n\n"
    else:
        summary += "<<COLOR:text>>No significant change points detected (threshold: P > 0.5)<</COLOR>>\n"

    # Per-regime capability table (when specs provided)
    if has_specs and any("cpk_median" in s for s in segments):
        spec_parts = []
        if _lsl is not None:
            spec_parts.append(f"LSL={_lsl}")
        if _usl is not None:
            spec_parts.append(f"USL={_usl}")
        summary += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PER-REGIME CAPABILITY<</COLOR>>  ({', '.join(spec_parts)})\n"
        summary += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"
        prev_p133 = None
        for idx, seg in enumerate(segments):
            if "cpk_median" not in seg:
                continue
            label = f"Regime {idx + 1} (obs {seg['start'] + 1}–{seg['end']})"
            cpk_str = f"{seg['cpk_median']:.2f} [{seg['cpk_ci'][0]:.2f}, {seg['cpk_ci'][1]:.2f}]"
            p133 = seg["p_cpk_gt_133"]
            p133_pct = p133 * 100
            # Color by capability confidence
            if p133 >= 0.95:
                color = "success"
            elif p133 >= 0.50:
                color = "highlight"
            else:
                color = "error"
            summary += f"  <<COLOR:{color}>>{label}:<</COLOR>>\n"
            summary += f"    Cpk = {cpk_str},  P(Cpk > 1.33) = {p133_pct:.0f}%,  n_eff = {seg['n_eff']:.0f}\n"
            # Value-of-information warning for small regimes
            if seg["n"] < 50:
                narrowing = seg["ci_narrowing_at_plus_30"]
                summary += f"    <<COLOR:highlight>>Confidence is low — {seg['n']} observations. "
                summary += f"Collect 30 more to narrow the credible interval by ~{narrowing:.0f}%.<</COLOR>>\n"
            if prev_p133 is not None:
                if p133 < prev_p133 - 0.15:
                    summary += "    <<COLOR:error>>Capability degraded by shift.<</COLOR>>\n"
                elif p133 > prev_p133 + 0.15:
                    summary += "    <<COLOR:success>>Capability improved after shift.<</COLOR>>\n"
            prev_p133 = p133
            summary += "\n"

    result["summary"] = summary
    result["statistics"] = {
        "n_changepoints": len(changepoints),
        "changepoints": changepoints,
        "segments": segments,
    }

    # Plot 1: Run-length posterior heatmap
    rl_display = min(max_rl, 100)
    heatmap_data = np.exp(log_R[1 : n + 1, :rl_display]).T
    result["plots"].append(
        {
            "title": "Run-Length Posterior",
            "data": [
                {
                    "type": "heatmap",
                    "z": heatmap_data.tolist(),
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "P(r)"},
                }
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Run Length"},
            },
        }
    )

    # Plot 2: Shift probability (1 - P(original run continues))
    result["plots"].append(
        {
            "title": "Shift Probability — P(process has changed)",
            "data": [
                {
                    "type": "scatter",
                    "y": shift_prob.tolist(),
                    "mode": "lines",
                    "line": {"color": "#d94a4a", "width": 2},
                    "name": "P(shifted)",
                },
                {
                    "type": "scatter",
                    "x": [0, n],
                    "y": [0.5, 0.5],
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "Threshold (50%)",
                },
                {
                    "type": "scatter",
                    "x": [0, n],
                    "y": [0.95, 0.95],
                    "mode": "lines",
                    "line": {"color": "#d94a4a", "dash": "dot", "width": 1},
                    "name": "Alarm (95%)",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "P(shifted)", "range": [0, 1.05]},
            },
        }
    )

    # Plot 3: Process data with detected changes
    proc_data = [
        {
            "type": "scatter",
            "y": data.tolist(),
            "mode": "lines+markers",
            "marker": {"size": 4, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e"},
            "name": col,
        }
    ]
    for i, cp in enumerate(changepoints):
        proc_data.append(
            {
                "type": "scatter",
                "x": [cp, cp],
                "y": [float(data.min()), float(data.max())],
                "mode": "lines",
                "line": {"color": "#e85747", "width": 2, "dash": "dash"},
                "name": f"Change {i + 1}",
            }
        )
    for seg in segments:
        proc_data.append(
            {
                "type": "scatter",
                "x": [seg["start"], seg["end"] - 1],
                "y": [seg["mean"], seg["mean"]],
                "mode": "lines",
                "line": {"color": "#e89547", "width": 2},
                "name": f"μ={seg['mean']:.2f}",
                "showlegend": False,
            }
        )
    result["plots"].append(
        {
            "title": "Process Data with Change Points",
            "data": proc_data,
            "layout": {
                "height": 350,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": col},
            },
        }
    )

    # Plot 4: Per-regime Cpk posteriors (when specs provided)
    if regime_cpk_samples:
        regime_colors = SVEND_COLORS[:6]
        cpk_traces = []
        for idx in sorted(regime_cpk_samples.keys()):
            seg = segments[idx]
            samp = regime_cpk_samples[idx]
            color = regime_colors[idx % len(regime_colors)]
            cpk_traces.append(
                {
                    "type": "histogram",
                    "x": samp.tolist(),
                    "name": f"Regime {idx + 1} (n={seg['n']})",
                    "opacity": 0.55,
                    "nbinsx": 60,
                    "marker": {"color": color},
                }
            )
        # Reference lines at 1.0 and 1.33
        cpk_shapes = [
            {
                "type": "line",
                "x0": 1.0,
                "x1": 1.0,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e89547", "width": 2, "dash": "dash"},
            },
            {
                "type": "line",
                "x0": 1.33,
                "x1": 1.33,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
            },
        ]
        cpk_annotations = [
            {
                "x": 1.0,
                "y": 1,
                "yref": "paper",
                "text": "Cpk=1.0",
                "showarrow": False,
                "yanchor": "bottom",
                "font": {"color": "#e89547", "size": 11},
            },
            {
                "x": 1.33,
                "y": 1,
                "yref": "paper",
                "text": "Cpk=1.33",
                "showarrow": False,
                "yanchor": "bottom",
                "font": {"color": "#d94a4a", "size": 11},
            },
        ]
        result["plots"].append(
            {
                "title": "Per-Regime Capability — Bayesian Cpk Posterior",
                "data": cpk_traces,
                "layout": {
                    "height": 300,
                    "barmode": "overlay",
                    "xaxis": {"title": "Cpk"},
                    "yaxis": {"title": "Density"},
                    "shapes": cpk_shapes,
                    "annotations": cpk_annotations,
                    "showlegend": True,
                },
            }
        )

    result["guide_observation"] = f"BOCPD detected {len(changepoints)} change point(s) in {n} observations"

    return result


def run_bayes_spc_control(df, config):
    result = {"plots": [], "summary": ""}

    # Bayesian Control Chart — two-state HMM forward filter
    col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
    data = df[col].dropna().values.astype(float)
    ref_mean = config.get("reference_mean")
    ref_std = config.get("reference_std")
    shift_size = float(config.get("shift_size", 1.5))
    trans_prob = float(config.get("transition_prob", 0.01))

    n = len(data)
    # Auto-estimate reference from first 20 obs if not provided
    n_ref = min(20, n)
    if ref_mean is None or ref_mean == "" or ref_mean == "null":
        ref_mean = float(np.mean(data[:n_ref]))
    else:
        ref_mean = float(ref_mean)
    if ref_std is None or ref_std == "" or ref_std == "null":
        ref_std = float(np.std(data[:n_ref], ddof=1)) if n_ref > 1 else float(np.std(data, ddof=1))
    else:
        ref_std = float(ref_std)
    ref_std = max(ref_std, 1e-10)

    # HMM parameters
    # State 0 (in-control): X ~ N(ref_mean, ref_std)
    # State 1 (shifted): marginalized over +δ and -δ shift direction
    delta = shift_size * ref_std
    p_recover = 0.05

    from scipy.stats import norm

    # Forward filter in log-space
    log_p_ic = np.zeros(n)  # P(in-control | data_{1:t})
    log_alpha_ic = 0.0  # log P(state=IC, data_{1:t})
    log_alpha_sh = np.log(1e-10)  # log P(state=shifted, data_{1:t})

    # Sequential NIG posterior for mu
    seq_mu = np.zeros(n)
    seq_ci_lo = np.zeros(n)
    seq_ci_hi = np.zeros(n)
    from scipy.stats import t as tdist_sc

    mu0_s, nu0_s, alpha0_s, beta0_s = ref_mean, 1.0, 2.0, max(ref_std**2, 1e-10)

    for t in range(n):
        x = data[t]

        # Emission likelihoods
        ll_ic = norm.logpdf(x, loc=ref_mean, scale=ref_std)
        # Shifted state: marginalize over +δ and -δ (equal probability)
        ll_sh_plus = norm.logpdf(x, loc=ref_mean + delta, scale=ref_std)
        ll_sh_minus = norm.logpdf(x, loc=ref_mean - delta, scale=ref_std)
        ll_sh = np.logaddexp(ll_sh_plus, ll_sh_minus) - np.log(2)

        # Transition
        log_t_ic_ic = np.log(1 - trans_prob)
        log_t_ic_sh = np.log(trans_prob)
        log_t_sh_ic = np.log(p_recover)
        log_t_sh_sh = np.log(1 - p_recover)

        # Forward step
        from scipy.special import logsumexp as _lse

        new_log_alpha_ic = _lse([log_alpha_ic + log_t_ic_ic, log_alpha_sh + log_t_sh_ic]) + ll_ic
        new_log_alpha_sh = _lse([log_alpha_ic + log_t_ic_sh, log_alpha_sh + log_t_sh_sh]) + ll_sh

        # Normalize
        log_evidence = _lse([new_log_alpha_ic, new_log_alpha_sh])
        log_alpha_ic = new_log_alpha_ic - log_evidence
        log_alpha_sh = new_log_alpha_sh - log_evidence

        log_p_ic[t] = log_alpha_ic
        1.0 - np.exp(log_alpha_ic)

        # Sequential NIG for mu
        seg_data = data[: t + 1]
        mu_n_s, nu_n_s, alpha_n_s, beta_n_s = _nig_posterior_update(seg_data, mu0_s, nu0_s, alpha0_s, beta0_s)
        seq_mu[t] = mu_n_s
        if alpha_n_s > 0 and nu_n_s > 0 and beta_n_s > 0:
            scale_s = np.sqrt(beta_n_s / (alpha_n_s * nu_n_s))
            ci_half = tdist_sc.ppf(0.975, df=2 * alpha_n_s) * scale_s
            seq_ci_lo[t] = mu_n_s - ci_half
            seq_ci_hi[t] = mu_n_s + ci_half
        else:
            seq_ci_lo[t] = mu_n_s
            seq_ci_hi[t] = mu_n_s

    p_shifted_arr = 1.0 - np.exp(log_p_ic)
    n_alarms = int(np.sum(p_shifted_arr > 0.5))

    # Summary
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CONTROL CHART<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    <<COLOR:highlight>>Ref μ:<</COLOR>> {ref_mean:.4f}    <<COLOR:highlight>>Ref σ:<</COLOR>> {ref_std:.4f}\n"
    summary += f"<<COLOR:highlight>>Shift size:<</COLOR>> {shift_size}σ    <<COLOR:highlight>>P(shift):<</COLOR>> {trans_prob}\n\n"

    if n_alarms > 0:
        first_alarm = int(np.argmax(p_shifted_arr > 0.5))
        summary += f"<<COLOR:error>>ALERT: {n_alarms} observations with P(shifted) > 0.5<</COLOR>>\n"
        summary += f"<<COLOR:highlight>>First alarm at observation {first_alarm}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Final posterior μ:<</COLOR>> {seq_mu[-1]:.4f} [{seq_ci_lo[-1]:.4f}, {seq_ci_hi[-1]:.4f}]\n"
    else:
        summary += "<<COLOR:success>>Process appears in control — no observations with P(shifted) > 0.5<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Final posterior μ:<</COLOR>> {seq_mu[-1]:.4f} [{seq_ci_lo[-1]:.4f}, {seq_ci_hi[-1]:.4f}]\n"

    result["summary"] = summary
    result["statistics"] = {
        "n_alarms": n_alarms,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "final_mu": float(seq_mu[-1]),
        "final_ci": [float(seq_ci_lo[-1]), float(seq_ci_hi[-1])],
    }

    # Plot 1: Process data colored by P(shifted)
    colors = [f"rgb({int(255 * p)},{int(255 * (1 - p))},80)" for p in p_shifted_arr]
    result["plots"].append(
        {
            "title": "Process Data — Colored by P(shifted)",
            "data": [
                {
                    "type": "scatter",
                    "y": data.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": colors,
                        "size": 6,
                        "line": {"color": "#333", "width": 0.5},
                    },
                    "name": col,
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "y": data.tolist(),
                    "mode": "lines",
                    "line": {"color": "rgba(150,150,150,0.3)", "width": 1},
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [ref_mean, ref_mean],
                    "mode": "lines",
                    "line": {"color": "#5b9bd5", "dash": "dash"},
                    "name": "Reference μ",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": col},
            },
        }
    )

    # Plot 2: Sequential posterior for μ
    x_idx = list(range(n))
    result["plots"].append(
        {
            "title": "Sequential Posterior for μ",
            "data": [
                {
                    "type": "scatter",
                    "x": x_idx,
                    "y": seq_ci_hi.tolist(),
                    "mode": "lines",
                    "line": {"color": "rgba(74,159,110,0.2)", "width": 0},
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": x_idx,
                    "y": seq_ci_lo.tolist(),
                    "mode": "lines",
                    "line": {"color": "rgba(74,159,110,0.2)", "width": 0},
                    "fill": "tonexty",
                    "fillcolor": "rgba(74,159,110,0.15)",
                    "name": "95% CI",
                },
                {
                    "type": "scatter",
                    "x": x_idx,
                    "y": seq_mu.tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "Posterior μ",
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [ref_mean, ref_mean],
                    "mode": "lines",
                    "line": {"color": "#5b9bd5", "dash": "dash"},
                    "name": "Reference",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "μ"},
            },
        }
    )

    # Plot 3: Alarm timeline
    alarm_colors = [f"rgba({int(255 * p)},{int(255 * (1 - p))},80,0.8)" for p in p_shifted_arr]
    result["plots"].append(
        {
            "title": "Shift Probability Timeline",
            "data": [
                {
                    "type": "bar",
                    "y": p_shifted_arr.tolist(),
                    "marker": {"color": alarm_colors},
                    "name": "P(shifted)",
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": [0, n - 1],
                    "y": [0.5, 0.5],
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    "name": "Alarm threshold",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "P(shifted)", "range": [0, 1.05]},
            },
        }
    )

    result["guide_observation"] = f"Bayesian control chart: {n_alarms} alarms in {n} observations (shift={shift_size}σ)"

    return result


def run_bayes_spc_acceptance(df, config):
    result = {"plots": [], "summary": ""}

    # Bayesian Acceptance Sampling — Beta-Binomial conjugate
    from scipy.stats import beta as betadist

    col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
    data = df[col].dropna().values.astype(float)

    aql = float(config.get("aql", 0.01))
    threshold = float(config.get("acceptance_threshold", 0.95))
    prior_alpha = float(config.get("prior_alpha", 1))
    prior_beta = float(config.get("prior_beta", 1))

    # Determine defectives: either from manual input or classify from spec limits
    manual_defectives = config.get("defectives")
    manual_sample = config.get("sample_size")
    if manual_defectives is not None and manual_sample is not None:
        k = int(manual_defectives)
        n_total = int(manual_sample)
    else:
        # Classify from measurements + spec limits
        usl_a = config.get("usl")
        lsl_a = config.get("lsl")
        n_total = len(data)
        k = 0
        if usl_a is not None and usl_a != "" and usl_a != "null":
            k += int(np.sum(data > float(usl_a)))
        if lsl_a is not None and lsl_a != "" and lsl_a != "null":
            k += int(np.sum(data < float(lsl_a)))

    # Posterior
    post_alpha = prior_alpha + k
    post_beta_param = prior_beta + n_total - k
    p_accept = float(betadist.cdf(aql, post_alpha, post_beta_param))
    post_mean = post_alpha / (post_alpha + post_beta_param)
    post_ci = (
        float(betadist.ppf(0.025, post_alpha, post_beta_param)),
        float(betadist.ppf(0.975, post_alpha, post_beta_param)),
    )

    # Decision
    if p_accept >= threshold:
        decision = "ACCEPT"
        decision_color = "success"
    elif p_accept <= (1 - threshold):
        decision = "REJECT"
        decision_color = "error"
    else:
        decision = "CONTINUE SAMPLING"
        decision_color = "highlight"

    # Sequential analysis: P(p<AQL) at each cumulative inspection
    seq_p_accept = []
    seq_k = 0
    earliest_accept = None
    earliest_reject = None
    for i in range(n_total):
        if manual_defectives is not None:
            # Simulate sequentially from overall rate
            seq_k_i = int(round(k * (i + 1) / n_total))
        else:
            # From actual data classification
            val = data[i] if i < len(data) else data[-1]
            is_def = False
            usl_a = config.get("usl")
            lsl_a = config.get("lsl")
            if usl_a is not None and usl_a != "" and usl_a != "null" and val > float(usl_a):
                is_def = True
            if lsl_a is not None and lsl_a != "" and lsl_a != "null" and val < float(lsl_a):
                is_def = True
            seq_k += int(is_def)
            seq_k_i = seq_k

        pa_i = float(betadist.cdf(aql, prior_alpha + seq_k_i, prior_beta + (i + 1) - seq_k_i))
        seq_p_accept.append(pa_i)

        if earliest_accept is None and pa_i >= threshold:
            earliest_accept = i + 1
        if earliest_reject is None and pa_i <= (1 - threshold):
            earliest_reject = i + 1

    # Decision boundaries: max k giving acceptance at each n
    boundary_n = list(range(1, n_total + 1))
    accept_boundary = []
    reject_boundary = []
    for ni in boundary_n:
        # Find max k where P(p<AQL) >= threshold
        max_k_accept = -1
        min_k_reject = ni + 1
        for ki in range(ni + 1):
            pa = betadist.cdf(aql, prior_alpha + ki, prior_beta + ni - ki)
            if pa >= threshold:
                max_k_accept = ki
            if pa <= (1 - threshold) and ki < min_k_reject:
                min_k_reject = ki
        accept_boundary.append(max_k_accept if max_k_accept >= 0 else None)
        reject_boundary.append(min_k_reject if min_k_reject <= ni else None)

    # Summary
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN ACCEPTANCE SAMPLING<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n_total}    <<COLOR:highlight>>Defectives:<</COLOR>> {k}    <<COLOR:highlight>>AQL:<</COLOR>> {aql}\n"
    summary += f"<<COLOR:highlight>>Prior:<</COLOR>> Beta({prior_alpha}, {prior_beta})    <<COLOR:highlight>>Threshold:<</COLOR>> {threshold}\n\n"
    summary += f"<<COLOR:accent>>{'─' * 40}<</COLOR>>\n"
    summary += "<<COLOR:title>>Posterior for Defect Rate<</COLOR>>\n"
    summary += f"  Mean: <<COLOR:highlight>>{post_mean:.6f}<</COLOR>>    95% CI: [{post_ci[0]:.6f}, {post_ci[1]:.6f}]\n"
    summary += f"  P(p < AQL) = <<COLOR:{'success' if p_accept > threshold else 'error'}>>{p_accept:.4f}<</COLOR>>\n\n"
    summary += f"<<COLOR:{decision_color}>>Decision: {decision}<</COLOR>>\n\n"

    if earliest_accept:
        summary += f"<<COLOR:success>>Earliest acceptance possible at n = {earliest_accept}<</COLOR>>\n"
    if earliest_reject:
        summary += f"<<COLOR:error>>Earliest rejection at n = {earliest_reject}<</COLOR>>\n"

    result["summary"] = summary
    result["statistics"] = {
        "defectives": k,
        "sample_size": n_total,
        "defect_rate_mean": post_mean,
        "defect_rate_ci": list(post_ci),
        "p_accept": p_accept,
        "decision": decision,
        "earliest_accept": earliest_accept,
        "earliest_reject": earliest_reject,
    }

    # Plot 1: Posterior for defect rate
    x_range = np.linspace(0, min(max(post_ci[1] * 3, aql * 5), 1.0), 300)
    post_pdf = betadist.pdf(x_range, post_alpha, post_beta_param)
    prior_pdf = (
        betadist.pdf(x_range, prior_alpha, prior_beta) if prior_alpha > 0 and prior_beta > 0 else np.zeros_like(x_range)
    )
    result["plots"].append(
        {
            "title": "Posterior for Defect Rate",
            "data": [
                {
                    "type": "scatter",
                    "x": x_range.tolist(),
                    "y": post_pdf.tolist(),
                    "mode": "lines",
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74,159,110,0.2)",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "Posterior",
                },
                {
                    "type": "scatter",
                    "x": x_range.tolist(),
                    "y": prior_pdf.tolist(),
                    "mode": "lines",
                    "line": {"color": "#888", "dash": "dash", "width": 1.5},
                    "name": "Prior",
                },
                {
                    "type": "scatter",
                    "x": [aql, aql],
                    "y": [0, max(post_pdf) if len(post_pdf) > 0 else 1],
                    "mode": "lines",
                    "line": {"color": "#e85747", "width": 2},
                    "name": f"AQL = {aql}",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Defect Rate (p)"},
                "yaxis": {"title": "Density"},
                "annotations": [
                    {
                        "x": post_mean,
                        "y": max(post_pdf) * 0.9 if len(post_pdf) > 0 else 0.5,
                        "text": f"P(p<AQL)={p_accept:.3f}",
                        "showarrow": True,
                        "font": {"color": "#4a9f6e"},
                    }
                ],
            },
        }
    )

    # Plot 2: Sequential posterior evolution
    result["plots"].append(
        {
            "title": "Sequential P(p < AQL) — Earliest Stopping",
            "data": [
                {
                    "type": "scatter",
                    "x": list(range(1, n_total + 1)),
                    "y": seq_p_accept,
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "P(p < AQL)",
                },
                {
                    "type": "scatter",
                    "x": [1, n_total],
                    "y": [threshold, threshold],
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "dash": "dash"},
                    "name": f"Accept ({threshold})",
                },
                {
                    "type": "scatter",
                    "x": [1, n_total],
                    "y": [1 - threshold, 1 - threshold],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash"},
                    "name": f"Reject ({1 - threshold:.2f})",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Items Inspected"},
                "yaxis": {"title": "P(p < AQL)", "range": [0, 1.05]},
                "annotations": (
                    [
                        {
                            "x": earliest_accept,
                            "y": threshold,
                            "text": f"Accept @ n={earliest_accept}",
                            "showarrow": True,
                            "font": {"color": "#4a9f6e"},
                        }
                    ]
                    if earliest_accept
                    else []
                ),
            },
        }
    )

    # Plot 3: Decision boundary
    accept_y = [b if b is not None else None for b in accept_boundary]
    reject_y = [b if b is not None else None for b in reject_boundary]
    boundary_plots = [
        {
            "type": "scatter",
            "x": boundary_n,
            "y": accept_y,
            "mode": "lines",
            "line": {"color": "#4a9f6e", "width": 2},
            "name": "Accept boundary",
            "connectgaps": False,
        },
        {
            "type": "scatter",
            "x": boundary_n,
            "y": reject_y,
            "mode": "lines",
            "line": {"color": "#e85747", "width": 2},
            "name": "Reject boundary",
            "connectgaps": False,
        },
    ]
    # Add actual trajectory point
    boundary_plots.append(
        {
            "type": "scatter",
            "x": [n_total],
            "y": [k],
            "mode": "markers",
            "marker": {"color": "#e89547", "size": 12, "symbol": "star"},
            "name": f"Observed ({n_total}, {k})",
        }
    )
    result["plots"].append(
        {
            "title": "Decision Boundaries",
            "data": boundary_plots,
            "layout": {
                "height": 300,
                "xaxis": {"title": "Sample Size (n)"},
                "yaxis": {"title": "Defectives (k)"},
            },
        }
    )

    result["guide_observation"] = (
        f"Bayesian acceptance: {k}/{n_total} defectives, P(p<AQL)={p_accept:.3f}, decision={decision}"
    )

    return result


_BAYES_SPC_DISPATCH = {
    "bayes_spc_capability": run_bayes_spc_capability,
    "bayes_spc_changepoint": run_bayes_spc_changepoint,
    "bayes_spc_control": run_bayes_spc_control,
    "bayes_spc_acceptance": run_bayes_spc_acceptance,
}
