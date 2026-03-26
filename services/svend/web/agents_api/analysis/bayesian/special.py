"""DSW Bayesian Special Analyses — changepoint, capability prediction, meta-analysis, EWMA."""

import numpy as np
from scipy import stats

from ..common import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
    _narrative,
    _rgba,
)


def run_bayes_changepoint(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian change point detection via BIC-approximated Bayes Factors
    var = config.get("var")
    time_col = config.get("time")
    max_cp = int(config.get("max_cp", 2))

    data = df[var].dropna().values
    n = len(data)

    if time_col:
        time_idx = df[time_col].loc[df[var].dropna().index].values
    else:
        time_idx = np.arange(n)

    def _seg_bic(segment):
        """BIC for a segment under N(mu_hat, sigma_hat^2): n*log(sigma_hat^2) + 2*log(n)."""
        m = len(segment)
        if m < 2:
            return 0.0
        ss = np.sum((segment - np.mean(segment)) ** 2)
        return m * np.log(max(ss / m, 1e-15)) + 2 * np.log(m)

    # Iteratively find change points by scanning within segments
    min_seg = max(3, n // 20)
    segments = [(0, n)]  # list of (start, end) boundaries
    changepoints = []  # list of (index, bayes_factor)

    for _ in range(max_cp):
        best_bf = 0.0
        best_cp = None
        best_seg_idx = None

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_data = data[seg_start:seg_end]
            seg_n = len(seg_data)

            if seg_n < 2 * min_seg:
                continue

            bic_null = _seg_bic(seg_data)

            for tau in range(min_seg, seg_n - min_seg + 1):
                bic_alt = _seg_bic(seg_data[:tau]) + _seg_bic(seg_data[tau:])
                bf = np.exp((bic_null - bic_alt) / 2)

                if bf > best_bf:
                    best_bf = bf
                    best_cp = seg_start + tau
                    best_seg_idx = seg_idx

        if best_cp is not None and best_bf > 3:  # moderate evidence
            changepoints.append((best_cp, best_bf))
            old_start, old_end = segments[best_seg_idx]
            segments[best_seg_idx] = (old_start, best_cp)
            segments.insert(best_seg_idx + 1, (best_cp, old_end))
        else:
            break

    changepoints.sort(key=lambda x: x[0])

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
    summary += "<<COLOR:dim>>Method: BIC-approximated Bayes Factor scan<</COLOR>>\n\n"

    if len(changepoints) > 0:
        summary += "<<COLOR:accent>>── Change Points ──<</COLOR>>\n"
        summary += f"  <<COLOR:success>>Detected {len(changepoints)} change point(s)<</COLOR>>\n\n"
        for i, (cp, bf) in enumerate(changepoints):
            before = data[:cp]
            after = data[cp:]
            shift = np.mean(after) - np.mean(before)
            pooled_std = (
                np.sqrt((np.var(before) + np.var(after)) / 2)
                if len(before) > 1 and len(after) > 1
                else 1.0
            )
            effect_d = abs(shift) / pooled_std if pooled_std > 0 else 0.0
            summary += f"  Point {i + 1}: index {cp}\n"
            summary += f"    BF₁₀ = {bf:.1f}"
            summary += f"  |  before μ = {np.mean(before):.4f}, after μ = {np.mean(after):.4f}\n"
            summary += (
                f"    Shift = {shift:+.4f}  |  Effect size (d) = {effect_d:.2f}\n"
            )
    else:
        summary += "<<COLOR:accent>>── Result ──<</COLOR>>\n"
        summary += "  <<COLOR:text>>No significant change points detected (BF₁₀ < 3)<</COLOR>>\n"

    summary += "\n<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
    if len(changepoints) > 0:
        best_bf = max(bf for _, bf in changepoints)
        if best_bf > 10:
            summary += "  <<COLOR:success>>Strong evidence for at least one process shift<</COLOR>>\n"
        else:
            summary += (
                "  <<COLOR:warning>>Moderate evidence for process shift(s)<</COLOR>>\n"
            )
    else:
        summary += "  <<COLOR:text>>Process appears stable — no evidence of mean shifts<</COLOR>>\n"

    result["summary"] = summary
    cp_indices = [cp for cp, _ in changepoints]
    cp_bfs = [bf for _, bf in changepoints]
    result["statistics"] = {
        "n_changepoints": len(changepoints),
        "changepoint_indices": cp_indices,
        "bayes_factors": cp_bfs,
    }

    # Guide observation
    if changepoints:
        result["guide_observation"] = (
            f"Bayesian changepoint: {len(changepoints)} shift(s) detected in {var}. Best BF₁₀={max(cp_bfs):.1f}."
        )
    else:
        result["guide_observation"] = (
            f"Bayesian changepoint: no significant shifts detected in {var} (n={n})."
        )

    # Narrative
    if changepoints:
        _bcp_best = max(changepoints, key=lambda x: x[1])
        _bcp_idx, _bcp_bf = _bcp_best
        _bcp_before = data[:_bcp_idx]
        _bcp_after = data[_bcp_idx:]
        _bcp_shift = float(np.mean(_bcp_after) - np.mean(_bcp_before))
        result["narrative"] = _narrative(
            f"Bayesian Changepoint — {len(changepoints)} shift{'s' if len(changepoints) > 1 else ''} detected",
            f"Strongest change at observation {_bcp_idx} (BF\u2081\u2080 = {_bcp_bf:.1f}): mean shifted by {_bcp_shift:+.4f}. "
            + (
                f"Total of {len(changepoints)} change points in {n} observations."
                if len(changepoints) > 1
                else ""
            ),
            next_steps="Investigate what happened at the change point(s). Align with process logs or external events.",
            chart_guidance="Red dashed vertical lines mark detected shifts. Compare the mean level before and after each line.",
        )
    else:
        result["narrative"] = _narrative(
            "Bayesian Changepoint — no shifts detected",
            f"No significant mean shifts found in {n} observations of {var}. The process appears stable.",
            next_steps="If you suspect a shift, try a smaller minimum segment size or add more data.",
        )

    # Time series plot with change points
    plot_data = [
        {
            "type": "scatter",
            "x": time_idx.tolist() if hasattr(time_idx, "tolist") else list(time_idx),
            "y": data.tolist(),
            "mode": "lines+markers",
            "marker": {"size": 4, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e"},
            "name": var,
        }
    ]

    for cp_idx, _cp_bf in changepoints:
        plot_data.append(
            {
                "type": "scatter",
                "x": [time_idx[cp_idx], time_idx[cp_idx]],
                "y": [min(data), max(data)],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
                "name": f"Change @ {cp_idx}",
            }
        )

    result["plots"].append(
        {
            "title": "Time Series with Change Points",
            "data": plot_data,
            "layout": {
                "height": 350,
                "xaxis": {"title": time_col or "Index"},
                "yaxis": {"title": var},
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Changepoint Detection",
        "content": (
            "<dl>"
            "<dt>What is changepoint detection?</dt>"
            "<dd>It identifies the most likely point(s) in a time series where the underlying "
            "process shifted — a change in mean, variance, or both. Unlike control chart alarms "
            "that flag individual points, this pinpoints <em>when</em> the regime changed.</dd>"
            "<dt>How does the Bayesian approach work?</dt>"
            "<dd>For each candidate split point, the model compares two hypotheses: 'one segment' "
            "vs 'two segments with different parameters'. BIC-approximated Bayes Factors rank "
            "all candidate points by evidence strength.</dd>"
            "<dt>What does the Bayes Factor mean here?</dt>"
            "<dd>A higher BF₁₀ at a candidate point means stronger evidence that a real shift "
            "occurred there. <strong>BF₁₀ &gt; 10</strong>: strong evidence of a changepoint. "
            "Multiple changepoints are detected iteratively by segmenting recursively.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>After process interventions (new material, equipment change, shift handover), "
            "to verify whether a suspected change actually occurred and locate it precisely. "
            "Also useful for segmenting historical data into stable regimes before running SPC.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_capability_prediction(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    """
    Bayesian Capability Prediction — posterior predictive distribution on Cp/Cpk.
    "Not just what IS my Cpk, but what WILL it be after N more samples."
    Uses Normal-Inverse-Chi-Squared conjugate prior for (mu, sigma²).
    """
    var = config.get("var") or config.get("var1")
    lsl = config.get("lsl")
    usl = config.get("usl")

    if lsl is None and usl is None:
        result["summary"] = "Error: Specify at least one spec limit (LSL and/or USL)."
        return result

    lsl = float(lsl) if lsl is not None and lsl != "" else None
    usl = float(usl) if usl is not None and usl != "" else None
    _target = float(
        config.get(
            "target",
            (
                (lsl + usl) / 2
                if lsl is not None and usl is not None
                else (lsl if lsl is not None else usl)
            ),
        )
    )

    x = df[var].dropna().values.astype(float)
    n = len(x)
    if n < 3:
        result["summary"] = "Need at least 3 data points."
        return result

    x_bar = float(np.mean(x))
    s2 = float(np.var(x, ddof=1))

    # Vague Normal-Inverse-Chi-Squared prior
    mu_0 = x_bar  # center on data (vague)
    kappa_0 = 0.01  # very low prior weight
    nu_0 = 0.01  # very low prior df
    s2_0 = s2  # prior variance centered on sample

    # Posterior parameters (conjugate update)
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
    nu_n = nu_0 + n
    s2_n = (
        nu_0 * s2_0 + (n - 1) * s2 + (kappa_0 * n * (x_bar - mu_0) ** 2) / kappa_n
    ) / nu_n

    # Draw from posterior predictive for Cpk
    n_draws = 10000
    rng = np.random.default_rng(42)

    # Sample sigma² from Inverse-Chi-Squared(nu_n, s2_n)
    sigma2_draws = nu_n * s2_n / rng.chisquare(nu_n, size=n_draws)
    sigma_draws = np.sqrt(sigma2_draws)

    # Sample mu from Normal(mu_n, sigma²/kappa_n)
    mu_draws = rng.normal(mu_n, np.sqrt(sigma2_draws / kappa_n))

    # Compute Cpk for each draw
    cpk_draws = np.zeros(n_draws)
    cp_draws = np.zeros(n_draws)
    for i in range(n_draws):
        if lsl is not None and usl is not None:
            cp_draws[i] = (usl - lsl) / (6 * sigma_draws[i])
            cpu = (usl - mu_draws[i]) / (3 * sigma_draws[i])
            cpl = (mu_draws[i] - lsl) / (3 * sigma_draws[i])
            cpk_draws[i] = min(cpu, cpl)
        elif usl is not None:
            cp_draws[i] = cpk_draws[i] = (usl - mu_draws[i]) / (3 * sigma_draws[i])
        else:
            cp_draws[i] = cpk_draws[i] = (mu_draws[i] - lsl) / (3 * sigma_draws[i])

    # Credible intervals
    cpk_mean = float(np.mean(cpk_draws))
    cpk_median = float(np.median(cpk_draws))
    cpk_ci = (
        float(np.percentile(cpk_draws, 2.5)),
        float(np.percentile(cpk_draws, 97.5)),
    )
    cp_mean = float(np.mean(cp_draws))

    # P(Cpk > threshold) for common targets
    cpk_targets = [1.0, 1.33, 1.5, 1.67, 2.0]
    prob_above = {t: float(np.mean(cpk_draws > t)) for t in cpk_targets}

    # Predictive: how many more samples to reach 95% confidence Cpk > 1.33?
    _future_ns = [10, 20, 50, 100, 200, 500]
    _future_probs = []
    for fn in _future_ns:
        kappa_f = kappa_n + fn
        nu_f = nu_n + fn
        sigma_f_draws = nu_f * s2_n / rng.chisquare(nu_f, size=3000)
        sigma_f = np.sqrt(sigma_f_draws)
        mu_f = rng.normal(mu_n, np.sqrt(sigma_f_draws / kappa_f))
        if lsl is not None and usl is not None:
            cpu_f = (usl - mu_f) / (3 * sigma_f)
            cpl_f = (mu_f - lsl) / (3 * sigma_f)
            cpk_f = np.minimum(cpu_f, cpl_f)
        elif usl is not None:
            cpk_f = (usl - mu_f) / (3 * sigma_f)
        else:
            cpk_f = (mu_f - lsl) / (3 * sigma_f)
        _future_probs.append(float(np.mean(cpk_f > 1.33)))

    # Summary
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CAPABILITY PREDICTION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n"
    if lsl is not None:
        summary += f"<<COLOR:highlight>>LSL:<</COLOR>> {lsl}\n"
    if usl is not None:
        summary += f"<<COLOR:highlight>>USL:<</COLOR>> {usl}\n"
    summary += "\n<<COLOR:accent>>── Posterior Capability ──<</COLOR>>\n"
    summary += f"  Cpk (posterior mean): {cpk_mean:.3f}\n"
    summary += f"  Cpk (posterior median): {cpk_median:.3f}\n"
    summary += f"  95% Credible Interval: [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}]\n"
    if lsl is not None and usl is not None:
        summary += f"  Cp (posterior mean): {cp_mean:.3f}\n"

    summary += "\n<<COLOR:accent>>── P(Cpk > target) ──<</COLOR>>\n"
    for t in cpk_targets:
        p = prob_above[t]
        color = "good" if p > 0.9 else ("highlight" if p > 0.5 else "bad")
        summary += f"  <<COLOR:{color}>>P(Cpk > {t:.2f}) = {p:.1%}<</COLOR>>\n"

    summary += "\n<<COLOR:accent>>── Sample Size Forecast ──<</COLOR>>\n"
    summary += "  Additional samples → P(Cpk > 1.33):\n"
    for fn, fp in zip(_future_ns, _future_probs):
        summary += f"    n + {fn:<5} → {fp:.1%}\n"

    result["summary"] = summary

    # Posterior Cpk distribution plot
    result["plots"].append(
        {
            "title": "Posterior Cpk Distribution",
            "data": [
                {
                    "type": "histogram",
                    "x": cpk_draws.tolist(),
                    "marker": {
                        "color": "rgba(74, 159, 110, 0.4)",
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                    "name": "Posterior Cpk",
                    "nbinsx": 60,
                }
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Cpk"},
                "yaxis": {"title": "Frequency"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 1.33,
                        "x1": 1.33,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    },
                    {
                        "type": "line",
                        "x0": cpk_ci[0],
                        "x1": cpk_ci[0],
                        "y0": 0,
                        "y1": 0.05,
                        "yref": "paper",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                    {
                        "type": "line",
                        "x0": cpk_ci[1],
                        "x1": cpk_ci[1],
                        "y0": 0,
                        "y1": 0.05,
                        "yref": "paper",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                ],
                "annotations": [
                    {
                        "x": 1.33,
                        "y": 1,
                        "yref": "paper",
                        "text": "Target 1.33",
                        "showarrow": False,
                        "font": {"color": "#e89547", "size": 10},
                    }
                ],
            },
        }
    )

    # Sample size forecast plot
    result["plots"].append(
        {
            "title": "Sample Size Forecast: P(Cpk > 1.33)",
            "data": [
                {
                    "type": "scatter",
                    "x": [n + fn for fn in _future_ns],
                    "y": [p * 100 for p in _future_probs],
                    "mode": "lines+markers",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "marker": {"size": 8},
                    "name": "P(Cpk > 1.33)",
                },
                {
                    "type": "scatter",
                    "x": [n + _future_ns[0], n + _future_ns[-1]],
                    "y": [95, 95],
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "95% confidence",
                },
            ],
            "layout": {
                "height": 280,
                "xaxis": {"title": "Total Sample Size"},
                "yaxis": {"title": "P(Cpk > 1.33) %", "range": [0, 105]},
            },
        }
    )

    # Narrative
    _cap_label = (
        "capable"
        if cpk_mean >= 1.33
        else ("marginally capable" if cpk_mean >= 1.0 else "not capable")
    )
    _confidence_133 = prob_above[1.33]
    result["narrative"] = _narrative(
        f"Process is {_cap_label} (Cpk = {cpk_mean:.3f}, 95% CI [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}])",
        f"There is a <strong>{_confidence_133:.0%}</strong> probability that true Cpk exceeds 1.33. "
        + (
            f"The 95% credible interval [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}] {'entirely exceeds' if cpk_ci[0] > 1.33 else 'straddles'} the 1.33 target."
            if lsl is not None and usl is not None
            else f"One-sided capability index = {cpk_mean:.3f}."
        ),
        next_steps="The sample size forecast shows how confidence improves with more data. "
        + (
            f"With {_future_ns[2]} more samples, P(Cpk > 1.33) reaches {_future_probs[2]:.0%}."
            if _future_probs[2] < 0.95
            else "Current sample size provides strong confidence."
        ),
        chart_guidance="The histogram shows the posterior belief about Cpk. The dashed line at 1.33 is the typical capability target. Points to the right of this line represent 'capable' outcomes.",
    )

    result["guide_observation"] = (
        f"Bayesian Cpk = {cpk_mean:.3f} (95% CI [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}]). P(Cpk > 1.33) = {_confidence_133:.1%}."
    )
    result["statistics"] = {
        "cpk_mean": cpk_mean,
        "cpk_median": cpk_median,
        "cpk_ci_low": cpk_ci[0],
        "cpk_ci_high": cpk_ci[1],
        "cp_mean": cp_mean,
        "prob_above_133": _confidence_133,
        "n": n,
    }

    result["education"] = {
        "title": "Understanding Bayesian Capability Prediction",
        "content": (
            "<dl>"
            "<dt>What is Bayesian Cpk prediction?</dt>"
            "<dd>Traditional Cpk is a point estimate from your current sample. Bayesian Cpk "
            "prediction uses a <em>Normal-Inverse-Chi-Squared</em> conjugate prior to produce "
            "a full posterior distribution on Cpk — telling you the probability that your "
            "process is truly capable, not just that a single number exceeded a threshold.</dd>"
            "<dt>What does P(Cpk &ge; 1.33) mean?</dt>"
            "<dd>The posterior probability that the true process capability exceeds the common "
            "threshold of 1.33. <strong>&gt; 90%</strong>: strong confidence in capability. "
            "<strong>50–90%</strong>: moderate confidence, consider collecting more data. "
            "<strong>&lt; 50%</strong>: the process is more likely incapable than capable.</dd>"
            "<dt>Why not just use the point Cpk?</dt>"
            "<dd>A Cpk of 1.4 from 20 samples is far less certain than 1.4 from 200 samples. "
            "The Bayesian approach captures this uncertainty — the posterior width shrinks "
            "as you add data, giving you an honest picture of how confident the estimate is.</dd>"
            "<dt>What is the predictive distribution?</dt>"
            "<dd>It forecasts what Cpk you would expect after collecting additional samples. "
            "This lets you plan sample sizes: how many more measurements do you need before "
            "the credible interval is narrow enough to make a confident capability decision?</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_meta(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Random-Effects Meta-Analysis (Normal-Normal hierarchical)
    effects_col = config.get("var1")
    se_col = config.get("var2")

    y = df[effects_col].dropna().values.astype(float)
    se = df[se_col].dropna().values.astype(float)
    k = len(y)

    if k < 2:
        result["summary"] = "Error: Need at least 2 studies."
        return result

    n_use = min(len(y), len(se))
    y, se = y[:n_use], se[:n_use]
    k = n_use

    # Grid posterior over tau (between-study SD)
    tau_max = max(3 * np.std(y), 1.0)
    n_tau = 200
    tau_range = np.linspace(0, tau_max, n_tau)
    log_marginal = np.zeros(n_tau)

    for i, tau in enumerate(tau_range):
        w = 1.0 / (se**2 + tau**2 + 1e-15)
        mu_hat = np.sum(w * y) / np.sum(w)
        var_total = se**2 + tau**2
        log_marginal[i] = -0.5 * np.sum(
            np.log(2 * np.pi * var_total) + (y - mu_hat) ** 2 / var_total
        )

    # Posterior on tau (flat prior)
    log_post_tau = log_marginal - log_marginal.max()
    post_tau = np.exp(log_post_tau)
    post_tau /= post_tau.sum() * (tau_range[1] - tau_range[0])

    tau_pmf = post_tau * (tau_range[1] - tau_range[0])
    tau_pmf /= tau_pmf.sum()
    tau_mean = float(np.sum(tau_range * tau_pmf))
    tau_cdf = np.cumsum(tau_pmf)
    tau_ci = (
        float(tau_range[np.searchsorted(tau_cdf, (1 - ci_level) / 2)]),
        float(tau_range[min(n_tau - 1, np.searchsorted(tau_cdf, (1 + ci_level) / 2))]),
    )

    # Posterior on mu (integrate over tau)
    n_mc = 10000
    rng = np.random.default_rng(42)
    tau_samples = tau_range[rng.choice(n_tau, size=n_mc, p=tau_pmf)]
    mu_samples = np.zeros(n_mc)
    for j in range(n_mc):
        w = 1.0 / (se**2 + tau_samples[j] ** 2 + 1e-15)
        mu_hat = np.sum(w * y) / np.sum(w)
        mu_se = 1.0 / np.sqrt(np.sum(w))
        mu_samples[j] = rng.normal(mu_hat, mu_se)

    mu_mean = float(np.mean(mu_samples))
    mu_ci = (
        float(np.percentile(mu_samples, (1 - ci_level) / 2 * 100)),
        float(np.percentile(mu_samples, (1 + ci_level) / 2 * 100)),
    )

    # Study-specific shrunken estimates
    shrunk = []
    for j_study in range(k):
        w_j = 1.0 / (se[j_study] ** 2 + tau_mean**2 + 1e-15)
        w_pool = np.sum(1.0 / (se**2 + tau_mean**2 + 1e-15))
        shrink_factor = w_j / (w_j + 1.0 / (1.0 / w_pool + 1e-15))
        est = shrink_factor * y[j_study] + (1 - shrink_factor) * mu_mean
        shrunk.append(float(est))

    i2 = (
        float(tau_mean**2 / (tau_mean**2 + np.mean(se**2)) * 100)
        if tau_mean > 0
        else 0.0
    )
    het_label = (
        "high"
        if i2 > 75
        else ("moderate" if i2 > 50 else ("low" if i2 > 25 else "negligible"))
    )

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN META-ANALYSIS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Studies:<</COLOR>> {k}\n\n"
    summary += (
        "<<COLOR:accent>>\u2500\u2500 Pooled Effect (\u03bc) \u2500\u2500<</COLOR>>\n"
    )
    summary += f"  Posterior mean: {mu_mean:.4f}\n"
    summary += f"  95% Credible Interval: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]\n\n"
    summary += (
        "<<COLOR:accent>>\u2500\u2500 Heterogeneity (\u03c4) \u2500\u2500<</COLOR>>\n"
    )
    summary += f"  Posterior mean: {tau_mean:.4f}\n"
    summary += f"  95% CI: [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]\n"
    summary += f"  I\u00b2 analog: {i2:.1f}% ({het_label})\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Study Estimates \u2500\u2500<</COLOR>>\n"
    for j_study in range(k):
        summary += f"  Study {j_study + 1}: {y[j_study]:.4f} \u00b1 {se[j_study]:.4f} \u2192 shrunk {shrunk[j_study]:.4f}\n"

    result["summary"] = summary
    result["statistics"] = {
        "mu_mean": mu_mean,
        "mu_ci": list(mu_ci),
        "tau_mean": tau_mean,
        "tau_ci": list(tau_ci),
        "i2": i2,
        "k": k,
        "shrunk_estimates": shrunk,
    }
    result["guide_observation"] = (
        f"Bayesian meta-analysis ({k} studies): pooled = {mu_mean:.4f} (95% CI: {mu_ci[0]:.4f}\u2013{mu_ci[1]:.4f}), \u03c4 = {tau_mean:.4f}, I\u00b2 = {i2:.0f}%."
    )

    result["narrative"] = _narrative(
        f"Bayesian Meta-Analysis \u2014 pooled effect = {mu_mean:.4f}, I\u00b2 = {i2:.0f}% ({het_label})",
        f"Across {k} studies, the pooled effect is {mu_mean:.4f} (95% credible interval: {mu_ci[0]:.4f} to {mu_ci[1]:.4f}). "
        f"Between-study heterogeneity \u03c4 = {tau_mean:.4f} (I\u00b2 \u2248 {i2:.0f}%, {het_label}). "
        + (
            "The CI excludes zero, supporting a real effect."
            if (mu_ci[0] > 0 or mu_ci[1] < 0)
            else "The CI includes zero \u2014 the overall effect is uncertain."
        ),
        next_steps="High I\u00b2 means the studies disagree. Investigate moderators (subgroup analysis) to explain the heterogeneity. "
        "The shrunken estimates show how each study's estimate is pulled toward the grand mean.",
        chart_guidance="The forest plot shows each study's estimate (with CI) and the shrunken Bayesian estimate. "
        "The diamond at the bottom is the pooled posterior.",
    )

    # Forest plot
    study_labels = [f"Study {i + 1}" for i in range(k)]
    result["plots"].append(
        {
            "title": "Bayesian Forest Plot",
            "data": [
                {
                    "type": "scatter",
                    "y": study_labels,
                    "x": y.tolist(),
                    "error_x": {"type": "data", "array": (z * se).tolist()},
                    "mode": "markers",
                    "marker": {"size": 8, "color": "#4a90d9"},
                    "name": "Observed",
                },
                {
                    "type": "scatter",
                    "y": study_labels,
                    "x": shrunk,
                    "mode": "markers",
                    "marker": {"size": 8, "symbol": "diamond", "color": "#d4a24a"},
                    "name": "Shrunken",
                },
                {
                    "type": "scatter",
                    "y": ["Pooled"],
                    "x": [mu_mean],
                    "error_x": {
                        "type": "data",
                        "symmetric": False,
                        "array": [mu_ci[1] - mu_mean],
                        "arrayminus": [mu_mean - mu_ci[0]],
                    },
                    "mode": "markers",
                    "marker": {"size": 14, "symbol": "diamond", "color": "#4a9f6e"},
                    "name": "Pooled",
                },
            ],
            "layout": {
                "height": max(200, k * 35 + 100),
                "xaxis": {"title": "Effect Size"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 0,
                        "x1": 0,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#888", "dash": "dash"},
                    }
                ],
            },
        }
    )

    # Posterior on tau
    result["plots"].append(
        {
            "title": "Posterior on Heterogeneity (\u03c4)",
            "data": [
                {
                    "type": "scatter",
                    "x": tau_range.tolist(),
                    "y": (post_tau * (tau_range[1] - tau_range[0])).tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(212, 162, 74, 0.3)",
                    "line": {"color": "#d4a24a", "width": 2},
                    "name": "\u03c4 Posterior",
                }
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Between-study SD (\u03c4)"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Meta-Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian meta-analysis?</dt>"
            "<dd>It combines effect sizes from multiple studies into a single pooled estimate "
            "using a <em>Normal-Normal hierarchical</em> model. Each study contributes in "
            "proportion to its precision, and between-study heterogeneity (\u03c4) is estimated.</dd>"
            "<dt>What is \u03c4 (tau)?</dt>"
            "<dd>The between-study standard deviation \u2014 how much true effect sizes vary across "
            "studies. <strong>\u03c4 \u2248 0</strong>: studies agree, fixed-effect model suffices. "
            "<strong>\u03c4 large</strong>: substantial heterogeneity, the pooled estimate is less "
            "certain and individual study contexts matter more.</dd>"
            "<dt>What is the pooled effect?</dt>"
            "<dd>The posterior mean of the overall effect size, accounting for both within-study "
            "uncertainty and between-study variability. Its credible interval is wider than "
            "any individual study because it honestly propagates heterogeneity.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Combining results across multiple experiments, plants, or time periods. "
            "For example: pooling Cpk estimates from 5 production lines, combining treatment "
            "effects from repeated trials, or synthesising defect rate studies.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_ewma(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian EWMA — EWMA smoothing with posterior inference for shift detection
    measurement = config.get("measurement")
    target = config.get("target")
    lambda_param = float(config.get("lambda_param", 0.2))
    L = float(config.get("L", 3))
    prior_scale_name = config.get("prior_scale", "medium")

    data = df[measurement].dropna().values
    n = len(data)

    if target is None or target == 0:
        target = float(np.mean(data))
    else:
        target = float(target)

    sigma = float(np.std(data, ddof=1))
    if sigma < 1e-15:
        sigma = 1.0

    # Prior precision scaling
    scale_map = {"tight": 5.0, "medium": 1.0, "wide": 0.2}
    kappa_0 = scale_map.get(prior_scale_name, 1.0)

    # ── EWMA smoothing (same as spc.py) ──
    ewma = np.zeros(n)
    ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target
    for i in range(1, n):
        ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i - 1]

    # Classical variable control limits
    factor = lambda_param / (2 - lambda_param)
    indices = np.arange(1, n + 1)
    cl_sigma = sigma * np.sqrt(factor * (1 - (1 - lambda_param) ** (2 * indices)))
    ucl = target + L * cl_sigma
    lcl = target - L * cl_sigma
    ucl_ss = target + L * sigma * np.sqrt(factor)
    lcl_ss = target - L * sigma * np.sqrt(factor)

    # OOC detection
    ooc_indices = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

    # ── Bayesian posterior inference ──
    # Conjugate Normal posterior: prior N(target, sigma^2/kappa_0)
    # At each step, update with EWMA observation as a pseudo-observation
    # with effective sample size proportional to smoothing weight
    ewma_var = sigma**2 * factor  # steady-state EWMA variance
    posterior_means = np.zeros(n)
    posterior_vars = np.zeros(n)
    shift_probs = np.zeros(n)

    prior_var = sigma**2 / kappa_0
    prior_mean = target

    for i in range(n):
        # Effective observation precision from EWMA
        obs_var = sigma**2 * factor * (1 - (1 - lambda_param) ** (2 * (i + 1)))
        if obs_var < 1e-15:
            obs_var = ewma_var

        obs_precision = 1.0 / obs_var
        prior_precision = 1.0 / prior_var

        # Posterior update
        post_precision = prior_precision + obs_precision
        post_var = 1.0 / post_precision
        post_mean = post_var * (prior_precision * prior_mean + obs_precision * ewma[i])

        posterior_means[i] = post_mean
        posterior_vars[i] = post_var

        # P(|mu - target| > 1sigma) — probability of meaningful shift
        delta = sigma  # 1-sigma shift threshold
        post_std = np.sqrt(post_var)
        if post_std > 0:
            p_above = 1 - stats.norm.cdf(target + delta, post_mean, post_std)
            p_below = stats.norm.cdf(target - delta, post_mean, post_std)
            shift_probs[i] = p_above + p_below
        else:
            shift_probs[i] = 1.0 if abs(post_mean - target) > delta else 0.0

        # Sequential update: current posterior becomes next prior
        prior_mean = post_mean
        prior_var = post_var + ewma_var * 0.1  # add process noise

    # ── Overall Bayes Factor ──
    # Compare H1: mean != target vs H0: mean = target
    # Using final posterior: BF10 ~ P(data|H1)/P(data|H0)
    # Savage-Dickey density ratio at mu = target
    final_mean = posterior_means[-1]
    final_std = np.sqrt(posterior_vars[-1])
    prior_std_0 = sigma / np.sqrt(kappa_0)

    # Density of posterior at target / density of prior at target
    post_density_at_target = stats.norm.pdf(target, final_mean, final_std)
    prior_density_at_target = stats.norm.pdf(target, target, prior_std_0)

    if post_density_at_target > 0:
        bf10 = prior_density_at_target / post_density_at_target
    else:
        bf10 = 100.0  # very strong evidence for shift

    bf10 = max(bf10, 0.01)  # floor

    # BF label
    if bf10 > 100:
        bf_label = "extreme"
    elif bf10 > 30:
        bf_label = "very strong"
    elif bf10 > 10:
        bf_label = "strong"
    elif bf10 > 3:
        bf_label = "moderate"
    elif bf10 > 1:
        bf_label = "anecdotal"
    else:
        bf_label = "supports null"

    # Credible intervals
    ci_mult = z  # z from ci_level at top of function
    ci_upper = posterior_means + ci_mult * np.sqrt(posterior_vars)
    ci_lower = posterior_means - ci_mult * np.sqrt(posterior_vars)

    max_shift_prob = float(np.max(shift_probs))

    # ── Statistics ──
    result["statistics"] = {
        "n": n,
        "lambda_param": lambda_param,
        "L": L,
        "target": float(target),
        "sigma": sigma,
        "n_ooc": len(ooc_indices),
        "ucl_steady": float(ucl_ss),
        "lcl_steady": float(lcl_ss),
        "max_shift_prob": max_shift_prob,
        "bf10": float(bf10),
        "bf_label": bf_label,
    }

    # ── Summary ──
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN EWMA ANALYSIS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {measurement}\n"
    summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
    summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target:.4f}\n"
    summary += f"<<COLOR:dim>>\u03bb = {lambda_param}, L = {L}, prior = {prior_scale_name}<</COLOR>>\n\n"
    summary += "<<COLOR:accent>>── Control Limits (steady-state) ──<</COLOR>>\n"
    summary += f"  UCL: {ucl_ss:.4f}\n"
    summary += f"  LCL: {lcl_ss:.4f}\n\n"
    summary += "<<COLOR:accent>>── Bayesian Inference ──<</COLOR>>\n"
    summary += f"  BF\u2081\u2080 = {bf10:.2f} ({bf_label} evidence)\n"
    summary += f"  Max P(shift) = {max_shift_prob:.4f}\n"
    summary += f"  Out-of-control points: {len(ooc_indices)}\n\n"

    if len(ooc_indices) == 0 and bf10 < 3:
        summary += "<<COLOR:success>>Process appears stable — no Bayesian evidence of shift<</COLOR>>\n"
    elif bf10 >= 10:
        summary += (
            "<<COLOR:error>>Strong Bayesian evidence of process shift<</COLOR>>\n"
        )
    elif bf10 >= 3:
        summary += (
            "<<COLOR:warning>>Moderate Bayesian evidence of process shift<</COLOR>>\n"
        )
    else:
        summary += f"<<COLOR:text>>OOC points detected but Bayesian evidence is {bf_label}<</COLOR>>\n"

    result["summary"] = summary

    # ── Guide observation ──
    if len(ooc_indices) == 0 and bf10 < 3:
        result["guide_observation"] = (
            f"Bayesian EWMA: process stable. BF\u2081\u2080={bf10:.2f} ({bf_label}), 0 OOC points."
        )
    else:
        result["guide_observation"] = (
            f"Bayesian EWMA: {len(ooc_indices)} OOC point{'s' if len(ooc_indices) != 1 else ''}. "
            f"BF\u2081\u2080={bf10:.2f} ({bf_label}), max P(shift)={max_shift_prob:.4f}."
        )

    # ── Narrative ──
    if len(ooc_indices) == 0 and bf10 < 3:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No EWMA points exceed control limits (\u03bb={lambda_param}, L={L}). "
            f"Bayesian analysis confirms stability: BF\u2081\u2080 = {bf10:.2f} ({bf_label}), "
            f"maximum posterior shift probability = {max_shift_prob:.4f}.",
            next_steps="Process is stable — EWMA is sensitive to small sustained shifts; "
            "the Bayesian posterior provides additional confidence in this conclusion.",
            chart_guidance="The EWMA line smooths out noise. The shaded band shows the 95% credible interval "
            "for the true process mean. The shift probability plot shows where the posterior "
            "concentrates away from target.",
        )
    else:
        result["narrative"] = _narrative(
            f"Bayesian EWMA — {'strong' if bf10 >= 10 else 'moderate' if bf10 >= 3 else 'weak'} evidence of shift",
            f"The smoothed mean has {'exceeded control limits' if ooc_indices else 'shifted'} "
            f"with BF\u2081\u2080 = {bf10:.2f} ({bf_label}). "
            f"Maximum posterior probability of a 1\u03c3 shift = {max_shift_prob:.4f}.",
            next_steps="Identify when the drift began (peak in shift probability plot) and "
            "correlate with process changes. The Bayesian credible intervals show "
            "where uncertainty about the true mean is highest.",
            chart_guidance="Red diamonds mark classical OOC points. The purple shift probability "
            "curve shows Bayesian confidence in a process shift at each observation. "
            "Values near 1.0 indicate near-certainty of drift.",
        )

    # ── Plots ──
    x_list = list(range(n))

    # Plot 1: EWMA Control Chart
    ewma_chart_data = [
        {
            "type": "scatter",
            "x": x_list,
            "y": ewma.tolist(),
            "mode": "lines+markers",
            "name": "EWMA",
            "marker": {
                "color": _rgba(COLOR_GOOD, 0.4),
                "size": 5,
                "line": {"color": COLOR_GOOD, "width": 1.5},
            },
            "line": {"color": COLOR_GOOD},
        },
        {
            "type": "scatter",
            "x": x_list,
            "y": [target] * n,
            "mode": "lines",
            "name": "Target",
            "line": {"color": COLOR_REFERENCE, "width": 1, "dash": "dot"},
        },
        {
            "type": "scatter",
            "x": x_list,
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
        },
        {
            "type": "scatter",
            "x": x_list,
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL",
            "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
        },
    ]

    # Add OOC markers
    if ooc_indices:
        ewma_chart_data.append(
            {
                "type": "scatter",
                "x": ooc_indices,
                "y": [ewma[i] for i in ooc_indices],
                "mode": "markers",
                "name": "OOC",
                "marker": {"color": COLOR_BAD, "size": 10, "symbol": "diamond"},
            }
        )

    result["plots"].append(
        {
            "title": "Bayesian EWMA Control Chart",
            "data": ewma_chart_data,
            "layout": {
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": measurement},
            },
        }
    )

    # Plot 2: Posterior Shift Probability
    result["plots"].append(
        {
            "title": "Posterior Probability of Shift",
            "data": [
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": shift_probs.tolist(),
                    "mode": "lines",
                    "name": "P(shift > 1\u03c3)",
                    "line": {"color": SVEND_COLORS[4], "width": 2},
                },
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": [0.95] * n,
                    "mode": "lines",
                    "name": "95% threshold",
                    "line": {"color": COLOR_WARNING, "width": 1, "dash": "dash"},
                },
            ],
            "layout": {
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "P(shift)", "range": [0, 1.05]},
            },
        }
    )

    # Plot 3: Credible Interval Band
    result["plots"].append(
        {
            "title": f"{int(ci_level * 100)}% Credible Interval for Process Mean",
            "data": [
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": ci_upper.tolist(),
                    "mode": "lines",
                    "name": f"Upper {int(ci_level * 100)}% CI",
                    "line": {"color": _rgba(SVEND_COLORS[1], 0.3), "width": 0},
                },
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": ci_lower.tolist(),
                    "mode": "lines",
                    "name": f"Lower {int(ci_level * 100)}% CI",
                    "line": {"color": _rgba(SVEND_COLORS[1], 0.3), "width": 0},
                    "fill": "tonexty",
                    "fillcolor": _rgba(SVEND_COLORS[1], 0.15),
                },
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": posterior_means.tolist(),
                    "mode": "lines",
                    "name": "Posterior Mean",
                    "line": {"color": SVEND_COLORS[1], "width": 2},
                },
                {
                    "type": "scatter",
                    "x": x_list,
                    "y": [target] * n,
                    "mode": "lines",
                    "name": "Target",
                    "line": {"color": COLOR_REFERENCE, "width": 1, "dash": "dot"},
                },
            ],
            "layout": {
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Process Mean"},
            },
        }
    )

    # ── Education ──
    result["education"] = {
        "title": "Understanding Bayesian EWMA",
        "content": (
            "<dl>"
            "<dt>What is Bayesian EWMA?</dt>"
            "<dd>It combines the classical EWMA (Exponentially Weighted Moving Average) chart "
            "with Bayesian posterior inference. Instead of just flagging points outside \u00b1L\u03c3 limits, "
            "it estimates the <em>posterior probability</em> that the process mean has shifted at "
            "each observation \u2014 giving you a continuous measure of confidence in process stability.</dd>"
            "<dt>How is it different from classical EWMA?</dt>"
            "<dd>Classical EWMA gives binary signals (in-control/out-of-control). Bayesian EWMA "
            "gives <em>probabilities</em>: 'there is a 92% posterior probability the process mean "
            "has shifted by more than 1\u03c3 from target.' It also provides credible intervals around "
            "the estimated process mean \u2014 the Bayesian analogue of confidence intervals.</dd>"
            "<dt>What is the Bayes Factor here?</dt>"
            "<dd>BF\u2081\u2080 compares two hypotheses: H\u2081 (the process has shifted from target) vs "
            "H\u2080 (the process is still at target). BF\u2081\u2080 &gt; 10 is strong evidence of shift; "
            "BF\u2081\u2080 &lt; 1/3 is evidence of stability. It uses the Savage-Dickey density ratio "
            "from the posterior distribution.</dd>"
            "<dt>When to use Bayesian EWMA over classical?</dt>"
            "<dd>When you need to <em>quantify</em> shift evidence rather than just detect it. "
            "Particularly useful for small sample sizes where classical control limits may be "
            "unreliable, or when you need to communicate shift risk probabilistically to "
            "decision-makers (e.g., 'P(shifted) = 0.87' vs 'one OOC point').</dd>"
            "</dl>"
        ),
    }

    return result
