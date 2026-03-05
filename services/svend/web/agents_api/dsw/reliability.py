"""DSW Reliability & Survival Analysis — Weibull, Kaplan-Meier, ALT, warranty, etc."""

import numpy as np
from scipy import stats as sp_stats

from .common import SVEND_COLORS, _narrative


def run_reliability_analysis(df, analysis_id, config):
    """Run reliability/survival analysis."""

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "weibull":
        # Weibull Distribution Analysis
        time_col = config.get("time")
        censor_col = config.get("censor")  # optional: 1=failed, 0=censored

        times = df[time_col].dropna().values
        times = times[times > 0]

        if censor_col and censor_col in df.columns:
            censor = df[censor_col].dropna().values[: len(times)]
            failed = times[censor == 1]
        else:
            failed = times

        # Fit Weibull: scipy uses (c, loc, scale) = (shape/beta, loc, scale/eta)
        shape, loc, scale = sp_stats.weibull_min.fit(failed, floc=0)

        # Probability plot data (Weibull)
        sorted_t = np.sort(failed)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)  # Bernard's approximation

        # Theoretical line
        t_range = np.linspace(sorted_t[0] * 0.8, sorted_t[-1] * 1.2, 200)
        cdf_fit = sp_stats.weibull_min.cdf(t_range, shape, 0, scale)

        # Probability plot (linearized)
        result["plots"].append(
            {
                "title": "Weibull Probability Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": np.log(sorted_t).tolist(),
                        "y": np.log(-np.log(1 - median_ranks)).tolist(),
                        "mode": "markers",
                        "name": "Data",
                        "marker": {"color": "#4a9f6e", "size": 7},
                    },
                    {
                        "type": "scatter",
                        "x": np.log(t_range[cdf_fit > 0]).tolist(),
                        "y": np.log(-np.log(1 - cdf_fit[cdf_fit > 0])).tolist() if np.any(cdf_fit > 0) else [],
                        "mode": "lines",
                        "name": f"Weibull (β={shape:.2f}, η={scale:.1f})",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "ln(Time)"},
                    "yaxis": {"title": "ln(-ln(1-F))"},
                    "showlegend": True,
                },
            }
        )

        # Reliability curve
        result["plots"].append(
            {
                "title": "Reliability Function",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": (1 - cdf_fit).tolist(),
                        "mode": "lines",
                        "name": "R(t)",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(74, 159, 110, 0.15)",
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability R(t)", "range": [0, 1]},
                },
            }
        )

        # B-life calculations
        b10 = sp_stats.weibull_min.ppf(0.10, shape, 0, scale)
        b50 = sp_stats.weibull_min.ppf(0.50, shape, 0, scale)
        import math

        mttf = scale * np.exp(math.lgamma(1 + 1 / shape))

        summary = f"Weibull Analysis\n\nShape (β): {shape:.4f}\nScale (η): {scale:.2f}\n\nB10 Life: {b10:.2f}\nB50 Life (median): {b50:.2f}\nMTTF: {mttf:.2f}\n\n"
        if shape < 1:
            summary += "β < 1: Decreasing failure rate (infant mortality)"
        elif shape == 1:
            summary += "β ≈ 1: Constant failure rate (random failures)"
        else:
            summary += "β > 1: Increasing failure rate (wear-out)"

        result["summary"] = summary
        result["guide_observation"] = f"Weibull β={shape:.2f}. " + (
            "Infant mortality pattern." if shape < 1 else "Wear-out pattern." if shape > 1 else "Random failures."
        )

        # Narrative
        _wb_phase = (
            "infant mortality (decreasing failure rate)"
            if shape < 1
            else ("random failures (constant rate)" if abs(shape - 1) < 0.1 else "wear-out (increasing failure rate)")
        )
        result["narrative"] = _narrative(
            f"Weibull Analysis — \u03b2 = {shape:.3f} ({_wb_phase})",
            f"Shape \u03b2 = {shape:.3f}, Scale \u03b7 = {scale:.2f}. B10 life (10% failures) = {b10:.2f}. Median life (B50) = {b50:.2f}. MTTF = {mttf:.2f}.",
            next_steps="For wear-out (\u03b2 > 1), schedule preventive replacement before B10. For infant mortality (\u03b2 < 1), improve screening/burn-in.",
            chart_guidance="Points on the probability plot diagonal = good Weibull fit. Curvature suggests a different distribution (try lognormal or mixed Weibull).",
        )

    elif analysis_id == "lognormal":
        # Lognormal Distribution Analysis
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]

        # Fit lognormal
        shape_ln, loc_ln, scale_ln = sp_stats.lognorm.fit(times, floc=0)
        mu = np.log(scale_ln)
        sigma = shape_ln

        sorted_t = np.sort(times)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

        t_range = np.linspace(sorted_t[0] * 0.5, sorted_t[-1] * 1.5, 200)
        cdf_fit = sp_stats.lognorm.cdf(t_range, shape_ln, 0, scale_ln)

        # Probability plot (lognormal linearized)
        result["plots"].append(
            {
                "title": "Lognormal Probability Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": np.log(sorted_t).tolist(),
                        "y": sp_stats.norm.ppf(median_ranks).tolist(),
                        "mode": "markers",
                        "name": "Data",
                        "marker": {"color": "#4a9f6e", "size": 7},
                    },
                    {
                        "type": "scatter",
                        "x": np.log(t_range[cdf_fit > 0]).tolist(),
                        "y": sp_stats.norm.ppf(np.clip(cdf_fit[cdf_fit > 0], 1e-10, 1 - 1e-10)).tolist(),
                        "mode": "lines",
                        "name": f"Lognormal (μ={mu:.2f}, σ={sigma:.2f})",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "ln(Time)"},
                    "yaxis": {"title": "Std Normal Quantile"},
                    "showlegend": True,
                },
            }
        )

        # Reliability curve
        result["plots"].append(
            {
                "title": "Reliability Function",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": (1 - cdf_fit).tolist(),
                        "mode": "lines",
                        "name": "R(t)",
                        "line": {"color": "#4a90d9", "width": 2},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(74, 144, 217, 0.15)",
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability R(t)", "range": [0, 1]},
                },
            }
        )

        b10 = sp_stats.lognorm.ppf(0.10, shape_ln, 0, scale_ln)
        b50 = sp_stats.lognorm.ppf(0.50, shape_ln, 0, scale_ln)
        mean_life = np.exp(mu + sigma**2 / 2)

        result["summary"] = (
            f"Lognormal Analysis\n\nμ (log mean): {mu:.4f}\nσ (log std): {sigma:.4f}\n\nB10 Life: {b10:.2f}\nB50 Life (median): {b50:.2f}\nMean Life: {mean_life:.2f}"
        )
        result["guide_observation"] = (
            f"Lognormal: \u03bc={mu:.3f}, \u03c3={sigma:.3f}. B10={b10:.2f}, median={b50:.2f}."
        )

        # Narrative
        _ln_skew = "highly right-skewed" if sigma > 1 else ("moderately skewed" if sigma > 0.5 else "near-symmetric")
        result["narrative"] = _narrative(
            f"Lognormal Analysis — B10 = {b10:.2f}, median = {b50:.2f}",
            f"\u03bc = {mu:.4f}, \u03c3 = {sigma:.4f}. Mean life = {mean_life:.2f}. Distribution is {_ln_skew} (log-std = {sigma:.3f}). "
            f"Note: mean ({mean_life:.2f}) exceeds median ({b50:.2f}) due to right skew — median is a better reliability planning metric.",
            next_steps="Use B10 life for warranty planning. If the probability plot shows curvature, try Weibull or mixed distributions.",
            chart_guidance="Points on the diagonal in the probability plot = good lognormal fit.",
        )

    elif analysis_id == "exponential":
        # Exponential Distribution Analysis
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]

        # MLE for exponential: rate = 1/mean
        mttf = np.mean(times)
        rate = 1.0 / mttf

        sorted_t = np.sort(times)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

        t_range = np.linspace(0, sorted_t[-1] * 1.5, 200)
        rel = np.exp(-rate * t_range)

        # Exponential probability plot (linearized: ln(1-F) vs t)
        result["plots"].append(
            {
                "title": "Exponential Probability Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": sorted_t.tolist(),
                        "y": (-np.log(1 - median_ranks)).tolist(),
                        "mode": "markers",
                        "name": "Data",
                        "marker": {"color": "#4a9f6e", "size": 7},
                    },
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": (rate * t_range).tolist(),
                        "mode": "lines",
                        "name": f"Exp (λ={rate:.4f})",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "-ln(1-F)"},
                    "showlegend": True,
                },
            }
        )

        # Reliability curve
        result["plots"].append(
            {
                "title": "Reliability Function",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": rel.tolist(),
                        "mode": "lines",
                        "name": "R(t)",
                        "line": {"color": "#e89547", "width": 2},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(232, 149, 71, 0.15)",
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability R(t)", "range": [0, 1]},
                },
            }
        )

        # Confidence interval on MTTF (chi-squared)
        chi2_lower = sp_stats.chi2.ppf(0.025, 2 * n)
        chi2_upper = sp_stats.chi2.ppf(0.975, 2 * n)
        mttf_lower = 2 * n * mttf / chi2_upper
        mttf_upper = 2 * n * mttf / chi2_lower

        result["summary"] = (
            f"Exponential Analysis\n\nFailure rate (λ): {rate:.6f}\nMTTF: {mttf:.2f}\n95% CI on MTTF: [{mttf_lower:.2f}, {mttf_upper:.2f}]\n\nSample size: {n}\n\nNote: Exponential assumes constant failure rate (no wear-out)."
        )
        result["guide_observation"] = (
            f"Exponential: \u03bb={rate:.6f}, MTTF={mttf:.2f} (95% CI: {mttf_lower:.2f}-{mttf_upper:.2f})."
        )

        # Narrative
        result["narrative"] = _narrative(
            f"Exponential Analysis — MTTF = {mttf:.2f}",
            f"Failure rate \u03bb = {rate:.6f} per unit time. MTTF = {mttf:.2f} (95% CI: {mttf_lower:.2f} to {mttf_upper:.2f}). "
            f"The exponential model assumes a constant failure rate — appropriate for random failures (Weibull \u03b2 \u2248 1), not wear-out.",
            next_steps="Verify the constant-rate assumption with a Weibull analysis. If \u03b2 departs significantly from 1, use Weibull instead.",
            chart_guidance="Straight line on the exponential probability plot confirms the constant-rate assumption. Curvature indicates wear-out or infant mortality.",
        )

    elif analysis_id == "kaplan_meier":
        # Kaplan-Meier Survival Analysis
        time_col = config.get("time")
        event_col = config.get("event")  # 1=event occurred, 0=censored

        times = df[time_col].dropna().values
        if event_col and event_col in df.columns:
            events = df[event_col].dropna().values[: len(times)]
        else:
            events = np.ones(len(times))

        # Sort by time
        order = np.argsort(times)
        times = times[order]
        events = events[order]

        # KM estimator
        unique_times = np.unique(times[events == 1])
        n_at_risk = len(times)
        s = 1.0
        km_times = [0]
        km_survival = [1.0]
        km_ci_lower = [1.0]
        km_ci_upper = [1.0]
        var_sum = 0

        for t in unique_times:
            d = np.sum((times == t) & (events == 1))
            c = np.sum((times == t) & (events == 0))
            n = n_at_risk
            if n > 0:
                s *= 1 - d / n
                if d > 0 and n > d:
                    var_sum += d / (n * (n - d))
            n_at_risk -= d + c

            # Greenwood confidence interval
            se = s * np.sqrt(var_sum) if var_sum > 0 else 0
            km_times.append(float(t))
            km_survival.append(float(s))
            km_ci_lower.append(max(0, float(s - 1.96 * se)))
            km_ci_upper.append(min(1, float(s + 1.96 * se)))

        # Censored points
        cens_t = times[events == 0]
        # Interpolate survival at censored times
        cens_s = []
        for ct in cens_t:
            idx = np.searchsorted(km_times, ct, side="right") - 1
            cens_s.append(km_survival[max(0, idx)])

        # Build censored mark customdata with at-risk counts
        _cens_cd = []
        for ct, cs in zip(cens_t, cens_s):
            _n_at_risk = int(np.sum(times >= ct))
            _cens_cd.append([float(ct), _n_at_risk])

        # Number-at-risk annotations at ~6 evenly spaced time points
        _nar_annotations = []
        if len(km_times) > 1:
            _t_min, _t_max = km_times[0], km_times[-1]
            _nar_ticks = np.linspace(_t_min, _t_max, min(6, len(km_times)))
            for _nt in _nar_ticks:
                _nar = int(np.sum(times >= _nt))
                _nar_annotations.append(
                    {
                        "x": float(_nt),
                        "y": -0.08,
                        "yref": "paper",
                        "text": str(_nar),
                        "showarrow": False,
                        "font": {"size": 10, "color": "#7A8F7A"},
                    }
                )
            # Label
            _nar_annotations.append(
                {
                    "x": float(_t_min) - (_t_max - _t_min) * 0.05,
                    "y": -0.08,
                    "yref": "paper",
                    "text": "At risk:",
                    "showarrow": False,
                    "font": {"size": 10, "color": "#7A8F7A", "weight": "bold"},
                    "xanchor": "right",
                }
            )

        _cens_trace = []
        if len(cens_t) > 0:
            _cens_trace = [
                {
                    "type": "scatter",
                    "x": cens_t.tolist(),
                    "y": cens_s,
                    "mode": "markers",
                    "name": "Censored",
                    "marker": {"color": "#4a90d9", "size": 8, "symbol": "cross"},
                    "customdata": _cens_cd,
                    "hovertemplate": "Time: %{x:.2f}<br>S(t): %{y:.4f}<br>At risk: %{customdata[1]}<extra>Censored</extra>",
                }
            ]

        result["plots"].append(
            {
                "title": "Kaplan-Meier Survival Curve",
                "data": [
                    {
                        "type": "scatter",
                        "x": km_times,
                        "y": km_survival,
                        "mode": "lines",
                        "name": "Survival",
                        "line": {"color": "#4a9f6e", "width": 2, "shape": "hv"},
                    },
                    {
                        "type": "scatter",
                        "x": km_times,
                        "y": km_ci_upper,
                        "mode": "lines",
                        "name": "95% CI",
                        "line": {"color": "#4a9f6e", "width": 1, "dash": "dot", "shape": "hv"},
                        "showlegend": False,
                    },
                    {
                        "type": "scatter",
                        "x": km_times,
                        "y": km_ci_lower,
                        "mode": "lines",
                        "name": "95% CI",
                        "line": {"color": "#4a9f6e", "width": 1, "dash": "dot", "shape": "hv"},
                        "fill": "tonexty",
                        "fillcolor": "rgba(74, 159, 110, 0.15)",
                    },
                ]
                + _cens_trace,
                "layout": {
                    "height": 340,
                    "margin": {"b": 60},
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                    "showlegend": True,
                    "annotations": _nar_annotations,
                },
            }
        )

        n_events = int(np.sum(events))
        n_censored = int(np.sum(events == 0))
        median_survival = "N/A"
        for i, s_val in enumerate(km_survival):
            if s_val <= 0.5:
                median_survival = f"{km_times[i]:.2f}"
                break

        result["summary"] = (
            f"Kaplan-Meier Survival Analysis\n\nTotal observations: {len(times)}\nEvents: {n_events}\nCensored: {n_censored}\n\nMedian survival time: {median_survival}\nFinal survival probability: {km_survival[-1]:.4f}"
        )
        result["guide_observation"] = (
            f"Kaplan-Meier: {n_events} events, {n_censored} censored. Median survival: {median_survival}."
        )

        # Narrative
        _km_cens_pct = n_censored / len(times) * 100 if len(times) > 0 else 0
        result["narrative"] = _narrative(
            f"Kaplan-Meier Survival — median = {median_survival}",
            f"{len(times)} observations: {n_events} events, {n_censored} censored ({_km_cens_pct:.0f}%). "
            f"Final survival probability = {km_survival[-1]:.4f}."
            + (
                f" Median survival time = {median_survival}."
                if median_survival != "N/A"
                else " Median not reached — more than 50% survived the observation period."
            ),
            next_steps="Compare groups with a log-rank test. Fit a parametric distribution (Weibull/lognormal) for extrapolation beyond observed times.",
            chart_guidance="The step function shows estimated survival over time. Crosses mark censored observations. Dotted lines are 95% confidence bands.",
        )

    elif analysis_id == "reliability_test_plan":
        # Reliability Test Planning — sample size for demonstration testing
        target_rel = float(config.get("target_reliability", 0.90))
        confidence = float(config.get("confidence", 0.95))
        test_duration = float(config.get("test_duration", 1000))
        dist = config.get("distribution", "exponential")

        if dist == "exponential":
            # For exponential: n = ln(1-C) / ln(R) where 0 failures
            n_required = int(np.ceil(np.log(1 - confidence) / np.log(target_rel)))

            # With allowed failures
            from scipy.special import comb

            results_table = []
            for failures in range(6):
                # Sum of binomial terms
                sum(
                    comb(n_required + failures, k) * (1 - target_rel) ** k * target_rel ** (n_required + failures - k)
                    for k in range(failures + 1)
                )
                n_for_f = n_required + failures
                # Adjust n until confidence is met
                n_adj = n_for_f
                while True:
                    cum = sum(
                        comb(n_adj, k) * (1 - target_rel) ** k * target_rel ** (n_adj - k) for k in range(failures + 1)
                    )
                    if 1 - cum >= confidence:
                        break
                    n_adj += 1
                    if n_adj > 10000:
                        break
                results_table.append({"failures": failures, "sample_size": n_adj})

            # Bar chart of sample sizes by allowed failures
            result["plots"].append(
                {
                    "title": "Required Sample Size vs Allowed Failures",
                    "data": [
                        {
                            "type": "bar",
                            "x": [f"{r['failures']} failures" for r in results_table],
                            "y": [r["sample_size"] for r in results_table],
                            "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a", "#9f4a4a", "#7a6a9a"]},
                            "text": [str(r["sample_size"]) for r in results_table],
                            "textposition": "outside",
                        }
                    ],
                    "layout": {"height": 340, "yaxis": {"title": "Sample Size"}},
                }
            )

            summary = f"Reliability Demonstration Test Plan\n\nTarget Reliability: {target_rel * 100:.1f}%\nConfidence Level: {confidence * 100:.1f}%\nTest Duration: {test_duration}\nDistribution: Exponential\n\nRequired Sample Sizes:\n"
            for r in results_table:
                summary += f"  {r['failures']} allowed failures: n = {r['sample_size']}\n"
            summary += f"\nZero-failure plan: Test {n_required} units for {test_duration} each with 0 failures."

            result["summary"] = summary
            result["guide_observation"] = (
                f"Reliability test plan: target R={target_rel * 100:.1f}% at {confidence * 100:.0f}% confidence. Zero-failure plan: n={n_required}."
            )
            result["narrative"] = _narrative(
                f"Reliability Test Plan — n = {n_required} (zero failures)",
                f"To demonstrate {target_rel * 100:.1f}% reliability at {confidence * 100:.0f}% confidence, test {n_required} units for {test_duration} each with zero failures allowed. "
                f"Allowing 1 failure increases n to {results_table[1]['sample_size']}."
                if len(results_table) > 1
                else "",
                next_steps="Fewer failures allowed = smaller sample but stricter pass criteria. Consider the cost trade-off.",
                chart_guidance="The bar chart shows how sample size increases as you allow more failures while maintaining the same confidence.",
            )

        elif dist == "weibull":
            beta = float(config.get("beta", 2.0))
            # For Weibull with shape beta
            n_required = int(np.ceil(np.log(1 - confidence) / np.log(target_rel)))
            (test_duration / test_duration) ** beta  # acceleration factor placeholder (ratio = 1 if test = use)

            result["plots"].append(
                {
                    "title": "Test Plan Parameters",
                    "data": [
                        {
                            "type": "bar",
                            "x": ["Sample Size", "Test Duration"],
                            "y": [n_required, test_duration],
                            "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                            "text": [str(n_required), str(test_duration)],
                            "textposition": "outside",
                        }
                    ],
                    "layout": {"height": 340, "yaxis": {"title": "Value"}},
                }
            )

            result["summary"] = (
                f"Reliability Test Plan (Weibull)\n\nTarget Reliability: {target_rel * 100:.1f}%\nConfidence: {confidence * 100:.1f}%\nWeibull β: {beta}\n\nZero-failure plan: Test {n_required} units for {test_duration} each."
            )
            result["guide_observation"] = (
                f"Weibull test plan: target R={target_rel * 100:.1f}%, \u03b2={beta}, n={n_required}."
            )
            result["narrative"] = _narrative(
                f"Reliability Test Plan (Weibull) — n = {n_required}",
                f"For Weibull \u03b2 = {beta}, test {n_required} units for {test_duration} each with zero failures to demonstrate {target_rel * 100:.1f}% reliability at {confidence * 100:.0f}% confidence.",
                next_steps="If \u03b2 is uncertain, use a conservative (higher) value to ensure adequate sample size.",
            )

    elif analysis_id == "distribution_id":
        """
        Distribution Identification — fits multiple distributions,
        ranks by goodness-of-fit, shows probability plots for top fits.
        """
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]
        n = len(times)

        distributions = {}

        # Normal
        mu, sigma = sp_stats.norm.fit(times)
        ks = sp_stats.kstest(times, "norm", args=(mu, sigma))
        distributions["Normal"] = {
            "dist": sp_stats.norm,
            "args": (mu, sigma),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"μ={mu:.2f}, σ={sigma:.2f}",
        }

        # Lognormal
        s, loc, scale = sp_stats.lognorm.fit(times, floc=0)
        ks = sp_stats.kstest(times, "lognorm", args=(s, 0, scale))
        distributions["Lognormal"] = {
            "dist": sp_stats.lognorm,
            "args": (s, 0, scale),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"μ={np.log(scale):.2f}, σ={s:.2f}",
        }

        # Weibull
        c, loc_w, scale_w = sp_stats.weibull_min.fit(times, floc=0)
        ks = sp_stats.kstest(times, "weibull_min", args=(c, 0, scale_w))
        distributions["Weibull"] = {
            "dist": sp_stats.weibull_min,
            "args": (c, 0, scale_w),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"β={c:.2f}, η={scale_w:.2f}",
        }

        # Exponential
        loc_e, scale_e = sp_stats.expon.fit(times)
        ks = sp_stats.kstest(times, "expon", args=(loc_e, scale_e))
        distributions["Exponential"] = {
            "dist": sp_stats.expon,
            "args": (loc_e, scale_e),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"λ={1 / scale_e:.4f}",
        }

        # Gamma
        a, loc_g, scale_g = sp_stats.gamma.fit(times, floc=0)
        ks = sp_stats.kstest(times, "gamma", args=(a, 0, scale_g))
        distributions["Gamma"] = {
            "dist": sp_stats.gamma,
            "args": (a, 0, scale_g),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"α={a:.2f}, β={scale_g:.2f}",
        }

        # Loglogistic (use fisk distribution in scipy)
        c_ll, loc_ll, scale_ll = sp_stats.fisk.fit(times, floc=0)
        ks = sp_stats.kstest(times, "fisk", args=(c_ll, 0, scale_ll))
        distributions["Loglogistic"] = {
            "dist": sp_stats.fisk,
            "args": (c_ll, 0, scale_ll),
            "ks_stat": ks.statistic,
            "ks_p": ks.pvalue,
            "params": f"μ={np.log(scale_ll):.2f}, σ={1 / c_ll:.2f}",
        }

        # Rank by KS p-value (higher = better fit)
        ranked = sorted(distributions.items(), key=lambda x: x[1]["ks_p"], reverse=True)

        summary = f"Distribution Identification\n\nSample size: {n}\n\n"
        summary += f"{'Distribution':<15} {'Parameters':<25} {'KS Stat':>10} {'p-value':>10}\n"
        summary += f"{'-' * 65}\n"
        for i, (name, info) in enumerate(ranked):
            marker = " <-- Best" if i == 0 else ""
            summary += f"{name:<15} {info['params']:<25} {info['ks_stat']:>10.4f} {info['ks_p']:>10.4f}{marker}\n"

        best_name, best_info = ranked[0]
        summary += f"\nRecommended: {best_name} ({best_info['params']})"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Distribution ID: best fit = {best_name} (KS p={best_info['ks_p']:.4f}). {n} data points."
        )

        # Narrative
        _di_second = ranked[1][0] if len(ranked) > 1 else "N/A"
        _di_good_fit = best_info["ks_p"] > 0.05
        result["narrative"] = _narrative(
            f"Distribution Identification — {best_name} (best fit)",
            f"Tested {len(ranked)} distributions on {n} data points. Best fit: <strong>{best_name}</strong> (KS p = {best_info['ks_p']:.4f}). "
            + (
                "Good fit (p > 0.05). "
                if _di_good_fit
                else "Marginal fit (p < 0.05) — consider transforming data or mixed distributions. "
            )
            + f"Runner-up: {_di_second}.",
            next_steps="Use the best-fit distribution for reliability predictions, capability analysis, and simulation inputs.",
            chart_guidance="Points on the diagonal in probability plots = good fit. The histogram overlay shows how well each distribution's PDF matches the data.",
        )

        # Probability plots for top 3 distributions
        sorted_t = np.sort(times)
        median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        theme_colors = SVEND_COLORS[:3]

        for idx, (name, info) in enumerate(ranked[:3]):
            dist = info["dist"]
            args = info["args"]
            theoretical = dist.ppf(median_ranks, *args)

            result["plots"].append(
                {
                    "title": f"Probability Plot — {name}" + (" (Best)" if idx == 0 else ""),
                    "data": [
                        {
                            "type": "scatter",
                            "x": theoretical.tolist(),
                            "y": sorted_t.tolist(),
                            "mode": "markers",
                            "name": "Data",
                            "marker": {"color": theme_colors[idx], "size": 5},
                        },
                        {
                            "type": "scatter",
                            "x": [float(min(theoretical)), float(max(theoretical))],
                            "y": [float(min(theoretical)), float(max(theoretical))],
                            "mode": "lines",
                            "name": "Reference",
                            "line": {"color": "#d94a4a", "dash": "dash"},
                        },
                    ],
                    "layout": {
                        "height": 340,
                        "showlegend": True,
                        "xaxis": {"title": f"Theoretical ({name})"},
                        "yaxis": {"title": "Observed"},
                    },
                }
            )

        # Overlay histogram with top 3 PDFs
        x_range = np.linspace(min(times), max(times), 200)
        hist_traces = [
            {
                "type": "histogram",
                "x": times.tolist(),
                "name": "Data",
                "marker": {"color": "rgba(74,159,110,0.3)", "line": {"color": "#4a9f6e", "width": 1}},
                "histnorm": "probability density",
            }
        ]
        for idx, (name, info) in enumerate(ranked[:3]):
            pdf = info["dist"].pdf(x_range, *info["args"])
            hist_traces.append(
                {
                    "type": "scatter",
                    "x": x_range.tolist(),
                    "y": pdf.tolist(),
                    "mode": "lines",
                    "name": name,
                    "line": {"color": theme_colors[idx], "width": 2},
                }
            )
        result["plots"].append(
            {
                "title": "Distribution Comparison",
                "data": hist_traces,
                "layout": {
                    "height": 340,
                    "showlegend": True,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

    elif analysis_id == "accelerated_life":
        """
        Accelerated Life Testing — fits life-stress model.
        Supports Arrhenius (temperature) and Inverse Power Law (voltage/stress).
        """
        time_col = config.get("time")
        stress_col = config.get("stress")
        model_type = config.get("model", "arrhenius")  # arrhenius or inverse_power
        use_stress = float(config.get("use_stress", 25))  # use condition

        times = df[time_col].dropna().values
        stresses = df[stress_col].dropna().values[: len(times)]

        unique_stresses = np.sort(np.unique(stresses))
        if len(unique_stresses) < 2:
            result["summary"] = "Error: Need at least 2 stress levels for ALT."
            return result

        # Fit Weibull at each stress level
        stress_results = []
        for stress in unique_stresses:
            mask = stresses == stress
            t_at_stress = times[mask]
            t_at_stress = t_at_stress[t_at_stress > 0]
            if len(t_at_stress) < 3:
                continue
            shape, _, scale = sp_stats.weibull_min.fit(t_at_stress, floc=0)
            stress_results.append({"stress": float(stress), "shape": shape, "scale": scale, "n": len(t_at_stress)})

        if len(stress_results) < 2:
            result["summary"] = "Error: Not enough data at each stress level (need 3+ per level)."
            return result

        # Common shape assumption (average)
        common_shape = np.mean([r["shape"] for r in stress_results])

        # Fit life-stress model: ln(scale) = a + b * transform(stress)
        log_scales = np.array([np.log(r["scale"]) for r in stress_results])
        stress_vals = np.array([r["stress"] for r in stress_results])

        if model_type == "arrhenius":
            # Arrhenius: ln(L) = a + b/T  (T in Kelvin)
            x_transform = 1.0 / (stress_vals + 273.15)  # assume Celsius
            x_use = 1.0 / (use_stress + 273.15)
            stress_label = "1/T (K)"
        else:
            # Inverse Power: ln(L) = a - b*ln(S)
            x_transform = np.log(stress_vals)
            x_use = np.log(use_stress)
            stress_label = "ln(Stress)"

        # Linear regression
        slope, intercept, r_value, _, _ = sp_stats.linregress(x_transform, log_scales)
        log_scale_use = intercept + slope * x_use
        scale_use = np.exp(log_scale_use)

        # Life at use conditions
        b10_use = sp_stats.weibull_min.ppf(0.10, common_shape, 0, scale_use)
        b50_use = sp_stats.weibull_min.ppf(0.50, common_shape, 0, scale_use)
        import math

        mttf_use = scale_use * np.exp(math.lgamma(1 + 1 / common_shape))

        summary = "Accelerated Life Testing\n\n"
        summary += f"Model: {'Arrhenius' if model_type == 'arrhenius' else 'Inverse Power Law'}\n"
        summary += f"Use Stress: {use_stress}\n"
        summary += f"Common Shape (β): {common_shape:.3f}\n\n"
        summary += "Stress Level Results:\n"
        summary += f"  {'Stress':>10} {'n':>5} {'Shape':>8} {'Scale':>10}\n"
        summary += f"  {'-' * 38}\n"
        for r in stress_results:
            summary += f"  {r['stress']:>10.1f} {r['n']:>5} {r['shape']:>8.3f} {r['scale']:>10.1f}\n"
        summary += f"\nLife-Stress Model: R² = {r_value**2:.4f}\n"
        summary += f"\nExtrapolated Life at Use Conditions ({use_stress}):\n"
        summary += f"  Scale (η): {scale_use:.1f}\n"
        summary += f"  B10 Life: {b10_use:.1f}\n"
        summary += f"  B50 Life: {b50_use:.1f}\n"
        summary += f"  MTTF: {mttf_use:.1f}"

        result["summary"] = summary
        result["guide_observation"] = (
            f"ALT ({model_type}): R\u00b2={r_value**2:.3f}. At use stress ({use_stress}): B10={b10_use:.1f}, MTTF={mttf_use:.1f}."
        )

        # Narrative
        _alt_r2 = r_value**2
        _alt_fit = "strong" if _alt_r2 > 0.9 else ("adequate" if _alt_r2 > 0.7 else "weak")
        result["narrative"] = _narrative(
            f"Accelerated Life Testing — MTTF at use = {mttf_use:.1f}",
            f"{'Arrhenius' if model_type == 'arrhenius' else 'Inverse Power Law'} model with {_alt_fit} fit (R\u00b2 = {_alt_r2:.4f}). "
            f"At use stress ({use_stress}): B10 = {b10_use:.1f}, B50 = {b50_use:.1f}, MTTF = {mttf_use:.1f}. Common shape \u03b2 = {common_shape:.3f}.",
            next_steps="Verify the life-stress relationship is linear on the transformed scale. Use the extrapolated B10 for warranty planning.",
            chart_guidance="The star marker shows the extrapolated life at use conditions. Confidence in this estimate depends on the R\u00b2 of the life-stress fit.",
        )

        # Life vs Stress plot
        x_plot = np.linspace(min(x_transform) * 0.9, max(max(x_transform), x_use) * 1.1, 100)
        y_plot = np.exp(intercept + slope * x_plot)

        result["plots"].append(
            {
                "title": "Life vs Stress Relationship",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_transform.tolist(),
                        "y": [r["scale"] for r in stress_results],
                        "mode": "markers",
                        "name": "Test Data",
                        "marker": {"color": "#4a9f6e", "size": 10},
                    },
                    {
                        "type": "scatter",
                        "x": x_plot.tolist(),
                        "y": y_plot.tolist(),
                        "mode": "lines",
                        "name": "Model Fit",
                        "line": {"color": "#4a90d9", "width": 2},
                    },
                    {
                        "type": "scatter",
                        "x": [float(x_use)],
                        "y": [float(scale_use)],
                        "mode": "markers",
                        "name": f"Use ({use_stress})",
                        "marker": {"color": "#d94a4a", "size": 12, "symbol": "star"},
                    },
                ],
                "layout": {
                    "height": 340,
                    "showlegend": True,
                    "xaxis": {"title": stress_label},
                    "yaxis": {"title": "Characteristic Life (η)", "type": "log"},
                },
            }
        )

        # Reliability at use stress
        t_range = np.linspace(0, scale_use * 2, 200)
        rel_use = 1 - sp_stats.weibull_min.cdf(t_range, common_shape, 0, scale_use)
        result["plots"].append(
            {
                "title": f"Reliability at Use Stress ({use_stress})",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": rel_use.tolist(),
                        "mode": "lines",
                        "name": "R(t)",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(74,159,110,0.15)",
                    },
                ],
                "layout": {
                    "height": 340,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability", "range": [0, 1.05]},
                },
            }
        )

    elif analysis_id == "repairable_systems":
        """
        Repairable Systems — Crow-AMSAA (Power Law NHPP).
        Models failure intensity for repairable systems.
        """
        time_col = config.get("time")
        system_col = config.get("system")

        if system_col and system_col in df.columns:
            # Multiple systems
            systems = df[system_col].unique()
            all_events = []
            for sys in systems:
                events = np.sort(df[df[system_col] == sys][time_col].dropna().values)
                all_events.append(events)
        else:
            # Single system — all events
            all_events = [np.sort(df[time_col].dropna().values)]

        n_systems = len(all_events)
        total_events = sum(len(e) for e in all_events)

        # Pool all events for single-system analysis
        pooled = np.sort(np.concatenate(all_events))
        n = len(pooled)
        T = pooled[-1]  # total observation time

        # Fit Power Law (Crow-AMSAA): N(t) = (t/θ)^β
        # MLE: β = n / Σln(T/ti), θ = T / n^(1/β)
        log_sum = np.sum(np.log(T / pooled[pooled > 0]))
        beta_crow = n / log_sum if log_sum > 0 else 1.0
        theta_crow = T / (n ** (1 / beta_crow))

        # Laplace test for trend
        laplace_stat = (np.mean(pooled) - T / 2) / (T / np.sqrt(12 * n))
        laplace_p = 2 * (1 - sp_stats.norm.cdf(abs(laplace_stat)))

        summary = "Repairable Systems Analysis (Crow-AMSAA)\n\n"
        summary += f"Systems: {n_systems}\n"
        summary += f"Total Events: {total_events}\n"
        summary += f"Observation Period: {T:.1f}\n\n"
        summary += "Power Law Parameters:\n"
        summary += f"  β (shape): {beta_crow:.4f}\n"
        summary += f"  θ (scale): {theta_crow:.2f}\n\n"
        summary += "Trend Test (Laplace):\n"
        summary += f"  Statistic: {laplace_stat:.4f}\n"
        summary += f"  p-value: {laplace_p:.4f}\n"

        if laplace_p < 0.05:
            if laplace_stat > 0:
                summary += "  Result: DETERIORATING — failure rate increasing\n"
            else:
                summary += "  Result: IMPROVING — failure rate decreasing\n"
        else:
            summary += "  Result: NO TREND — stable failure rate (HPP)\n"

        if beta_crow > 1:
            summary += "\nβ > 1: System deteriorating (wear-out)"
        elif beta_crow < 1:
            summary += "\nβ < 1: System improving (reliability growth)"
        else:
            summary += "\nβ ≈ 1: Constant failure rate"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Crow-AMSAA: \u03b2={beta_crow:.3f}, \u03b8={theta_crow:.1f}. {'Deteriorating' if beta_crow > 1 else ('Improving' if beta_crow < 1 else 'Stable')} (Laplace p={laplace_p:.4f})."
        )

        # Narrative
        _rs_trend = (
            "deteriorating (failure rate increasing)"
            if beta_crow > 1
            else ("improving (reliability growth)" if beta_crow < 1 else "stable (constant failure rate)")
        )
        _rs_sig = "statistically significant" if laplace_p < 0.05 else "not statistically significant"
        result["narrative"] = _narrative(
            f"Repairable Systems — {_rs_trend}",
            f"Crow-AMSAA power law: \u03b2 = {beta_crow:.4f}, \u03b8 = {theta_crow:.2f}. {total_events} events across {n_systems} system{'s' if n_systems > 1 else ''} over {T:.1f} time units. "
            f"Laplace trend test: {_rs_sig} (p = {laplace_p:.4f}).",
            next_steps="For \u03b2 > 1 (deteriorating), consider preventive maintenance or design changes. For \u03b2 < 1, reliability growth is occurring — document what changed.",
            chart_guidance="The MCF shows cumulative failures over time. A concave-up curve = deteriorating. Concave-down = improving. Linear = constant rate.",
        )

        # MCF (Mean Cumulative Function) plot
        mcf_t = [0] + pooled.tolist()
        mcf_n = list(range(len(mcf_t)))
        # Fitted model: E[N(t)] = (t/θ)^β
        t_fit = np.linspace(0, T * 1.2, 200)
        n_fit = (t_fit / theta_crow) ** beta_crow

        result["plots"].append(
            {
                "title": "Mean Cumulative Function (MCF)",
                "data": [
                    {
                        "type": "scatter",
                        "x": mcf_t,
                        "y": mcf_n,
                        "mode": "lines",
                        "name": "Observed",
                        "line": {"color": "#4a9f6e", "width": 2, "shape": "hv"},
                    },
                    {
                        "type": "scatter",
                        "x": t_fit.tolist(),
                        "y": n_fit.tolist(),
                        "mode": "lines",
                        "name": f"Crow-AMSAA (β={beta_crow:.2f})",
                        "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
                    },
                ],
                "layout": {
                    "height": 340,
                    "showlegend": True,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Cumulative Events"},
                },
            }
        )

        # Instantaneous failure rate: λ(t) = (β/θ)(t/θ)^(β-1)
        t_rate = np.linspace(pooled[0] * 0.5, T * 1.1, 200)
        rate = (beta_crow / theta_crow) * (t_rate / theta_crow) ** (beta_crow - 1)
        result["plots"].append(
            {
                "title": "Failure Intensity (ROCOF)",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_rate.tolist(),
                        "y": rate.tolist(),
                        "mode": "lines",
                        "name": "λ(t)",
                        "line": {"color": "#e89547", "width": 2},
                    },
                ],
                "layout": {"height": 340, "xaxis": {"title": "Time"}, "yaxis": {"title": "Failure Rate"}},
            }
        )

    elif analysis_id == "warranty":
        """
        Warranty Prediction — forecasts future returns from field data.
        """
        time_col = config.get("time")  # time-to-return (age at return)
        warranty_period = float(config.get("warranty_period", 365))
        fleet_size = int(config.get("fleet_size", 1000))

        times = df[time_col].dropna().values
        times = times[times > 0]
        n_returns = len(times)

        # Fit Weibull to return times
        shape, _, scale = sp_stats.weibull_min.fit(times, floc=0)

        # Return rate function: F(t)
        t_range = np.linspace(0, warranty_period * 1.5, 300)
        cdf = sp_stats.weibull_min.cdf(t_range, shape, 0, scale)

        # Projected returns
        projected_in_warranty = fleet_size * sp_stats.weibull_min.cdf(warranty_period, shape, 0, scale)

        # Monthly projection
        months = int(warranty_period / 30)
        monthly_returns = []
        for m in range(months + 1):
            t = m * 30
            cum_returns = fleet_size * sp_stats.weibull_min.cdf(t, shape, 0, scale)
            monthly_returns.append({"month": m, "cumulative": cum_returns, "incremental": 0})
        for i in range(1, len(monthly_returns)):
            monthly_returns[i]["incremental"] = monthly_returns[i]["cumulative"] - monthly_returns[i - 1]["cumulative"]

        summary = "Warranty Prediction\n\n"
        summary += f"Observed Returns: {n_returns}\n"
        summary += f"Fleet Size: {fleet_size}\n"
        summary += f"Warranty Period: {warranty_period:.0f}\n\n"
        summary += "Fitted Distribution: Weibull\n"
        summary += f"  Shape (β): {shape:.3f}\n"
        summary += f"  Scale (η): {scale:.1f}\n\n"
        summary += f"Projected Returns in Warranty: {projected_in_warranty:.0f} ({projected_in_warranty / fleet_size * 100:.2f}%)\n\n"
        summary += "Monthly Forecast (next 6 months):\n"
        summary += f"  {'Month':>6} {'Incremental':>13} {'Cumulative':>12}\n"
        summary += f"  {'-' * 35}\n"
        for mr in monthly_returns[:7]:
            summary += f"  {mr['month']:>6} {mr['incremental']:>13.1f} {mr['cumulative']:>12.1f}\n"

        result["summary"] = summary
        _wr_pct = projected_in_warranty / fleet_size * 100
        result["guide_observation"] = (
            f"Warranty: {projected_in_warranty:.0f} projected returns ({_wr_pct:.2f}%) from {fleet_size} fleet. Weibull \u03b2={shape:.3f}."
        )

        # Narrative
        _wr_phase = "infant mortality" if shape < 1 else ("random" if abs(shape - 1) < 0.1 else "wear-out")
        result["narrative"] = _narrative(
            f"Warranty Prediction — {projected_in_warranty:.0f} returns ({_wr_pct:.2f}%)",
            f"Based on {n_returns} observed returns, projecting {projected_in_warranty:.0f} total returns within the {warranty_period:.0f}-day warranty for a fleet of {fleet_size:,}. "
            f"Weibull \u03b2 = {shape:.3f} indicates {_wr_phase} failure pattern. Scale \u03b7 = {scale:.1f}.",
            next_steps="Monitor actual vs projected returns monthly. If actuals exceed projections, investigate root cause.",
            chart_guidance="The cumulative curve shows expected return rate over product age. The dashed line marks warranty expiration.",
        )

        # Cumulative return rate
        result["plots"].append(
            {
                "title": "Cumulative Return Rate",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": (cdf * 100).tolist(),
                        "mode": "lines",
                        "name": "Projected %",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "fill": "tozeroy",
                        "fillcolor": "rgba(74,159,110,0.15)",
                    },
                    {
                        "type": "scatter",
                        "x": [warranty_period, warranty_period],
                        "y": [0, float(cdf[-1] * 100)],
                        "mode": "lines",
                        "name": "Warranty End",
                        "line": {"color": "#d94a4a", "dash": "dash", "width": 2},
                    },
                ],
                "layout": {
                    "height": 340,
                    "showlegend": True,
                    "xaxis": {"title": "Age (days)"},
                    "yaxis": {"title": "Cumulative Return Rate (%)"},
                },
            }
        )

        # Monthly incremental returns
        result["plots"].append(
            {
                "title": "Monthly Incremental Returns",
                "data": [
                    {
                        "type": "bar",
                        "x": [f"M{mr['month']}" for mr in monthly_returns[1:]],
                        "y": [mr["incremental"] for mr in monthly_returns[1:]],
                        "marker": {"color": "#4a90d9"},
                    }
                ],
                "layout": {"height": 340, "xaxis": {"title": "Month"}, "yaxis": {"title": "Returns"}},
            }
        )

    elif analysis_id == "competing_risks":
        """
        Competing Risks Analysis — estimates cumulative incidence functions (CIF)
        when multiple failure modes exist. Uses Aalen-Johansen estimator.
        """
        time_col = config.get("time") or config.get("var")
        event_col = config.get("event") or config.get("failure_mode")
        try:
            data = df[[time_col, event_col]].dropna()
            times = data[time_col].values.astype(float)
            events = data[event_col].values
            N = len(data)

            # Event types: 0 = censored, others are failure modes
            unique_events = sorted(
                [e for e in np.unique(events) if e != 0 and str(e) != "0" and str(e).lower() != "censored"], key=str
            )
            n_events = len(unique_events)

            if n_events < 1:
                result["summary"] = (
                    "No failure events found. Ensure event column has non-zero values for failure modes."
                )
                return result

            # Sort by time
            order = np.argsort(times)
            times = times[order]
            events = events[order]

            # Unique event times
            unique_times = np.sort(np.unique(times))

            # Kaplan-Meier overall survival for denominator
            np.zeros(len(unique_times))
            np.ones(len(unique_times) + 1)

            # Compute CIF for each event type
            cifs = {}
            for event_type in unique_events:
                cif_vals = [0.0]
                cif_times = [0.0]
                surv_prev = 1.0

                for ti, t in enumerate(unique_times):
                    at_risk = np.sum(times >= t)
                    if at_risk == 0:
                        continue

                    # Count events of any type and specific type at this time
                    d_j = np.sum((times == t) & (events == event_type))
                    d_all = np.sum(
                        (times == t)
                        & (events != 0)
                        & (np.array([str(e) for e in events]) != "0")
                        & (np.array([str(e).lower() for e in events]) != "censored")
                    )

                    # Cause-specific hazard
                    h_j = d_j / at_risk
                    h_all = d_all / at_risk

                    # CIF increment = S(t-) * h_j(t)
                    cif_increment = surv_prev * h_j

                    cif_vals.append(cif_vals[-1] + cif_increment)
                    cif_times.append(t)

                    # Update overall survival
                    surv_prev = surv_prev * (1 - h_all)

                cifs[str(event_type)] = {"times": cif_times, "values": cif_vals}

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += "<<COLOR:title>>COMPETING RISKS ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Time variable:<</COLOR>> {time_col}\n"
            summary_text += f"<<COLOR:highlight>>Event variable:<</COLOR>> {event_col}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n"
            summary_text += f"<<COLOR:highlight>>Failure modes:<</COLOR>> {n_events}\n\n"

            summary_text += "<<COLOR:text>>Cumulative Incidence at Final Time:<</COLOR>>\n"
            summary_text += f"{'Failure Mode':<20} {'Events':>8} {'CIF (final)':>12}\n"
            summary_text += f"{'─' * 42}\n"
            for event_type in unique_events:
                n_events_type = int(np.sum(events == event_type))
                final_cif = cifs[str(event_type)]["values"][-1]
                summary_text += f"{str(event_type):<20} {n_events_type:>8} {final_cif:>12.4f}\n"

            n_censored = int(
                np.sum(
                    (events == 0)
                    | (np.array([str(e) for e in events]) == "0")
                    | (np.array([str(e).lower() for e in events]) == "censored")
                )
            )
            summary_text += f"\n<<COLOR:text>>Censored observations:<</COLOR>> {n_censored}"

            result["summary"] = summary_text

            # CIF plot
            traces = []
            colors = SVEND_COLORS[:5]
            for ei, event_type in enumerate(unique_events):
                cif = cifs[str(event_type)]
                traces.append(
                    {
                        "x": cif["times"],
                        "y": cif["values"],
                        "mode": "lines",
                        "name": f"CIF: {event_type}",
                        "line": {"color": colors[ei % len(colors)], "width": 2, "shape": "hv"},
                    }
                )

            result["plots"].append(
                {
                    "title": "Cumulative Incidence Functions",
                    "data": traces,
                    "layout": {
                        "height": 340,
                        "xaxis": {"title": time_col},
                        "yaxis": {"title": "Cumulative Incidence", "range": [0, 1.05]},
                    },
                }
            )

            # Stacked area plot
            stacked_traces = []
            for ei, event_type in enumerate(unique_events):
                cif = cifs[str(event_type)]
                stacked_traces.append(
                    {
                        "x": cif["times"],
                        "y": cif["values"],
                        "mode": "lines",
                        "name": str(event_type),
                        "stackgroup": "one",
                        "line": {"color": colors[ei % len(colors)]},
                        "fillcolor": colors[ei % len(colors)] + "40",
                    }
                )
            result["plots"].append(
                {
                    "title": "Stacked Cumulative Incidence",
                    "data": stacked_traces,
                    "layout": {"height": 340, "xaxis": {"title": time_col}, "yaxis": {"title": "Cumulative Incidence"}},
                }
            )

            result["statistics"] = {
                "n": N,
                "n_censored": n_censored,
                "n_failure_modes": n_events,
                "cif_final": {str(et): cifs[str(et)]["values"][-1] for et in unique_events},
                "event_counts": {str(et): int(np.sum(events == et)) for et in unique_events},
            }
            result["guide_observation"] = (
                f"Competing risks: {n_events} failure modes. "
                + ", ".join([f"{et}: CIF={cifs[str(et)]['values'][-1]:.3f}" for et in unique_events])
                + "."
            )

            # Narrative
            _cr_dominant = max(unique_events, key=lambda et: cifs[str(et)]["values"][-1])
            _cr_dom_cif = cifs[str(_cr_dominant)]["values"][-1]
            result["narrative"] = _narrative(
                f"Competing Risks — {n_events} failure modes",
                f"{N} observations with {n_censored} censored. Dominant failure mode: <strong>{_cr_dominant}</strong> (CIF = {_cr_dom_cif:.3f}). "
                + "Cumulative incidence accounts for competing events — each mode's CIF is the probability of failing from that specific cause.",
                next_steps=f"Focus improvement on <strong>{_cr_dominant}</strong> as the primary failure mode. Reducing it will shift the risk profile.",
                chart_guidance="The stacked area chart shows how total failure probability is partitioned among competing modes. Taller bands indicate more dominant failure causes.",
            )

        except Exception as e:
            result["summary"] = f"Competing risks error: {str(e)}"

    return result
