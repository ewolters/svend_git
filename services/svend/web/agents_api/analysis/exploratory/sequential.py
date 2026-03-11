"""DSW Exploratory — sequential and tolerance analyses."""

import logging

import numpy as np
from scipy import stats

from ..common import (
    _narrative,
)

logger = logging.getLogger(__name__)


def run_run_chart(df, config):
    """Run chart — time-ordered individual values with median line and runs tests."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")
    time_col = config.get("time_col")

    vals = df[var].dropna().values
    n = len(vals)

    if n < 5:
        result["summary"] = "Run chart requires at least 5 observations."
    else:
        median_val = float(np.median(vals))

        # X-axis: time column or row index
        if time_col and time_col != "" and time_col != "None":
            x_vals = df[time_col].loc[df[var].dropna().index].tolist()
        else:
            x_vals = list(range(1, n + 1))

        # Count runs above/below median
        above = vals > median_val
        # Exclude values exactly at the median for runs test
        not_on = vals != median_val
        filtered = above[not_on]
        n_above = int(np.sum(filtered))
        n_below = int(len(filtered) - n_above)

        if n_above > 0 and n_below > 0:
            # Count runs
            runs = 1
            for i in range(1, len(filtered)):
                if filtered[i] != filtered[i - 1]:
                    runs += 1

            # Expected runs and standard deviation
            n1, n2 = n_above, n_below
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))

            z_score = (runs - expected_runs) / std_runs if std_runs > 0 else 0
            p_clustering = stats.norm.cdf(z_score)  # Too few runs = clustering
            p_mixtures = 1 - stats.norm.cdf(z_score)  # Too many runs = mixtures

            # Longest run (trend indicator)
            longest_run = 1
            current_run = 1
            for i in range(1, len(filtered)):
                if filtered[i] == filtered[i - 1]:
                    current_run += 1
                    longest_run = max(longest_run, current_run)
                else:
                    current_run = 1

            cluster_flag = "<<COLOR:danger>>Yes<</COLOR>>" if p_clustering < 0.05 else "<<COLOR:success>>No<</COLOR>>"
            mixture_flag = "<<COLOR:danger>>Yes<</COLOR>>" if p_mixtures < 0.05 else "<<COLOR:success>>No<</COLOR>>"

            summary = f"""<<COLOR:title>>RUN CHART<</COLOR>>
{"=" * 50}
<<COLOR:highlight>>Variable:<</COLOR>> {var}
<<COLOR:highlight>>Observations:<</COLOR>> {n}
<<COLOR:highlight>>Median:<</COLOR>> {median_val:.6g}

<<COLOR:accent>>Runs Test Results<</COLOR>>
  Number of runs:    {runs}
  Expected runs:     {expected_runs:.1f}
  Longest run:       {longest_run}
  Points above median: {n_above}
  Points below median: {n_below}

  Clustering (too few runs)?   {cluster_flag}  (p = {p_clustering:.4f})
  Mixtures (too many runs)?    {mixture_flag}  (p = {p_mixtures:.4f})"""
        else:
            summary = f"""<<COLOR:title>>RUN CHART<</COLOR>>
{"=" * 50}
Variable: {var}  |  N = {n}  |  Median = {median_val:.6g}
<<COLOR:warning>>All values on same side of median — runs test not applicable<</COLOR>>"""
            runs = 0

        result["summary"] = summary
        result["guide_observation"] = f"Run chart: {n} obs, median={median_val:.4g}, {runs} runs."
        result["narrative"] = _narrative(
            f"Run Chart: {n} observations, {runs} runs",
            f"Median = {median_val:.4g}. A run chart monitors process behavior over time without requiring control limits.",
            next_steps="Look for trends (6+ consecutive increasing/decreasing), runs (8+ points on one side of median), or cycles. These indicate non-random behavior.",
        )

        # Plot
        traces = [
            {
                "type": "scatter",
                "x": x_vals,
                "y": vals.tolist(),
                "mode": "lines+markers",
                "marker": {"color": "#4a9f6e", "size": 5},
                "line": {"color": "#4a9f6e", "width": 1.5},
                "name": var,
            },
            {
                "type": "scatter",
                "x": [x_vals[0], x_vals[-1]],
                "y": [median_val, median_val],
                "mode": "lines",
                "line": {"color": "#e89547", "dash": "dash", "width": 2},
                "name": f"Median = {median_val:.4g}",
            },
        ]
        result["plots"].append(
            {
                "title": f"Run Chart: {var}",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": time_col if time_col else "Observation"},
                    "yaxis": {"title": var},
                    "showlegend": True,
                },
            }
        )

    return result


def run_sprt(df, config):
    """Sequential Probability Ratio Test (Wald)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    col = config.get("var") or config.get("variable") or config.get("column")
    # Frontend sends mu0/mu1; backend also accepts target/delta
    mu0 = config.get("mu0")
    mu1 = config.get("mu1")
    if mu0 is not None and mu1 is not None:
        target = float(mu0)
        delta = float(mu1) - float(mu0)
    else:
        target = float(config.get("target", 0))
        delta = float(config.get("delta", 1.0))
    sigma = config.get("sigma")
    alpha = float(config.get("alpha", 0.05))
    beta = float(config.get("beta", 0.10))

    if not col or col not in df.columns:
        result["summary"] = "Error: Specify a numeric column."
        return result

    data = df[col].dropna().values.astype(float)
    n = len(data)

    if sigma is not None:
        sigma = float(sigma)
    else:
        sigma = float(np.std(data, ddof=1))

    # SPRT for H0: mu = target vs H1: mu = target + delta
    # Log-likelihood ratio for each observation
    mu0 = target
    mu1 = target + delta

    # Wald boundaries
    A = np.log((1 - beta) / alpha)  # upper boundary (reject H0)
    B = np.log(beta / (1 - alpha))  # lower boundary (accept H0)

    # Cumulative log-likelihood ratio
    ll_ratio = np.cumsum((data - mu0) * delta / sigma**2 - delta**2 / (2 * sigma**2))

    # Find decision point
    decision_idx = None
    decision = "Continue sampling"
    for i in range(n):
        if ll_ratio[i] >= A:
            decision_idx = i
            decision = "Reject H0 (effect detected)"
            break
        elif ll_ratio[i] <= B:
            decision_idx = i
            decision = "Accept H0 (no effect)"
            break

    samples_used = decision_idx + 1 if decision_idx is not None else n

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>SEQUENTIAL PROBABILITY RATIO TEST (SPRT)<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Variable:<</COLOR>> {col}    N: {n}\n"
    summary += f"<<COLOR:text>>H\u2080:<</COLOR>> \u03bc = {mu0:.4f}\n"
    summary += f"<<COLOR:text>>H\u2081:<</COLOR>> \u03bc = {mu1:.4f} (shift = {delta:.4f})\n"
    summary += f"<<COLOR:text>>\u03c3:<</COLOR>> {sigma:.4f}\n"
    summary += f"<<COLOR:text>>\u03b1:<</COLOR>> {alpha}    \u03b2: {beta}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Decision Boundaries \u2500\u2500<</COLOR>>\n"
    summary += f"  Upper (reject H\u2080): {A:.3f}\n"
    summary += f"  Lower (accept H\u2080): {B:.3f}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Result \u2500\u2500<</COLOR>>\n"
    summary += f"  {decision}\n"
    summary += f"  Samples used: {samples_used} (of {n} available)\n"
    if decision_idx is not None and decision_idx < n - 1:
        saved = n - samples_used
        summary += f"  Samples saved vs fixed-n: {saved} ({saved / n * 100:.0f}%)\n"

    result["summary"] = summary
    result["statistics"] = {
        "decision": decision,
        "samples_used": samples_used,
        "upper_boundary": float(A),
        "lower_boundary": float(B),
        "final_llr": float(ll_ratio[min(decision_idx or n - 1, n - 1)]),
    }
    result["guide_observation"] = f"SPRT: {decision} after {samples_used} samples (of {n})."

    _sprt_savings = (
        f" Saved {n - samples_used} inspections ({(n - samples_used) / n * 100:.0f}%) vs fixed-sample testing."
        if decision_idx is not None and decision_idx < n - 1
        else ""
    )
    result["narrative"] = _narrative(
        f"SPRT \u2014 {decision} (n = {samples_used})",
        f"Testing H\u2080: \u03bc = {mu0:.4f} vs H\u2081: \u03bc = {mu1:.4f} (shift = {delta:.4f}). "
        f"The cumulative evidence {f'crossed the upper boundary at observation {samples_used}' if 'Reject' in decision else f'crossed the lower boundary at observation {samples_used}' if 'Accept' in decision else 'did not reach either boundary'}. "
        f"{decision}.{_sprt_savings}",
        next_steps="SPRT is the most sample-efficient hypothesis test \u2014 it reaches decisions with fewer samples than fixed-n tests. "
        "For incoming inspection, this means fewer units tested per lot.",
        chart_guidance="The path shows cumulative evidence. Crossing the upper red line = reject H\u2080 (shift detected). "
        "Crossing the lower green line = accept H\u2080 (no shift). Staying between = undecided.",
    )

    # Plot: SPRT path with boundaries
    plot_n = min(n, samples_used + 20) if decision_idx is not None else n
    result["plots"].append(
        {
            "title": "SPRT Evidence Path",
            "data": [
                {
                    "type": "scatter",
                    "x": list(range(1, plot_n + 1)),
                    "y": ll_ratio[:plot_n].tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a90d9", "width": 2},
                    "name": "Log-LR",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Sample Number"},
                "yaxis": {"title": "Cumulative Log-Likelihood Ratio"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 1,
                        "x1": plot_n,
                        "y0": float(A),
                        "y1": float(A),
                        "line": {"color": "#dc5050", "dash": "dash", "width": 2},
                    },
                    {
                        "type": "line",
                        "x0": 1,
                        "x1": plot_n,
                        "y0": float(B),
                        "y1": float(B),
                        "line": {"color": "#4a9f6e", "dash": "dash", "width": 2},
                    },
                    {
                        "type": "line",
                        "x0": 1,
                        "x1": plot_n,
                        "y0": 0,
                        "y1": 0,
                        "line": {"color": "#888", "width": 1, "dash": "dot"},
                    },
                ],
                "annotations": [
                    {
                        "x": plot_n,
                        "y": float(A),
                        "text": "Reject H\u2080",
                        "showarrow": False,
                        "xanchor": "right",
                        "font": {"color": "#dc5050", "size": 10},
                    },
                    {
                        "x": plot_n,
                        "y": float(B),
                        "text": "Accept H\u2080",
                        "showarrow": False,
                        "xanchor": "right",
                        "font": {"color": "#4a9f6e", "size": 10},
                    },
                ],
            },
        }
    )

    return result


def run_tolerance_interval(df, config):
    """Tolerance intervals — contain a proportion of the population."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")
    proportion = float(config.get("proportion", 0.95))  # Proportion of population
    confidence = float(config.get("confidence", 0.95))  # Confidence level
    _method = config.get("method", "normal")  # normal or nonparametric

    data = df[var].dropna().values
    n = len(data)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>TOLERANCE INTERVALS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n}\n"
    summary += f"<<COLOR:highlight>>Coverage:<</COLOR>> {proportion * 100:.0f}% of population\n"
    summary += f"<<COLOR:highlight>>Confidence:<</COLOR>> {confidence * 100:.0f}%\n\n"

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    summary += "<<COLOR:accent>>── Sample Statistics ──<</COLOR>>\n"
    summary += f"  Mean: {mean:.4f}\n"
    summary += f"  Std Dev: {std:.4f}\n\n"

    # Normal-based tolerance interval
    # k factor from tolerance interval tables (approximation)
    z_p = stats.norm.ppf((1 + proportion) / 2)
    chi2_val = stats.chi2.ppf(1 - confidence, n - 1)

    # Two-sided tolerance factor
    k_normal = z_p * np.sqrt((n - 1) * (1 + 1 / n) / chi2_val)

    tol_lower_normal = mean - k_normal * std
    tol_upper_normal = mean + k_normal * std

    summary += "<<COLOR:accent>>Normal-Based Tolerance Interval:<</COLOR>>\n"
    summary += f"  k factor: {k_normal:.4f}\n"
    summary += f"  Interval: ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})\n\n"

    # Non-parametric tolerance interval
    # Uses order statistics
    # For 95/95, need approximately n >= 59 for two-sided
    # Coverage probability for (X(r), X(n-r+1)) where r is chosen appropriately

    # Simple approach: use percentiles
    _alpha = 1 - confidence
    _beta = 1 - proportion

    # Find r such that P(at least proportion*100% between X(r) and X(n-r+1)) >= confidence
    # Using binomial distribution
    from scipy.special import comb

    r_found = None
    for r in range(1, n // 2 + 1):
        # Probability that at least proportion of population is between order statistics
        prob = 0
        for j in range(r, n - r + 2):
            prob += comb(n, j, exact=True) * (proportion ** (j)) * ((1 - proportion) ** (n - j))
        if prob >= confidence:
            r_found = r
            break

    if r_found:
        sorted_data = np.sort(data)
        tol_lower_np = sorted_data[r_found - 1]
        tol_upper_np = sorted_data[n - r_found]
        summary += "<<COLOR:accent>>Non-Parametric Tolerance Interval:<</COLOR>>\n"
        summary += f"  Uses order statistics X({r_found}) and X({n - r_found + 1})\n"
        summary += f"  Interval: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"
    else:
        tol_lower_np = np.min(data)
        tol_upper_np = np.max(data)
        summary += "<<COLOR:warning>>Non-Parametric: Sample too small for exact interval.<</COLOR>>\n"
        summary += f"  Using min/max: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"

    # Comparison with confidence interval
    se = std / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se

    summary += f"<<COLOR:dim>>For comparison - {confidence * 100:.0f}% CI for mean:<</COLOR>>\n"
    summary += f"  ({ci_lower:.4f}, {ci_upper:.4f})\n\n"

    summary += "<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
    summary += f"  We are {confidence * 100:.0f}% confident that at least {proportion * 100:.0f}%\n"
    summary += "  of the population falls within the tolerance interval.\n"
    summary += "\n<<COLOR:dim>>Note: Tolerance intervals are WIDER than confidence intervals<</COLOR>>\n"
    summary += "<<COLOR:dim>>because they cover the population, not just the mean.<</COLOR>>\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"Tolerance interval ({proportion * 100:.0f}%/{confidence * 100:.0f}%): ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})"
    )
    result["narrative"] = _narrative(
        f"Tolerance Interval: ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})",
        f"We are {confidence * 100:.0f}% confident that at least {proportion * 100:.0f}% of the population falls within this interval.",
        next_steps="Tolerance intervals are wider than confidence intervals because they cover the population, not just the mean. Compare with specification limits.",
    )
    result["statistics"] = {
        "tol_lower_normal": float(tol_lower_normal),
        "tol_upper_normal": float(tol_upper_normal),
        "tol_lower_np": float(tol_lower_np),
        "tol_upper_np": float(tol_upper_np),
        "k_factor": float(k_normal),
        "mean": float(mean),
        "std": float(std),
    }

    # Plot showing intervals
    result["plots"].append(
        {
            "title": "Tolerance vs Confidence Intervals",
            "data": [
                {
                    "type": "histogram",
                    "x": data.tolist(),
                    "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": var},
                "shapes": [
                    # Tolerance interval (normal)
                    {
                        "type": "rect",
                        "x0": tol_lower_normal,
                        "x1": tol_upper_normal,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "fillcolor": "rgba(232, 149, 71, 0.2)",
                        "line": {"color": "#e89547", "width": 2},
                    },
                    # Confidence interval
                    {
                        "type": "rect",
                        "x0": ci_lower,
                        "x1": ci_upper,
                        "y0": 0.4,
                        "y1": 0.6,
                        "yref": "paper",
                        "fillcolor": "rgba(71, 165, 232, 0.4)",
                        "line": {"color": "#47a5e8", "width": 2},
                    },
                    # Mean line
                    {
                        "type": "line",
                        "x0": mean,
                        "x1": mean,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#4a9f6e", "width": 2, "dash": "dash"},
                    },
                ],
                "annotations": [
                    {
                        "x": (tol_lower_normal + tol_upper_normal) / 2,
                        "y": 0.95,
                        "yref": "paper",
                        "text": "Tolerance",
                        "showarrow": False,
                        "font": {"color": "#e89547"},
                    },
                    {
                        "x": (ci_lower + ci_upper) / 2,
                        "y": 0.5,
                        "yref": "paper",
                        "text": "CI",
                        "showarrow": False,
                        "font": {"color": "#47a5e8"},
                    },
                ],
            },
        }
    )

    return result
