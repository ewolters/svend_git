"""DSW Exploratory — univariate analyses."""

import logging

import numpy as np
from scipy import stats
from scipy import stats as sp_stats

from ..common import (
    _bayesian_shadow,
    _check_normality,
    _evidence_grade,
    _narrative,
)

logger = logging.getLogger(__name__)


def run_descriptive(df, config):
    """Descriptive statistics for numeric variables."""
    import pandas as pd  # noqa: F401

    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Get selected vars from config, fall back to all numeric
    vars_from_config = config.get("vars", [])
    if isinstance(vars_from_config, list) and len(vars_from_config) > 0:
        vars_to_analyze = [v for v in vars_from_config if v in df.columns]
    else:
        vars_to_analyze = numeric_cols

    if not vars_to_analyze:
        result["summary"] = "No numeric variables found to analyze."
        return result

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>DESCRIPTIVE STATISTICS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {len(vars_to_analyze)}    "
    summary += f"<<COLOR:highlight>>Total rows:<</COLOR>> {len(df)}\n\n"

    # Add explicit statistics for Synara integration
    result["statistics"] = {}
    obs_parts = []

    for var in vars_to_analyze:
        col = df[var].dropna()
        n = len(col)
        mean = col.mean()
        std = col.std()
        median = col.median()
        skew = col.skew()
        kurt = col.kurtosis()
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        missing = len(df[var]) - n
        cv = (std / abs(mean) * 100) if mean != 0 else float("inf")

        summary += f"<<COLOR:accent>>── {var} ──<</COLOR>>\n"
        summary += f"  N: {n}"
        if missing > 0:
            summary += f"  (<<COLOR:warning>>{missing} missing<</COLOR>>)"
        summary += f"\n  Mean: {mean:.4f}    Std Dev: {std:.4f}    CV: {cv:.1f}%\n"
        summary += f"  Median: {median:.4f}    IQR: {iqr:.4f}    [{q1:.4f}, {q3:.4f}]\n"
        summary += f"  Min: {col.min():.4f}    Max: {col.max():.4f}    Range: {col.max() - col.min():.4f}\n"
        summary += f"  Skewness: {skew:.3f}    Kurtosis: {kurt:.3f}\n"

        # Distribution shape interpretation
        if abs(skew) < 0.5:
            shape = "approximately symmetric"
        elif skew > 0:
            shape = f"right-skewed (skew={skew:.2f})"
        else:
            shape = f"left-skewed (skew={skew:.2f})"

        if abs(kurt) > 2:
            shape += ", heavy-tailed" if kurt > 0 else ", light-tailed"

        summary += f"  <<COLOR:dim>>Shape: {shape}<</COLOR>>\n\n"

        result["statistics"][f"mean({var})"] = float(mean)
        result["statistics"][f"std({var})"] = float(std)
        result["statistics"][f"min({var})"] = float(col.min())
        result["statistics"][f"max({var})"] = float(col.max())
        result["statistics"][f"median({var})"] = float(median)
        result["statistics"][f"n({var})"] = int(n)

        obs_parts.append(f"{var}: μ={mean:.3f}, σ={std:.3f}, n={n}")

    result["summary"] = summary
    result["guide_observation"] = (
        f"Descriptive statistics for {len(vars_to_analyze)} variable(s). "
        + "; ".join(obs_parts[:5])
    )

    # Add histogram for each variable
    for var in vars_to_analyze:
        try:
            data = df[var].dropna().tolist()
            if len(data) > 0:
                result["plots"].append(
                    {
                        "title": f"Distribution of {var}",
                        "data": [
                            {
                                "type": "histogram",
                                "x": data,
                                "name": var,
                                "marker": {
                                    "color": "rgba(74, 159, 110, 0.4)",
                                    "line": {"color": "#4a9f6e", "width": 1.5},
                                },
                            }
                        ],
                        "layout": {"height": 200},
                    }
                )
        except Exception as plot_err:
            logger.warning(f"Could not create histogram for {var}: {plot_err}")

    return result


def run_prop_1sample(df, config):
    """
    One-Proportion Z-Test — test if an observed proportion equals a hypothesized value.
    Uses normal approximation to the binomial; reports Z, p-value, and Wilson CI.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    event = config.get("event")  # value to count as success
    p0 = float(config.get("p0", 0.5))  # hypothesized proportion
    alt = config.get("alternative", "two-sided")  # two-sided, greater, less
    alpha = 1 - float(config.get("conf", 95)) / 100

    col = df[var].dropna()
    n = len(col)
    if event is not None and str(event) != "":
        x = int((col.astype(str) == str(event)).sum())
    else:
        # If binary 0/1, count 1s
        x = (
            int((col == 1).sum())
            if col.dtype in ["int64", "float64"]
            else int(col.value_counts().iloc[0])
        )
    p_hat = x / n if n > 0 else 0

    # Z-test
    se0 = np.sqrt(p0 * (1 - p0) / n) if n > 0 else 1
    z_stat = (p_hat - p0) / se0 if se0 > 0 else 0

    if alt == "greater":
        p_val = float(1 - stats.norm.cdf(z_stat))
    elif alt == "less":
        p_val = float(stats.norm.cdf(z_stat))
    else:
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Wilson confidence interval
    z_crit = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z_crit**2 / n
    center = (p_hat + z_crit**2 / (2 * n)) / denom
    margin = z_crit * np.sqrt((p_hat * (1 - p_hat) + z_crit**2 / (4 * n)) / n) / denom
    ci_lo, ci_hi = max(0, center - margin), min(1, center + margin)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>ONE-PROPORTION Z-TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    if event is not None and str(event) != "":
        summary += f"<<COLOR:highlight>>Event:<</COLOR>> {event}\n"
    summary += f"<<COLOR:highlight>>H₀:<</COLOR>> p = {p0}\n"
    summary += f"<<COLOR:highlight>>H₁:<</COLOR>> p {'≠' if alt == 'two-sided' else '>' if alt == 'greater' else '<'} {p0}\n\n"
    summary += "<<COLOR:accent>>── Sample Results ──<</COLOR>>\n"
    summary += f"  N: {n}\n"
    summary += f"  Successes: {x}\n"
    summary += f"  p̂: {p_hat:.4f}\n\n"
    summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
    summary += f"  Z-statistic: {z_stat:.4f}\n"
    summary += f"  p-value: {p_val:.4f}\n"
    summary += f"  {100 * (1 - alpha):.0f}% CI (Wilson): ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

    if p_val < alpha:
        summary += f"<<COLOR:good>>Proportion differs significantly from {p0} (p < {alpha})<</COLOR>>"
    else:
        summary += (
            f"<<COLOR:text>>No significant difference from {p0} (p ≥ {alpha})<</COLOR>>"
        )

    result["summary"] = summary

    # Proportion bar with CI and reference line
    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": ["Observed"],
                    "y": [p_hat],
                    "marker": {"color": "#4a9f6e"},
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_hi - p_hat],
                        "arrayminus": [p_hat - ci_lo],
                        "color": "#5a6a5a",
                    },
                    "name": f"p̂ = {p_hat:.4f}",
                }
            ],
            "layout": {
                "title": "Observed Proportion vs Hypothesized",
                "yaxis": {
                    "title": "Proportion",
                    "range": [0, min(1.05, max(ci_hi + 0.1, p0 + 0.2))],
                },
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": 0.5,
                        "y0": p0,
                        "y1": p0,
                        "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    }
                ],
                "annotations": [
                    {
                        "x": 0.5,
                        "y": p0,
                        "text": f"H₀: p={p0}",
                        "showarrow": False,
                        "xanchor": "left",
                        "font": {"color": "#e89547"},
                    }
                ],
            },
        }
    )

    result["guide_observation"] = (
        f"1-prop Z-test: p̂={p_hat:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. "
        + ("Significant." if p_val < alpha else "Not significant.")
    )
    result["statistics"] = {
        "n": n,
        "successes": x,
        "p_hat": p_hat,
        "p0": p0,
        "z_statistic": float(z_stat),
        "p_value": p_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "alternative": alt,
    }

    # Narrative
    if p_val < alpha:
        verdict = (
            f"Proportion differs from {p0} (p\u0302 = {p_hat:.4f}, p = {p_val:.4f})"
        )
        body = (
            f"The observed proportion {p_hat:.4f} ({x}/{n}) is significantly different from the hypothesized value of {p0}. "
            f"Wilson {100 * (1 - alpha):.0f}% CI: ({ci_lo:.4f}, {ci_hi:.4f})."
        )
        nxt = "Investigate why the proportion deviates from the target. If it's a defect rate, identify root causes."
    else:
        verdict = (
            f"Proportion consistent with {p0} (p\u0302 = {p_hat:.4f}, p = {p_val:.4f})"
        )
        body = (
            f"The observed proportion {p_hat:.4f} ({x}/{n}) is not significantly different from {p0}. "
            f"Wilson {100 * (1 - alpha):.0f}% CI: ({ci_lo:.4f}, {ci_hi:.4f}) includes the hypothesized value."
        )
        nxt = "No evidence of departure from the target. Continue monitoring."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="The bar shows the observed proportion with CI error bars. The dashed line is the hypothesized value.",
    )

    # --- Bayesian Insurance ---
    try:
        _shadow = _bayesian_shadow("proportion", x=x, n=n, p0=p0)
        if _shadow:
            result["bayesian_shadow"] = _shadow
        _grade = _evidence_grade(p_val, bf10=_shadow.get("bf10") if _shadow else None)
        if _grade:
            result["evidence_grade"] = _grade
    except Exception:
        pass

    return result


def run_poisson_1sample(df, config):
    """
    One-Sample Poisson Rate Test — test if an observed event rate equals a hypothesized rate.
    Uses exact Poisson test (conditional) or normal approximation for large counts.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    rate0 = float(config.get("rate0", 1.0))  # hypothesized rate
    exposure = float(config.get("exposure", 1.0))  # time/area/units of exposure
    alt = config.get("alternative", "two-sided")
    alpha = 1 - float(config.get("conf", 95)) / 100

    col = df[var].dropna()
    total_count = float(col.sum())
    n = len(col)
    observed_rate = total_count / exposure if exposure > 0 else 0
    expected_count = rate0 * exposure

    # Exact Poisson test
    if alt == "greater":
        p_val = float(1 - stats.poisson.cdf(int(total_count) - 1, expected_count))
    elif alt == "less":
        p_val = float(stats.poisson.cdf(int(total_count), expected_count))
    else:
        # Two-sided: 2 * min(left, right)
        p_left = stats.poisson.cdf(int(total_count), expected_count)
        p_right = 1 - stats.poisson.cdf(int(total_count) - 1, expected_count)
        p_val = float(min(1.0, 2 * min(p_left, p_right)))

    # Exact Poisson CI for rate
    z_crit = stats.norm.ppf(1 - alpha / 2)  # noqa: F841
    if total_count > 0:
        ci_lo = stats.chi2.ppf(alpha / 2, 2 * total_count) / (2 * exposure)
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (total_count + 1)) / (2 * exposure)
    else:
        ci_lo = 0
        ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * exposure)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>ONE-SAMPLE POISSON RATE TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>H₀:<</COLOR>> rate = {rate0}\n"
    summary += f"<<COLOR:highlight>>Exposure:<</COLOR>> {exposure}\n\n"
    summary += "<<COLOR:accent>>── Sample Results ──<</COLOR>>\n"
    summary += f"  Total count: {total_count:.0f}\n"
    summary += f"  Observed rate: {observed_rate:.4f}\n"
    summary += f"  Expected count (under H₀): {expected_count:.1f}\n\n"
    summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
    summary += f"  p-value (exact): {p_val:.4f}\n"
    summary += f"  {100 * (1 - alpha):.0f}% CI for rate: ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

    if p_val < alpha:
        summary += f"<<COLOR:good>>Rate differs significantly from {rate0} (p < {alpha})<</COLOR>>"
    else:
        summary += f"<<COLOR:text>>No significant difference from {rate0} (p ≥ {alpha})<</COLOR>>"

    result["summary"] = summary

    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": ["Observed"],
                    "y": [observed_rate],
                    "marker": {"color": "#4a9f6e"},
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_hi - observed_rate],
                        "arrayminus": [observed_rate - ci_lo],
                        "color": "#5a6a5a",
                    },
                    "name": f"Rate = {observed_rate:.4f}",
                }
            ],
            "layout": {
                "title": "Observed Rate vs Hypothesized",
                "yaxis": {"title": "Rate"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": 0.5,
                        "y0": rate0,
                        "y1": rate0,
                        "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    }
                ],
                "annotations": [
                    {
                        "x": 0.5,
                        "y": rate0,
                        "text": f"H₀: λ={rate0}",
                        "showarrow": False,
                        "xanchor": "left",
                        "font": {"color": "#e89547"},
                    }
                ],
            },
        }
    )

    # Distribution plot
    x_range = list(range(max(0, int(total_count) - 15), int(total_count) + 16))
    pmf_vals = [float(stats.poisson.pmf(k, expected_count)) for k in x_range]
    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": x_range,
                    "y": pmf_vals,
                    "name": f"Poisson(λ={expected_count:.1f})",
                    "marker": {
                        "color": [
                            "#d94a4a" if k == int(total_count) else "#4a9f6e"
                            for k in x_range
                        ],
                        "opacity": 0.7,
                    },
                }
            ],
            "layout": {
                "title": f"Poisson Distribution under H₀ (observed = {int(total_count)})",
                "xaxis": {"title": "Count"},
                "yaxis": {"title": "Probability"},
            },
        }
    )

    result["guide_observation"] = (
        f"Poisson rate test: observed rate={observed_rate:.4f}, H₀ rate={rate0}, p={p_val:.4f}. "
        + ("Significant." if p_val < alpha else "Not significant.")
    )
    if p_val < alpha:
        verdict = f"Rate differs from {rate0} (observed = {observed_rate:.4f}, p = {p_val:.4f})"
        body = f"Observed count = {total_count:.0f} over {n} observations. Rate {observed_rate:.4f} is significantly different from hypothesized rate {rate0}."
    else:
        verdict = f"Rate consistent with {rate0} (p = {p_val:.4f})"
        body = f"Observed rate {observed_rate:.4f} is not significantly different from {rate0}."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps="If rate is higher than target, investigate common causes. Poisson tests assume events occur independently at a constant rate.",
    )
    result["statistics"] = {
        "total_count": total_count,
        "exposure": exposure,
        "observed_rate": observed_rate,
        "hypothesized_rate": rate0,
        "p_value": p_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "alternative": alt,
    }

    return result


def run_box_cox(df, config):
    """
    Box-Cox Transformation - find optimal power transformation.
    Transforms data to approximate normality.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")

    data = df[var].dropna().values

    # Box-Cox requires positive data
    if np.any(data <= 0):
        # Shift data to be positive
        shift = -np.min(data) + 1
        data_shifted = data + shift
        shifted = True
    else:
        data_shifted = data
        shift = 0
        shifted = False

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BOX-COX TRANSFORMATION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
    if shifted:
        summary += f"<<COLOR:warning>>Data shifted by {shift:.4f} (original had non-positive values)<</COLOR>>\n"
    summary += "\n"

    # Find optimal lambda
    transformed, optimal_lambda = stats.boxcox(data_shifted)

    # Test common transformations
    lambdas = [-2, -1, -0.5, 0, 0.5, 1, 2]
    lambda_names = ["1/x²", "1/x", "1/√x", "ln(x)", "√x", "x (none)", "x²"]

    summary += (
        "<<COLOR:accent>>── Common Transformations (Log-Likelihood) ──<</COLOR>>\n"
    )
    for lam, name in zip(lambdas, lambda_names):
        if lam == 0:
            trans = np.log(data_shifted)
        else:
            trans = (data_shifted**lam - 1) / lam
        # Calculate log-likelihood
        ll = -len(data) / 2 * np.log(np.var(trans)) + (lam - 1) * np.sum(
            np.log(data_shifted)
        )
        summary += f"  λ = {lam:>5} ({name:<8}): LL = {ll:.2f}\n"

    summary += "\n<<COLOR:success>>OPTIMAL TRANSFORMATION:<</COLOR>>\n"
    summary += f"  λ = {optimal_lambda:.4f}\n"

    # Interpret lambda
    if abs(optimal_lambda) < 0.1:
        suggestion = "ln(x) - logarithmic"
    elif abs(optimal_lambda - 0.5) < 0.1:
        suggestion = "√x - square root"
    elif abs(optimal_lambda - 1) < 0.1:
        suggestion = "x - no transformation needed"
    elif abs(optimal_lambda + 1) < 0.1:
        suggestion = "1/x - reciprocal"
    elif optimal_lambda < 0:
        suggestion = f"x^{optimal_lambda:.2f} - inverse power"
    else:
        suggestion = f"x^{optimal_lambda:.2f} - power transformation"

    summary += f"  Suggested: {suggestion}\n\n"

    # Normality tests before and after
    _, p_before = stats.shapiro(data[: min(5000, len(data))])
    _, p_after = stats.shapiro(transformed[: min(5000, len(transformed))])

    summary += "<<COLOR:accent>>── Normality Tests (Shapiro-Wilk) ──<</COLOR>>\n"
    summary += f"  Original: p = {p_before:.4f} {'(normal)' if p_before > 0.05 else '(non-normal)'}\n"
    summary += f"  Transformed: p = {p_after:.4f} {'(normal)' if p_after > 0.05 else '(non-normal)'}\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"Box-Cox optimal λ = {optimal_lambda:.3f}. {suggestion}."
    )
    result["narrative"] = _narrative(
        f"Box-Cox: optimal \u03bb = {optimal_lambda:.3f}",
        f"{suggestion}. The Box-Cox transformation finds the power that best normalizes the data.",
        next_steps="Apply the transformation before running parametric tests (t-test, ANOVA, regression) if data is non-normal.",
        chart_guidance="The log-likelihood curve shows which \u03bb best normalizes the data. The 95% CI indicates the range of acceptable values.",
    )
    result["statistics"] = {
        "optimal_lambda": float(optimal_lambda),
        "p_before": float(p_before),
        "p_after": float(p_after),
        "shift_applied": float(shift),
    }

    # Plot: original vs transformed distributions
    result["plots"].append(
        {
            "title": "Original Distribution",
            "data": [
                {
                    "type": "histogram",
                    "x": data.tolist(),
                    "marker": {
                        "color": "rgba(232, 87, 71, 0.4)",
                        "line": {"color": "#e85747", "width": 1},
                    },
                }
            ],
            "layout": {"height": 200, "xaxis": {"title": var}},
        }
    )

    result["plots"].append(
        {
            "title": f"Transformed (λ = {optimal_lambda:.2f})",
            "data": [
                {
                    "type": "histogram",
                    "x": transformed.tolist(),
                    "marker": {
                        "color": "rgba(74, 159, 110, 0.4)",
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                }
            ],
            "layout": {"height": 200, "xaxis": {"title": f"Box-Cox({var})"}},
        }
    )

    # Lambda vs log-likelihood profile
    lambda_range = np.linspace(
        max(-3, optimal_lambda - 2), min(3, optimal_lambda + 2), 50
    )
    log_likelihoods = []
    for lam in lambda_range:
        if abs(lam) < 1e-10:
            trans = np.log(data_shifted)
        else:
            trans = (data_shifted**lam - 1) / lam
        ll = -len(data) / 2 * np.log(np.var(trans)) + (lam - 1) * np.sum(
            np.log(data_shifted)
        )
        log_likelihoods.append(float(ll))
    result["plots"].append(
        {
            "title": "Lambda vs Log-Likelihood",
            "data": [
                {
                    "type": "scatter",
                    "x": lambda_range.tolist(),
                    "y": log_likelihoods,
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "Log-Likelihood",
                },
                {
                    "type": "scatter",
                    "x": [float(optimal_lambda)],
                    "y": [max(log_likelihoods)],
                    "mode": "markers",
                    "marker": {"color": "#d94a4a", "size": 10, "symbol": "diamond"},
                    "name": f"Optimal λ = {optimal_lambda:.3f}",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Lambda (λ)"},
                "yaxis": {"title": "Log-Likelihood"},
            },
        }
    )

    return result


def run_johnson_transform(df, config):
    """
    Johnson Transformation — finds optimal Johnson family (SB, SL, SU) to normalize data.
    More general than Box-Cox (handles bounded and unbounded distributions).
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")

    data = df[var].dropna().values
    n = len(data)

    if n < 10:
        result["summary"] = "Johnson transformation requires at least 10 observations."
    else:
        summary = f"<<COLOR:title>>JOHNSON TRANSFORMATION<</COLOR>>\n{'=' * 50}\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"

        # Test each Johnson family
        families = {}

        # SU (unbounded)
        try:
            params_su = stats.johnsonsu.fit(data)
            transformed_su = stats.johnsonsu.cdf(data, *params_su)
            transformed_su = stats.norm.ppf(np.clip(transformed_su, 0.001, 0.999))
            _, p_su = stats.shapiro(transformed_su[: min(5000, len(transformed_su))])
            families["SU"] = {
                "params": params_su,
                "p_value": float(p_su),
                "transformed": transformed_su,
            }
        except Exception:
            pass

        # SB (bounded)
        try:
            params_sb = stats.johnsonsb.fit(data)
            transformed_sb = stats.johnsonsb.cdf(data, *params_sb)
            transformed_sb = stats.norm.ppf(np.clip(transformed_sb, 0.001, 0.999))
            _, p_sb = stats.shapiro(transformed_sb[: min(5000, len(transformed_sb))])
            families["SB"] = {
                "params": params_sb,
                "p_value": float(p_sb),
                "transformed": transformed_sb,
            }
        except Exception:
            pass

        # SL (lognormal — just use log transform)
        try:
            if np.all(data > 0):
                transformed_sl = np.log(data)
                _, p_sl = stats.shapiro(
                    transformed_sl[: min(5000, len(transformed_sl))]
                )
                families["SL"] = {
                    "params": None,
                    "p_value": float(p_sl),
                    "transformed": transformed_sl,
                }
        except Exception:
            pass

        # Original normality
        _, p_orig = stats.shapiro(data[: min(5000, n)])

        summary += f"<<COLOR:accent>>── Original Shapiro-Wilk p-value ──<</COLOR>> {p_orig:.4f}"
        summary += f" {'(normal)' if p_orig > 0.05 else '(non-normal)'}\n\n"

        if families:
            best_family = max(families.keys(), key=lambda k: families[k]["p_value"])
            best = families[best_family]

            summary += "<<COLOR:accent>>Family Results:<</COLOR>>\n"
            for fam_name, fam_data in sorted(
                families.items(), key=lambda x: -x[1]["p_value"]
            ):
                marker = " ← Best" if fam_name == best_family else ""
                p = fam_data["p_value"]
                status = (
                    "<<COLOR:success>>normal<</COLOR>>"
                    if p > 0.05
                    else "<<COLOR:warning>>non-normal<</COLOR>>"
                )
                summary += f"  Johnson {fam_name}: Shapiro-Wilk p = {p:.4f} ({status}){marker}\n"

            summary += f"\n<<COLOR:success>>Best transformation: Johnson {best_family}<</COLOR>>\n"

            result["summary"] = summary
            result["guide_observation"] = (
                f"Johnson transform: best family={best_family}, p={best['p_value']:.4f}."
            )
            result["statistics"] = {
                "best_family": best_family,
                "p_original": float(p_orig),
                "p_transformed": best["p_value"],
            }
            result["narrative"] = _narrative(
                f"Johnson Transform: {best_family} family (p = {best['p_value']:.4f})",
                f"Original data normality p = {p_orig:.4f}. After {best_family} transformation, p = {best['p_value']:.4f}.",
                next_steps="Use the Johnson-transformed data for parametric analyses requiring normality. Unlike Box-Cox, Johnson handles bounded and unbounded distributions.",
            )

            # Plots: before and after
            result["plots"].append(
                {
                    "title": f"Original: {var}",
                    "data": [
                        {
                            "type": "histogram",
                            "x": data.tolist(),
                            "marker": {
                                "color": "rgba(232,87,71,0.4)",
                                "line": {"color": "#e85747", "width": 1},
                            },
                        }
                    ],
                    "layout": {"height": 200, "xaxis": {"title": var}},
                }
            )
            result["plots"].append(
                {
                    "title": f"Johnson {best_family} Transformed",
                    "data": [
                        {
                            "type": "histogram",
                            "x": best["transformed"].tolist(),
                            "marker": {
                                "color": "rgba(74,159,110,0.4)",
                                "line": {"color": "#4a9f6e", "width": 1},
                            },
                        }
                    ],
                    "layout": {
                        "height": 200,
                        "xaxis": {"title": f"Johnson {best_family}({var})"},
                    },
                }
            )
        else:
            summary += "\n<<COLOR:warning>>Could not fit any Johnson family to this data.<</COLOR>>"
            result["summary"] = summary

    return result


def run_grubbs_test(df, config):
    """
    Grubbs' test for a single outlier.
    Tests whether the most extreme value is significantly different.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")
    alpha = config.get("alpha", 0.05)

    vals = df[var].dropna().values
    n = len(vals)

    if n < 3:
        result["summary"] = "Grubbs' test requires at least 3 observations."
    else:
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))

        if std_val == 0:
            result["summary"] = "All values are identical — no outlier possible."
        else:
            # Find most extreme value
            deviations = np.abs(vals - mean_val)
            max_idx = int(np.argmax(deviations))
            suspect = float(vals[max_idx])
            G = float(deviations[max_idx] / std_val)

            # Critical value from t-distribution
            t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

            # Two-sided p-value (approximation)
            _t_stat = G * np.sqrt(n) / np.sqrt(n - 1)  # noqa: F841
            _p_val = (
                min(
                    1.0,
                    2
                    * n
                    * (
                        1
                        - stats.t.cdf(
                            np.sqrt(
                                (n * (n - 2) * G**2) / (n - 1 - G**2 * (n - 1) / n)
                            ),
                            n - 2,
                        )
                    ),
                )
                if G**2 < n * (n - 1) / n
                else 0.0
            )

            is_outlier = G > G_crit
            verdict = (
                "<<COLOR:danger>>Yes — significant outlier<</COLOR>>"
                if is_outlier
                else "<<COLOR:success>>No — not a significant outlier<</COLOR>>"
            )

            summary = f"""<<COLOR:title>>GRUBBS' OUTLIER TEST<</COLOR>>
{"=" * 50}
<<COLOR:highlight>>Variable:<</COLOR>> {var}
<<COLOR:highlight>>N:<</COLOR>> {n}
<<COLOR:highlight>>Significance level:<</COLOR>> {alpha}

<<COLOR:accent>>Results<</COLOR>>
  Suspect value: {suspect:.6g}
  Mean:          {mean_val:.6g}
  StDev:         {std_val:.6g}

  G statistic:   {G:.4f}
  G critical:    {G_crit:.4f}

  Outlier? {verdict}"""

            result["summary"] = summary
            result["guide_observation"] = (
                f"Grubbs' test on {var}: suspect={suspect:.4g}, G={G:.3f}, {'outlier' if is_outlier else 'not outlier'} at α={alpha}."
            )
            result["statistics"] = {
                "G": G,
                "G_critical": G_crit,
                "suspect_value": suspect,
                "is_outlier": is_outlier,
            }
            if is_outlier:
                result["narrative"] = _narrative(
                    f"Outlier detected: {suspect:.4g} (G = {G:.3f})",
                    f"Value {suspect:.4g} is a statistical outlier at \u03b1 = {alpha}. G = {G:.3f} exceeds critical value {G_crit:.3f}.",
                    next_steps="Investigate the outlier. If it's a data entry error, correct it. If real, consider robust methods.",
                )
            else:
                result["narrative"] = _narrative(
                    f"No outlier detected (G = {G:.3f})",
                    f"Most extreme value {suspect:.4g} is not a statistical outlier. G = {G:.3f} < {G_crit:.3f}.",
                    next_steps="All values are within expected range for a normal distribution.",
                )

            # ── Diagnostics ──
            diagnostics = []
            # Normality check — Grubbs assumes normality
            _norm = _check_normality(vals, label=var, alpha=alpha)
            if _norm:
                _norm["detail"] = (
                    "Grubbs' test assumes normality. Non-normal data may produce false outlier flags. Consider IQR or robust methods."
                )
                diagnostics.append(_norm)
            # Sample size warning
            if n < 7:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": f"Very small sample (n = {n})",
                        "detail": "Grubbs' test has very low power with fewer than 7 observations. The test may fail to detect true outliers.",
                    }
                )
            # Outlier-specific diagnostics
            if is_outlier:
                diagnostics.append(
                    {
                        "level": "info",
                        "title": "Investigate outlier impact",
                        "detail": f"Value {suspect:.4g} was flagged as an outlier. Determine if it is a data entry error, measurement artifact, or genuine extreme observation before removing it.",
                        "action": {
                            "label": "Investigate Impact",
                            "type": "stats",
                            "analysis": "robust_regression",
                        },
                    }
                )
                # Masking warning
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": "Potential masking effect",
                        "detail": "Grubbs' test examines one outlier at a time. If multiple outliers exist, they can mask each other — the presence of one extreme value may prevent detection of others. Re-run after addressing this outlier.",
                    }
                )
            result["diagnostics"] = diagnostics

            # Highlight plot
            colors = ["#4a9f6e" if i != max_idx else "#d94a4a" for i in range(n)]
            sizes = [5 if i != max_idx else 12 for i in range(n)]
            result["plots"].append(
                {
                    "title": f"Grubbs' Test: {var}",
                    "data": [
                        {
                            "type": "scatter",
                            "x": list(range(1, n + 1)),
                            "y": vals.tolist(),
                            "mode": "markers",
                            "marker": {"color": colors, "size": sizes},
                            "name": var,
                        },
                        {
                            "type": "scatter",
                            "x": [1, n],
                            "y": [mean_val, mean_val],
                            "mode": "lines",
                            "line": {"color": "#e89547", "dash": "dash"},
                            "name": f"Mean = {mean_val:.4g}",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Observation"},
                        "yaxis": {"title": var},
                    },
                }
            )

    return result


def run_distribution_fit(df, config):
    """
    General-purpose distribution fitting — fits 12+ distributions,
    ranks by AIC/BIC, provides probability plots for top fits.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    x = df[var].dropna().values.astype(float)
    n = len(x)

    if n < 5:
        result["summary"] = "Need at least 5 data points for distribution fitting."
        return result

    # Candidate distributions with scipy names and display names
    candidates = [
        ("norm", "Normal", {}),
        ("lognorm", "Lognormal", {}),
        ("weibull_min", "Weibull", {}),
        ("gamma", "Gamma", {}),
        ("beta", "Beta", {}),
        ("expon", "Exponential", {}),
        ("logistic", "Logistic", {}),
        ("rayleigh", "Rayleigh", {}),
        ("invgauss", "Inverse Gaussian", {}),
    ]

    # Only try lognormal/gamma/weibull/beta/exponential/rayleigh/invgauss if data > 0
    has_negative = np.any(x <= 0)

    fit_results = []
    for dist_name, display_name, extra_kwargs in candidates:
        # Skip distributions that require positive data
        if has_negative and dist_name in (
            "lognorm",
            "weibull_min",
            "gamma",
            "beta",
            "expon",
            "rayleigh",
            "invgauss",
        ):
            continue
        # Beta requires data in (0, 1) range
        if dist_name == "beta" and (np.min(x) <= 0 or np.max(x) >= 1):
            continue
        try:
            dist_obj = getattr(stats, dist_name)
            params = dist_obj.fit(x, **extra_kwargs)
            # Log-likelihood
            ll = np.sum(dist_obj.logpdf(x, *params))
            if not np.isfinite(ll):
                continue
            k = len(params)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            # Anderson-Darling statistic (compare CDF)
            sorted_x = np.sort(x)
            cdf_vals = dist_obj.cdf(sorted_x, *params)
            cdf_vals = np.clip(cdf_vals, 1e-15, 1 - 1e-15)
            ad_stat = (
                -n
                - np.sum(
                    (2 * np.arange(1, n + 1) - 1)
                    * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
                )
                / n
            )
            # KS test
            ks_stat, ks_pval = stats.kstest(x, dist_name, args=params)
            fit_results.append(
                {
                    "dist_name": dist_name,
                    "display_name": display_name,
                    "params": params,
                    "param_names": (
                        dist_obj.shapes.split(", ") if dist_obj.shapes else []
                    ),
                    "ll": ll,
                    "aic": aic,
                    "bic": bic,
                    "ad_stat": ad_stat,
                    "ks_stat": ks_stat,
                    "ks_pval": ks_pval,
                }
            )
        except Exception:
            continue

    if not fit_results:
        result["summary"] = "No distributions could be fit to this data."
        return result

    # Sort by AIC
    fit_results.sort(key=lambda r: r["aic"])
    best = fit_results[0]

    # Summary
    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>DISTRIBUTION FITTING<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n"
    summary += (
        f"<<COLOR:highlight>>Distributions tested:<</COLOR>> {len(fit_results)}\n\n"
    )
    summary += "<<COLOR:accent>>── AIC/BIC Ranking (lower = better) ──<</COLOR>>\n"
    summary += f"  {'Rank':<5} {'Distribution':<22} {'AIC':>10} {'BIC':>10} {'KS p':>8} {'AD':>8}\n"
    summary += f"  {'-' * 65}\n"
    for i, fr in enumerate(fit_results):
        marker = "<<COLOR:good>>★<</COLOR>>" if i == 0 else f"  {i + 1}"
        ks_color = "good" if fr["ks_pval"] >= 0.05 else "bad"
        summary += f"  {marker:<5} {fr['display_name']:<22} {fr['aic']:>10.2f} {fr['bic']:>10.2f} <<COLOR:{ks_color}>>{fr['ks_pval']:>8.4f}<</COLOR>> {fr['ad_stat']:>8.3f}\n"

    # Parameter table for top 3
    summary += "\n<<COLOR:accent>>── Parameter Estimates (Top 3) ──<</COLOR>>\n"
    for i, fr in enumerate(fit_results[:3]):
        dist_obj = getattr(stats, fr["dist_name"])
        param_names = list(fr["param_names"]) + ["loc", "scale"]
        param_str = ", ".join(
            f"{name}={val:.4f}" for name, val in zip(param_names, fr["params"])
        )
        summary += f"  {fr['display_name']}: {param_str}\n"

    result["summary"] = summary

    # Histogram with top 3 PDF overlays
    x_range = np.linspace(float(x.min()), float(x.max()), 200)
    bin_width = (x.max() - x.min()) / min(30, max(5, int(np.sqrt(n))))
    hist_trace = {
        "type": "histogram",
        "x": x.tolist(),
        "marker": {
            "color": "rgba(74, 159, 110, 0.3)",
            "line": {"color": "#4a9f6e", "width": 1},
        },
        "name": "Data",
    }
    pdf_colors = ["#d94a4a", "#47a5e8", "#e89547"]
    pdf_traces = []
    for i, fr in enumerate(fit_results[:3]):
        dist_obj = getattr(stats, fr["dist_name"])
        pdf_vals = dist_obj.pdf(x_range, *fr["params"]) * n * bin_width
        pdf_traces.append(
            {
                "type": "scatter",
                "x": x_range.tolist(),
                "y": pdf_vals.tolist(),
                "mode": "lines",
                "line": {"color": pdf_colors[i], "width": 2},
                "name": f"{fr['display_name']} (AIC={fr['aic']:.0f})",
            }
        )
    result["plots"].append(
        {
            "title": f"Distribution Fit: {var}",
            "data": [hist_trace] + pdf_traces,
            "layout": {
                "height": 320,
                "xaxis": {"title": var},
                "yaxis": {"title": "Count"},
                "barmode": "overlay",
            },
        }
    )

    # Probability plots for top 4
    for i, fr in enumerate(fit_results[:4]):
        dist_obj = getattr(stats, fr["dist_name"])
        sorted_x = np.sort(x)
        theoretical = dist_obj.ppf((np.arange(1, n + 1) - 0.5) / n, *fr["params"])
        if not np.all(np.isfinite(theoretical)):
            continue
        result["plots"].append(
            {
                "title": f"Probability Plot: {fr['display_name']}",
                "data": [
                    {
                        "type": "scatter",
                        "x": theoretical.tolist(),
                        "y": sorted_x.tolist(),
                        "mode": "markers",
                        "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 5},
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": [
                            float(min(theoretical.min(), sorted_x.min())),
                            float(max(theoretical.max(), sorted_x.max())),
                        ],
                        "y": [
                            float(min(theoretical.min(), sorted_x.min())),
                            float(max(theoretical.max(), sorted_x.max())),
                        ],
                        "mode": "lines",
                        "line": {"color": "#ff7675", "dash": "dash"},
                        "name": "Perfect fit",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": f"Theoretical ({fr['display_name']})"},
                    "yaxis": {"title": "Sample"},
                },
            }
        )

    # Shape interpretation for best fit
    _shape_desc = ""
    if best["dist_name"] == "norm":
        _shape_desc = "symmetric, bell-shaped"
    elif best["dist_name"] == "lognorm":
        _shape_desc = "right-skewed with a long upper tail — common in cycle times, financial data, and natural phenomena"
    elif best["dist_name"] == "weibull_min":
        shape_param = best["params"][0]
        if shape_param < 1:
            _shape_desc = "decreasing failure rate (infant mortality pattern)"
        elif abs(shape_param - 1) < 0.1:
            _shape_desc = "constant failure rate (random / exponential-like)"
        else:
            _shape_desc = (
                f"increasing failure rate (wear-out pattern, shape = {shape_param:.2f})"
            )
    elif best["dist_name"] == "gamma":
        _shape_desc = "right-skewed, flexible shape — common in wait times and queuing"
    elif best["dist_name"] == "expon":
        _shape_desc = "memoryless / constant hazard rate — common in reliability"
    elif best["dist_name"] == "logistic":
        _shape_desc = "symmetric but heavier-tailed than normal"
    elif best["dist_name"] == "beta":
        _shape_desc = "bounded on [0,1] — common for proportions and probabilities"
    elif best["dist_name"] == "rayleigh":
        _shape_desc = (
            "right-skewed, useful for magnitudes (e.g., wind speed, vibration)"
        )
    elif best["dist_name"] == "invgauss":
        _shape_desc = (
            "right-skewed with heavy upper tail — common in first-passage times"
        )
    else:
        _shape_desc = "see probability plot for shape assessment"

    _aic_delta = fit_results[1]["aic"] - best["aic"] if len(fit_results) > 1 else 0
    _aic_strength = (
        "decisively"
        if _aic_delta > 10
        else ("substantially" if _aic_delta > 4 else "marginally")
    )

    result["guide_observation"] = (
        f"Best fit: {best['display_name']} (AIC = {best['aic']:.1f}). {_shape_desc.capitalize()}."
    )
    result["narrative"] = _narrative(
        f"{best['display_name']} provides the best fit (\u0394AIC = {_aic_delta:.1f} {_aic_strength} better than {fit_results[1]['display_name'] if len(fit_results) > 1 else 'N/A'})",
        f"The data is best described by a <strong>{best['display_name']}</strong> distribution (AIC = {best['aic']:.1f}, KS p = {best['ks_pval']:.4f}). Shape: {_shape_desc}.",
        next_steps="Use the fitted distribution for reliability analysis, non-normal capability, or simulation inputs. If KS p < 0.05, consider transformations or mixture models.",
        chart_guidance="Points on the probability plot diagonal = good fit. Systematic curvature = misfit. The histogram overlay compares the top 3 fitted PDFs to the data.",
    )
    result["statistics"] = {
        "best_distribution": best["display_name"],
        "best_aic": float(best["aic"]),
        "best_bic": float(best["bic"]),
        "best_ks_pval": float(best["ks_pval"]),
        "n_distributions_tested": len(fit_results),
    }

    # ── Diagnostics ──
    diagnostics = []
    # AIC ambiguity — top 2 distributions indistinguishable
    if len(fit_results) > 1:
        _aic_gap = fit_results[1]["aic"] - best["aic"]
        if _aic_gap < 2:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": "Top distributions are statistically indistinguishable",
                    "detail": f"AIC difference between {best['display_name']} and {fit_results[1]['display_name']} is only {_aic_gap:.1f} (< 2). Choose based on domain knowledge.",
                }
            )
    # Best fit is Normal — parametric methods appropriate
    if best["dist_name"] == "norm":
        diagnostics.append(
            {
                "level": "info",
                "title": "Normal distribution confirmed",
                "detail": "Parametric methods (t-tests, ANOVA, control charts) are appropriate for this data.",
            }
        )
    # Best fit is non-Normal — suggest non-normal capability and transformation
    if best["dist_name"] != "norm":
        diagnostics.append(
            {
                "level": "info",
                "title": f"Data follows {best['display_name']} distribution",
                "detail": "Standard parametric assumptions may not hold. Consider non-normal capability analysis or transforming to normal.",
                "action": {
                    "label": "Non-Normal Capability",
                    "type": "stats",
                    "analysis": "nonnormal_capability_np",
                    "config": {"var": var},
                },
            }
        )
        diagnostics.append(
            {
                "level": "info",
                "title": "Transform to Normal",
                "detail": "A Box-Cox or Johnson transformation may normalize the data for parametric analysis.",
                "action": {
                    "label": "Transform to Normal",
                    "type": "stats",
                    "analysis": "box_cox",
                    "config": {"var": var},
                },
            }
        )
    # All fits are poor — no standard distribution fits well
    _all_poor = all(fr["ks_pval"] < 0.05 for fr in fit_results)
    if _all_poor:
        diagnostics.append(
            {
                "level": "warning",
                "title": "No standard distribution fits well",
                "detail": "All candidate distributions have KS p < 0.05. The data may come from a mixture of populations. Consider mixture models.",
                "action": {
                    "label": "Mixture Model",
                    "type": "stats",
                    "analysis": "mixture_model",
                    "config": {"var": var},
                },
            }
        )
    result["diagnostics"] = diagnostics

    return result


def run_mixture_model(df, config):
    """Gaussian Mixture Model — detect hidden subpopulations."""
    from sklearn.mixture import GaussianMixture

    result = {"plots": [], "summary": "", "guide_observation": ""}

    col = config.get("var") or config.get("variable") or config.get("column")
    max_k = min(int(config.get("max_k") or config.get("max_components") or 6), 10)

    if not col or col not in df.columns:
        result["summary"] = "Error: Specify a numeric column."
        return result

    data = df[col].dropna().values.astype(float).reshape(-1, 1)
    n = len(data)

    if n < 20:
        result["summary"] = "Error: Need at least 20 observations."
        return result

    # Fit GMMs with k=1..max_k, select by BIC
    results_k = []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(data)
        bic = gmm.bic(data)
        aic = gmm.aic(data)
        results_k.append({"k": k, "bic": float(bic), "aic": float(aic), "model": gmm})

    best_k_idx = int(np.argmin([r["bic"] for r in results_k]))
    best = results_k[best_k_idx]
    best_gmm = best["model"]
    k_best = best["k"]

    # Extract component parameters
    components = []
    for j in range(k_best):
        components.append(
            {
                "mean": float(best_gmm.means_[j, 0]),
                "std": float(np.sqrt(best_gmm.covariances_[j, 0, 0])),
                "weight": float(best_gmm.weights_[j]),
            }
        )
    components.sort(key=lambda c: c["mean"])

    # Assign labels
    best_gmm.predict(data)  # labels assigned for side-effect verification

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>GAUSSIAN MIXTURE MODEL<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Variable:<</COLOR>> {col}    N: {n}\n"
    summary += f"<<COLOR:text>>Best k:<</COLOR>> {k_best} components (by BIC)\n\n"

    if k_best == 1:
        summary += (
            "<<COLOR:success>>Data is consistent with a single population.<</COLOR>>\n"
        )
        summary += (
            f"  Mean: {components[0]['mean']:.4f}    Std: {components[0]['std']:.4f}\n"
        )
    else:
        summary += f"<<COLOR:warning>>Data is best described as {k_best} overlapping populations:<</COLOR>>\n\n"
        for j, c in enumerate(components):
            summary += f"  Component {j + 1}: \u03bc={c['mean']:.4f}, \u03c3={c['std']:.4f}, weight={c['weight']:.1%}\n"

    summary += (
        "\n<<COLOR:accent>>\u2500\u2500 Model Comparison (BIC) \u2500\u2500<</COLOR>>\n"
    )
    for r in results_k:
        marker = " \u2190 best" if r["k"] == k_best else ""
        summary += f"  k={r['k']}: BIC={r['bic']:.1f}{marker}\n"

    result["summary"] = summary
    result["statistics"] = {
        "best_k": k_best,
        "components": components,
        "bic_values": {r["k"]: r["bic"] for r in results_k},
    }

    if k_best == 1:
        result["guide_observation"] = (
            f"Mixture model: single population (\u03bc={components[0]['mean']:.3f}, \u03c3={components[0]['std']:.3f})."
        )
        result["narrative"] = _narrative(
            "Mixture Model \u2014 single population",
            f"BIC selects k=1: the data is consistent with a single Gaussian (\u03bc={components[0]['mean']:.4f}, \u03c3={components[0]['std']:.4f}). "
            "No evidence of hidden subpopulations.",
            next_steps="If you suspect stratification, check whether grouping by a categorical variable (shift, supplier, machine) reveals separation.",
        )
    else:
        _mm_desc = "; ".join(
            f"one at {c['mean']:.3f} ({c['weight']:.0%})" for c in components
        )
        result["guide_observation"] = (
            f"Mixture model: {k_best} populations detected \u2014 {_mm_desc}."
        )
        _mm_gap = max(
            abs(components[i + 1]["mean"] - components[i]["mean"])
            for i in range(len(components) - 1)
        )
        result["narrative"] = _narrative(
            f"Mixture Model \u2014 {k_best} populations detected",
            f"BIC selects k={k_best}: the data is best described as {k_best} overlapping Gaussians. "
            + " ".join(
                f"<strong>Component {j + 1}</strong>: \u03bc={c['mean']:.4f}, \u03c3={c['std']:.4f} ({c['weight']:.0%} of data)."
                for j, c in enumerate(components)
            )
            + f" The largest gap between means is {_mm_gap:.4f}.",
            next_steps="This often indicates an uncontrolled stratification variable (shift, supplier, machine, cavity). "
            "Investigate what separates the subpopulations. If intentional, analyze each component's capability separately.",
            chart_guidance="The histogram shows the overall data. Overlaid curves show each fitted component. "
            "If the components are well-separated, a stratification variable is likely driving the split.",
        )

    # Plot: histogram with overlaid component densities
    x_plot = np.linspace(
        float(data.min()) - 2 * components[-1]["std"],
        float(data.max()) + 2 * components[-1]["std"],
        300,
    )
    plot_data_list = [
        {
            "type": "histogram",
            "x": data.ravel().tolist(),
            "nbinsx": min(50, n // 3),
            "marker": {
                "color": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e", "width": 1},
            },
            "name": "Data",
            "yaxis": "y2",
        },
    ]
    colors = ["#4a90d9", "#dc5050", "#d4a24a", "#6ab7d4", "#9b59b6", "#e67e22"]
    total_density = np.zeros_like(x_plot)
    for j, c in enumerate(components):
        dens = c["weight"] * sp_stats.norm.pdf(x_plot, c["mean"], c["std"])
        total_density += dens
        if k_best > 1:
            plot_data_list.append(
                {
                    "type": "scatter",
                    "x": x_plot.tolist(),
                    "y": dens.tolist(),
                    "line": {
                        "color": colors[j % len(colors)],
                        "width": 2,
                        "dash": "dash",
                    },
                    "name": f"Component {j + 1} ({c['weight']:.0%})",
                }
            )
    plot_data_list.append(
        {
            "type": "scatter",
            "x": x_plot.tolist(),
            "y": total_density.tolist(),
            "line": {"color": "#4a9f6e", "width": 2},
            "name": "Mixture",
        }
    )

    result["plots"].append(
        {
            "title": f"Mixture Model ({col}) \u2014 {k_best} component{'s' if k_best > 1 else ''}",
            "data": plot_data_list,
            "layout": {
                "height": 320,
                "xaxis": {"title": col},
                "yaxis": {"title": "Density", "side": "left"},
                "yaxis2": {
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "title": "Count",
                },
                "barmode": "overlay",
            },
        }
    )

    # Plot: BIC curve
    result["plots"].append(
        {
            "title": "BIC vs Number of Components",
            "data": [
                {
                    "type": "scatter",
                    "x": [r["k"] for r in results_k],
                    "y": [r["bic"] for r in results_k],
                    "mode": "lines+markers",
                    "marker": {
                        "size": 8,
                        "color": [
                            "#4a9f6e" if r["k"] == k_best else "#999" for r in results_k
                        ],
                    },
                    "line": {"color": "#4a90d9"},
                }
            ],
            "layout": {
                "height": 220,
                "xaxis": {"title": "k (components)", "dtick": 1},
                "yaxis": {"title": "BIC"},
            },
        }
    )

    return result
