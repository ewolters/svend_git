"""Forge-backed exploratory analysis handlers.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .forge_stats import _alpha, _col, _col2, _pval_str

logger = logging.getLogger(__name__)


# =============================================================================
# Univariate / Transformations
# =============================================================================


def forge_poisson_1sample(df, config):
    """One-sample Poisson test."""
    data, col_name = _col(df, config, "column", "var1")
    mu0 = float(config.get("test_value", config.get("mu0", data.mean())))
    alpha = _alpha(config)

    observed = data.sum()
    n = len(data)
    rate = data.mean()
    # Exact Poisson test: P(X >= observed | mu0*n) or P(X <= observed | mu0*n)
    expected_total = mu0 * n
    p_upper = 1 - sp_stats.poisson.cdf(observed - 1, expected_total)
    p_lower = sp_stats.poisson.cdf(observed, expected_total)
    p_value = 2 * min(p_upper, p_lower)  # two-sided

    ci_lo, ci_hi = sp_stats.poisson.interval(1 - alpha, rate * n)
    ci_lo /= n
    ci_hi /= n

    return {
        "plots": [],
        "statistics": {
            "observed_rate": round(rate, 4),
            "test_value": mu0,
            "total_count": int(observed),
            "n": n,
            "p_value": round(p_value, 6),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>One-Sample Poisson Test<</COLOR>>\n\n"
            f"<<COLOR:text>>Observed rate = {rate:.4f}, H\u2080: \u03bb = {mu0}<</COLOR>>\n"
            f"<<COLOR:text>>p = {_pval_str(p_value)}<</COLOR>>\n"
            f"<<COLOR:text>>{(1 - alpha) * 100:.0f}% CI: [{ci_lo:.4f}, {ci_hi:.4f}]<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Rate = {rate:.4f} vs \u03bb\u2080 = {mu0}, p = {_pval_str(p_value)}",
            "body": f"Observed rate {rate:.4f} {'differs' if p_value < alpha else 'does not differ'} significantly from {mu0}.",
            "next_steps": "Check for overdispersion if variance >> mean.",
            "chart_guidance": "",
        },
        "guide_observation": f"Poisson 1-sample: rate={rate:.4f}, p={_pval_str(p_value)}.",
        "diagnostics": [],
    }


def forge_poisson_2sample(df, config):
    """Two-sample Poisson rate comparison."""
    c1, n1, c2, n2 = _col2(df, config)
    alpha = _alpha(config)

    rate1, rate2 = c1.mean(), c2.mean()
    len1, len2 = len(c1), len(c2)

    # Z-test for two Poisson rates
    rate_diff = rate1 - rate2
    se = math.sqrt(rate1 / len1 + rate2 / len2) if (rate1 > 0 or rate2 > 0) else 1e-10
    z = rate_diff / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {
        "plots": [],
        "statistics": {
            "rate1": round(rate1, 4),
            "rate2": round(rate2, 4),
            "rate_diff": round(rate_diff, 4),
            "z_statistic": round(z, 4),
            "p_value": round(p_value, 6),
            "n1": len1,
            "n2": len2,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Two-Sample Poisson Test<</COLOR>>\n\n"
            f"<<COLOR:text>>{n1}: rate = {rate1:.4f}, {n2}: rate = {rate2:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Z = {z:.4f}, p = {_pval_str(p_value)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Rate difference = {rate_diff:.4f}, p = {_pval_str(p_value)}",
            "body": f"Rates {'differ' if p_value < alpha else 'do not differ'} significantly.",
            "next_steps": "Consider negative binomial if overdispersed.",
            "chart_guidance": "",
        },
        "guide_observation": f"Poisson 2-sample: Z={z:.4f}, p={_pval_str(p_value)}.",
        "diagnostics": [],
    }


def forge_box_cox(df, config):
    """Box-Cox transformation via forgestat."""
    from forgestat.core.distributions import box_cox

    data, col_name = _col(df, config, "column", "var1")
    # Box-Cox requires positive data
    shift = 0
    if data.min() <= 0:
        shift = abs(data.min()) + 1
        data = data + shift

    transformed, lam = box_cox(data.tolist())

    # Normality test on transformed data
    _, p_before = sp_stats.shapiro(data[: min(len(data), 5000)])
    _, p_after = sp_stats.shapiro(transformed[: min(len(transformed), 5000)])

    return {
        "plots": [],
        "statistics": {
            "lambda": round(float(lam), 4),
            "shift": shift,
            "normality_p_before": round(p_before, 6),
            "normality_p_after": round(p_after, 6),
            "n": len(data),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Box-Cox Transformation<</COLOR>>\n\n"
            f"<<COLOR:text>>\u03bb = {lam:.4f}" + (f" (shifted +{shift})" if shift else "") + f"<</COLOR>>\n"
            f"<<COLOR:text>>Normality p-value: before = {_pval_str(p_before)}, after = {_pval_str(p_after)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Box-Cox \u03bb = {lam:.4f}",
            "body": f"Optimal \u03bb = {lam:.4f}. Normality {'improved' if p_after > p_before else 'not improved'} by transformation.",
            "next_steps": "Use transformed data for analyses assuming normality.",
            "chart_guidance": "Compare histograms before/after transformation.",
        },
        "guide_observation": f"Box-Cox: \u03bb={lam:.4f}, normality p: {p_before:.4f}\u2192{p_after:.4f}.",
        "diagnostics": [],
    }


def forge_johnson_transform(df, config):
    """Johnson transformation via forgestat."""
    from forgestat.core.distributions import johnson_transform

    data, col_name = _col(df, config, "column", "var1")
    transformed, family, params = johnson_transform(data.tolist())

    _, p_before = sp_stats.shapiro(data[: min(len(data), 5000)])
    _, p_after = sp_stats.shapiro(transformed[: min(len(transformed), 5000)])

    return {
        "plots": [],
        "statistics": {
            "family": family,
            "params": [round(float(p), 4) for p in params] if params else [],
            "normality_p_before": round(p_before, 6),
            "normality_p_after": round(p_after, 6),
            "n": len(data),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Johnson Transformation<</COLOR>>\n\n"
            f"<<COLOR:text>>Family: {family}<</COLOR>>\n"
            f"<<COLOR:text>>Normality p-value: before = {_pval_str(p_before)}, after = {_pval_str(p_after)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Johnson {family} transform",
            "body": f"Johnson {family} family selected. Normality {'improved' if p_after > p_before else 'not improved'}.",
            "next_steps": "Use transformed data for capability analysis or parametric tests.",
            "chart_guidance": "Compare histograms before/after.",
        },
        "guide_observation": f"Johnson: family={family}, normality p: {p_before:.4f}\u2192{p_after:.4f}.",
        "diagnostics": [],
    }


def forge_grubbs_test(df, config):
    """Grubbs' test for outliers."""
    data, col_name = _col(df, config, "column", "var1")
    alpha = _alpha(config)
    n = len(data)

    mean, std = data.mean(), data.std(ddof=1)
    # Grubbs statistic = max|x_i - mean| / std
    deviations = np.abs(data - mean)
    max_idx = np.argmax(deviations)
    g_stat = deviations[max_idx] / std

    # Critical value from t-distribution
    t_crit = sp_stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / math.sqrt(n)) * math.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    outlier_detected = g_stat > g_crit
    outlier_value = float(data[max_idx])

    return {
        "plots": [],
        "statistics": {
            "grubbs_statistic": round(g_stat, 4),
            "critical_value": round(g_crit, 4),
            "outlier_detected": outlier_detected,
            "outlier_value": round(outlier_value, 4),
            "outlier_index": int(max_idx),
            "n": n,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Grubbs' Test for Outliers<</COLOR>>\n\n"
            f"<<COLOR:text>>G = {g_stat:.4f}, critical = {g_crit:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Outlier {'detected' if outlier_detected else 'not detected'}: {outlier_value:.4f}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{'Outlier detected' if outlier_detected else 'No outlier'} (G = {g_stat:.4f})",
            "body": f"Most extreme value = {outlier_value:.4f}. G = {g_stat:.4f} {'>' if outlier_detected else '<='} {g_crit:.4f}.",
            "next_steps": "If outlier detected, investigate root cause before removing.",
            "chart_guidance": "",
        },
        "guide_observation": f"Grubbs: G={g_stat:.4f}, {'outlier' if outlier_detected else 'no outlier'} at {outlier_value:.4f}.",
        "diagnostics": [],
    }


def forge_distribution_fit(df, config):
    """Distribution fitting via forgestat."""
    from forgestat.core.distributions import fit_best

    data, col_name = _col(df, config, "column", "var1")
    result = fit_best(data.tolist())

    best = result.best
    fits_summary = []
    for fit in result.all_fits[:5]:
        fits_summary.append(
            f"  <<COLOR:highlight>>{fit.name}:<</COLOR>> AIC={fit.aic:.1f}, KS p={_pval_str(fit.ks_p_value)}"
        )

    return {
        "plots": [],
        "statistics": {
            "best_distribution": best.name,
            "best_aic": round(best.aic, 2),
            "best_bic": round(best.bic, 2) if best.bic else None,
            "best_ks_statistic": round(best.ks_statistic, 4),
            "best_ks_p_value": round(best.ks_p_value, 6),
            "n_distributions_tested": len(result.all_fits),
            "n": len(data),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Distribution Fit<</COLOR>>\n\n"
            f"<<COLOR:text>>Best: {best.name} (AIC = {best.aic:.1f})<</COLOR>>\n"
            f"<<COLOR:text>>KS test: D = {best.ks_statistic:.4f}, p = {_pval_str(best.ks_p_value)}<</COLOR>>\n\n"
            + "\n".join(fits_summary)
        ),
        "narrative": {
            "verdict": f"Best fit: {best.name} (AIC = {best.aic:.1f})",
            "body": f"Tested {len(result.all_fits)} distributions. {best.name} provides the best fit (AIC = {best.aic:.1f}).",
            "next_steps": "Use Q-Q plot to visually confirm fit. Consider physical plausibility.",
            "chart_guidance": "Histogram with fitted PDF overlay. Q-Q plot for goodness-of-fit.",
        },
        "guide_observation": f"Best fit: {best.name}, AIC={best.aic:.1f}, KS p={_pval_str(best.ks_p_value)}.",
        "diagnostics": [],
    }


def forge_mixture_model(df, config):
    """Gaussian mixture model."""
    from sklearn.mixture import GaussianMixture

    data, col_name = _col(df, config, "column", "var1")
    n_components = int(config.get("n_components", config.get("k", 2)))

    X = data.reshape(-1, 1)
    gm = GaussianMixture(n_components=n_components, random_state=42)
    gm.fit(X)

    means = gm.means_.flatten().tolist()
    variances = gm.covariances_.flatten().tolist()
    weights = gm.weights_.tolist()
    bic = gm.bic(X)
    aic = gm.aic(X)

    comp_lines = []
    for i in range(n_components):
        comp_lines.append(
            f"  <<COLOR:highlight>>Component {i + 1}:<</COLOR>> \u03bc={means[i]:.4f}, \u03c3\u00b2={variances[i]:.4f}, weight={weights[i]:.3f}"
        )

    return {
        "plots": [],
        "statistics": {
            "n_components": n_components,
            "means": [round(m, 4) for m in means],
            "variances": [round(v, 4) for v in variances],
            "weights": [round(w, 4) for w in weights],
            "bic": round(bic, 2),
            "aic": round(aic, 2),
            "n": len(data),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Gaussian Mixture Model (k={n_components})<</COLOR>>\n\n"
            f"<<COLOR:text>>BIC = {bic:.1f}, AIC = {aic:.1f}<</COLOR>>\n\n" + "\n".join(comp_lines)
        ),
        "narrative": {
            "verdict": f"GMM with {n_components} components, BIC = {bic:.1f}",
            "body": f"Fitted {n_components}-component Gaussian mixture. BIC = {bic:.1f}.",
            "next_steps": "Compare BIC across different k values. Lower BIC = better model.",
            "chart_guidance": "Density plot with component PDFs overlaid.",
        },
        "guide_observation": f"GMM: k={n_components}, BIC={bic:.1f}.",
        "diagnostics": [],
    }


def forge_tolerance_interval(df, config):
    """Tolerance interval via forgestat."""
    from forgestat.exploratory.univariate import tolerance_interval

    data, col_name = _col(df, config, "column", "var1")
    coverage = float(config.get("coverage", 0.95))
    confidence = float(config.get("confidence", 0.95))
    method = config.get("method", "normal")

    result = tolerance_interval(data.tolist(), coverage=coverage, confidence=confidence, method=method)

    return {
        "plots": [],
        "statistics": {
            "lower": round(result.lower, 4),
            "upper": round(result.upper, 4),
            "coverage": result.coverage,
            "confidence": result.confidence,
            "k_factor": round(result.k_factor, 4),
            "method": result.method,
            "n": len(data),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Tolerance Interval ({method})<</COLOR>>\n\n"
            f"<<COLOR:text>>[{result.lower:.4f}, {result.upper:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>Coverage = {coverage * 100:.0f}%, Confidence = {confidence * 100:.0f}%<</COLOR>>\n"
            f"<<COLOR:text>>k-factor = {result.k_factor:.4f}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Tolerance interval: [{result.lower:.4f}, {result.upper:.4f}]",
            "body": f"With {confidence * 100:.0f}% confidence, at least {coverage * 100:.0f}% of the population falls within [{result.lower:.4f}, {result.upper:.4f}].",
            "next_steps": "Compare with specification limits for capability assessment.",
            "chart_guidance": "",
        },
        "guide_observation": f"Tolerance: [{result.lower:.4f}, {result.upper:.4f}], k={result.k_factor:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Sequential / Run chart / SPRT
# =============================================================================


def forge_run_chart(df, config):
    """Run chart analysis."""
    data, col_name = _col(df, config, "column", "var1")
    median = np.median(data)

    # Count runs
    above = data > median
    runs = 1
    for i in range(1, len(above)):
        if above[i] != above[i - 1]:
            runs += 1

    n_above = above.sum()
    n_below = len(above) - n_above
    n = len(data)

    # Expected runs under randomness
    if n_above > 0 and n_below > 0:
        expected = 1 + 2 * n_above * n_below / n
        var_runs = 2 * n_above * n_below * (2 * n_above * n_below - n) / (n**2 * (n - 1))
        z = (runs - expected) / math.sqrt(max(var_runs, 1e-10))
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    else:
        expected = 1
        z = 0
        p_value = 1.0

    return {
        "plots": [],
        "statistics": {
            "n": n,
            "median": round(float(median), 4),
            "n_runs": runs,
            "expected_runs": round(expected, 2),
            "z_statistic": round(z, 4),
            "p_value": round(p_value, 6),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Run Chart<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {n}, median = {median:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Runs: {runs} (expected {expected:.1f}), Z = {z:.4f}, p = {_pval_str(p_value)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{'Non-random pattern' if p_value < 0.05 else 'Random'} (p = {_pval_str(p_value)})",
            "body": f"Observed {runs} runs vs {expected:.1f} expected. Z = {z:.4f}.",
            "next_steps": "Too few runs = trend/shift. Too many = oscillation.",
            "chart_guidance": "Points plotted in time order with median line.",
        },
        "guide_observation": f"Run chart: {runs} runs, Z={z:.4f}, p={_pval_str(p_value)}.",
        "diagnostics": [],
    }


def forge_sprt(df, config):
    """Sequential Probability Ratio Test."""
    data, col_name = _col(df, config, "column", "var1")
    mu0 = float(config.get("mu0", 0))
    mu1 = float(config.get("mu1", 1))
    sigma = float(config.get("sigma", data.std(ddof=1)))
    alpha_val = float(config.get("alpha", 0.05))
    beta = float(config.get("beta", 0.10))

    # Log-likelihood ratio boundaries
    A = math.log((1 - beta) / alpha_val)  # Accept H1
    B = math.log(beta / (1 - alpha_val))  # Accept H0

    cum_llr = 0
    llr_values = []
    decision = "continue"
    decision_idx = len(data)

    for i, x in enumerate(data):
        ll1 = sp_stats.norm.logpdf(x, mu1, sigma)
        ll0 = sp_stats.norm.logpdf(x, mu0, sigma)
        cum_llr += ll1 - ll0
        llr_values.append(cum_llr)
        if cum_llr >= A:
            decision = "reject H0"
            decision_idx = i + 1
            break
        elif cum_llr <= B:
            decision = "accept H0"
            decision_idx = i + 1
            break

    return {
        "plots": [],
        "statistics": {
            "decision": decision,
            "samples_used": decision_idx,
            "final_llr": round(cum_llr, 4),
            "upper_boundary": round(A, 4),
            "lower_boundary": round(B, 4),
            "mu0": mu0,
            "mu1": mu1,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Sequential Probability Ratio Test<</COLOR>>\n\n"
            f"<<COLOR:text>>Decision: {decision} after {decision_idx} samples<</COLOR>>\n"
            f"<<COLOR:text>>H\u2080: \u03bc = {mu0}, H\u2081: \u03bc = {mu1}<</COLOR>>\n"
            f"<<COLOR:text>>Final LLR = {cum_llr:.4f}, bounds = [{B:.4f}, {A:.4f}]<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"SPRT: {decision} after {decision_idx} observations",
            "body": f"Sequential test {'stopped early' if decision != 'continue' else 'did not reach a decision'} after {decision_idx} of {len(data)} observations.",
            "next_steps": "SPRT typically uses fewer samples than fixed-sample tests.",
            "chart_guidance": "LLR path between two boundaries. Crossing upper = reject H0, lower = accept H0.",
        },
        "guide_observation": f"SPRT: {decision} after {decision_idx} samples, LLR={cum_llr:.4f}.",
        "diagnostics": [],
    }


def forge_effect_size_calculator(df, config):
    """Effect size calculator."""
    data, col_name = _col(df, config, "column", "var1")

    # Try to get two groups
    group_col = config.get("factor") or config.get("group_var")
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        if len(groups) >= 2:
            g1 = pd.to_numeric(df.loc[df[group_col] == groups[0], col_name], errors="coerce").dropna().values
            g2 = pd.to_numeric(df.loc[df[group_col] == groups[1], col_name], errors="coerce").dropna().values
            pooled_std = math.sqrt(
                ((len(g1) - 1) * g1.std(ddof=1) ** 2 + (len(g2) - 1) * g2.std(ddof=1) ** 2) / (len(g1) + len(g2) - 2)
            )
            cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0
            glass_delta = (g1.mean() - g2.mean()) / g2.std(ddof=1) if g2.std(ddof=1) > 0 else 0
            hedges_g = cohens_d * (1 - 3 / (4 * (len(g1) + len(g2)) - 9))
            r = cohens_d / math.sqrt(cohens_d**2 + 4)

            return {
                "plots": [],
                "statistics": {
                    "cohens_d": round(cohens_d, 4),
                    "hedges_g": round(hedges_g, 4),
                    "glass_delta": round(glass_delta, 4),
                    "r": round(r, 4),
                    "n1": len(g1),
                    "n2": len(g2),
                },
                "assumptions": {},
                "summary": (
                    f"<<COLOR:header>>Effect Size Calculator<</COLOR>>\n\n"
                    f"<<COLOR:text>>Cohen's d = {cohens_d:.4f}, Hedges' g = {hedges_g:.4f}<</COLOR>>\n"
                    f"<<COLOR:text>>Glass's \u0394 = {glass_delta:.4f}, r = {r:.4f}<</COLOR>>"
                ),
                "narrative": {
                    "verdict": f"Cohen's d = {cohens_d:.4f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'})",
                    "body": f"Effect size between groups: d = {cohens_d:.4f}.",
                    "next_steps": "Report effect size alongside p-value for practical significance.",
                    "chart_guidance": "",
                },
                "guide_observation": f"Effect: d={cohens_d:.4f}, g={hedges_g:.4f}.",
                "diagnostics": [],
            }

    # Single sample: effect vs 0
    d = data.mean() / data.std(ddof=1) if data.std(ddof=1) > 0 else 0
    return {
        "plots": [],
        "statistics": {"cohens_d": round(d, 4), "n": len(data)},
        "assumptions": {},
        "summary": f"<<COLOR:header>>Effect Size<</COLOR>>\n\n<<COLOR:text>>Cohen's d = {d:.4f}<</COLOR>>",
        "narrative": {
            "verdict": f"d = {d:.4f}",
            "body": f"One-sample effect size = {d:.4f}.",
            "next_steps": "",
            "chart_guidance": "",
        },
        "guide_observation": f"Effect: d={d:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Data Quality / Profiling
# =============================================================================


def forge_data_profile(df, config):
    """Data profiling summary."""
    cols = config.get("columns") or list(df.columns)
    n_rows = len(df)
    profiles = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        missing = int(s.isna().sum())
        unique = int(s.nunique())
        dtype = str(s.dtype)
        entry = {
            "column": col,
            "dtype": dtype,
            "missing": missing,
            "missing_pct": round(missing / n_rows * 100, 1),
            "unique": unique,
        }
        if np.issubdtype(s.dtype, np.number):
            clean = pd.to_numeric(s, errors="coerce").dropna()
            entry.update(
                {
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                }
            )
        profiles.append(entry)

    total_missing = sum(p["missing"] for p in profiles)
    return {
        "plots": [],
        "statistics": {
            "n_rows": n_rows,
            "n_columns": len(profiles),
            "total_missing": total_missing,
            "total_missing_pct": round(total_missing / (n_rows * len(profiles)) * 100, 1) if profiles else 0,
        },
        "assumptions": {},
        "summary": f"<<COLOR:header>>Data Profile<</COLOR>>\n\n<<COLOR:text>>{n_rows} rows, {len(profiles)} columns, {total_missing} missing values<</COLOR>>",
        "narrative": {
            "verdict": f"{n_rows} rows, {len(profiles)} columns",
            "body": f"Data profile of {len(profiles)} columns. Overall {total_missing} missing values.",
            "next_steps": "Address columns with high missing rates before analysis.",
            "chart_guidance": "",
        },
        "guide_observation": f"Profile: {n_rows}x{len(profiles)}, {total_missing} missing.",
        "diagnostics": [],
        "profiles": profiles,
    }


def forge_auto_profile(df, config):
    """Automated data profiling — delegates to data_profile."""
    return forge_data_profile(df, config)


def forge_graphical_summary(df, config):
    """Graphical summary of a variable."""
    data, col_name = _col(df, config, "column", "var1")
    n = len(data)

    desc = {
        "mean": round(float(data.mean()), 4),
        "std": round(float(data.std(ddof=1)), 4),
        "median": round(float(np.median(data)), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "q1": round(float(np.percentile(data, 25)), 4),
        "q3": round(float(np.percentile(data, 75)), 4),
        "skewness": round(float(sp_stats.skew(data)), 4),
        "kurtosis": round(float(sp_stats.kurtosis(data)), 4),
        "n": n,
    }
    _, ad_p = sp_stats.normaltest(data) if n >= 20 else (0, 1)
    desc["normality_p"] = round(float(ad_p), 6)

    ci_lo = desc["mean"] - sp_stats.t.ppf(0.975, n - 1) * desc["std"] / math.sqrt(n)
    ci_hi = desc["mean"] + sp_stats.t.ppf(0.975, n - 1) * desc["std"] / math.sqrt(n)
    desc["ci_mean_lower"] = round(ci_lo, 4)
    desc["ci_mean_upper"] = round(ci_hi, 4)

    return {
        "plots": [],
        "statistics": desc,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Graphical Summary: {col_name}<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {n}, Mean = {desc['mean']}, Std = {desc['std']}<</COLOR>>\n"
            f"<<COLOR:text>>95% CI for mean: [{ci_lo:.4f}, {ci_hi:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>Normality p = {_pval_str(ad_p)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Mean = {desc['mean']}, Std = {desc['std']}",
            "body": f"N = {n}. Skewness = {desc['skewness']}, kurtosis = {desc['kurtosis']}. {'Approximately normal' if ad_p > 0.05 else 'Not normally distributed'}.",
            "next_steps": "Review histogram and probability plot for distribution shape.",
            "chart_guidance": "Four-panel: histogram, boxplot, probability plot, CI for mean.",
        },
        "guide_observation": f"Summary: mean={desc['mean']}, std={desc['std']}, n={n}.",
        "diagnostics": [],
    }


def forge_missing_data_analysis(df, config):
    """Missing data analysis."""
    cols = config.get("columns") or list(df.columns)
    n_rows = len(df)

    results = {}
    for col in cols:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        results[col] = {"missing": missing, "pct": round(missing / n_rows * 100, 1)}

    total_missing = sum(r["missing"] for r in results.values())
    n_complete = int(df[cols].dropna().shape[0]) if all(c in df.columns for c in cols) else 0

    return {
        "plots": [],
        "statistics": {
            "n_rows": n_rows,
            "n_complete_rows": n_complete,
            "total_missing": total_missing,
            "by_column": results,
        },
        "assumptions": {},
        "summary": f"<<COLOR:header>>Missing Data Analysis<</COLOR>>\n\n<<COLOR:text>>{total_missing} missing values across {len(results)} columns. {n_complete}/{n_rows} complete rows.<</COLOR>>",
        "narrative": {
            "verdict": f"{total_missing} missing values, {n_complete} complete rows",
            "body": f"Missing data analysis across {len(results)} columns.",
            "next_steps": "Consider imputation strategy based on missingness pattern (MCAR/MAR/MNAR).",
            "chart_guidance": "Heatmap shows missing data patterns.",
        },
        "guide_observation": f"Missing: {total_missing} values, {n_complete}/{n_rows} complete.",
        "diagnostics": [],
    }


def forge_outlier_analysis(df, config):
    """Outlier detection using IQR and Z-score methods."""
    data, col_name = _col(df, config, "column", "var1")
    n = len(data)
    threshold = float(config.get("threshold", 3.0))

    # IQR method
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    iqr_lower, iqr_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outliers = int(((data < iqr_lower) | (data > iqr_upper)).sum())

    # Z-score method
    z_scores = np.abs((data - data.mean()) / data.std(ddof=1)) if data.std(ddof=1) > 0 else np.zeros(n)
    z_outliers = int((z_scores > threshold).sum())

    return {
        "plots": [],
        "statistics": {
            "n": n,
            "iqr_outliers": iqr_outliers,
            "z_outliers": z_outliers,
            "iqr_lower": round(float(iqr_lower), 4),
            "iqr_upper": round(float(iqr_upper), 4),
            "z_threshold": threshold,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Outlier Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>IQR method: {iqr_outliers} outliers (bounds: [{iqr_lower:.4f}, {iqr_upper:.4f}])<</COLOR>>\n"
            f"<<COLOR:text>>Z-score method: {z_outliers} outliers (|Z| > {threshold})<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{iqr_outliers} IQR outliers, {z_outliers} Z-score outliers",
            "body": f"Of {n} observations, IQR flags {iqr_outliers} and Z-score flags {z_outliers} outliers.",
            "next_steps": "Investigate flagged points. Outliers may indicate special causes or data errors.",
            "chart_guidance": "Box plot with outliers highlighted.",
        },
        "guide_observation": f"Outliers: {iqr_outliers} (IQR), {z_outliers} (Z).",
        "diagnostics": [],
    }


def forge_duplicate_analysis(df, config):
    """Duplicate row detection."""
    cols = config.get("columns") or list(df.columns)
    subset = [c for c in cols if c in df.columns]
    n_rows = len(df)
    n_dupes = int(df[subset].duplicated().sum()) if subset else 0
    n_unique = n_rows - n_dupes

    return {
        "plots": [],
        "statistics": {
            "n_rows": n_rows,
            "n_duplicates": n_dupes,
            "n_unique": n_unique,
            "duplicate_pct": round(n_dupes / n_rows * 100, 1) if n_rows > 0 else 0,
        },
        "assumptions": {},
        "summary": f"<<COLOR:header>>Duplicate Analysis<</COLOR>>\n\n<<COLOR:text>>{n_dupes} duplicates ({n_dupes / n_rows * 100:.1f}%) of {n_rows} rows<</COLOR>>",
        "narrative": {
            "verdict": f"{n_dupes} duplicates found",
            "body": f"{n_dupes} duplicate rows detected across {len(subset)} columns.",
            "next_steps": "Review duplicates for data entry errors vs legitimate repeated observations.",
            "chart_guidance": "",
        },
        "guide_observation": f"Duplicates: {n_dupes}/{n_rows} ({n_dupes / n_rows * 100:.1f}%).",
        "diagnostics": [],
    }


def forge_copula(df, config):
    """Bivariate copula analysis."""
    c1, n1, c2, n2 = _col2(df, config)
    n = len(c1)

    # Empirical copula: rank-transform to uniform marginals
    u = sp_stats.rankdata(c1) / (n + 1)
    v = sp_stats.rankdata(c2) / (n + 1)

    # Kendall's tau and Spearman's rho
    tau, tau_p = sp_stats.kendalltau(c1, c2)
    rho, rho_p = sp_stats.spearmanr(c1, c2)

    # Tail dependence (empirical)
    q = 0.05
    lower_tail = np.mean((u <= q) & (v <= q)) / q if q > 0 else 0
    upper_tail = np.mean((u >= 1 - q) & (v >= 1 - q)) / q if q > 0 else 0

    return {
        "plots": [],
        "statistics": {
            "kendall_tau": round(float(tau), 4),
            "spearman_rho": round(float(rho), 4),
            "lower_tail_dependence": round(float(lower_tail), 4),
            "upper_tail_dependence": round(float(upper_tail), 4),
            "n": n,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Copula Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>Kendall's \u03c4 = {tau:.4f}, Spearman's \u03c1 = {rho:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Lower tail dep = {lower_tail:.4f}, Upper tail dep = {upper_tail:.4f}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Dependence: \u03c4 = {tau:.4f}, tail dep: lower={lower_tail:.4f}, upper={upper_tail:.4f}",
            "body": f"Copula analysis of {n1} and {n2}. Kendall \u03c4 = {tau:.4f}.",
            "next_steps": "Tail dependence > 0 indicates joint extreme behavior.",
            "chart_guidance": "Scatter plot in uniform margins shows copula structure.",
        },
        "guide_observation": f"Copula: \u03c4={tau:.4f}, tail dep L={lower_tail:.4f} U={upper_tail:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Dispatch
# =============================================================================

FORGE_EXPLORATORY_HANDLERS = {
    # Univariate / Transformations
    "poisson_1sample": forge_poisson_1sample,
    "poisson_2sample": forge_poisson_2sample,
    "box_cox": forge_box_cox,
    "johnson_transform": forge_johnson_transform,
    "grubbs_test": forge_grubbs_test,
    "distribution_fit": forge_distribution_fit,
    "mixture_model": forge_mixture_model,
    "tolerance_interval": forge_tolerance_interval,
    # Sequential
    "run_chart": forge_run_chart,
    "sprt": forge_sprt,
    "effect_size_calculator": forge_effect_size_calculator,
    # Data quality
    "data_profile": forge_data_profile,
    "auto_profile": forge_auto_profile,
    "graphical_summary": forge_graphical_summary,
    "missing_data_analysis": forge_missing_data_analysis,
    "outlier_analysis": forge_outlier_analysis,
    "duplicate_analysis": forge_duplicate_analysis,
    # Other
    "copula": forge_copula,
}
