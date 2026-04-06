"""Forge-backed statistical analysis handlers.

Replaces inline stats code with forgestat computation + ForgeViz charts.
Returns the same dict schema as the legacy handlers so dispatch.py and
standardize.py work without changes.

Object 271 — incremental migration. Each function here replaces one
analysis_id in the legacy stats/ package. When all IDs in a category
are ported, the legacy file can be deleted.

CR: pending
"""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _col(df, config, key, fallback_key=None):
    """Extract a clean numeric array from df using config key.

    Coerces to numeric, drops NaN/non-parseable values.
    """
    name = config.get(key) or config.get(fallback_key or key)
    if not name:
        nums = df.select_dtypes(include="number").columns
        if len(nums) == 0:
            raise ValueError("No numeric columns in dataset")
        name = nums[0]
    if name not in df.columns:
        raise ValueError(f"Column '{name}' not found")
    series = pd.to_numeric(df[name], errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Column '{name}' has no valid numeric values")
    return series.values, name


def _col2(df, config):
    """Extract two clean numeric arrays."""
    c1, n1 = _col(df, config, "column1", "var1")
    c2, n2 = _col(df, config, "column2", "var2")
    return c1, n1, c2, n2


def _alpha(config):
    return config.get("alpha", 0.05)


def _to_chart(spec):
    """Convert ForgeViz ChartSpec to dict for the result schema.

    Returns the native ChartSpec dict — the ForgeViz JS client renders
    this directly with full interactive utilities (toolbar, color picker,
    title/axis editing, etc).
    """
    return spec.to_dict()


def _assumption_to_dict(check):
    """Convert forgestat AssumptionCheck dataclass to dispatch-compatible dict."""
    passed = getattr(check, "passed", True)
    return {
        "level": "info" if passed else "warning",
        "title": getattr(check, "name", ""),
        "detail": getattr(check, "detail", ""),
        "pass": passed,
        "p": getattr(check, "p_value", None),
        "test": getattr(check, "test_name", ""),
        "suggestion": getattr(check, "suggestion", ""),
    }


def _assumptions_dict(checks):
    """Convert list of AssumptionCheck to the assumptions dict the frontend expects."""
    result = {}
    for c in checks:
        name = getattr(c, "name", "check")
        result[name] = {
            "pass": getattr(c, "passed", True),
            "p": getattr(c, "p_value", None),
            "test": getattr(c, "test_name", ""),
            "detail": getattr(c, "detail", ""),
        }
    return result


def _effect_label(d):
    """Classify Cohen's d magnitude."""
    if d is None or not math.isfinite(d):
        return "unknown"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _pval_str(p):
    """Format p-value for summary text."""
    if p is None:
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _education(analysis_type, analysis_id):
    """Fetch hand-written education content for this analysis."""
    try:
        from .education import get_education

        return get_education(analysis_type, analysis_id)
    except Exception:
        return None


def _bayesian_shadow_ttest1(data, mu=0.0):
    """Compute Bayesian shadow for one-sample t-test."""
    try:
        from forgestat.bayesian.tests import bayesian_ttest_one_sample

        r = bayesian_ttest_one_sample(data, mu=mu)
        return {
            "shadow_type": "ttest_1samp",
            "bf10": round(r.bf10, 4),
            "bf_label": r.bf_label,
            "credible_interval": {
                "param": "Cohen's d",
                "estimate": round(r.posterior_mean, 4),
                "ci_low": round(r.credible_interval[0], 4),
                "ci_high": round(r.credible_interval[1], 4),
                "level": r.ci_level,
            },
            "interpretation": (
                f"BF₁₀ = {r.bf10:.1f} — the data are {r.bf10:.1f}× more likely under H₁ than H₀ "
                f"({r.bf_label} evidence). "
                f"{r.ci_level * 100:.0f}% CrI for d: [{r.credible_interval[0]:.3f}, {r.credible_interval[1]:.3f}]"
                + (" (excludes zero)." if r.credible_interval[0] > 0 or r.credible_interval[1] < 0 else ".")
            ),
        }
    except Exception:
        return None


def _bayesian_shadow_ttest2(data1, data2):
    """Compute Bayesian shadow for two-sample t-test."""
    try:
        from forgestat.bayesian.tests import bayesian_ttest_two_sample

        r = bayesian_ttest_two_sample(data1, data2)
        return {
            "shadow_type": "ttest_2samp",
            "bf10": round(r.bf10, 4),
            "bf_label": r.bf_label,
            "credible_interval": {
                "param": "Cohen's d",
                "estimate": round(r.posterior_mean, 4),
                "ci_low": round(r.credible_interval[0], 4),
                "ci_high": round(r.credible_interval[1], 4),
                "level": r.ci_level,
            },
            "interpretation": (
                f"BF₁₀ = {r.bf10:.1f} ({r.bf_label} evidence). "
                f"{r.ci_level * 100:.0f}% CrI: [{r.credible_interval[0]:.3f}, {r.credible_interval[1]:.3f}]."
            ),
        }
    except Exception:
        return None


def _bayesian_shadow_from_t(t_stat, n):
    """Quick Bayesian shadow from t-statistic and sample size."""
    try:
        from forgestat.bayesian.tests import bayes_factor_shadow

        s = bayes_factor_shadow(t_stat, n)
        return {
            "shadow_type": "t_approx",
            "bf10": round(s.get("bf10", 0), 4),
            "bf_label": s.get("bf_label", ""),
            "interpretation": (f"BF₁₀ ≈ {s.get('bf10', 0):.1f} ({s.get('bf_label', '')} evidence)."),
        }
    except Exception:
        return None


def _rich_summary(title, sections):
    """Build a rich <<COLOR:>> formatted summary matching legacy output."""
    lines = [
        "<<COLOR:accent>>══════════════════════════════════════════════════════════════════════<</COLOR>>",
        f"<<COLOR:title>>{title}<</COLOR>>",
        "<<COLOR:accent>>══════════════════════════════════════════════════════════════════════<</COLOR>>",
        "",
    ]
    for heading, items in sections:
        lines.append(f"<<COLOR:accent>>── {heading} ──<</COLOR>>")
        for label, value in items:
            lines.append(f"  <<COLOR:highlight>>{label}:<</COLOR>> {value}")
        lines.append("")
    return "\n".join(lines)


def _practical_significance(effect_size, effect_label, significant):
    """Generate practical significance commentary."""
    if not significant:
        return "The result is not statistically significant. The observed difference could be due to chance."
    if effect_label in ("negligible", "small"):
        return (
            f"Statistically significant but the effect is <strong>{effect_label}</strong> "
            f"(d = {effect_size:.3f}). The difference may be too small to justify action — "
            f"consider whether the cost of intervention is worth it."
        )
    if effect_label == "medium":
        return (
            f"The effect is <strong>medium</strong> (d = {effect_size:.3f}), suggesting a "
            f"meaningful practical difference worth investigating further."
        )
    return (
        f"The effect is <strong>large</strong> (d = {effect_size:.3f}), indicating a "
        f"substantial practical difference that warrants action."
    )


# =============================================================================
# T-Tests
# =============================================================================


def forge_ttest(df, config):
    """One-sample t-test via forgestat."""
    from forgestat.parametric.ttest import one_sample
    from forgeviz.charts.distribution import histogram

    data, col_name = _col(df, config, "column", "var1")
    mu = config.get("test_value", config.get("mu", 0))
    alpha = _alpha(config)

    result = one_sample(data, mu=float(mu), alpha=alpha)

    # Chart: histogram with test value reference line
    spec = histogram(
        data=data.tolist(),
        bins=min(30, max(8, len(data) // 5)),
        title=f"Distribution of {col_name}",
        target=float(mu),
        show_normal=True,
    )
    chart = _to_chart(spec)

    el = _effect_label(result.effect_size)
    sig = "significant" if result.significant else "not significant"
    std = float(np.std(data, ddof=1))

    # Bayesian shadow
    bayes = _bayesian_shadow_ttest1(data, mu=float(mu))

    # Cross-check: nonparametric
    cross_check = None
    try:
        from forgestat.nonparametric.rank_tests import wilcoxon_signed_rank

        w = wilcoxon_signed_rank(data - float(mu))
        cross_check = {
            "level": "info",
            "title": f"Cross-check: Wilcoxon signed-rank {'agrees' if w.significant == result.significant else 'disagrees'} (p = {_pval_str(w.p_value)})",
            "detail": "Both parametric and non-parametric tests reach the same conclusion."
            if w.significant == result.significant
            else "Parametric and nonparametric tests disagree — check normality assumption.",
        }
    except Exception:
        pass

    diagnostics = [_assumption_to_dict(a) for a in result.assumptions]
    if cross_check:
        diagnostics.append(cross_check)
    if result.effect_size is not None:
        diagnostics.append(
            {
                "level": "info",
                "title": f"{'Large' if el == 'large' else el.title()} practical effect (Cohen's d = {result.effect_size:.2f})",
                "detail": _practical_significance(result.effect_size, el, result.significant),
            }
        )

    return {
        "plots": [chart],
        "statistics": {
            f"mean({col_name})": round(result.mean1, 4),
            f"std({col_name})": round(std, 4),
            "n": result.n1,
            "t_statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "df": result.df,
            "cohens_d": round(result.effect_size, 4) if result.effect_size is not None else None,
            "effect_size_label": el,
            f"ci_lower({col_name})": round(result.ci_lower, 4),
            f"ci_upper({col_name})": round(result.ci_upper, 4),
            "se": round(result.se, 4),
        },
        "assumptions": _assumptions_dict(result.assumptions),
        "summary": _rich_summary(
            "ONE-SAMPLE T-TEST",
            [
                ("Variable", [(col_name, f"n = {result.n1}")]),
                ("Hypothesized mean", [("μ₀", str(mu))]),
                (
                    "Sample Statistics",
                    [
                        ("Mean", f"{result.mean1:.4f}"),
                        ("Std Dev", f"{std:.4f}"),
                        ("SE Mean", f"{result.se:.4f}"),
                    ],
                ),
                (
                    "Test Results",
                    [
                        ("t-statistic", f"{result.statistic:.4f}"),
                        ("df", f"{result.df:.0f}"),
                        ("p-value", _pval_str(result.p_value)),
                        ("Cohen's d", f"{result.effect_size:.4f} ({el})"),
                        (f"{int((1 - alpha) * 100)}% CI", f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"),
                    ],
                ),
                (
                    "Conclusion",
                    [
                        ("Decision", f"{'Reject' if result.significant else 'Fail to reject'} H₀ at α = {alpha}"),
                        (
                            "Practical",
                            _practical_significance(result.effect_size, el, result.significant)
                            .replace("<strong>", "")
                            .replace("</strong>", ""),
                        ),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": (
                f"The mean of {col_name} is significantly {'higher' if result.mean_diff > 0 else 'lower'} than {mu}"
                if result.significant
                else f"No significant difference between {col_name} and {mu}"
            ),
            "body": (
                f"The sample mean ({result.mean1:.4f}) differs from the hypothesized value by "
                f"{abs(result.mean_diff):.4f} units &mdash; a <strong>{el}</strong> effect "
                f"(Cohen's d = {result.effect_size:.2f}). "
                + _practical_significance(result.effect_size, el, result.significant)
                + (
                    f" The {int((1 - alpha) * 100)}% CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}] "
                    f"{'excludes' if result.significant else 'includes'} the hypothesized value."
                )
            ),
            "next_steps": (
                "Investigate what is causing the shift from the target value."
                if result.significant and el in ("medium", "large")
                else "Consider the practical significance of this difference. "
                "A power analysis can determine if the sample size was adequate."
                if result.significant
                else "The null hypothesis cannot be rejected. Consider whether the sample size "
                "provides sufficient power to detect a meaningful effect."
            ),
            "chart_guidance": (
                f"The histogram shows the distribution of {col_name}. "
                f"The green line marks the target value ({mu}). "
                f"The normal overlay helps assess distributional shape."
            ),
        },
        "guide_observation": (
            f"One-sample t-test: mean={result.mean1:.3f} vs μ₀={mu}, "
            f"p={_pval_str(result.p_value)}, Cohen's d={result.effect_size:.3f} ({el}) "
            f"{'Statistically significant but ' + el + ' effect.' if result.significant and el in ('negligible', 'small') else sig.capitalize() + '.'}"
        ),
        "education": _education("stats", "ttest"),
        "bayesian_shadow": bayes,
        "power_explorer": {
            "test_type": "ttest",
            "observed_effect": abs(result.mean_diff),
            "observed_std": std,
            "observed_n": result.n1,
            "alpha": alpha,
            "cohens_d": abs(result.effect_size) if result.effect_size else 0,
        },
        "diagnostics": diagnostics,
    }


def forge_ttest2(df, config):
    """Two-sample t-test via forgestat."""
    from forgestat.parametric.ttest import two_sample
    from forgeviz.charts.distribution import box_plot
    from forgeviz.charts.statistical import individual_value_plot

    data1, n1, data2, n2 = _col2(df, config)
    alpha = _alpha(config)
    equal_var = config.get("equal_var", False)

    result = two_sample(data1, data2, equal_var=equal_var, alpha=alpha)

    charts = [
        _to_chart(box_plot(datasets={n1: data1.tolist(), n2: data2.tolist()}, title=f"{n1} vs {n2}")),
        _to_chart(individual_value_plot(groups={n1: data1.tolist(), n2: data2.tolist()}, title="Individual Values")),
    ]

    el = _effect_label(result.effect_size)
    sig = "significant" if result.significant else "not significant"
    method = "Welch's" if not equal_var else "Pooled"

    bayes = _bayesian_shadow_ttest2(data1, data2)

    # Cross-check: Mann-Whitney
    cross_check = None
    try:
        from forgestat.nonparametric.rank_tests import mann_whitney

        mw = mann_whitney(data1, data2)
        cross_check = {
            "level": "info",
            "title": f"Cross-check: Mann-Whitney {'agrees' if mw.significant == result.significant else 'disagrees'} (p = {_pval_str(mw.p_value)})",
            "detail": "Both parametric and non-parametric tests reach the same conclusion."
            if mw.significant == result.significant
            else "Tests disagree — check normality and equal variance assumptions.",
        }
    except Exception:
        pass

    diagnostics = [_assumption_to_dict(a) for a in result.assumptions]
    if cross_check:
        diagnostics.append(cross_check)
    diagnostics.append(
        {
            "level": "info",
            "title": f"{'Large' if el == 'large' else el.title()} practical effect (Cohen's d = {result.effect_size:.2f})",
            "detail": _practical_significance(result.effect_size, el, result.significant),
        }
    )

    return {
        "plots": charts,
        "statistics": {
            f"mean({n1})": round(result.mean1, 4),
            f"std({n1})": round(float(np.std(data1, ddof=1)), 4),
            f"mean({n2})": round(result.mean2, 4),
            f"std({n2})": round(float(np.std(data2, ddof=1)), 4),
            "difference": round(result.mean_diff, 4),
            f"n({n1})": result.n1,
            f"n({n2})": result.n2,
            "t_statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "df": round(result.df, 2),
            "cohens_d": round(result.effect_size, 4) if result.effect_size is not None else None,
            "effect_size_label": el,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
            "method": result.method,
        },
        "assumptions": _assumptions_dict(result.assumptions),
        "summary": _rich_summary(
            f"{method.upper()} TWO-SAMPLE T-TEST",
            [
                (
                    "Groups",
                    [
                        (n1, f"n = {result.n1}, mean = {result.mean1:.4f}, std = {float(np.std(data1, ddof=1)):.4f}"),
                        (n2, f"n = {result.n2}, mean = {result.mean2:.4f}, std = {float(np.std(data2, ddof=1)):.4f}"),
                    ],
                ),
                (
                    "Test Results",
                    [
                        ("Difference", f"{result.mean_diff:.4f}"),
                        ("t-statistic", f"{result.statistic:.4f}"),
                        ("df", f"{result.df:.1f}"),
                        ("p-value", _pval_str(result.p_value)),
                        ("Cohen's d", f"{result.effect_size:.4f} ({el})"),
                        (f"{int((1 - alpha) * 100)}% CI", f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": (
                f"The mean of {n1} is significantly {'higher' if result.mean_diff > 0 else 'lower'} than {n2}"
                if result.significant
                else f"No significant difference between {n1} and {n2}"
            ),
            "body": (
                f"{method} t-test comparing {n1} (mean={result.mean1:.4f}, n={result.n1}) and "
                f"{n2} (mean={result.mean2:.4f}, n={result.n2}). Difference = {result.mean_diff:.4f}. "
                f"t({result.df:.1f}) = {result.statistic:.3f}, p = {_pval_str(result.p_value)}. "
                + _practical_significance(result.effect_size, el, result.significant)
            ),
            "next_steps": (
                "Investigate the source of the difference. Run post-hoc analysis if comparing multiple groups."
                if result.significant
                else "Groups do not differ significantly. Check power if a meaningful difference was expected."
            ),
            "chart_guidance": f"Box plots show distributions of {n1} and {n2}. Individual value plot shows every observation.",
        },
        "guide_observation": (
            f"Two-sample t-test: {n1} vs {n2}, diff={result.mean_diff:.3f}, "
            f"p={_pval_str(result.p_value)}, d={result.effect_size:.3f} ({el}). {sig.capitalize()}."
        ),
        "education": _education("stats", "ttest2"),
        "bayesian_shadow": bayes,
        "diagnostics": diagnostics,
    }


def forge_paired_t(df, config):
    """Paired t-test via forgestat."""
    from forgestat.parametric.ttest import paired
    from forgeviz.charts.distribution import histogram

    data1, n1, data2, n2 = _col2(df, config)
    alpha = _alpha(config)

    result = paired(data1, data2, alpha=alpha)
    diffs = (data1 - data2).tolist()

    spec = histogram(
        data=diffs,
        bins=min(25, max(6, len(diffs) // 4)),
        title=f"Paired Differences ({n1} − {n2})",
        target=0.0,
        show_normal=True,
    )
    chart = _to_chart(spec)

    el = _effect_label(result.effect_size)
    sig = "significant" if result.significant else "not significant"

    return {
        "plots": [chart],
        "statistics": {
            f"mean({n1})": round(result.mean1, 4),
            f"mean({n2})": round(result.mean2, 4),
            "mean_diff": round(result.mean_diff, 4),
            "n_pairs": result.n1,
            "t_statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "df": result.df,
            "cohens_d": round(result.effect_size, 4) if result.effect_size is not None else None,
            "effect_size_label": el,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
        },
        "assumptions": _assumptions_dict(result.assumptions),
        "summary": (
            f"Paired t-test: mean difference = {result.mean_diff:.3f}, "
            f"t({result.df:.0f})={result.statistic:.3f}, p={_pval_str(result.p_value)}, "
            f"d={result.effect_size:.3f} ({el}). {sig.capitalize()} at α={alpha}."
        ),
        "narrative": {
            "verdict": f"Paired difference is {sig}" if result.significant else "No significant paired difference",
            "body": (
                f"Paired comparison of {n1} and {n2} (n={result.n1} pairs). "
                f"Mean difference = {result.mean_diff:.3f}. "
                f"t({result.df:.0f}) = {result.statistic:.3f}, p = {_pval_str(result.p_value)}. "
                f"Cohen's d = {result.effect_size:.3f} ({el})."
            ),
            "next_steps": "Evaluate practical significance of the mean difference.",
            "chart_guidance": (
                "Histogram shows the distribution of paired differences. The green line marks zero (no difference)."
            ),
        },
        "guide_observation": (
            f"Paired t-test: diff={result.mean_diff:.3f}, "
            f"p={_pval_str(result.p_value)}, d={result.effect_size:.3f}. {sig.capitalize()}."
        ),
        "diagnostics": [_assumption_to_dict(a) for a in result.assumptions],
    }


# =============================================================================
# ANOVA
# =============================================================================


def forge_anova(df, config):
    """One-way ANOVA via forgestat."""
    from forgestat.parametric.anova import one_way_from_dict
    from forgeviz.charts.distribution import box_plot
    from forgeviz.charts.statistical import interval_plot

    response = config.get("response") or config.get("column")
    factor = config.get("factor")

    if not response or not factor:
        raise ValueError("ANOVA requires 'response' and 'factor' columns")
    if response not in df.columns or factor not in df.columns:
        raise ValueError(f"Columns not found: {response}, {factor}")

    groups = {}
    for name, group_df in df.groupby(factor):
        vals = pd.to_numeric(group_df[response], errors="coerce").dropna().values
        if len(vals) > 0:
            groups[str(name)] = vals.tolist()

    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    alpha = _alpha(config)
    result = one_way_from_dict(groups, alpha=alpha)

    # Charts: box plot + interval plot
    spec_box = box_plot(datasets=groups, title=f"{response} by {factor}")
    spec_int = interval_plot(groups=groups, confidence=1 - alpha, title=f"Means with {int((1 - alpha) * 100)}% CI")

    charts = [_to_chart(spec_box), _to_chart(spec_int)]

    # Effect size classification
    eta_sq = result.effect_size if hasattr(result, "effect_size") else None
    if eta_sq is None:
        eta_sq = getattr(result, "eta_squared", None)
        if eta_sq is None and result.ss_total > 0:
            eta_sq = result.ss_between / result.ss_total

    if eta_sq is not None and math.isfinite(eta_sq):
        if eta_sq < 0.01:
            eta_label = "negligible"
        elif eta_sq < 0.06:
            eta_label = "small"
        elif eta_sq < 0.14:
            eta_label = "medium"
        else:
            eta_label = "large"
    else:
        eta_label = "unknown"

    sig = "significant" if result.significant else "not significant"
    n_groups = len(groups)
    n_total = sum(len(v) for v in groups.values())

    stats = {
        "f_statistic": round(result.statistic, 4),
        "p_value": result.p_value,
        "df_between": result.df_between,
        "df_within": result.df_within,
        "ss_between": round(result.ss_between, 4),
        "ss_within": round(result.ss_within, 4),
        "ms_between": round(result.ms_between, 4),
        "ms_within": round(result.ms_within, 4),
        "n_groups": n_groups,
        "n_total": n_total,
    }
    if eta_sq is not None:
        stats["eta_squared"] = round(eta_sq, 4)
        stats["effect_size_label"] = eta_label
    omega_sq = getattr(result, "omega_squared", None)
    if omega_sq is not None:
        stats["omega_squared"] = round(omega_sq, 4)

    # Bayesian shadow from F → t approximation
    bayes = _bayesian_shadow_from_t(math.sqrt(max(result.statistic, 0)), n_total) if result.statistic else None

    # Group means for narrative
    means_str = ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in groups.items())

    diagnostics = [_assumption_to_dict(a) for a in result.assumptions] if hasattr(result, "assumptions") else []
    diagnostics.append(
        {
            "level": "info",
            "title": f"Factor explains {eta_sq * 100:.1f}% of variation ({eta_label})",
            "detail": f"η² = {eta_sq:.4f}, ω² = {omega_sq:.4f}" if omega_sq else f"η² = {eta_sq:.4f}",
        }
    )

    return {
        "plots": charts,
        "statistics": stats,
        "assumptions": _assumptions_dict(result.assumptions) if hasattr(result, "assumptions") else {},
        "summary": _rich_summary(
            "ONE-WAY ANOVA",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Factor", f"{factor} ({n_groups} levels)"),
                        ("Total N", str(n_total)),
                    ],
                ),
                ("Group Means", [(k, f"{np.mean(v):.4f} (n={len(v)})") for k, v in groups.items()]),
                (
                    "ANOVA Table",
                    [
                        ("F-statistic", f"{result.statistic:.4f}"),
                        ("df", f"({result.df_between:.0f}, {result.df_within:.0f})"),
                        ("p-value", _pval_str(result.p_value)),
                        ("η²", f"{eta_sq:.4f} ({eta_label})"),
                    ],
                ),
                (
                    "Conclusion",
                    [
                        ("Decision", f"{'Reject' if result.significant else 'Fail to reject'} H₀ at α = {alpha}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": (
                f"{factor} has a significant effect on {response}"
                if result.significant
                else f"No significant effect of {factor} on {response}"
            ),
            "body": (
                f"One-way ANOVA with {n_groups} groups ({means_str}). "
                f"F({result.df_between:.0f},{result.df_within:.0f}) = {result.statistic:.3f}, "
                f"p = {_pval_str(result.p_value)}. "
                f"The factor explains <strong>{eta_sq * 100:.1f}%</strong> of the variation "
                f"&mdash; a <strong>{eta_label}</strong> effect."
            ),
            "next_steps": (
                "Run <strong>Tukey HSD</strong> or <strong>Games-Howell</strong> post-hoc tests "
                "to identify which specific groups differ."
                if result.significant
                else "No significant differences detected. Consider increasing sample size or "
                "checking if the effect is too small to detect."
            ),
            "chart_guidance": (
                f"Box plot shows distributions by {factor}. "
                f"Interval plot shows group means with {int((1 - alpha) * 100)}% confidence intervals — "
                f"non-overlapping intervals suggest significant differences."
            ),
        },
        "guide_observation": (
            f"One-way ANOVA: F={result.statistic:.3f}, p={_pval_str(result.p_value)}, "
            f"η²={eta_sq:.3f} ({eta_label}). {sig.capitalize()}."
        ),
        "education": _education("stats", "anova"),
        "bayesian_shadow": bayes,
        "power_explorer": {
            "test_type": "anova",
            "observed_effect": math.sqrt(eta_sq / (1 - eta_sq)) if eta_sq and eta_sq < 1 else 0,
            "observed_n": n_total,
            "alpha": alpha,
            "n_groups": n_groups,
        },
        "diagnostics": diagnostics,
    }


# =============================================================================
# Chi-Square
# =============================================================================


def forge_chi2(df, config):
    """Chi-square test of independence via forgestat."""
    from forgestat.parametric.chi_square import chi_square_independence
    from forgeviz.charts.statistical import heatmap

    col1 = config.get("column1") or config.get("var1")
    col2 = config.get("column2") or config.get("var2")

    if not col1 or not col2:
        # Auto-pick first two categorical columns
        cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cats) < 2:
            raise ValueError("Chi-square requires two categorical columns")
        col1, col2 = cats[0], cats[1]

    ct = pd.crosstab(df[col1], df[col2])
    observed = ct.values.tolist()
    row_labels = [str(x) for x in ct.index.tolist()]
    col_labels = [str(x) for x in ct.columns.tolist()]

    alpha = _alpha(config)
    result = chi_square_independence(observed, row_labels=row_labels, col_labels=col_labels, alpha=alpha)

    spec = heatmap(
        x_labels=col_labels,
        y_labels=row_labels,
        z_matrix=observed,
        title=f"{col1} vs {col2} (Observed Frequencies)",
    )
    chart = _to_chart(spec)

    sig = "significant" if result.significant else "not significant"
    cv = result.cramers_v if hasattr(result, "cramers_v") else None

    stats_dict = {
        "chi2_statistic": round(result.statistic, 4),
        "p_value": result.p_value,
        "df": result.df,
    }
    if cv is not None:
        stats_dict["cramers_v"] = round(cv, 4)

    return {
        "plots": [chart],
        "statistics": stats_dict,
        "assumptions": {},
        "summary": (
            f"Chi-square test of independence: χ²({result.df})={result.statistic:.3f}, "
            f"p={_pval_str(result.p_value)}"
            + (f", Cramér's V={cv:.3f}" if cv else "")
            + f". Association is {sig} at α={alpha}."
        ),
        "narrative": {
            "verdict": f"{col1} and {col2} are {'associated' if result.significant else 'independent'}",
            "body": (
                f"Chi-square test of independence between {col1} and {col2}. "
                f"χ²({result.df}) = {result.statistic:.3f}, p = {_pval_str(result.p_value)}."
                + (f" Cramér's V = {cv:.3f}." if cv else "")
            ),
            "next_steps": (
                "Examine the heatmap to identify which cells contribute most to the chi-square statistic."
                if result.significant
                else "No significant association detected."
            ),
            "chart_guidance": "Heatmap shows observed frequencies. Darker cells indicate higher counts.",
        },
        "guide_observation": (
            f"Chi-square: χ²={result.statistic:.3f}, p={_pval_str(result.p_value)}"
            + (f", V={cv:.3f}" if cv else "")
            + f". {sig.capitalize()}."
        ),
        "diagnostics": [],
    }


# =============================================================================
# Correlation
# =============================================================================


def forge_correlation(df, config):
    """Correlation analysis via forgestat."""
    from forgestat.parametric.correlation import correlation
    from forgeviz.charts.statistical import heatmap

    method = config.get("method", "pearson")
    cols = config.get("columns")
    if not cols:
        cols = df.select_dtypes(include="number").columns.tolist()
    if len(cols) < 2:
        raise ValueError("Correlation requires at least 2 numeric columns")

    data_dict = {c: pd.to_numeric(df[c], errors="coerce").dropna().values.tolist() for c in cols}
    alpha = _alpha(config)

    result = correlation(data_dict, method=method, alpha=alpha)

    # Build correlation matrix for heatmap
    matrix_vals = []
    for r in cols:
        row = []
        for c in cols:
            row.append(round(result.matrix.get(r, {}).get(c, 0), 3))
        matrix_vals.append(row)

    spec = heatmap(
        x_labels=cols,
        y_labels=cols,
        z_matrix=matrix_vals,
        title=f"{method.title()} Correlation Matrix",
    )
    chart = _to_chart(spec)

    stats_dict = {}
    for pair in result.pairs[:10]:  # top 10 pairs
        key = f"r({pair.var1},{pair.var2})"
        stats_dict[key] = round(pair.r, 4)
    if result.pairs:
        top = result.pairs[0]
        stats_dict["strongest_r"] = round(top.r, 4)
        stats_dict["strongest_pair"] = f"{top.var1} × {top.var2}"
        stats_dict["strongest_p"] = top.p_value

    return {
        "plots": [chart],
        "statistics": stats_dict,
        "assumptions": _assumptions_dict(result.assumptions) if hasattr(result, "assumptions") else {},
        "summary": (
            f"{method.title()} correlation matrix ({len(cols)} variables). "
            + (
                f"Strongest: r({result.pairs[0].var1},{result.pairs[0].var2})="
                f"{result.pairs[0].r:.3f}, p={_pval_str(result.pairs[0].p_value)}."
                if result.pairs
                else "No pairs computed."
            )
        ),
        "narrative": {
            "verdict": (
                f"Strongest correlation: {result.pairs[0].var1} × {result.pairs[0].var2} (r={result.pairs[0].r:.3f})"
                if result.pairs
                else "No significant correlations"
            ),
            "body": (
                f"Computed {method} correlations for {len(cols)} variables. "
                + (
                    f"Top pair: {result.pairs[0].var1} and {result.pairs[0].var2} "
                    f"with r={result.pairs[0].r:.3f} (p={_pval_str(result.pairs[0].p_value)}), "
                    f"R²={result.pairs[0].r_squared:.3f}."
                    if result.pairs
                    else ""
                )
            ),
            "next_steps": "Consider regression analysis for the strongest relationships.",
            "chart_guidance": "Heatmap shows pairwise correlations. Green = positive, red = negative.",
        },
        "guide_observation": (
            f"{method.title()} correlation ({len(cols)} vars). "
            + (f"Strongest: r={result.pairs[0].r:.3f}." if result.pairs else "")
        ),
        "diagnostics": [],
    }


# =============================================================================
# Descriptive Statistics
# =============================================================================


def forge_descriptive(df, config):
    """Descriptive statistics via forgestat."""
    from forgestat.exploratory.univariate import describe
    from forgeviz.charts.distribution import box_plot, histogram

    data, col_name = _col(df, config, "column", "var1")
    result = describe(data)

    spec_hist = histogram(
        data=data.tolist(),
        bins=min(30, max(8, len(data) // 5)),
        title=f"Distribution of {col_name}",
        show_normal=True,
    )
    spec_box = box_plot(
        datasets={col_name: data.tolist()},
        title=f"Box Plot of {col_name}",
    )

    stats_dict = {
        "n": result.n,
        "n_missing": result.n_missing,
        "mean": round(result.mean, 4),
        "se_mean": round(result.se_mean, 4),
        "std": round(result.std, 4),
        "variance": round(result.variance, 4),
        "median": round(result.median, 4),
        "q1": round(result.q1, 4),
        "q3": round(result.q3, 4),
        "iqr": round(result.iqr, 4),
        "min": round(result.min, 4),
        "max": round(result.max, 4),
        "range": round(result.range, 4),
        "skewness": round(result.skewness, 4),
        "kurtosis": round(result.kurtosis, 4),
        "cv_pct": round(result.cv, 2),
        "n_outliers": result.n_outliers,
    }

    return {
        "plots": [_to_chart(spec_hist), _to_chart(spec_box)],
        "statistics": stats_dict,
        "assumptions": {},
        "summary": _rich_summary(
            "DESCRIPTIVE STATISTICS",
            [
                ("Variable", [(col_name, f"n = {result.n}")]),
                (
                    "Central Tendency",
                    [
                        ("Mean", f"{result.mean:.4f}"),
                        ("Median", f"{result.median:.4f}"),
                        ("SE Mean", f"{result.se_mean:.4f}"),
                    ],
                ),
                (
                    "Dispersion",
                    [
                        ("Std Dev", f"{result.std:.4f}"),
                        ("Variance", f"{result.variance:.4f}"),
                        ("IQR", f"{result.iqr:.4f}"),
                        ("Range", f"[{result.min:.4f}, {result.max:.4f}]"),
                    ],
                ),
                (
                    "Shape",
                    [
                        ("Skewness", f"{result.skewness:.4f}"),
                        ("Kurtosis", f"{result.kurtosis:.4f}"),
                        ("Description", result.shape_description),
                    ],
                ),
                ("Outliers", [(f"{result.n_outliers} detected", "via IQR method")]),
            ],
        ),
        "narrative": {
            "verdict": f"{col_name}: {result.shape_description}",
            "body": (
                f"n={result.n}, mean={result.mean:.3f} ± {result.std:.3f}, "
                f"median={result.median:.3f}. IQR=[{result.q1:.3f}, {result.q3:.3f}]. "
                f"Skewness={result.skewness:.3f}, kurtosis={result.kurtosis:.3f}."
            ),
            "next_steps": (
                "Data appears non-normal — consider nonparametric tests."
                if abs(result.skewness) > 1
                else "Distribution is approximately symmetric."
            ),
            "chart_guidance": "Histogram with normal overlay shows distribution shape. Box plot shows quartiles and outliers.",
        },
        "guide_observation": f"Descriptive: {col_name}, n={result.n}, mean={result.mean:.3f}, std={result.std:.3f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nonparametric
# =============================================================================


def _forge_rank_test(df, config, func_name, test_label, two_col=True):
    """Generic wrapper for forgestat rank tests (mann_whitney, wilcoxon)."""
    from forgestat.nonparametric import rank_tests
    from forgeviz.charts.distribution import box_plot

    func = getattr(rank_tests, func_name)

    if two_col:
        data1, n1, data2, n2 = _col2(df, config)
        result = func(data1, data2, alpha=_alpha(config))
        spec = box_plot(datasets={n1: data1.tolist(), n2: data2.tolist()}, title=f"{n1} vs {n2}")
        label = f"{n1} vs {n2}"
    else:
        data, col_name = _col(df, config, "column", "var1")
        result = func(data, alpha=_alpha(config))
        from forgeviz.charts.distribution import histogram

        spec = histogram(data=data.tolist(), bins=min(25, max(6, len(data) // 4)), title=f"{col_name}")
        label = col_name

    sig = "significant" if result.significant else "not significant"
    el = getattr(result, "effect_label", "")

    stats = {
        "statistic": round(result.statistic, 4),
        "p_value": result.p_value,
    }
    if hasattr(result, "effect_size") and result.effect_size is not None:
        stats["effect_size"] = round(result.effect_size, 4)
        stats["effect_size_type"] = getattr(result, "effect_size_type", "")
        stats["effect_size_label"] = el
    if hasattr(result, "median1"):
        stats["median1"] = round(result.median1, 4)
    if hasattr(result, "median2") and result.median2 is not None:
        stats["median2"] = round(result.median2, 4)
    if hasattr(result, "n1"):
        stats["n1"] = result.n1
    if hasattr(result, "n2") and result.n2 is not None:
        stats["n2"] = result.n2

    return {
        "plots": [_to_chart(spec)],
        "statistics": stats,
        "assumptions": {},
        "summary": _rich_summary(
            test_label.upper(),
            [
                ("Test", [("Method", test_label), ("Groups" if two_col else "Variable", label)]),
                (
                    "Results",
                    [
                        ("Statistic", f"{result.statistic:.4f}"),
                        ("p-value", _pval_str(result.p_value)),
                    ]
                    + (
                        [("Effect", f"{result.effect_size:.4f} ({el})")]
                        if hasattr(result, "effect_size") and result.effect_size is not None
                        else []
                    ),
                ),
                ("Conclusion", [("Decision", sig.capitalize())]),
            ],
        ),
        "narrative": {
            "verdict": f"{test_label}: {sig}",
            "body": f"{test_label} on {label}. Test statistic = {result.statistic:.3f}, p = {_pval_str(result.p_value)}"
            + (f". Effect size: {el} ({result.effect_size:.3f})" if el else "")
            + ".",
            "next_steps": "Review effect size for practical significance."
            if result.significant
            else "No significant difference detected.",
            "chart_guidance": "Distribution comparison shown.",
        },
        "guide_observation": f"{test_label}: p={_pval_str(result.p_value)}. {sig.capitalize()}.",
        "diagnostics": [],
    }


def forge_mann_whitney(df, config):
    return _forge_rank_test(df, config, "mann_whitney", "Mann-Whitney U")


def forge_wilcoxon(df, config):
    return _forge_rank_test(df, config, "wilcoxon_signed_rank", "Wilcoxon Signed-Rank")


def _forge_k_group_test(df, config, func_name, test_label):
    """Generic wrapper for k-group nonparametric tests (kruskal, friedman, mood)."""
    from forgestat.nonparametric import rank_tests
    from forgeviz.charts.distribution import box_plot

    func = getattr(rank_tests, func_name)
    response = config.get("response") or config.get("column")
    factor = config.get("factor")

    if not response or not factor:
        raise ValueError(f"{test_label} requires 'response' and 'factor'")

    groups = {}
    for name, g in df.groupby(factor):
        vals = pd.to_numeric(g[response], errors="coerce").dropna().values
        if len(vals) > 0:
            groups[str(name)] = vals.tolist()

    if len(groups) < 2:
        raise ValueError(f"{test_label} requires at least 2 groups")

    arrays = [np.array(v) for v in groups.values()]
    labels = list(groups.keys())
    result = func(*arrays, labels=labels, alpha=_alpha(config))

    spec = box_plot(datasets=groups, title=f"{response} by {factor}")
    sig = "significant" if result.significant else "not significant"

    means_str = ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in groups.items())

    return {
        "plots": [_to_chart(spec)],
        "statistics": {
            "statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "df": getattr(result, "df", None),
            "n_groups": len(groups),
        },
        "assumptions": {},
        "summary": _rich_summary(
            test_label.upper(),
            [
                ("Design", [("Response", response), ("Factor", f"{factor} ({len(groups)} levels)")]),
                ("Group Medians", [(k, f"{np.median(v):.4f} (n={len(v)})") for k, v in groups.items()]),
                ("Results", [("Statistic", f"{result.statistic:.4f}"), ("p-value", _pval_str(result.p_value))]),
                ("Conclusion", [("Decision", f"{'Reject' if result.significant else 'Fail to reject'} H₀")]),
            ],
        ),
        "narrative": {
            "verdict": f"{factor} effect is {sig} ({test_label})",
            "body": f"{test_label} with {len(groups)} groups ({means_str}). Statistic = {result.statistic:.3f}, p = {_pval_str(result.p_value)}.",
            "next_steps": "Run post-hoc pairwise comparisons (Dunn's test) to identify differing groups."
            if result.significant
            else "No significant differences detected.",
            "chart_guidance": f"Box plots show distributions of {response} by {factor}.",
        },
        "guide_observation": f"{test_label}: p={_pval_str(result.p_value)}. {sig.capitalize()}.",
        "diagnostics": [],
    }


def forge_kruskal(df, config):
    return _forge_k_group_test(df, config, "kruskal_wallis", "Kruskal-Wallis")


def forge_friedman(df, config):
    return _forge_k_group_test(df, config, "friedman", "Friedman")


def forge_mood_median(df, config):
    return _forge_k_group_test(df, config, "mood_median", "Mood's Median")


def forge_spearman(df, config):
    """Spearman rank correlation via forgestat."""
    config = {**config, "method": "spearman"}
    return forge_correlation(df, config)


# =============================================================================
# Proportions
# =============================================================================


def forge_prop_1sample(df, config):
    """One-proportion test via forgestat."""
    from forgestat.parametric.proportion import one_proportion
    from forgeviz.charts.generic import bar

    successes = int(config.get("successes", 0))
    n = int(config.get("n", config.get("trials", 0)))
    p0 = float(config.get("p0", config.get("test_value", 0.5)))

    if n <= 0:
        raise ValueError("Sample size n must be positive")

    result = one_proportion(successes, n, p0=p0, alpha=_alpha(config))
    p_hat = result.p_hat

    spec = bar(
        categories=["Observed", "Expected"],
        values=[p_hat, p0],
        title="Proportion: Observed vs Expected",
    )

    sig = "significant" if result.significant else "not significant"

    return {
        "plots": [_to_chart(spec)],
        "statistics": {
            "p_hat": round(p_hat, 4),
            "p0": p0,
            "n": n,
            "successes": successes,
            "z_statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
        },
        "assumptions": {},
        "summary": f"1-proportion test: p̂={p_hat:.4f} vs p₀={p0}, z={result.statistic:.3f}, p={_pval_str(result.p_value)}. {sig.capitalize()}.",
        "narrative": {
            "verdict": f"Proportion is {sig}ly different from {p0}",
            "body": f"Observed proportion {p_hat:.4f} ({successes}/{n}) tested against {p0}. z = {result.statistic:.3f}, p = {_pval_str(result.p_value)}.",
            "next_steps": "Evaluate practical significance."
            if result.significant
            else "No significant departure from expected proportion.",
            "chart_guidance": "Bar chart compares observed vs expected proportion.",
        },
        "guide_observation": f"1-prop test: p̂={p_hat:.4f} vs {p0}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


def forge_prop_2sample(df, config):
    """Two-proportion test via forgestat."""
    from forgestat.parametric.proportion import two_proportions
    from forgeviz.charts.generic import grouped_bar

    s1 = int(config.get("successes1", 0))
    n1 = int(config.get("n1", 0))
    s2 = int(config.get("successes2", 0))
    n2 = int(config.get("n2", 0))

    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive")

    result = two_proportions(s1, n1, s2, n2, alpha=_alpha(config))
    sig = "significant" if result.significant else "not significant"

    spec = grouped_bar(
        categories=["Group 1", "Group 2"],
        series={"Proportion": [result.p_hat, result.p_hat2 or 0]},
        title="Two-Proportion Comparison",
    )

    return {
        "plots": [_to_chart(spec)],
        "statistics": {
            "p_hat1": round(result.p_hat, 4),
            "p_hat2": round(result.p_hat2, 4) if result.p_hat2 else None,
            "difference": round(result.p_diff, 4) if result.p_diff else None,
            "z_statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
        },
        "assumptions": {},
        "summary": f"2-proportion test: p₁={result.p_hat:.4f} vs p₂={result.p_hat2:.4f}, z={result.statistic:.3f}, p={_pval_str(result.p_value)}. {sig.capitalize()}.",
        "narrative": {
            "verdict": f"Proportions {'differ' if result.significant else 'do not differ'} significantly",
            "body": f"Group 1: {s1}/{n1} = {result.p_hat:.4f}. Group 2: {s2}/{n2} = {result.p_hat2:.4f}. Difference = {result.p_diff:.4f}.",
            "next_steps": "Investigate the source of the difference."
            if result.significant
            else "No significant difference.",
            "chart_guidance": "Bar chart compares the two proportions.",
        },
        "guide_observation": f"2-prop: diff={result.p_diff:.4f}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


# =============================================================================
# Variance / Equivalence
# =============================================================================


def forge_variance_test(df, config):
    """Test for equal variances via forgestat."""
    from forgestat.parametric.variance import variance_test
    from forgeviz.charts.distribution import box_plot

    response = config.get("response") or config.get("column")
    factor = config.get("factor")
    if not response or not factor:
        raise ValueError("Variance test requires 'response' and 'factor'")

    groups = {}
    for name, g in df.groupby(factor):
        vals = pd.to_numeric(g[response], errors="coerce").dropna().values
        if len(vals) > 0:
            groups[str(name)] = vals.tolist()

    arrays = [np.array(v) for v in groups.values()]
    labels = list(groups.keys())
    result = variance_test(*arrays, labels=labels, alpha=_alpha(config))
    sig = "significant" if result.significant else "not significant"

    spec = box_plot(datasets=groups, title=f"Variance of {response} by {factor}")

    return {
        "plots": [_to_chart(spec)],
        "statistics": {
            "statistic": round(result.statistic, 4),
            "p_value": result.p_value,
            "df": getattr(result, "df", None),
        },
        "assumptions": {},
        "summary": f"Equal variances test: stat={result.statistic:.3f}, p={_pval_str(result.p_value)}. Variances are {sig}ly different.",
        "narrative": {
            "verdict": f"Variances are {'unequal' if result.significant else 'equal'}",
            "body": f"Levene's test: statistic={result.statistic:.3f}, p={_pval_str(result.p_value)}.",
            "next_steps": "Use Welch's t-test if variances are unequal."
            if result.significant
            else "Pooled variance methods are appropriate.",
            "chart_guidance": "Box plots show spread by group.",
        },
        "guide_observation": f"Variance test: p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


def forge_equivalence(df, config):
    """TOST equivalence test via forgestat."""
    from forgestat.parametric.equivalence import tost
    from forgeviz.charts.distribution import box_plot

    data1, n1, data2, n2 = _col2(df, config)
    margin = float(config.get("margin", config.get("equivalence_margin", 1.0)))
    result = tost(data1, data2, margin=margin, alpha=_alpha(config))

    spec = box_plot(datasets={n1: data1.tolist(), n2: data2.tolist()}, title=f"Equivalence: {n1} vs {n2}")

    equiv = "equivalent" if result.equivalent else "not equivalent"

    return {
        "plots": [_to_chart(spec)],
        "statistics": {
            "mean_diff": round(result.mean_diff, 4),
            "margin": margin,
            "t_lower": round(result.t_lower, 4),
            "t_upper": round(result.t_upper, 4),
            "p_lower": result.p_lower,
            "p_upper": result.p_upper,
            "p_tost": result.p_tost,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
        },
        "assumptions": _assumptions_dict(result.assumptions)
        if hasattr(result, "assumptions") and result.assumptions
        else {},
        "summary": f"TOST equivalence: diff={result.mean_diff:.3f}, margin=±{margin}, p_TOST={_pval_str(result.p_tost)}. Groups are {equiv}.",
        "narrative": {
            "verdict": f"Groups are {equiv} (margin=±{margin})",
            "body": f"Mean difference = {result.mean_diff:.3f}. CI = [{result.ci_lower:.3f}, {result.ci_upper:.3f}]. TOST p = {_pval_str(result.p_tost)}.",
            "next_steps": "The difference is within the equivalence margin."
            if result.equivalent
            else "Cannot declare equivalence.",
            "chart_guidance": "Box plots show the two distributions.",
        },
        "guide_observation": f"TOST: diff={result.mean_diff:.3f}, p={_pval_str(result.p_tost)}. {equiv.capitalize()}.",
        "diagnostics": [],
    }


# =============================================================================
# Post-Hoc Tests
# =============================================================================


def _forge_posthoc(df, config, func_name, test_label):
    """Generic post-hoc test wrapper."""
    from forgestat.posthoc import comparisons
    from forgeviz.charts.statistical import interval_plot

    func = getattr(comparisons, func_name)
    response = config.get("response") or config.get("column")
    factor = config.get("factor")
    if not response or not factor:
        raise ValueError(f"{test_label} requires 'response' and 'factor'")

    groups = {}
    for name, g in df.groupby(factor):
        vals = pd.to_numeric(g[response], errors="coerce").dropna().values
        if len(vals) > 0:
            groups[str(name)] = vals.tolist()

    arrays = [np.array(v) for v in groups.values()]
    labels = list(groups.keys())
    result = func(*arrays, labels=labels, alpha=_alpha(config))

    spec = interval_plot(groups=groups, confidence=1 - _alpha(config), title=f"{test_label}: {response} by {factor}")

    # Build comparisons table
    comp_stats = {}
    for c in result.comparisons:
        key = f"{c.group1} vs {c.group2}"
        comp_stats[f"diff({key})"] = round(c.mean_diff, 4)
        comp_stats[f"p({key})"] = c.p_value
        comp_stats[f"sig({key})"] = "Yes" if c.significant else "No"

    n_sig = sum(1 for c in result.comparisons if c.significant)

    return {
        "plots": [_to_chart(spec)],
        "statistics": comp_stats,
        "assumptions": {},
        "summary": f"{test_label}: {n_sig}/{len(result.comparisons)} pairwise comparisons significant at α={_alpha(config)}.",
        "narrative": {
            "verdict": f"{n_sig} significant differences found",
            "body": f"{test_label} on {len(labels)} groups. {n_sig} of {len(result.comparisons)} pairs differ significantly.",
            "next_steps": "Focus on significantly different pairs for investigation.",
            "chart_guidance": "Interval plot shows group means with confidence intervals.",
        },
        "guide_observation": f"{test_label}: {n_sig}/{len(result.comparisons)} pairs significant.",
        "diagnostics": [],
    }


def forge_tukey_hsd(df, config):
    return _forge_posthoc(df, config, "tukey_hsd", "Tukey HSD")


def forge_games_howell(df, config):
    return _forge_posthoc(df, config, "games_howell", "Games-Howell")


def forge_dunn(df, config):
    return _forge_posthoc(df, config, "dunn", "Dunn's Test")


def forge_bonferroni(df, config):
    return _forge_posthoc(df, config, "bonferroni", "Bonferroni")


def forge_scheffe(df, config):
    return _forge_posthoc(df, config, "scheffe", "Scheffé")


# =============================================================================
# Normality Test
# =============================================================================


def forge_normality(df, config):
    """Normality test via forgestat assumptions module."""
    from forgestat.core.assumptions import check_normality
    from forgeviz.charts.diagnostic import qq_plot
    from forgeviz.charts.distribution import histogram

    data, col_name = _col(df, config, "column", "var1")
    norm = check_normality(data)

    spec_qq = qq_plot(residuals=data.tolist(), title=f"Normal Q-Q: {col_name}")
    spec_hist = histogram(
        data=data.tolist(), bins=min(25, max(6, len(data) // 4)), title=f"Distribution of {col_name}", show_normal=True
    )

    return {
        "plots": [_to_chart(spec_hist), _to_chart(spec_qq)],
        "statistics": {
            "test": norm.test_name,
            "statistic": round(norm.statistic, 4) if norm.statistic else None,
            "p_value": norm.p_value,
            "normal": norm.passed,
        },
        "assumptions": {"normality": {"pass": norm.passed, "p": norm.p_value, "test": norm.test_name}},
        "summary": f"Normality test ({norm.test_name}): p={_pval_str(norm.p_value)}. Data {'appears normal' if norm.passed else 'is not normal'}.",
        "narrative": {
            "verdict": f"Data {'is' if norm.passed else 'is not'} normally distributed",
            "body": norm.detail,
            "next_steps": "Parametric tests are appropriate."
            if norm.passed
            else "Consider nonparametric alternatives or data transformation.",
            "chart_guidance": "Q-Q plot: points should follow the diagonal if normal. Histogram shows shape.",
        },
        "guide_observation": f"Normality: p={_pval_str(norm.p_value)}. {'Normal' if norm.passed else 'Non-normal'}.",
        "diagnostics": [],
    }


# =============================================================================
# Power Analysis
# =============================================================================


def forge_power_z(df, config):
    """Power/sample size for z-test."""
    from forgestat.power.sample_size import power_z_test

    result = power_z_test(
        effect_size=float(config.get("effect_size", 0.5)),
        n=config.get("n"),
        alpha=_alpha(config),
        power=config.get("power"),
    )
    return _power_result(result, "Z-test")


def forge_power_ttest(df, config):
    """Power/sample size for t-test."""
    from forgestat.power.sample_size import power_t_test

    result = power_t_test(
        effect_size=float(config.get("effect_size", 0.5)),
        n=config.get("n"),
        alpha=_alpha(config),
        power=config.get("power"),
        test_type=config.get("test_type", "one_sample"),
    )
    return _power_result(result, "T-test")


def forge_power_anova(df, config):
    """Power/sample size for ANOVA."""
    from forgestat.power.sample_size import power_anova

    result = power_anova(
        effect_size=float(config.get("effect_size", 0.25)),
        k=int(config.get("k", config.get("n_groups", 3))),
        n_per_group=config.get("n_per_group") or config.get("n"),
        alpha=_alpha(config),
        power=config.get("power"),
    )
    return _power_result(result, "ANOVA")


def forge_power_proportion(df, config):
    """Power/sample size for proportion test."""
    from forgestat.power.sample_size import power_proportion

    result = power_proportion(
        p1=float(config.get("p1", 0.5)),
        p2=config.get("p2"),
        p0=config.get("p0"),
        n=config.get("n"),
        alpha=_alpha(config),
        power=config.get("power"),
    )
    return _power_result(result, "Proportion")


def forge_power_equivalence(df, config):
    """Power/sample size for equivalence test."""
    from forgestat.power.sample_size import power_equivalence

    result = power_equivalence(
        effect_size=float(config.get("effect_size", 0.5)),
        margin=float(config.get("margin", 1.0)),
        n=config.get("n"),
        alpha=_alpha(config),
        power=config.get("power"),
    )
    return _power_result(result, "Equivalence")


def forge_sample_size_ci(df, config):
    """Sample size for confidence interval."""
    from forgestat.power.sample_size import sample_size_for_ci

    n = sample_size_for_ci(
        target_width=float(config.get("target_width", 1.0)),
        std=config.get("std"),
        proportion=config.get("proportion"),
        conf=config.get("conf", 0.95),
    )
    return {
        "plots": [],
        "statistics": {
            "required_n": n,
            "target_width": config.get("target_width", 1.0),
            "confidence": config.get("conf", 0.95),
        },
        "assumptions": {},
        "summary": f"Required sample size for CI: n = {n}.",
        "narrative": {
            "verdict": f"Need n = {n}",
            "body": f"To achieve target CI width of {config.get('target_width', 1.0)} at {config.get('conf', 0.95) * 100:.0f}% confidence.",
            "next_steps": "Collect the required sample.",
            "chart_guidance": "",
        },
        "guide_observation": f"Sample size: n={n}.",
        "diagnostics": [],
    }


def _power_result(result, test_label):
    """Format a PowerResult into the dispatch schema."""
    stats = {
        "power": round(result.power, 4) if result.power else None,
        "sample_size": result.sample_size,
        "alpha": result.alpha,
        "effect_size": round(result.effect_size, 4) if result.effect_size else None,
    }
    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": f"Power analysis ({test_label}): power={result.power:.3f}, n={result.sample_size}, effect={result.effect_size:.3f}."
        if result.power
        else f"Required n={result.sample_size} for {test_label}.",
        "narrative": {
            "verdict": result.detail or f"Power = {result.power:.3f}" if result.power else f"n = {result.sample_size}",
            "body": result.detail or "",
            "next_steps": "Adjust effect size or sample size to meet power targets.",
            "chart_guidance": "",
        },
        "guide_observation": f"Power ({test_label}): {result.power:.3f}, n={result.sample_size}."
        if result.power
        else f"n={result.sample_size}.",
        "diagnostics": [],
    }


# =============================================================================
# Power — Variance Tests
# =============================================================================


def forge_power_1variance(df, config):
    """Power/sample size for 1-variance chi-square test."""
    from scipy.stats import chi2 as chi2_dist

    sigma0 = float(config.get("sigma0", 1.0))
    sigma1 = float(config.get("sigma1", 1.5))
    alpha = _alpha(config)
    target_power = float(config.get("power", 0.80))
    alt = config.get("alternative", "two-sided")
    ratio_sq = (sigma1 / sigma0) ** 2

    n_req = 10000
    for n_try in range(2, 10000):
        df_val = n_try - 1
        if alt == "two-sided":
            lo = chi2_dist.ppf(alpha / 2, df_val)
            hi = chi2_dist.ppf(1 - alpha / 2, df_val)
            pw = chi2_dist.cdf(lo / ratio_sq, df_val) + 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
        elif alt == "greater":
            hi = chi2_dist.ppf(1 - alpha, df_val)
            pw = 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
        else:
            lo = chi2_dist.ppf(alpha, df_val)
            pw = chi2_dist.cdf(lo / ratio_sq, df_val)
        if pw >= target_power:
            n_req = n_try
            break

    return {
        "plots": [],
        "statistics": {
            "required_n": n_req,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "variance_ratio": round(ratio_sq, 4),
            "alpha": alpha,
            "power": target_power,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Power Analysis — 1-Variance Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H\u2080: \u03c3 = {sigma0}, H\u2081: \u03c3 = {sigma1} \u2192 ratio \u03c3\u00b2\u2081/\u03c3\u00b2\u2080 = {ratio_sq:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>\u03b1 = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        ),
        "narrative": {
            "verdict": f"Power Analysis \u2014 n = {n_req} for variance test",
            "body": f"To detect \u03c3\u2081 = {sigma1} vs \u03c3\u2080 = {sigma0} (ratio = {ratio_sq:.3f}) with {target_power * 100:.0f}% power: <strong>n = {n_req}</strong>.",
            "next_steps": "Variance tests require larger samples than mean tests. Consider if a practical change in spread matters.",
            "chart_guidance": "Power curve shows how detection probability increases with n.",
        },
        "guide_observation": f"1-variance power: need n={n_req} to detect \u03c3\u2081={sigma1} vs \u03c3\u2080={sigma0} at power={target_power}.",
        "diagnostics": [],
    }


def forge_power_2variance(df, config):
    """Power/sample size for 2-variance F-test."""
    from scipy.stats import f as f_dist

    var_ratio = float(config.get("variance_ratio", 2.0))
    alpha = _alpha(config)
    target_power = float(config.get("power", 0.80))
    ratio_n = float(config.get("ratio", 1.0))
    alt = config.get("alternative", "two-sided")

    n_req = 10000
    for n_try in range(2, 10000):
        n2_try = max(2, int(n_try * ratio_n))
        df1, df2 = n_try - 1, n2_try - 1
        if alt == "two-sided":
            f_lo = f_dist.ppf(alpha / 2, df1, df2)
            f_hi = f_dist.ppf(1 - alpha / 2, df1, df2)
            pw = f_dist.cdf(f_lo / var_ratio, df1, df2) + 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
        else:
            f_hi = f_dist.ppf(1 - alpha, df1, df2)
            pw = 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
        if pw >= target_power:
            n_req = n_try
            break

    n2_req = max(2, int(n_req * ratio_n))
    return {
        "plots": [],
        "statistics": {
            "n1": n_req,
            "n2": n2_req,
            "total_n": n_req + n2_req,
            "variance_ratio": var_ratio,
            "alpha": alpha,
            "power": target_power,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Power Analysis — 2-Variance F-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H\u2080: \u03c3\u00b2\u2081/\u03c3\u00b2\u2082 = 1, H\u2081: \u03c3\u00b2\u2081/\u03c3\u00b2\u2082 = {var_ratio}<</COLOR>>\n"
            f"<<COLOR:text>>\u03b1 = {alpha}, desired power = {target_power}, n\u2082/n\u2081 = {ratio_n}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n\u2081 = {n_req}, n\u2082 = {n2_req} (total = {n_req + n2_req})<</COLOR>>\n"
        ),
        "narrative": {
            "verdict": f"Power Analysis \u2014 n\u2081 = {n_req}, n\u2082 = {n2_req} for F-test",
            "body": f"To detect a variance ratio of {var_ratio} with {target_power * 100:.0f}% power: <strong>n\u2081 = {n_req}, n\u2082 = {n2_req}</strong>.",
            "next_steps": "F-tests are sensitive to non-normality. Consider Levene's test as a robust alternative.",
            "chart_guidance": "Equal group sizes maximize power.",
        },
        "guide_observation": f"2-variance power: need n\u2081={n_req}, n\u2082={n2_req} for ratio={var_ratio} at power={target_power}.",
        "diagnostics": [],
    }


def forge_power_2prop(df, config):
    """Power/sample size for 2-proportion test."""
    from scipy.stats import norm

    p1 = float(config.get("p1", 0.5))
    p2 = float(config.get("p2", 0.6))
    alpha = _alpha(config)
    target_power = float(config.get("power", 0.80))
    ratio = float(config.get("ratio", 1.0))
    alt = config.get("alternative", "two-sided")

    z_a = norm.ppf(1 - alpha / 2) if alt == "two-sided" else norm.ppf(1 - alpha)
    z_b = norm.ppf(target_power)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    h = abs(2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

    numer = z_a * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar)) + z_b * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2) / ratio
    )
    n1 = math.ceil((numer / (p1 - p2)) ** 2) if p1 != p2 else 9999
    n2 = math.ceil(n1 * ratio)

    return {
        "plots": [],
        "statistics": {
            "n1": n1,
            "n2": n2,
            "total_n": n1 + n2,
            "p1": p1,
            "p2": p2,
            "cohens_h": round(h, 4),
            "alpha": alpha,
            "power": target_power,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Power Analysis — 2-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>p\u2081 = {p1}, p\u2082 = {p2} \u2192 Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>\u03b1 = {alpha}, desired power = {target_power}, ratio n\u2082/n\u2081 = {ratio}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n\u2081 = {n1}, n\u2082 = {n2} (total = {n1 + n2})<</COLOR>>\n"
        ),
        "narrative": {
            "verdict": f"Power Analysis \u2014 n\u2081 = {n1}, n\u2082 = {n2} per group",
            "body": f"To detect |p\u2081 \u2212 p\u2082| = {abs(p1 - p2):.3f} (h = {h:.3f}) with {target_power * 100:.0f}% power: <strong>n\u2081 = {n1}, n\u2082 = {n2}</strong>.",
            "next_steps": "Equal allocation (ratio = 1) is most efficient. Unequal ratios require larger total n.",
            "chart_guidance": "The power curve shows power vs n\u2081.",
        },
        "guide_observation": f"2-prop power: need n\u2081={n1}, n\u2082={n2} for |\u0394p|={abs(p1 - p2):.3f} at power={target_power}.",
        "diagnostics": [],
    }


def forge_power_doe(df, config):
    """Power/sample size for 2-level factorial DOE."""
    from scipy.stats import t as t_dist

    n_factors = int(config.get("factors", config.get("n_factors", 3)))
    delta = float(config.get("delta", config.get("effect_size", 1.0)))
    sigma = float(config.get("sigma", 1.0))
    alpha = _alpha(config)
    target_power = float(config.get("power", 0.80))

    n_runs_base = 2**n_factors

    req_reps = 100
    for reps in range(1, 100):
        n_total = n_runs_base * reps
        df_error = n_total - n_runs_base
        if df_error < 1:
            continue
        se = sigma * math.sqrt(4.0 / n_total)
        if se <= 0:
            continue
        t_crit = t_dist.ppf(1 - alpha / 2, df_error)
        ncp = abs(delta) / se
        pw = 1 - t_dist.cdf(t_crit - ncp, df_error) + t_dist.cdf(-t_crit - ncp, df_error)
        if pw >= target_power:
            req_reps = reps
            break

    n_total_req = n_runs_base * req_reps
    return {
        "plots": [],
        "statistics": {
            "factors": n_factors,
            "base_runs": n_runs_base,
            "required_reps": req_reps,
            "total_runs": n_total_req,
            "delta": delta,
            "sigma": sigma,
            "alpha": alpha,
            "power": target_power,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Power Analysis — 2^{n_factors} Factorial DOE<</COLOR>>\n\n"
            f"<<COLOR:text>>Factors: {n_factors}, base runs: {n_runs_base}<</COLOR>>\n"
            f"<<COLOR:text>>Minimum detectable effect: \u0394 = {delta}, \u03c3 = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>\u03b1 = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required replicates: {req_reps} \u2192 total runs = {n_total_req}<</COLOR>>\n"
        ),
        "narrative": {
            "verdict": f"DOE Power \u2014 {req_reps} replicates ({n_total_req} runs)",
            "body": f"A 2^{n_factors} factorial design needs <strong>{req_reps} replicates</strong> ({n_total_req} total runs) to detect \u0394 = {delta} (\u03c3 = {sigma}) with {target_power * 100:.0f}% power.",
            "next_steps": "If too many runs, consider a fractional factorial or screen fewer factors first.",
            "chart_guidance": "Power increases with replicates.",
        },
        "guide_observation": f"DOE power: 2^{n_factors} design needs {req_reps} reps ({n_total_req} runs) to detect \u0394={delta} at power={target_power}.",
        "diagnostics": [],
    }


def forge_sample_size_tolerance(df, config):
    """Sample size for tolerance interval."""
    from forgestat.power.sample_size import sample_size_tolerance

    coverage = float(config.get("coverage", 0.95))
    confidence = float(config.get("confidence", 0.95))
    std = float(config.get("std", config.get("sigma", 1.0)))
    target_width = config.get("target_width")
    kwargs = {"coverage": coverage, "confidence": confidence, "std": std}
    if target_width is not None:
        kwargs["target_width"] = float(target_width)

    n = sample_size_tolerance(**kwargs)
    return {
        "plots": [],
        "statistics": {
            "required_n": n,
            "coverage": coverage,
            "confidence": confidence,
        },
        "assumptions": {},
        "summary": f"Required sample size for tolerance interval: n = {n} (coverage={coverage}, confidence={confidence}).",
        "narrative": {
            "verdict": f"Need n = {n}",
            "body": f"To achieve {coverage * 100:.0f}% coverage with {confidence * 100:.0f}% confidence.",
            "next_steps": "Collect the required sample.",
            "chart_guidance": "",
        },
        "guide_observation": f"Tolerance interval sample size: n={n}.",
        "diagnostics": [],
    }


# =============================================================================
# Dispatch — imports from split modules
# =============================================================================

from .forge_stats_advanced import FORGE_ADVANCED_HANDLERS  # noqa: E402
from .forge_stats_anova import FORGE_ANOVA_HANDLERS  # noqa: E402
from .forge_stats_exploratory import FORGE_EXPLORATORY_HANDLERS  # noqa: E402
from .forge_stats_msa import FORGE_MSA_HANDLERS  # noqa: E402
from .forge_stats_quality import FORGE_QUALITY_HANDLERS  # noqa: E402
from .forge_stats_regression import FORGE_REGRESSION_HANDLERS  # noqa: E402

FORGE_HANDLERS = {
    # Parametric
    "ttest": forge_ttest,
    "ttest2": forge_ttest2,
    "paired_t": forge_paired_t,
    "anova": forge_anova,
    "chi2": forge_chi2,
    "correlation": forge_correlation,
    "descriptive": forge_descriptive,
    "normality": forge_normality,
    "equivalence": forge_equivalence,
    "variance_test": forge_variance_test,
    "prop_1sample": forge_prop_1sample,
    "prop_2sample": forge_prop_2sample,
    # Nonparametric
    "mann_whitney": forge_mann_whitney,
    "wilcoxon": forge_wilcoxon,
    "kruskal": forge_kruskal,
    "friedman": forge_friedman,
    "mood_median": forge_mood_median,
    "spearman": forge_spearman,
    # Post-hoc (basic — via _forge_posthoc)
    "tukey_hsd": forge_tukey_hsd,
    "games_howell": forge_games_howell,
    "dunn": forge_dunn,
    "bonferroni_test": forge_bonferroni,
    "scheffe_test": forge_scheffe,
    # Power & sample size
    "power_z": forge_power_z,
    "power_ttest": forge_power_ttest,
    "power_anova": forge_power_anova,
    "power_1prop": forge_power_proportion,
    "power_equivalence": forge_power_equivalence,
    "sample_size_ci": forge_sample_size_ci,
    "power_1variance": forge_power_1variance,
    "power_2variance": forge_power_2variance,
    "power_2prop": forge_power_2prop,
    "power_doe": forge_power_doe,
    "sample_size_tolerance": forge_sample_size_tolerance,
}

# Merge in split-module handlers
FORGE_HANDLERS.update(FORGE_REGRESSION_HANDLERS)
FORGE_HANDLERS.update(FORGE_ANOVA_HANDLERS)
FORGE_HANDLERS.update(FORGE_QUALITY_HANDLERS)
FORGE_HANDLERS.update(FORGE_ADVANCED_HANDLERS)
FORGE_HANDLERS.update(FORGE_EXPLORATORY_HANDLERS)
FORGE_HANDLERS.update(FORGE_MSA_HANDLERS)


def run_forge_stats(analysis_id, df, config):
    """Run a forge-backed statistical analysis.

    Returns the result dict, or None if analysis_id is not yet ported to forge.
    Automatically enriches every result with education content.
    """
    handler = FORGE_HANDLERS.get(analysis_id)
    if handler is None:
        return None
    try:
        result = handler(df, config)
    except Exception:
        logger.exception(f"Forge handler failed for {analysis_id}, falling back to legacy")
        return None

    # ── Enrich: education (applies to ALL handlers) ──
    if "education" not in result or result["education"] is None:
        # Map analysis_id to the (type, id) the education system uses
        edu_type = "stats"
        edu_id = analysis_id
        # Some IDs need remapping for the education lookup
        _EDU_REMAP = {
            "paired_t": ("stats", "paired_t"),
            "kruskal": ("stats", "kruskal"),
            "mann_whitney": ("stats", "mann_whitney"),
            "wilcoxon": ("stats", "wilcoxon"),
            "friedman": ("stats", "friedman"),
            "tukey_hsd": ("stats", "tukey_hsd"),
            "games_howell": ("stats", "games_howell"),
            "dunn": ("stats", "dunn"),
            "bonferroni_test": ("stats", "bonferroni"),
            "scheffe_test": ("stats", "scheffe"),
            "regression": ("stats", "regression"),
            "logistic": ("stats", "logistic"),
            "prop_1sample": ("stats", "prop_1sample"),
            "prop_2sample": ("stats", "prop_2sample"),
            "normality": ("stats", "normality"),
            "equivalence": ("stats", "equivalence"),
            "variance_test": ("stats", "variance_test"),
            "power_ttest": ("stats", "power_z"),
            "power_z": ("stats", "power_z"),
            "power_anova": ("stats", "power_z"),
            "power_1prop": ("stats", "power_1prop"),
            "power_equivalence": ("stats", "power_equivalence"),
            "anova2": ("stats", "anova2"),
            "f_test": ("stats", "f_test"),
            "repeated_measures_anova": ("stats", "repeated_measures_anova"),
            "sign_test": ("stats", "sign_test"),
            "split_plot_anova": ("stats", "split_plot_anova"),
            "robust_regression": ("stats", "robust_regression"),
            "poisson_regression": ("stats", "poisson_regression"),
            "nonlinear_regression": ("stats", "nonlinear_regression"),
            "best_subsets": ("stats", "best_subsets"),
            "glm": ("stats", "glm"),
            "ordinal_logistic": ("stats", "ordinal_logistic"),
            "orthogonal_regression": ("stats", "orthogonal_regression"),
            "nominal_logistic": ("stats", "nominal_logistic"),
            "dunnett": ("stats", "dunnett"),
            "hsu_mcb": ("stats", "hsu_mcb"),
            "main_effects": ("stats", "main_effects"),
            "interaction": ("stats", "interaction"),
            "anom": ("stats", "anom"),
            "attribute_capability": ("stats", "attribute_capability"),
            "nonnormal_capability_np": ("stats", "nonnormal_capability_np"),
            "acceptance_sampling": ("stats", "acceptance_sampling"),
            "variable_acceptance_sampling": ("stats", "variable_acceptance_sampling"),
            "variance_components": ("stats", "variance_components"),
            "capability_sixpack": ("stats", "capability_sixpack"),
            "multiple_plan_comparison": ("stats", "multiple_plan_comparison"),
        }
        if analysis_id in _EDU_REMAP:
            edu_type, edu_id = _EDU_REMAP[analysis_id]
        result["education"] = _education(edu_type, edu_id)

    # ── Enrich: wrap plain summary in COLOR format if needed ──
    summary = result.get("summary", "")
    if summary and "<<COLOR:" not in summary:
        # Auto-wrap: extract a title from the analysis label and format
        title = analysis_id.replace("_", " ").upper()
        result["summary"] = (
            "<<COLOR:accent>>══════════════════════════════════════════════════════════════════════<</COLOR>>\n"
            f"<<COLOR:title>>{title}<</COLOR>>\n"
            "<<COLOR:accent>>══════════════════════════════════════════════════════════════════════<</COLOR>>\n\n"
            + summary
        )

    # ── Enrich: ensure required keys exist ──
    result.setdefault("bayesian_shadow", None)
    result.setdefault("power_explorer", None)
    result.setdefault("diagnostics", [])
    result.setdefault("education", None)

    return result
