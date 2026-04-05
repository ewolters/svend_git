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
# Regression
# =============================================================================


def forge_regression(df, config):
    """Linear regression via forgestat."""
    from forgestat.regression.linear import ols
    from forgeviz.charts.diagnostic import four_in_one
    from forgeviz.charts.scatter import scatter

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Regression requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]
    if not predictors:
        raise ValueError("No predictors available")

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for regression")
    y = sub[response].values
    X = sub[predictors].values

    result = ols(X, y, feature_names=predictors, alpha=_alpha(config))

    charts = []
    # Scatter + regression line for simple regression
    if len(predictors) == 1:
        spec = scatter(
            x=X[:, 0].tolist(),
            y=y.tolist(),
            x_label=predictors[0],
            y_label=response,
            title=f"{response} vs {predictors[0]}",
            show_regression=True,
        )
        charts.append(_to_chart(spec))

    # Diagnostic 4-in-1
    if result.fitted is not None and result.residuals is not None:
        diag_specs = four_in_one(
            fitted=result.fitted.tolist() if hasattr(result.fitted, "tolist") else list(result.fitted),
            residuals=result.residuals.tolist() if hasattr(result.residuals, "tolist") else list(result.residuals),
        )
        for s in diag_specs:
            charts.append(_to_chart(s))

    stats = {
        "r_squared": round(result.r_squared, 4),
        "adj_r_squared": round(result.adj_r_squared, 4),
        "f_statistic": round(result.f_statistic, 4) if result.f_statistic else None,
        "f_p_value": result.f_p_value,
        "durbin_watson": round(result.durbin_watson, 4) if result.durbin_watson else None,
        "rmse": round(result.rmse, 4),
        "n": result.n,
    }
    coeffs = result.coefficients if isinstance(result.coefficients, dict) else {}
    pvals = result.p_values if isinstance(result.p_values, dict) else {}
    for key, val in coeffs.items():
        stats[f"coeff({key})"] = round(val, 4)
        if key in pvals:
            stats[f"p({key})"] = pvals[key]

    return {
        "plots": charts,
        "statistics": stats,
        "assumptions": {},
        "summary": f"Regression: R²={result.r_squared:.4f}, adj R²={result.adj_r_squared:.4f}, F={result.f_statistic:.3f}, p={_pval_str(result.f_p_value)}.",
        "narrative": {
            "verdict": f"Model explains {result.r_squared * 100:.1f}% of variation",
            "body": f"Linear regression with {len(predictors)} predictor(s). R²={result.r_squared:.4f}, RMSE={result.rmse:.4f}.",
            "next_steps": "Check residual diagnostics for model adequacy.",
            "chart_guidance": "Diagnostic plots show residual patterns. Look for non-random patterns.",
        },
        "guide_observation": f"Regression: R²={result.r_squared:.4f}, p={_pval_str(result.f_p_value)}.",
        "diagnostics": [],
    }


def forge_logistic(df, config):
    """Logistic regression via forgestat."""
    from forgestat.regression.logistic import logistic_regression

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Logistic regression requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for logistic regression")
    y = sub[response].values.astype(int)
    X = sub[predictors].values

    result = logistic_regression(X, y, feature_names=predictors)

    stats = {
        "pseudo_r_squared": round(result.pseudo_r_squared, 4),
        "aic": round(result.aic, 2),
        "n": result.n,
        "converged": result.converged,
    }
    coeffs = result.coefficients if isinstance(result.coefficients, dict) else {}
    odds = result.odds_ratios if isinstance(result.odds_ratios, dict) else {}
    pvals = result.p_values if isinstance(result.p_values, dict) else {}
    for key, val in coeffs.items():
        stats[f"coeff({key})"] = round(val, 4)
        if key in odds:
            stats[f"OR({key})"] = round(odds[key], 4)
        if key in pvals:
            stats[f"p({key})"] = pvals[key]

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": f"Logistic regression: pseudo R²={result.pseudo_r_squared:.4f}, AIC={result.aic:.1f}, n={result.n}.",
        "narrative": {
            "verdict": f"Model {'converged' if result.converged else 'did not converge'}",
            "body": f"Logistic regression with {len(predictors)} predictors. Pseudo R²={result.pseudo_r_squared:.4f}.",
            "next_steps": "Examine odds ratios for effect interpretation.",
            "chart_guidance": "",
        },
        "guide_observation": f"Logistic: pseudo R²={result.pseudo_r_squared:.4f}.",
        "diagnostics": [],
    }


def forge_stepwise(df, config):
    """Stepwise regression via forgestat."""
    from forgestat.regression.stepwise import stepwise

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Stepwise requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for stepwise")
    y = sub[response].values
    X = sub[predictors].values

    result = stepwise(X, y, feature_names=predictors, method=config.get("method", "both"))

    stats = {"selected_features": result.selected_features, "n_steps": len(result.steps)}
    if result.final_model:
        stats["r_squared"] = round(result.final_model.r_squared, 4)
        stats["adj_r_squared"] = round(result.final_model.adj_r_squared, 4)

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": f"Stepwise ({result.method}): selected {len(result.selected_features)} of {len(predictors)} predictors"
        + (f", R²={result.final_model.r_squared:.4f}" if result.final_model else "")
        + ".",
        "narrative": {
            "verdict": f"Selected: {', '.join(result.selected_features) if result.selected_features else 'none'}",
            "body": f"Stepwise selection completed in {len(result.steps)} steps.",
            "next_steps": "Validate with holdout data.",
            "chart_guidance": "",
        },
        "guide_observation": f"Stepwise: {len(result.selected_features)} predictors selected.",
        "diagnostics": [],
    }


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
# Two-way ANOVA
# =============================================================================


def forge_anova2(df, config):
    """Two-way ANOVA via forgestat."""
    from forgestat.parametric.anova import two_way
    from forgeviz.charts.generic import multi_line

    response = config.get("response") or config.get("var")
    factor_a = config.get("factor_a")
    factor_b = config.get("factor_b")
    alpha = _alpha(config)

    if not all([response, factor_a, factor_b]):
        raise ValueError("Two-way ANOVA requires response, factor_a, and factor_b")

    # Build data dict from DataFrame
    data_dict = {
        response: pd.to_numeric(df[response], errors="coerce").tolist(),
        factor_a: df[factor_a].astype(str).tolist(),
        factor_b: df[factor_b].astype(str).tolist(),
    }

    result = two_way(data_dict, response=response, factor_a=factor_a, factor_b=factor_b, alpha=alpha)

    # Build stats from sources
    effect_stats = {}
    for src in result.sources:
        effect_stats[src.source] = {
            "ss": round(src.ss, 4),
            "df": int(src.df),
            "ms": round(src.ms, 4),
            "f_statistic": round(src.f_statistic, 4),
            "p_value": src.p_value,
            "partial_eta_squared": round(src.partial_eta_sq, 4),
        }

    # Classify effects
    sig_effects = [s for s in result.sources if s.p_value < alpha]
    strongest = max(result.sources, key=lambda s: s.partial_eta_sq) if result.sources else None
    ix_source = next((s for s in result.sources if ":" in s.source or "×" in s.source), None)
    has_interaction = ix_source is not None and ix_source.p_value < alpha

    # Interaction plot via ForgeViz multi_line
    a_levels = sorted(df[factor_a].astype(str).unique())
    b_levels = sorted(df[factor_b].astype(str).unique())
    series = {}
    for b_lev in b_levels:
        means = []
        for a_lev in a_levels:
            cell = df[(df[factor_a].astype(str) == a_lev) & (df[factor_b].astype(str) == b_lev)][response]
            cell_num = pd.to_numeric(cell, errors="coerce").dropna()
            means.append(round(float(cell_num.mean()), 4) if len(cell_num) > 0 else 0.0)
        series[str(b_lev)] = means

    ix_chart = multi_line(
        x=a_levels,
        series=series,
        title=f"Interaction Plot: {factor_a} × {factor_b}",
        x_label=factor_a,
        y_label=f"Mean {response}",
        show_markers=True,
    )
    plots = [_to_chart(ix_chart)]

    # Summary sections
    sections = [
        (
            "Design",
            [
                ("Response", response),
                ("Factor A", f"{factor_a} ({len(a_levels)} levels)"),
                ("Factor B", f"{factor_b} ({len(b_levels)} levels)"),
                ("N", str(len(df[response].dropna()))),
            ],
        ),
        (
            "ANOVA Table",
            [
                (src.source, f"F = {src.f_statistic:.3f}, p = {_pval_str(src.p_value)}, η²p = {src.partial_eta_sq:.3f}")
                for src in result.sources
            ],
        ),
        (
            "Residual",
            [
                ("df", str(int(result.residual_df))),
                ("SS", f"{result.residual_ss:.4f}"),
                ("MS", f"{result.residual_ms:.4f}"),
            ],
        ),
    ]

    # Narrative
    if sig_effects:
        if has_interaction:
            verdict = (
                f"Interaction is significant (p = {_pval_str(ix_source.p_value)}, η²p = {ix_source.partial_eta_sq:.3f})"
            )
            body = (
                f"The {factor_a}×{factor_b} interaction is significant — the effect of {factor_a} "
                f"depends on {factor_b}. Interpret main effects with caution. "
                f"Significant effects: <strong>{', '.join(s.source for s in sig_effects)}</strong>."
            )
            nxt = "Run post-hoc tests (Tukey HSD) on simple effects within each level of the interacting factor."
        else:
            verdict = f"{strongest.source} has the largest effect (η²p = {strongest.partial_eta_sq:.3f})"
            body = (
                f"Significant main effects: <strong>{', '.join(s.source for s in sig_effects)}</strong>. "
                f"No significant interaction — factors act independently on {response}."
            )
            nxt = "Run post-hoc tests (Tukey HSD) on significant main effects to identify which levels differ."
    else:
        verdict = "No significant effects detected"
        body = f"Neither {factor_a}, {factor_b}, nor their interaction significantly affects {response}."
        nxt = "Check sample sizes and effect sizes. The study may lack power."

    # Diagnostics from assumptions
    diagnostics = [_assumption_to_dict(a) for a in result.assumptions]
    if has_interaction:
        diagnostics.append(
            {
                "level": "info",
                "title": "Interaction detected — interpret main effects with caution",
                "detail": (
                    f"The {factor_a}×{factor_b} interaction is significant (p = {_pval_str(ix_source.p_value)}). "
                    f"Main effects alone do not tell the full story — the effect of one factor depends on the other."
                ),
            }
        )
    if not sig_effects:
        diagnostics.append(
            {
                "level": "warning",
                "title": "No detectable effects — consider increasing sample size",
                "detail": f"Neither {factor_a}, {factor_b}, nor their interaction reached significance.",
            }
        )

    return {
        "plots": plots,
        "statistics": {
            "effects": effect_stats,
            "residual_df": int(result.residual_df),
            "residual_ss": round(result.residual_ss, 4),
            "residual_ms": round(result.residual_ms, 4),
        },
        "assumptions": _assumptions_dict(result.assumptions),
        "summary": _rich_summary("TWO-WAY ANOVA", sections),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": nxt,
            "chart_guidance": "Non-parallel lines in the interaction plot suggest an interaction between factors.",
        },
        "guide_observation": "Two-way ANOVA: "
        + "; ".join(f"{s.source}: p={_pval_str(s.p_value)}, η²p={s.partial_eta_sq:.3f}" for s in result.sources),
        "diagnostics": diagnostics,
    }


# =============================================================================
# F-test for equality of variances
# =============================================================================


def forge_f_test(df, config):
    """F-test for equality of two variances via forgestat."""
    from forgestat.parametric.variance import f_test
    from forgeviz.charts.distribution import box_plot
    from forgeviz.charts.generic import bar

    var = config.get("var") or config.get("column")
    group_var = config.get("group_var") or config.get("factor")
    alpha = _alpha(config)

    if not var or not group_var:
        raise ValueError("F-test requires var and group_var")

    groups = df[group_var].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"F-test requires exactly 2 groups. Found {len(groups)}.")

    g1_data = pd.to_numeric(df[df[group_var] == groups[0]][var], errors="coerce").dropna().values
    g2_data = pd.to_numeric(df[df[group_var] == groups[1]][var], errors="coerce").dropna().values

    result = f_test(g1_data, g2_data, alpha=alpha)
    extra = result.extra or {}
    var1 = extra.get("var1", float(np.var(g1_data, ddof=1)))
    var2 = extra.get("var2", float(np.var(g2_data, ddof=1)))
    n1, n2 = len(g1_data), len(g2_data)

    # Charts: variance comparison bar + side-by-side box
    bar_chart = bar(
        categories=[str(groups[0]), str(groups[1])],
        values=[var1, var2],
        title=f"Variance Comparison: {groups[0]} vs {groups[1]}",
    )
    box_chart = box_plot(
        datasets={str(groups[0]): g1_data.tolist(), str(groups[1]): g2_data.tolist()},
        title=f"Distribution by Group: {var}",
    )
    plots = [_to_chart(bar_chart), _to_chart(box_chart)]

    # CI for variance ratio
    from scipy import stats as sp_stats

    df1, df2 = extra.get("df1", n1 - 1), extra.get("df2", n2 - 1)
    f_stat = result.statistic
    ci_lo = f_stat / sp_stats.f.ppf(0.975, df1, df2) if df1 > 0 and df2 > 0 else 0
    ci_hi = f_stat / sp_stats.f.ppf(0.025, df1, df2) if df1 > 0 and df2 > 0 else 0
    vr = max(var1, var2) / min(var1, var2) if min(var1, var2) > 0 else float("inf")

    if result.significant:
        verdict = f"Variances differ significantly (F = {f_stat:.3f}, p = {_pval_str(result.p_value)})"
        body = (
            f"Variance ratio = {vr:.2f}. Use Welch's t-test (not pooled) or non-parametric tests for group comparisons."
        )
        nxt = "F-test is sensitive to non-normality. For robust alternatives, use Levene's test."
    else:
        verdict = f"Variances are similar (F = {f_stat:.3f}, p = {_pval_str(result.p_value)})"
        body = f"Variance ratio = {vr:.2f}. The equal-variance assumption is reasonable for pooled t-tests and ANOVA."
        nxt = "Equal variance confirmed — pooled tests are appropriate."

    return {
        "plots": plots,
        "statistics": {
            "F_statistic": round(f_stat, 4),
            "p_value": result.p_value,
            "variance_ratio": round(vr, 4),
            f"variance({groups[0]})": round(var1, 4),
            f"variance({groups[1]})": round(var2, 4),
            f"n({groups[0]})": n1,
            f"n({groups[1]})": n2,
            "df1": df1,
            "df2": df2,
            "ci_ratio_lower": round(ci_lo, 4),
            "ci_ratio_upper": round(ci_hi, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "F-TEST FOR EQUALITY OF VARIANCES",
            [
                ("Variable", [(var, f"{groups[0]} vs {groups[1]}")]),
                (
                    "Group Statistics",
                    [
                        (str(groups[0]), f"n={n1}, variance={var1:.4f}, StDev={np.sqrt(var1):.4f}"),
                        (str(groups[1]), f"n={n2}, variance={var2:.4f}, StDev={np.sqrt(var2):.4f}"),
                    ],
                ),
                (
                    "Test Results",
                    [
                        ("F statistic", f"{f_stat:.4f}"),
                        ("p-value", _pval_str(result.p_value)),
                        ("95% CI for ratio", f"[{ci_lo:.4f}, {ci_hi:.4f}]"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": nxt,
            "chart_guidance": "Bar chart compares group variances. Box plots show spread and outliers per group.",
        },
        "guide_observation": f"F-test: F={f_stat:.3f}, p={_pval_str(result.p_value)}. "
        + ("Variances differ significantly." if result.significant else "Variances are similar."),
        "diagnostics": [],
    }


# =============================================================================
# Repeated Measures ANOVA
# =============================================================================


def forge_repeated_measures_anova(df, config):
    """Repeated measures ANOVA via forgestat."""
    from forgestat.parametric.repeated_measures import repeated_measures_anova
    from forgeviz.charts.generic import multi_line

    response = config.get("response") or config.get("var")
    subject_col = config.get("subject") or config.get("subject_id")
    within_factor = config.get("within_factor") or config.get("condition")
    alpha = _alpha(config)

    if not all([response, subject_col, within_factor]):
        raise ValueError("Repeated measures ANOVA requires response, subject, and within_factor")

    data_rm = df[[response, subject_col, within_factor]].dropna()
    data_rm[subject_col] = data_rm[subject_col].astype(str)
    data_rm[within_factor] = data_rm[within_factor].astype(str)

    conditions = sorted(data_rm[within_factor].unique())
    k = len(conditions)
    if k < 2:
        raise ValueError("Need at least 2 levels of within-subject factor")

    # Pivot to subject × condition, drop incomplete subjects
    pivot = data_rm.pivot_table(
        index=subject_col,
        columns=within_factor,
        values=response,
        aggfunc="mean",
    ).dropna()

    if len(pivot) < 3:
        raise ValueError("Need at least 3 complete subjects (all conditions measured)")

    # Build forgestat input: {condition_name: [subject_scores]}
    rm_data = {str(cond): pivot[cond].tolist() for cond in pivot.columns}

    result = repeated_measures_anova(rm_data)
    n_subj = result.n_subjects

    # Use GG-corrected p when sphericity violated
    best_p = result.p_value_gg if (k > 2 and not result.sphericity_met) else result.p_value
    eta_label = "large" if result.partial_eta_sq > 0.14 else "medium" if result.partial_eta_sq > 0.06 else "small"

    # Profile plot: condition means ± SE
    cond_names = list(result.condition_means.keys())
    cond_means = list(result.condition_means.values())
    cond_stds = [float(np.std(pivot[c].values, ddof=1)) for c in pivot.columns]
    cond_ses = [s / np.sqrt(n_subj) for s in cond_stds]  # noqa: F841

    profile_chart = multi_line(
        x=cond_names,
        series={"Mean ± SE": cond_means},
        title=f"Profile Plot: {within_factor} Means",
        x_label=within_factor,
        y_label=f"Mean {response}",
        show_markers=True,
    )
    plots = [_to_chart(profile_chart)]

    # Spaghetti plot (individual subject trajectories) — build as raw ChartSpec dict
    # since ForgeViz multi_line doesn't support opacity per trace
    spaghetti_series = {}
    for si, subj in enumerate(list(pivot.index)[:30]):
        spaghetti_series[str(subj)] = pivot.loc[subj].tolist()
    spaghetti_chart = multi_line(
        x=cond_names,
        series=spaghetti_series,
        title="Subject Trajectories",
        x_label=within_factor,
        y_label=response,
        show_markers=False,
    )
    plots.append(_to_chart(spaghetti_chart))

    # Summary sections
    sections = [
        (
            "Design",
            [
                ("Response", response),
                ("Within-subject factor", f"{within_factor} ({k} levels)"),
                ("Subjects", str(n_subj)),
            ],
        ),
        (
            "Within-Subjects ANOVA",
            [
                ("F", f"{result.f_statistic:.3f}"),
                ("df", f"({int(result.df_condition)}, {int(result.df_error)})"),
                ("p-value", _pval_str(result.p_value)),
                ("Partial η²", f"{result.partial_eta_sq:.4f} ({eta_label})"),
            ],
        ),
    ]

    if k > 2:
        sph_status = "met" if result.sphericity_met else "VIOLATED"
        sections.append(
            (
                "Sphericity",
                [
                    ("Mauchly's p", _pval_str(result.mauchly_p) if result.mauchly_p is not None else "N/A"),
                    ("Status", sph_status),
                    ("GG ε", f"{result.epsilon_gg:.4f}"),
                    ("GG-corrected p", _pval_str(result.p_value_gg)),
                ],
            )
        )

    sections.append(
        (
            "Condition Means",
            [
                (name, f"{mean:.4f} (SD = {cond_stds[i]:.4f})")
                for i, (name, mean) in enumerate(result.condition_means.items())
            ],
        )
    )

    # Narrative
    if best_p < alpha:
        verdict = f"Conditions differ significantly (F = {result.f_statistic:.3f}, p = {_pval_str(best_p)})"
        body = (
            f"η² = {result.partial_eta_sq:.4f} ({eta_label} effect). "
            f"The repeated-measures factor significantly affects {response}."
            + (
                f" Sphericity violated — using Greenhouse-Geisser correction (ε = {result.epsilon_gg:.3f})."
                if k > 2 and not result.sphericity_met
                else ""
            )
        )
        nxt = "Run post-hoc paired comparisons to identify which conditions differ."
    else:
        verdict = f"No significant effect across conditions (p = {_pval_str(best_p)})"
        body = f"η² = {result.partial_eta_sq:.4f}. The conditions do not significantly differ."
        nxt = "Consider increasing sample size if a meaningful difference was expected."

    diagnostics = []
    if k > 2 and not result.sphericity_met:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Sphericity violated (Mauchly's p = {_pval_str(result.mauchly_p)})",
                "detail": f"Use Greenhouse-Geisser corrected p-value ({_pval_str(result.p_value_gg)}) instead of uncorrected ({_pval_str(result.p_value)}).",
            }
        )
    if result.partial_eta_sq > 0.14 and best_p < alpha:
        diagnostics.append(
            {
                "level": "info",
                "title": f"Large practical effect (η² = {result.partial_eta_sq:.3f})",
                "detail": f"The within-subject factor explains {result.partial_eta_sq * 100:.1f}% of variance — a substantial effect.",
            }
        )

    return {
        "plots": plots,
        "statistics": {
            "f_statistic": round(result.f_statistic, 4),
            "p_value": result.p_value,
            "p_value_gg": result.p_value_gg,
            "df_condition": int(result.df_condition),
            "df_error": int(result.df_error),
            "ss_condition": round(result.ss_condition, 4),
            "ss_error": round(result.ss_error, 4),
            "partial_eta_squared": round(result.partial_eta_sq, 4),
            "n_subjects": n_subj,
            "k_levels": k,
            "mauchly_p": result.mauchly_p,
            "gg_epsilon": round(result.epsilon_gg, 4),
            "sphericity_met": result.sphericity_met,
            "condition_means": result.condition_means,
        },
        "assumptions": {},
        "summary": _rich_summary("REPEATED MEASURES ANOVA", sections),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": nxt,
            "chart_guidance": "Profile plot shows condition means. Spaghetti plot shows individual subject trajectories — look for consistent patterns.",
        },
        "guide_observation": (
            f"Repeated measures ANOVA: F({int(result.df_condition)},{int(result.df_error)})={result.f_statistic:.3f}, "
            f"p={_pval_str(best_p)}, η²={result.partial_eta_sq:.4f}."
        ),
        "diagnostics": diagnostics,
    }


# =============================================================================
# Sign Test
# =============================================================================


def forge_sign_test(df, config):
    """Sign test for median via forgestat."""
    from forgestat.nonparametric.rank_tests import sign_test
    from forgeviz.charts.distribution import histogram

    data, col_name = _col(df, config, "column", "var")
    h0_median = float(config.get("hypothesized_median", config.get("mu", 0)))
    alpha = _alpha(config)

    result = sign_test(data, median0=h0_median, alpha=alpha)
    extra = result.extra or {}
    above = extra.get("above", int(np.sum(data > h0_median)))
    below = extra.get("below", int(np.sum(data < h0_median)))
    ties = extra.get("ties", int(np.sum(data == h0_median)))
    sample_median = float(np.median(data))

    # CI on median (binomial-based)
    from scipy import stats as sp_stats

    sorted_data = np.sort(data)
    n_total = len(data)
    ci_idx = max(0, int(sp_stats.binom.ppf(0.025, n_total, 0.5)) - 1)
    ci_lower = float(sorted_data[ci_idx]) if ci_idx < n_total else float(sorted_data[0])
    ci_upper = float(sorted_data[n_total - 1 - ci_idx]) if ci_idx < n_total else float(sorted_data[-1])

    # Chart: histogram with median lines
    spec = histogram(
        data=data.tolist(),
        bins=min(30, max(8, len(data) // 5)),
        title=f"Sign Test: {col_name}",
        target=h0_median,
        show_normal=False,
    )
    plots = [_to_chart(spec)]

    if result.significant:
        verdict = f"Median differs from {h0_median} (p = {_pval_str(result.p_value)})"
        body = f"Sample median = {sample_median:.4f}. {above} values above and {below} below {h0_median}. The imbalance is significant."
        nxt = "The sign test uses only directions, not magnitudes — it is the most robust nonparametric test."
    else:
        verdict = f"Median consistent with {h0_median} (p = {_pval_str(result.p_value)})"
        body = f"Sample median = {sample_median:.4f}. {above} above and {below} below. No significant departure from {h0_median}."
        nxt = "Consider Wilcoxon signed-rank test if you want to use magnitude information for more power."

    return {
        "plots": plots,
        "statistics": {
            "sample_median": round(sample_median, 4),
            "above": above,
            "below": below,
            "ties": ties,
            "n_used": above + below,
            "p_value": result.p_value,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "SIGN TEST FOR MEDIAN",
            [
                ("Variable", [(col_name, f"n = {n_total}")]),
                ("Hypothesized median", [("H₀", str(h0_median))]),
                (
                    "Sample Statistics",
                    [
                        ("Sample median", f"{sample_median:.4f}"),
                        ("95% CI", f"[{ci_lower:.4f}, {ci_upper:.4f}]"),
                    ],
                ),
                (
                    "Sign Counts",
                    [
                        ("Above H₀", str(above)),
                        ("Below H₀", str(below)),
                        ("Ties (excluded)", str(ties)),
                    ],
                ),
                (
                    "Test Result",
                    [
                        ("p-value (two-sided)", _pval_str(result.p_value)),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": nxt,
            "chart_guidance": "Histogram shows the data distribution. The reference line marks the hypothesized median.",
        },
        "guide_observation": f"Sign test: median={sample_median:.4f} vs H₀={h0_median}, p={_pval_str(result.p_value)}. "
        + ("Median differs." if result.significant else "No evidence of difference."),
        "diagnostics": [],
    }


# =============================================================================
# Split-Plot ANOVA
# =============================================================================


def forge_split_plot_anova(df, config):
    """Split-plot ANOVA via forgestat."""
    from forgestat.parametric.split_plot import split_plot_anova
    from forgeviz.charts.generic import multi_line

    response = config.get("response") or config.get("var")
    wp_factors = config.get("whole_plot_factors", [])
    sp_factors = config.get("sub_plot_factors", [])
    block_col = config.get("block") or config.get("whole_plot_id")
    alpha = _alpha(config)

    if isinstance(wp_factors, str):
        wp_factors = [wp_factors]
    if isinstance(sp_factors, str):
        sp_factors = [sp_factors]

    if not response or not wp_factors or not sp_factors:
        raise ValueError("Split-plot ANOVA requires response, whole_plot_factors, and sub_plot_factors")

    # Use first WP and SP factor for forgestat (single WP × SP design)
    wp_factor = wp_factors[0]
    sp_factor = sp_factors[0]

    # Build data dict
    needed = [response, wp_factor, sp_factor]
    if block_col:
        needed.append(block_col)
    data_sp = df[needed].dropna()
    for f in [wp_factor, sp_factor]:
        data_sp[f] = data_sp[f].astype(str)

    data_dict = {
        response: pd.to_numeric(data_sp[response], errors="coerce").tolist(),
        wp_factor: data_sp[wp_factor].tolist(),
        sp_factor: data_sp[sp_factor].tolist(),
    }

    result = split_plot_anova(
        data_dict, response=response, whole_plot_factor=wp_factor, sub_plot_factor=sp_factor, block=block_col
    )

    # Build ANOVA table from sources
    anova_rows = []
    for src in result.sources:
        anova_rows.append(
            {
                "source": src.source,
                "ss": round(src.ss, 4),
                "df": int(src.df),
                "ms": round(src.ms, 4),
                "f_statistic": round(src.f_statistic, 4) if src.f_statistic else None,
                "p_value": src.p_value if src.p_value else None,
                "error_term": src.error_term,
                "significant": src.p_value < alpha if src.p_value else False,
            }
        )

    n_sig = sum(1 for r in anova_rows if r.get("significant"))

    # Main effects plot via multi_line
    all_factors = [wp_factor, sp_factor]
    series = {}
    x_labels = []
    for factor in all_factors:
        grp = data_sp.groupby(factor)[response].apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
        for lev, mean_val in grp.items():
            x_labels.append(f"{factor}={lev}")
            series.setdefault(factor, []).append(float(mean_val))

    # Interaction chart: SP levels across WP levels
    wp_levels = sorted(data_sp[wp_factor].unique())
    sp_levels = sorted(data_sp[sp_factor].unique())
    ix_series = {}
    for s_lev in sp_levels:
        means = []
        for w_lev in wp_levels:
            cell = data_sp[(data_sp[wp_factor] == w_lev) & (data_sp[sp_factor] == s_lev)][response]
            cell_num = pd.to_numeric(cell, errors="coerce").dropna()
            means.append(round(float(cell_num.mean()), 4) if len(cell_num) > 0 else 0.0)
        ix_series[str(s_lev)] = means

    ix_chart = multi_line(
        x=[str(w) for w in wp_levels],
        series=ix_series,
        title=f"Split-Plot Interaction: {wp_factor} × {sp_factor}",
        x_label=wp_factor,
        y_label=f"Mean {response}",
        show_markers=True,
    )
    plots = [_to_chart(ix_chart)]

    # Summary
    table_items = []
    for r in anova_rows:
        if r["f_statistic"] is not None:
            table_items.append(
                (r["source"], f"F = {r['f_statistic']:.3f}, p = {_pval_str(r['p_value'])}, Error: {r['error_term']}")
            )
        else:
            table_items.append((r["source"], f"SS = {r['ss']:.4f}, df = {r['df']}, MS = {r['ms']:.4f}"))

    sections = [
        (
            "Design",
            [
                ("Response", response),
                ("Whole-plot factor", ", ".join(wp_factors)),
                ("Sub-plot factor", ", ".join(sp_factors)),
                ("N", str(len(data_sp))),
            ],
        ),
        ("ANOVA Table", table_items),
    ]

    verdict = f"Split-Plot ANOVA: {n_sig} significant term{'s' if n_sig != 1 else ''}"
    body = (
        f"Whole-plot factors tested against WP error (MS = {result.whole_plot_error_ms:.4f}), "
        f"sub-plot factors against residual (MS = {result.sub_plot_error_ms:.4f})."
    )

    return {
        "plots": plots,
        "statistics": {
            "anova_table": anova_rows,
            "n": len(data_sp),
            "n_whole_plots": result.n_whole_plots,
            "n_sub_plots": result.n_sub_plots,
        },
        "assumptions": {},
        "summary": _rich_summary("SPLIT-PLOT ANOVA", sections),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": "Split-plot designs arise when some factors are harder to change. Check both error terms for proper inference.",
            "chart_guidance": "Interaction plot shows sub-plot factor means across whole-plot levels. Non-parallel lines indicate interaction.",
        },
        "guide_observation": f"Split-plot ANOVA: {n_sig} significant terms. WP error MS={result.whole_plot_error_ms:.4f}, SP error MS={result.sub_plot_error_ms:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Robust Regression
# =============================================================================


def forge_robust_regression(df, config):
    """Robust regression via forgestat."""
    from forgestat.regression.robust import robust_regression
    from forgeviz.charts.scatter import scatter

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))
    method = config.get("method", "huber")

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = robust_regression(X, y, feature_names=predictors, method=method)

    # Chart: fitted vs actual
    fitted = y - np.array(result.residuals)
    chart = scatter(
        x=fitted.tolist(),
        y=y.tolist(),
        title=f"Robust Regression: Fitted vs Actual ({method})",
        x_label="Fitted",
        y_label="Actual",
    )
    plots = [_to_chart(chart)]

    coef_items = [
        (
            name,
            f"{val:.4f} (OLS: {result.ols_coefficients.get(name, 0):.4f}, Δ={result.coefficient_changes.get(name, 0):.1f}%)",
        )
        for name, val in result.coefficients.items()
    ]

    return {
        "plots": plots,
        "statistics": {
            "method": result.method,
            "coefficients": result.coefficients,
            "ols_coefficients": result.ols_coefficients,
            "coefficient_changes_pct": result.coefficient_changes,
            "r_squared": round(result.r_squared, 4),
            "n_downweighted": result.n_downweighted,
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"ROBUST REGRESSION ({method.upper()})",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Predictors", ", ".join(predictors)),
                        ("Method", method),
                        ("N", str(len(y))),
                    ],
                ),
                ("Coefficients (Robust vs OLS)", coef_items),
                (
                    "Fit",
                    [
                        ("R²", f"{result.r_squared:.4f}"),
                        ("Downweighted obs", str(result.n_downweighted)),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Robust regression ({method}): R² = {result.r_squared:.3f}, {result.n_downweighted} observations downweighted",
            "body": (
                f"Comparing robust vs OLS coefficients shows where outliers pull the fit. "
                f"{result.n_downweighted} observations received reduced weight. "
                + (
                    "Large coefficient changes suggest OLS is sensitive to outliers here."
                    if any(abs(v) > 20 for v in result.coefficient_changes.values())
                    else "Coefficients are stable — outlier influence is minimal."
                )
            ),
            "next_steps": "Compare robust and OLS residuals to identify influential observations.",
            "chart_guidance": "Scatter plot shows fitted vs actual values. Points near the diagonal indicate good fit.",
        },
        "guide_observation": f"Robust regression ({method}): R²={result.r_squared:.3f}, {result.n_downweighted} downweighted.",
        "diagnostics": [],
    }


# =============================================================================
# Poisson Regression
# =============================================================================


def forge_poisson_regression(df, config):
    """Poisson regression via forgestat."""
    from forgestat.regression.logistic import poisson_regression
    from forgeviz.charts.generic import bar

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values.astype(int)
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = poisson_regression(X, y, feature_names=predictors)

    # Chart: IRR bar chart
    irr_names = [k for k in result.irr.keys() if k != "intercept"]
    irr_vals = [result.irr[k] for k in irr_names]
    chart = bar(
        categories=irr_names,
        values=irr_vals,
        title="Incidence Rate Ratios",
        y_label="IRR",
    )
    plots = [_to_chart(chart)]

    coef_items = [
        (
            name,
            f"β={result.coefficients.get(name, 0):.4f}, IRR={result.irr.get(name, 0):.4f}, p={_pval_str(result.p_values.get(name, 1))}",
        )
        for name in irr_names
    ]

    sig_predictors = [k for k in irr_names if result.p_values.get(k, 1) < 0.05]

    return {
        "plots": plots,
        "statistics": {
            "coefficients": result.coefficients,
            "irr": result.irr,
            "p_values": result.p_values,
            "deviance": round(result.deviance, 4),
            "pearson_chi2": round(result.pearson_chi2, 4),
            "aic": round(result.aic, 4),
            "n": result.n,
        },
        "assumptions": {},
        "summary": _rich_summary(
            "POISSON REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} (count data)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                (
                    "Model Fit",
                    [
                        ("Deviance", f"{result.deviance:.4f}"),
                        ("Pearson χ²", f"{result.pearson_chi2:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Poisson regression: {len(sig_predictors)} significant predictors, AIC = {result.aic:.1f}",
            "body": (
                "IRR > 1 means increased rate; IRR < 1 means decreased rate. "
                + (
                    f"Significant: <strong>{', '.join(sig_predictors)}</strong>."
                    if sig_predictors
                    else "No predictors reached significance."
                )
                + (
                    f" Check for overdispersion: Pearson χ²/df = {result.pearson_chi2 / (result.n - len(predictors) - 1):.2f} "
                    f"({'overdispersed — consider negative binomial' if result.pearson_chi2 / (result.n - len(predictors) - 1) > 1.5 else 'acceptable'})."
                    if result.n > len(predictors) + 1
                    else ""
                )
            ),
            "next_steps": "Check for overdispersion. If Pearson χ²/df >> 1, switch to negative binomial regression.",
            "chart_guidance": "Bar chart shows incidence rate ratios. IRR = 1 means no effect; > 1 means increased count rate.",
        },
        "guide_observation": f"Poisson regression: {len(sig_predictors)} significant, AIC={result.aic:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nonlinear Regression (Curve Fitting)
# =============================================================================


def forge_nonlinear_regression(df, config):
    """Nonlinear regression (curve fitting) via forgestat."""
    from forgestat.regression.nonlinear import curve_fit
    from forgeviz.charts.scatter import scatter

    x_var = config.get("x_var") or config.get("var1") or config.get("predictor")
    y_var = config.get("y_var") or config.get("var2") or config.get("response")
    model_type = config.get("model", "exponential")

    if not x_var or not y_var:
        raise ValueError("Nonlinear regression requires x_var and y_var")

    x_data = pd.to_numeric(df[x_var], errors="coerce").dropna()
    y_data = pd.to_numeric(df[y_var], errors="coerce").dropna()
    # Align indices
    common = x_data.index.intersection(y_data.index)
    x_vals = x_data.loc[common].values
    y_vals = y_data.loc[common].values

    result = curve_fit(x_vals, y_vals, model=model_type)

    if not result.converged:
        return {
            "plots": [],
            "statistics": {"converged": False, "model": model_type},
            "summary": f"Model '{model_type}' did not converge. Try a different model or initial parameters.",
            "narrative": {
                "verdict": "Model did not converge",
                "body": "",
                "next_steps": "Try a different model type.",
                "chart_guidance": "",
            },
            "guide_observation": f"Nonlinear fit ({model_type}) did not converge.",
            "diagnostics": [],
        }

    # Chart: data + fitted curve
    chart = scatter(
        x=x_vals.tolist(),
        y=y_vals.tolist(),
        title=f"Nonlinear Fit: {model_type}",
        x_label=x_var,
        y_label=y_var,
    )
    plots = [_to_chart(chart)]

    param_items = [
        (name, f"{val:.6f}" + (f" ± {result.std_errors.get(name, 0):.6f}" if result.std_errors.get(name) else ""))
        for name, val in result.parameters.items()
    ]

    return {
        "plots": plots,
        "statistics": {
            "model": result.model,
            "parameters": result.parameters,
            "std_errors": result.std_errors,
            "r_squared": round(result.r_squared, 4),
            "rmse": round(result.rmse, 4),
            "aic": round(result.aic, 4),
            "bic": round(result.bic, 4),
            "n": result.n,
            "converged": result.converged,
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"NONLINEAR REGRESSION ({model_type.upper()})",
            [
                (
                    "Design",
                    [
                        ("X", x_var),
                        ("Y", y_var),
                        ("Model", model_type),
                        ("N", str(result.n)),
                    ],
                ),
                ("Parameters", param_items),
                (
                    "Fit",
                    [
                        ("R²", f"{result.r_squared:.4f}"),
                        ("RMSE", f"{result.rmse:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                        ("BIC", f"{result.bic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Nonlinear fit ({model_type}): R² = {result.r_squared:.3f}",
            "body": f"The {model_type} model explains {result.r_squared * 100:.1f}% of variance. RMSE = {result.rmse:.4f}.",
            "next_steps": "Compare against other model types (exponential, logistic, power) to find the best fit.",
            "chart_guidance": "Scatter shows observed data. The fitted curve overlays the model prediction.",
        },
        "guide_observation": f"Nonlinear ({model_type}): R²={result.r_squared:.3f}, RMSE={result.rmse:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Best Subsets Regression
# =============================================================================


def forge_best_subsets(df, config):
    """Best subsets regression via forgestat."""
    from forgestat.regression.best_subsets import best_subsets

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = best_subsets(X, y, feature_names=predictors)

    # Build summary from best models
    sections = [
        (
            "Design",
            [
                ("Response", response),
                ("Candidate predictors", ", ".join(predictors)),
                ("N", str(len(y))),
            ],
        ),
    ]

    if result.best_aic:
        sections.append(
            (
                "Best AIC Model",
                [
                    ("Predictors", ", ".join(result.best_aic.features)),
                    ("AIC", f"{result.best_aic.aic:.4f}"),
                    ("R²", f"{result.best_aic.r_squared:.4f}"),
                    ("Adj R²", f"{result.best_aic.adj_r_squared:.4f}"),
                ],
            )
        )
    if result.best_bic:
        sections.append(
            (
                "Best BIC Model",
                [
                    ("Predictors", ", ".join(result.best_bic.features)),
                    ("BIC", f"{result.best_bic.bic:.4f}"),
                    ("R²", f"{result.best_bic.r_squared:.4f}"),
                    ("Adj R²", f"{result.best_bic.adj_r_squared:.4f}"),
                ],
            )
        )
    if result.best_adj_r2:
        sections.append(
            (
                "Best Adj R² Model",
                [
                    ("Predictors", ", ".join(result.best_adj_r2.features)),
                    ("Adj R²", f"{result.best_adj_r2.adj_r_squared:.4f}"),
                ],
            )
        )

    best = result.best_bic or result.best_aic or result.best_adj_r2
    best_feats = ", ".join(best.features) if best else "none"

    return {
        "plots": [],
        "statistics": {
            "best_aic": {
                "features": result.best_aic.features,
                "aic": round(result.best_aic.aic, 4),
                "r_squared": round(result.best_aic.r_squared, 4),
            }
            if result.best_aic
            else None,
            "best_bic": {
                "features": result.best_bic.features,
                "bic": round(result.best_bic.bic, 4),
                "r_squared": round(result.best_bic.r_squared, 4),
            }
            if result.best_bic
            else None,
            "best_adj_r2": {
                "features": result.best_adj_r2.features,
                "adj_r_squared": round(result.best_adj_r2.adj_r_squared, 4),
            }
            if result.best_adj_r2
            else None,
            "n_subsets_evaluated": len(result.all_subsets),
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary("BEST SUBSETS REGRESSION", sections),
        "narrative": {
            "verdict": f"Best model uses {len(best.features) if best else 0} of {len(predictors)} predictors",
            "body": f"Best BIC model: <strong>{best_feats}</strong>." if best else "No valid model found.",
            "next_steps": "Use the recommended subset as starting predictors. Validate with out-of-sample data.",
            "chart_guidance": "",
        },
        "guide_observation": f"Best subsets: {best_feats} (of {len(predictors)} candidates).",
        "diagnostics": [],
    }


# =============================================================================
# GLM (Generalized Linear Model)
# =============================================================================


def forge_glm(df, config):
    """Generalized linear model via forgestat."""
    from forgestat.regression.glm import glm

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))
    family = config.get("family", "gaussian")

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = glm(X, y, feature_names=predictors, family=family)

    coef_items = [
        (name, f"{val:.4f} (p = {_pval_str(result.p_values.get(name, 1))})")
        for name, val in result.coefficients.items()
    ]
    sig_preds = [k for k in predictors if result.p_values.get(k, 1) < 0.05]

    return {
        "plots": [],
        "statistics": {
            "family": result.family,
            "coefficients": result.coefficients,
            "std_errors": result.std_errors,
            "p_values": result.p_values,
            "deviance": round(result.deviance, 4),
            "aic": round(result.aic, 4),
            "n": result.n,
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"GLM ({family.upper()})",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Predictors", ", ".join(predictors)),
                        ("Family", family),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                (
                    "Model Fit",
                    [
                        ("Deviance", f"{result.deviance:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"GLM ({family}): {len(sig_preds)} significant predictors, AIC = {result.aic:.1f}",
            "body": (
                f"Significant: <strong>{', '.join(sig_preds)}</strong>."
                if sig_preds
                else "No predictors reached significance."
            ),
            "next_steps": "Consider alternative link functions or families if deviance is large relative to df.",
            "chart_guidance": "",
        },
        "guide_observation": f"GLM ({family}): {len(sig_preds)} significant, AIC={result.aic:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Ordinal Logistic Regression
# =============================================================================


def forge_ordinal_logistic(df, config):
    """Ordinal logistic regression via forgestat."""
    from forgestat.regression.glm import ordinal_logistic

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = data_clean[response].values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = ordinal_logistic(X, y, feature_names=predictors)

    coef_items = [(name, f"{val:.4f}") for name, val in result.coefficients.items()]

    return {
        "plots": [],
        "statistics": {
            "coefficients": result.coefficients,
            "thresholds": result.thresholds,
            "categories": result.categories,
            "n": result.n,
            "log_likelihood": round(result.log_likelihood, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ORDINAL LOGISTIC REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} ({len(result.categories)} ordered levels)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                ("Thresholds", [(f"τ{i + 1}", f"{t:.4f}") for i, t in enumerate(result.thresholds)]),
                ("Fit", [("Log-likelihood", f"{result.log_likelihood:.4f}")]),
            ],
        ),
        "narrative": {
            "verdict": f"Ordinal logistic: {len(result.categories)} response levels, {len(predictors)} predictors",
            "body": f"Positive coefficients increase probability of higher categories. Response levels: {', '.join(str(c) for c in result.categories)}.",
            "next_steps": "Check proportional odds assumption. Consider nominal logistic if assumption is violated.",
            "chart_guidance": "",
        },
        "guide_observation": f"Ordinal logistic: {len(result.categories)} levels, LL={result.log_likelihood:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Orthogonal Regression (Deming)
# =============================================================================


def forge_orthogonal_regression(df, config):
    """Orthogonal (Deming) regression via forgestat."""
    from forgestat.regression.glm import orthogonal_regression
    from forgeviz.charts.scatter import scatter

    x_var = config.get("x_var") or config.get("var1")
    y_var = config.get("y_var") or config.get("var2")
    error_ratio = float(config.get("error_ratio", 1.0))

    if not x_var or not y_var:
        raise ValueError("Orthogonal regression requires x_var and y_var")

    x_data = pd.to_numeric(df[x_var], errors="coerce").dropna()
    y_data = pd.to_numeric(df[y_var], errors="coerce").dropna()
    common = x_data.index.intersection(y_data.index)
    x_vals = x_data.loc[common].values
    y_vals = y_data.loc[common].values

    result = orthogonal_regression(x_vals, y_vals, error_ratio=error_ratio)

    chart = scatter(
        x=x_vals.tolist(),
        y=y_vals.tolist(),
        title=f"Orthogonal Regression: {y_var} vs {x_var}",
        x_label=x_var,
        y_label=y_var,
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "slope_orthogonal": round(result.slope, 6),
            "intercept_orthogonal": round(result.intercept, 6),
            "slope_ols": round(result.slope_ols, 6),
            "intercept_ols": round(result.intercept_ols, 6),
            "error_ratio": result.error_ratio,
            "n": len(x_vals),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ORTHOGONAL (DEMING) REGRESSION",
            [
                (
                    "Design",
                    [
                        ("X", x_var),
                        ("Y", y_var),
                        ("Error ratio (σ²_x/σ²_y)", f"{error_ratio}"),
                        ("N", str(len(x_vals))),
                    ],
                ),
                (
                    "Orthogonal Fit",
                    [
                        ("Slope", f"{result.slope:.6f}"),
                        ("Intercept", f"{result.intercept:.6f}"),
                    ],
                ),
                (
                    "OLS Comparison",
                    [
                        ("Slope (OLS)", f"{result.slope_ols:.6f}"),
                        ("Intercept (OLS)", f"{result.intercept_ols:.6f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Orthogonal slope = {result.slope:.4f} (OLS: {result.slope_ols:.4f})",
            "body": (
                f"Orthogonal regression accounts for measurement error in both X and Y. "
                f"Slope difference from OLS: {abs(result.slope - result.slope_ols):.4f}. "
                + (
                    "Substantial difference — measurement error in X biases OLS."
                    if abs(result.slope - result.slope_ols) > 0.05 * abs(result.slope_ols)
                    else "OLS and orthogonal fits are similar — X measurement error is minor."
                )
            ),
            "next_steps": "Use orthogonal regression when both variables have measurement error (e.g., method comparison studies).",
            "chart_guidance": "Scatter plot shows the data. Orthogonal regression minimizes perpendicular distances, not vertical.",
        },
        "guide_observation": f"Orthogonal regression: slope={result.slope:.4f} vs OLS={result.slope_ols:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nominal Logistic Regression
# =============================================================================


def forge_nominal_logistic(df, config):
    """Nominal (multinomial) logistic regression via sklearn."""
    from forgeviz.charts.generic import bar
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y_raw = data_clean[response]
    classes = sorted(y_raw.unique().tolist(), key=str)

    if len(classes) < 2:
        raise ValueError(f"Response '{response}' has only {len(classes)} unique value(s). Need at least 2.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    X = data_clean[predictors].copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").values

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = float((y_pred == y).mean())
    cm = confusion_matrix(y, y_pred)

    # Chart: accuracy by class
    per_class_acc = []
    for i, cn in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_acc.append(round(float(cm[i, i] / cm[i].sum()), 4))
        else:
            per_class_acc.append(0.0)

    chart = bar(
        categories=[str(c) for c in class_names],
        values=per_class_acc,
        title="Per-Class Accuracy",
        y_label="Accuracy",
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "accuracy": round(accuracy, 4),
            "classes": [str(c) for c in class_names],
            "n_classes": len(class_names),
            "per_class_accuracy": {str(c): a for c, a in zip(class_names, per_class_acc)},
            "confusion_matrix": cm.tolist(),
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "NOMINAL LOGISTIC REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} ({len(class_names)} categories)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(len(y))),
                    ],
                ),
                (
                    "Overall",
                    [
                        ("Accuracy", f"{accuracy:.1%}"),
                    ],
                ),
                ("Per-class Accuracy", [(str(c), f"{a:.1%}") for c, a in zip(class_names, per_class_acc)]),
            ],
        ),
        "narrative": {
            "verdict": f"Nominal logistic: {accuracy:.1%} overall accuracy across {len(class_names)} categories",
            "body": (
                f"Multinomial model classifying {response} into {len(class_names)} categories. "
                f"Weakest class: {class_names[per_class_acc.index(min(per_class_acc))]} ({min(per_class_acc):.1%})."
            ),
            "next_steps": "Check confusion matrix for systematic misclassifications. Consider ordinal logistic if categories are ordered.",
            "chart_guidance": "Bar chart shows per-class classification accuracy.",
        },
        "guide_observation": f"Nominal logistic: {accuracy:.1%} accuracy, {len(class_names)} classes.",
        "diagnostics": [],
    }


# =============================================================================
# Registry: maps analysis_id → forge handler
# =============================================================================

FORGE_HANDLERS = {
    # Parametric
    "ttest": forge_ttest,
    "ttest2": forge_ttest2,
    "paired_t": forge_paired_t,
    "anova": forge_anova,
    "anova2": forge_anova2,
    "chi2": forge_chi2,
    "correlation": forge_correlation,
    "descriptive": forge_descriptive,
    "normality": forge_normality,
    "equivalence": forge_equivalence,
    "variance_test": forge_variance_test,
    "f_test": forge_f_test,
    "prop_1sample": forge_prop_1sample,
    "prop_2sample": forge_prop_2sample,
    "repeated_measures_anova": forge_repeated_measures_anova,
    "sign_test": forge_sign_test,
    "split_plot_anova": forge_split_plot_anova,
    # Nonparametric
    "mann_whitney": forge_mann_whitney,
    "wilcoxon": forge_wilcoxon,
    "kruskal": forge_kruskal,
    "friedman": forge_friedman,
    "mood_median": forge_mood_median,
    "spearman": forge_spearman,
    # Post-hoc
    "tukey_hsd": forge_tukey_hsd,
    "games_howell": forge_games_howell,
    "dunn": forge_dunn,
    "bonferroni_test": forge_bonferroni,
    "scheffe_test": forge_scheffe,
    # Regression
    "regression": forge_regression,
    "logistic": forge_logistic,
    "stepwise": forge_stepwise,
    "robust_regression": forge_robust_regression,
    "poisson_regression": forge_poisson_regression,
    "nonlinear_regression": forge_nonlinear_regression,
    "best_subsets": forge_best_subsets,
    "glm": forge_glm,
    "ordinal_logistic": forge_ordinal_logistic,
    "orthogonal_regression": forge_orthogonal_regression,
    "nominal_logistic": forge_nominal_logistic,
    # Power & sample size
    "power_z": forge_power_z,
    "power_ttest": forge_power_ttest,
    "power_anova": forge_power_anova,
    "power_1prop": forge_power_proportion,
    "power_equivalence": forge_power_equivalence,
    "sample_size_ci": forge_sample_size_ci,
}


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
