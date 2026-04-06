"""Forge-backed advanced ANOVA and post-hoc handlers.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging

import numpy as np
import pandas as pd

from .forge_stats import (
    _alpha,
    _assumption_to_dict,
    _assumptions_dict,
    _col,
    _pval_str,
    _rich_summary,
    _to_chart,
)

logger = logging.getLogger(__name__)


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
# Dunnett's Test (vs Control)
# =============================================================================


def forge_dunnett(df, config):
    """Dunnett's test via forgestat."""
    from forgestat.posthoc.comparisons import dunnett
    from forgeviz.charts.generic import bar

    response = config.get("response") or config.get("var")
    factor = config.get("factor") or config.get("group_var")
    control = config.get("control")
    alpha = _alpha(config)

    if not response or not factor:
        raise ValueError("Dunnett's test requires response and factor")

    data = df[[response, factor]].dropna()
    levels = sorted(data[factor].unique().tolist(), key=str)

    if control is None or control not in levels:
        control = levels[0]

    control_data = pd.to_numeric(data[data[factor] == control][response], errors="coerce").dropna().values
    treatments = [lev for lev in levels if lev != control]
    treatment_data = [
        pd.to_numeric(data[data[factor] == lev][response], errors="coerce").dropna().values for lev in treatments
    ]

    result = dunnett(
        control_data,
        *treatment_data,
        control_name=str(control),
        treatment_names=[str(t) for t in treatments],
        alpha=alpha,
    )

    # Build comparisons from result
    comparisons = []
    for comp in result.comparisons:
        comparisons.append(
            {
                "treatment": comp.group2,
                "control": comp.group1,
                "mean_diff": round(comp.mean_diff, 4),
                "p_value": comp.p_value,
                "reject": comp.significant or comp.reject,
            }
        )

    n_sig = sum(1 for c in comparisons if c["reject"])

    # Bar chart of differences
    chart = bar(
        categories=[c["treatment"] for c in comparisons],
        values=[c["mean_diff"] for c in comparisons],
        title=f"Dunnett's Test — Difference from Control ({control})",
        y_label=f"Difference from {control}",
    )
    plots = [_to_chart(chart)]

    comp_items = [
        (c["treatment"], f"diff = {c['mean_diff']:.4f}, p = {_pval_str(c['p_value'])}" + (" *" if c["reject"] else ""))
        for c in comparisons
    ]

    return {
        "plots": plots,
        "statistics": {"comparisons": comparisons, "control": str(control)},
        "assumptions": {},
        "summary": _rich_summary(
            "DUNNETT'S TEST (vs Control)",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Factor", factor),
                        ("Control group", str(control)),
                        ("N (control)", str(len(control_data))),
                    ],
                ),
                ("Comparisons", comp_items),
                ("Result", [(f"{n_sig}/{len(comparisons)}", "treatments differ from control")]),
            ],
        ),
        "narrative": {
            "verdict": f"Dunnett's: {n_sig} of {len(comparisons)} treatments differ from control ({control})",
            "body": (
                f"Comparing all treatments against <strong>{control}</strong>. "
                + (f"{n_sig} show significant differences." if n_sig else "No treatments differ from control.")
            ),
            "next_steps": "Dunnett's test is more powerful than Tukey when comparing only to a control group.",
            "chart_guidance": "Bar chart shows mean difference from control for each treatment. Significant differences are highlighted.",
        },
        "guide_observation": f"Dunnett's vs {control}: {n_sig}/{len(comparisons)} treatments differ.",
        "diagnostics": [],
    }


# =============================================================================
# Hsu's MCB (Multiple Comparisons with the Best)
# =============================================================================


def forge_hsu_mcb(df, config):
    """Hsu's MCB — which groups could be the best."""
    from scipy import stats as sp_stats

    response = config.get("response") or config.get("var")
    factor = config.get("factor") or config.get("group_var")
    alpha = _alpha(config)
    direction = config.get("direction", "max")

    if not response or not factor:
        raise ValueError("Hsu's MCB requires response and factor")

    data = df[[response, factor]].dropna()
    groups = sorted(data[factor].unique(), key=str)
    k = len(groups)

    group_data = {
        str(g): pd.to_numeric(data[data[factor] == g][response], errors="coerce").dropna().values for g in groups
    }
    group_means = {g: float(np.mean(v)) for g, v in group_data.items()}
    group_ns = {g: len(v) for g, v in group_data.items()}
    n_total = sum(group_ns.values())

    ss_w = sum(float(np.sum((v - np.mean(v)) ** 2)) for v in group_data.values())
    df_w = n_total - k
    mse = ss_w / df_w if df_w > 0 else 0

    t_crit = sp_stats.t.ppf(1 - alpha / (2 * (k - 1)), df_w) if df_w > 0 else 2.0

    best_group = max(group_means, key=group_means.get) if direction == "max" else min(group_means, key=group_means.get)

    comparisons = []
    for g in group_data:
        mean_g = group_means[g]
        mean_best = group_means[best_group]
        diff = mean_g - mean_best if direction == "max" else mean_best - mean_g
        se = np.sqrt(mse * (1 / group_ns[g] + 1 / group_ns[best_group])) if mse > 0 else 0
        lower = diff - t_crit * se
        upper = diff + t_crit * se

        if g == best_group:
            could_be_best = True
            lower_mcb, upper_mcb = 0.0, max(0, upper)
        else:
            lower_mcb = min(0, lower)
            upper_mcb = min(0, upper)
            could_be_best = upper_mcb >= 0

        comparisons.append(
            {
                "group": g,
                "mean": round(float(mean_g), 4),
                "diff_from_best": round(float(diff), 4),
                "lower": round(float(lower_mcb), 4),
                "upper": round(float(upper_mcb), 4),
                "could_be_best": could_be_best,
                "is_best": g == best_group,
            }
        )

    n_could = sum(1 for c in comparisons if c["could_be_best"])

    comp_items = [
        (
            c["group"],
            f"mean={c['mean']:.4f}, diff={c['diff_from_best']:.4f}, CI=[{c['lower']:.4f}, {c['upper']:.4f}] {'← COULD BE BEST' if c['could_be_best'] else ''}",
        )
        for c in comparisons
    ]

    return {
        "plots": [],
        "statistics": {
            "comparisons": comparisons,
            "best_group": best_group,
            "direction": direction,
            "mse": round(mse, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "HSU'S MCB (MULTIPLE COMPARISONS WITH THE BEST)",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Factor", f"{factor} ({k} groups)"),
                        ("Direction", "Higher is better" if direction == "max" else "Lower is better"),
                        ("Best group", f"{best_group} (mean = {group_means[best_group]:.4f})"),
                    ],
                ),
                ("MCB Intervals", comp_items),
                ("Result", [(f"{n_could}/{k}", "groups could be the best")]),
            ],
        ),
        "narrative": {
            "verdict": f"Hsu's MCB: {n_could} of {k} groups could be the best",
            "body": f"Current best: <strong>{best_group}</strong> ({direction}). {n_could} group{'s' if n_could != 1 else ''} cannot be ruled out as best.",
            "next_steps": "MCB identifies which groups could plausibly be the best. Narrow the field with more data.",
            "chart_guidance": "Groups with CI including zero could be the best. Groups entirely below zero are eliminated.",
        },
        "guide_observation": f"Hsu's MCB: {n_could}/{k} groups could be best. Best={best_group}.",
        "diagnostics": [],
    }


# =============================================================================
# Main Effects Plot
# =============================================================================


def forge_main_effects(df, config):
    """Main effects plot for DOE analysis."""
    from forgeviz.charts.generic import multi_line

    response = config.get("response") or config.get("var")
    factors = config.get("factors", [])

    if not response or not factors:
        raise ValueError("Main effects requires response and factors")

    y = pd.to_numeric(df[response], errors="coerce").dropna()
    grand_mean = float(y.mean())

    plots = []
    effect_ranges = []
    effect_items = []

    for factor in factors:
        factor_means = df.groupby(factor)[response].apply(lambda x: float(pd.to_numeric(x, errors="coerce").mean()))
        levels = [str(lv) for lv in factor_means.index]
        means = list(factor_means.values)

        chart = multi_line(
            x=levels,
            series={"Mean": means, "Grand Mean": [grand_mean] * len(levels)},
            title=f"Main Effect: {factor}",
            x_label=factor,
            y_label=f"Mean {response}",
            show_markers=True,
        )
        plots.append(_to_chart(chart))

        f_range = max(means) - min(means)
        effect_ranges.append((factor, f_range))

        for lv, m in zip(levels, means):
            effect = m - grand_mean
            effect_items.append((f"{factor}={lv}", f"{m:.4f} (effect: {effect:+.4f})"))

    effect_ranges.sort(key=lambda x: x[1], reverse=True)
    top_factor, top_range = effect_ranges[0]

    return {
        "plots": plots,
        "statistics": {
            "grand_mean": round(grand_mean, 4),
            "effects": {f: round(r, 4) for f, r in effect_ranges},
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "MAIN EFFECTS PLOT",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Factors", ", ".join(factors)),
                        ("Grand Mean", f"{grand_mean:.4f}"),
                    ],
                ),
                ("Effects (deviation from grand mean)", effect_items),
                ("Largest Effect", [(top_factor, f"range = {top_range:.4f}")]),
            ],
        ),
        "narrative": {
            "verdict": f"Main Effects — {top_factor} has the largest effect (range = {top_range:.4f})",
            "body": (
                f"Across {len(factors)} factors, <strong>{top_factor}</strong> produces the widest swing "
                f"in mean {response} ({top_range:.4f}). Grand mean = {grand_mean:.4f}."
                + (
                    f" Second: {effect_ranges[1][0]} (range = {effect_ranges[1][1]:.4f})."
                    if len(effect_ranges) > 1
                    else ""
                )
            ),
            "next_steps": "Steep slopes = strong effects, flat lines = negligible. Run interaction plot to check independence.",
            "chart_guidance": "Each panel shows how one factor's levels shift the response mean. Dashed line is grand mean.",
        },
        "guide_observation": f"Main effects: {top_factor} has largest effect (range={top_range:.4f}).",
        "diagnostics": [],
    }


# =============================================================================
# Interaction Plot
# =============================================================================


def forge_interaction(df, config):
    """Interaction plot for two factors."""
    from forgeviz.charts.generic import multi_line

    response = config.get("response") or config.get("var")
    factor1 = config.get("factor1")
    factor2 = config.get("factor2")

    if not all([response, factor1, factor2]):
        raise ValueError("Interaction plot requires response, factor1, and factor2")

    # Calculate cell means
    interaction_means = (
        df.groupby([factor1, factor2])[response]
        .apply(lambda x: float(pd.to_numeric(x, errors="coerce").mean()))
        .unstack()
    )

    levels1 = [str(lv) for lv in interaction_means.index]
    levels2 = [str(lv) for lv in interaction_means.columns]

    series = {}
    for lev2 in interaction_means.columns:
        series[str(lev2)] = [round(float(v), 4) for v in interaction_means[lev2].values]

    chart = multi_line(
        x=levels1,
        series=series,
        title=f"Interaction: {factor1} × {factor2}",
        x_label=factor1,
        y_label=f"Mean {response}",
        show_markers=True,
    )
    plots = [_to_chart(chart)]

    # Simple interaction detection
    has_interaction = False
    if len(levels2) >= 2 and len(levels1) >= 2:
        slopes = []
        for lev2 in interaction_means.columns:
            vals = interaction_means[lev2].values
            slope = (vals[-1] - vals[0]) / (len(vals) - 1) if len(vals) > 1 else 0
            slopes.append(float(slope))
        slope_diff = max(slopes) - min(slopes)
        mean_slope = float(np.mean(slopes))
        has_interaction = slope_diff > 0.1 * abs(mean_slope) if mean_slope != 0 else slope_diff > 0.1

    if has_interaction:
        verdict = f"Interaction detected — {factor1} × {factor2}"
        body = (
            f"The effect of <strong>{factor1}</strong> on {response} depends on <strong>{factor2}</strong> "
            f"(non-parallel lines). Optimize jointly, not independently."
        )
        nxt = "Factors interact — optimize jointly. Confirm with ANOVA interaction term."
    else:
        verdict = f"No strong interaction — {factor1} × {factor2}"
        body = (
            f"Lines are approximately parallel — <strong>{factor1}</strong> and <strong>{factor2}</strong> "
            f"act independently on {response}."
        )
        nxt = "Factors appear independent — optimize each separately."

    return {
        "plots": plots,
        "statistics": {
            "cell_means": {str(k): round(float(v), 4) for k, v in interaction_means.stack().items()},
            "has_interaction": has_interaction,
        },
        "assumptions": {},
        "summary": _rich_summary(
            "INTERACTION PLOT",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("X-axis factor", factor1),
                        ("Trace factor", factor2),
                    ],
                ),
                (
                    "Cell Means",
                    [
                        (f"{factor1}={l1}, {factor2}={l2}", f"{interaction_means.loc[l1_raw, l2_raw]:.4f}")
                        for l1, l1_raw in zip(levels1, interaction_means.index)
                        for l2, l2_raw in zip(levels2, interaction_means.columns)
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": nxt,
            "chart_guidance": "Parallel lines = no interaction. Crossing or diverging lines = interaction.",
        },
        "guide_observation": f"Interaction {factor1}×{factor2}: {'detected' if has_interaction else 'not detected'}.",
        "diagnostics": [],
    }


FORGE_ANOVA_HANDLERS = {
    "anova2": forge_anova2,
    "f_test": forge_f_test,
    "repeated_measures_anova": forge_repeated_measures_anova,
    "sign_test": forge_sign_test,
    "split_plot_anova": forge_split_plot_anova,
    "dunnett": forge_dunnett,
    "hsu_mcb": forge_hsu_mcb,
    "main_effects": forge_main_effects,
    "interaction": forge_interaction,
}
