"""DSW Exploratory — comparative analyses (chi2, proportions, Fisher, Poisson 2-sample, bootstrap)."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from ..common import (
    _bayesian_shadow,
    _check_outliers,
    _effect_magnitude,
    _evidence_grade,
    _narrative,
    _practical_block,
)

logger = logging.getLogger(__name__)


def run_chi2(df, config):
    """Chi-Square Test for Independence."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    row_var = config.get("row_var") or config.get("var1") or config.get("var")
    col_var = config.get("col_var") or config.get("var2") or config.get("group_var")

    # Create contingency table
    contingency = pd.crosstab(df[row_var], df[col_var])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        result["summary"] = (
            f"<<COLOR:danger>>Chi-square requires at least a 2×2 contingency table.<</COLOR>>\n\n"
            f"Got {contingency.shape[0]} row(s) × {contingency.shape[1]} column(s).\n"
            f"'{row_var}' has {contingency.shape[0]} unique value(s), "
            f"'{col_var}' has {contingency.shape[1]} unique value(s).\n\n"
            "Both variables must have at least 2 distinct values."
        )
        result["guide_observation"] = f"Chi-square test not applicable: {row_var} or {col_var} has fewer than 2 levels."
        return result
    chi2, pval, dof, expected = stats.chi2_contingency(contingency)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>CHI-SQUARE TEST FOR INDEPENDENCE<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Row Variable:<</COLOR>> {row_var}\n"
    summary += f"<<COLOR:highlight>>Column Variable:<</COLOR>> {col_var}\n\n"

    summary += "<<COLOR:accent>>── Contingency Table (Observed) ──<</COLOR>>\n"
    summary += contingency.to_string() + "\n\n"

    # Cramér's V effect size
    n_obs = contingency.values.sum()
    min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
    cramers_v = np.sqrt(chi2 / (n_obs * min_dim)) if (n_obs > 0 and min_dim > 0) else 0.0
    v_label, v_meaningful = _effect_magnitude(cramers_v, "cramers_v")

    summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
    summary += f"  Chi-square statistic: {chi2:.4f}\n"
    summary += f"  Degrees of freedom: {dof}\n"
    summary += f"  p-value: {pval:.4f}\n"
    summary += f"  Cramér's V: {cramers_v:.3f} ({v_label} association)\n"
    if contingency.shape == (2, 2):
        _a, _b, _c, _d = (
            contingency.iloc[0, 0],
            contingency.iloc[0, 1],
            contingency.iloc[1, 0],
            contingency.iloc[1, 1],
        )
        if min(_a, _b, _c, _d) > 0:
            _or = (_a * _d) / (_b * _c)
            _log_se = np.sqrt(1 / _a + 1 / _b + 1 / _c + 1 / _d)
            _or_lo, _or_hi = np.exp(np.log(_or) - 1.96 * _log_se), np.exp(np.log(_or) + 1.96 * _log_se)
            summary += f"  Odds Ratio: {_or:.3f}, 95% CI [{_or_lo:.3f}, {_or_hi:.3f}]\n"
    summary += "\n"

    if pval < 0.05:
        summary += "<<COLOR:good>>Variables are significantly associated (p < 0.05)<</COLOR>>"
    else:
        summary += "<<COLOR:text>>No significant association found (p >= 0.05)<</COLOR>>"

    summary += _practical_block(
        "Cramér's V",
        cramers_v,
        "cramers_v",
        pval,
        context=f"The association between '{row_var}' and '{col_var}' is {v_label}. V=0 means no association, V=1 means perfect association.",
    )

    result["summary"] = summary
    obs_parts = [f"Chi-square test: χ²={chi2:.4f}, p={pval:.4f}, Cramér's V={cramers_v:.3f} ({v_label})"]
    if pval < 0.05 and v_meaningful:
        obs_parts.append(f"'{row_var}' and '{col_var}' are meaningfully associated.")
    elif pval < 0.05:
        obs_parts.append("Significant but weak association.")
    else:
        obs_parts.append("No significant association.")
    result["guide_observation"] = " ".join(obs_parts)

    # Narrative
    # Find the cell with largest standardized residual for specific insight
    std_resid = (contingency.values - expected) / np.sqrt(np.maximum(expected, 1))
    max_idx = np.unravel_index(np.abs(std_resid).argmax(), std_resid.shape)
    max_row_label = contingency.index[max_idx[0]]
    max_col_label = contingency.columns[max_idx[1]]
    max_direction = "over" if std_resid[max_idx] > 0 else "under"
    if pval < 0.05 and v_meaningful:
        verdict = f"{v_label.title()} association between {row_var} and {col_var}"
        body = f"Cram&eacute;r's V = {cramers_v:.3f} ({v_label}). The most notable pattern: <strong>{max_row_label}</strong> is {max_direction}-represented in <strong>{max_col_label}</strong> relative to what independence would predict."
        nexts = "Examine the cells with the largest residuals to understand the pattern driving the association."
    elif pval < 0.05:
        verdict = f"Weak but significant association between {row_var} and {col_var}"
        body = f"The variables are statistically associated (p = {pval:.4f}), but the effect is {v_label} (V = {cramers_v:.3f}). Largest departure: {max_row_label} &times; {max_col_label}."
        nexts = "The association is real but may not be strong enough to act on."
    else:
        verdict = f"No significant association between {row_var} and {col_var}"
        body = f"The chi-square test does not detect a meaningful relationship (p = {pval:.4f}, V = {cramers_v:.3f}). These variables appear to be independent."
        nexts = None
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nexts,
        chart_guidance="The grouped bar chart shows observed counts by category. Bars that deviate from the overall pattern indicate cells driving the association.",
    )

    result["statistics"] = {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(pval),
        "cramers_v": float(cramers_v),
        "effect_size_label": v_label,
        "n": int(n_obs),
    }

    # Interactive Power Explorer metadata
    # For chi-square, use Cramér's V as effect size, dof for power calc
    result["power_explorer"] = {
        "test_type": "chi2",
        "observed_effect": float(cramers_v),
        "observed_std": 1.0,
        "observed_n": int(n_obs),
        "alpha": 0.05,
        "cohens_d": float(cramers_v),
        "dof": int(dof),
    }

    # ── Diagnostics ──
    diagnostics = []
    # Expected cell count check
    _low_expected = int(np.sum(expected < 5))
    _pct_low = _low_expected / expected.size * 100 if expected.size > 0 else 0
    if _pct_low > 20:
        diagnostics.append(
            {
                "level": "error",
                "title": f"{_low_expected} cells ({_pct_low:.0f}%) have expected count < 5",
                "detail": "Chi-square approximation is unreliable. Consider Fisher's exact test or collapsing categories.",
                "action": (
                    {
                        "label": "Run Fisher Exact",
                        "type": "stats",
                        "analysis": "fisher_exact",
                        "config": {"row_var": row_var, "col_var": col_var},
                    }
                    if contingency.shape == (2, 2)
                    else None
                ),
            }
        )
    elif _low_expected > 0:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"{_low_expected} cells have expected count < 5",
                "detail": "Some expected counts are low. Results may be approximate.",
            }
        )
    # Effect size emphasis
    if cramers_v >= 0.3 and pval < 0.05:
        diagnostics.append(
            {
                "level": "info",
                "title": f"Meaningful association (Cram\u00e9r's V = {cramers_v:.3f})",
                "detail": f"The relationship between {row_var} and {col_var} is strong enough to be practically relevant.",
            }
        )
    elif cramers_v < 0.1 and pval < 0.05:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Significant but negligible association (V = {cramers_v:.3f})",
                "detail": "Statistical significance with trivial effect. Large sample sizes can detect meaningless associations.",
            }
        )
    # Clean None actions
    diagnostics = [d for d in diagnostics if d is not None]
    for d in diagnostics:
        if d.get("action") is None and "action" in d:
            del d["action"]
    result["diagnostics"] = diagnostics

    # --- Bayesian Insurance ---
    try:
        _shadow = _bayesian_shadow("chi2", contingency=contingency.values, chi2_stat=chi2, dof=dof, n_obs=n_obs)
        if _shadow:
            result["bayesian_shadow"] = _shadow
        _grade = _evidence_grade(
            pval,
            bf10=_shadow.get("bf10") if _shadow else None,
            effect_magnitude=v_label,
        )
        if _grade:
            result["evidence_grade"] = _grade
    except Exception:
        pass

    # Heatmap of observed counts
    result["plots"].append(
        {
            "title": f"Contingency Table: {row_var} × {col_var}",
            "data": [
                {
                    "type": "heatmap",
                    "z": contingency.values.tolist(),
                    "x": contingency.columns.astype(str).tolist(),
                    "y": contingency.index.astype(str).tolist(),
                    "colorscale": "Blues",
                    "text": contingency.values.tolist(),
                    "texttemplate": "%{text}",
                    "textfont": {"size": 12},
                }
            ],
            "layout": {"height": 300},
        }
    )

    return result


def run_prop_2sample(df, config):
    """
    Two-Proportion Z-Test — compare proportions between two groups.
    Tests H₀: p₁ = p₂. Reports pooled Z, individual CIs, and difference CI.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    group_var = config.get("group_var") or config.get("var2") or config.get("factor")
    event = config.get("event")
    alt = config.get("alternative", "two-sided")
    alpha = 1 - float(config.get("conf", 95)) / 100

    data = df[[var, group_var]].dropna()
    groups = sorted(data[group_var].unique().tolist(), key=str)
    if len(groups) != 2:
        result["summary"] = f"Two-proportion test requires exactly 2 groups. Found {len(groups)}."
        return result

    g1 = data[data[group_var] == groups[0]][var]
    g2 = data[data[group_var] == groups[1]][var]
    n1, n2 = len(g1), len(g2)

    if event is not None and str(event) != "":
        x1 = int((g1.astype(str) == str(event)).sum())
        x2 = int((g2.astype(str) == str(event)).sum())
    else:
        x1 = int((g1 == 1).sum()) if g1.dtype in ["int64", "float64"] else int(g1.value_counts().iloc[0])
        x2 = int((g2 == 1).sum()) if g2.dtype in ["int64", "float64"] else int(g2.value_counts().iloc[0])

    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0
    p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2)) if (n1 > 0 and n2 > 0) else 1
    z_stat = (p1 - p2) / se_pooled if se_pooled > 0 else 0

    if alt == "greater":
        p_val = float(1 - stats.norm.cdf(z_stat))
    elif alt == "less":
        p_val = float(stats.norm.cdf(z_stat))
    else:
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # Difference CI (unpooled SE)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if (n1 > 0 and n2 > 0) else 0
    diff = p1 - p2
    ci_lo = diff - z_crit * se_diff
    ci_hi = diff + z_crit * se_diff

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>TWO-PROPORTION Z-TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {group_var}\n\n"

    summary += "<<COLOR:accent>>── Sample Results ──<</COLOR>>\n"
    summary += f"  {'Group':<15} {'N':>6} {'Events':>8} {'Proportion':>12}\n"
    summary += f"  {'─' * 45}\n"
    summary += f"  {str(groups[0]):<15} {n1:>6} {x1:>8} {p1:>12.4f}\n"
    summary += f"  {str(groups[1]):<15} {n2:>6} {x2:>8} {p2:>12.4f}\n\n"
    summary += f"<<COLOR:accent>>── Difference (p₁ − p₂) ──<</COLOR>> {diff:.4f}\n"
    summary += f"<<COLOR:text>>{100 * (1 - alpha):.0f}% CI for difference:<</COLOR>> ({ci_lo:.4f}, {ci_hi:.4f})\n\n"
    summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
    summary += f"  Z-statistic: {z_stat:.4f}\n"
    summary += f"  p-value: {p_val:.4f}\n\n"

    if p_val < alpha:
        summary += f"<<COLOR:good>>Proportions differ significantly (p < {alpha})<</COLOR>>"
    else:
        summary += f"<<COLOR:text>>No significant difference in proportions (p ≥ {alpha})<</COLOR>>"

    result["summary"] = summary

    # Side-by-side bar chart
    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": [str(groups[0]), str(groups[1])],
                    "y": [p1, p2],
                    "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                    "text": [f"{p1:.3f}", f"{p2:.3f}"],
                    "textposition": "outside",
                }
            ],
            "layout": {
                "title": "Proportions by Group",
                "yaxis": {
                    "title": "Proportion",
                    "range": [0, max(p1, p2) * 1.3 + 0.05],
                },
            },
        }
    )

    # Difference CI plot
    result["plots"].append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": [diff],
                    "y": ["p₁ − p₂"],
                    "mode": "markers",
                    "marker": {"size": 12, "color": "#4a9f6e"},
                    "error_x": {
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_hi - diff],
                        "arrayminus": [diff - ci_lo],
                        "color": "#5a6a5a",
                    },
                }
            ],
            "layout": {
                "title": f"Difference in Proportions ({100 * (1 - alpha):.0f}% CI)",
                "xaxis": {
                    "title": "p₁ − p₂",
                    "zeroline": True,
                    "zerolinecolor": "#e89547",
                    "zerolinewidth": 2,
                },
                "shapes": [
                    {
                        "type": "line",
                        "x0": 0,
                        "x1": 0,
                        "y0": -0.5,
                        "y1": 0.5,
                        "line": {"color": "#e89547", "dash": "dash"},
                    }
                ],
                "height": 200,
            },
        }
    )

    result["guide_observation"] = f"2-prop Z-test: p₁={p1:.4f} vs p₂={p2:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. " + (
        "Significant." if p_val < alpha else "Not significant."
    )
    result["statistics"] = {
        "n1": n1,
        "n2": n2,
        "x1": x1,
        "x2": x2,
        "p1": p1,
        "p2": p2,
        "difference": diff,
        "z_statistic": float(z_stat),
        "p_value": p_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "alternative": alt,
    }

    # Narrative
    if p_val < alpha:
        higher = str(groups[0]) if p1 > p2 else str(groups[1])
        verdict = f"Proportions differ significantly (p = {p_val:.4f})"
        body = (
            f"<strong>{groups[0]}</strong>: {p1:.4f} ({x1}/{n1}) vs <strong>{groups[1]}</strong>: {p2:.4f} ({x2}/{n2}). "
            f"Difference = {diff:.4f}, {100 * (1 - alpha):.0f}% CI ({ci_lo:.4f}, {ci_hi:.4f}). "
            f"<strong>{higher}</strong> has the higher rate."
        )
        nxt = "Investigate what differs between the groups. If these are defect rates, focus improvement on the higher-rate group."
    else:
        verdict = f"No significant difference in proportions (p = {p_val:.4f})"
        body = (
            f"<strong>{groups[0]}</strong>: {p1:.4f} ({x1}/{n1}) vs <strong>{groups[1]}</strong>: {p2:.4f} ({x2}/{n2}). "
            f"Difference = {diff:.4f}, {100 * (1 - alpha):.0f}% CI ({ci_lo:.4f}, {ci_hi:.4f}) includes zero."
        )
        nxt = "No evidence the groups differ. If you expected a difference, consider increasing sample sizes."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="Bars show group proportions. The CI plot shows whether the difference CI excludes zero (significant) or includes it (not significant).",
    )

    return result


def run_fisher_exact(df, config):
    """
    Fisher's Exact Test — exact test for 2×2 contingency tables.
    Preferred over chi-square when expected cell counts are small (<5).
    Reports odds ratio, exact p-value, and odds ratio CI.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var1 = config.get("var") or config.get("var1") or config.get("row_var")
    var2 = config.get("var2") or config.get("group_var") or config.get("col_var")
    alt = config.get("alternative", "two-sided")
    alpha = 1 - float(config.get("conf", 95)) / 100

    ct = pd.crosstab(df[var1], df[var2])
    if ct.shape != (2, 2):
        result["summary"] = (
            f"Fisher's exact test requires a 2×2 table. Got {ct.shape[0]}×{ct.shape[1]}. Ensure both variables have exactly 2 levels."
        )
        return result

    table = ct.values
    odds_ratio, p_val = stats.fisher_exact(table, alternative=alt)

    # Odds ratio CI via log method
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    if all(v > 0 for v in [a, b, c, d]):
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        or_ci_lo = np.exp(log_or - z_crit * se_log_or)
        or_ci_hi = np.exp(log_or + z_crit * se_log_or)
    else:
        or_ci_lo, or_ci_hi = 0, float("inf")

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>FISHER'S EXACT TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Row variable:<</COLOR>> {var1}\n"
    summary += f"<<COLOR:highlight>>Column variable:<</COLOR>> {var2}\n\n"

    summary += "<<COLOR:accent>>── 2×2 Contingency Table ──<</COLOR>>\n"
    summary += f"  {'':>15} {str(ct.columns[0]):>10} {str(ct.columns[1]):>10}\n"
    summary += f"  {str(ct.index[0]):>15} {a:>10} {b:>10}\n"
    summary += f"  {str(ct.index[1]):>15} {c:>10} {d:>10}\n\n"
    summary += "<<COLOR:accent>>── Results ──<</COLOR>>\n"
    summary += f"  Odds Ratio: {odds_ratio:.4f}\n"
    summary += f"  {100 * (1 - alpha):.0f}% CI: ({or_ci_lo:.4f}, {or_ci_hi:.4f})\n"
    summary += f"  p-value (exact): {p_val:.4f}\n\n"

    if p_val < alpha:
        summary += f"<<COLOR:good>>Significant association (p < {alpha}). Odds ratio ≠ 1.<</COLOR>>"
    else:
        summary += f"<<COLOR:text>>No significant association (p ≥ {alpha})<</COLOR>>"

    result["summary"] = summary

    # Mosaic-like stacked bar
    col_labels = [str(c) for c in ct.columns]
    row_labels = [str(r) for r in ct.index]
    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": col_labels,
                    "y": [int(table[0, 0]), int(table[0, 1])],
                    "name": row_labels[0],
                    "marker": {"color": "#4a9f6e"},
                },
                {
                    "type": "bar",
                    "x": col_labels,
                    "y": [int(table[1, 0]), int(table[1, 1])],
                    "name": row_labels[1],
                    "marker": {"color": "#4a90d9"},
                },
            ],
            "layout": {
                "title": "Contingency Table",
                "barmode": "stack",
                "yaxis": {"title": "Count"},
            },
        }
    )

    # Odds ratio forest-style plot
    if odds_ratio > 0 and or_ci_hi < 1e6:
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": [odds_ratio],
                        "y": ["OR"],
                        "mode": "markers",
                        "marker": {"size": 12, "color": "#4a9f6e"},
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [or_ci_hi - odds_ratio],
                            "arrayminus": [odds_ratio - or_ci_lo],
                            "color": "#5a6a5a",
                        },
                    }
                ],
                "layout": {
                    "title": f"Odds Ratio ({100 * (1 - alpha):.0f}% CI)",
                    "xaxis": {"title": "Odds Ratio", "type": "log"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": -0.5,
                            "y1": 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                    "height": 180,
                },
            }
        )

    result["guide_observation"] = f"Fisher's exact: OR={odds_ratio:.3f}, p={p_val:.4f}. " + (
        "Significant association." if p_val < alpha else "No association."
    )
    result["statistics"] = {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_val),
        "or_ci_lower": float(or_ci_lo),
        "or_ci_upper": float(or_ci_hi),
        "table": table.tolist(),
        "alternative": alt,
    }

    if p_val < alpha:
        verdict = f"Significant association (OR = {odds_ratio:.2f}, p = {p_val:.4f})"
        _dir = "higher" if odds_ratio > 1 else "lower"
        body = f"The odds of {ct.index[0]} are <strong>{odds_ratio:.2f}x</strong> {_dir} in {ct.columns[0]} vs {ct.columns[1]}. CI: ({or_ci_lo:.2f}, {or_ci_hi:.2f})."
        nxt = "Investigate what drives the association. Consider confounders that may explain the relationship."
    else:
        verdict = f"No significant association (p = {p_val:.4f})"
        body = f"OR = {odds_ratio:.2f}, CI: ({or_ci_lo:.2f}, {or_ci_hi:.2f}) — the CI includes 1, indicating no significant odds difference."
        nxt = "No evidence of association. If expected, increase sample size for more power."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="OR > 1 means higher odds in the first group. CI not crossing 1 (dashed line) = significant.",
    )

    return result


def run_poisson_2sample(df, config):
    """
    Two-Sample Poisson Rate Test — compare event rates between two groups.
    Modes:
      1) Two count columns + optional exposure per group
      2) Response column + grouping factor (auto-sum per group)
    Reports exact conditional test, rate ratio with CI.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var1 = config.get("var1") or config.get("var")
    var2 = config.get("var2")
    alpha = 1 - float(config.get("conf", 95)) / 100
    conf_pct = float(config.get("conf", 95))
    alt = config.get("alternative", "two-sided")

    if config.get("data_format") == "factor" or config.get("group_var"):
        response_col = config.get("response") or var1
        factor_col = config.get("group_var") or config.get("factor") or var2
        data_clean = df[[response_col, factor_col]].dropna()
        levels = sorted(data_clean[factor_col].unique().tolist(), key=str)
        if len(levels) != 2:
            result["summary"] = f"Two-sample Poisson test requires exactly 2 groups. Found {len(levels)}."
            return result
        g1 = data_clean[data_clean[factor_col] == levels[0]][response_col]
        g2 = data_clean[data_clean[factor_col] == levels[1]][response_col]
        c1, c2 = float(g1.sum()), float(g2.sum())
        e1 = float(config.get("exposure1", len(g1)))
        e2 = float(config.get("exposure2", len(g2)))
        label1, label2 = str(levels[0]), str(levels[1])
    else:
        col1 = df[var1].dropna()
        col2 = df[var2].dropna()
        c1, c2 = float(col1.sum()), float(col2.sum())
        e1 = float(config.get("exposure1", len(col1)))
        e2 = float(config.get("exposure2", len(col2)))
        label1, label2 = str(var1), str(var2)

    r1 = c1 / e1 if e1 > 0 else 0
    r2 = c2 / e2 if e2 > 0 else 0
    rate_ratio = r1 / r2 if r2 > 0 else float("inf")

    # Exact conditional test (condition on total count)
    total = c1 + c2
    e_ratio = e1 / (e1 + e2) if (e1 + e2) > 0 else 0.5
    if total > 0:
        if alt == "greater":
            p_val = float(1 - stats.binom.cdf(int(c1) - 1, int(total), e_ratio))
        elif alt == "less":
            p_val = float(stats.binom.cdf(int(c1), int(total), e_ratio))
        else:
            p_left = stats.binom.cdf(int(c1), int(total), e_ratio)
            p_right = 1 - stats.binom.cdf(int(c1) - 1, int(total), e_ratio)
            p_val = float(min(1.0, 2 * min(p_left, p_right)))
    else:
        p_val = 1.0

    # CI for rate ratio (log-normal approximation)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    if c1 > 0 and c2 > 0:
        se_ln = np.sqrt(1 / c1 + 1 / c2)
        ln_rr = np.log(rate_ratio)
        rr_lo = float(np.exp(ln_rr - z_crit * se_ln))
        rr_hi = float(np.exp(ln_rr + z_crit * se_ln))
    else:
        rr_lo, rr_hi = 0.0, float("inf")

    # CI for rate difference
    diff = r1 - r2
    se_diff = np.sqrt(r1 / e1 + r2 / e2) if (e1 > 0 and e2 > 0) else 0
    diff_lo = diff - z_crit * se_diff
    diff_hi = diff + z_crit * se_diff

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>TWO-SAMPLE POISSON RATE TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += "<<COLOR:accent>>── Sample Results ──<</COLOR>>\n"
    summary += f"  {'Group':<20} {'Count':>8} {'Exposure':>10} {'Rate':>10}\n"
    summary += f"  {'─' * 52}\n"
    summary += f"  {label1:<20} {c1:>8.0f} {e1:>10.1f} {r1:>10.4f}\n"
    summary += f"  {label2:<20} {c2:>8.0f} {e2:>10.1f} {r2:>10.4f}\n\n"
    summary += f"<<COLOR:accent>>── Rate Ratio (r₁/r₂) ──<</COLOR>> {rate_ratio:.4f}\n"
    summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for ratio:<</COLOR>> ({rr_lo:.4f}, {rr_hi:.4f})\n"
    summary += f"<<COLOR:accent>>── Rate Difference (r₁ − r₂) ──<</COLOR>> {diff:.4f}\n"
    summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for difference:<</COLOR>> ({diff_lo:.4f}, {diff_hi:.4f})\n\n"
    summary += "<<COLOR:accent>>── Exact Conditional Test ──<</COLOR>>\n"
    summary += f"  p-value: {p_val:.4f}\n\n"

    if p_val < alpha:
        summary += f"<<COLOR:good>>Rates differ significantly (p < {alpha})<</COLOR>>"
    else:
        summary += f"<<COLOR:text>>No significant difference in rates (p ≥ {alpha})<</COLOR>>"

    result["summary"] = summary

    result["plots"].append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": [label1, label2],
                    "y": [r1, r2],
                    "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                    "text": [f"{r1:.4f}", f"{r2:.4f}"],
                    "textposition": "outside",
                }
            ],
            "layout": {
                "title": "Rates by Group",
                "yaxis": {"title": "Rate", "rangemode": "tozero"},
                "height": 280,
            },
        }
    )

    # Rate ratio CI plot
    if rate_ratio < float("inf"):
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": [rate_ratio],
                        "y": ["Rate Ratio"],
                        "mode": "markers",
                        "marker": {"size": 12, "color": "#4a9f6e"},
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [rr_hi - rate_ratio],
                            "arrayminus": [rate_ratio - rr_lo],
                            "color": "#5a6a5a",
                        },
                    }
                ],
                "layout": {
                    "title": f"Rate Ratio ({conf_pct:.0f}% CI)",
                    "xaxis": {"title": "r₁ / r₂", "type": "log"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": -0.5,
                            "y1": 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                    "height": 180,
                },
            }
        )

    result["guide_observation"] = (
        f"Two-sample Poisson: r₁={r1:.4f} vs r₂={r2:.4f}, ratio={rate_ratio:.3f}, p={p_val:.4f}. "
        + ("Rates differ." if p_val < alpha else "Not significant.")
    )
    if p_val < alpha:
        verdict = f"Rates differ significantly (ratio = {rate_ratio:.3f}, p = {p_val:.4f})"
        body = f"Rate 1 = {r1:.4f} vs Rate 2 = {r2:.4f}. The rates are significantly different."
    else:
        verdict = f"No significant difference in rates (p = {p_val:.4f})"
        body = f"Rate 1 = {r1:.4f} vs Rate 2 = {r2:.4f} (ratio = {rate_ratio:.3f}). Cannot conclude the rates differ."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps="If rates differ, investigate what process or environmental differences drive the gap.",
    )
    result["statistics"] = {
        "count1": c1,
        "count2": c2,
        "exposure1": e1,
        "exposure2": e2,
        "rate1": r1,
        "rate2": r2,
        "rate_ratio": float(rate_ratio),
        "rate_difference": diff,
        "p_value": p_val,
        "ratio_ci_lower": rr_lo,
        "ratio_ci_upper": rr_hi,
        "diff_ci_lower": diff_lo,
        "diff_ci_upper": diff_hi,
    }

    return result


def run_bootstrap_ci(df, config):
    """
    Bootstrap Confidence Intervals - non-parametric inference.
    Resampling-based confidence intervals for statistics.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var")
    statistic = config.get("statistic", "mean")  # mean, median, std, correlation
    var2 = config.get("var2")  # For correlation
    n_bootstrap = int(config.get("n_bootstrap", 1000))
    conf_level = float(config.get("conf", 95)) / 100

    data = df[var].dropna().values
    n = len(data)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BOOTSTRAP CONFIDENCE INTERVALS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
    summary += f"<<COLOR:highlight>>Statistic:<</COLOR>> {statistic}\n"
    summary += f"<<COLOR:highlight>>Bootstrap samples:<</COLOR>> {n_bootstrap}\n"
    summary += f"<<COLOR:highlight>>Confidence level:<</COLOR>> {conf_level * 100:.0f}%\n\n"

    np.random.seed(42)

    # Calculate bootstrap distribution
    boot_stats = []

    if statistic == "correlation" and var2:
        data2 = df[var2].dropna().values
        min_len = min(len(data), len(data2))
        data = data[:min_len]
        data2 = data2[:min_len]
        observed = np.corrcoef(data, data2)[0, 1]

        for _ in range(n_bootstrap):
            idx = np.random.choice(min_len, min_len, replace=True)
            boot_stats.append(np.corrcoef(data[idx], data2[idx])[0, 1])
    else:
        if statistic == "mean":
            observed = np.mean(data)
            stat_func = np.mean
        elif statistic == "median":
            observed = np.median(data)
            stat_func = np.median
        elif statistic == "std":
            observed = np.std(data, ddof=1)

            def stat_func(x):
                return np.std(x, ddof=1)

        elif statistic == "trimmed_mean":
            from scipy.stats import trim_mean

            observed = trim_mean(data, 0.1)

            def stat_func(x):
                return trim_mean(x, 0.1)

        else:
            observed = np.mean(data)
            stat_func = np.mean

        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(data, n, replace=True)
            boot_stats.append(stat_func(boot_sample))

    boot_stats = np.array(boot_stats)

    # Calculate CI using percentile method
    alpha = 1 - conf_level
    ci_lower = np.percentile(boot_stats, alpha / 2 * 100)
    ci_upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    # BCa (Bias-Corrected and Accelerated) interval
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_stats < observed))

    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        if statistic == "mean":
            jackknife_stats.append(np.mean(jack_sample))
        elif statistic == "median":
            jackknife_stats.append(np.median(jack_sample))
        else:
            jackknife_stats.append(np.mean(jack_sample))

    jackknife_stats = np.array(jackknife_stats)
    jack_mean = np.mean(jackknife_stats)
    a = np.sum((jack_mean - jackknife_stats) ** 3) / (6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5 + 1e-10)

    # BCa quantiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)

    bca_low_q = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
    bca_high_q = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

    bca_lower = np.percentile(boot_stats, bca_low_q * 100)
    bca_upper = np.percentile(boot_stats, bca_high_q * 100)

    summary += "<<COLOR:accent>>── Sample Statistics ──<</COLOR>>\n"
    summary += f"  Observed {statistic}: {observed:.4f}\n"
    summary += f"  Bootstrap SE: {np.std(boot_stats):.4f}\n"
    summary += f"  Bootstrap Bias: {np.mean(boot_stats) - observed:.4f}\n\n"

    summary += "<<COLOR:accent>>── Confidence Intervals ──<</COLOR>>\n"
    summary += f"  Percentile: ({ci_lower:.4f}, {ci_upper:.4f})\n"
    summary += f"  BCa:        ({bca_lower:.4f}, {bca_upper:.4f})\n\n"

    summary += "<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
    summary += f"  We are {conf_level * 100:.0f}% confident the true {statistic}\n"
    summary += f"  lies between {bca_lower:.4f} and {bca_upper:.4f}.\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"Bootstrap {conf_level * 100:.0f}% CI for {statistic}: ({bca_lower:.4f}, {bca_upper:.4f})"
    )
    result["statistics"] = {
        f"observed_{statistic}": float(observed),
        "ci_lower": float(bca_lower),
        "ci_upper": float(bca_upper),
        "bootstrap_se": float(np.std(boot_stats)),
    }

    # Narrative
    _boot_bias = float(np.mean(boot_stats) - observed)
    _boot_se = float(np.std(boot_stats))
    _ci_width = bca_upper - bca_lower
    verdict = f"Bootstrap {conf_level * 100:.0f}% CI for {statistic}: ({bca_lower:.4f}, {bca_upper:.4f})"
    body = (
        f"Observed {statistic} = <strong>{observed:.4f}</strong>. "
        f"After {n_bootstrap} resamples, the BCa interval is ({bca_lower:.4f}, {bca_upper:.4f}). "
        f"Bootstrap SE = {_boot_se:.4f}, bias = {_boot_bias:.4f}."
        + (
            " The BCa method corrects for bias and skewness in the bootstrap distribution."
            if abs(_boot_bias) > 0.01 * abs(observed)
            else ""
        )
    )
    nxt = f"The interval width ({_ci_width:.4f}) reflects estimation precision. To narrow it, increase the sample size (currently n = {n})."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="The histogram shows the bootstrap sampling distribution. Orange dashed line = observed value. Red triangles = CI bounds.",
    )

    # ── Diagnostics ──
    diagnostics = []
    # Small sample warning
    if n < 20:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Small sample (n = {n})",
                "detail": "Bootstrap reliability degrades with fewer than 20 observations. CI coverage may be inaccurate.",
            }
        )
    # Outlier check — outliers can dominate bootstrap resamples
    _out = _check_outliers(data, label=var)
    if _out:
        _out["detail"] += " Outliers can dominate bootstrap resamples, inflating or deflating the CI."
        diagnostics.append(_out)
    # Compare bootstrap CI with parametric CI (for mean statistic)
    if statistic == "mean" and n >= 3:
        _se_param = np.std(data, ddof=1) / np.sqrt(n)
        _t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        _param_lower = observed - _t_crit * _se_param
        _param_upper = observed + _t_crit * _se_param
        _param_width = _param_upper - _param_lower
        _boot_width = bca_upper - bca_lower
        _rel_diff = abs(_boot_width - _param_width) / (_param_width + 1e-15)
        if _rel_diff < 0.15:
            diagnostics.append(
                {
                    "level": "info",
                    "title": "Bootstrap confirms parametric assumptions",
                    "detail": f"Bootstrap CI width ({_boot_width:.4f}) closely matches parametric t-interval ({_param_width:.4f}). Normal assumption appears reasonable.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": "Bootstrap and parametric CIs disagree",
                    "detail": f"Bootstrap CI width ({_boot_width:.4f}) differs from parametric t-interval ({_param_width:.4f}) by {_rel_diff * 100:.0f}%. Distribution may be non-normal. Trust the bootstrap.",
                }
            )
    # BCa vs percentile divergence
    _perc_width = ci_upper - ci_lower
    _bca_width = bca_upper - bca_lower
    if abs(_bca_width - _perc_width) / (_perc_width + 1e-15) > 0.10:
        diagnostics.append(
            {
                "level": "info",
                "title": "BCa correction is meaningful",
                "detail": f"BCa interval ({bca_lower:.4f}, {bca_upper:.4f}) differs from percentile interval ({ci_lower:.4f}, {ci_upper:.4f}). The BCa method is correcting for bias and/or skewness in the bootstrap distribution.",
            }
        )
    result["diagnostics"] = diagnostics

    # Histogram of bootstrap distribution
    result["plots"].append(
        {
            "title": f"Bootstrap Distribution of {statistic.title()}",
            "data": [
                {
                    "type": "histogram",
                    "x": boot_stats.tolist(),
                    "marker": {
                        "color": "rgba(74, 159, 110, 0.4)",
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                },
                {
                    "type": "scatter",
                    "x": [observed, observed],
                    "y": [0, n_bootstrap / 20],
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2, "dash": "dash"},
                    "name": "Observed",
                },
                {
                    "type": "scatter",
                    "x": [bca_lower, bca_upper],
                    "y": [0, 0],
                    "mode": "markers",
                    "marker": {"color": "#e85747", "size": 12, "symbol": "triangle-up"},
                    "name": "CI bounds",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": statistic.title()},
                "yaxis": {"title": "Frequency"},
            },
        }
    )

    return result
