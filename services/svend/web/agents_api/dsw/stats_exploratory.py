"""DSW Statistical Analysis — exploratory analysis (descriptive, distribution, multivariate)."""

import logging

import numpy as np
from scipy import stats as sp_stats

from .common import (
    _bayesian_shadow,
    _check_normality,
    _check_outliers,
    _effect_magnitude,
    _evidence_grade,
    _narrative,
    _practical_block,
)

logger = logging.getLogger(__name__)


def _run_exploratory(analysis_id, df, config):
    """Run exploratory analysis."""
    import pandas as pd
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "descriptive":
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
            summary += (
                f"  Median: {median:.4f}    IQR: {iqr:.4f}    [{q1:.4f}, {q3:.4f}]\n"
            )
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

    elif analysis_id == "chi2":
        row_var = config.get("row_var") or config.get("var1") or config.get("var")
        col_var = config.get("col_var") or config.get("var2") or config.get("group_var")

        # Create contingency table
        contingency = pd.crosstab(df[row_var], df[col_var])
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
        cramers_v = (
            np.sqrt(chi2 / (n_obs * min_dim)) if (n_obs > 0 and min_dim > 0) else 0.0
        )
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
                _or_lo, _or_hi = np.exp(np.log(_or) - 1.96 * _log_se), np.exp(
                    np.log(_or) + 1.96 * _log_se
                )
                summary += (
                    f"  Odds Ratio: {_or:.3f}, 95% CI [{_or_lo:.3f}, {_or_hi:.3f}]\n"
                )
        summary += "\n"

        if pval < 0.05:
            summary += "<<COLOR:good>>Variables are significantly associated (p < 0.05)<</COLOR>>"
        else:
            summary += (
                "<<COLOR:text>>No significant association found (p >= 0.05)<</COLOR>>"
            )

        summary += _practical_block(
            "Cramér's V",
            cramers_v,
            "cramers_v",
            pval,
            context=f"The association between '{row_var}' and '{col_var}' is {v_label}. V=0 means no association, V=1 means perfect association.",
        )

        result["summary"] = summary
        obs_parts = [
            f"Chi-square test: χ²={chi2:.4f}, p={pval:.4f}, Cramér's V={cramers_v:.3f} ({v_label})"
        ]
        if pval < 0.05 and v_meaningful:
            obs_parts.append(
                f"'{row_var}' and '{col_var}' are meaningfully associated."
            )
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
            verdict = (
                f"Weak but significant association between {row_var} and {col_var}"
            )
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
            _shadow = _bayesian_shadow(
                "chi2",
                contingency=contingency.values,
                chi2_stat=chi2,
                dof=dof,
                n_obs=n_obs,
            )
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

    elif analysis_id == "prop_1sample":
        """
        One-Proportion Z-Test — test if an observed proportion equals a hypothesized value.
        Uses normal approximation to the binomial; reports Z, p-value, and Wilson CI.
        """
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
        margin = (
            z_crit * np.sqrt((p_hat * (1 - p_hat) + z_crit**2 / (4 * n)) / n) / denom
        )
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
        summary += (
            f"  {100 * (1 - alpha):.0f}% CI (Wilson): ({ci_lo:.4f}, {ci_hi:.4f})\n\n"
        )

        if p_val < alpha:
            summary += f"<<COLOR:good>>Proportion differs significantly from {p0} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference from {p0} (p ≥ {alpha})<</COLOR>>"

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
            verdict = f"Proportion consistent with {p0} (p\u0302 = {p_hat:.4f}, p = {p_val:.4f})"
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
            _grade = _evidence_grade(
                p_val, bf10=_shadow.get("bf10") if _shadow else None
            )
            if _grade:
                result["evidence_grade"] = _grade
        except Exception:
            pass

    elif analysis_id == "prop_2sample":
        """
        Two-Proportion Z-Test — compare proportions between two groups.
        Tests H₀: p₁ = p₂. Reports pooled Z, individual CIs, and difference CI.
        """
        var = config.get("var") or config.get("var1")
        group_var = (
            config.get("group_var") or config.get("var2") or config.get("factor")
        )
        event = config.get("event")
        alt = config.get("alternative", "two-sided")
        alpha = 1 - float(config.get("conf", 95)) / 100

        data = df[[var, group_var]].dropna()
        groups = sorted(data[group_var].unique().tolist(), key=str)
        if len(groups) != 2:
            result["summary"] = (
                f"Two-proportion test requires exactly 2 groups. Found {len(groups)}."
            )
            return result

        g1 = data[data[group_var] == groups[0]][var]
        g2 = data[data[group_var] == groups[1]][var]
        n1, n2 = len(g1), len(g2)

        if event is not None and str(event) != "":
            x1 = int((g1.astype(str) == str(event)).sum())
            x2 = int((g2.astype(str) == str(event)).sum())
        else:
            x1 = (
                int((g1 == 1).sum())
                if g1.dtype in ["int64", "float64"]
                else int(g1.value_counts().iloc[0])
            )
            x2 = (
                int((g2 == 1).sum())
                if g2.dtype in ["int64", "float64"]
                else int(g2.value_counts().iloc[0])
            )

        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0
        p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

        se_pooled = (
            np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
            if (n1 > 0 and n2 > 0)
            else 1
        )
        z_stat = (p1 - p2) / se_pooled if se_pooled > 0 else 0

        if alt == "greater":
            p_val = float(1 - stats.norm.cdf(z_stat))
        elif alt == "less":
            p_val = float(stats.norm.cdf(z_stat))
        else:
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Difference CI (unpooled SE)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se_diff = (
            np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            if (n1 > 0 and n2 > 0)
            else 0
        )
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

        result["guide_observation"] = (
            f"2-prop Z-test: p₁={p1:.4f} vs p₂={p2:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. "
            + ("Significant." if p_val < alpha else "Not significant.")
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

    elif analysis_id == "fisher_exact":
        """
        Fisher's Exact Test — exact test for 2×2 contingency tables.
        Preferred over chi-square when expected cell counts are small (<5).
        Reports odds ratio, exact p-value, and odds ratio CI.
        """
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
            summary += (
                f"<<COLOR:text>>No significant association (p ≥ {alpha})<</COLOR>>"
            )

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

        result["guide_observation"] = (
            f"Fisher's exact: OR={odds_ratio:.3f}, p={p_val:.4f}. "
            + ("Significant association." if p_val < alpha else "No association.")
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
            verdict = (
                f"Significant association (OR = {odds_ratio:.2f}, p = {p_val:.4f})"
            )
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

    elif analysis_id == "poisson_1sample":
        """
        One-Sample Poisson Rate Test — test if an observed event rate equals a hypothesized rate.
        Uses exact Poisson test (conditional) or normal approximation for large counts.
        """
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
        z_crit = stats.norm.ppf(1 - alpha / 2)
        if total_count > 0:
            ci_lo = stats.chi2.ppf(alpha / 2, 2 * total_count) / (2 * exposure)
            ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (total_count + 1)) / (
                2 * exposure
            )
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
        summary += (
            f"  {100 * (1 - alpha):.0f}% CI for rate: ({ci_lo:.4f}, {ci_hi:.4f})\n\n"
        )

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

    elif analysis_id == "poisson_2sample":
        """
        Two-Sample Poisson Rate Test — compare event rates between two groups.
        Modes:
          1) Two count columns + optional exposure per group
          2) Response column + grouping factor (auto-sum per group)
        Reports exact conditional test, rate ratio with CI.
        """
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
                result["summary"] = (
                    f"Two-sample Poisson test requires exactly 2 groups. Found {len(levels)}."
                )
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
        summary += (
            f"<<COLOR:accent>>── Rate Ratio (r₁/r₂) ──<</COLOR>> {rate_ratio:.4f}\n"
        )
        summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for ratio:<</COLOR>> ({rr_lo:.4f}, {rr_hi:.4f})\n"
        summary += (
            f"<<COLOR:accent>>── Rate Difference (r₁ − r₂) ──<</COLOR>> {diff:.4f}\n"
        )
        summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for difference:<</COLOR>> ({diff_lo:.4f}, {diff_hi:.4f})\n\n"
        summary += "<<COLOR:accent>>── Exact Conditional Test ──<</COLOR>>\n"
        summary += f"  p-value: {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += (
                f"<<COLOR:good>>Rates differ significantly (p < {alpha})<</COLOR>>"
            )
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

    elif analysis_id == "bootstrap_ci":
        """
        Bootstrap Confidence Intervals - non-parametric inference.
        Resampling-based confidence intervals for statistics.
        """
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
        a = np.sum((jack_mean - jackknife_stats) ** 3) / (
            6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5 + 1e-10
        )

        # BCa quantiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        bca_low_q = stats.norm.cdf(
            z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low))
        )
        bca_high_q = stats.norm.cdf(
            z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high))
        )

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
            _out[
                "detail"
            ] += " Outliers can dominate bootstrap resamples, inflating or deflating the CI."
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
                        "marker": {
                            "color": "#e85747",
                            "size": 12,
                            "symbol": "triangle-up",
                        },
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

    elif analysis_id == "box_cox":
        """
        Box-Cox Transformation - find optimal power transformation.
        Transforms data to approximate normality.
        """
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

    # ── Run Chart ──────────────────────────────────────────────────────────
    elif analysis_id == "run_chart":
        """
        Run Chart — time-ordered individual values with median line and runs tests.
        Tests for clustering, mixtures, trends, and oscillation.
        """
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
                std_runs = np.sqrt(
                    (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2))
                    / ((n1 + n2) ** 2 * (n1 + n2 - 1))
                )

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

                cluster_flag = (
                    "<<COLOR:danger>>Yes<</COLOR>>"
                    if p_clustering < 0.05
                    else "<<COLOR:success>>No<</COLOR>>"
                )
                mixture_flag = (
                    "<<COLOR:danger>>Yes<</COLOR>>"
                    if p_mixtures < 0.05
                    else "<<COLOR:success>>No<</COLOR>>"
                )

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
            result["guide_observation"] = (
                f"Run chart: {n} obs, median={median_val:.4g}, {runs} runs."
            )
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

    # ── Grubbs' Outlier Test ─────────────────────────────────────────────
    elif analysis_id == "grubbs_test":
        """
        Grubbs' test for a single outlier.
        Tests whether the most extreme value is significantly different.
        """
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
                G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(
                    t_crit**2 / (n - 2 + t_crit**2)
                )

                # Two-sided p-value (approximation)
                _t_stat = G * np.sqrt(n) / np.sqrt(n - 1)  # noqa: F841
                p_val = (
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

    # ── Cross-Correlation Function ───────────────────────────────────────
    elif analysis_id == "johnson_transform":
        """
        Johnson Transformation — finds optimal Johnson family (SB, SL, SU) to normalize data.
        More general than Box-Cox (handles bounded and unbounded distributions).
        """
        var = config.get("var")

        data = df[var].dropna().values
        n = len(data)

        if n < 10:
            result["summary"] = (
                "Johnson transformation requires at least 10 observations."
            )
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
                _, p_su = stats.shapiro(
                    transformed_su[: min(5000, len(transformed_su))]
                )
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
                _, p_sb = stats.shapiro(
                    transformed_sb[: min(5000, len(transformed_sb))]
                )
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

    elif analysis_id == "tolerance_interval":
        """
        Tolerance Intervals - contain a proportion of the population.
        Both normal-based and non-parametric methods.
        """
        var = config.get("var")
        proportion = float(config.get("proportion", 0.95))  # Proportion of population
        confidence = float(config.get("confidence", 0.95))  # Confidence level
        method = config.get("method", "normal")  # normal or nonparametric

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>TOLERANCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Coverage:<</COLOR>> {proportion * 100:.0f}% of population\n"
        summary += (
            f"<<COLOR:highlight>>Confidence:<</COLOR>> {confidence * 100:.0f}%\n\n"
        )

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
        alpha = 1 - confidence
        beta = 1 - proportion

        # Find r such that P(at least proportion*100% between X(r) and X(n-r+1)) >= confidence
        # Using binomial distribution
        from scipy.special import comb

        r_found = None
        for r in range(1, n // 2 + 1):
            # Probability that at least proportion of population is between order statistics
            prob = 0
            for j in range(r, n - r + 2):
                prob += (
                    comb(n, j, exact=True)
                    * (proportion ** (j))
                    * ((1 - proportion) ** (n - j))
                )
            if prob >= confidence:
                r_found = r
                break

        if r_found:
            sorted_data = np.sort(data)
            tol_lower_np = sorted_data[r_found - 1]
            tol_upper_np = sorted_data[n - r_found]
            summary += "<<COLOR:accent>>Non-Parametric Tolerance Interval:<</COLOR>>\n"
            summary += (
                f"  Uses order statistics X({r_found}) and X({n - r_found + 1})\n"
            )
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
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.3)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
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

    elif analysis_id == "hotelling_t2":
        """
        Hotelling's T² Test — multivariate extension of the two-sample t-test.
        Tests whether two groups have different mean vectors across multiple response variables.
        """
        responses = config.get("responses", [])
        group_var = config.get("group_var") or config.get("factor")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[responses + [group_var]].dropna()
        groups = sorted(data[group_var].unique().tolist(), key=str)

        if len(groups) != 2:
            result["summary"] = (
                f"Hotelling's T² requires exactly 2 groups. Found {len(groups)}: {groups}"
            )
            return result

        g1_data = data[data[group_var] == groups[0]][responses].values
        g2_data = data[data[group_var] == groups[1]][responses].values
        n1, n2 = len(g1_data), len(g2_data)
        p = len(responses)

        mean1 = np.mean(g1_data, axis=0)
        mean2 = np.mean(g2_data, axis=0)
        diff = mean1 - mean2

        # Pooled covariance matrix
        S1 = np.cov(g1_data, rowvar=False, ddof=1)
        S2 = np.cov(g2_data, rowvar=False, ddof=1)
        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

        # Hotelling's T²
        S_inv = np.linalg.inv(S_pooled * (1.0 / n1 + 1.0 / n2))
        T2 = float(diff @ S_inv @ diff)

        # Convert to F-statistic
        df1 = p
        df2 = n1 + n2 - p - 1
        F_stat = T2 * df2 / (p * (n1 + n2 - 2))
        p_value = float(1 - stats.f.cdf(F_stat, df1, df2))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>HOTELLING'S T² TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Group variable:<</COLOR>> {group_var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += "<<COLOR:accent>>── Group Means ──<</COLOR>>\n"
        summary += f"{'Variable':<20} {str(groups[0]):>12} {str(groups[1]):>12} {'Difference':>12}\n"
        summary += f"{'─' * 58}\n"
        for i, var in enumerate(responses):
            summary += (
                f"{var:<20} {mean1[i]:>12.4f} {mean2[i]:>12.4f} {diff[i]:>12.4f}\n"
            )

        summary += "\n<<COLOR:accent>>── Test Statistics ──<</COLOR>>\n"
        summary += f"  Hotelling's T²: {T2:.4f}\n"
        summary += f"  F-statistic: {F_stat:.4f} (df1={df1}, df2={df2})\n"
        summary += f"  p-value: {p_value:.4f}\n\n"

        if p_value < alpha:
            summary += f"<<COLOR:good>>Mean vectors differ significantly (p < {alpha})<</COLOR>>\n"
            summary += "<<COLOR:text>>The groups have different multivariate profiles across the response variables.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in mean vectors (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Radar/profile plot of group means
        traces = []
        for idx, grp in enumerate(groups):
            grp_means = [
                float(mean1[i]) if idx == 0 else float(mean2[i]) for i in range(p)
            ]
            colors = ["#4a9f6e", "#47a5e8"]
            traces.append(
                {
                    "type": "scatterpolar",
                    "r": grp_means + [grp_means[0]],
                    "theta": responses + [responses[0]],
                    "name": str(grp),
                    "fill": "toself",
                    "fillcolor": f"rgba({','.join(str(int(c, 16)) for c in [colors[idx][1:3], colors[idx][3:5], colors[idx][5:7]])}, 0.15)",
                    "line": {"color": colors[idx]},
                }
            )
        result["plots"].append(
            {
                "title": "Multivariate Profile — Group Means",
                "data": traces,
                "layout": {"height": 350, "polar": {"radialaxis": {"visible": True}}},
            }
        )

        # Per-variable box plots by group
        box_traces = []
        grp_colors = ["#4a9f6e", "#47a5e8"]
        for gi, grp in enumerate(groups):
            grp_data = data[data[group_var] == grp]
            for vi, var in enumerate(responses):
                box_traces.append(
                    {
                        "type": "box",
                        "y": grp_data[var].tolist(),
                        "x": [var] * len(grp_data),
                        "name": str(grp),
                        "marker": {"color": grp_colors[gi]},
                        "legendgroup": str(grp),
                        "showlegend": vi == 0,
                    }
                )
        result["plots"].append(
            {
                "title": "Response Distributions by Group",
                "data": box_traces,
                "layout": {
                    "height": 300,
                    "boxmode": "group",
                    "xaxis": {"title": "Response Variable"},
                    "yaxis": {"title": "Value"},
                },
            }
        )

        result["guide_observation"] = (
            f"Hotelling's T² = {T2:.2f}, F = {F_stat:.2f}, p = {p_value:.4f}. "
            + ("Groups differ." if p_value < alpha else "No difference.")
        )
        result["statistics"] = {
            "T2": T2,
            "F_statistic": F_stat,
            "p_value": p_value,
            "df1": df1,
            "df2": df2,
            "mean_diff": diff.tolist(),
        }
        if p_value < alpha:
            result["narrative"] = _narrative(
                f"Multivariate means differ (T\u00b2 = {T2:.2f}, p = {p_value:.4f})",
                "The groups differ when considering all response variables simultaneously. Hotelling's T\u00b2 is the multivariate extension of the t-test.",
                next_steps="Examine individual variables to identify which drive the difference. Consider MANOVA for more than 2 groups.",
            )
        else:
            result["narrative"] = _narrative(
                f"No multivariate difference (p = {p_value:.4f})",
                "The groups do not differ significantly across the combined set of variables.",
                next_steps="Even if individual variables differ, the multivariate test considers their correlation structure.",
            )

    elif analysis_id == "manova":
        """
        One-Way MANOVA — Multivariate Analysis of Variance.
        Tests whether group means differ across multiple response variables simultaneously.
        Reports Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Largest Root.
        """
        responses = config.get("responses", [])
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[responses + [factor]].dropna()
        groups = sorted(data[factor].unique().tolist(), key=str)
        k = len(groups)
        p = len(responses)
        N = len(data)

        if k < 2:
            result["summary"] = f"MANOVA requires at least 2 groups. Found {k}."
            return result

        # Overall mean
        grand_mean = data[responses].values.mean(axis=0)

        # Between-groups (hypothesis) and within-groups (error) SSCP matrices
        H = np.zeros((p, p))  # Hypothesis SSCP
        E = np.zeros((p, p))  # Error SSCP

        group_means = {}
        for grp in groups:
            grp_data = data[data[factor] == grp][responses].values
            n_g = len(grp_data)
            grp_mean = grp_data.mean(axis=0)
            group_means[grp] = {"mean": grp_mean, "n": n_g}

            diff = (grp_mean - grand_mean).reshape(-1, 1)
            H += n_g * (diff @ diff.T)

            centered = grp_data - grp_mean
            E += centered.T @ centered

        # Four test statistics
        try:
            E_inv = np.linalg.inv(E)
            HE_inv = H @ E_inv
            eigenvalues = np.real(np.linalg.eigvals(HE_inv))
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]
        except np.linalg.LinAlgError:
            result["summary"] = (
                "MANOVA: Error matrix is singular. Check for collinear responses or insufficient data."
            )
            return result

        s = min(p, k - 1)

        # 1. Wilks' Lambda
        wilks = float(np.linalg.det(E) / np.linalg.det(E + H))

        # Wilks' Lambda → F approximation (Rao's F)
        df_h = p * (k - 1)
        df_e = N - k
        if p == 1 or k == 2:
            F_wilks = ((1 - wilks) / wilks) * (df_e / df_h)
            df1_w, df2_w = df_h, df_e
        elif p == 2 or k == 3:
            wilks_sqrt = np.sqrt(wilks) if wilks > 0 else 0
            r = df_e - (p - k + 2) / 2
            F_wilks = (
                ((1 - wilks_sqrt) / wilks_sqrt) * (r / df_h) if wilks_sqrt > 0 else 0
            )
            df1_w, df2_w = df_h, 2 * (r - 1) if r > 1 else 1
        else:
            # General case: Chi-square approximation
            t = (
                np.sqrt((p**2 * (k - 1) ** 2 - 4) / (p**2 + (k - 1) ** 2 - 5))
                if (p**2 + (k - 1) ** 2 - 5) > 0
                else 1
            )
            df1_w = p * (k - 1)
            ms = N - 1 - (p + k) / 2
            df2_w = ms * t - df1_w / 2 + 1
            wilks_t = wilks ** (1 / t) if wilks > 0 and t > 0 else 0
            F_wilks = ((1 - wilks_t) / wilks_t) * (df2_w / df1_w) if wilks_t > 0 else 0

        p_wilks = float(1 - stats.f.cdf(max(F_wilks, 0), max(df1_w, 1), max(df2_w, 1)))

        # 2. Pillai's Trace
        pillai = float(np.sum(eigenvalues / (1 + eigenvalues)))
        df1_p = s * max(p, k - 1)
        df2_p = s * (N - k - p + s)
        F_pillai = (
            (pillai / s) * (df2_p / (max(p, k - 1))) / (1 - pillai / s)
            if (1 - pillai / s) > 0
            else 0
        )
        p_pillai = float(
            1 - stats.f.cdf(max(F_pillai, 0), max(df1_p, 1), max(df2_p, 1))
        )

        # 3. Hotelling-Lawley Trace
        hl_trace = float(np.sum(eigenvalues))
        df1_hl = s * max(p, k - 1)
        df2_hl = s * (N - k - p - 1) + 2
        F_hl = (hl_trace / s) * (df2_hl / max(p, k - 1)) if max(p, k - 1) > 0 else 0
        p_hl = float(1 - stats.f.cdf(max(F_hl, 0), max(df1_hl, 1), max(df2_hl, 1)))

        # 4. Roy's Largest Root
        roy = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
        df1_r = max(p, k - 1)
        df2_r = N - k - max(p, k - 1) + 1
        F_roy = roy * df2_r / df1_r if df1_r > 0 else 0
        p_roy = float(1 - stats.f.cdf(max(F_roy, 0), max(df1_r, 1), max(df2_r, 1)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ONE-WAY MANOVA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

        summary += "<<COLOR:accent>>── Group Means ──<</COLOR>>\n"
        header = f"{'Variable':<20}" + "".join(f"{str(g):>12}" for g in groups)
        summary += header + "\n" + "─" * len(header) + "\n"
        for i, var in enumerate(responses):
            row = f"{var:<20}" + "".join(
                f"{group_means[g]['mean'][i]:>12.4f}" for g in groups
            )
            summary += row + "\n"
        summary += "\n"

        summary += "<<COLOR:accent>>── MANOVA Test Statistics ──<</COLOR>>\n"
        summary += f"{'Test':<25} {'Value':>10} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
        summary += f"{'─' * 62}\n"

        tests = [
            ("Wilks' Lambda", wilks, F_wilks, p_wilks),
            ("Pillai's Trace", pillai, F_pillai, p_pillai),
            ("Hotelling-Lawley Trace", hl_trace, F_hl, p_hl),
            ("Roy's Largest Root", roy, F_roy, p_roy),
        ]
        for name, val, f_val, p_val in tests:
            sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
            summary += (
                f"{name:<25} {val:>10.4f} {f_val:>10.4f} {p_val:>10.4f} {sig:>5}\n"
            )

        summary += f"\n<<COLOR:accent>>── Eigenvalues of H·E⁻¹ ──<</COLOR>> {', '.join(f'{e:.4f}' for e in eigenvalues)}\n\n"

        # Overall interpretation (use Pillai's — most robust)
        if p_pillai < alpha:
            summary += f"<<COLOR:good>>Significant multivariate effect (Pillai's Trace, p < {alpha})<</COLOR>>\n"
            summary += "<<COLOR:text>>Group means differ across the response variables considered jointly.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant multivariate effect (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Group centroid plot (first 2 responses, or first 2 discriminant functions)
        if p >= 2:
            traces = []
            colors = [
                "#4a9f6e",
                "#47a5e8",
                "#e89547",
                "#9f4a4a",
                "#6c5ce7",
                "#e84393",
                "#00b894",
                "#fdcb6e",
            ]
            for i, grp in enumerate(groups):
                grp_data = data[data[factor] == grp]
                traces.append(
                    {
                        "type": "scatter",
                        "x": grp_data[responses[0]].tolist(),
                        "y": grp_data[responses[1]].tolist(),
                        "mode": "markers",
                        "name": str(grp),
                        "marker": {
                            "color": colors[i % len(colors)],
                            "size": 6,
                            "opacity": 0.6,
                        },
                    }
                )
                # Centroid
                traces.append(
                    {
                        "type": "scatter",
                        "x": [float(group_means[grp]["mean"][0])],
                        "y": [float(group_means[grp]["mean"][1])],
                        "mode": "markers",
                        "marker": {
                            "color": colors[i % len(colors)],
                            "size": 14,
                            "symbol": "diamond",
                            "line": {"color": "white", "width": 2},
                        },
                        "showlegend": False,
                    }
                )
            result["plots"].append(
                {
                    "title": f"Group Centroids — {responses[0]} vs {responses[1]}",
                    "data": traces,
                    "layout": {
                        "height": 350,
                        "xaxis": {"title": responses[0]},
                        "yaxis": {"title": responses[1]},
                    },
                }
            )

        # Per-response box plots by group
        box_traces_m = []
        m_colors = [
            "#4a9f6e",
            "#47a5e8",
            "#e89547",
            "#9f4a4a",
            "#6c5ce7",
            "#e84393",
            "#00b894",
            "#fdcb6e",
        ]
        for gi, grp in enumerate(groups):
            grp_d = data[data[factor] == grp]
            for vi, var in enumerate(responses):
                box_traces_m.append(
                    {
                        "type": "box",
                        "y": grp_d[var].tolist(),
                        "x": [var] * len(grp_d),
                        "name": str(grp),
                        "marker": {"color": m_colors[gi % len(m_colors)]},
                        "legendgroup": str(grp),
                        "showlegend": vi == 0,
                    }
                )
        result["plots"].append(
            {
                "title": "Response Distributions by Group",
                "data": box_traces_m,
                "layout": {
                    "height": 300,
                    "boxmode": "group",
                    "xaxis": {"title": "Response"},
                    "yaxis": {"title": "Value"},
                },
            }
        )

        # Correlation heatmap of response variables
        corr_mat = data[responses].corr().values
        result["plots"].append(
            {
                "data": [
                    {
                        "z": corr_mat.tolist(),
                        "x": responses,
                        "y": responses,
                        "type": "heatmap",
                        "colorscale": [
                            [0, "#d94a4a"],
                            [0.5, "#f0f4f0"],
                            [1, "#2c5f2d"],
                        ],
                        "zmin": -1,
                        "zmax": 1,
                        "text": [
                            [f"{corr_mat[i][j]:.3f}" for j in range(len(responses))]
                            for i in range(len(responses))
                        ],
                        "texttemplate": "%{text}",
                        "showscale": True,
                    }
                ],
                "layout": {"title": "Response Correlation Matrix", "height": 300},
            }
        )

        result["guide_observation"] = (
            f"MANOVA: Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}; Pillai's V = {pillai:.4f}, p = {p_pillai:.4f}. "
            + (
                "Multivariate effect detected."
                if p_pillai < alpha
                else "No multivariate effect."
            )
        )
        result["statistics"] = {
            "wilks_lambda": wilks,
            "wilks_F": F_wilks,
            "wilks_p": p_wilks,
            "pillai_trace": pillai,
            "pillai_F": F_pillai,
            "pillai_p": p_pillai,
            "hotelling_lawley": hl_trace,
            "hl_F": F_hl,
            "hl_p": p_hl,
            "roys_root": roy,
            "roy_F": F_roy,
            "roy_p": p_roy,
            "eigenvalues": eigenvalues.tolist(),
            "n_groups": k,
            "n_responses": p,
            "N": N,
        }

        # Narrative
        _mv_sig = p_pillai < alpha
        _mv_eta2 = float(pillai)  # Pillai's trace approximates multivariate η²
        _mv_mag = (
            "large" if _mv_eta2 > 0.25 else ("medium" if _mv_eta2 > 0.10 else "small")
        )
        result["narrative"] = _narrative(
            f"MANOVA — {'Significant' if _mv_sig else 'No significant'} multivariate effect (Pillai's V = {pillai:.4f})",
            f"Testing {p} response variables across {k} groups (N = {N}). "
            + (
                f"The factor has a {_mv_mag} multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}). Examine per-response ANOVAs to identify which variables drive the difference."
                if _mv_sig
                else f"No evidence of group differences across the response variables jointly (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f})."
            ),
            next_steps=(
                "Run univariate ANOVAs per response with Bonferroni correction to identify which variables differ."
                if _mv_sig
                else "Check individual ANOVAs — marginal effects may exist that the joint test misses."
            ),
            chart_guidance="The scatter plot shows group separation in the first two response dimensions. Non-overlapping clusters confirm a multivariate effect.",
        )

    elif analysis_id == "nested_anova":
        """
        Nested (Hierarchical) ANOVA — random effects model.
        Tests fixed factor effect while accounting for a random nesting factor.
        E.g., operators nested within machines, batches within suppliers.
        Uses linear mixed-effects model (statsmodels mixedlm).
        """
        response = config.get("response") or config.get("var")
        fixed_factor = config.get("fixed_factor") or config.get("factor")
        random_factor = config.get("random_factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        try:
            from statsmodels.formula.api import mixedlm

            data = df[[response, fixed_factor, random_factor]].dropna()
            N = len(data)

            # Fit mixed model: response ~ fixed_factor with random intercept for random_factor
            formula = f"{response} ~ C({fixed_factor})"
            model = mixedlm(formula, data, groups=data[random_factor])
            fit = model.fit(reml=True)

            # Extract results
            fixed_effects = {}
            for name, val in fit.fe_params.items():
                pval = float(fit.pvalues[name]) if name in fit.pvalues else None
                se = float(fit.bse[name]) if name in fit.bse else None
                fixed_effects[name] = {"coef": float(val), "se": se, "p_value": pval}

            # Variance components
            var_random = (
                float(fit.cov_re.iloc[0, 0])
                if hasattr(fit.cov_re, "iloc")
                else float(fit.cov_re)
            )
            var_residual = float(fit.scale)
            var_total = var_random + var_residual
            icc = var_random / var_total if var_total > 0 else 0

            # Group stats
            fixed_levels = sorted(data[fixed_factor].unique().tolist(), key=str)
            random_levels = sorted(data[random_factor].unique().tolist(), key=str)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += "<<COLOR:title>>NESTED ANOVA (Mixed-Effects Model)<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Fixed factor:<</COLOR>> {fixed_factor} ({len(fixed_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>Random factor (nesting):<</COLOR>> {random_factor} ({len(random_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary += "<<COLOR:accent>>── Fixed Effects ──<</COLOR>>\n"
            summary += f"{'Term':<30} {'Coef':>8} {'SE':>8} {'p-value':>8} {'Sig':>5}\n"
            summary += f"{'─' * 62}\n"
            for name, fe in fixed_effects.items():
                sig = (
                    "<<COLOR:good>>*<</COLOR>>"
                    if fe["p_value"] is not None and fe["p_value"] < alpha
                    else ""
                )
                p_str = f"{fe['p_value']:.4f}" if fe["p_value"] is not None else "N/A"
                se_str = f"{fe['se']:.4f}" if fe["se"] is not None else "N/A"
                summary += (
                    f"{name:<30} {fe['coef']:>8.4f} {se_str:>8} {p_str:>8} {sig:>5}\n"
                )

            summary += "\n<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
            summary += f"  {random_factor} (random): {var_random:.4f} ({icc * 100:.1f}% of total)\n"
            summary += (
                f"  Residual: {var_residual:.4f} ({(1 - icc) * 100:.1f}% of total)\n"
            )
            summary += f"  Total: {var_total:.4f}\n"
            summary += f"  ICC (Intraclass Correlation): {icc:.4f}\n\n"

            if icc > 0.1:
                summary += f"<<COLOR:good>>ICC = {icc:.3f} — substantial variation attributed to {random_factor}.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The nesting structure accounts for {icc * 100:.1f}% of the variance. Ignoring it would inflate Type I error.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:text>>ICC = {icc:.3f} — low variation from {random_factor}. A standard ANOVA may suffice.<</COLOR>>\n"

            # Check if fixed factor is significant
            sig_fixed = any(
                fe["p_value"] is not None and fe["p_value"] < alpha
                for name, fe in fixed_effects.items()
                if name != "Intercept"
            )
            if sig_fixed:
                summary += f"<<COLOR:good>>Fixed factor {fixed_factor} has significant effect.<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>Fixed factor {fixed_factor} not significant after accounting for {random_factor}.<</COLOR>>"

            result["summary"] = summary

            # Box plot with nesting structure
            traces = []
            for i, fl in enumerate(fixed_levels):
                subset = data[data[fixed_factor] == fl]
                traces.append(
                    {
                        "type": "box",
                        "y": subset[response].tolist(),
                        "x": [str(fl)] * len(subset),
                        "name": str(fl),
                        "boxpoints": "all",
                        "jitter": 0.3,
                        "pointpos": 0,
                        "marker": {"size": 4, "opacity": 0.5},
                    }
                )

            result["plots"].append(
                {
                    "title": f"Nested ANOVA: {response} by {fixed_factor} (nested in {random_factor})",
                    "data": traces,
                    "layout": {
                        "height": 300,
                        "yaxis": {"title": response},
                        "xaxis": {"title": fixed_factor},
                    },
                }
            )

            # Residuals vs fitted values
            fitted_vals = fit.fittedvalues
            resid_vals = fit.resid
            result["plots"].append(
                {
                    "title": "Residuals vs Fitted Values",
                    "data": [
                        {
                            "x": fitted_vals.tolist(),
                            "y": resid_vals.tolist(),
                            "mode": "markers",
                            "type": "scatter",
                            "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6},
                        }
                    ],
                    "layout": {
                        "height": 280,
                        "xaxis": {"title": "Fitted Values"},
                        "yaxis": {"title": "Residuals"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": float(fitted_vals.min()),
                                "x1": float(fitted_vals.max()),
                                "y0": 0,
                                "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            # Normal Q-Q plot of residuals
            from scipy import stats as qstats

            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [
                float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)
            ]
            result["plots"].append(
                {
                    "title": "Normal Q-Q Plot of Residuals",
                    "data": [
                        {
                            "x": theoretical_q,
                            "y": sorted_resid.tolist(),
                            "mode": "markers",
                            "type": "scatter",
                            "marker": {"color": "#4a9f6e", "size": 4},
                            "name": "Residuals",
                        },
                        {
                            "x": [theoretical_q[0], theoretical_q[-1]],
                            "y": [
                                theoretical_q[0] * np.std(sorted_resid)
                                + np.mean(sorted_resid),
                                theoretical_q[-1] * np.std(sorted_resid)
                                + np.mean(sorted_resid),
                            ],
                            "mode": "lines",
                            "line": {"color": "#e89547", "dash": "dash"},
                            "name": "Reference",
                        },
                    ],
                    "layout": {
                        "height": 280,
                        "xaxis": {"title": "Theoretical Quantiles"},
                        "yaxis": {"title": "Sample Quantiles"},
                    },
                }
            )

            result["guide_observation"] = (
                f"Nested ANOVA: ICC = {icc:.3f} ({icc * 100:.1f}% variance from {random_factor}). "
                + (
                    "Fixed effect significant."
                    if sig_fixed
                    else "Fixed effect not significant."
                )
            )
            if sig_fixed:
                result["narrative"] = _narrative(
                    f"Nested ANOVA: fixed effect significant, ICC = {icc:.3f}",
                    f"{icc * 100:.1f}% of variation comes from <strong>{random_factor}</strong>. The fixed effect significantly affects the response.",
                    next_steps="The ICC tells you how much of the variation is between groups vs within. High ICC = groups are very different.",
                )
            else:
                result["narrative"] = _narrative(
                    f"Nested ANOVA: no significant fixed effect (ICC = {icc:.3f})",
                    f"{icc * 100:.1f}% of variation comes from {random_factor}. The fixed effect is not significant.",
                    next_steps="Low ICC and non-significant fixed effect suggests most variation is within groups.",
                )
            result["statistics"] = {
                "fixed_effects": fixed_effects,
                "var_random": var_random,
                "var_residual": var_residual,
                "icc": icc,
                "aic": float(fit.aic) if hasattr(fit, "aic") else None,
                "bic": float(fit.bic) if hasattr(fit, "bic") else None,
                "converged": fit.converged if hasattr(fit, "converged") else True,
            }

        except ImportError:
            result["summary"] = (
                "Nested ANOVA requires statsmodels. Install with: pip install statsmodels"
            )
        except Exception as e:
            result["summary"] = f"Nested ANOVA error: {str(e)}"

    elif analysis_id == "manova":
        """
        Multivariate ANOVA — tests group differences across multiple response variables.
        Uses Pillai's trace, Wilks' lambda, Hotelling-Lawley, Roy's greatest root.
        """
        responses = config.get("responses", [])
        factor = config.get("factor") or config.get("group_var") or config.get("group")
        alpha = 1 - config.get("conf", 95) / 100

        if not responses and config.get("response"):
            responses = [config["response"]]

        try:
            all_cols = responses + [factor]
            data = df[all_cols].dropna()
            N = len(data)
            groups = sorted(data[factor].unique().tolist(), key=str)
            k = len(groups)
            p = len(responses)

            # Compute group means and overall mean
            overall_mean = data[responses].mean().values
            group_data = {g: data[data[factor] == g][responses].values for g in groups}
            group_means = {g: v.mean(axis=0) for g, v in group_data.items()}
            group_ns = {g: len(v) for g, v in group_data.items()}

            # Between-group SSCP matrix (H)
            H = np.zeros((p, p))
            for g in groups:
                diff = (group_means[g] - overall_mean).reshape(-1, 1)
                H += group_ns[g] * diff @ diff.T

            # Within-group SSCP matrix (E)
            E = np.zeros((p, p))
            for g in groups:
                centered = group_data[g] - group_means[g]
                E += centered.T @ centered

            # Test statistics
            df_h = k - 1
            df_e = N - k

            # Eigenvalues of E^-1 H
            try:
                E_inv = np.linalg.inv(E)
                eigvals = np.real(np.linalg.eigvals(E_inv @ H))
                eigvals = np.sort(eigvals)[::-1]
            except np.linalg.LinAlgError:
                eigvals = np.array([0.0] * p)

            # Pillai's trace
            pillai = np.sum(eigvals / (1 + eigvals))

            # Wilks' lambda
            wilks = np.prod(1 / (1 + eigvals))

            # Hotelling-Lawley trace
            hotelling = np.sum(eigvals)

            # Roy's greatest root
            roy = eigvals[0] if len(eigvals) > 0 else 0

            # Approximate F-test for Wilks' lambda
            s = min(p, df_h)
            _m = (abs(p - df_h) - 1) / 2  # noqa: F841
            (df_e - p - 1) / 2
            if s > 0 and wilks > 0:
                r = df_e - (p - df_h + 1) / 2
                u = (p * df_h - 2) / 4
                if p**2 + df_h**2 - 5 > 0:
                    t = np.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5))
                else:
                    t = 1
                df1 = p * df_h
                df2 = r * t - 2 * u
                if df2 > 0:
                    f_wilks = ((1 - wilks ** (1 / t)) / (wilks ** (1 / t))) * (
                        df2 / df1
                    )
                    from scipy import stats as fstats

                    p_wilks = 1 - fstats.f.cdf(f_wilks, df1, df2)
                else:
                    f_wilks = None
                    p_wilks = None
            else:
                f_wilks = None
                p_wilks = None

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += (
                "<<COLOR:title>>MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)<</COLOR>>\n"
            )
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += (
                f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
            )
            summary_text += (
                f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
            )
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += (
                "<<COLOR:accent>>── Multivariate Test Statistics ──<</COLOR>>\n"
            )
            summary_text += (
                f"{'Test':<25} {'Value':>10} {'Approx F':>10} {'p-value':>10}\n"
            )
            summary_text += f"{'─' * 57}\n"
            pillai_label = "Pillai's Trace"
            summary_text += f"{pillai_label:<25} {pillai:>10.4f} {'':>10} {'':>10}\n"
            wilks_f_str = f"{f_wilks:.4f}" if f_wilks is not None else "N/A"
            wilks_p_str = f"{p_wilks:.4f}" if p_wilks is not None else "N/A"
            wilks_label = "Wilks' Lambda"
            summary_text += f"{wilks_label:<25} {wilks:>10.4f} {wilks_f_str:>10} {wilks_p_str:>10}\n"
            summary_text += (
                f"{'Hotelling-Lawley':<25} {hotelling:>10.4f} {'':>10} {'':>10}\n"
            )
            roy_label = "Roy's Greatest Root"
            summary_text += f"{roy_label:<25} {roy:>10.4f} {'':>10} {'':>10}\n\n"

            # Univariate ANOVAs
            summary_text += (
                "<<COLOR:accent>>── Univariate ANOVA per Response ──<</COLOR>>\n"
            )
            summary_text += f"{'Response':<20} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
            summary_text += f"{'─' * 47}\n"
            from scipy import stats as fstats

            for resp in responses:
                group_vals = [data[data[factor] == g][resp].values for g in groups]
                f_stat, p_val = fstats.f_oneway(*group_vals)
                sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
                summary_text += f"{resp:<20} {f_stat:>10.4f} {p_val:>10.4f} {sig:>5}\n"

            if p_wilks is not None and p_wilks < alpha:
                summary_text += f"\n<<COLOR:good>>Significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"
            elif p_wilks is not None:
                summary_text += f"\n<<COLOR:text>>No significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"

            result["summary"] = summary_text

            # Mean profiles plot
            for resp in responses:
                means = [float(data[data[factor] == g][resp].mean()) for g in groups]
                sds = [float(data[data[factor] == g][resp].std()) for g in groups]
                result["plots"].append(
                    {
                        "title": f"Group Means: {resp} by {factor}",
                        "data": [
                            {
                                "x": [str(g) for g in groups],
                                "y": means,
                                "error_y": {
                                    "type": "data",
                                    "array": sds,
                                    "visible": True,
                                },
                                "type": "bar",
                                "marker": {"color": "#4a90d9"},
                            }
                        ],
                        "layout": {
                            "height": 250,
                            "yaxis": {"title": resp},
                            "xaxis": {"title": factor},
                        },
                    }
                )

            result["statistics"] = {
                "pillai": float(pillai),
                "wilks_lambda": float(wilks),
                "hotelling_lawley": float(hotelling),
                "roys_greatest_root": float(roy),
                "f_wilks": float(f_wilks) if f_wilks else None,
                "p_wilks": float(p_wilks) if p_wilks else None,
                "n_groups": k,
                "n_responses": p,
                "n": N,
            }
            result["guide_observation"] = (
                f"MANOVA: {', '.join(responses)} by {factor}. Wilks' Λ={wilks:.4f}"
                + (f", p={p_wilks:.4f}" if p_wilks else "")
                + "."
            )

            # Narrative
            try:
                _mv2_sig = p_wilks is not None and p_wilks < alpha
                result["narrative"] = _narrative(
                    f"MANOVA — {'Significant' if _mv2_sig else 'No significant'} multivariate effect",
                    f"Testing {', '.join(responses)} jointly by {factor} ({k} groups, N = {N}). "
                    + (
                        f"Wilks' Λ = {wilks:.4f}"
                        + (f" (p = {p_wilks:.4f})" if p_wilks else "")
                        + ". "
                    )
                    + (
                        "The factor significantly affects the responses jointly."
                        if _mv2_sig
                        else "No evidence of a joint multivariate effect."
                    ),
                    next_steps=(
                        "Examine the per-response bar charts to see which variables drive the group separation."
                        if _mv2_sig
                        else None
                    ),
                    chart_guidance="Bar charts show group means ± SD for each response. Large non-overlapping error bars suggest meaningful differences.",
                )
            except Exception:
                pass

        except Exception as e:
            result["summary"] = f"MANOVA error: {str(e)}"

    elif analysis_id == "tolerance_interval":
        """
        Tolerance Intervals — bounds containing a specified proportion of the population
        with a given confidence level. Normal-based and non-parametric methods.
        """
        var = config.get("var") or config.get("response")
        conf = config.get("conf", 95) / 100
        coverage = config.get("coverage", 95) / 100
        method = config.get("method", "normal")

        try:
            vals = df[var].dropna().values.astype(float)
            n = len(vals)
            xbar = float(np.mean(vals))
            s = float(np.std(vals, ddof=1))

            from scipy import stats as tstats

            if method == "nonparametric":
                # Non-parametric: order statistics
                sorted_vals = np.sort(vals)
                # Find r such that P(X_(r) to X_(n-r+1) covers coverage% with conf% confidence)
                from scipy.stats import beta as beta_dist

                best_r = 1
                for r in range(1, n // 2):
                    prob = beta_dist.cdf(coverage, n - 2 * r + 1, 2 * r)
                    if prob >= conf:
                        best_r = r
                        break
                lower = float(sorted_vals[best_r - 1])
                upper = float(sorted_vals[n - best_r])
                k_factor = None
                method_desc = f"Non-parametric (order statistics r={best_r})"
            else:
                # Normal-based: k-factor from chi-squared and normal quantiles
                z_p = float(tstats.norm.ppf((1 + coverage) / 2))
                chi2_val = float(tstats.chi2.ppf(1 - conf, n - 1))
                k_factor = z_p * np.sqrt((n - 1) * (1 + 1 / n) / chi2_val)
                lower = xbar - k_factor * s
                upper = xbar + k_factor * s
                method_desc = f"Normal (k={k_factor:.4f})"

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += "<<COLOR:title>>TOLERANCE INTERVAL<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method_desc}\n\n"
            summary_text += (
                f"<<COLOR:highlight>>Confidence:<</COLOR>> {conf * 100:.0f}%\n"
            )
            summary_text += (
                f"<<COLOR:highlight>>Coverage:<</COLOR>> {coverage * 100:.0f}%\n\n"
            )
            summary_text += f"<<COLOR:accent>>── Tolerance Interval ──<</COLOR>> [{lower:.4f}, {upper:.4f}]\n"
            summary_text += f"<<COLOR:text>>Mean:<</COLOR>> {xbar:.4f}\n"
            summary_text += f"<<COLOR:text>>Std Dev:<</COLOR>> {s:.4f}\n\n"
            summary_text += f"<<COLOR:highlight>>Interpretation:<</COLOR>> With {conf * 100:.0f}% confidence, at least {coverage * 100:.0f}% of the population falls between {lower:.4f} and {upper:.4f}."

            result["summary"] = summary_text

            # Histogram with tolerance bounds
            hist_vals, bin_edges = np.histogram(vals, bins="auto")
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append(
                {
                    "title": f"Tolerance Interval: {var}",
                    "data": [
                        {
                            "type": "bar",
                            "x": bin_centers,
                            "y": hist_vals.tolist(),
                            "marker": {"color": "#4a90d9", "opacity": 0.7},
                            "name": "Data",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": var},
                        "yaxis": {"title": "Frequency"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": lower,
                                "x1": lower,
                                "y0": 0,
                                "y1": max(hist_vals) * 1.1,
                                "line": {
                                    "color": "#d94a4a",
                                    "width": 2,
                                    "dash": "dash",
                                },
                            },
                            {
                                "type": "line",
                                "x0": upper,
                                "x1": upper,
                                "y0": 0,
                                "y1": max(hist_vals) * 1.1,
                                "line": {
                                    "color": "#d94a4a",
                                    "width": 2,
                                    "dash": "dash",
                                },
                            },
                            {
                                "type": "line",
                                "x0": xbar,
                                "x1": xbar,
                                "y0": 0,
                                "y1": max(hist_vals) * 1.1,
                                "line": {"color": "#4a9f6e", "width": 2},
                            },
                        ],
                        "annotations": [
                            {
                                "x": lower,
                                "y": max(hist_vals) * 1.05,
                                "text": f"Lower: {lower:.2f}",
                                "showarrow": False,
                                "font": {"color": "#d94a4a"},
                            },
                            {
                                "x": upper,
                                "y": max(hist_vals) * 1.05,
                                "text": f"Upper: {upper:.2f}",
                                "showarrow": False,
                                "font": {"color": "#d94a4a"},
                            },
                        ],
                    },
                }
            )

            result["statistics"] = {
                "mean": xbar,
                "std": s,
                "n": n,
                "lower": lower,
                "upper": upper,
                "confidence": conf,
                "coverage": coverage,
                "k_factor": k_factor,
                "method": method,
            }
            result["guide_observation"] = (
                f"Tolerance interval for {var}: [{lower:.4f}, {upper:.4f}] ({conf * 100:.0f}% conf, {coverage * 100:.0f}% coverage)."
            )

        except Exception as e:
            result["summary"] = f"Tolerance interval error: {str(e)}"

    elif analysis_id == "data_profile":
        try:
            n_rows, n_cols = df.shape
            mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

            # Summary stats table
            desc = df.describe(include="all").T
            desc["missing"] = df.isnull().sum()
            desc["missing%"] = (df.isnull().sum() / n_rows * 100).round(1)
            desc["unique"] = df.nunique()
            desc["dtype"] = df.dtypes

            tbl_rows = []
            for col in df.columns:
                r = desc.loc[col] if col in desc.index else {}
                tbl_rows.append(
                    f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td>"
                    f"<td>{r.get('mean', ''):.4g}</td>"
                    if col in numeric_cols
                    else f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td><td>-</td>"
                    f"<td>{r.get('top', '-')}</td></tr>"
                )

            table_html = "<table class='result-table'><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing%</th><th>Unique</th><th>Mean/Top</th></tr>"
            for col in df.columns:
                miss = int(df[col].isnull().sum())
                miss_pct = miss / n_rows * 100 if n_rows > 0 else 0
                uniq = df[col].nunique()
                if col in numeric_cols:
                    val = f"{df[col].mean():.4g}"
                else:
                    top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                    val = str(top)[:20]
                table_html += f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{miss}</td><td>{miss_pct:.1f}%</td><td>{uniq}</td><td>{val}</td></tr>"
            table_html += "</table>"

            # Top correlations
            corr_text = ""
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        pairs.append(
                            (
                                numeric_cols[i],
                                numeric_cols[j],
                                abs(corr_matrix.iloc[i, j]),
                                corr_matrix.iloc[i, j],
                            )
                        )
                pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = pairs[:10]
                corr_lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, _, v in top_pairs]
                corr_text = "\n".join(corr_lines)

            total_missing = int(df.isnull().sum().sum())
            total_cells = n_rows * n_cols
            complete_rows = int((~df.isnull().any(axis=1)).sum())

            summary = f"""DATA PROFILE
{"=" * 50}
Shape: {n_rows} rows x {n_cols} columns
Memory: {mem_mb:.2f} MB

Column Types:
  Numeric:     {len(numeric_cols)}
  Categorical: {len(cat_cols)}
  Datetime:    {len(dt_cols)}

Missing Data:
  Total missing cells: {total_missing} / {total_cells} ({total_missing / total_cells * 100:.1f}%)
  Complete rows: {complete_rows} / {n_rows} ({complete_rows / n_rows * 100:.1f}%)

Top Correlations:
{corr_text if corr_text else "  (need 2+ numeric columns)"}"""

            result["summary"] = summary
            result["tables"] = [{"title": "Column Summary", "html": table_html}]

            # Plot 1: Correlation heatmap
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "heatmap",
                                "z": corr_matrix.values.tolist(),
                                "x": numeric_cols,
                                "y": numeric_cols,
                                "colorscale": "RdBu_r",
                                "zmin": -1,
                                "zmax": 1,
                                "text": [
                                    [f"{v:.2f}" for v in row]
                                    for row in corr_matrix.values
                                ],
                                "texttemplate": "%{text}",
                                "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>",
                            }
                        ],
                        "layout": {"title": "Correlation Matrix", "height": 400},
                    }
                )

            # Plot 2: Missing percentage bar chart
            miss_cols = df.columns[df.isnull().any()].tolist()
            if miss_cols:
                miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "bar",
                                "x": [round(p, 1) for p in miss_pcts],
                                "y": miss_cols,
                                "orientation": "h",
                                "marker": {"color": "rgba(232,71,71,0.6)"},
                            }
                        ],
                        "layout": {
                            "title": "Missing Data by Column (%)",
                            "height": max(250, len(miss_cols) * 25),
                            "xaxis": {"title": "% Missing"},
                            "margin": {"l": 120},
                        },
                    }
                )

            # Plot 3: Distribution grid (top 6 numeric)
            for col in numeric_cols[:6]:
                vals = df[col].dropna().tolist()
                if len(vals) > 0:
                    result["plots"].append(
                        {
                            "data": [
                                {
                                    "type": "histogram",
                                    "x": vals,
                                    "marker": {"color": "rgba(74,144,217,0.6)"},
                                    "nbinsx": 30,
                                }
                            ],
                            "layout": {
                                "title": f"Distribution: {col}",
                                "height": 250,
                                "xaxis": {"title": col},
                                "yaxis": {"title": "Count"},
                            },
                        }
                    )

            result["guide_observation"] = (
                f"Data profile: {n_rows} rows, {n_cols} cols, {total_missing} missing cells, {len(numeric_cols)} numeric columns."
            )

            # Narrative
            _dp_miss_note = (
                f" {total_missing} cells are missing ({total_missing / total_cells * 100:.1f}%)."
                if total_missing > 0
                else " No missing data detected."
            )
            _dp_top_corr = ""
            if len(numeric_cols) >= 2:
                _corr = df[numeric_cols].corr()
                _tri = _corr.where(np.triu(np.ones(_corr.shape, dtype=bool), k=1))
                _max_pair = (
                    _tri.stack().abs().idxmax()
                    if _tri.stack().abs().max() > 0.5
                    else None
                )
                if _max_pair:
                    _dp_top_corr = f" Strongest correlation: <strong>{_max_pair[0]}</strong> ↔ <strong>{_max_pair[1]}</strong> (r = {_corr.loc[_max_pair[0], _max_pair[1]]:.3f})."
            result["narrative"] = _narrative(
                f"Dataset: {n_rows:,} rows × {n_cols} columns ({len(numeric_cols)} numeric, {len(df.select_dtypes(include=['object', 'category']).columns)} categorical)",
                f"The dataset contains {n_rows:,} observations across {n_cols} variables.{_dp_miss_note}{_dp_top_corr}",
                next_steps="Check distributions for skewness, then run normality tests before parametric analysis.",
                chart_guidance="The correlation heatmap highlights linear relationships. Red/blue extremes indicate strong positive/negative associations.",
            )

        except Exception as e:
            result["summary"] = f"Data profile error: {str(e)}"

    # ── Auto Profile (lightweight, runs on import) ───────────────────────
    elif analysis_id == "auto_profile":
        try:
            n_rows, n_cols = df.shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
            total_missing = int(df.isnull().sum().sum())
            total_cells = n_rows * n_cols
            miss_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

            # Per-column one-liner stats
            col_lines = []
            for col in df.columns:
                miss = int(df[col].isnull().sum())
                miss_p = miss / n_rows * 100 if n_rows > 0 else 0
                if col in numeric_cols:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        col_lines.append(
                            f"  <<COLOR:accent>>{col}<</COLOR>>  N={len(vals)}  Mean={vals.mean():.4g}  StDev={vals.std():.4g}  Min={vals.min():.4g}  Max={vals.max():.4g}  Missing={miss_p:.1f}%"
                        )
                    else:
                        col_lines.append(
                            f"  <<COLOR:warning>>{col}<</COLOR>>  (all missing)"
                        )
                else:
                    uniq = df[col].nunique()
                    top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                    col_lines.append(
                        f'  <<COLOR:accent>>{col}<</COLOR>>  Unique={uniq}  Top="{str(top)[:20]}"  Missing={miss_p:.1f}%'
                    )

            summary = f"""<<COLOR:title>>DATA OVERVIEW<</COLOR>>
{"=" * 50}
<<COLOR:highlight>>{n_rows}<</COLOR>> rows × <<COLOR:highlight>>{n_cols}<</COLOR>> columns
Numeric: {len(numeric_cols)}  |  Categorical: {len(cat_cols)}  |  Datetime: {len(dt_cols)}
Missing: {total_missing} / {total_cells} ({miss_pct:.1f}%)

<<COLOR:title>>Column Summary<</COLOR>>
{chr(10).join(col_lines)}"""

            result["summary"] = summary

            # Correlation heatmap (if 2+ numeric)
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "heatmap",
                                "z": corr_matrix.values.tolist(),
                                "x": numeric_cols,
                                "y": numeric_cols,
                                "colorscale": "RdBu_r",
                                "zmin": -1,
                                "zmax": 1,
                                "text": [
                                    [f"{v:.2f}" for v in row]
                                    for row in corr_matrix.values
                                ],
                                "texttemplate": "%{text}",
                                "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>",
                            }
                        ],
                        "layout": {"title": "Correlation Matrix", "height": 350},
                    }
                )

            # Distribution histograms (up to 12 numeric)
            for col in numeric_cols[:12]:
                vals = df[col].dropna().tolist()
                if len(vals) > 0:
                    result["plots"].append(
                        {
                            "data": [
                                {
                                    "type": "histogram",
                                    "x": vals,
                                    "marker": {"color": "rgba(74,144,217,0.6)"},
                                    "nbinsx": 30,
                                }
                            ],
                            "layout": {
                                "title": f"{col}",
                                "height": 220,
                                "xaxis": {"title": col},
                                "yaxis": {"title": "Count"},
                                "margin": {"t": 30, "b": 40, "l": 50, "r": 20},
                            },
                        }
                    )

            # Missing bar if any
            miss_cols = df.columns[df.isnull().any()].tolist()
            if miss_cols:
                miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "bar",
                                "x": [round(p, 1) for p in miss_pcts],
                                "y": miss_cols,
                                "orientation": "h",
                                "marker": {"color": "rgba(232,71,71,0.6)"},
                            }
                        ],
                        "layout": {
                            "title": "Missing Data (%)",
                            "height": max(200, len(miss_cols) * 22),
                            "xaxis": {"title": "% Missing"},
                            "margin": {"l": 120},
                        },
                    }
                )

            result["guide_observation"] = (
                f"Auto-profile: {n_rows} rows, {n_cols} cols, {total_missing} missing, {len(numeric_cols)} numeric."
            )

            # Narrative
            _ap_miss = (
                f" {total_missing} missing cells ({miss_pct:.1f}%) — check the missing data bar chart."
                if total_missing > 0
                else " Dataset is complete (no missing values)."
            )
            result["narrative"] = _narrative(
                f"Data Overview: {n_rows:,} rows × {n_cols} columns",
                f"{len(numeric_cols)} numeric, {len(cat_cols)} categorical, {len(dt_cols)} datetime columns.{_ap_miss}",
                next_steps="Review distributions for normality. Address missing data before hypothesis testing.",
                chart_guidance="Histograms show the shape of each numeric variable. Look for skewness, bimodality, or outliers.",
            )

        except Exception as e:
            result["summary"] = f"Auto-profile error: {str(e)}"

    # ── Graphical Summary (Minitab-style) ────────────────────────────────
    elif analysis_id == "graphical_summary":
        try:
            conf_level = config.get("confidence", 0.95)
            selected = config.get("vars", [])
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not selected:
                selected = numeric_cols
            selected = [c for c in selected if c in numeric_cols]

            if not selected:
                result["summary"] = (
                    "No numeric columns selected or available for Graphical Summary."
                )
            else:
                all_summaries = []
                for col in selected:
                    vals = df[col].dropna().values
                    n = len(vals)
                    n_star = int(df[col].isnull().sum())

                    if n < 3:
                        all_summaries.append(
                            f"<<COLOR:title>>{col}<</COLOR>>: insufficient data (N={n})"
                        )
                        continue

                    # Descriptive stats
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1))
                    var_val = std_val**2
                    se = std_val / np.sqrt(n)
                    skew_val = float(sp_stats.skew(vals, bias=False))
                    kurt_val = float(sp_stats.kurtosis(vals, bias=False))
                    q1 = float(np.percentile(vals, 25))
                    median_val = float(np.median(vals))
                    q3 = float(np.percentile(vals, 75))
                    min_val = float(np.min(vals))
                    max_val = float(np.max(vals))

                    # Anderson-Darling test
                    ad_result = sp_stats.anderson(vals, dist="norm")
                    ad_stat = ad_result.statistic
                    # Use 5% significance level (index 2 in critical_values)
                    ad_crit = (
                        ad_result.critical_values[2]
                        if len(ad_result.critical_values) > 2
                        else ad_result.critical_values[-1]
                    )
                    ad_sig = (
                        ad_result.significance_level[2]
                        if len(ad_result.significance_level) > 2
                        else ad_result.significance_level[-1]
                    )
                    ad_pass = ad_stat < ad_crit
                    ad_verdict = (
                        "<<COLOR:success>>Yes (fail to reject H₀)<</COLOR>>"
                        if ad_pass
                        else "<<COLOR:danger>>No (reject H₀)<</COLOR>>"
                    )

                    # CI for mean (t-interval)
                    ci_mean = sp_stats.t.interval(
                        conf_level, df=n - 1, loc=mean_val, scale=se
                    )

                    # CI for median (nonparametric sign-test inversion)
                    alpha = 1 - conf_level
                    from scipy.stats import binom

                    j = 0
                    for k_idx in range(n):
                        if binom.cdf(k_idx, n, 0.5) >= alpha / 2:
                            j = k_idx
                            break
                    sorted_vals = np.sort(vals)
                    ci_median_lo = (
                        float(sorted_vals[j]) if j < n else float(sorted_vals[0])
                    )
                    ci_median_hi = (
                        float(sorted_vals[n - 1 - j])
                        if (n - 1 - j) >= 0
                        else float(sorted_vals[-1])
                    )

                    # CI for StDev (chi-square)
                    chi2_lo = sp_stats.chi2.ppf((1 + conf_level) / 2, n - 1)
                    chi2_hi = sp_stats.chi2.ppf((1 - conf_level) / 2, n - 1)
                    ci_std_lo = np.sqrt((n - 1) * var_val / chi2_lo)
                    ci_std_hi = np.sqrt((n - 1) * var_val / chi2_hi)

                    pct_str = f"{conf_level * 100:.0f}%"

                    # Summary text
                    summ = f"""<<COLOR:title>>{"═" * 50}
{col}
{"═" * 50}<</COLOR>>

<<COLOR:accent>>Anderson-Darling Normality Test<</COLOR>>
  A² = {ad_stat:.4f}    Critical ({ad_sig:.0f}%) = {ad_crit:.4f}
  Normal? {ad_verdict}

<<COLOR:accent>>Descriptive Statistics<</COLOR>>
  N = {n}    N* = {n_star}
  Mean     = {mean_val:.6g}       StDev    = {std_val:.6g}
  Variance = {var_val:.6g}       Skewness = {skew_val:.4f}
  Kurtosis = {kurt_val:.4f}
  Minimum  = {min_val:.6g}       Q1       = {q1:.6g}
  Median   = {median_val:.6g}       Q3       = {q3:.6g}
  Maximum  = {max_val:.6g}

<<COLOR:accent>>Confidence Intervals ({pct_str})<</COLOR>>
  Mean:   ({ci_mean[0]:.6g}, {ci_mean[1]:.6g})
  Median: ({ci_median_lo:.6g}, {ci_median_hi:.6g})
  StDev:  ({ci_std_lo:.6g}, {ci_std_hi:.6g})"""
                    all_summaries.append(summ)

                    # ── Plotly figure: 3-row subplot ──
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        row_heights=[0.55, 0.20, 0.25],
                        shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=[
                            f"{col} — Histogram + Normal Fit",
                            "Boxplot",
                            f"{pct_str} Confidence Intervals",
                        ],
                    )

                    # Row 1: Histogram + normal curve
                    nbins = min(max(int(np.sqrt(n)), 10), 50)
                    counts, bin_edges = np.histogram(vals, bins=nbins)
                    bin_width = bin_edges[1] - bin_edges[0]

                    fig.add_trace(
                        go.Histogram(
                            x=vals.tolist(),
                            nbinsx=nbins,
                            marker=dict(
                                color="rgba(74,144,217,0.6)",
                                line=dict(color="rgba(74,144,217,1)", width=1),
                            ),
                            name="Data",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                    # Normal PDF overlay scaled to histogram
                    x_fit = np.linspace(
                        min_val - 0.5 * std_val, max_val + 0.5 * std_val, 200
                    )
                    y_fit = sp_stats.norm.pdf(x_fit, mean_val, std_val) * n * bin_width
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit.tolist(),
                            y=y_fit.tolist(),
                            mode="lines",
                            line=dict(color="rgba(232,71,71,0.9)", width=2),
                            name="Normal Fit",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                    # Row 2: Boxplot
                    fig.add_trace(
                        go.Box(
                            x=vals.tolist(),
                            orientation="h",
                            marker=dict(color="rgba(74,144,217,0.7)"),
                            line=dict(color="rgba(74,144,217,1)"),
                            fillcolor="rgba(74,144,217,0.3)",
                            name="Box",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

                    # Row 3: CI bars (mean and median)
                    fig.add_trace(
                        go.Scatter(
                            x=[ci_mean[0], mean_val, ci_mean[1]],
                            y=["Mean", "Mean", "Mean"],
                            mode="lines+markers",
                            marker=dict(
                                size=[8, 12, 8],
                                color=[
                                    "rgba(232,71,71,0.8)",
                                    "rgba(232,71,71,1)",
                                    "rgba(232,71,71,0.8)",
                                ],
                                symbol=["line-ns", "diamond", "line-ns"],
                            ),
                            line=dict(color="rgba(232,71,71,0.8)", width=2),
                            name="Mean CI",
                            showlegend=False,
                        ),
                        row=3,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[ci_median_lo, median_val, ci_median_hi],
                            y=["Median", "Median", "Median"],
                            mode="lines+markers",
                            marker=dict(
                                size=[8, 12, 8],
                                color=[
                                    "rgba(74,144,217,0.8)",
                                    "rgba(74,144,217,1)",
                                    "rgba(74,144,217,0.8)",
                                ],
                                symbol=["line-ns", "diamond", "line-ns"],
                            ),
                            line=dict(color="rgba(74,144,217,0.8)", width=2),
                            name="Median CI",
                            showlegend=False,
                        ),
                        row=3,
                        col=1,
                    )

                    fig.update_layout(height=520, margin=dict(t=40, b=30, l=60, r=20))
                    fig.update_yaxes(showticklabels=False, row=2, col=1)

                    # Convert to JSON-serializable dict
                    fig_dict = fig.to_dict()
                    result["plots"].append(
                        {
                            "data": [t for t in fig_dict["data"]],
                            "layout": fig_dict["layout"],
                        }
                    )

                    # HTML table for this variable
                    tbl = f"""<table class='result-table'>
<tr><th colspan='4' style='text-align:center;'>{col} — Descriptive Statistics</th></tr>
<tr><td>N</td><td>{n}</td><td>N* (missing)</td><td>{n_star}</td></tr>
<tr><td>Mean</td><td>{mean_val:.6g}</td><td>StDev</td><td>{std_val:.6g}</td></tr>
<tr><td>Variance</td><td>{var_val:.6g}</td><td>Skewness</td><td>{skew_val:.4f}</td></tr>
<tr><td>Kurtosis</td><td>{kurt_val:.4f}</td><td>A² Statistic</td><td>{ad_stat:.4f}</td></tr>
<tr><td>Minimum</td><td>{min_val:.6g}</td><td>Q1</td><td>{q1:.6g}</td></tr>
<tr><td>Median</td><td>{median_val:.6g}</td><td>Q3</td><td>{q3:.6g}</td></tr>
<tr><td>Maximum</td><td>{max_val:.6g}</td><td>IQR</td><td>{q3 - q1:.6g}</td></tr>
<tr><th colspan='4'>Confidence Intervals ({pct_str})</th></tr>
<tr><td>Mean</td><td>({ci_mean[0]:.6g}, {ci_mean[1]:.6g})</td><td>Median</td><td>({ci_median_lo:.6g}, {ci_median_hi:.6g})</td></tr>
<tr><td>StDev</td><td>({ci_std_lo:.6g}, {ci_std_hi:.6g})</td><td></td><td></td></tr>
</table>"""
                    result["tables"].append({"title": f"{col} Statistics", "html": tbl})

                result["summary"] = "\n\n".join(all_summaries)
                result["guide_observation"] = (
                    f"Graphical summary for {len(selected)} variable(s) at {conf_level * 100:.0f}% confidence."
                )

                # Narrative — use first variable's stats as lead
                if selected and len(all_summaries) > 0:
                    result["narrative"] = _narrative(
                        f"Graphical Summary — {len(selected)} variable{'s' if len(selected) > 1 else ''} at {conf_level * 100:.0f}% confidence",
                        "Each panel shows the histogram with normal fit overlay, boxplot, and confidence intervals for mean and median. Review the Anderson-Darling test result to determine if the normal distribution is appropriate.",
                        next_steps="If data departs from normality, consider non-parametric tests or a Box-Cox / Johnson transform.",
                        chart_guidance="Points in the boxplot beyond the whiskers are potential outliers. If the histogram shape is skewed or bimodal, the normal fit (red curve) will visually misfit.",
                    )

        except Exception as e:
            import traceback

            result["summary"] = (
                f"Graphical Summary error: {str(e)}\n{traceback.format_exc()}"
            )

    # ── Missing Data Analysis ─────────────────────────────────────────────
    elif analysis_id == "missing_data_analysis":
        try:
            n_rows, n_cols = df.shape
            miss_count = df.isnull().sum()
            miss_pct = (miss_count / n_rows * 100).round(2)
            cols_with_missing = miss_count[miss_count > 0].sort_values(ascending=False)

            # Row completeness
            row_completeness = (~df.isnull()).sum(axis=1) / n_cols * 100
            complete_rows = int((row_completeness == 100).sum())

            # Missing patterns
            miss_indicator = df.isnull().astype(int)
            pattern_strs = miss_indicator.apply(
                lambda r: "".join(str(v) for v in r), axis=1
            )
            pattern_counts = pattern_strs.value_counts()
            n_patterns = len(pattern_counts)

            # Pattern table
            pattern_rows = []
            for pat, cnt in pattern_counts.head(15).items():
                cols_missing = [df.columns[i] for i, v in enumerate(pat) if v == "1"]
                desc = ", ".join(cols_missing) if cols_missing else "(complete)"
                pattern_rows.append(
                    f"  {cnt:>6} rows ({cnt / n_rows * 100:.1f}%): {desc}"
                )
            pattern_text = "\n".join(pattern_rows)

            # Little's MCAR test approximation
            mcar_text = ""
            if len(cols_with_missing) >= 2 and len(cols_with_missing) < n_cols:
                try:
                    observed_counts = pattern_counts.values
                    # Expected under MCAR: each column missing independently
                    col_miss_rates = miss_count / n_rows
                    expected_probs = []
                    for pat in pattern_counts.index:
                        prob = 1.0
                        for i, v in enumerate(pat):
                            p_miss = col_miss_rates.iloc[i]
                            prob *= p_miss if v == "1" else (1 - p_miss)
                        expected_probs.append(prob)
                    expected_counts = np.array(expected_probs) * n_rows
                    # Filter out zero-expected
                    mask = expected_counts > 0.5
                    if mask.sum() >= 2:
                        from scipy.stats import chi2

                        obs = observed_counts[mask]
                        exp = expected_counts[mask]
                        chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
                        dof = int(mask.sum() - 1)
                        p_val = float(1 - chi2.cdf(chi2_stat, dof))
                        conclusion = (
                            "Data appears MCAR (p >= 0.05)"
                            if p_val >= 0.05
                            else "Data may NOT be MCAR (p < 0.05)"
                        )
                        mcar_text = f"""
MCAR Test (Chi-squared approximation):
  Chi-squared: {chi2_stat:.2f}  (df={dof})
  p-value: {p_val:.4f}
  Conclusion: {conclusion}"""
                except Exception:
                    mcar_text = "\nMCAR Test: Could not compute (insufficient patterns)"

            # Missing correlation
            miss_corr_text = ""
            if len(cols_with_missing) >= 2:
                miss_corr = miss_indicator[cols_with_missing.index].corr()
                pairs = []
                idx_list = cols_with_missing.index.tolist()
                for i in range(len(idx_list)):
                    for j in range(i + 1, len(idx_list)):
                        pairs.append(
                            (
                                idx_list[i],
                                idx_list[j],
                                miss_corr.loc[idx_list[i], idx_list[j]],
                            )
                        )
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, v in pairs[:5]]
                    miss_corr_text = "\nMissing Correlation (top pairs):\n" + "\n".join(
                        lines
                    )

            summary_lines = ["MISSING DATA ANALYSIS", "=" * 50]
            summary_lines.append(f"Dataset: {n_rows} rows x {n_cols} columns")
            summary_lines.append(
                f"Total missing: {int(miss_count.sum())} / {n_rows * n_cols} ({miss_count.sum() / (n_rows * n_cols) * 100:.1f}%)"
            )
            summary_lines.append(
                f"Complete rows: {complete_rows} / {n_rows} ({complete_rows / n_rows * 100:.1f}%)"
            )
            summary_lines.append(
                f"\nColumns with missing data ({len(cols_with_missing)}):"
            )
            for col, cnt in cols_with_missing.items():
                summary_lines.append(f"  {col:<30} {cnt:>6} ({miss_pct[col]:.1f}%)")
            summary_lines.append(f"\nMissing Patterns ({n_patterns} unique):")
            summary_lines.append(pattern_text)
            if mcar_text:
                summary_lines.append(mcar_text)
            if miss_corr_text:
                summary_lines.append(miss_corr_text)

            result["summary"] = "\n".join(summary_lines)

            # Plot 1: Missing pattern heatmap
            if len(cols_with_missing) > 0:
                top_patterns = pattern_counts.head(20)
                z_data = []
                y_labels = []
                for pat, cnt in top_patterns.items():
                    z_data.append([int(v) for v in pat])
                    cols_miss = sum(1 for v in pat if v == "1")
                    y_labels.append(f"{cnt} rows ({cols_miss} missing)")
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "heatmap",
                                "z": z_data,
                                "x": df.columns.tolist(),
                                "y": y_labels,
                                "colorscale": [
                                    [0, "rgba(74,144,217,0.15)"],
                                    [1, "rgba(232,71,71,0.7)"],
                                ],
                                "showscale": False,
                                "hovertemplate": "%{x}: %{z}<extra>0=present, 1=missing</extra>",
                            }
                        ],
                        "layout": {
                            "title": "Missing Data Patterns",
                            "height": max(300, len(z_data) * 25 + 100),
                            "xaxis": {"tickangle": -45},
                            "margin": {"b": 100, "l": 150},
                        },
                    }
                )

            # Plot 2: Row completeness histogram
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "histogram",
                            "x": row_completeness.tolist(),
                            "nbinsx": 20,
                            "marker": {"color": "rgba(74,159,110,0.6)"},
                        }
                    ],
                    "layout": {
                        "title": "Row Completeness Distribution",
                        "height": 280,
                        "xaxis": {"title": "% Complete", "range": [0, 105]},
                        "yaxis": {"title": "Row Count"},
                    },
                }
            )

            result["guide_observation"] = (
                f"Missing data: {int(miss_count.sum())} cells across {len(cols_with_missing)} columns. {n_patterns} unique patterns."
            )

            # Narrative
            _md_total = int(miss_count.sum())
            _md_pct = _md_total / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0
            _md_worst = (
                cols_with_missing.index[0] if len(cols_with_missing) > 0 else "N/A"
            )
            _md_worst_pct = (
                float(miss_pct[_md_worst]) if len(cols_with_missing) > 0 else 0
            )
            _md_severity = (
                "minimal"
                if _md_pct < 5
                else ("moderate" if _md_pct < 20 else "substantial")
            )
            result["narrative"] = _narrative(
                f"Missing Data — {_md_severity} ({_md_pct:.1f}% of cells)",
                f"{_md_total:,} missing cells across {len(cols_with_missing)} columns in {n_patterns} unique patterns. "
                + (
                    f"Worst column: <strong>{_md_worst}</strong> ({_md_worst_pct:.1f}% missing). "
                    if len(cols_with_missing) > 0
                    else ""
                )
                + f"{complete_rows} of {n_rows} rows ({complete_rows / n_rows * 100:.1f}%) are complete.",
                next_steps="If missingness is random (MCAR), listwise deletion is safe. Otherwise consider multiple imputation.",
                chart_guidance="The heatmap shows which columns are missing together (correlated patterns suggest non-random missingness).",
            )

        except Exception as e:
            result["summary"] = f"Missing data analysis error: {str(e)}"

    # ── Outlier Analysis ──────────────────────────────────────────────────
    elif analysis_id == "outlier_analysis":
        try:
            columns = config.get("columns", [])
            methods = config.get("methods", ["iqr"])
            iqr_mult = float(config.get("iqr_multiplier", 1.5))
            z_thresh = float(config.get("zscore_threshold", 3.0))

            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [
                c
                for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

            if not columns:
                result["summary"] = "No numeric columns found for outlier analysis."
                return result

            all_results = {}
            consensus = np.zeros(len(df), dtype=int)

            for col in columns:
                vals = df[col].dropna()
                col_results = {}

                if "iqr" in methods:
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                    mask = (df[col] < lower) | (df[col] > upper)
                    col_results["IQR"] = {
                        "count": int(mask.sum()),
                        "pct": round(mask.sum() / len(df) * 100, 1),
                        "bounds": f"[{lower:.4g}, {upper:.4g}]",
                    }
                    consensus += mask.astype(int).values

                if "zscore" in methods:
                    z = (
                        np.abs((df[col] - vals.mean()) / vals.std())
                        if vals.std() > 0
                        else pd.Series(0, index=df.index)
                    )
                    mask = z > z_thresh
                    col_results["Z-score"] = {
                        "count": int(mask.sum()),
                        "pct": round(mask.sum() / len(df) * 100, 1),
                        "threshold": z_thresh,
                    }
                    consensus += mask.astype(int).values

                if "modified_zscore" in methods:
                    median = vals.median()
                    mad = np.median(np.abs(vals - median))
                    if mad > 0:
                        modified_z = 0.6745 * np.abs(df[col] - median) / mad
                        mask = modified_z > 3.5
                    else:
                        mask = pd.Series(False, index=df.index)
                    col_results["Modified Z"] = {
                        "count": int(mask.sum()),
                        "pct": round(mask.sum() / len(df) * 100, 1),
                        "threshold": 3.5,
                    }
                    consensus += mask.astype(int).values

                if "mahalanobis" in methods and len(columns) >= 2:
                    try:
                        from scipy.spatial.distance import mahalanobis as mah_dist

                        sub = df[columns].dropna()
                        if len(sub) > len(columns):
                            cov = np.cov(sub.T)
                            cov_inv = np.linalg.inv(cov + np.eye(len(columns)) * 1e-6)
                            mean = sub.mean().values
                            dists = sub.apply(
                                lambda r: mah_dist(r.values, mean, cov_inv), axis=1
                            )
                            from scipy.stats import chi2 as chi2_dist

                            threshold = chi2_dist.ppf(0.975, df=len(columns))
                            mask_full = dists > np.sqrt(threshold)
                            col_results["Mahalanobis"] = {
                                "count": int(mask_full.sum()),
                                "pct": round(mask_full.sum() / len(sub) * 100, 1),
                                "threshold": f"chi2(p=0.975, df={len(columns)})",
                            }
                    except Exception:
                        col_results["Mahalanobis"] = {
                            "count": 0,
                            "pct": 0,
                            "error": "Could not compute",
                        }

                all_results[col] = col_results

            # Build summary
            summary_lines = ["OUTLIER ANALYSIS", "=" * 50]
            summary_lines.append(f"Columns: {', '.join(columns)}")
            summary_lines.append(f"Methods: {', '.join(methods)}")
            summary_lines.append(f"Rows: {len(df)}\n")

            for col, methods_res in all_results.items():
                summary_lines.append(f"{col}:")
                for method, info in methods_res.items():
                    summary_lines.append(
                        f"  {method:<18} {info['count']:>5} outliers ({info['pct']}%)"
                    )
                summary_lines.append("")

            # Consensus
            n_methods_used = len(
                [m for m in methods if m != "mahalanobis" or len(columns) >= 2]
            )
            if n_methods_used >= 2:
                (
                    int((consensus >= n_methods_used * len(columns)).sum())
                    if n_methods_used > 0
                    else 0
                )
                flagged_majority = int(
                    (consensus >= max(1, n_methods_used * len(columns) // 2)).sum()
                )
                summary_lines.append("Consensus:")
                summary_lines.append(
                    f"  Flagged by majority of methods: {flagged_majority} rows"
                )

            result["summary"] = "\n".join(summary_lines)

            # Table
            table_html = "<table class='result-table'><tr><th>Column</th><th>Method</th><th>Outliers</th><th>%</th><th>Details</th></tr>"
            for col, methods_res in all_results.items():
                for method, info in methods_res.items():
                    detail = info.get("bounds", info.get("threshold", ""))
                    table_html += f"<tr><td>{col}</td><td>{method}</td><td>{info['count']}</td><td>{info['pct']}%</td><td>{detail}</td></tr>"
            table_html += "</table>"
            result["tables"] = [{"title": "Outlier Summary", "html": table_html}]

            # Plot: Boxplots
            for col in columns[:6]:
                vals = df[col].dropna().tolist()
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "box",
                                "y": vals,
                                "name": col,
                                "boxpoints": "outliers",
                                "marker": {
                                    "color": "rgba(74,144,217,0.6)",
                                    "outliercolor": "rgba(232,71,71,0.8)",
                                },
                                "line": {"color": "rgba(74,144,217,0.8)"},
                            }
                        ],
                        "layout": {
                            "title": f"Outlier Detection: {col}",
                            "height": 300,
                            "yaxis": {"title": col},
                        },
                    }
                )

            result["guide_observation"] = (
                f"Outlier analysis on {len(columns)} columns with {len(methods)} methods."
            )

            # Narrative
            _oa_total = sum(
                info["count"]
                for col_res in all_results.values()
                for info in col_res.values()
            )
            _oa_worst_col = (
                max(
                    all_results.keys(),
                    key=lambda c: max(
                        info["count"] for info in all_results[c].values()
                    ),
                )
                if all_results
                else ""
            )
            _oa_worst_n = (
                max(info["count"] for info in all_results[_oa_worst_col].values())
                if _oa_worst_col
                else 0
            )
            result["narrative"] = _narrative(
                f"Outlier Analysis — {len(columns)} columns, {len(methods)} method{'s' if len(methods) > 1 else ''}",
                (
                    f"Most outliers detected in <strong>{_oa_worst_col}</strong> ({_oa_worst_n} points). "
                    if _oa_worst_col
                    else ""
                )
                + f"Methods used: {', '.join(methods)}. Review boxplots for visual confirmation before removing any points.",
                next_steps="Investigate outliers for data entry errors or special causes. Do not blindly remove — they may carry process information.",
                chart_guidance="Points beyond whiskers in the boxplots are flagged. Compare across methods for consensus.",
            )

        except Exception as e:
            result["summary"] = f"Outlier analysis error: {str(e)}"

    # ── Duplicate Analysis ────────────────────────────────────────────────
    elif analysis_id == "duplicate_analysis":
        try:
            mode = config.get("mode", "exact")
            subset_cols = config.get("subset_columns", [])

            if mode == "subset" and subset_cols:
                check_cols = [c for c in subset_cols if c in df.columns]
            else:
                check_cols = df.columns.tolist()

            duplicated_mask = df.duplicated(subset=check_cols, keep=False)
            n_dup_rows = int(duplicated_mask.sum())
            n_dup_groups = (
                int(df[duplicated_mask].groupby(check_cols).ngroups)
                if n_dup_rows > 0
                else 0
            )
            first_dup_mask = df.duplicated(subset=check_cols, keep="first")
            n_extra = int(first_dup_mask.sum())

            summary_lines = ["DUPLICATE ANALYSIS", "=" * 50]
            summary_lines.append(
                f"Mode: {'Exact (all columns)' if mode == 'exact' else 'Subset (' + ', '.join(check_cols) + ')'}"
            )
            summary_lines.append(f"Total rows: {len(df)}")
            summary_lines.append(f"Unique rows: {len(df) - n_extra}")
            summary_lines.append(
                f"Duplicate rows: {n_dup_rows} ({n_dup_rows / len(df) * 100:.1f}%)"
            )
            summary_lines.append(f"Duplicate groups: {n_dup_groups}")
            summary_lines.append(f"Extra copies (removable): {n_extra}")

            # Show top duplicate groups
            if n_dup_rows > 0:
                dup_df = df[duplicated_mask].copy()
                group_sizes = (
                    dup_df.groupby(check_cols).size().sort_values(ascending=False)
                )
                summary_lines.append("\nLargest duplicate groups:")
                for i, (vals, cnt) in enumerate(group_sizes.head(10).items()):
                    if isinstance(vals, tuple):
                        desc = ", ".join(f"{c}={v}" for c, v in zip(check_cols, vals))
                    else:
                        desc = f"{check_cols[0]}={vals}"
                    summary_lines.append(f"  Group {i + 1}: {cnt} copies — {desc[:80]}")

            result["summary"] = "\n".join(summary_lines)

            # Table of sample duplicates
            if n_dup_rows > 0:
                sample = df[duplicated_mask].head(20)
                table_html = "<table class='result-table'><tr>"
                for c in sample.columns:
                    table_html += f"<th>{c}</th>"
                table_html += "<th>Row#</th></tr>"
                for idx, row in sample.iterrows():
                    table_html += "<tr>"
                    for c in sample.columns:
                        table_html += f"<td>{row[c]}</td>"
                    table_html += f"<td>{idx}</td></tr>"
                table_html += "</table>"
                result["tables"] = [
                    {"title": "Sample Duplicate Rows", "html": table_html}
                ]

            # Plot: duplicate group size histogram
            if n_dup_groups > 0:
                group_sizes_list = dup_df.groupby(check_cols).size().tolist()
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "histogram",
                                "x": group_sizes_list,
                                "marker": {"color": "rgba(232,149,71,0.6)"},
                            }
                        ],
                        "layout": {
                            "title": "Duplicate Group Sizes",
                            "height": 280,
                            "xaxis": {"title": "Copies per Group"},
                            "yaxis": {"title": "Number of Groups"},
                        },
                    }
                )

            result["guide_observation"] = (
                f"Duplicate analysis ({mode}): {n_dup_rows} duplicate rows in {n_dup_groups} groups."
            )

            # Narrative
            _da_pct = n_dup_rows / len(df) * 100 if len(df) > 0 else 0
            if n_dup_rows == 0:
                _da_verdict = "No duplicates found"
                _da_body = (
                    f"All {len(df):,} rows are unique across the checked columns."
                )
            else:
                _da_verdict = f"{n_dup_rows:,} duplicate rows ({_da_pct:.1f}%)"
                _da_body = f"{n_dup_groups} duplicate groups found. {n_extra} rows are extra copies that could be removed, leaving {len(df) - n_extra:,} unique rows."
            result["narrative"] = _narrative(
                f"Duplicate Analysis — {_da_verdict}",
                _da_body,
                next_steps=(
                    "Verify duplicates are true repeats (not valid repeat measurements) before removing."
                    if n_dup_rows > 0
                    else None
                ),
                chart_guidance=(
                    "The histogram shows how many copies exist per duplicate group."
                    if n_dup_rows > 0
                    else None
                ),
            )

        except Exception as e:
            result["summary"] = f"Duplicate analysis error: {str(e)}"

    # ── Meta-Analysis ─────────────────────────────────────────────────────
    elif analysis_id == "meta_analysis":
        try:
            mode = config.get("mode", "precomputed")
            config.get("subgroup_col", "")

            if mode == "precomputed":
                effect_col = config.get("effect_col", "")
                se_col = config.get("se_col", "")
                study_col = config.get("study_col", "")
                if not all([effect_col, se_col, study_col]):
                    result["summary"] = (
                        "Please specify effect size, SE, and study label columns."
                    )
                    return result
                effects = df[effect_col].dropna().values.astype(float)
                ses = df[se_col].dropna().values.astype(float)
                studies = df[study_col].values[: len(effects)]
            else:
                # Raw mode: compute Cohen's d
                m1c, s1c, n1c = (
                    config.get("mean1_col", ""),
                    config.get("sd1_col", ""),
                    config.get("n1_col", ""),
                )
                m2c, s2c, n2c = (
                    config.get("mean2_col", ""),
                    config.get("sd2_col", ""),
                    config.get("n2_col", ""),
                )
                study_col = config.get("study_col", "")
                if not all([m1c, s1c, n1c, m2c, s2c, n2c]):
                    result["summary"] = (
                        "Please specify all 6 raw data columns (mean, SD, n for each group)."
                    )
                    return result
                m1 = df[m1c].values.astype(float)
                s1 = df[s1c].values.astype(float)
                n1 = df[n1c].values.astype(float)
                m2 = df[m2c].values.astype(float)
                s2 = df[s2c].values.astype(float)
                n2 = df[n2c].values.astype(float)
                sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                effects = (m1 - m2) / sp
                ses = np.sqrt(1 / n1 + 1 / n2 + effects**2 / (2 * (n1 + n2)))
                studies = (
                    df[study_col].values[: len(effects)]
                    if study_col
                    else np.arange(1, len(effects) + 1)
                )

            k = len(effects)
            if k < 2:
                result["summary"] = "Need at least 2 studies for meta-analysis."
                return result

            # Weights
            w = 1.0 / (ses**2)

            # Fixed effects
            fe_est = float(np.sum(w * effects) / np.sum(w))
            fe_se = float(np.sqrt(1.0 / np.sum(w)))
            fe_ci_lo = fe_est - 1.96 * fe_se
            fe_ci_hi = fe_est + 1.96 * fe_se
            fe_z = fe_est / fe_se
            from scipy.stats import norm as norm_dist

            fe_p = float(2 * (1 - norm_dist.cdf(abs(fe_z))))

            # Heterogeneity
            Q = float(np.sum(w * (effects - fe_est) ** 2))
            df_q = k - 1
            from scipy.stats import chi2 as chi2_dist_fn

            Q_p = float(1 - chi2_dist_fn.cdf(Q, df_q))
            I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0
            H2 = Q / df_q if df_q > 0 else 1

            # DerSimonian-Laird tau²
            C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
            tau2 = max(0, (Q - df_q) / C) if C > 0 else 0

            # Random effects
            w_re = 1.0 / (ses**2 + tau2)
            re_est = float(np.sum(w_re * effects) / np.sum(w_re))
            re_se = float(np.sqrt(1.0 / np.sum(w_re)))
            re_ci_lo = re_est - 1.96 * re_se
            re_ci_hi = re_est + 1.96 * re_se
            re_z = re_est / re_se
            re_p = float(2 * (1 - norm_dist.cdf(abs(re_z))))

            interpretation = (
                "Low heterogeneity"
                if I2 < 25
                else ("Moderate heterogeneity" if I2 < 75 else "High heterogeneity")
            )
            if I2 < 40:
                interpretation += (
                    " — fixed and random effects models give similar results."
                )
            else:
                interpretation += " — random effects model is more appropriate."

            summary = f"""META-ANALYSIS
{"=" * 50}
Studies: {k}

Fixed Effects Model:
  Pooled estimate: {fe_est:.4f} (95% CI: {fe_ci_lo:.4f}, {fe_ci_hi:.4f})
  z = {fe_z:.3f}, p = {fe_p:.4f}

Random Effects Model (DerSimonian-Laird):
  Pooled estimate: {re_est:.4f} (95% CI: {re_ci_lo:.4f}, {re_ci_hi:.4f})
  z = {re_z:.3f}, p = {re_p:.4f}

Heterogeneity:
  Q = {Q:.2f} (df={df_q}, p={Q_p:.4f})
  I² = {I2:.1f}% (variation due to heterogeneity)
  tau² = {tau2:.4f} (between-study variance)
  H² = {H2:.2f}

Interpretation:
  {interpretation}"""

            result["summary"] = summary

            # Table
            table_html = "<table class='result-table'><tr><th>Study</th><th>Effect</th><th>SE</th><th>95% CI</th><th>Weight (FE)</th><th>Weight (RE)</th></tr>"
            for i in range(k):
                ci_lo = effects[i] - 1.96 * ses[i]
                ci_hi = effects[i] + 1.96 * ses[i]
                w_fe_pct = w[i] / np.sum(w) * 100
                w_re_pct = w_re[i] / np.sum(w_re) * 100
                table_html += f"<tr><td>{studies[i]}</td><td>{effects[i]:.4f}</td><td>{ses[i]:.4f}</td>"
                table_html += f"<td>[{ci_lo:.4f}, {ci_hi:.4f}]</td><td>{w_fe_pct:.1f}%</td><td>{w_re_pct:.1f}%</td></tr>"
            table_html += f"<tr style='font-weight:bold;border-top:2px solid #666;'><td>Fixed Effects</td><td>{fe_est:.4f}</td><td>{fe_se:.4f}</td><td>[{fe_ci_lo:.4f}, {fe_ci_hi:.4f}]</td><td>100%</td><td>-</td></tr>"
            table_html += f"<tr style='font-weight:bold;'><td>Random Effects</td><td>{re_est:.4f}</td><td>{re_se:.4f}</td><td>[{re_ci_lo:.4f}, {re_ci_hi:.4f}]</td><td>-</td><td>100%</td></tr>"
            table_html += "</table>"
            result["tables"] = [{"title": "Study Results", "html": table_html}]

            # Forest plot
            forest_y = list(range(k, 0, -1))
            ci_lows = [effects[i] - 1.96 * ses[i] for i in range(k)]
            ci_highs = [effects[i] + 1.96 * ses[i] for i in range(k)]
            study_labels = [str(s) for s in studies]

            forest_data = [
                # Study CIs (horizontal lines)
                {
                    "type": "scatter",
                    "x": list(effects),
                    "y": forest_y,
                    "mode": "markers",
                    "name": "Studies",
                    "marker": {
                        "size": [
                            max(4, min(16, float(w_re[i] / np.max(w_re) * 16)))
                            for i in range(k)
                        ],
                        "color": "rgba(74,144,217,0.8)",
                    },
                    "error_x": {
                        "type": "data",
                        "symmetric": False,
                        "array": [ci_highs[i] - effects[i] for i in range(k)],
                        "arrayminus": [effects[i] - ci_lows[i] for i in range(k)],
                        "color": "rgba(74,144,217,0.5)",
                        "thickness": 2,
                    },
                    "text": study_labels,
                    "hovertemplate": "%{text}: %{x:.4f}<extra></extra>",
                },
                # Pooled diamond (RE)
                {
                    "type": "scatter",
                    "x": [re_ci_lo, re_est, re_ci_hi, re_est, re_ci_lo],
                    "y": [0, -0.3, 0, 0.3, 0],
                    "mode": "lines",
                    "fill": "toself",
                    "fillcolor": "rgba(232,71,71,0.3)",
                    "line": {"color": "rgba(232,71,71,0.8)"},
                    "name": "Pooled (RE)",
                    "hoverinfo": "skip",
                },
                # Zero line
                {
                    "type": "scatter",
                    "x": [0, 0],
                    "y": [-1, k + 1],
                    "mode": "lines",
                    "line": {"color": "gray", "dash": "dash", "width": 1},
                    "showlegend": False,
                    "hoverinfo": "skip",
                },
            ]
            result["plots"].append(
                {
                    "data": forest_data,
                    "layout": {
                        "title": "Forest Plot",
                        "height": max(350, k * 30 + 120),
                        "yaxis": {
                            "tickvals": forest_y + [0],
                            "ticktext": study_labels + ["Pooled (RE)"],
                            "range": [-1, k + 1],
                        },
                        "xaxis": {"title": "Effect Size", "zeroline": True},
                        "showlegend": False,
                        "margin": {"l": 120},
                    },
                }
            )

            # Funnel plot
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "scatter",
                            "x": list(effects),
                            "y": list(ses),
                            "mode": "markers",
                            "marker": {"size": 8, "color": "rgba(74,144,217,0.7)"},
                            "name": "Studies",
                            "text": study_labels,
                            "hovertemplate": "%{text}: effect=%{x:.4f}, SE=%{y:.4f}<extra></extra>",
                        },
                        # Funnel lines
                        {
                            "type": "scatter",
                            "x": [
                                re_est - 1.96 * max(ses),
                                re_est,
                                re_est + 1.96 * max(ses),
                            ],
                            "y": [max(ses), 0, max(ses)],
                            "mode": "lines",
                            "line": {"color": "gray", "dash": "dash"},
                            "name": "95% CI",
                            "hoverinfo": "skip",
                        },
                        {
                            "type": "scatter",
                            "x": [re_est, re_est],
                            "y": [0, max(ses) * 1.1],
                            "mode": "lines",
                            "line": {"color": "rgba(232,71,71,0.5)", "dash": "dot"},
                            "name": "Pooled",
                            "hoverinfo": "skip",
                        },
                    ],
                    "layout": {
                        "title": "Funnel Plot",
                        "height": 350,
                        "xaxis": {"title": "Effect Size"},
                        "yaxis": {"title": "Standard Error", "autorange": "reversed"},
                    },
                }
            )

            result["guide_observation"] = (
                f"Meta-analysis of {k} studies. RE pooled: {re_est:.4f}. I²={I2:.1f}%. {interpretation}"
            )

            # Narrative
            _ma_sig = (
                "statistically significant"
                if re_p < 0.05
                else "not statistically significant"
            )
            _ma_het = "low" if I2 < 25 else ("moderate" if I2 < 75 else "high")
            _ma_model = (
                "Fixed effects may suffice."
                if I2 < 25
                else "Random effects model is preferred due to heterogeneity."
            )
            result["narrative"] = _narrative(
                f"Meta-Analysis — {k} studies, pooled effect = {re_est:.4f} (RE)",
                f"The random-effects pooled estimate is {re_est:.4f} (95% CI: {re_ci_lo:.4f} to {re_ci_hi:.4f}), {_ma_sig} (p = {re_p:.4f}). "
                f"Heterogeneity is {_ma_het} (I² = {I2:.1f}%, τ² = {tau2:.4f}). {_ma_model}",
                next_steps="Inspect the funnel plot for asymmetry (publication bias). Consider subgroup analysis if I² is high.",
                chart_guidance="In the forest plot, study size is proportional to marker area. The red diamond is the pooled estimate. The funnel plot should be symmetric if no publication bias exists.",
            )

        except Exception as e:
            result["summary"] = f"Meta-analysis error: {str(e)}"

    # ── Effect Size Calculator ────────────────────────────────────────────
    elif analysis_id == "effect_size_calculator":
        try:
            effect_type = config.get("effect_type", "cohens_d")
            results_list = []  # noqa: F841

            if effect_type in ("cohens_d", "hedges_g", "glass_delta"):
                m1 = float(config.get("mean1", 0))
                s1 = float(config.get("sd1", 1))
                n1 = int(config.get("n1", 10))
                m2 = float(config.get("mean2", 0))
                s2 = float(config.get("sd2", 1))
                n2 = int(config.get("n2", 10))

                sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

                if effect_type == "cohens_d":
                    d = (m1 - m2) / sp if sp > 0 else 0
                    se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2)))
                    name = "Cohen's d"
                elif effect_type == "hedges_g":
                    d_raw = (m1 - m2) / sp if sp > 0 else 0
                    J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                    d = d_raw * J
                    se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2))) * J
                    name = "Hedges' g"
                else:  # glass_delta
                    d = (m1 - m2) / s2 if s2 > 0 else 0
                    se = np.sqrt(1 / n1 + d**2 / (2 * n2))
                    name = "Glass's delta"

                ci_lo = d - 1.96 * se
                ci_hi = d + 1.96 * se

                if abs(d) < 0.2:
                    magnitude = "Negligible"
                elif abs(d) < 0.5:
                    magnitude = "Small"
                elif abs(d) < 0.8:
                    magnitude = "Medium"
                else:
                    magnitude = "Large"

                summary = f"""EFFECT SIZE CALCULATOR
{"=" * 50}
Type: {name}

Input:
  Group 1: M={m1}, SD={s1}, n={n1}
  Group 2: M={m2}, SD={s2}, n={n2}

Result:
  {name} = {d:.4f}
  Standard Error = {se:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})

Interpretation:
  Magnitude: {magnitude} (Cohen's benchmarks: 0.2=small, 0.5=medium, 0.8=large)
  Direction: {"Group 1 > Group 2" if d > 0 else "Group 2 > Group 1" if d < 0 else "No difference"}"""

                # Plot
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "scatter",
                                "x": [d],
                                "y": [name],
                                "mode": "markers",
                                "marker": {"size": 14, "color": "rgba(74,144,217,0.8)"},
                                "error_x": {
                                    "type": "data",
                                    "symmetric": False,
                                    "array": [ci_hi - d],
                                    "arrayminus": [d - ci_lo],
                                    "color": "rgba(74,144,217,0.5)",
                                    "thickness": 3,
                                },
                            },
                            {
                                "type": "scatter",
                                "x": [0, 0],
                                "y": [-0.5, 1.5],
                                "mode": "lines",
                                "line": {"color": "gray", "dash": "dash"},
                                "showlegend": False,
                            },
                        ],
                        "layout": {
                            "title": f"{name} with 95% CI",
                            "height": 200,
                            "xaxis": {"title": "Effect Size"},
                            "showlegend": False,
                        },
                    }
                )

            elif effect_type in ("odds_ratio", "risk_ratio"):
                a = int(config.get("a", 0))
                b = int(config.get("b", 0))
                c = int(config.get("c", 0))
                dd = int(config.get("d", 0))

                if effect_type == "odds_ratio":
                    if b * c > 0:
                        es = (a * dd) / (b * c)
                        se_ln = np.sqrt(
                            1 / max(a, 0.5)
                            + 1 / max(b, 0.5)
                            + 1 / max(c, 0.5)
                            + 1 / max(dd, 0.5)
                        )
                    else:
                        es, se_ln = 0, 0
                    name = "Odds Ratio"
                else:
                    r1 = a / (a + b) if (a + b) > 0 else 0
                    r2 = c / (c + dd) if (c + dd) > 0 else 0
                    es = r1 / r2 if r2 > 0 else 0
                    se_ln = np.sqrt(
                        1 / max(a, 0.5)
                        - 1 / max(a + b, 1)
                        + 1 / max(c, 0.5)
                        - 1 / max(c + dd, 1)
                    )
                    name = "Risk Ratio"

                ci_lo = es * np.exp(-1.96 * se_ln) if es > 0 else 0
                ci_hi = es * np.exp(1.96 * se_ln) if es > 0 else 0

                summary = f"""EFFECT SIZE CALCULATOR
{"=" * 50}
Type: {name}

Input (2x2 table):
               Outcome+  Outcome-
  Exposed      {a:<9} {b}
  Not Exposed  {c:<9} {dd}

Result:
  {name} = {es:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})
  ln({name}) SE = {se_ln:.4f}

Interpretation:
  {"No association (= 1.0)" if abs(es - 1.0) < 0.01 else ("Positive association" if es > 1 else "Negative association")}"""

                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "scatter",
                                "x": [es],
                                "y": [name],
                                "mode": "markers",
                                "marker": {"size": 14, "color": "rgba(232,149,71,0.8)"},
                                "error_x": {
                                    "type": "data",
                                    "symmetric": False,
                                    "array": [ci_hi - es],
                                    "arrayminus": [es - ci_lo],
                                    "color": "rgba(232,149,71,0.5)",
                                    "thickness": 3,
                                },
                            },
                            {
                                "type": "scatter",
                                "x": [1, 1],
                                "y": [-0.5, 1.5],
                                "mode": "lines",
                                "line": {"color": "gray", "dash": "dash"},
                                "showlegend": False,
                            },
                        ],
                        "layout": {
                            "title": f"{name} with 95% CI",
                            "height": 200,
                            "xaxis": {"title": name},
                            "showlegend": False,
                        },
                    }
                )
            else:
                summary = f"Unknown effect size type: {effect_type}"

            result["summary"] = summary
            result["guide_observation"] = f"Effect size calculated: {effect_type}"

            # Narrative
            try:
                if effect_type in ("cohens_d", "hedges_g", "glass_delta"):
                    result["narrative"] = _narrative(
                        f"{name} = {d:.4f} — {magnitude} effect",
                        f"The standardized difference between groups is {d:.4f} (95% CI: {ci_lo:.4f} to {ci_hi:.4f}). "
                        f"By Cohen's benchmarks this is a <strong>{magnitude.lower()}</strong> effect. "
                        + (
                            "The CI excludes zero, confirming a meaningful difference."
                            if (ci_lo > 0 or ci_hi < 0)
                            else "The CI includes zero — the difference may not be meaningful."
                        ),
                        next_steps="Report this alongside the p-value for a complete picture of both statistical and practical significance.",
                    )
                elif effect_type in ("odds_ratio", "risk_ratio"):
                    result["narrative"] = _narrative(
                        f"{name} = {es:.4f}",
                        f"The {name.lower()} is {es:.4f} (95% CI: {ci_lo:.4f} to {ci_hi:.4f}). "
                        + (
                            "The CI excludes 1.0, indicating a significant association."
                            if (ci_lo > 1 or ci_hi < 1)
                            else "The CI includes 1.0 — the association is not statistically significant."
                        ),
                        next_steps="Consider confounding variables. An adjusted analysis (logistic regression) may be more appropriate.",
                    )
            except Exception:
                pass

        except Exception as e:
            result["summary"] = f"Effect size error: {str(e)}"

    elif analysis_id == "distribution_fit":
        """
        General-purpose distribution fitting — fits 12+ distributions,
        ranks by AIC/BIC, provides probability plots for top fits.
        """
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
                _shape_desc = f"increasing failure rate (wear-out pattern, shape = {shape_param:.2f})"
        elif best["dist_name"] == "gamma":
            _shape_desc = (
                "right-skewed, flexible shape — common in wait times and queuing"
            )
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

    elif analysis_id == "mixture_model":
        # Gaussian Mixture Model — detect hidden subpopulations
        from sklearn.mixture import GaussianMixture

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
            results_k.append(
                {"k": k, "bic": float(bic), "aic": float(aic), "model": gmm}
            )

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
            summary += "<<COLOR:success>>Data is consistent with a single population.<</COLOR>>\n"
            summary += f"  Mean: {components[0]['mean']:.4f}    Std: {components[0]['std']:.4f}\n"
        else:
            summary += f"<<COLOR:warning>>Data is best described as {k_best} overlapping populations:<</COLOR>>\n\n"
            for j, c in enumerate(components):
                summary += f"  Component {j + 1}: \u03bc={c['mean']:.4f}, \u03c3={c['std']:.4f}, weight={c['weight']:.1%}\n"

        summary += "\n<<COLOR:accent>>\u2500\u2500 Model Comparison (BIC) \u2500\u2500<</COLOR>>\n"
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
                                "#4a9f6e" if r["k"] == k_best else "#999"
                                for r in results_k
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

    elif analysis_id == "sprt":
        # Sequential Probability Ratio Test (Wald)
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
        ll_ratio = np.cumsum(
            (data - mu0) * delta / sigma**2 - delta**2 / (2 * sigma**2)
        )

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
        summary += (
            "<<COLOR:accent>>\u2500\u2500 Decision Boundaries \u2500\u2500<</COLOR>>\n"
        )
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
        result["guide_observation"] = (
            f"SPRT: {decision} after {samples_used} samples (of {n})."
        )

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

    elif analysis_id == "copula":
        # Copula-Based Dependency Modeling
        var1 = config.get("var1") or config.get("variable1")
        var2 = config.get("var2") or config.get("variable2")

        if not var1 or not var2 or var1 not in df.columns or var2 not in df.columns:
            result["summary"] = "Error: Specify two numeric columns."
            return result

        x = df[var1].dropna()
        y = df[var2].loc[x.index].dropna()
        x = x.loc[y.index].values.astype(float)
        y = y.values.astype(float)
        n = len(x)

        if n < 20:
            result["summary"] = "Error: Need at least 20 paired observations."
            return result

        # Transform to pseudo-observations (empirical CDF / ranks)
        from scipy.stats import rankdata

        u = rankdata(x) / (n + 1)
        v = rankdata(y) / (n + 1)

        # Pearson and Spearman for reference
        pearson_r, pearson_p = sp_stats.pearsonr(x, y)
        spearman_r, spearman_p = sp_stats.spearmanr(x, y)
        kendall_tau, kendall_p = sp_stats.kendalltau(x, y)

        # Fit copulas via maximum pseudo-likelihood
        # 1. Gaussian copula: parameter = correlation on normal quantiles
        z_u = sp_stats.norm.ppf(np.clip(u, 0.001, 0.999))
        z_v = sp_stats.norm.ppf(np.clip(v, 0.001, 0.999))
        rho_gauss = float(np.corrcoef(z_u, z_v)[0, 1])
        # Log-likelihood for Gaussian copula
        ll_gauss = 0.0
        for i in range(n):
            rho2 = rho_gauss**2
            if rho2 < 1:
                ll_gauss += (
                    -0.5 * np.log(1 - rho2)
                    + rho2 * (z_u[i] ** 2 + z_v[i] ** 2) / (2 * (1 - rho2))
                    - rho_gauss * z_u[i] * z_v[i] / (1 - rho2)
                    + rho_gauss * z_u[i] * z_v[i] / (1 - rho2)
                )
        # Simplified: use the bivariate normal copula density
        ll_gauss = float(
            np.sum(
                sp_stats.multivariate_normal.logpdf(
                    np.column_stack([z_u, z_v]),
                    mean=[0, 0],
                    cov=[[1, rho_gauss], [rho_gauss, 1]],
                )
                - sp_stats.norm.logpdf(z_u)
                - sp_stats.norm.logpdf(z_v)
            )
        )

        # 2. Clayton copula: theta > 0 for positive dependence
        # Kendall's tau = theta / (theta + 2)
        if kendall_tau > 0.01:
            theta_clayton = max(2 * kendall_tau / (1 - kendall_tau), 0.01)
        else:
            theta_clayton = 0.01
        # Clayton copula density: c(u,v) = (1+theta) * (u*v)^(-1-theta) * (u^-theta + v^-theta - 1)^(-1/theta - 2)
        try:
            t = theta_clayton
            term = np.clip(u ** (-t) + v ** (-t) - 1, 1e-15, None)
            ll_clayton = float(
                np.sum(
                    np.log(1 + t)
                    + (-1 - t) * (np.log(u) + np.log(v))
                    + (-1 / t - 2) * np.log(term)
                )
            )
        except Exception:
            ll_clayton = -np.inf

        # 3. Frank copula: use Kendall's tau to estimate theta
        # tau = 1 - 4/theta * (1 - D1(theta)/theta) where D1 is Debye function
        # Approximate: theta ≈ solve tau equation via bisection
        def _frank_tau(theta):
            if abs(theta) < 1e-6:
                return 0.0
            # Debye function D1(x) = (1/x) * integral_0^x t/(exp(t)-1) dt
            from scipy.integrate import quad as _quad

            d1, _ = _quad(lambda t: t / (np.exp(t) - 1 + 1e-15), 0, abs(theta))
            d1 /= abs(theta)
            return 1 - 4 / theta * (1 - d1)

        # Bisection search for Frank theta
        theta_frank = 0.0
        if abs(kendall_tau) > 0.01:
            lo_f, hi_f = -30.0, 30.0
            for _ in range(50):
                mid = (lo_f + hi_f) / 2
                if abs(mid) < 1e-6:
                    mid = 0.01
                tau_mid = _frank_tau(mid)
                if tau_mid < kendall_tau:
                    lo_f = mid
                else:
                    hi_f = mid
            theta_frank = (lo_f + hi_f) / 2

        # Frank copula density
        try:
            t = theta_frank
            if abs(t) > 0.01:
                num = -t * (np.exp(-t) - 1) * np.exp(-t * (u + v))
                den = (
                    (np.exp(-t * u) - 1) * (np.exp(-t * v) - 1) + (np.exp(-t) - 1)
                ) ** 2
                ll_frank = float(np.sum(np.log(np.clip(num / den, 1e-300, None))))
            else:
                ll_frank = 0.0
        except Exception:
            ll_frank = -np.inf

        # Compare by AIC (each copula has 1 parameter)
        copulas = [
            {
                "name": "Gaussian",
                "param": rho_gauss,
                "param_name": "\u03c1",
                "ll": ll_gauss,
                "aic": -2 * ll_gauss + 2,
            },
            {
                "name": "Clayton",
                "param": theta_clayton,
                "param_name": "\u03b8",
                "ll": ll_clayton,
                "aic": -2 * ll_clayton + 2,
            },
            {
                "name": "Frank",
                "param": theta_frank,
                "param_name": "\u03b8",
                "ll": ll_frank,
                "aic": -2 * ll_frank + 2,
            },
        ]
        copulas.sort(key=lambda c: c["aic"])
        best_cop = copulas[0]

        # Tail dependence
        # Clayton: lower tail = 2^(-1/theta), upper = 0
        # Gaussian: both = 0 (for |rho| < 1)
        # Frank: both = 0
        lower_tail = 2 ** (-1 / theta_clayton) if theta_clayton > 0.01 else 0.0
        tail_note = ""
        if best_cop["name"] == "Clayton":
            tail_note = f"Clayton copula has lower-tail dependence = {lower_tail:.3f} \u2014 variables are more correlated in the low/defect region."
        elif best_cop["name"] == "Gaussian":
            tail_note = "Gaussian copula has no tail dependence \u2014 correlation is symmetric across the distribution."
        elif best_cop["name"] == "Frank":
            tail_note = "Frank copula has no tail dependence \u2014 dependency is symmetric and concentrated in the middle."

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>COPULA DEPENDENCY MODELING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>{var1}<</COLOR>> \u00d7 <<COLOR:text>>{var2}<</COLOR>>    N: {n}\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Marginal Correlations \u2500\u2500<</COLOR>>\n"
        summary += f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.4f})\n"
        summary += f"  Spearman \u03c1:  {spearman_r:.4f} (p={spearman_p:.4f})\n"
        summary += f"  Kendall \u03c4:   {kendall_tau:.4f} (p={kendall_p:.4f})\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Copula Fit (ranked by AIC) \u2500\u2500<</COLOR>>\n"
        for j, c in enumerate(copulas):
            marker = " \u2190 best" if j == 0 else ""
            summary += f"  {c['name']:<12} {c['param_name']}={c['param']:.4f}  AIC={c['aic']:.1f}{marker}\n"
        summary += f"\n{tail_note}\n"

        result["summary"] = summary
        result["statistics"] = {
            "best_copula": best_cop["name"],
            "best_param": best_cop["param"],
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "kendall_tau": float(kendall_tau),
            "lower_tail_dep": float(lower_tail),
            "copulas": [{k: v for k, v in c.items() if k != "ll"} for c in copulas],
        }
        result["guide_observation"] = (
            f"Copula: best fit = {best_cop['name']} ({best_cop['param_name']}={best_cop['param']:.3f}). Kendall \u03c4 = {kendall_tau:.3f}."
        )

        result["narrative"] = _narrative(
            f"Copula \u2014 {best_cop['name']} fits best ({best_cop['param_name']} = {best_cop['param']:.3f})",
            f"The {best_cop['name']} copula best describes the dependency between <strong>{var1}</strong> and <strong>{var2}</strong> "
            f"(AIC = {best_cop['aic']:.1f}). Kendall \u03c4 = {kendall_tau:.4f}. {tail_note}",
            next_steps="Copulas separate dependency structure from marginal distributions. "
            "Use this for joint probability calculations (e.g., P(both variables exceed spec)) "
            "that would be wrong under a bivariate normal assumption.",
            chart_guidance="The pseudo-observation scatter shows the dependency structure in [0,1]\u00b2 space. "
            "Clustering in corners = tail dependence. Uniform spread = independence.",
        )

        # Plot 1: pseudo-observation scatter
        result["plots"].append(
            {
                "title": "Pseudo-Observations (Copula Space)",
                "data": [
                    {
                        "type": "scatter",
                        "x": u.tolist(),
                        "y": v.tolist(),
                        "mode": "markers",
                        "marker": {"size": 4, "color": "#4a90d9", "opacity": 0.5},
                        "name": "Pseudo-obs",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": f"F({var1})", "range": [0, 1]},
                    "yaxis": {"title": f"F({var2})", "range": [0, 1]},
                },
            }
        )

        # Plot 2: original scatter with marginal densities
        result["plots"].append(
            {
                "title": f"{var1} vs {var2}",
                "data": [
                    {
                        "type": "scatter",
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "mode": "markers",
                        "marker": {"size": 4, "color": "#4a9f6e", "opacity": 0.5},
                    }
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": var1},
                    "yaxis": {"title": var2},
                },
            }
        )

    return result

    return result
