"""DSW Statistical Analysis — parametric tests (t-tests, ANOVA, correlation)."""

import logging

import numpy as np
from scipy import stats as sp_stats

from ..common import (
    _bayesian_shadow,
    _check_equal_variance,
    _check_normality,
    _check_outliers,
    _cross_validate,
    _effect_magnitude,
    _evidence_grade,
    _narrative,
    _practical_block,
)

logger = logging.getLogger(__name__)


def _run_parametric(analysis_id, df, config):
    """Run parametric analysis."""
    import pandas as pd
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "ttest":
        # One-sample t-test
        var1 = config.get("var1")
        mu = float(config.get("mu", 0))
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        n = len(x)
        if n < 2:
            return {
                "summary": f"Insufficient data: t-test requires at least 2 observations (got {n}).",
                "plots": [],
                "statistics": {"n": n},
            }
        stat, pval = stats.ttest_1samp(x, mu)

        # Confidence interval
        se = x.std() / np.sqrt(n)
        ci = stats.t.interval(conf / 100, n - 1, loc=x.mean(), scale=se)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ONE-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Hypothesized mean:<</COLOR>> {mu}\n\n"
        summary += "<<COLOR:accent>>── Sample Statistics ──<</COLOR>>\n"
        summary += f"  Mean: {x.mean():.4f}\n"
        summary += f"  Std Dev: {x.std():.4f}\n"
        summary += f"  SE Mean: {se:.4f}\n\n"
        summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  {conf}% CI: ({ci[0]:.4f}, {ci[1]:.4f})\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Reject H₀: Mean differs significantly from {mu} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>Fail to reject H₀ (p >= {alpha})<</COLOR>>"

        # Effect size: Cohen's d
        d = (x.mean() - mu) / x.std() if x.std() > 0 else 0.0
        diff_val = x.mean() - mu
        label, meaningful = _effect_magnitude(d, "cohens_d")
        summary += _practical_block(
            "Cohen's d",
            d,
            "cohens_d",
            pval,
            alpha,
            context=f"The sample mean differs from {mu} by {abs(diff_val):.4f} units ({abs(d):.2f} standard deviations).",
        )

        result["summary"] = summary
        obs_parts = [
            f"One-sample t-test: mean={x.mean():.4f} vs μ₀={mu}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"
        ]
        if pval < alpha and meaningful:
            obs_parts.append("Practically significant — act on this difference.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)

        # Narrative
        direction = "higher" if x.mean() > mu else "lower"
        if pval < alpha and meaningful:
            verdict = f"The mean of {var1} is significantly {direction} than {mu}"
            body = f"The sample mean ({x.mean():.4f}) differs from the hypothesized value by {abs(diff_val):.4f} units &mdash; a <strong>{label}</strong> effect (Cohen's d = {abs(d):.2f}). This difference is both statistically and practically significant."
            nexts = "Investigate what is causing the shift from the target value."
        elif pval < alpha:
            verdict = f"Statistically significant but small difference in {var1}"
            body = f"The sample mean ({x.mean():.4f}) differs from {mu} (p = {pval:.4f}), but the effect size is {label} (d = {abs(d):.2f}). The difference may be too small to justify action."
            nexts = f"Evaluate whether the cost of intervention is worth a {label} improvement."
        else:
            verdict = f"No significant difference from {mu}"
            body = f"The sample mean ({x.mean():.4f}) is not significantly different from {mu} (p = {pval:.4f}). The effect size is {label} (d = {abs(d):.2f})."
            nexts = (
                "If you expected a difference, consider increasing the sample size."
                if label in ("medium", "large")
                else None
            )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nexts,
            chart_guidance="The histogram shows the data distribution. The dashed line marks the hypothesized mean; the shaded band is the 95% confidence interval.",
        )

        # Explicit statistics for Synara
        result["statistics"] = {
            f"mean({var1})": float(x.mean()),
            f"std({var1})": float(x.std()),
            f"n({var1})": int(n),
            "t_statistic": float(stat),
            "p_value": float(pval),
            "cohens_d": float(d),
            "effect_size_label": label,
            f"ci_lower({var1})": float(ci[0]),
            f"ci_upper({var1})": float(ci[1]),
        }

        # Interactive Power Explorer metadata
        result["power_explorer"] = {
            "test_type": "ttest",
            "observed_effect": float(abs(x.mean() - mu)),
            "observed_std": float(x.std()) if x.std() > 0 else 1.0,
            "observed_n": int(n),
            "alpha": float(alpha),
            "cohens_d": float(abs(d)),
        }

        # ── Diagnostics: assumption checks + cross-validation ──
        diagnostics = []
        _norm = _check_normality(x.values, label=var1, alpha=alpha)
        if _norm:
            _norm["detail"] += f" With n={n}, consider a Wilcoxon signed-rank test."
            _norm["action"] = {
                "label": "Run Wilcoxon Signed-Rank",
                "type": "stats",
                "analysis": "wilcoxon_1samp",
                "config": {"var": var1, "mu": mu},
            }
            diagnostics.append(_norm)
        _out = _check_outliers(x.values, label=var1)
        if _out:
            diagnostics.append(_out)
        # Cross-validate with Wilcoxon signed-rank
        _cv_agrees = None
        try:
            _wsr_stat, _wsr_p = stats.wilcoxon(x - mu)
            _cv_result = _cross_validate(
                pval,
                _wsr_p,
                "t-test",
                "Wilcoxon signed-rank",
                alpha=alpha,
                normality_failed=bool(_norm),
            )
            _cv_agrees = _cv_result.get("level") == "info"
            diagnostics.append(_cv_result)
        except Exception:
            pass
        # Effect size emphasis
        if abs(d) >= 0.8 and pval < alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (Cohen's d = {abs(d):.2f})",
                    "detail": "This difference is large enough to be practically meaningful regardless of p-value.",
                }
            )
        elif abs(d) < 0.2 and pval < alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (d = {abs(d):.2f})",
                    "detail": "Statistical significance with negligible practical effect. The difference may not justify action.",
                }
            )
        result["diagnostics"] = diagnostics

        # --- Bayesian Insurance ---
        try:
            _shadow = _bayesian_shadow("ttest_1samp", x=x.values, mu=mu)
            if _shadow:
                result["bayesian_shadow"] = _shadow
            _grade = _evidence_grade(
                pval,
                bf10=_shadow.get("bf10") if _shadow else None,
                effect_magnitude=label,
                cross_val_agrees=_cv_agrees,
            )
            if _grade:
                result["evidence_grade"] = _grade
        except Exception:
            pass

        # Histogram with mean line and CI band
        result["plots"].append(
            {
                "title": f"Distribution of {var1} with Mean & {conf}% CI",
                "data": [
                    {
                        "type": "histogram",
                        "x": x.tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": var1,
                    },
                    {
                        "type": "scatter",
                        "x": [float(x.mean()), float(x.mean())],
                        "y": [0, n / 5],
                        "mode": "lines",
                        "line": {"color": "#4a90d9", "width": 2},
                        "name": f"Mean ({x.mean():.3f})",
                    },
                    {
                        "type": "scatter",
                        "x": [mu, mu],
                        "y": [0, n / 5],
                        "mode": "lines",
                        "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
                        "name": f"H₀ μ = {mu}",
                    },
                    {
                        "type": "scatter",
                        "x": [
                            float(ci[0]),
                            float(ci[1]),
                            float(ci[1]),
                            float(ci[0]),
                            float(ci[0]),
                        ],
                        "y": [0, 0, n / 5, n / 5, 0],
                        "fill": "toself",
                        "fillcolor": "rgba(74, 144, 217, 0.15)",
                        "line": {"color": "rgba(74, 144, 217, 0.3)"},
                        "name": f"{conf}% CI",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": var1},
                    "yaxis": {"title": "Count"},
                    "barmode": "overlay",
                },
            }
        )

    elif analysis_id == "ttest2":
        # Two-sample t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        # Support stacked/factor format: response column + grouping column
        if config.get("data_format") == "factor":
            response_col = config.get("response") or var1
            factor_col = config.get("group_var") or config.get("factor") or var2
            levels = df[factor_col].dropna().unique()
            if len(levels) != 2:
                result["summary"] = f"Two-sample t-test requires exactly 2 groups. Found {len(levels)}: {list(levels)}"
                return result
            x = df[df[factor_col] == levels[0]][response_col].dropna()
            y = df[df[factor_col] == levels[1]][response_col].dropna()
            var1, var2 = (
                f"{response_col} [{levels[0]}]",
                f"{response_col} [{levels[1]}]",
            )
        else:
            x = df[var1].dropna()
            y = df[var2].dropna()
        stat, pval = stats.ttest_ind(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>TWO-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += (
            f"<<COLOR:text>>Sample 1:<</COLOR>> {var1} (n = {len(x)}, mean = {x.mean():.4f}, std = {x.std():.4f})\n"
        )
        summary += (
            f"<<COLOR:text>>Sample 2:<</COLOR>> {var2} (n = {len(y)}, mean = {y.mean():.4f}, std = {y.std():.4f})\n\n"
        )
        summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
        summary += f"  Difference of means: {x.mean() - y.mean():.4f}\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        _se_diff = np.sqrt(x.std() ** 2 / len(x) + y.std() ** 2 / len(y))
        # Welch-Satterthwaite degrees of freedom for unequal variances
        _s1_n = x.std() ** 2 / len(x)
        _s2_n = y.std() ** 2 / len(y)
        _welch_df = (
            (_s1_n + _s2_n) ** 2 / (_s1_n**2 / (len(x) - 1) + _s2_n**2 / (len(y) - 1))
            if (_s1_n + _s2_n) > 0
            else len(x) + len(y) - 2
        )
        _t_crit = stats.t.ppf(1 - alpha / 2, _welch_df)
        _ci_lo = (x.mean() - y.mean()) - _t_crit * _se_diff
        _ci_hi = (x.mean() - y.mean()) + _t_crit * _se_diff
        summary += f"  {conf}% CI for difference: [{_ci_lo:.4f}, {_ci_hi:.4f}]\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Means are significantly different (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        # Effect size: Cohen's d (pooled)
        nx, ny = len(x), len(y)
        pooled_std = (
            np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / (nx + ny - 2)) if (nx + ny > 2) else 1.0
        )
        d = (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0.0
        diff_val = x.mean() - y.mean()
        label, meaningful = _effect_magnitude(d, "cohens_d")
        summary += _practical_block(
            "Cohen's d",
            d,
            "cohens_d",
            pval,
            alpha,
            context=f"{var1} is {abs(diff_val):.4f} units {'higher' if diff_val > 0 else 'lower'} than {var2} ({abs(d):.2f} pooled SDs).",
        )

        result["summary"] = summary
        obs_parts = [f"Two-sample t-test: diff={diff_val:.4f}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"]
        if pval < alpha and meaningful:
            obs_parts.append("Practically significant difference.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)

        # Narrative
        higher = var1 if diff_val > 0 else var2
        lower = var2 if diff_val > 0 else var1
        if pval < alpha and meaningful:
            verdict = f"{higher} is significantly higher than {lower}"
            body = f"The mean difference is {abs(diff_val):.4f} units &mdash; a <strong>{label}</strong> effect (Cohen's d = {abs(d):.2f}). This is both statistically and practically significant."
            nexts = "Investigate root causes for the difference between groups."
        elif pval < alpha:
            verdict = f"Statistically significant but {label} difference"
            body = f"The groups differ (p = {pval:.4f}), but the effect size is {label} (d = {abs(d):.2f}). The {abs(diff_val):.4f}-unit gap may be too small to act on."
            nexts = "Evaluate practical importance before allocating resources."
        else:
            verdict = "No significant difference between groups"
            body = f"The means of {var1} ({x.mean():.4f}) and {var2} ({y.mean():.4f}) are not significantly different (p = {pval:.4f})."
            nexts = (
                "If you expected a difference, consider increasing sample size or reducing measurement noise."
                if label in ("medium", "large")
                else None
            )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nexts,
            chart_guidance="Side-by-side box plots show the distribution of each group. Overlapping boxes suggest similar distributions; separated boxes suggest a real difference.",
        )

        result["statistics"] = {
            f"mean({var1})": float(x.mean()),
            f"mean({var2})": float(y.mean()),
            "difference": float(diff_val),
            "t_statistic": float(stat),
            "p_value": float(pval),
            "cohens_d": float(d),
            "effect_size_label": label,
        }

        # Interactive Power Explorer metadata
        result["power_explorer"] = {
            "test_type": "ttest2",
            "observed_effect": float(abs(diff_val)),
            "observed_std": float(pooled_std) if pooled_std > 0 else 1.0,
            "observed_n": int(nx + ny),
            "alpha": float(alpha),
            "cohens_d": float(abs(d)),
        }

        # ── Diagnostics: assumption checks + cross-validation ──
        diagnostics = []
        _norm1 = _check_normality(x.values, label=var1, alpha=alpha)
        _norm2 = _check_normality(y.values, label=var2, alpha=alpha)
        _any_nonnormal = bool(_norm1 or _norm2)
        if _norm1:
            diagnostics.append(_norm1)
        if _norm2:
            diagnostics.append(_norm2)
        _eq_var = _check_equal_variance(x.values, y.values, labels=[var1, var2], alpha=alpha)
        if _eq_var:
            _eq_var["detail"] += " Welch's t-test (default) handles this correctly."
            diagnostics.append(_eq_var)
        _out1 = _check_outliers(x.values, label=var1)
        _out2 = _check_outliers(y.values, label=var2)
        if _out1:
            diagnostics.append(_out1)
        if _out2:
            diagnostics.append(_out2)
        # Cross-validate with Mann-Whitney
        _cv_agrees = None
        try:
            _mw_u, _mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
            _cv = _cross_validate(
                pval,
                _mw_p,
                "t-test",
                "Mann-Whitney U",
                alpha=alpha,
                normality_failed=_any_nonnormal,
            )
            _cv_agrees = _cv.get("level") == "info"
            # Enrich with effect size
            if abs(d) >= 0.5:
                _cv["detail"] += f" Effect size is {label} (d = {abs(d):.2f})."
            _cv["action"] = {
                "label": "Run Mann-Whitney",
                "type": "stats",
                "analysis": "mann_whitney",
                "config": {
                    "var": config.get("response") or config.get("var1", ""),
                    "group_var": config.get("group_var") or config.get("factor") or config.get("var2", ""),
                },
            }
            diagnostics.append(_cv)
        except Exception:
            pass
        # Effect size emphasis
        if abs(d) >= 0.8 and pval < alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (Cohen's d = {abs(d):.2f})",
                    "detail": f"The {abs(diff_val):.4f}-unit difference is {label} \u2014 large enough to be practically meaningful.",
                }
            )
        elif abs(d) < 0.2 and pval < alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (d = {abs(d):.2f})",
                    "detail": "Statistical significance with negligible practical effect. Large sample sizes can make tiny differences significant.",
                }
            )
        elif abs(d) >= 0.5 and pval >= alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate effect not reaching significance (d = {abs(d):.2f})",
                    "detail": "The effect size suggests a real difference, but the sample may be too small to detect it. Consider collecting more data.",
                    "action": {
                        "label": "Power Analysis",
                        "type": "stats",
                        "analysis": "power_sample_size",
                        "config": {
                            "test_type": "ttest2",
                            "effect_size": float(abs(d)),
                            "alpha": float(alpha),
                        },
                    },
                }
            )
        result["diagnostics"] = diagnostics

        # --- Bayesian Insurance ---
        try:
            _shadow = _bayesian_shadow("ttest_2samp", x=x.values, y=y.values)
            if _shadow:
                result["bayesian_shadow"] = _shadow
            _grade = _evidence_grade(
                pval,
                bf10=_shadow.get("bf10") if _shadow else None,
                effect_magnitude=label,
                cross_val_agrees=_cv_agrees,
            )
            if _grade:
                result["evidence_grade"] = _grade
        except Exception:
            pass

        # Side-by-side box plots
        result["plots"].append(
            {
                "title": f"Comparison: {var1} vs {var2}",
                "data": [
                    {
                        "type": "box",
                        "y": x.tolist(),
                        "name": var1,
                        "marker": {"color": "#4a9f6e"},
                        "boxpoints": "outliers",
                    },
                    {
                        "type": "box",
                        "y": y.tolist(),
                        "name": var2,
                        "marker": {"color": "#4a90d9"},
                        "boxpoints": "outliers",
                    },
                ],
                "layout": {"height": 300, "yaxis": {"title": "Value"}},
            }
        )

    elif analysis_id == "paired_t":
        # Paired t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        # Support stacked/factor format: response column + grouping column
        if config.get("data_format") == "factor":
            response_col = config.get("response") or var1
            factor_col = config.get("group_var") or config.get("factor") or var2
            levels = df[factor_col].dropna().unique()
            if len(levels) != 2:
                result["summary"] = f"Paired t-test requires exactly 2 groups. Found {len(levels)}: {list(levels)}"
                return result
            x = df[df[factor_col] == levels[0]][response_col].dropna().reset_index(drop=True)
            y = df[df[factor_col] == levels[1]][response_col].dropna().reset_index(drop=True)
            var1, var2 = (
                f"{response_col} [{levels[0]}]",
                f"{response_col} [{levels[1]}]",
            )
            # Align by row order (pairs must match by position)
            min_len = min(len(x), len(y))
            x = x.iloc[:min_len]
            y = y.iloc[:min_len]
        else:
            x = df[var1].dropna()
            y = df[var2].dropna()
            # Align samples
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            y = y.loc[common_idx]

        diff = x - y
        stat, pval = stats.ttest_rel(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>PAIRED T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1 (Before):<</COLOR>> {var1}\n"
        summary += f"<<COLOR:text>>Sample 2 (After):<</COLOR>> {var2}\n"
        summary += f"<<COLOR:accent>>── Pairs ──<</COLOR>> {len(x)}\n\n"
        summary += "<<COLOR:accent>>── Difference Statistics ──<</COLOR>>\n"
        summary += f"  Mean difference: {diff.mean():.4f}\n"
        summary += f"  Std of differences: {diff.std():.4f}\n\n"
        summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        _se_diff = diff.std() / np.sqrt(len(diff))
        _t_crit = stats.t.ppf(1 - alpha / 2, len(diff) - 1)
        _ci_lo = diff.mean() - _t_crit * _se_diff
        _ci_hi = diff.mean() + _t_crit * _se_diff
        summary += f"  {conf}% CI for mean difference: [{_ci_lo:.4f}, {_ci_hi:.4f}]\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Significant difference between paired observations (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        # Effect size: Cohen's d for paired data
        d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
        label, meaningful = _effect_magnitude(d, "cohens_d")
        direction = "improved" if diff.mean() > 0 else "decreased"
        summary += _practical_block(
            "Cohen's d",
            d,
            "cohens_d",
            pval,
            alpha,
            context=f"Values {direction} by {abs(diff.mean()):.4f} units on average ({abs(d):.2f} SDs of within-subject change).",
        )

        result["summary"] = summary
        obs_parts = [f"Paired t-test: mean diff={diff.mean():.4f}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"]
        if pval < alpha and meaningful:
            obs_parts.append(f"Practically significant — values {direction} meaningfully.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)

        # Narrative
        if pval < alpha and meaningful:
            verdict = f"Paired values {direction} significantly"
            body = f"The mean difference is {abs(diff.mean()):.4f} units &mdash; a <strong>{label}</strong> effect (d = {abs(d):.2f}). The change is both statistically and practically significant."
            nexts = "This confirms the intervention had a meaningful impact."
        elif pval < alpha:
            verdict = "Small but statistically significant change"
            body = f"Paired values {direction} by {abs(diff.mean()):.4f} units on average (p = {pval:.4f}), but the effect is {label} (d = {abs(d):.2f})."
            nexts = "The change is real but may not justify the cost of the intervention."
        else:
            verdict = "No significant change between paired observations"
            body = f"The mean difference ({diff.mean():.4f}) is not statistically significant (p = {pval:.4f}). The effect size is {label}."
            nexts = (
                "If you expected improvement, check whether the intervention was applied consistently."
                if label in ("medium", "large")
                else None
            )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nexts,
            chart_guidance="The histogram shows the distribution of paired differences. If centered away from zero, the treatment had a systematic effect.",
        )

        result["statistics"] = {
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std()),
            "n_pairs": int(len(x)),
            "t_statistic": float(stat),
            "p_value": float(pval),
            "cohens_d": float(d),
            "effect_size_label": label,
        }

        # Interactive Power Explorer metadata
        result["power_explorer"] = {
            "test_type": "ttest",
            "observed_effect": float(abs(diff.mean())),
            "observed_std": float(diff.std()) if diff.std() > 0 else 1.0,
            "observed_n": int(len(x)),
            "alpha": float(alpha),
            "cohens_d": float(abs(d)),
        }

        # ── Diagnostics ──
        diagnostics = []
        _norm_d = _check_normality(diff.values, label="Differences", alpha=alpha)
        if _norm_d:
            _norm_d["detail"] += " For paired tests, normality of differences matters."
            _norm_d["action"] = {
                "label": "Run Wilcoxon Signed-Rank",
                "type": "stats",
                "analysis": "wilcoxon_signed",
                "config": {
                    "var1": config.get("var1", ""),
                    "var2": config.get("var2", ""),
                },
            }
            diagnostics.append(_norm_d)
        _out_d = _check_outliers(diff.values, label="Differences")
        if _out_d:
            diagnostics.append(_out_d)
        # Cross-validate with Wilcoxon signed-rank
        _cv_agrees = None
        try:
            _wsr_stat, _wsr_p = stats.wilcoxon(diff)
            _cv = _cross_validate(
                pval,
                _wsr_p,
                "Paired t-test",
                "Wilcoxon signed-rank",
                alpha=alpha,
                normality_failed=bool(_norm_d),
            )
            _cv_agrees = _cv.get("level") == "info"
            if abs(d) >= 0.5:
                _cv["detail"] += f" Effect size is {label} (d = {abs(d):.2f})."
            diagnostics.append(_cv)
        except Exception:
            pass
        if abs(d) >= 0.8 and pval < alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (d = {abs(d):.2f})",
                    "detail": f"The intervention produced a {label} change of {abs(diff.mean()):.4f} units.",
                }
            )
        elif abs(d) < 0.2 and pval < alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (d = {abs(d):.2f})",
                    "detail": "The change is statistically detectable but may not justify the intervention cost.",
                }
            )
        result["diagnostics"] = diagnostics

        # --- Bayesian Insurance ---
        try:
            _shadow = _bayesian_shadow("ttest_paired", x=x.values, y=y.values)
            if _shadow:
                result["bayesian_shadow"] = _shadow
            _grade = _evidence_grade(
                pval,
                bf10=_shadow.get("bf10") if _shadow else None,
                effect_magnitude=label,
                cross_val_agrees=_cv_agrees,
            )
            if _grade:
                result["evidence_grade"] = _grade
        except Exception:
            pass

        # Histogram of differences
        result["plots"].append(
            {
                "title": f"Distribution of Differences ({var1} − {var2})",
                "data": [
                    {
                        "type": "histogram",
                        "x": diff.tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Differences",
                    },
                    {
                        "type": "scatter",
                        "x": [float(diff.mean()), float(diff.mean())],
                        "y": [0, len(x) / 5],
                        "mode": "lines",
                        "line": {"color": "#4a90d9", "width": 2},
                        "name": f"Mean diff ({diff.mean():.3f})",
                    },
                    {
                        "type": "scatter",
                        "x": [0, 0],
                        "y": [0, len(x) / 5],
                        "mode": "lines",
                        "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
                        "name": "Zero (no diff)",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Difference"},
                    "yaxis": {"title": "Count"},
                },
            }
        )

    elif analysis_id == "anova":
        response = config.get("response")
        # Support both 'factor' (single) and 'factors' (list) for backwards compatibility
        factor = config.get("factor")
        factors = config.get("factors", [])
        if factor and not factors:
            factors = [factor]

        if len(factors) >= 1:
            # One-way ANOVA
            factor_col = factors[0]
            groups = [df[df[factor_col] == level][response].dropna() for level in df[factor_col].unique()]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups

            stat, pval = stats.f_oneway(*groups)

            # Calculate group statistics
            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += "<<COLOR:title>>ONE-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"

            summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
            for level in df[factor_col].unique():
                grp = df[df[factor_col] == level][response].dropna()
                _ci = (
                    stats.t.interval(
                        0.95,
                        len(grp) - 1,
                        loc=grp.mean(),
                        scale=grp.std() / np.sqrt(len(grp)),
                    )
                    if len(grp) > 1
                    else (grp.mean(), grp.mean())
                )
                summary += f"  {level}: n={len(grp)}, mean={grp.mean():.4f}, std={grp.std():.4f}, 95% CI [{_ci[0]:.4f}, {_ci[1]:.4f}]\n"

            # Compute eta-squared
            grand_mean = df[response].dropna().mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = sum((df[response].dropna() - grand_mean) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            n_total = sum(len(g) for g in groups)
            k = len(groups)
            # Omega-squared (less biased)
            omega_sq = (
                (ss_between - (k - 1) * (ss_total - ss_between) / (n_total - k))
                / (ss_total + (ss_total - ss_between) / (n_total - k))
                if (n_total > k and ss_total > 0)
                else 0.0
            )
            omega_sq = max(0, omega_sq)
            eta_label, eta_meaningful = _effect_magnitude(eta_sq, "eta_squared")

            summary += "\n<<COLOR:accent>>── ANOVA Results ──<</COLOR>>\n"
            summary += f"  F-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n\n"

            if pval < 0.05:
                summary += "<<COLOR:good>>Significant difference between groups (p < 0.05)<</COLOR>>\n"
                summary += "<<COLOR:text>>Run post-hoc tests (Tukey HSD, Games-Howell, or Dunnett) to identify which groups differ.<</COLOR>>"
            else:
                summary += "<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>"

            summary += _practical_block(
                "Eta-squared (η²)",
                eta_sq,
                "eta_squared",
                pval,
                context=f"The factor '{factor_col}' explains {eta_sq * 100:.1f}% of the variation in '{response}'. Omega-squared (less biased): {omega_sq:.3f}.",
            )

            result["summary"] = summary
            obs_parts = [f"One-way ANOVA: F={stat:.4f}, p={pval:.4f}, η²={eta_sq:.3f} ({eta_label})"]
            if pval < 0.05 and eta_meaningful:
                obs_parts.append(f"'{factor_col}' has a {eta_label}, practically significant effect on '{response}'.")
            elif pval < 0.05:
                obs_parts.append(f"Significant but {eta_label} effect.")
            else:
                obs_parts.append("Not significant.")
            result["guide_observation"] = " ".join(obs_parts)

            # Narrative
            if pval < 0.05 and eta_meaningful:
                verdict = f"{factor_col} has a significant effect on {response}"
                body = f"The factor explains <strong>{eta_sq * 100:.1f}%</strong> of the variation in {response} &mdash; a <strong>{eta_label}</strong> effect ({k} groups, F = {stat:.2f}, p = {pval:.4f})."
                nexts = "Run <strong>Tukey HSD</strong> or <strong>Games-Howell</strong> post-hoc tests to identify which specific groups differ."
            elif pval < 0.05:
                verdict = f"Statistically significant but {eta_label} effect of {factor_col}"
                body = f"At least one group mean differs (p = {pval:.4f}), but {factor_col} explains only {eta_sq * 100:.1f}% of the variation ({eta_label} effect). Other factors likely dominate."
                nexts = "Consider whether the small effect justifies further investigation. Look for other sources of variation."
            else:
                verdict = f"No significant difference across {factor_col} groups"
                body = f"The {k} groups do not differ significantly (F = {stat:.2f}, p = {pval:.4f}). {factor_col} explains only {eta_sq * 100:.1f}% of the variation."
                nexts = (
                    "If you expected differences, check sample sizes and measurement precision."
                    if eta_meaningful
                    else None
                )
            result["narrative"] = _narrative(
                verdict,
                body,
                next_steps=nexts,
                chart_guidance=f"The box plot shows the distribution of {response} by {factor_col}. Non-overlapping boxes suggest real differences; look for groups with notably different medians.",
            )

            result["statistics"] = {
                "f_statistic": float(stat),
                "p_value": float(pval),
                "eta_squared": float(eta_sq),
                "omega_squared": float(omega_sq),
                "effect_size_label": eta_label,
                "n_groups": k,
                "n_total": n_total,
            }

            # Interactive Power Explorer metadata
            # Convert eta-squared to Cohen's f: f = sqrt(eta2 / (1 - eta2))
            cohens_f = np.sqrt(eta_sq / (1 - eta_sq)) if eta_sq < 1 else 1.0
            result["power_explorer"] = {
                "test_type": "anova",
                "observed_effect": float(cohens_f),
                "observed_std": 1.0,
                "observed_n": int(n_total),
                "alpha": 0.05,
                "cohens_d": float(cohens_f),
                "n_groups": int(k),
            }

            # ── Diagnostics ──
            diagnostics = []
            # Normality of residuals (approximate: check each group)
            _any_nonnormal = False
            for level in df[factor_col].unique():
                grp = df[df[factor_col] == level][response].dropna()
                _norm_g = _check_normality(grp.values, label=f"{response} [{level}]", alpha=0.05)
                if _norm_g:
                    _any_nonnormal = True
                    diagnostics.append(_norm_g)
                    break  # Report once, not per-group
            if _any_nonnormal:
                diagnostics[-1]["detail"] = (
                    "ANOVA assumes normality within groups. Consider Kruskal-Wallis for non-normal data."
                )
                diagnostics[-1]["action"] = {
                    "label": "Run Kruskal-Wallis",
                    "type": "stats",
                    "analysis": "kruskal_wallis",
                    "config": {"response": response, "factor": factor_col},
                }
            # Equal variance
            _eq_var = _check_equal_variance(
                *[g.values for g in groups],
                labels=[str(lbl) for lbl in df[factor_col].unique()],
                alpha=0.05,
            )
            if _eq_var:
                _eq_var["detail"] += " Consider Welch's ANOVA or Games-Howell post-hoc."
                diagnostics.append(_eq_var)
            # Cross-validate with Kruskal-Wallis
            _cv_agrees = None
            try:
                _kw_stat, _kw_p = stats.kruskal(*groups)
                _cv = _cross_validate(
                    pval,
                    _kw_p,
                    "ANOVA",
                    "Kruskal-Wallis",
                    alpha=0.05,
                    normality_failed=_any_nonnormal,
                )
                _cv_agrees = _cv.get("level") == "info"
                if eta_sq >= 0.06:
                    _cv["detail"] += f" Effect size is {eta_label} (\u03b7\u00b2 = {eta_sq:.3f})."
                diagnostics.append(_cv)
            except Exception:
                pass
            # Effect size emphasis
            if eta_sq >= 0.14 and pval < 0.05:
                diagnostics.append(
                    {
                        "level": "info",
                        "title": f"Large practical effect (\u03b7\u00b2 = {eta_sq:.3f})",
                        "detail": f"{factor_col} explains {eta_sq * 100:.1f}% of variation \u2014 a meaningful source of differences.",
                    }
                )
            elif eta_sq < 0.01 and pval < 0.05:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": f"Significant but negligible effect (\u03b7\u00b2 = {eta_sq:.3f})",
                        "detail": f"{factor_col} explains only {eta_sq * 100:.1f}% of variation. The difference is real but practically irrelevant.",
                    }
                )
            elif eta_sq >= 0.06 and pval >= 0.05:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": f"Moderate effect not reaching significance (\u03b7\u00b2 = {eta_sq:.3f})",
                        "detail": "The effect size suggests real group differences but the sample may be too small.",
                        "action": {
                            "label": "Power Analysis",
                            "type": "stats",
                            "analysis": "power_sample_size",
                            "config": {
                                "test_type": "anova",
                                "effect_size": float(eta_sq),
                                "alpha": 0.05,
                                "n_groups": k,
                            },
                        },
                    }
                )
            # Post-hoc suggestion
            if pval < 0.05 and k > 2:
                diagnostics.append(
                    {
                        "level": "info",
                        "title": f"Post-hoc needed: {k} groups \u2014 ANOVA only says 'at least one differs'",
                        "detail": "Run pairwise comparisons to identify which specific groups differ.",
                        "action": {
                            "label": "Run Tukey HSD",
                            "type": "stats",
                            "analysis": "tukey_hsd",
                            "config": {"response": response, "factor": factor_col},
                        },
                    }
                )
            result["diagnostics"] = diagnostics

            # --- Bayesian Insurance ---
            try:
                _shadow = _bayesian_shadow("anova", groups=[g.values for g in groups])
                if _shadow:
                    result["bayesian_shadow"] = _shadow
                _grade = _evidence_grade(
                    pval,
                    bf10=_shadow.get("bf10") if _shadow else None,
                    effect_magnitude=eta_label,
                    cross_val_agrees=_cv_agrees,
                )
                if _grade:
                    result["evidence_grade"] = _grade
            except Exception:
                pass

            # Box plot
            result["plots"].append(
                {
                    "title": f"{response} by {factor_col}",
                    "data": [
                        {
                            "type": "box",
                            "y": df[response].tolist(),
                            "x": df[factor_col].astype(str).tolist(),
                            "marker": {
                                "color": "rgba(74, 159, 110, 0.4)",
                                "line": {"color": "#4a9f6e", "width": 1.5},
                            },
                        }
                    ],
                    "layout": {"height": 300},
                }
            )
        else:
            result["summary"] = "Please select a factor column for ANOVA."

    elif analysis_id == "anova2":
        # Two-way ANOVA
        response = config.get("response")
        factor_a = config.get("factor_a")
        factor_b = config.get("factor_b")

        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            # Build formula
            formula = f"{response} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += "<<COLOR:title>>TWO-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor A:<</COLOR>> {factor_a}\n"
            summary += f"<<COLOR:highlight>>Factor B:<</COLOR>> {factor_b}\n\n"

            summary += "<<COLOR:accent>>── ANOVA Table ──<</COLOR>>\n"
            summary += anova_table.to_string() + "\n\n"

            # Compute partial eta-squared and interpret each factor
            ss_resid = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 0
            effect_stats = {}
            for idx in anova_table.index:
                if idx == "Residual":
                    continue
                if "PR(>F)" in anova_table.columns:
                    p = anova_table.loc[idx, "PR(>F)"]
                    ss = anova_table.loc[idx, "sum_sq"] if "sum_sq" in anova_table.columns else 0
                    partial_eta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                    eta_label, eta_meaningful = _effect_magnitude(partial_eta, "eta_squared")
                    if not np.isnan(p):
                        sig = "<<COLOR:good>>*<</COLOR>>" if p < 0.05 else ""
                        summary += f"{idx}: p = {p:.4f} {sig}  |  partial η² = {partial_eta:.3f} ({eta_label})\n"
                        effect_stats[idx] = {
                            "p_value": float(p),
                            "partial_eta_squared": float(partial_eta),
                            "label": eta_label,
                        }

            # Practical significance block for strongest effect
            if effect_stats:
                strongest = max(effect_stats.items(), key=lambda x: x[1]["partial_eta_squared"])
                s_name, s_vals = strongest
                summary += _practical_block(
                    f"Partial η² ({s_name})",
                    s_vals["partial_eta_squared"],
                    "eta_squared",
                    s_vals["p_value"],
                    context=f"'{s_name}' explains {s_vals['partial_eta_squared'] * 100:.1f}% of the remaining variation in '{response}'.",
                )

            result["summary"] = summary
            result["guide_observation"] = "Two-way ANOVA: " + "; ".join(
                f"{k}: p={v['p_value']:.4f}, η²={v['partial_eta_squared']:.3f} ({v['label']})"
                for k, v in effect_stats.items()
            )
            result["statistics"] = {"effects": effect_stats}

            # Narrative
            if effect_stats:
                _sig_effects = {k: v for k, v in effect_stats.items() if v["p_value"] < 0.05}
                strongest = max(effect_stats.items(), key=lambda x: x[1]["partial_eta_squared"])
                s_name, s_vals = strongest
                if _sig_effects:
                    _ix_key = f"C({factor_a}):C({factor_b})"
                    _has_ix = _ix_key in _sig_effects
                    verdict = f"{'Interaction' if _has_ix else s_name} is significant (η² = {s_vals['partial_eta_squared']:.3f})"
                    body = f"Significant effects: <strong>{', '.join(_sig_effects.keys())}</strong>."
                    if _has_ix:
                        body += (
                            f" The interaction means the effect of {factor_a} depends on {factor_b} — optimize jointly."
                        )
                    nxt = "Run post-hoc tests (Tukey HSD) on significant main effects to identify which levels differ."
                else:
                    verdict = "No significant effects detected"
                    body = f"Neither {factor_a}, {factor_b}, nor their interaction significantly affects {response}."
                    nxt = "Check sample sizes and effect sizes. The study may lack power."
                result["narrative"] = _narrative(
                    verdict,
                    body,
                    next_steps=nxt,
                    chart_guidance="Non-parallel lines in the interaction plot suggest an interaction between factors.",
                )

            # ── Diagnostics ──
            diagnostics = []
            # Normality of residuals
            try:
                _resids = model.resid.values
                _norm_r = _check_normality(_resids, label="Model residuals", alpha=0.05)
                if _norm_r:
                    _norm_r["detail"] = (
                        "Two-way ANOVA assumes normality of residuals. Consider a non-parametric alternative or data transformation."
                    )
                    diagnostics.append(_norm_r)
            except Exception:
                # Fallback: check response within each factor combination
                for _a_lev in df[factor_a].unique():
                    for _b_lev in df[factor_b].unique():
                        _cell = df[(df[factor_a] == _a_lev) & (df[factor_b] == _b_lev)][response].dropna()
                        if len(_cell) >= 8:
                            _norm_c = _check_normality(
                                _cell.values,
                                label=f"{response} [{_a_lev}×{_b_lev}]",
                                alpha=0.05,
                            )
                            if _norm_c:
                                _norm_c["detail"] = (
                                    "Non-normal data in at least one cell. ANOVA is robust for large samples but consider transformations."
                                )
                                diagnostics.append(_norm_c)
                                break
                    else:
                        continue
                    break
            # Equal variances across factor combinations (Levene's)
            _cell_groups = []
            _cell_labels = []
            for _a_lev in df[factor_a].unique():
                for _b_lev in df[factor_b].unique():
                    _cell = df[(df[factor_a] == _a_lev) & (df[factor_b] == _b_lev)][response].dropna()
                    if len(_cell) >= 2:
                        _cell_groups.append(_cell.values)
                        _cell_labels.append(f"{_a_lev}×{_b_lev}")
            if len(_cell_groups) >= 2:
                _eq_var = _check_equal_variance(*_cell_groups, labels=_cell_labels, alpha=0.05)
                if _eq_var:
                    _eq_var["detail"] += (
                        " Unequal variances across cells may inflate Type I error. Consider data transformation."
                    )
                    diagnostics.append(_eq_var)
            # Effect size emphasis: partial eta-squared per effect
            _ix_key = f"C({factor_a}):C({factor_b})"
            for _eff_name, _eff_vals in effect_stats.items():
                _peta = _eff_vals["partial_eta_squared"]
                _pval = _eff_vals["p_value"]
                if _peta > 0.14 and _pval < 0.05:
                    diagnostics.append(
                        {
                            "level": "info",
                            "title": f"Large practical effect for {_eff_name} (η²p = {_peta:.3f})",
                            "detail": f"{_eff_name} explains {_peta * 100:.1f}% of variation after accounting for other effects — a meaningful source of differences.",
                        }
                    )
            # Interaction warning
            if _ix_key in effect_stats and effect_stats[_ix_key]["p_value"] < 0.05:
                _ix_p = effect_stats[_ix_key]["p_value"]
                diagnostics.append(
                    {
                        "level": "info",
                        "title": "Interaction detected \u2014 interpret main effects with caution",
                        "detail": f"The {factor_a}\u00d7{factor_b} interaction is significant (p = {_ix_p:.4f}). Main effects alone do not tell the full story \u2014 the effect of one factor depends on the level of the other.",
                    }
                )
            # No effects significant
            if not any(v["p_value"] < 0.05 for v in effect_stats.values()):
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": "No detectable effects — consider increasing sample size or redesigning",
                        "detail": f"Neither {factor_a}, {factor_b}, nor their interaction reached significance. The study may lack statistical power.",
                    }
                )
            result["diagnostics"] = diagnostics

            # Interaction plot
            means = df.groupby([factor_a, factor_b])[response].mean().unstack()
            traces = []
            for col in means.columns:
                traces.append(
                    {
                        "type": "scatter",
                        "x": means.index.astype(str).tolist(),
                        "y": means[col].tolist(),
                        "mode": "lines+markers",
                        "name": str(col),
                    }
                )

            result["plots"].append(
                {
                    "title": "Interaction Plot",
                    "data": traces,
                    "layout": {"height": 300, "xaxis": {"title": factor_a}},
                }
            )

        except ImportError:
            result["summary"] = "Two-way ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Two-way ANOVA error: {str(e)}"

    elif analysis_id == "correlation":
        vars_list = config.get("variables") or config.get("vars") or []
        method = config.get("method", "pearson")

        if vars_list:
            numeric_cols = [v for v in vars_list if v in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[numeric_cols].corr(method=method)

        # Build formatted output
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CORRELATION ANALYSIS ({method.upper()})<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(numeric_cols)}\n\n"
        summary += "<<COLOR:accent>>── Correlation Matrix ──<</COLOR>>\n"
        summary += corr_matrix.to_string() + "\n"

        # Find and report strongest correlations with practical interpretation
        len(df[numeric_cols].dropna())  # n_obs computed for validation
        strong_pairs = []
        stat_dict = {}
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    r = corr_matrix.loc[col1, col2]
                    # Compute p-value for each pair
                    pair_data = df[[col1, col2]].dropna()
                    if len(pair_data) > 2:
                        if method == "spearman":
                            r_val, p_val = stats.spearmanr(pair_data[col1], pair_data[col2])
                        elif method == "kendall":
                            r_val, p_val = stats.kendalltau(pair_data[col1], pair_data[col2])
                        else:
                            r_val, p_val = stats.pearsonr(pair_data[col1], pair_data[col2])
                    else:
                        r_val, p_val = float(r), 1.0
                    r2_val = r_val**2
                    label, meaningful = _effect_magnitude(r2_val, "r_squared")
                    stat_dict[f"r({col1},{col2})"] = float(r_val)
                    stat_dict[f"p({col1},{col2})"] = float(p_val)
                    if abs(r_val) >= 0.3:
                        strong_pairs.append(
                            (
                                col1,
                                col2,
                                float(r_val),
                                float(p_val),
                                label,
                                len(pair_data),
                            )
                        )

        if strong_pairs:
            strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            summary += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
            summary += "<<COLOR:title>>KEY RELATIONSHIPS<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"
            for col1, col2, r_val, p_val, label, n_pair in strong_pairs[:8]:
                direction = "positive" if r_val > 0 else "negative"
                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                r2_pct = r_val**2 * 100
                # Fisher z-transform CI for r
                _z_r = np.arctanh(r_val) if abs(r_val) < 0.9999 else np.sign(r_val) * 3
                _se_z = 1 / np.sqrt(n_pair - 3) if n_pair > 3 else 1
                _ci_r = (np.tanh(_z_r - 1.96 * _se_z), np.tanh(_z_r + 1.96 * _se_z))
                summary += f"<<COLOR:highlight>>{col1} ↔ {col2}:<</COLOR>> r = {r_val:+.3f} {sig} 95% CI [{_ci_r[0]:+.3f}, {_ci_r[1]:+.3f}] — {label} {direction} ({r2_pct:.0f}% shared variance)\n"
            summary += f"\n<<COLOR:text>>Strongest: {strong_pairs[0][0]} and {strong_pairs[0][1]} share {strong_pairs[0][2] ** 2 * 100:.0f}% of their variation.<</COLOR>>"
        else:
            summary += "\n<<COLOR:text>>No strong correlations found (all |r| < 0.3).<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Correlation ({method}): {len(strong_pairs)} pairs with |r| ≥ 0.3." + (
            f" Strongest: {strong_pairs[0][0]} ↔ {strong_pairs[0][1]} (r={strong_pairs[0][2]:.3f})."
            if strong_pairs
            else " No strong relationships found."
        )

        # Narrative (enhanced — spurious warning, sample size context)
        _n_obs = len(df[numeric_cols[0]].dropna()) if numeric_cols else 0
        _spur_warn = ""
        if _n_obs < 30 and strong_pairs:
            _spur_warn = " <em>Warning: with n &lt; 30, correlations can be inflated by outliers. Verify with a larger sample.</em>"
        elif len(numeric_cols) > 10 and strong_pairs:
            _n_tests = len(numeric_cols) * (len(numeric_cols) - 1) // 2
            _spur_warn = f" <em>Note: {_n_tests} pairwise tests — some correlations may be spurious. Apply Bonferroni correction (α = {0.05 / _n_tests:.4f}) for strict interpretation.</em>"

        if strong_pairs:
            top = strong_pairs[0]
            r2_pct = top[2] ** 2 * 100
            verdict = f"{len(strong_pairs)} strong correlation{'s' if len(strong_pairs) > 1 else ''} found"
            body = f"The strongest relationship is <strong>{top[0]} &harr; {top[1]}</strong> (r = {top[2]:+.3f}), sharing {r2_pct:.0f}% of their variation. Method: {method.title()}.{_spur_warn}"
            if len(strong_pairs) > 1:
                body += f" Second: {strong_pairs[1][0]} &harr; {strong_pairs[1][1]} (r = {strong_pairs[1][2]:+.3f})."
            nexts = "Correlation &ne; causation. Use <strong>Causal Discovery</strong> tools or a designed experiment to test directionality. Consider partial correlations to control for confounders."
        else:
            verdict = "No strong correlations detected"
            body = f"All pairwise correlations are below |r| = 0.3. The variables in this dataset appear to be largely independent ({method.title()} method, n = {_n_obs})."
            nexts = "If you expected relationships, check for non-linear associations (try Spearman) or confounding variables that may mask true relationships."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nexts,
            chart_guidance="In the heatmap, darker colors indicate stronger correlations. Green = positive (both increase together), red = negative (one increases as the other decreases).",
        )

        result["statistics"] = stat_dict

        # ── Diagnostics ──
        diagnostics = []
        # If Pearson, check normality and suggest Spearman if non-normal
        if method == "pearson" and numeric_cols:
            _any_nonnorm = False
            for _cc in numeric_cols[:4]:  # check first few
                _nd = _check_normality(df[_cc].dropna().values, label=_cc, alpha=0.05)
                if _nd:
                    _any_nonnorm = True
                    break
            if _any_nonnorm:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": "Non-normal data detected (Pearson assumes normality)",
                        "detail": "Pearson's r is sensitive to outliers and non-normality. Spearman's rank correlation is more robust.",
                        "action": {
                            "label": "Run Spearman Correlation",
                            "type": "stats",
                            "analysis": "correlation",
                            "config": {"vars": numeric_cols, "method": "spearman"},
                        },
                    }
                )
                # Cross-validate: run Spearman internally
                try:
                    _sp_corr = df[numeric_cols].corr(method="spearman")
                    if strong_pairs:
                        _sp_r = _sp_corr.loc[strong_pairs[0][0], strong_pairs[0][1]]
                        _cv = _cross_validate(
                            strong_pairs[0][3],
                            strong_pairs[0][3],
                            "Pearson",
                            "Spearman",
                            alpha=0.05,
                        )
                        _cv["detail"] = (
                            f"Pearson r = {strong_pairs[0][2]:.3f}, Spearman \u03c1 = {_sp_r:.3f} for {strong_pairs[0][0]} \u2194 {strong_pairs[0][1]}."
                        )
                        if abs(strong_pairs[0][2] - _sp_r) > 0.15:
                            _cv["level"] = "warning"
                            _cv["title"] = "Pearson and Spearman differ notably"
                            _cv["detail"] += " This suggests outliers or non-linearity are affecting Pearson's r."
                        diagnostics.append(_cv)
                except Exception:
                    pass
        # Multiple testing warning
        if len(numeric_cols) > 5:
            _n_tests = len(numeric_cols) * (len(numeric_cols) - 1) // 2
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"{_n_tests} pairwise comparisons \u2014 multiple testing risk",
                    "detail": f"With {_n_tests} tests, expect ~{_n_tests * 0.05:.0f} false positives at \u03b1=0.05. Apply Bonferroni (\u03b1 = {0.05 / _n_tests:.4f}) for strict control.",
                }
            )
        # Effect size emphasis
        if strong_pairs and strong_pairs[0][2] ** 2 >= 0.5:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Strong shared variation: {strong_pairs[0][0]} \u2194 {strong_pairs[0][1]} ({strong_pairs[0][2] ** 2 * 100:.0f}%)",
                    "detail": "These variables share over half their variation \u2014 one may be predictable from the other.",
                }
            )
        result["diagnostics"] = diagnostics

        # --- Bayesian Insurance (strongest pair only) ---
        try:
            if strong_pairs:
                _sp = strong_pairs[0]
                _sp_data = df[[_sp[0], _sp[1]]].dropna()
                _shadow = _bayesian_shadow("correlation", x=_sp_data[_sp[0]].values, y=_sp_data[_sp[1]].values)
                if _shadow:
                    result["bayesian_shadow"] = _shadow
                _r2_label, _ = _effect_magnitude(_sp[2] ** 2, "r_squared")
                _grade = _evidence_grade(
                    _sp[3],
                    bf10=_shadow.get("bf10") if _shadow else None,
                    effect_magnitude=_r2_label,
                )
                if _grade:
                    result["evidence_grade"] = _grade
        except Exception:
            pass

        # Build p-value customdata matrix for heatmap hover
        p_matrix = []
        for r_col in numeric_cols:
            row = []
            for c_col in numeric_cols:
                if r_col == c_col:
                    row.append("\u2014")
                else:
                    pv = stat_dict.get(f"p({r_col},{c_col})") or stat_dict.get(f"p({c_col},{r_col})")
                    row.append(f"p = {pv:.4f}" if pv is not None else "\u2014")
            p_matrix.append(row)

        # Heatmap
        result["plots"].append(
            {
                "title": "Correlation Heatmap",
                "data": [
                    {
                        "type": "heatmap",
                        "z": corr_matrix.values.tolist(),
                        "x": numeric_cols,
                        "y": numeric_cols,
                        "colorscale": "RdBu",
                        "zmid": 0,
                        "text": [[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                        "texttemplate": "%{text}",
                        "textfont": {"size": 10},
                        "customdata": p_matrix,
                        "hovertemplate": "r(%{x}, %{y}) = %{text}<br>%{customdata}<extra></extra>",
                    }
                ],
                "layout": {"height": 400},
                "interactive": {"type": "correlation_heatmap", "columns": numeric_cols},
            }
        )

    elif analysis_id == "normality":
        var = config.get("var")
        test_type = config.get("test", "anderson")

        x = df[var].dropna()
        n = len(x)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>NORMALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n\n"

        if test_type == "anderson":
            stat_result = stats.anderson(x)
            summary += "<<COLOR:text>>Anderson-Darling Test<</COLOR>>\n"
            summary += f"  Statistic: {stat_result.statistic:.4f}\n"
            summary += "  Critical Values:\n"
            for cv, sl in zip(stat_result.critical_values, stat_result.significance_level):
                marker = "<<COLOR:good>>✓<</COLOR>>" if stat_result.statistic < cv else "<<COLOR:bad>>✗<</COLOR>>"
                summary += f"    {marker} {sl}%: {cv:.4f}\n"
        elif test_type == "shapiro":
            stat, pval = stats.shapiro(x)
            summary += "<<COLOR:text>>Shapiro-Wilk Test<</COLOR>>\n"
            summary += f"  W-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += "\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += "\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"
        elif test_type == "ks":
            stat, pval = stats.kstest(x, "norm", args=(x.mean(), x.std()))
            summary += "<<COLOR:text>>Kolmogorov-Smirnov Test<</COLOR>>\n"
            summary += f"  D-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += "\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += "\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Q-Q plot — rank-based quantiles (Hazen plotting position)
        sorted_data = np.sort(x)
        n_qq = len(x)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n_qq + 1) - 0.5) / n_qq)

        result["plots"].append(
            {
                "title": f"Normal Q-Q Plot: {var}",
                "data": [
                    {
                        "type": "scatter",
                        "x": theoretical_quantiles.tolist(),
                        "y": sorted_data.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.5)",
                            "size": 6,
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": [
                            float(theoretical_quantiles.min()),
                            float(theoretical_quantiles.max()),
                        ],
                        "y": [float(sorted_data.min()), float(sorted_data.max())],
                        "mode": "lines",
                        "line": {"color": "#ff7675", "dash": "dash"},
                        "name": "Reference",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Theoretical Quantiles"},
                    "yaxis": {"title": "Sample Quantiles"},
                },
            }
        )

        # Histogram with normal curve overlay
        x_range = np.linspace(float(x.min()), float(x.max()), 100)
        normal_pdf = stats.norm.pdf(x_range, x.mean(), x.std())
        bin_width = (x.max() - x.min()) / min(30, max(5, int(np.sqrt(n))))
        normal_scaled = normal_pdf * n * bin_width

        result["plots"].append(
            {
                "title": f"Histogram with Normal Curve: {var}",
                "data": [
                    {
                        "type": "histogram",
                        "x": x.tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": x_range.tolist(),
                        "y": normal_scaled.tolist(),
                        "mode": "lines",
                        "line": {"color": "#d94a4a", "width": 2},
                        "name": "Normal fit",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": var},
                    "yaxis": {"title": "Count"},
                    "barmode": "overlay",
                },
            }
        )

        # Shape descriptors
        _skew = float(x.skew()) if hasattr(x, "skew") else float(pd.Series(x).skew())
        _kurt = float(x.kurtosis()) if hasattr(x, "kurtosis") else float(pd.Series(x).kurtosis())
        _shape_parts = []
        if abs(_skew) > 1:
            _shape_parts.append(f"{'right' if _skew > 0 else 'left'}-skewed (skewness = {_skew:.2f})")
        elif abs(_skew) > 0.5:
            _shape_parts.append(f"moderately {'right' if _skew > 0 else 'left'}-skewed (skewness = {_skew:.2f})")
        if _kurt > 1:
            _shape_parts.append(f"heavy-tailed (excess kurtosis = {_kurt:.2f})")
        elif _kurt < -1:
            _shape_parts.append(f"light-tailed (excess kurtosis = {_kurt:.2f})")
        _shape_desc = ", ".join(_shape_parts) if _shape_parts else "approximately symmetric with normal tail weight"

        # Significance and test label depend on test type
        if test_type == "anderson":
            _is_sig = stat_result.statistic > stat_result.critical_values[2]  # 5% level
            _test_label = "Anderson-Darling"
            _stat_str = (
                f"A\u00b2 = {stat_result.statistic:.4f}, 5% critical value = {stat_result.critical_values[2]:.4f}"
            )
        elif test_type == "shapiro":
            _is_sig = pval < 0.05
            _test_label = "Shapiro-Wilk"
            _stat_str = f"W = {stat:.4f}, p = {pval:.4f}"
        else:
            _is_sig = pval < 0.05
            _test_label = "Kolmogorov-Smirnov"
            _stat_str = f"D = {stat:.4f}, p = {pval:.4f}"

        if _is_sig:
            _n_verdict = "Data departs significantly from normality"
            _n_body = f"The {_test_label} test ({_stat_str}) rejects the normality assumption at \u03b1 = 0.05. The distribution is {_shape_desc}."
            _n_next = "For hypothesis tests, consider non-parametric alternatives (Mann-Whitney, Kruskal-Wallis). For capability analysis, use non-normal methods or fit an appropriate distribution."
        else:
            _n_verdict = "Data appears normally distributed"
            _n_body = f"The {_test_label} test ({_stat_str}) does not reject normality at \u03b1 = 0.05. The distribution is {_shape_desc}."
            _n_next = "Standard parametric methods (t-tests, ANOVA, normal capability) are appropriate for this data."

        result["guide_observation"] = (
            f"Normality test ({_test_label}): {_stat_str}. {'Data is NOT normal.' if _is_sig else 'Data appears normal.'} Distribution is {_shape_desc}."
        )
        result["narrative"] = _narrative(
            _n_verdict,
            _n_body,
            next_steps=_n_next,
            chart_guidance="Points falling off the Q-Q diagonal indicate departures from normality. The histogram overlay shows how well the normal curve fits the data.",
        )

        # ── Diagnostics ──
        diagnostics = []
        _out = _check_outliers(x.values, label=var)
        if _out:
            _out["detail"] += " Outliers inflate test statistics and can cause false rejection of normality."
            diagnostics.append(_out)

        # Effect size: deviation from normality via skewness/kurtosis
        _dev_parts = []
        if abs(_skew) > 0.5:
            _dev_parts.append(f"skewness = {_skew:.2f}")
        if abs(_kurt) > 1:
            _dev_parts.append(f"excess kurtosis = {_kurt:.2f}")
        if _dev_parts:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Shape deviation: {', '.join(_dev_parts)}",
                    "detail": f"Skewness = {_skew:.2f} (0 = symmetric), excess kurtosis = {_kurt:.2f} (0 = normal tails). Large departures indicate non-normality even if the test is marginal.",
                }
            )

        # Parametric / non-parametric guidance
        if not _is_sig:
            diagnostics.append(
                {
                    "level": "info",
                    "title": "Data supports parametric methods",
                    "detail": "The normality assumption holds. t-tests, ANOVA, and regression are appropriate for this data.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "action",
                    "title": "Non-normal data — consider distribution fitting or transformation",
                    "detail": "Fit an alternative distribution or apply a variance-stabilizing transformation to enable parametric analysis.",
                    "actions": [
                        {
                            "label": "Fit Distribution",
                            "type": "stats",
                            "analysis": "distribution_fit",
                            "config": {"var": var},
                        },
                        {
                            "label": "Box-Cox Transform",
                            "type": "stats",
                            "analysis": "box_cox",
                            "config": {"var": var},
                        },
                    ],
                }
            )

        # Sample size advisory
        if n < 20:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Low sample size (n = {n})",
                    "detail": "Normality tests have low power with fewer than 20 observations. Failure to reject normality does not mean the data is normal — inspect the Q-Q plot visually.",
                }
            )
        elif n > 5000:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Very large sample (n = {n:,})",
                    "detail": "With n > 5,000, normality tests are over-sensitive and will reject even trivial departures. Focus on the Q-Q plot and practical effect of skewness/kurtosis.",
                }
            )

        result["diagnostics"] = diagnostics

    elif analysis_id == "split_plot_anova":
        """
        Split-Plot ANOVA — for designs with hard-to-change (whole-plot) and
        easy-to-change (sub-plot) factors. Uses restricted error terms:
        whole-plot factors tested against whole-plot error, sub-plot factors
        against residual error.
        """
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        response_sp = config.get("response") or config.get("var")
        whole_plot_factors_sp = config.get("whole_plot_factors", [])
        sub_plot_factors_sp = config.get("sub_plot_factors", [])
        block_col_sp = config.get("block") or config.get("whole_plot_id")

        if isinstance(whole_plot_factors_sp, str):
            whole_plot_factors_sp = [whole_plot_factors_sp]
        if isinstance(sub_plot_factors_sp, str):
            sub_plot_factors_sp = [sub_plot_factors_sp]

        all_factors_sp = whole_plot_factors_sp + sub_plot_factors_sp
        needed_cols = [response_sp] + all_factors_sp + ([block_col_sp] if block_col_sp else [])
        data_sp = df[needed_cols].dropna()

        # Ensure factors are categorical
        for f in all_factors_sp:
            data_sp[f] = data_sp[f].astype(str)
        if block_col_sp:
            data_sp[block_col_sp] = data_sp[block_col_sp].astype(str)

        try:
            # Build model formula with interactions
            wp_terms = " + ".join([f"C({f})" for f in whole_plot_factors_sp])
            sp_terms = " + ".join([f"C({f})" for f in sub_plot_factors_sp])
            # Whole-plot × sub-plot interactions
            interactions = []
            for wp in whole_plot_factors_sp:
                for sp in sub_plot_factors_sp:
                    interactions.append(f"C({wp}):C({sp})")
            interaction_terms = " + ".join(interactions) if interactions else ""

            formula_parts = [wp_terms]
            if block_col_sp:
                formula_parts.append(f"C({block_col_sp})")
            formula_parts.append(sp_terms)
            if interaction_terms:
                formula_parts.append(interaction_terms)
            formula_sp = f"{response_sp} ~ " + " + ".join(formula_parts)

            model_sp = ols(formula_sp, data=data_sp).fit()

            # Full ANOVA table
            anova_table = sm.stats.anova_lm(model_sp, typ=2)

            # Separate whole-plot and sub-plot errors
            # Whole-plot error = block(whole-plot) interaction, or pooled if no blocks
            # For proper split-plot: test WP factors against WP error
            anova_rows = []
            ss_total = anova_table["sum_sq"].sum()
            ms_resid = (
                anova_table.loc["Residual", "mean_sq"]
                if "Residual" in anova_table.index
                else anova_table.iloc[-1]["mean_sq"]
            )
            df_resid_sp = (
                anova_table.loc["Residual", "df"] if "Residual" in anova_table.index else anova_table.iloc[-1]["df"]
            )

            # If we have blocks, compute whole-plot error from block SS
            wp_error_ms = ms_resid  # default to residual
            wp_error_df = df_resid_sp
            if block_col_sp and f"C({block_col_sp})" in anova_table.index:
                wp_error_ms = anova_table.loc[f"C({block_col_sp})", "mean_sq"]
                wp_error_df = anova_table.loc[f"C({block_col_sp})", "df"]

            for idx_row in anova_table.index:
                if idx_row == "Residual":
                    continue
                ss = float(anova_table.loc[idx_row, "sum_sq"])
                df_val = float(anova_table.loc[idx_row, "df"])
                ms = float(anova_table.loc[idx_row, "mean_sq"])

                # Determine error term
                clean_name = idx_row.replace("C(", "").replace(")", "").replace(":", " × ")
                is_wp = any(f in idx_row for f in whole_plot_factors_sp) and not any(
                    f in idx_row for f in sub_plot_factors_sp
                )

                if is_wp and block_col_sp:
                    error_ms = wp_error_ms
                    error_df = wp_error_df
                    error_term = "WP Error"
                else:
                    error_ms = ms_resid
                    error_df = df_resid_sp
                    error_term = "Residual"

                f_val = ms / error_ms if error_ms > 0 else 0
                p_val = 1 - sp_stats.f.cdf(f_val, df_val, error_df) if f_val > 0 else 1.0
                pct_contrib = ss / ss_total * 100

                anova_rows.append(
                    {
                        "source": clean_name,
                        "ss": ss,
                        "df": int(df_val),
                        "ms": ms,
                        "f": f_val,
                        "p": p_val,
                        "pct": pct_contrib,
                        "error_term": error_term,
                        "significant": p_val < 0.05,
                    }
                )

            # Add error rows
            if block_col_sp and f"C({block_col_sp})" in anova_table.index:
                anova_rows.append(
                    {
                        "source": "Whole-Plot Error",
                        "ss": float(wp_error_ms * wp_error_df),
                        "df": int(wp_error_df),
                        "ms": float(wp_error_ms),
                        "f": None,
                        "p": None,
                        "pct": float(wp_error_ms * wp_error_df / ss_total * 100),
                        "error_term": "",
                    }
                )
            anova_rows.append(
                {
                    "source": "Sub-Plot Error (Residual)",
                    "ss": float(ms_resid * df_resid_sp),
                    "df": int(df_resid_sp),
                    "ms": float(ms_resid),
                    "f": None,
                    "p": None,
                    "pct": float(ms_resid * df_resid_sp / ss_total * 100),
                    "error_term": "",
                }
            )

            # Summary
            summary_sp = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_sp += "<<COLOR:title>>SPLIT-PLOT ANOVA<</COLOR>>\n"
            summary_sp += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_sp += f"<<COLOR:highlight>>Response:<</COLOR>> {response_sp}\n"
            summary_sp += f"<<COLOR:highlight>>Whole-plot factors:<</COLOR>> {', '.join(whole_plot_factors_sp)}\n"
            summary_sp += f"<<COLOR:highlight>>Sub-plot factors:<</COLOR>> {', '.join(sub_plot_factors_sp)}\n"
            if block_col_sp:
                summary_sp += f"<<COLOR:highlight>>Block (whole-plot ID):<</COLOR>> {block_col_sp}\n"
            summary_sp += f"<<COLOR:highlight>>N:<</COLOR>> {len(data_sp)}\n\n"

            summary_sp += "<<COLOR:accent>>── ANOVA Table ──<</COLOR>>\n"
            summary_sp += (
                f"{'Source':<30} {'SS':>10} {'df':>4} {'MS':>10} {'F':>8} {'p':>8} {'%Contrib':>8} {'Error Term':<12}\n"
            )
            summary_sp += f"{'─' * 95}\n"
            for r in anova_rows:
                f_str = f"{r['f']:>8.3f}" if r["f"] is not None else f"{'':>8}"
                p_str = f"{r['p']:>8.4f}" if r["p"] is not None else f"{'':>8}"
                sig = " <<COLOR:good>>*<</COLOR>>" if r.get("significant") else ""
                summary_sp += f"{r['source']:<30} {r['ss']:>10.2f} {r['df']:>4} {r['ms']:>10.3f} {f_str} {p_str} {r['pct']:>7.1f}% {r.get('error_term', ''):<12}{sig}\n"

            result["summary"] = summary_sp

            # Residual plots
            resids_sp = model_sp.resid
            fitted_sp = model_sp.fittedvalues
            result["plots"].append(
                {
                    "title": "Residuals vs Fitted Values",
                    "data": [
                        {
                            "type": "scatter",
                            "mode": "markers",
                            "x": fitted_sp.tolist(),
                            "y": resids_sp.tolist(),
                            "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6},
                            "showlegend": False,
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Fitted"},
                        "yaxis": {"title": "Residual"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": min(fitted_sp),
                                "x1": max(fitted_sp),
                                "y0": 0,
                                "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            # Main effects plot
            me_traces = []
            for fi, factor in enumerate(all_factors_sp):
                grp = data_sp.groupby(factor)[response_sp].mean()
                me_traces.append(
                    {
                        "x": [str(lv) for lv in grp.index],
                        "y": grp.values.tolist(),
                        "mode": "lines+markers",
                        "name": factor,
                        "marker": {"size": 8},
                    }
                )
            result["plots"].append(
                {
                    "title": "Main Effects Plot",
                    "data": me_traces,
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Factor Level"},
                        "yaxis": {"title": f"Mean {response_sp}"},
                    },
                }
            )

            n_sig_sp = sum(1 for r in anova_rows if r.get("significant"))
            result["guide_observation"] = (
                f"Split-plot ANOVA: {n_sig_sp} significant terms. WP factors tested against WP error, SP factors against residual."
            )
            result["statistics"] = {
                "anova_table": anova_rows,
                "r_squared": float(model_sp.rsquared),
                "n": len(data_sp),
            }
            result["narrative"] = _narrative(
                f"Split-Plot ANOVA: {n_sig_sp} significant term{'s' if n_sig_sp != 1 else ''}",
                f"R\u00b2 = {model_sp.rsquared:.3f}. Whole-plot factors are tested against the whole-plot error, subplot factors against the residual error.",
                next_steps="Split-plot designs arise when some factors are harder to change than others. Check both error terms for proper inference.",
            )

        except Exception as e:
            result["summary"] = f"Split-plot ANOVA error: {str(e)}"

    # =====================================================================
    # Repeated Measures ANOVA
    # =====================================================================
    elif analysis_id == "repeated_measures_anova":
        """
        Repeated Measures ANOVA — tests within-subject effects across time points
        or conditions. Includes Mauchly's sphericity test and Greenhouse-Geisser /
        Huynh-Feldt corrections when sphericity is violated.
        """
        response_rm = config.get("response") or config.get("var")
        subject_col = config.get("subject") or config.get("subject_id")
        within_factor = config.get("within_factor") or config.get("condition")
        between_factor = config.get("between_factor")  # optional

        needed_cols_rm = [response_rm, subject_col, within_factor]
        if between_factor:
            needed_cols_rm.append(between_factor)
        data_rm = df[needed_cols_rm].dropna()
        data_rm[subject_col] = data_rm[subject_col].astype(str)
        data_rm[within_factor] = data_rm[within_factor].astype(str)

        try:
            subjects = data_rm[subject_col].unique()
            conditions = sorted(data_rm[within_factor].unique())
            k_rm = len(conditions)
            len(subjects)  # n_subj computed for validation

            if k_rm < 2:
                result["summary"] = "Need at least 2 levels of the within-subject factor."
                return result

            # Build subject × condition matrix
            pivot_rm = data_rm.pivot_table(
                index=subject_col,
                columns=within_factor,
                values=response_rm,
                aggfunc="mean",
            )
            # Drop subjects with missing conditions
            pivot_rm = pivot_rm.dropna()
            n_complete = len(pivot_rm)

            if n_complete < 3:
                result["summary"] = "Need at least 3 complete subjects (all conditions measured)."
                return result

            Y_rm = pivot_rm.values  # n_subj × k
            grand_mean_rm = float(np.mean(Y_rm))
            subj_means = np.mean(Y_rm, axis=1)
            cond_means = np.mean(Y_rm, axis=0)

            # SS decomposition
            ss_total_rm = float(np.sum((Y_rm - grand_mean_rm) ** 2))
            ss_between_subj = k_rm * float(np.sum((subj_means - grand_mean_rm) ** 2))
            ss_within_subj = ss_total_rm - ss_between_subj
            ss_condition = n_complete * float(np.sum((cond_means - grand_mean_rm) ** 2))
            ss_error_rm = ss_within_subj - ss_condition

            df_condition = k_rm - 1
            df_subjects = n_complete - 1
            df_error_rm = df_condition * df_subjects

            ms_condition = ss_condition / df_condition if df_condition > 0 else 0
            ms_error_rm = ss_error_rm / df_error_rm if df_error_rm > 0 else 0

            f_val_rm = ms_condition / ms_error_rm if ms_error_rm > 0 else 0
            from scipy import stats as rm_stats

            p_val_rm = 1 - rm_stats.f.cdf(f_val_rm, df_condition, df_error_rm) if f_val_rm > 0 else 1.0

            # Mauchly's test of sphericity
            # Compute covariance matrix of differences
            if k_rm > 2:
                # Orthogonal contrasts
                C_mat = np.zeros((k_rm, k_rm - 1))
                for j in range(k_rm - 1):
                    C_mat[j, j] = 1
                    C_mat[j + 1, j] = -1
                S_diff = Y_rm @ C_mat  # n × (k-1)
                cov_diff = np.cov(S_diff.T)
                det_cov = np.linalg.det(cov_diff)
                trace_cov = np.trace(cov_diff)
                p_rm = k_rm - 1

                # Mauchly's W
                W_mauchly = det_cov / ((trace_cov / p_rm) ** p_rm) if trace_cov > 0 else 0
                # Chi-square approximation
                f_coeff = (2 * p_rm**2 + p_rm + 2) / (6 * p_rm * df_subjects)
                chi2_mauchly = -(df_subjects - f_coeff) * np.log(max(W_mauchly, 1e-15))
                df_mauchly = p_rm * (p_rm + 1) / 2 - 1
                p_mauchly = 1 - rm_stats.chi2.cdf(chi2_mauchly, df_mauchly) if df_mauchly > 0 else 1.0

                # Greenhouse-Geisser epsilon
                eigenvals_cov = np.linalg.eigvalsh(cov_diff)
                eigenvals_cov = eigenvals_cov[eigenvals_cov > 0]
                gg_epsilon = (
                    (np.sum(eigenvals_cov) ** 2) / (p_rm * np.sum(eigenvals_cov**2)) if len(eigenvals_cov) > 0 else 1.0
                )
                gg_epsilon = min(1.0, max(1.0 / p_rm, gg_epsilon))

                # Huynh-Feldt epsilon
                hf_epsilon = (n_complete * p_rm * gg_epsilon - 2) / (p_rm * (df_subjects - p_rm * gg_epsilon))
                hf_epsilon = min(1.0, max(gg_epsilon, hf_epsilon))

                # Corrected p-values
                p_gg = 1 - rm_stats.f.cdf(f_val_rm, df_condition * gg_epsilon, df_error_rm * gg_epsilon)
                p_hf = 1 - rm_stats.f.cdf(f_val_rm, df_condition * hf_epsilon, df_error_rm * hf_epsilon)
            else:
                W_mauchly = 1.0
                p_mauchly = 1.0
                gg_epsilon = 1.0
                hf_epsilon = 1.0
                p_gg = p_val_rm
                p_hf = p_val_rm

            # Summary
            summary_rm = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_rm += "<<COLOR:title>>REPEATED MEASURES ANOVA<</COLOR>>\n"
            summary_rm += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_rm += f"<<COLOR:highlight>>Response:<</COLOR>> {response_rm}\n"
            summary_rm += f"<<COLOR:highlight>>Within-subject factor:<</COLOR>> {within_factor} ({k_rm} levels)\n"
            summary_rm += f"<<COLOR:highlight>>Subjects:<</COLOR>> {n_complete}\n\n"

            summary_rm += "<<COLOR:accent>>── Within-Subjects ANOVA ──<</COLOR>>\n"
            summary_rm += f"{'Source':<25} {'SS':>10} {'df':>4} {'MS':>10} {'F':>8} {'p':>8}\n"
            summary_rm += f"{'─' * 70}\n"
            sig_mark = " <<COLOR:good>>*<</COLOR>>" if p_val_rm < 0.05 else ""
            summary_rm += f"{'Condition':<25} {ss_condition:>10.2f} {df_condition:>4} {ms_condition:>10.3f} {f_val_rm:>8.3f} {p_val_rm:>8.4f}{sig_mark}\n"
            summary_rm += f"{'Error':<25} {ss_error_rm:>10.2f} {df_error_rm:>4} {ms_error_rm:>10.3f}\n"

            if k_rm > 2:
                summary_rm += "\n<<COLOR:accent>>── Mauchly's Test of Sphericity ──<</COLOR>>\n"
                summary_rm += f"  W = {W_mauchly:.4f},  χ² = {chi2_mauchly:.3f},  p = {p_mauchly:.4f}\n"
                if p_mauchly < 0.05:
                    summary_rm += "  <<COLOR:warning>>Sphericity violated — use corrected tests below<</COLOR>>\n"
                else:
                    summary_rm += "  <<COLOR:good>>Sphericity assumption met<</COLOR>>\n"

                summary_rm += "\n<<COLOR:accent>>── Epsilon Corrections ──<</COLOR>>\n"
                summary_rm += f"  Greenhouse-Geisser ε = {gg_epsilon:.4f}  →  p = {p_gg:.4f}\n"
                summary_rm += f"  Huynh-Feldt ε = {hf_epsilon:.4f}  →  p = {p_hf:.4f}\n"

            # Condition means
            summary_rm += "\n<<COLOR:accent>>── Condition Means ──<</COLOR>>\n"
            for ci, cond in enumerate(pivot_rm.columns):
                summary_rm += f"  {cond}: {cond_means[ci]:.4f} (SD = {np.std(Y_rm[:, ci], ddof=1):.4f})\n"

            # Partial eta-squared
            eta_sq = ss_condition / (ss_condition + ss_error_rm) if (ss_condition + ss_error_rm) > 0 else 0
            summary_rm += f"\n<<COLOR:accent>>── Effect Size ──<</COLOR>> partial η² = {eta_sq:.4f}\n"

            result["summary"] = summary_rm

            # Profile plot (condition means with SE)
            se_conds = np.std(Y_rm, axis=0, ddof=1) / np.sqrt(n_complete)
            result["plots"].append(
                {
                    "title": "Profile Plot — Condition Means (±SE)",
                    "data": [
                        {
                            "x": [str(c) for c in pivot_rm.columns],
                            "y": cond_means.tolist(),
                            "mode": "lines+markers",
                            "name": "Mean",
                            "marker": {"color": "#4a9f6e", "size": 10},
                            "line": {"color": "#4a9f6e", "width": 2},
                            "error_y": {
                                "type": "data",
                                "array": se_conds.tolist(),
                                "visible": True,
                                "color": "#5a6a5a",
                            },
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": within_factor},
                        "yaxis": {"title": f"Mean {response_rm}"},
                    },
                }
            )

            # Individual subject trajectories (spaghetti plot)
            spaghetti_traces = []
            for si, subj in enumerate(pivot_rm.index[:30]):  # limit to 30 subjects for readability
                spaghetti_traces.append(
                    {
                        "x": [str(c) for c in pivot_rm.columns],
                        "y": pivot_rm.loc[subj].tolist(),
                        "mode": "lines",
                        "line": {"color": "#5a6a5a", "width": 0.5},
                        "opacity": 0.3,
                        "showlegend": False,
                    }
                )
            # Overlay mean
            spaghetti_traces.append(
                {
                    "x": [str(c) for c in pivot_rm.columns],
                    "y": cond_means.tolist(),
                    "mode": "lines+markers",
                    "name": "Grand Mean",
                    "marker": {"color": "#4a9f6e", "size": 8},
                    "line": {"color": "#4a9f6e", "width": 3},
                }
            )
            result["plots"].append(
                {
                    "title": "Subject Trajectories (spaghetti plot)",
                    "data": spaghetti_traces,
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": within_factor},
                        "yaxis": {"title": response_rm},
                    },
                }
            )

            best_p = p_gg if (k_rm > 2 and p_mauchly < 0.05) else p_val_rm
            result["guide_observation"] = (
                f"Repeated measures ANOVA: F({df_condition},{df_error_rm})={f_val_rm:.3f}, p={best_p:.4f}, η²={eta_sq:.4f}."
            )
            _eta_label = "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small"
            if best_p < 0.05:
                result["narrative"] = _narrative(
                    f"Conditions differ significantly (F = {f_val_rm:.3f}, p = {best_p:.4f})",
                    f"\u03b7\u00b2 = {eta_sq:.4f} ({_eta_label} effect). The repeated-measures factor significantly affects the response.",
                    next_steps="Run post-hoc paired comparisons to identify which conditions differ.",
                )
            else:
                result["narrative"] = _narrative(
                    f"No significant effect across conditions (p = {best_p:.4f})",
                    f"\u03b7\u00b2 = {eta_sq:.4f}. The conditions do not significantly differ.",
                    next_steps="If sphericity is violated (Mauchly's test), use Greenhouse-Geisser or Huynh-Feldt correction.",
                )
            result["statistics"] = {
                "f_value": f_val_rm,
                "p_value": p_val_rm,
                "df_condition": df_condition,
                "df_error": df_error_rm,
                "ss_condition": ss_condition,
                "ss_error": ss_error_rm,
                "partial_eta_squared": eta_sq,
                "n_subjects": n_complete,
                "k_levels": k_rm,
                "mauchly_w": float(W_mauchly),
                "mauchly_p": float(p_mauchly),
                "gg_epsilon": float(gg_epsilon),
                "hf_epsilon": float(hf_epsilon),
                "p_gg": float(p_gg),
                "p_hf": float(p_hf),
                "condition_means": {str(c): float(m) for c, m in zip(pivot_rm.columns, cond_means)},
            }

        except Exception as e:
            result["summary"] = f"Repeated measures ANOVA error: {str(e)}"

    # ── Power & Sample Size Calculators ──────────────────────────────────────

    elif analysis_id == "f_test":
        """
        F-test for equality of variances between two groups.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"F-test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        var1, var2 = g1.var(), g2.var()
        n1, n2 = len(g1), len(g2)

        # F statistic (larger variance / smaller variance)
        if var1 >= var2:
            F = var1 / var2
            df1, df2 = n1 - 1, n2 - 1
        else:
            F = var2 / var1
            df1, df2 = n2 - 1, n1 - 1

        from scipy import stats

        p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>F-TEST FOR EQUALITY OF VARIANCES<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n\n"
        summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
        summary += f"  {groups[0]}: n={n1}, variance={var1:.4f}, StDev={np.sqrt(var1):.4f}\n"
        summary += f"  {groups[1]}: n={n2}, variance={var2:.4f}, StDev={np.sqrt(var2):.4f}\n\n"
        summary += f"<<COLOR:highlight>>F statistic:<</COLOR>> {F:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n"
        _f_ci_lo = F / stats.f.ppf(0.975, df1, df2)
        _f_ci_hi = F / stats.f.ppf(0.025, df1, df2)
        summary += f"<<COLOR:highlight>>95% CI for variance ratio:<</COLOR>> [{_f_ci_lo:.4f}, {_f_ci_hi:.4f}]\n"
        _ln_ratio = np.log(max(var1, var2) / min(var1, var2))
        summary += f"<<COLOR:highlight>>Log variance ratio:<</COLOR>> {_ln_ratio:.4f} (0 = equal variances)\n\n"

        if p_value < 0.05:
            summary += "<<COLOR:warning>>Variances are SIGNIFICANTLY DIFFERENT (p < 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>Consider using Welch's t-test or non-parametric alternatives.<</COLOR>>\n"
        else:
            summary += "<<COLOR:good>>No significant difference in variances (p >= 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>Equal variance assumption is reasonable.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"F = {F:.2f}, p = {p_value:.4f}. " + (
            "Variances differ significantly." if p_value < 0.05 else "Variances are similar."
        )
        result["statistics"] = {
            "F_statistic": float(F),
            "p_value": float(p_value),
            "variance_ratio": float(max(var1, var2) / min(var1, var2)),
        }
        _vr = float(max(var1, var2) / min(var1, var2)) if min(var1, var2) > 0 else 0
        if p_value < 0.05:
            verdict = f"Variances differ significantly (F = {F:.2f}, p = {p_value:.4f})"
            body = f"Variance ratio = {_vr:.2f}. Use Welch's t-test (not pooled) or non-parametric tests for group comparisons."
        else:
            verdict = f"Variances are similar (F = {F:.2f}, p = {p_value:.4f})"
            body = (
                f"Variance ratio = {_vr:.2f}. The equal-variance assumption is reasonable for pooled t-tests and ANOVA."
            )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="F-test is sensitive to non-normality. For robust alternatives, use Levene's test.",
        )

        # Variance comparison bar chart + side-by-side box plots
        result["plots"].append(
            {
                "title": f"Variance Comparison: {groups[0]} vs {groups[1]}",
                "data": [
                    {
                        "type": "bar",
                        "x": [str(groups[0]), str(groups[1])],
                        "y": [float(var1), float(var2)],
                        "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                        "name": "Variance",
                    },
                ],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": "Variance"},
                    "xaxis": {"title": group_var},
                },
            }
        )
        result["plots"].append(
            {
                "title": f"Distribution by Group: {var}",
                "data": [
                    {
                        "type": "box",
                        "y": g1.tolist(),
                        "name": str(groups[0]),
                        "marker": {"color": "#4a9f6e"},
                        "boxpoints": "outliers",
                    },
                    {
                        "type": "box",
                        "y": g2.tolist(),
                        "name": str(groups[1]),
                        "marker": {"color": "#4a90d9"},
                        "boxpoints": "outliers",
                    },
                ],
                "layout": {"height": 300, "yaxis": {"title": var}},
            }
        )

    elif analysis_id == "equivalence":
        """
        TOST (Two One-Sided Tests) for equivalence testing.
        Tests whether two means are equivalent within a specified margin.
        """
        var = config.get("var")
        group_var = config.get("group_var")
        margin = float(config.get("margin", 0.5))  # Equivalence margin

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Equivalence test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        from scipy import stats

        mean1, mean2 = g1.mean(), g2.mean()
        std1, std2 = g1.std(), g2.std()
        n1, n2 = len(g1), len(g2)

        diff = mean1 - mean2
        se = np.sqrt(std1**2 / n1 + std2**2 / n2)
        df_val = n1 + n2 - 2

        # TOST: Two one-sided tests
        t_lower = (diff - (-margin)) / se
        t_upper = (diff - margin) / se

        p_lower = 1 - stats.t.cdf(t_lower, df_val)
        p_upper = stats.t.cdf(t_upper, df_val)
        p_tost = max(p_lower, p_upper)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>EQUIVALENCE TEST (TOST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n"
        summary += f"<<COLOR:highlight>>Equivalence margin:<</COLOR>> ±{margin}\n\n"
        summary += "<<COLOR:accent>>── Group Means ──<</COLOR>>\n"
        summary += f"  {groups[0]}: {mean1:.4f} (n={n1})\n"
        summary += f"  {groups[1]}: {mean2:.4f} (n={n2})\n"
        summary += f"  Difference: {diff:.4f}\n\n"
        summary += f"<<COLOR:highlight>>TOST p-value:<</COLOR>> {p_tost:.4f}\n"
        _ci90_lo = diff - stats.t.ppf(0.95, df_val) * se
        _ci90_hi = diff + stats.t.ppf(0.95, df_val) * se
        summary += f"<<COLOR:highlight>>90% CI for difference:<</COLOR>> [{_ci90_lo:.4f}, {_ci90_hi:.4f}]  (must fall within ±{margin} for equivalence)\n"
        _ci95_lo = diff - stats.t.ppf(0.975, df_val) * se
        _ci95_hi = diff + stats.t.ppf(0.975, df_val) * se
        summary += f"<<COLOR:highlight>>95% CI for difference:<</COLOR>> [{_ci95_lo:.4f}, {_ci95_hi:.4f}]\n\n"

        if p_tost < 0.05:
            summary += f"<<COLOR:good>>EQUIVALENT within ±{margin} (p < 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>The difference is small enough to be considered equivalent.<</COLOR>>\n"
        else:
            summary += "<<COLOR:warning>>NOT EQUIVALENT (p >= 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>Cannot conclude equivalence within the specified margin.<</COLOR>>\n"

        # Equivalence plot
        result["plots"].append(
            {
                "title": "Equivalence Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": [diff],
                        "y": [0.5],
                        "mode": "markers",
                        "marker": {
                            "size": 15,
                            "color": "#4a9f6e" if p_tost < 0.05 else "#e85747",
                        },
                        "error_x": {
                            "type": "constant",
                            "value": 1.96 * se,
                            "color": "#4a9f6e",
                        },
                        "name": "Mean Difference",
                    }
                ],
                "layout": {
                    "height": 200,
                    "xaxis": {"title": "Difference", "zeroline": True},
                    "yaxis": {"visible": False, "range": [0, 1]},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": -margin,
                            "x1": -margin,
                            "y0": 0,
                            "y1": 1,
                            "line": {"color": "#e89547", "dash": "dash"},
                        },
                        {
                            "type": "line",
                            "x0": margin,
                            "x1": margin,
                            "y0": 0,
                            "y1": 1,
                            "line": {"color": "#e89547", "dash": "dash"},
                        },
                        {
                            "type": "rect",
                            "x0": -margin,
                            "x1": margin,
                            "y0": 0,
                            "y1": 1,
                            "fillcolor": "rgba(74,159,110,0.1)",
                            "line": {"width": 0},
                        },
                    ],
                },
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"TOST p = {p_tost:.4f}. " + (
            f"Groups equivalent within ±{margin}." if p_tost < 0.05 else "Cannot confirm equivalence."
        )
        result["statistics"] = {
            "TOST_p_value": float(p_tost),
            "mean_difference": float(diff),
            "margin": float(margin),
        }

        # Narrative
        if p_tost < 0.05:
            verdict = f"Groups are equivalent within ±{margin} (TOST p = {p_tost:.4f})"
            body = (
                f"The mean difference ({diff:.4f}) falls within the equivalence margin of ±{margin}. "
                f"90% CI [{_ci90_lo:.4f}, {_ci90_hi:.4f}] is entirely inside the equivalence bounds. "
                f"This provides statistical evidence that the groups are practically equivalent."
            )
            nxt = "Equivalence confirmed. The groups can be treated as interchangeable within the specified margin."
        else:
            verdict = f"Cannot confirm equivalence (TOST p = {p_tost:.4f})"
            body = (
                f"The mean difference is {diff:.4f}, but the 90% CI [{_ci90_lo:.4f}, {_ci90_hi:.4f}] "
                f"extends beyond the ±{margin} equivalence margin. Cannot conclude the groups are equivalent."
            )
            nxt = "Consider: (1) widening the margin if scientifically justified, (2) increasing sample size for more power, or (3) the groups may genuinely differ."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nxt,
            chart_guidance="The green shaded region is the equivalence zone (±margin). If the CI (error bars) falls entirely within it, equivalence is confirmed.",
        )

        # ── Diagnostics ──
        diagnostics = []
        _norm1 = _check_normality(g1.values, label=f"{var} [{groups[0]}]")
        if _norm1:
            diagnostics.append(_norm1)
        _norm2 = _check_normality(g2.values, label=f"{var} [{groups[1]}]")
        if _norm2:
            diagnostics.append(_norm2)
        _out1 = _check_outliers(g1.values, label=f"{var} [{groups[0]}]")
        if _out1:
            diagnostics.append(_out1)
        _out2 = _check_outliers(g2.values, label=f"{var} [{groups[1]}]")
        if _out2:
            diagnostics.append(_out2)

        # Cohen's d alongside TOST
        _pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
        _d = abs(diff) / _pooled_std if _pooled_std > 0 else 0.0
        _d_label, _d_meaningful = _effect_magnitude(_d, "cohens_d")

        if p_tost < 0.05 and not _d_meaningful:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Groups are both statistically and practically equivalent (d = {_d:.2f}, {_d_label})",
                    "detail": f"TOST confirms equivalence and Cohen's d = {_d:.2f} indicates a {_d_label} effect — the groups are interchangeable.",
                }
            )
        elif p_tost < 0.05 and _d_meaningful:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Formally equivalent but meaningful difference exists (d = {_d:.2f}, {_d_label})",
                    "detail": f"TOST p = {p_tost:.4f} confirms equivalence within ±{margin}, but Cohen's d = {_d:.2f} ({_d_label}) suggests a practically relevant difference. Consider tightening the equivalence margin.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Effect size: Cohen's d = {_d:.2f} ({_d_label})",
                    "detail": f"Equivalence not confirmed (TOST p = {p_tost:.4f}). The observed effect is {_d_label}. Consider widening the equivalence margin if scientifically justified, or increasing sample size (current n1={n1}, n2={n2}).",
                }
            )

        result["diagnostics"] = diagnostics

    elif analysis_id == "sign_test":
        """
        One-sample sign test for median.
        Non-parametric alternative to one-sample t-test.
        """
        var = config.get("var")
        h0_median = float(config.get("hypothesized_median", 0))
        data = df[var].dropna().values

        from scipy import stats

        # Count values above and below hypothesized median
        above = np.sum(data > h0_median)
        below = np.sum(data < h0_median)
        ties = np.sum(data == h0_median)
        n_used = above + below  # ties excluded

        # Two-sided binomial test
        k = min(above, below)
        p_value = 2 * stats.binom.cdf(k, n_used, 0.5) if n_used > 0 else 1.0
        p_value = min(p_value, 1.0)

        sample_median = np.median(data)

        # Confidence interval on median (binomial-based)
        # ppf gives smallest k with cdf(k) >= 0.025; we need largest k with cdf(k) < 0.025
        sorted_data = np.sort(data)
        n_total = len(data)
        ci_idx = max(0, int(stats.binom.ppf(0.025, n_total, 0.5)) - 1)
        ci_lower = sorted_data[ci_idx] if ci_idx < n_total else sorted_data[0]
        ci_upper = sorted_data[n_total - 1 - ci_idx] if ci_idx < n_total else sorted_data[-1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>SIGN TEST FOR MEDIAN<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>H₀ Median:<</COLOR>> {h0_median}\n\n"
        summary += "<<COLOR:accent>>── Sample Statistics ──<</COLOR>>\n"
        summary += f"  N: {len(data)}\n"
        summary += f"  Sample median: {sample_median:.4f}\n"
        summary += f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
        summary += "<<COLOR:accent>>── Sign Counts ──<</COLOR>>\n"
        summary += f"  Above H₀: {above}\n"
        summary += f"  Below H₀: {below}\n"
        summary += f"  Ties (excluded): {ties}\n\n"
        summary += f"<<COLOR:highlight>>p-value (two-sided):<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>REJECT H₀ — Median differs from {h0_median}<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>FAIL TO REJECT H₀ — No evidence median differs from {h0_median}<</COLOR>>\n"

        # Dot plot with median lines
        result["plots"].append(
            {
                "title": f"Sign Test: {var}",
                "data": [
                    {
                        "type": "scatter",
                        "y": data.tolist(),
                        "mode": "markers",
                        "name": "Data",
                        "marker": {
                            "color": [
                                ("#4a9f6e" if v > h0_median else "#d94a4a" if v < h0_median else "#e89547")
                                for v in data
                            ],
                            "size": 6,
                        },
                    },
                    {
                        "type": "scatter",
                        "y": [sample_median] * len(data),
                        "mode": "lines",
                        "name": f"Sample Median ({sample_median:.2f})",
                        "line": {"color": "#4a9f6e", "dash": "dash"},
                    },
                    {
                        "type": "scatter",
                        "y": [h0_median] * len(data),
                        "mode": "lines",
                        "name": f"H₀ ({h0_median})",
                        "line": {"color": "#d94a4a", "dash": "dot"},
                    },
                ],
                "layout": {
                    "height": 250,
                    "showlegend": True,
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": var},
                },
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Sign test p = {p_value:.4f}. " + (
            f"Median differs from {h0_median}." if p_value < 0.05 else f"No evidence median differs from {h0_median}."
        )
        result["statistics"] = {
            "sample_median": float(sample_median),
            "above": int(above),
            "below": int(below),
            "p_value": float(p_value),
        }
        if p_value < 0.05:
            verdict = f"Median differs from {h0_median} (p = {p_value:.4f})"
            body = f"Sample median = {sample_median:.4f}. {above} values above and {below} below {h0_median}. The imbalance is significant."
        else:
            verdict = f"Median consistent with {h0_median} (p = {p_value:.4f})"
            body = f"Sample median = {sample_median:.4f}. {above} above and {below} below. No significant departure from {h0_median}."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="The sign test is the most robust non-parametric test — it only uses the direction of differences, not magnitudes.",
        )

    return result
