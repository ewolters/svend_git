"""DSW Statistical Analysis — nonparametric tests (Mann-Whitney, Kruskal-Wallis, etc.)."""

import logging

import numpy as np

from .common import (
    _check_normality,
    _check_outliers,
    _cross_validate,
    _narrative,
)

logger = logging.getLogger(__name__)


def _run_nonparametric(analysis_id, df, config):
    """Run nonparametric analysis."""
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "mann_whitney":
        """
        Mann-Whitney U Test - non-parametric alternative to 2-sample t-test.
        Tests if two groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Mann-Whitney U requires exactly 2 groups. Found {len(groups)}: {list(groups)}"
            return result

        group1 = df[df[group_var] == groups[0]][var].dropna()
        group2 = df[df[group_var] == groups[1]][var].dropna()

        stat, pval = stats.mannwhitneyu(group1, group2, alternative="two-sided")

        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>MANN-WHITNEY U TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
        summary += f"  {groups[0]}: median = {group1.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[:n1].mean():.1f}\n"
        summary += f"  {groups[1]}: median = {group2.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[n1:].mean():.1f}\n\n"

        summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
        summary += f"  U statistic: {stat:.2f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  Effect size (r): {effect_size:.3f}\n"
        # Hodges-Lehmann median difference + CI
        if n1 * n2 <= 500000:
            _all_diffs = np.subtract.outer(group1.values, group2.values).ravel()
            _hl_med = float(np.median(_all_diffs))
            _sorted_d = np.sort(_all_diffs)
            _C_a = stats.norm.ppf(0.975) * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            _lo_i = max(0, int(np.floor(n1 * n2 / 2 - _C_a)))
            _hi_i = min(len(_sorted_d) - 1, int(np.ceil(n1 * n2 / 2 + _C_a)))
            summary += f"  Hodges-Lehmann median diff: {_hl_med:.4f}, 95% CI [{_sorted_d[_lo_i]:.4f}, {_sorted_d[_hi_i]:.4f}]\n"
        summary += "\n"

        if pval < 0.05:
            summary += "<<COLOR:good>>Groups differ significantly (p < 0.05)<</COLOR>>\n"
        else:
            summary += "<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots
        result["plots"].append(
            {
                "title": f"Mann-Whitney: {var} by {group_var}",
                "data": [
                    {
                        "type": "box",
                        "y": group1.tolist(),
                        "name": str(groups[0]),
                        "marker": {"color": "#4a9f6e"},
                        "fillcolor": "rgba(74, 159, 110, 0.3)",
                    },
                    {
                        "type": "box",
                        "y": group2.tolist(),
                        "name": str(groups[1]),
                        "marker": {"color": "#47a5e8"},
                        "fillcolor": "rgba(71, 165, 232, 0.3)",
                    },
                ],
                "layout": {"height": 300, "yaxis": {"title": var}},
            }
        )

        result["guide_observation"] = f"Mann-Whitney U test p = {pval:.4f}. " + (
            "Groups differ significantly." if pval < 0.05 else "No significant difference."
        )
        result["statistics"] = {"U_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_size)}

        # ── Diagnostics: assumption checks + cross-validation ──
        diagnostics = []
        _alpha = 0.05
        _out1 = _check_outliers(group1.values, label=str(groups[0]))
        _out2 = _check_outliers(group2.values, label=str(groups[1]))
        if _out1:
            diagnostics.append(_out1)
        if _out2:
            diagnostics.append(_out2)
        # Cross-validate with t-test (if data is roughly normal)
        try:
            _norm1 = _check_normality(group1.values, label=str(groups[0]), alpha=_alpha)
            _norm2 = _check_normality(group2.values, label=str(groups[1]), alpha=_alpha)
            _any_nonnormal = bool(_norm1 or _norm2)
            _t_stat, _t_p = stats.ttest_ind(group1, group2, equal_var=False)
            _cv = _cross_validate(
                pval, _t_p, "Mann-Whitney U", "Welch's t-test", alpha=_alpha, normality_failed=_any_nonnormal
            )
            _cv["action"] = {
                "label": "Run t-test",
                "type": "stats",
                "analysis": "ttest2",
                "config": {"var": config.get("var", ""), "group_var": config.get("group_var", "")},
            }
            diagnostics.append(_cv)
        except Exception:
            pass
        # Effect size emphasis (rank-biserial r)
        _mw_r_abs = abs(effect_size)
        if _mw_r_abs >= 0.5 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (rank-biserial r = {effect_size:.3f})",
                    "detail": f"The rank-biserial correlation of {effect_size:.3f} indicates a large separation between groups — practically meaningful.",
                }
            )
        elif _mw_r_abs < 0.1 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (r = {effect_size:.3f})",
                    "detail": "Statistical significance with negligible practical effect. Large sample sizes can make tiny rank differences significant.",
                }
            )
        elif _mw_r_abs >= 0.3 and pval >= _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate effect not reaching significance (r = {effect_size:.3f})",
                    "detail": "The effect size suggests a real difference, but the sample may be too small to detect it. Consider collecting more data.",
                    "action": {
                        "label": "Power Analysis",
                        "type": "stats",
                        "analysis": "power_sample_size",
                        "config": {
                            "test_type": "mann_whitney",
                            "effect_size": float(_mw_r_abs),
                            "alpha": float(_alpha),
                        },
                    },
                }
            )
        result["diagnostics"] = diagnostics

        # Narrative
        _mw_eff_abs = abs(effect_size)
        _mw_eff_label = "large" if _mw_eff_abs >= 0.5 else ("medium" if _mw_eff_abs >= 0.3 else "small")
        _mw_higher = groups[0] if group1.median() > group2.median() else groups[1]
        if pval < 0.05:
            _mw_verdict = f"Groups differ significantly (rank-biserial r = {effect_size:.3f}, {_mw_eff_label} effect)"
            _mw_body = f"Values in <strong>{_mw_higher}</strong> tend to be higher (median {max(group1.median(), group2.median()):.4f} vs {min(group1.median(), group2.median()):.4f}). The Mann-Whitney U test (p = {pval:.4f}) confirms this difference is unlikely due to chance."
            _mw_next = "Investigate the root cause of the difference between groups."
        else:
            _mw_verdict = "No significant difference between groups"
            _mw_body = f"The Mann-Whitney U test (p = {pval:.4f}) does not detect a significant difference between {groups[0]} and {groups[1]}. Effect size r = {effect_size:.3f} ({_mw_eff_label})."
            _mw_next = "Consider whether the sample size provides adequate power, or use a bootstrap CI for distribution-free inference."
        result["narrative"] = _narrative(
            _mw_verdict,
            _mw_body,
            next_steps=_mw_next,
            chart_guidance="Box plots show the median (center line), IQR (box), and range (whiskers). Non-overlapping boxes suggest different central tendencies.",
        )

    elif analysis_id == "kruskal":
        """
        Kruskal-Wallis H Test - non-parametric alternative to one-way ANOVA.
        Tests if multiple groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        group_data = [df[df[group_var] == g][var].dropna().values for g in groups]

        stat, pval = stats.kruskal(*group_data)

        # Effect size (epsilon squared), clamped to [0, 1]
        n_total = sum(len(g) for g in group_data)
        epsilon_sq = (
            max(0.0, min(1.0, (stat - len(groups) + 1) / (n_total - len(groups)))) if n_total > len(groups) else 0.0
        )

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>KRUSKAL-WALLIS H TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(groups)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n\n"

        summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
        for g, gdata in zip(groups, group_data):
            _n_g = len(gdata)
            _med = np.median(gdata)
            if _n_g > 1:
                _sorted_g = np.sort(gdata)
                _j = max(0, int(np.floor(_n_g / 2 - 1.96 * np.sqrt(_n_g) / 2)))
                _k = min(_n_g - 1, int(np.ceil(_n_g / 2 + 1.96 * np.sqrt(_n_g) / 2)))
                summary += f"  {g}: n={_n_g}, median={_med:.4f}, 95% CI [{_sorted_g[_j]:.4f}, {_sorted_g[_k]:.4f}]\n"
            else:
                summary += f"  {g}: n={_n_g}, median={_med:.4f}\n"

        summary += "\n<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
        summary += f"  H statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  ε² (effect size): {epsilon_sq:.4f}\n\n"

        if pval < 0.05:
            summary += "<<COLOR:good>>At least one group differs significantly (p < 0.05)<</COLOR>>\n"
            summary += (
                "<<COLOR:text>>Consider post-hoc tests (Dunn's test) to identify which groups differ.<</COLOR>>\n"
            )
        else:
            summary += "<<COLOR:text>>No significant difference among groups (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for all groups
        result["plots"].append(
            {
                "title": f"Kruskal-Wallis: {var} by {group_var}",
                "data": [
                    {
                        "type": "box",
                        "y": data.tolist(),
                        "name": str(g),
                        "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"][i % 5]},
                        "fillcolor": [
                            "rgba(74,159,110,0.3)",
                            "rgba(71,165,232,0.3)",
                            "rgba(232,149,71,0.3)",
                            "rgba(159,74,74,0.3)",
                            "rgba(108,92,231,0.3)",
                        ][i % 5],
                    }
                    for i, (g, data) in enumerate(zip(groups, group_data))
                ],
                "layout": {"height": 300, "yaxis": {"title": var}},
            }
        )

        result["guide_observation"] = f"Kruskal-Wallis H = {stat:.2f}, p = {pval:.4f}. " + (
            "Groups differ." if pval < 0.05 else "No difference."
        )
        result["statistics"] = {
            "H_statistic": float(stat),
            "p_value": float(pval),
            "epsilon_squared": float(epsilon_sq),
        }

        # ── Diagnostics: assumption checks + cross-validation ──
        diagnostics = []
        _alpha = 0.05
        for _g_name, _g_data in zip(groups, group_data):
            _out = _check_outliers(_g_data, label=str(_g_name))
            if _out:
                diagnostics.append(_out)
        # Cross-validate with one-way ANOVA (if groups roughly normal)
        try:
            _norms = [_check_normality(gd, label=str(gn), alpha=_alpha) for gn, gd in zip(groups, group_data)]
            _any_nonnormal = any(_norms)
            _f_stat, _f_p = stats.f_oneway(*group_data)
            _cv = _cross_validate(
                pval, _f_p, "Kruskal-Wallis", "One-way ANOVA", alpha=_alpha, normality_failed=_any_nonnormal
            )
            _cv["action"] = {
                "label": "Run ANOVA",
                "type": "stats",
                "analysis": "anova",
                "config": {"var": config.get("var", ""), "group_var": config.get("group_var", "")},
            }
            diagnostics.append(_cv)
        except Exception:
            pass
        # Effect size emphasis (η² = H / (n - 1))
        _eta_sq = stat / (n_total - 1) if n_total > 1 else 0.0
        if _eta_sq > 0.14 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (\u03b7\u00b2 = {_eta_sq:.4f})",
                    "detail": f"The eta-squared of {_eta_sq:.4f} indicates a large effect — group membership explains a substantial share of rank variance.",
                }
            )
        elif _eta_sq < 0.01 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (\u03b7\u00b2 = {_eta_sq:.4f})",
                    "detail": "Statistical significance with negligible practical effect. Large combined sample size can make tiny rank differences significant.",
                }
            )
        elif _eta_sq >= 0.06 and pval >= _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate effect not reaching significance (\u03b7\u00b2 = {_eta_sq:.4f})",
                    "detail": "The effect size suggests real group differences, but the sample may be too small. Consider collecting more data.",
                    "action": {
                        "label": "Power Analysis",
                        "type": "stats",
                        "analysis": "power_sample_size",
                        "config": {"test_type": "kruskal", "effect_size": float(_eta_sq), "alpha": float(_alpha)},
                    },
                }
            )
        result["diagnostics"] = diagnostics

        # Narrative
        _es_label = "large" if epsilon_sq > 0.14 else "medium" if epsilon_sq > 0.06 else "small"
        if pval < 0.05:
            verdict = f"Groups differ significantly (H = {stat:.2f}, p = {pval:.4f})"
            body = (
                f"At least one of {len(groups)} groups has a different distribution of <strong>{var}</strong> "
                f"(ε² = {epsilon_sq:.3f}, {_es_label} effect). This is the non-parametric equivalent of one-way ANOVA."
            )
            nxt = "Run Dunn's post-hoc test to identify which specific groups differ."
        else:
            verdict = f"No significant difference among groups (p = {pval:.4f})"
            body = (
                f"The {len(groups)} groups of <strong>{var}</strong> do not differ significantly "
                f"(H = {stat:.2f}, ε² = {epsilon_sq:.3f})."
            )
            nxt = "If you expected a difference, check sample sizes — the test may lack power with small groups."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nxt,
            chart_guidance="Box plots show group distributions. Compare medians (middle lines) and spreads (box widths). Outlier dots may drive rank differences.",
        )

    elif analysis_id == "wilcoxon":
        """
        Wilcoxon Signed-Rank Test - non-parametric alternative to paired t-test.
        Tests if paired differences are symmetrically distributed around zero.
        """
        var1 = config.get("var1")
        var2 = config.get("var2")

        # Support stacked/factor format: response column + grouping column
        if config.get("data_format") == "factor":
            response_col = config.get("response") or var1
            factor_col = config.get("group_var") or config.get("factor") or var2
            levels = df[factor_col].dropna().unique()
            if len(levels) != 2:
                result["summary"] = f"Wilcoxon test requires exactly 2 groups. Found {len(levels)}: {list(levels)}"
                return result
            sample1 = df[df[factor_col] == levels[0]][response_col].dropna().reset_index(drop=True)
            sample2 = df[df[factor_col] == levels[1]][response_col].dropna().reset_index(drop=True)
            var1, var2 = f"{response_col} [{levels[0]}]", f"{response_col} [{levels[1]}]"
            min_len = min(len(sample1), len(sample2))
            sample1 = sample1.iloc[:min_len]
            sample2 = sample2.iloc[:min_len]
        else:
            sample1 = df[var1].dropna()
            sample2 = df[var2].dropna()
            # Align lengths for paired data
            min_len = min(len(sample1), len(sample2))
            sample1 = sample1.iloc[:min_len]
            sample2 = sample2.iloc[:min_len]

        if min_len < 6:
            result["summary"] = f"Wilcoxon signed-rank requires at least 6 paired observations. Got {min_len}."
            return result

        diffs = sample1.values - sample2.values
        stat, pval = stats.wilcoxon(sample1, sample2)

        # Effect size: r = Z / sqrt(N) — compute Z directly from test statistic
        mean_w = min_len * (min_len + 1) / 4
        std_w = np.sqrt(min_len * (min_len + 1) * (2 * min_len + 1) / 24)
        z_score = (stat - mean_w) / std_w if std_w > 0 else 0.0
        effect_r = abs(z_score) / np.sqrt(min_len)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>WILCOXON SIGNED-RANK TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Pair:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N pairs:<</COLOR>> {min_len}\n\n"
        summary += "<<COLOR:accent>>── Differences (var1 - var2) ──<</COLOR>>\n"
        summary += f"  Median diff: {np.median(diffs):.4f}\n"
        summary += f"  Mean diff:   {np.mean(diffs):.4f}\n"
        summary += f"  Std diff:    {np.std(diffs, ddof=1):.4f}\n\n"
        summary += f"<<COLOR:highlight>>Test Statistic (W):<</COLOR>> {stat:.2f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>Effect Size (r):<</COLOR>> {effect_r:.4f}\n\n"

        if pval < 0.05:
            summary += "<<COLOR:accent>>Significant difference between paired samples (p < 0.05)<</COLOR>>\n"
        else:
            summary += "<<COLOR:text>>No significant difference between paired samples (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Histogram of differences
        result["plots"].append(
            {
                "title": f"Paired Differences: {var1} - {var2}",
                "data": [
                    {
                        "type": "histogram",
                        "x": diffs.tolist(),
                        "name": "Differences",
                        "marker": {"color": "rgba(71,165,232,0.7)", "line": {"color": "#47a5e8", "width": 1}},
                    },
                    {
                        "type": "scatter",
                        "x": [0, 0],
                        "y": [0, min_len // 3],
                        "mode": "lines",
                        "name": "Zero",
                        "line": {"color": "#e89547", "dash": "dash", "width": 2},
                    },
                ],
                "layout": {"height": 300, "xaxis": {"title": "Difference"}, "yaxis": {"title": "Count"}},
            }
        )

        result["guide_observation"] = f"Wilcoxon signed-rank W = {stat:.2f}, p = {pval:.4f}. " + (
            "Paired samples differ." if pval < 0.05 else "No paired difference."
        )
        result["statistics"] = {
            "W_statistic": float(stat),
            "p_value": float(pval),
            "effect_size_r": float(effect_r),
            "median_diff": float(np.median(diffs)),
            "n_pairs": int(min_len),
        }

        # ── Diagnostics: assumption checks + cross-validation ──
        diagnostics = []
        _alpha = 0.05
        _out_diffs = _check_outliers(diffs, label="Paired differences")
        if _out_diffs:
            diagnostics.append(_out_diffs)
        # Cross-validate with paired t-test
        try:
            _norm_diffs = _check_normality(diffs, label="Paired differences", alpha=_alpha)
            _any_nonnormal = bool(_norm_diffs)
            _t_stat, _t_p = stats.ttest_rel(sample1, sample2)
            _cv = _cross_validate(
                pval, _t_p, "Wilcoxon signed-rank", "Paired t-test", alpha=_alpha, normality_failed=_any_nonnormal
            )
            _cv["action"] = {
                "label": "Run paired t-test",
                "type": "stats",
                "analysis": "ttest_paired",
                "config": {"var1": config.get("var1", ""), "var2": config.get("var2", "")},
            }
            diagnostics.append(_cv)
        except Exception:
            pass
        # Effect size emphasis (r = Z / sqrt(N))
        if effect_r >= 0.5 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Large practical effect (r = {effect_r:.4f})",
                    "detail": f"The effect size r = {effect_r:.4f} indicates a large paired difference — practically meaningful.",
                }
            )
        elif effect_r < 0.1 and pval < _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but trivial effect (r = {effect_r:.4f})",
                    "detail": "Statistical significance with negligible practical effect. Large sample sizes can make tiny paired differences significant.",
                }
            )
        elif effect_r >= 0.3 and pval >= _alpha:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate effect not reaching significance (r = {effect_r:.4f})",
                    "detail": "The effect size suggests a real paired difference, but the sample may be too small to detect it. Consider collecting more data.",
                    "action": {
                        "label": "Power Analysis",
                        "type": "stats",
                        "analysis": "power_sample_size",
                        "config": {"test_type": "wilcoxon", "effect_size": float(effect_r), "alpha": float(_alpha)},
                    },
                }
            )
        result["diagnostics"] = diagnostics

        _er_label = "large" if effect_r > 0.5 else "medium" if effect_r > 0.3 else "small"
        if pval < 0.05:
            verdict = f"Paired samples differ (W = {stat:.1f}, p = {pval:.4f})"
            body = f"Median difference = {np.median(diffs):.4f} ({_er_label} effect, r = {effect_r:.3f}). The paired measurements are significantly different."
        else:
            verdict = f"No significant paired difference (p = {pval:.4f})"
            body = f"Median difference = {np.median(diffs):.4f} (r = {effect_r:.3f}). Cannot conclude the paired measurements differ."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="If significant, investigate what systematic factor drives the difference between paired measurements.",
            chart_guidance="The distribution of paired differences. Values centered away from zero indicate a systematic shift.",
        )

    elif analysis_id == "friedman":
        """
        Friedman Test - non-parametric alternative to repeated measures ANOVA.
        Tests if k related samples have different distributions.
        Requires multiple measurement columns (repeated measures).
        """
        vars_list = config.get("vars", [])
        if not vars_list:
            # Fallback: use var1 and var2 as minimum
            var1 = config.get("var1")
            var2 = config.get("var2")
            if var1 and var2:
                vars_list = [var1, var2]

        if len(vars_list) < 3:
            result["summary"] = (
                f"Friedman test requires at least 3 related samples (repeated measures). Got {len(vars_list)}.\n\nSelect 3+ measurement columns (e.g., Time1, Time2, Time3)."
            )
            return result

        # Drop rows with any missing values across all vars
        clean_df = df[vars_list].dropna()
        n_subjects = len(clean_df)

        if n_subjects < 5:
            result["summary"] = f"Friedman test requires at least 5 complete observations. Got {n_subjects}."
            return result

        groups = [clean_df[v].values for v in vars_list]
        stat, pval = stats.friedmanchisquare(*groups)

        # Effect size: Kendall's W = chi2 / (N * (k - 1))
        k = len(vars_list)
        kendall_w = stat / (n_subjects * (k - 1))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>FRIEDMAN TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Repeated Measures:<</COLOR>> {', '.join(vars_list)}\n"
        summary += f"<<COLOR:highlight>>N subjects:<</COLOR>> {n_subjects}\n"
        summary += f"<<COLOR:highlight>>k conditions:<</COLOR>> {k}\n\n"

        for v in vars_list:
            col = clean_df[v]
            summary += f"  {v}: median={col.median():.4f}, mean={col.mean():.4f}, sd={col.std():.4f}\n"
        summary += "\n"

        summary += f"<<COLOR:highlight>>Chi-square:<</COLOR>> {stat:.4f}\n"
        summary += f"<<COLOR:highlight>>df:<</COLOR>> {k - 1}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>Kendall's W:<</COLOR>> {kendall_w:.4f}\n\n"

        if pval < 0.05:
            summary += "<<COLOR:accent>>Significant difference across conditions (p < 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>Consider Wilcoxon signed-rank tests for pairwise comparisons (with Bonferroni correction).<</COLOR>>\n"
        else:
            summary += "<<COLOR:text>>No significant difference across conditions (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for each condition
        result["plots"].append(
            {
                "title": "Friedman: Repeated Measures Comparison",
                "data": [
                    {
                        "type": "box",
                        "y": clean_df[v].tolist(),
                        "name": v,
                        "marker": {
                            "color": [
                                "#4a9f6e",
                                "#47a5e8",
                                "#e89547",
                                "#9f4a4a",
                                "#6c5ce7",
                                "#e84747",
                                "#47e8c4",
                                "#c4e847",
                            ][i % 8]
                        },
                        "fillcolor": [
                            "rgba(74,159,110,0.3)",
                            "rgba(71,165,232,0.3)",
                            "rgba(232,149,71,0.3)",
                            "rgba(159,74,74,0.3)",
                            "rgba(108,92,231,0.3)",
                            "rgba(232,71,71,0.3)",
                            "rgba(71,232,196,0.3)",
                            "rgba(196,232,71,0.3)",
                        ][i % 8],
                    }
                    for i, v in enumerate(vars_list)
                ],
                "layout": {"height": 300, "yaxis": {"title": "Value"}},
            }
        )

        result["guide_observation"] = f"Friedman chi2 = {stat:.2f}, p = {pval:.4f}, W = {kendall_w:.3f}. " + (
            "Conditions differ." if pval < 0.05 else "No difference."
        )
        result["statistics"] = {
            "chi2_statistic": float(stat),
            "p_value": float(pval),
            "kendall_w": float(kendall_w),
            "df": int(k - 1),
            "n_subjects": int(n_subjects),
        }
        _w_label = "strong" if kendall_w > 0.7 else "moderate" if kendall_w > 0.3 else "weak"
        if pval < 0.05:
            verdict = f"Conditions differ significantly (\u03c7\u00b2 = {stat:.2f}, p = {pval:.4f})"
            body = f"At least one of {k} conditions has a different distribution ({n_subjects} subjects). Kendall's W = {kendall_w:.3f} ({_w_label} concordance)."
            nxt = "Run post-hoc Wilcoxon signed-rank tests with Bonferroni correction to identify which conditions differ."
        else:
            verdict = f"No significant difference across conditions (p = {pval:.4f})"
            body = f"The {k} conditions do not differ significantly (W = {kendall_w:.3f}). No evidence of a treatment effect."
            nxt = "If expected, increase sample size. The Friedman test has lower power than repeated-measures ANOVA."
        result["narrative"] = _narrative(verdict, body, next_steps=nxt)

    elif analysis_id == "spearman":
        """
        Spearman Rank Correlation - non-parametric measure of monotonic association.
        Returns rho, p-value, and confidence interval (unlike matrix-only correlation).
        """
        var1 = config.get("var1", config.get("var"))
        var2 = config.get("var2", config.get("group_var"))

        x = df[var1].dropna()
        y = df[var2].dropna()
        # Align
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        n = len(x)

        if n < 5:
            result["summary"] = f"Spearman correlation requires at least 5 observations. Got {n}."
            return result

        rho, pval = stats.spearmanr(x, y)

        # Fisher z-transform CI
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
        ci_low = np.tanh(z - 1.96 * se)
        ci_high = np.tanh(z + 1.96 * se)

        # Interpret strength
        abs_rho = abs(rho)
        if abs_rho >= 0.7:
            strength = "strong"
        elif abs_rho >= 0.4:
            strength = "moderate"
        elif abs_rho >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        direction = "positive" if rho > 0 else "negative"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>SPEARMAN RANK CORRELATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:highlight>>Spearman rho:<</COLOR>> {rho:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>95% CI:<</COLOR>> [{ci_low:.4f}, {ci_high:.4f}]\n\n"
        summary += (
            f"<<COLOR:text>>Interpretation: {strength.capitalize()} {direction} monotonic association<</COLOR>>\n"
        )

        if pval < 0.05:
            summary += "<<COLOR:accent>>Statistically significant (p < 0.05)<</COLOR>>\n"
        else:
            summary += "<<COLOR:text>>Not statistically significant (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Scatter with rank overlay
        result["plots"].append(
            {
                "title": f"Spearman: {var1} vs {var2} (rho={rho:.3f})",
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "marker": {"color": "#47a5e8", "size": 6, "opacity": 0.7},
                        "name": "Data",
                    }
                ],
                "layout": {"height": 300, "xaxis": {"title": var1}, "yaxis": {"title": var2}},
            }
        )

        result["guide_observation"] = (
            f"Spearman rho = {rho:.3f}, p = {pval:.4f}. {strength.capitalize()} {direction} monotonic association."
        )
        result["statistics"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
            "n": int(n),
        }
        if pval < 0.05:
            verdict = f"{strength.capitalize()} {direction} monotonic relationship (\u03c1 = {rho:+.3f})"
            body = f"<strong>{var1}</strong> and <strong>{var2}</strong> have a {strength} {direction} monotonic association (p = {pval:.4f}, 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]). Spearman's rank correlation is robust to outliers and non-linear relationships."
        else:
            verdict = f"No significant monotonic relationship (\u03c1 = {rho:+.3f}, p = {pval:.4f})"
            body = f"No significant monotonic association between <strong>{var1}</strong> and <strong>{var2}</strong>."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Spearman captures monotonic (not just linear) relationships. For linear-only, use Pearson. For causal testing, use designed experiments.",
            chart_guidance="Points along a rising/falling curve = monotonic relationship. Spearman doesn't require linearity.",
        )

    elif analysis_id == "runs_test":
        """
        Runs test for randomness in a sequence.
        Tests whether the sequence is random or has patterns.
        """
        var = config.get("var")
        data = df[var].dropna().values

        from scipy import stats

        # Convert to binary (above/below median)
        median = np.median(data)
        binary = (data > median).astype(int)

        # Count runs
        n_runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                n_runs += 1

        n_pos = np.sum(binary)
        n_neg = len(binary) - n_pos
        n = len(binary)

        # Expected runs and variance
        expected_runs = (2 * n_pos * n_neg / n) + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        # Z-score
        z = (n_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>RUNS TEST FOR RANDOMNESS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"
        summary += "<<COLOR:accent>>── Run Statistics ──<</COLOR>>\n"
        summary += f"  Observed runs: {n_runs}\n"
        summary += f"  Expected runs: {expected_runs:.2f}\n"
        summary += f"  Above median: {n_pos}\n"
        summary += f"  Below median: {n_neg}\n\n"
        summary += f"<<COLOR:highlight>>Z-statistic:<</COLOR>> {z:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            if n_runs < expected_runs:
                summary += "<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too few runs (clustering)<</COLOR>>\n"
                summary += "<<COLOR:text>>Values tend to cluster together, suggesting trends or patterns.<</COLOR>>\n"
            else:
                summary += "<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too many runs (oscillation)<</COLOR>>\n"
                summary += (
                    "<<COLOR:text>>Values alternate too frequently, suggesting negative autocorrelation.<</COLOR>>\n"
                )
        else:
            summary += "<<COLOR:good>>SEQUENCE APPEARS RANDOM (p >= 0.05)<</COLOR>>\n"
            summary += "<<COLOR:text>>No evidence of patterns or trends in the data.<</COLOR>>\n"

        # Plot the sequence with run coloring
        colors = []
        current_color = "#4a9f6e"
        for i in range(len(binary)):
            if i > 0 and binary[i] != binary[i - 1]:
                current_color = "#47a5e8" if current_color == "#4a9f6e" else "#4a9f6e"
            colors.append(current_color)

        result["plots"].append(
            {
                "title": f"Sequence Plot ({n_runs} runs)",
                "data": [
                    {
                        "type": "scatter",
                        "y": data.tolist(),
                        "mode": "lines+markers",
                        "marker": {"color": colors, "size": 6},
                        "line": {"color": "rgba(74,159,110,0.3)"},
                    },
                    {
                        "type": "scatter",
                        "y": [median] * len(data),
                        "mode": "lines",
                        "name": "Median",
                        "line": {"color": "#e89547", "dash": "dash"},
                    },
                ],
                "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Runs test: {n_runs} runs, p = {p_value:.4f}. " + (
            "Non-random pattern detected." if p_value < 0.05 else "Sequence appears random."
        )
        result["statistics"] = {
            "runs": int(n_runs),
            "expected_runs": float(expected_runs),
            "Z_statistic": float(z),
            "p_value": float(p_value),
        }
        if p_value < 0.05:
            verdict = f"Non-random pattern detected ({n_runs} runs, p = {p_value:.4f})"
            body = f"Expected {expected_runs:.1f} runs but observed {n_runs}. " + (
                "Too few runs = clustering/trends."
                if n_runs < expected_runs
                else "Too many runs = oscillation/over-correction."
            )
            nxt = "Investigate the cause of non-randomness: time trends, autocorrelation, or cyclical patterns."
        else:
            verdict = f"Sequence appears random (p = {p_value:.4f})"
            body = f"Observed {n_runs} runs vs {expected_runs:.1f} expected. No evidence of non-random patterns."
            nxt = None
        result["narrative"] = _narrative(verdict, body, next_steps=nxt)

    elif analysis_id == "mood_median":
        """
        Mood's Median Test - k-sample test comparing medians.
        Non-parametric alternative to one-way ANOVA for medians.
        """
        var = config.get("var")
        group_col = config.get("group") or config.get("group_var") or config.get("factor")

        from scipy import stats

        groups = df[group_col].dropna().unique()
        data_by_group = [df[df[group_col] == g][var].dropna().values for g in groups]
        all_data = np.concatenate(data_by_group)
        grand_median = np.median(all_data)

        # Contingency table: above/below grand median per group
        table = np.zeros((2, len(groups)), dtype=int)
        for j, grp_data in enumerate(data_by_group):
            table[0, j] = np.sum(grp_data > grand_median)  # above
            table[1, j] = np.sum(grp_data <= grand_median)  # at or below

        # Chi-squared test on contingency table
        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        group_medians = {str(g): float(np.median(d)) for g, d in zip(groups, data_by_group)}

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>MOOD'S MEDIAN TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {group_col}\n"
        summary += f"<<COLOR:highlight>>Grand Median:<</COLOR>> {grand_median:.4f}\n\n"
        summary += "<<COLOR:accent>>── Group Medians ──<</COLOR>>\n"
        for g, m in group_medians.items():
            summary += f"  {g}: {m:.4f}\n"
        summary += "\n<<COLOR:accent>>── Contingency Table (above/at-or-below grand median) ──<</COLOR>>\n"
        summary += f"  {'Group':<15} {'Above':>8} {'At/Below':>10} {'N':>6}\n"
        summary += f"  {'-' * 42}\n"
        for j, g in enumerate(groups):
            summary += f"  {str(g):<15} {table[0, j]:>8} {table[1, j]:>10} {table[0, j] + table[1, j]:>6}\n"
        summary += f"\n<<COLOR:highlight>>Chi-squared:<</COLOR>> {chi2:.4f}\n"
        summary += f"<<COLOR:highlight>>df:<</COLOR>> {dof}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += "<<COLOR:warning>>SIGNIFICANT — At least one group median differs<</COLOR>>\n"
        else:
            summary += "<<COLOR:good>>NOT SIGNIFICANT — No evidence of median differences<</COLOR>>\n"

        # Box plots by group
        traces = []
        theme_colors = ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a", "#9f4a4a", "#7a6a9a"]
        for i, (g, d) in enumerate(zip(groups, data_by_group)):
            traces.append(
                {
                    "type": "box",
                    "y": d.tolist(),
                    "name": str(g),
                    "marker": {"color": theme_colors[i % len(theme_colors)]},
                }
            )
        traces.append(
            {
                "type": "scatter",
                "x": [str(g) for g in groups],
                "y": [grand_median] * len(groups),
                "mode": "lines",
                "name": f"Grand Median ({grand_median:.2f})",
                "line": {"color": "#e89547", "dash": "dash", "width": 2},
            }
        )
        result["plots"].append(
            {
                "title": f"Mood's Median Test: {var} by {group_col}",
                "data": traces,
                "layout": {"height": 280, "showlegend": True},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Mood's median test: χ² = {chi2:.2f}, p = {p_value:.4f}. " + (
            "Medians differ." if p_value < 0.05 else "No evidence of median differences."
        )
        result["statistics"] = {
            "chi_squared": float(chi2),
            "df": int(dof),
            "p_value": float(p_value),
            "grand_median": float(grand_median),
        }
        if p_value < 0.05:
            verdict = f"Group medians differ (\u03c7\u00b2 = {chi2:.2f}, p = {p_value:.4f})"
            body = f"At least one group's median differs from the grand median of {grand_median:.4f}."
        else:
            verdict = f"No significant median differences (p = {p_value:.4f})"
            body = f"All groups have medians consistent with the grand median of {grand_median:.4f}."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Mood's test is more robust than Kruskal-Wallis to outliers but has less power. Use when outlier resistance is critical.",
        )

    elif analysis_id == "multi_vari":
        """
        Multi-Vari Chart - shows variation across multiple factors.
        Essential for understanding sources of variation.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if len(factors) < 1:
            result["summary"] = "Please select at least one factor."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>MULTI-VARI CHART<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n\n"

        colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"]

        if len(factors) == 1:
            # Single factor multi-vari
            factor = factors[0]
            groups = df.groupby(factor)[response]

            plot_data = []
            x_positions = []
            x_labels = []

            for i, (level, group_data) in enumerate(groups):
                vals = group_data.dropna().values
                x_pos = i
                x_positions.append(x_pos)
                x_labels.append(str(level))

                # Individual points
                plot_data.append(
                    {
                        "type": "scatter",
                        "x": [x_pos + np.random.uniform(-0.1, 0.1) for _ in vals],
                        "y": vals.tolist(),
                        "mode": "markers",
                        "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6},
                        "showlegend": False,
                    }
                )

                # Mean marker
                plot_data.append(
                    {
                        "type": "scatter",
                        "x": [x_pos],
                        "y": [vals.mean()],
                        "mode": "markers",
                        "marker": {"color": colors[i % len(colors)], "size": 12, "symbol": "diamond"},
                        "showlegend": False,
                    }
                )

                summary += f"  {level}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n"

            # Connect means
            means = [groups.get_group(lvl).mean() for lvl in df[factor].dropna().unique()]
            plot_data.append(
                {
                    "type": "scatter",
                    "x": x_positions,
                    "y": means,
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2},
                    "showlegend": False,
                }
            )

            result["plots"].append(
                {
                    "title": f"Multi-Vari: {response} by {factor}",
                    "data": plot_data,
                    "layout": {
                        "height": 300,
                        "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor},
                        "yaxis": {"title": response},
                    },
                }
            )

        else:
            # Two or more factors - nested multi-vari
            factor1, factor2 = factors[0], factors[1]
            levels1 = df[factor1].dropna().unique()
            levels2 = df[factor2].dropna().unique()

            plot_data = []
            x_positions = []
            x_labels = []
            pos = 0

            for i, lev1 in enumerate(levels1):
                group_means = []
                for j, lev2 in enumerate(levels2):
                    mask = (df[factor1] == lev1) & (df[factor2] == lev2)
                    vals = df.loc[mask, response].dropna().values

                    if len(vals) > 0:
                        x_positions.append(pos)
                        x_labels.append(f"{lev2}")

                        # Individual points
                        plot_data.append(
                            {
                                "type": "scatter",
                                "x": [pos + np.random.uniform(-0.15, 0.15) for _ in vals],
                                "y": vals.tolist(),
                                "mode": "markers",
                                "marker": {"color": colors[i % len(colors)], "size": 5, "opacity": 0.5},
                                "showlegend": False,
                            }
                        )

                        # Mean
                        group_means.append((pos, vals.mean()))
                        pos += 1

                # Connect means within factor1 level
                if group_means:
                    plot_data.append(
                        {
                            "type": "scatter",
                            "x": [g[0] for g in group_means],
                            "y": [g[1] for g in group_means],
                            "mode": "lines+markers",
                            "marker": {"color": colors[i % len(colors)], "size": 10, "symbol": "diamond"},
                            "line": {"color": colors[i % len(colors)], "width": 2},
                            "name": str(lev1),
                        }
                    )

                pos += 0.5  # Gap between factor1 levels

            result["plots"].append(
                {
                    "title": f"Multi-Vari: {response} by {factor1}/{factor2}",
                    "data": plot_data,
                    "layout": {
                        "height": 350,
                        "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor2},
                        "yaxis": {"title": response},
                        "showlegend": True,
                    },
                }
            )

            summary += f"<<COLOR:text>>Nested structure: {factor1} (colors) → {factor2} (x-axis)<</COLOR>>\n\n"

        summary += "\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += "  • Vertical spread within groups = within-group variation\n"
        summary += "  • Differences between group means = between-group variation\n"
        summary += "  • Compare spreads to identify dominant sources of variation\n"

        result["summary"] = summary

        # Variance decomposition for narrative
        _mv_all = df[response].dropna()
        _mv_total_var = float(_mv_all.var())
        if len(factors) == 1 and _mv_total_var > 0:
            _mv_grand = float(_mv_all.mean())
            _mv_grp = df.groupby(factors[0])[response]
            _mv_means = _mv_grp.mean()
            _mv_sizes = _mv_grp.count()
            _mv_ss_between = sum(float(_mv_sizes[g]) * (float(_mv_means[g]) - _mv_grand) ** 2 for g in _mv_means.index)
            _mv_ss_total = float(np.sum((_mv_all.values - _mv_grand) ** 2))
            _mv_pct = (_mv_ss_between / _mv_ss_total * 100) if _mv_ss_total > 0 else 0
            _mv_dominant = f"Between-{factors[0]}" if _mv_pct > 50 else f"Within-{factors[0]}"
            _mv_verdict = f"{_mv_dominant} variation dominates ({_mv_pct:.0f}% between-group)"
            _mv_body = f"Variation between levels of <strong>{factors[0]}</strong> accounts for {_mv_pct:.0f}% of total variation in {response}. {'Focus improvement on the between-group differences.' if _mv_pct > 50 else 'Focus improvement on reducing within-group variation (consistency).'}"
            _mv_next = (
                "Run a one-way ANOVA to formally test if the between-group difference is statistically significant."
            )
        elif len(factors) >= 2 and _mv_total_var > 0:
            _mv_grand = float(_mv_all.mean())
            _mv_f1_means = df.groupby(factors[0])[response].mean()
            _mv_f2_means = df.groupby(factors[1])[response].mean()
            _mv_f1_sizes = df.groupby(factors[0])[response].count()
            _mv_f2_sizes = df.groupby(factors[1])[response].count()
            _mv_ss1 = sum(
                float(_mv_f1_sizes[g]) * (float(_mv_f1_means[g]) - _mv_grand) ** 2 for g in _mv_f1_means.index
            )
            _mv_ss2 = sum(
                float(_mv_f2_sizes[g]) * (float(_mv_f2_means[g]) - _mv_grand) ** 2 for g in _mv_f2_means.index
            )
            _mv_ss_total = float(np.sum((_mv_all.values - _mv_grand) ** 2))
            _mv_pct1 = (_mv_ss1 / _mv_ss_total * 100) if _mv_ss_total > 0 else 0
            _mv_pct2 = (_mv_ss2 / _mv_ss_total * 100) if _mv_ss_total > 0 else 0
            _mv_dom = factors[0] if _mv_pct1 > _mv_pct2 else factors[1]
            _mv_dom_pct = max(_mv_pct1, _mv_pct2)
            _mv_residual_pct = max(0, 100 - _mv_pct1 - _mv_pct2)
            _mv_verdict = f"{_mv_dom} explains the most variation ({_mv_dom_pct:.0f}%)"
            _mv_body = (
                f"<strong>{factors[0]}</strong> accounts for {_mv_pct1:.0f}% and <strong>{factors[1]}</strong> for {_mv_pct2:.0f}% "
                f"of total variation in {response}. Residual (within-group + interaction) = {_mv_residual_pct:.0f}%. "
                f"Focus improvement on <strong>{_mv_dom}</strong>."
            )
            if _mv_residual_pct > 50:
                _mv_body += " High residual variation suggests interactions or unmeasured factors may be important."
            _mv_next = f"Run a two-way ANOVA to test if the {factors[0]} × {factors[1]} interaction is significant. If so, optimize the factors jointly."
        else:
            _mv_verdict = f"Multi-vari analysis of {response}"
            _mv_body = f"Showing variation across {len(factors)} factor(s)."
            _mv_next = None

        result["guide_observation"] = (
            f"Multi-vari chart showing variation in {response} across {len(factors)} factor(s)."
        )
        result["narrative"] = _narrative(
            _mv_verdict,
            _mv_body,
            next_steps=_mv_next,
            chart_guidance="Wider vertical spread = more within-group variation. Differences between group means (connected line) = between-group variation. The dominant source of variation is where improvement effort should focus.",
        )

    return result
