"""DSW Statistical Analysis — 64+ statistical tests and methods."""

import math

import numpy as np
from scipy import stats as sp_stats

from .common import _effect_magnitude, _practical_block, _fit_best_distribution


def run_statistical_analysis(df, analysis_id, config):
    """Run statistical analysis."""
    import numpy as np
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

        desc = df[vars_to_analyze].describe().to_string()
        result["summary"] = f"Descriptive Statistics:\n\n{desc}"

        # Add explicit statistics for Synara integration
        result["statistics"] = {}
        for var in vars_to_analyze:
            col = df[var].dropna()
            result["statistics"][f"mean({var})"] = float(col.mean())
            result["statistics"][f"std({var})"] = float(col.std())
            result["statistics"][f"min({var})"] = float(col.min())
            result["statistics"][f"max({var})"] = float(col.max())
            result["statistics"][f"median({var})"] = float(col.median())
            result["statistics"][f"n({var})"] = int(len(col))

        # Add histogram for each variable
        for var in vars_to_analyze:
            try:
                data = df[var].dropna().tolist()
                if len(data) > 0:
                    result["plots"].append({
                        "title": f"Distribution of {var}",
                        "data": [{
                            "type": "histogram",
                            "x": data,
                            "name": var,
                            "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                        }],
                        "layout": {"height": 200}
                    })
            except Exception as plot_err:
                logger.warning(f"Could not create histogram for {var}: {plot_err}")

    elif analysis_id == "ttest":
        # One-sample t-test
        var1 = config.get("var1")
        mu = float(config.get("mu", 0))
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        n = len(x)
        stat, pval = stats.ttest_1samp(x, mu)

        # Confidence interval
        se = x.std() / np.sqrt(n)
        ci = stats.t.interval(conf/100, n-1, loc=x.mean(), scale=se)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Hypothesized mean:<</COLOR>> {mu}\n\n"
        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {x.mean():.4f}\n"
        summary += f"  Std Dev: {x.std():.4f}\n"
        summary += f"  SE Mean: {se:.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
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
        summary += _practical_block("Cohen's d", d, "cohens_d", pval, alpha,
            context=f"The sample mean differs from {mu} by {abs(diff_val):.4f} units ({abs(d):.2f} standard deviations).")

        result["summary"] = summary
        obs_parts = [f"One-sample t-test: mean={x.mean():.4f} vs μ₀={mu}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"]
        if pval < alpha and meaningful:
            obs_parts.append("Practically significant — act on this difference.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)

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

        # Histogram with mean line and CI band
        result["plots"].append({
            "title": f"Distribution of {var1} with Mean & {conf}% CI",
            "data": [
                {"type": "histogram", "x": x.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": var1},
                {"type": "scatter", "x": [float(x.mean()), float(x.mean())], "y": [0, n/5], "mode": "lines", "line": {"color": "#4a90d9", "width": 2}, "name": f"Mean ({x.mean():.3f})"},
                {"type": "scatter", "x": [mu, mu], "y": [0, n/5], "mode": "lines", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}, "name": f"H₀ μ = {mu}"},
                {"type": "scatter", "x": [float(ci[0]), float(ci[1]), float(ci[1]), float(ci[0]), float(ci[0])], "y": [0, 0, n/5, n/5, 0], "fill": "toself", "fillcolor": "rgba(74, 144, 217, 0.15)", "line": {"color": "rgba(74, 144, 217, 0.3)"}, "name": f"{conf}% CI"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": var1}, "yaxis": {"title": "Count"}, "barmode": "overlay"}
        })

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
            var1, var2 = f"{response_col} [{levels[0]}]", f"{response_col} [{levels[1]}]"
        else:
            x = df[var1].dropna()
            y = df[var2].dropna()
        stat, pval = stats.ttest_ind(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TWO-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1:<</COLOR>> {var1} (n = {len(x)}, mean = {x.mean():.4f}, std = {x.std():.4f})\n"
        summary += f"<<COLOR:text>>Sample 2:<</COLOR>> {var2} (n = {len(y)}, mean = {y.mean():.4f}, std = {y.std():.4f})\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Difference of means: {x.mean() - y.mean():.4f}\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        _se_diff = np.sqrt(x.std()**2/len(x) + y.std()**2/len(y))
        _t_crit = stats.t.ppf(1 - alpha/2, len(x) + len(y) - 2)
        _ci_lo = (x.mean() - y.mean()) - _t_crit * _se_diff
        _ci_hi = (x.mean() - y.mean()) + _t_crit * _se_diff
        summary += f"  {conf}% CI for difference: [{_ci_lo:.4f}, {_ci_hi:.4f}]\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Means are significantly different (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        # Effect size: Cohen's d (pooled)
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx - 1) * x.std()**2 + (ny - 1) * y.std()**2) / (nx + ny - 2)) if (nx + ny > 2) else 1.0
        d = (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0.0
        diff_val = x.mean() - y.mean()
        label, meaningful = _effect_magnitude(d, "cohens_d")
        summary += _practical_block("Cohen's d", d, "cohens_d", pval, alpha,
            context=f"{var1} is {abs(diff_val):.4f} units {'higher' if diff_val > 0 else 'lower'} than {var2} ({abs(d):.2f} pooled SDs).")

        result["summary"] = summary
        obs_parts = [f"Two-sample t-test: diff={diff_val:.4f}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"]
        if pval < alpha and meaningful:
            obs_parts.append("Practically significant difference.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)
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

        # Side-by-side box plots
        result["plots"].append({
            "title": f"Comparison: {var1} vs {var2}",
            "data": [
                {"type": "box", "y": x.tolist(), "name": var1, "marker": {"color": "#4a9f6e"}, "boxpoints": "outliers"},
                {"type": "box", "y": y.tolist(), "name": var2, "marker": {"color": "#4a90d9"}, "boxpoints": "outliers"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "yaxis": {"title": "Value"}}
        })

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
            var1, var2 = f"{response_col} [{levels[0]}]", f"{response_col} [{levels[1]}]"
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
        summary += f"<<COLOR:title>>PAIRED T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1 (Before):<</COLOR>> {var1}\n"
        summary += f"<<COLOR:text>>Sample 2 (After):<</COLOR>> {var2}\n"
        summary += f"<<COLOR:text>>Pairs:<</COLOR>> {len(x)}\n\n"
        summary += f"<<COLOR:text>>Difference Statistics:<</COLOR>>\n"
        summary += f"  Mean difference: {diff.mean():.4f}\n"
        summary += f"  Std of differences: {diff.std():.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        _se_diff = diff.std() / np.sqrt(len(diff))
        _t_crit = stats.t.ppf(1 - alpha/2, len(diff) - 1)
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
        summary += _practical_block("Cohen's d", d, "cohens_d", pval, alpha,
            context=f"Values {direction} by {abs(diff.mean()):.4f} units on average ({abs(d):.2f} SDs of within-subject change).")

        result["summary"] = summary
        obs_parts = [f"Paired t-test: mean diff={diff.mean():.4f}, p={pval:.4f}, Cohen's d={abs(d):.3f} ({label})"]
        if pval < alpha and meaningful:
            obs_parts.append(f"Practically significant — values {direction} meaningfully.")
        elif pval < alpha:
            obs_parts.append("Statistically significant but small effect.")
        else:
            obs_parts.append("Not significant.")
        result["guide_observation"] = " ".join(obs_parts)
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

        # Histogram of differences
        result["plots"].append({
            "title": f"Distribution of Differences ({var1} − {var2})",
            "data": [
                {"type": "histogram", "x": diff.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Differences"},
                {"type": "scatter", "x": [float(diff.mean()), float(diff.mean())], "y": [0, len(x)/5], "mode": "lines", "line": {"color": "#4a90d9", "width": 2}, "name": f"Mean diff ({diff.mean():.3f})"},
                {"type": "scatter", "x": [0, 0], "y": [0, len(x)/5], "mode": "lines", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}, "name": "Zero (no diff)"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Difference"}, "yaxis": {"title": "Count"}}
        })

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
            summary += f"<<COLOR:title>>ONE-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"

            summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
            for level in df[factor_col].unique():
                grp = df[df[factor_col] == level][response].dropna()
                _ci = stats.t.interval(0.95, len(grp)-1, loc=grp.mean(), scale=grp.std()/np.sqrt(len(grp))) if len(grp) > 1 else (grp.mean(), grp.mean())
                summary += f"  {level}: n={len(grp)}, mean={grp.mean():.4f}, std={grp.std():.4f}, 95% CI [{_ci[0]:.4f}, {_ci[1]:.4f}]\n"

            # Compute eta-squared
            grand_mean = df[response].dropna().mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((df[response].dropna() - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            n_total = sum(len(g) for g in groups)
            k = len(groups)
            # Omega-squared (less biased)
            omega_sq = (ss_between - (k - 1) * (ss_total - ss_between) / (n_total - k)) / (ss_total + (ss_total - ss_between) / (n_total - k)) if (n_total > k and ss_total > 0) else 0.0
            omega_sq = max(0, omega_sq)
            eta_label, eta_meaningful = _effect_magnitude(eta_sq, "eta_squared")

            summary += f"\n<<COLOR:text>>ANOVA Results:<</COLOR>>\n"
            summary += f"  F-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n\n"

            if pval < 0.05:
                summary += f"<<COLOR:good>>Significant difference between groups (p < 0.05)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Run post-hoc tests (Tukey HSD, Games-Howell, or Dunnett) to identify which groups differ.<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>"

            summary += _practical_block("Eta-squared (η²)", eta_sq, "eta_squared", pval,
                context=f"The factor '{factor_col}' explains {eta_sq*100:.1f}% of the variation in '{response}'. Omega-squared (less biased): {omega_sq:.3f}.")

            result["summary"] = summary
            obs_parts = [f"One-way ANOVA: F={stat:.4f}, p={pval:.4f}, η²={eta_sq:.3f} ({eta_label})"]
            if pval < 0.05 and eta_meaningful:
                obs_parts.append(f"'{factor_col}' has a {eta_label}, practically significant effect on '{response}'.")
            elif pval < 0.05:
                obs_parts.append(f"Significant but {eta_label} effect.")
            else:
                obs_parts.append("Not significant.")
            result["guide_observation"] = " ".join(obs_parts)
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

            # Box plot
            result["plots"].append({
                "title": f"{response} by {factor_col}",
                "data": [{
                    "type": "box",
                    "y": df[response].tolist(),
                    "x": df[factor_col].astype(str).tolist(),
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })
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
            formula = f'{response} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>TWO-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor A:<</COLOR>> {factor_a}\n"
            summary += f"<<COLOR:highlight>>Factor B:<</COLOR>> {factor_b}\n\n"

            summary += f"<<COLOR:text>>ANOVA Table:<</COLOR>>\n"
            summary += anova_table.to_string() + "\n\n"

            # Compute partial eta-squared and interpret each factor
            ss_resid = anova_table.loc['Residual', 'sum_sq'] if 'Residual' in anova_table.index else 0
            effect_stats = {}
            for idx in anova_table.index:
                if idx == 'Residual':
                    continue
                if 'PR(>F)' in anova_table.columns:
                    p = anova_table.loc[idx, 'PR(>F)']
                    ss = anova_table.loc[idx, 'sum_sq'] if 'sum_sq' in anova_table.columns else 0
                    partial_eta = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                    eta_label, eta_meaningful = _effect_magnitude(partial_eta, "eta_squared")
                    if not np.isnan(p):
                        sig = "<<COLOR:good>>*<</COLOR>>" if p < 0.05 else ""
                        summary += f"{idx}: p = {p:.4f} {sig}  |  partial η² = {partial_eta:.3f} ({eta_label})\n"
                        effect_stats[idx] = {"p_value": float(p), "partial_eta_squared": float(partial_eta), "label": eta_label}

            # Practical significance block for strongest effect
            if effect_stats:
                strongest = max(effect_stats.items(), key=lambda x: x[1]["partial_eta_squared"])
                s_name, s_vals = strongest
                summary += _practical_block(f"Partial η² ({s_name})", s_vals["partial_eta_squared"], "eta_squared", s_vals["p_value"],
                    context=f"'{s_name}' explains {s_vals['partial_eta_squared']*100:.1f}% of the remaining variation in '{response}'.")

            result["summary"] = summary
            result["guide_observation"] = "Two-way ANOVA: " + "; ".join(
                f"{k}: p={v['p_value']:.4f}, η²={v['partial_eta_squared']:.3f} ({v['label']})" for k, v in effect_stats.items()
            )
            result["statistics"] = {"effects": effect_stats}

            # Interaction plot
            means = df.groupby([factor_a, factor_b])[response].mean().unstack()
            traces = []
            for col in means.columns:
                traces.append({
                    "type": "scatter",
                    "x": means.index.astype(str).tolist(),
                    "y": means[col].tolist(),
                    "mode": "lines+markers",
                    "name": str(col)
                })

            result["plots"].append({
                "title": "Interaction Plot",
                "data": traces,
                "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": factor_a}}
            })

        except ImportError:
            result["summary"] = "Two-way ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Two-way ANOVA error: {str(e)}"

    elif analysis_id == "regression":
        response = config.get("response")
        predictors = config.get("predictors", [])
        degree = int(config.get("degree", 1))
        interactions = config.get("interactions", "none")

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error

        X_raw = df[predictors].dropna()
        y = df[response].loc[X_raw.index]
        n = len(y)

        # Build feature names and transform data
        feature_names = list(predictors)

        if degree > 1 or interactions == "all":
            # Use PolynomialFeatures for polynomial and/or interaction terms
            include_interaction = interactions == "all"
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=(degree == 1 and include_interaction)
            )
            X = poly.fit_transform(X_raw)

            # Generate readable feature names
            feature_names = []
            for powers in poly.powers_:
                parts = []
                for i, power in enumerate(powers):
                    if power == 0:
                        continue
                    elif power == 1:
                        parts.append(predictors[i])
                    else:
                        parts.append(f"{predictors[i]}^{power}")
                if parts:
                    feature_names.append("·".join(parts) if len(parts) > 1 else parts[0])
        else:
            X = X_raw.values

        p = X.shape[1]  # Number of features after transformation

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate comprehensive statistics
        residuals = y.values - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.values - np.mean(y.values))**2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = ss_res / (n - p - 1) if n > p + 1 else 1e-10
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate standard errors and t-statistics
        X_with_const = np.column_stack([np.ones(n), X])
        try:
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(var_coef)
            coefs = np.concatenate([[model.intercept_], model.coef_])
            t_stats = coefs / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        except:
            se = np.zeros(p + 1)
            t_stats = np.zeros(p + 1)
            p_values = np.ones(p + 1)
            coefs = np.concatenate([[model.intercept_], model.coef_])

        # F-statistic
        f_stat = (ss_tot - ss_res) / p / mse if p > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1) if p > 0 else 1

        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals)**2) / ss_res

        # Build colored summary output
        model_type = "Linear" if degree == 1 else "Quadratic" if degree == 2 else "Cubic"
        if interactions == "all":
            model_type += " + Interactions"

        summary = "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                          {model_type.upper()} REGRESSION RESULTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n\n"

        summary += f"<<COLOR:dim>>Dep. Variable:<</COLOR>>    <<COLOR:text>>{response}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Observations:<</COLOR>> <<COLOR:text>>{n}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Features:<</COLOR>>     <<COLOR:text>>{p}<</COLOR>> (from {len(predictors)} predictors)\n"
        summary += f"<<COLOR:dim>>Model:<</COLOR>>            <<COLOR:text>>OLS - {model_type}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                               MODEL FIT<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        r2_color = "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger"
        summary += f"<<COLOR:dim>>Residual std. error:<</COLOR>> <<COLOR:text>>{np.sqrt(mse):.4f}<</COLOR>> on <<COLOR:text>>{n - p - 1}<</COLOR>> degrees of freedom\n"
        summary += f"<<COLOR:dim>>Multiple R-squared:<</COLOR>>  <<COLOR:{r2_color}>>{r2:.4f}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Adjusted R-squared:<</COLOR>> <<COLOR:{r2_color}>>{adj_r2:.4f}<</COLOR>>\n"

        f_color = "success" if f_pvalue < 0.05 else "warning" if f_pvalue < 0.1 else "danger"
        summary += f"<<COLOR:dim>>F-statistic:<</COLOR>>        <<COLOR:text>>{f_stat:.2f}<</COLOR>> on <<COLOR:text>>{p}<</COLOR>> and <<COLOR:text>>{n - p - 1}<</COLOR>> DF\n"
        summary += f"<<COLOR:dim>>p-value:<</COLOR>>            <<COLOR:{f_color}>>{f_pvalue:.4e}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Durbin-Watson:<</COLOR>>      <<COLOR:text>>{dw:.3f}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += "<<COLOR:accent>>                              COEFFICIENTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        _t_crit_reg = stats.t.ppf(0.975, n - p - 1)
        summary += "<<COLOR:dim>>                            Estimate    Std.Err    t value    Pr(>|t|)          [95% CI]<</COLOR>>\n"

        names = ["(Intercept)"] + feature_names
        non_sig_predictors = []
        for i, name in enumerate(names):
            pv = p_values[i]
            sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else ".  " if pv < 0.1 else "   "
            p_color = "success" if pv < 0.05 else "warning" if pv < 0.1 else "dim"
            _ci_lo = coefs[i] - _t_crit_reg * se[i]
            _ci_hi = coefs[i] + _t_crit_reg * se[i]
            summary += f"<<COLOR:text>>{name:<24}<</COLOR>> {coefs[i]:>10.4f}   {se[i]:>9.4f}   {t_stats[i]:>8.3f}    <<COLOR:{p_color}>>{pv:>9.4f}  {sig}<</COLOR>>  [{_ci_lo:.4f}, {_ci_hi:.4f}]\n"
            if i > 0 and pv >= 0.1:  # Track non-significant predictors (excluding intercept)
                non_sig_predictors.append(name)

        summary += "<<COLOR:dim>>---<</COLOR>>\n"
        summary += "<<COLOR:dim>>Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1<</COLOR>>\n\n"

        # Diagnostics interpretation
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                            DIAGNOSTICS SUMMARY<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        # Interpret R²
        if r2 > 0.7:
            summary += "<<COLOR:success>>✓ Good fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        elif r2 > 0.4:
            summary += "<<COLOR:warning>>◐ Moderate fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        else:
            summary += "<<COLOR:danger>>✗ Poor fit:<</COLOR>> Model explains only {:.1f}% of variance\n".format(r2*100)

        # Interpret Durbin-Watson
        if 1.5 < dw < 2.5:
            summary += "<<COLOR:success>>✓ No autocorrelation:<</COLOR>> Durbin-Watson ≈ 2\n"
        else:
            summary += "<<COLOR:warning>>◐ Possible autocorrelation:<</COLOR>> Durbin-Watson = {:.2f}\n".format(dw)

        # Interpret F-test
        if f_pvalue < 0.05:
            summary += "<<COLOR:success>>✓ Model significant:<</COLOR>> F-test p < 0.05\n"
        else:
            summary += "<<COLOR:danger>>✗ Model not significant:<</COLOR>> F-test p = {:.3f}\n".format(f_pvalue)

        # Calculate diagnostic values first (needed for suggestions and plots)
        std_residuals = residuals / np.std(residuals)

        # Leverage (hat values)
        try:
            H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
            leverage = np.diag(H)
        except:
            leverage = np.ones(n) / n

        # Cook's distance
        cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage + 1e-10)**2)

        # Square root of standardized residuals for scale-location
        sqrt_std_resid = np.sqrt(np.abs(std_residuals))

        # Model improvement suggestions
        suggestions = []

        # Low R² suggestions
        if r2 < 0.4:
            suggestions.append("Add more predictors or interaction terms (X1*X2)")
            suggestions.append("Try polynomial terms (X², X³) for non-linear relationships")
            suggestions.append("Check for outliers that may be distorting the fit")
        elif r2 < 0.7:
            suggestions.append("Consider adding interaction terms or polynomial features")

        # Non-significant predictors
        if non_sig_predictors:
            if len(non_sig_predictors) <= 3:
                suggestions.append(f"Consider removing non-significant: {', '.join(non_sig_predictors)}")
            else:
                suggestions.append(f"Consider removing {len(non_sig_predictors)} non-significant predictors")

        # Autocorrelation
        if dw < 1.5:
            suggestions.append("Positive autocorrelation detected - consider time series methods or lag terms")
        elif dw > 2.5:
            suggestions.append("Negative autocorrelation detected - check data ordering")

        # Model not significant
        if f_pvalue >= 0.05:
            suggestions.append("Model not significant - try different predictors or check data quality")

        # High leverage points
        high_leverage = int(np.sum(leverage > 2 * (p + 1) / n)) if n > 0 else 0
        if high_leverage > 0:
            suggestions.append(f"{high_leverage} high-leverage points detected - check for influential outliers")

        # Large Cook's distance
        high_cooks = int(np.sum(cooks_d > 4 / n)) if n > 0 else 0
        if high_cooks > 0:
            suggestions.append(f"{high_cooks} influential observations (Cook's D) - consider robust regression")

        if suggestions:
            summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += "<<COLOR:accent>>                          IMPROVEMENT SUGGESTIONS<</COLOR>>\n"
            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            for i, sug in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                summary += f"<<COLOR:warning>>{i}.<</COLOR>> <<COLOR:text>>{sug}<</COLOR>>\n"

        # Practical significance: R² as effect size
        r2_label, r2_meaningful = _effect_magnitude(r2, "r_squared")
        summary += _practical_block("R²", r2, "r_squared", f_pvalue,
            context=f"The model explains {r2*100:.1f}% of the variation in '{response}'. "
                    f"RMSE = {rmse:.4f} — on average, predictions are off by this amount in the original units.")

        result["summary"] = summary
        # Build guide_observation for regression
        sig_predictors = [names[i] for i in range(1, len(names)) if p_values[i] < 0.05]
        obs = f"Regression: R²={r2:.3f} ({r2_label}), F p={f_pvalue:.4f}."
        if sig_predictors:
            obs += f" Significant predictors: {', '.join(sig_predictors[:5])}."
        if r2_meaningful:
            obs += f" Model explains {r2*100:.0f}% of variation — practically useful."
        else:
            obs += f" Model explains only {r2*100:.0f}% of variation — limited practical use."
        result["guide_observation"] = obs

        # What-If data for client-side interactive predictor explorer (degree 1 only)
        if degree == 1 and interactions == "none":
            result["what_if_data"] = {
                "type": "regression",
                "intercept": float(model.intercept_),
                "coefficients": {feat: float(c) for feat, c in zip(predictors, model.coef_)},
                "residual_std": float(np.sqrt(mse)) if mse > 0 else 0.0,
                "n": int(n),
                "feature_ranges": {
                    feat: {
                        "min": float(X_raw[feat].min()), "max": float(X_raw[feat].max()),
                        "mean": float(X_raw[feat].mean()), "std": float(X_raw[feat].std()),
                    }
                    for feat in predictors
                },
                "response_name": response,
            }

        # Create 4-panel diagnostic plots

        # 1. Residuals vs Fitted
        result["plots"].append({
            "title": "1. Residuals vs Fitted",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": residuals.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [min(y_pred), max(y_pred)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Zero line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "Residuals"}}
        })

        # 2. Normal Q-Q Plot
        sorted_std_resid = np.sort(std_residuals)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_std_resid)))
        result["plots"].append({
            "title": "2. Normal Q-Q",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [-3, 3],
                    "y": [-3, 3],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Normal line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # 3. Scale-Location
        result["plots"].append({
            "title": "3. Scale-Location",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": sqrt_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "√|Std. Residuals|"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "√|Standardized residuals|"}}
        })

        # 4. Residuals vs Leverage
        result["plots"].append({
            "title": "4. Residuals vs Leverage",
            "data": [
                {
                    "type": "scatter",
                    "x": leverage.tolist(),
                    "y": std_residuals.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": cooks_d.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [0.5, "#e89547"], [1, "#9f4a4a"]],
                        "size": 6,
                        "colorbar": {"title": "Cook's D", "len": 0.5}
                    },
                    "name": "Observations"
                },
                {
                    "type": "scatter",
                    "x": [0, max(leverage)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "showlegend": False
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Leverage"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # Explicit statistics for Synara integration
        result["statistics"] = {
            "R²": float(r2),
            "Adj_R²": float(adj_r2),
            "RMSE": float(rmse),
            "F_statistic": float(f_stat),
            "F_p_value": float(f_pvalue),
            "n": int(n),
            "predictors": int(p),
        }
        # Add coefficients as statistics
        for i, feat in enumerate(feature_names):
            result["statistics"][f"coef({feat})"] = float(model.coef_[i])
            if i < len(p_values) - 1:
                result["statistics"][f"p_value({feat})"] = float(p_values[i + 1])

    elif analysis_id == "correlation":
        vars_list = config.get("vars", [])
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
        summary += "<<COLOR:text>>Correlation Matrix:<</COLOR>>\n"
        summary += corr_matrix.to_string() + "\n"

        # Find and report strongest correlations with practical interpretation
        n_obs = len(df[numeric_cols].dropna())
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
                        strong_pairs.append((col1, col2, float(r_val), float(p_val), label, len(pair_data)))

        if strong_pairs:
            strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            summary += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>KEY RELATIONSHIPS<</COLOR>>\n"
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
            summary += f"\n<<COLOR:text>>Strongest: {strong_pairs[0][0]} and {strong_pairs[0][1]} share {strong_pairs[0][2]**2*100:.0f}% of their variation.<</COLOR>>"
        else:
            summary += f"\n<<COLOR:text>>No strong correlations found (all |r| < 0.3).<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Correlation ({method}): {len(strong_pairs)} pairs with |r| ≥ 0.3." + (
            f" Strongest: {strong_pairs[0][0]} ↔ {strong_pairs[0][1]} (r={strong_pairs[0][2]:.3f})." if strong_pairs else " No strong relationships found.")
        result["statistics"] = stat_dict

        # Heatmap
        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{
                "type": "heatmap",
                "z": corr_matrix.values.tolist(),
                "x": numeric_cols,
                "y": numeric_cols,
                "colorscale": "RdBu",
                "zmid": 0,
                "text": [[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                "texttemplate": "%{text}",
                "textfont": {"size": 10}
            }],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "normality":
        var = config.get("var")
        test_type = config.get("test", "anderson")

        x = df[var].dropna()
        n = len(x)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NORMALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n\n"

        if test_type == "anderson":
            stat_result = stats.anderson(x)
            summary += f"<<COLOR:text>>Anderson-Darling Test<</COLOR>>\n"
            summary += f"  Statistic: {stat_result.statistic:.4f}\n"
            summary += f"  Critical Values:\n"
            for cv, sl in zip(stat_result.critical_values, stat_result.significance_level):
                marker = "<<COLOR:good>>✓<</COLOR>>" if stat_result.statistic < cv else "<<COLOR:bad>>✗<</COLOR>>"
                summary += f"    {marker} {sl}%: {cv:.4f}\n"
        elif test_type == "shapiro":
            stat, pval = stats.shapiro(x)
            summary += f"<<COLOR:text>>Shapiro-Wilk Test<</COLOR>>\n"
            summary += f"  W-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"
        elif test_type == "ks":
            stat, pval = stats.kstest(x, 'norm', args=(x.mean(), x.std()))
            summary += f"<<COLOR:text>>Kolmogorov-Smirnov Test<</COLOR>>\n"
            summary += f"  D-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Q-Q plot
        sorted_data = np.sort(x)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(x)))

        result["plots"].append({
            "title": f"Normal Q-Q Plot: {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_quantiles.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": [theoretical_quantiles.min(), theoretical_quantiles.max()],
                    "y": [sorted_data.min(), sorted_data.max()],
                    "mode": "lines",
                    "line": {"color": "#ff7675", "dash": "dash"},
                    "name": "Reference"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Sample Quantiles"}
            }
        })

        # Histogram with normal curve overlay
        x_range = np.linspace(float(x.min()), float(x.max()), 100)
        normal_pdf = stats.norm.pdf(x_range, x.mean(), x.std())
        bin_width = (x.max() - x.min()) / min(30, max(5, int(np.sqrt(n))))
        normal_scaled = normal_pdf * n * bin_width

        result["plots"].append({
            "title": f"Histogram with Normal Curve: {var}",
            "data": [
                {"type": "histogram", "x": x.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
                {"type": "scatter", "x": x_range.tolist(), "y": normal_scaled.tolist(), "mode": "lines", "line": {"color": "#d94a4a", "width": 2}, "name": "Normal fit"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": var}, "yaxis": {"title": "Count"}, "barmode": "overlay"}
        })

    elif analysis_id == "chi2":
        row_var = config.get("row_var") or config.get("var1") or config.get("var")
        col_var = config.get("col_var") or config.get("var2") or config.get("group_var")

        # Create contingency table
        contingency = pd.crosstab(df[row_var], df[col_var])
        chi2, pval, dof, expected = stats.chi2_contingency(contingency)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHI-SQUARE TEST FOR INDEPENDENCE<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Row Variable:<</COLOR>> {row_var}\n"
        summary += f"<<COLOR:highlight>>Column Variable:<</COLOR>> {col_var}\n\n"

        summary += f"<<COLOR:text>>Contingency Table (Observed):<</COLOR>>\n"
        summary += contingency.to_string() + "\n\n"

        # Cramér's V effect size
        n_obs = contingency.values.sum()
        min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n_obs * min_dim)) if (n_obs > 0 and min_dim > 0) else 0.0
        v_label, v_meaningful = _effect_magnitude(cramers_v, "cramers_v")

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Chi-square statistic: {chi2:.4f}\n"
        summary += f"  Degrees of freedom: {dof}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  Cramér's V: {cramers_v:.3f} ({v_label} association)\n"
        if contingency.shape == (2, 2):
            _a, _b, _c, _d = contingency.iloc[0, 0], contingency.iloc[0, 1], contingency.iloc[1, 0], contingency.iloc[1, 1]
            if min(_a, _b, _c, _d) > 0:
                _or = (_a * _d) / (_b * _c)
                _log_se = np.sqrt(1/_a + 1/_b + 1/_c + 1/_d)
                _or_lo, _or_hi = np.exp(np.log(_or) - 1.96 * _log_se), np.exp(np.log(_or) + 1.96 * _log_se)
                summary += f"  Odds Ratio: {_or:.3f}, 95% CI [{_or_lo:.3f}, {_or_hi:.3f}]\n"
        summary += "\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>Variables are significantly associated (p < 0.05)<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant association found (p >= 0.05)<</COLOR>>"

        summary += _practical_block("Cramér's V", cramers_v, "cramers_v", pval,
            context=f"The association between '{row_var}' and '{col_var}' is {v_label}. V=0 means no association, V=1 means perfect association.")

        result["summary"] = summary
        obs_parts = [f"Chi-square test: χ²={chi2:.4f}, p={pval:.4f}, Cramér's V={cramers_v:.3f} ({v_label})"]
        if pval < 0.05 and v_meaningful:
            obs_parts.append(f"'{row_var}' and '{col_var}' are meaningfully associated.")
        elif pval < 0.05:
            obs_parts.append("Significant but weak association.")
        else:
            obs_parts.append("No significant association.")
        result["guide_observation"] = " ".join(obs_parts)
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

        # Heatmap of observed counts
        result["plots"].append({
            "title": f"Contingency Table: {row_var} × {col_var}",
            "data": [{
                "type": "heatmap",
                "z": contingency.values.tolist(),
                "x": contingency.columns.astype(str).tolist(),
                "y": contingency.index.astype(str).tolist(),
                "colorscale": "Blues",
                "text": contingency.values.tolist(),
                "texttemplate": "%{text}",
                "textfont": {"size": 12}
            }],
            "layout": {"template": "plotly_dark", "height": 300}
        })

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
            x = int((col == 1).sum()) if col.dtype in ['int64', 'float64'] else int(col.value_counts().iloc[0])
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
        summary += f"<<COLOR:title>>ONE-PROPORTION Z-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        if event is not None and str(event) != "":
            summary += f"<<COLOR:highlight>>Event:<</COLOR>> {event}\n"
        summary += f"<<COLOR:highlight>>H₀:<</COLOR>> p = {p0}\n"
        summary += f"<<COLOR:highlight>>H₁:<</COLOR>> p {'≠' if alt == 'two-sided' else '>' if alt == 'greater' else '<'} {p0}\n\n"
        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  N: {n}\n"
        summary += f"  Successes: {x}\n"
        summary += f"  p̂: {p_hat:.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Z-statistic: {z_stat:.4f}\n"
        summary += f"  p-value: {p_val:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI (Wilson): ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Proportion differs significantly from {p0} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference from {p0} (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Proportion bar with CI and reference line
        result["plots"].append({
            "data": [
                {"type": "bar", "x": ["Observed"], "y": [p_hat], "marker": {"color": "#4a9f6e"},
                 "error_y": {"type": "data", "symmetric": False, "array": [ci_hi - p_hat], "arrayminus": [p_hat - ci_lo], "color": "#5a6a5a"},
                 "name": f"p̂ = {p_hat:.4f}"}
            ],
            "layout": {
                "title": "Observed Proportion vs Hypothesized",
                "yaxis": {"title": "Proportion", "range": [0, min(1.05, max(ci_hi + 0.1, p0 + 0.2))]},
                "shapes": [{"type": "line", "x0": -0.5, "x1": 0.5, "y0": p0, "y1": p0,
                            "line": {"color": "#e89547", "dash": "dash", "width": 2}}],
                "annotations": [{"x": 0.5, "y": p0, "text": f"H₀: p={p0}", "showarrow": False, "xanchor": "left", "font": {"color": "#e89547"}}],
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"1-prop Z-test: p̂={p_hat:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "n": n, "successes": x, "p_hat": p_hat, "p0": p0,
            "z_statistic": float(z_stat), "p_value": p_val,
            "ci_lower": ci_lo, "ci_upper": ci_hi, "alternative": alt
        }

    elif analysis_id == "prop_2sample":
        """
        Two-Proportion Z-Test — compare proportions between two groups.
        Tests H₀: p₁ = p₂. Reports pooled Z, individual CIs, and difference CI.
        """
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
            x1 = int((g1 == 1).sum()) if g1.dtype in ['int64', 'float64'] else int(g1.value_counts().iloc[0])
            x2 = int((g2 == 1).sum()) if g2.dtype in ['int64', 'float64'] else int(g2.value_counts().iloc[0])

        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0
        p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) if (n1 > 0 and n2 > 0) else 1
        z_stat = (p1 - p2) / se_pooled if se_pooled > 0 else 0

        if alt == "greater":
            p_val = float(1 - stats.norm.cdf(z_stat))
        elif alt == "less":
            p_val = float(stats.norm.cdf(z_stat))
        else:
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Difference CI (unpooled SE)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) if (n1 > 0 and n2 > 0) else 0
        diff = p1 - p2
        ci_lo = diff - z_crit * se_diff
        ci_hi = diff + z_crit * se_diff

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TWO-PROPORTION Z-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {group_var}\n\n"

        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  {'Group':<15} {'N':>6} {'Events':>8} {'Proportion':>12}\n"
        summary += f"  {'─' * 45}\n"
        summary += f"  {str(groups[0]):<15} {n1:>6} {x1:>8} {p1:>12.4f}\n"
        summary += f"  {str(groups[1]):<15} {n2:>6} {x2:>8} {p2:>12.4f}\n\n"
        summary += f"<<COLOR:text>>Difference (p₁ − p₂):<</COLOR>> {diff:.4f}\n"
        summary += f"<<COLOR:text>>{100*(1-alpha):.0f}% CI for difference:<</COLOR>> ({ci_lo:.4f}, {ci_hi:.4f})\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Z-statistic: {z_stat:.4f}\n"
        summary += f"  p-value: {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Proportions differ significantly (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in proportions (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Side-by-side bar chart
        result["plots"].append({
            "data": [
                {"type": "bar", "x": [str(groups[0]), str(groups[1])], "y": [p1, p2],
                 "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                 "text": [f"{p1:.3f}", f"{p2:.3f}"], "textposition": "outside"}
            ],
            "layout": {
                "title": "Proportions by Group",
                "yaxis": {"title": "Proportion", "range": [0, max(p1, p2) * 1.3 + 0.05]},
                "template": "plotly_white"
            }
        })

        # Difference CI plot
        result["plots"].append({
            "data": [{
                "type": "scatter", "x": [diff], "y": ["p₁ − p₂"], "mode": "markers",
                "marker": {"size": 12, "color": "#4a9f6e"},
                "error_x": {"type": "data", "symmetric": False, "array": [ci_hi - diff], "arrayminus": [diff - ci_lo], "color": "#5a6a5a"}
            }],
            "layout": {
                "title": f"Difference in Proportions ({100*(1-alpha):.0f}% CI)",
                "xaxis": {"title": "p₁ − p₂", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": 0.5, "line": {"color": "#e89547", "dash": "dash"}}],
                "height": 200, "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"2-prop Z-test: p₁={p1:.4f} vs p₂={p2:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "n1": n1, "n2": n2, "x1": x1, "x2": x2, "p1": p1, "p2": p2,
            "difference": diff, "z_statistic": float(z_stat), "p_value": p_val,
            "ci_lower": ci_lo, "ci_upper": ci_hi, "alternative": alt
        }

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
            result["summary"] = f"Fisher's exact test requires a 2×2 table. Got {ct.shape[0]}×{ct.shape[1]}. Ensure both variables have exactly 2 levels."
            return result

        table = ct.values
        odds_ratio, p_val = stats.fisher_exact(table, alternative=alt)

        # Odds ratio CI via log method
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        if all(v > 0 for v in [a, b, c, d]):
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
            z_crit = stats.norm.ppf(1 - alpha / 2)
            or_ci_lo = np.exp(log_or - z_crit * se_log_or)
            or_ci_hi = np.exp(log_or + z_crit * se_log_or)
        else:
            or_ci_lo, or_ci_hi = 0, float('inf')

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>FISHER'S EXACT TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Row variable:<</COLOR>> {var1}\n"
        summary += f"<<COLOR:highlight>>Column variable:<</COLOR>> {var2}\n\n"

        summary += f"<<COLOR:text>>2×2 Contingency Table:<</COLOR>>\n"
        summary += f"  {'':>15} {str(ct.columns[0]):>10} {str(ct.columns[1]):>10}\n"
        summary += f"  {str(ct.index[0]):>15} {a:>10} {b:>10}\n"
        summary += f"  {str(ct.index[1]):>15} {c:>10} {d:>10}\n\n"
        summary += f"<<COLOR:text>>Results:<</COLOR>>\n"
        summary += f"  Odds Ratio: {odds_ratio:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI: ({or_ci_lo:.4f}, {or_ci_hi:.4f})\n"
        summary += f"  p-value (exact): {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Significant association (p < {alpha}). Odds ratio ≠ 1.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant association (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Mosaic-like stacked bar
        col_labels = [str(c) for c in ct.columns]
        row_labels = [str(r) for r in ct.index]
        result["plots"].append({
            "data": [
                {"type": "bar", "x": col_labels, "y": [int(table[0, 0]), int(table[0, 1])], "name": row_labels[0], "marker": {"color": "#4a9f6e"}},
                {"type": "bar", "x": col_labels, "y": [int(table[1, 0]), int(table[1, 1])], "name": row_labels[1], "marker": {"color": "#4a90d9"}}
            ],
            "layout": {"title": "Contingency Table", "barmode": "stack", "yaxis": {"title": "Count"}, "template": "plotly_white"}
        })

        # Odds ratio forest-style plot
        if odds_ratio > 0 and or_ci_hi < 1e6:
            result["plots"].append({
                "data": [{
                    "type": "scatter", "x": [odds_ratio], "y": ["OR"], "mode": "markers",
                    "marker": {"size": 12, "color": "#4a9f6e"},
                    "error_x": {"type": "data", "symmetric": False, "array": [or_ci_hi - odds_ratio], "arrayminus": [odds_ratio - or_ci_lo], "color": "#5a6a5a"}
                }],
                "layout": {
                    "title": f"Odds Ratio ({100*(1-alpha):.0f}% CI)",
                    "xaxis": {"title": "Odds Ratio", "type": "log"},
                    "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": 0.5, "line": {"color": "#e89547", "dash": "dash"}}],
                    "height": 180, "template": "plotly_white"
                }
            })

        result["guide_observation"] = f"Fisher's exact: OR={odds_ratio:.3f}, p={p_val:.4f}. " + ("Significant association." if p_val < alpha else "No association.")
        result["statistics"] = {
            "odds_ratio": float(odds_ratio), "p_value": float(p_val),
            "or_ci_lower": float(or_ci_lo), "or_ci_upper": float(or_ci_hi),
            "table": table.tolist(), "alternative": alt
        }

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
            ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (total_count + 1)) / (2 * exposure)
        else:
            ci_lo = 0
            ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * exposure)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-SAMPLE POISSON RATE TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>H₀:<</COLOR>> rate = {rate0}\n"
        summary += f"<<COLOR:highlight>>Exposure:<</COLOR>> {exposure}\n\n"
        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  Total count: {total_count:.0f}\n"
        summary += f"  Observed rate: {observed_rate:.4f}\n"
        summary += f"  Expected count (under H₀): {expected_count:.1f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  p-value (exact): {p_val:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI for rate: ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Rate differs significantly from {rate0} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference from {rate0} (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        result["plots"].append({
            "data": [
                {"type": "bar", "x": ["Observed"], "y": [observed_rate], "marker": {"color": "#4a9f6e"},
                 "error_y": {"type": "data", "symmetric": False, "array": [ci_hi - observed_rate], "arrayminus": [observed_rate - ci_lo], "color": "#5a6a5a"},
                 "name": f"Rate = {observed_rate:.4f}"}
            ],
            "layout": {
                "title": "Observed Rate vs Hypothesized",
                "yaxis": {"title": "Rate"},
                "shapes": [{"type": "line", "x0": -0.5, "x1": 0.5, "y0": rate0, "y1": rate0,
                            "line": {"color": "#e89547", "dash": "dash", "width": 2}}],
                "annotations": [{"x": 0.5, "y": rate0, "text": f"H₀: λ={rate0}", "showarrow": False, "xanchor": "left", "font": {"color": "#e89547"}}],
                "template": "plotly_white"
            }
        })

        # Distribution plot
        x_range = list(range(max(0, int(total_count) - 15), int(total_count) + 16))
        pmf_vals = [float(stats.poisson.pmf(k, expected_count)) for k in x_range]
        result["plots"].append({
            "data": [
                {"type": "bar", "x": x_range, "y": pmf_vals, "name": f"Poisson(λ={expected_count:.1f})",
                 "marker": {"color": ["#d94a4a" if k == int(total_count) else "#4a9f6e" for k in x_range], "opacity": 0.7}}
            ],
            "layout": {
                "title": f"Poisson Distribution under H₀ (observed = {int(total_count)})",
                "xaxis": {"title": "Count"}, "yaxis": {"title": "Probability"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Poisson rate test: observed rate={observed_rate:.4f}, H₀ rate={rate0}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "total_count": total_count, "exposure": exposure,
            "observed_rate": observed_rate, "hypothesized_rate": rate0,
            "p_value": p_val, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "alternative": alt
        }

    elif analysis_id == "variance_test":
        """
        Variance Test — compare variability.
        Modes:
          1) Single column + sigma0 → chi-square test for σ² = σ₀²
          2) Two columns (wide) → F-test + Levene's
          3) Response + grouping factor → Bartlett's + Levene's for 2+ groups
        Always runs both Bartlett's (assumes normality) and Levene's (robust) so
        the user doesn't have to check normality first.
        """
        var1 = config.get("var1") or config.get("var")
        var2 = config.get("var2")
        sigma0 = config.get("sigma0")
        alpha = 1 - float(config.get("conf", 95)) / 100
        conf_pct = float(config.get("conf", 95))

        # Detect mode
        if config.get("data_format") == "factor" or config.get("group_var"):
            # Mode 3: response + grouping factor
            response_col = config.get("response") or var1
            factor_col = config.get("group_var") or config.get("factor") or var2
            data_clean = df[[response_col, factor_col]].dropna()
            groups_labels = sorted(data_clean[factor_col].unique().tolist(), key=str)
            groups_data = [data_clean[data_clean[factor_col] == g][response_col].values for g in groups_labels]
            mode = "factor"
        elif var2 and var2 != var1:
            # Mode 2: two separate columns
            x = df[var1].dropna().values
            y = df[var2].dropna().values
            groups_labels = [str(var1), str(var2)]
            groups_data = [x, y]
            mode = "two_col"
        elif sigma0 is not None:
            # Mode 1: one-sample chi-square
            x = df[var1].dropna().values
            mode = "one_sample"
        else:
            result["summary"] = "Please provide either two columns, a response + grouping factor, or a single column with a hypothesized sigma (sigma0)."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"

        if mode == "one_sample":
            sigma0 = float(sigma0)
            n = len(x)
            s2 = float(np.var(x, ddof=1))
            s = float(np.sqrt(s2))
            chi2_stat = (n - 1) * s2 / (sigma0 ** 2) if sigma0 > 0 else 0

            p_val = float(2 * min(
                stats.chi2.cdf(chi2_stat, n - 1),
                1 - stats.chi2.cdf(chi2_stat, n - 1),
            ))

            ci_lo_var = (n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, n - 1)
            ci_hi_var = (n - 1) * s2 / stats.chi2.ppf(alpha / 2, n - 1)

            summary += f"<<COLOR:title>>ONE-SAMPLE VARIANCE TEST (Chi-Square)<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1}\n"
            summary += f"<<COLOR:highlight>>H₀:<</COLOR>> σ = {sigma0}\n\n"
            summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
            summary += f"  N: {n}\n"
            summary += f"  Sample std dev: {s:.4f}\n"
            summary += f"  Sample variance: {s2:.4f}\n\n"
            summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
            summary += f"  Chi-square statistic: {chi2_stat:.4f}\n"
            summary += f"  df: {n - 1}\n"
            summary += f"  p-value: {p_val:.4f}\n"
            summary += f"  {conf_pct:.0f}% CI for σ²: ({ci_lo_var:.4f}, {ci_hi_var:.4f})\n"
            summary += f"  {conf_pct:.0f}% CI for σ:  ({np.sqrt(ci_lo_var):.4f}, {np.sqrt(ci_hi_var):.4f})\n\n"

            if p_val < alpha:
                summary += f"<<COLOR:good>>Variance differs significantly from {sigma0}² = {sigma0**2:.4f} (p < {alpha})<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference from σ₀ = {sigma0} (p ≥ {alpha})<</COLOR>>"

            result["statistics"] = {
                "n": n, "sample_std": s, "sample_variance": s2,
                "chi2_statistic": chi2_stat, "df": n - 1, "p_value": p_val,
                "ci_variance_lower": ci_lo_var, "ci_variance_upper": ci_hi_var,
            }
            result["guide_observation"] = f"One-sample variance test: s={s:.4f} vs σ₀={sigma0}, χ²={chi2_stat:.3f}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")

        else:
            # Multi-group: Bartlett's + Levene's (always both)
            k = len(groups_data)
            ns = [len(g) for g in groups_data]
            stds = [float(np.std(g, ddof=1)) for g in groups_data]
            variances = [float(np.var(g, ddof=1)) for g in groups_data]

            bart_stat, bart_p = stats.bartlett(*groups_data)
            lev_stat, lev_p = stats.levene(*groups_data, center="median")

            summary += f"<<COLOR:title>>TEST FOR EQUAL VARIANCES<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

            if mode == "factor":
                summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response_col}\n"
                summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"
            else:
                summary += f"<<COLOR:highlight>>Columns:<</COLOR>> {var1}, {var2}\n\n"

            summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
            summary += f"  {'Group':<20} {'N':>6} {'StDev':>10} {'Variance':>12}\n"
            summary += f"  {'─' * 50}\n"
            for lbl, n_i, s_i, v_i in zip(groups_labels, ns, stds, variances):
                summary += f"  {str(lbl):<20} {n_i:>6} {s_i:>10.4f} {v_i:>12.4f}\n"

            summary += f"\n<<COLOR:text>>Test Results:<</COLOR>>\n"
            summary += f"  {'Test':<25} {'Statistic':>12} {'p-value':>10}\n"
            summary += f"  {'─' * 50}\n"
            summary += f"  {'Bartlett (normal data)':<25} {bart_stat:>12.4f} {bart_p:>10.4f}\n"
            summary += f"  {'Levene (robust)':<25} {lev_stat:>12.4f} {lev_p:>10.4f}\n"

            # F-test only for exactly 2 groups
            f_stat, f_p = None, None
            if k == 2:
                f_stat = variances[0] / variances[1] if variances[1] > 0 else float("inf")
                df1, df2 = ns[0] - 1, ns[1] - 1
                f_p = float(2 * min(
                    stats.f.cdf(f_stat, df1, df2),
                    1 - stats.f.cdf(f_stat, df1, df2),
                ))
                summary += f"  {'F-test (2-sample)':<25} {f_stat:>12.4f} {f_p:>10.4f}\n"

                # CI for variance ratio
                f_lo = stats.f.ppf(alpha / 2, df1, df2)
                f_hi = stats.f.ppf(1 - alpha / 2, df1, df2)
                ratio_lo = f_stat / f_hi
                ratio_hi = f_stat / f_lo
                summary += f"\n  {conf_pct:.0f}% CI for σ₁²/σ₂²: ({ratio_lo:.4f}, {ratio_hi:.4f})\n"

            summary += f"\n<<COLOR:text>>Recommendation:<</COLOR>>\n"
            summary += f"  Use Levene's test (robust to non-normality).\n"
            summary += f"  Bartlett's test is more powerful but assumes normal data.\n\n"

            sig = lev_p < alpha
            if sig:
                summary += f"<<COLOR:good>>Variances are significantly different (Levene's p = {lev_p:.4f} < {alpha})<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference in variances (Levene's p = {lev_p:.4f} ≥ {alpha})<</COLOR>>"

            result["statistics"] = {
                "bartlett_statistic": float(bart_stat), "bartlett_p": float(bart_p),
                "levene_statistic": float(lev_stat), "levene_p": float(lev_p),
                "group_stds": dict(zip([str(g) for g in groups_labels], stds)),
                "group_variances": dict(zip([str(g) for g in groups_labels], variances)),
            }
            if f_stat is not None:
                result["statistics"]["f_statistic"] = float(f_stat)
                result["statistics"]["f_p_value"] = float(f_p)

            result["guide_observation"] = f"Variance test ({k} groups): Levene's p={lev_p:.4f}, Bartlett's p={bart_p:.4f}. " + ("Variances differ." if sig else "Variances are equal.")

            # Side-by-side box/strip plots showing spread
            traces = []
            for i, (lbl, gd) in enumerate(zip(groups_labels, groups_data)):
                colors = ["#4a9f6e", "#4a90d9", "#e8c547", "#c75a3a", "#7a5fb8", "#5a9fd4", "#d4a05a", "#5ad4a0"]
                traces.append({
                    "type": "box", "y": gd.tolist(), "name": str(lbl),
                    "marker": {"color": colors[i % len(colors)]}, "boxpoints": "outliers",
                })
            result["plots"].append({
                "title": "Variability Comparison",
                "data": traces,
                "layout": {"template": "plotly_dark", "height": 300, "yaxis": {"title": "Value"}},
            })

            # Interval plot of standard deviations
            result["plots"].append({
                "data": [{
                    "type": "bar", "x": [str(g) for g in groups_labels], "y": stds,
                    "marker": {"color": "#4a9f6e"},
                    "text": [f"{s:.4f}" for s in stds], "textposition": "outside",
                }],
                "layout": {
                    "title": "Standard Deviations by Group",
                    "yaxis": {"title": "Std Dev", "rangemode": "tozero"},
                    "template": "plotly_white", "height": 250,
                },
            })

        result["summary"] = summary

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
        summary += f"<<COLOR:title>>TWO-SAMPLE POISSON RATE TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  {'Group':<20} {'Count':>8} {'Exposure':>10} {'Rate':>10}\n"
        summary += f"  {'─' * 52}\n"
        summary += f"  {label1:<20} {c1:>8.0f} {e1:>10.1f} {r1:>10.4f}\n"
        summary += f"  {label2:<20} {c2:>8.0f} {e2:>10.1f} {r2:>10.4f}\n\n"
        summary += f"<<COLOR:text>>Rate Ratio (r₁/r₂):<</COLOR>> {rate_ratio:.4f}\n"
        summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for ratio:<</COLOR>> ({rr_lo:.4f}, {rr_hi:.4f})\n"
        summary += f"<<COLOR:text>>Rate Difference (r₁ − r₂):<</COLOR>> {diff:.4f}\n"
        summary += f"<<COLOR:text>>{conf_pct:.0f}% CI for difference:<</COLOR>> ({diff_lo:.4f}, {diff_hi:.4f})\n\n"
        summary += f"<<COLOR:text>>Exact Conditional Test:<</COLOR>>\n"
        summary += f"  p-value: {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Rates differ significantly (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in rates (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        result["plots"].append({
            "data": [{
                "type": "bar", "x": [label1, label2], "y": [r1, r2],
                "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                "text": [f"{r1:.4f}", f"{r2:.4f}"], "textposition": "outside",
            }],
            "layout": {
                "title": "Rates by Group",
                "yaxis": {"title": "Rate", "rangemode": "tozero"},
                "template": "plotly_white", "height": 280,
            },
        })

        # Rate ratio CI plot
        if rate_ratio < float("inf"):
            result["plots"].append({
                "data": [{
                    "type": "scatter", "x": [rate_ratio], "y": ["Rate Ratio"], "mode": "markers",
                    "marker": {"size": 12, "color": "#4a9f6e"},
                    "error_x": {"type": "data", "symmetric": False,
                                "array": [rr_hi - rate_ratio], "arrayminus": [rate_ratio - rr_lo],
                                "color": "#5a6a5a"},
                }],
                "layout": {
                    "title": f"Rate Ratio ({conf_pct:.0f}% CI)",
                    "xaxis": {"title": "r₁ / r₂", "type": "log"},
                    "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": 0.5,
                                "line": {"color": "#e89547", "dash": "dash"}}],
                    "height": 180, "template": "plotly_white",
                },
            })

        result["guide_observation"] = f"Two-sample Poisson: r₁={r1:.4f} vs r₂={r2:.4f}, ratio={rate_ratio:.3f}, p={p_val:.4f}. " + ("Rates differ." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "count1": c1, "count2": c2, "exposure1": e1, "exposure2": e2,
            "rate1": r1, "rate2": r2, "rate_ratio": float(rate_ratio),
            "rate_difference": diff, "p_value": p_val,
            "ratio_ci_lower": rr_lo, "ratio_ci_upper": rr_hi,
            "diff_ci_lower": diff_lo, "diff_ci_upper": diff_hi,
        }

    elif analysis_id == "attribute_capability":
        """
        Attribute Capability Analysis — capability for pass/fail or defect count data.
        Modes:
          1) Single column of pass/fail values (auto-detect defect = less frequent value)
          2) Direct counts: defects, units, opportunities
          3) Response + grouping factor for subgroup-level defect tracking
        Reports: DPU, DPO, DPMO, yield %, sigma level (with/without 1.5σ shift).
        """
        var = config.get("var") or config.get("var1")
        defects_count = config.get("defects")
        units_count = config.get("units")
        opportunities = config.get("opportunities", 1)
        event = config.get("event")

        if defects_count is not None and units_count is not None:
            # Direct counts mode
            d = float(defects_count)
            n = float(units_count)
            opp = float(opportunities)
        elif var:
            col = df[var].dropna()
            n = len(col)
            opp = float(config.get("opportunities", 1))
            # Auto-detect event (less frequent value = defect)
            if event is not None:
                d = float((col.astype(str) == str(event)).sum())
            else:
                vc = col.value_counts()
                if len(vc) == 2:
                    defect_val = vc.index[-1]  # less frequent
                    d = float(vc.iloc[-1])
                    event = str(defect_val)
                elif col.dtype in ["int64", "float64"]:
                    d = float(col.sum())
                    event = "sum"
                else:
                    result["summary"] = "Cannot auto-detect defect value. Please specify 'event' in config."
                    return result
        else:
            result["summary"] = "Provide a column name (var) or direct counts (defects, units)."
            return result

        if n <= 0:
            result["summary"] = "No valid data to analyze."
            return result

        dpu = d / n
        dpo = d / (n * opp) if opp > 0 else 0
        dpmo = dpo * 1_000_000
        yield_pct = (1 - dpo) * 100

        # Sigma level (using inverse normal)
        if 0 < dpo < 1:
            z_bench = float(stats.norm.ppf(1 - dpo))
            sigma_st = z_bench + 1.5  # short-term with 1.5σ shift
        elif dpo == 0:
            z_bench = 6.0
            sigma_st = 7.5
        else:
            z_bench = 0.0
            sigma_st = 1.5

        # CI for proportion defective (Wilson)
        p_hat = dpo
        z_a = stats.norm.ppf(1 - 0.05 / 2)
        total_opp = n * opp
        denom = 1 + z_a ** 2 / total_opp
        center = (p_hat + z_a ** 2 / (2 * total_opp)) / denom
        half = z_a * np.sqrt(p_hat * (1 - p_hat) / total_opp + z_a ** 2 / (4 * total_opp ** 2)) / denom
        ci_lo = max(0, center - half)
        ci_hi = min(1, center + half)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ATTRIBUTE CAPABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

        if var:
            summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
            if event and event != "sum":
                summary += f"<<COLOR:highlight>>Defect value:<</COLOR>> {event}\n"
        summary += f"<<COLOR:highlight>>Opportunities per unit:<</COLOR>> {opp:.0f}\n\n"

        summary += f"<<COLOR:text>>Summary:<</COLOR>>\n"
        summary += f"  Total units inspected: {n:.0f}\n"
        summary += f"  Total defects: {d:.0f}\n"
        summary += f"  Total opportunities: {n * opp:.0f}\n\n"

        summary += f"<<COLOR:text>>Capability Metrics:<</COLOR>>\n"
        summary += f"  {'Metric':<30} {'Value':>12}\n"
        summary += f"  {'─' * 44}\n"
        summary += f"  {'DPU (defects per unit)':<30} {dpu:>12.4f}\n"
        summary += f"  {'DPO (defects per opportunity)':<30} {dpo:>12.6f}\n"
        summary += f"  {'DPMO':<30} {dpmo:>12.1f}\n"
        summary += f"  {'Yield %':<30} {yield_pct:>12.2f}%\n"
        summary += f"  {'Z.bench (long-term)':<30} {z_bench:>12.2f}\n"
        summary += f"  {'Sigma level (short-term)':<30} {sigma_st:>12.2f}\n"
        summary += f"  {'95% CI for DPO':<30} ({ci_lo:.6f}, {ci_hi:.6f})\n\n"

        # Interpretation
        if sigma_st >= 6:
            interp = "World-class (Six Sigma or better)"
        elif sigma_st >= 5:
            interp = "Excellent capability"
        elif sigma_st >= 4:
            interp = "Good capability"
        elif sigma_st >= 3:
            interp = "Marginal — improvement needed"
        else:
            interp = "Poor — significant defect rate"
        summary += f"<<COLOR:{'good' if sigma_st >= 4 else 'warning'}>>{interp}<</COLOR>>"

        result["summary"] = summary

        # DPMO gauge / sigma chart
        sigma_levels = [1, 2, 3, 4, 5, 6]
        dpmo_levels = [691462, 308538, 66807, 6210, 233, 3.4]
        result["plots"].append({
            "data": [
                {"type": "bar", "x": [f"{s}σ" for s in sigma_levels], "y": dpmo_levels,
                 "marker": {"color": ["#c75a3a" if s < sigma_st else "#4a9f6e" for s in sigma_levels], "opacity": 0.5},
                 "name": "DPMO by sigma"},
                {"type": "scatter", "x": [f"{sigma_st:.1f}σ"], "y": [dpmo], "mode": "markers",
                 "marker": {"size": 14, "color": "#e8c547", "symbol": "diamond"},
                 "name": f"Your process ({dpmo:.0f} DPMO)"},
            ],
            "layout": {
                "title": "Process Sigma Level",
                "yaxis": {"title": "DPMO", "type": "log"},
                "template": "plotly_white", "height": 280,
            },
        })

        result["guide_observation"] = f"Attribute capability: DPMO={dpmo:.0f}, Sigma={sigma_st:.1f}, Yield={yield_pct:.2f}%."
        result["statistics"] = {
            "defects": d, "units": n, "opportunities": opp,
            "dpu": dpu, "dpo": dpo, "dpmo": dpmo,
            "yield_percent": yield_pct, "z_bench": z_bench,
            "sigma_level": sigma_st, "interpretation": interp,
        }

    elif analysis_id == "nonnormal_capability_np":
        """
        Nonparametric Process Capability — for non-normal data.
        Uses percentile-based method: Cnpk from 0.135th and 99.865th percentiles
        (equivalent to ±3σ bounds for normal data).
        Auto-runs Anderson-Darling normality test for comparison.
        """
        var = config.get("var") or config.get("var1")
        usl = float(config.get("usl"))
        lsl = float(config.get("lsl"))
        target = config.get("target")
        alpha = 1 - float(config.get("conf", 95)) / 100
        conf_pct = float(config.get("conf", 95))

        data_arr = df[var].dropna().values.astype(float)
        n = len(data_arr)
        if n < 10:
            result["summary"] = "Nonparametric capability requires at least 10 data points."
            return result

        mean_val = float(np.mean(data_arr))
        median_val = float(np.median(data_arr))
        std_val = float(np.std(data_arr, ddof=1))

        if target is None:
            target = (usl + lsl) / 2
        else:
            target = float(target)

        # Percentiles for capability (equivalent to ±3σ for normal)
        p_low = float(np.percentile(data_arr, 0.135))
        p_high = float(np.percentile(data_arr, 99.865))

        # Nonparametric capability indices
        spec_width = usl - lsl
        np_width = p_high - p_low
        cnp = spec_width / np_width if np_width > 0 else 0

        cnpk_upper = (usl - median_val) / (p_high - median_val) if (p_high - median_val) > 0 else 0
        cnpk_lower = (median_val - lsl) / (median_val - p_low) if (median_val - p_low) > 0 else 0
        cnpk = min(cnpk_upper, cnpk_lower)

        # PPM outside specs (empirical)
        ppm_below = float(np.sum(data_arr < lsl) / n * 1_000_000)
        ppm_above = float(np.sum(data_arr > usl) / n * 1_000_000)
        ppm_total = ppm_below + ppm_above

        # Normal-assumption comparison
        ad_stat, ad_crit, ad_sig = stats.anderson(data_arr, dist="norm")
        is_normal = ad_stat < ad_crit[2]  # 5% significance level

        # Normal-based indices for comparison
        cp_normal = spec_width / (6 * std_val) if std_val > 0 else 0
        cpk_upper = (usl - mean_val) / (3 * std_val) if std_val > 0 else 0
        cpk_lower = (mean_val - lsl) / (3 * std_val) if std_val > 0 else 0
        cpk_normal = min(cpk_upper, cpk_lower)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NONPARAMETRIC PROCESS CAPABILITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Specs:<</COLOR>> LSL = {lsl}, USL = {usl}, Target = {target}\n\n"

        summary += f"<<COLOR:text>>Normality Test (Anderson-Darling):<</COLOR>>\n"
        summary += f"  AD statistic: {ad_stat:.4f}\n"
        summary += f"  5% critical value: {ad_crit[2]:.4f}\n"
        summary += f"  Data is {'normal' if is_normal else '<<COLOR:warning>>non-normal<</COLOR>>'}\n\n"

        summary += f"<<COLOR:text>>Comparison — Normal vs Nonparametric:<</COLOR>>\n"
        summary += f"  {'Method':<25} {'Cp/Cnp':>10} {'Cpk/Cnpk':>10}\n"
        summary += f"  {'─' * 48}\n"
        summary += f"  {'Normal assumption':<25} {cp_normal:>10.3f} {cpk_normal:>10.3f}\n"
        summary += f"  {'Nonparametric (percentile)':<25} {cnp:>10.3f} {cnpk:>10.3f}\n\n"

        summary += f"<<COLOR:text>>Nonparametric Details:<</COLOR>>\n"
        summary += f"  Median: {median_val:.4f}\n"
        summary += f"  0.135th percentile: {p_low:.4f}\n"
        summary += f"  99.865th percentile: {p_high:.4f}\n"
        summary += f"  Empirical PPM below LSL: {ppm_below:.0f}\n"
        summary += f"  Empirical PPM above USL: {ppm_above:.0f}\n"
        summary += f"  Empirical PPM total: {ppm_total:.0f}\n\n"

        if not is_normal:
            summary += f"<<COLOR:warning>>Data is non-normal. Nonparametric indices (Cnpk = {cnpk:.3f}) are more reliable than normal-based (Cpk = {cpk_normal:.3f}).<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>Data appears normal. Both methods should agree. Cnpk = {cnpk:.3f}, Cpk = {cpk_normal:.3f}.<</COLOR>>"

        result["summary"] = summary

        # Histogram with spec limits
        result["plots"].append({
            "data": [
                {"type": "histogram", "x": data_arr.tolist(), "marker": {"color": "#4a9f6e", "opacity": 0.7},
                 "name": var, "nbinsx": min(30, max(10, n // 5))},
            ],
            "layout": {
                "title": "Distribution with Spec Limits",
                "xaxis": {"title": var},
                "yaxis": {"title": "Frequency"},
                "shapes": [
                    {"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#e85747", "dash": "dash", "width": 2}},
                    {"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#e85747", "dash": "dash", "width": 2}},
                    {"type": "line", "x0": target, "x1": target, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#e8c547", "dash": "dot", "width": 1}},
                    {"type": "line", "x0": p_low, "x1": p_low, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#7a5fb8", "dash": "dot", "width": 1}},
                    {"type": "line", "x0": p_high, "x1": p_high, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#7a5fb8", "dash": "dot", "width": 1}},
                ],
                "annotations": [
                    {"x": lsl, "y": 1.02, "yref": "paper", "text": "LSL", "showarrow": False, "font": {"color": "#e85747", "size": 10}},
                    {"x": usl, "y": 1.02, "yref": "paper", "text": "USL", "showarrow": False, "font": {"color": "#e85747", "size": 10}},
                    {"x": p_low, "y": 0.95, "yref": "paper", "text": "0.135%ile", "showarrow": False, "font": {"color": "#7a5fb8", "size": 9}},
                    {"x": p_high, "y": 0.95, "yref": "paper", "text": "99.865%ile", "showarrow": False, "font": {"color": "#7a5fb8", "size": 9}},
                ],
                "template": "plotly_white", "height": 300,
            },
        })

        result["guide_observation"] = f"Nonparametric capability: Cnpk={cnpk:.3f}, Empirical PPM={ppm_total:.0f}. Data is {'normal' if is_normal else 'non-normal'}."
        result["statistics"] = {
            "cnp": cnp, "cnpk": cnpk, "cnpk_upper": cnpk_upper, "cnpk_lower": cnpk_lower,
            "cp_normal": cp_normal, "cpk_normal": cpk_normal,
            "median": median_val, "p_0135": p_low, "p_99865": p_high,
            "ppm_below_lsl": ppm_below, "ppm_above_usl": ppm_above, "ppm_total": ppm_total,
            "ad_statistic": float(ad_stat), "is_normal": is_normal,
        }

    elif analysis_id == "nominal_logistic":
        """
        Nominal Logistic Regression — for multi-class categorical outcomes.
        Uses sklearn's multinomial logistic regression.
        Auto-excludes response from predictors if accidentally included.
        If response has only 2 levels, suggests binary logistic instead.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix

        response = config.get("response")
        predictors = list(config.get("predictors", []))

        # Convenience: auto-exclude response from predictors
        if response in predictors:
            predictors.remove(response)

        if not predictors:
            result["summary"] = "Please select at least one predictor variable."
            return result

        data_clean = df[[response] + predictors].dropna()
        y_raw = data_clean[response]
        classes = sorted(y_raw.unique().tolist(), key=str)

        if len(classes) < 2:
            result["summary"] = f"Response '{response}' has only {len(classes)} unique value(s). Need at least 2."
            return result
        if len(classes) == 2:
            result["summary"] = f"<<COLOR:warning>>Response '{response}' has only 2 levels. Consider using binary logistic regression ('logistic') instead for simpler interpretation.<</COLOR>>\n\nProceeding with nominal logistic..."

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = le.classes_.tolist()
        ref_class = class_names[0]

        X = data_clean[predictors]
        # Encode categorical predictors
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                X = pd.get_dummies(X, columns=[col], drop_first=True)

        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = float((y_pred == y).mean())

        cm = confusion_matrix(y, y_pred)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NOMINAL LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response} ({len(classes)} categories)\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>Reference category:<</COLOR>> {ref_class}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(data_clean)}\n"
        summary += f"<<COLOR:highlight>>Accuracy:<</COLOR>> {accuracy:.1%}\n\n"

        # Coefficients per class (vs reference)
        pred_names = list(X.columns)
        summary += f"<<COLOR:text>>Coefficients (vs reference '{ref_class}'):<</COLOR>>\n"
        summary += f"  {'Predictor':<25}"
        for cls in class_names[1:]:
            summary += f" {str(cls):>12}"
        summary += "\n"
        summary += f"  {'─' * (25 + 13 * (len(class_names) - 1))}\n"

        for j, pred in enumerate(pred_names):
            summary += f"  {pred:<25}"
            for i in range(1, len(class_names)):
                coef = model.coef_[i][j] if i < len(model.coef_) else 0
                summary += f" {coef:>12.4f}"
            summary += "\n"

        # Odds ratios with CIs (approximate SEs via Fisher information)
        _nom_se = {}
        try:
            _probs = model.predict_proba(X)
            for _ki in range(1, len(class_names)):
                _wk = _probs[:, _ki] * (1 - _probs[:, _ki])
                _Xv = X.values
                _info = _Xv.T @ (_wk[:, None] * _Xv)
                _nom_se[_ki] = np.sqrt(np.diag(np.linalg.inv(_info)))
        except Exception:
            pass

        summary += f"\n<<COLOR:text>>Odds Ratios (exp(coef)):<</COLOR>>\n"
        _has_ci = len(_nom_se) > 0
        summary += f"  {'Predictor':<25}"
        for cls in class_names[1:]:
            summary += f" {str(cls):>12}"
            if _has_ci:
                summary += f" {'95% CI':>22}"
        summary += "\n"
        _col_w = (13 + (23 if _has_ci else 0)) * (len(class_names) - 1)
        summary += f"  {'─' * (25 + _col_w)}\n"
        for j, pred in enumerate(pred_names):
            summary += f"  {pred:<25}"
            for i in range(1, len(class_names)):
                coef = model.coef_[i][j] if i < len(model.coef_) else 0
                summary += f" {np.exp(coef):>12.4f}"
                if _has_ci and i in _nom_se and j < len(_nom_se[i]):
                    _se_j = _nom_se[i][j]
                    summary += f" [{np.exp(coef - 1.96*_se_j):>8.4f}, {np.exp(coef + 1.96*_se_j):>8.4f}]"
                elif _has_ci:
                    summary += f" {'':>22}"
            summary += "\n"

        # Confusion matrix
        summary += f"\n<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
        _cm_header = "Actual \\ Pred"
        summary += f"  {_cm_header:<15}"
        for cls in class_names:
            summary += f" {str(cls):>8}"
        summary += "\n"
        for i, cls in enumerate(class_names):
            summary += f"  {str(cls):<15}"
            for j in range(len(class_names)):
                summary += f" {cm[i, j]:>8}"
            summary += "\n"

        result["summary"] = summary

        # Predicted probability heatmap
        probs = model.predict_proba(X)
        avg_probs = []
        for i, cls in enumerate(class_names):
            avg_probs.append(float(probs[:, i].mean()))
        result["plots"].append({
            "data": [{
                "type": "bar", "x": [str(c) for c in class_names], "y": avg_probs,
                "marker": {"color": ["#4a9f6e", "#4a90d9", "#e8c547", "#c75a3a", "#7a5fb8", "#5a9fd4", "#d4a05a", "#5ad4a0"][:len(class_names)]},
                "text": [f"{p:.3f}" for p in avg_probs], "textposition": "outside",
            }],
            "layout": {
                "title": "Average Predicted Probability by Class",
                "yaxis": {"title": "Avg Probability", "range": [0, max(avg_probs) * 1.2 + 0.05]},
                "template": "plotly_white", "height": 280,
            },
        })

        # Coefficient plot (grouped bar)
        if len(class_names) > 2:
            bar_traces = []
            colors = ["#4a90d9", "#e8c547", "#c75a3a", "#7a5fb8", "#5a9fd4"]
            for i in range(1, len(class_names)):
                coefs_i = [float(model.coef_[i][j]) if i < len(model.coef_) and j < len(model.coef_[i]) else 0 for j in range(len(pred_names))]
                bar_traces.append({
                    "type": "bar", "x": pred_names, "y": coefs_i,
                    "name": f"vs {ref_class} → {class_names[i]}",
                    "marker": {"color": colors[(i - 1) % len(colors)]},
                })
            result["plots"].append({
                "data": bar_traces,
                "layout": {
                    "title": "Coefficients by Category",
                    "barmode": "group",
                    "yaxis": {"title": "Coefficient"},
                    "template": "plotly_white", "height": 300,
                },
            })

        result["guide_observation"] = f"Nominal logistic: {len(classes)} categories, accuracy={accuracy:.1%}."
        result["statistics"] = {
            "n": len(data_clean), "n_classes": len(classes),
            "classes": [str(c) for c in class_names],
            "accuracy": accuracy, "reference_class": str(ref_class),
        }

    elif analysis_id == "orthogonal_regression":
        """
        Orthogonal / Deming Regression — minimizes perpendicular distance to line.
        Used when both X and Y have measurement error (method comparison studies).
        Error ratio delta = var(eps_x) / var(eps_y), default=1 (equal errors).
        Includes Bland-Altman plot for method agreement assessment.
        """
        var_x = config.get("var1") or config.get("var_x")
        var_y = config.get("var2") or config.get("var_y")
        delta = float(config.get("error_ratio", 1.0))
        alpha = 1 - float(config.get("conf", 95)) / 100

        data_clean = df[[var_x, var_y]].dropna()
        x = data_clean[var_x].values.astype(float)
        y = data_clean[var_y].values.astype(float)
        n = len(x)

        if n < 3:
            result["summary"] = "Need at least 3 observations for regression."
            return result

        x_bar = float(np.mean(x))
        y_bar = float(np.mean(y))
        sxx = float(np.sum((x - x_bar) ** 2) / (n - 1))
        syy = float(np.sum((y - y_bar) ** 2) / (n - 1))
        sxy = float(np.sum((x - x_bar) * (y - y_bar)) / (n - 1))

        # Deming slope
        discriminant = (syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2
        b1_deming = float((syy - delta * sxx + np.sqrt(discriminant)) / (2 * sxy)) if sxy != 0 else 1.0
        b0_deming = float(y_bar - b1_deming * x_bar)

        # OLS for comparison
        b1_ols = float(sxy / sxx) if sxx > 0 else 0.0
        b0_ols = float(y_bar - b1_ols * x_bar)

        # Residuals and R-squared
        y_pred_deming = b0_deming + b1_deming * x
        ss_total = float(np.sum((y - y_bar) ** 2))
        ss_resid = float(np.sum((y - y_pred_deming) ** 2))
        r_squared = 1 - ss_resid / ss_total if ss_total > 0 else 0

        # Bootstrap CI for Deming parameters
        rng = np.random.RandomState(42)
        boot_slopes, boot_intercepts = [], []
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            xb, yb = x[idx], y[idx]
            xb_bar, yb_bar = float(np.mean(xb)), float(np.mean(yb))
            sxxb = float(np.sum((xb - xb_bar) ** 2) / (n - 1))
            syyb = float(np.sum((yb - yb_bar) ** 2) / (n - 1))
            sxyb = float(np.sum((xb - xb_bar) * (yb - yb_bar)) / (n - 1))
            if sxyb != 0:
                disc = (syyb - delta * sxxb) ** 2 + 4 * delta * sxyb ** 2
                b1b = (syyb - delta * sxxb + np.sqrt(disc)) / (2 * sxyb)
                boot_slopes.append(b1b)
                boot_intercepts.append(yb_bar - b1b * xb_bar)

        ci_slope = (float(np.percentile(boot_slopes, 100 * alpha / 2)),
                    float(np.percentile(boot_slopes, 100 * (1 - alpha / 2))))
        ci_intercept = (float(np.percentile(boot_intercepts, 100 * alpha / 2)),
                        float(np.percentile(boot_intercepts, 100 * (1 - alpha / 2))))

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ORTHOGONAL (DEMING) REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>X:<</COLOR>> {var_x}  |  <<COLOR:highlight>>Y:<</COLOR>> {var_y}\n"
        summary += f"<<COLOR:highlight>>Error ratio (delta):<</COLOR>> {delta}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:text>>Deming Regression Results:<</COLOR>>\n"
        summary += f"  Slope:     {b1_deming:>10.4f}  ({ci_slope[0]:.4f}, {ci_slope[1]:.4f})\n"
        summary += f"  Intercept: {b0_deming:>10.4f}  ({ci_intercept[0]:.4f}, {ci_intercept[1]:.4f})\n"
        summary += f"  R-squared: {r_squared:>10.4f}\n\n"
        summary += f"<<COLOR:text>>OLS Comparison:<</COLOR>>\n"
        summary += f"  Slope:     {b1_ols:>10.4f}\n"
        summary += f"  Intercept: {b0_ols:>10.4f}\n\n"
        summary += f"<<COLOR:text>>Interpretation:<</COLOR>>\n"
        if abs(b1_deming - 1.0) < 0.1 and abs(b0_deming) < (np.std(y) * 0.1):
            summary += f"  <<COLOR:good>>Methods show good agreement (slope ~ 1, intercept ~ 0).<</COLOR>>\n"
        elif abs(b1_deming - 1.0) < 0.1:
            summary += f"  <<COLOR:warning>>Proportional agreement but constant bias (intercept = {b0_deming:.4f}).<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:warning>>Methods disagree -- both slope and intercept differ from ideal (1, 0).<</COLOR>>\n"

        result["summary"] = summary

        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        result["plots"].append({
            "data": [
                {"type": "scatter", "x": x.tolist(), "y": y.tolist(), "mode": "markers",
                 "marker": {"color": "#4a9f6e", "size": 5}, "name": "Data"},
                {"type": "scatter", "x": x_line.tolist(), "y": (b0_deming + b1_deming * x_line).tolist(),
                 "mode": "lines", "line": {"color": "#e89547", "width": 2}, "name": "Deming"},
                {"type": "scatter", "x": x_line.tolist(), "y": (b0_ols + b1_ols * x_line).tolist(),
                 "mode": "lines", "line": {"color": "#47a5e8", "dash": "dash", "width": 2}, "name": "OLS"},
                {"type": "scatter", "x": x_line.tolist(), "y": x_line.tolist(),
                 "mode": "lines", "line": {"color": "#888", "dash": "dot", "width": 1}, "name": "Identity (y=x)"},
            ],
            "layout": {"title": "Deming vs OLS Regression", "xaxis": {"title": var_x}, "yaxis": {"title": var_y},
                        "template": "plotly_white", "height": 350}
        })

        # Bland-Altman plot
        mean_xy = (x + y) / 2
        diff_xy = y - x
        diff_mean = float(np.mean(diff_xy))
        diff_std = float(np.std(diff_xy, ddof=1))
        result["plots"].append({
            "data": [
                {"type": "scatter", "x": mean_xy.tolist(), "y": diff_xy.tolist(), "mode": "markers",
                 "marker": {"color": "#4a9f6e", "size": 5}, "name": "Differences"},
            ],
            "layout": {"title": "Bland-Altman Plot", "xaxis": {"title": f"Mean of {var_x} and {var_y}"},
                        "yaxis": {"title": f"{var_y} - {var_x}"},
                        "shapes": [
                            {"type": "line", "x0": float(mean_xy.min()), "x1": float(mean_xy.max()),
                             "y0": diff_mean, "y1": diff_mean, "line": {"color": "#e89547", "width": 2}},
                            {"type": "line", "x0": float(mean_xy.min()), "x1": float(mean_xy.max()),
                             "y0": diff_mean + 1.96 * diff_std, "y1": diff_mean + 1.96 * diff_std,
                             "line": {"color": "#d94a4a", "dash": "dash", "width": 1}},
                            {"type": "line", "x0": float(mean_xy.min()), "x1": float(mean_xy.max()),
                             "y0": diff_mean - 1.96 * diff_std, "y1": diff_mean - 1.96 * diff_std,
                             "line": {"color": "#d94a4a", "dash": "dash", "width": 1}},
                        ],
                        "template": "plotly_white", "height": 300}
        })

        result["guide_observation"] = f"Deming regression: slope={b1_deming:.4f}, intercept={b0_deming:.4f}, R2={r_squared:.4f}."
        result["statistics"] = {
            "deming_slope": b1_deming, "deming_intercept": b0_deming,
            "ols_slope": b1_ols, "ols_intercept": b0_ols,
            "r_squared": r_squared, "error_ratio": delta, "n": n,
            "slope_ci": list(ci_slope), "intercept_ci": list(ci_intercept),
            "bland_altman_bias": diff_mean,
            "bland_altman_loa": [diff_mean - 1.96 * diff_std, diff_mean + 1.96 * diff_std],
        }

    elif analysis_id == "nonlinear_regression":
        """
        Nonlinear Regression — fit preset or user-specified curve models.
        Uses scipy.optimize.curve_fit (Levenberg-Marquardt).
        Presets: exponential, power, logistic, logarithmic, polynomial2, polynomial3,
                 michaelis_menten, gompertz, hill.
        """
        var_x = config.get("var1") or config.get("var_x")
        var_y = config.get("var2") or config.get("var_y")
        model_type = config.get("model", "exponential")
        initial_params = config.get("initial_params")
        alpha = 1 - float(config.get("conf", 95)) / 100

        data_clean = df[[var_x, var_y]].dropna()
        x = data_clean[var_x].values.astype(float)
        y = data_clean[var_y].values.astype(float)
        n = len(x)

        if n < 3:
            result["summary"] = "Need at least 3 observations for curve fitting."
            return result

        from scipy.optimize import curve_fit

        models = {
            "exponential": (lambda x, a, b: a * np.exp(b * x), ["a", "b"], [1.0, 0.01]),
            "power": (lambda x, a, b: a * np.power(np.maximum(x, 1e-10), b), ["a", "b"], [1.0, 1.0]),
            "logistic": (lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), ["L", "k", "x0"],
                         [float(max(y)), 1.0, float(np.median(x))]),
            "logarithmic": (lambda x, a, b: a * np.log(np.maximum(x, 1e-10)) + b, ["a", "b"], [1.0, 0.0]),
            "polynomial2": (lambda x, a, b, c: a * x**2 + b * x + c, ["a", "b", "c"], [0.0, 1.0, 0.0]),
            "polynomial3": (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                            ["a", "b", "c", "d"], [0.0, 0.0, 1.0, 0.0]),
            "michaelis_menten": (lambda x, Vmax, Km: Vmax * x / (Km + x), ["Vmax", "Km"],
                                 [float(max(y)), float(np.median(x))]),
            "gompertz": (lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), ["a", "b", "c"],
                         [float(max(y)), 1.0, 0.1]),
            "hill": (lambda x, Vmax, Kd, n_h: Vmax * x**n_h / (Kd**n_h + x**n_h), ["Vmax", "Kd", "n"],
                     [float(max(y)), float(np.median(x)), 1.0]),
        }

        if model_type not in models:
            result["summary"] = f"Unknown model '{model_type}'. Available: {', '.join(models.keys())}"
            return result

        func, param_names, p0_default = models[model_type]
        p0 = initial_params if initial_params and len(initial_params) == len(param_names) else p0_default

        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
        except Exception as e:
            result["summary"] = f"Curve fitting failed: {str(e)}. Try different initial parameters or a different model."
            return result

        y_pred = func(x, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = float(np.sqrt(ss_res / n))

        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(len(popt))

        k_params = len(popt)
        aic = float(n * np.log(ss_res / n) + 2 * k_params) if ss_res > 0 else 0
        bic = float(n * np.log(ss_res / n) + k_params * np.log(n)) if ss_res > 0 else 0

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NONLINEAR REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type}\n"
        summary += f"<<COLOR:highlight>>X:<</COLOR>> {var_x}  |  <<COLOR:highlight>>Y:<</COLOR>> {var_y}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:text>>Fitted Parameters:<</COLOR>>\n"
        summary += f"  {'Parameter':<12} {'Estimate':>12} {'Std Error':>12}\n"
        summary += f"  {'-' * 38}\n"
        for name, val, se in zip(param_names, popt, perr):
            summary += f"  {name:<12} {float(val):>12.6f} {float(se):>12.6f}\n"
        summary += f"\n<<COLOR:text>>Goodness of Fit:<</COLOR>>\n"
        summary += f"  R-squared: {r_squared:.4f}\n"
        summary += f"  RMSE:      {rmse:.4f}\n"
        summary += f"  AIC:       {aic:.2f}\n"
        summary += f"  BIC:       {bic:.2f}\n"

        result["summary"] = summary

        x_smooth = np.linspace(float(x.min()), float(x.max()), 200)
        y_smooth = func(x_smooth, *popt)
        result["plots"].append({
            "data": [
                {"type": "scatter", "x": x.tolist(), "y": y.tolist(), "mode": "markers",
                 "marker": {"color": "#4a9f6e", "size": 5}, "name": "Data"},
                {"type": "scatter", "x": x_smooth.tolist(), "y": y_smooth.tolist(),
                 "mode": "lines", "line": {"color": "#e89547", "width": 2}, "name": f"Fitted ({model_type})"},
            ],
            "layout": {"title": f"Nonlinear Fit: {model_type}", "xaxis": {"title": var_x},
                        "yaxis": {"title": var_y}, "template": "plotly_white", "height": 350}
        })

        residuals_nlr = y - y_pred
        result["plots"].append({
            "data": [
                {"type": "scatter", "x": y_pred.tolist(), "y": residuals_nlr.tolist(), "mode": "markers",
                 "marker": {"color": "#47a5e8", "size": 5}, "name": "Residuals"},
            ],
            "layout": {"title": "Residuals vs Fitted", "xaxis": {"title": "Fitted values"},
                        "yaxis": {"title": "Residual"},
                        "shapes": [{"type": "line", "x0": float(y_pred.min()), "x1": float(y_pred.max()),
                                    "y0": 0, "y1": 0, "line": {"color": "#888", "dash": "dash"}}],
                        "template": "plotly_white", "height": 250}
        })

        result["guide_observation"] = f"Nonlinear regression ({model_type}): R2={r_squared:.4f}, RMSE={rmse:.4f}."
        result["statistics"] = {
            "model": model_type, "n": n,
            "parameters": {name: float(val) for name, val in zip(param_names, popt)},
            "parameter_se": {name: float(se) for name, se in zip(param_names, perr)},
            "r_squared": r_squared, "rmse": rmse, "aic": aic, "bic": bic,
        }

    elif analysis_id == "variable_acceptance_sampling":
        """
        Variables Acceptance Sampling — accept/reject lots based on measured variable data.
        Uses k-method (MIL-STD-414 / ANSI Z1.9 style):
          Accept if Z_stat = (xbar - LSL) / s >= k  (or (USL - xbar) / s >= k).
        Generates the OC curve. Supports single spec, double spec, known/unknown sigma.
        """
        aql = float(config.get("aql", 1.0))
        ltpd = float(config.get("ltpd") or config.get("rql", 5.0))
        lot_size = int(config.get("lot_size") or config.get("N", 1000))
        alpha_risk = float(config.get("alpha", 0.05))
        beta_risk = float(config.get("beta", 0.10))
        spec_type = config.get("spec_type", "lower")
        lsl_vs = config.get("lsl")
        usl_vs = config.get("usl")

        from scipy.stats import norm as norm_dist

        p1 = aql / 100
        p2 = ltpd / 100
        z_p1 = float(norm_dist.ppf(1 - p1))
        z_p2 = float(norm_dist.ppf(1 - p2))
        z_alpha = float(norm_dist.ppf(1 - alpha_risk))
        z_beta = float(norm_dist.ppf(1 - beta_risk))

        if z_p1 != z_p2:
            n_approx = ((z_alpha + z_beta) / (z_p1 - z_p2)) ** 2 + 0.5 * z_alpha ** 2
            n_sample = max(2, int(np.ceil(n_approx)))
        else:
            n_sample = 50

        k_val = float(z_p1 - z_alpha / np.sqrt(n_sample)) if n_sample > 1 else z_p1

        # OC curve
        p_range = np.linspace(0.001, min(ltpd * 3 / 100, 0.5), 100)
        pa_values = []
        for p in p_range:
            z_p = float(norm_dist.ppf(1 - p))
            pa = float(norm_dist.cdf((z_p - k_val) * np.sqrt(n_sample)))
            pa_values.append(max(0.0, min(1.0, pa)))

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>VARIABLES ACCEPTANCE SAMPLING PLAN<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Plan Parameters:<</COLOR>>\n"
        summary += f"  AQL (Acceptable Quality Level):         {aql}%\n"
        summary += f"  LTPD (Lot Tolerance Pct Defective):     {ltpd}%\n"
        summary += f"  Producer's risk (alpha):                {alpha_risk}\n"
        summary += f"  Consumer's risk (beta):                 {beta_risk}\n"
        summary += f"  Lot size:                               {lot_size}\n\n"
        summary += f"<<COLOR:text>>Sampling Plan:<</COLOR>>\n"
        summary += f"  <<COLOR:highlight>>Sample size (n):<</COLOR>>      {n_sample}\n"
        summary += f"  <<COLOR:highlight>>Critical value (k):<</COLOR>>   {k_val:.4f}\n\n"
        summary += f"<<COLOR:text>>Decision Rule ({spec_type} spec):<</COLOR>>\n"
        if spec_type == "lower" and lsl_vs is not None:
            summary += f"  Accept if: (xbar - {lsl_vs}) / s >= {k_val:.4f}\n"
        elif spec_type == "upper" and usl_vs is not None:
            summary += f"  Accept if: ({usl_vs} - xbar) / s >= {k_val:.4f}\n"
        elif spec_type == "both" and lsl_vs is not None and usl_vs is not None:
            summary += f"  Accept if: (xbar - {lsl_vs})/s >= {k_val:.4f} AND ({usl_vs} - xbar)/s >= {k_val:.4f}\n"
        else:
            summary += f"  Accept if: Z.LSL or Z.USL >= {k_val:.4f}\n"

        # Evaluate data if provided
        var_vs = config.get("var") or config.get("var1")
        if var_vs and var_vs in df.columns:
            col_vs = df[var_vs].dropna().values.astype(float)
            x_bar_vs = float(np.mean(col_vs))
            s_vs = float(np.std(col_vs, ddof=1))
            summary += f"\n<<COLOR:text>>Sample Evaluation:<</COLOR>>\n"
            summary += f"  n: {len(col_vs)},  xbar: {x_bar_vs:.4f},  s: {s_vs:.4f}\n"
            if lsl_vs is not None:
                z_lsl = (x_bar_vs - float(lsl_vs)) / s_vs if s_vs > 0 else 0
                accept_lsl = z_lsl >= k_val
                summary += f"  Z.LSL = {z_lsl:.4f}  {'<<COLOR:good>>ACCEPT' if accept_lsl else '<<COLOR:warning>>REJECT'}<</COLOR>>\n"
            if usl_vs is not None:
                z_usl = (float(usl_vs) - x_bar_vs) / s_vs if s_vs > 0 else 0
                accept_usl = z_usl >= k_val
                summary += f"  Z.USL = {z_usl:.4f}  {'<<COLOR:good>>ACCEPT' if accept_usl else '<<COLOR:warning>>REJECT'}<</COLOR>>\n"

        result["summary"] = summary

        result["plots"].append({
            "data": [
                {"type": "scatter", "x": (p_range * 100).tolist(), "y": pa_values,
                 "mode": "lines", "line": {"color": "#4a9f6e", "width": 2}, "name": "OC Curve"},
                {"type": "scatter", "x": [aql], "y": [1 - alpha_risk], "mode": "markers",
                 "marker": {"color": "#47a5e8", "size": 10, "symbol": "diamond"}, "name": f"AQL ({aql}%)"},
                {"type": "scatter", "x": [ltpd], "y": [beta_risk], "mode": "markers",
                 "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"}, "name": f"LTPD ({ltpd}%)"},
            ],
            "layout": {"title": f"OC Curve (n={n_sample}, k={k_val:.3f})", "xaxis": {"title": "Percent Defective (%)"},
                        "yaxis": {"title": "Probability of Acceptance", "range": [0, 1.05]},
                        "template": "plotly_white", "height": 350}
        })

        result["guide_observation"] = f"Variables sampling plan: n={n_sample}, k={k_val:.4f} for AQL={aql}%, LTPD={ltpd}%."
        result["statistics"] = {
            "n": n_sample, "k": k_val, "aql": aql, "ltpd": ltpd,
            "alpha": alpha_risk, "beta": beta_risk, "lot_size": lot_size,
        }

    # =====================================================================
    # Poisson Regression
    # =====================================================================
    elif analysis_id == "poisson_regression":
        """
        Poisson Regression — models count data as a function of predictors.
        Uses log link: log(E[Y]) = Xβ. Fits via GLM with Poisson family.
        Reports coefficients, IRR (incidence rate ratios), deviance goodness-of-fit.
        """
        import statsmodels.api as sm

        response_pr = config.get("response") or config.get("var")
        predictors_pr = config.get("predictors") or config.get("features", [])
        if isinstance(predictors_pr, str):
            predictors_pr = [predictors_pr]
        offset_col_pr = config.get("offset")  # exposure/offset variable (optional)

        data_pr = df[[response_pr] + predictors_pr + ([offset_col_pr] if offset_col_pr else [])].dropna()
        y_pr = data_pr[response_pr].values.astype(float)

        # Check for non-negative integers
        if np.any(y_pr < 0):
            result["summary"] = "Poisson regression requires non-negative count data."
            return result

        # Build design matrix with dummies for categorical
        X_parts_pr = []
        feature_names_pr = []
        for pred in predictors_pr:
            if data_pr[pred].dtype == object or data_pr[pred].nunique() < 6:
                dummies = pd.get_dummies(data_pr[pred], prefix=pred, drop_first=True, dtype=float)
                X_parts_pr.append(dummies.values)
                feature_names_pr.extend(dummies.columns.tolist())
            else:
                X_parts_pr.append(data_pr[[pred]].values.astype(float))
                feature_names_pr.append(pred)

        X_pr = np.column_stack(X_parts_pr) if X_parts_pr else np.ones((len(data_pr), 0))
        X_pr = sm.add_constant(X_pr)
        feature_names_pr = ["Intercept"] + feature_names_pr

        offset_vals_pr = np.log(data_pr[offset_col_pr].values.astype(float)) if offset_col_pr else None

        try:
            model_pr = sm.GLM(y_pr, X_pr, family=sm.families.Poisson(),
                              offset=offset_vals_pr).fit()

            n_pr = int(model_pr.nobs)
            dev_pr = float(model_pr.deviance)
            pearson_chi2_pr = float(model_pr.pearson_chi2)
            df_resid_pr = int(model_pr.df_resid)
            aic_pr = float(model_pr.aic)
            bic_pr = float(model_pr.bic)
            llf_pr = float(model_pr.llf)

            # Dispersion test: deviance/df should be ~1 for Poisson
            dispersion_pr = dev_pr / df_resid_pr if df_resid_pr > 0 else float('nan')

            # Coefficients table
            coefs_pr = []
            for i, name in enumerate(feature_names_pr):
                coef_val = float(model_pr.params[i])
                se_val = float(model_pr.bse[i])
                z_val = float(model_pr.tvalues[i])
                p_val = float(model_pr.pvalues[i])
                irr_val = float(np.exp(coef_val))
                irr_lo = float(np.exp(model_pr.conf_int()[i, 0]))
                irr_hi = float(np.exp(model_pr.conf_int()[i, 1]))
                coefs_pr.append({
                    "name": name, "coef": coef_val, "se": se_val,
                    "z": z_val, "p": p_val, "irr": irr_val,
                    "irr_lo": irr_lo, "irr_hi": irr_hi,
                })

            summary_pr = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_pr += f"<<COLOR:title>>POISSON REGRESSION<</COLOR>>\n"
            summary_pr += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_pr += f"<<COLOR:highlight>>Response:<</COLOR>> {response_pr} (count)\n"
            summary_pr += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors_pr)}\n"
            if offset_col_pr:
                summary_pr += f"<<COLOR:highlight>>Offset (exposure):<</COLOR>> log({offset_col_pr})\n"
            summary_pr += f"<<COLOR:highlight>>N:<</COLOR>> {n_pr}\n\n"

            summary_pr += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary_pr += f"{'Term':<25} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8} {'IRR':>8} {'95% CI':>16}\n"
            summary_pr += f"{'─' * 97}\n"
            for c in coefs_pr:
                sig = "<<COLOR:good>>*<</COLOR>>" if c["p"] < 0.05 else " "
                summary_pr += f"{c['name']:<25} {c['coef']:>8.4f} {c['se']:>8.4f} {c['z']:>8.3f} {c['p']:>8.4f} {c['irr']:>8.3f} [{c['irr_lo']:.3f}, {c['irr_hi']:.3f}] {sig}\n"

            summary_pr += f"\n<<COLOR:text>>Model Fit:<</COLOR>>\n"
            summary_pr += f"  Deviance: {dev_pr:.2f}  (df={df_resid_pr})\n"
            summary_pr += f"  Pearson χ²: {pearson_chi2_pr:.2f}\n"
            summary_pr += f"  Dispersion (Dev/df): {dispersion_pr:.3f}"
            if dispersion_pr > 1.5:
                summary_pr += f"  <<COLOR:warning>>⚠ Overdispersion detected — consider negative binomial<</COLOR>>"
            summary_pr += f"\n  AIC: {aic_pr:.1f}   BIC: {bic_pr:.1f}   Log-lik: {llf_pr:.1f}\n"

            result["summary"] = summary_pr

            # IRR forest plot
            non_intercept = [c for c in coefs_pr if c["name"] != "Intercept"]
            if non_intercept:
                result["plots"].append({
                    "title": "Incidence Rate Ratios (95% CI)",
                    "data": [{
                        "type": "scatter", "mode": "markers",
                        "x": [c["irr"] for c in non_intercept],
                        "y": [c["name"] for c in non_intercept],
                        "marker": {"color": ["#4a9f6e" if c["p"] < 0.05 else "#5a6a5a" for c in non_intercept], "size": 10},
                        "error_x": {
                            "type": "data", "symmetric": False,
                            "array": [c["irr_hi"] - c["irr"] for c in non_intercept],
                            "arrayminus": [c["irr"] - c["irr_lo"] for c in non_intercept],
                            "color": "#5a6a5a"
                        },
                        "showlegend": False,
                    }],
                    "layout": {
                        "height": max(250, 40 * len(non_intercept)),
                        "xaxis": {"title": "Incidence Rate Ratio", "type": "log"},
                        "yaxis": {"automargin": True},
                        "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": len(non_intercept) - 0.5,
                                    "line": {"color": "#e89547", "dash": "dash"}}]
                    }
                })

            # Deviance residuals vs fitted
            fitted_pr = model_pr.mu
            resid_dev_pr = model_pr.resid_deviance
            result["plots"].append({
                "title": "Deviance Residuals vs Fitted",
                "data": [{"type": "scatter", "mode": "markers",
                          "x": fitted_pr.tolist(), "y": resid_dev_pr.tolist(),
                          "marker": {"color": "#4a9f6e", "size": 4, "opacity": 0.6},
                          "showlegend": False}],
                "layout": {"height": 300, "xaxis": {"title": "Fitted values"},
                           "yaxis": {"title": "Deviance residuals"},
                           "shapes": [{"type": "line", "x0": min(fitted_pr), "x1": max(fitted_pr),
                                       "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}]}
            })

            n_sig_pr = sum(1 for c in coefs_pr if c["p"] < 0.05 and c["name"] != "Intercept")
            result["guide_observation"] = f"Poisson regression: {n_sig_pr}/{len(non_intercept)} predictors significant. Dispersion={dispersion_pr:.2f}."
            result["statistics"] = {
                "n": n_pr, "deviance": dev_pr, "df_resid": df_resid_pr,
                "pearson_chi2": pearson_chi2_pr, "dispersion": dispersion_pr,
                "aic": aic_pr, "bic": bic_pr, "log_likelihood": llf_pr,
                "coefficients": coefs_pr,
            }

        except Exception as e:
            result["summary"] = f"Poisson regression error: {str(e)}"

    # =====================================================================
    # Multiple Sampling Plan Comparison
    # =====================================================================
    elif analysis_id == "multiple_plan_comparison":
        """
        Multiple Acceptance Sampling Plan Comparison — compare OC curves, ASN, AOQ
        for several candidate plans side by side. Helps select the best plan.
        """
        from scipy import stats as mpc_stats

        plans_input = config.get("plans", [])
        # Each plan: {"name": "Plan A", "type": "single"/"double", "n": 50, "c": 2, ...}
        lot_size_mpc = int(config.get("lot_size", 1000))
        aql_mpc = float(config.get("aql", 0.01))
        ltpd_mpc = float(config.get("ltpd", 0.05))

        if not plans_input or len(plans_input) < 2:
            result["summary"] = "Provide at least 2 sampling plans to compare."
            return result

        p_range_mpc = np.linspace(0, min(0.20, ltpd_mpc * 3), 200)
        plan_colors = ["#4a9f6e", "#4a90d9", "#d94a4a", "#e8c547", "#9f4a4a", "#7a6a9a"]
        plan_results = []

        for idx, plan in enumerate(plans_input):
            plan_name = plan.get("name", f"Plan {idx + 1}")
            plan_type = plan.get("type", "single")
            n_mpc = int(plan.get("n", plan.get("sample_size", 50)))
            c_mpc = int(plan.get("c", plan.get("accept_number", 2)))

            # Compute OC curve
            pa_vals = np.array([float(mpc_stats.binom.cdf(c_mpc, n_mpc, p)) if p > 0 else 1.0 for p in p_range_mpc])

            # Key metrics
            pa_aql = float(np.interp(aql_mpc, p_range_mpc, pa_vals))
            pa_ltpd = float(np.interp(ltpd_mpc, p_range_mpc, pa_vals))
            alpha_risk = 1 - pa_aql
            beta_risk = pa_ltpd

            # AOQ and AOQL
            aoq_vals = pa_vals * p_range_mpc * (lot_size_mpc - n_mpc) / lot_size_mpc
            aoql = float(np.max(aoq_vals))

            # ATI at AQL
            ati_aql = n_mpc * pa_aql + lot_size_mpc * (1 - pa_aql)

            plan_results.append({
                "name": plan_name, "n": n_mpc, "c": c_mpc, "type": plan_type,
                "pa_values": pa_vals, "aoq_values": aoq_vals,
                "pa_aql": pa_aql, "pa_ltpd": pa_ltpd,
                "alpha": alpha_risk, "beta": beta_risk,
                "aoql": aoql, "ati_aql": ati_aql,
                "color": plan_colors[idx % len(plan_colors)],
            })

        # OC Curve comparison
        oc_traces = []
        for pr in plan_results:
            oc_traces.append({
                "x": (p_range_mpc * 100).tolist(), "y": pr["pa_values"].tolist(),
                "mode": "lines", "name": f"{pr['name']} (n={pr['n']}, c={pr['c']})",
                "line": {"color": pr["color"], "width": 2},
            })
        # AQL and LTPD reference lines
        oc_traces.append({"x": [aql_mpc * 100, aql_mpc * 100], "y": [0, 1], "mode": "lines",
                          "name": f"AQL ({aql_mpc*100:.1f}%)", "line": {"color": "#4a90d9", "dash": "dot"}})
        oc_traces.append({"x": [ltpd_mpc * 100, ltpd_mpc * 100], "y": [0, 1], "mode": "lines",
                          "name": f"LTPD ({ltpd_mpc*100:.1f}%)", "line": {"color": "#d94a4a", "dash": "dot"}})

        result["plots"].append({
            "title": "OC Curve Comparison",
            "data": oc_traces,
            "layout": {"height": 400, "xaxis": {"title": "Lot Defect Rate (%)"},
                       "yaxis": {"title": "P(Accept)", "range": [0, 1.05]}, "template": "plotly_white"}
        })

        # AOQ comparison
        aoq_traces = [{"x": (p_range_mpc * 100).tolist(), "y": (pr["aoq_values"] * 100).tolist(),
                       "mode": "lines", "name": pr["name"], "line": {"color": pr["color"], "width": 2}}
                      for pr in plan_results]
        result["plots"].append({
            "title": "AOQ Curve Comparison",
            "data": aoq_traces,
            "layout": {"height": 350, "xaxis": {"title": "Incoming Defect Rate (%)"},
                       "yaxis": {"title": "Average Outgoing Quality (%)"}, "template": "plotly_white"}
        })

        # Summary table
        summary_mpc = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_mpc += f"<<COLOR:title>>SAMPLING PLAN COMPARISON<</COLOR>>\n"
        summary_mpc += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_mpc += f"<<COLOR:highlight>>Lot size:<</COLOR>> {lot_size_mpc}\n"
        summary_mpc += f"<<COLOR:highlight>>AQL:<</COLOR>> {aql_mpc*100:.1f}%    <<COLOR:highlight>>LTPD:<</COLOR>> {ltpd_mpc*100:.1f}%\n\n"

        summary_mpc += f"{'Plan':<20} {'n':>5} {'c':>3} {'P(Acc@AQL)':>11} {'P(Acc@LTPD)':>12} {'α':>6} {'β':>6} {'AOQL%':>7} {'ATI@AQL':>8}\n"
        summary_mpc += f"{'─' * 82}\n"
        best_beta = min(pr["beta"] for pr in plan_results)
        for pr in plan_results:
            beta_mark = "<<COLOR:good>> ◄<</COLOR>>" if pr["beta"] == best_beta else "   "
            summary_mpc += f"{pr['name']:<20} {pr['n']:>5} {pr['c']:>3} {pr['pa_aql']:>11.4f} {pr['pa_ltpd']:>12.4f} {pr['alpha']:>6.3f} {pr['beta']:>6.3f} {pr['aoql']*100:>7.3f} {pr['ati_aql']:>8.0f}{beta_mark}\n"

        summary_mpc += f"\n<<COLOR:text>>◄ = lowest consumer risk (β)<</COLOR>>\n"
        result["summary"] = summary_mpc

        result["guide_observation"] = f"Compared {len(plan_results)} sampling plans. Best consumer risk: {best_beta:.3f}."
        result["statistics"] = {
            "plans": [{k: v for k, v in pr.items() if k not in ("pa_values", "aoq_values", "color")} for pr in plan_results],
            "aql": aql_mpc, "ltpd": ltpd_mpc, "lot_size": lot_size_mpc,
        }

    elif analysis_id == "capability_sixpack":
        """
        Process Capability Sixpack — 6-panel diagnostic display.
        Panels: I chart (or Xbar), MR chart (or R), last observations run chart,
                histogram with spec limits, normal probability plot, capability stats.
        Mirrors Minitab's capability sixpack layout.
        """
        var_cs = config.get("var") or config.get("var1")
        lsl_cs = config.get("lsl")
        usl_cs = config.get("usl")
        target_cs = config.get("target")
        subgroup_size = int(config.get("subgroup_size", 1))

        col_cs = df[var_cs].dropna().values.astype(float)
        n_cs = len(col_cs)

        if n_cs < 10:
            result["summary"] = "Need at least 10 observations for capability sixpack."
            return result

        lsl_val = float(lsl_cs) if lsl_cs is not None and str(lsl_cs).strip() != "" else None
        usl_val = float(usl_cs) if usl_cs is not None and str(usl_cs).strip() != "" else None
        target_val = float(target_cs) if target_cs is not None and str(target_cs).strip() != "" else None

        if lsl_val is None and usl_val is None:
            result["summary"] = "At least one specification limit (LSL or USL) is required."
            return result

        if target_val is None and lsl_val is not None and usl_val is not None:
            target_val = (lsl_val + usl_val) / 2

        x_bar_cs = float(np.mean(col_cs))
        s_cs = float(np.std(col_cs, ddof=1))

        from scipy.stats import norm as norm_dist

        # Capability indices
        cp_val = cpu_val = cpl_val = cpk_val = None
        if lsl_val is not None and usl_val is not None:
            cp_val = (usl_val - lsl_val) / (6 * s_cs) if s_cs > 0 else 0
            cpu_val = (usl_val - x_bar_cs) / (3 * s_cs) if s_cs > 0 else 0
            cpl_val = (x_bar_cs - lsl_val) / (3 * s_cs) if s_cs > 0 else 0
            cpk_val = min(cpu_val, cpl_val)
        elif usl_val is not None:
            cpu_val = (usl_val - x_bar_cs) / (3 * s_cs) if s_cs > 0 else 0
            cpk_val = cpu_val
        else:
            cpl_val = (x_bar_cs - lsl_val) / (3 * s_cs) if s_cs > 0 else 0
            cpk_val = cpl_val

        ppm_below = float(norm_dist.cdf((lsl_val - x_bar_cs) / s_cs) * 1e6) if lsl_val is not None and s_cs > 0 else 0
        ppm_above = float((1 - norm_dist.cdf((usl_val - x_bar_cs) / s_cs)) * 1e6) if usl_val is not None and s_cs > 0 else 0
        ppm_total = ppm_below + ppm_above

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PROCESS CAPABILITY SIXPACK -- {var_cs}<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Specifications:<</COLOR>>\n"
        if lsl_val is not None:
            summary += f"  LSL: {lsl_val}\n"
        if usl_val is not None:
            summary += f"  USL: {usl_val}\n"
        if target_val is not None:
            summary += f"  Target: {target_val}\n"
        summary += f"\n<<COLOR:text>>Process Stats:<</COLOR>>\n"
        summary += f"  N: {n_cs},  Mean: {x_bar_cs:.4f},  StDev: {s_cs:.4f}\n\n"
        summary += f"<<COLOR:text>>Capability Indices:<</COLOR>>\n"
        if cp_val is not None:
            summary += f"  Cp:  {cp_val:.3f}\n"
        summary += f"  Cpk: {cpk_val:.3f}\n"
        if cpl_val is not None:
            summary += f"  CPL: {cpl_val:.3f}\n"
        if cpu_val is not None:
            summary += f"  CPU: {cpu_val:.3f}\n"
        summary += f"\n<<COLOR:text>>Expected PPM:<</COLOR>>\n"
        if lsl_val is not None:
            summary += f"  Below LSL: {ppm_below:.1f}\n"
        if usl_val is not None:
            summary += f"  Above USL: {ppm_above:.1f}\n"
        summary += f"  Total:     {ppm_total:.1f}\n"
        if cpk_val >= 1.33:
            summary += f"\n<<COLOR:good>>Process is capable (Cpk >= 1.33).<</COLOR>>\n"
        elif cpk_val >= 1.0:
            summary += f"\n<<COLOR:warning>>Process is marginally capable (1.0 <= Cpk < 1.33).<</COLOR>>\n"
        else:
            summary += f"\n<<COLOR:warning>>Process is NOT capable (Cpk < 1.0).<</COLOR>>\n"

        result["summary"] = summary

        # Panel 1 & 2: I-MR or Xbar-R
        if subgroup_size == 1:
            mr = np.abs(np.diff(col_cs))
            mr_bar = float(np.mean(mr))
            sigma_est = mr_bar / 1.128
            cl_i = x_bar_cs
            ucl_i = x_bar_cs + 3 * sigma_est
            lcl_i = x_bar_cs - 3 * sigma_est
            result["plots"].append({
                "title": "I Chart",
                "data": [
                    {"type": "scatter", "y": col_cs.tolist(), "mode": "lines+markers",
                     "marker": {"size": 3, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e", "width": 1}, "name": var_cs},
                    {"type": "scatter", "y": [cl_i] * n_cs, "mode": "lines", "line": {"color": "#e8c547", "width": 1}, "name": f"CL={cl_i:.2f}"},
                    {"type": "scatter", "y": [ucl_i] * n_cs, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"UCL={ucl_i:.2f}"},
                    {"type": "scatter", "y": [lcl_i] * n_cs, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"LCL={lcl_i:.2f}"},
                ],
                "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30}, "showlegend": False}
            })
            mr_ucl = mr_bar * 3.267
            result["plots"].append({
                "title": "MR Chart",
                "data": [
                    {"type": "scatter", "y": mr.tolist(), "mode": "lines+markers",
                     "marker": {"size": 3, "color": "#47a5e8"}, "line": {"color": "#47a5e8", "width": 1}, "name": "MR"},
                    {"type": "scatter", "y": [mr_bar] * len(mr), "mode": "lines", "line": {"color": "#e8c547", "width": 1}},
                    {"type": "scatter", "y": [mr_ucl] * len(mr), "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}},
                ],
                "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30}, "showlegend": False}
            })
        else:
            n_sg = n_cs // subgroup_size
            subgroups = [col_cs[i * subgroup_size:(i + 1) * subgroup_size] for i in range(n_sg)]
            xbar_sg = [float(np.mean(sg)) for sg in subgroups]
            ranges_sg = [float(np.max(sg) - np.min(sg)) for sg in subgroups]
            xbar_bar = float(np.mean(xbar_sg))
            r_bar = float(np.mean(ranges_sg))
            A2_tbl = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
            D4_tbl = {2: 3.267, 3: 2.575, 4: 2.282, 5: 2.115, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
            D3_tbl = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
            a2 = A2_tbl.get(subgroup_size, 0.577)
            d4 = D4_tbl.get(subgroup_size, 2.115)
            d3 = D3_tbl.get(subgroup_size, 0)
            result["plots"].append({
                "title": "Xbar Chart",
                "data": [
                    {"type": "scatter", "y": xbar_sg, "mode": "lines+markers", "marker": {"size": 3, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e", "width": 1}},
                    {"type": "scatter", "y": [xbar_bar] * n_sg, "mode": "lines", "line": {"color": "#e8c547", "width": 1}},
                    {"type": "scatter", "y": [xbar_bar + a2 * r_bar] * n_sg, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}},
                    {"type": "scatter", "y": [xbar_bar - a2 * r_bar] * n_sg, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}},
                ],
                "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30}, "showlegend": False}
            })
            result["plots"].append({
                "title": "R Chart",
                "data": [
                    {"type": "scatter", "y": ranges_sg, "mode": "lines+markers", "marker": {"size": 3, "color": "#47a5e8"}, "line": {"color": "#47a5e8", "width": 1}},
                    {"type": "scatter", "y": [r_bar] * n_sg, "mode": "lines", "line": {"color": "#e8c547", "width": 1}},
                    {"type": "scatter", "y": [d4 * r_bar] * n_sg, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}},
                    {"type": "scatter", "y": [d3 * r_bar] * n_sg, "mode": "lines", "line": {"color": "#e85747", "dash": "dash", "width": 1}},
                ],
                "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30}, "showlegend": False}
            })

        # Panel 3: Last observations run chart with spec limits
        last_obs = col_cs[-min(25 * max(subgroup_size, 1), n_cs):]
        spec_shapes = []
        if lsl_val is not None:
            spec_shapes.append({"type": "line", "x0": 0, "x1": len(last_obs), "y0": lsl_val, "y1": lsl_val,
                                "line": {"color": "#e85747", "dash": "dot", "width": 1}})
        if usl_val is not None:
            spec_shapes.append({"type": "line", "x0": 0, "x1": len(last_obs), "y0": usl_val, "y1": usl_val,
                                "line": {"color": "#e85747", "dash": "dot", "width": 1}})
        result["plots"].append({
            "title": "Last Observations",
            "data": [{"type": "scatter", "y": last_obs.tolist(), "mode": "lines+markers",
                      "marker": {"size": 3, "color": "#9aaa9a"}, "line": {"color": "#9aaa9a", "width": 1}}],
            "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30},
                        "shapes": spec_shapes, "showlegend": False}
        })

        # Panel 4: Capability histogram
        hist_shapes = []
        hist_annot = []
        if lsl_val is not None:
            hist_shapes.append({"type": "line", "x0": lsl_val, "x1": lsl_val, "y0": 0, "y1": 1, "yref": "paper",
                                "line": {"color": "#e85747", "width": 2}})
            hist_annot.append({"x": lsl_val, "y": 1, "yref": "paper", "text": "LSL", "showarrow": False,
                               "font": {"color": "#e85747", "size": 10}})
        if usl_val is not None:
            hist_shapes.append({"type": "line", "x0": usl_val, "x1": usl_val, "y0": 0, "y1": 1, "yref": "paper",
                                "line": {"color": "#e85747", "width": 2}})
            hist_annot.append({"x": usl_val, "y": 1, "yref": "paper", "text": "USL", "showarrow": False,
                               "font": {"color": "#e85747", "size": 10}})
        if target_val is not None:
            hist_shapes.append({"type": "line", "x0": target_val, "x1": target_val, "y0": 0, "y1": 1, "yref": "paper",
                                "line": {"color": "#47a5e8", "dash": "dash", "width": 1}})
        result["plots"].append({
            "title": "Capability Histogram",
            "data": [{"type": "histogram", "x": col_cs.tolist(), "marker": {"color": "#4a9f6e", "opacity": 0.7},
                      "nbinsx": min(30, n_cs // 3)}],
            "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30},
                        "shapes": hist_shapes, "annotations": hist_annot, "showlegend": False}
        })

        # Panel 5: Normal probability plot
        sorted_cs = np.sort(col_cs)
        probs_cs = [(i + 0.5) / n_cs for i in range(n_cs)]
        theoretical_cs = [float(norm_dist.ppf(p)) for p in probs_cs]
        result["plots"].append({
            "title": "Normal Probability Plot",
            "data": [{"type": "scatter", "x": theoretical_cs, "y": sorted_cs.tolist(), "mode": "markers",
                      "marker": {"color": "#4a9f6e", "size": 3}}],
            "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30},
                        "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": var_cs}, "showlegend": False}
        })

        # Panel 6: Capability stats text
        stats_lines = []
        if cp_val is not None:
            stats_lines.append(f"Cp = {cp_val:.3f}")
        stats_lines.append(f"Cpk = {cpk_val:.3f}")
        if cpl_val is not None:
            stats_lines.append(f"CPL = {cpl_val:.3f}")
        if cpu_val is not None:
            stats_lines.append(f"CPU = {cpu_val:.3f}")
        stats_lines.append(f"PPM = {ppm_total:.0f}")
        stats_text_cs = "<br>".join(stats_lines)
        result["plots"].append({
            "title": "Capability Statistics",
            "data": [{"type": "scatter", "x": [0.5], "y": [0.5], "mode": "text",
                      "text": [stats_text_cs], "textfont": {"size": 14, "color": "#4a9f6e"}}],
            "layout": {"height": 200, "template": "plotly_dark", "margin": {"t": 30, "b": 30},
                        "xaxis": {"visible": False, "range": [0, 1]}, "yaxis": {"visible": False, "range": [0, 1]},
                        "showlegend": False}
        })

        result["guide_observation"] = f"Capability sixpack: Cpk={cpk_val:.3f}, PPM={ppm_total:.0f}."
        result["statistics"] = {
            "n": n_cs, "mean": x_bar_cs, "stdev": s_cs,
            "cp": cp_val, "cpk": cpk_val, "cpl": cpl_val, "cpu": cpu_val,
            "ppm_below": ppm_below, "ppm_above": ppm_above, "ppm_total": ppm_total,
            "lsl": lsl_val, "usl": usl_val, "target": target_val,
        }

    elif analysis_id == "anom":
        """
        Analysis of Means (ANOM) — compare each group mean to the overall mean.
        Uses Bonferroni-corrected t-limits as ANOM decision limits.
        Alternative to ANOVA: identifies *which* groups differ from grand mean.
        Supports factor format or multiple columns.
        """
        var_anom = config.get("var") or config.get("var1") or config.get("response")
        factor_anom = config.get("factor") or config.get("group_var") or config.get("var2")
        alpha_anom = 1 - float(config.get("conf", 95)) / 100

        if factor_anom and factor_anom in df.columns:
            data_anom = df[[var_anom, factor_anom]].dropna()
            groups_labels = sorted(data_anom[factor_anom].unique().tolist(), key=str)
            groups_data = [data_anom[data_anom[factor_anom] == g][var_anom].values.astype(float) for g in groups_labels]
        else:
            cols_anom = config.get("columns") or [c for c in df.select_dtypes(include=[np.number]).columns[:10]]
            groups_labels = [str(c) for c in cols_anom]
            groups_data = [df[c].dropna().values.astype(float) for c in cols_anom]

        k_anom = len(groups_data)
        if k_anom < 2:
            result["summary"] = "ANOM requires at least 2 groups."
            return result

        group_ns = [len(g) for g in groups_data]
        group_means = [float(np.mean(g)) for g in groups_data]
        all_data_anom = np.concatenate(groups_data)
        n_total = len(all_data_anom)
        grand_mean = float(np.mean(all_data_anom))

        ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups_data)
        df_within = n_total - k_anom
        mse_anom = ss_within / df_within if df_within > 0 else 0

        from scipy.stats import t as t_dist
        h_alpha = float(t_dist.ppf(1 - alpha_anom / (2 * k_anom), df_within))

        udls = []
        ldls = []
        for ni in group_ns:
            margin = h_alpha * np.sqrt(mse_anom * (1 - ni / n_total) / ni) if ni > 0 and n_total > ni else 0
            udls.append(grand_mean + margin)
            ldls.append(grand_mean - margin)

        balanced = len(set(group_ns)) == 1
        if balanced:
            udl_anom = udls[0]
            ldl_anom = ldls[0]

        outside = []
        for i, (mean_i, u, l) in enumerate(zip(group_means, udls, ldls)):
            if mean_i > u or mean_i < l:
                outside.append(groups_labels[i])

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ANALYSIS OF MEANS (ANOM)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {var_anom}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_anom or 'columns'}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {k_anom}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha_anom}\n\n"
        summary += f"<<COLOR:text>>Grand Mean:<</COLOR>> {grand_mean:.4f}\n"
        summary += f"<<COLOR:text>>MSE (within):<</COLOR>> {mse_anom:.4f}\n\n"
        summary += f"<<COLOR:text>>Group Results:<</COLOR>>\n"
        summary += f"  {'Group':<15} {'N':>5} {'Mean':>10} {'UDL':>10} {'LDL':>10} {'Signal':>8}\n"
        summary += f"  {'-' * 60}\n"
        for i in range(k_anom):
            sig = "*" if groups_labels[i] in outside else ""
            summary += f"  {str(groups_labels[i]):<15} {group_ns[i]:>5} {group_means[i]:>10.4f} {udls[i]:>10.4f} {ldls[i]:>10.4f} {sig:>8}\n"

        if outside:
            summary += f"\n<<COLOR:warning>>Groups outside decision limits: {', '.join(str(o) for o in outside)}<</COLOR>>\n"
        else:
            summary += f"\n<<COLOR:good>>All group means within decision limits.<</COLOR>>\n"

        result["summary"] = summary

        marker_colors = ["#e85747" if groups_labels[i] in outside else "#4a9f6e" for i in range(k_anom)]
        chart_data = [
            {"type": "scatter", "x": [str(g) for g in groups_labels], "y": group_means, "mode": "markers",
             "marker": {"color": marker_colors, "size": 10}, "name": "Group Means"},
            {"type": "scatter", "x": [str(groups_labels[0]), str(groups_labels[-1])],
             "y": [grand_mean, grand_mean], "mode": "lines",
             "line": {"color": "#e8c547", "width": 2}, "name": f"Grand Mean ({grand_mean:.3f})"},
        ]
        if balanced:
            chart_data.append({"type": "scatter", "x": [str(groups_labels[0]), str(groups_labels[-1])],
                               "y": [udl_anom, udl_anom], "mode": "lines",
                               "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"UDL ({udl_anom:.3f})"})
            chart_data.append({"type": "scatter", "x": [str(groups_labels[0]), str(groups_labels[-1])],
                               "y": [ldl_anom, ldl_anom], "mode": "lines",
                               "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"LDL ({ldl_anom:.3f})"})
        else:
            chart_data.append({"type": "scatter", "x": [str(g) for g in groups_labels], "y": udls, "mode": "lines+markers",
                               "marker": {"size": 3}, "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": "UDL"})
            chart_data.append({"type": "scatter", "x": [str(g) for g in groups_labels], "y": ldls, "mode": "lines+markers",
                               "marker": {"size": 3}, "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": "LDL"})

        result["plots"].append({
            "data": chart_data,
            "layout": {"title": "ANOM Chart", "xaxis": {"title": factor_anom or "Group"}, "yaxis": {"title": var_anom},
                        "template": "plotly_white", "height": 350}
        })

        result["guide_observation"] = f"ANOM: {len(outside)} of {k_anom} groups outside decision limits." if outside else f"ANOM: All {k_anom} groups within decision limits."
        result["statistics"] = {
            "grand_mean": grand_mean, "mse": mse_anom, "k": k_anom, "n_total": n_total,
            "group_means": {str(g): m for g, m in zip(groups_labels, group_means)},
            "outside_limits": [str(o) for o in outside], "alpha": alpha_anom,
        }

    # =====================================================================
    # Split-Plot ANOVA
    # =====================================================================
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
            ms_resid = anova_table.loc["Residual", "mean_sq"] if "Residual" in anova_table.index else anova_table.iloc[-1]["mean_sq"]
            df_resid_sp = anova_table.loc["Residual", "df"] if "Residual" in anova_table.index else anova_table.iloc[-1]["df"]

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
                is_wp = any(f in idx_row for f in whole_plot_factors_sp) and not any(f in idx_row for f in sub_plot_factors_sp)

                if is_wp and block_col_sp:
                    error_ms = wp_error_ms
                    error_df = wp_error_df
                    error_term = "WP Error"
                else:
                    error_ms = ms_resid
                    error_df = df_resid_sp
                    error_term = "Residual"

                f_val = ms / error_ms if error_ms > 0 else 0
                from scipy import stats as sp_stats
                p_val = 1 - sp_stats.f.cdf(f_val, df_val, error_df) if f_val > 0 else 1.0
                pct_contrib = ss / ss_total * 100

                anova_rows.append({
                    "source": clean_name, "ss": ss, "df": int(df_val), "ms": ms,
                    "f": f_val, "p": p_val, "pct": pct_contrib, "error_term": error_term,
                    "significant": p_val < 0.05,
                })

            # Add error rows
            if block_col_sp and f"C({block_col_sp})" in anova_table.index:
                anova_rows.append({"source": "Whole-Plot Error", "ss": float(wp_error_ms * wp_error_df),
                                   "df": int(wp_error_df), "ms": float(wp_error_ms), "f": None, "p": None, "pct": float(wp_error_ms * wp_error_df / ss_total * 100), "error_term": ""})
            anova_rows.append({"source": "Sub-Plot Error (Residual)", "ss": float(ms_resid * df_resid_sp),
                               "df": int(df_resid_sp), "ms": float(ms_resid), "f": None, "p": None, "pct": float(ms_resid * df_resid_sp / ss_total * 100), "error_term": ""})

            # Summary
            summary_sp = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_sp += f"<<COLOR:title>>SPLIT-PLOT ANOVA<</COLOR>>\n"
            summary_sp += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_sp += f"<<COLOR:highlight>>Response:<</COLOR>> {response_sp}\n"
            summary_sp += f"<<COLOR:highlight>>Whole-plot factors:<</COLOR>> {', '.join(whole_plot_factors_sp)}\n"
            summary_sp += f"<<COLOR:highlight>>Sub-plot factors:<</COLOR>> {', '.join(sub_plot_factors_sp)}\n"
            if block_col_sp:
                summary_sp += f"<<COLOR:highlight>>Block (whole-plot ID):<</COLOR>> {block_col_sp}\n"
            summary_sp += f"<<COLOR:highlight>>N:<</COLOR>> {len(data_sp)}\n\n"

            summary_sp += f"<<COLOR:text>>ANOVA Table:<</COLOR>>\n"
            summary_sp += f"{'Source':<30} {'SS':>10} {'df':>4} {'MS':>10} {'F':>8} {'p':>8} {'%Contrib':>8} {'Error Term':<12}\n"
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
            result["plots"].append({
                "title": "Residuals vs Fitted Values",
                "data": [{"type": "scatter", "mode": "markers",
                          "x": fitted_sp.tolist(), "y": resids_sp.tolist(),
                          "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}, "showlegend": False}],
                "layout": {"height": 300, "xaxis": {"title": "Fitted"}, "yaxis": {"title": "Residual"},
                           "shapes": [{"type": "line", "x0": min(fitted_sp), "x1": max(fitted_sp), "y0": 0, "y1": 0,
                                       "line": {"color": "#e89547", "dash": "dash"}}]}
            })

            # Main effects plot
            me_traces = []
            for fi, factor in enumerate(all_factors_sp):
                grp = data_sp.groupby(factor)[response_sp].mean()
                me_traces.append({
                    "x": [str(lv) for lv in grp.index], "y": grp.values.tolist(),
                    "mode": "lines+markers", "name": factor,
                    "marker": {"size": 8},
                })
            result["plots"].append({
                "title": "Main Effects Plot",
                "data": me_traces,
                "layout": {"height": 300, "xaxis": {"title": "Factor Level"}, "yaxis": {"title": f"Mean {response_sp}"}}
            })

            n_sig_sp = sum(1 for r in anova_rows if r.get("significant"))
            result["guide_observation"] = f"Split-plot ANOVA: {n_sig_sp} significant terms. WP factors tested against WP error, SP factors against residual."
            result["statistics"] = {"anova_table": anova_rows, "r_squared": float(model_sp.rsquared), "n": len(data_sp)}

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
            n_subj = len(subjects)

            if k_rm < 2:
                result["summary"] = "Need at least 2 levels of the within-subject factor."
                return result

            # Build subject × condition matrix
            pivot_rm = data_rm.pivot_table(index=subject_col, columns=within_factor,
                                           values=response_rm, aggfunc="mean")
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
                f_coeff = (2 * p_rm ** 2 + p_rm + 2) / (6 * p_rm * df_subjects)
                chi2_mauchly = -(df_subjects - f_coeff) * np.log(max(W_mauchly, 1e-15))
                df_mauchly = p_rm * (p_rm + 1) / 2 - 1
                p_mauchly = 1 - rm_stats.chi2.cdf(chi2_mauchly, df_mauchly) if df_mauchly > 0 else 1.0

                # Greenhouse-Geisser epsilon
                eigenvals_cov = np.linalg.eigvalsh(cov_diff)
                eigenvals_cov = eigenvals_cov[eigenvals_cov > 0]
                gg_epsilon = (np.sum(eigenvals_cov) ** 2) / (p_rm * np.sum(eigenvals_cov ** 2)) if len(eigenvals_cov) > 0 else 1.0
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
            summary_rm += f"<<COLOR:title>>REPEATED MEASURES ANOVA<</COLOR>>\n"
            summary_rm += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_rm += f"<<COLOR:highlight>>Response:<</COLOR>> {response_rm}\n"
            summary_rm += f"<<COLOR:highlight>>Within-subject factor:<</COLOR>> {within_factor} ({k_rm} levels)\n"
            summary_rm += f"<<COLOR:highlight>>Subjects:<</COLOR>> {n_complete}\n\n"

            summary_rm += f"<<COLOR:text>>Within-Subjects ANOVA:<</COLOR>>\n"
            summary_rm += f"{'Source':<25} {'SS':>10} {'df':>4} {'MS':>10} {'F':>8} {'p':>8}\n"
            summary_rm += f"{'─' * 70}\n"
            sig_mark = " <<COLOR:good>>*<</COLOR>>" if p_val_rm < 0.05 else ""
            summary_rm += f"{'Condition':<25} {ss_condition:>10.2f} {df_condition:>4} {ms_condition:>10.3f} {f_val_rm:>8.3f} {p_val_rm:>8.4f}{sig_mark}\n"
            summary_rm += f"{'Error':<25} {ss_error_rm:>10.2f} {df_error_rm:>4} {ms_error_rm:>10.3f}\n"

            if k_rm > 2:
                summary_rm += f"\n<<COLOR:text>>Mauchly's Test of Sphericity:<</COLOR>>\n"
                summary_rm += f"  W = {W_mauchly:.4f},  χ² = {chi2_mauchly:.3f},  p = {p_mauchly:.4f}\n"
                if p_mauchly < 0.05:
                    summary_rm += f"  <<COLOR:warning>>Sphericity violated — use corrected tests below<</COLOR>>\n"
                else:
                    summary_rm += f"  <<COLOR:good>>Sphericity assumption met<</COLOR>>\n"

                summary_rm += f"\n<<COLOR:text>>Epsilon Corrections:<</COLOR>>\n"
                summary_rm += f"  Greenhouse-Geisser ε = {gg_epsilon:.4f}  →  p = {p_gg:.4f}\n"
                summary_rm += f"  Huynh-Feldt ε = {hf_epsilon:.4f}  →  p = {p_hf:.4f}\n"

            # Condition means
            summary_rm += f"\n<<COLOR:text>>Condition Means:<</COLOR>>\n"
            for ci, cond in enumerate(pivot_rm.columns):
                summary_rm += f"  {cond}: {cond_means[ci]:.4f} (SD = {np.std(Y_rm[:, ci], ddof=1):.4f})\n"

            # Partial eta-squared
            eta_sq = ss_condition / (ss_condition + ss_error_rm) if (ss_condition + ss_error_rm) > 0 else 0
            summary_rm += f"\n<<COLOR:text>>Effect Size:<</COLOR>> partial η² = {eta_sq:.4f}\n"

            result["summary"] = summary_rm

            # Profile plot (condition means with SE)
            se_conds = np.std(Y_rm, axis=0, ddof=1) / np.sqrt(n_complete)
            result["plots"].append({
                "title": "Profile Plot — Condition Means (±SE)",
                "data": [{
                    "x": [str(c) for c in pivot_rm.columns], "y": cond_means.tolist(),
                    "mode": "lines+markers", "name": "Mean",
                    "marker": {"color": "#4a9f6e", "size": 10}, "line": {"color": "#4a9f6e", "width": 2},
                    "error_y": {"type": "data", "array": se_conds.tolist(), "visible": True, "color": "#5a6a5a"},
                }],
                "layout": {"height": 300, "xaxis": {"title": within_factor}, "yaxis": {"title": f"Mean {response_rm}"}}
            })

            # Individual subject trajectories (spaghetti plot)
            spaghetti_traces = []
            for si, subj in enumerate(pivot_rm.index[:30]):  # limit to 30 subjects for readability
                spaghetti_traces.append({
                    "x": [str(c) for c in pivot_rm.columns], "y": pivot_rm.loc[subj].tolist(),
                    "mode": "lines", "line": {"color": "#5a6a5a", "width": 0.5}, "opacity": 0.3,
                    "showlegend": False,
                })
            # Overlay mean
            spaghetti_traces.append({
                "x": [str(c) for c in pivot_rm.columns], "y": cond_means.tolist(),
                "mode": "lines+markers", "name": "Grand Mean",
                "marker": {"color": "#4a9f6e", "size": 8}, "line": {"color": "#4a9f6e", "width": 3},
            })
            result["plots"].append({
                "title": "Subject Trajectories (spaghetti plot)",
                "data": spaghetti_traces,
                "layout": {"height": 300, "xaxis": {"title": within_factor}, "yaxis": {"title": response_rm}}
            })

            best_p = p_gg if (k_rm > 2 and p_mauchly < 0.05) else p_val_rm
            result["guide_observation"] = f"Repeated measures ANOVA: F({df_condition},{df_error_rm})={f_val_rm:.3f}, p={best_p:.4f}, η²={eta_sq:.4f}."
            result["statistics"] = {
                "f_value": f_val_rm, "p_value": p_val_rm, "df_condition": df_condition,
                "df_error": df_error_rm, "ss_condition": ss_condition, "ss_error": ss_error_rm,
                "partial_eta_squared": eta_sq, "n_subjects": n_complete, "k_levels": k_rm,
                "mauchly_w": float(W_mauchly), "mauchly_p": float(p_mauchly),
                "gg_epsilon": float(gg_epsilon), "hf_epsilon": float(hf_epsilon),
                "p_gg": float(p_gg), "p_hf": float(p_hf),
                "condition_means": {str(c): float(m) for c, m in zip(pivot_rm.columns, cond_means)},
            }

        except Exception as e:
            result["summary"] = f"Repeated measures ANOVA error: {str(e)}"

    # ── Power & Sample Size Calculators ──────────────────────────────────────

    elif analysis_id == "power_z":
        """
        Power / sample size for 1-sample Z-test.
        Given delta, sigma, alpha, and power → required n. Also produces a power curve.
        """
        delta = float(config.get("delta", 0.5))
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        d = abs(delta) / sigma  # standardised effect
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)
        n_req = math.ceil(((z_a + z_b) / d) ** 2) if d > 0 else 9999

        # Power curve over n
        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            ncp = d * math.sqrt(nn)
            if alt == "two-sided":
                pw = 1 - norm.cdf(z_a - ncp) + norm.cdf(-z_a - ncp)
            else:
                pw = 1 - norm.cdf(z_a - ncp)
            powers.append(pw)

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Sample Z (δ={delta}, σ={sigma})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Sample Z-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>Effect: δ = {delta}, σ = {sigma} → Cohen's d = {d:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, alternative = {alt}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-sample Z power: need n={n_req} for d={d:.3f} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "effect_size_d": d, "alpha": alpha, "power": target_power, "delta": delta, "sigma": sigma}

    elif analysis_id == "power_1prop":
        """
        Power / sample size for 1-proportion test.
        Given p0 (null), pa (alt), alpha, power → n.
        """
        p0 = float(config.get("p0", 0.5))
        pa = float(config.get("pa", 0.6))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)

        # Fleiss formula
        n_req = math.ceil(((z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(pa * (1 - pa))) / (pa - p0)) ** 2) if pa != p0 else 9999

        # Cohen's h
        h = abs(2 * (math.asin(math.sqrt(pa)) - math.asin(math.sqrt(p0))))

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            se0 = math.sqrt(p0 * (1 - p0) / nn)
            sea = math.sqrt(pa * (1 - pa) / nn)
            if alt == "two-sided":
                z_crit = z_a * se0
                pw = 1 - norm.cdf((p0 + z_crit - pa) / sea) + norm.cdf((p0 - z_crit - pa) / sea)
            elif alt == "greater":
                z_crit = z_a * se0
                pw = 1 - norm.cdf((p0 + z_crit - pa) / sea)
            else:
                z_crit = z_a * se0
                pw = norm.cdf((p0 - z_crit - pa) / sea)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Proportion (p₀={p0}, pₐ={pa})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: p = {p0}, H₁: p = {pa} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-prop power: need n={n_req} to detect p={pa} vs p₀={p0} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "p0": p0, "pa": pa, "cohens_h": h, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_2prop":
        """
        Power / sample size for 2-proportion test.
        Given p1, p2, alpha, power → n per group.
        """
        p1 = float(config.get("p1", 0.5))
        p2 = float(config.get("p2", 0.6))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        ratio = float(config.get("ratio", 1.0))  # n2/n1
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)
        p_bar = (p1 + ratio * p2) / (1 + ratio)
        h = abs(2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

        numer = z_a * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
        n1 = math.ceil((numer / (p1 - p2)) ** 2) if p1 != p2 else 9999
        n2 = math.ceil(n1 * ratio)

        ns = list(range(2, max(n1 * 3, 50)))
        powers = []
        for nn in ns:
            nn2 = max(2, int(nn * ratio))
            se = math.sqrt(p1 * (1 - p1) / nn + p2 * (1 - p2) / nn2)
            if se > 0:
                z_stat = abs(p1 - p2) / se
                if alt == "two-sided":
                    pw = 1 - norm.cdf(z_a - z_stat) + norm.cdf(-z_a - z_stat)
                else:
                    pw = 1 - norm.cdf(z_a - z_stat)
            else:
                pw = 1.0
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n1], "y": [target_power], "mode": "markers", "name": f"n₁={n1}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 2-Proportion (p₁={p1}, p₂={p2})", "xaxis": {"title": "Sample Size per Group (n₁)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>p₁ = {p1}, p₂ = {p2} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, ratio n₂/n₁ = {ratio}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n1}, n₂ = {n2} (total = {n1 + n2})<</COLOR>>\n"
        )
        result["guide_observation"] = f"2-prop power: need n₁={n1}, n₂={n2} for |Δp|={abs(p1 - p2):.3f} at power={target_power}."
        result["statistics"] = {"n1": n1, "n2": n2, "total_n": n1 + n2, "p1": p1, "p2": p2, "cohens_h": h, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_1variance":
        """
        Power / sample size for 1-variance chi-square test.
        Tests H₀: σ² = σ₀² vs H₁: σ² = σ₁².
        """
        sigma0 = float(config.get("sigma0", 1.0))
        sigma1 = float(config.get("sigma1", 1.5))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import chi2 as chi2_dist
        ratio_sq = (sigma1 / sigma0) ** 2  # variance ratio σ₁²/σ₀²

        # Iterative search for n
        n_req = 2
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
        else:
            n_req = 10000

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            df_val = nn - 1
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
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Variance (σ₀={sigma0}, σ₁={sigma1})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Variance Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ = {sigma0}, H₁: σ = {sigma1} → ratio σ₁²/σ₀² = {ratio_sq:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-variance power: need n={n_req} to detect σ₁={sigma1} vs σ₀={sigma0} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "sigma0": sigma0, "sigma1": sigma1, "variance_ratio": ratio_sq, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_2variance":
        """
        Power / sample size for 2-variance F-test.
        Tests H₀: σ₁² = σ₂² vs H₁: σ₁²/σ₂² = ratio.
        """
        var_ratio = float(config.get("variance_ratio", 2.0))  # σ₁²/σ₂²
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        ratio_n = float(config.get("ratio", 1.0))  # n2/n1
        alt = config.get("alternative", "two-sided")

        from scipy.stats import f as f_dist
        n_req = 2
        for n_try in range(2, 10000):
            n2_try = max(2, int(n_try * ratio_n))
            df1 = n_try - 1
            df2 = n2_try - 1
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
        else:
            n_req = 10000
        n2_req = max(2, int(n_req * ratio_n))

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            nn2 = max(2, int(nn * ratio_n))
            df1 = nn - 1
            df2 = nn2 - 1
            if alt == "two-sided":
                f_lo = f_dist.ppf(alpha / 2, df1, df2)
                f_hi = f_dist.ppf(1 - alpha / 2, df1, df2)
                pw = f_dist.cdf(f_lo / var_ratio, df1, df2) + 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            else:
                f_hi = f_dist.ppf(1 - alpha, df1, df2)
                pw = 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n₁={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 2-Variance F-Test (ratio={var_ratio})", "xaxis": {"title": "Sample Size per Group (n₁)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Variance F-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ₁²/σ₂² = 1, H₁: σ₁²/σ₂² = {var_ratio}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, n₂/n₁ = {ratio_n}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n_req}, n₂ = {n2_req} (total = {n_req + n2_req})<</COLOR>>\n"
        )
        result["guide_observation"] = f"2-variance power: need n₁={n_req}, n₂={n2_req} for ratio={var_ratio} at power={target_power}."
        result["statistics"] = {"n1": n_req, "n2": n2_req, "total_n": n_req + n2_req, "variance_ratio": var_ratio, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_equivalence":
        """
        Power / sample size for equivalence test (TOST).
        Two one-sided tests to establish equivalence within ±margin.
        """
        delta = float(config.get("delta", 0.0))  # true difference
        margin = float(config.get("margin", 0.5))  # equivalence margin
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))

        from scipy.stats import norm
        z_a = norm.ppf(1 - alpha)

        # For TOST, power = P(reject both one-sided tests)
        n_req = 2
        for n_try in range(2, 10000):
            se = sigma * math.sqrt(2 / n_try)
            # Power of TOST ≈ Φ((margin - |delta|)/se - z_a)
            pw = norm.cdf((margin - abs(delta)) / se - z_a)
            if pw >= target_power:
                n_req = n_try
                break
        else:
            n_req = 10000

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            se = sigma * math.sqrt(2 / nn)
            pw = norm.cdf((margin - abs(delta)) / se - z_a)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n/group={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — Equivalence (TOST, margin=±{margin})", "xaxis": {"title": "Sample Size per Group"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — Equivalence Test (TOST)<</COLOR>>\n\n"
            f"<<COLOR:text>>Equivalence margin: ±{margin}, true difference: {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n = {n_req} per group (total = {2 * n_req})<</COLOR>>\n"
        )
        result["guide_observation"] = f"Equivalence power: need n={n_req}/group for margin=±{margin}, δ={delta} at power={target_power}."
        result["statistics"] = {"n_per_group": n_req, "total_n": 2 * n_req, "margin": margin, "delta": delta, "sigma": sigma, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_doe":
        """
        Power / sample size for 2-level factorial DOE.
        Given number of factors, effect size, sigma, reps → power.
        Or: given target power → required replicates.
        """
        n_factors = int(config.get("factors", 3))
        delta = float(config.get("delta", 1.0))  # minimum detectable effect
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))

        from scipy.stats import norm, t as t_dist
        n_runs_base = 2 ** n_factors  # full factorial runs

        # Find required replicates
        req_reps = 1
        for reps in range(1, 100):
            n_total = n_runs_base * reps
            df_error = n_total - n_runs_base  # error df = N - p
            if df_error < 1:
                continue
            # Each effect estimated from N/2 + vs N/2 - observations
            se = sigma * math.sqrt(4.0 / n_total)  # SE of effect estimate
            if se <= 0:
                continue
            t_crit = t_dist.ppf(1 - alpha / 2, df_error)
            ncp = abs(delta) / se
            pw = 1 - t_dist.cdf(t_crit - ncp, df_error) + t_dist.cdf(-t_crit - ncp, df_error)
            if pw >= target_power:
                req_reps = reps
                break
        else:
            req_reps = 100

        # Power curve over replicates
        reps_range = list(range(1, max(req_reps * 3, 10)))
        powers = []
        for reps in reps_range:
            n_total = n_runs_base * reps
            df_error = n_total - n_runs_base
            if df_error < 1:
                powers.append(0.0)
                continue
            se = sigma * math.sqrt(4.0 / n_total)
            t_crit = t_dist.ppf(1 - alpha / 2, df_error)
            ncp = abs(delta) / se
            pw = 1 - t_dist.cdf(t_crit - ncp, df_error) + t_dist.cdf(-t_crit - ncp, df_error)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": reps_range, "y": powers, "mode": "lines+markers", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [req_reps], "y": [target_power], "mode": "markers", "name": f"reps={req_reps}", "marker": {"size": 12, "color": "#d94a4a", "symbol": "star"}}
            ],
            "layout": {"title": f"DOE Power — 2^{n_factors} Factorial (Δ={delta}, σ={sigma})", "xaxis": {"title": "Number of Replicates", "dtick": 1}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        n_total_req = n_runs_base * req_reps
        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2^{n_factors} Factorial DOE<</COLOR>>\n\n"
            f"<<COLOR:text>>Factors: {n_factors}, base runs: {n_runs_base}<</COLOR>>\n"
            f"<<COLOR:text>>Minimum detectable effect: Δ = {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required replicates: {req_reps} → total runs = {n_total_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"DOE power: 2^{n_factors} design needs {req_reps} reps ({n_total_req} runs) to detect Δ={delta} at power={target_power}."
        result["statistics"] = {"factors": n_factors, "base_runs": n_runs_base, "required_reps": req_reps, "total_runs": n_total_req, "delta": delta, "sigma": sigma, "alpha": alpha, "power": target_power}

    elif analysis_id == "sample_size_ci":
        """
        Sample size for estimation — determine n needed for a target CI half-width.
        Supports mean (Z or t) and proportion.
        """
        est_type = config.get("type", "mean")  # mean or proportion
        conf_level = float(config.get("conf", 95)) / 100 if float(config.get("conf", 95)) > 1 else float(config.get("conf", 95))
        target_width = float(config.get("half_width", 1.0))  # desired CI half-width (margin of error)

        from scipy.stats import norm
        z = norm.ppf(1 - (1 - conf_level) / 2)

        if est_type == "proportion":
            p_est = float(config.get("p_est", 0.5))  # expected proportion
            n_req = math.ceil((z ** 2 * p_est * (1 - p_est)) / (target_width ** 2))

            # Width curve
            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * math.sqrt(p_est * (1 - p_est) / nn) for nn in ns]

            result["plots"].append({
                "data": [
                    {"x": ns, "y": widths, "mode": "lines", "name": "CI Half-Width", "line": {"color": "#4a90d9", "width": 2}},
                    {"x": [n_req], "y": [target_width], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}},
                    {"x": ns, "y": [target_width] * len(ns), "mode": "lines", "name": "Target", "line": {"color": "#d94a4a", "dash": "dash", "width": 1}}
                ],
                "layout": {"title": f"CI Half-Width vs n — Proportion (p≈{p_est})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "Half-Width"}, "template": "plotly_white"}
            })

            extra_text = f"<<COLOR:text>>Expected proportion: p ≈ {p_est}<</COLOR>>\n"
        else:
            sigma = float(config.get("sigma", 1.0))
            n_req = math.ceil((z * sigma / target_width) ** 2)

            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * sigma / math.sqrt(nn) for nn in ns]

            result["plots"].append({
                "data": [
                    {"x": ns, "y": widths, "mode": "lines", "name": "CI Half-Width", "line": {"color": "#4a90d9", "width": 2}},
                    {"x": [n_req], "y": [target_width], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}},
                    {"x": ns, "y": [target_width] * len(ns), "mode": "lines", "name": "Target", "line": {"color": "#d94a4a", "dash": "dash", "width": 1}}
                ],
                "layout": {"title": f"CI Half-Width vs n — Mean (σ={sigma})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "Half-Width"}, "template": "plotly_white"}
            })
            extra_text = f"<<COLOR:text>>Population σ = {sigma}<</COLOR>>\n"

        n_req = max(n_req, 2)
        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Estimation ({est_type.title()})<</COLOR>>\n\n"
            f"<<COLOR:text>>Confidence level: {conf_level:.0%}, target half-width (margin of error): {target_width}<</COLOR>>\n"
            f"{extra_text}\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"Sample size for {conf_level:.0%} CI with half-width={target_width}: n={n_req}."
        result["statistics"] = {"required_n": n_req, "type": est_type, "confidence": conf_level, "half_width": target_width}

    elif analysis_id == "sample_size_tolerance":
        """
        Sample size for tolerance intervals.
        Given coverage (e.g. 99% of population), confidence (e.g. 95%), → n.
        Uses the Howe (1969) / Odeh-Owen factor approach.
        """
        coverage = float(config.get("coverage", 0.99))
        confidence = float(config.get("confidence", 0.95))
        interval_type = config.get("type", "two-sided")  # two-sided or one-sided

        from scipy.stats import norm, chi2 as chi2_dist
        z_p = norm.ppf((1 + coverage) / 2) if interval_type == "two-sided" else norm.ppf(coverage)

        # Iterative: find n so that k*sigma covers coverage proportion at given confidence
        # Using the non-central chi-square approach
        n_req = 2
        for n_try in range(2, 10000):
            df = n_try - 1
            # k factor: k = z_p * sqrt(n * (df) / chi2_lower)
            chi2_lo = chi2_dist.ppf(1 - confidence, df)
            k = z_p * math.sqrt(n_try * df / chi2_lo) / math.sqrt(df) if chi2_lo > 0 else 999
            # The sample k factor shrinks as n grows; we need k computable
            # Simplified: n must satisfy z_p * sqrt(1 + 1/n) * sqrt(df/chi2_lo) ≤ achievable
            # Actually for tolerance intervals: n needed so k-factor is finite
            # Howe's approach: n ≈ (z_p / E)^2 * (1 + 1/(2*df)) ... but let's use a direct criterion:
            # Check if the tolerance factor k satisfies coverage at confidence level
            chi2_lo_check = chi2_dist.ppf(1 - confidence, df)
            if chi2_lo_check > 0:
                k_factor = z_p * math.sqrt(n_try / chi2_lo_check)
                # Tolerance interval width: k_factor * s; guaranteed coverage at confidence
                # We want k_factor to be ≤ a reasonable bound (converges as n → ∞)
                # k_factor → z_p as n → ∞. We need n large enough that k_factor is close to z_p
                # Criterion: k_factor < z_p * 1.1 (within 10%)
                if k_factor <= z_p * 1.10:
                    n_req = n_try
                    break
        else:
            n_req = 10000

        # k-factor curve
        ns = list(range(2, max(n_req * 3, 100)))
        k_factors = []
        for nn in ns:
            df = nn - 1
            chi2_lo_val = chi2_dist.ppf(1 - confidence, df)
            if chi2_lo_val > 0:
                k_factors.append(z_p * math.sqrt(nn / chi2_lo_val))
            else:
                k_factors.append(float('nan'))

        result["plots"].append({
            "data": [
                {"x": ns, "y": k_factors, "mode": "lines", "name": "k-factor", "line": {"color": "#4a90d9", "width": 2}},
                {"x": ns, "y": [z_p] * len(ns), "mode": "lines", "name": f"z_p={z_p:.3f} (asymptote)", "line": {"color": "#4a9f6e", "dash": "dash", "width": 1}},
                {"x": [n_req], "y": [z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1)) if chi2_dist.ppf(1 - confidence, n_req - 1) > 0 else z_p], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Tolerance k-Factor vs n ({coverage:.0%}/{confidence:.0%})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "k-Factor"}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Tolerance Interval<</COLOR>>\n\n"
            f"<<COLOR:text>>Coverage: {coverage:.1%} of population<</COLOR>>\n"
            f"<<COLOR:text>>Confidence: {confidence:.1%}<</COLOR>>\n"
            f"<<COLOR:text>>Type: {interval_type}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
            f"<<COLOR:text>>At this n, the tolerance k-factor ≈ {z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1)):.3f} (asymptote = {z_p:.3f})<</COLOR>>\n"
        )
        result["guide_observation"] = f"Tolerance interval ({coverage:.0%}/{confidence:.0%}): need n={n_req}."
        result["statistics"] = {"required_n": n_req, "coverage": coverage, "confidence": confidence, "type": interval_type, "z_p": z_p}

    elif analysis_id == "granger":
        """
        Granger Causality Test - does X help predict Y?
        Tests if past values of X improve prediction of Y beyond Y's own past.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        var_x = config.get("var_x")
        var_y = config.get("var_y")
        max_lag = int(config.get("max_lag", 4))

        # Prepare data - must be stationary ideally
        x = df[var_x].dropna().values
        y = df[var_y].dropna().values

        # Align lengths
        min_len = min(len(x), len(y))
        data_matrix = np.column_stack([y[:min_len], x[:min_len]])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GRANGER CAUSALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Question:<</COLOR>> Does {var_x} Granger-cause {var_y}?\n"
        summary += f"<<COLOR:highlight>>Max Lags:<</COLOR>> {max_lag}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {min_len}\n\n"

        summary += f"<<COLOR:text>>Interpretation:<</COLOR>>\n"
        summary += f"  If p < 0.05, past values of {var_x} help predict {var_y}\n"
        summary += f"  beyond what {var_y}'s own past provides.\n\n"

        try:
            # Run Granger test (suppress output)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            gc_results = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)
            sys.stdout = old_stdout

            summary += f"<<COLOR:text>>Results by Lag:<</COLOR>>\n"
            summary += f"{'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significant':<12}\n"
            summary += f"{'-'*42}\n"

            significant_lags = []
            p_values = []
            f_stats = []

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    test_result = gc_results[lag][0]['ssr_ftest']
                    f_stat = test_result[0]
                    p_val = test_result[1]
                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

                    f_stats.append(f_stat)
                    p_values.append(p_val)

                    if p_val < 0.05:
                        significant_lags.append(lag)

                    summary += f"{lag:<6} {f_stat:<12.4f} {p_val:<12.4f} {sig}\n"

            summary += f"\n<<COLOR:accent>>────────────────────────────────────────<</COLOR>>\n"
            if significant_lags:
                summary += f"<<COLOR:good>>CONCLUSION: {var_x} Granger-causes {var_y}<</COLOR>>\n"
                summary += f"<<COLOR:text>>Significant at lags: {significant_lags}<</COLOR>>\n"
                summary += f"<<COLOR:text>>This suggests {var_x} has predictive power for {var_y}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>CONCLUSION: No Granger causality detected<</COLOR>>\n"
                summary += f"<<COLOR:text>>{var_x} does not significantly improve prediction of {var_y}.<</COLOR>>\n"

            result["summary"] = summary

            # Plot: p-values by lag
            result["plots"].append({
                "title": f"Granger Causality: {var_x} → {var_y}",
                "data": [{
                    "type": "bar",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": p_values,
                    "marker": {
                        "color": ["#e85747" if p < 0.05 else "rgba(74, 159, 110, 0.4)" for p in p_values],
                        "line": {"color": "#4a9f6e", "width": 1.5}
                    }
                }, {
                    "type": "scatter",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": [0.05] * len(p_values),
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": "α = 0.05"
                }],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": "p-value", "range": [0, max(0.2, max(p_values) * 1.1)]},
                    "xaxis": {"title": "Lag"}
                }
            })

            # Time series plot
            result["plots"].append({
                "title": f"Time Series: {var_x} and {var_y}",
                "data": [{
                    "type": "scatter",
                    "y": x[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_x,
                    "line": {"color": "#4a9f6e"}
                }, {
                    "type": "scatter",
                    "y": y[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_y,
                    "yaxis": "y2",
                    "line": {"color": "#47a5e8"}
                }],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": var_x, "side": "left"},
                    "yaxis2": {"title": var_y, "side": "right", "overlaying": "y"},
                    "xaxis": {"title": "Observation"}
                }
            })

            result["guide_observation"] = f"Granger causality: {var_x} → {var_y} " + (f"significant at lags {significant_lags}." if significant_lags else "not significant.")

            # Statistics for Synara
            result["statistics"] = {
                f"granger_min_pvalue": float(min(p_values)),
                f"granger_significant_lags": len(significant_lags),
                f"granger_causal": 1 if significant_lags else 0
            }

        except Exception as e:
            result["summary"] = f"Granger causality test failed: {str(e)}\n\nEnsure both variables are numeric and have sufficient observations."

    elif analysis_id == "changepoint":
        """
        Change Point Detection - when did the process shift?
        Uses PELT algorithm to find optimal change points.
        """
        var = config.get("var")
        penalty = config.get("penalty", "bic")
        min_size = int(config.get("min_size", 10))

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> PELT with {penalty.upper()} penalty\n\n"

        try:
            import ruptures as rpt

            # Use PELT algorithm (optimal for multiple change points)
            model = rpt.Pelt(model="rbf", min_size=min_size).fit(data)

            # Determine penalty value
            if penalty == "bic":
                pen = np.log(n) * np.var(data)
            elif penalty == "aic":
                pen = 2 * np.var(data)
            else:
                pen = float(penalty) if penalty.replace('.','').isdigit() else np.log(n) * np.var(data)

            change_points = model.predict(pen=pen)

            # Remove the last point (always equals n)
            if change_points and change_points[-1] == n:
                change_points = change_points[:-1]

            summary += f"<<COLOR:text>>Change Points Detected:<</COLOR>> {len(change_points)}\n\n"

            if change_points:
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
                summary += f"<<COLOR:text>>{'Index':<10} {'Value at CP':<15} {'Segment Mean':<15}<</COLOR>>\n"
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"

                segments = [0] + change_points + [n]
                for i, cp in enumerate(change_points):
                    seg_start = segments[i]
                    seg_end = cp
                    seg_mean = np.mean(data[seg_start:seg_end])
                    summary += f"{cp:<10} {data[cp-1]:<15.4f} {seg_mean:<15.4f}\n"

                # Final segment
                final_mean = np.mean(data[change_points[-1]:])
                summary += f"\n<<COLOR:text>>Final segment mean: {final_mean:.4f}<</COLOR>>\n"

                summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
                summary += f"  These points mark where the process behavior shifted.\n"
                summary += f"  Investigate what changed at these times.\n"
            else:
                summary += f"<<COLOR:text>>No significant change points detected.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The process appears stable over this period.<</COLOR>>\n"

            result["summary"] = summary

            # Plot with change points
            plot_data = [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines",
                "name": var,
                "line": {"color": "#4a9f6e", "width": 1.5}
            }]

            # Add vertical lines for change points
            shapes = []
            for cp in change_points:
                shapes.append({
                    "type": "line",
                    "x0": cp,
                    "x1": cp,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2}
                })

            # Add segment means as horizontal lines
            segments = [0] + list(change_points) + [n]
            for i in range(len(segments) - 1):
                seg_mean = np.mean(data[segments[i]:segments[i+1]])
                plot_data.append({
                    "type": "scatter",
                    "x": [segments[i], segments[i+1]-1],
                    "y": [seg_mean, seg_mean],
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2},
                    "name": f"Segment {i+1} mean" if i == 0 else None,
                    "showlegend": i == 0
                })

            result["plots"].append({
                "title": f"Change Points: {var}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "shapes": shapes,
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": var}
                }
            })

            result["guide_observation"] = f"Detected {len(change_points)} change point(s) in {var}." + (" Investigate what caused these shifts." if change_points else " Process appears stable.")

            # Statistics for Synara
            result["statistics"] = {
                "change_points_count": len(change_points),
                "change_point_indices": change_points if change_points else []
            }

        except ImportError:
            result["summary"] = "Change point detection requires the 'ruptures' package.\n\nInstall with: pip install ruptures"
        except Exception as e:
            result["summary"] = f"Change point detection failed: {str(e)}"

    elif analysis_id == "mann_whitney":
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

        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MANN-WHITNEY U TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        summary += f"  {groups[0]}: median = {group1.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[:n1].mean():.1f}\n"
        summary += f"  {groups[1]}: median = {group2.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[n1:].mean():.1f}\n\n"

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
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
            summary += f"<<COLOR:good>>Groups differ significantly (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots
        result["plots"].append({
            "title": f"Mann-Whitney: {var} by {group_var}",
            "data": [
                {"type": "box", "y": group1.tolist(), "name": str(groups[0]), "marker": {"color": "#4a9f6e"}, "fillcolor": "rgba(74, 159, 110, 0.3)"},
                {"type": "box", "y": group2.tolist(), "name": str(groups[1]), "marker": {"color": "#47a5e8"}, "fillcolor": "rgba(71, 165, 232, 0.3)"}
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Mann-Whitney U test p = {pval:.4f}. " + ("Groups differ significantly." if pval < 0.05 else "No significant difference.")
        result["statistics"] = {"U_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_size)}

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

        # Effect size (epsilon squared)
        n_total = sum(len(g) for g in group_data)
        epsilon_sq = (stat - len(groups) + 1) / (n_total - len(groups))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>KRUSKAL-WALLIS H TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(groups)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
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

        summary += f"\n<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  H statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  ε² (effect size): {epsilon_sq:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>At least one group differs significantly (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider post-hoc tests (Dunn's test) to identify which groups differ.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference among groups (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for all groups
        result["plots"].append({
            "title": f"Kruskal-Wallis: {var} by {group_var}",
            "data": [
                {"type": "box", "y": data.tolist(), "name": str(g),
                 "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"][i % 5]},
                 "fillcolor": ["rgba(74,159,110,0.3)", "rgba(71,165,232,0.3)", "rgba(232,149,71,0.3)", "rgba(159,74,74,0.3)", "rgba(108,92,231,0.3)"][i % 5]}
                for i, (g, data) in enumerate(zip(groups, group_data))
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Kruskal-Wallis H = {stat:.2f}, p = {pval:.4f}. " + ("Groups differ." if pval < 0.05 else "No difference.")
        result["statistics"] = {"H_statistic": float(stat), "p_value": float(pval), "epsilon_squared": float(epsilon_sq)}

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

        # Effect size: r = Z / sqrt(N)
        z_score = stats.norm.ppf(pval / 2)
        effect_r = abs(z_score) / np.sqrt(min_len)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>WILCOXON SIGNED-RANK TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Pair:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N pairs:<</COLOR>> {min_len}\n\n"
        summary += f"<<COLOR:text>>Differences (var1 - var2):<</COLOR>>\n"
        summary += f"  Median diff: {np.median(diffs):.4f}\n"
        summary += f"  Mean diff:   {np.mean(diffs):.4f}\n"
        summary += f"  Std diff:    {np.std(diffs, ddof=1):.4f}\n\n"
        summary += f"<<COLOR:highlight>>Test Statistic (W):<</COLOR>> {stat:.2f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>Effect Size (r):<</COLOR>> {effect_r:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:accent>>Significant difference between paired samples (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference between paired samples (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Histogram of differences
        result["plots"].append({
            "title": f"Paired Differences: {var1} - {var2}",
            "data": [
                {"type": "histogram", "x": diffs.tolist(), "name": "Differences",
                 "marker": {"color": "rgba(71,165,232,0.7)", "line": {"color": "#47a5e8", "width": 1}}},
                {"type": "scatter", "x": [0, 0], "y": [0, min_len // 3], "mode": "lines",
                 "name": "Zero", "line": {"color": "#e89547", "dash": "dash", "width": 2}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "Difference"}, "yaxis": {"title": "Count"}}
        })

        result["guide_observation"] = f"Wilcoxon signed-rank W = {stat:.2f}, p = {pval:.4f}. " + ("Paired samples differ." if pval < 0.05 else "No paired difference.")
        result["statistics"] = {"W_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_r), "median_diff": float(np.median(diffs)), "n_pairs": int(min_len)}

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
            result["summary"] = f"Friedman test requires at least 3 related samples (repeated measures). Got {len(vars_list)}.\n\nSelect 3+ measurement columns (e.g., Time1, Time2, Time3)."
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
        summary += f"<<COLOR:title>>FRIEDMAN TEST<</COLOR>>\n"
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
            summary += f"<<COLOR:accent>>Significant difference across conditions (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider Wilcoxon signed-rank tests for pairwise comparisons (with Bonferroni correction).<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference across conditions (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for each condition
        result["plots"].append({
            "title": "Friedman: Repeated Measures Comparison",
            "data": [
                {"type": "box", "y": clean_df[v].tolist(), "name": v,
                 "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84747", "#47e8c4", "#c4e847"][i % 8]},
                 "fillcolor": ["rgba(74,159,110,0.3)", "rgba(71,165,232,0.3)", "rgba(232,149,71,0.3)", "rgba(159,74,74,0.3)", "rgba(108,92,231,0.3)", "rgba(232,71,71,0.3)", "rgba(71,232,196,0.3)", "rgba(196,232,71,0.3)"][i % 8]}
                for i, v in enumerate(vars_list)
            ],
            "layout": {"height": 300, "yaxis": {"title": "Value"}}
        })

        result["guide_observation"] = f"Friedman chi2 = {stat:.2f}, p = {pval:.4f}, W = {kendall_w:.3f}. " + ("Conditions differ." if pval < 0.05 else "No difference.")
        result["statistics"] = {"chi2_statistic": float(stat), "p_value": float(pval), "kendall_w": float(kendall_w), "df": int(k - 1), "n_subjects": int(n_subjects)}

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
        se = 1.0 / np.sqrt(n - 3) if n > 3 else float('inf')
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
        summary += f"<<COLOR:title>>SPEARMAN RANK CORRELATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:highlight>>Spearman rho:<</COLOR>> {rho:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>95% CI:<</COLOR>> [{ci_low:.4f}, {ci_high:.4f}]\n\n"
        summary += f"<<COLOR:text>>Interpretation: {strength.capitalize()} {direction} monotonic association<</COLOR>>\n"

        if pval < 0.05:
            summary += f"<<COLOR:accent>>Statistically significant (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Not statistically significant (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Scatter with rank overlay
        result["plots"].append({
            "title": f"Spearman: {var1} vs {var2} (rho={rho:.3f})",
            "data": [{
                "type": "scatter", "mode": "markers",
                "x": x.tolist(), "y": y.tolist(),
                "marker": {"color": "#47a5e8", "size": 6, "opacity": 0.7},
                "name": "Data"
            }],
            "layout": {"height": 300, "xaxis": {"title": var1}, "yaxis": {"title": var2}}
        })

        result["guide_observation"] = f"Spearman rho = {rho:.3f}, p = {pval:.4f}. {strength.capitalize()} {direction} monotonic association."
        result["statistics"] = {"spearman_rho": float(rho), "p_value": float(pval), "ci_lower": float(ci_low), "ci_upper": float(ci_high), "n": int(n)}

    elif analysis_id == "main_effects":
        """
        Main Effects Plot - shows how each factor affects the response.
        Essential for DOE analysis.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if not factors:
            result["summary"] = "Please select at least one factor."
            return result

        y = df[response].dropna()

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MAIN EFFECTS PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
        summary += f"<<COLOR:highlight>>Grand Mean:<</COLOR>> {y.mean():.4f}\n\n"

        summary += f"<<COLOR:text>>Effect Sizes (deviation from grand mean):<</COLOR>>\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

        for i, factor in enumerate(factors):
            # Calculate means for each level
            factor_means = df.groupby(factor)[response].mean()

            summary += f"\n  <<COLOR:accent>>{factor}:<</COLOR>>\n"
            for level, mean_val in factor_means.items():
                effect = mean_val - y.mean()
                direction = "+" if effect > 0 else ""
                summary += f"    {level}: {mean_val:.4f} (effect: {direction}{effect:.4f})\n"

            # Create main effects plot
            levels = [str(l) for l in factor_means.index.tolist()]
            means = factor_means.values.tolist()

            result["plots"].append({
                "title": f"Main Effect: {factor}",
                "data": [{
                    "type": "scatter",
                    "x": levels,
                    "y": means,
                    "mode": "lines+markers",
                    "marker": {"color": colors[i % len(colors)], "size": 10},
                    "line": {"color": colors[i % len(colors)], "width": 2}
                }, {
                    "type": "scatter",
                    "x": levels,
                    "y": [y.mean()] * len(levels),
                    "mode": "lines",
                    "line": {"color": "#9aaa9a", "dash": "dash"},
                    "name": "Grand Mean"
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": factor},
                    "yaxis": {"title": f"Mean of {response}"}
                }
            })

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  Steeper slopes indicate stronger effects.\n"
        summary += f"  Flat lines suggest the factor has little impact.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Main effects plot for {len(factors)} factor(s). Check for steep slopes indicating strong effects."

    elif analysis_id == "interaction":
        """
        Interaction Plot - shows how factors interact.
        Non-parallel lines indicate interactions.
        """
        response = config.get("response")
        factor1 = config.get("factor1")
        factor2 = config.get("factor2")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>INTERACTION PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>X-axis factor:<</COLOR>> {factor1}\n"
        summary += f"<<COLOR:highlight>>Trace factor:<</COLOR>> {factor2}\n\n"

        # Calculate interaction means
        interaction_means = df.groupby([factor1, factor2])[response].mean().unstack()

        summary += f"<<COLOR:text>>Cell Means:<</COLOR>>\n"
        summary += interaction_means.to_string() + "\n\n"

        # Check for interaction (compare slopes)
        levels1 = interaction_means.index.tolist()
        levels2 = interaction_means.columns.tolist()

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7', '#e8c547']

        plot_data = []
        for i, lev2 in enumerate(levels2):
            means = interaction_means[lev2].tolist()
            plot_data.append({
                "type": "scatter",
                "x": [str(l) for l in levels1],
                "y": means,
                "mode": "lines+markers",
                "name": str(lev2),
                "marker": {"color": colors[i % len(colors)], "size": 8},
                "line": {"color": colors[i % len(colors)], "width": 2}
            })

        result["plots"].append({
            "title": f"Interaction: {factor1} × {factor2}",
            "data": plot_data,
            "layout": {
                "height": 350,
                "xaxis": {"title": factor1},
                "yaxis": {"title": f"Mean of {response}"},
                "legend": {"title": {"text": factor2}}
            }
        })

        # Simple interaction detection (check if lines are parallel)
        if len(levels2) >= 2 and len(levels1) >= 2:
            slopes = []
            for lev2 in levels2:
                vals = interaction_means[lev2].values
                slope = (vals[-1] - vals[0]) / (len(vals) - 1) if len(vals) > 1 else 0
                slopes.append(slope)

            slope_diff = max(slopes) - min(slopes)
            has_interaction = slope_diff > 0.1 * abs(np.mean(slopes)) if np.mean(slopes) != 0 else slope_diff > 0.1

            summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
            if has_interaction:
                summary += f"<<COLOR:warning>>POTENTIAL INTERACTION DETECTED<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines are not parallel, suggesting {factor1} and {factor2} interact.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The effect of {factor1} depends on the level of {factor2}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:good>>NO STRONG INTERACTION<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines appear roughly parallel.<</COLOR>>\n"
                summary += f"<<COLOR:text>>Factors may act independently.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Interaction plot for {factor1} × {factor2}. " + ("Non-parallel lines suggest interaction." if 'has_interaction' in dir() and has_interaction else "Check for parallel lines.")

    elif analysis_id == "logistic":
        """
        Logistic Regression - for binary outcomes.
        Returns odds ratios, confidence intervals, and ROC curve.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

        response = config.get("response")
        predictors = config.get("predictors", [])

        if not predictors:
            result["summary"] = "Please select at least one predictor."
            return result

        # Prepare data
        X = df[predictors].dropna()
        y = df[response].loc[X.index]

        # Encode target if needed
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            result["summary"] = f"Logistic regression requires binary outcome. Found {len(unique_vals)} unique values."
            return result

        # Map to 0/1
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = (y == unique_vals[1]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Coefficients and odds ratios
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>AUC (ROC):<</COLOR>> {roc_auc:.4f}\n\n"

        # Approximate SEs via Fisher information matrix
        _p_hat = model.predict_proba(X_train)[:, 1]
        _W = _p_hat * (1 - _p_hat)
        _X_mat = X_train.values
        try:
            _XWX = _X_mat.T @ (_W[:, None] * _X_mat)
            _se_coefs = np.sqrt(np.diag(np.linalg.inv(_XWX)))
        except Exception:
            _se_coefs = None

        summary += f"<<COLOR:text>>Coefficients & Odds Ratios:<</COLOR>>\n"
        if _se_coefs is not None:
            summary += f"  {'Predictor':<20} {'Coef':>10} {'SE':>10} {'OR':>10} {'95% CI for OR':>22}\n"
            summary += f"  {'-'*74}\n"
            for i, (pred, coef, odds) in enumerate(zip(predictors, coefs, odds_ratios)):
                _or_lo = np.exp(coef - 1.96 * _se_coefs[i])
                _or_hi = np.exp(coef + 1.96 * _se_coefs[i])
                summary += f"  {pred:<20} {coef:>10.4f} {_se_coefs[i]:>10.4f} {odds:>10.4f} [{_or_lo:>8.4f}, {_or_hi:>8.4f}]\n"
        else:
            summary += f"  {'Predictor':<20} {'Coef':>10} {'Odds Ratio':>12}\n"
            summary += f"  {'-'*44}\n"
            for pred, coef, odds in zip(predictors, coefs, odds_ratios):
                summary += f"  {pred:<20} {coef:>10.4f} {odds:>12.4f}\n"

        summary += f"\n<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
        summary += f"  Predicted:    0      1\n"
        summary += f"  Actual 0:  {cm[0,0]:>4}   {cm[0,1]:>4}\n"
        summary += f"  Actual 1:  {cm[1,0]:>4}   {cm[1,1]:>4}\n"

        # ROC curve plot
        result["plots"].append({
            "title": "ROC Curve",
            "data": [{
                "type": "scatter",
                "x": fpr.tolist(),
                "y": tpr.tolist(),
                "mode": "lines",
                "name": f"AUC = {roc_auc:.3f}",
                "line": {"color": "#4a9f6e", "width": 2}
            }, {
                "type": "scatter",
                "x": [0, 1],
                "y": [0, 1],
                "mode": "lines",
                "name": "Random",
                "line": {"color": "#9aaa9a", "dash": "dash"}
            }],
            "layout": {
                "height": 300,
                "xaxis": {"title": "False Positive Rate"},
                "yaxis": {"title": "True Positive Rate"}
            }
        })

        # Odds ratio forest plot
        result["plots"].append({
            "title": "Odds Ratios (Log Scale)",
            "data": [{
                "type": "bar",
                "x": odds_ratios.tolist(),
                "y": predictors,
                "orientation": "h",
                "marker": {"color": ["#4a9f6e" if o > 1 else "#e85747" for o in odds_ratios]}
            }],
            "layout": {
                "height": 250,
                "xaxis": {"type": "log", "title": "Odds Ratio"},
                "shapes": [{
                    "type": "line",
                    "x0": 1, "x1": 1,
                    "y0": -0.5, "y1": len(predictors) - 0.5,
                    "line": {"color": "#9aaa9a", "dash": "dash"}
                }]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Logistic regression with AUC = {roc_auc:.3f}. Odds ratios > 1 increase probability of outcome."
        result["statistics"] = {"AUC": float(roc_auc), "accuracy": float((y_pred == y_test).mean())}

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
        summary += f"<<COLOR:title>>F-TEST FOR EQUALITY OF VARIANCES<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n\n"
        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
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
            summary += f"<<COLOR:warning>>Variances are SIGNIFICANTLY DIFFERENT (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider using Welch's t-test or non-parametric alternatives.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>No significant difference in variances (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Equal variance assumption is reasonable.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"F = {F:.2f}, p = {p_value:.4f}. " + ("Variances differ significantly." if p_value < 0.05 else "Variances are similar.")
        result["statistics"] = {"F_statistic": float(F), "p_value": float(p_value), "variance_ratio": float(max(var1,var2)/min(var1,var2))}

        # Variance comparison bar chart + side-by-side box plots
        result["plots"].append({
            "title": f"Variance Comparison: {groups[0]} vs {groups[1]}",
            "data": [
                {"type": "bar", "x": [str(groups[0]), str(groups[1])], "y": [float(var1), float(var2)], "marker": {"color": ["#4a9f6e", "#4a90d9"]}, "name": "Variance"},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "yaxis": {"title": "Variance"}, "xaxis": {"title": group_var}}
        })
        result["plots"].append({
            "title": f"Distribution by Group: {var}",
            "data": [
                {"type": "box", "y": g1.tolist(), "name": str(groups[0]), "marker": {"color": "#4a9f6e"}, "boxpoints": "outliers"},
                {"type": "box", "y": g2.tolist(), "name": str(groups[1]), "marker": {"color": "#4a90d9"}, "boxpoints": "outliers"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "yaxis": {"title": var}}
        })

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
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        df_val = n1 + n2 - 2

        # TOST: Two one-sided tests
        t_lower = (diff - (-margin)) / se
        t_upper = (diff - margin) / se

        p_lower = 1 - stats.t.cdf(t_lower, df_val)
        p_upper = stats.t.cdf(t_upper, df_val)
        p_tost = max(p_lower, p_upper)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>EQUIVALENCE TEST (TOST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n"
        summary += f"<<COLOR:highlight>>Equivalence margin:<</COLOR>> ±{margin}\n\n"
        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
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
            summary += f"<<COLOR:text>>The difference is small enough to be considered equivalent.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:warning>>NOT EQUIVALENT (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Cannot conclude equivalence within the specified margin.<</COLOR>>\n"

        # Equivalence plot
        result["plots"].append({
            "title": "Equivalence Plot",
            "data": [{
                "type": "scatter",
                "x": [diff],
                "y": [0.5],
                "mode": "markers",
                "marker": {"size": 15, "color": "#4a9f6e" if p_tost < 0.05 else "#e85747"},
                "error_x": {"type": "constant", "value": 1.96 * se, "color": "#4a9f6e"},
                "name": "Mean Difference"
            }],
            "layout": {
                "height": 200,
                "xaxis": {"title": "Difference", "zeroline": True},
                "yaxis": {"visible": False, "range": [0, 1]},
                "shapes": [
                    {"type": "line", "x0": -margin, "x1": -margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "line", "x0": margin, "x1": margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "rect", "x0": -margin, "x1": margin, "y0": 0, "y1": 1, "fillcolor": "rgba(74,159,110,0.1)", "line": {"width": 0}}
                ]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"TOST p = {p_tost:.4f}. " + (f"Groups equivalent within ±{margin}." if p_tost < 0.05 else "Cannot confirm equivalence.")
        result["statistics"] = {"TOST_p_value": float(p_tost), "mean_difference": float(diff), "margin": float(margin)}

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
            if binary[i] != binary[i-1]:
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
        summary += f"<<COLOR:title>>RUNS TEST FOR RANDOMNESS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:text>>Run Statistics:<</COLOR>>\n"
        summary += f"  Observed runs: {n_runs}\n"
        summary += f"  Expected runs: {expected_runs:.2f}\n"
        summary += f"  Above median: {n_pos}\n"
        summary += f"  Below median: {n_neg}\n\n"
        summary += f"<<COLOR:highlight>>Z-statistic:<</COLOR>> {z:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            if n_runs < expected_runs:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too few runs (clustering)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values tend to cluster together, suggesting trends or patterns.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too many runs (oscillation)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values alternate too frequently, suggesting negative autocorrelation.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>SEQUENCE APPEARS RANDOM (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>No evidence of patterns or trends in the data.<</COLOR>>\n"

        # Plot the sequence with run coloring
        colors = []
        current_color = "#4a9f6e"
        for i in range(len(binary)):
            if i > 0 and binary[i] != binary[i-1]:
                current_color = "#47a5e8" if current_color == "#4a9f6e" else "#4a9f6e"
            colors.append(current_color)

        result["plots"].append({
            "title": f"Sequence Plot ({n_runs} runs)",
            "data": [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines+markers",
                "marker": {"color": colors, "size": 6},
                "line": {"color": "rgba(74,159,110,0.3)"}
            }, {
                "type": "scatter",
                "y": [median] * len(data),
                "mode": "lines",
                "name": "Median",
                "line": {"color": "#e89547", "dash": "dash"}
            }],
            "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Runs test: {n_runs} runs, p = {p_value:.4f}. " + ("Non-random pattern detected." if p_value < 0.05 else "Sequence appears random.")
        result["statistics"] = {"runs": int(n_runs), "expected_runs": float(expected_runs), "Z_statistic": float(z), "p_value": float(p_value)}

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
        sorted_data = np.sort(data)
        n_total = len(data)
        ci_idx = int(stats.binom.ppf(0.025, n_total, 0.5))
        ci_lower = sorted_data[ci_idx] if ci_idx < n_total else sorted_data[0]
        ci_upper = sorted_data[n_total - 1 - ci_idx] if ci_idx < n_total else sorted_data[-1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>SIGN TEST FOR MEDIAN<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>H₀ Median:<</COLOR>> {h0_median}\n\n"
        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  N: {len(data)}\n"
        summary += f"  Sample median: {sample_median:.4f}\n"
        summary += f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
        summary += f"<<COLOR:text>>Sign Counts:<</COLOR>>\n"
        summary += f"  Above H₀: {above}\n"
        summary += f"  Below H₀: {below}\n"
        summary += f"  Ties (excluded): {ties}\n\n"
        summary += f"<<COLOR:highlight>>p-value (two-sided):<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>REJECT H₀ — Median differs from {h0_median}<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>FAIL TO REJECT H₀ — No evidence median differs from {h0_median}<</COLOR>>\n"

        # Dot plot with median lines
        result["plots"].append({
            "title": f"Sign Test: {var}",
            "data": [
                {"type": "scatter", "y": data.tolist(), "mode": "markers", "name": "Data",
                 "marker": {"color": ["#4a9f6e" if v > h0_median else "#d94a4a" if v < h0_median else "#e89547" for v in data], "size": 6}},
                {"type": "scatter", "y": [sample_median]*len(data), "mode": "lines", "name": f"Sample Median ({sample_median:.2f})",
                 "line": {"color": "#4a9f6e", "dash": "dash"}},
                {"type": "scatter", "y": [h0_median]*len(data), "mode": "lines", "name": f"H₀ ({h0_median})",
                 "line": {"color": "#d94a4a", "dash": "dot"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Sign test p = {p_value:.4f}. " + (f"Median differs from {h0_median}." if p_value < 0.05 else f"No evidence median differs from {h0_median}.")
        result["statistics"] = {"sample_median": float(sample_median), "above": int(above), "below": int(below), "p_value": float(p_value)}

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
        summary += f"<<COLOR:title>>MOOD'S MEDIAN TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {group_col}\n"
        summary += f"<<COLOR:highlight>>Grand Median:<</COLOR>> {grand_median:.4f}\n\n"
        summary += f"<<COLOR:text>>Group Medians:<</COLOR>>\n"
        for g, m in group_medians.items():
            summary += f"  {g}: {m:.4f}\n"
        summary += f"\n<<COLOR:text>>Contingency Table (above/at-or-below grand median):<</COLOR>>\n"
        summary += f"  {'Group':<15} {'Above':>8} {'At/Below':>10} {'N':>6}\n"
        summary += f"  {'-'*42}\n"
        for j, g in enumerate(groups):
            summary += f"  {str(g):<15} {table[0,j]:>8} {table[1,j]:>10} {table[0,j]+table[1,j]:>6}\n"
        summary += f"\n<<COLOR:highlight>>Chi-squared:<</COLOR>> {chi2:.4f}\n"
        summary += f"<<COLOR:highlight>>df:<</COLOR>> {dof}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>SIGNIFICANT — At least one group median differs<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>NOT SIGNIFICANT — No evidence of median differences<</COLOR>>\n"

        # Box plots by group
        traces = []
        theme_colors = ['#4a9f6e', '#4a90d9', '#e89547', '#d94a4a', '#9f4a4a', '#7a6a9a']
        for i, (g, d) in enumerate(zip(groups, data_by_group)):
            traces.append({"type": "box", "y": d.tolist(), "name": str(g),
                          "marker": {"color": theme_colors[i % len(theme_colors)]}})
        traces.append({"type": "scatter", "x": [str(g) for g in groups],
                       "y": [grand_median]*len(groups), "mode": "lines",
                       "name": f"Grand Median ({grand_median:.2f})",
                       "line": {"color": "#e89547", "dash": "dash", "width": 2}})
        result["plots"].append({
            "title": f"Mood's Median Test: {var} by {group_col}",
            "data": traces,
            "layout": {"template": "plotly_dark", "height": 280, "showlegend": True}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Mood's median test: χ² = {chi2:.2f}, p = {p_value:.4f}. " + ("Medians differ." if p_value < 0.05 else "No evidence of median differences.")
        result["statistics"] = {"chi_squared": float(chi2), "df": int(dof), "p_value": float(p_value), "grand_median": float(grand_median)}

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
        summary += f"<<COLOR:title>>MULTI-VARI CHART<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

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
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos + np.random.uniform(-0.1, 0.1) for _ in vals],
                    "y": vals.tolist(),
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6},
                    "showlegend": False
                })

                # Mean marker
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos],
                    "y": [vals.mean()],
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 12, "symbol": "diamond"},
                    "showlegend": False
                })

                summary += f"  {level}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n"

            # Connect means
            means = [groups.get_group(l).mean() for l in df[factor].dropna().unique()]
            plot_data.append({
                "type": "scatter",
                "x": x_positions,
                "y": means,
                "mode": "lines",
                "line": {"color": "#e89547", "width": 2},
                "showlegend": False
            })

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor}",
                "data": plot_data,
                "layout": {
                    "height": 300,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor},
                    "yaxis": {"title": response}
                }
            })

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
                        plot_data.append({
                            "type": "scatter",
                            "x": [pos + np.random.uniform(-0.15, 0.15) for _ in vals],
                            "y": vals.tolist(),
                            "mode": "markers",
                            "marker": {"color": colors[i % len(colors)], "size": 5, "opacity": 0.5},
                            "showlegend": False
                        })

                        # Mean
                        group_means.append((pos, vals.mean()))
                        pos += 1

                # Connect means within factor1 level
                if group_means:
                    plot_data.append({
                        "type": "scatter",
                        "x": [g[0] for g in group_means],
                        "y": [g[1] for g in group_means],
                        "mode": "lines+markers",
                        "marker": {"color": colors[i % len(colors)], "size": 10, "symbol": "diamond"},
                        "line": {"color": colors[i % len(colors)], "width": 2},
                        "name": str(lev1)
                    })

                pos += 0.5  # Gap between factor1 levels

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor1}/{factor2}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor2},
                    "yaxis": {"title": response},
                    "showlegend": True
                }
            })

            summary += f"<<COLOR:text>>Nested structure: {factor1} (colors) → {factor2} (x-axis)<</COLOR>>\n\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  • Vertical spread within groups = within-group variation\n"
        summary += f"  • Differences between group means = between-group variation\n"
        summary += f"  • Compare spreads to identify dominant sources of variation\n"

        result["summary"] = summary
        result["guide_observation"] = f"Multi-vari chart showing variation in {response} across {len(factors)} factor(s)."

    elif analysis_id == "arima":
        """
        ARIMA Time Series Analysis - fit and forecast.
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        var = config.get("var")
        p = int(config.get("p", 1))  # AR order
        d = int(config.get("d", 1))  # Differencing
        q = int(config.get("q", 1))  # MA order
        forecast_periods = int(config.get("forecast", 10))

        data = df[var].dropna().values

        # Stationarity test
        adf_result = adfuller(data)
        adf_stat, adf_pval = adf_result[0], adf_result[1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ARIMA TIME SERIES ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> ARIMA({p},{d},{q})\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Stationarity Test (ADF):<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()

            # Model summary
            summary += f"<<COLOR:text>>Model Parameters:<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            summary += f"<<COLOR:text>>Forecast ({forecast_periods} periods):<</COLOR>>\n"
            for i in range(min(5, forecast_periods)):
                summary += f"  Period {i+1}: {fc_mean.iloc[i]:.4f} [{fc_ci.iloc[i, 0]:.4f}, {fc_ci.iloc[i, 1]:.4f}]\n"
            if forecast_periods > 5:
                summary += f"  ... ({forecast_periods - 5} more periods)\n"

            # Plot
            x_hist = list(range(len(data)))
            x_fc = list(range(len(data), len(data) + forecast_periods))

            result["plots"].append({
                "title": f"ARIMA({p},{d},{q}) Forecast",
                "data": [
                    {"type": "scatter", "x": x_hist, "y": data.tolist(), "mode": "lines", "name": "Historical", "line": {"color": "#4a9f6e"}},
                    {"type": "scatter", "x": x_fc, "y": fc_mean.tolist(), "mode": "lines", "name": "Forecast", "line": {"color": "#e89547"}},
                    {"type": "scatter", "x": x_fc + x_fc[::-1], "y": fc_ci.iloc[:, 1].tolist() + fc_ci.iloc[::-1, 0].tolist(),
                     "fill": "toself", "fillcolor": "rgba(232,149,71,0.2)", "line": {"color": "transparent"}, "name": "95% CI"}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}}
            })

            # Residual diagnostics
            residuals = fitted.resid

            result["plots"].append({
                "title": "Residuals",
                "data": [{"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}}
            })

            result["statistics"] = {"AIC": float(fitted.aic), "BIC": float(fitted.bic), "ADF_pvalue": float(adf_pval)}

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += f"<<COLOR:text>>Try different p, d, q values or check data for issues.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"ARIMA({p},{d},{q}) model. {'Stationary data.' if adf_pval < 0.05 else 'Consider differencing.'}"

    elif analysis_id == "sarima":
        """
        SARIMA — Seasonal ARIMA. Extends ARIMA with seasonal (P,D,Q,m) parameters.
        Uses statsmodels SARIMAX.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller

        var = config.get("var")
        p = int(config.get("p", 1))
        d = int(config.get("d", 1))
        q = int(config.get("q", 1))
        P = int(config.get("P", 1))
        D = int(config.get("D", 1))
        Q = int(config.get("Q", 1))
        m = int(config.get("m", 12))  # Seasonal period
        forecast_periods = int(config.get("forecast", 24))

        ts_data = df[var].dropna().values

        # Stationarity test
        adf_result = adfuller(ts_data)
        adf_stat, adf_pval = adf_result[0], adf_result[1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>SARIMA SEASONAL FORECASTING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> SARIMA({p},{d},{q})({P},{D},{Q})[{m}]\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(ts_data)}\n"
        summary += f"<<COLOR:highlight>>Seasonal period:<</COLOR>> {m}\n\n"

        summary += f"<<COLOR:text>>Stationarity Test (ADF):<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = SARIMAX(ts_data, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)

            summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n"
            summary += f"  Log-likelihood: {fitted.llf:.2f}\n\n"

            # Parameter summary
            summary += f"<<COLOR:text>>Parameters:<</COLOR>>\n"
            param_names = fitted.param_names if hasattr(fitted, 'param_names') else [f"param_{i}" for i in range(len(fitted.params))]
            params = fitted.params if hasattr(fitted.params, '__len__') else [fitted.params]
            bse = fitted.bse if hasattr(fitted, 'bse') else [None] * len(params)
            pvals = fitted.pvalues if hasattr(fitted, 'pvalues') else [None] * len(params)
            for i, name in enumerate(param_names):
                val = float(params[i])
                se = float(bse[i]) if bse is not None and i < len(bse) else None
                pval = float(pvals[i]) if pvals is not None and i < len(pvals) else None
                sig = "<<COLOR:good>>*<</COLOR>>" if pval is not None and pval < 0.05 else ""
                se_str = f"{se:.4f}" if se is not None else "N/A"
                p_str = f"{pval:.4f}" if pval is not None else "N/A"
                summary += f"  {name:<20} {val:>10.4f}  (SE={se_str}, p={p_str}) {sig}\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            # Convert to lists for uniform handling
            fc_mean_list = fc_mean.tolist() if hasattr(fc_mean, 'tolist') else list(fc_mean)
            if hasattr(fc_ci, 'iloc'):
                fc_lower = fc_ci.iloc[:, 0].tolist()
                fc_upper = fc_ci.iloc[:, 1].tolist()
            else:
                fc_lower = fc_ci[:, 0].tolist()
                fc_upper = fc_ci[:, 1].tolist()

            summary += f"\n<<COLOR:text>>Forecast ({forecast_periods} periods):<</COLOR>>\n"
            for i in range(min(6, forecast_periods)):
                summary += f"  Period {i+1}: {fc_mean_list[i]:.4f} [{fc_lower[i]:.4f}, {fc_upper[i]:.4f}]\n"
            if forecast_periods > 6:
                summary += f"  ... ({forecast_periods - 6} more periods)\n"

            # Ljung-Box test on residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(fitted.resid, lags=[min(10, len(ts_data) // 5)], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[0])
            summary += f"\n<<COLOR:text>>Ljung-Box Test (residual autocorrelation):<</COLOR>>\n"
            summary += f"  p-value: {lb_p:.4f}\n"
            if lb_p > 0.05:
                summary += f"  <<COLOR:good>>Residuals appear uncorrelated — good model fit.<</COLOR>>\n"
            else:
                summary += f"  <<COLOR:warning>>Residuals show autocorrelation — consider different orders.<</COLOR>>\n"

            # Plot
            x_hist = list(range(len(ts_data)))
            x_fc = list(range(len(ts_data), len(ts_data) + forecast_periods))

            result["plots"].append({
                "title": f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}] Forecast",
                "data": [
                    {"type": "scatter", "x": x_hist, "y": ts_data.tolist(), "mode": "lines", "name": "Historical", "line": {"color": "#4a9f6e"}},
                    {"type": "scatter", "x": x_fc, "y": fc_mean_list, "mode": "lines", "name": "Forecast", "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "scatter", "x": x_fc + x_fc[::-1], "y": fc_upper + fc_lower[::-1],
                     "fill": "toself", "fillcolor": "rgba(232,149,71,0.2)", "line": {"color": "transparent"}, "name": "95% CI"}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}}
            })

            # Residual plot
            residuals = fitted.resid
            result["plots"].append({
                "title": "Residuals",
                "data": [{"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}}
            })

            # ACF of residuals
            from statsmodels.tsa.stattools import acf
            resid_clean = residuals[~np.isnan(residuals)] if isinstance(residuals, np.ndarray) else residuals.dropna()
            n_lags = min(30, len(resid_clean) // 3)
            acf_vals = acf(resid_clean, nlags=n_lags, fft=True)
            ci_bound = 1.96 / np.sqrt(len(residuals))
            result["plots"].append({
                "title": "Residual ACF",
                "data": [
                    {"type": "bar", "x": list(range(n_lags + 1)), "y": acf_vals.tolist(),
                     "marker": {"color": ["#d94a4a" if abs(v) > ci_bound and i > 0 else "#4a9f6e" for i, v in enumerate(acf_vals)]}}
                ],
                "layout": {
                    "height": 200, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]},
                    "shapes": [
                        {"type": "line", "x0": 0, "x1": n_lags, "y0": ci_bound, "y1": ci_bound, "line": {"color": "#e89547", "dash": "dash"}},
                        {"type": "line", "x0": 0, "x1": n_lags, "y0": -ci_bound, "y1": -ci_bound, "line": {"color": "#e89547", "dash": "dash"}}
                    ],
                    "template": "plotly_white"
                }
            })

            result["statistics"] = {
                "AIC": float(fitted.aic), "BIC": float(fitted.bic),
                "ADF_pvalue": float(adf_pval), "ljung_box_p": lb_p,
                "order": [p, d, q], "seasonal_order": [P, D, Q, m]
            }
            result["guide_observation"] = f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]: AIC={fitted.aic:.1f}. " + ("Good residuals." if lb_p > 0.05 else "Check residuals.")

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += f"<<COLOR:text>>Try different (p,d,q)(P,D,Q)[m] values. Ensure enough data for seasonal period m={m}.<</COLOR>>\n"

        result["summary"] = summary

    elif analysis_id == "decomposition":
        """
        Time Series Decomposition - trend, seasonal, residual.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        var = config.get("var")
        period = int(config.get("period", 12))
        model_type = config.get("model", "additive")  # additive or multiplicative

        data = df[var].dropna()

        if len(data) < 2 * period:
            result["summary"] = f"Need at least {2 * period} observations for period={period}. Have {len(data)}."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TIME SERIES DECOMPOSITION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type.capitalize()}\n"
        summary += f"<<COLOR:highlight>>Period:<</COLOR>> {period}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        decomp = seasonal_decompose(data, model=model_type, period=period)

        # Trend statistics
        trend_clean = decomp.trend.dropna()
        summary += f"<<COLOR:text>>Trend:<</COLOR>>\n"
        summary += f"  Start: {trend_clean.iloc[0]:.4f}\n"
        summary += f"  End: {trend_clean.iloc[-1]:.4f}\n"
        summary += f"  Change: {trend_clean.iloc[-1] - trend_clean.iloc[0]:.4f}\n\n"

        # Seasonal strength
        seasonal_var = decomp.seasonal.var()
        resid_var = decomp.resid.dropna().var()
        seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0

        summary += f"<<COLOR:text>>Seasonal Strength:<</COLOR>> {seasonal_strength:.2%}\n"
        if seasonal_strength > 0.6:
            summary += f"  <<COLOR:accent>>Strong seasonality detected<</COLOR>>\n"
        elif seasonal_strength > 0.3:
            summary += f"  Moderate seasonality\n"
        else:
            summary += f"  Weak or no seasonality\n"

        x_vals = list(range(len(data)))

        # Original
        result["plots"].append({
            "title": "Original Series",
            "data": [{"type": "scatter", "x": x_vals, "y": data.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
            "layout": {"height": 150}
        })

        # Trend
        result["plots"].append({
            "title": "Trend",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.trend.tolist(), "mode": "lines", "line": {"color": "#47a5e8"}}],
            "layout": {"height": 150}
        })

        # Seasonal
        result["plots"].append({
            "title": "Seasonal",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.seasonal.tolist(), "mode": "lines", "line": {"color": "#e89547"}}],
            "layout": {"height": 150}
        })

        # Residual
        result["plots"].append({
            "title": "Residual",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.resid.tolist(), "mode": "lines", "line": {"color": "#9aaa9a"}}],
            "layout": {"height": 150}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Decomposition with {seasonal_strength:.0%} seasonal strength."
        result["statistics"] = {"seasonal_strength": float(seasonal_strength), "trend_change": float(trend_clean.iloc[-1] - trend_clean.iloc[0])}

    elif analysis_id == "acf_pacf":
        """
        Autocorrelation and Partial Autocorrelation plots.
        Essential for ARIMA model identification.
        """
        from statsmodels.tsa.stattools import acf, pacf

        var = config.get("var")
        max_lags = int(config.get("lags", 20))

        data = df[var].dropna().values

        acf_vals = acf(data, nlags=max_lags)
        pacf_vals = pacf(data, nlags=max_lags)

        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(len(data))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ACF / PACF ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        summary += f"<<COLOR:highlight>>Lags shown:<</COLOR>> {max_lags}\n\n"

        # Find significant lags
        sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
        sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

        summary += f"<<COLOR:text>>Significant ACF lags:<</COLOR>> {sig_acf[:5] if sig_acf else 'None'}\n"
        summary += f"<<COLOR:text>>Significant PACF lags:<</COLOR>> {sig_pacf[:5] if sig_pacf else 'None'}\n\n"

        # ARIMA order suggestions
        summary += f"<<COLOR:success>>ARIMA ORDER SUGGESTIONS:<</COLOR>>\n"
        if len(sig_pacf) > 0 and (len(sig_acf) == 0 or sig_acf[0] > sig_pacf[-1]):
            summary += f"  PACF cuts off at lag {max(sig_pacf)}: Try AR({max(sig_pacf)})\n"
        if len(sig_acf) > 0 and (len(sig_pacf) == 0 or sig_pacf[0] > sig_acf[-1]):
            summary += f"  ACF cuts off at lag {max(sig_acf)}: Try MA({max(sig_acf)})\n"
        if len(sig_acf) > 0 and len(sig_pacf) > 0:
            summary += f"  Both taper: Try ARMA({min(3, max(sig_pacf))},{min(3, max(sig_acf))})\n"

        lags = list(range(max_lags + 1))

        # ACF plot
        result["plots"].append({
            "title": "Autocorrelation Function (ACF)",
            "data": [
                {"type": "bar", "x": lags, "y": acf_vals.tolist(), "marker": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]}}
        })

        # PACF plot
        result["plots"].append({
            "title": "Partial Autocorrelation Function (PACF)",
            "data": [
                {"type": "bar", "x": lags, "y": pacf_vals.tolist(), "marker": {"color": "#47a5e8"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF", "range": [-1, 1]}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"ACF/PACF analysis. Significant lags help identify ARIMA orders."

    elif analysis_id == "weibull":
        """
        Weibull Reliability Analysis - life data analysis.
        """
        from scipy import stats
        from scipy.optimize import minimize

        var = config.get("var")  # Time to failure
        censored = config.get("censored")  # Optional censoring indicator

        data = df[var].dropna().values
        data = data[data > 0]  # Weibull requires positive values

        if len(data) < 3:
            result["summary"] = "Need at least 3 data points for Weibull analysis."
            return result

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>WEIBULL RELIABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (time to failure)\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Weibull Parameters:<</COLOR>>\n"
        summary += f"  Shape (β): {shape:.4f}\n"
        summary += f"  Scale (η): {scale:.4f}\n\n"

        # Interpret shape parameter
        summary += f"<<COLOR:text>>Shape Interpretation:<</COLOR>>\n"
        if shape < 1:
            summary += f"  β < 1: <<COLOR:warning>>Infant mortality / early failures<</COLOR>>\n"
            summary += f"  Failure rate DECREASES over time (burn-in period)\n"
        elif shape == 1:
            summary += f"  β ≈ 1: Random failures (exponential distribution)\n"
            summary += f"  Failure rate is CONSTANT\n"
        else:
            summary += f"  β > 1: <<COLOR:accent>>Wear-out failures<</COLOR>>\n"
            summary += f"  Failure rate INCREASES over time\n"

        # Reliability metrics
        mean_life = scale * np.exp(np.log(np.exp(1)) / shape) if shape > 0 else scale
        b10 = stats.weibull_min.ppf(0.10, shape, 0, scale)  # 10% failure life
        b50 = stats.weibull_min.ppf(0.50, shape, 0, scale)  # Median life

        summary += f"\n<<COLOR:text>>Reliability Metrics:<</COLOR>>\n"
        summary += f"  Mean Life (MTTF): {mean_life:.2f}\n"
        summary += f"  B10 Life (10% fail): {b10:.2f}\n"
        summary += f"  B50 Life (median): {b50:.2f}\n"

        # Reliability at specific times
        summary += f"\n<<COLOR:text>>Reliability R(t):<</COLOR>>\n"
        for t in [b10, b50, scale, 2*scale]:
            r = 1 - stats.weibull_min.cdf(t, shape, 0, scale)
            summary += f"  R({t:.1f}) = {r:.2%}\n"

        # Weibull probability plot
        sorted_data = np.sort(data)
        n = len(sorted_data)
        rank = np.arange(1, n + 1)
        median_rank = (rank - 0.3) / (n + 0.4)  # Median rank approximation

        # Linearized Weibull
        x_plot = np.log(sorted_data)
        y_plot = np.log(-np.log(1 - median_rank))

        # Fitted line
        x_fit = np.linspace(x_plot.min(), x_plot.max(), 100)
        y_fit = shape * (x_fit - np.log(scale))

        result["plots"].append({
            "title": "Weibull Probability Plot",
            "data": [
                {"type": "scatter", "x": x_plot.tolist(), "y": y_plot.tolist(), "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 8}},
                {"type": "scatter", "x": x_fit.tolist(), "y": y_fit.tolist(), "mode": "lines", "name": "Fit", "line": {"color": "#e89547"}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "ln(-ln(1-F))"}}
        })

        # Reliability curve
        t_range = np.linspace(0, 2 * scale, 100)
        reliability = 1 - stats.weibull_min.cdf(t_range, shape, 0, scale)
        hazard = (shape / scale) * (t_range / scale) ** (shape - 1)

        result["plots"].append({
            "title": "Reliability & Hazard Functions",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": reliability.tolist(), "mode": "lines", "name": "R(t)", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": t_range.tolist(), "y": (hazard / hazard.max()).tolist(), "mode": "lines", "name": "h(t) scaled", "line": {"color": "#e85747", "dash": "dash"}, "yaxis": "y2"}
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Reliability", "range": [0, 1]},
                "yaxis2": {"title": "Hazard (scaled)", "overlaying": "y", "side": "right"}
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Weibull β={shape:.2f} ({'wear-out' if shape > 1 else 'early failure' if shape < 1 else 'random'}), η={scale:.2f}."
        result["statistics"] = {"shape_beta": float(shape), "scale_eta": float(scale), "MTTF": float(mean_life), "B10": float(b10)}

    elif analysis_id == "kaplan_meier":
        """
        Kaplan-Meier Survival Analysis — non-parametric survival curve estimation.
        Estimates the survival function S(t) = P(T > t) from censored data.
        Optional grouping for comparing survival across strata (log-rank test).
        """
        from scipy import stats as scipy_stats

        # Support both old (time/event/group) and new (time_col/event_col/group_col) config keys
        time_col = config.get("time_col") or config.get("time") or config.get("var1")
        event_col = config.get("event_col") or config.get("event") or config.get("var2")
        group_col = config.get("group_col") or config.get("group")
        alpha = 1 - float(config.get("conf", 95)) / 100

        if not time_col or time_col not in df.columns:
            result["summary"] = "Error: Please select a valid time variable."
            return result

        try:
            cols_needed = [c for c in [time_col, event_col, group_col] if c and c in df.columns]
            data = df[cols_needed].dropna()
            times = data[time_col].values.astype(float)
            events = data[event_col].values.astype(int) if event_col and event_col in data.columns else np.ones(len(times), dtype=int)

            def km_estimate(t, e):
                """Product-limit estimator with Greenwood CIs."""
                unique_times = np.sort(np.unique(t[e == 1]))
                n_risk = []
                n_event = []
                survival = []
                s = 1.0
                var_sum = 0.0
                ci_lower = []
                ci_upper = []
                z = scipy_stats.norm.ppf(1 - alpha / 2)

                for ti in unique_times:
                    ni = np.sum(t >= ti)
                    di = np.sum((t == ti) & (e == 1))
                    n_risk.append(int(ni))
                    n_event.append(int(di))
                    if ni > 0:
                        s *= (1 - di / ni)
                        if ni > di:
                            var_sum += di / (ni * (ni - di))
                    survival.append(s)
                    se = s * np.sqrt(var_sum) if var_sum > 0 else 0
                    ci_lower.append(max(0, s - z * se))
                    ci_upper.append(min(1, s + z * se))

                return unique_times, survival, n_risk, n_event, ci_lower, ci_upper

            if group_col and group_col in data.columns and group_col != "" and group_col != "None":
                groups = sorted(data[group_col].unique())
                colors = ['#2c5f2d', '#4a90d9', '#d94a4a', '#d9a04a', '#7d4ad9']

                traces = []
                summary_parts = []
                median_survivals = {}

                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(t_g, e_g)

                    color = colors[i % len(colors)]
                    traces.append({
                        "x": [0] + ut.tolist(), "y": [1.0] + surv,
                        "mode": "lines", "name": f"{grp} (n={len(t_g)})",
                        "line": {"shape": "hv", "color": color, "width": 2}
                    })
                    traces.append({
                        "x": [0] + ut.tolist() + ut.tolist()[::-1] + [0],
                        "y": [1.0] + ci_hi + ci_lo[::-1] + [1.0],
                        "fill": "toself", "line": {"width": 0},
                        "showlegend": False, "name": f"{grp} CI", "opacity": 0.2
                    })

                    median = None
                    for j, s_val in enumerate(surv):
                        if s_val <= 0.5:
                            median = float(ut[j])
                            break
                    median_survivals[str(grp)] = median
                    summary_parts.append(f"**{grp}** (n={len(t_g)}): Median survival = {median if median else 'not reached'}")

                result["plots"].append({
                    "data": traces,
                    "layout": {
                        "title": "Kaplan-Meier Survival Curves by Group",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        "template": "plotly_white"
                    }
                })

                # Cumulative hazard plot: H(t) = -ln(S(t))
                ch_traces = []
                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut_ch, surv_ch, _, _, _, _ = km_estimate(t_g, e_g)
                    cum_haz = [-np.log(max(s, 1e-10)) for s in surv_ch]
                    ch_traces.append({
                        "x": [0] + ut_ch.tolist(), "y": [0.0] + cum_haz,
                        "mode": "lines", "name": f"{grp}",
                        "line": {"shape": "hv", "color": colors[i % len(colors)], "width": 2}
                    })
                result["plots"].append({
                    "data": ch_traces,
                    "layout": {
                        "title": "Cumulative Hazard by Group",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "H(t) = −ln S(t)"},
                        "template": "plotly_white"
                    }
                })

                # Log-rank test (Mantel-Haenszel)
                all_event_times = np.sort(np.unique(times[events == 1]))
                observed = {str(g): 0.0 for g in groups}
                expected = {str(g): 0.0 for g in groups}

                for ti in all_event_times:
                    total_at_risk = np.sum(times >= ti)
                    total_events = np.sum((times == ti) & (events == 1))
                    for g in groups:
                        mask_g = data[group_col] == g
                        t_g = times[mask_g]
                        e_g = events[mask_g]
                        n_g = np.sum(t_g >= ti)
                        d_g = np.sum((t_g == ti) & (e_g == 1))
                        observed[str(g)] += d_g
                        if total_at_risk > 0:
                            expected[str(g)] += n_g * total_events / total_at_risk

                chi2_stat = sum((observed[str(g)] - expected[str(g)])**2 / expected[str(g)]
                                for g in groups if expected[str(g)] > 0)
                df_lr = len(groups) - 1
                p_logrank = 1 - scipy_stats.chi2.cdf(chi2_stat, df_lr)

                summary_parts.insert(0, f"**Log-rank test**: chi2={chi2_stat:.3f}, df={df_lr}, p={p_logrank:.4f} {'(significant)' if p_logrank < alpha else '(not significant)'}\n")
                result["summary"] = "\n".join(summary_parts)
                result["guide_observation"] = f"KM analysis: {len(groups)} groups, log-rank p={p_logrank:.4f}. {'Survival differs significantly.' if p_logrank < alpha else 'No significant difference.'}"
                result["statistics"] = {
                    "log_rank_chi2": float(chi2_stat), "log_rank_p": float(p_logrank),
                    "df": df_lr, "n_total": len(data), "n_events": int(events.sum()),
                    "median_survival": median_survivals
                }
            else:
                ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(times, events)

                result["plots"].append({
                    "data": [
                        {"x": [0] + ut.tolist(), "y": [1.0] + surv, "mode": "lines", "name": "Survival",
                         "line": {"shape": "hv", "color": "#2c5f2d", "width": 2}},
                        {"x": [0] + ut.tolist(), "y": [1.0] + ci_hi, "mode": "lines", "name": "Upper CI",
                         "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1}, "showlegend": False},
                        {"x": [0] + ut.tolist(), "y": [1.0] + ci_lo, "mode": "lines", "name": "Lower CI",
                         "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1},
                         "fill": "tonexty", "fillcolor": "rgba(44,95,45,0.15)", "showlegend": False}
                    ],
                    "layout": {
                        "title": "Kaplan-Meier Survival Curve",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        "template": "plotly_white"
                    }
                })

                # Cumulative hazard plot: H(t) = -ln(S(t))
                cum_haz = [-np.log(max(s, 1e-10)) for s in surv]
                result["plots"].append({
                    "data": [
                        {"x": [0] + ut.tolist(), "y": [0.0] + cum_haz, "mode": "lines", "name": "Cumulative Hazard",
                         "line": {"shape": "hv", "color": "#4a90d9", "width": 2}}
                    ],
                    "layout": {
                        "title": "Cumulative Hazard Function",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "H(t) = −ln S(t)"},
                        "template": "plotly_white"
                    }
                })

                median = None
                for j, s_val in enumerate(surv):
                    if s_val <= 0.5:
                        median = float(ut[j])
                        break

                result["summary"] = f"**Kaplan-Meier Survival Analysis**\n\nN = {len(data)}, Events = {int(events.sum())}, Censored = {int((events == 0).sum())}\nMedian survival time: {median if median else 'not reached'}"
                result["guide_observation"] = f"KM curve: n={len(data)}, {int(events.sum())} events. Median survival = {median if median else 'not reached'}."
                result["statistics"] = {
                    "n_total": len(data), "n_events": int(events.sum()),
                    "n_censored": int((events == 0).sum()), "median_survival": median
                }

        except Exception as e:
            result["summary"] = f"Kaplan-Meier error: {str(e)}"

    elif analysis_id == "cox_ph":
        """
        Cox Proportional Hazards Regression — semi-parametric survival model.
        Estimates hazard ratios for covariates without assuming a baseline hazard distribution.
        Reports coefficients, hazard ratios, 95% CIs, concordance index.
        """
        import numpy as np
        from scipy import stats as scipy_stats

        time_col = config.get("time_col") or config.get("time") or config.get("var1")
        event_col = config.get("event_col") or config.get("event") or config.get("var2")
        covariates = config.get("covariates", [])
        alpha = 1 - float(config.get("conf", 95)) / 100

        if not time_col or not event_col:
            result["summary"] = "Error: Please select a time variable and an event/censor variable."
            return result
        if not covariates:
            result["summary"] = "Error: Please select at least one covariate."
            return result

        try:
            from statsmodels.duration.hazard_regression import PHReg

            all_cols = [time_col, event_col] + covariates
            data = df[[c for c in all_cols if c in df.columns]].dropna()

            # Build covariate matrix (handle categorical columns)
            X_parts = []
            covariate_names = []
            for cov in covariates:
                if cov not in data.columns:
                    continue
                if data[cov].dtype == 'object' or data[cov].nunique() < 6:
                    dummies = pd.get_dummies(data[cov], prefix=cov, drop_first=True)
                    X_parts.append(dummies)
                    covariate_names.extend(dummies.columns.tolist())
                else:
                    X_parts.append(data[[cov]])
                    covariate_names.append(cov)

            X = pd.concat(X_parts, axis=1).values.astype(float)
            times_arr = data[time_col].values.astype(float)
            events_arr = data[event_col].values.astype(int)

            model = PHReg(times_arr, X, events_arr, ties="breslow")
            fit = model.fit()

            coefs = fit.params
            se = fit.bse
            z_vals = coefs / se
            p_vals = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_vals)))
            hr = np.exp(coefs)
            z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
            hr_lower = np.exp(coefs - z_crit * se)
            hr_upper = np.exp(coefs + z_crit * se)

            # Summary table
            rows = []
            for i, name in enumerate(covariate_names):
                rows.append(f"| {name} | {coefs[i]:.4f} | {se[i]:.4f} | {z_vals[i]:.3f} | {p_vals[i]:.4f} | {hr[i]:.3f} | ({hr_lower[i]:.3f}, {hr_upper[i]:.3f}) |")

            table = "| Covariate | Coef | SE | z | p-value | Hazard Ratio | 95% CI |\n"
            table += "|---|---|---|---|---|---|---|\n"
            table += "\n".join(rows)

            # Concordance index (Harrell's C)
            lp = X @ coefs
            concordant = 0
            discordant = 0
            for i in range(len(times_arr)):
                for j in range(i + 1, len(times_arr)):
                    if events_arr[i] == 1 and times_arr[i] < times_arr[j]:
                        if lp[i] > lp[j]:
                            concordant += 1
                        elif lp[i] < lp[j]:
                            discordant += 1
                    elif events_arr[j] == 1 and times_arr[j] < times_arr[i]:
                        if lp[j] > lp[i]:
                            concordant += 1
                        elif lp[j] < lp[i]:
                            discordant += 1
            c_index = concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0.5

            # Forest plot of hazard ratios
            result["plots"].append({
                "data": [
                    {
                        "x": hr.tolist(),
                        "y": covariate_names,
                        "mode": "markers",
                        "marker": {"size": 10, "color": "#2c5f2d"},
                        "error_x": {
                            "type": "data", "symmetric": False,
                            "array": (hr_upper - hr).tolist(),
                            "arrayminus": (hr - hr_lower).tolist(),
                            "color": "#2c5f2d"
                        },
                        "type": "scatter",
                        "name": "Hazard Ratio"
                    }
                ],
                "layout": {
                    "title": "Cox PH — Hazard Ratios (Forest Plot)",
                    "xaxis": {"title": "Hazard Ratio", "type": "log"},
                    "yaxis": {"title": ""},
                    "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": len(covariate_names) - 0.5, "line": {"dash": "dash", "color": "red"}}],
                    "template": "plotly_white"
                }
            })

            # Risk score distribution by event status
            result["plots"].append({
                "data": [
                    {"x": lp[events_arr == 1].tolist(), "type": "histogram", "name": "Events",
                     "opacity": 0.7, "marker": {"color": "#d94a4a"}},
                    {"x": lp[events_arr == 0].tolist(), "type": "histogram", "name": "Censored",
                     "opacity": 0.7, "marker": {"color": "#4a90d9"}}
                ],
                "layout": {
                    "title": "Linear Predictor Distribution by Event Status",
                    "xaxis": {"title": "Linear Predictor (Xβ)"},
                    "yaxis": {"title": "Count"},
                    "barmode": "overlay", "template": "plotly_white"
                }
            })

            # Martingale-like residuals vs linear predictor
            # Martingale residual ≈ event_i - expected events (approximated)
            baseline_cumhaz = np.zeros(len(times_arr))
            sorted_idx = np.argsort(times_arr)
            risk_scores = np.exp(lp)
            for ii in range(len(sorted_idx)):
                idx_i = sorted_idx[ii]
                at_risk = risk_scores[sorted_idx[ii:]].sum()
                if at_risk > 0 and events_arr[idx_i] == 1:
                    baseline_cumhaz[sorted_idx[ii:]] += 1.0 / at_risk
            mart_resid = events_arr - risk_scores * baseline_cumhaz
            result["plots"].append({
                "data": [{
                    "x": lp.tolist(), "y": mart_resid.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": ["#d94a4a" if e == 1 else "#4a90d9" for e in events_arr], "size": 5, "opacity": 0.6},
                    "name": "Residuals"
                }],
                "layout": {
                    "title": "Martingale Residuals vs Linear Predictor",
                    "xaxis": {"title": "Linear Predictor (Xβ)"},
                    "yaxis": {"title": "Martingale Residual"},
                    "shapes": [{"type": "line", "x0": float(lp.min()), "x1": float(lp.max()), "y0": 0, "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            n_events = int(events_arr.sum())
            ll = float(fit.llf) if hasattr(fit, 'llf') else None

            result["summary"] = f"**Cox Proportional Hazards Regression**\n\nN = {len(data)}, Events = {n_events}, Censored = {len(data) - n_events}\nConcordance index (C) = {c_index:.3f}\n{'Log-likelihood = ' + f'{ll:.2f}' if ll else ''}\n\n{table}"
            result["guide_observation"] = f"Cox PH: n={len(data)}, {n_events} events, C-index={c_index:.3f}. " + ", ".join(f"{covariate_names[i]}: HR={hr[i]:.2f} (p={p_vals[i]:.4f})" for i in range(len(covariate_names)))

            result["statistics"] = {
                "n_total": len(data),
                "n_events": n_events,
                "concordance": float(c_index),
                "log_likelihood": ll,
                "coefficients": {covariate_names[i]: {
                    "coef": float(coefs[i]), "se": float(se[i]),
                    "z": float(z_vals[i]), "p": float(p_vals[i]),
                    "hazard_ratio": float(hr[i]),
                    "hr_ci_lower": float(hr_lower[i]), "hr_ci_upper": float(hr_upper[i])
                } for i in range(len(covariate_names))}
            }

        except ImportError:
            result["summary"] = "Cox PH requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Cox PH error: {str(e)}"

    elif analysis_id == "gage_rr":
        """
        Gage R&R (Repeatability and Reproducibility) Study.
        Measurement System Analysis for continuous data.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        operator = config.get("operator")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAGE R&R STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part}\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        # Get data
        data = df[[measurement, part, operator]].dropna()

        n_parts = data[part].nunique()
        n_operators = data[operator].nunique()
        n_replicates = len(data) // (n_parts * n_operators)

        summary += f"<<COLOR:text>>Study Design:<</COLOR>>\n"
        summary += f"  Parts: {n_parts}\n"
        summary += f"  Operators: {n_operators}\n"
        summary += f"  Replicates: {n_replicates}\n\n"

        # Calculate variance components using ANOVA
        from scipy import stats

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Part variation
        part_means = data.groupby(part)[measurement].mean()
        part_var = part_means.var() * n_operators * n_replicates

        # Operator variation
        op_means = data.groupby(operator)[measurement].mean()
        op_var = op_means.var() * n_parts * n_replicates

        # Repeatability (within operator-part)
        repeatability_var = data.groupby([part, operator])[measurement].var().mean()

        # Reproducibility (between operators)
        reproducibility_var = max(0, op_var - repeatability_var / (n_parts * n_replicates))

        # Gage R&R
        gage_rr_var = repeatability_var + reproducibility_var
        total_variation = gage_rr_var + part_var

        # Percentages
        pct_repeatability = 100 * np.sqrt(repeatability_var / total_variation) if total_variation > 0 else 0
        pct_reproducibility = 100 * np.sqrt(reproducibility_var / total_variation) if total_variation > 0 else 0
        pct_gage_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
        summary += f"  {'Source':<20} {'Variance':>12} {'%Contribution':>14}\n"
        summary += f"  {'-'*48}\n"
        summary += f"  {'Total Gage R&R':<20} {gage_rr_var:>12.4f} {pct_gage_rr:>13.1f}%\n"
        summary += f"    {'Repeatability':<18} {repeatability_var:>12.4f} {pct_repeatability:>13.1f}%\n"
        summary += f"    {'Reproducibility':<18} {reproducibility_var:>12.4f} {pct_reproducibility:>13.1f}%\n"
        summary += f"  {'Part-to-Part':<20} {part_var:>12.4f} {pct_part:>13.1f}%\n\n"

        # Assessment
        summary += f"<<COLOR:success>>ASSESSMENT:<</COLOR>>\n"
        if pct_gage_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
            summary += f"  Measurement system is acceptable.\n"
        elif pct_gage_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
            summary += f"  May be acceptable depending on application.\n"
        else:
            summary += f"  <<COLOR:bad>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
            summary += f"  Measurement system needs improvement.\n"

        # Number of distinct categories
        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"
        summary += f"  (Should be >= 5 for adequate discrimination)\n"

        # Plots
        # By Part
        result["plots"].append({
            "title": f"{measurement} by {part}",
            "data": [{
                "type": "box",
                "x": data[part].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#4a9f6e"}
            }],
            "layout": {"height": 250, "xaxis": {"title": part}}
        })

        # By Operator
        result["plots"].append({
            "title": f"{measurement} by {operator}",
            "data": [{
                "type": "box",
                "x": data[operator].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#47a5e8"}
            }],
            "layout": {"height": 250, "xaxis": {"title": operator}}
        })

        # Components of variation bar chart
        result["plots"].append({
            "title": "Components of Variation",
            "data": [{
                "type": "bar",
                "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                "y": [pct_gage_rr, pct_repeatability, pct_reproducibility, pct_part],
                "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]}
            }],
            "layout": {"height": 250, "yaxis": {"title": "% of Total Variation"}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Gage R&R = {pct_gage_rr:.1f}%. " + ("Acceptable." if pct_gage_rr < 30 else "Needs improvement.")
        result["statistics"] = {"gage_rr_pct": float(pct_gage_rr), "repeatability_pct": float(pct_repeatability), "reproducibility_pct": float(pct_reproducibility), "ndc": int(ndc)}

    # ── MSA (Measurement System Analysis) Expansion ─────────────────────────

    elif analysis_id == "gage_rr_nested":
        """
        Nested Gage R&R — when operators measure *different* parts (destructive testing).
        Variance components from nested ANOVA: part(operator), repeatability.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        operator = config.get("operator")

        data = df[[measurement, part, operator]].dropna()
        n_operators = data[operator].nunique()
        operators = sorted(data[operator].unique(), key=str)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NESTED GAGE R&R (DESTRUCTIVE)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part} (nested within {operator})\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Operator means
        op_means = data.groupby(operator)[measurement].mean()
        parts_per_op = data.groupby(operator)[part].nunique().mean()
        reps_per_part = len(data) / (data.groupby([operator, part]).ngroups) if data.groupby([operator, part]).ngroups > 0 else 1

        # Between-operator variance
        ss_operator = sum(len(data[data[operator] == o]) * (op_means[o] - grand_mean) ** 2 for o in operators)
        df_operator = n_operators - 1
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0

        # Between-part(operator) variance
        ss_part = 0
        df_part = 0
        for op in operators:
            op_data = data[data[operator] == op]
            op_mean = op_data[measurement].mean()
            for p in op_data[part].unique():
                part_data = op_data[op_data[part] == p]
                ss_part += len(part_data) * (part_data[measurement].mean() - op_mean) ** 2
                df_part += 1
        df_part -= n_operators  # df for part(operator)
        ms_part = ss_part / df_part if df_part > 0 else 0

        # Repeatability (within)
        ss_error = sum((data[measurement] - data.groupby([operator, part])[measurement].transform('mean')) ** 2)
        df_error = len(data) - data.groupby([operator, part]).ngroups
        ms_error = float(ss_error / df_error) if df_error > 0 else 0

        # Variance components
        repeat_var = ms_error
        r = reps_per_part
        part_var = max(0, (ms_part - ms_error) / r) if r > 0 else 0
        n_p = parts_per_op
        reprod_var = max(0, (ms_operator - ms_part) / (n_p * r)) if n_p * r > 0 else 0
        gage_rr_var = repeat_var + reprod_var
        total_variation = gage_rr_var + part_var

        pct_gage_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_repeat = 100 * np.sqrt(repeat_var / total_variation) if total_variation > 0 else 0
        pct_reprod = 100 * np.sqrt(reprod_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components (Nested ANOVA):<</COLOR>>\n"
        summary += f"  {'Source':<25} {'Variance':>12} {'%Study Var':>12}\n"
        summary += f"  {'-'*52}\n"
        summary += f"  {'Total Gage R&R':<25} {gage_rr_var:>12.4f} {pct_gage_rr:>11.1f}%\n"
        summary += f"    {'Repeatability':<23} {repeat_var:>12.4f} {pct_repeat:>11.1f}%\n"
        summary += f"    {'Reproducibility':<23} {reprod_var:>12.4f} {pct_reprod:>11.1f}%\n"
        summary += f"  {'Part-to-Part':<25} {part_var:>12.4f} {pct_part:>11.1f}%\n\n"

        if pct_gage_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT — Gage R&R < 10%<</COLOR>>\n"
        elif pct_gage_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL — Gage R&R 10-30%<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:bad>>UNACCEPTABLE — Gage R&R > 30%<</COLOR>>\n"

        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"

        # Plots
        result["plots"].append({
            "data": [{"type": "box", "x": data[operator].astype(str).tolist(), "y": data[measurement].tolist(), "marker": {"color": "#4a9f6e"}}],
            "layout": {"title": f"{measurement} by {operator}", "xaxis": {"title": operator}, "template": "plotly_white"}
        })
        result["plots"].append({
            "data": [{"type": "bar", "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                      "y": [pct_gage_rr, pct_repeat, pct_reprod, pct_part],
                      "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]}}],
            "layout": {"title": "Components of Variation", "yaxis": {"title": "% Study Var"}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Nested Gage R&R = {pct_gage_rr:.1f}%, NDC={ndc}. " + ("Acceptable." if pct_gage_rr < 30 else "Needs improvement.")
        result["statistics"] = {"gage_rr_pct": float(pct_gage_rr), "repeatability_pct": float(pct_repeat), "reproducibility_pct": float(pct_reprod), "part_pct": float(pct_part), "ndc": ndc}

    elif analysis_id == "gage_rr_expanded":
        """
        Expanded Gage R&R — GLM-based MSA with up to 8 factors.
        Beyond the standard part/operator model, includes additional factors
        (fixture, environment, time, etc.) to identify all sources of measurement variation.
        Uses Type II sums of squares and EMS rules for variance decomposition.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        factors_list = config.get("factors") or []  # additional factors beyond part
        operator = config.get("operator")

        # Build factor list: part is always included; operator and others are additional
        all_factors = [part]
        if operator:
            all_factors.append(operator)
        all_factors.extend([f for f in factors_list if f and f not in all_factors and f != part])

        cols_needed = [measurement] + [f for f in all_factors if f]
        cols_needed = [c for c in cols_needed if c in df.columns]
        data_grr = df[cols_needed].dropna()

        if len(data_grr) < 10:
            result["summary"] = "Need at least 10 observations for expanded Gage R&R."
            return result

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>EXPANDED GAGE R&R (Multi-Factor MSA)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(f for f in all_factors if f != part)}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(data_grr)}\n\n"

        grand_mean = float(data_grr[measurement].mean())
        total_var = float(data_grr[measurement].var(ddof=1))

        # Compute variance components for each factor using ANOVA-style decomposition
        var_components = {}

        # Part variation
        part_means = data_grr.groupby(part)[measurement].mean()
        n_parts = data_grr[part].nunique()

        # Factor variations
        for f_name in all_factors:
            if f_name not in data_grr.columns:
                continue
            f_means = data_grr.groupby(f_name)[measurement].mean()
            n_levels = data_grr[f_name].nunique()
            n_per_level = len(data_grr) / n_levels
            ss_f = float(np.sum((f_means - grand_mean) ** 2) * n_per_level)
            df_f = n_levels - 1
            ms_f = ss_f / df_f if df_f > 0 else 0
            var_components[f_name] = {
                "n_levels": n_levels, "ss": ss_f, "df": df_f, "ms": ms_f,
            }

        # Repeatability: residual after all factors
        # Group by all factors, compute within-cell variance
        if len(all_factors) > 1:
            cell_vars = data_grr.groupby(all_factors)[measurement].var(dropna=True)
            cell_counts = data_grr.groupby(all_factors)[measurement].count()
            repeatability_var = float(cell_vars.mean()) if not cell_vars.isna().all() else 0
        else:
            within_var = data_grr.groupby(part)[measurement].var(dropna=True)
            repeatability_var = float(within_var.mean()) if not within_var.isna().all() else 0

        # Estimate variance component for each factor
        for f_name, comp in var_components.items():
            n_per = len(data_grr) / comp["n_levels"]
            est_var = max(0, (comp["ms"] - repeatability_var) / n_per) if n_per > 0 else 0
            comp["var_est"] = est_var

        # Part variation
        part_var = var_components.get(part, {}).get("var_est", 0)

        # Reproducibility: sum of all non-part factor variances
        reproducibility_var = sum(comp["var_est"] for f_name, comp in var_components.items() if f_name != part)

        # Gage R&R
        gage_rr_var = repeatability_var + reproducibility_var
        total_variation = gage_rr_var + part_var
        if total_variation <= 0:
            total_variation = total_var

        # Study var percentages (using std dev ratios)
        pct_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_repeat = 100 * np.sqrt(repeatability_var / total_variation) if total_variation > 0 else 0
        pct_reprod = 100 * np.sqrt(reproducibility_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
        summary += f"  {'Source':<20} {'VarComp':>12} {'%StudyVar':>12}\n"
        summary += f"  {'-' * 46}\n"
        summary += f"  {'Total Gage R&R':<20} {gage_rr_var:>12.4f} {pct_rr:>11.1f}%\n"
        summary += f"    {'Repeatability':<18} {repeatability_var:>12.4f} {pct_repeat:>11.1f}%\n"
        summary += f"    {'Reproducibility':<18} {reproducibility_var:>12.4f} {pct_reprod:>11.1f}%\n"

        for f_name, comp in var_components.items():
            if f_name != part:
                pct_f = 100 * np.sqrt(comp["var_est"] / total_variation) if total_variation > 0 else 0
                summary += f"      {f_name:<16} {comp['var_est']:>12.4f} {pct_f:>11.1f}%\n"

        summary += f"  {'Part-to-Part':<20} {part_var:>12.4f} {pct_part:>11.1f}%\n\n"

        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0

        summary += f"<<COLOR:text>>Assessment:<</COLOR>>\n"
        if pct_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
        elif pct_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:warning>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories (NDC):<</COLOR>> {ndc}\n"

        # Identify largest source of measurement variation
        if reproducibility_var > repeatability_var * 1.5 and len(var_components) > 1:
            worst_factor = max(
                [(f, c["var_est"]) for f, c in var_components.items() if f != part],
                key=lambda x: x[1], default=(None, 0)
            )
            if worst_factor[0]:
                summary += f"\n<<COLOR:text>>Largest reproducibility source:<</COLOR>> {worst_factor[0]}\n"
                summary += f"  Consider standardizing {worst_factor[0]} to reduce measurement variation.\n"
        elif repeatability_var > reproducibility_var * 1.5:
            summary += f"\n<<COLOR:text>>Repeatability dominates — improve gage precision or measurement procedure.<</COLOR>>\n"

        result["summary"] = summary

        # Components of variation bar chart
        bar_labels = ["Gage R&R", "Repeatability", "Reproducibility"]
        bar_vals = [pct_rr, pct_repeat, pct_reprod]
        bar_colors = ["#e89547", "#4a9f6e", "#47a5e8"]
        for f_name, comp in var_components.items():
            if f_name != part:
                pct_f = 100 * np.sqrt(comp["var_est"] / total_variation) if total_variation > 0 else 0
                bar_labels.append(f_name)
                bar_vals.append(pct_f)
                bar_colors.append("#9aaa9a")
        bar_labels.append("Part-to-Part")
        bar_vals.append(pct_part)
        bar_colors.append("#d9a04a")

        result["plots"].append({
            "data": [{"type": "bar", "x": bar_labels, "y": bar_vals, "marker": {"color": bar_colors}}],
            "layout": {"title": "Components of Variation", "yaxis": {"title": "% Study Variation"},
                        "template": "plotly_white", "height": 300}
        })

        # Measurement by Part
        result["plots"].append({
            "data": [{"type": "box", "x": data_grr[part].astype(str).tolist(), "y": data_grr[measurement].tolist(),
                      "marker": {"color": "#4a9f6e"}}],
            "layout": {"title": f"{measurement} by {part}", "height": 250, "template": "plotly_white"}
        })

        # Measurement by each factor
        for f_name in all_factors:
            if f_name != part and f_name in data_grr.columns:
                result["plots"].append({
                    "data": [{"type": "box", "x": data_grr[f_name].astype(str).tolist(),
                              "y": data_grr[measurement].tolist(), "marker": {"color": "#47a5e8"}}],
                    "layout": {"title": f"{measurement} by {f_name}", "height": 250, "template": "plotly_white"}
                })

        result["guide_observation"] = f"Expanded Gage R&R = {pct_rr:.1f}%, NDC={ndc}. {len(all_factors)} factors analyzed."
        result["statistics"] = {
            "gage_rr_pct": float(pct_rr), "repeatability_pct": float(pct_repeat),
            "reproducibility_pct": float(pct_reprod), "part_pct": float(pct_part),
            "ndc": ndc, "n_factors": len(all_factors),
            "factor_variances": {f: float(c["var_est"]) for f, c in var_components.items()},
        }

    elif analysis_id == "gage_linearity_bias":
        """
        Gage Linearity & Bias Study — measures how bias changes across the measurement range.
        Bias = average measured − reference. Linearity = slope of bias vs reference.
        """
        measurement = config.get("measurement")
        reference = config.get("reference")  # known/reference values

        data = df[[measurement, reference]].dropna()
        bias = data[measurement] - data[reference]
        data_with_bias = data.copy()
        data_with_bias["bias"] = bias

        # Linearity regression: bias = a + b * reference
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(data[reference], bias)

        ref_mean = data[reference].mean()
        overall_bias = bias.mean()
        bias_pct = 100 * abs(overall_bias) / data[reference].std() if data[reference].std() > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAGE LINEARITY & BIAS STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference:<</COLOR>> {reference}\n"
        summary += f"<<COLOR:highlight>>N observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>BIAS:<</COLOR>>\n"
        summary += f"  Overall Bias = {overall_bias:.4f}\n"
        summary += f"  Bias as % of Process Variation ≈ {bias_pct:.1f}%\n"
        summary += f"  {'Bias is significant' if abs(overall_bias) / (bias.std() / np.sqrt(len(bias))) > 2 else 'Bias is not significant'}\n\n"

        summary += f"<<COLOR:text>>LINEARITY:<</COLOR>>\n"
        summary += f"  Bias = {intercept:.4f} + {slope:.4f} × Reference\n"
        summary += f"  Slope = {slope:.4f} (p = {p_value:.4f})\n"
        summary += f"  R² = {r_value**2:.4f}\n"
        summary += f"  {'Linearity is significant (bias changes across range)' if p_value < 0.05 else 'Linearity is not significant (bias is constant)'}\n"

        # Scatter: bias vs reference
        ref_range = np.linspace(data[reference].min(), data[reference].max(), 100)
        fit_line = intercept + slope * ref_range

        result["plots"].append({
            "data": [
                {"x": data[reference].tolist(), "y": bias.tolist(), "mode": "markers", "name": "Bias", "marker": {"color": "#4a90d9", "size": 6}},
                {"x": ref_range.tolist(), "y": fit_line.tolist(), "mode": "lines", "name": f"Fit (slope={slope:.4f})", "line": {"color": "#d94a4a", "width": 2}},
                {"x": ref_range.tolist(), "y": [0] * len(ref_range), "mode": "lines", "name": "Zero bias", "line": {"color": "#4a9f6e", "dash": "dash", "width": 1}}
            ],
            "layout": {"title": "Gage Linearity (Bias vs Reference)", "xaxis": {"title": "Reference Value"}, "yaxis": {"title": "Bias (Measured − Reference)"}, "template": "plotly_white"}
        })

        # Bias by reference level (grouped)
        ref_groups = pd.qcut(data[reference], min(5, data[reference].nunique()), duplicates='drop')
        grouped_bias = data_with_bias.groupby(ref_groups, observed=False)["bias"].agg(["mean", "std", "count"])

        result["plots"].append({
            "data": [{"type": "bar", "x": [str(g) for g in grouped_bias.index], "y": grouped_bias["mean"].tolist(),
                      "error_y": {"type": "data", "array": (grouped_bias["std"] / np.sqrt(grouped_bias["count"])).tolist(), "visible": True},
                      "marker": {"color": "#e89547"}}],
            "layout": {"title": "Average Bias by Reference Level", "xaxis": {"title": "Reference Range"}, "yaxis": {"title": "Average Bias"}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Linearity: slope={slope:.4f} (p={p_value:.4f}), overall bias={overall_bias:.4f}. " + ("Linearity issue detected." if p_value < 0.05 else "No linearity issue.")
        result["statistics"] = {"bias": float(overall_bias), "slope": float(slope), "intercept": float(intercept), "r_squared": float(r_value**2), "p_value": float(p_value), "bias_pct": float(bias_pct)}

    elif analysis_id == "gage_type1":
        """
        Type 1 Gage Study — single part measured repeatedly by one operator.
        Assesses basic repeatability and bias against a known reference value.
        Cg and Cgk indices.
        """
        measurement = config.get("measurement")
        ref_value = float(config.get("reference", 0))
        tolerance = float(config.get("tolerance", 1.0))  # total tolerance = USL - LSL

        data = df[measurement].dropna()
        n = len(data)
        mean_val = data.mean()
        std_val = data.std(ddof=1)
        bias = mean_val - ref_value
        t_stat = bias / (std_val / np.sqrt(n)) if std_val > 0 else 0
        from scipy.stats import t as t_dist
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), n - 1))

        # Cg and Cgk indices
        k = 0.2  # 20% of tolerance (industry standard)
        cg = (k * tolerance) / (6 * std_val) if std_val > 0 else 0
        cgk = (k * tolerance / 2 - abs(bias)) / (3 * std_val) if std_val > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TYPE 1 GAGE STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference value:<</COLOR>> {ref_value}\n"
        summary += f"<<COLOR:highlight>>Tolerance:<</COLOR>> {tolerance}\n"
        summary += f"<<COLOR:highlight>>N measurements:<</COLOR>> {n}\n\n"

        summary += f"<<COLOR:text>>Descriptive Statistics:<</COLOR>>\n"
        summary += f"  Mean = {mean_val:.4f}\n"
        summary += f"  Std Dev = {std_val:.4f}\n"
        summary += f"  Bias = {bias:.4f} (Mean − Ref)\n"
        summary += f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}\n"
        summary += f"  {'Bias is significant' if p_val < 0.05 else 'Bias is not significant'}\n\n"

        summary += f"<<COLOR:text>>Capability Indices:<</COLOR>>\n"
        summary += f"  Cg  = {cg:.3f} {'(≥ 1.33 required)' if cg < 1.33 else '✓'}\n"
        summary += f"  Cgk = {cgk:.3f} {'(≥ 1.33 required)' if cgk < 1.33 else '✓'}\n\n"

        if cg >= 1.33 and cgk >= 1.33:
            summary += f"  <<COLOR:good>>ACCEPTABLE — both Cg and Cgk ≥ 1.33<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:bad>>NOT ACCEPTABLE — improve repeatability or reduce bias<</COLOR>>\n"

        # Histogram with reference line
        result["plots"].append({
            "data": [
                {"type": "histogram", "x": data.tolist(), "marker": {"color": "#4a90d9", "opacity": 0.7}, "name": "Measurements"},
                {"type": "scatter", "x": [ref_value, ref_value], "y": [0, n * 0.3], "mode": "lines", "name": f"Ref = {ref_value}", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}}
            ],
            "layout": {"title": "Type 1 Gage Study — Measurement Distribution", "xaxis": {"title": measurement}, "yaxis": {"title": "Count"}, "template": "plotly_white"}
        })

        # Run chart
        result["plots"].append({
            "data": [
                {"x": list(range(1, n + 1)), "y": data.tolist(), "mode": "lines+markers", "name": "Measurements", "line": {"color": "#4a90d9", "width": 1}, "marker": {"size": 4}},
                {"x": [1, n], "y": [ref_value, ref_value], "mode": "lines", "name": "Reference", "line": {"color": "#d94a4a", "dash": "dash", "width": 2}},
                {"x": [1, n], "y": [mean_val, mean_val], "mode": "lines", "name": f"Mean = {mean_val:.4f}", "line": {"color": "#4a9f6e", "width": 1}}
            ],
            "layout": {"title": "Run Chart", "xaxis": {"title": "Observation"}, "yaxis": {"title": measurement}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Type 1 Gage: Cg={cg:.3f}, Cgk={cgk:.3f}, bias={bias:.4f} (p={p_val:.4f}). " + ("Acceptable." if cg >= 1.33 and cgk >= 1.33 else "Not acceptable.")
        result["statistics"] = {"mean": float(mean_val), "std": float(std_val), "bias": float(bias), "p_value": float(p_val), "cg": float(cg), "cgk": float(cgk)}

    elif analysis_id == "attribute_gage":
        """
        Attribute Gage Study (Binary) — evaluate an attribute measurement system.
        Each appraiser classifies parts as pass/fail. Compare to known reference.
        Reports % agreement, % effectiveness (detection rate), false alarm rate.
        """
        result_col = config.get("result") or config.get("measurement")  # appraiser's call
        reference_col = config.get("reference")  # known good/bad
        appraiser_col = config.get("appraiser") or config.get("operator")

        data = df[[result_col, reference_col]].dropna()
        if appraiser_col and appraiser_col in df.columns:
            data = df[[result_col, reference_col, appraiser_col]].dropna()
        else:
            appraiser_col = None

        # Ensure binary
        result_vals = data[result_col].astype(str)
        ref_vals = data[reference_col].astype(str)
        unique_vals = sorted(set(result_vals.unique()) | set(ref_vals.unique()), key=str)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ATTRIBUTE GAGE STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Appraisal:<</COLOR>> {result_col}\n"
        summary += f"<<COLOR:highlight>>Reference:<</COLOR>> {reference_col}\n"
        summary += f"<<COLOR:highlight>>Categories:<</COLOR>> {unique_vals}\n"
        summary += f"<<COLOR:highlight>>N assessments:<</COLOR>> {len(data)}\n\n"

        # Overall agreement
        agree = (result_vals == ref_vals).sum()
        total = len(data)
        pct_agree = 100 * agree / total if total > 0 else 0

        # If binary (2 categories): compute sensitivity/specificity
        if len(unique_vals) == 2:
            pos_label = unique_vals[1]  # assume second value is "positive/fail/defective"
            tp = ((result_vals == pos_label) & (ref_vals == pos_label)).sum()
            tn = ((result_vals != pos_label) & (ref_vals != pos_label)).sum()
            fp = ((result_vals == pos_label) & (ref_vals != pos_label)).sum()
            fn = ((result_vals != pos_label) & (ref_vals == pos_label)).sum()

            sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
            false_alarm = 100 * fp / (fp + tn) if (fp + tn) > 0 else 0
            miss_rate = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0

            summary += f"<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
            summary += f"  {'':>15} {'Ref: ' + str(unique_vals[0]):>15} {'Ref: ' + str(unique_vals[1]):>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[0]):>15} {tn:>15} {fn:>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[1]):>15} {fp:>15} {tp:>15}\n\n"

            summary += f"<<COLOR:text>>Effectiveness Metrics:<</COLOR>>\n"
            summary += f"  Overall Agreement:  {pct_agree:.1f}%\n"
            summary += f"  Sensitivity (detect): {sensitivity:.1f}%\n"
            summary += f"  Specificity (accept): {specificity:.1f}%\n"
            summary += f"  False Alarm Rate:   {false_alarm:.1f}%\n"
            summary += f"  Miss Rate:          {miss_rate:.1f}%\n\n"

            if pct_agree >= 90:
                summary += f"  <<COLOR:good>>ACCEPTABLE — agreement ≥ 90%<</COLOR>>\n"
            else:
                summary += f"  <<COLOR:bad>>NOT ACCEPTABLE — agreement < 90%<</COLOR>>\n"

            # Confusion matrix heatmap
            result["plots"].append({
                "data": [{"type": "heatmap", "z": [[tn, fn], [fp, tp]], "x": [str(unique_vals[0]), str(unique_vals[1])],
                          "y": [str(unique_vals[0]), str(unique_vals[1])], "text": [[str(tn), str(fn)], [str(fp), str(tp)]],
                          "texttemplate": "%{text}", "colorscale": [[0, "#f0f0f0"], [1, "#4a90d9"]], "showscale": False}],
                "layout": {"title": "Confusion Matrix", "xaxis": {"title": "Reference"}, "yaxis": {"title": "Appraiser Call"}, "template": "plotly_white"}
            })

            result["statistics"] = {"agreement_pct": float(pct_agree), "sensitivity": float(sensitivity), "specificity": float(specificity), "false_alarm_rate": float(false_alarm), "miss_rate": float(miss_rate)}
        else:
            summary += f"<<COLOR:text>>Overall Agreement: {pct_agree:.1f}% ({agree}/{total})<</COLOR>>\n"
            result["statistics"] = {"agreement_pct": float(pct_agree)}

        # By appraiser if available
        if appraiser_col:
            appraisers = sorted(data[appraiser_col].unique(), key=str)
            app_agree = []
            for app in appraisers:
                mask = data[appraiser_col] == app
                a = (data.loc[mask, result_col].astype(str) == data.loc[mask, reference_col].astype(str)).mean() * 100
                app_agree.append(a)
            result["plots"].append({
                "data": [{"type": "bar", "x": [str(a) for a in appraisers], "y": app_agree,
                          "marker": {"color": ["#4a9f6e" if a >= 90 else "#d94a4a" for a in app_agree]}}],
                "layout": {"title": "Agreement % by Appraiser", "xaxis": {"title": "Appraiser"}, "yaxis": {"title": "% Agreement", "range": [0, 100]}, "template": "plotly_white"}
            })

        result["summary"] = summary
        result["guide_observation"] = f"Attribute gage: {pct_agree:.1f}% overall agreement. " + ("Acceptable." if pct_agree >= 90 else "Needs improvement.")

    elif analysis_id == "attribute_agreement":
        """
        Attribute Agreement Analysis — Kappa and Kendall statistics for multiple appraisers.
        Cohen's Kappa (2 raters) or Fleiss' Kappa (3+ raters).
        """
        appraiser_col = config.get("appraiser") or config.get("operator")
        part_col = config.get("part") or config.get("item")
        rating_col = config.get("rating") or config.get("measurement")

        data = df[[appraiser_col, part_col, rating_col]].dropna()
        appraisers = sorted(data[appraiser_col].unique(), key=str)
        parts = sorted(data[part_col].unique(), key=str)
        categories = sorted(data[rating_col].unique(), key=str)

        n_appraisers = len(appraisers)
        n_parts = len(parts)
        n_categories = len(categories)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ATTRIBUTE AGREEMENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Appraisers:<</COLOR>> {n_appraisers} ({', '.join(str(a) for a in appraisers)})\n"
        summary += f"<<COLOR:highlight>>Parts:<</COLOR>> {n_parts}\n"
        summary += f"<<COLOR:highlight>>Categories:<</COLOR>> {categories}\n\n"

        # Build rating matrix: rows = parts, columns = appraisers
        pivot = data.pivot_table(index=part_col, columns=appraiser_col, values=rating_col, aggfunc='first')

        # Within-appraiser agreement (if repeated trials)
        trial_counts = data.groupby([appraiser_col, part_col]).size()
        has_repeats = (trial_counts > 1).any()

        # Between-appraiser agreement
        pairwise_kappas = []
        if n_appraisers == 2:
            # Cohen's Kappa
            r1 = pivot.iloc[:, 0].astype(str).values
            r2 = pivot.iloc[:, 1].astype(str).values
            mask = ~pd.isna(r1) & ~pd.isna(r2)
            r1, r2 = r1[mask], r2[mask]
            n = len(r1)
            po = (r1 == r2).mean()
            pe = sum((r1 == c).mean() * (r2 == c).mean() for c in categories)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
            pairwise_kappas.append(("All", kappa))

            summary += f"<<COLOR:text>>Cohen's Kappa (2 raters):<</COLOR>>\n"
            summary += f"  κ = {kappa:.4f}\n"
            summary += f"  Observed agreement: {po:.1%}\n"
            summary += f"  Expected agreement: {pe:.1%}\n\n"
        else:
            # Fleiss' Kappa for 3+ raters
            # Build category count matrix
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            n_matrix = np.zeros((n_parts, n_categories))
            for i, p in enumerate(parts):
                part_data = data[data[part_col] == p][rating_col].astype(str)
                for val in part_data:
                    if val in cat_to_idx:
                        n_matrix[i, cat_to_idx[val]] += 1

            N = n_parts
            n_raters = n_matrix.sum(axis=1).mean()  # average raters per item
            p_j = n_matrix.sum(axis=0) / (N * n_raters)  # proportion in each category
            Pe = (p_j ** 2).sum()

            Pi = (n_matrix ** 2).sum(axis=1) - n_raters
            Pi = Pi / (n_raters * (n_raters - 1)) if n_raters > 1 else Pi
            Po = Pi.mean()

            kappa = (Po - Pe) / (1 - Pe) if Pe < 1 else 1.0

            summary += f"<<COLOR:text>>Fleiss' Kappa ({n_appraisers} raters):<</COLOR>>\n"
            summary += f"  κ = {kappa:.4f}\n"
            summary += f"  Observed agreement: {Po:.1%}\n"
            summary += f"  Expected agreement: {Pe:.1%}\n\n"

            # Also compute pairwise Cohens
            for i in range(n_appraisers):
                for j in range(i + 1, n_appraisers):
                    r1 = pivot.iloc[:, i].astype(str).values
                    r2 = pivot.iloc[:, j].astype(str).values
                    mask = ~pd.isna(r1) & ~pd.isna(r2)
                    if mask.sum() > 0:
                        r1m, r2m = r1[mask], r2[mask]
                        po_pair = (r1m == r2m).mean()
                        pe_pair = sum((r1m == c).mean() * (r2m == c).mean() for c in categories)
                        k_pair = (po_pair - pe_pair) / (1 - pe_pair) if pe_pair < 1 else 1.0
                        pairwise_kappas.append((f"{appraisers[i]} vs {appraisers[j]}", k_pair))

        # Interpret kappa
        kappa_val = kappa if 'kappa' in dir() else (pairwise_kappas[0][1] if pairwise_kappas else 0)
        if kappa_val >= 0.81:
            interp = "Almost perfect"
        elif kappa_val >= 0.61:
            interp = "Substantial"
        elif kappa_val >= 0.41:
            interp = "Moderate"
        elif kappa_val >= 0.21:
            interp = "Fair"
        else:
            interp = "Slight/Poor"
        summary += f"<<COLOR:text>>Interpretation:<</COLOR>> {interp} agreement (Landis & Koch)\n"

        # Agreement by appraiser
        app_agreements = []
        for app in appraisers:
            app_data = data[data[appraiser_col] == app]
            if has_repeats:
                # Within-appraiser: across trials for same part
                within_agree = app_data.groupby(part_col)[rating_col].apply(lambda g: g.astype(str).nunique() == 1).mean() * 100
            else:
                within_agree = 100.0  # single trial = always agrees with self
            app_agreements.append(within_agree)

        result["plots"].append({
            "data": [{"type": "bar", "x": [str(a) for a in appraisers], "y": app_agreements,
                      "marker": {"color": "#4a90d9"}}],
            "layout": {"title": "Within-Appraiser Agreement %", "xaxis": {"title": "Appraiser"}, "yaxis": {"title": "% Self-Consistent", "range": [0, 100]}, "template": "plotly_white"}
        })

        if pairwise_kappas:
            result["plots"].append({
                "data": [{"type": "bar", "x": [p[0] for p in pairwise_kappas], "y": [p[1] for p in pairwise_kappas],
                          "marker": {"color": ["#4a9f6e" if p[1] >= 0.6 else "#e89547" if p[1] >= 0.4 else "#d94a4a" for p in pairwise_kappas]}}],
                "layout": {"title": "Pairwise Cohen's Kappa", "xaxis": {"title": "Pair"}, "yaxis": {"title": "κ", "range": [-0.2, 1]}, "template": "plotly_white"}
            })

        result["summary"] = summary
        result["guide_observation"] = f"Attribute agreement: κ={kappa_val:.3f} ({interp}). " + ("Good agreement." if kappa_val >= 0.6 else "Agreement needs improvement.")
        result["statistics"] = {"kappa": float(kappa_val), "interpretation": interp, "n_appraisers": n_appraisers, "n_parts": n_parts}

    elif analysis_id == "stepwise":
        """
        Stepwise Regression - automatic variable selection.
        Uses forward selection, backward elimination, or both.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import statsmodels.api as sm

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "both")  # forward, backward, both
        alpha_enter = float(config.get("alpha_enter", 0.05))
        alpha_remove = float(config.get("alpha_remove", 0.10))

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>STEPWISE REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {len(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method}\n"
        summary += f"<<COLOR:highlight>>α to enter:<</COLOR>> {alpha_enter}\n"
        summary += f"<<COLOR:highlight>>α to remove:<</COLOR>> {alpha_remove}\n\n"

        selected = []
        remaining = list(predictors)
        step_history = []

        def get_pvalue(X_subset, y_data, var_name):
            """Get p-value for adding a variable."""
            X_with_const = sm.add_constant(X_subset)
            try:
                model = sm.OLS(y_data, X_with_const).fit()
                idx = list(X_subset.columns).index(var_name) + 1  # +1 for constant
                return model.pvalues.iloc[idx]
            except:
                return 1.0

        # Stepwise selection
        step = 0
        while True:
            step += 1
            changed = False

            # Forward step: try adding variables
            if method in ["forward", "both"] and remaining:
                best_pval = 1.0
                best_var = None

                for var in remaining:
                    if selected:
                        X_test = X_full[selected + [var]]
                    else:
                        X_test = X_full[[var]]

                    pval = get_pvalue(X_test, y, var)
                    if pval < best_pval:
                        best_pval = pval
                        best_var = var

                if best_var and best_pval < alpha_enter:
                    selected.append(best_var)
                    remaining.remove(best_var)
                    step_history.append(f"Step {step}: ADD {best_var} (p={best_pval:.4f})")
                    changed = True

            # Backward step: try removing variables
            if method in ["backward", "both"] and selected:
                worst_pval = 0.0
                worst_var = None

                X_current = X_full[selected]
                X_with_const = sm.add_constant(X_current)
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    for i, var in enumerate(selected):
                        pval = model.pvalues.iloc[i + 1]  # +1 for constant
                        if pval > worst_pval:
                            worst_pval = pval
                            worst_var = var
                except:
                    pass

                if worst_var and worst_pval > alpha_remove:
                    selected.remove(worst_var)
                    remaining.append(worst_var)
                    step_history.append(f"Step {step}: REMOVE {worst_var} (p={worst_pval:.4f})")
                    changed = True

            if not changed:
                break

            if step > 50:  # Safety limit
                break

        summary += f"<<COLOR:text>>Selection History:<</COLOR>>\n"
        for hist in step_history:
            summary += f"  {hist}\n"
        summary += "\n"

        # Final model
        if selected:
            X_final = X_full[selected]
            X_with_const = sm.add_constant(X_final)
            final_model = sm.OLS(y, X_with_const).fit()

            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:accent>>                              FINAL MODEL<</COLOR>>\n"
            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:highlight>>Selected Variables:<</COLOR>> {', '.join(selected)}\n"
            summary += f"<<COLOR:highlight>>R²:<</COLOR>> {final_model.rsquared:.4f}\n"
            summary += f"<<COLOR:highlight>>Adjusted R²:<</COLOR>> {final_model.rsquared_adj:.4f}\n"
            summary += f"<<COLOR:highlight>>F-statistic:<</COLOR>> {final_model.fvalue:.4f} (p={final_model.f_pvalue:.4e})\n\n"

            summary += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}\n"
            summary += f"  {'-'*66}\n"
            for i, name in enumerate(['const'] + selected):
                summary += f"  {name:<20} {final_model.params.iloc[i]:>12.4f} {final_model.bse.iloc[i]:>12.4f} {final_model.tvalues.iloc[i]:>10.3f} {final_model.pvalues.iloc[i]:>10.4f}\n"

            result["statistics"] = {
                "R²": float(final_model.rsquared),
                "Adj_R²": float(final_model.rsquared_adj),
                "n_selected": len(selected),
                "selected_vars": selected
            }

            # Coefficient plot
            result["plots"].append({
                "title": "Stepwise Selected Coefficients",
                "data": [{
                    "type": "bar",
                    "x": selected,
                    "y": [float(final_model.params.iloc[i+1]) for i in range(len(selected))],
                    "marker": {"color": "#4a9f6e"}
                }],
                "layout": {"height": 250, "yaxis": {"title": "Coefficient"}}
            })
        else:
            summary += f"<<COLOR:warning>>No variables met the selection criteria.<</COLOR>>\n"
            result["statistics"] = {"n_selected": 0, "selected_vars": []}

        result["summary"] = summary
        result["guide_observation"] = f"Stepwise regression selected {len(selected)} of {len(predictors)} predictors."

    elif analysis_id == "best_subsets":
        """
        Best Subsets Regression - evaluate all possible models.
        Compares models by R², Adjusted R², Cp, BIC.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from itertools import combinations

        response = config.get("response")
        predictors = config.get("predictors", [])
        max_predictors = min(int(config.get("max_predictors", 8)), len(predictors), 8)

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)
        p_full = len(predictors)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BEST SUBSETS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {p_full}\n"
        summary += f"<<COLOR:highlight>>Max subset size:<</COLOR>> {max_predictors}\n\n"

        # Calculate full model MSE for Cp
        model_full = LinearRegression().fit(X_full, y)
        mse_full = np.mean((y - model_full.predict(X_full))**2)

        results_list = []

        # Evaluate all subsets up to max_predictors
        for k in range(1, max_predictors + 1):
            for combo in combinations(predictors, k):
                X_sub = X_full[list(combo)]
                model = LinearRegression().fit(X_sub, y)
                y_pred = model.predict(X_sub)

                sse = np.sum((y - y_pred)**2)
                r2 = 1 - sse / np.sum((y - y.mean())**2)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                # Mallows' Cp
                cp = sse / mse_full - n + 2 * (k + 1)

                # BIC
                bic = n * np.log(sse / n) + k * np.log(n)

                results_list.append({
                    "vars": combo,
                    "k": k,
                    "r2": r2,
                    "adj_r2": adj_r2,
                    "cp": cp,
                    "bic": bic
                })

        # Sort by different criteria
        by_r2 = sorted(results_list, key=lambda x: -x["r2"])
        by_adj_r2 = sorted(results_list, key=lambda x: -x["adj_r2"])
        by_cp = sorted(results_list, key=lambda x: x["cp"])
        by_bic = sorted(results_list, key=lambda x: x["bic"])

        summary += f"<<COLOR:text>>Best Models by Criterion:<</COLOR>>\n\n"

        summary += f"<<COLOR:accent>>By Adjusted R² (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10} {'Predictors'}\n"
        summary += f"  {'-'*70}\n"
        for res in by_adj_r2[:5]:
            vars_str = ', '.join(res['vars'][:3]) + ('...' if len(res['vars']) > 3 else '')
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f} {vars_str}\n"

        summary += f"\n<<COLOR:accent>>By Mallows' Cp (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10}\n"
        summary += f"  {'-'*50}\n"
        for res in by_cp[:5]:
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f}\n"

        # Best model recommendation
        best = by_adj_r2[0]
        summary += f"\n<<COLOR:success>>RECOMMENDED MODEL:<</COLOR>>\n"
        summary += f"  Variables: {', '.join(best['vars'])}\n"
        summary += f"  R² = {best['r2']:.4f}, Adj R² = {best['adj_r2']:.4f}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Best subsets: recommended {best['k']}-variable model with Adj R² = {best['adj_r2']:.4f}"
        result["statistics"] = {
            "best_r2": float(best["r2"]),
            "best_adj_r2": float(best["adj_r2"]),
            "best_vars": list(best["vars"]),
            "models_evaluated": len(results_list)
        }

        # Plot: R² and Adj R² by number of variables
        k_values = sorted(set(r["k"] for r in results_list))
        best_r2_by_k = [max(r["r2"] for r in results_list if r["k"] == k) for k in k_values]
        best_adj_r2_by_k = [max(r["adj_r2"] for r in results_list if r["k"] == k) for k in k_values]

        result["plots"].append({
            "title": "Best Subsets: R² by Model Size",
            "data": [
                {"type": "scatter", "x": k_values, "y": best_r2_by_k, "mode": "lines+markers", "name": "R²", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": k_values, "y": best_adj_r2_by_k, "mode": "lines+markers", "name": "Adj R²", "line": {"color": "#47a5e8"}}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Number of Predictors"}, "yaxis": {"title": "R²"}}
        })

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
        summary += f"<<COLOR:title>>BOOTSTRAP CONFIDENCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Statistic:<</COLOR>> {statistic}\n"
        summary += f"<<COLOR:highlight>>Bootstrap samples:<</COLOR>> {n_bootstrap}\n"
        summary += f"<<COLOR:highlight>>Confidence level:<</COLOR>> {conf_level*100:.0f}%\n\n"

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
                stat_func = lambda x: np.std(x, ddof=1)
            elif statistic == "trimmed_mean":
                from scipy.stats import trim_mean
                observed = trim_mean(data, 0.1)
                stat_func = lambda x: trim_mean(x, 0.1)
            else:
                observed = np.mean(data)
                stat_func = np.mean

            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data, n, replace=True)
                boot_stats.append(stat_func(boot_sample))

        boot_stats = np.array(boot_stats)

        # Calculate CI using percentile method
        alpha = 1 - conf_level
        ci_lower = np.percentile(boot_stats, alpha/2 * 100)
        ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)

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
        a = np.sum((jack_mean - jackknife_stats)**3) / (6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5 + 1e-10)

        # BCa quantiles
        z_alpha_low = stats.norm.ppf(alpha/2)
        z_alpha_high = stats.norm.ppf(1 - alpha/2)

        bca_low_q = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        bca_high_q = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        bca_lower = np.percentile(boot_stats, bca_low_q * 100)
        bca_upper = np.percentile(boot_stats, bca_high_q * 100)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Observed {statistic}: {observed:.4f}\n"
        summary += f"  Bootstrap SE: {np.std(boot_stats):.4f}\n"
        summary += f"  Bootstrap Bias: {np.mean(boot_stats) - observed:.4f}\n\n"

        summary += f"<<COLOR:text>>Confidence Intervals:<</COLOR>>\n"
        summary += f"  Percentile: ({ci_lower:.4f}, {ci_upper:.4f})\n"
        summary += f"  BCa:        ({bca_lower:.4f}, {bca_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {conf_level*100:.0f}% confident the true {statistic}\n"
        summary += f"  lies between {bca_lower:.4f} and {bca_upper:.4f}.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Bootstrap {conf_level*100:.0f}% CI for {statistic}: ({bca_lower:.4f}, {bca_upper:.4f})"
        result["statistics"] = {
            f"observed_{statistic}": float(observed),
            "ci_lower": float(bca_lower),
            "ci_upper": float(bca_upper),
            "bootstrap_se": float(np.std(boot_stats))
        }

        # Histogram of bootstrap distribution
        result["plots"].append({
            "title": f"Bootstrap Distribution of {statistic.title()}",
            "data": [
                {"type": "histogram", "x": boot_stats.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}},
                {"type": "scatter", "x": [observed, observed], "y": [0, n_bootstrap/20], "mode": "lines", "line": {"color": "#e89547", "width": 2, "dash": "dash"}, "name": "Observed"},
                {"type": "scatter", "x": [bca_lower, bca_upper], "y": [0, 0], "mode": "markers", "marker": {"color": "#e85747", "size": 12, "symbol": "triangle-up"}, "name": "CI bounds"}
            ],
            "layout": {"height": 250, "xaxis": {"title": statistic.title()}, "yaxis": {"title": "Frequency"}}
        })

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
        summary += f"<<COLOR:title>>BOX-COX TRANSFORMATION<</COLOR>>\n"
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

        summary += f"<<COLOR:text>>Common Transformations (Log-Likelihood):<</COLOR>>\n"
        for lam, name in zip(lambdas, lambda_names):
            if lam == 0:
                trans = np.log(data_shifted)
            else:
                trans = (data_shifted**lam - 1) / lam
            # Calculate log-likelihood
            ll = -len(data)/2 * np.log(np.var(trans)) + (lam - 1) * np.sum(np.log(data_shifted))
            summary += f"  λ = {lam:>5} ({name:<8}): LL = {ll:.2f}\n"

        summary += f"\n<<COLOR:success>>OPTIMAL TRANSFORMATION:<</COLOR>>\n"
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
        _, p_before = stats.shapiro(data[:min(5000, len(data))])
        _, p_after = stats.shapiro(transformed[:min(5000, len(transformed))])

        summary += f"<<COLOR:text>>Normality Tests (Shapiro-Wilk):<</COLOR>>\n"
        summary += f"  Original: p = {p_before:.4f} {'(normal)' if p_before > 0.05 else '(non-normal)'}\n"
        summary += f"  Transformed: p = {p_after:.4f} {'(normal)' if p_after > 0.05 else '(non-normal)'}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Box-Cox optimal λ = {optimal_lambda:.3f}. {suggestion}."
        result["statistics"] = {
            "optimal_lambda": float(optimal_lambda),
            "p_before": float(p_before),
            "p_after": float(p_after),
            "shift_applied": float(shift)
        }

        # Plot: original vs transformed distributions
        result["plots"].append({
            "title": "Original Distribution",
            "data": [{"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(232, 87, 71, 0.4)", "line": {"color": "#e85747", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": var}}
        })

        result["plots"].append({
            "title": f"Transformed (λ = {optimal_lambda:.2f})",
            "data": [{"type": "histogram", "x": transformed.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": f"Box-Cox({var})"}}
        })

        # Lambda vs log-likelihood profile
        lambda_range = np.linspace(max(-3, optimal_lambda - 2), min(3, optimal_lambda + 2), 50)
        log_likelihoods = []
        for lam in lambda_range:
            if abs(lam) < 1e-10:
                trans = np.log(data_shifted)
            else:
                trans = (data_shifted**lam - 1) / lam
            ll = -len(data)/2 * np.log(np.var(trans)) + (lam - 1) * np.sum(np.log(data_shifted))
            log_likelihoods.append(float(ll))
        result["plots"].append({
            "title": "Lambda vs Log-Likelihood",
            "data": [
                {"type": "scatter", "x": lambda_range.tolist(), "y": log_likelihoods, "mode": "lines", "line": {"color": "#4a9f6e", "width": 2}, "name": "Log-Likelihood"},
                {"type": "scatter", "x": [float(optimal_lambda)], "y": [max(log_likelihoods)], "mode": "markers", "marker": {"color": "#d94a4a", "size": 10, "symbol": "diamond"}, "name": f"Optimal λ = {optimal_lambda:.3f}"}
            ],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Lambda (λ)"}, "yaxis": {"title": "Log-Likelihood"}}
        })

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
                std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))

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

                cluster_flag = "<<COLOR:danger>>Yes<<\/COLOR>>" if p_clustering < 0.05 else "<<COLOR:success>>No<<\/COLOR>>"
                mixture_flag = "<<COLOR:danger>>Yes<<\/COLOR>>" if p_mixtures < 0.05 else "<<COLOR:success>>No<<\/COLOR>>"

                summary = f"""<<COLOR:title>>RUN CHART<<\/COLOR>>
{'='*50}
<<COLOR:highlight>>Variable:<<\/COLOR>> {var}
<<COLOR:highlight>>Observations:<<\/COLOR>> {n}
<<COLOR:highlight>>Median:<<\/COLOR>> {median_val:.6g}

<<COLOR:accent>>Runs Test Results<<\/COLOR>>
  Number of runs:    {runs}
  Expected runs:     {expected_runs:.1f}
  Longest run:       {longest_run}
  Points above median: {n_above}
  Points below median: {n_below}

  Clustering (too few runs)?   {cluster_flag}  (p = {p_clustering:.4f})
  Mixtures (too many runs)?    {mixture_flag}  (p = {p_mixtures:.4f})"""
            else:
                summary = f"""<<COLOR:title>>RUN CHART<<\/COLOR>>
{'='*50}
Variable: {var}  |  N = {n}  |  Median = {median_val:.6g}
<<COLOR:warning>>All values on same side of median — runs test not applicable<<\/COLOR>>"""
                runs = 0

            result["summary"] = summary
            result["guide_observation"] = f"Run chart: {n} obs, median={median_val:.4g}, {runs} runs."

            # Plot
            traces = [
                {"type": "scatter", "x": x_vals, "y": vals.tolist(), "mode": "lines+markers",
                 "marker": {"color": "#4a9f6e", "size": 5}, "line": {"color": "#4a9f6e", "width": 1.5}, "name": var},
                {"type": "scatter", "x": [x_vals[0], x_vals[-1]], "y": [median_val, median_val],
                 "mode": "lines", "line": {"color": "#e89547", "dash": "dash", "width": 2}, "name": f"Median = {median_val:.4g}"}
            ]
            result["plots"].append({
                "title": f"Run Chart: {var}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"title": time_col if time_col else "Observation"}, "yaxis": {"title": var}, "showlegend": True}
            })

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
                G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

                # Two-sided p-value (approximation)
                t_stat = G * np.sqrt(n) / np.sqrt(n - 1)
                p_val = min(1.0, 2 * n * (1 - stats.t.cdf(np.sqrt((n * (n - 2) * G**2) / (n - 1 - G**2 * (n - 1) / n)), n - 2))) if G**2 < n * (n - 1) / n else 0.0

                is_outlier = G > G_crit
                verdict = "<<COLOR:danger>>Yes — significant outlier<<\/COLOR>>" if is_outlier else "<<COLOR:success>>No — not a significant outlier<<\/COLOR>>"

                summary = f"""<<COLOR:title>>GRUBBS' OUTLIER TEST<<\/COLOR>>
{'='*50}
<<COLOR:highlight>>Variable:<<\/COLOR>> {var}
<<COLOR:highlight>>N:<<\/COLOR>> {n}
<<COLOR:highlight>>Significance level:<<\/COLOR>> {alpha}

<<COLOR:accent>>Results<<\/COLOR>>
  Suspect value: {suspect:.6g}
  Mean:          {mean_val:.6g}
  StDev:         {std_val:.6g}

  G statistic:   {G:.4f}
  G critical:    {G_crit:.4f}

  Outlier? {verdict}"""

                result["summary"] = summary
                result["guide_observation"] = f"Grubbs' test on {var}: suspect={suspect:.4g}, G={G:.3f}, {'outlier' if is_outlier else 'not outlier'} at α={alpha}."
                result["statistics"] = {"G": G, "G_critical": G_crit, "suspect_value": suspect, "is_outlier": is_outlier}

                # Highlight plot
                colors = ["#4a9f6e" if i != max_idx else "#d94a4a" for i in range(n)]
                sizes = [5 if i != max_idx else 12 for i in range(n)]
                result["plots"].append({
                    "title": f"Grubbs' Test: {var}",
                    "data": [
                        {"type": "scatter", "x": list(range(1, n + 1)), "y": vals.tolist(),
                         "mode": "markers", "marker": {"color": colors, "size": sizes}, "name": var},
                        {"type": "scatter", "x": [1, n], "y": [mean_val, mean_val],
                         "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": f"Mean = {mean_val:.4g}"}
                    ],
                    "layout": {"height": 300, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
                })

    # ── Cross-Correlation Function ───────────────────────────────────────
    elif analysis_id == "ccf":
        """
        Cross-Correlation Function — correlation between two time series at various lags.
        """
        var1 = config.get("var1")
        var2 = config.get("var2")
        max_lag = int(config.get("max_lag", 20))

        x = df[var1].dropna().values
        y = df[var2].dropna().values
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if n < 5:
            result["summary"] = "Cross-correlation requires at least 5 observations."
        else:
            # Standardize
            x_std = (x - np.mean(x)) / np.std(x)
            y_std = (y - np.mean(y)) / np.std(y)

            lags = list(range(-max_lag, max_lag + 1))
            ccf_vals = []
            for lag in lags:
                if lag >= 0:
                    cc = np.mean(x_std[:n - lag] * y_std[lag:]) if lag < n else 0.0
                else:
                    cc = np.mean(x_std[-lag:] * y_std[:n + lag]) if -lag < n else 0.0
                ccf_vals.append(float(cc))

            sig_bound = 2.0 / np.sqrt(n)

            # Find significant lags
            sig_lags = [(lag, cc) for lag, cc in zip(lags, ccf_vals) if abs(cc) > sig_bound]

            summary = f"""<<COLOR:title>>CROSS-CORRELATION FUNCTION<<\/COLOR>>
{'='*50}
<<COLOR:highlight>>Series 1:<<\/COLOR>> {var1}
<<COLOR:highlight>>Series 2:<<\/COLOR>> {var2}
<<COLOR:highlight>>N:<<\/COLOR>> {n}
<<COLOR:highlight>>Max lag:<<\/COLOR>> ±{max_lag}
<<COLOR:highlight>>Significance bound:<<\/COLOR>> ±{sig_bound:.4f}

<<COLOR:accent>>Lag 0 correlation:<<\/COLOR>> {ccf_vals[max_lag]:.4f}"""

            if sig_lags:
                summary += f"\n\n<<COLOR:accent>>Significant lags:<<\/COLOR>>"
                for lag, cc in sorted(sig_lags, key=lambda x: abs(x[1]), reverse=True)[:10]:
                    summary += f"\n  Lag {lag:>3d}: r = {cc:+.4f}"

            result["summary"] = summary
            result["guide_observation"] = f"CCF({var1}, {var2}): lag-0 r={ccf_vals[max_lag]:.3f}, {len(sig_lags)} significant lags."
            result["statistics"] = {"lag_0_correlation": ccf_vals[max_lag], "significant_lags": len(sig_lags)}

            result["plots"].append({
                "title": f"Cross-Correlation: {var1} vs {var2}",
                "data": [
                    {"type": "bar", "x": lags, "y": ccf_vals, "marker": {"color": ["#d94a4a" if abs(c) > sig_bound else "#4a9f6e" for c in ccf_vals]}, "name": "CCF"},
                    {"type": "scatter", "x": [-max_lag, max_lag], "y": [sig_bound, sig_bound], "mode": "lines", "line": {"color": "#e89547", "dash": "dash", "width": 1}, "showlegend": False},
                    {"type": "scatter", "x": [-max_lag, max_lag], "y": [-sig_bound, -sig_bound], "mode": "lines", "line": {"color": "#e89547", "dash": "dash", "width": 1}, "showlegend": False}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Lag"}, "yaxis": {"title": "Correlation", "range": [-1.05, 1.05]}}
            })

    # ── Johnson Transformation ───────────────────────────────────────────
    elif analysis_id == "johnson_transform":
        """
        Johnson Transformation — finds optimal Johnson family (SB, SL, SU) to normalize data.
        More general than Box-Cox (handles bounded and unbounded distributions).
        """
        var = config.get("var")

        data = df[var].dropna().values
        n = len(data)

        if n < 10:
            result["summary"] = "Johnson transformation requires at least 10 observations."
        else:
            summary = f"<<COLOR:title>>JOHNSON TRANSFORMATION<<\/COLOR>>\n{'='*50}\n"
            summary += f"<<COLOR:highlight>>Variable:<<\/COLOR>> {var}\n"
            summary += f"<<COLOR:highlight>>N:<<\/COLOR>> {n}\n\n"

            # Test each Johnson family
            families = {}

            # SU (unbounded)
            try:
                params_su = stats.johnsonsu.fit(data)
                transformed_su = stats.johnsonsu.cdf(data, *params_su)
                transformed_su = stats.norm.ppf(np.clip(transformed_su, 0.001, 0.999))
                _, p_su = stats.shapiro(transformed_su[:min(5000, len(transformed_su))])
                families["SU"] = {"params": params_su, "p_value": float(p_su), "transformed": transformed_su}
            except Exception:
                pass

            # SB (bounded)
            try:
                params_sb = stats.johnsonsb.fit(data)
                transformed_sb = stats.johnsonsb.cdf(data, *params_sb)
                transformed_sb = stats.norm.ppf(np.clip(transformed_sb, 0.001, 0.999))
                _, p_sb = stats.shapiro(transformed_sb[:min(5000, len(transformed_sb))])
                families["SB"] = {"params": params_sb, "p_value": float(p_sb), "transformed": transformed_sb}
            except Exception:
                pass

            # SL (lognormal — just use log transform)
            try:
                if np.all(data > 0):
                    transformed_sl = np.log(data)
                    _, p_sl = stats.shapiro(transformed_sl[:min(5000, len(transformed_sl))])
                    families["SL"] = {"params": None, "p_value": float(p_sl), "transformed": transformed_sl}
            except Exception:
                pass

            # Original normality
            _, p_orig = stats.shapiro(data[:min(5000, n)])

            summary += f"<<COLOR:text>>Original Shapiro-Wilk p-value:<<\/COLOR>> {p_orig:.4f}"
            summary += f" {'(normal)' if p_orig > 0.05 else '(non-normal)'}\n\n"

            if families:
                best_family = max(families.keys(), key=lambda k: families[k]["p_value"])
                best = families[best_family]

                summary += f"<<COLOR:accent>>Family Results:<<\/COLOR>>\n"
                for fam_name, fam_data in sorted(families.items(), key=lambda x: -x[1]["p_value"]):
                    marker = " ← Best" if fam_name == best_family else ""
                    p = fam_data["p_value"]
                    status = "<<COLOR:success>>normal<<\/COLOR>>" if p > 0.05 else "<<COLOR:warning>>non-normal<<\/COLOR>>"
                    summary += f"  Johnson {fam_name}: Shapiro-Wilk p = {p:.4f} ({status}){marker}\n"

                summary += f"\n<<COLOR:success>>Best transformation: Johnson {best_family}<<\/COLOR>>\n"

                result["summary"] = summary
                result["guide_observation"] = f"Johnson transform: best family={best_family}, p={best['p_value']:.4f}."
                result["statistics"] = {"best_family": best_family, "p_original": float(p_orig), "p_transformed": best["p_value"]}

                # Plots: before and after
                result["plots"].append({
                    "title": f"Original: {var}",
                    "data": [{"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(232,87,71,0.4)", "line": {"color": "#e85747", "width": 1}}}],
                    "layout": {"height": 200, "xaxis": {"title": var}}
                })
                result["plots"].append({
                    "title": f"Johnson {best_family} Transformed",
                    "data": [{"type": "histogram", "x": best["transformed"].tolist(), "marker": {"color": "rgba(74,159,110,0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
                    "layout": {"height": 200, "xaxis": {"title": f"Johnson {best_family}({var})"}}
                })
            else:
                summary += "\n<<COLOR:warning>>Could not fit any Johnson family to this data.<<\/COLOR>>"
                result["summary"] = summary

    elif analysis_id == "robust_regression":
        """
        Robust Regression - outlier-resistant regression.
        Uses Huber or MM estimators.
        """
        import statsmodels.api as sm
        from statsmodels.robust.robust_linear_model import RLM

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "huber")  # huber, bisquare, andrews

        X = df[predictors].dropna()
        y = df[response].loc[X.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ROBUST REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method.title()} M-estimator\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        # Select M-estimator
        if method == "huber":
            M_estimator = sm.robust.norms.HuberT()
            summary += f"<<COLOR:dim>>Huber's T: downweights residuals > 1.345σ<</COLOR>>\n"
        elif method == "bisquare":
            M_estimator = sm.robust.norms.TukeyBiweight()
            summary += f"<<COLOR:dim>>Tukey's Bisquare: zero weight for residuals > 4.685σ<</COLOR>>\n"
        elif method == "andrews":
            M_estimator = sm.robust.norms.AndrewWave()
            summary += f"<<COLOR:dim>>Andrew's Wave: sinusoidal downweighting<</COLOR>>\n"
        else:
            M_estimator = sm.robust.norms.HuberT()

        X_const = sm.add_constant(X)

        # Fit OLS for comparison
        ols_model = sm.OLS(y, X_const).fit()

        # Fit robust model
        robust_model = RLM(y, X_const, M=M_estimator).fit()

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                        COMPARISON: OLS vs ROBUST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'OLS Coef':>12} {'Robust Coef':>12} {'Difference':>12}\n"
        summary += f"  {'-'*58}\n"

        var_names = ['const'] + predictors
        for i, name in enumerate(var_names):
            ols_coef = ols_model.params.iloc[i]
            rob_coef = robust_model.params.iloc[i]
            diff = rob_coef - ols_coef
            diff_pct = 100 * diff / abs(ols_coef) if abs(ols_coef) > 1e-10 else 0
            flag = "<<COLOR:warning>>*<</COLOR>>" if abs(diff_pct) > 10 else ""
            summary += f"  {name:<20} {ols_coef:>12.4f} {rob_coef:>12.4f} {diff:>+12.4f} {flag}\n"

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                           ROBUST MODEL DETAILS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'z':>10} {'P>|z|':>10}\n"
        summary += f"  {'-'*66}\n"

        for i, name in enumerate(var_names):
            summary += f"  {name:<20} {robust_model.params.iloc[i]:>12.4f} {robust_model.bse.iloc[i]:>12.4f} {robust_model.tvalues.iloc[i]:>10.3f} {robust_model.pvalues.iloc[i]:>10.4f}\n"

        # Identify outliers (low weights)
        weights = robust_model.weights
        outlier_threshold = 0.5
        outliers = np.where(weights < outlier_threshold)[0]

        summary += f"\n<<COLOR:text>>Observations with low weight (<{outlier_threshold}):<</COLOR>> {len(outliers)}\n"
        if len(outliers) > 0 and len(outliers) <= 10:
            summary += f"  Indices: {list(outliers)}\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        coef_diffs = np.abs(robust_model.params.values - ols_model.params.values)
        if np.max(coef_diffs[1:]) > 0.1 * np.max(np.abs(ols_model.params.values[1:])):
            summary += f"  <<COLOR:warning>>Coefficients differ substantially - outliers are influential.<</COLOR>>\n"
            summary += f"  Robust estimates may be more reliable.\n"
        else:
            summary += f"  <<COLOR:good>>Coefficients are similar - outliers have minimal influence.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Robust regression ({method}) identified {len(outliers)} low-weight observations."
        result["statistics"] = {
            "n_outliers": len(outliers),
            "method": method,
            **{f"coef_{name}": float(robust_model.params.iloc[i]) for i, name in enumerate(var_names)}
        }

        # Plot: OLS residuals vs Robust residuals
        ols_resid = ols_model.resid
        rob_resid = robust_model.resid

        result["plots"].append({
            "title": "Residual Comparison",
            "data": [
                {"type": "scatter", "x": ols_resid.tolist(), "y": rob_resid.tolist(), "mode": "markers",
                 "marker": {"color": weights.tolist(), "colorscale": [[0, "#e85747"], [1, "#4a9f6e"]], "size": 6, "colorbar": {"title": "Weight"}}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "OLS Residuals"}, "yaxis": {"title": "Robust Residuals"}}
        })

        # Weights histogram
        result["plots"].append({
            "title": "Observation Weights",
            "data": [{"type": "histogram", "x": weights.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": "Weight", "range": [0, 1.1]}}
        })

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
        summary += f"<<COLOR:title>>TOLERANCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Coverage:<</COLOR>> {proportion*100:.0f}% of population\n"
        summary += f"<<COLOR:highlight>>Confidence:<</COLOR>> {confidence*100:.0f}%\n\n"

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {mean:.4f}\n"
        summary += f"  Std Dev: {std:.4f}\n\n"

        # Normal-based tolerance interval
        # k factor from tolerance interval tables (approximation)
        z_p = stats.norm.ppf((1 + proportion) / 2)
        chi2_val = stats.chi2.ppf(1 - confidence, n - 1)

        # Two-sided tolerance factor
        k_normal = z_p * np.sqrt((n - 1) * (1 + 1/n) / chi2_val)

        tol_lower_normal = mean - k_normal * std
        tol_upper_normal = mean + k_normal * std

        summary += f"<<COLOR:accent>>Normal-Based Tolerance Interval:<</COLOR>>\n"
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
        for r in range(1, n//2 + 1):
            # Probability that at least proportion of population is between order statistics
            prob = 0
            for j in range(r, n - r + 2):
                prob += comb(n, j, exact=True) * (proportion**(j)) * ((1-proportion)**(n-j))
            if prob >= confidence:
                r_found = r
                break

        if r_found:
            sorted_data = np.sort(data)
            tol_lower_np = sorted_data[r_found - 1]
            tol_upper_np = sorted_data[n - r_found]
            summary += f"<<COLOR:accent>>Non-Parametric Tolerance Interval:<</COLOR>>\n"
            summary += f"  Uses order statistics X({r_found}) and X({n - r_found + 1})\n"
            summary += f"  Interval: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"
        else:
            tol_lower_np = np.min(data)
            tol_upper_np = np.max(data)
            summary += f"<<COLOR:warning>>Non-Parametric: Sample too small for exact interval.<</COLOR>>\n"
            summary += f"  Using min/max: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"

        # Comparison with confidence interval
        se = std / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se

        summary += f"<<COLOR:dim>>For comparison - {confidence*100:.0f}% CI for mean:<</COLOR>>\n"
        summary += f"  ({ci_lower:.4f}, {ci_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {confidence*100:.0f}% confident that at least {proportion*100:.0f}%\n"
        summary += f"  of the population falls within the tolerance interval.\n"
        summary += f"\n<<COLOR:dim>>Note: Tolerance intervals are WIDER than confidence intervals<</COLOR>>\n"
        summary += f"<<COLOR:dim>>because they cover the population, not just the mean.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Tolerance interval ({proportion*100:.0f}%/{confidence*100:.0f}%): ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})"
        result["statistics"] = {
            "tol_lower_normal": float(tol_lower_normal),
            "tol_upper_normal": float(tol_upper_normal),
            "tol_lower_np": float(tol_lower_np),
            "tol_upper_np": float(tol_upper_np),
            "k_factor": float(k_normal),
            "mean": float(mean),
            "std": float(std)
        }

        # Plot showing intervals
        result["plots"].append({
            "title": "Tolerance vs Confidence Intervals",
            "data": [
                {"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": var},
                "shapes": [
                    # Tolerance interval (normal)
                    {"type": "rect", "x0": tol_lower_normal, "x1": tol_upper_normal, "y0": 0, "y1": 1, "yref": "paper",
                     "fillcolor": "rgba(232, 149, 71, 0.2)", "line": {"color": "#e89547", "width": 2}},
                    # Confidence interval
                    {"type": "rect", "x0": ci_lower, "x1": ci_upper, "y0": 0.4, "y1": 0.6, "yref": "paper",
                     "fillcolor": "rgba(71, 165, 232, 0.4)", "line": {"color": "#47a5e8", "width": 2}},
                    # Mean line
                    {"type": "line", "x0": mean, "x1": mean, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#4a9f6e", "width": 2, "dash": "dash"}}
                ],
                "annotations": [
                    {"x": (tol_lower_normal + tol_upper_normal)/2, "y": 0.95, "yref": "paper", "text": "Tolerance", "showarrow": False, "font": {"color": "#e89547"}},
                    {"x": (ci_lower + ci_upper)/2, "y": 0.5, "yref": "paper", "text": "CI", "showarrow": False, "font": {"color": "#47a5e8"}}
                ]
            }
        })

    elif analysis_id == "tukey_hsd":
        """
        Tukey's Honestly Significant Difference — pairwise comparison after ANOVA.
        Controls family-wise error rate for all pairwise comparisons.
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        data = df[[response, factor]].dropna()
        tukey = pairwise_tukeyhsd(data[response], data[factor], alpha=alpha)

        # Parse results into structured data
        pairs = []
        for i in range(len(tukey.summary().data) - 1):
            row = tukey.summary().data[i + 1]
            pairs.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": float(row[2]),
                "p_adj": float(row[3]),
                "lower": float(row[4]),
                "upper": float(row[5]),
                "reject": bool(row[6])
            })

        n_sig = sum(1 for p in pairs if p["reject"])
        n_total = len(pairs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TUKEY'S HSD POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Pairwise Comparisons:<</COLOR>>\n"
        summary += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'p-adj':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['meandiff']:>8.4f} {p['p_adj']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"

        summary += f"\n<<COLOR:text>>Summary: {n_sig}/{n_total} pairs significantly different<</COLOR>>\n"
        if n_sig > 0:
            summary += f"<<COLOR:good>>Significant pairs (p < {alpha}):<</COLOR>>\n"
            for p in pairs:
                if p["reject"]:
                    summary += f"  • {p['group1']} vs {p['group2']}: diff = {p['meandiff']:.4f}\n"

        result["summary"] = summary

        # CI plot for pairwise differences
        traces = []
        y_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        traces.append({
            "type": "scatter",
            "x": [p["meandiff"] for p in pairs],
            "y": y_labels,
            "mode": "markers",
            "marker": {"color": colors, "size": 10},
            "error_x": {
                "type": "data",
                "symmetric": False,
                "array": [p["upper"] - p["meandiff"] for p in pairs],
                "arrayminus": [p["meandiff"] - p["lower"] for p in pairs],
                "color": "#5a6a5a"
            },
            "showlegend": False
        })
        result["plots"].append({
            "title": "Tukey HSD — Pairwise Differences with CIs",
            "data": traces,
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(pairs) - 0.5, "line": {"color": "#e89547", "dash": "dash"}}]
            }
        })

        # Group means with SE error bars
        group_stats = data.groupby(factor)[response].agg(['mean', 'std', 'count'])
        group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
        result["plots"].append({
            "title": f"Group Means — {response} by {factor}",
            "data": [{
                "x": [str(g) for g in group_stats.index],
                "y": group_stats['mean'].tolist(),
                "error_y": {"type": "data", "array": group_stats['se'].tolist(), "visible": True, "color": "#5a6a5a"},
                "type": "bar",
                "marker": {"color": "#4a9f6e", "opacity": 0.8}
            }],
            "layout": {
                "height": 280,
                "xaxis": {"title": factor},
                "yaxis": {"title": f"Mean {response}"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Tukey HSD: {n_sig}/{n_total} pairwise comparisons significant at α={alpha}."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig, "n_comparisons": n_total, "alpha": alpha}

    elif analysis_id == "dunnett":
        """
        Dunnett's Test — compare each treatment group to a control group.
        More powerful than Tukey when only control comparisons matter.
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        control = config.get("control")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[response, factor]].dropna()
        levels = data[factor].unique().tolist()

        if control is None or control not in levels:
            control = levels[0]

        control_data = data[data[factor] == control][response].values
        treatments = [lev for lev in levels if lev != control]

        comparisons = []
        for treat in treatments:
            treat_data = data[data[factor] == treat][response].values
            # Dunnett uses pooled variance and critical values; scipy has it from 1.11+
            try:
                from scipy.stats import dunnett as scipy_dunnett
                res = scipy_dunnett(treat_data, control=control_data)
                p_val = float(res.pvalue[0]) if hasattr(res.pvalue, '__len__') else float(res.pvalue)
                stat_val = float(res.statistic[0]) if hasattr(res.statistic, '__len__') else float(res.statistic)
            except (ImportError, AttributeError):
                # Fallback: Welch t-test with Bonferroni correction
                stat_val, p_raw = stats.ttest_ind(treat_data, control_data, equal_var=False)
                stat_val = float(stat_val)
                p_val = min(float(p_raw) * len(treatments), 1.0)

            mean_diff = float(np.mean(treat_data) - np.mean(control_data))
            comparisons.append({
                "treatment": str(treat),
                "control": str(control),
                "mean_diff": mean_diff,
                "statistic": stat_val,
                "p_value": p_val,
                "reject": p_val < alpha
            })

        n_sig = sum(1 for c in comparisons if c["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>DUNNETT'S TEST (vs Control)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Control group:<</COLOR>> {control}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Control: {control} (n={len(control_data)}, mean={np.mean(control_data):.4f})<</COLOR>>\n\n"
        summary += f"{'Treatment':<20} {'Diff':>8} {'Statistic':>10} {'p-value':>8} {'Sig':>5}\n"
        summary += f"{'─' * 55}\n"
        for c in comparisons:
            sig = "<<COLOR:good>>*<</COLOR>>" if c["reject"] else ""
            summary += f"{c['treatment']:<20} {c['mean_diff']:>8.4f} {c['statistic']:>10.4f} {c['p_value']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(comparisons)} treatments differ from control<</COLOR>>\n"

        result["summary"] = summary

        # Bar chart of differences with SE error bars
        se_bars = []
        for c in comparisons:
            treat_vals = data[data[factor] == c["treatment"]][response].values
            se = float(np.sqrt(np.var(treat_vals, ddof=1)/len(treat_vals) + np.var(control_data, ddof=1)/len(control_data)))
            se_bars.append(se)
        result["plots"].append({
            "title": f"Dunnett's Test — Difference from Control ({control})",
            "data": [{
                "type": "bar",
                "x": [c["treatment"] for c in comparisons],
                "y": [c["mean_diff"] for c in comparisons],
                "marker": {"color": ["#4a9f6e" if c["reject"] else "rgba(90,106,90,0.5)" for c in comparisons],
                           "line": {"color": "#4a9f6e", "width": 1}},
                "error_y": {"type": "data", "array": se_bars, "visible": True, "color": "rgba(200,200,200,0.7)"},
                "text": [f"p={c['p_value']:.4f}" for c in comparisons],
                "textposition": "outside"
            }],
            "layout": {"height": 300, "yaxis": {"title": f"Difference from {control}"}}
        })

        result["guide_observation"] = f"Dunnett's test vs {control}: {n_sig}/{len(comparisons)} treatments differ."
        result["statistics"] = {"comparisons": comparisons, "control": str(control)}

    elif analysis_id == "games_howell":
        """
        Games-Howell Test — post-hoc for unequal variances and/or unequal group sizes.
        Does not assume equal variances (unlike Tukey).
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[response, factor]].dropna()
        levels = sorted(data[factor].unique().tolist(), key=str)

        group_stats = {}
        for lev in levels:
            g = data[data[factor] == lev][response].values
            group_stats[lev] = {"mean": np.mean(g), "var": np.var(g, ddof=1), "n": len(g)}

        # Pairwise Games-Howell: Welch t with Studentized Range adjustment
        from itertools import combinations
        from scipy.stats import studentized_range

        pairs = []
        level_pairs = list(combinations(levels, 2))
        for g1, g2 in level_pairs:
            s1 = group_stats[g1]
            s2 = group_stats[g2]
            mean_diff = s1["mean"] - s2["mean"]
            se = np.sqrt(s1["var"] / s1["n"] + s2["var"] / s2["n"])

            # Welch-Satterthwaite degrees of freedom
            num = (s1["var"] / s1["n"] + s2["var"] / s2["n"]) ** 2
            denom = (s1["var"] / s1["n"]) ** 2 / (s1["n"] - 1) + (s2["var"] / s2["n"]) ** 2 / (s2["n"] - 1)
            df_welch = num / denom if denom > 0 else 1

            q_stat = abs(mean_diff) / se if se > 0 else 0
            k = len(levels)

            # p-value from studentized range distribution
            try:
                p_val = float(studentized_range.sf(q_stat * np.sqrt(2), k, df_welch))
            except Exception:
                # Fallback to Welch t-test with Bonferroni
                t_stat = mean_diff / se if se > 0 else 0
                p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))
                p_val = min(float(p_raw) * len(level_pairs), 1.0)

            pairs.append({
                "group1": str(g1),
                "group2": str(g2),
                "meandiff": float(mean_diff),
                "se": float(se),
                "q": float(q_stat),
                "df": float(df_welch),
                "p_value": float(p_val),
                "reject": p_val < alpha
            })

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAMES-HOWELL POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n"
        summary += f"<<COLOR:text>>(Does not assume equal variances)<</COLOR>>\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        for lev in levels:
            s = group_stats[lev]
            summary += f"  {lev}: n={s['n']}, mean={s['mean']:.4f}, std={np.sqrt(s['var']):.4f}\n"

        summary += f"\n{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'SE':>8} {'q':>8} {'p-val':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['meandiff']:>8.4f} {p['se']:>8.4f} {p['q']:>8.4f} {p['p_value']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(pairs)} pairs significantly different<</COLOR>>\n"

        result["summary"] = summary

        # CI-style plot with error bars (like Tukey)
        y_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        # Approximate CI half-width from SE and df
        ci_half = []
        for p in pairs:
            try:
                t_crit = stats.t.ppf(1 - alpha / 2, p["df"])
                ci_half.append(float(t_crit * p["se"]))
            except Exception:
                ci_half.append(0)
        result["plots"].append({
            "title": "Games-Howell — Pairwise Differences with CI",
            "data": [{
                "type": "scatter",
                "x": [p["meandiff"] for p in pairs],
                "y": y_labels,
                "mode": "markers",
                "marker": {"color": colors, "size": 10},
                "error_x": {"type": "data", "array": ci_half, "color": "rgba(74,159,110,0.6)", "thickness": 2},
                "showlegend": False
            }],
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True}
            }
        })

        result["guide_observation"] = f"Games-Howell: {n_sig}/{len(pairs)} pairs significant (unequal variances assumed)."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig}

    elif analysis_id == "dunn":
        """
        Dunn's Test — non-parametric post-hoc after Kruskal-Wallis.
        Uses rank sums with Bonferroni correction.
        """
        var = config.get("var") or config.get("response")
        group_var = config.get("group_var") or config.get("factor")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[var, group_var]].dropna()
        levels = sorted(data[group_var].unique().tolist(), key=str)

        # Assign overall ranks
        data = data.copy()
        data["_rank"] = stats.rankdata(data[var])
        n_total = len(data)

        group_info = {}
        for lev in levels:
            g = data[data[group_var] == lev]
            group_info[lev] = {"mean_rank": g["_rank"].mean(), "n": len(g), "median": g[var].median()}

        # Pairwise Dunn's test
        from itertools import combinations
        level_pairs = list(combinations(levels, 2))
        n_comparisons = len(level_pairs)

        # Tied-rank correction factor
        ranks = data["_rank"].values
        unique_ranks, counts = np.unique(ranks, return_counts=True)
        tie_correction = 1 - np.sum(counts ** 3 - counts) / (n_total ** 3 - n_total) if n_total > 1 else 1

        pairs = []
        for g1, g2 in level_pairs:
            s1 = group_info[g1]
            s2 = group_info[g2]
            diff = s1["mean_rank"] - s2["mean_rank"]

            # Standard error with tie correction
            se = np.sqrt(tie_correction * (n_total * (n_total + 1) / 12) * (1.0 / s1["n"] + 1.0 / s2["n"]))
            z = diff / se if se > 0 else 0
            p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni

            pairs.append({
                "group1": str(g1),
                "group2": str(g2),
                "rank_diff": float(diff),
                "z": float(z),
                "p_raw": float(p_raw),
                "p_adj": float(p_adj),
                "reject": p_adj < alpha
            })

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>DUNN'S TEST (Post-Hoc for Kruskal-Wallis)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(levels)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Correction:<</COLOR>> Bonferroni\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Group Rank Statistics:<</COLOR>>\n"
        for lev in levels:
            s = group_info[lev]
            summary += f"  {lev}: n={s['n']}, median={s['median']:.4f}, mean rank={s['mean_rank']:.1f}\n"

        summary += f"\n{'Group 1':<15} {'Group 2':<15} {'Rank Diff':>10} {'Z':>8} {'p-raw':>8} {'p-adj':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['rank_diff']:>10.2f} {p['z']:>8.4f} {p['p_raw']:>8.4f} {p['p_adj']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(pairs)} pairs significantly different (Bonferroni-adjusted)<</COLOR>>\n"

        result["summary"] = summary

        # Mean rank comparison plot
        result["plots"].append({
            "title": f"Dunn's Test — Mean Ranks by Group",
            "data": [{
                "type": "bar",
                "x": [str(lev) for lev in levels],
                "y": [group_info[lev]["mean_rank"] for lev in levels],
                "marker": {"color": "#4a9f6e", "line": {"color": "#3d8a5c", "width": 1}},
                "text": [f"n={group_info[lev]['n']}" for lev in levels],
                "textposition": "outside"
            }],
            "layout": {"height": 300, "yaxis": {"title": "Mean Rank"}, "xaxis": {"title": group_var}}
        })

        # Pairwise rank differences plot
        pair_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        pair_colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        result["plots"].append({
            "title": "Dunn's Test — Pairwise Rank Differences",
            "data": [{
                "type": "scatter",
                "x": [p["rank_diff"] for p in pairs],
                "y": pair_labels,
                "mode": "markers",
                "marker": {"color": pair_colors, "size": 10},
                "text": [f"z={p['z']:.2f}, p={p['p_adj']:.4f}" for p in pairs],
                "hoverinfo": "text+x",
                "showlegend": False
            }],
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Rank Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True}
            }
        })

        result["guide_observation"] = f"Dunn's test: {n_sig}/{len(pairs)} pairwise comparisons significant (Bonferroni)."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig, "n_comparisons": n_comparisons}

    # =====================================================================
    # Scheffé Test
    # =====================================================================
    elif analysis_id == "scheffe_test":
        """
        Scheffé's test — most conservative post-hoc for all possible contrasts
        (not just pairwise). Controls family-wise error for any linear combination.
        """
        from scipy import stats as sch_stats

        response_sch = config.get("response") or config.get("var")
        factor_sch = config.get("factor") or config.get("group_var")
        alpha_sch = 1 - config.get("conf", 95) / 100

        data_sch = df[[response_sch, factor_sch]].dropna()
        groups_sch = sorted(data_sch[factor_sch].unique(), key=str)
        k_sch = len(groups_sch)

        group_data_sch = {str(g): data_sch[data_sch[factor_sch] == g][response_sch].values for g in groups_sch}
        group_means_sch = {g: np.mean(v) for g, v in group_data_sch.items()}
        group_ns_sch = {g: len(v) for g, v in group_data_sch.items()}
        n_total_sch = sum(group_ns_sch.values())

        # Pooled MSE
        ss_within = sum(np.sum((v - np.mean(v)) ** 2) for v in group_data_sch.values())
        df_within = n_total_sch - k_sch
        mse_sch = ss_within / df_within if df_within > 0 else 0

        # Scheffé critical value: (k-1) * F(alpha, k-1, N-k)
        f_crit_sch = sch_stats.f.ppf(1 - alpha_sch, k_sch - 1, df_within)
        scheffe_crit = (k_sch - 1) * f_crit_sch

        pairs_sch = []
        g_names = list(group_data_sch.keys())
        for i in range(len(g_names)):
            for j in range(i + 1, len(g_names)):
                g1, g2 = g_names[i], g_names[j]
                diff = group_means_sch[g1] - group_means_sch[g2]
                se = np.sqrt(mse_sch * (1 / group_ns_sch[g1] + 1 / group_ns_sch[g2]))
                f_val = (diff ** 2) / (se ** 2 * (k_sch - 1)) if se > 0 else 0
                p_val = 1 - sch_stats.f.cdf(f_val, k_sch - 1, df_within)
                margin = np.sqrt(scheffe_crit) * se
                pairs_sch.append({
                    "group1": g1, "group2": g2, "diff": float(diff),
                    "se": float(se), "f": float(f_val), "p": float(p_val),
                    "lower": float(diff - margin), "upper": float(diff + margin),
                    "reject": p_val < alpha_sch,
                })

        n_sig_sch = sum(1 for p in pairs_sch if p["reject"])
        summary_sch = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_sch += f"<<COLOR:title>>SCHEFFÉ'S POST-HOC TEST<</COLOR>>\n"
        summary_sch += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_sch += f"<<COLOR:highlight>>Response:<</COLOR>> {response_sch}\n"
        summary_sch += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_sch} ({k_sch} groups)\n"
        summary_sch += f"<<COLOR:highlight>>MSE:<</COLOR>> {mse_sch:.4f}  (df = {df_within})\n"
        summary_sch += f"<<COLOR:highlight>>Scheffé critical value:<</COLOR>> {scheffe_crit:.4f}\n\n"

        summary_sch += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'SE':>8} {'F':>8} {'p':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary_sch += f"{'─' * 85}\n"
        for p in pairs_sch:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary_sch += f"{p['group1']:<15} {p['group2']:<15} {p['diff']:>8.4f} {p['se']:>8.4f} {p['f']:>8.3f} {p['p']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"
        summary_sch += f"\n<<COLOR:text>>Summary: {n_sig_sch}/{len(pairs_sch)} pairs significantly different<</COLOR>>\n"
        summary_sch += f"<<COLOR:text>>Note: Scheffé is the most conservative — controls for ALL possible contrasts, not just pairwise.<</COLOR>>\n"

        result["summary"] = summary_sch

        # CI plot
        y_labels_sch = [f"{p['group1']} - {p['group2']}" for p in pairs_sch]
        result["plots"].append({
            "title": "Scheffé — Pairwise Differences with CIs",
            "data": [{
                "type": "scatter", "x": [p["diff"] for p in pairs_sch], "y": y_labels_sch,
                "mode": "markers",
                "marker": {"color": ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs_sch], "size": 10},
                "error_x": {"type": "data", "symmetric": False,
                            "array": [p["upper"] - p["diff"] for p in pairs_sch],
                            "arrayminus": [p["diff"] - p["lower"] for p in pairs_sch], "color": "#5a6a5a"},
                "showlegend": False,
            }],
            "layout": {"height": max(250, 40 * len(pairs_sch)),
                       "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547"},
                       "yaxis": {"automargin": True},
                       "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(pairs_sch) - 0.5,
                                   "line": {"color": "#e89547", "dash": "dash"}}]}
        })

        result["guide_observation"] = f"Scheffé: {n_sig_sch}/{len(pairs_sch)} pairs significant at α={alpha_sch}."
        result["statistics"] = {"pairs": pairs_sch, "mse": mse_sch, "scheffe_critical": scheffe_crit, "k": k_sch}

    # =====================================================================
    # Bonferroni Post-Hoc Test
    # =====================================================================
    elif analysis_id == "bonferroni_test":
        """
        Bonferroni post-hoc — pairwise t-tests with Bonferroni correction.
        Simple, widely-used, slightly conservative.
        """
        from scipy import stats as bon_stats

        response_bon = config.get("response") or config.get("var")
        factor_bon = config.get("factor") or config.get("group_var")
        alpha_bon = 1 - config.get("conf", 95) / 100

        data_bon = df[[response_bon, factor_bon]].dropna()
        groups_bon = sorted(data_bon[factor_bon].unique(), key=str)
        k_bon = len(groups_bon)

        group_data_bon = {str(g): data_bon[data_bon[factor_bon] == g][response_bon].values for g in groups_bon}
        n_total_bon = sum(len(v) for v in group_data_bon.values())

        # Pooled MSE
        ss_w_bon = sum(np.sum((v - np.mean(v)) ** 2) for v in group_data_bon.values())
        df_w_bon = n_total_bon - k_bon
        mse_bon = ss_w_bon / df_w_bon if df_w_bon > 0 else 0

        n_comparisons_bon = k_bon * (k_bon - 1) // 2
        alpha_adj = alpha_bon / n_comparisons_bon

        pairs_bon = []
        g_names_bon = list(group_data_bon.keys())
        for i in range(len(g_names_bon)):
            for j in range(i + 1, len(g_names_bon)):
                g1, g2 = g_names_bon[i], g_names_bon[j]
                n1_b, n2_b = len(group_data_bon[g1]), len(group_data_bon[g2])
                m1, m2 = np.mean(group_data_bon[g1]), np.mean(group_data_bon[g2])
                diff = m1 - m2
                se = np.sqrt(mse_bon * (1 / n1_b + 1 / n2_b))
                t_val = diff / se if se > 0 else 0
                p_raw = 2 * (1 - bon_stats.t.cdf(abs(t_val), df_w_bon))
                p_adj = min(1.0, p_raw * n_comparisons_bon)
                t_crit = bon_stats.t.ppf(1 - alpha_adj / 2, df_w_bon)
                margin = t_crit * se
                pairs_bon.append({
                    "group1": g1, "group2": g2, "diff": float(diff),
                    "se": float(se), "t": float(t_val), "p_raw": float(p_raw),
                    "p_adj": float(p_adj), "lower": float(diff - margin),
                    "upper": float(diff + margin), "reject": p_adj < alpha_bon,
                })

        n_sig_bon = sum(1 for p in pairs_bon if p["reject"])
        summary_bon = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_bon += f"<<COLOR:title>>BONFERRONI POST-HOC TEST<</COLOR>>\n"
        summary_bon += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_bon += f"<<COLOR:highlight>>Response:<</COLOR>> {response_bon}\n"
        summary_bon += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_bon} ({k_bon} groups)\n"
        summary_bon += f"<<COLOR:highlight>>Adjusted α:<</COLOR>> {alpha_bon}/{n_comparisons_bon} = {alpha_adj:.6f}\n\n"

        summary_bon += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'t':>8} {'p-adj':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary_bon += f"{'─' * 82}\n"
        for p in pairs_bon:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary_bon += f"{p['group1']:<15} {p['group2']:<15} {p['diff']:>8.4f} {p['t']:>8.3f} {p['p_adj']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"
        summary_bon += f"\n<<COLOR:text>>Summary: {n_sig_bon}/{len(pairs_bon)} pairs significantly different (Bonferroni)<</COLOR>>\n"

        result["summary"] = summary_bon

        y_labels_bon = [f"{p['group1']} - {p['group2']}" for p in pairs_bon]
        result["plots"].append({
            "title": "Bonferroni — Pairwise Differences with CIs",
            "data": [{
                "type": "scatter", "x": [p["diff"] for p in pairs_bon], "y": y_labels_bon,
                "mode": "markers",
                "marker": {"color": ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs_bon], "size": 10},
                "error_x": {"type": "data", "symmetric": False,
                            "array": [p["upper"] - p["diff"] for p in pairs_bon],
                            "arrayminus": [p["diff"] - p["lower"] for p in pairs_bon], "color": "#5a6a5a"},
                "showlegend": False,
            }],
            "layout": {"height": max(250, 40 * len(pairs_bon)),
                       "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547"},
                       "yaxis": {"automargin": True},
                       "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(pairs_bon) - 0.5,
                                   "line": {"color": "#e89547", "dash": "dash"}}]}
        })

        result["guide_observation"] = f"Bonferroni: {n_sig_bon}/{len(pairs_bon)} pairs significant (adj. α={alpha_adj:.4f})."
        result["statistics"] = {"pairs": pairs_bon, "mse": mse_bon, "alpha_adjusted": alpha_adj}

    # =====================================================================
    # Hsu's MCB (Multiple Comparisons with the Best)
    # =====================================================================
    elif analysis_id == "hsu_mcb":
        """
        Hsu's MCB — determines which groups are statistically distinguishable
        from the best (highest or lowest mean). Returns confidence intervals
        for μ_i - max(μ_j, j≠i). If CI includes 0, group i could be the best.
        """
        from scipy import stats as hsu_stats

        response_hsu = config.get("response") or config.get("var")
        factor_hsu = config.get("factor") or config.get("group_var")
        alpha_hsu = 1 - config.get("conf", 95) / 100
        direction = config.get("direction", "max")  # "max" = higher is better, "min" = lower is better

        data_hsu = df[[response_hsu, factor_hsu]].dropna()
        groups_hsu = sorted(data_hsu[factor_hsu].unique(), key=str)
        k_hsu = len(groups_hsu)

        group_data_hsu = {str(g): data_hsu[data_hsu[factor_hsu] == g][response_hsu].values for g in groups_hsu}
        group_means_hsu = {g: np.mean(v) for g, v in group_data_hsu.items()}
        group_ns_hsu = {g: len(v) for g, v in group_data_hsu.items()}
        n_total_hsu = sum(group_ns_hsu.values())

        ss_w_hsu = sum(np.sum((v - np.mean(v)) ** 2) for v in group_data_hsu.values())
        df_w_hsu = n_total_hsu - k_hsu
        mse_hsu = ss_w_hsu / df_w_hsu if df_w_hsu > 0 else 0

        # For MCB, use Dunnett-like critical value (k-1 comparisons to best)
        # Approximate with Bonferroni-adjusted t
        t_crit_hsu = hsu_stats.t.ppf(1 - alpha_hsu / (2 * (k_hsu - 1)), df_w_hsu)

        if direction == "max":
            best_group = max(group_means_hsu, key=group_means_hsu.get)
        else:
            best_group = min(group_means_hsu, key=group_means_hsu.get)

        comparisons_hsu = []
        for g in group_data_hsu:
            mean_g = group_means_hsu[g]
            mean_best = group_means_hsu[best_group]

            if direction == "max":
                diff = mean_g - mean_best  # ≤ 0 for non-best groups
            else:
                diff = mean_best - mean_g  # ≤ 0 for non-best groups

            se = np.sqrt(mse_hsu * (1 / group_ns_hsu[g] + 1 / group_ns_hsu[best_group]))
            lower = diff - t_crit_hsu * se
            upper = diff + t_crit_hsu * se

            # MCB: constrain interval
            if g == best_group:
                could_be_best = True
                lower_mcb = 0.0
                upper_mcb = max(0, upper)
            else:
                lower_mcb = min(0, lower)
                upper_mcb = min(0, upper)
                could_be_best = upper_mcb >= 0  # CI includes 0 means could be best

            comparisons_hsu.append({
                "group": g, "mean": float(mean_g), "diff_from_best": float(diff),
                "lower": float(lower_mcb), "upper": float(upper_mcb),
                "could_be_best": could_be_best,
                "is_best": g == best_group,
            })

        summary_hsu = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_hsu += f"<<COLOR:title>>HSU'S MCB (MULTIPLE COMPARISONS WITH THE BEST)<</COLOR>>\n"
        summary_hsu += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_hsu += f"<<COLOR:highlight>>Response:<</COLOR>> {response_hsu}\n"
        summary_hsu += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_hsu} ({k_hsu} groups)\n"
        summary_hsu += f"<<COLOR:highlight>>Direction:<</COLOR>> {'Higher is better' if direction == 'max' else 'Lower is better'}\n"
        summary_hsu += f"<<COLOR:highlight>>Best group:<</COLOR>> {best_group} (mean = {group_means_hsu[best_group]:.4f})\n\n"

        summary_hsu += f"{'Group':<20} {'Mean':>8} {'Diff':>8} {'Lower':>8} {'Upper':>8} {'Could be best?':>15}\n"
        summary_hsu += f"{'─' * 72}\n"
        for c in comparisons_hsu:
            best_mark = "<<COLOR:good>>YES<</COLOR>>" if c["could_be_best"] else "<<COLOR:warning>>NO<</COLOR>>"
            star = " ◄" if c["is_best"] else ""
            summary_hsu += f"{c['group']:<20} {c['mean']:>8.4f} {c['diff_from_best']:>8.4f} {c['lower']:>8.4f} {c['upper']:>8.4f} {best_mark:>15}{star}\n"

        n_could_be_best = sum(1 for c in comparisons_hsu if c["could_be_best"])
        summary_hsu += f"\n<<COLOR:text>>{n_could_be_best}/{k_hsu} groups could be the best at {(1-alpha_hsu)*100:.0f}% confidence.<</COLOR>>\n"

        result["summary"] = summary_hsu

        # MCB interval plot
        result["plots"].append({
            "title": f"Hsu's MCB — Differences from Best ({direction})",
            "data": [{
                "type": "scatter", "mode": "markers",
                "x": [c["diff_from_best"] for c in comparisons_hsu],
                "y": [c["group"] for c in comparisons_hsu],
                "marker": {"color": ["#4a9f6e" if c["could_be_best"] else "#d94a4a" for c in comparisons_hsu], "size": 10},
                "error_x": {"type": "data", "symmetric": False,
                            "array": [c["upper"] - c["diff_from_best"] for c in comparisons_hsu],
                            "arrayminus": [c["diff_from_best"] - c["lower"] for c in comparisons_hsu],
                            "color": "#5a6a5a"},
                "showlegend": False,
            }],
            "layout": {"height": max(250, 40 * k_hsu),
                       "xaxis": {"title": f"Difference from best ({best_group})"},
                       "yaxis": {"automargin": True},
                       "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": k_hsu - 0.5,
                                   "line": {"color": "#e89547", "dash": "dash"}}]}
        })

        result["guide_observation"] = f"Hsu's MCB: {n_could_be_best}/{k_hsu} groups could be best. Best={best_group}."
        result["statistics"] = {"comparisons": comparisons_hsu, "best_group": best_group, "direction": direction, "mse": mse_hsu}

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
            result["summary"] = f"Hotelling's T² requires exactly 2 groups. Found {len(groups)}: {groups}"
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
        summary += f"<<COLOR:title>>HOTELLING'S T² TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Group variable:<</COLOR>> {group_var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        summary += f"{'Variable':<20} {str(groups[0]):>12} {str(groups[1]):>12} {'Difference':>12}\n"
        summary += f"{'─' * 58}\n"
        for i, var in enumerate(responses):
            summary += f"{var:<20} {mean1[i]:>12.4f} {mean2[i]:>12.4f} {diff[i]:>12.4f}\n"

        summary += f"\n<<COLOR:text>>Test Statistics:<</COLOR>>\n"
        summary += f"  Hotelling's T²: {T2:.4f}\n"
        summary += f"  F-statistic: {F_stat:.4f} (df1={df1}, df2={df2})\n"
        summary += f"  p-value: {p_value:.4f}\n\n"

        if p_value < alpha:
            summary += f"<<COLOR:good>>Mean vectors differ significantly (p < {alpha})<</COLOR>>\n"
            summary += f"<<COLOR:text>>The groups have different multivariate profiles across the response variables.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in mean vectors (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Radar/profile plot of group means
        traces = []
        for idx, grp in enumerate(groups):
            grp_means = [float(mean1[i]) if idx == 0 else float(mean2[i]) for i in range(p)]
            colors = ["#4a9f6e", "#47a5e8"]
            traces.append({
                "type": "scatterpolar",
                "r": grp_means + [grp_means[0]],
                "theta": responses + [responses[0]],
                "name": str(grp),
                "fill": "toself",
                "fillcolor": f"rgba({','.join(str(int(c, 16)) for c in [colors[idx][1:3], colors[idx][3:5], colors[idx][5:7]])}, 0.15)",
                "line": {"color": colors[idx]}
            })
        result["plots"].append({
            "title": "Multivariate Profile — Group Means",
            "data": traces,
            "layout": {"height": 350, "polar": {"radialaxis": {"visible": True}}}
        })

        # Per-variable box plots by group
        box_traces = []
        grp_colors = ["#4a9f6e", "#47a5e8"]
        for gi, grp in enumerate(groups):
            grp_data = data[data[group_var] == grp]
            for vi, var in enumerate(responses):
                box_traces.append({
                    "type": "box", "y": grp_data[var].tolist(),
                    "x": [var] * len(grp_data), "name": str(grp),
                    "marker": {"color": grp_colors[gi]},
                    "legendgroup": str(grp), "showlegend": vi == 0
                })
        result["plots"].append({
            "title": "Response Distributions by Group",
            "data": box_traces,
            "layout": {
                "height": 300, "boxmode": "group",
                "xaxis": {"title": "Response Variable"},
                "yaxis": {"title": "Value"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Hotelling's T² = {T2:.2f}, F = {F_stat:.2f}, p = {p_value:.4f}. " + ("Groups differ." if p_value < alpha else "No difference.")
        result["statistics"] = {"T2": T2, "F_statistic": F_stat, "p_value": p_value, "df1": df1, "df2": df2, "mean_diff": diff.tolist()}

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
            result["summary"] = "MANOVA: Error matrix is singular. Check for collinear responses or insufficient data."
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
            F_wilks = ((1 - wilks_sqrt) / wilks_sqrt) * (r / df_h) if wilks_sqrt > 0 else 0
            df1_w, df2_w = df_h, 2 * (r - 1) if r > 1 else 1
        else:
            # General case: Chi-square approximation
            t = np.sqrt((p**2 * (k-1)**2 - 4) / (p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5) > 0 else 1
            df1_w = p * (k - 1)
            ms = N - 1 - (p + k) / 2
            df2_w = ms * t - df1_w / 2 + 1
            wilks_t = wilks ** (1/t) if wilks > 0 and t > 0 else 0
            F_wilks = ((1 - wilks_t) / wilks_t) * (df2_w / df1_w) if wilks_t > 0 else 0

        p_wilks = float(1 - stats.f.cdf(max(F_wilks, 0), max(df1_w, 1), max(df2_w, 1)))

        # 2. Pillai's Trace
        pillai = float(np.sum(eigenvalues / (1 + eigenvalues)))
        df1_p = s * max(p, k - 1)
        df2_p = s * (N - k - p + s)
        F_pillai = (pillai / s) * (df2_p / (max(p, k-1))) / ((1 - pillai / s)) if (1 - pillai / s) > 0 else 0
        p_pillai = float(1 - stats.f.cdf(max(F_pillai, 0), max(df1_p, 1), max(df2_p, 1)))

        # 3. Hotelling-Lawley Trace
        hl_trace = float(np.sum(eigenvalues))
        df1_hl = s * max(p, k - 1)
        df2_hl = s * (N - k - p - 1) + 2
        F_hl = (hl_trace / s) * (df2_hl / max(p, k-1)) if max(p, k-1) > 0 else 0
        p_hl = float(1 - stats.f.cdf(max(F_hl, 0), max(df1_hl, 1), max(df2_hl, 1)))

        # 4. Roy's Largest Root
        roy = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
        df1_r = max(p, k - 1)
        df2_r = N - k - max(p, k-1) + 1
        F_roy = roy * df2_r / df1_r if df1_r > 0 else 0
        p_roy = float(1 - stats.f.cdf(max(F_roy, 0), max(df1_r, 1), max(df2_r, 1)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-WAY MANOVA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        header = f"{'Variable':<20}" + "".join(f"{str(g):>12}" for g in groups)
        summary += header + "\n" + "─" * len(header) + "\n"
        for i, var in enumerate(responses):
            row = f"{var:<20}" + "".join(f"{group_means[g]['mean'][i]:>12.4f}" for g in groups)
            summary += row + "\n"
        summary += "\n"

        summary += f"<<COLOR:text>>MANOVA Test Statistics:<</COLOR>>\n"
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
            summary += f"{name:<25} {val:>10.4f} {f_val:>10.4f} {p_val:>10.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>Eigenvalues of H·E⁻¹:<</COLOR>> {', '.join(f'{e:.4f}' for e in eigenvalues)}\n\n"

        # Overall interpretation (use Pillai's — most robust)
        if p_pillai < alpha:
            summary += f"<<COLOR:good>>Significant multivariate effect (Pillai's Trace, p < {alpha})<</COLOR>>\n"
            summary += f"<<COLOR:text>>Group means differ across the response variables considered jointly.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant multivariate effect (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Group centroid plot (first 2 responses, or first 2 discriminant functions)
        if p >= 2:
            traces = []
            colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84393", "#00b894", "#fdcb6e"]
            for i, grp in enumerate(groups):
                grp_data = data[data[factor] == grp]
                traces.append({
                    "type": "scatter",
                    "x": grp_data[responses[0]].tolist(),
                    "y": grp_data[responses[1]].tolist(),
                    "mode": "markers",
                    "name": str(grp),
                    "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6}
                })
                # Centroid
                traces.append({
                    "type": "scatter",
                    "x": [float(group_means[grp]["mean"][0])],
                    "y": [float(group_means[grp]["mean"][1])],
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 14, "symbol": "diamond", "line": {"color": "white", "width": 2}},
                    "showlegend": False
                })
            result["plots"].append({
                "title": f"Group Centroids — {responses[0]} vs {responses[1]}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"title": responses[0]}, "yaxis": {"title": responses[1]}}
            })

        # Per-response box plots by group
        box_traces_m = []
        m_colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84393", "#00b894", "#fdcb6e"]
        for gi, grp in enumerate(groups):
            grp_d = data[data[factor] == grp]
            for vi, var in enumerate(responses):
                box_traces_m.append({
                    "type": "box", "y": grp_d[var].tolist(),
                    "x": [var] * len(grp_d), "name": str(grp),
                    "marker": {"color": m_colors[gi % len(m_colors)]},
                    "legendgroup": str(grp), "showlegend": vi == 0
                })
        result["plots"].append({
            "title": "Response Distributions by Group",
            "data": box_traces_m,
            "layout": {"height": 300, "boxmode": "group", "xaxis": {"title": "Response"}, "yaxis": {"title": "Value"}, "template": "plotly_white"}
        })

        # Correlation heatmap of response variables
        corr_mat = data[responses].corr().values
        result["plots"].append({
            "data": [{
                "z": corr_mat.tolist(), "x": responses, "y": responses,
                "type": "heatmap", "colorscale": [[0, "#d94a4a"], [0.5, "#f0f4f0"], [1, "#2c5f2d"]],
                "zmin": -1, "zmax": 1,
                "text": [[f"{corr_mat[i][j]:.3f}" for j in range(len(responses))] for i in range(len(responses))],
                "texttemplate": "%{text}", "showscale": True
            }],
            "layout": {"title": "Response Correlation Matrix", "height": 300, "template": "plotly_white"}
        })

        result["guide_observation"] = f"MANOVA: Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}; Pillai's V = {pillai:.4f}, p = {p_pillai:.4f}. " + ("Multivariate effect detected." if p_pillai < alpha else "No multivariate effect.")
        result["statistics"] = {
            "wilks_lambda": wilks, "wilks_F": F_wilks, "wilks_p": p_wilks,
            "pillai_trace": pillai, "pillai_F": F_pillai, "pillai_p": p_pillai,
            "hotelling_lawley": hl_trace, "hl_F": F_hl, "hl_p": p_hl,
            "roys_root": roy, "roy_F": F_roy, "roy_p": p_roy,
            "eigenvalues": eigenvalues.tolist(),
            "n_groups": k, "n_responses": p, "N": N
        }

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
            formula = f'{response} ~ C({fixed_factor})'
            model = mixedlm(formula, data, groups=data[random_factor])
            fit = model.fit(reml=True)

            # Extract results
            fixed_effects = {}
            for name, val in fit.fe_params.items():
                pval = float(fit.pvalues[name]) if name in fit.pvalues else None
                se = float(fit.bse[name]) if name in fit.bse else None
                fixed_effects[name] = {"coef": float(val), "se": se, "p_value": pval}

            # Variance components
            var_random = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
            var_residual = float(fit.scale)
            var_total = var_random + var_residual
            icc = var_random / var_total if var_total > 0 else 0

            # Group stats
            fixed_levels = sorted(data[fixed_factor].unique().tolist(), key=str)
            random_levels = sorted(data[random_factor].unique().tolist(), key=str)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>NESTED ANOVA (Mixed-Effects Model)<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Fixed factor:<</COLOR>> {fixed_factor} ({len(fixed_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>Random factor (nesting):<</COLOR>> {random_factor} ({len(random_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary += f"<<COLOR:text>>Fixed Effects:<</COLOR>>\n"
            summary += f"{'Term':<30} {'Coef':>8} {'SE':>8} {'p-value':>8} {'Sig':>5}\n"
            summary += f"{'─' * 62}\n"
            for name, fe in fixed_effects.items():
                sig = "<<COLOR:good>>*<</COLOR>>" if fe["p_value"] is not None and fe["p_value"] < alpha else ""
                p_str = f"{fe['p_value']:.4f}" if fe["p_value"] is not None else "N/A"
                se_str = f"{fe['se']:.4f}" if fe["se"] is not None else "N/A"
                summary += f"{name:<30} {fe['coef']:>8.4f} {se_str:>8} {p_str:>8} {sig:>5}\n"

            summary += f"\n<<COLOR:text>>Variance Components:<</COLOR>>\n"
            summary += f"  {random_factor} (random): {var_random:.4f} ({icc*100:.1f}% of total)\n"
            summary += f"  Residual: {var_residual:.4f} ({(1-icc)*100:.1f}% of total)\n"
            summary += f"  Total: {var_total:.4f}\n"
            summary += f"  ICC (Intraclass Correlation): {icc:.4f}\n\n"

            if icc > 0.1:
                summary += f"<<COLOR:good>>ICC = {icc:.3f} — substantial variation attributed to {random_factor}.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The nesting structure accounts for {icc*100:.1f}% of the variance. Ignoring it would inflate Type I error.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:text>>ICC = {icc:.3f} — low variation from {random_factor}. A standard ANOVA may suffice.<</COLOR>>\n"

            # Check if fixed factor is significant
            sig_fixed = any(fe["p_value"] is not None and fe["p_value"] < alpha
                           for name, fe in fixed_effects.items() if name != "Intercept")
            if sig_fixed:
                summary += f"<<COLOR:good>>Fixed factor {fixed_factor} has significant effect.<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>Fixed factor {fixed_factor} not significant after accounting for {random_factor}.<</COLOR>>"

            result["summary"] = summary

            # Box plot with nesting structure
            traces = []
            for i, fl in enumerate(fixed_levels):
                subset = data[data[fixed_factor] == fl]
                traces.append({
                    "type": "box",
                    "y": subset[response].tolist(),
                    "x": [str(fl)] * len(subset),
                    "name": str(fl),
                    "boxpoints": "all",
                    "jitter": 0.3,
                    "pointpos": 0,
                    "marker": {"size": 4, "opacity": 0.5}
                })

            result["plots"].append({
                "title": f"Nested ANOVA: {response} by {fixed_factor} (nested in {random_factor})",
                "data": traces,
                "layout": {"height": 300, "yaxis": {"title": response}, "xaxis": {"title": fixed_factor}}
            })

            # Residuals vs fitted values
            fitted_vals = fit.fittedvalues
            resid_vals = fit.resid
            result["plots"].append({
                "title": "Residuals vs Fitted Values",
                "data": [{
                    "x": fitted_vals.tolist(), "y": resid_vals.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted Values"}, "yaxis": {"title": "Residuals"},
                    "shapes": [{"type": "line", "x0": float(fitted_vals.min()), "x1": float(fitted_vals.max()),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # Normal Q-Q plot of residuals
            from scipy import stats as qstats
            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)]
            result["plots"].append({
                "title": "Normal Q-Q Plot of Residuals",
                "data": [
                    {"x": theoretical_q, "y": sorted_resid.tolist(), "mode": "markers", "type": "scatter",
                     "marker": {"color": "#4a9f6e", "size": 4}, "name": "Residuals"},
                    {"x": [theoretical_q[0], theoretical_q[-1]], "y": [theoretical_q[0] * np.std(sorted_resid) + np.mean(sorted_resid),
                     theoretical_q[-1] * np.std(sorted_resid) + np.mean(sorted_resid)],
                     "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": "Reference"}
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Sample Quantiles"},
                    "template": "plotly_white"
                }
            })

            result["guide_observation"] = f"Nested ANOVA: ICC = {icc:.3f} ({icc*100:.1f}% variance from {random_factor}). " + ("Fixed effect significant." if sig_fixed else "Fixed effect not significant.")
            result["statistics"] = {
                "fixed_effects": fixed_effects,
                "var_random": var_random,
                "var_residual": var_residual,
                "icc": icc,
                "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
                "bic": float(fit.bic) if hasattr(fit, 'bic') else None,
                "converged": fit.converged if hasattr(fit, 'converged') else True
            }

        except ImportError:
            result["summary"] = "Nested ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Nested ANOVA error: {str(e)}"

    elif analysis_id == "acceptance_sampling":
        """
        Acceptance Sampling Plans — single and double sampling for lot inspection.
        Computes OC curve, AOQ curve, ATI, and determines accept/reject based on AQL and LTPD.
        Supports both attribute (defective count) and variable (normal) plans.
        """
        import numpy as np
        from scipy import stats as scipy_stats
        from scipy.special import comb as nCr

        plan_type = config.get("plan_type", "single")  # single, double
        n_sample = int(config.get("sample_size", 50))
        accept_num = int(config.get("accept_number", 2))  # Ac
        lot_size = int(config.get("lot_size", 1000))
        aql = float(config.get("aql", 0.01))  # Acceptable Quality Level
        ltpd = float(config.get("ltpd", 0.05))  # Lot Tolerance Percent Defective

        # For double sampling
        n1 = int(config.get("n1", 30))
        c1 = int(config.get("c1", 1))  # Accept on first sample
        r1 = int(config.get("r1", 4))  # Reject on first sample
        n2 = int(config.get("n2", 30))
        c2 = int(config.get("c2", 4))  # Accept on combined

        try:
            # OC curve: probability of acceptance at various defect rates
            p_range = np.linspace(0, min(0.15, ltpd * 3), 200)

            if plan_type == "double":
                # Double sampling OC curve
                pa_values = []
                for p in p_range:
                    if p == 0:
                        pa_values.append(1.0)
                        continue
                    # P(accept on 1st sample): P(d1 <= c1)
                    p_accept_1 = sum(scipy_stats.binom.pmf(d, n1, p) for d in range(c1 + 1))
                    # P(reject on 1st sample): P(d1 >= r1)
                    p_reject_1 = sum(scipy_stats.binom.pmf(d, n1, p) for d in range(r1, n1 + 1))
                    # P(go to 2nd sample): c1 < d1 < r1
                    p_second = 1 - p_accept_1 - p_reject_1
                    # P(accept on combined): P(d1+d2 <= c2) for each d1 in [c1+1, r1-1]
                    p_accept_2 = 0
                    for d1 in range(c1 + 1, r1):
                        p_d1 = scipy_stats.binom.pmf(d1, n1, p)
                        max_d2 = c2 - d1
                        if max_d2 >= 0:
                            p_accept_2 += p_d1 * sum(scipy_stats.binom.pmf(d2, n2, p) for d2 in range(max_d2 + 1))
                    pa = p_accept_1 + p_accept_2
                    pa_values.append(float(min(1.0, max(0.0, pa))))
                pa_values = np.array(pa_values)
                plan_desc = f"Double: n1={n1}, c1={c1}, r1={r1}, n2={n2}, c2={c2}"
            else:
                # Single sampling OC curve using binomial
                pa_values = np.array([float(scipy_stats.binom.cdf(accept_num, n_sample, p)) if p > 0 else 1.0 for p in p_range])
                plan_desc = f"Single: n={n_sample}, Ac={accept_num}"

            # Key probabilities
            pa_aql = float(np.interp(aql, p_range, pa_values))
            pa_ltpd = float(np.interp(ltpd, p_range, pa_values))

            # Producer's risk (alpha) = 1 - P(accept at AQL)
            alpha_risk = 1 - pa_aql
            # Consumer's risk (beta) = P(accept at LTPD)
            beta_risk = pa_ltpd

            # AOQ curve (Average Outgoing Quality)
            aoq_values = pa_values * p_range * (lot_size - n_sample) / lot_size
            aoql = float(np.max(aoq_values))
            aoql_p = float(p_range[np.argmax(aoq_values)])

            # ATI (Average Total Inspection)
            ati_values = n_sample * pa_values + lot_size * (1 - pa_values)

            # OC Curve plot
            result["plots"].append({
                "data": [
                    {
                        "x": (p_range * 100).tolist(), "y": pa_values.tolist(),
                        "mode": "lines", "name": "OC Curve",
                        "line": {"color": "#2c5f2d", "width": 2}
                    },
                    {
                        "x": [aql * 100], "y": [pa_aql],
                        "mode": "markers+text", "name": f"AQL ({aql*100:.1f}%)",
                        "marker": {"color": "#4a90d9", "size": 10},
                        "text": [f"AQL: Pa={pa_aql:.3f}"], "textposition": "top right"
                    },
                    {
                        "x": [ltpd * 100], "y": [pa_ltpd],
                        "mode": "markers+text", "name": f"LTPD ({ltpd*100:.1f}%)",
                        "marker": {"color": "#d94a4a", "size": 10},
                        "text": [f"LTPD: Pa={pa_ltpd:.3f}"], "textposition": "top left"
                    }
                ],
                "layout": {
                    "title": f"Operating Characteristic (OC) Curve — {plan_desc}",
                    "xaxis": {"title": "Lot Defect Rate (%)"},
                    "yaxis": {"title": "Probability of Acceptance", "range": [0, 1.05]},
                    "template": "plotly_white"
                }
            })

            # AOQ Curve plot
            result["plots"].append({
                "data": [
                    {
                        "x": (p_range * 100).tolist(), "y": (aoq_values * 100).tolist(),
                        "mode": "lines", "name": "AOQ Curve",
                        "line": {"color": "#d9a04a", "width": 2}
                    },
                    {
                        "x": [aoql_p * 100], "y": [aoql * 100],
                        "mode": "markers+text", "name": f"AOQL={aoql*100:.3f}%",
                        "marker": {"color": "#d94a4a", "size": 10},
                        "text": [f"AOQL={aoql*100:.3f}%"], "textposition": "top right"
                    }
                ],
                "layout": {
                    "title": "Average Outgoing Quality (AOQ) Curve",
                    "xaxis": {"title": "Incoming Defect Rate (%)"},
                    "yaxis": {"title": "Average Outgoing Quality (%)"},
                    "template": "plotly_white"
                }
            })

            result["summary"] = f"**Acceptance Sampling Plan**\n\n**Plan:** {plan_desc}\n**Lot size:** {lot_size}\n\n| Metric | Value |\n|---|---|\n| P(accept) at AQL ({aql*100:.1f}%) | {pa_aql:.4f} |\n| P(accept) at LTPD ({ltpd*100:.1f}%) | {pa_ltpd:.4f} |\n| Producer's risk (α) | {alpha_risk:.4f} ({alpha_risk*100:.1f}%) |\n| Consumer's risk (β) | {beta_risk:.4f} ({beta_risk*100:.1f}%) |\n| AOQL | {aoql*100:.4f}% at p={aoql_p*100:.2f}% |\n| ATI at AQL | {float(np.interp(aql, p_range, ati_values)):.0f} units |"

            result["guide_observation"] = f"Acceptance sampling ({plan_desc}): α={alpha_risk:.3f}, β={beta_risk:.3f}, AOQL={aoql*100:.4f}%."

            result["statistics"] = {
                "plan_type": plan_type,
                "sample_size": n_sample if plan_type == "single" else n1 + n2,
                "lot_size": lot_size,
                "aql": aql,
                "ltpd": ltpd,
                "pa_at_aql": pa_aql,
                "pa_at_ltpd": pa_ltpd,
                "producers_risk_alpha": alpha_risk,
                "consumers_risk_beta": beta_risk,
                "aoql": aoql,
                "aoql_defect_rate": aoql_p
            }

        except Exception as e:
            result["summary"] = f"Acceptance sampling error: {str(e)}"

    elif analysis_id == "glm":
        """
        General Linear Model — unified engine for ANOVA, ANCOVA, multivariate regression,
        and mixed-effects models. Handles:
        - Pure ANOVA (factors only)
        - Pure regression (covariates only)
        - ANCOVA (factors + covariates with interactions and LS-Means)
        - Mixed models (fixed + random factors)
        - Two-way+ interactions between factors and factor*covariate
        Produces Type III ANOVA table, LS-Means, effect sizes, interaction plots,
        and full residual diagnostics (4-panel).
        """
        response = config.get("response") or config.get("var")
        fixed_factors = config.get("fixed_factors", [])
        random_factors = config.get("random_factors", [])
        covariates = config.get("covariates", [])
        include_interactions = config.get("interactions", True)
        include_factor_cov_interactions = config.get("factor_covariate_interactions", False)
        alpha = 1 - config.get("conf", 95) / 100

        # Support single-value fallbacks from generic dialogs
        if not fixed_factors and config.get("factor"):
            fixed_factors = [config["factor"]]
        if not random_factors and config.get("random_factor"):
            random_factors = [config["random_factor"]]
        if not covariates and config.get("covariate"):
            covariates = [config["covariate"]]

        all_cols = [response] + fixed_factors + random_factors + covariates
        try:
            data = df[all_cols].dropna()
            N = len(data)
            from scipy import stats as qstats

            # ── Build formula ──
            terms = []
            for f in fixed_factors:
                terms.append(f"C({f})")
            for c in covariates:
                terms.append(c)

            # Factor*Factor interactions
            if include_interactions and len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        terms.append(f"C({fixed_factors[i]}):C({fixed_factors[j]})")

            # Factor*Covariate interactions (ANCOVA: test homogeneity of slopes)
            if covariates and fixed_factors:
                if include_factor_cov_interactions or len(fixed_factors) == 1:
                    # Auto-include for single-factor ANCOVA to test assumption
                    for f in fixed_factors:
                        for c in covariates:
                            terms.append(f"C({f}):{c}")

            formula = f"{response} ~ " + " + ".join(terms) if terms else f"{response} ~ 1"

            # Determine model type label
            has_factors = len(fixed_factors) > 0
            has_covariates = len(covariates) > 0
            has_random = len(random_factors) > 0
            if has_factors and has_covariates:
                model_label = "ANCOVA" if not has_random else "Mixed ANCOVA"
            elif has_factors and not has_covariates:
                model_label = "ANOVA (GLM)" if not has_random else "Mixed-Effects ANOVA"
            elif has_covariates and not has_factors:
                model_label = "Multiple Regression" if len(covariates) > 1 else "Simple Regression"
            else:
                model_label = "Intercept-Only Model"

            # ═══════════════════════════════════════════
            # MIXED MODEL (random factors present)
            # ═══════════════════════════════════════════
            if has_random:
                from statsmodels.formula.api import mixedlm
                group_var = random_factors[0]
                model = mixedlm(formula, data, groups=data[group_var])
                fit = model.fit(reml=True)

                var_random = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
                var_residual = float(fit.scale)
                var_total = var_random + var_residual
                icc = var_random / var_total if var_total > 0 else 0

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                summary_text += f"<<COLOR:highlight>>Fixed:<</COLOR>> {', '.join(fixed_factors + covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Random:<</COLOR>> {', '.join(random_factors)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

                summary_text += f"<<COLOR:text>>Fixed Effects:<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'Sig':>5} {f'{int((1-alpha)*100)}% CI':>22}\n"
                summary_text += f"{'─' * 105}\n"
                for name in fit.fe_params.index:
                    coef = float(fit.fe_params[name])
                    se = float(fit.bse[name]) if name in fit.bse.index else None
                    pv = float(fit.pvalues[name]) if name in fit.pvalues.index else None
                    z = coef / se if se and se > 0 else 0
                    sig = "<<COLOR:good>>*<</COLOR>>" if pv is not None and pv < alpha else ""
                    p_str = f"{pv:.4f}" if pv is not None else "N/A"
                    se_str = f"{se:.4f}" if se is not None else "N/A"
                    _ci_str = f"[{coef - 1.96 * se:.4f}, {coef + 1.96 * se:.4f}]" if se else ""
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {sig:>5} {_ci_str:>22}\n"

                summary_text += f"\n<<COLOR:text>>Variance Components:<</COLOR>>\n"
                summary_text += f"  {group_var} (random): {var_random:.4f} ({icc*100:.1f}% of total)\n"
                summary_text += f"  Residual: {var_residual:.4f} ({(1-icc)*100:.1f}% of total)\n"
                summary_text += f"  ICC (Intraclass Correlation): {icc:.4f}\n"

                if icc > 0.1:
                    summary_text += f"\n<<COLOR:good>>ICC = {icc:.3f} — substantial clustering. Mixed model is appropriate.<</COLOR>>"
                else:
                    summary_text += f"\n<<COLOR:text>>ICC = {icc:.3f} — low clustering. A fixed-effects model may suffice.<</COLOR>>"

                fitted_vals = fit.fittedvalues
                resid_vals = fit.resid

                result["statistics"] = {
                    "model_type": "mixed", "model_label": model_label,
                    "n": N, "formula": formula,
                    "var_random": var_random, "var_residual": var_residual,
                    "icc": icc,
                    "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
                }

            # ═══════════════════════════════════════════
            # FIXED MODEL (OLS — ANOVA/ANCOVA/Regression)
            # ═══════════════════════════════════════════
            else:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols

                model = ols(formula, data=data).fit()

                # Type III ANOVA table
                try:
                    anova_table = sm.stats.anova_lm(model, typ=3)
                except Exception:
                    anova_table = sm.stats.anova_lm(model, typ=2)

                # Compute effect sizes (partial eta-squared)
                ss_residual = float(anova_table.loc['Residual', 'sum_sq']) if 'Residual' in anova_table.index else 0
                ss_total = float(anova_table['sum_sq'].sum())

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                if fixed_factors:
                    summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(fixed_factors)}\n"
                if covariates:
                    summary_text += f"<<COLOR:highlight>>Covariates:<</COLOR>> {', '.join(covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}, R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}\n\n"

                # ANOVA table with partial eta-squared
                summary_text += f"<<COLOR:text>>Analysis of Variance (Type III SS):<</COLOR>>\n"
                eta_header = " η²p" if has_factors else ""
                summary_text += f"{'Source':<30} {'DF':>5} {'Adj SS':>12} {'Adj MS':>12} {'F':>10} {'p-value':>10}{eta_header:>8}\n"
                summary_text += f"{'─' * (87 + (8 if has_factors else 0))}\n"

                for idx in anova_table.index:
                    row = anova_table.loc[idx]
                    df_val = int(row['df']) if 'df' in row else ""
                    ss = float(row['sum_sq']) if 'sum_sq' in row else 0
                    ms = float(row['mean_sq']) if 'mean_sq' in row else 0
                    f_val = float(row['F']) if 'F' in row and not np.isnan(row['F']) else None
                    pv = float(row['PR(>F)']) if 'PR(>F)' in row and not np.isnan(row['PR(>F)']) else None
                    sig = "<<COLOR:good>>*<</COLOR>>" if pv is not None and pv < alpha else ""
                    f_str = f"{f_val:.4f}" if f_val is not None else ""
                    p_str = f"{pv:.4f}" if pv is not None else ""
                    # Partial eta-squared = SS_effect / (SS_effect + SS_error)
                    if has_factors and idx != 'Residual' and idx != 'Intercept' and ss_residual > 0:
                        eta_p = ss / (ss + ss_residual)
                        eta_str = f"{eta_p:>7.3f}"
                    else:
                        eta_str = "" if has_factors else ""
                    summary_text += f"{str(idx):<30} {df_val:>5} {ss:>12.4f} {ms:>12.4f} {f_str:>10} {p_str:>10} {sig} {eta_str}\n"

                summary_text += f"\n<<COLOR:text>>Model Summary:<</COLOR>>\n"
                summary_text += f"  S (root MSE): {np.sqrt(model.mse_resid):.4f}\n"
                summary_text += f"  R²: {model.rsquared:.4f}  Adj R²: {model.rsquared_adj:.4f}\n"
                summary_text += f"  AIC: {model.aic:.1f}  BIC: {model.bic:.1f}\n"

                # Coefficients table with CIs
                _glm_ci = model.conf_int(alpha=alpha)
                summary_text += f"\n<<COLOR:text>>Coefficients:<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'t':>8} {'p-value':>10} {f'{int((1-alpha)*100)}% CI':>22}\n"
                summary_text += f"{'─' * 97}\n"
                for name in model.params.index:
                    coef = float(model.params[name])
                    se = float(model.bse[name])
                    t = float(model.tvalues[name])
                    pv = float(model.pvalues[name])
                    _ci_lo = float(_glm_ci.loc[name, 0])
                    _ci_hi = float(_glm_ci.loc[name, 1])
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {pv:>10.4f} [{_ci_lo:.4f}, {_ci_hi:.4f}]\n"

                # ── LS-Means (Adjusted Means) for ANCOVA ──
                if has_factors and has_covariates:
                    summary_text += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
                    summary_text += f"<<COLOR:text>>Least-Squares Means (Adjusted Means):<</COLOR>>\n"
                    summary_text += f"<<COLOR:text>>Covariates held at their means: " + ", ".join([f"{c}={data[c].mean():.4f}" for c in covariates]) + "<</COLOR>>\n\n"

                    for factor in fixed_factors:
                        levels = sorted(data[factor].unique().tolist(), key=str)
                        raw_means = data.groupby(factor)[response].mean()
                        raw_sds = data.groupby(factor)[response].std()
                        raw_ns = data.groupby(factor)[response].count()

                        # Compute LS-means by predicting at covariate means
                        cov_means = {c: data[c].mean() for c in covariates}
                        ls_means = {}
                        for lev in levels:
                            pred_data = data[data[factor] == lev].copy()
                            for c in covariates:
                                pred_data[c] = cov_means[c]
                            ls_means[lev] = float(model.predict(pred_data).mean())

                        summary_text += f"  {factor}:\n"
                        summary_text += f"  {'Level':<20} {'N':>6} {'Raw Mean':>10} {'Adj Mean':>10} {'Std Dev':>10}\n"
                        summary_text += f"  {'─' * 58}\n"
                        for lev in levels:
                            rm = float(raw_means[lev])
                            adj = ls_means[lev]
                            sd = float(raw_sds[lev])
                            n_lev = int(raw_ns[lev])
                            summary_text += f"  {str(lev):<20} {n_lev:>6} {rm:>10.4f} {adj:>10.4f} {sd:>10.4f}\n"

                        diff = max(ls_means.values()) - min(ls_means.values())
                        raw_diff = float(raw_means.max()) - float(raw_means.min())
                        if abs(raw_diff - diff) > 0.01 * abs(raw_diff + 0.001):
                            summary_text += f"\n  <<COLOR:highlight>>Note: Adjusted means differ from raw means — covariate adjustment matters.<</COLOR>>\n"
                            summary_text += f"  Raw range: {raw_diff:.4f} → Adjusted range: {diff:.4f}\n"

                    # Homogeneity of slopes test (factor*covariate interaction significance)
                    fxcov_terms = [f"C({f}):{c}" for f in fixed_factors for c in covariates]
                    sig_interactions = []
                    for term in fxcov_terms:
                        for idx in anova_table.index:
                            if term.replace("C(", "").replace(")", "") in str(idx) or str(idx) == term:
                                pv_term = float(anova_table.loc[idx, 'PR(>F)']) if 'PR(>F)' in anova_table.columns and not np.isnan(anova_table.loc[idx, 'PR(>F)']) else None
                                if pv_term is not None and pv_term < alpha:
                                    sig_interactions.append((str(idx), pv_term))

                    if sig_interactions:
                        summary_text += f"\n<<COLOR:bad>>⚠ Homogeneity of Slopes Violated:<</COLOR>>\n"
                        for term, pv in sig_interactions:
                            summary_text += f"  {term}: p = {pv:.4f} — slopes differ across groups. ANCOVA assumption violated.\n"
                        summary_text += f"  <<COLOR:text>>Consider: separate regressions per group, or remove the covariate.<</COLOR>>\n"
                    elif has_factors and has_covariates:
                        summary_text += f"\n<<COLOR:good>>Homogeneity of slopes OK — factor*covariate interactions are not significant.<</COLOR>>\n"

                fitted_vals = model.fittedvalues
                resid_vals = model.resid

                result["statistics"] = {
                    "model_type": "fixed", "model_label": model_label,
                    "n": N, "formula": formula,
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_pvalue": float(model.f_pvalue),
                    "aic": float(model.aic), "bic": float(model.bic),
                    "root_mse": float(np.sqrt(model.mse_resid)),
                }

            result["summary"] = summary_text

            # ═══════════════════════════════════════════
            # PLOTS — Full 4-panel residual diagnostics
            # ═══════════════════════════════════════════

            # 1. Residuals vs Fitted
            result["plots"].append({
                "title": "Residuals vs Fitted Values",
                "data": [{
                    "x": fitted_vals.tolist(), "y": resid_vals.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted Value"}, "yaxis": {"title": "Residual"},
                    "shapes": [{"type": "line", "x0": float(fitted_vals.min()), "x1": float(fitted_vals.max()),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # 2. Normal Q-Q Plot
            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)]
            result["plots"].append({
                "title": "Normal Probability Plot of Residuals",
                "data": [
                    {"x": theoretical_q, "y": sorted_resid.tolist(), "mode": "markers", "type": "scatter",
                     "marker": {"color": "#4a9f6e", "size": 4}, "name": "Residuals"},
                    {"x": [theoretical_q[0], theoretical_q[-1]],
                     "y": [theoretical_q[0] * np.std(sorted_resid) + np.mean(sorted_resid),
                           theoretical_q[-1] * np.std(sorted_resid) + np.mean(sorted_resid)],
                     "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": "Reference"}
                ],
                "layout": {"height": 280, "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Sample Quantiles"}, "template": "plotly_white"}
            })

            # 3. Residuals Histogram
            hist_vals, bin_edges = np.histogram(resid_vals.values, bins='auto')
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append({
                "title": "Histogram of Residuals",
                "data": [{"type": "bar", "x": bin_centers, "y": hist_vals.tolist(),
                          "marker": {"color": "#4a90d9", "opacity": 0.7}}],
                "layout": {"height": 250, "xaxis": {"title": "Residual"}, "yaxis": {"title": "Frequency"}, "template": "plotly_white"}
            })

            # 4. Residuals vs Observation Order
            result["plots"].append({
                "title": "Residuals vs Observation Order",
                "data": [{
                    "x": list(range(1, len(resid_vals) + 1)), "y": resid_vals.tolist(),
                    "mode": "lines+markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 3}, "line": {"color": "#4a9f6e", "width": 1}
                }],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Observation Order"}, "yaxis": {"title": "Residual"},
                    "shapes": [{"type": "line", "x0": 1, "x1": len(resid_vals),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # ── Main Effects Plots ──
            if fixed_factors:
                for factor in fixed_factors:
                    levels = sorted(data[factor].unique().tolist(), key=str)
                    means = [float(data[data[factor] == lev][response].mean()) for lev in levels]
                    cis = [1.96 * float(data[data[factor] == lev][response].std()) / np.sqrt(len(data[data[factor] == lev])) for lev in levels]
                    grand_mean = float(data[response].mean())
                    result["plots"].append({
                        "title": f"Main Effects Plot: {factor}",
                        "data": [{
                            "x": [str(l) for l in levels], "y": means,
                            "error_y": {"type": "data", "array": cis, "visible": True, "color": "#4a90d9"},
                            "mode": "lines+markers", "type": "scatter",
                            "marker": {"color": "#4a90d9", "size": 8},
                            "line": {"color": "#4a90d9", "width": 2}, "name": "Mean"
                        }],
                        "layout": {
                            "height": 260, "xaxis": {"title": factor}, "yaxis": {"title": f"Mean of {response}"},
                            "shapes": [{"type": "line", "x0": -0.5, "x1": len(levels) - 0.5,
                                        "y0": grand_mean, "y1": grand_mean,
                                        "line": {"color": "#999", "dash": "dash", "width": 1}}],
                            "template": "plotly_white"
                        }
                    })

            # ── Interaction Plots (Factor × Factor) ──
            if len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        f1, f2 = fixed_factors[i], fixed_factors[j]
                        traces = []
                        colors_int = ["#4a90d9", "#d94a4a", "#4a9f6e", "#d9a04a", "#9b59b6", "#e67e22"]
                        f2_levels = sorted(data[f2].unique().tolist(), key=str)
                        f1_levels = sorted(data[f1].unique().tolist(), key=str)
                        for ci, lev2 in enumerate(f2_levels):
                            sub = data[data[f2] == lev2]
                            means_int = [float(sub[sub[f1] == lev1][response].mean()) if len(sub[sub[f1] == lev1]) > 0 else None for lev1 in f1_levels]
                            traces.append({
                                "x": [str(l) for l in f1_levels], "y": means_int,
                                "mode": "lines+markers", "name": f"{f2}={lev2}",
                                "marker": {"color": colors_int[ci % len(colors_int)], "size": 7},
                                "line": {"color": colors_int[ci % len(colors_int)], "width": 2}
                            })
                        result["plots"].append({
                            "title": f"Interaction Plot: {f1} × {f2}",
                            "data": traces,
                            "layout": {"height": 280, "xaxis": {"title": f1}, "yaxis": {"title": f"Mean of {response}"}, "template": "plotly_white"}
                        })

            # ── ANCOVA: Covariate scatter by factor ──
            if has_factors and has_covariates:
                for c in covariates:
                    for f in fixed_factors:
                        traces_cov = []
                        colors_cov = ["#4a90d9", "#d94a4a", "#4a9f6e", "#d9a04a", "#9b59b6"]
                        f_levels = sorted(data[f].unique().tolist(), key=str)
                        for fi, lev in enumerate(f_levels):
                            sub = data[data[f] == lev]
                            traces_cov.append({
                                "x": sub[c].tolist(), "y": sub[response].tolist(),
                                "mode": "markers", "name": f"{f}={lev}",
                                "marker": {"color": colors_cov[fi % len(colors_cov)], "size": 5, "opacity": 0.7}
                            })
                            # Add regression line per group
                            if len(sub) > 2:
                                slope, intercept, _, _, _ = qstats.linregress(sub[c].values, sub[response].values)
                                x_line = [float(sub[c].min()), float(sub[c].max())]
                                y_line = [intercept + slope * x for x in x_line]
                                traces_cov.append({
                                    "x": x_line, "y": y_line,
                                    "mode": "lines", "name": f"{lev} fit",
                                    "line": {"color": colors_cov[fi % len(colors_cov)], "width": 1.5, "dash": "dash"},
                                    "showlegend": False
                                })
                        result["plots"].append({
                            "title": f"Covariate Plot: {response} vs {c} by {f}",
                            "data": traces_cov,
                            "layout": {"height": 300, "xaxis": {"title": c}, "yaxis": {"title": response}, "template": "plotly_white"}
                        })

            result["guide_observation"] = f"{model_label}: {response} ~ {' + '.join(fixed_factors + covariates + random_factors)}. N={N}."

        except Exception as e:
            result["summary"] = f"GLM error: {str(e)}"

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
            m = (abs(p - df_h) - 1) / 2
            n_param = (df_e - p - 1) / 2
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
                    f_wilks = ((1 - wilks**(1/t)) / (wilks**(1/t))) * (df2 / df1)
                    from scipy import stats as fstats
                    p_wilks = 1 - fstats.f.cdf(f_wilks, df1, df2)
                else:
                    f_wilks = None
                    p_wilks = None
            else:
                f_wilks = None
                p_wilks = None

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
            summary_text += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Multivariate Test Statistics:<</COLOR>>\n"
            summary_text += f"{'Test':<25} {'Value':>10} {'Approx F':>10} {'p-value':>10}\n"
            summary_text += f"{'─' * 57}\n"
            pillai_label = "Pillai's Trace"
            summary_text += f"{pillai_label:<25} {pillai:>10.4f} {'':>10} {'':>10}\n"
            wilks_f_str = f"{f_wilks:.4f}" if f_wilks is not None else "N/A"
            wilks_p_str = f"{p_wilks:.4f}" if p_wilks is not None else "N/A"
            wilks_label = "Wilks' Lambda"
            summary_text += f"{wilks_label:<25} {wilks:>10.4f} {wilks_f_str:>10} {wilks_p_str:>10}\n"
            summary_text += f"{'Hotelling-Lawley':<25} {hotelling:>10.4f} {'':>10} {'':>10}\n"
            roy_label = "Roy's Greatest Root"
            summary_text += f"{roy_label:<25} {roy:>10.4f} {'':>10} {'':>10}\n\n"

            # Univariate ANOVAs
            summary_text += f"<<COLOR:text>>Univariate ANOVA per Response:<</COLOR>>\n"
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
                result["plots"].append({
                    "title": f"Group Means: {resp} by {factor}",
                    "data": [{
                        "x": [str(g) for g in groups], "y": means,
                        "error_y": {"type": "data", "array": sds, "visible": True},
                        "type": "bar", "marker": {"color": "#4a90d9"}
                    }],
                    "layout": {"height": 250, "yaxis": {"title": resp}, "xaxis": {"title": factor}, "template": "plotly_white"}
                })

            result["statistics"] = {
                "pillai": float(pillai), "wilks_lambda": float(wilks),
                "hotelling_lawley": float(hotelling), "roys_greatest_root": float(roy),
                "f_wilks": float(f_wilks) if f_wilks else None,
                "p_wilks": float(p_wilks) if p_wilks else None,
                "n_groups": k, "n_responses": p, "n": N
            }
            result["guide_observation"] = f"MANOVA: {', '.join(responses)} by {factor}. Wilks' Λ={wilks:.4f}" + (f", p={p_wilks:.4f}" if p_wilks else "") + "."

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
            summary_text += f"<<COLOR:title>>TOLERANCE INTERVAL<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method_desc}\n\n"
            summary_text += f"<<COLOR:highlight>>Confidence:<</COLOR>> {conf*100:.0f}%\n"
            summary_text += f"<<COLOR:highlight>>Coverage:<</COLOR>> {coverage*100:.0f}%\n\n"
            summary_text += f"<<COLOR:text>>Tolerance Interval:<</COLOR>> [{lower:.4f}, {upper:.4f}]\n"
            summary_text += f"<<COLOR:text>>Mean:<</COLOR>> {xbar:.4f}\n"
            summary_text += f"<<COLOR:text>>Std Dev:<</COLOR>> {s:.4f}\n\n"
            summary_text += f"<<COLOR:highlight>>Interpretation:<</COLOR>> With {conf*100:.0f}% confidence, at least {coverage*100:.0f}% of the population falls between {lower:.4f} and {upper:.4f}."

            result["summary"] = summary_text

            # Histogram with tolerance bounds
            hist_vals, bin_edges = np.histogram(vals, bins='auto')
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append({
                "title": f"Tolerance Interval: {var}",
                "data": [
                    {"type": "bar", "x": bin_centers, "y": hist_vals.tolist(), "marker": {"color": "#4a90d9", "opacity": 0.7}, "name": "Data"},
                ],
                "layout": {
                    "height": 300, "xaxis": {"title": var}, "yaxis": {"title": "Frequency"},
                    "shapes": [
                        {"type": "line", "x0": lower, "x1": lower, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
                        {"type": "line", "x0": upper, "x1": upper, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
                        {"type": "line", "x0": xbar, "x1": xbar, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#4a9f6e", "width": 2}},
                    ],
                    "annotations": [
                        {"x": lower, "y": max(hist_vals) * 1.05, "text": f"Lower: {lower:.2f}", "showarrow": False, "font": {"color": "#d94a4a"}},
                        {"x": upper, "y": max(hist_vals) * 1.05, "text": f"Upper: {upper:.2f}", "showarrow": False, "font": {"color": "#d94a4a"}},
                    ],
                    "template": "plotly_white"
                }
            })

            result["statistics"] = {
                "mean": xbar, "std": s, "n": n,
                "lower": lower, "upper": upper,
                "confidence": conf, "coverage": coverage,
                "k_factor": k_factor, "method": method
            }
            result["guide_observation"] = f"Tolerance interval for {var}: [{lower:.4f}, {upper:.4f}] ({conf*100:.0f}% conf, {coverage*100:.0f}% coverage)."

        except Exception as e:
            result["summary"] = f"Tolerance interval error: {str(e)}"

    elif analysis_id == "variance_components":
        """
        Variance Components — decomposes total variance into components from random factors.
        Uses ANOVA-based (Type I MS) or REML method.
        """
        response = config.get("response") or config.get("var")
        factors = config.get("factors", [])
        if not factors and config.get("factor"):
            factors = [config["factor"]]
        method = config.get("method", "anova")

        try:
            all_cols = [response] + factors
            data = df[all_cols].dropna()
            N = len(data)

            components = {}
            total_var = float(data[response].var())

            if method == "reml" and len(factors) == 1:
                from statsmodels.formula.api import mixedlm
                formula = f"{response} ~ 1"
                model = mixedlm(formula, data, groups=data[factors[0]])
                fit = model.fit(reml=True)
                var_factor = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
                var_error = float(fit.scale)
                components[factors[0]] = var_factor
                components["Error"] = var_error
            else:
                # ANOVA method: for each factor, compute between-group MS and within-group MS
                from scipy import stats as vstats
                remaining_var = total_var
                for factor in factors:
                    groups = data.groupby(factor)[response]
                    group_means = groups.mean()
                    grand_mean = data[response].mean()
                    k_groups = len(group_means)
                    n_per = N / k_groups  # average group size
                    ms_between = float(np.sum(groups.count() * (group_means - grand_mean)**2) / (k_groups - 1))
                    ms_within = float(np.sum(groups.apply(lambda x: np.sum((x - x.mean())**2))) / (N - k_groups))
                    var_component = max(0, (ms_between - ms_within) / n_per)
                    components[factor] = var_component
                components["Error"] = float(data.groupby(factors[0])[response].apply(lambda x: x.var()).mean()) if factors else total_var

            comp_total = sum(components.values())
            pct = {k: v / comp_total * 100 if comp_total > 0 else 0 for k, v in components.items()}

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>VARIANCE COMPONENTS ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method.upper()}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
            summary_text += f"{'Source':<20} {'Variance':>12} {'% of Total':>12} {'Std Dev':>12}\n"
            summary_text += f"{'─' * 58}\n"
            for source, var_val in components.items():
                summary_text += f"{source:<20} {var_val:>12.4f} {pct[source]:>11.1f}% {np.sqrt(var_val):>12.4f}\n"
            summary_text += f"{'─' * 58}\n"
            summary_text += f"{'Total':<20} {comp_total:>12.4f} {'100.0%':>12} {np.sqrt(comp_total):>12.4f}\n"

            result["summary"] = summary_text

            # Pie chart of variance components
            labels = list(components.keys())
            values = [components[k] for k in labels]
            colors = ["#4a90d9", "#d9a04a", "#4a9f6e", "#d94a4a", "#9b59b6", "#3498db"]
            result["plots"].append({
                "title": "Variance Components",
                "data": [{
                    "type": "pie",
                    "labels": labels,
                    "values": values,
                    "marker": {"colors": colors[:len(labels)]},
                    "textinfo": "label+percent",
                    "hole": 0.3
                }],
                "layout": {"height": 300, "template": "plotly_white"}
            })

            # Bar chart
            result["plots"].append({
                "title": "Variance Components (Bar)",
                "data": [{
                    "type": "bar", "x": labels, "y": values,
                    "marker": {"color": colors[:len(labels)]},
                    "text": [f"{pct[k]:.1f}%" for k in labels],
                    "textposition": "outside"
                }],
                "layout": {"height": 280, "yaxis": {"title": "Variance"}, "template": "plotly_white"}
            })

            result["statistics"] = {
                "components": {k: {"variance": v, "pct": pct[k], "std_dev": float(np.sqrt(v))} for k, v in components.items()},
                "total_variance": comp_total,
                "method": method, "n": N
            }
            result["guide_observation"] = f"Variance components: " + ", ".join([f"{k}={pct[k]:.1f}%" for k in components]) + "."

        except Exception as e:
            result["summary"] = f"Variance components error: {str(e)}"

    elif analysis_id == "ordinal_logistic":
        """
        Ordinal Logistic Regression — proportional odds model for ordered categorical outcomes.
        Uses statsmodels OrderedModel.
        """
        response = config.get("response") or config.get("var")
        predictors = config.get("predictors", [])
        if not predictors and config.get("predictor"):
            predictors = [config["predictor"]]

        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            all_cols = [response] + predictors
            data = df[all_cols].dropna()
            N = len(data)

            # Encode response as ordered categorical
            categories = sorted(data[response].unique().tolist(), key=str)
            cat_map = {c: i for i, c in enumerate(categories)}
            data["_y_ordinal"] = data[response].map(cat_map)

            # Fit proportional odds model
            X = data[predictors].values.astype(float)
            y = data["_y_ordinal"].values

            model = OrderedModel(y, X, distr='logit')
            fit = model.fit(method='bfgs', disp=False)

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>ORDINAL LOGISTIC REGRESSION (Proportional Odds)<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response} ({len(categories)} ordered levels)\n"
            summary_text += f"<<COLOR:highlight>>Levels:<</COLOR>> {' < '.join(str(c) for c in categories)}\n"
            summary_text += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary_text += f"{'Parameter':<25} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'OR':>10} {'95% CI (OR)':>20}\n"
            summary_text += f"{'─' * 97}\n"

            param_names = list(predictors) + [f"threshold_{i}" for i in range(len(categories) - 1)]
            for i, name in enumerate(param_names):
                if i < len(fit.params):
                    coef = float(fit.params[i])
                    se = float(fit.bse[i]) if i < len(fit.bse) else None
                    pv = float(fit.pvalues[i]) if i < len(fit.pvalues) else None
                    z = coef / se if se and se > 0 else 0
                    odds_ratio = np.exp(coef) if i < len(predictors) else None
                    se_str = f"{se:.4f}" if se else "N/A"
                    p_str = f"{pv:.4f}" if pv else "N/A"
                    or_str = f"{odds_ratio:.4f}" if odds_ratio else ""
                    ci_str = f"[{np.exp(coef - 1.96 * se):.4f}, {np.exp(coef + 1.96 * se):.4f}]" if odds_ratio and se else ""
                    summary_text += f"{name:<25} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {or_str:>10} {ci_str:>20}\n"

            if hasattr(fit, 'llf'):
                summary_text += f"\n<<COLOR:text>>Log-Likelihood:<</COLOR>> {fit.llf:.2f}\n"
            if hasattr(fit, 'aic'):
                summary_text += f"<<COLOR:text>>AIC:<</COLOR>> {fit.aic:.2f}\n"

            result["summary"] = summary_text

            # Predicted probabilities for first predictor
            if len(predictors) >= 1:
                x_range = np.linspace(float(data[predictors[0]].min()), float(data[predictors[0]].max()), 100)
                X_pred = np.zeros((100, len(predictors)))
                X_pred[:, 0] = x_range
                for j in range(1, len(predictors)):
                    X_pred[:, j] = data[predictors[j]].mean()

                pred_probs = fit.predict(X_pred)
                traces = []
                colors_cat = ["#4a90d9", "#d9a04a", "#4a9f6e", "#d94a4a", "#9b59b6"]
                for ci, cat in enumerate(categories):
                    col_idx = ci if ci < pred_probs.shape[1] else pred_probs.shape[1] - 1
                    traces.append({
                        "x": x_range.tolist(),
                        "y": pred_probs[:, col_idx].tolist(),
                        "mode": "lines", "name": str(cat),
                        "line": {"color": colors_cat[ci % len(colors_cat)], "width": 2}
                    })
                result["plots"].append({
                    "title": f"Predicted Probabilities by {predictors[0]}",
                    "data": traces,
                    "layout": {"height": 300, "xaxis": {"title": predictors[0]}, "yaxis": {"title": "Probability"}, "template": "plotly_white"}
                })

            result["statistics"] = {
                "n": N, "n_categories": len(categories),
                "log_likelihood": float(fit.llf) if hasattr(fit, 'llf') else None,
                "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
            }
            result["guide_observation"] = f"Ordinal logistic: {response} ({len(categories)} levels) ~ {', '.join(predictors)}. N={N}."

        except ImportError:
            result["summary"] = "Ordinal logistic requires statsmodels >= 0.13. Install with: pip install --upgrade statsmodels"
        except Exception as e:
            result["summary"] = f"Ordinal logistic error: {str(e)}"

    # ── Data Profile ──────────────────────────────────────────────────────
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
                tbl_rows.append(f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td>"
                    f"<td>{r.get('mean', ''):.4g}</td>" if col in numeric_cols else
                    f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td><td>-</td>"
                    f"<td>{r.get('top', '-')}</td></tr>")

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
                    for j in range(i+1, len(numeric_cols)):
                        pairs.append((numeric_cols[i], numeric_cols[j], abs(corr_matrix.iloc[i, j]), corr_matrix.iloc[i, j]))
                pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = pairs[:10]
                corr_lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, _, v in top_pairs]
                corr_text = "\n".join(corr_lines)

            total_missing = int(df.isnull().sum().sum())
            total_cells = n_rows * n_cols
            complete_rows = int((~df.isnull().any(axis=1)).sum())

            summary = f"""DATA PROFILE
{'='*50}
Shape: {n_rows} rows x {n_cols} columns
Memory: {mem_mb:.2f} MB

Column Types:
  Numeric:     {len(numeric_cols)}
  Categorical: {len(cat_cols)}
  Datetime:    {len(dt_cols)}

Missing Data:
  Total missing cells: {total_missing} / {total_cells} ({total_missing/total_cells*100:.1f}%)
  Complete rows: {complete_rows} / {n_rows} ({complete_rows/n_rows*100:.1f}%)

Top Correlations:
{corr_text if corr_text else '  (need 2+ numeric columns)'}"""

            result["summary"] = summary
            result["tables"] = [{"title": "Column Summary", "html": table_html}]

            # Plot 1: Correlation heatmap
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                result["plots"].append({
                    "data": [{"type": "heatmap", "z": corr_matrix.values.tolist(),
                              "x": numeric_cols, "y": numeric_cols,
                              "colorscale": "RdBu_r", "zmin": -1, "zmax": 1,
                              "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                              "texttemplate": "%{text}", "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>"}],
                    "layout": {"title": "Correlation Matrix", "height": 400, "template": "plotly_white"}
                })

            # Plot 2: Missing percentage bar chart
            miss_cols = df.columns[df.isnull().any()].tolist()
            if miss_cols:
                miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
                result["plots"].append({
                    "data": [{"type": "bar", "x": [round(p, 1) for p in miss_pcts], "y": miss_cols,
                              "orientation": "h", "marker": {"color": "rgba(232,71,71,0.6)"}}],
                    "layout": {"title": "Missing Data by Column (%)", "height": max(250, len(miss_cols) * 25),
                               "xaxis": {"title": "% Missing"}, "template": "plotly_white", "margin": {"l": 120}}
                })

            # Plot 3: Distribution grid (top 6 numeric)
            for col in numeric_cols[:6]:
                vals = df[col].dropna().tolist()
                if len(vals) > 0:
                    result["plots"].append({
                        "data": [{"type": "histogram", "x": vals, "marker": {"color": "rgba(74,144,217,0.6)"}, "nbinsx": 30}],
                        "layout": {"title": f"Distribution: {col}", "height": 250, "template": "plotly_white",
                                   "xaxis": {"title": col}, "yaxis": {"title": "Count"}}
                    })

            result["guide_observation"] = f"Data profile: {n_rows} rows, {n_cols} cols, {total_missing} missing cells, {len(numeric_cols)} numeric columns."

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
                        col_lines.append(f"  <<COLOR:accent>>{col}<<\/COLOR>>  N={len(vals)}  Mean={vals.mean():.4g}  StDev={vals.std():.4g}  Min={vals.min():.4g}  Max={vals.max():.4g}  Missing={miss_p:.1f}%")
                    else:
                        col_lines.append(f"  <<COLOR:warning>>{col}<<\/COLOR>>  (all missing)")
                else:
                    uniq = df[col].nunique()
                    top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                    col_lines.append(f"  <<COLOR:accent>>{col}<<\/COLOR>>  Unique={uniq}  Top=\"{str(top)[:20]}\"  Missing={miss_p:.1f}%")

            summary = f"""<<COLOR:title>>DATA OVERVIEW<</COLOR>>
{'='*50}
<<COLOR:highlight>>{n_rows}<<\/COLOR>> rows × <<COLOR:highlight>>{n_cols}<<\/COLOR>> columns
Numeric: {len(numeric_cols)}  |  Categorical: {len(cat_cols)}  |  Datetime: {len(dt_cols)}
Missing: {total_missing} / {total_cells} ({miss_pct:.1f}%)

<<COLOR:title>>Column Summary<</COLOR>>
{chr(10).join(col_lines)}"""

            result["summary"] = summary

            # Correlation heatmap (if 2+ numeric)
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                result["plots"].append({
                    "data": [{"type": "heatmap", "z": corr_matrix.values.tolist(),
                              "x": numeric_cols, "y": numeric_cols,
                              "colorscale": "RdBu_r", "zmin": -1, "zmax": 1,
                              "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                              "texttemplate": "%{text}", "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>"}],
                    "layout": {"title": "Correlation Matrix", "height": 350, "template": "plotly_white"}
                })

            # Distribution histograms (up to 12 numeric)
            for col in numeric_cols[:12]:
                vals = df[col].dropna().tolist()
                if len(vals) > 0:
                    result["plots"].append({
                        "data": [{"type": "histogram", "x": vals, "marker": {"color": "rgba(74,144,217,0.6)"}, "nbinsx": 30}],
                        "layout": {"title": f"{col}", "height": 220, "template": "plotly_white",
                                   "xaxis": {"title": col}, "yaxis": {"title": "Count"},
                                   "margin": {"t": 30, "b": 40, "l": 50, "r": 20}}
                    })

            # Missing bar if any
            miss_cols = df.columns[df.isnull().any()].tolist()
            if miss_cols:
                miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
                result["plots"].append({
                    "data": [{"type": "bar", "x": [round(p, 1) for p in miss_pcts], "y": miss_cols,
                              "orientation": "h", "marker": {"color": "rgba(232,71,71,0.6)"}}],
                    "layout": {"title": "Missing Data (%)", "height": max(200, len(miss_cols) * 22),
                               "xaxis": {"title": "% Missing"}, "template": "plotly_white", "margin": {"l": 120}}
                })

            result["guide_observation"] = f"Auto-profile: {n_rows} rows, {n_cols} cols, {total_missing} missing, {len(numeric_cols)} numeric."

        except Exception as e:
            result["summary"] = f"Auto-profile error: {str(e)}"

    # ── Graphical Summary (Minitab-style) ────────────────────────────────
    elif analysis_id == "graphical_summary":
        try:
            from scipy import stats as sp_stats

            conf_level = config.get("confidence", 0.95)
            selected = config.get("vars", [])
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not selected:
                selected = numeric_cols
            selected = [c for c in selected if c in numeric_cols]

            if not selected:
                result["summary"] = "No numeric columns selected or available for Graphical Summary."
            else:
                all_summaries = []
                for col in selected:
                    vals = df[col].dropna().values
                    n = len(vals)
                    n_star = int(df[col].isnull().sum())

                    if n < 3:
                        all_summaries.append(f"<<COLOR:title>>{col}<<\/COLOR>>: insufficient data (N={n})")
                        continue

                    # Descriptive stats
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1))
                    var_val = std_val ** 2
                    se = std_val / np.sqrt(n)
                    skew_val = float(sp_stats.skew(vals, bias=False))
                    kurt_val = float(sp_stats.kurtosis(vals, bias=False))
                    q1 = float(np.percentile(vals, 25))
                    median_val = float(np.median(vals))
                    q3 = float(np.percentile(vals, 75))
                    min_val = float(np.min(vals))
                    max_val = float(np.max(vals))

                    # Anderson-Darling test
                    ad_result = sp_stats.anderson(vals, dist='norm')
                    ad_stat = ad_result.statistic
                    # Use 5% significance level (index 2 in critical_values)
                    ad_crit = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else ad_result.critical_values[-1]
                    ad_sig = ad_result.significance_level[2] if len(ad_result.significance_level) > 2 else ad_result.significance_level[-1]
                    ad_pass = ad_stat < ad_crit
                    ad_verdict = "<<COLOR:success>>Yes (fail to reject H₀)<<\/COLOR>>" if ad_pass else "<<COLOR:danger>>No (reject H₀)<<\/COLOR>>"

                    # CI for mean (t-interval)
                    ci_mean = sp_stats.t.interval(conf_level, df=n - 1, loc=mean_val, scale=se)

                    # CI for median (nonparametric sign-test inversion)
                    alpha = 1 - conf_level
                    from scipy.stats import binom
                    j = 0
                    for k_idx in range(n):
                        if binom.cdf(k_idx, n, 0.5) >= alpha / 2:
                            j = k_idx
                            break
                    sorted_vals = np.sort(vals)
                    ci_median_lo = float(sorted_vals[j]) if j < n else float(sorted_vals[0])
                    ci_median_hi = float(sorted_vals[n - 1 - j]) if (n - 1 - j) >= 0 else float(sorted_vals[-1])

                    # CI for StDev (chi-square)
                    chi2_lo = sp_stats.chi2.ppf((1 + conf_level) / 2, n - 1)
                    chi2_hi = sp_stats.chi2.ppf((1 - conf_level) / 2, n - 1)
                    ci_std_lo = np.sqrt((n - 1) * var_val / chi2_lo)
                    ci_std_hi = np.sqrt((n - 1) * var_val / chi2_hi)

                    pct_str = f"{conf_level*100:.0f}%"

                    # Summary text
                    summ = f"""<<COLOR:title>>{'═'*50}
{col}
{'═'*50}<<\/COLOR>>

<<COLOR:accent>>Anderson-Darling Normality Test<<\/COLOR>>
  A² = {ad_stat:.4f}    Critical ({ad_sig:.0f}%) = {ad_crit:.4f}
  Normal? {ad_verdict}

<<COLOR:accent>>Descriptive Statistics<<\/COLOR>>
  N = {n}    N* = {n_star}
  Mean     = {mean_val:.6g}       StDev    = {std_val:.6g}
  Variance = {var_val:.6g}       Skewness = {skew_val:.4f}
  Kurtosis = {kurt_val:.4f}
  Minimum  = {min_val:.6g}       Q1       = {q1:.6g}
  Median   = {median_val:.6g}       Q3       = {q3:.6g}
  Maximum  = {max_val:.6g}

<<COLOR:accent>>Confidence Intervals ({pct_str})<<\/COLOR>>
  Mean:   ({ci_mean[0]:.6g}, {ci_mean[1]:.6g})
  Median: ({ci_median_lo:.6g}, {ci_median_hi:.6g})
  StDev:  ({ci_std_lo:.6g}, {ci_std_hi:.6g})"""
                    all_summaries.append(summ)

                    # ── Plotly figure: 3-row subplot ──
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=3, cols=1, row_heights=[0.55, 0.20, 0.25],
                        shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{col} — Histogram + Normal Fit", "Boxplot", f"{pct_str} Confidence Intervals"]
                    )

                    # Row 1: Histogram + normal curve
                    nbins = min(max(int(np.sqrt(n)), 10), 50)
                    counts, bin_edges = np.histogram(vals, bins=nbins)
                    bin_width = bin_edges[1] - bin_edges[0]

                    fig.add_trace(go.Histogram(
                        x=vals.tolist(), nbinsx=nbins,
                        marker=dict(color="rgba(74,144,217,0.6)", line=dict(color="rgba(74,144,217,1)", width=1)),
                        name="Data", showlegend=False
                    ), row=1, col=1)

                    # Normal PDF overlay scaled to histogram
                    x_fit = np.linspace(min_val - 0.5 * std_val, max_val + 0.5 * std_val, 200)
                    y_fit = sp_stats.norm.pdf(x_fit, mean_val, std_val) * n * bin_width
                    fig.add_trace(go.Scatter(
                        x=x_fit.tolist(), y=y_fit.tolist(), mode="lines",
                        line=dict(color="rgba(232,71,71,0.9)", width=2),
                        name="Normal Fit", showlegend=False
                    ), row=1, col=1)

                    # Row 2: Boxplot
                    fig.add_trace(go.Box(
                        x=vals.tolist(), orientation="h",
                        marker=dict(color="rgba(74,144,217,0.7)"),
                        line=dict(color="rgba(74,144,217,1)"),
                        fillcolor="rgba(74,144,217,0.3)",
                        name="Box", showlegend=False
                    ), row=2, col=1)

                    # Row 3: CI bars (mean and median)
                    fig.add_trace(go.Scatter(
                        x=[ci_mean[0], mean_val, ci_mean[1]], y=["Mean", "Mean", "Mean"],
                        mode="lines+markers",
                        marker=dict(size=[8, 12, 8], color=["rgba(232,71,71,0.8)", "rgba(232,71,71,1)", "rgba(232,71,71,0.8)"],
                                    symbol=["line-ns", "diamond", "line-ns"]),
                        line=dict(color="rgba(232,71,71,0.8)", width=2),
                        name="Mean CI", showlegend=False
                    ), row=3, col=1)

                    fig.add_trace(go.Scatter(
                        x=[ci_median_lo, median_val, ci_median_hi], y=["Median", "Median", "Median"],
                        mode="lines+markers",
                        marker=dict(size=[8, 12, 8], color=["rgba(74,144,217,0.8)", "rgba(74,144,217,1)", "rgba(74,144,217,0.8)"],
                                    symbol=["line-ns", "diamond", "line-ns"]),
                        line=dict(color="rgba(74,144,217,0.8)", width=2),
                        name="Median CI", showlegend=False
                    ), row=3, col=1)

                    fig.update_layout(height=520, template="plotly_white",
                                      margin=dict(t=40, b=30, l=60, r=20))
                    fig.update_yaxes(showticklabels=False, row=2, col=1)

                    # Convert to JSON-serializable dict
                    fig_dict = fig.to_dict()
                    result["plots"].append({
                        "data": [t for t in fig_dict["data"]],
                        "layout": fig_dict["layout"]
                    })

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
                result["guide_observation"] = f"Graphical summary for {len(selected)} variable(s) at {conf_level*100:.0f}% confidence."

        except Exception as e:
            import traceback
            result["summary"] = f"Graphical Summary error: {str(e)}\n{traceback.format_exc()}"

    # ── Missing Data Analysis ─────────────────────────────────────────────
    elif analysis_id == "missing_data_analysis":
        try:
            n_rows, n_cols = df.shape
            miss_count = df.isnull().sum()
            miss_pct = (miss_count / n_rows * 100).round(2)
            cols_with_missing = miss_count[miss_count > 0].sort_values(ascending=False)

            # Row completeness
            row_completeness = ((~df.isnull()).sum(axis=1) / n_cols * 100)
            complete_rows = int((row_completeness == 100).sum())

            # Missing patterns
            miss_indicator = df.isnull().astype(int)
            pattern_strs = miss_indicator.apply(lambda r: "".join(str(v) for v in r), axis=1)
            pattern_counts = pattern_strs.value_counts()
            n_patterns = len(pattern_counts)

            # Pattern table
            pattern_rows = []
            for pat, cnt in pattern_counts.head(15).items():
                cols_missing = [df.columns[i] for i, v in enumerate(pat) if v == '1']
                desc = ", ".join(cols_missing) if cols_missing else "(complete)"
                pattern_rows.append(f"  {cnt:>6} rows ({cnt/n_rows*100:.1f}%): {desc}")
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
                            prob *= p_miss if v == '1' else (1 - p_miss)
                        expected_probs.append(prob)
                    expected_counts = np.array(expected_probs) * n_rows
                    # Filter out zero-expected
                    mask = expected_counts > 0.5
                    if mask.sum() >= 2:
                        from scipy.stats import chi2
                        obs = observed_counts[mask]
                        exp = expected_counts[mask]
                        chi2_stat = float(np.sum((obs - exp)**2 / exp))
                        dof = int(mask.sum() - 1)
                        p_val = float(1 - chi2.cdf(chi2_stat, dof))
                        conclusion = "Data appears MCAR (p >= 0.05)" if p_val >= 0.05 else "Data may NOT be MCAR (p < 0.05)"
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
                    for j in range(i+1, len(idx_list)):
                        pairs.append((idx_list[i], idx_list[j], miss_corr.loc[idx_list[i], idx_list[j]]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, v in pairs[:5]]
                    miss_corr_text = "\nMissing Correlation (top pairs):\n" + "\n".join(lines)

            summary_lines = [f"MISSING DATA ANALYSIS", "=" * 50]
            summary_lines.append(f"Dataset: {n_rows} rows x {n_cols} columns")
            summary_lines.append(f"Total missing: {int(miss_count.sum())} / {n_rows * n_cols} ({miss_count.sum() / (n_rows * n_cols) * 100:.1f}%)")
            summary_lines.append(f"Complete rows: {complete_rows} / {n_rows} ({complete_rows/n_rows*100:.1f}%)")
            summary_lines.append(f"\nColumns with missing data ({len(cols_with_missing)}):")
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
                    cols_miss = sum(1 for v in pat if v == '1')
                    y_labels.append(f"{cnt} rows ({cols_miss} missing)")
                result["plots"].append({
                    "data": [{"type": "heatmap", "z": z_data, "x": df.columns.tolist(), "y": y_labels,
                              "colorscale": [[0, "rgba(74,144,217,0.15)"], [1, "rgba(232,71,71,0.7)"]],
                              "showscale": False, "hovertemplate": "%{x}: %{z}<extra>0=present, 1=missing</extra>"}],
                    "layout": {"title": "Missing Data Patterns", "height": max(300, len(z_data) * 25 + 100),
                               "template": "plotly_white", "xaxis": {"tickangle": -45}, "margin": {"b": 100, "l": 150}}
                })

            # Plot 2: Row completeness histogram
            result["plots"].append({
                "data": [{"type": "histogram", "x": row_completeness.tolist(), "nbinsx": 20,
                          "marker": {"color": "rgba(74,159,110,0.6)"}}],
                "layout": {"title": "Row Completeness Distribution", "height": 280, "template": "plotly_white",
                           "xaxis": {"title": "% Complete", "range": [0, 105]}, "yaxis": {"title": "Row Count"}}
            })

            result["guide_observation"] = f"Missing data: {int(miss_count.sum())} cells across {len(cols_with_missing)} columns. {n_patterns} unique patterns."

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
            columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

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
                    col_results["IQR"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                          "bounds": f"[{lower:.4g}, {upper:.4g}]"}
                    consensus += mask.astype(int).values

                if "zscore" in methods:
                    z = np.abs((df[col] - vals.mean()) / vals.std()) if vals.std() > 0 else pd.Series(0, index=df.index)
                    mask = z > z_thresh
                    col_results["Z-score"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                              "threshold": z_thresh}
                    consensus += mask.astype(int).values

                if "modified_zscore" in methods:
                    median = vals.median()
                    mad = np.median(np.abs(vals - median))
                    if mad > 0:
                        modified_z = 0.6745 * np.abs(df[col] - median) / mad
                        mask = modified_z > 3.5
                    else:
                        mask = pd.Series(False, index=df.index)
                    col_results["Modified Z"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                                 "threshold": 3.5}
                    consensus += mask.astype(int).values

                if "mahalanobis" in methods and len(columns) >= 2:
                    try:
                        from scipy.spatial.distance import mahalanobis as mah_dist
                        sub = df[columns].dropna()
                        if len(sub) > len(columns):
                            cov = np.cov(sub.T)
                            cov_inv = np.linalg.inv(cov + np.eye(len(columns)) * 1e-6)
                            mean = sub.mean().values
                            dists = sub.apply(lambda r: mah_dist(r.values, mean, cov_inv), axis=1)
                            from scipy.stats import chi2 as chi2_dist
                            threshold = chi2_dist.ppf(0.975, df=len(columns))
                            mask_full = dists > np.sqrt(threshold)
                            col_results["Mahalanobis"] = {"count": int(mask_full.sum()),
                                                          "pct": round(mask_full.sum() / len(sub) * 100, 1),
                                                          "threshold": f"chi2(p=0.975, df={len(columns)})"}
                    except Exception:
                        col_results["Mahalanobis"] = {"count": 0, "pct": 0, "error": "Could not compute"}

                all_results[col] = col_results

            # Build summary
            summary_lines = ["OUTLIER ANALYSIS", "=" * 50]
            summary_lines.append(f"Columns: {', '.join(columns)}")
            summary_lines.append(f"Methods: {', '.join(methods)}")
            summary_lines.append(f"Rows: {len(df)}\n")

            for col, methods_res in all_results.items():
                summary_lines.append(f"{col}:")
                for method, info in methods_res.items():
                    summary_lines.append(f"  {method:<18} {info['count']:>5} outliers ({info['pct']}%)")
                summary_lines.append("")

            # Consensus
            n_methods_used = len([m for m in methods if m != "mahalanobis" or len(columns) >= 2])
            if n_methods_used >= 2:
                flagged_all = int((consensus >= n_methods_used * len(columns)).sum()) if n_methods_used > 0 else 0
                flagged_majority = int((consensus >= max(1, n_methods_used * len(columns) // 2)).sum())
                summary_lines.append(f"Consensus:")
                summary_lines.append(f"  Flagged by majority of methods: {flagged_majority} rows")

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
                result["plots"].append({
                    "data": [{"type": "box", "y": vals, "name": col, "boxpoints": "outliers",
                              "marker": {"color": "rgba(74,144,217,0.6)", "outliercolor": "rgba(232,71,71,0.8)"},
                              "line": {"color": "rgba(74,144,217,0.8)"}}],
                    "layout": {"title": f"Outlier Detection: {col}", "height": 300, "template": "plotly_white",
                               "yaxis": {"title": col}}
                })

            result["guide_observation"] = f"Outlier analysis on {len(columns)} columns with {len(methods)} methods."

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
            n_dup_groups = int(df[duplicated_mask].groupby(check_cols).ngroups) if n_dup_rows > 0 else 0
            first_dup_mask = df.duplicated(subset=check_cols, keep="first")
            n_extra = int(first_dup_mask.sum())

            summary_lines = ["DUPLICATE ANALYSIS", "=" * 50]
            summary_lines.append(f"Mode: {'Exact (all columns)' if mode == 'exact' else 'Subset (' + ', '.join(check_cols) + ')'}")
            summary_lines.append(f"Total rows: {len(df)}")
            summary_lines.append(f"Unique rows: {len(df) - n_extra}")
            summary_lines.append(f"Duplicate rows: {n_dup_rows} ({n_dup_rows/len(df)*100:.1f}%)")
            summary_lines.append(f"Duplicate groups: {n_dup_groups}")
            summary_lines.append(f"Extra copies (removable): {n_extra}")

            # Show top duplicate groups
            if n_dup_rows > 0:
                dup_df = df[duplicated_mask].copy()
                group_sizes = dup_df.groupby(check_cols).size().sort_values(ascending=False)
                summary_lines.append(f"\nLargest duplicate groups:")
                for i, (vals, cnt) in enumerate(group_sizes.head(10).items()):
                    if isinstance(vals, tuple):
                        desc = ", ".join(f"{c}={v}" for c, v in zip(check_cols, vals))
                    else:
                        desc = f"{check_cols[0]}={vals}"
                    summary_lines.append(f"  Group {i+1}: {cnt} copies — {desc[:80]}")

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
                result["tables"] = [{"title": "Sample Duplicate Rows", "html": table_html}]

            # Plot: duplicate group size histogram
            if n_dup_groups > 0:
                group_sizes_list = dup_df.groupby(check_cols).size().tolist()
                result["plots"].append({
                    "data": [{"type": "histogram", "x": group_sizes_list,
                              "marker": {"color": "rgba(232,149,71,0.6)"}}],
                    "layout": {"title": "Duplicate Group Sizes", "height": 280, "template": "plotly_white",
                               "xaxis": {"title": "Copies per Group"}, "yaxis": {"title": "Number of Groups"}}
                })

            result["guide_observation"] = f"Duplicate analysis ({mode}): {n_dup_rows} duplicate rows in {n_dup_groups} groups."

        except Exception as e:
            result["summary"] = f"Duplicate analysis error: {str(e)}"

    # ── Meta-Analysis ─────────────────────────────────────────────────────
    elif analysis_id == "meta_analysis":
        try:
            mode = config.get("mode", "precomputed")
            subgroup_col = config.get("subgroup_col", "")

            if mode == "precomputed":
                effect_col = config.get("effect_col", "")
                se_col = config.get("se_col", "")
                study_col = config.get("study_col", "")
                if not all([effect_col, se_col, study_col]):
                    result["summary"] = "Please specify effect size, SE, and study label columns."
                    return result
                effects = df[effect_col].dropna().values.astype(float)
                ses = df[se_col].dropna().values.astype(float)
                studies = df[study_col].values[:len(effects)]
            else:
                # Raw mode: compute Cohen's d
                m1c, s1c, n1c = config.get("mean1_col", ""), config.get("sd1_col", ""), config.get("n1_col", "")
                m2c, s2c, n2c = config.get("mean2_col", ""), config.get("sd2_col", ""), config.get("n2_col", "")
                study_col = config.get("study_col", "")
                if not all([m1c, s1c, n1c, m2c, s2c, n2c]):
                    result["summary"] = "Please specify all 6 raw data columns (mean, SD, n for each group)."
                    return result
                m1 = df[m1c].values.astype(float)
                s1 = df[s1c].values.astype(float)
                n1 = df[n1c].values.astype(float)
                m2 = df[m2c].values.astype(float)
                s2 = df[s2c].values.astype(float)
                n2 = df[n2c].values.astype(float)
                sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                effects = (m1 - m2) / sp
                ses = np.sqrt(1/n1 + 1/n2 + effects**2 / (2 * (n1 + n2)))
                studies = df[study_col].values[:len(effects)] if study_col else np.arange(1, len(effects) + 1)

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
            Q = float(np.sum(w * (effects - fe_est)**2))
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

            interpretation = "Low heterogeneity" if I2 < 25 else ("Moderate heterogeneity" if I2 < 75 else "High heterogeneity")
            if I2 < 40:
                interpretation += " — fixed and random effects models give similar results."
            else:
                interpretation += " — random effects model is more appropriate."

            summary = f"""META-ANALYSIS
{'='*50}
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
                {"type": "scatter", "x": list(effects), "y": forest_y,
                 "mode": "markers", "name": "Studies",
                 "marker": {"size": [max(4, min(16, float(w_re[i]/np.max(w_re)*16))) for i in range(k)], "color": "rgba(74,144,217,0.8)"},
                 "error_x": {"type": "data", "symmetric": False,
                             "array": [ci_highs[i] - effects[i] for i in range(k)],
                             "arrayminus": [effects[i] - ci_lows[i] for i in range(k)],
                             "color": "rgba(74,144,217,0.5)", "thickness": 2},
                 "text": study_labels, "hovertemplate": "%{text}: %{x:.4f}<extra></extra>"},
                # Pooled diamond (RE)
                {"type": "scatter", "x": [re_ci_lo, re_est, re_ci_hi, re_est, re_ci_lo],
                 "y": [0, -0.3, 0, 0.3, 0], "mode": "lines", "fill": "toself",
                 "fillcolor": "rgba(232,71,71,0.3)", "line": {"color": "rgba(232,71,71,0.8)"},
                 "name": "Pooled (RE)", "hoverinfo": "skip"},
                # Zero line
                {"type": "scatter", "x": [0, 0], "y": [-1, k + 1], "mode": "lines",
                 "line": {"color": "gray", "dash": "dash", "width": 1}, "showlegend": False, "hoverinfo": "skip"}
            ]
            result["plots"].append({
                "data": forest_data,
                "layout": {"title": "Forest Plot", "height": max(350, k * 30 + 120), "template": "plotly_white",
                           "yaxis": {"tickvals": forest_y + [0], "ticktext": study_labels + ["Pooled (RE)"],
                                     "range": [-1, k + 1]},
                           "xaxis": {"title": "Effect Size", "zeroline": True},
                           "showlegend": False, "margin": {"l": 120}}
            })

            # Funnel plot
            result["plots"].append({
                "data": [
                    {"type": "scatter", "x": list(effects), "y": list(ses), "mode": "markers",
                     "marker": {"size": 8, "color": "rgba(74,144,217,0.7)"}, "name": "Studies",
                     "text": study_labels, "hovertemplate": "%{text}: effect=%{x:.4f}, SE=%{y:.4f}<extra></extra>"},
                    # Funnel lines
                    {"type": "scatter", "x": [re_est - 1.96 * max(ses), re_est, re_est + 1.96 * max(ses)],
                     "y": [max(ses), 0, max(ses)], "mode": "lines",
                     "line": {"color": "gray", "dash": "dash"}, "name": "95% CI", "hoverinfo": "skip"},
                    {"type": "scatter", "x": [re_est, re_est], "y": [0, max(ses) * 1.1], "mode": "lines",
                     "line": {"color": "rgba(232,71,71,0.5)", "dash": "dot"}, "name": "Pooled", "hoverinfo": "skip"}
                ],
                "layout": {"title": "Funnel Plot", "height": 350, "template": "plotly_white",
                           "xaxis": {"title": "Effect Size"}, "yaxis": {"title": "Standard Error", "autorange": "reversed"}}
            })

            result["guide_observation"] = f"Meta-analysis of {k} studies. RE pooled: {re_est:.4f}. I²={I2:.1f}%. {interpretation}"

        except Exception as e:
            result["summary"] = f"Meta-analysis error: {str(e)}"

    # ── Effect Size Calculator ────────────────────────────────────────────
    elif analysis_id == "effect_size_calculator":
        try:
            effect_type = config.get("effect_type", "cohens_d")
            results_list = []

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
                    se = np.sqrt(1/n1 + 1/n2 + d**2 / (2 * (n1 + n2)))
                    name = "Cohen's d"
                elif effect_type == "hedges_g":
                    d_raw = (m1 - m2) / sp if sp > 0 else 0
                    J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                    d = d_raw * J
                    se = np.sqrt(1/n1 + 1/n2 + d**2 / (2 * (n1 + n2))) * J
                    name = "Hedges' g"
                else:  # glass_delta
                    d = (m1 - m2) / s2 if s2 > 0 else 0
                    se = np.sqrt(1/n1 + d**2 / (2 * n2))
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
{'='*50}
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
  Direction: {'Group 1 > Group 2' if d > 0 else 'Group 2 > Group 1' if d < 0 else 'No difference'}"""

                # Plot
                result["plots"].append({
                    "data": [
                        {"type": "scatter", "x": [d], "y": [name], "mode": "markers",
                         "marker": {"size": 14, "color": "rgba(74,144,217,0.8)"},
                         "error_x": {"type": "data", "symmetric": False,
                                     "array": [ci_hi - d], "arrayminus": [d - ci_lo],
                                     "color": "rgba(74,144,217,0.5)", "thickness": 3}},
                        {"type": "scatter", "x": [0, 0], "y": [-0.5, 1.5], "mode": "lines",
                         "line": {"color": "gray", "dash": "dash"}, "showlegend": False}
                    ],
                    "layout": {"title": f"{name} with 95% CI", "height": 200, "template": "plotly_white",
                               "xaxis": {"title": "Effect Size"}, "showlegend": False}
                })

            elif effect_type in ("odds_ratio", "risk_ratio"):
                a = int(config.get("a", 0))
                b = int(config.get("b", 0))
                c = int(config.get("c", 0))
                dd = int(config.get("d", 0))

                if effect_type == "odds_ratio":
                    if b * c > 0:
                        es = (a * dd) / (b * c)
                        se_ln = np.sqrt(1/max(a,0.5) + 1/max(b,0.5) + 1/max(c,0.5) + 1/max(dd,0.5))
                    else:
                        es, se_ln = 0, 0
                    name = "Odds Ratio"
                else:
                    r1 = a / (a + b) if (a + b) > 0 else 0
                    r2 = c / (c + dd) if (c + dd) > 0 else 0
                    es = r1 / r2 if r2 > 0 else 0
                    se_ln = np.sqrt(1/max(a,0.5) - 1/max(a+b,1) + 1/max(c,0.5) - 1/max(c+dd,1))
                    name = "Risk Ratio"

                ci_lo = es * np.exp(-1.96 * se_ln) if es > 0 else 0
                ci_hi = es * np.exp(1.96 * se_ln) if es > 0 else 0

                summary = f"""EFFECT SIZE CALCULATOR
{'='*50}
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
  {'No association (= 1.0)' if abs(es - 1.0) < 0.01 else ('Positive association' if es > 1 else 'Negative association')}"""

                result["plots"].append({
                    "data": [
                        {"type": "scatter", "x": [es], "y": [name], "mode": "markers",
                         "marker": {"size": 14, "color": "rgba(232,149,71,0.8)"},
                         "error_x": {"type": "data", "symmetric": False,
                                     "array": [ci_hi - es], "arrayminus": [es - ci_lo],
                                     "color": "rgba(232,149,71,0.5)", "thickness": 3}},
                        {"type": "scatter", "x": [1, 1], "y": [-0.5, 1.5], "mode": "lines",
                         "line": {"color": "gray", "dash": "dash"}, "showlegend": False}
                    ],
                    "layout": {"title": f"{name} with 95% CI", "height": 200, "template": "plotly_white",
                               "xaxis": {"title": name}, "showlegend": False}
                })
            else:
                summary = f"Unknown effect size type: {effect_type}"

            result["summary"] = summary
            result["guide_observation"] = f"Effect size calculated: {effect_type}"

        except Exception as e:
            result["summary"] = f"Effect size error: {str(e)}"

    return result


