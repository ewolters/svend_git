"""DSW Statistical Analysis — post-hoc tests (Tukey, Dunnett, Games-Howell, etc.)."""

import logging

import numpy as np

from ..common import (
    _narrative,
)

logger = logging.getLogger(__name__)


def _run_posthoc(analysis_id, df, config):
    """Run posthoc analysis."""
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "main_effects":
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
        summary += "<<COLOR:title>>MAIN EFFECTS PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
        summary += f"<<COLOR:highlight>>Grand Mean:<</COLOR>> {y.mean():.4f}\n\n"

        summary += (
            "<<COLOR:accent>>── Effect Sizes (deviation from grand mean) ──<</COLOR>>\n"
        )

        colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"]

        for i, factor in enumerate(factors):
            # Calculate means for each level
            factor_means = df.groupby(factor)[response].mean()

            summary += f"\n  <<COLOR:accent>>{factor}:<</COLOR>>\n"
            for level, mean_val in factor_means.items():
                effect = mean_val - y.mean()
                direction = "+" if effect > 0 else ""
                summary += (
                    f"    {level}: {mean_val:.4f} (effect: {direction}{effect:.4f})\n"
                )

            # Create main effects plot
            levels = [str(lvl) for lvl in factor_means.index.tolist()]
            means = factor_means.values.tolist()

            result["plots"].append(
                {
                    "title": f"Main Effect: {factor}",
                    "data": [
                        {
                            "type": "scatter",
                            "x": levels,
                            "y": means,
                            "mode": "lines+markers",
                            "marker": {"color": colors[i % len(colors)], "size": 10},
                            "line": {"color": colors[i % len(colors)], "width": 2},
                        },
                        {
                            "type": "scatter",
                            "x": levels,
                            "y": [y.mean()] * len(levels),
                            "mode": "lines",
                            "line": {"color": "#9aaa9a", "dash": "dash"},
                            "name": "Grand Mean",
                        },
                    ],
                    "layout": {
                        "height": 280,
                        "xaxis": {"title": factor},
                        "yaxis": {"title": f"Mean of {response}"},
                    },
                }
            )

        summary += "\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += "  Steeper slopes indicate stronger effects.\n"
        summary += "  Flat lines suggest the factor has little impact.\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Main effects plot for {len(factors)} factor(s). Check for steep slopes indicating strong effects."
        )

        # Narrative — find factor with largest effect range
        effect_ranges = []
        for factor in factors:
            fmeans = df.groupby(factor)[response].mean()
            effect_ranges.append((factor, float(fmeans.max() - fmeans.min())))
        effect_ranges.sort(key=lambda x: x[1], reverse=True)
        top_factor, top_range = effect_ranges[0]
        grand_mean = float(y.mean())
        verdict = f"Main Effects — <strong>{top_factor}</strong> has the largest effect (range = {top_range:.4f})"
        body = (
            f"Across {len(factors)} factor{'s' if len(factors) > 1 else ''}, <strong>{top_factor}</strong> "
            f"produces the widest swing in mean {response} ({top_range:.4f}). Grand mean = {grand_mean:.4f}."
        )
        if len(effect_ranges) > 1:
            body += (
                f" Second: {effect_ranges[1][0]} (range = {effect_ranges[1][1]:.4f})."
            )
        nxt = "Steep slopes = strong effects, flat lines = negligible. Run an interaction plot to check whether factors act independently."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nxt,
            chart_guidance="Each panel shows how one factor's levels shift the response mean. The dashed line is the grand mean.",
        )

    elif analysis_id == "interaction":
        """
        Interaction Plot - shows how factors interact.
        Non-parallel lines indicate interactions.
        """
        response = config.get("response")
        factor1 = config.get("factor1")
        factor2 = config.get("factor2")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>INTERACTION PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>X-axis factor:<</COLOR>> {factor1}\n"
        summary += f"<<COLOR:highlight>>Trace factor:<</COLOR>> {factor2}\n\n"

        # Calculate interaction means
        interaction_means = df.groupby([factor1, factor2])[response].mean().unstack()

        summary += "<<COLOR:accent>>── Cell Means ──<</COLOR>>\n"
        summary += interaction_means.to_string() + "\n\n"

        # Check for interaction (compare slopes)
        levels1 = interaction_means.index.tolist()
        levels2 = interaction_means.columns.tolist()

        colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e8c547"]

        plot_data = []
        for i, lev2 in enumerate(levels2):
            means = interaction_means[lev2].tolist()
            plot_data.append(
                {
                    "type": "scatter",
                    "x": [str(lvl) for lvl in levels1],
                    "y": means,
                    "mode": "lines+markers",
                    "name": str(lev2),
                    "marker": {"color": colors[i % len(colors)], "size": 8},
                    "line": {"color": colors[i % len(colors)], "width": 2},
                }
            )

        result["plots"].append(
            {
                "title": f"Interaction: {factor1} × {factor2}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": factor1},
                    "yaxis": {"title": f"Mean of {response}"},
                    "legend": {"title": {"text": factor2}},
                },
            }
        )

        # Simple interaction detection (check if lines are parallel)
        if len(levels2) >= 2 and len(levels1) >= 2:
            slopes = []
            for lev2 in levels2:
                vals = interaction_means[lev2].values
                slope = (vals[-1] - vals[0]) / (len(vals) - 1) if len(vals) > 1 else 0
                slopes.append(slope)

            slope_diff = max(slopes) - min(slopes)
            has_interaction = (
                slope_diff > 0.1 * abs(np.mean(slopes))
                if np.mean(slopes) != 0
                else slope_diff > 0.1
            )

            summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
            if has_interaction:
                summary += "<<COLOR:warning>>POTENTIAL INTERACTION DETECTED<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines are not parallel, suggesting {factor1} and {factor2} interact.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The effect of {factor1} depends on the level of {factor2}.<</COLOR>>\n"
            else:
                summary += "<<COLOR:good>>NO STRONG INTERACTION<</COLOR>>\n"
                summary += "<<COLOR:text>>Lines appear roughly parallel.<</COLOR>>\n"
                summary += "<<COLOR:text>>Factors may act independently.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Interaction plot for {factor1} × {factor2}. "
            + (
                "Non-parallel lines suggest interaction."
                if "has_interaction" in dir() and has_interaction
                else "Check for parallel lines."
            )
        )

        # Narrative
        _has_ix = "has_interaction" in dir() and has_interaction
        if _has_ix:
            verdict = f"Interaction detected — {factor1} × {factor2}"
            body = (
                f"The effect of <strong>{factor1}</strong> on {response} depends on the level of <strong>{factor2}</strong> "
                f"(non-parallel lines). This means you cannot optimize {factor1} without considering {factor2}."
            )
            nxt = "Factors interact — optimize them jointly, not independently. Consider a full factorial DOE to quantify the interaction effect."
        else:
            verdict = f"No strong interaction — {factor1} × {factor2}"
            body = (
                f"Lines are approximately parallel, suggesting <strong>{factor1}</strong> and <strong>{factor2}</strong> "
                f"act independently on {response}. Each factor's effect is consistent across levels of the other."
            )
            nxt = "Factors appear independent — you can optimize each separately. Confirm with ANOVA interaction term if needed."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nxt,
            chart_guidance="Parallel lines = no interaction (factors act independently). Crossing or diverging lines = interaction (combined effect differs from individual effects).",
        )

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
            pairs.append(
                {
                    "group1": str(row[0]),
                    "group2": str(row[1]),
                    "meandiff": float(row[2]),
                    "p_adj": float(row[3]),
                    "lower": float(row[4]),
                    "upper": float(row[5]),
                    "reject": bool(row[6]),
                }
            )

        n_sig = sum(1 for p in pairs if p["reject"])
        n_total = len(pairs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>TUKEY'S HSD POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += "<<COLOR:accent>>── Pairwise Comparisons ──<</COLOR>>\n"
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
        traces.append(
            {
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
                    "color": "#5a6a5a",
                },
                "showlegend": False,
            }
        )
        result["plots"].append(
            {
                "title": "Tukey HSD — Pairwise Differences with CIs",
                "data": traces,
                "layout": {
                    "height": max(250, 40 * len(pairs)),
                    "xaxis": {
                        "title": "Mean Difference",
                        "zeroline": True,
                        "zerolinecolor": "#e89547",
                        "zerolinewidth": 2,
                    },
                    "yaxis": {"automargin": True},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": -0.5,
                            "y1": len(pairs) - 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # Group means with SE error bars
        group_stats = data.groupby(factor)[response].agg(["mean", "std", "count"])
        group_stats["se"] = group_stats["std"] / np.sqrt(group_stats["count"])
        result["plots"].append(
            {
                "title": f"Group Means — {response} by {factor}",
                "data": [
                    {
                        "x": [str(g) for g in group_stats.index],
                        "y": group_stats["mean"].tolist(),
                        "error_y": {
                            "type": "data",
                            "array": group_stats["se"].tolist(),
                            "visible": True,
                            "color": "#5a6a5a",
                        },
                        "type": "bar",
                        "marker": {"color": "#4a9f6e", "opacity": 0.8},
                    }
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": factor},
                    "yaxis": {"title": f"Mean {response}"},
                },
            }
        )

        result["guide_observation"] = (
            f"Tukey HSD: {n_sig}/{n_total} pairwise comparisons significant at α={alpha}."
        )
        result["statistics"] = {
            "pairs": pairs,
            "n_significant": n_sig,
            "n_comparisons": n_total,
            "alpha": alpha,
        }
        verdict = f"Tukey HSD: {n_sig} of {n_total} pairs differ significantly"
        body = (
            f"<strong>{n_sig}</strong> pairwise comparisons are significant at \u03b1 = {alpha} (family-wise error controlled)."
            if n_sig > 0
            else f"No pairwise differences are significant at \u03b1 = {alpha}."
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Tukey HSD controls the family-wise error rate. Pairs with non-overlapping CIs differ significantly.",
        )

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
                p_val = (
                    float(res.pvalue[0])
                    if hasattr(res.pvalue, "__len__")
                    else float(res.pvalue)
                )
                stat_val = (
                    float(res.statistic[0])
                    if hasattr(res.statistic, "__len__")
                    else float(res.statistic)
                )
            except (ImportError, AttributeError):
                # Fallback: Welch t-test with Bonferroni correction
                stat_val, p_raw = stats.ttest_ind(
                    treat_data, control_data, equal_var=False
                )
                stat_val = float(stat_val)
                p_val = min(float(p_raw) * len(treatments), 1.0)

            mean_diff = float(np.mean(treat_data) - np.mean(control_data))
            comparisons.append(
                {
                    "treatment": str(treat),
                    "control": str(control),
                    "mean_diff": mean_diff,
                    "statistic": stat_val,
                    "p_value": p_val,
                    "reject": p_val < alpha,
                }
            )

        n_sig = sum(1 for c in comparisons if c["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>DUNNETT'S TEST (vs Control)<</COLOR>>\n"
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
            se = float(
                np.sqrt(
                    np.var(treat_vals, ddof=1) / len(treat_vals)
                    + np.var(control_data, ddof=1) / len(control_data)
                )
            )
            se_bars.append(se)
        result["plots"].append(
            {
                "title": f"Dunnett's Test — Difference from Control ({control})",
                "data": [
                    {
                        "type": "bar",
                        "x": [c["treatment"] for c in comparisons],
                        "y": [c["mean_diff"] for c in comparisons],
                        "marker": {
                            "color": [
                                "#4a9f6e" if c["reject"] else "rgba(90,106,90,0.5)"
                                for c in comparisons
                            ],
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "error_y": {
                            "type": "data",
                            "array": se_bars,
                            "visible": True,
                            "color": "rgba(200,200,200,0.7)",
                        },
                        "text": [f"p={c['p_value']:.4f}" for c in comparisons],
                        "textposition": "outside",
                    }
                ],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": f"Difference from {control}"},
                },
            }
        )

        result["guide_observation"] = (
            f"Dunnett's test vs {control}: {n_sig}/{len(comparisons)} treatments differ."
        )
        result["statistics"] = {"comparisons": comparisons, "control": str(control)}
        verdict = f"Dunnett's: {n_sig} of {len(comparisons)} treatments differ from control ({control})"
        body = (
            f"Comparing all treatments against the control group <strong>{control}</strong>."
            + (
                f" {n_sig} treatment{'s' if n_sig > 1 else ''} show{'s' if n_sig == 1 else ''} a significant difference."
                if n_sig
                else " No treatments differ from control."
            )
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Dunnett's test is more powerful than Tukey when comparing only to a control group.",
        )

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
            group_stats[lev] = {
                "mean": np.mean(g),
                "var": np.var(g, ddof=1),
                "n": len(g),
            }

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
            denom = (s1["var"] / s1["n"]) ** 2 / (s1["n"] - 1) + (
                s2["var"] / s2["n"]
            ) ** 2 / (s2["n"] - 1)
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

            pairs.append(
                {
                    "group1": str(g1),
                    "group2": str(g2),
                    "meandiff": float(mean_diff),
                    "se": float(se),
                    "q": float(q_stat),
                    "df": float(df_welch),
                    "p_value": float(p_val),
                    "reject": p_val < alpha,
                }
            )

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>GAMES-HOWELL POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n"
        summary += "<<COLOR:text>>(Does not assume equal variances)<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
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
        result["plots"].append(
            {
                "title": "Games-Howell — Pairwise Differences with CI",
                "data": [
                    {
                        "type": "scatter",
                        "x": [p["meandiff"] for p in pairs],
                        "y": y_labels,
                        "mode": "markers",
                        "marker": {"color": colors, "size": 10},
                        "error_x": {
                            "type": "data",
                            "array": ci_half,
                            "color": "rgba(74,159,110,0.6)",
                            "thickness": 2,
                        },
                        "showlegend": False,
                    }
                ],
                "layout": {
                    "height": max(250, 40 * len(pairs)),
                    "xaxis": {
                        "title": "Mean Difference",
                        "zeroline": True,
                        "zerolinecolor": "#e89547",
                        "zerolinewidth": 2,
                    },
                    "yaxis": {"automargin": True},
                },
            }
        )

        result["guide_observation"] = (
            f"Games-Howell: {n_sig}/{len(pairs)} pairs significant (unequal variances assumed)."
        )
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig}
        verdict = f"Games-Howell: {n_sig} of {len(pairs)} pairs differ"
        body = (
            f"Post-hoc comparison assuming unequal variances. <strong>{n_sig}</strong> pair{'s' if n_sig != 1 else ''} significant."
            if n_sig
            else "No pairs differ significantly."
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Games-Howell is preferred when group variances are unequal (Levene's test significant).",
        )

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
            group_info[lev] = {
                "mean_rank": g["_rank"].mean(),
                "n": len(g),
                "median": g[var].median(),
            }

        # Pairwise Dunn's test
        from itertools import combinations

        level_pairs = list(combinations(levels, 2))
        n_comparisons = len(level_pairs)

        # Tied-rank correction factor
        ranks = data["_rank"].values
        unique_ranks, counts = np.unique(ranks, return_counts=True)
        tie_correction = (
            1 - np.sum(counts**3 - counts) / (n_total**3 - n_total)
            if n_total > 1
            else 1
        )

        pairs = []
        for g1, g2 in level_pairs:
            s1 = group_info[g1]
            s2 = group_info[g2]
            diff = s1["mean_rank"] - s2["mean_rank"]

            # Standard error with tie correction
            se = np.sqrt(
                tie_correction
                * (n_total * (n_total + 1) / 12)
                * (1.0 / s1["n"] + 1.0 / s2["n"])
            )
            z = diff / se if se > 0 else 0
            p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni

            pairs.append(
                {
                    "group1": str(g1),
                    "group2": str(g2),
                    "rank_diff": float(diff),
                    "z": float(z),
                    "p_raw": float(p_raw),
                    "p_adj": float(p_adj),
                    "reject": p_adj < alpha,
                }
            )

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += (
            "<<COLOR:title>>DUNN'S TEST (Post-Hoc for Kruskal-Wallis)<</COLOR>>\n"
        )
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(levels)} levels of {group_var}\n"
        summary += "<<COLOR:highlight>>Correction:<</COLOR>> Bonferroni\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += "<<COLOR:accent>>── Group Rank Statistics ──<</COLOR>>\n"
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
        result["plots"].append(
            {
                "title": "Dunn's Test — Mean Ranks by Group",
                "data": [
                    {
                        "type": "bar",
                        "x": [str(lev) for lev in levels],
                        "y": [group_info[lev]["mean_rank"] for lev in levels],
                        "marker": {
                            "color": "#4a9f6e",
                            "line": {"color": "#3d8a5c", "width": 1},
                        },
                        "text": [f"n={group_info[lev]['n']}" for lev in levels],
                        "textposition": "outside",
                    }
                ],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": "Mean Rank"},
                    "xaxis": {"title": group_var},
                },
            }
        )

        # Pairwise rank differences plot
        pair_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        pair_colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        result["plots"].append(
            {
                "title": "Dunn's Test — Pairwise Rank Differences",
                "data": [
                    {
                        "type": "scatter",
                        "x": [p["rank_diff"] for p in pairs],
                        "y": pair_labels,
                        "mode": "markers",
                        "marker": {"color": pair_colors, "size": 10},
                        "text": [f"z={p['z']:.2f}, p={p['p_adj']:.4f}" for p in pairs],
                        "hoverinfo": "text+x",
                        "showlegend": False,
                    }
                ],
                "layout": {
                    "height": max(250, 40 * len(pairs)),
                    "xaxis": {
                        "title": "Rank Difference",
                        "zeroline": True,
                        "zerolinecolor": "#e89547",
                        "zerolinewidth": 2,
                    },
                    "yaxis": {"automargin": True},
                },
            }
        )

        result["guide_observation"] = (
            f"Dunn's test: {n_sig}/{len(pairs)} pairwise comparisons significant (Bonferroni)."
        )
        result["statistics"] = {
            "pairs": pairs,
            "n_significant": n_sig,
            "n_comparisons": n_comparisons,
        }
        verdict = (
            f"Dunn's test: {n_sig} of {len(pairs)} pairs differ (Bonferroni corrected)"
        )
        body = (
            f"Non-parametric post-hoc following Kruskal-Wallis. <strong>{n_sig}</strong> pair{'s' if n_sig != 1 else ''} significant after Bonferroni correction."
            if n_sig
            else "No pairs differ after correction."
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Dunn's test is the appropriate post-hoc for Kruskal-Wallis (non-parametric). Bonferroni controls family-wise error.",
        )

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

        group_data_sch = {
            str(g): data_sch[data_sch[factor_sch] == g][response_sch].values
            for g in groups_sch
        }
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
                f_val = (diff**2) / (se**2 * (k_sch - 1)) if se > 0 else 0
                p_val = 1 - sch_stats.f.cdf(f_val, k_sch - 1, df_within)
                margin = np.sqrt(scheffe_crit) * se
                pairs_sch.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "diff": float(diff),
                        "se": float(se),
                        "f": float(f_val),
                        "p": float(p_val),
                        "lower": float(diff - margin),
                        "upper": float(diff + margin),
                        "reject": p_val < alpha_sch,
                    }
                )

        n_sig_sch = sum(1 for p in pairs_sch if p["reject"])
        summary_sch = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_sch += "<<COLOR:title>>SCHEFFÉ'S POST-HOC TEST<</COLOR>>\n"
        summary_sch += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_sch += f"<<COLOR:highlight>>Response:<</COLOR>> {response_sch}\n"
        summary_sch += (
            f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_sch} ({k_sch} groups)\n"
        )
        summary_sch += (
            f"<<COLOR:highlight>>MSE:<</COLOR>> {mse_sch:.4f}  (df = {df_within})\n"
        )
        summary_sch += f"<<COLOR:highlight>>Scheffé critical value:<</COLOR>> {scheffe_crit:.4f}\n\n"

        summary_sch += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'SE':>8} {'F':>8} {'p':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary_sch += f"{'─' * 85}\n"
        for p in pairs_sch:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary_sch += f"{p['group1']:<15} {p['group2']:<15} {p['diff']:>8.4f} {p['se']:>8.4f} {p['f']:>8.3f} {p['p']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"
        summary_sch += f"\n<<COLOR:text>>Summary: {n_sig_sch}/{len(pairs_sch)} pairs significantly different<</COLOR>>\n"
        summary_sch += "<<COLOR:text>>Note: Scheffé is the most conservative — controls for ALL possible contrasts, not just pairwise.<</COLOR>>\n"

        result["summary"] = summary_sch

        # CI plot
        y_labels_sch = [f"{p['group1']} - {p['group2']}" for p in pairs_sch]
        result["plots"].append(
            {
                "title": "Scheffé — Pairwise Differences with CIs",
                "data": [
                    {
                        "type": "scatter",
                        "x": [p["diff"] for p in pairs_sch],
                        "y": y_labels_sch,
                        "mode": "markers",
                        "marker": {
                            "color": [
                                "#4a9f6e" if p["reject"] else "#5a6a5a"
                                for p in pairs_sch
                            ],
                            "size": 10,
                        },
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [p["upper"] - p["diff"] for p in pairs_sch],
                            "arrayminus": [p["diff"] - p["lower"] for p in pairs_sch],
                            "color": "#5a6a5a",
                        },
                        "showlegend": False,
                    }
                ],
                "layout": {
                    "height": max(250, 40 * len(pairs_sch)),
                    "xaxis": {
                        "title": "Mean Difference",
                        "zeroline": True,
                        "zerolinecolor": "#e89547",
                    },
                    "yaxis": {"automargin": True},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": -0.5,
                            "y1": len(pairs_sch) - 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        result["guide_observation"] = (
            f"Scheffé: {n_sig_sch}/{len(pairs_sch)} pairs significant at α={alpha_sch}."
        )
        result["statistics"] = {
            "pairs": pairs_sch,
            "mse": mse_sch,
            "scheffe_critical": scheffe_crit,
            "k": k_sch,
        }
        verdict = f"Scheff\u00e9: {n_sig_sch} of {len(pairs_sch)} contrasts significant"
        body = (
            f"The most conservative post-hoc test. <strong>{n_sig_sch}</strong> comparison{'s' if n_sig_sch != 1 else ''} significant."
            if n_sig_sch
            else "No comparisons significant — Scheffé is the most conservative test."
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Scheffé controls error for ALL possible contrasts (not just pairwise). Use Tukey for more power on pairwise comparisons only.",
        )

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

        group_data_bon = {
            str(g): data_bon[data_bon[factor_bon] == g][response_bon].values
            for g in groups_bon
        }
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
                pairs_bon.append(
                    {
                        "group1": g1,
                        "group2": g2,
                        "diff": float(diff),
                        "se": float(se),
                        "t": float(t_val),
                        "p_raw": float(p_raw),
                        "p_adj": float(p_adj),
                        "lower": float(diff - margin),
                        "upper": float(diff + margin),
                        "reject": p_adj < alpha_bon,
                    }
                )

        n_sig_bon = sum(1 for p in pairs_bon if p["reject"])
        summary_bon = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_bon += "<<COLOR:title>>BONFERRONI POST-HOC TEST<</COLOR>>\n"
        summary_bon += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_bon += f"<<COLOR:highlight>>Response:<</COLOR>> {response_bon}\n"
        summary_bon += (
            f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_bon} ({k_bon} groups)\n"
        )
        summary_bon += f"<<COLOR:highlight>>Adjusted α:<</COLOR>> {alpha_bon}/{n_comparisons_bon} = {alpha_adj:.6f}\n\n"

        summary_bon += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'t':>8} {'p-adj':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary_bon += f"{'─' * 82}\n"
        for p in pairs_bon:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary_bon += f"{p['group1']:<15} {p['group2']:<15} {p['diff']:>8.4f} {p['t']:>8.3f} {p['p_adj']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"
        summary_bon += f"\n<<COLOR:text>>Summary: {n_sig_bon}/{len(pairs_bon)} pairs significantly different (Bonferroni)<</COLOR>>\n"

        result["summary"] = summary_bon

        y_labels_bon = [f"{p['group1']} - {p['group2']}" for p in pairs_bon]
        result["plots"].append(
            {
                "title": "Bonferroni — Pairwise Differences with CIs",
                "data": [
                    {
                        "type": "scatter",
                        "x": [p["diff"] for p in pairs_bon],
                        "y": y_labels_bon,
                        "mode": "markers",
                        "marker": {
                            "color": [
                                "#4a9f6e" if p["reject"] else "#5a6a5a"
                                for p in pairs_bon
                            ],
                            "size": 10,
                        },
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [p["upper"] - p["diff"] for p in pairs_bon],
                            "arrayminus": [p["diff"] - p["lower"] for p in pairs_bon],
                            "color": "#5a6a5a",
                        },
                        "showlegend": False,
                    }
                ],
                "layout": {
                    "height": max(250, 40 * len(pairs_bon)),
                    "xaxis": {
                        "title": "Mean Difference",
                        "zeroline": True,
                        "zerolinecolor": "#e89547",
                    },
                    "yaxis": {"automargin": True},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": -0.5,
                            "y1": len(pairs_bon) - 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        result["guide_observation"] = (
            f"Bonferroni: {n_sig_bon}/{len(pairs_bon)} pairs significant (adj. α={alpha_adj:.4f})."
        )
        result["statistics"] = {
            "pairs": pairs_bon,
            "mse": mse_bon,
            "alpha_adjusted": alpha_adj,
        }
        verdict = f"Bonferroni: {n_sig_bon} of {len(pairs_bon)} pairs significant (adj. \u03b1 = {alpha_adj:.4f})"
        body = (
            f"Family-wise error controlled by dividing \u03b1 by {len(pairs_bon)} comparisons."
            + (
                f" <strong>{n_sig_bon}</strong> pair{'s' if n_sig_bon != 1 else ''} survive correction."
                if n_sig_bon
                else " No pairs survive correction."
            )
        )
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="Bonferroni is simple but conservative. For more power, use Tukey HSD (pairwise) or Holm-Bonferroni (step-down).",
        )

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
        direction = config.get(
            "direction", "max"
        )  # "max" = higher is better, "min" = lower is better

        data_hsu = df[[response_hsu, factor_hsu]].dropna()
        groups_hsu = sorted(data_hsu[factor_hsu].unique(), key=str)
        k_hsu = len(groups_hsu)

        group_data_hsu = {
            str(g): data_hsu[data_hsu[factor_hsu] == g][response_hsu].values
            for g in groups_hsu
        }
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

            comparisons_hsu.append(
                {
                    "group": g,
                    "mean": float(mean_g),
                    "diff_from_best": float(diff),
                    "lower": float(lower_mcb),
                    "upper": float(upper_mcb),
                    "could_be_best": could_be_best,
                    "is_best": g == best_group,
                }
            )

        summary_hsu = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_hsu += (
            "<<COLOR:title>>HSU'S MCB (MULTIPLE COMPARISONS WITH THE BEST)<</COLOR>>\n"
        )
        summary_hsu += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_hsu += f"<<COLOR:highlight>>Response:<</COLOR>> {response_hsu}\n"
        summary_hsu += (
            f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_hsu} ({k_hsu} groups)\n"
        )
        summary_hsu += f"<<COLOR:highlight>>Direction:<</COLOR>> {'Higher is better' if direction == 'max' else 'Lower is better'}\n"
        summary_hsu += f"<<COLOR:highlight>>Best group:<</COLOR>> {best_group} (mean = {group_means_hsu[best_group]:.4f})\n\n"

        summary_hsu += f"{'Group':<20} {'Mean':>8} {'Diff':>8} {'Lower':>8} {'Upper':>8} {'Could be best?':>15}\n"
        summary_hsu += f"{'─' * 72}\n"
        for c in comparisons_hsu:
            best_mark = (
                "<<COLOR:good>>YES<</COLOR>>"
                if c["could_be_best"]
                else "<<COLOR:warning>>NO<</COLOR>>"
            )
            star = " ◄" if c["is_best"] else ""
            summary_hsu += f"{c['group']:<20} {c['mean']:>8.4f} {c['diff_from_best']:>8.4f} {c['lower']:>8.4f} {c['upper']:>8.4f} {best_mark:>15}{star}\n"

        n_could_be_best = sum(1 for c in comparisons_hsu if c["could_be_best"])
        summary_hsu += f"\n<<COLOR:text>>{n_could_be_best}/{k_hsu} groups could be the best at {(1 - alpha_hsu) * 100:.0f}% confidence.<</COLOR>>\n"

        result["summary"] = summary_hsu

        # MCB interval plot
        result["plots"].append(
            {
                "title": f"Hsu's MCB — Differences from Best ({direction})",
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": [c["diff_from_best"] for c in comparisons_hsu],
                        "y": [c["group"] for c in comparisons_hsu],
                        "marker": {
                            "color": [
                                "#4a9f6e" if c["could_be_best"] else "#d94a4a"
                                for c in comparisons_hsu
                            ],
                            "size": 10,
                        },
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                c["upper"] - c["diff_from_best"]
                                for c in comparisons_hsu
                            ],
                            "arrayminus": [
                                c["diff_from_best"] - c["lower"]
                                for c in comparisons_hsu
                            ],
                            "color": "#5a6a5a",
                        },
                        "showlegend": False,
                    }
                ],
                "layout": {
                    "height": max(250, 40 * k_hsu),
                    "xaxis": {"title": f"Difference from best ({best_group})"},
                    "yaxis": {"automargin": True},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": -0.5,
                            "y1": k_hsu - 0.5,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        result["guide_observation"] = (
            f"Hsu's MCB: {n_could_be_best}/{k_hsu} groups could be best. Best={best_group}."
        )
        result["statistics"] = {
            "comparisons": comparisons_hsu,
            "best_group": best_group,
            "direction": direction,
            "mse": mse_hsu,
        }
        verdict = f"Hsu's MCB: {n_could_be_best} of {k_hsu} groups could be the best"
        body = f"Current best group: <strong>{best_group}</strong> ({direction}). {n_could_be_best} group{'s' if n_could_be_best != 1 else ''} cannot be ruled out as the best."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="MCB (Multiple Comparisons with the Best) identifies which groups could plausibly be the best. Narrow the field with more data.",
        )

    return result
