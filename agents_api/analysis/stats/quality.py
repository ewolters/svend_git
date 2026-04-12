"""DSW Statistical Analysis — quality analysis (capability, acceptance sampling, ANOM)."""

import logging

import numpy as np

from ..common import (
    _check_normality,
    _check_outliers,
    _narrative,
)

logger = logging.getLogger(__name__)


def _run_quality(analysis_id, df, config):
    """Run quality analysis."""
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "variance_test":
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
            result["summary"] = (
                "Please provide either two columns, a response + grouping factor, or a single column with a hypothesized sigma (sigma0)."
            )
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"

        if mode == "one_sample":
            sigma0 = float(sigma0)
            n = len(x)
            s2 = float(np.var(x, ddof=1))
            s = float(np.sqrt(s2))
            chi2_stat = (n - 1) * s2 / (sigma0**2) if sigma0 > 0 else 0

            p_val = float(
                2
                * min(
                    stats.chi2.cdf(chi2_stat, n - 1),
                    1 - stats.chi2.cdf(chi2_stat, n - 1),
                )
            )

            ci_lo_var = (n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, n - 1)
            ci_hi_var = (n - 1) * s2 / stats.chi2.ppf(alpha / 2, n - 1)

            summary += "<<COLOR:title>>ONE-SAMPLE VARIANCE TEST (Chi-Square)<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1}\n"
            summary += f"<<COLOR:highlight>>H₀:<</COLOR>> σ = {sigma0}\n\n"
            summary += "<<COLOR:accent>>── Sample Results ──<</COLOR>>\n"
            summary += f"  N: {n}\n"
            summary += f"  Sample std dev: {s:.4f}\n"
            summary += f"  Sample variance: {s2:.4f}\n\n"
            summary += "<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
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
                "n": n,
                "sample_std": s,
                "sample_variance": s2,
                "chi2_statistic": chi2_stat,
                "df": n - 1,
                "p_value": p_val,
                "ci_variance_lower": ci_lo_var,
                "ci_variance_upper": ci_hi_var,
            }
            result["guide_observation"] = (
                f"One-sample variance test: s={s:.4f} vs σ₀={sigma0}, χ²={chi2_stat:.3f}, p={p_val:.4f}. "
                + ("Significant." if p_val < alpha else "Not significant.")
            )
            if p_val < alpha:
                result["narrative"] = _narrative(
                    f"Variance differs from \u03c3\u2080\u00b2 = {sigma0**2:.4f} (p = {p_val:.4f})",
                    f"Sample std dev s = {s:.4f} is significantly different from hypothesized \u03c3\u2080 = {sigma0:.4f}.",
                    next_steps="Investigate process changes that may have increased or decreased variability.",
                )
            else:
                result["narrative"] = _narrative(
                    f"Variance consistent with \u03c3\u2080 = {sigma0} (p = {p_val:.4f})",
                    f"Sample std dev s = {s:.4f} is consistent with \u03c3\u2080 = {sigma0:.4f}.",
                )

        else:
            # Multi-group: Bartlett's + Levene's (always both)
            k = len(groups_data)
            ns = [len(g) for g in groups_data]
            stds = [float(np.std(g, ddof=1)) for g in groups_data]
            variances = [float(np.var(g, ddof=1)) for g in groups_data]

            bart_stat, bart_p = stats.bartlett(*groups_data)
            lev_stat, lev_p = stats.levene(*groups_data, center="median")

            summary += "<<COLOR:title>>TEST FOR EQUAL VARIANCES<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

            if mode == "factor":
                summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response_col}\n"
                summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"
            else:
                summary += f"<<COLOR:highlight>>Columns:<</COLOR>> {var1}, {var2}\n\n"

            summary += "<<COLOR:accent>>── Sample Statistics ──<</COLOR>>\n"
            summary += f"  {'Group':<20} {'N':>6} {'StDev':>10} {'Variance':>12}\n"
            summary += f"  {'─' * 50}\n"
            for lbl, n_i, s_i, v_i in zip(groups_labels, ns, stds, variances):
                summary += f"  {str(lbl):<20} {n_i:>6} {s_i:>10.4f} {v_i:>12.4f}\n"

            summary += "\n<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
            summary += f"  {'Test':<25} {'Statistic':>12} {'p-value':>10}\n"
            summary += f"  {'─' * 50}\n"
            summary += f"  {'Bartlett (normal data)':<25} {bart_stat:>12.4f} {bart_p:>10.4f}\n"
            summary += f"  {'Levene (robust)':<25} {lev_stat:>12.4f} {lev_p:>10.4f}\n"

            # F-test only for exactly 2 groups
            f_stat, f_p = None, None
            if k == 2:
                f_stat = variances[0] / variances[1] if variances[1] > 0 else float("inf")
                df1, df2 = ns[0] - 1, ns[1] - 1
                f_p = float(
                    2
                    * min(
                        stats.f.cdf(f_stat, df1, df2),
                        1 - stats.f.cdf(f_stat, df1, df2),
                    )
                )
                summary += f"  {'F-test (2-sample)':<25} {f_stat:>12.4f} {f_p:>10.4f}\n"

                # CI for variance ratio
                f_lo = stats.f.ppf(alpha / 2, df1, df2)
                f_hi = stats.f.ppf(1 - alpha / 2, df1, df2)
                ratio_lo = f_stat / f_hi
                ratio_hi = f_stat / f_lo
                summary += f"\n  {conf_pct:.0f}% CI for σ₁²/σ₂²: ({ratio_lo:.4f}, {ratio_hi:.4f})\n"

            summary += "\n<<COLOR:accent>>── Recommendation ──<</COLOR>>\n"
            summary += "  Use Levene's test (robust to non-normality).\n"
            summary += "  Bartlett's test is more powerful but assumes normal data.\n\n"

            sig = lev_p < alpha
            if sig:
                summary += f"<<COLOR:good>>Variances are significantly different (Levene's p = {lev_p:.4f} < {alpha})<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference in variances (Levene's p = {lev_p:.4f} ≥ {alpha})<</COLOR>>"

            result["statistics"] = {
                "bartlett_statistic": float(bart_stat),
                "bartlett_p": float(bart_p),
                "levene_statistic": float(lev_stat),
                "levene_p": float(lev_p),
                "group_stds": dict(zip([str(g) for g in groups_labels], stds)),
                "group_variances": dict(zip([str(g) for g in groups_labels], variances)),
            }
            if f_stat is not None:
                result["statistics"]["f_statistic"] = float(f_stat)
                result["statistics"]["f_p_value"] = float(f_p)

            result["guide_observation"] = (
                f"Variance test ({k} groups): Levene's p={lev_p:.4f}, Bartlett's p={bart_p:.4f}. "
                + ("Variances differ." if sig else "Variances are equal.")
            )
            if sig:
                result["narrative"] = _narrative(
                    "Group variances differ significantly",
                    f"Levene's p = {lev_p:.4f}, Bartlett's p = {bart_p:.4f}. At least one group has a different spread.",
                    next_steps="Use Welch's ANOVA or Games-Howell post-hoc (both handle unequal variances).",
                )
            else:
                result["narrative"] = _narrative(
                    "Group variances are equal",
                    f"Levene's p = {lev_p:.4f}, Bartlett's p = {bart_p:.4f}. Equal-variance assumption is reasonable.",
                    next_steps="Proceed with standard ANOVA and Tukey HSD post-hoc.",
                )

            # Side-by-side box/strip plots showing spread
            traces = []
            for i, (lbl, gd) in enumerate(zip(groups_labels, groups_data)):
                colors = [
                    "#4a9f6e",
                    "#4a90d9",
                    "#e8c547",
                    "#c75a3a",
                    "#7a5fb8",
                    "#5a9fd4",
                    "#d4a05a",
                    "#5ad4a0",
                ]
                traces.append(
                    {
                        "type": "box",
                        "y": gd.tolist(),
                        "name": str(lbl),
                        "marker": {"color": colors[i % len(colors)]},
                        "boxpoints": "outliers",
                    }
                )
            result["plots"].append(
                {
                    "title": "Variability Comparison",
                    "data": traces,
                    "layout": {"height": 300, "yaxis": {"title": "Value"}},
                }
            )

            # Interval plot of standard deviations
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": [str(g) for g in groups_labels],
                            "y": stds,
                            "marker": {"color": "#4a9f6e"},
                            "text": [f"{s:.4f}" for s in stds],
                            "textposition": "outside",
                        }
                    ],
                    "layout": {
                        "title": "Standard Deviations by Group",
                        "yaxis": {"title": "Std Dev", "rangemode": "tozero"},
                        "height": 250,
                    },
                }
            )

            # ── Diagnostics ──
            diagnostics = []
            # Check normality of each group (F-test / Bartlett are sensitive to non-normality)
            _any_non_normal = False
            for _gl, _gd in zip(groups_labels, groups_data):
                _gn = _check_normality(_gd, label=str(_gl))
                if _gn:
                    _any_non_normal = True
                    diagnostics.append(_gn)
            if _any_non_normal:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": "Non-normal data detected — F-test and Bartlett's are unreliable",
                        "detail": "The F-test and Bartlett's test assume normality. Use Levene's (median-centered) or Brown-Forsythe test for robust variance comparison.",
                    }
                )

            # Check outliers per group
            for _gl, _gd in zip(groups_labels, groups_data):
                _go = _check_outliers(_gd, label=str(_gl))
                if _go:
                    diagnostics.append(_go)

            # Variance ratio interpretation (for 2-group case)
            if k == 2 and variances[1] > 0:
                _vratio = variances[0] / variances[1]
                _vratio_display = _vratio if _vratio >= 1 else 1 / _vratio
                if _vratio_display > 4:
                    diagnostics.append(
                        {
                            "level": "warning",
                            "title": f"Large variance inequality (ratio = {_vratio_display:.2f})",
                            "detail": "Variance ratio exceeds 4:1. For means comparisons, use Welch's t-test which does not assume equal variances.",
                        }
                    )
                elif 0.5 <= _vratio <= 2 and sig:
                    diagnostics.append(
                        {
                            "level": "info",
                            "title": f"Significant but practically similar variances (ratio = {_vratio_display:.2f})",
                            "detail": "The test detects a statistically significant difference, but the variance ratio is near 1. The practical impact on downstream analyses (t-tests, ANOVA) is minimal.",
                        }
                    )

            result["diagnostics"] = diagnostics

        result["summary"] = summary

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
        denom = 1 + z_a**2 / total_opp
        center = (p_hat + z_a**2 / (2 * total_opp)) / denom
        half = z_a * np.sqrt(p_hat * (1 - p_hat) / total_opp + z_a**2 / (4 * total_opp**2)) / denom
        ci_lo = max(0, center - half)
        ci_hi = min(1, center + half)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ATTRIBUTE CAPABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

        if var:
            summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
            if event and event != "sum":
                summary += f"<<COLOR:highlight>>Defect value:<</COLOR>> {event}\n"
        summary += f"<<COLOR:highlight>>Opportunities per unit:<</COLOR>> {opp:.0f}\n\n"

        summary += "<<COLOR:accent>>── Summary ──<</COLOR>>\n"
        summary += f"  Total units inspected: {n:.0f}\n"
        summary += f"  Total defects: {d:.0f}\n"
        summary += f"  Total opportunities: {n * opp:.0f}\n\n"

        summary += "<<COLOR:accent>>── Capability Metrics ──<</COLOR>>\n"
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
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [f"{s}σ" for s in sigma_levels],
                        "y": dpmo_levels,
                        "marker": {
                            "color": ["#c75a3a" if s < sigma_st else "#4a9f6e" for s in sigma_levels],
                            "opacity": 0.5,
                        },
                        "name": "DPMO by sigma",
                    },
                    {
                        "type": "scatter",
                        "x": [f"{sigma_st:.1f}σ"],
                        "y": [dpmo],
                        "mode": "markers",
                        "marker": {"size": 14, "color": "#e8c547", "symbol": "diamond"},
                        "name": f"Your process ({dpmo:.0f} DPMO)",
                    },
                ],
                "layout": {
                    "title": "Process Sigma Level",
                    "yaxis": {"title": "DPMO", "type": "log"},
                    "height": 280,
                },
            }
        )

        result["guide_observation"] = (
            f"Attribute capability: DPMO={dpmo:.0f}, Sigma={sigma_st:.1f}, Yield={yield_pct:.2f}%."
        )
        _sigma_label = (
            "world-class"
            if sigma_st >= 6
            else (
                "excellent"
                if sigma_st >= 5
                else ("good" if sigma_st >= 4 else "needs improvement" if sigma_st >= 3 else "poor")
            )
        )
        result["narrative"] = _narrative(
            f"Attribute Capability: {sigma_st:.1f}\u03c3 ({_sigma_label})",
            f"DPMO = {dpmo:.0f}, Yield = {yield_pct:.2f}%. Process sigma level = {sigma_st:.1f}.",
            next_steps=(
                "Focus on reducing DPMO. Pareto the top defect types to prioritize improvement."
                if sigma_st < 4
                else "Strong capability. Monitor and maintain."
            ),
            chart_guidance="The sigma scale: 3\u03c3 = 66,807 DPMO, 4\u03c3 = 6,210 DPMO, 6\u03c3 = 3.4 DPMO.",
        )
        result["statistics"] = {
            "defects": d,
            "units": n,
            "opportunities": opp,
            "dpu": dpu,
            "dpo": dpo,
            "dpmo": dpmo,
            "yield_percent": yield_pct,
            "z_bench": z_bench,
            "sigma_level": sigma_st,
            "interpretation": interp,
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
        summary += "<<COLOR:title>>NONPARAMETRIC PROCESS CAPABILITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Specs:<</COLOR>> LSL = {lsl}, USL = {usl}, Target = {target}\n\n"

        summary += "<<COLOR:accent>>── Normality Test (Anderson-Darling) ──<</COLOR>>\n"
        summary += f"  AD statistic: {ad_stat:.4f}\n"
        summary += f"  5% critical value: {ad_crit[2]:.4f}\n"
        summary += f"  Data is {'normal' if is_normal else '<<COLOR:warning>>non-normal<</COLOR>>'}\n\n"

        summary += "<<COLOR:accent>>── Comparison — Normal vs Nonparametric ──<</COLOR>>\n"
        summary += f"  {'Method':<25} {'Cp/Cnp':>10} {'Cpk/Cnpk':>10}\n"
        summary += f"  {'─' * 48}\n"
        summary += f"  {'Normal assumption':<25} {cp_normal:>10.3f} {cpk_normal:>10.3f}\n"
        summary += f"  {'Nonparametric (percentile)':<25} {cnp:>10.3f} {cnpk:>10.3f}\n\n"

        summary += "<<COLOR:accent>>── Nonparametric Details ──<</COLOR>>\n"
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
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "histogram",
                        "x": data_arr.tolist(),
                        "marker": {"color": "#4a9f6e", "opacity": 0.7},
                        "name": var,
                        "nbinsx": min(30, max(10, n // 5)),
                    },
                ],
                "layout": {
                    "title": "Distribution with Spec Limits",
                    "xaxis": {"title": var},
                    "yaxis": {"title": "Frequency"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": lsl,
                            "x1": lsl,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e85747", "dash": "dash", "width": 2},
                        },
                        {
                            "type": "line",
                            "x0": usl,
                            "x1": usl,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e85747", "dash": "dash", "width": 2},
                        },
                        {
                            "type": "line",
                            "x0": target,
                            "x1": target,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e8c547", "dash": "dot", "width": 1},
                        },
                        {
                            "type": "line",
                            "x0": p_low,
                            "x1": p_low,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#7a5fb8", "dash": "dot", "width": 1},
                        },
                        {
                            "type": "line",
                            "x0": p_high,
                            "x1": p_high,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#7a5fb8", "dash": "dot", "width": 1},
                        },
                    ],
                    "annotations": [
                        {
                            "x": lsl,
                            "y": 1.02,
                            "yref": "paper",
                            "text": "LSL",
                            "showarrow": False,
                            "font": {"color": "#e85747", "size": 10},
                        },
                        {
                            "x": usl,
                            "y": 1.02,
                            "yref": "paper",
                            "text": "USL",
                            "showarrow": False,
                            "font": {"color": "#e85747", "size": 10},
                        },
                        {
                            "x": p_low,
                            "y": 0.95,
                            "yref": "paper",
                            "text": "0.135%ile",
                            "showarrow": False,
                            "font": {"color": "#7a5fb8", "size": 9},
                        },
                        {
                            "x": p_high,
                            "y": 0.95,
                            "yref": "paper",
                            "text": "99.865%ile",
                            "showarrow": False,
                            "font": {"color": "#7a5fb8", "size": 9},
                        },
                    ],
                    "height": 300,
                },
            }
        )

        result["guide_observation"] = (
            f"Nonparametric capability: Cnpk={cnpk:.3f}, Empirical PPM={ppm_total:.0f}. Data is {'normal' if is_normal else 'non-normal'}."
        )
        _cnpk_label = "capable" if cnpk >= 1.33 else "marginal" if cnpk >= 1.0 else "not capable"
        result["narrative"] = _narrative(
            f"Non-parametric Cpk = {cnpk:.3f} ({_cnpk_label})",
            f"Distribution-free capability using percentile-based estimates. PPM = {ppm_total:.0f}. Data is {'normal' if is_normal else 'non-normal — this method is more appropriate than standard Cpk'}.",
            next_steps=(
                "Non-normal capability avoids distributional assumptions. If Cnpk < 1.33, reduce variation or center the process."
                if cnpk < 1.33
                else "Process is capable. Monitor with control charts."
            ),
        )
        result["statistics"] = {
            "cnp": cnp,
            "cnpk": cnpk,
            "cnpk_upper": cnpk_upper,
            "cnpk_lower": cnpk_lower,
            "cp_normal": cp_normal,
            "cpk_normal": cpk_normal,
            "median": median_val,
            "p_0135": p_low,
            "p_99865": p_high,
            "ppm_below_lsl": ppm_below,
            "ppm_above_usl": ppm_above,
            "ppm_total": ppm_total,
            "ad_statistic": float(ad_stat),
            "is_normal": is_normal,
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
            n_approx = ((z_alpha + z_beta) / (z_p1 - z_p2)) ** 2 + 0.5 * z_alpha**2
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
        summary += "<<COLOR:title>>VARIABLES ACCEPTANCE SAMPLING PLAN<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += "<<COLOR:accent>>── Plan Parameters ──<</COLOR>>\n"
        summary += f"  AQL (Acceptable Quality Level):         {aql}%\n"
        summary += f"  LTPD (Lot Tolerance Pct Defective):     {ltpd}%\n"
        summary += f"  Producer's risk (alpha):                {alpha_risk}\n"
        summary += f"  Consumer's risk (beta):                 {beta_risk}\n"
        summary += f"  Lot size:                               {lot_size}\n\n"
        summary += "<<COLOR:accent>>── Sampling Plan ──<</COLOR>>\n"
        summary += f"  <<COLOR:highlight>>Sample size (n):<</COLOR>>      {n_sample}\n"
        summary += f"  <<COLOR:highlight>>Critical value (k):<</COLOR>>   {k_val:.4f}\n\n"
        summary += f"<<COLOR:accent>>── Decision Rule ({spec_type} spec) ──<</COLOR>>\n"
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
            summary += "\n<<COLOR:accent>>── Sample Evaluation ──<</COLOR>>\n"
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

        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": (p_range * 100).tolist(),
                        "y": pa_values,
                        "mode": "lines",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "name": "OC Curve",
                    },
                    {
                        "type": "scatter",
                        "x": [aql],
                        "y": [1 - alpha_risk],
                        "mode": "markers",
                        "marker": {"color": "#47a5e8", "size": 10, "symbol": "diamond"},
                        "name": f"AQL ({aql}%)",
                    },
                    {
                        "type": "scatter",
                        "x": [ltpd],
                        "y": [beta_risk],
                        "mode": "markers",
                        "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"},
                        "name": f"LTPD ({ltpd}%)",
                    },
                ],
                "layout": {
                    "title": f"OC Curve (n={n_sample}, k={k_val:.3f})",
                    "xaxis": {"title": "Percent Defective (%)"},
                    "yaxis": {"title": "Probability of Acceptance", "range": [0, 1.05]},
                    "height": 350,
                },
            }
        )

        result["guide_observation"] = (
            f"Variables sampling plan: n={n_sample}, k={k_val:.4f} for AQL={aql}%, LTPD={ltpd}%."
        )
        result["narrative"] = _narrative(
            f"Variables Sampling Plan: n = {n_sample}, k = {k_val:.4f}",
            f"Sample {n_sample} items and accept if the sample mean is within k\u00d7s of the spec limit. AQL = {aql}%, LTPD = {ltpd}%.",
            next_steps="This plan assumes normally distributed measurements. Verify normality before use.",
        )
        result["statistics"] = {
            "n": n_sample,
            "k": k_val,
            "aql": aql,
            "ltpd": ltpd,
            "alpha": alpha_risk,
            "beta": beta_risk,
            "lot_size": lot_size,
        }

    # =====================================================================
    # Poisson Regression
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

            plan_results.append(
                {
                    "name": plan_name,
                    "n": n_mpc,
                    "c": c_mpc,
                    "type": plan_type,
                    "pa_values": pa_vals,
                    "aoq_values": aoq_vals,
                    "pa_aql": pa_aql,
                    "pa_ltpd": pa_ltpd,
                    "alpha": alpha_risk,
                    "beta": beta_risk,
                    "aoql": aoql,
                    "ati_aql": ati_aql,
                    "color": plan_colors[idx % len(plan_colors)],
                }
            )

        # OC Curve comparison
        oc_traces = []
        for pr in plan_results:
            oc_traces.append(
                {
                    "x": (p_range_mpc * 100).tolist(),
                    "y": pr["pa_values"].tolist(),
                    "mode": "lines",
                    "name": f"{pr['name']} (n={pr['n']}, c={pr['c']})",
                    "line": {"color": pr["color"], "width": 2},
                }
            )
        # AQL and LTPD reference lines
        oc_traces.append(
            {
                "x": [aql_mpc * 100, aql_mpc * 100],
                "y": [0, 1],
                "mode": "lines",
                "name": f"AQL ({aql_mpc * 100:.1f}%)",
                "line": {"color": "#4a90d9", "dash": "dot"},
            }
        )
        oc_traces.append(
            {
                "x": [ltpd_mpc * 100, ltpd_mpc * 100],
                "y": [0, 1],
                "mode": "lines",
                "name": f"LTPD ({ltpd_mpc * 100:.1f}%)",
                "line": {"color": "#d94a4a", "dash": "dot"},
            }
        )

        result["plots"].append(
            {
                "title": "OC Curve Comparison",
                "data": oc_traces,
                "layout": {
                    "height": 400,
                    "xaxis": {"title": "Lot Defect Rate (%)"},
                    "yaxis": {"title": "P(Accept)", "range": [0, 1.05]},
                },
            }
        )

        # AOQ comparison
        aoq_traces = [
            {
                "x": (p_range_mpc * 100).tolist(),
                "y": (pr["aoq_values"] * 100).tolist(),
                "mode": "lines",
                "name": pr["name"],
                "line": {"color": pr["color"], "width": 2},
            }
            for pr in plan_results
        ]
        result["plots"].append(
            {
                "title": "AOQ Curve Comparison",
                "data": aoq_traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Incoming Defect Rate (%)"},
                    "yaxis": {"title": "Average Outgoing Quality (%)"},
                },
            }
        )

        # Summary table
        summary_mpc = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_mpc += "<<COLOR:title>>SAMPLING PLAN COMPARISON<</COLOR>>\n"
        summary_mpc += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_mpc += f"<<COLOR:highlight>>Lot size:<</COLOR>> {lot_size_mpc}\n"
        summary_mpc += f"<<COLOR:highlight>>AQL:<</COLOR>> {aql_mpc * 100:.1f}%    <<COLOR:highlight>>LTPD:<</COLOR>> {ltpd_mpc * 100:.1f}%\n\n"

        summary_mpc += f"{'Plan':<20} {'n':>5} {'c':>3} {'P(Acc@AQL)':>11} {'P(Acc@LTPD)':>12} {'α':>6} {'β':>6} {'AOQL%':>7} {'ATI@AQL':>8}\n"
        summary_mpc += f"{'─' * 82}\n"
        best_beta = min(pr["beta"] for pr in plan_results)
        for pr in plan_results:
            beta_mark = "<<COLOR:good>> ◄<</COLOR>>" if pr["beta"] == best_beta else "   "
            summary_mpc += f"{pr['name']:<20} {pr['n']:>5} {pr['c']:>3} {pr['pa_aql']:>11.4f} {pr['pa_ltpd']:>12.4f} {pr['alpha']:>6.3f} {pr['beta']:>6.3f} {pr['aoql'] * 100:>7.3f} {pr['ati_aql']:>8.0f}{beta_mark}\n"

        summary_mpc += "\n<<COLOR:text>>◄ = lowest consumer risk (β)<</COLOR>>\n"
        result["summary"] = summary_mpc

        result["guide_observation"] = (
            f"Compared {len(plan_results)} sampling plans. Best consumer risk: {best_beta:.3f}."
        )
        result["narrative"] = _narrative(
            f"Sampling Plan Comparison: {len(plan_results)} plans evaluated",
            f"Best consumer risk (\u03b2) = {best_beta:.3f}. Compare OC curves to find the best trade-off between sample size and protection.",
            next_steps="Choose the plan that balances inspection cost (sample size) with risk. Lower \u03b2 = better consumer protection.",
        )
        result["statistics"] = {
            "plans": [
                {k: v for k, v in pr.items() if k not in ("pa_values", "aoq_values", "color")} for pr in plan_results
            ],
            "aql": aql_mpc,
            "ltpd": ltpd_mpc,
            "lot_size": lot_size_mpc,
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
        ppm_above = (
            float((1 - norm_dist.cdf((usl_val - x_bar_cs) / s_cs)) * 1e6) if usl_val is not None and s_cs > 0 else 0
        )
        ppm_total = ppm_below + ppm_above

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PROCESS CAPABILITY SIXPACK -- {var_cs}<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += "<<COLOR:accent>>── Specifications ──<</COLOR>>\n"
        if lsl_val is not None:
            summary += f"  LSL: {lsl_val}\n"
        if usl_val is not None:
            summary += f"  USL: {usl_val}\n"
        if target_val is not None:
            summary += f"  Target: {target_val}\n"
        summary += "\n<<COLOR:accent>>── Process Stats ──<</COLOR>>\n"
        summary += f"  N: {n_cs},  Mean: {x_bar_cs:.4f},  StDev: {s_cs:.4f}\n\n"
        summary += "<<COLOR:accent>>── Capability Indices ──<</COLOR>>\n"
        if cp_val is not None:
            summary += f"  Cp:  {cp_val:.3f}\n"
        summary += f"  Cpk: {cpk_val:.3f}\n"
        if cpl_val is not None:
            summary += f"  CPL: {cpl_val:.3f}\n"
        if cpu_val is not None:
            summary += f"  CPU: {cpu_val:.3f}\n"
        summary += "\n<<COLOR:accent>>── Expected PPM ──<</COLOR>>\n"
        if lsl_val is not None:
            summary += f"  Below LSL: {ppm_below:.1f}\n"
        if usl_val is not None:
            summary += f"  Above USL: {ppm_above:.1f}\n"
        summary += f"  Total:     {ppm_total:.1f}\n"
        if cpk_val >= 1.33:
            summary += "\n<<COLOR:good>>Process is capable (Cpk >= 1.33).<</COLOR>>\n"
        elif cpk_val >= 1.0:
            summary += "\n<<COLOR:warning>>Process is marginally capable (1.0 <= Cpk < 1.33).<</COLOR>>\n"
        else:
            summary += "\n<<COLOR:warning>>Process is NOT capable (Cpk < 1.0).<</COLOR>>\n"

        result["summary"] = summary

        # Panel 1 & 2: I-MR or Xbar-R
        if subgroup_size == 1:
            mr = np.abs(np.diff(col_cs))
            mr_bar = float(np.mean(mr))
            sigma_est = mr_bar / 1.128
            cl_i = x_bar_cs
            ucl_i = x_bar_cs + 3 * sigma_est
            lcl_i = x_bar_cs - 3 * sigma_est
            result["plots"].append(
                {
                    "title": "I Chart",
                    "data": [
                        {
                            "type": "scatter",
                            "y": col_cs.tolist(),
                            "mode": "lines+markers",
                            "marker": {"size": 3, "color": "#4a9f6e"},
                            "line": {"color": "#4a9f6e", "width": 1},
                            "name": var_cs,
                        },
                        {
                            "type": "scatter",
                            "y": [cl_i] * n_cs,
                            "mode": "lines",
                            "line": {"color": "#e8c547", "width": 1},
                            "name": f"CL={cl_i:.2f}",
                        },
                        {
                            "type": "scatter",
                            "y": [ucl_i] * n_cs,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                            "name": f"UCL={ucl_i:.2f}",
                        },
                        {
                            "type": "scatter",
                            "y": [lcl_i] * n_cs,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                            "name": f"LCL={lcl_i:.2f}",
                        },
                    ],
                    "layout": {
                        "height": 200,
                        "margin": {"t": 30, "b": 30},
                        "showlegend": False,
                    },
                    "group": "Control Charts",
                }
            )
            mr_ucl = mr_bar * 3.267
            result["plots"].append(
                {
                    "title": "MR Chart",
                    "data": [
                        {
                            "type": "scatter",
                            "y": mr.tolist(),
                            "mode": "lines+markers",
                            "marker": {"size": 3, "color": "#47a5e8"},
                            "line": {"color": "#47a5e8", "width": 1},
                            "name": "MR",
                        },
                        {
                            "type": "scatter",
                            "y": [mr_bar] * len(mr),
                            "mode": "lines",
                            "line": {"color": "#e8c547", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [mr_ucl] * len(mr),
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        },
                    ],
                    "layout": {
                        "height": 200,
                        "margin": {"t": 30, "b": 30},
                        "showlegend": False,
                    },
                    "group": "Control Charts",
                }
            )
        else:
            n_sg = n_cs // subgroup_size
            subgroups = [col_cs[i * subgroup_size : (i + 1) * subgroup_size] for i in range(n_sg)]
            xbar_sg = [float(np.mean(sg)) for sg in subgroups]
            ranges_sg = [float(np.max(sg) - np.min(sg)) for sg in subgroups]
            xbar_bar = float(np.mean(xbar_sg))
            r_bar = float(np.mean(ranges_sg))
            A2_tbl = {
                2: 1.880,
                3: 1.023,
                4: 0.729,
                5: 0.577,
                6: 0.483,
                7: 0.419,
                8: 0.373,
                9: 0.337,
                10: 0.308,
            }
            D4_tbl = {
                2: 3.267,
                3: 2.575,
                4: 2.282,
                5: 2.115,
                6: 2.004,
                7: 1.924,
                8: 1.864,
                9: 1.816,
                10: 1.777,
            }
            D3_tbl = {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0.076,
                8: 0.136,
                9: 0.184,
                10: 0.223,
            }
            a2 = A2_tbl.get(subgroup_size, 0.577)
            d4 = D4_tbl.get(subgroup_size, 2.115)
            d3 = D3_tbl.get(subgroup_size, 0)
            result["plots"].append(
                {
                    "title": "Xbar Chart",
                    "data": [
                        {
                            "type": "scatter",
                            "y": xbar_sg,
                            "mode": "lines+markers",
                            "marker": {"size": 3, "color": "#4a9f6e"},
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [xbar_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e8c547", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [xbar_bar + a2 * r_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [xbar_bar - a2 * r_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        },
                    ],
                    "layout": {
                        "height": 200,
                        "margin": {"t": 30, "b": 30},
                        "showlegend": False,
                    },
                    "group": "Control Charts",
                }
            )
            result["plots"].append(
                {
                    "title": "R Chart",
                    "data": [
                        {
                            "type": "scatter",
                            "y": ranges_sg,
                            "mode": "lines+markers",
                            "marker": {"size": 3, "color": "#47a5e8"},
                            "line": {"color": "#47a5e8", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [r_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e8c547", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [d4 * r_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        },
                        {
                            "type": "scatter",
                            "y": [d3 * r_bar] * n_sg,
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        },
                    ],
                    "layout": {
                        "height": 200,
                        "margin": {"t": 30, "b": 30},
                        "showlegend": False,
                    },
                    "group": "Control Charts",
                }
            )

        # Panel 3: Last observations run chart with spec limits
        last_obs = col_cs[-min(25 * max(subgroup_size, 1), n_cs) :]
        spec_shapes = []
        if lsl_val is not None:
            spec_shapes.append(
                {
                    "type": "line",
                    "x0": 0,
                    "x1": len(last_obs),
                    "y0": lsl_val,
                    "y1": lsl_val,
                    "line": {"color": "#e85747", "dash": "dot", "width": 1},
                }
            )
        if usl_val is not None:
            spec_shapes.append(
                {
                    "type": "line",
                    "x0": 0,
                    "x1": len(last_obs),
                    "y0": usl_val,
                    "y1": usl_val,
                    "line": {"color": "#e85747", "dash": "dot", "width": 1},
                }
            )
        result["plots"].append(
            {
                "title": "Last Observations",
                "data": [
                    {
                        "type": "scatter",
                        "y": last_obs.tolist(),
                        "mode": "lines+markers",
                        "marker": {"size": 3, "color": "#9aaa9a"},
                        "line": {"color": "#9aaa9a", "width": 1},
                    }
                ],
                "layout": {
                    "height": 200,
                    "margin": {"t": 30, "b": 30},
                    "shapes": spec_shapes,
                    "showlegend": False,
                },
                "group": "Control Charts",
            }
        )

        # Panel 4: Capability histogram
        hist_shapes = []
        hist_annot = []
        if lsl_val is not None:
            hist_shapes.append(
                {
                    "type": "line",
                    "x0": lsl_val,
                    "x1": lsl_val,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#e85747", "width": 2},
                }
            )
            hist_annot.append(
                {
                    "x": lsl_val,
                    "y": 1,
                    "yref": "paper",
                    "text": "LSL",
                    "showarrow": False,
                    "font": {"color": "#e85747", "size": 10},
                }
            )
        if usl_val is not None:
            hist_shapes.append(
                {
                    "type": "line",
                    "x0": usl_val,
                    "x1": usl_val,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#e85747", "width": 2},
                }
            )
            hist_annot.append(
                {
                    "x": usl_val,
                    "y": 1,
                    "yref": "paper",
                    "text": "USL",
                    "showarrow": False,
                    "font": {"color": "#e85747", "size": 10},
                }
            )
        if target_val is not None:
            hist_shapes.append(
                {
                    "type": "line",
                    "x0": target_val,
                    "x1": target_val,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#47a5e8", "dash": "dash", "width": 1},
                }
            )
        result["plots"].append(
            {
                "title": "Capability Histogram",
                "data": [
                    {
                        "type": "histogram",
                        "x": col_cs.tolist(),
                        "marker": {"color": "#4a9f6e", "opacity": 0.7},
                        "nbinsx": min(30, n_cs // 3),
                    }
                ],
                "layout": {
                    "height": 200,
                    "margin": {"t": 30, "b": 30},
                    "shapes": hist_shapes,
                    "annotations": hist_annot,
                    "showlegend": False,
                },
                "group": "Capability",
            }
        )

        # Panel 5: Normal probability plot
        sorted_cs = np.sort(col_cs)
        probs_cs = [(i + 0.5) / n_cs for i in range(n_cs)]
        theoretical_cs = [float(norm_dist.ppf(p)) for p in probs_cs]
        result["plots"].append(
            {
                "title": "Normal Probability Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": theoretical_cs,
                        "y": sorted_cs.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 3},
                    }
                ],
                "layout": {
                    "height": 200,
                    "margin": {"t": 30, "b": 30},
                    "xaxis": {"title": "Theoretical Quantiles"},
                    "yaxis": {"title": var_cs},
                    "showlegend": False,
                },
                "group": "Capability",
            }
        )

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
        result["plots"].append(
            {
                "title": "Capability Statistics",
                "data": [
                    {
                        "type": "scatter",
                        "x": [0.5],
                        "y": [0.5],
                        "mode": "text",
                        "text": [stats_text_cs],
                        "textfont": {"size": 14, "color": "#4a9f6e"},
                    }
                ],
                "layout": {
                    "height": 200,
                    "margin": {"t": 30, "b": 30},
                    "xaxis": {"visible": False, "range": [0, 1]},
                    "yaxis": {"visible": False, "range": [0, 1]},
                    "showlegend": False,
                },
                "group": "Capability",
            }
        )

        result["guide_observation"] = f"Capability sixpack: Cpk={cpk_val:.3f}, PPM={ppm_total:.0f}."
        _cpk_label = "capable" if cpk_val >= 1.33 else "marginal" if cpk_val >= 1.0 else "not capable"
        result["narrative"] = _narrative(
            f"Capability Sixpack: Cpk = {cpk_val:.3f} ({_cpk_label})",
            f"PPM = {ppm_total:.0f}. The sixpack combines control charts, capability histogram, normal probability plot, and capability indices in one view.",
            next_steps="The sixpack verifies two prerequisites: (1) process stability (control chart) and (2) normality (probability plot). Only trust Cpk if both are satisfied.",
            chart_guidance="Six panels: I chart (stability), moving range (variation stability), last 25 observations, capability histogram, normal probability plot, and capability summary.",
        )
        result["statistics"] = {
            "n": n_cs,
            "mean": x_bar_cs,
            "stdev": s_cs,
            "cp": cp_val,
            "cpk": cpk_val,
            "cpl": cpl_val,
            "cpu": cpu_val,
            "ppm_below": ppm_below,
            "ppm_above": ppm_above,
            "ppm_total": ppm_total,
            "lsl": lsl_val,
            "usl": usl_val,
            "target": target_val,
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
        for i, (mean_i, u, lo) in enumerate(zip(group_means, udls, ldls)):
            if mean_i > u or mean_i < lo:
                outside.append(groups_labels[i])

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ANALYSIS OF MEANS (ANOM)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {var_anom}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_anom or 'columns'}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {k_anom}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha_anom}\n\n"
        summary += f"<<COLOR:text>>Grand Mean:<</COLOR>> {grand_mean:.4f}\n"
        summary += f"<<COLOR:text>>MSE (within):<</COLOR>> {mse_anom:.4f}\n\n"
        summary += "<<COLOR:accent>>── Group Results ──<</COLOR>>\n"
        summary += f"  {'Group':<15} {'N':>5} {'Mean':>10} {'UDL':>10} {'LDL':>10} {'Signal':>8}\n"
        summary += f"  {'-' * 60}\n"
        for i in range(k_anom):
            sig = "*" if groups_labels[i] in outside else ""
            summary += f"  {str(groups_labels[i]):<15} {group_ns[i]:>5} {group_means[i]:>10.4f} {udls[i]:>10.4f} {ldls[i]:>10.4f} {sig:>8}\n"

        if outside:
            summary += (
                f"\n<<COLOR:warning>>Groups outside decision limits: {', '.join(str(o) for o in outside)}<</COLOR>>\n"
            )
        else:
            summary += "\n<<COLOR:good>>All group means within decision limits.<</COLOR>>\n"

        result["summary"] = summary

        marker_colors = ["#e85747" if groups_labels[i] in outside else "#4a9f6e" for i in range(k_anom)]
        chart_data = [
            {
                "type": "scatter",
                "x": [str(g) for g in groups_labels],
                "y": group_means,
                "mode": "markers",
                "marker": {"color": marker_colors, "size": 10},
                "name": "Group Means",
            },
            {
                "type": "scatter",
                "x": [str(groups_labels[0]), str(groups_labels[-1])],
                "y": [grand_mean, grand_mean],
                "mode": "lines",
                "line": {"color": "#e8c547", "width": 2},
                "name": f"Grand Mean ({grand_mean:.3f})",
            },
        ]
        if balanced:
            chart_data.append(
                {
                    "type": "scatter",
                    "x": [str(groups_labels[0]), str(groups_labels[-1])],
                    "y": [udl_anom, udl_anom],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"UDL ({udl_anom:.3f})",
                }
            )
            chart_data.append(
                {
                    "type": "scatter",
                    "x": [str(groups_labels[0]), str(groups_labels[-1])],
                    "y": [ldl_anom, ldl_anom],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"LDL ({ldl_anom:.3f})",
                }
            )
        else:
            chart_data.append(
                {
                    "type": "scatter",
                    "x": [str(g) for g in groups_labels],
                    "y": udls,
                    "mode": "lines+markers",
                    "marker": {"size": 3},
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": "UDL",
                }
            )
            chart_data.append(
                {
                    "type": "scatter",
                    "x": [str(g) for g in groups_labels],
                    "y": ldls,
                    "mode": "lines+markers",
                    "marker": {"size": 3},
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": "LDL",
                }
            )

        result["plots"].append(
            {
                "data": chart_data,
                "layout": {
                    "title": "ANOM Chart",
                    "xaxis": {"title": factor_anom or "Group"},
                    "yaxis": {"title": var_anom},
                    "height": 350,
                },
            }
        )

        result["guide_observation"] = (
            f"ANOM: {len(outside)} of {k_anom} groups outside decision limits."
            if outside
            else f"ANOM: All {k_anom} groups within decision limits."
        )
        if outside:
            result["narrative"] = _narrative(
                f"ANOM: {len(outside)} of {k_anom} groups differ from overall mean",
                f"Groups outside decision limits: <strong>{', '.join(str(o) for o in outside[:5])}</strong>. These are significantly different from the grand mean.",
                next_steps="ANOM is a graphical alternative to ANOVA. Groups outside the limits warrant investigation.",
                chart_guidance="Points outside the dashed decision limits differ significantly from the overall average.",
            )
        else:
            result["narrative"] = _narrative(
                f"ANOM: All {k_anom} groups within decision limits",
                "No group means are significantly different from the overall mean.",
                chart_guidance="All points within the decision limits = no significant group differences.",
            )
        result["statistics"] = {
            "grand_mean": grand_mean,
            "mse": mse_anom,
            "k": k_anom,
            "n_total": n_total,
            "group_means": {str(g): m for g, m in zip(groups_labels, group_means)},
            "outside_limits": [str(o) for o in outside],
            "alpha": alpha_anom,
        }

    # =====================================================================
    # Split-Plot ANOVA
    # =====================================================================
    elif analysis_id == "acceptance_sampling":
        """
        Acceptance Sampling Plans — single and double sampling for lot inspection.
        Computes OC curve, AOQ curve, ATI, and determines accept/reject based on AQL and LTPD.
        Supports both attribute (defective count) and variable (normal) plans.
        """
        from scipy import stats as scipy_stats

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
                    1 - p_accept_1 - p_reject_1
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
                pa_values = np.array(
                    [(float(scipy_stats.binom.cdf(accept_num, n_sample, p)) if p > 0 else 1.0) for p in p_range]
                )
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
            result["plots"].append(
                {
                    "data": [
                        {
                            "x": (p_range * 100).tolist(),
                            "y": pa_values.tolist(),
                            "mode": "lines",
                            "name": "OC Curve",
                            "line": {"color": "#2c5f2d", "width": 2},
                        },
                        {
                            "x": [aql * 100],
                            "y": [pa_aql],
                            "mode": "markers+text",
                            "name": f"AQL ({aql * 100:.1f}%)",
                            "marker": {"color": "#4a90d9", "size": 10},
                            "text": [f"AQL: Pa={pa_aql:.3f}"],
                            "textposition": "top right",
                        },
                        {
                            "x": [ltpd * 100],
                            "y": [pa_ltpd],
                            "mode": "markers+text",
                            "name": f"LTPD ({ltpd * 100:.1f}%)",
                            "marker": {"color": "#d94a4a", "size": 10},
                            "text": [f"LTPD: Pa={pa_ltpd:.3f}"],
                            "textposition": "top left",
                        },
                    ],
                    "layout": {
                        "title": f"Operating Characteristic (OC) Curve — {plan_desc}",
                        "xaxis": {"title": "Lot Defect Rate (%)"},
                        "yaxis": {
                            "title": "Probability of Acceptance",
                            "range": [0, 1.05],
                        },
                    },
                }
            )

            # AOQ Curve plot
            result["plots"].append(
                {
                    "data": [
                        {
                            "x": (p_range * 100).tolist(),
                            "y": (aoq_values * 100).tolist(),
                            "mode": "lines",
                            "name": "AOQ Curve",
                            "line": {"color": "#d9a04a", "width": 2},
                        },
                        {
                            "x": [aoql_p * 100],
                            "y": [aoql * 100],
                            "mode": "markers+text",
                            "name": f"AOQL={aoql * 100:.3f}%",
                            "marker": {"color": "#d94a4a", "size": 10},
                            "text": [f"AOQL={aoql * 100:.3f}%"],
                            "textposition": "top right",
                        },
                    ],
                    "layout": {
                        "title": "Average Outgoing Quality (AOQ) Curve",
                        "xaxis": {"title": "Incoming Defect Rate (%)"},
                        "yaxis": {"title": "Average Outgoing Quality (%)"},
                    },
                }
            )

            result["summary"] = (
                f"**Acceptance Sampling Plan**\n\n**Plan:** {plan_desc}\n**Lot size:** {lot_size}\n\n| Metric | Value |\n|---|---|\n| P(accept) at AQL ({aql * 100:.1f}%) | {pa_aql:.4f} |\n| P(accept) at LTPD ({ltpd * 100:.1f}%) | {pa_ltpd:.4f} |\n| Producer's risk (α) | {alpha_risk:.4f} ({alpha_risk * 100:.1f}%) |\n| Consumer's risk (β) | {beta_risk:.4f} ({beta_risk * 100:.1f}%) |\n| AOQL | {aoql * 100:.4f}% at p={aoql_p * 100:.2f}% |\n| ATI at AQL | {float(np.interp(aql, p_range, ati_values)):.0f} units |"
            )

            result["guide_observation"] = (
                f"Acceptance sampling ({plan_desc}): α={alpha_risk:.3f}, β={beta_risk:.3f}, AOQL={aoql * 100:.4f}%."
            )
            result["narrative"] = _narrative(
                f"Acceptance Sampling: {plan_desc}",
                f"Producer's risk (\u03b1) = {alpha_risk:.3f}, Consumer's risk (\u03b2) = {beta_risk:.3f}, AOQL = {aoql * 100:.4f}%.",
                next_steps="The OC curve shows acceptance probability vs lot quality. AOQL is the worst average quality that passes through.",
                chart_guidance="OC curve: steeper = better discrimination between good and bad lots. The curve should drop sharply at the quality threshold.",
            )

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
                "aoql_defect_rate": aoql_p,
            }

        except Exception as e:
            result["summary"] = f"Acceptance sampling error: {str(e)}"

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
                var_factor = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, "iloc") else float(fit.cov_re)
                var_error = float(fit.scale)
                components[factors[0]] = var_factor
                components["Error"] = var_error
            else:
                # ANOVA method: for each factor, compute between-group MS and within-group MS
                for factor in factors:
                    groups = data.groupby(factor)[response]
                    group_means = groups.mean()
                    grand_mean = data[response].mean()
                    k_groups = len(group_means)
                    n_per = N / k_groups  # average group size
                    ms_between = float(np.sum(groups.count() * (group_means - grand_mean) ** 2) / (k_groups - 1))
                    ms_within = float(np.sum(groups.apply(lambda x: np.sum((x - x.mean()) ** 2))) / (N - k_groups))
                    var_component = max(0, (ms_between - ms_within) / n_per)
                    components[factor] = var_component
                components["Error"] = (
                    float(data.groupby(factors[0])[response].apply(lambda x: x.var()).mean()) if factors else total_var
                )

            comp_total = sum(components.values())
            pct = {k: v / comp_total * 100 if comp_total > 0 else 0 for k, v in components.items()}

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += "<<COLOR:title>>VARIANCE COMPONENTS ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method.upper()}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += "<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
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
            result["plots"].append(
                {
                    "title": "Variance Components",
                    "data": [
                        {
                            "type": "pie",
                            "labels": labels,
                            "values": values,
                            "marker": {"colors": colors[: len(labels)]},
                            "textinfo": "label+percent",
                            "hole": 0.3,
                        }
                    ],
                    "layout": {"height": 300},
                }
            )

            # Bar chart
            result["plots"].append(
                {
                    "title": "Variance Components (Bar)",
                    "data": [
                        {
                            "type": "bar",
                            "x": labels,
                            "y": values,
                            "marker": {"color": colors[: len(labels)]},
                            "text": [f"{pct[k]:.1f}%" for k in labels],
                            "textposition": "outside",
                        }
                    ],
                    "layout": {"height": 280, "yaxis": {"title": "Variance"}},
                }
            )

            result["statistics"] = {
                "components": {
                    k: {"variance": v, "pct": pct[k], "std_dev": float(np.sqrt(v))} for k, v in components.items()
                },
                "total_variance": comp_total,
                "method": method,
                "n": N,
            }
            result["guide_observation"] = (
                "Variance components: " + ", ".join([f"{k}={pct[k]:.1f}%" for k in components]) + "."
            )
            _top_comp = max(pct.items(), key=lambda x: x[1]) if pct else ("", 0)
            result["narrative"] = _narrative(
                f"Variance Components: {_top_comp[0]} dominates ({_top_comp[1]:.1f}%)",
                f"Breakdown: {', '.join(f'{k} = {v:.1f}%' for k, v in pct.items())}.",
                next_steps=f"Focus improvement on <strong>{_top_comp[0]}</strong> — reducing the largest variance component gives the most impact.",
            )

        except Exception as e:
            result["summary"] = f"Variance components error: {str(e)}"

    return result
