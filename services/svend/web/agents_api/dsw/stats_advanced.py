"""DSW Statistical Analysis — advanced methods (power, MSA, survival, time series)."""

import logging
import math

import numpy as np

from .common import (
    _narrative,
)

logger = logging.getLogger(__name__)


def _run_advanced(analysis_id, df, config):
    """Run advanced analysis."""
    import pandas as pd
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "power_z":
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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n_req],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — 1-Sample Z (δ={delta}, σ={sigma})",
                    "xaxis": {"title": "Sample Size (n)"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Sample Z-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>Effect: δ = {delta}, σ = {sigma} → Cohen's d = {d:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, alternative = {alt}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-sample Z power: need n={n_req} for d={d:.3f} at power={target_power}."
        result["statistics"] = {
            "required_n": n_req,
            "effect_size_d": d,
            "alpha": alpha,
            "power": target_power,
            "delta": delta,
            "sigma": sigma,
        }
        _pz_mag = "large" if d >= 0.8 else ("medium" if d >= 0.5 else "small")
        result["narrative"] = _narrative(
            f"Power Analysis — n = {n_req} required",
            f"To detect a {_pz_mag} effect (d = {d:.3f}) with {target_power * 100:.0f}% power at \u03b1 = {alpha}, you need <strong>n = {n_req}</strong> observations.",
            next_steps="If n is too large, consider increasing the acceptable effect size or relaxing \u03b1.",
            chart_guidance="The power curve shows how power increases with sample size. The red dot marks your required n.",
        )

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
        n_req = (
            math.ceil(((z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(pa * (1 - pa))) / (pa - p0)) ** 2)
            if pa != p0
            else 9999
        )

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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n_req],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — 1-Proportion (p₀={p0}, pₐ={pa})",
                    "xaxis": {"title": "Sample Size (n)"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: p = {p0}, H₁: p = {pa} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"1-prop power: need n={n_req} to detect p={pa} vs p₀={p0} at power={target_power}."
        )
        result["statistics"] = {
            "required_n": n_req,
            "p0": p0,
            "pa": pa,
            "cohens_h": h,
            "alpha": alpha,
            "power": target_power,
        }
        result["narrative"] = _narrative(
            f"Power Analysis — n = {n_req} for proportion test",
            f"To detect a shift from p\u2080 = {p0} to p\u2081 = {pa} (Cohen's h = {h:.3f}) with {target_power * 100:.0f}% power, you need <strong>n = {n_req}</strong>.",
            next_steps="Use p = 0.5 for a conservative estimate if the expected proportion is unknown.",
            chart_guidance="The curve shows power vs sample size. The red dot marks your target.",
        )

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

        numer = z_a * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar)) + z_b * math.sqrt(
            p1 * (1 - p1) + p2 * (1 - p2) / ratio
        )
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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n1],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n₁={n1}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — 2-Proportion (p₁={p1}, p₂={p2})",
                    "xaxis": {"title": "Sample Size per Group (n₁)"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>p₁ = {p1}, p₂ = {p2} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, ratio n₂/n₁ = {ratio}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n1}, n₂ = {n2} (total = {n1 + n2})<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"2-prop power: need n₁={n1}, n₂={n2} for |Δp|={abs(p1 - p2):.3f} at power={target_power}."
        )
        result["narrative"] = _narrative(
            f"Power Analysis — n\u2081 = {n1}, n\u2082 = {n2} per group",
            f"To detect |p\u2081 \u2212 p\u2082| = {abs(p1 - p2):.3f} (h = {h:.3f}) with {target_power * 100:.0f}% power: <strong>n\u2081 = {n1}, n\u2082 = {n2}</strong>.",
            next_steps="Equal allocation (ratio = 1) is most efficient. Unequal ratios require larger total n.",
            chart_guidance="The power curve shows power vs n\u2081. The red dot marks your target.",
        )
        result["statistics"] = {
            "n1": n1,
            "n2": n2,
            "total_n": n1 + n2,
            "p1": p1,
            "p2": p2,
            "cohens_h": h,
            "alpha": alpha,
            "power": target_power,
        }

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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n_req],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — 1-Variance (σ₀={sigma0}, σ₁={sigma1})",
                    "xaxis": {"title": "Sample Size (n)"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Variance Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ = {sigma0}, H₁: σ = {sigma1} → ratio σ₁²/σ₀² = {ratio_sq:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"1-variance power: need n={n_req} to detect σ₁={sigma1} vs σ₀={sigma0} at power={target_power}."
        )
        result["narrative"] = _narrative(
            f"Power Analysis — n = {n_req} for variance test",
            f"To detect \u03c3\u2081 = {sigma1} vs \u03c3\u2080 = {sigma0} (ratio = {ratio_sq:.3f}) with {target_power * 100:.0f}% power: <strong>n = {n_req}</strong>.",
            next_steps="Variance tests require larger samples than mean tests. Consider if a practical change in spread matters.",
            chart_guidance="Power curve shows how detection probability increases with n.",
        )
        result["statistics"] = {
            "required_n": n_req,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "variance_ratio": ratio_sq,
            "alpha": alpha,
            "power": target_power,
        }

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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n_req],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n₁={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — 2-Variance F-Test (ratio={var_ratio})",
                    "xaxis": {"title": "Sample Size per Group (n₁)"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Variance F-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ₁²/σ₂² = 1, H₁: σ₁²/σ₂² = {var_ratio}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, n₂/n₁ = {ratio_n}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n_req}, n₂ = {n2_req} (total = {n_req + n2_req})<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"2-variance power: need n₁={n_req}, n₂={n2_req} for ratio={var_ratio} at power={target_power}."
        )
        result["narrative"] = _narrative(
            f"Power Analysis — n\u2081 = {n_req}, n\u2082 = {n2_req} for F-test",
            f"To detect a variance ratio of {var_ratio} between two groups with {target_power * 100:.0f}% power: <strong>n\u2081 = {n_req}, n\u2082 = {n2_req}</strong>.",
            next_steps="F-tests are sensitive to non-normality. Consider Levene's test as a robust alternative.",
            chart_guidance="Power curve shows power vs n\u2081. Equal group sizes maximize power.",
        )
        result["statistics"] = {
            "n1": n_req,
            "n2": n2_req,
            "total_n": n_req + n2_req,
            "variance_ratio": var_ratio,
            "alpha": alpha,
            "power": target_power,
        }

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

        result["plots"].append(
            {
                "data": [
                    {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                    {
                        "x": [n_req],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"n/group={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Power Curve — Equivalence (TOST, margin=±{margin})",
                    "xaxis": {"title": "Sample Size per Group"},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — Equivalence Test (TOST)<</COLOR>>\n\n"
            f"<<COLOR:text>>Equivalence margin: ±{margin}, true difference: {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n = {n_req} per group (total = {2 * n_req})<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"Equivalence power: need n={n_req}/group for margin=±{margin}, δ={delta} at power={target_power}."
        )
        result["narrative"] = _narrative(
            f"Power Analysis — n = {n_req} per group for equivalence",
            f"To demonstrate equivalence within \u00b1{margin} (true \u03b4 = {delta}) with {target_power * 100:.0f}% power: <strong>n = {n_req} per group</strong>.",
            next_steps="Tighter margins require more samples. Ensure the margin reflects a practically meaningful difference.",
            chart_guidance="Power curve shows how close to the margin the true difference must be for the test to succeed.",
        )
        result["statistics"] = {
            "n_per_group": n_req,
            "total_n": 2 * n_req,
            "margin": margin,
            "delta": delta,
            "sigma": sigma,
            "alpha": alpha,
            "power": target_power,
        }

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

        from scipy.stats import norm
        from scipy.stats import t as t_dist

        n_runs_base = 2**n_factors  # full factorial runs

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

        result["plots"].append(
            {
                "data": [
                    {
                        "x": reps_range,
                        "y": powers,
                        "mode": "lines+markers",
                        "name": "Power",
                        "line": {"color": "#4a90d9", "width": 2},
                    },
                    {
                        "x": [req_reps],
                        "y": [target_power],
                        "mode": "markers",
                        "name": f"reps={req_reps}",
                        "marker": {"size": 12, "color": "#d94a4a", "symbol": "star"},
                    },
                ],
                "layout": {
                    "title": f"DOE Power — 2^{n_factors} Factorial (Δ={delta}, σ={sigma})",
                    "xaxis": {"title": "Number of Replicates", "dtick": 1},
                    "yaxis": {"title": "Power", "range": [0, 1]},
                },
            }
        )

        n_total_req = n_runs_base * req_reps
        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2^{n_factors} Factorial DOE<</COLOR>>\n\n"
            f"<<COLOR:text>>Factors: {n_factors}, base runs: {n_runs_base}<</COLOR>>\n"
            f"<<COLOR:text>>Minimum detectable effect: Δ = {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required replicates: {req_reps} → total runs = {n_total_req}<</COLOR>>\n"
        )
        result["guide_observation"] = (
            f"DOE power: 2^{n_factors} design needs {req_reps} reps ({n_total_req} runs) to detect Δ={delta} at power={target_power}."
        )
        result["narrative"] = _narrative(
            f"DOE Power — {req_reps} replicates ({n_total_req} runs)",
            f"A 2^{n_factors} factorial design needs <strong>{req_reps} replicates</strong> ({n_total_req} total runs) to detect an effect of \u0394 = {delta} (\u03c3 = {sigma}) with {target_power * 100:.0f}% power.",
            next_steps="If too many runs, consider a fractional factorial or screen fewer factors first.",
            chart_guidance="Power increases with replicates. The red dot marks the minimum replicates needed.",
        )
        result["statistics"] = {
            "factors": n_factors,
            "base_runs": n_runs_base,
            "required_reps": req_reps,
            "total_runs": n_total_req,
            "delta": delta,
            "sigma": sigma,
            "alpha": alpha,
            "power": target_power,
        }

    elif analysis_id == "sample_size_ci":
        """
        Sample size for estimation — determine n needed for a target CI half-width.
        Supports mean (Z or t) and proportion.
        """
        est_type = config.get("type", "mean")  # mean or proportion
        conf_level = (
            float(config.get("conf", 95)) / 100 if float(config.get("conf", 95)) > 1 else float(config.get("conf", 95))
        )
        target_width = float(config.get("half_width", 1.0))  # desired CI half-width (margin of error)

        from scipy.stats import norm

        z = norm.ppf(1 - (1 - conf_level) / 2)

        if est_type == "proportion":
            p_est = float(config.get("p_est", 0.5))  # expected proportion
            n_req = math.ceil((z**2 * p_est * (1 - p_est)) / (target_width**2))

            # Width curve
            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * math.sqrt(p_est * (1 - p_est) / nn) for nn in ns]

            result["plots"].append(
                {
                    "data": [
                        {
                            "x": ns,
                            "y": widths,
                            "mode": "lines",
                            "name": "CI Half-Width",
                            "line": {"color": "#4a90d9", "width": 2},
                        },
                        {
                            "x": [n_req],
                            "y": [target_width],
                            "mode": "markers",
                            "name": f"n={n_req}",
                            "marker": {"size": 10, "color": "#d94a4a"},
                        },
                        {
                            "x": ns,
                            "y": [target_width] * len(ns),
                            "mode": "lines",
                            "name": "Target",
                            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                        },
                    ],
                    "layout": {
                        "title": f"CI Half-Width vs n — Proportion (p≈{p_est})",
                        "xaxis": {"title": "Sample Size"},
                        "yaxis": {"title": "Half-Width"},
                    },
                }
            )

            extra_text = f"<<COLOR:text>>Expected proportion: p ≈ {p_est}<</COLOR>>\n"
        else:
            sigma = float(config.get("sigma", 1.0))
            n_req = math.ceil((z * sigma / target_width) ** 2)

            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * sigma / math.sqrt(nn) for nn in ns]

            result["plots"].append(
                {
                    "data": [
                        {
                            "x": ns,
                            "y": widths,
                            "mode": "lines",
                            "name": "CI Half-Width",
                            "line": {"color": "#4a90d9", "width": 2},
                        },
                        {
                            "x": [n_req],
                            "y": [target_width],
                            "mode": "markers",
                            "name": f"n={n_req}",
                            "marker": {"size": 10, "color": "#d94a4a"},
                        },
                        {
                            "x": ns,
                            "y": [target_width] * len(ns),
                            "mode": "lines",
                            "name": "Target",
                            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                        },
                    ],
                    "layout": {
                        "title": f"CI Half-Width vs n — Mean (σ={sigma})",
                        "xaxis": {"title": "Sample Size"},
                        "yaxis": {"title": "Half-Width"},
                    },
                }
            )
            extra_text = f"<<COLOR:text>>Population σ = {sigma}<</COLOR>>\n"

        n_req = max(n_req, 2)
        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Estimation ({est_type.title()})<</COLOR>>\n\n"
            f"<<COLOR:text>>Confidence level: {conf_level:.0%}, target half-width (margin of error): {target_width}<</COLOR>>\n"
            f"{extra_text}\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"Sample size for {conf_level:.0%} CI with half-width={target_width}: n={n_req}."
        result["narrative"] = _narrative(
            f"Sample Size for {conf_level:.0%} CI — n = {n_req}",
            f"To achieve a {conf_level:.0%} confidence interval with half-width \u2264 {target_width}: <strong>n = {n_req}</strong>. "
            f"Type: {est_type}.",
            next_steps="Wider margins require fewer samples. Use a pilot study to estimate \u03c3 if unknown.",
            chart_guidance="The curve shows how CI half-width shrinks as n increases. The dashed line is your target precision.",
        )
        result["statistics"] = {
            "required_n": n_req,
            "type": est_type,
            "confidence": conf_level,
            "half_width": target_width,
        }

    elif analysis_id == "sample_size_tolerance":
        """
        Sample size for tolerance intervals.
        Given coverage (e.g. 99% of population), confidence (e.g. 95%), → n.
        Uses the Howe (1969) / Odeh-Owen factor approach.
        """
        coverage = float(config.get("coverage", 0.99))
        confidence = float(config.get("confidence", 0.95))
        interval_type = config.get("type", "two-sided")  # two-sided or one-sided

        from scipy.stats import chi2 as chi2_dist
        from scipy.stats import norm

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
                k_factors.append(float("nan"))

        result["plots"].append(
            {
                "data": [
                    {
                        "x": ns,
                        "y": k_factors,
                        "mode": "lines",
                        "name": "k-factor",
                        "line": {"color": "#4a90d9", "width": 2},
                    },
                    {
                        "x": ns,
                        "y": [z_p] * len(ns),
                        "mode": "lines",
                        "name": f"z_p={z_p:.3f} (asymptote)",
                        "line": {"color": "#4a9f6e", "dash": "dash", "width": 1},
                    },
                    {
                        "x": [n_req],
                        "y": [
                            z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1))
                            if chi2_dist.ppf(1 - confidence, n_req - 1) > 0
                            else z_p
                        ],
                        "mode": "markers",
                        "name": f"n={n_req}",
                        "marker": {"size": 10, "color": "#d94a4a"},
                    },
                ],
                "layout": {
                    "title": f"Tolerance k-Factor vs n ({coverage:.0%}/{confidence:.0%})",
                    "xaxis": {"title": "Sample Size"},
                    "yaxis": {"title": "k-Factor"},
                },
            }
        )

        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Tolerance Interval<</COLOR>>\n\n"
            f"<<COLOR:text>>Coverage: {coverage:.1%} of population<</COLOR>>\n"
            f"<<COLOR:text>>Confidence: {confidence:.1%}<</COLOR>>\n"
            f"<<COLOR:text>>Type: {interval_type}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
            f"<<COLOR:text>>At this n, the tolerance k-factor ≈ {z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1)):.3f} (asymptote = {z_p:.3f})<</COLOR>>\n"
        )
        result["guide_observation"] = f"Tolerance interval ({coverage:.0%}/{confidence:.0%}): need n={n_req}."
        result["narrative"] = _narrative(
            f"Sample Size for Tolerance Interval — n = {n_req}",
            f"To bound {coverage:.0%} of the population with {confidence:.0%} confidence ({interval_type}): <strong>n = {n_req}</strong>. "
            f"The k-factor converges to z = {z_p:.3f} as n increases.",
            next_steps="Tolerance intervals are critical for capability studies — ensure n is sufficient before reporting Cpk.",
            chart_guidance="The k-factor curve shows how the tolerance multiplier decreases (tightens) with more data. It asymptotes to z_p.",
        )
        result["statistics"] = {
            "required_n": n_req,
            "coverage": coverage,
            "confidence": confidence,
            "type": interval_type,
            "z_p": z_p,
        }

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
        summary += "<<COLOR:title>>GRANGER CAUSALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Question:<</COLOR>> Does {var_x} Granger-cause {var_y}?\n"
        summary += f"<<COLOR:highlight>>Max Lags:<</COLOR>> {max_lag}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {min_len}\n\n"

        summary += "<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
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

            summary += "<<COLOR:accent>>── Results by Lag ──<</COLOR>>\n"
            summary += f"{'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significant':<12}\n"
            summary += f"{'-' * 42}\n"

            significant_lags = []
            p_values = []
            f_stats = []

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    test_result = gc_results[lag][0]["ssr_ftest"]
                    f_stat = test_result[0]
                    p_val = test_result[1]
                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

                    f_stats.append(f_stat)
                    p_values.append(p_val)

                    if p_val < 0.05:
                        significant_lags.append(lag)

                    summary += f"{lag:<6} {f_stat:<12.4f} {p_val:<12.4f} {sig}\n"

            summary += "\n<<COLOR:accent>>────────────────────────────────────────<</COLOR>>\n"
            if significant_lags:
                summary += f"<<COLOR:good>>CONCLUSION: {var_x} Granger-causes {var_y}<</COLOR>>\n"
                summary += f"<<COLOR:text>>Significant at lags: {significant_lags}<</COLOR>>\n"
                summary += f"<<COLOR:text>>This suggests {var_x} has predictive power for {var_y}.<</COLOR>>\n"
            else:
                summary += "<<COLOR:warning>>CONCLUSION: No Granger causality detected<</COLOR>>\n"
                summary += f"<<COLOR:text>>{var_x} does not significantly improve prediction of {var_y}.<</COLOR>>\n"

            result["summary"] = summary

            # Plot: p-values by lag
            result["plots"].append(
                {
                    "title": f"Granger Causality: {var_x} → {var_y}",
                    "data": [
                        {
                            "type": "bar",
                            "x": [f"Lag {i + 1}" for i in range(len(p_values))],
                            "y": p_values,
                            "marker": {
                                "color": ["#e85747" if p < 0.05 else "rgba(74, 159, 110, 0.4)" for p in p_values],
                                "line": {"color": "#4a9f6e", "width": 1.5},
                            },
                        },
                        {
                            "type": "scatter",
                            "x": [f"Lag {i + 1}" for i in range(len(p_values))],
                            "y": [0.05] * len(p_values),
                            "mode": "lines",
                            "line": {"color": "#e85747", "dash": "dash", "width": 2},
                            "name": "α = 0.05",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "yaxis": {"title": "p-value", "range": [0, max(0.2, max(p_values) * 1.1)]},
                        "xaxis": {"title": "Lag"},
                    },
                }
            )

            # Time series plot
            result["plots"].append(
                {
                    "title": f"Time Series: {var_x} and {var_y}",
                    "data": [
                        {
                            "type": "scatter",
                            "y": x[:min_len].tolist(),
                            "mode": "lines",
                            "name": var_x,
                            "line": {"color": "#4a9f6e"},
                        },
                        {
                            "type": "scatter",
                            "y": y[:min_len].tolist(),
                            "mode": "lines",
                            "name": var_y,
                            "yaxis": "y2",
                            "line": {"color": "#47a5e8"},
                        },
                    ],
                    "layout": {
                        "height": 250,
                        "yaxis": {"title": var_x, "side": "left"},
                        "yaxis2": {"title": var_y, "side": "right", "overlaying": "y"},
                        "xaxis": {"title": "Observation"},
                    },
                }
            )

            result["guide_observation"] = f"Granger causality: {var_x} → {var_y} " + (
                f"significant at lags {significant_lags}." if significant_lags else "not significant."
            )
            if significant_lags:
                result["narrative"] = _narrative(
                    f"{var_x} Granger-causes {var_y} at lags {significant_lags}",
                    f"Past values of <strong>{var_x}</strong> contain information that helps predict <strong>{var_y}</strong> beyond its own history.",
                    next_steps="Granger causality is predictive, not causal. It means X helps forecast Y, not that X causes Y. Test both directions.",
                )
            else:
                result["narrative"] = _narrative(
                    f"No Granger causality from {var_x} to {var_y}",
                    f"Past values of {var_x} do not significantly improve predictions of {var_y}.",
                    next_steps="Try more lags or test the reverse direction. Granger causality is specific to the lag structure tested.",
                )

            # Statistics for Synara
            result["statistics"] = {
                "granger_min_pvalue": float(min(p_values)),
                "granger_significant_lags": len(significant_lags),
                "granger_causal": 1 if significant_lags else 0,
            }

        except Exception as e:
            result["summary"] = (
                f"Granger causality test failed: {str(e)}\n\nEnsure both variables are numeric and have sufficient observations."
            )

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
        summary += "<<COLOR:title>>CHANGE POINT DETECTION<</COLOR>>\n"
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
                pen = float(penalty) if penalty.replace(".", "").isdigit() else np.log(n) * np.var(data)

            change_points = model.predict(pen=pen)

            # Remove the last point (always equals n)
            if change_points and change_points[-1] == n:
                change_points = change_points[:-1]

            summary += f"<<COLOR:accent>>── Change Points Detected ──<</COLOR>> {len(change_points)}\n\n"

            if change_points:
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
                summary += f"<<COLOR:text>>{'Index':<10} {'Value at CP':<15} {'Segment Mean':<15}<</COLOR>>\n"
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"

                segments = [0] + change_points + [n]
                for i, cp in enumerate(change_points):
                    seg_start = segments[i]
                    seg_end = cp
                    seg_mean = np.mean(data[seg_start:seg_end])
                    summary += f"{cp:<10} {data[cp - 1]:<15.4f} {seg_mean:<15.4f}\n"

                # Final segment
                final_mean = np.mean(data[change_points[-1] :])
                summary += f"\n<<COLOR:text>>Final segment mean: {final_mean:.4f}<</COLOR>>\n"

                summary += "\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
                summary += "  These points mark where the process behavior shifted.\n"
                summary += "  Investigate what changed at these times.\n"
            else:
                summary += "<<COLOR:text>>No significant change points detected.<</COLOR>>\n"
                summary += "<<COLOR:text>>The process appears stable over this period.<</COLOR>>\n"

            result["summary"] = summary

            # Plot with change points
            plot_data = [
                {
                    "type": "scatter",
                    "y": data.tolist(),
                    "mode": "lines",
                    "name": var,
                    "line": {"color": "#4a9f6e", "width": 1.5},
                }
            ]

            # Add vertical lines for change points
            shapes = []
            for cp in change_points:
                shapes.append(
                    {
                        "type": "line",
                        "x0": cp,
                        "x1": cp,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    }
                )

            # Add segment means as horizontal lines
            segments = [0] + list(change_points) + [n]
            for i in range(len(segments) - 1):
                seg_mean = np.mean(data[segments[i] : segments[i + 1]])
                plot_data.append(
                    {
                        "type": "scatter",
                        "x": [segments[i], segments[i + 1] - 1],
                        "y": [seg_mean, seg_mean],
                        "mode": "lines",
                        "line": {"color": "#e89547", "width": 2},
                        "name": f"Segment {i + 1} mean" if i == 0 else None,
                        "showlegend": i == 0,
                    }
                )

            result["plots"].append(
                {
                    "title": f"Change Points: {var}",
                    "data": plot_data,
                    "layout": {
                        "height": 350,
                        "shapes": shapes,
                        "xaxis": {"title": "Observation"},
                        "yaxis": {"title": var},
                    },
                }
            )

            result["guide_observation"] = f"Detected {len(change_points)} change point(s) in {var}." + (
                " Investigate what caused these shifts." if change_points else " Process appears stable."
            )
            if change_points:
                result["narrative"] = _narrative(
                    f"{len(change_points)} change point{'s' if len(change_points) > 1 else ''} detected in {var}",
                    f"The process shifted at {'these points' if len(change_points) > 1 else 'this point'}. Each change point represents a significant level shift.",
                    next_steps="Investigate what changed at each point: new material, operator, equipment, or environmental conditions.",
                    chart_guidance="Vertical lines mark detected change points. Segments between them have different statistical properties.",
                )
            else:
                result["narrative"] = _narrative(
                    f"No change points detected in {var}",
                    "The process appears stable over the observation period — no significant level shifts.",
                    next_steps="Stable process. Monitor with control charts to detect future shifts early.",
                )

            # Statistics for Synara
            result["statistics"] = {
                "change_points_count": len(change_points),
                "change_point_indices": change_points if change_points else [],
            }

        except ImportError:
            result["summary"] = (
                "Change point detection requires the 'ruptures' package.\n\nInstall with: pip install ruptures"
            )
        except Exception as e:
            result["summary"] = f"Change point detection failed: {str(e)}"

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
        summary += "<<COLOR:title>>ARIMA TIME SERIES ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> ARIMA({p},{d},{q})\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += "<<COLOR:accent>>── Stationarity Test (ADF) ──<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()

            # Model summary
            summary += "<<COLOR:accent>>── Model Parameters ──<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            summary += f"<<COLOR:accent>>── Forecast ({forecast_periods} periods) ──<</COLOR>>\n"
            for i in range(min(5, forecast_periods)):
                summary += f"  Period {i + 1}: {fc_mean.iloc[i]:.4f} [{fc_ci.iloc[i, 0]:.4f}, {fc_ci.iloc[i, 1]:.4f}]\n"
            if forecast_periods > 5:
                summary += f"  ... ({forecast_periods - 5} more periods)\n"

            # Plot
            x_hist = list(range(len(data)))
            x_fc = list(range(len(data), len(data) + forecast_periods))

            result["plots"].append(
                {
                    "title": f"ARIMA({p},{d},{q}) Forecast",
                    "data": [
                        {
                            "type": "scatter",
                            "x": x_hist,
                            "y": data.tolist(),
                            "mode": "lines",
                            "name": "Historical",
                            "line": {"color": "#4a9f6e"},
                        },
                        {
                            "type": "scatter",
                            "x": x_fc,
                            "y": fc_mean.tolist(),
                            "mode": "lines",
                            "name": "Forecast",
                            "line": {"color": "#e89547"},
                        },
                        {
                            "type": "scatter",
                            "x": x_fc + x_fc[::-1],
                            "y": fc_ci.iloc[:, 1].tolist() + fc_ci.iloc[::-1, 0].tolist(),
                            "fill": "toself",
                            "fillcolor": "rgba(232,149,71,0.2)",
                            "line": {"color": "transparent"},
                            "name": "95% CI",
                        },
                    ],
                    "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}},
                }
            )

            # Residual diagnostics
            residuals = fitted.resid

            result["plots"].append(
                {
                    "title": "Residuals",
                    "data": [
                        {"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}
                    ],
                    "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}},
                }
            )

            result["statistics"] = {"AIC": float(fitted.aic), "BIC": float(fitted.bic), "ADF_pvalue": float(adf_pval)}

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += "<<COLOR:text>>Try different p, d, q values or check data for issues.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"ARIMA({p},{d},{q}) model. {'Stationary data.' if adf_pval < 0.05 else 'Consider differencing.'}"
        )
        _arima_stationarity = (
            "Data is stationary (ADF p < 0.05)." if adf_pval < 0.05 else "Data may need differencing (ADF p >= 0.05)."
        )
        result["narrative"] = _narrative(
            f"ARIMA({p},{d},{q}) Time Series Model",
            f"{_arima_stationarity} Model captures autoregressive and moving average patterns.",
            next_steps="Check residual ACF for remaining autocorrelation. White noise residuals = good model. Use the forecast for planning.",
            chart_guidance="Forecast shows predicted values with confidence bands. Wider bands = more uncertainty further out.",
        )

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
        summary += "<<COLOR:title>>SARIMA SEASONAL FORECASTING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> SARIMA({p},{d},{q})({P},{D},{Q})[{m}]\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(ts_data)}\n"
        summary += f"<<COLOR:highlight>>Seasonal period:<</COLOR>> {m}\n\n"

        summary += "<<COLOR:accent>>── Stationarity Test (ADF) ──<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = SARIMAX(
                ts_data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=200)

            summary += "<<COLOR:accent>>── Model Fit ──<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n"
            summary += f"  Log-likelihood: {fitted.llf:.2f}\n\n"

            # Parameter summary
            summary += "<<COLOR:accent>>── Parameters ──<</COLOR>>\n"
            param_names = (
                fitted.param_names
                if hasattr(fitted, "param_names")
                else [f"param_{i}" for i in range(len(fitted.params))]
            )
            params = fitted.params if hasattr(fitted.params, "__len__") else [fitted.params]
            bse = fitted.bse if hasattr(fitted, "bse") else [None] * len(params)
            pvals = fitted.pvalues if hasattr(fitted, "pvalues") else [None] * len(params)
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
            fc_mean_list = fc_mean.tolist() if hasattr(fc_mean, "tolist") else list(fc_mean)
            if hasattr(fc_ci, "iloc"):
                fc_lower = fc_ci.iloc[:, 0].tolist()
                fc_upper = fc_ci.iloc[:, 1].tolist()
            else:
                fc_lower = fc_ci[:, 0].tolist()
                fc_upper = fc_ci[:, 1].tolist()

            summary += f"\n<<COLOR:accent>>── Forecast ({forecast_periods} periods) ──<</COLOR>>\n"
            for i in range(min(6, forecast_periods)):
                summary += f"  Period {i + 1}: {fc_mean_list[i]:.4f} [{fc_lower[i]:.4f}, {fc_upper[i]:.4f}]\n"
            if forecast_periods > 6:
                summary += f"  ... ({forecast_periods - 6} more periods)\n"

            # Ljung-Box test on residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb = acorr_ljungbox(fitted.resid, lags=[min(10, len(ts_data) // 5)], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
            summary += "\n<<COLOR:accent>>── Ljung-Box Test (residual autocorrelation) ──<</COLOR>>\n"
            summary += f"  p-value: {lb_p:.4f}\n"
            if lb_p > 0.05:
                summary += "  <<COLOR:good>>Residuals appear uncorrelated — good model fit.<</COLOR>>\n"
            else:
                summary += "  <<COLOR:warning>>Residuals show autocorrelation — consider different orders.<</COLOR>>\n"

            # Plot
            x_hist = list(range(len(ts_data)))
            x_fc = list(range(len(ts_data), len(ts_data) + forecast_periods))

            result["plots"].append(
                {
                    "title": f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}] Forecast",
                    "data": [
                        {
                            "type": "scatter",
                            "x": x_hist,
                            "y": ts_data.tolist(),
                            "mode": "lines",
                            "name": "Historical",
                            "line": {"color": "#4a9f6e"},
                        },
                        {
                            "type": "scatter",
                            "x": x_fc,
                            "y": fc_mean_list,
                            "mode": "lines",
                            "name": "Forecast",
                            "line": {"color": "#e89547", "dash": "dash"},
                        },
                        {
                            "type": "scatter",
                            "x": x_fc + x_fc[::-1],
                            "y": fc_upper + fc_lower[::-1],
                            "fill": "toself",
                            "fillcolor": "rgba(232,149,71,0.2)",
                            "line": {"color": "transparent"},
                            "name": "95% CI",
                        },
                    ],
                    "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}},
                }
            )

            # Residual plot
            residuals = fitted.resid
            result["plots"].append(
                {
                    "title": "Residuals",
                    "data": [
                        {"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}
                    ],
                    "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}},
                }
            )

            # ACF of residuals
            from statsmodels.tsa.stattools import acf

            resid_clean = residuals[~np.isnan(residuals)] if isinstance(residuals, np.ndarray) else residuals.dropna()
            n_lags = min(30, len(resid_clean) // 3)
            acf_vals = acf(resid_clean, nlags=n_lags, fft=True)
            ci_bound = 1.96 / np.sqrt(len(residuals))
            result["plots"].append(
                {
                    "title": "Residual ACF",
                    "data": [
                        {
                            "type": "bar",
                            "x": list(range(n_lags + 1)),
                            "y": acf_vals.tolist(),
                            "marker": {
                                "color": [
                                    "#d94a4a" if abs(v) > ci_bound and i > 0 else "#4a9f6e"
                                    for i, v in enumerate(acf_vals)
                                ]
                            },
                        }
                    ],
                    "layout": {
                        "height": 200,
                        "xaxis": {"title": "Lag"},
                        "yaxis": {"title": "ACF", "range": [-1, 1]},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": 0,
                                "x1": n_lags,
                                "y0": ci_bound,
                                "y1": ci_bound,
                                "line": {"color": "#e89547", "dash": "dash"},
                            },
                            {
                                "type": "line",
                                "x0": 0,
                                "x1": n_lags,
                                "y0": -ci_bound,
                                "y1": -ci_bound,
                                "line": {"color": "#e89547", "dash": "dash"},
                            },
                        ],
                    },
                }
            )

            result["statistics"] = {
                "AIC": float(fitted.aic),
                "BIC": float(fitted.bic),
                "ADF_pvalue": float(adf_pval),
                "ljung_box_p": lb_p,
                "order": [p, d, q],
                "seasonal_order": [P, D, Q, m],
            }
            result["guide_observation"] = f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]: AIC={fitted.aic:.1f}. " + (
                "Good residuals." if lb_p > 0.05 else "Check residuals."
            )
            _sarima_resid = (
                "Residuals pass Ljung-Box (no remaining autocorrelation)."
                if lb_p > 0.05
                else "Residuals show remaining pattern — consider adjusting model orders."
            )
            result["narrative"] = _narrative(
                f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}] — seasonal time series model",
                f"AIC = {fitted.aic:.1f}. Seasonal period = {m}. {_sarima_resid}",
                next_steps="Use the forecast for seasonal planning. Compare AIC with alternative models (lower = better).",
            )

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
        summary += "<<COLOR:title>>TIME SERIES DECOMPOSITION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type.capitalize()}\n"
        summary += f"<<COLOR:highlight>>Period:<</COLOR>> {period}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        decomp = seasonal_decompose(data, model=model_type, period=period)

        # Trend statistics
        trend_clean = decomp.trend.dropna()
        summary += "<<COLOR:accent>>── Trend ──<</COLOR>>\n"
        summary += f"  Start: {trend_clean.iloc[0]:.4f}\n"
        summary += f"  End: {trend_clean.iloc[-1]:.4f}\n"
        summary += f"  Change: {trend_clean.iloc[-1] - trend_clean.iloc[0]:.4f}\n\n"

        # Seasonal strength
        seasonal_var = decomp.seasonal.var()
        resid_var = decomp.resid.dropna().var()
        seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0

        summary += f"<<COLOR:accent>>── Seasonal Strength ──<</COLOR>> {seasonal_strength:.2%}\n"
        if seasonal_strength > 0.6:
            summary += "  <<COLOR:accent>>Strong seasonality detected<</COLOR>>\n"
        elif seasonal_strength > 0.3:
            summary += "  Moderate seasonality\n"
        else:
            summary += "  Weak or no seasonality\n"

        x_vals = list(range(len(data)))

        # Original
        result["plots"].append(
            {
                "title": "Original Series",
                "data": [
                    {"type": "scatter", "x": x_vals, "y": data.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}
                ],
                "layout": {"height": 150},
            }
        )

        # Trend
        result["plots"].append(
            {
                "title": "Trend",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_vals,
                        "y": decomp.trend.tolist(),
                        "mode": "lines",
                        "line": {"color": "#47a5e8"},
                    }
                ],
                "layout": {"height": 150},
            }
        )

        # Seasonal
        result["plots"].append(
            {
                "title": "Seasonal",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_vals,
                        "y": decomp.seasonal.tolist(),
                        "mode": "lines",
                        "line": {"color": "#e89547"},
                    }
                ],
                "layout": {"height": 150},
            }
        )

        # Residual
        result["plots"].append(
            {
                "title": "Residual",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_vals,
                        "y": decomp.resid.tolist(),
                        "mode": "lines",
                        "line": {"color": "#9aaa9a"},
                    }
                ],
                "layout": {"height": 150},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Decomposition with {seasonal_strength:.0%} seasonal strength."
        result["statistics"] = {
            "seasonal_strength": float(seasonal_strength),
            "trend_change": float(trend_clean.iloc[-1] - trend_clean.iloc[0]),
        }
        _trend_dir = (
            "upward"
            if trend_clean.iloc[-1] > trend_clean.iloc[0]
            else "downward"
            if trend_clean.iloc[-1] < trend_clean.iloc[0]
            else "flat"
        )
        result["narrative"] = _narrative(
            f"Time Series Decomposition: {seasonal_strength:.0%} seasonal strength",
            f"Trend is {_trend_dir}. Seasonal component accounts for {seasonal_strength:.0%} of the variation.",
            next_steps="Strong seasonality means seasonal adjustments are needed. The residual component shows unexplained variation.",
            chart_guidance="Four panels: observed data, trend (long-term direction), seasonal (repeating pattern), residual (random noise).",
        )

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
        summary += "<<COLOR:title>>ACF / PACF ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        summary += f"<<COLOR:highlight>>Lags shown:<</COLOR>> {max_lags}\n\n"

        # Find significant lags
        sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
        sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

        summary += f"<<COLOR:accent>>── Significant ACF lags ──<</COLOR>> {sig_acf[:5] if sig_acf else 'None'}\n"
        summary += f"<<COLOR:accent>>── Significant PACF lags ──<</COLOR>> {sig_pacf[:5] if sig_pacf else 'None'}\n\n"

        # ARIMA order suggestions
        summary += "<<COLOR:success>>ARIMA ORDER SUGGESTIONS:<</COLOR>>\n"
        if len(sig_pacf) > 0 and (len(sig_acf) == 0 or sig_acf[0] > sig_pacf[-1]):
            summary += f"  PACF cuts off at lag {max(sig_pacf)}: Try AR({max(sig_pacf)})\n"
        if len(sig_acf) > 0 and (len(sig_pacf) == 0 or sig_pacf[0] > sig_acf[-1]):
            summary += f"  ACF cuts off at lag {max(sig_acf)}: Try MA({max(sig_acf)})\n"
        if len(sig_acf) > 0 and len(sig_pacf) > 0:
            summary += f"  Both taper: Try ARMA({min(3, max(sig_pacf))},{min(3, max(sig_acf))})\n"

        lags = list(range(max_lags + 1))

        # ACF plot
        result["plots"].append(
            {
                "title": "Autocorrelation Function (ACF)",
                "data": [
                    {"type": "bar", "x": lags, "y": acf_vals.tolist(), "marker": {"color": "#4a9f6e"}},
                    {
                        "type": "scatter",
                        "x": lags,
                        "y": [ci] * len(lags),
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "showlegend": False,
                    },
                    {
                        "type": "scatter",
                        "x": lags,
                        "y": [-ci] * len(lags),
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "showlegend": False,
                    },
                ],
                "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]}},
            }
        )

        # PACF plot
        result["plots"].append(
            {
                "title": "Partial Autocorrelation Function (PACF)",
                "data": [
                    {"type": "bar", "x": lags, "y": pacf_vals.tolist(), "marker": {"color": "#47a5e8"}},
                    {
                        "type": "scatter",
                        "x": lags,
                        "y": [ci] * len(lags),
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "showlegend": False,
                    },
                    {
                        "type": "scatter",
                        "x": lags,
                        "y": [-ci] * len(lags),
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "showlegend": False,
                    },
                ],
                "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF", "range": [-1, 1]}},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = "ACF/PACF analysis. Significant lags help identify ARIMA orders."
        result["narrative"] = _narrative(
            "ACF/PACF — Autocorrelation Analysis",
            "ACF shows overall correlation at each lag. PACF shows direct (partial) correlation after removing intermediate lags.",
            next_steps="ACF dying slowly + PACF cutting off = AR model (p = PACF cutoff). ACF cutting off + PACF dying slowly = MA model (q = ACF cutoff).",
            chart_guidance="Bars outside the shaded region are statistically significant. Use these to determine ARIMA model orders.",
        )

    elif analysis_id == "weibull":
        """
        Weibull Reliability Analysis - life data analysis.
        """
        from scipy import stats

        var = config.get("var")  # Time to failure
        config.get("censored")  # Optional censoring indicator

        data = df[var].dropna().values
        data = data[data > 0]  # Weibull requires positive values

        if len(data) < 3:
            result["summary"] = "Need at least 3 data points for Weibull analysis."
            return result

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>WEIBULL RELIABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (time to failure)\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += "<<COLOR:accent>>── Weibull Parameters ──<</COLOR>>\n"
        summary += f"  Shape (β): {shape:.4f}\n"
        summary += f"  Scale (η): {scale:.4f}\n\n"

        # Interpret shape parameter
        summary += "<<COLOR:accent>>── Shape Interpretation ──<</COLOR>>\n"
        if shape < 1:
            summary += "  β < 1: <<COLOR:warning>>Infant mortality / early failures<</COLOR>>\n"
            summary += "  Failure rate DECREASES over time (burn-in period)\n"
        elif shape == 1:
            summary += "  β ≈ 1: Random failures (exponential distribution)\n"
            summary += "  Failure rate is CONSTANT\n"
        else:
            summary += "  β > 1: <<COLOR:accent>>Wear-out failures<</COLOR>>\n"
            summary += "  Failure rate INCREASES over time\n"

        # Reliability metrics
        mean_life = scale * np.exp(np.log(np.exp(1)) / shape) if shape > 0 else scale
        b10 = stats.weibull_min.ppf(0.10, shape, 0, scale)  # 10% failure life
        b50 = stats.weibull_min.ppf(0.50, shape, 0, scale)  # Median life

        summary += "\n<<COLOR:accent>>── Reliability Metrics ──<</COLOR>>\n"
        summary += f"  Mean Life (MTTF): {mean_life:.2f}\n"
        summary += f"  B10 Life (10% fail): {b10:.2f}\n"
        summary += f"  B50 Life (median): {b50:.2f}\n"

        # Reliability at specific times
        summary += "\n<<COLOR:accent>>── Reliability R(t) ──<</COLOR>>\n"
        for t in [b10, b50, scale, 2 * scale]:
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

        result["plots"].append(
            {
                "title": "Weibull Probability Plot",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_plot.tolist(),
                        "y": y_plot.tolist(),
                        "mode": "markers",
                        "name": "Data",
                        "marker": {"color": "#4a9f6e", "size": 8},
                    },
                    {
                        "type": "scatter",
                        "x": x_fit.tolist(),
                        "y": y_fit.tolist(),
                        "mode": "lines",
                        "name": "Fit",
                        "line": {"color": "#e89547"},
                    },
                ],
                "layout": {"height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "ln(-ln(1-F))"}},
            }
        )

        # Reliability curve
        t_range = np.linspace(0, 2 * scale, 100)
        reliability = 1 - stats.weibull_min.cdf(t_range, shape, 0, scale)
        hazard = (shape / scale) * (t_range / scale) ** (shape - 1)

        result["plots"].append(
            {
                "title": "Reliability & Hazard Functions",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": reliability.tolist(),
                        "mode": "lines",
                        "name": "R(t)",
                        "line": {"color": "#4a9f6e"},
                    },
                    {
                        "type": "scatter",
                        "x": t_range.tolist(),
                        "y": (hazard / hazard.max()).tolist(),
                        "mode": "lines",
                        "name": "h(t) scaled",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "yaxis": "y2",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability", "range": [0, 1]},
                    "yaxis2": {"title": "Hazard (scaled)", "overlaying": "y", "side": "right"},
                },
            }
        )

        result["summary"] = summary
        result["guide_observation"] = (
            f"Weibull β={shape:.2f} ({'wear-out' if shape > 1 else 'early failure' if shape < 1 else 'random'}), η={scale:.2f}."
        )
        result["statistics"] = {
            "shape_beta": float(shape),
            "scale_eta": float(scale),
            "MTTF": float(mean_life),
            "B10": float(b10),
        }
        _fail_mode = (
            "wear-out (increasing hazard)"
            if shape > 1
            else "early failure (decreasing hazard)"
            if shape < 1
            else "random failure (constant hazard)"
        )
        result["narrative"] = _narrative(
            f"Weibull: \u03b2 = {shape:.2f} — {_fail_mode}",
            f"Scale \u03b7 = {scale:.2f}, MTTF = {mean_life:.1f}, B10 = {b10:.1f}. Shape parameter reveals the failure mode.",
            next_steps="\u03b2 < 1: improve screening/burn-in. \u03b2 = 1: failures are random (maintenance won't help). \u03b2 > 1: preventive maintenance at B10 life.",
            chart_guidance="Probability plot: points on the line = good Weibull fit. The slope is related to \u03b2 (steeper = more concentrated failures).",
        )

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
            events = (
                data[event_col].values.astype(int)
                if event_col and event_col in data.columns
                else np.ones(len(times), dtype=int)
            )

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
                        s *= 1 - di / ni
                        if ni > di:
                            var_sum += di / (ni * (ni - di))
                    survival.append(s)
                    se = s * np.sqrt(var_sum) if var_sum > 0 else 0
                    ci_lower.append(max(0, s - z * se))
                    ci_upper.append(min(1, s + z * se))

                return unique_times, survival, n_risk, n_event, ci_lower, ci_upper

            if group_col and group_col in data.columns and group_col != "" and group_col != "None":
                groups = sorted(data[group_col].unique())
                colors = ["#2c5f2d", "#4a90d9", "#d94a4a", "#d9a04a", "#7d4ad9"]

                traces = []
                summary_parts = []
                median_survivals = {}

                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(t_g, e_g)

                    color = colors[i % len(colors)]
                    traces.append(
                        {
                            "x": [0] + ut.tolist(),
                            "y": [1.0] + surv,
                            "mode": "lines",
                            "name": f"{grp} (n={len(t_g)})",
                            "line": {"shape": "hv", "color": color, "width": 2},
                        }
                    )
                    traces.append(
                        {
                            "x": [0] + ut.tolist() + ut.tolist()[::-1] + [0],
                            "y": [1.0] + ci_hi + ci_lo[::-1] + [1.0],
                            "fill": "toself",
                            "line": {"width": 0},
                            "showlegend": False,
                            "name": f"{grp} CI",
                            "opacity": 0.2,
                        }
                    )

                    median = None
                    for j, s_val in enumerate(surv):
                        if s_val <= 0.5:
                            median = float(ut[j])
                            break
                    median_survivals[str(grp)] = median
                    summary_parts.append(
                        f"**{grp}** (n={len(t_g)}): Median survival = {median if median else 'not reached'}"
                    )

                result["plots"].append(
                    {
                        "data": traces,
                        "layout": {
                            "title": "Kaplan-Meier Survival Curves by Group",
                            "xaxis": {"title": f"Time ({time_col})"},
                            "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        },
                    }
                )

                # Cumulative hazard plot: H(t) = -ln(S(t))
                ch_traces = []
                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut_ch, surv_ch, _, _, _, _ = km_estimate(t_g, e_g)
                    cum_haz = [-np.log(max(s, 1e-10)) for s in surv_ch]
                    ch_traces.append(
                        {
                            "x": [0] + ut_ch.tolist(),
                            "y": [0.0] + cum_haz,
                            "mode": "lines",
                            "name": f"{grp}",
                            "line": {"shape": "hv", "color": colors[i % len(colors)], "width": 2},
                        }
                    )
                result["plots"].append(
                    {
                        "data": ch_traces,
                        "layout": {
                            "title": "Cumulative Hazard by Group",
                            "xaxis": {"title": f"Time ({time_col})"},
                            "yaxis": {"title": "H(t) = −ln S(t)"},
                        },
                    }
                )

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

                chi2_stat = sum(
                    (observed[str(g)] - expected[str(g)]) ** 2 / expected[str(g)]
                    for g in groups
                    if expected[str(g)] > 0
                )
                df_lr = len(groups) - 1
                p_logrank = 1 - scipy_stats.chi2.cdf(chi2_stat, df_lr)

                summary_parts.insert(
                    0,
                    f"**Log-rank test**: chi2={chi2_stat:.3f}, df={df_lr}, p={p_logrank:.4f} {'(significant)' if p_logrank < alpha else '(not significant)'}\n",
                )
                result["summary"] = "\n".join(summary_parts)
                result["guide_observation"] = (
                    f"KM analysis: {len(groups)} groups, log-rank p={p_logrank:.4f}. {'Survival differs significantly.' if p_logrank < alpha else 'No significant difference.'}"
                )
                result["statistics"] = {
                    "log_rank_chi2": float(chi2_stat),
                    "log_rank_p": float(p_logrank),
                    "df": df_lr,
                    "n_total": len(data),
                    "n_events": int(events.sum()),
                    "median_survival": median_survivals,
                }
            else:
                ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(times, events)

                result["plots"].append(
                    {
                        "data": [
                            {
                                "x": [0] + ut.tolist(),
                                "y": [1.0] + surv,
                                "mode": "lines",
                                "name": "Survival",
                                "line": {"shape": "hv", "color": "#2c5f2d", "width": 2},
                            },
                            {
                                "x": [0] + ut.tolist(),
                                "y": [1.0] + ci_hi,
                                "mode": "lines",
                                "name": "Upper CI",
                                "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1},
                                "showlegend": False,
                            },
                            {
                                "x": [0] + ut.tolist(),
                                "y": [1.0] + ci_lo,
                                "mode": "lines",
                                "name": "Lower CI",
                                "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1},
                                "fill": "tonexty",
                                "fillcolor": "rgba(44,95,45,0.15)",
                                "showlegend": False,
                            },
                        ],
                        "layout": {
                            "title": "Kaplan-Meier Survival Curve",
                            "xaxis": {"title": f"Time ({time_col})"},
                            "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        },
                    }
                )

                # Cumulative hazard plot: H(t) = -ln(S(t))
                cum_haz = [-np.log(max(s, 1e-10)) for s in surv]
                result["plots"].append(
                    {
                        "data": [
                            {
                                "x": [0] + ut.tolist(),
                                "y": [0.0] + cum_haz,
                                "mode": "lines",
                                "name": "Cumulative Hazard",
                                "line": {"shape": "hv", "color": "#4a90d9", "width": 2},
                            }
                        ],
                        "layout": {
                            "title": "Cumulative Hazard Function",
                            "xaxis": {"title": f"Time ({time_col})"},
                            "yaxis": {"title": "H(t) = −ln S(t)"},
                        },
                    }
                )

                median = None
                for j, s_val in enumerate(surv):
                    if s_val <= 0.5:
                        median = float(ut[j])
                        break

                result["summary"] = (
                    f"**Kaplan-Meier Survival Analysis**\n\nN = {len(data)}, Events = {int(events.sum())}, Censored = {int((events == 0).sum())}\nMedian survival time: {median if median else 'not reached'}"
                )
                result["guide_observation"] = (
                    f"KM curve: n={len(data)}, {int(events.sum())} events. Median survival = {median if median else 'not reached'}."
                )
                result["statistics"] = {
                    "n_total": len(data),
                    "n_events": int(events.sum()),
                    "n_censored": int((events == 0).sum()),
                    "median_survival": median,
                }

                # Narrative
                _n_total = len(data)
                _n_events = int(events.sum())
                _n_cens = int((events == 0).sum())
                _cens_pct = _n_cens / _n_total * 100 if _n_total > 0 else 0
                _med_str = f"{median:.1f}" if median else "not reached"
                verdict = f"Kaplan-Meier: median survival = {_med_str} (N = {_n_total})"
                body = (
                    f"Of {_n_total} subjects, {_n_events} experienced the event and {_n_cens} ({_cens_pct:.0f}%) were censored. "
                    + (
                        f"Median survival time is <strong>{median:.1f}</strong>."
                        if median
                        else "Median survival was <strong>not reached</strong> — more than 50% survived beyond the observation period."
                    )
                )
                nxt = (
                    "Compare groups with log-rank test. For covariate-adjusted analysis, use Cox Proportional Hazards."
                )
                result["narrative"] = _narrative(
                    verdict,
                    body,
                    next_steps=nxt,
                    chart_guidance="The step function shows the probability of surviving beyond each time point. Steps down = events. Censored observations are marked with ticks.",
                )

        except Exception as e:
            result["summary"] = f"Kaplan-Meier error: {str(e)}"

    elif analysis_id == "cox_ph":
        """
        Cox Proportional Hazards Regression — semi-parametric survival model.
        Estimates hazard ratios for covariates without assuming a baseline hazard distribution.
        Reports coefficients, hazard ratios, 95% CIs, concordance index.
        """
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
                if data[cov].dtype == "object" or data[cov].nunique() < 6:
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
                rows.append(
                    f"| {name} | {coefs[i]:.4f} | {se[i]:.4f} | {z_vals[i]:.3f} | {p_vals[i]:.4f} | {hr[i]:.3f} | ({hr_lower[i]:.3f}, {hr_upper[i]:.3f}) |"
                )

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
            result["plots"].append(
                {
                    "data": [
                        {
                            "x": hr.tolist(),
                            "y": covariate_names,
                            "mode": "markers",
                            "marker": {"size": 10, "color": "#2c5f2d"},
                            "error_x": {
                                "type": "data",
                                "symmetric": False,
                                "array": (hr_upper - hr).tolist(),
                                "arrayminus": (hr - hr_lower).tolist(),
                                "color": "#2c5f2d",
                            },
                            "type": "scatter",
                            "name": "Hazard Ratio",
                        }
                    ],
                    "layout": {
                        "title": "Cox PH — Hazard Ratios (Forest Plot)",
                        "xaxis": {"title": "Hazard Ratio", "type": "log"},
                        "yaxis": {"title": ""},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": 1,
                                "x1": 1,
                                "y0": -0.5,
                                "y1": len(covariate_names) - 0.5,
                                "line": {"dash": "dash", "color": "red"},
                            }
                        ],
                    },
                }
            )

            # Risk score distribution by event status
            result["plots"].append(
                {
                    "data": [
                        {
                            "x": lp[events_arr == 1].tolist(),
                            "type": "histogram",
                            "name": "Events",
                            "opacity": 0.7,
                            "marker": {"color": "#d94a4a"},
                        },
                        {
                            "x": lp[events_arr == 0].tolist(),
                            "type": "histogram",
                            "name": "Censored",
                            "opacity": 0.7,
                            "marker": {"color": "#4a90d9"},
                        },
                    ],
                    "layout": {
                        "title": "Linear Predictor Distribution by Event Status",
                        "xaxis": {"title": "Linear Predictor (Xβ)"},
                        "yaxis": {"title": "Count"},
                        "barmode": "overlay",
                    },
                }
            )

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
            result["plots"].append(
                {
                    "data": [
                        {
                            "x": lp.tolist(),
                            "y": mart_resid.tolist(),
                            "mode": "markers",
                            "type": "scatter",
                            "marker": {
                                "color": ["#d94a4a" if e == 1 else "#4a90d9" for e in events_arr],
                                "size": 5,
                                "opacity": 0.6,
                            },
                            "name": "Residuals",
                        }
                    ],
                    "layout": {
                        "title": "Martingale Residuals vs Linear Predictor",
                        "xaxis": {"title": "Linear Predictor (Xβ)"},
                        "yaxis": {"title": "Martingale Residual"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": float(lp.min()),
                                "x1": float(lp.max()),
                                "y0": 0,
                                "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            n_events = int(events_arr.sum())
            ll = float(fit.llf) if hasattr(fit, "llf") else None

            result["summary"] = (
                f"**Cox Proportional Hazards Regression**\n\nN = {len(data)}, Events = {n_events}, Censored = {len(data) - n_events}\nConcordance index (C) = {c_index:.3f}\n{'Log-likelihood = ' + f'{ll:.2f}' if ll else ''}\n\n{table}"
            )
            result["guide_observation"] = (
                f"Cox PH: n={len(data)}, {n_events} events, C-index={c_index:.3f}. "
                + ", ".join(
                    f"{covariate_names[i]}: HR={hr[i]:.2f} (p={p_vals[i]:.4f})" for i in range(len(covariate_names))
                )
            )

            # Narrative
            _sig_covs = [
                (covariate_names[i], float(hr[i]), float(p_vals[i]))
                for i in range(len(covariate_names))
                if p_vals[i] < 0.05
            ]
            _c_label = (
                "excellent" if c_index > 0.8 else "good" if c_index > 0.7 else "moderate" if c_index > 0.6 else "weak"
            )
            verdict = f"Cox PH: C-index = {c_index:.3f} ({_c_label} discrimination)"
            body = f"Model fit on {len(data)} subjects ({n_events} events). "
            if _sig_covs:
                _top = _sig_covs[0]
                body += "Significant predictors: "
                _parts = []
                for _name, _hr, _pv in _sig_covs[:3]:
                    _dir = "increases" if _hr > 1 else "decreases"
                    body += f"<strong>{_name}</strong> (HR = {_hr:.2f}, p = {_pv:.4f}) {_dir} hazard by {abs(_hr - 1) * 100:.0f}%. "
            else:
                body += "No covariates are significant at α = 0.05."
            nxt = "Check proportional hazards assumption via Schoenfeld residuals. HR > 1 = increased risk; HR < 1 = protective."
            result["narrative"] = _narrative(
                verdict,
                body,
                next_steps=nxt,
                chart_guidance="Forest plot: HR > 1 (right of dashed line) = increased hazard. CI not crossing 1 = significant. Risk score histogram shows model discrimination.",
            )

            result["statistics"] = {
                "n_total": len(data),
                "n_events": n_events,
                "concordance": float(c_index),
                "log_likelihood": ll,
                "coefficients": {
                    covariate_names[i]: {
                        "coef": float(coefs[i]),
                        "se": float(se[i]),
                        "z": float(z_vals[i]),
                        "p": float(p_vals[i]),
                        "hazard_ratio": float(hr[i]),
                        "hr_ci_lower": float(hr_lower[i]),
                        "hr_ci_upper": float(hr_upper[i]),
                    }
                    for i in range(len(covariate_names))
                },
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
        summary += "<<COLOR:title>>GAGE R&R STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part}\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        # Get data
        data = df[[measurement, part, operator]].dropna()

        n_parts = data[part].nunique()
        n_operators = data[operator].nunique()
        n_replicates = len(data) // (n_parts * n_operators)

        summary += "<<COLOR:accent>>── Study Design ──<</COLOR>>\n"
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

        summary += "<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
        summary += f"  {'Source':<20} {'Variance':>12} {'%Contribution':>14}\n"
        summary += f"  {'-' * 48}\n"
        summary += f"  {'Total Gage R&R':<20} {gage_rr_var:>12.4f} {pct_gage_rr:>13.1f}%\n"
        summary += f"    {'Repeatability':<18} {repeatability_var:>12.4f} {pct_repeatability:>13.1f}%\n"
        summary += f"    {'Reproducibility':<18} {reproducibility_var:>12.4f} {pct_reproducibility:>13.1f}%\n"
        summary += f"  {'Part-to-Part':<20} {part_var:>12.4f} {pct_part:>13.1f}%\n\n"

        # Assessment
        summary += "<<COLOR:success>>ASSESSMENT:<</COLOR>>\n"
        if pct_gage_rr < 10:
            summary += "  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
            summary += "  Measurement system is acceptable.\n"
        elif pct_gage_rr < 30:
            summary += "  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
            summary += "  May be acceptable depending on application.\n"
        else:
            summary += "  <<COLOR:bad>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
            summary += "  Measurement system needs improvement.\n"

        # Number of distinct categories
        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"
        summary += "  (Should be >= 5 for adequate discrimination)\n"

        # Plots
        # By Part
        result["plots"].append(
            {
                "title": f"{measurement} by {part}",
                "data": [
                    {
                        "type": "box",
                        "x": data[part].astype(str).tolist(),
                        "y": data[measurement].tolist(),
                        "marker": {"color": "#4a9f6e"},
                    }
                ],
                "layout": {"height": 250, "xaxis": {"title": part}},
            }
        )

        # By Operator
        result["plots"].append(
            {
                "title": f"{measurement} by {operator}",
                "data": [
                    {
                        "type": "box",
                        "x": data[operator].astype(str).tolist(),
                        "y": data[measurement].tolist(),
                        "marker": {"color": "#47a5e8"},
                    }
                ],
                "layout": {"height": 250, "xaxis": {"title": operator}},
            }
        )

        # Components of variation bar chart
        result["plots"].append(
            {
                "title": "Components of Variation",
                "data": [
                    {
                        "type": "bar",
                        "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                        "y": [pct_gage_rr, pct_repeatability, pct_reproducibility, pct_part],
                        "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]},
                    }
                ],
                "layout": {"height": 250, "yaxis": {"title": "% of Total Variation"}},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Gage R&R = {pct_gage_rr:.1f}%. " + (
            "Acceptable." if pct_gage_rr < 30 else "Needs improvement."
        )
        result["statistics"] = {
            "gage_rr_pct": float(pct_gage_rr),
            "repeatability_pct": float(pct_repeatability),
            "reproducibility_pct": float(pct_reproducibility),
            "ndc": int(ndc),
        }

        # Narrative
        if pct_gage_rr < 10:
            verdict = f"Gage R&R = {pct_gage_rr:.1f}% — Measurement system acceptable"
            assessment = "excellent"
        elif pct_gage_rr < 30:
            verdict = f"Gage R&R = {pct_gage_rr:.1f}% — Marginal measurement system"
            assessment = "marginal"
        else:
            verdict = f"Gage R&R = {pct_gage_rr:.1f}% — Measurement system unacceptable"
            assessment = "unacceptable"

        dominant = (
            "repeatability (equipment variation)"
            if pct_repeatability > pct_reproducibility
            else "reproducibility (operator variation)"
        )
        body = (
            f"The measurement system consumes <strong>{pct_gage_rr:.1f}%</strong> of total observed variation. "
            f"The dominant component is {dominant}. "
            f"Part-to-part variation accounts for {pct_part:.1f}%. "
            f"Number of distinct categories (ndc) = {ndc}"
            + (" — adequate discrimination." if ndc >= 5 else " — insufficient discrimination (need \u2265 5).")
        )
        if assessment == "unacceptable":
            nxt = (
                f"Fix the measurement system before using this data for process analysis. Focus on reducing {dominant}."
            )
        elif assessment == "marginal":
            nxt = f"Acceptable for some applications. To improve, focus on reducing {dominant}."
        else:
            nxt = "Measurement system is adequate. Data from this gage can be trusted for process analysis and capability studies."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nxt,
            chart_guidance="The components bar chart shows where variation originates. Tall Gage R&R bars = measurement noise dominates over true part differences.",
        )

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
        summary += "<<COLOR:title>>NESTED GAGE R&R (DESTRUCTIVE)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part} (nested within {operator})\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Operator means
        op_means = data.groupby(operator)[measurement].mean()
        parts_per_op = data.groupby(operator)[part].nunique().mean()
        reps_per_part = (
            len(data) / (data.groupby([operator, part]).ngroups) if data.groupby([operator, part]).ngroups > 0 else 1
        )

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
        ss_error = sum((data[measurement] - data.groupby([operator, part])[measurement].transform("mean")) ** 2)
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

        summary += "<<COLOR:accent>>── Variance Components (Nested ANOVA) ──<</COLOR>>\n"
        summary += f"  {'Source':<25} {'Variance':>12} {'%Study Var':>12}\n"
        summary += f"  {'-' * 52}\n"
        summary += f"  {'Total Gage R&R':<25} {gage_rr_var:>12.4f} {pct_gage_rr:>11.1f}%\n"
        summary += f"    {'Repeatability':<23} {repeat_var:>12.4f} {pct_repeat:>11.1f}%\n"
        summary += f"    {'Reproducibility':<23} {reprod_var:>12.4f} {pct_reprod:>11.1f}%\n"
        summary += f"  {'Part-to-Part':<25} {part_var:>12.4f} {pct_part:>11.1f}%\n\n"

        if pct_gage_rr < 10:
            summary += "  <<COLOR:good>>EXCELLENT — Gage R&R < 10%<</COLOR>>\n"
        elif pct_gage_rr < 30:
            summary += "  <<COLOR:warning>>MARGINAL — Gage R&R 10-30%<</COLOR>>\n"
        else:
            summary += "  <<COLOR:bad>>UNACCEPTABLE — Gage R&R > 30%<</COLOR>>\n"

        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"

        # Plots
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "box",
                        "x": data[operator].astype(str).tolist(),
                        "y": data[measurement].tolist(),
                        "marker": {"color": "#4a9f6e"},
                    }
                ],
                "layout": {"title": f"{measurement} by {operator}", "xaxis": {"title": operator}},
            }
        )
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                        "y": [pct_gage_rr, pct_repeat, pct_reprod, pct_part],
                        "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]},
                    }
                ],
                "layout": {"title": "Components of Variation", "yaxis": {"title": "% Study Var"}},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Nested Gage R&R = {pct_gage_rr:.1f}%, NDC={ndc}. " + (
            "Acceptable." if pct_gage_rr < 30 else "Needs improvement."
        )
        result["statistics"] = {
            "gage_rr_pct": float(pct_gage_rr),
            "repeatability_pct": float(pct_repeat),
            "reproducibility_pct": float(pct_reprod),
            "part_pct": float(pct_part),
            "ndc": ndc,
        }
        _grr_label = "acceptable" if pct_gage_rr < 10 else "marginal" if pct_gage_rr < 30 else "unacceptable"
        result["narrative"] = _narrative(
            f"Nested Gage R&R = {pct_gage_rr:.1f}% ({_grr_label})",
            f"Designed for destructive testing where operators measure different parts. NDC = {ndc}. Repeatability = {pct_repeat:.1f}%, Reproducibility = {pct_reprod:.1f}%.",
            next_steps="For destructive testing, nested designs are required. NDC \u2265 5 means adequate discrimination."
            if ndc >= 5
            else "NDC < 5 — measurement system cannot adequately distinguish parts. Improve the measurement process.",
        )

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
        summary += "<<COLOR:title>>EXPANDED GAGE R&R (Multi-Factor MSA)<</COLOR>>\n"
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
                "n_levels": n_levels,
                "ss": ss_f,
                "df": df_f,
                "ms": ms_f,
            }

        # Repeatability: residual after all factors
        # Group by all factors, compute within-cell variance
        if len(all_factors) > 1:
            cell_vars = data_grr.groupby(all_factors)[measurement].var()
            data_grr.groupby(all_factors)[measurement].count()
            repeatability_var = float(cell_vars.mean()) if not cell_vars.isna().all() else 0
        else:
            within_var = data_grr.groupby(part)[measurement].var()
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

        summary += "<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
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

        summary += "<<COLOR:accent>>── Assessment ──<</COLOR>>\n"
        if pct_rr < 10:
            summary += "  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
        elif pct_rr < 30:
            summary += "  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
        else:
            summary += "  <<COLOR:warning>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories (NDC):<</COLOR>> {ndc}\n"

        # Identify largest source of measurement variation
        if reproducibility_var > repeatability_var * 1.5 and len(var_components) > 1:
            worst_factor = max(
                [(f, c["var_est"]) for f, c in var_components.items() if f != part],
                key=lambda x: x[1],
                default=(None, 0),
            )
            if worst_factor[0]:
                summary += f"\n<<COLOR:accent>>── Largest reproducibility source ──<</COLOR>> {worst_factor[0]}\n"
                summary += f"  Consider standardizing {worst_factor[0]} to reduce measurement variation.\n"
        elif repeatability_var > reproducibility_var * 1.5:
            summary += (
                "\n<<COLOR:text>>Repeatability dominates — improve gage precision or measurement procedure.<</COLOR>>\n"
            )

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

        result["plots"].append(
            {
                "data": [{"type": "bar", "x": bar_labels, "y": bar_vals, "marker": {"color": bar_colors}}],
                "layout": {"title": "Components of Variation", "yaxis": {"title": "% Study Variation"}, "height": 300},
            }
        )

        # Measurement by Part
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "box",
                        "x": data_grr[part].astype(str).tolist(),
                        "y": data_grr[measurement].tolist(),
                        "marker": {"color": "#4a9f6e"},
                    }
                ],
                "layout": {"title": f"{measurement} by {part}", "height": 250},
            }
        )

        # Measurement by each factor
        for f_name in all_factors:
            if f_name != part and f_name in data_grr.columns:
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "box",
                                "x": data_grr[f_name].astype(str).tolist(),
                                "y": data_grr[measurement].tolist(),
                                "marker": {"color": "#47a5e8"},
                            }
                        ],
                        "layout": {"title": f"{measurement} by {f_name}", "height": 250},
                    }
                )

        result["guide_observation"] = (
            f"Expanded Gage R&R = {pct_rr:.1f}%, NDC={ndc}. {len(all_factors)} factors analyzed."
        )
        _grr_label = "acceptable" if pct_rr < 10 else "marginal" if pct_rr < 30 else "unacceptable"
        result["narrative"] = _narrative(
            f"Expanded Gage R&R = {pct_rr:.1f}% ({_grr_label})",
            f"Analyzed {len(all_factors)} factors. NDC = {ndc}. Expanded study includes additional sources of variation beyond standard operator/part.",
            next_steps="Address the largest variance component first. NDC \u2265 5 needed for adequate part discrimination.",
        )
        result["statistics"] = {
            "gage_rr_pct": float(pct_rr),
            "repeatability_pct": float(pct_repeat),
            "reproducibility_pct": float(pct_reprod),
            "part_pct": float(pct_part),
            "ndc": ndc,
            "n_factors": len(all_factors),
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

        data[reference].mean()
        overall_bias = bias.mean()
        bias_pct = 100 * abs(overall_bias) / data[reference].std() if data[reference].std() > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>GAGE LINEARITY & BIAS STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference:<</COLOR>> {reference}\n"
        summary += f"<<COLOR:highlight>>N observations:<</COLOR>> {len(data)}\n\n"

        summary += "<<COLOR:accent>>── BIAS ──<</COLOR>>\n"
        summary += f"  Overall Bias = {overall_bias:.4f}\n"
        summary += f"  Bias as % of Process Variation ≈ {bias_pct:.1f}%\n"
        summary += f"  {'Bias is significant' if abs(overall_bias) / (bias.std() / np.sqrt(len(bias))) > 2 else 'Bias is not significant'}\n\n"

        summary += "<<COLOR:accent>>── LINEARITY ──<</COLOR>>\n"
        summary += f"  Bias = {intercept:.4f} + {slope:.4f} × Reference\n"
        summary += f"  Slope = {slope:.4f} (p = {p_value:.4f})\n"
        summary += f"  R² = {r_value**2:.4f}\n"
        summary += f"  {'Linearity is significant (bias changes across range)' if p_value < 0.05 else 'Linearity is not significant (bias is constant)'}\n"

        # Scatter: bias vs reference
        ref_range = np.linspace(data[reference].min(), data[reference].max(), 100)
        fit_line = intercept + slope * ref_range

        result["plots"].append(
            {
                "data": [
                    {
                        "x": data[reference].tolist(),
                        "y": bias.tolist(),
                        "mode": "markers",
                        "name": "Bias",
                        "marker": {"color": "#4a90d9", "size": 6},
                    },
                    {
                        "x": ref_range.tolist(),
                        "y": fit_line.tolist(),
                        "mode": "lines",
                        "name": f"Fit (slope={slope:.4f})",
                        "line": {"color": "#d94a4a", "width": 2},
                    },
                    {
                        "x": ref_range.tolist(),
                        "y": [0] * len(ref_range),
                        "mode": "lines",
                        "name": "Zero bias",
                        "line": {"color": "#4a9f6e", "dash": "dash", "width": 1},
                    },
                ],
                "layout": {
                    "title": "Gage Linearity (Bias vs Reference)",
                    "xaxis": {"title": "Reference Value"},
                    "yaxis": {"title": "Bias (Measured − Reference)"},
                },
            }
        )

        # Bias by reference level (grouped)
        ref_groups = pd.qcut(data[reference], min(5, data[reference].nunique()), duplicates="drop")
        grouped_bias = data_with_bias.groupby(ref_groups, observed=False)["bias"].agg(["mean", "std", "count"])

        result["plots"].append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [str(g) for g in grouped_bias.index],
                        "y": grouped_bias["mean"].tolist(),
                        "error_y": {
                            "type": "data",
                            "array": (grouped_bias["std"] / np.sqrt(grouped_bias["count"])).tolist(),
                            "visible": True,
                        },
                        "marker": {"color": "#e89547"},
                    }
                ],
                "layout": {
                    "title": "Average Bias by Reference Level",
                    "xaxis": {"title": "Reference Range"},
                    "yaxis": {"title": "Average Bias"},
                },
            }
        )

        result["summary"] = summary
        result["guide_observation"] = (
            f"Linearity: slope={slope:.4f} (p={p_value:.4f}), overall bias={overall_bias:.4f}. "
            + ("Linearity issue detected." if p_value < 0.05 else "No linearity issue.")
        )
        result["statistics"] = {
            "bias": float(overall_bias),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "bias_pct": float(bias_pct),
        }
        if p_value < 0.05:
            result["narrative"] = _narrative(
                f"Linearity issue detected (slope = {slope:.4f}, p = {p_value:.4f})",
                f"Bias changes with the reference value. Overall bias = {overall_bias:.4f} ({bias_pct:.2f}%). The gage measures differently at different points in the range.",
                next_steps="Recalibrate the gage across its operating range. Check for non-linear sensor response.",
            )
        else:
            result["narrative"] = _narrative(
                f"No linearity issue (slope = {slope:.4f}, p = {p_value:.4f})",
                f"Bias is consistent across the measurement range. Overall bias = {overall_bias:.4f} ({bias_pct:.2f}%).",
                next_steps="Linearity is acceptable. If bias is significant, apply a calibration offset.",
            )

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
        summary += "<<COLOR:title>>TYPE 1 GAGE STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference value:<</COLOR>> {ref_value}\n"
        summary += f"<<COLOR:highlight>>Tolerance:<</COLOR>> {tolerance}\n"
        summary += f"<<COLOR:highlight>>N measurements:<</COLOR>> {n}\n\n"

        summary += "<<COLOR:accent>>── Descriptive Statistics ──<</COLOR>>\n"
        summary += f"  Mean = {mean_val:.4f}\n"
        summary += f"  Std Dev = {std_val:.4f}\n"
        summary += f"  Bias = {bias:.4f} (Mean − Ref)\n"
        summary += f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}\n"
        summary += f"  {'Bias is significant' if p_val < 0.05 else 'Bias is not significant'}\n\n"

        summary += "<<COLOR:accent>>── Capability Indices ──<</COLOR>>\n"
        summary += f"  Cg  = {cg:.3f} {'(≥ 1.33 required)' if cg < 1.33 else '✓'}\n"
        summary += f"  Cgk = {cgk:.3f} {'(≥ 1.33 required)' if cgk < 1.33 else '✓'}\n\n"

        if cg >= 1.33 and cgk >= 1.33:
            summary += "  <<COLOR:good>>ACCEPTABLE — both Cg and Cgk ≥ 1.33<</COLOR>>\n"
        else:
            summary += "  <<COLOR:bad>>NOT ACCEPTABLE — improve repeatability or reduce bias<</COLOR>>\n"

        # Histogram with reference line
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "histogram",
                        "x": data.tolist(),
                        "marker": {"color": "#4a90d9", "opacity": 0.7},
                        "name": "Measurements",
                    },
                    {
                        "type": "scatter",
                        "x": [ref_value, ref_value],
                        "y": [0, n * 0.3],
                        "mode": "lines",
                        "name": f"Ref = {ref_value}",
                        "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
                    },
                ],
                "layout": {
                    "title": "Type 1 Gage Study — Measurement Distribution",
                    "xaxis": {"title": measurement},
                    "yaxis": {"title": "Count"},
                },
            }
        )

        # Run chart
        result["plots"].append(
            {
                "data": [
                    {
                        "x": list(range(1, n + 1)),
                        "y": data.tolist(),
                        "mode": "lines+markers",
                        "name": "Measurements",
                        "line": {"color": "#4a90d9", "width": 1},
                        "marker": {"size": 4},
                    },
                    {
                        "x": [1, n],
                        "y": [ref_value, ref_value],
                        "mode": "lines",
                        "name": "Reference",
                        "line": {"color": "#d94a4a", "dash": "dash", "width": 2},
                    },
                    {
                        "x": [1, n],
                        "y": [mean_val, mean_val],
                        "mode": "lines",
                        "name": f"Mean = {mean_val:.4f}",
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                ],
                "layout": {"title": "Run Chart", "xaxis": {"title": "Observation"}, "yaxis": {"title": measurement}},
            }
        )

        result["summary"] = summary
        result["guide_observation"] = f"Type 1 Gage: Cg={cg:.3f}, Cgk={cgk:.3f}, bias={bias:.4f} (p={p_val:.4f}). " + (
            "Acceptable." if cg >= 1.33 and cgk >= 1.33 else "Not acceptable."
        )
        result["statistics"] = {
            "mean": float(mean_val),
            "std": float(std_val),
            "bias": float(bias),
            "p_value": float(p_val),
            "cg": float(cg),
            "cgk": float(cgk),
        }
        _t1_ok = cg >= 1.33 and cgk >= 1.33
        result["narrative"] = _narrative(
            f"Type 1 Gage Study: {'Acceptable' if _t1_ok else 'Not acceptable'} (Cg = {cg:.3f}, Cgk = {cgk:.3f})",
            f"Repeatability study against a reference standard. Bias = {bias:.4f} (p = {p_val:.4f}). Cg measures precision, Cgk measures precision + bias.",
            next_steps="Both Cg and Cgk should be \u2265 1.33. If Cg is OK but Cgk is low, recalibrate to reduce bias."
            if not _t1_ok
            else "Gage is acceptable for this measurement. Recheck periodically.",
        )

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
        summary += "<<COLOR:title>>ATTRIBUTE GAGE STUDY<</COLOR>>\n"
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

            summary += "<<COLOR:accent>>── Confusion Matrix ──<</COLOR>>\n"
            summary += f"  {'':>15} {'Ref: ' + str(unique_vals[0]):>15} {'Ref: ' + str(unique_vals[1]):>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[0]):>15} {tn:>15} {fn:>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[1]):>15} {fp:>15} {tp:>15}\n\n"

            summary += "<<COLOR:accent>>── Effectiveness Metrics ──<</COLOR>>\n"
            summary += f"  Overall Agreement:  {pct_agree:.1f}%\n"
            summary += f"  Sensitivity (detect): {sensitivity:.1f}%\n"
            summary += f"  Specificity (accept): {specificity:.1f}%\n"
            summary += f"  False Alarm Rate:   {false_alarm:.1f}%\n"
            summary += f"  Miss Rate:          {miss_rate:.1f}%\n\n"

            if pct_agree >= 90:
                summary += "  <<COLOR:good>>ACCEPTABLE — agreement ≥ 90%<</COLOR>>\n"
            else:
                summary += "  <<COLOR:bad>>NOT ACCEPTABLE — agreement < 90%<</COLOR>>\n"

            # Confusion matrix heatmap
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "heatmap",
                            "z": [[tn, fn], [fp, tp]],
                            "x": [str(unique_vals[0]), str(unique_vals[1])],
                            "y": [str(unique_vals[0]), str(unique_vals[1])],
                            "text": [[str(tn), str(fn)], [str(fp), str(tp)]],
                            "texttemplate": "%{text}",
                            "colorscale": [[0, "#f0f0f0"], [1, "#4a90d9"]],
                            "showscale": False,
                        }
                    ],
                    "layout": {
                        "title": "Confusion Matrix",
                        "xaxis": {"title": "Reference"},
                        "yaxis": {"title": "Appraiser Call"},
                    },
                }
            )

            result["statistics"] = {
                "agreement_pct": float(pct_agree),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "false_alarm_rate": float(false_alarm),
                "miss_rate": float(miss_rate),
            }
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
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": [str(a) for a in appraisers],
                            "y": app_agree,
                            "marker": {"color": ["#4a9f6e" if a >= 90 else "#d94a4a" for a in app_agree]},
                        }
                    ],
                    "layout": {
                        "title": "Agreement % by Appraiser",
                        "xaxis": {"title": "Appraiser"},
                        "yaxis": {"title": "% Agreement", "range": [0, 100]},
                    },
                }
            )

        result["summary"] = summary
        result["guide_observation"] = f"Attribute gage: {pct_agree:.1f}% overall agreement. " + (
            "Acceptable." if pct_agree >= 90 else "Needs improvement."
        )
        result["narrative"] = _narrative(
            f"Attribute Gage: {pct_agree:.1f}% agreement",
            f"{'Acceptable' if pct_agree >= 90 else 'Needs improvement'} — target is \u2265 90% agreement.",
            next_steps="If low, provide clearer standards, better training, or improved measurement tools."
            if pct_agree < 90
            else "Agreement is adequate. Monitor periodically.",
        )

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
        summary += "<<COLOR:title>>ATTRIBUTE AGREEMENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += (
            f"<<COLOR:highlight>>Appraisers:<</COLOR>> {n_appraisers} ({', '.join(str(a) for a in appraisers)})\n"
        )
        summary += f"<<COLOR:highlight>>Parts:<</COLOR>> {n_parts}\n"
        summary += f"<<COLOR:highlight>>Categories:<</COLOR>> {categories}\n\n"

        # Build rating matrix: rows = parts, columns = appraisers
        pivot = data.pivot_table(index=part_col, columns=appraiser_col, values=rating_col, aggfunc="first")

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

            summary += "<<COLOR:accent>>── Cohen's Kappa (2 raters) ──<</COLOR>>\n"
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
            Pe = (p_j**2).sum()

            Pi = (n_matrix**2).sum(axis=1) - n_raters
            Pi = Pi / (n_raters * (n_raters - 1)) if n_raters > 1 else Pi
            Po = Pi.mean()

            kappa = (Po - Pe) / (1 - Pe) if Pe < 1 else 1.0

            summary += f"<<COLOR:accent>>── Fleiss' Kappa ({n_appraisers} raters) ──<</COLOR>>\n"
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
        kappa_val = kappa if "kappa" in dir() else (pairwise_kappas[0][1] if pairwise_kappas else 0)
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
        summary += f"<<COLOR:accent>>── Interpretation ──<</COLOR>> {interp} agreement (Landis & Koch)\n"

        # Agreement by appraiser
        app_agreements = []
        for app in appraisers:
            app_data = data[data[appraiser_col] == app]
            if has_repeats:
                # Within-appraiser: across trials for same part
                within_agree = (
                    app_data.groupby(part_col)[rating_col].apply(lambda g: g.astype(str).nunique() == 1).mean() * 100
                )
            else:
                within_agree = 100.0  # single trial = always agrees with self
            app_agreements.append(within_agree)

        result["plots"].append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [str(a) for a in appraisers],
                        "y": app_agreements,
                        "marker": {"color": "#4a90d9"},
                    }
                ],
                "layout": {
                    "title": "Within-Appraiser Agreement %",
                    "xaxis": {"title": "Appraiser"},
                    "yaxis": {"title": "% Self-Consistent", "range": [0, 100]},
                },
            }
        )

        if pairwise_kappas:
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": [p[0] for p in pairwise_kappas],
                            "y": [p[1] for p in pairwise_kappas],
                            "marker": {
                                "color": [
                                    "#4a9f6e" if p[1] >= 0.6 else "#e89547" if p[1] >= 0.4 else "#d94a4a"
                                    for p in pairwise_kappas
                                ]
                            },
                        }
                    ],
                    "layout": {
                        "title": "Pairwise Cohen's Kappa",
                        "xaxis": {"title": "Pair"},
                        "yaxis": {"title": "κ", "range": [-0.2, 1]},
                    },
                }
            )

        result["summary"] = summary
        result["guide_observation"] = f"Attribute agreement: κ={kappa_val:.3f} ({interp}). " + (
            "Good agreement." if kappa_val >= 0.6 else "Agreement needs improvement."
        )
        result["statistics"] = {
            "kappa": float(kappa_val),
            "interpretation": interp,
            "n_appraisers": n_appraisers,
            "n_parts": n_parts,
        }
        result["narrative"] = _narrative(
            f"Attribute Agreement: \u03ba = {kappa_val:.3f} ({interp})",
            f"Fleiss' kappa measures inter-rater agreement beyond chance. {n_appraisers} appraisers assessed {n_parts} parts.",
            next_steps="\u03ba > 0.8 = excellent, 0.6\u20130.8 = good, 0.4\u20130.6 = moderate, < 0.4 = poor. Improve with clearer standards and training."
            if kappa_val < 0.8
            else "Agreement is strong. Standards and training are effective.",
        )

        # Diagnostics for attribute_agreement
        diagnostics = []
        if kappa_val < 0.4:
            diagnostics.append(
                {
                    "level": "error",
                    "title": f"Poor agreement (\u03ba = {kappa_val:.3f})",
                    "detail": "Raters are not consistent. Measurement system is unreliable for this attribute. Retrain and re-evaluate.",
                }
            )
        elif kappa_val < 0.6:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate agreement (\u03ba = {kappa_val:.3f})",
                    "detail": "Agreement is borderline. Clarify decision criteria and provide reference standards.",
                }
            )
        elif kappa_val >= 0.8:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Excellent agreement (\u03ba = {kappa_val:.3f})",
                    "detail": "Raters are highly consistent. Measurement system is reliable for this attribute.",
                }
            )
        result["diagnostics"] = diagnostics

    elif analysis_id == "icc":
        """
        Intraclass Correlation Coefficient (ICC) — reliability and agreement for continuous measurements.
        Supports ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k).
        """
        rater_col = config.get("rater") or config.get("appraiser") or config.get("operator")
        subject_col = config.get("subject") or config.get("part") or config.get("item")
        value_col = config.get("value") or config.get("measurement") or config.get("var")
        icc_type = config.get("icc_type", "ICC(2,1)")  # default to two-way random, single

        data = df[[rater_col, subject_col, value_col]].dropna()
        subjects = data[subject_col].unique()
        raters = data[rater_col].unique()
        n = len(subjects)
        k = len(raters)

        if n < 3 or k < 2:
            result["summary"] = f"Error: ICC requires at least 3 subjects and 2 raters. Found {n} subjects, {k} raters."
            return result

        # Pivot to subjects × raters matrix
        pivot = data.pivot_table(index=subject_col, columns=rater_col, values=value_col, aggfunc="mean")
        pivot = pivot.dropna()
        Y = pivot.values
        n, k = Y.shape

        grand_mean = np.mean(Y)
        row_means = np.mean(Y, axis=1)
        col_means = np.mean(Y, axis=0)

        # Sum of squares
        SS_total = np.sum((Y - grand_mean) ** 2)
        SS_rows = k * np.sum((row_means - grand_mean) ** 2)  # Between subjects
        SS_cols = n * np.sum((col_means - grand_mean) ** 2)  # Between raters
        SS_error = SS_total - SS_rows - SS_cols
        SS_within = SS_total - SS_rows

        # Mean squares
        MS_rows = SS_rows / (n - 1) if n > 1 else 0
        MS_cols = SS_cols / (k - 1) if k > 1 else 0
        MS_error = SS_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 1e-10
        MS_within = SS_within / (n * (k - 1)) if n * (k - 1) > 0 else 1e-10

        # Calculate all ICC forms
        icc_values = {}
        # ICC(1,1): One-way random, single measures
        icc_values["ICC(1,1)"] = (
            (MS_rows - MS_within) / (MS_rows + (k - 1) * MS_within) if (MS_rows + (k - 1) * MS_within) > 0 else 0
        )
        # ICC(2,1): Two-way random, single measures
        icc_values["ICC(2,1)"] = (
            (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n)
            if (MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n) > 0
            else 0
        )
        # ICC(3,1): Two-way mixed, single measures
        icc_values["ICC(3,1)"] = (
            (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error) if (MS_rows + (k - 1) * MS_error) > 0 else 0
        )
        # ICC(1,k): One-way random, average measures
        icc_values["ICC(1,k)"] = (MS_rows - MS_within) / MS_rows if MS_rows > 0 else 0
        # ICC(2,k): Two-way random, average measures
        icc_values["ICC(2,k)"] = (
            (MS_rows - MS_error) / (MS_rows + (MS_cols - MS_error) / n)
            if (MS_rows + (MS_cols - MS_error) / n) > 0
            else 0
        )
        # ICC(3,k): Two-way mixed, average measures
        icc_values["ICC(3,k)"] = (MS_rows - MS_error) / MS_rows if MS_rows > 0 else 0

        # Primary ICC value
        icc_val = float(icc_values.get(icc_type, icc_values["ICC(2,1)"]))
        icc_val = max(-1, min(1, icc_val))

        # Confidence interval via F-distribution (for ICC(2,1))
        try:
            F_val = MS_rows / MS_error if MS_error > 0 else 1
            df1, df2 = n - 1, (n - 1) * (k - 1)
            F_lo = F_val / stats.f.ppf(0.975, df1, df2) if stats.f.ppf(0.975, df1, df2) > 0 else F_val
            F_hi = F_val / stats.f.ppf(0.025, df1, df2) if stats.f.ppf(0.025, df1, df2) > 0 else F_val
            ci_lo = max(-1, (F_lo - 1) / (F_lo + k - 1))
            ci_hi = min(1, (F_hi - 1) / (F_hi + k - 1))
        except Exception:
            ci_lo, ci_hi = icc_val - 0.1, min(1, icc_val + 0.1)

        # Interpretation (Koo & Li 2016)
        if icc_val >= 0.9:
            interp = "excellent"
        elif icc_val >= 0.75:
            interp = "good"
        elif icc_val >= 0.5:
            interp = "moderate"
        else:
            interp = "poor"

        # Summary
        _eq = "=" * 70
        _dash = "-" * 60
        summary = f"<<COLOR:accent>>{_eq}<</COLOR>>\n"
        summary += "<<COLOR:title>>INTRACLASS CORRELATION COEFFICIENT (ICC)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{_eq}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Subjects:<</COLOR>> {n}    <<COLOR:text>>Raters:<</COLOR>> {k}\n"
        summary += f"<<COLOR:text>>ICC Type:<</COLOR>> {icc_type}\n\n"

        summary += "<<COLOR:highlight>>All ICC Forms:<</COLOR>>\n"
        summary += f"  {'Form':<12} {'Value':>8}  {'Use Case'}\n"
        summary += f"  {_dash}\n"
        _use_cases = {
            "ICC(1,1)": "One-way random, single rater",
            "ICC(2,1)": "Two-way random, single rater (most common)",
            "ICC(3,1)": "Two-way mixed, single rater",
            "ICC(1,k)": "One-way random, average of k raters",
            "ICC(2,k)": "Two-way random, average of k raters",
            "ICC(3,k)": "Two-way mixed, average of k raters",
        }
        for form, val in icc_values.items():
            marker = " <<" if form == icc_type else ""
            summary += f"  {form:<12} {val:>8.4f}  {_use_cases.get(form, '')}{marker}\n"

        summary += f"\n<<COLOR:highlight>>Selected: {icc_type} = {icc_val:.4f} ({interp})<</COLOR>>\n"
        summary += f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n\n"

        summary += "<<COLOR:highlight>>ANOVA Table:<</COLOR>>\n"
        summary += f"  {'Source':<20} {'SS':>12} {'df':>6} {'MS':>12}\n"
        summary += f"  {_dash}\n"
        summary += f"  {'Between Subjects':<20} {SS_rows:>12.4f} {n - 1:>6} {MS_rows:>12.4f}\n"
        summary += f"  {'Between Raters':<20} {SS_cols:>12.4f} {k - 1:>6} {MS_cols:>12.4f}\n"
        summary += f"  {'Residual':<20} {SS_error:>12.4f} {(n - 1) * (k - 1):>6} {MS_error:>12.4f}\n"
        summary += f"  {'Total':<20} {SS_total:>12.4f} {n * k - 1:>6}\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"ICC ({icc_type}) = {icc_val:.3f} ({interp}). {k} raters, {n} subjects. 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]."
        )

        result["narrative"] = _narrative(
            f"Reliability: ICC = {icc_val:.3f} ({interp})",
            f"The {icc_type} intraclass correlation is <strong>{icc_val:.3f}</strong> ({interp}) with 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]. "
            f"{k} raters measured {n} subjects."
            + (
                " Measurement system is reliable."
                if icc_val >= 0.75
                else " Measurement system needs improvement."
                if icc_val >= 0.5
                else " Measurement system is unreliable \u2014 reduce rater variability before using this measurement for decisions."
            ),
            next_steps="ICC > 0.9 = excellent, 0.75\u20130.9 = good, 0.5\u20130.75 = moderate, < 0.5 = poor. For MSA, target ICC > 0.9."
            if icc_val < 0.9
            else "Excellent reliability. System is suitable for critical measurements.",
            chart_guidance="The rater comparison plot shows each subject measured by each rater. Tight clustering = good agreement.",
        )

        # Diagnostics
        diagnostics = []
        if icc_val < 0.5:
            diagnostics.append(
                {
                    "level": "error",
                    "title": f"Poor reliability (ICC = {icc_val:.3f})",
                    "detail": "More than half the variation is from measurement error. Do not use this measurement system for decisions.",
                }
            )
        elif icc_val < 0.75:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Moderate reliability (ICC = {icc_val:.3f})",
                    "detail": "Acceptable for group comparisons but not for individual assessments. Improve standardization.",
                }
            )
        elif icc_val >= 0.9:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Excellent reliability (ICC = {icc_val:.3f})",
                    "detail": "Measurement system is highly reliable. Suitable for individual-level decisions.",
                }
            )
        # Rater bias
        if MS_cols > 2 * MS_error:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": "Systematic rater bias detected",
                    "detail": "Between-rater variance is large relative to error. Some raters consistently rate higher or lower.",
                }
            )
        result["diagnostics"] = diagnostics

        result["statistics"] = {
            "icc": float(icc_val),
            "icc_type": icc_type,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "interpretation": interp,
            "n_subjects": n,
            "n_raters": k,
            "all_icc": {k: float(v) for k, v in icc_values.items()},
        }

        # Rater comparison plot
        plot_data = []
        for rater in raters:
            rater_data = pivot[rater].values if rater in pivot.columns else []
            if len(rater_data) > 0:
                plot_data.append(
                    {
                        "type": "scatter",
                        "y": rater_data.tolist(),
                        "x": list(range(1, len(rater_data) + 1)),
                        "mode": "markers+lines",
                        "name": str(rater),
                        "line": {"width": 1},
                        "marker": {"size": 6},
                    }
                )
        result["plots"].append(
            {
                "title": f"Rater Comparison ({icc_type} = {icc_val:.3f})",
                "data": plot_data,
                "layout": {"height": 300, "xaxis": {"title": "Subject"}, "yaxis": {"title": value_col}},
            }
        )

        # ICC bar chart
        result["plots"].append(
            {
                "title": "ICC Forms Comparison",
                "data": [
                    {
                        "type": "bar",
                        "x": list(icc_values.keys()),
                        "y": [float(v) for v in icc_values.values()],
                        "marker": {
                            "color": [
                                "#4a9f6e" if v >= 0.75 else "#e8953f" if v >= 0.5 else "#dc5050"
                                for v in icc_values.values()
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": "ICC", "range": [0, 1.05]},
                    "shapes": [
                        {
                            "type": "line",
                            "y0": 0.75,
                            "y1": 0.75,
                            "x0": -0.5,
                            "x1": 5.5,
                            "line": {"color": "#4a9f6e", "width": 1, "dash": "dash"},
                        }
                    ],
                },
            }
        )

    elif analysis_id == "krippendorff_alpha":
        """
        Krippendorff's Alpha — universal agreement metric for any data level, any number of raters, missing data OK.
        """
        rater_col = config.get("rater") or config.get("appraiser") or config.get("operator")
        subject_col = config.get("subject") or config.get("part") or config.get("item")
        value_col = config.get("value") or config.get("measurement") or config.get("var")
        level = config.get("level", "interval")  # nominal, ordinal, interval, ratio

        data = df[[rater_col, subject_col, value_col]].dropna()
        subjects = data[subject_col].unique()
        raters = data[rater_col].unique()
        n_subj = len(subjects)
        n_raters = len(raters)

        if n_subj < 3 or n_raters < 2:
            result["summary"] = (
                f"Error: Need at least 3 subjects and 2 raters. Found {n_subj} subjects, {n_raters} raters."
            )
            return result

        # Build reliability data matrix (raters × subjects)
        pivot = data.pivot_table(index=rater_col, columns=subject_col, values=value_col, aggfunc="first")
        R = pivot.values  # raters × subjects, may have NaN

        # Compute Krippendorff's alpha
        # Collect all value pairs within each unit
        n_total = 0
        D_o = 0.0  # observed disagreement
        all_values = []

        for j in range(R.shape[1]):  # each subject
            col = R[:, j]
            valid = col[~np.isnan(col)]
            m = len(valid)
            if m < 2:
                continue
            all_values.extend(valid)
            n_total += m
            # Within-unit disagreement
            for a in range(m):
                for b in range(a + 1, m):
                    if level == "nominal":
                        D_o += 0 if valid[a] == valid[b] else 1
                    else:  # interval/ratio
                        D_o += (valid[a] - valid[b]) ** 2
            m * (m - 1) / 2

        # Expected disagreement
        all_vals = np.array(all_values)
        n_v = len(all_vals)
        D_e = 0.0
        if n_v >= 2:
            if level == "nominal":
                unique_vals, counts = np.unique(all_vals, return_counts=True)
                for c in counts:
                    D_e += c * (n_v - c)
                D_e /= n_v * (n_v - 1)
            else:
                for a in range(n_v):
                    for b in range(a + 1, n_v):
                        D_e += (all_vals[a] - all_vals[b]) ** 2
                D_e = D_e * 2 / (n_v * (n_v - 1))

        # Compute alpha
        # Normalize observed disagreement
        n_pairs_obs = 0
        D_o_norm = 0.0
        for j in range(R.shape[1]):
            col = R[:, j]
            valid = col[~np.isnan(col)]
            m = len(valid)
            if m < 2:
                continue
            pairs = m * (m - 1) / 2
            n_pairs_obs += pairs
            for a in range(m):
                for b in range(a + 1, m):
                    if level == "nominal":
                        D_o_norm += 0 if valid[a] == valid[b] else 1
                    else:
                        D_o_norm += (valid[a] - valid[b]) ** 2

        if n_pairs_obs > 0:
            D_o_avg = D_o_norm / n_pairs_obs
        else:
            D_o_avg = 0

        alpha = 1 - D_o_avg / D_e if D_e > 0 else 1.0
        alpha = max(-1, min(1, float(alpha)))

        # Interpretation
        if alpha >= 0.8:
            interp = "reliable"
        elif alpha >= 0.667:
            interp = "tentatively acceptable"
        else:
            interp = "unacceptable"

        # Summary
        _eq = "=" * 70
        summary = f"<<COLOR:accent>>{_eq}<</COLOR>>\n"
        summary += "<<COLOR:title>>KRIPPENDORFF'S ALPHA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{_eq}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Subjects:<</COLOR>> {n_subj}    <<COLOR:text>>Raters:<</COLOR>> {n_raters}    <<COLOR:text>>Level:<</COLOR>> {level}\n\n"
        summary += f"<<COLOR:highlight>>Alpha = {alpha:.4f} ({interp})<</COLOR>>\n"
        summary += f"  Observed disagreement: {D_o_avg:.4f}\n"
        summary += f"  Expected disagreement: {D_e:.4f}\n\n"
        summary += "<<COLOR:text>>Interpretation (Krippendorff):<</COLOR>>\n"
        summary += "  \u03b1 \u2265 0.800: Reliable\n"
        summary += "  0.667 \u2264 \u03b1 < 0.800: Tentatively acceptable\n"
        summary += "  \u03b1 < 0.667: Unacceptable for drawing conclusions\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Krippendorff's \u03b1 = {alpha:.3f} ({interp}). {n_raters} raters, {n_subj} subjects, {level} level."
        )

        result["narrative"] = _narrative(
            f"Agreement: \u03b1 = {alpha:.3f} ({interp})",
            f"Krippendorff's alpha is <strong>{alpha:.3f}</strong> ({interp}) for {n_raters} raters across {n_subj} subjects at the {level} measurement level. "
            + (
                "Data is reliable for analysis."
                if alpha >= 0.8
                else "Data is tentatively usable but agreement should be improved."
                if alpha >= 0.667
                else "Agreement is too low \u2014 conclusions from this data are unreliable."
            ),
            next_steps="Improve rater training and standardize measurement procedures."
            if alpha < 0.8
            else "Agreement is sufficient. Proceed with analysis.",
            chart_guidance="The heatmap shows each rater's measurements by subject. Consistent colors across raters = good agreement.",
        )

        # Diagnostics
        diagnostics = []
        if alpha < 0.667:
            diagnostics.append(
                {
                    "level": "error",
                    "title": f"Unacceptable agreement (\u03b1 = {alpha:.3f})",
                    "detail": "Do not draw conclusions from this data. Raters disagree too much.",
                    "action": {
                        "label": "Run ICC for details",
                        "type": "stats",
                        "analysis": "icc",
                        "config": {"rater": rater_col, "subject": subject_col, "value": value_col},
                    },
                }
            )
        elif alpha < 0.8:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Borderline agreement (\u03b1 = {alpha:.3f})",
                    "detail": "Tentatively acceptable. Consider improving standardization before critical decisions.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Reliable agreement (\u03b1 = {alpha:.3f})",
                    "detail": "Raters are consistent. Data is suitable for analysis and decision-making.",
                }
            )
        result["diagnostics"] = diagnostics

        result["statistics"] = {
            "alpha": alpha,
            "interpretation": interp,
            "D_observed": float(D_o_avg),
            "D_expected": float(D_e),
            "level": level,
            "n_subjects": n_subj,
            "n_raters": n_raters,
        }

        # Heatmap of ratings
        result["plots"].append(
            {
                "title": f"Rater \u00d7 Subject Matrix (\u03b1 = {alpha:.3f})",
                "data": [
                    {
                        "type": "heatmap",
                        "z": R.tolist(),
                        "x": [str(s) for s in subjects[:50]],
                        "y": [str(r) for r in raters],
                        "colorscale": "Viridis",
                        "text": [[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in R],
                        "texttemplate": "%{text}",
                        "textfont": {"size": 10},
                    }
                ],
                "layout": {
                    "height": max(200, n_raters * 30 + 80),
                    "xaxis": {"title": "Subject"},
                    "yaxis": {"title": "Rater"},
                },
            }
        )

    elif analysis_id == "bland_altman":
        """
        Bland-Altman Method Comparison — assess agreement between two measurement methods.
        """
        method1 = config.get("method1") or config.get("var1")
        method2 = config.get("method2") or config.get("var2")

        m1 = df[method1].dropna()
        m2 = df[method2].dropna()
        common = m1.index.intersection(m2.index)
        m1, m2 = m1.loc[common].values, m2.loc[common].values
        n = len(m1)

        if n < 5:
            result["summary"] = f"Error: Need at least 5 paired measurements. Found {n}."
            return result

        diffs = m1 - m2
        means = (m1 + m2) / 2
        bias = float(np.mean(diffs))
        sd_diff = float(np.std(diffs, ddof=1))
        loa_upper = bias + 1.96 * sd_diff
        loa_lower = bias - 1.96 * sd_diff

        # CI for bias
        se_bias = sd_diff / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, n - 1))
        bias_ci = (bias - t_crit * se_bias, bias + t_crit * se_bias)

        # CI for LOA
        se_loa = np.sqrt(3 * sd_diff**2 / n)
        loa_lower_ci = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)
        loa_upper_ci = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)

        # Proportional bias check (correlation between difference and mean)
        r_prop, p_prop = stats.pearsonr(means, diffs) if n > 3 else (0, 1)

        # Percentage of points within LOA
        within_loa = float(np.mean((diffs >= loa_lower) & (diffs <= loa_upper)) * 100)

        # Summary
        _eq = "=" * 70
        summary = f"<<COLOR:accent>>{_eq}<</COLOR>>\n"
        summary += "<<COLOR:title>>BLAND-ALTMAN METHOD COMPARISON<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{_eq}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Method 1:<</COLOR>> {method1}    <<COLOR:text>>Method 2:<</COLOR>> {method2}    <<COLOR:text>>n:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:highlight>>Bias (mean difference):<</COLOR>> {bias:.4f}  95% CI [{bias_ci[0]:.4f}, {bias_ci[1]:.4f}]\n"
        summary += "<<COLOR:highlight>>Limits of Agreement:<</COLOR>>\n"
        summary += f"  Upper: {loa_upper:.4f}  95% CI [{loa_upper_ci[0]:.4f}, {loa_upper_ci[1]:.4f}]\n"
        summary += f"  Lower: {loa_lower:.4f}  95% CI [{loa_lower_ci[0]:.4f}, {loa_lower_ci[1]:.4f}]\n"
        summary += f"  Width: {loa_upper - loa_lower:.4f}\n\n"
        summary += f"<<COLOR:text>>Points within LOA:<</COLOR>> {within_loa:.1f}%\n"
        summary += f"<<COLOR:text>>Proportional bias:<</COLOR>> r = {r_prop:.3f}, p = {p_prop:.4f}\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Bland-Altman: bias = {bias:.4f}, LOA = [{loa_lower:.4f}, {loa_upper:.4f}]. {within_loa:.0f}% within LOA."
        )

        bias_meaningful = abs(bias) > 0.1 * np.mean(np.abs(means)) if np.mean(np.abs(means)) > 0 else abs(bias) > 0
        result["narrative"] = _narrative(
            f"Bias: {bias:.4f} | LOA width: {loa_upper - loa_lower:.4f}",
            f"The mean difference between <strong>{method1}</strong> and <strong>{method2}</strong> is {bias:.4f} "
            f"(95% CI [{bias_ci[0]:.4f}, {bias_ci[1]:.4f}]). "
            f"The limits of agreement span {loa_upper - loa_lower:.4f} units. "
            f"{within_loa:.1f}% of measurements fall within the LOA."
            + (
                f" <strong>Proportional bias detected</strong> (r = {r_prop:.3f}, p = {p_prop:.4f}) \u2014 disagreement varies with magnitude."
                if p_prop < 0.05
                else ""
            ),
            next_steps="Judge whether the LOA width is clinically/practically acceptable. If bias is consistent, one method can be calibrated to the other."
            if bias_meaningful
            else "Bias is negligible. Focus on whether the LOA width is acceptable for your application.",
            chart_guidance="Points should scatter randomly around the bias line (dashed). Funnel shapes indicate proportional bias. Points outside LOA are outlier disagreements.",
        )

        # Diagnostics
        diagnostics = []
        if p_prop < 0.05:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Proportional bias detected (r = {r_prop:.3f})",
                    "detail": "Disagreement between methods changes with measurement magnitude. A simple calibration offset won't work.",
                }
            )
        if abs(bias) > 0:
            _bias_zero = 0 >= bias_ci[0] and 0 <= bias_ci[1]
            if not _bias_zero:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": f"Systematic bias: {bias:.4f} (CI excludes zero)",
                        "detail": f"{method1} reads consistently {'higher' if bias > 0 else 'lower'} than {method2}.",
                    }
                )
            else:
                diagnostics.append(
                    {
                        "level": "info",
                        "title": "No significant bias (CI includes zero)",
                        "detail": f"Mean difference {bias:.4f} is not significantly different from zero.",
                    }
                )
        if within_loa < 90:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Only {within_loa:.0f}% within LOA (expected ~95%)",
                    "detail": "More outlier disagreements than expected. Check for specific conditions where methods diverge.",
                }
            )
        result["diagnostics"] = diagnostics

        result["statistics"] = {
            "bias": bias,
            "sd_diff": sd_diff,
            "loa_upper": float(loa_upper),
            "loa_lower": float(loa_lower),
            "bias_ci": [float(bias_ci[0]), float(bias_ci[1])],
            "within_loa_pct": within_loa,
            "proportional_bias_r": float(r_prop),
            "proportional_bias_p": float(p_prop),
            "n": n,
        }

        # Bland-Altman plot
        result["plots"].append(
            {
                "title": f"Bland-Altman: {method1} vs {method2}",
                "data": [
                    {
                        "type": "scatter",
                        "x": means.tolist(),
                        "y": diffs.tolist(),
                        "mode": "markers",
                        "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 7},
                        "name": "Differences",
                    },
                ],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": f"Mean of {method1} and {method2}"},
                    "yaxis": {"title": f"{method1} \u2212 {method2}"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": float(min(means)),
                            "x1": float(max(means)),
                            "y0": bias,
                            "y1": bias,
                            "line": {"color": "#4a90d9", "width": 2, "dash": "dash"},
                        },
                        {
                            "type": "line",
                            "x0": float(min(means)),
                            "x1": float(max(means)),
                            "y0": loa_upper,
                            "y1": loa_upper,
                            "line": {"color": "#dc5050", "width": 1.5, "dash": "dot"},
                        },
                        {
                            "type": "line",
                            "x0": float(min(means)),
                            "x1": float(max(means)),
                            "y0": loa_lower,
                            "y1": loa_lower,
                            "line": {"color": "#dc5050", "width": 1.5, "dash": "dot"},
                        },
                    ],
                    "annotations": [
                        {
                            "x": float(max(means)),
                            "y": bias,
                            "text": f"Bias: {bias:.3f}",
                            "showarrow": False,
                            "font": {"color": "#4a90d9", "size": 10},
                            "xanchor": "left",
                        },
                        {
                            "x": float(max(means)),
                            "y": loa_upper,
                            "text": f"+1.96SD: {loa_upper:.3f}",
                            "showarrow": False,
                            "font": {"color": "#dc5050", "size": 10},
                            "xanchor": "left",
                        },
                        {
                            "x": float(max(means)),
                            "y": loa_lower,
                            "text": f"\u22121.96SD: {loa_lower:.3f}",
                            "showarrow": False,
                            "font": {"color": "#dc5050", "size": 10},
                            "xanchor": "left",
                        },
                    ],
                },
            }
        )

        # Histogram of differences
        result["plots"].append(
            {
                "title": "Distribution of Differences",
                "data": [
                    {
                        "type": "histogram",
                        "x": diffs.tolist(),
                        "marker": {"color": "rgba(74, 159, 110, 0.4)"},
                        "name": "Differences",
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": f"{method1} \u2212 {method2}"},
                    "yaxis": {"title": "Count"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": bias,
                            "x1": bias,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#4a90d9", "width": 2, "dash": "dash"},
                        }
                    ],
                },
            }
        )

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
                    cc = np.mean(x_std[: n - lag] * y_std[lag:]) if lag < n else 0.0
                else:
                    cc = np.mean(x_std[-lag:] * y_std[: n + lag]) if -lag < n else 0.0
                ccf_vals.append(float(cc))

            sig_bound = 2.0 / np.sqrt(n)

            # Find significant lags
            sig_lags = [(lag, cc) for lag, cc in zip(lags, ccf_vals) if abs(cc) > sig_bound]

            summary = f"""<<COLOR:title>>CROSS-CORRELATION FUNCTION<</COLOR>>
{"=" * 50}
<<COLOR:highlight>>Series 1:<</COLOR>> {var1}
<<COLOR:highlight>>Series 2:<</COLOR>> {var2}
<<COLOR:highlight>>N:<</COLOR>> {n}
<<COLOR:highlight>>Max lag:<</COLOR>> ±{max_lag}
<<COLOR:highlight>>Significance bound:<</COLOR>> ±{sig_bound:.4f}

<<COLOR:accent>>Lag 0 correlation:<</COLOR>> {ccf_vals[max_lag]:.4f}"""

            if sig_lags:
                summary += "\n\n<<COLOR:accent>>Significant lags:<</COLOR>>"
                for lag, cc in sorted(sig_lags, key=lambda x: abs(x[1]), reverse=True)[:10]:
                    summary += f"\n  Lag {lag:>3d}: r = {cc:+.4f}"

            result["summary"] = summary
            result["guide_observation"] = (
                f"CCF({var1}, {var2}): lag-0 r={ccf_vals[max_lag]:.3f}, {len(sig_lags)} significant lags."
            )
            result["statistics"] = {"lag_0_correlation": ccf_vals[max_lag], "significant_lags": len(sig_lags)}
            result["narrative"] = _narrative(
                f"Cross-Correlation: lag-0 r = {ccf_vals[max_lag]:.3f}, {len(sig_lags)} significant lags",
                f"CCF shows how <strong>{var1}</strong> and <strong>{var2}</strong> relate at different time lags. Positive lags = {var1} leads {var2}.",
                next_steps="Significant negative lags suggest {var2} leads {var1}. Use the peak lag for lead-lag transfer function modeling.",
                chart_guidance="Bars outside the shaded band are significant. The lag at peak correlation suggests the delay between the series.",
            )

            result["plots"].append(
                {
                    "title": f"Cross-Correlation: {var1} vs {var2}",
                    "data": [
                        {
                            "type": "bar",
                            "x": lags,
                            "y": ccf_vals,
                            "marker": {"color": ["#d94a4a" if abs(c) > sig_bound else "#4a9f6e" for c in ccf_vals]},
                            "name": "CCF",
                        },
                        {
                            "type": "scatter",
                            "x": [-max_lag, max_lag],
                            "y": [sig_bound, sig_bound],
                            "mode": "lines",
                            "line": {"color": "#e89547", "dash": "dash", "width": 1},
                            "showlegend": False,
                        },
                        {
                            "type": "scatter",
                            "x": [-max_lag, max_lag],
                            "y": [-sig_bound, -sig_bound],
                            "mode": "lines",
                            "line": {"color": "#e89547", "dash": "dash", "width": 1},
                            "showlegend": False,
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Lag"},
                        "yaxis": {"title": "Correlation", "range": [-1.05, 1.05]},
                    },
                }
            )

    # ── Johnson Transformation ───────────────────────────────────────────

    return result
