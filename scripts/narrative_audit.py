"""Narrative Quality Audit — compares forgenarr output against canonical handler inline narratives.

Run: set -a && source /etc/svend/env && set +a && cd ~/kjerne && python3 scripts/narrative_audit.py

Exercises every forgenarr template with representative statistics and scores quality.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/forgenarr/src"))

from forgenarr import narrate
from forgenarr.engine import _REGISTRY, _TYPE_DEFAULTS

# ─── Representative statistics for each (type, id) ────────────────────────

TEST_CASES = {
    # ── Stats ──
    ("stats", "ttest"): {
        "stats": {
            "p_value": 0.003,
            "statistic": 3.21,
            "significant": True,
            "mean1": 10.4,
            "mean2": None,
            "mean_diff": 0.4,
            "effect_size": 0.62,
            "effect_label": "Cohen's d",
            "ci_lower": 0.15,
            "ci_upper": 0.65,
            "n": 45,
        },
        "config": {"column": "diameter"},
    },
    ("stats", "ttest2"): {
        "stats": {
            "p_value": 0.12,
            "statistic": 1.58,
            "significant": False,
            "mean1": 5.2,
            "mean2": 5.0,
            "mean_diff": 0.2,
            "effect_size": 0.18,
            "effect_label": "Cohen's d",
            "ci_lower": -0.05,
            "ci_upper": 0.45,
            "n": 30,
        },
        "config": {"column": "weight"},
    },
    ("stats", "anova"): {
        "stats": {
            "p_value": 0.001,
            "statistic": 6.7,
            "significant": True,
            "n_groups": 4,
            "eta_squared": 0.28,
            "effect_size": 0.28,
            "n": 80,
        },
        "config": {},
    },
    ("stats", "correlation"): {
        "stats": {
            "p_value": 0.002,
            "r": 0.74,
            "r_squared": 0.55,
            "statistic": 4.2,
            "significant": True,
            "n": 25,
            "effect_size": 0.74,
        },
        "config": {"column": "temp", "column2": "viscosity"},
    },
    ("stats", "regression"): {
        "stats": {
            "p_value": 0.0001,
            "r_squared": 0.82,
            "adj_r_squared": 0.80,
            "n": 50,
            "n_predictors": 3,
            "f_statistic": 45.2,
            "rmse": 1.23,
            "significant": True,
        },
        "config": {},
    },
    ("stats", "normality"): {
        "stats": {
            "p_value": 0.04,
            "statistic": 0.95,
            "test": "Shapiro-Wilk",
            "significant": True,
            "n": 30,
            "skewness": 0.8,
            "kurtosis": 4.2,
        },
        "config": {"column": "cycle_time"},
    },
    ("stats", "chi2"): {
        "stats": {"p_value": 0.03, "statistic": 8.9, "df": 3, "significant": True, "cramers_v": 0.22, "n": 100},
        "config": {},
    },
    ("stats", "descriptive"): {
        "stats": {
            "mean": 25.4,
            "std": 3.2,
            "median": 25.0,
            "n": 50,
            "skewness": 0.3,
            "kurtosis": 2.8,
            "min": 18.1,
            "max": 33.2,
            "iqr": 4.5,
        },
        "config": {"column": "thickness"},
    },
    ("stats", "gage_rr"): {
        "stats": {
            "total_grr_pct": 18.5,
            "repeatability_pct": 12.0,
            "reproducibility_pct": 6.5,
            "part_to_part_pct": 81.5,
            "ndc": 7,
            "n_operators": 3,
            "n_parts": 10,
        },
        "config": {},
    },
    # ── SPC ──
    ("spc", "imr"): {
        "stats": {
            "n_ooc": 2,
            "in_control": False,
            "ucl": 10.5,
            "cl": 7.2,
            "lcl": 3.9,
            "n_run_violations": 1,
            "run_violations": [{"rule": 2, "points": [15, 16, 17]}],
        },
        "config": {"column": "pressure"},
    },
    ("spc", "capability"): {
        "stats": {
            "cpk": 1.45,
            "cp": 1.62,
            "ppk": 1.38,
            "pp": 1.55,
            "pct_out": 0.12,
            "mean": 50.2,
            "std": 0.82,
            "usl": 52.5,
            "lsl": 47.5,
            "sigma_level": 4.35,
        },
        "config": {"column": "bore_diameter"},
    },
    ("spc", "cusum"): {
        "stats": {"n_ooc": 3, "in_control": False, "ucl": 5.0, "lcl": -5.0, "h": 5.0, "k": 0.5, "n_run_violations": 0},
        "config": {"column": "pH"},
    },
    ("spc", "ewma"): {
        "stats": {
            "n_ooc": 0,
            "in_control": True,
            "ucl": 3.2,
            "cl": 0.0,
            "lcl": -3.2,
            "lambda_": 0.2,
            "n_run_violations": 0,
        },
        "config": {"column": "temp"},
    },
    ("spc", "xbar_r"): {
        "stats": {
            "n_ooc": 1,
            "in_control": False,
            "ucl": 25.3,
            "cl": 24.8,
            "lcl": 24.3,
            "r_bar": 1.2,
            "n_run_violations": 0,
        },
        "config": {"column": "length", "subgroup_size": 5},
    },
    ("spc", "p_chart"): {
        "stats": {
            "n_ooc": 0,
            "in_control": True,
            "ucl": 0.12,
            "cl": 0.05,
            "lcl": 0.0,
            "p_bar": 0.05,
            "n_run_violations": 0,
        },
        "config": {"column": "defective"},
    },
    # ── PBS ──
    ("pbs", "pbs_belief"): {
        "stats": {"shift_probability": 0.97, "status": "ALARM", "n_observations": 120},
        "config": {"column": "torque"},
    },
    ("pbs", "pbs_edetector"): {
        "stats": {"alarm_detected": True, "peak_log_E": 8.4, "peak_observation": 87, "threshold": 4.6, "n_alarms": 2},
        "config": {"column": "vibration"},
    },
    ("pbs", "pbs_cpk"): {
        "stats": {
            "cpk_posterior_mean": 1.32,
            "p_above_133": 0.42,
            "p_above_1": 0.91,
            "cpk_hdi_lower": 1.05,
            "cpk_hdi_upper": 1.58,
        },
        "config": {"column": "bore"},
    },
    ("pbs", "pbs_evidence"): {
        "stats": {"cumulative_evidence": 25.4, "threshold": 20, "above_threshold": True, "n_observations": 200},
        "config": {"column": "width"},
    },
    ("pbs", "pbs_health"): {
        "stats": {"overall_health": 0.72, "n_streams": 5, "n_alarming": 1, "n_caution": 2, "n_stable": 2},
        "config": {},
    },
    # ── Bayesian ──
    ("bayesian", "bayes_ttest"): {
        "stats": {"bf10": 12.4, "posterior_mean": 0.45, "hdi_lower": 0.12, "hdi_upper": 0.78, "effect_size": 0.45},
        "config": {},
    },
    ("bayesian", "bayes_proportion"): {
        "stats": {"bf10": 3.2, "posterior_mean": 0.08, "hdi_lower": 0.04, "hdi_upper": 0.12, "n": 200, "successes": 16},
        "config": {},
    },
    ("bayesian", "bayes_correlation"): {
        "stats": {"bf10": 45.0, "posterior_rho": 0.68, "hdi_lower": 0.42, "hdi_upper": 0.85, "n": 40},
        "config": {},
    },
    ("bayesian", "bayes_equivalence"): {
        "stats": {
            "bf10": 0.3,
            "p_in_rope": 0.82,
            "rope_lower": -0.1,
            "rope_upper": 0.1,
            "posterior_mean": 0.02,
            "hdi_lower": -0.05,
            "hdi_upper": 0.09,
        },
        "config": {},
    },
    # ── Reliability ──
    ("reliability", "weibull"): {
        "stats": {"shape": 2.1, "scale": 5000, "mttf": 4430, "b10_life": 2100, "r_squared": 0.97},
        "config": {},
    },
    # ── Causal ──
    ("causal", "pc_algorithm"): {
        "stats": {"n_edges": 5, "n_nodes": 4, "algorithm": "PC", "alpha": 0.05},
        "config": {},
    },
    # ── Quality Economics ──
    ("quality_econ", "taguchi_loss"): {
        "stats": {"total_loss": 12500, "avg_loss_per_unit": 2.5, "n_units": 5000, "k": 100, "target": 10.0},
        "config": {},
    },
    # ── Misc ──
    ("simulation", "tolerance_stackup"): {
        "stats": {"rss_sigma": 0.85, "mc_std": 0.92, "pct_out_of_spec": 2.3, "n_components": 5},
        "config": {},
    },
    ("drift", "feature_drift"): {
        "stats": {"n_drifted": 3, "n_features": 12},
        "config": {},
    },
    ("anytime", "e_value"): {
        "stats": {"cumulative_e": 15.2, "threshold": 20, "above_threshold": False, "n": 100},
        "config": {},
    },
    # ── Viz (type default) ──
    ("viz", "histogram"): {
        "stats": {"n_bins": 20, "mean": 4.5, "std": 1.2},
        "config": {"column": "response_time"},
    },
}


# ─── Quality scoring ──────────────────────────────────────────────────────


def score_narrative(narr, analysis_type, analysis_id, stats):
    """Score a narrative dict on quality criteria. Returns (score, issues)."""
    issues = []
    score = 0
    max_score = 20

    verdict = narr.get("verdict", "")
    body = narr.get("body", "")
    next_steps = narr.get("next_steps", "")
    chart_guidance = narr.get("chart_guidance", "")

    # 1. Verdict exists and is conditional (not static) — 5 pts
    if not verdict:
        issues.append("CRITICAL: Empty verdict")
    elif verdict in ("Analysis complete.", "No data.", ""):
        issues.append("WEAK: Generic/static verdict — not conditional on results")
        score += 1
    elif any(
        v in verdict
        for v in (
            str(stats.get("p_value", "NONE")),
            str(stats.get("shift_probability", "NONE")),
            str(stats.get("bf10", "NONE")),
            str(stats.get("cpk", "NONE")),
            "significant",
            "ALARM",
            "control",
            "detected",
            "drift",
            "evidence",
        )
    ):
        score += 5  # Dynamic, references actual values
    else:
        score += 3  # Present but may not reference specific values
        issues.append("OK: Verdict present but doesn't reference specific stat values")

    # 2. Body is substantive (>50 chars, references numbers) — 5 pts
    if not body:
        issues.append("CRITICAL: Empty body")
    elif len(body) < 50:
        issues.append("WEAK: Body too short (<50 chars)")
        score += 1
    else:
        has_numbers = any(c.isdigit() for c in body)
        if has_numbers:
            score += 5
        else:
            score += 3
            issues.append("OK: Body present but lacks numeric values")

    # 3. next_steps is actionable (>20 chars, not empty) — 5 pts
    if not next_steps:
        # Some analyses legitimately have no next steps (process stable, no action needed)
        stable_indicators = ("in_control", "stable", "no drift", "no change")
        if any(s in verdict.lower() for s in stable_indicators):
            score += 4  # Acceptable empty for stable processes
        else:
            issues.append("WEAK: Empty next_steps for a finding that warrants action")
            score += 1
    elif len(next_steps) < 20:
        issues.append("WEAK: next_steps too short (<20 chars)")
        score += 2
    else:
        score += 5

    # 4. chart_guidance is specific (references axes, lines, colors) — 5 pts
    if not chart_guidance:
        issues.append("MISSING: Empty chart_guidance")
    elif len(chart_guidance) < 30:
        issues.append("WEAK: chart_guidance too short")
        score += 2
    else:
        chart_specifics = (
            "line",
            "bar",
            "axis",
            "red",
            "green",
            "dashed",
            "histogram",
            "scatter",
            "box",
            "limit",
            "threshold",
            "UCL",
            "LCL",
            "marker",
            "overlay",
            "distribution",
        )
        if any(s.lower() in chart_guidance.lower() for s in chart_specifics):
            score += 5
        else:
            score += 3
            issues.append("OK: chart_guidance present but generic")

    return score, max_score, issues


# ─── Comparison with old handler narratives (representative samples) ──────

OLD_HANDLER_SAMPLES = {
    ("stats", "ttest"): {
        "verdict": "The mean of diameter is significantly higher than 10.0",
        "body": "The sample mean (10.400) differs from the hypothesized value by 0.400 units — a medium effect (Cohen's d = 0.62). The 95% CI [0.150, 0.650] excludes the hypothesized value.",
        "next_steps": "Investigate what is causing the shift from the target value.",
        "chart_guidance": "The histogram shows the distribution of diameter. The green line marks the target value (10.0). The normal overlay helps assess distributional shape.",
    },
    ("spc", "capability"): {
        "verdict": "Process is capable — Cpk = 1.45, exceeds 1.33 threshold",
        "body": "Cpk = 1.45 (short-term), Ppk = 1.38 (long-term). Process is centered and within spec. 0.12% predicted out of specification.",
        "next_steps": "Process is performing well. Monitor for shifts using control charts. Consider reducing variation further for Cpk > 1.67.",
        "chart_guidance": "The capability histogram shows measurement distribution vs spec limits (red dashed). Green area = within spec. Numbers below show short/long-term indices.",
    },
    ("pbs", "pbs_belief"): {
        "verdict": "ALARM — P(shift) = 0.970 on 'torque'. Investigate immediately.",
        "body": "The belief chart tracks the Bayesian posterior probability that the process mean has shifted from its baseline. After 120 observations, the current shift probability is 0.970.",
        "next_steps": "Identify the assignable cause. Check recent material lots, operator changes, or equipment events.",
        "chart_guidance": "Red line shows P(shift) at each observation. Dashed line at 0.95 = alarm threshold. Dotted at 0.50 = watch threshold. Red markers indicate alarm points.",
    },
}


# ─── Main audit ───────────────────────────────────────────────────────────


def run_audit():
    print("=" * 80)
    print("NARRATIVE QUALITY AUDIT — forgenarr vs canonical handler baseline")
    print("=" * 80)
    print()

    # Report registered templates
    print(f"Registry: {len(_REGISTRY)} specific templates, {len(_TYPE_DEFAULTS)} type defaults")
    print()

    results = []
    total_score = 0
    total_max = 0

    for (atype, aid), case in sorted(TEST_CASES.items()):
        stats = case["stats"]
        config = case["config"]

        try:
            narr = narrate(atype, aid, stats, config)
        except Exception as e:
            print(f"  ERROR: {atype}/{aid} — {e}")
            results.append((atype, aid, 0, 20, [f"CRASH: {e}"]))
            total_max += 20
            continue

        score, max_score, issues = score_narrative(narr, atype, aid, stats)
        total_score += score
        total_max += max_score
        results.append((atype, aid, score, max_score, issues))

    # Print results sorted by score (worst first)
    results.sort(key=lambda x: x[2])

    print("─" * 80)
    print(f"{'Type':<12} {'ID':<25} {'Score':<8} {'Issues'}")
    print("─" * 80)

    critical_count = 0
    weak_count = 0
    ok_count = 0

    for atype, aid, score, max_score, issues in results:
        pct = score / max_score * 100
        if pct < 50:
            marker = "🔴"
            critical_count += 1
        elif pct < 75:
            marker = "🟡"
            weak_count += 1
        else:
            marker = "🟢"
            ok_count += 1

        issue_str = "; ".join(issues) if issues else "✓"
        print(f"{marker} {atype:<10} {aid:<25} {score}/{max_score:<4} {issue_str}")

    print("─" * 80)
    print(f"\nOverall: {total_score}/{total_max} ({total_score / total_max * 100:.0f}%)")
    print(f"  🟢 Good (≥75%): {ok_count}")
    print(f"  🟡 Weak (50-74%): {weak_count}")
    print(f"  🔴 Critical (<50%): {critical_count}")

    # Comparison with old handler samples
    print("\n" + "=" * 80)
    print("COMPARISON — forgenarr vs old inline handlers (select samples)")
    print("=" * 80)

    for (atype, aid), old_narr in OLD_HANDLER_SAMPLES.items():
        if (atype, aid) not in TEST_CASES:
            continue
        case = TEST_CASES[(atype, aid)]
        new_narr = narrate(atype, aid, case["stats"], case["config"])

        print(f"\n── {atype}/{aid} ──")
        for key in ("verdict", "body", "next_steps", "chart_guidance"):
            old_val = old_narr.get(key, "")
            new_val = new_narr.get(key, "")
            if old_val == new_val:
                print(f"  {key}: ✓ MATCH")
            else:
                old_len = len(old_val)
                new_len = len(new_val)
                delta = new_len - old_len
                quality = "BETTER" if new_len >= old_len * 0.8 else "SHORTER"
                print(f"  {key}: {quality} ({new_len} chars vs {old_len} old, Δ{delta:+d})")
                # Show content if notably different
                if new_len < old_len * 0.5:
                    print(f"    OLD: {old_val[:120]}...")
                    print(f"    NEW: {new_val[:120]}")

    # List unregistered analysis IDs (in registry but no forgenarr template)
    print("\n" + "=" * 80)
    print("COVERAGE GAPS — analysis IDs with no forgenarr template")
    print("=" * 80)

    # Check what IDs exist in the workbench but have no forgenarr coverage
    missing_specific = []

    # Common IDs from the ANALYSIS_REGISTRY that should have specific templates
    EXPECTED_IDS = [
        ("stats", "mann_whitney"),
        ("stats", "wilcoxon"),
        ("stats", "kruskal"),
        ("stats", "friedman"),
        ("stats", "spearman"),
        ("stats", "robust_regression"),
        ("stats", "stepwise"),
        ("stats", "glm"),
        ("stats", "auto_profile"),
        ("stats", "data_profile"),
        ("spc", "laney_p"),
        ("spc", "laney_u"),
        ("spc", "mewma"),
        ("spc", "conformal_control"),
        ("spc", "entropy_spc"),
        ("spc", "capability_sixpack"),
        ("spc", "nonnormal_capability"),
        ("bayesian", "bayes_ab"),
        ("bayesian", "bayes_anova"),
        ("bayesian", "bayes_regression"),
        ("bayesian", "bayes_changepoint"),
        ("ml", "classification"),
        ("ml", "clustering"),
        ("ml", "pca"),
        ("ml", "shap_explain"),
    ]

    for atype, aid in EXPECTED_IDS:
        if (atype, aid) not in _REGISTRY:
            has_default = atype in _TYPE_DEFAULTS
            missing_specific.append((atype, aid, has_default))

    if missing_specific:
        print(f"\n{len(missing_specific)} expected IDs without specific templates:")
        for atype, aid, has_default in sorted(missing_specific):
            fallback = "→ type default" if has_default else "→ GENERIC FALLBACK"
            print(f"  ({atype}, {aid}) {fallback}")
    else:
        print("\nAll expected IDs have specific templates. ✓")


if __name__ == "__main__":
    run_audit()
