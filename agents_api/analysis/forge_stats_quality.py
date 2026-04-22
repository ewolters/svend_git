"""Forge-backed quality and capability analysis handlers.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging

import numpy as np
import pandas as pd

from .forge_stats import _alpha, _rich_summary, _to_chart

logger = logging.getLogger(__name__)


# =============================================================================
# ANOM (Analysis of Means)
# =============================================================================


def forge_anom(df, config):
    """Analysis of Means via forgestat."""
    from forgestat.quality.anom import anom
    from forgeviz.charts.generic import bar

    response = config.get("response") or config.get("var") or config.get("var1")
    factor = config.get("factor") or config.get("group_var") or config.get("var2")
    alpha = _alpha(config)

    if not response or not factor:
        raise ValueError("ANOM requires response and factor")

    data_clean = df[[response, factor]].dropna()
    labels = sorted(data_clean[factor].unique().tolist(), key=str)
    groups_data = [
        pd.to_numeric(data_clean[data_clean[factor] == g][response], errors="coerce").dropna().values for g in labels
    ]

    result = anom(*groups_data, labels=[str(lb) for lb in labels], alpha=alpha)

    outside = [g.name for g in result.groups if g.exceeds_upper or g.exceeds_lower]
    group_means = [g.mean for g in result.groups]

    chart = bar(
        categories=[g.name for g in result.groups],
        values=group_means,
        title="ANOM Chart",
        y_label=response,
    )
    plots = [_to_chart(chart)]

    comp_items = [
        (g.name, f"mean={g.mean:.4f}, n={g.n}" + (" *OUTSIDE*" if g.exceeds_upper or g.exceeds_lower else ""))
        for g in result.groups
    ]

    if outside:
        verdict = f"ANOM: {len(outside)} of {result.n_groups} groups differ from overall mean"
        body = f"Groups outside decision limits: <strong>{', '.join(outside)}</strong>."
    else:
        verdict = f"ANOM: All {result.n_groups} groups within decision limits"
        body = "No group means are significantly different from the overall mean."

    return {
        "plots": plots,
        "statistics": {
            "grand_mean": round(result.grand_mean, 4),
            "mse": round(result.mse, 4),
            "upper_limit": round(result.upper_limit, 4),
            "lower_limit": round(result.lower_limit, 4),
            "k": result.n_groups,
            "outside_limits": outside,
            "group_means": {g.name: round(g.mean, 4) for g in result.groups},
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ANALYSIS OF MEANS (ANOM)",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Factor", str(factor)),
                        ("Groups", str(result.n_groups)),
                        ("Grand Mean", f"{result.grand_mean:.4f}"),
                    ],
                ),
                ("Groups", comp_items),
                (
                    "Decision Limits",
                    [
                        ("UDL", f"{result.upper_limit:.4f}"),
                        ("LDL", f"{result.lower_limit:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": verdict,
            "body": body,
            "next_steps": "ANOM is a graphical alternative to ANOVA. Groups outside the limits warrant investigation.",
            "chart_guidance": "Points outside the decision limits differ significantly from the overall average.",
        },
        "guide_observation": f"ANOM: {len(outside)} of {result.n_groups} groups outside limits.",
        "diagnostics": [],
    }


# =============================================================================
# Attribute Capability
# =============================================================================


def forge_attribute_capability(df, config):
    """Attribute capability analysis via forgestat."""
    from forgestat.quality.capability import attribute_capability

    var = config.get("var") or config.get("var1")
    defects_count = config.get("defects")
    units_count = config.get("units")
    opportunities = int(config.get("opportunities", 1))
    event = config.get("event")

    if defects_count is not None and units_count is not None:
        d = int(float(defects_count))
        n = int(float(units_count))
    elif var:
        col = df[var].dropna()
        n = len(col)
        if event is not None:
            d = int((col.astype(str) == str(event)).sum())
        else:
            vc = col.value_counts()
            if len(vc) == 2:
                d = int(vc.iloc[-1])
            elif col.dtype in ["int64", "float64"]:
                d = int(col.sum())
            else:
                raise ValueError("Cannot auto-detect defect value. Specify 'event' in config.")
    else:
        raise ValueError("Provide var or (defects, units)")

    result = attribute_capability(d, n, opportunities)

    sigma_label = (
        "world-class"
        if result.sigma_short_term >= 6
        else "excellent"
        if result.sigma_short_term >= 5
        else "good"
        if result.sigma_short_term >= 4
        else "marginal"
        if result.sigma_short_term >= 3
        else "poor"
    )

    return {
        "plots": [],
        "statistics": {
            "defects": d,
            "units": n,
            "opportunities": opportunities,
            "dpu": round(result.dpu, 4),
            "dpo": round(result.dpo, 6),
            "dpmo": round(result.dpmo, 1),
            "yield_percent": round(result.yield_pct, 2),
            "z_bench": round(result.z_bench, 2),
            "sigma_level": round(result.sigma_short_term, 2),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ATTRIBUTE CAPABILITY ANALYSIS",
            [
                (
                    "Input",
                    [
                        ("Units", str(n)),
                        ("Defects", str(d)),
                        ("Opportunities/unit", str(opportunities)),
                    ],
                ),
                (
                    "Capability Metrics",
                    [
                        ("DPU", f"{result.dpu:.4f}"),
                        ("DPO", f"{result.dpo:.6f}"),
                        ("DPMO", f"{result.dpmo:.1f}"),
                        ("Yield %", f"{result.yield_pct:.2f}%"),
                        ("Z.bench", f"{result.z_bench:.2f}"),
                        ("Sigma level", f"{result.sigma_short_term:.2f} ({sigma_label})"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Attribute Capability: {result.sigma_short_term:.1f}σ ({sigma_label})",
            "body": f"DPMO = {result.dpmo:.0f}, Yield = {result.yield_pct:.2f}%.",
            "next_steps": "Focus on reducing DPMO. Pareto the top defect types."
            if result.sigma_short_term < 4
            else "Strong capability. Monitor and maintain.",
            "chart_guidance": "3σ = 66,807 DPMO, 4σ = 6,210, 6σ = 3.4.",
        },
        "guide_observation": f"Attribute capability: DPMO={result.dpmo:.0f}, Sigma={result.sigma_short_term:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nonparametric Capability
# =============================================================================


def forge_nonnormal_capability_np(df, config):
    """Nonparametric process capability via forgestat."""
    from forgestat.quality.capability import nonnormal_capability
    from forgeviz.charts.distribution import histogram

    var = config.get("var") or config.get("var1")
    usl = float(config.get("usl"))
    lsl = float(config.get("lsl"))

    if not var:
        raise ValueError("Nonnormal capability requires var, usl, lsl")

    data_arr = pd.to_numeric(df[var], errors="coerce").dropna().values
    if len(data_arr) < 10:
        raise ValueError("Need at least 10 data points")

    result = nonnormal_capability(data_arr, lsl=lsl, usl=usl)

    # Normal-based comparison
    std_val = float(np.std(data_arr, ddof=1))
    mean_val = float(np.mean(data_arr))
    cp_normal = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
    cpk_normal = min((usl - mean_val) / (3 * std_val), (mean_val - lsl) / (3 * std_val)) if std_val > 0 else 0

    cnpk_label = "capable" if result.cnpk >= 1.33 else "marginal" if result.cnpk >= 1.0 else "not capable"

    chart = histogram(
        data=data_arr.tolist(),
        bins=min(30, max(10, len(data_arr) // 5)),
        title=f"Distribution: {var} with Spec Limits",
        target=(usl + lsl) / 2,
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "cnp": round(result.cnp, 3),
            "cnpk": round(result.cnpk, 3),
            "cp_normal": round(cp_normal, 3),
            "cpk_normal": round(cpk_normal, 3),
            "median": round(result.median, 4),
            "p_0135": round(result.p_low, 4),
            "p_99865": round(result.p_high, 4),
            "ppm_out": round(result.ppm_out, 0),
            "is_normal": result.is_normal,
            "n": len(data_arr),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "NONPARAMETRIC PROCESS CAPABILITY",
            [
                ("Design", [("Variable", f"{var} (n={len(data_arr)})"), ("LSL", str(lsl)), ("USL", str(usl))]),
                (
                    "Comparison",
                    [
                        ("Normal Cp/Cpk", f"{cp_normal:.3f} / {cpk_normal:.3f}"),
                        ("Nonparametric Cnp/Cnpk", f"{result.cnp:.3f} / {result.cnpk:.3f}"),
                    ],
                ),
                (
                    "Nonparametric Details",
                    [
                        ("Median", f"{result.median:.4f}"),
                        ("0.135th %ile", f"{result.p_low:.4f}"),
                        ("99.865th %ile", f"{result.p_high:.4f}"),
                        ("PPM outside", f"{result.ppm_out:.0f}"),
                        ("Data normality", "normal" if result.is_normal else "NON-NORMAL"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Nonparametric Cpk = {result.cnpk:.3f} ({cnpk_label})",
            "body": f"Percentile-based capability. PPM = {result.ppm_out:.0f}. Data is {'normal' if result.is_normal else 'non-normal — this method is more appropriate'}.",
            "next_steps": "Reduce variation or center the process."
            if result.cnpk < 1.33
            else "Process is capable. Monitor with control charts.",
            "chart_guidance": "Histogram shows data distribution relative to spec limits.",
        },
        "guide_observation": f"Nonparametric capability: Cnpk={result.cnpk:.3f}, PPM={result.ppm_out:.0f}.",
        "diagnostics": [],
    }


# =============================================================================
# Acceptance Sampling (Attribute)
# =============================================================================


def forge_acceptance_sampling(df, config):
    """Acceptance sampling plan via forgestat."""
    from forgestat.quality.acceptance import attribute_plan

    aql = float(config.get("aql", 0.01))
    ltpd = float(config.get("ltpd", 0.05))
    lot_size = int(config.get("lot_size", 1000))
    alpha_risk = float(config.get("alpha", 0.05))
    beta_risk = float(config.get("beta", 0.10))

    result = attribute_plan(aql=aql, ltpd=ltpd, producer_risk=alpha_risk, consumer_risk=beta_risk, lot_size=lot_size)

    return {
        "plots": [],
        "statistics": {
            "plan_type": result.plan_type,
            "sample_size": result.sample_size,
            "acceptance_number": result.acceptance_number,
            "aql": result.aql,
            "ltpd": result.ltpd,
            "producer_risk": round(result.producer_risk, 4),
            "consumer_risk": round(result.consumer_risk, 4),
            "aoql": round(result.aoql, 6) if result.aoql else None,
            "lot_size": lot_size,
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ACCEPTANCE SAMPLING PLAN",
            [
                (
                    "Plan Parameters",
                    [
                        ("AQL", f"{aql * 100:.1f}%"),
                        ("LTPD", f"{ltpd * 100:.1f}%"),
                        ("Producer risk (α)", f"{alpha_risk}"),
                        ("Consumer risk (β)", f"{beta_risk}"),
                        ("Lot size", str(lot_size)),
                    ],
                ),
                (
                    "Sampling Plan",
                    [
                        ("Sample size (n)", str(result.sample_size)),
                        ("Accept number (c)", str(result.acceptance_number)),
                    ],
                ),
                (
                    "Risk",
                    [
                        ("Producer risk", f"{result.producer_risk:.4f}"),
                        ("Consumer risk", f"{result.consumer_risk:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Acceptance plan: n={result.sample_size}, c={result.acceptance_number}",
            "body": f"Inspect {result.sample_size} items. Accept if ≤ {result.acceptance_number} defectives found.",
            "next_steps": "The OC curve shows acceptance probability vs lot quality.",
            "chart_guidance": "",
        },
        "guide_observation": f"Sampling plan: n={result.sample_size}, c={result.acceptance_number}, AQL={aql * 100:.1f}%.",
        "diagnostics": [],
    }


# =============================================================================
# Variable Acceptance Sampling
# =============================================================================


def forge_variable_acceptance_sampling(df, config):
    """Variables acceptance sampling plan via forgestat."""
    from forgestat.quality.acceptance import variable_plan

    aql = float(config.get("aql", 1.0)) / 100  # input as percentage
    ltpd = float(config.get("ltpd", config.get("rql", 5.0))) / 100
    alpha_risk = float(config.get("alpha", 0.05))
    beta_risk = float(config.get("beta", 0.10))

    result = variable_plan(aql=aql, ltpd=ltpd, producer_risk=alpha_risk, consumer_risk=beta_risk)

    return {
        "plots": [],
        "statistics": {
            "plan_type": result.plan_type,
            "sample_size": result.sample_size,
            "k_value": round(result.k_value, 4),
            "aql": aql * 100,
            "ltpd": ltpd * 100,
            "producer_risk": round(result.producer_risk, 4),
            "consumer_risk": round(result.consumer_risk, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "VARIABLES ACCEPTANCE SAMPLING PLAN",
            [
                (
                    "Parameters",
                    [
                        ("AQL", f"{aql * 100:.1f}%"),
                        ("LTPD", f"{ltpd * 100:.1f}%"),
                        ("α", str(alpha_risk)),
                        ("β", str(beta_risk)),
                    ],
                ),
                (
                    "Plan",
                    [
                        ("Sample size (n)", str(result.sample_size)),
                        ("Critical value (k)", f"{result.k_value:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Variables plan: n={result.sample_size}, k={result.k_value:.4f}",
            "body": f"Sample {result.sample_size} items. Accept if Z_stat >= {result.k_value:.4f}.",
            "next_steps": "Assumes normally distributed measurements. Verify normality before use.",
            "chart_guidance": "",
        },
        "guide_observation": f"Variables sampling: n={result.sample_size}, k={result.k_value:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Variance Components
# =============================================================================


def forge_variance_components(df, config):
    """Variance components via forgestat."""
    from forgestat.quality.variance_components import one_way_random
    from forgeviz.charts.generic import bar

    response = config.get("response") or config.get("var")
    factors = config.get("factors", [])
    if not factors and config.get("factor"):
        factors = [config["factor"]]

    if not response or not factors:
        raise ValueError("Variance components requires response and at least one factor")

    factor = factors[0]  # forgestat supports one-way
    data_clean = df[[response, factor]].dropna()

    groups = {}
    for g in sorted(data_clean[factor].unique(), key=str):
        vals = pd.to_numeric(data_clean[data_clean[factor] == g][response], errors="coerce").dropna()
        groups[str(g)] = vals.tolist()

    result = one_way_random(groups, factor_name=factor)

    comp_items = [
        (c.source, f"variance={c.variance:.4f}, {c.pct_contribution:.1f}%, StDev={c.std_dev:.4f}")
        for c in result.components
    ]

    top_comp = max(result.components, key=lambda c: c.pct_contribution) if result.components else None

    chart = bar(
        categories=[c.source for c in result.components],
        values=[c.pct_contribution for c in result.components],
        title="Variance Components (%)",
        y_label="% of Total",
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "components": {
                c.source: {
                    "variance": round(c.variance, 4),
                    "pct": round(c.pct_contribution, 1),
                    "std_dev": round(c.std_dev, 4),
                }
                for c in result.components
            },
            "total_variance": round(result.total_variance, 4),
            "icc": round(result.icc, 4),
            "n": len(data_clean),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "VARIANCE COMPONENTS ANALYSIS",
            [
                ("Design", [("Response", response), ("Factor", factor), ("N", str(len(data_clean)))]),
                ("Components", comp_items),
                ("Total", [("Total variance", f"{result.total_variance:.4f}"), ("ICC", f"{result.icc:.4f}")]),
            ],
        ),
        "narrative": {
            "verdict": f"Variance Components: {top_comp.source} dominates ({top_comp.pct_contribution:.1f}%)"
            if top_comp
            else "No components",
            "body": f"ICC = {result.icc:.4f}. "
            + ", ".join(f"{c.source}={c.pct_contribution:.1f}%" for c in result.components)
            + ".",
            "next_steps": f"Focus improvement on <strong>{top_comp.source}</strong> — reducing the largest component gives the most impact."
            if top_comp
            else "",
            "chart_guidance": "Bar chart shows each source's contribution to total variance.",
        },
        "guide_observation": "Variance components: "
        + ", ".join(f"{c.source}={c.pct_contribution:.1f}%" for c in result.components)
        + ".",
        "diagnostics": [],
    }


# =============================================================================
# Capability Sixpack
# =============================================================================


def forge_capability_sixpack(df, config):
    """Process capability sixpack — 6-panel diagnostic display."""
    from forgeviz.charts.distribution import histogram
    from scipy.stats import norm as norm_dist

    var = config.get("var") or config.get("column") or config.get("var1")
    lsl_raw = config.get("lsl")
    usl_raw = config.get("usl")
    target_raw = config.get("target")

    if not var:
        raise ValueError("Capability sixpack requires var")

    data = pd.to_numeric(df[var], errors="coerce").dropna().values
    n = len(data)
    if n < 10:
        raise ValueError("Need at least 10 observations")

    lsl = float(lsl_raw) if lsl_raw is not None and str(lsl_raw).strip() else None
    usl = float(usl_raw) if usl_raw is not None and str(usl_raw).strip() else None
    target = float(target_raw) if target_raw is not None and str(target_raw).strip() else None

    if lsl is None and usl is None:
        raise ValueError("At least one spec limit (LSL or USL) required")
    if target is None and lsl is not None and usl is not None:
        target = (lsl + usl) / 2

    x_bar = float(np.mean(data))
    s = float(np.std(data, ddof=1))

    # Capability indices
    cp = cpu = cpl = cpk = None
    if lsl is not None and usl is not None and s > 0:
        cp = (usl - lsl) / (6 * s)
        cpu = (usl - x_bar) / (3 * s)
        cpl = (x_bar - lsl) / (3 * s)
        cpk = min(cpu, cpl)
    elif usl is not None and s > 0:
        cpu = (usl - x_bar) / (3 * s)
        cpk = cpu
    elif lsl is not None and s > 0:
        cpl = (x_bar - lsl) / (3 * s)
        cpk = cpl

    ppm_below = float(norm_dist.cdf((lsl - x_bar) / s) * 1e6) if lsl is not None and s > 0 else 0
    ppm_above = float((1 - norm_dist.cdf((usl - x_bar) / s)) * 1e6) if usl is not None and s > 0 else 0
    ppm_total = ppm_below + ppm_above

    # Histogram chart
    chart = histogram(
        data=data.tolist(),
        bins=min(30, max(10, n // 5)),
        title=f"Capability: {var}",
        target=target,
    )
    plots = [_to_chart(chart)]

    cap_items = []
    if cp is not None:
        cap_items.append(("Cp", f"{cp:.3f}"))
    if cpk is not None:
        cap_items.append(("Cpk", f"{cpk:.3f}"))
    if cpl is not None:
        cap_items.append(("CPL", f"{cpl:.3f}"))
    if cpu is not None:
        cap_items.append(("CPU", f"{cpu:.3f}"))

    cpk_label = "capable" if cpk and cpk >= 1.33 else "marginal" if cpk and cpk >= 1.0 else "not capable"

    return {
        "plots": plots,
        "statistics": {
            "n": n,
            "mean": round(x_bar, 4),
            "std_dev": round(s, 4),
            "cp": round(cp, 3) if cp else None,
            "cpk": round(cpk, 3) if cpk else None,
            "cpl": round(cpl, 3) if cpl else None,
            "cpu": round(cpu, 3) if cpu else None,
            "ppm_below": round(ppm_below, 1),
            "ppm_above": round(ppm_above, 1),
            "ppm_total": round(ppm_total, 1),
            "lsl": lsl,
            "usl": usl,
            "target": target,
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"PROCESS CAPABILITY SIXPACK — {var}",
            [
                ("Specs", [("LSL", str(lsl)), ("USL", str(usl)), ("Target", str(target))]),
                ("Process Stats", [("N", str(n)), ("Mean", f"{x_bar:.4f}"), ("StDev", f"{s:.4f}")]),
                ("Capability", cap_items),
                (
                    "Expected PPM",
                    [
                        ("Below LSL", f"{ppm_below:.1f}"),
                        ("Above USL", f"{ppm_above:.1f}"),
                        ("Total", f"{ppm_total:.1f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Capability Sixpack: Cpk = {cpk:.3f} ({cpk_label})" if cpk else "Capability computed",
            "body": f"PPM total = {ppm_total:.1f}."
            + (" Process is capable." if cpk and cpk >= 1.33 else " Improvement needed." if cpk else ""),
            "next_steps": "Review I-MR chart for stability before interpreting capability indices.",
            "chart_guidance": "Histogram shows distribution vs spec limits. Check for centering and spread.",
        },
        "guide_observation": f"Sixpack: Cpk={cpk:.3f}, PPM={ppm_total:.1f}." if cpk else "Sixpack computed.",
        "diagnostics": [],
    }


# =============================================================================
# Multiple Plan Comparison
# =============================================================================


def forge_multiple_plan_comparison(df, config):
    """Compare multiple acceptance sampling plans."""
    from scipy import stats as sp_stats

    plans_input = config.get("plans", [])
    lot_size = int(config.get("lot_size", 1000))
    aql = float(config.get("aql", 0.01))
    ltpd = float(config.get("ltpd", 0.05))

    if not plans_input or len(plans_input) < 2:
        raise ValueError("Provide at least 2 sampling plans to compare")

    p_range = np.linspace(0, min(0.20, ltpd * 3), 200)
    plan_results = []

    for idx, plan in enumerate(plans_input):
        plan_name = plan.get("name", f"Plan {idx + 1}")
        n_plan = int(plan.get("n", plan.get("sample_size", 50)))
        c_plan = int(plan.get("c", plan.get("accept_number", 2)))

        pa_vals = np.array([float(sp_stats.binom.cdf(c_plan, n_plan, p)) if p > 0 else 1.0 for p in p_range])
        pa_aql = float(np.interp(aql, p_range, pa_vals))
        pa_ltpd = float(np.interp(ltpd, p_range, pa_vals))
        alpha_r = 1 - pa_aql
        beta_r = pa_ltpd
        aoq_vals = pa_vals * p_range * (lot_size - n_plan) / lot_size
        aoql = float(np.max(aoq_vals))

        plan_results.append(
            {
                "name": plan_name,
                "n": n_plan,
                "c": c_plan,
                "pa_aql": round(pa_aql, 4),
                "pa_ltpd": round(pa_ltpd, 4),
                "alpha": round(alpha_r, 4),
                "beta": round(beta_r, 4),
                "aoql": round(aoql, 6),
            }
        )

    best_beta = min(pr["beta"] for pr in plan_results)

    comp_items = [
        (pr["name"], f"n={pr['n']}, c={pr['c']}, α={pr['alpha']:.3f}, β={pr['beta']:.3f}, AOQL={pr['aoql'] * 100:.3f}%")
        for pr in plan_results
    ]

    return {
        "plots": [],
        "statistics": {
            "plans": plan_results,
            "aql": aql,
            "ltpd": ltpd,
            "lot_size": lot_size,
        },
        "assumptions": {},
        "summary": _rich_summary(
            "SAMPLING PLAN COMPARISON",
            [
                (
                    "Parameters",
                    [("Lot size", str(lot_size)), ("AQL", f"{aql * 100:.1f}%"), ("LTPD", f"{ltpd * 100:.1f}%")],
                ),
                ("Plans", comp_items),
                ("Best consumer risk", [("β", f"{best_beta:.3f}")]),
            ],
        ),
        "narrative": {
            "verdict": f"Compared {len(plan_results)} plans. Best β = {best_beta:.3f}",
            "body": "Compare OC curves to find the best trade-off between sample size and protection.",
            "next_steps": "Choose the plan that balances inspection cost with risk. Lower β = better consumer protection.",
            "chart_guidance": "",
        },
        "guide_observation": f"Plan comparison: {len(plan_results)} plans, best β={best_beta:.3f}.",
        "diagnostics": [],
    }


# =============================================================================
# Registry: maps analysis_id → forge handler
# =============================================================================


FORGE_QUALITY_HANDLERS = {
    "anom": forge_anom,
    "attribute_capability": forge_attribute_capability,
    "nonnormal_capability_np": forge_nonnormal_capability_np,
    "acceptance_sampling": forge_acceptance_sampling,
    "variable_acceptance_sampling": forge_variable_acceptance_sampling,
    "variance_components": forge_variance_components,
    "capability_sixpack": forge_capability_sixpack,
    "multiple_plan_comparison": forge_multiple_plan_comparison,
}
