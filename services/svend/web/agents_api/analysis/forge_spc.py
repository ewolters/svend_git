"""Forge-backed SPC analysis handlers.

Replaces inline SPC code with forgespc computation + ForgeViz charts.
Returns the same dict schema as the legacy handlers so dispatch.py and
standardize.py work without changes.

Mirrors forge_stats.py structure. Each function here replaces one
analysis_id in the legacy spc/ package. When all IDs in a category
are ported, the legacy file can be deleted.

CR: pending
"""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _col(df, config, key, fallback_key=None):
    """Extract a clean numeric array from df using config key.

    Coerces to numeric, drops NaN/non-parseable values.
    """
    name = config.get(key) or config.get(fallback_key or key)
    if not name:
        nums = df.select_dtypes(include="number").columns
        if len(nums) == 0:
            raise ValueError("No numeric columns in dataset")
        name = nums[0]
    if name not in df.columns:
        raise ValueError(f"Column '{name}' not found")
    series = pd.to_numeric(df[name], errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Column '{name}' has no valid numeric values")
    return series.values, name


def _int_col(df, config, key):
    """Extract a clean integer array from df using config key.

    For attribute data (defective counts, sample sizes).
    """
    name = config.get(key)
    if not name:
        raise ValueError(f"Config key '{key}' is required")
    if name not in df.columns:
        raise ValueError(f"Column '{name}' not found")
    series = pd.to_numeric(df[name], errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Column '{name}' has no valid values")
    return series.astype(int).values.tolist(), name


def _float_col(df, config, key):
    """Extract a clean float array from df using config key."""
    name = config.get(key)
    if not name:
        raise ValueError(f"Config key '{key}' is required")
    if name not in df.columns:
        raise ValueError(f"Column '{name}' not found")
    series = pd.to_numeric(df[name], errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Column '{name}' has no valid values")
    return series.values.tolist(), name


def _to_chart(spec):
    """Convert ForgeViz ChartSpec to dict for the result schema."""
    return spec.to_dict()


def _pval_str(p):
    """Format p-value for summary text."""
    if p is None:
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


def _education(analysis_type, analysis_id):
    """Fetch hand-written education content for this analysis."""
    try:
        from .education import get_education

        return get_education(analysis_type, analysis_id)
    except Exception:
        return None


def _rich_summary(title, sections):
    """Build a rich <<COLOR:>> formatted summary matching legacy output."""
    lines = [
        "<<COLOR:accent>>======================================================================<</COLOR>>",
        f"<<COLOR:title>>{title}<</COLOR>>",
        "<<COLOR:accent>>======================================================================<</COLOR>>",
        "",
    ]
    for heading, items in sections:
        lines.append(f"<<COLOR:accent>>-- {heading} --<</COLOR>>")
        for label, value in items:
            lines.append(f"  <<COLOR:highlight>>{label}:<</COLOR>> {value}")
        lines.append("")
    return "\n".join(lines)


def _jsonify(val):
    """Ensure value is JSON-serializable (no numpy types)."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _jsonify(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonify(v) for v in val]
    if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
        return None
    return val


def _build_subgroups(data, subgroup_size):
    """Split flat data array into subgroups."""
    n = len(data)
    sg_size = int(subgroup_size)
    if sg_size < 2:
        raise ValueError("Subgroup size must be >= 2")
    n_full = (n // sg_size) * sg_size
    if n_full < sg_size * 2:
        raise ValueError(f"Need at least {sg_size * 2} observations for subgroup size {sg_size}")
    trimmed = data[:n_full]
    return [trimmed[i : i + sg_size].tolist() for i in range(0, n_full, sg_size)]


def _nelson_diagnostics(result):
    """Extract Nelson rule violations from a ControlChartResult as diagnostics list."""
    diags = []

    # Out-of-control points
    ooc = result.out_of_control or []
    if ooc:
        ooc_indices = [p["index"] for p in ooc]
        reasons = list({p.get("reason", "OOC") for p in ooc})
        diags.append(
            {
                "level": "warning",
                "title": f"{len(ooc)} out-of-control point(s) detected",
                "detail": f"Indices: {ooc_indices[:10]}{'...' if len(ooc_indices) > 10 else ''}. "
                f"Reasons: {', '.join(reasons)}",
            }
        )

    # Run rule violations (Nelson/Western Electric)
    violations = result.run_violations or []
    for v in violations:
        rule = v.get("rule", v.get("description", "Run rule"))
        indices = v.get("indices", [])
        desc = v.get("description", "")
        diags.append(
            {
                "level": "warning",
                "title": f"Nelson rule violation: {rule}",
                "detail": f"{desc}. Indices: {indices[:10]}{'...' if len(indices) > 10 else ''}" if indices else desc,
            }
        )

    if not diags:
        diags.append(
            {
                "level": "info",
                "title": "Process appears stable",
                "detail": "No out-of-control points or run rule violations detected.",
            }
        )

    return diags


def _control_status(result):
    """Return human-readable control status string."""
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    n_viol = len(result.run_violations) if result.run_violations else 0
    if n_ooc == 0 and n_viol == 0:
        return "IN CONTROL"
    parts = []
    if n_ooc:
        parts.append(f"{n_ooc} OOC point(s)")
    if n_viol:
        parts.append(f"{n_viol} run rule violation(s)")
    return f"OUT OF CONTROL - {', '.join(parts)}"


def _cpk_interpretation(cpk):
    """Interpret Cpk value."""
    if cpk is None:
        return "N/A"
    if cpk >= 2.0:
        return f"Cpk = {cpk:.3f} - World class (Six Sigma)"
    if cpk >= 1.67:
        return f"Cpk = {cpk:.3f} - Excellent capability"
    if cpk >= 1.33:
        return f"Cpk = {cpk:.3f} - Capable process"
    if cpk >= 1.0:
        return f"Cpk = {cpk:.3f} - Barely capable, improvement needed"
    return f"Cpk = {cpk:.3f} - NOT capable, immediate action required"


def _grr_assessment(grr_pct):
    """Interpret %GRR value per AIAG guidelines."""
    if grr_pct < 10:
        return f"%GRR = {grr_pct:.1f}% - Acceptable measurement system"
    if grr_pct < 30:
        return f"%GRR = {grr_pct:.1f}% - Marginal, may be acceptable depending on application"
    return f"%GRR = {grr_pct:.1f}% - Unacceptable, measurement system needs improvement"


# =============================================================================
# I-MR Chart
# =============================================================================


def forge_imr(df, config):
    """I-MR (Individuals and Moving Range) chart."""
    from forgespc.charts import individuals_moving_range_chart
    from forgeviz.charts.control import from_spc_result_pair

    data, col_name = _col(df, config, "column")
    result = individuals_moving_range_chart(data.tolist())

    chart_specs = from_spc_result_pair(result, title=f"I-MR Chart: {col_name}")
    plots = [_to_chart(s) for s in chart_specs]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    limits = result.limits
    mean_val = float(limits.cl)
    ucl_val = float(limits.ucl)
    lcl_val = float(limits.lcl)

    # Secondary (MR) stats
    mr_stats = {}
    if result.secondary_chart:
        mr = result.secondary_chart
        mr_stats = {
            "MR_bar": float(mr.limits.cl),
            "MR_UCL": float(mr.limits.ucl),
            "MR_LCL": float(mr.limits.lcl),
            "MR_OOC_count": len(mr.out_of_control) if mr.out_of_control else 0,
        }

    diagnostics = _nelson_diagnostics(result)
    if result.secondary_chart:
        mr_diags = _nelson_diagnostics(result.secondary_chart)
        for d in mr_diags:
            d["title"] = f"MR Chart: {d['title']}"
        diagnostics.extend(mr_diags)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n": len(data),
                "mean": round(mean_val, 4),
                "UCL": round(ucl_val, 4),
                "LCL": round(lcl_val, 4),
                "std_dev": round(float(np.std(data, ddof=1)), 4),
                "OOC_count": n_ooc,
                "status": status,
                **mr_stats,
            },
            "summary": _rich_summary(
                "I-MR CONTROL CHART",
                [
                    ("Variable", [(col_name, f"n = {len(data)}")]),
                    (
                        "I Chart Limits",
                        [
                            ("UCL", f"{ucl_val:.4f}"),
                            ("CL (Mean)", f"{mean_val:.4f}"),
                            ("LCL", f"{lcl_val:.4f}"),
                        ],
                    ),
                    (
                        "MR Chart",
                        [
                            ("MR-bar", f"{mr_stats.get('MR_bar', 0):.4f}"),
                            ("MR UCL", f"{mr_stats.get('MR_UCL', 0):.4f}"),
                        ],
                    )
                    if mr_stats
                    else ("MR Chart", [("Status", "N/A")]),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Process is {status}",
                "body": (
                    f"The I-MR chart for {col_name} shows the process "
                    f"{'is stable with no out-of-control signals' if n_ooc == 0 else f'has {n_ooc} out-of-control point(s) requiring investigation'}. "
                    f"The process mean is {mean_val:.4f} with control limits at [{lcl_val:.4f}, {ucl_val:.4f}]."
                ),
                "next_steps": (
                    "Continue monitoring. The process is stable."
                    if n_ooc == 0
                    else "Investigate out-of-control points for assignable causes. "
                    "Consider 5-Why or fishbone analysis on flagged observations."
                ),
                "chart_guidance": "Top panel is the Individuals chart (each observation). Bottom panel is the Moving Range chart (consecutive differences).",
            },
            "assumptions": {
                "independence": {
                    "pass": True,
                    "detail": "Observations assumed independent. Check for autocorrelation if time-ordered.",
                },
                "normality": {
                    "pass": True,
                    "detail": "Control charts are robust to mild non-normality with sufficient data.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"I-MR chart for {col_name}: process is {status.lower()}. Mean = {mean_val:.4f}, {n_ooc} OOC points."
            ),
        }
    )


# =============================================================================
# X-bar/R Chart
# =============================================================================


def forge_xbar_r(df, config):
    """X-bar/R control chart."""
    from forgespc.charts import xbar_r_chart
    from forgeviz.charts.control import from_spc_result_pair

    data, col_name = _col(df, config, "column")
    sg_size = config.get("subgroup_size", 5)
    subgroups = _build_subgroups(data, sg_size)

    result = xbar_r_chart(subgroups)

    chart_specs = from_spc_result_pair(result, title=f"X-bar/R Chart: {col_name}")
    plots = [_to_chart(s) for s in chart_specs]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    limits = result.limits

    r_stats = {}
    if result.secondary_chart:
        r = result.secondary_chart
        r_stats = {
            "R_bar": float(r.limits.cl),
            "R_UCL": float(r.limits.ucl),
            "R_LCL": float(r.limits.lcl),
            "R_OOC_count": len(r.out_of_control) if r.out_of_control else 0,
        }

    diagnostics = _nelson_diagnostics(result)
    if result.secondary_chart:
        r_diags = _nelson_diagnostics(result.secondary_chart)
        for d in r_diags:
            d["title"] = f"R Chart: {d['title']}"
        diagnostics.extend(r_diags)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_observations": len(data),
                "n_subgroups": len(subgroups),
                "subgroup_size": sg_size,
                "grand_mean": round(float(limits.cl), 4),
                "UCL": round(float(limits.ucl), 4),
                "LCL": round(float(limits.lcl), 4),
                "OOC_count": n_ooc,
                "status": status,
                **r_stats,
            },
            "summary": _rich_summary(
                "X-BAR/R CONTROL CHART",
                [
                    ("Variable", [(col_name, f"n = {len(data)}, {len(subgroups)} subgroups of {sg_size}")]),
                    (
                        "X-bar Limits",
                        [
                            ("UCL", f"{float(limits.ucl):.4f}"),
                            ("CL (Grand Mean)", f"{float(limits.cl):.4f}"),
                            ("LCL", f"{float(limits.lcl):.4f}"),
                        ],
                    ),
                    (
                        "R Chart",
                        [
                            ("R-bar", f"{r_stats.get('R_bar', 0):.4f}"),
                            ("R UCL", f"{r_stats.get('R_UCL', 0):.4f}"),
                            ("R LCL", f"{r_stats.get('R_LCL', 0):.4f}"),
                        ],
                    )
                    if r_stats
                    else ("R Chart", [("Status", "N/A")]),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Process is {status}",
                "body": (
                    f"X-bar/R analysis of {col_name} with subgroup size {sg_size}. "
                    f"Grand mean = {float(limits.cl):.4f}. "
                    f"{'Process is stable.' if n_ooc == 0 else f'{n_ooc} subgroup(s) exceed control limits.'}"
                ),
                "next_steps": (
                    "Process is stable. Use for baseline capability analysis."
                    if n_ooc == 0
                    else "Investigate out-of-control subgroups. Look for special cause variation."
                ),
                "chart_guidance": "Top: X-bar chart (subgroup means). Bottom: R chart (subgroup ranges).",
            },
            "assumptions": {
                "subgroup_rational": {
                    "pass": True,
                    "detail": f"Subgroups of size {sg_size}. Ensure subgroups represent rational groupings.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"X-bar/R chart for {col_name}: {len(subgroups)} subgroups of {sg_size}, "
                f"grand mean = {float(limits.cl):.4f}, {status.lower()}."
            ),
        }
    )


# =============================================================================
# X-bar/S Chart
# =============================================================================


def forge_xbar_s(df, config):
    """X-bar/S control chart."""
    from forgespc.advanced import xbar_s_chart
    from forgeviz.charts.control import from_spc_result_pair

    data, col_name = _col(df, config, "column")
    sg_size = config.get("subgroup_size", 5)
    subgroups = _build_subgroups(data, sg_size)

    result = xbar_s_chart(subgroups)

    chart_specs = from_spc_result_pair(result, title=f"X-bar/S Chart: {col_name}")
    plots = [_to_chart(s) for s in chart_specs]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    limits = result.limits

    s_stats = {}
    if result.secondary_chart:
        s = result.secondary_chart
        s_stats = {
            "S_bar": float(s.limits.cl),
            "S_UCL": float(s.limits.ucl),
            "S_LCL": float(s.limits.lcl),
            "S_OOC_count": len(s.out_of_control) if s.out_of_control else 0,
        }

    diagnostics = _nelson_diagnostics(result)
    if result.secondary_chart:
        s_diags = _nelson_diagnostics(result.secondary_chart)
        for d in s_diags:
            d["title"] = f"S Chart: {d['title']}"
        diagnostics.extend(s_diags)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_observations": len(data),
                "n_subgroups": len(subgroups),
                "subgroup_size": sg_size,
                "grand_mean": round(float(limits.cl), 4),
                "UCL": round(float(limits.ucl), 4),
                "LCL": round(float(limits.lcl), 4),
                "OOC_count": n_ooc,
                "status": status,
                **s_stats,
            },
            "summary": _rich_summary(
                "X-BAR/S CONTROL CHART",
                [
                    ("Variable", [(col_name, f"n = {len(data)}, {len(subgroups)} subgroups of {sg_size}")]),
                    (
                        "X-bar Limits",
                        [
                            ("UCL", f"{float(limits.ucl):.4f}"),
                            ("CL (Grand Mean)", f"{float(limits.cl):.4f}"),
                            ("LCL", f"{float(limits.lcl):.4f}"),
                        ],
                    ),
                    (
                        "S Chart",
                        [
                            ("S-bar", f"{s_stats.get('S_bar', 0):.4f}"),
                            ("S UCL", f"{s_stats.get('S_UCL', 0):.4f}"),
                            ("S LCL", f"{s_stats.get('S_LCL', 0):.4f}"),
                        ],
                    )
                    if s_stats
                    else ("S Chart", [("Status", "N/A")]),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Process is {status}",
                "body": (
                    f"X-bar/S analysis of {col_name} with subgroup size {sg_size}. "
                    f"Grand mean = {float(limits.cl):.4f}. "
                    f"X-bar/S is preferred over X-bar/R for subgroups > 10 due to better sigma estimation. "
                    f"{'Process is stable.' if n_ooc == 0 else f'{n_ooc} subgroup(s) exceed control limits.'}"
                ),
                "next_steps": (
                    "Process is stable. Consider capability analysis."
                    if n_ooc == 0
                    else "Investigate OOC subgroups for assignable causes."
                ),
                "chart_guidance": "Top: X-bar chart (subgroup means). Bottom: S chart (subgroup standard deviations).",
            },
            "assumptions": {
                "subgroup_rational": {
                    "pass": True,
                    "detail": f"Subgroups of size {sg_size}. X-bar/S is preferred when n > 10.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"X-bar/S chart for {col_name}: {len(subgroups)} subgroups of {sg_size}, "
                f"grand mean = {float(limits.cl):.4f}, {status.lower()}."
            ),
        }
    )


# =============================================================================
# p Chart
# =============================================================================


def forge_p_chart(df, config):
    """p chart for proportion defective."""
    from forgespc.charts import p_chart
    from forgeviz.charts.control import from_spc_result

    defectives, def_col = _int_col(df, config, "defectives_column")
    sample_sizes, sz_col = _int_col(df, config, "sample_size_column")

    # Ensure equal length after cleaning
    min_len = min(len(defectives), len(sample_sizes))
    defectives = defectives[:min_len]
    sample_sizes = sample_sizes[:min_len]

    result = p_chart(defectives, sample_sizes)

    spec = from_spc_result(result, title=f"p Chart: {def_col}")
    plots = [_to_chart(spec)]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    p_bar = float(result.limits.cl)

    diagnostics = _nelson_diagnostics(result)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_samples": len(defectives),
                "total_defectives": sum(defectives),
                "total_inspected": sum(sample_sizes),
                "p_bar": round(p_bar, 6),
                "UCL": round(float(result.limits.ucl), 6),
                "LCL": round(float(result.limits.lcl), 6),
                "OOC_count": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "P CHART (PROPORTION DEFECTIVE)",
                [
                    (
                        "Data",
                        [
                            (f"Defectives ({def_col})", f"{sum(defectives)} total across {len(defectives)} samples"),
                            (f"Sample sizes ({sz_col})", f"{sum(sample_sizes)} total inspected"),
                        ],
                    ),
                    (
                        "Control Limits",
                        [
                            ("p-bar", f"{p_bar:.6f}"),
                            ("UCL", f"{float(result.limits.ucl):.6f}"),
                            ("LCL", f"{float(result.limits.lcl):.6f}"),
                        ],
                    ),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Proportion defective process is {status}",
                "body": (
                    f"p chart monitors the proportion defective across {len(defectives)} samples. "
                    f"Average proportion defective (p-bar) = {p_bar:.6f} "
                    f"({p_bar * 100:.2f}%). "
                    f"{'No out-of-control signals detected.' if n_ooc == 0 else f'{n_ooc} sample(s) exceed control limits.'}"
                ),
                "next_steps": (
                    "Process is stable. Focus on reducing the baseline defective rate."
                    if n_ooc == 0
                    else "Investigate samples with abnormal defective rates for special causes."
                ),
                "chart_guidance": "Each point is the proportion defective for one sample. Limits vary with sample size.",
            },
            "assumptions": {
                "binomial": {
                    "pass": True,
                    "detail": "p chart assumes defectives follow a binomial distribution. Items are classified as defective/non-defective.",
                },
                "independence": {
                    "pass": True,
                    "detail": "Samples assumed independent.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"p chart: p-bar = {p_bar:.4f} ({p_bar * 100:.2f}%), {n_ooc} OOC points, {status.lower()}."
            ),
        }
    )


# =============================================================================
# c Chart
# =============================================================================


def forge_c_chart(df, config):
    """c chart for defect counts per unit."""
    from forgespc.charts import c_chart
    from forgeviz.charts.control import from_spc_result

    data, col_name = _int_col(df, config, "column")

    result = c_chart(data)

    spec = from_spc_result(result, title=f"c Chart: {col_name}")
    plots = [_to_chart(spec)]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    c_bar = float(result.limits.cl)

    diagnostics = _nelson_diagnostics(result)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_samples": len(data),
                "c_bar": round(c_bar, 4),
                "total_defects": sum(data),
                "UCL": round(float(result.limits.ucl), 4),
                "LCL": round(float(result.limits.lcl), 4),
                "OOC_count": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "C CHART (DEFECT COUNT)",
                [
                    ("Data", [(col_name, f"{len(data)} samples, {sum(data)} total defects")]),
                    (
                        "Control Limits",
                        [
                            ("c-bar", f"{c_bar:.4f}"),
                            ("UCL", f"{float(result.limits.ucl):.4f}"),
                            ("LCL", f"{float(result.limits.lcl):.4f}"),
                        ],
                    ),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Defect count process is {status}",
                "body": (
                    f"c chart monitors defect counts in {col_name} across {len(data)} equal-sized units. "
                    f"Average defects per unit (c-bar) = {c_bar:.4f}. "
                    f"{'Process is stable.' if n_ooc == 0 else f'{n_ooc} unit(s) have abnormal defect counts.'}"
                ),
                "next_steps": (
                    "Process stable. Pareto analysis of defect types may identify improvement opportunities."
                    if n_ooc == 0
                    else "Investigate units with high defect counts. Consider stratification by defect type."
                ),
                "chart_guidance": "Each point is the total defect count for one unit of constant size.",
            },
            "assumptions": {
                "poisson": {
                    "pass": True,
                    "detail": "c chart assumes defect counts follow a Poisson distribution. Units must be equal size.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"c chart for {col_name}: c-bar = {c_bar:.4f}, {n_ooc} OOC points, {status.lower()}."
            ),
        }
    )


# =============================================================================
# u Chart
# =============================================================================


def forge_u_chart(df, config):
    """u chart for defects per unit (variable inspection size)."""
    from forgespc.charts import u_chart
    from forgeviz.charts.control import from_spc_result

    defects, def_col = _int_col(df, config, "defects_column")
    units, units_col = _float_col(df, config, "units_column")

    min_len = min(len(defects), len(units))
    defects = defects[:min_len]
    units = units[:min_len]

    result = u_chart(defects, units)

    spec = from_spc_result(result, title=f"u Chart: {def_col}")
    plots = [_to_chart(spec)]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    u_bar = float(result.limits.cl)

    diagnostics = _nelson_diagnostics(result)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_samples": len(defects),
                "total_defects": sum(defects),
                "total_units": round(sum(units), 2),
                "u_bar": round(u_bar, 6),
                "UCL": round(float(result.limits.ucl), 6),
                "LCL": round(float(result.limits.lcl), 6),
                "OOC_count": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "U CHART (DEFECTS PER UNIT)",
                [
                    (
                        "Data",
                        [
                            (f"Defects ({def_col})", f"{sum(defects)} total"),
                            (f"Units ({units_col})", f"{sum(units):.2f} total"),
                        ],
                    ),
                    (
                        "Control Limits",
                        [
                            ("u-bar", f"{u_bar:.6f}"),
                            ("UCL", f"{float(result.limits.ucl):.6f}"),
                            ("LCL", f"{float(result.limits.lcl):.6f}"),
                        ],
                    ),
                    (
                        "Status",
                        [
                            ("Process", status),
                            ("OOC Points", str(n_ooc)),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"Defects-per-unit process is {status}",
                "body": (
                    f"u chart monitors defect rates across variable inspection sizes. "
                    f"Average defects per unit (u-bar) = {u_bar:.6f}. "
                    f"{'Process is stable.' if n_ooc == 0 else f'{n_ooc} sample(s) have abnormal defect rates.'}"
                ),
                "next_steps": (
                    "Process stable. Consider defect classification for targeted improvement."
                    if n_ooc == 0
                    else "Investigate high-rate samples. The u chart adjusts for varying inspection sizes."
                ),
                "chart_guidance": "Each point is defects/unit for that sample. Limits may vary with sample size.",
            },
            "assumptions": {
                "poisson": {
                    "pass": True,
                    "detail": "u chart assumes defects follow a Poisson distribution. Inspection sizes may vary.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (f"u chart: u-bar = {u_bar:.4f}, {n_ooc} OOC points, {status.lower()}."),
        }
    )


# =============================================================================
# CUSUM Chart
# =============================================================================


def forge_cusum(df, config):
    """CUSUM (Cumulative Sum) chart for detecting small sustained shifts."""
    from forgespc.advanced import cusum_chart
    from forgeviz.charts.control import from_spc_result

    data, col_name = _col(df, config, "column")
    target = config.get("target")
    k = config.get("k", 0.5)
    h = config.get("h", 5.0)

    if target is not None:
        target = float(target)

    cusum_result = cusum_chart(data.tolist(), target=target, k=float(k), h=float(h))

    # Convert to ControlChartResult for ForgeViz
    chart_result = cusum_result.to_chart_result()
    spec = from_spc_result(chart_result, title=f"CUSUM Chart: {col_name}")
    plots = [_to_chart(spec)]

    n_signals = cusum_result.n_signals
    status = "IN CONTROL" if cusum_result.in_control else f"OUT OF CONTROL - {n_signals} signal(s)"

    diagnostics = []
    if cusum_result.signals_up:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Upward shift detected ({len(cusum_result.signals_up)} signals)",
                "detail": f"CUSUM+ exceeded h={h} at indices: {cusum_result.signals_up[:10]}",
            }
        )
    if cusum_result.signals_down:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Downward shift detected ({len(cusum_result.signals_down)} signals)",
                "detail": f"CUSUM- exceeded h={h} at indices: {cusum_result.signals_down[:10]}",
            }
        )
    if not diagnostics:
        diagnostics.append(
            {
                "level": "info",
                "title": "No sustained shifts detected",
                "detail": f"CUSUM chart with target={cusum_result.target:.4f}, k={k}, h={h} shows stable process.",
            }
        )

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n": cusum_result.n,
                "target": round(cusum_result.target, 4),
                "sigma": round(cusum_result.sigma, 4),
                "k": k,
                "h": h,
                "n_signals_up": len(cusum_result.signals_up),
                "n_signals_down": len(cusum_result.signals_down),
                "n_signals_total": n_signals,
                "status": status,
            },
            "summary": _rich_summary(
                "CUSUM CONTROL CHART",
                [
                    ("Variable", [(col_name, f"n = {cusum_result.n}")]),
                    (
                        "Parameters",
                        [
                            ("Target", f"{cusum_result.target:.4f}"),
                            ("Sigma", f"{cusum_result.sigma:.4f}"),
                            ("Slack (k)", str(k)),
                            ("Decision Interval (h)", str(h)),
                        ],
                    ),
                    (
                        "Results",
                        [
                            ("Upward Shifts", str(len(cusum_result.signals_up))),
                            ("Downward Shifts", str(len(cusum_result.signals_down))),
                            ("Status", status),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"CUSUM analysis: {status}",
                "body": (
                    f"CUSUM chart for {col_name} with target = {cusum_result.target:.4f}. "
                    f"CUSUM is sensitive to small sustained shifts that Shewhart charts may miss. "
                    f"{'No sustained process shifts detected.' if n_signals == 0 else f'{n_signals} signal(s) indicate a process shift.'}"
                ),
                "next_steps": (
                    "Process is stable. CUSUM provides early warning of small shifts."
                    if n_signals == 0
                    else "Process shift detected. Investigate timing of shift using the CUSUM chart. "
                    "The point where CUSUM begins rising indicates when the shift started."
                ),
                "chart_guidance": "CUSUM plots cumulative deviations from target. Signals occur when the statistic exceeds h.",
            },
            "assumptions": {
                "independence": {
                    "pass": True,
                    "detail": "CUSUM assumes independent observations. Check for autocorrelation.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"CUSUM for {col_name}: target = {cusum_result.target:.4f}, {n_signals} signal(s), {status.lower()}."
            ),
        }
    )


# =============================================================================
# EWMA Chart
# =============================================================================


def forge_ewma(df, config):
    """EWMA (Exponentially Weighted Moving Average) chart."""
    from forgespc.advanced import ewma_chart
    from forgeviz.charts.control import from_spc_result

    data, col_name = _col(df, config, "column")
    target = config.get("target")
    lambda_param = config.get("lambda", config.get("lambda_param", 0.2))
    L = config.get("L", 3.0)

    if target is not None:
        target = float(target)

    ewma_result = ewma_chart(data.tolist(), target=target, lambda_param=float(lambda_param), L=float(L))

    chart_result = ewma_result.to_chart_result()
    spec = from_spc_result(chart_result, title=f"EWMA Chart: {col_name}")
    plots = [_to_chart(spec)]

    n_ooc = len(ewma_result.out_of_control_indices)
    status = "IN CONTROL" if ewma_result.in_control else f"OUT OF CONTROL - {n_ooc} signal(s)"

    diagnostics = []
    if n_ooc > 0:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"{n_ooc} EWMA signal(s) detected",
                "detail": f"EWMA exceeded time-varying limits at indices: {ewma_result.out_of_control_indices[:10]}",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "info",
                "title": "No shifts detected",
                "detail": f"EWMA with lambda={lambda_param}, L={L} shows stable process.",
            }
        )

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n": ewma_result.n,
                "target": round(ewma_result.target, 4),
                "sigma": round(ewma_result.sigma, 4),
                "lambda": lambda_param,
                "L": L,
                "UCL_steady": round(ewma_result.ucl_steady, 4),
                "LCL_steady": round(ewma_result.lcl_steady, 4),
                "n_OOC": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "EWMA CONTROL CHART",
                [
                    ("Variable", [(col_name, f"n = {ewma_result.n}")]),
                    (
                        "Parameters",
                        [
                            ("Target", f"{ewma_result.target:.4f}"),
                            ("Sigma", f"{ewma_result.sigma:.4f}"),
                            ("Lambda", str(lambda_param)),
                            ("L (sigma multiplier)", str(L)),
                        ],
                    ),
                    (
                        "Steady-State Limits",
                        [
                            ("UCL", f"{ewma_result.ucl_steady:.4f}"),
                            ("LCL", f"{ewma_result.lcl_steady:.4f}"),
                        ],
                    ),
                    (
                        "Results",
                        [
                            ("OOC Points", str(n_ooc)),
                            ("Status", status),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": f"EWMA analysis: {status}",
                "body": (
                    f"EWMA chart for {col_name} with lambda = {lambda_param} and L = {L}. "
                    f"EWMA gives more weight to recent observations (lambda controls memory). "
                    f"{'Process is stable.' if n_ooc == 0 else f'{n_ooc} signal(s) detected, indicating a process shift.'}"
                ),
                "next_steps": (
                    "Process stable. EWMA is good for detecting small to moderate shifts."
                    if n_ooc == 0
                    else "Investigate the timing of the shift. Consider reducing lambda for more sensitivity."
                ),
                "chart_guidance": "EWMA smooths data with exponential weighting. Control limits widen over time to steady state.",
            },
            "assumptions": {
                "independence": {
                    "pass": True,
                    "detail": "EWMA assumes independent observations.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"EWMA for {col_name}: lambda = {lambda_param}, {n_ooc} OOC points, {status.lower()}."
            ),
        }
    )


# =============================================================================
# Process Capability (Sixpack)
# =============================================================================


def forge_capability(df, config):
    """Process capability analysis with sixpack visualization."""
    from forgespc.capability import calculate_capability
    from forgeviz.charts.capability import capability_histogram, capability_sixpack

    data, col_name = _col(df, config, "column")
    lsl = config.get("lsl")
    usl = config.get("usl")
    target = config.get("target")
    sg_size = config.get("subgroup_size", 1)

    if lsl is None or usl is None:
        raise ValueError("Both LSL and USL are required for capability analysis")

    lsl = float(lsl)
    usl = float(usl)
    if target is not None:
        target = float(target)

    cap = calculate_capability(data.tolist(), usl=usl, lsl=lsl, target=target, subgroup_size=int(sg_size))

    # Generate sixpack charts
    sixpack_specs = capability_sixpack(
        data.tolist(),
        usl=usl,
        lsl=lsl,
        target=target,
        cp=cap.cp,
        cpk=cap.cpk,
        pp=cap.pp,
        ppk=cap.ppk,
    )
    plots = [_to_chart(s) for s in sixpack_specs]

    # Also add standalone histogram
    hist_spec = capability_histogram(
        data.tolist(),
        usl=usl,
        lsl=lsl,
        target=target,
        cp=cap.cp,
        cpk=cap.cpk,
    )
    plots.append(_to_chart(hist_spec))

    cpk_interp = _cpk_interpretation(cap.cpk)

    # Bayesian shadow for capability
    bayes_cap = None
    try:
        from forgespc.bayesian import bayesian_capability

        bc = bayesian_capability(data.tolist(), usl=usl, lsl=lsl, target=target)
        bayes_cap = {
            "cpk_mean": round(float(bc.cpk_mean), 4),
            "cpk_ci": [round(float(bc.cpk_ci[0]), 4), round(float(bc.cpk_ci[1]), 4)],
            "prob_capable": round(float(bc.prob_capable), 4),
            "interpretation": (
                f"Bayesian Cpk: mean = {bc.cpk_mean:.3f}, "
                f"95% CrI [{bc.cpk_ci[0]:.3f}, {bc.cpk_ci[1]:.3f}]. "
                f"P(Cpk >= 1.33) = {bc.prob_capable:.1%}."
            ),
        }
    except Exception:
        pass

    diagnostics = [
        {
            "level": "info" if cap.cpk >= 1.33 else "warning" if cap.cpk >= 1.0 else "error",
            "title": cpk_interp,
            "detail": (
                f"Cp = {cap.cp:.3f}, Cpk = {cap.cpk:.3f}. "
                f"Pp = {cap.pp:.3f}, Ppk = {cap.ppk:.3f}. "
                f"Sigma level: {cap.sigma_level:.2f}. "
                f"DPMO: {cap.dpmo:.0f}. Yield: {cap.yield_percent:.2f}%."
            ),
        }
    ]

    if cap.cp > 1.33 and cap.cpk < 1.0:
        diagnostics.append(
            {
                "level": "warning",
                "title": "Process is capable but not centered",
                "detail": (
                    f"Cp ({cap.cp:.3f}) indicates adequate spread, but "
                    f"Cpk ({cap.cpk:.3f}) shows the process mean is off-center. "
                    f"Centering the process would improve capability."
                ),
            }
        )

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n": cap.n_samples,
                "mean": round(cap.mean, 4),
                "sigma_within": round(cap.sigma_within, 4),
                "sigma_overall": round(cap.sigma_overall, 4),
                "Cp": round(cap.cp, 4),
                "Cpk": round(cap.cpk, 4),
                "Cpu": round(cap.cpu, 4),
                "Cpl": round(cap.cpl, 4),
                "Pp": round(cap.pp, 4),
                "Ppk": round(cap.ppk, 4),
                "Ppu": round(cap.ppu, 4),
                "Ppl": round(cap.ppl, 4),
                "sigma_level": round(cap.sigma_level, 2),
                "DPMO": round(cap.dpmo, 0),
                "yield_percent": round(cap.yield_percent, 2),
                "USL": usl,
                "LSL": lsl,
                "target": target,
            },
            "summary": _rich_summary(
                "PROCESS CAPABILITY ANALYSIS",
                [
                    ("Variable", [(col_name, f"n = {cap.n_samples}")]),
                    (
                        "Specifications",
                        [
                            ("USL", f"{usl:.4f}"),
                            ("Target", f"{target:.4f}" if target else "Midpoint"),
                            ("LSL", f"{lsl:.4f}"),
                        ],
                    ),
                    (
                        "Process Statistics",
                        [
                            ("Mean", f"{cap.mean:.4f}"),
                            ("Sigma (within)", f"{cap.sigma_within:.4f}"),
                            ("Sigma (overall)", f"{cap.sigma_overall:.4f}"),
                        ],
                    ),
                    (
                        "Capability Indices (Within)",
                        [
                            ("Cp", f"{cap.cp:.4f}"),
                            ("Cpk", f"{cap.cpk:.4f}"),
                            ("Cpu", f"{cap.cpu:.4f}"),
                            ("Cpl", f"{cap.cpl:.4f}"),
                        ],
                    ),
                    (
                        "Performance Indices (Overall)",
                        [
                            ("Pp", f"{cap.pp:.4f}"),
                            ("Ppk", f"{cap.ppk:.4f}"),
                        ],
                    ),
                    (
                        "Six Sigma Metrics",
                        [
                            ("Sigma Level", f"{cap.sigma_level:.2f}"),
                            ("DPMO", f"{cap.dpmo:.0f}"),
                            ("Yield", f"{cap.yield_percent:.2f}%"),
                        ],
                    ),
                    (
                        "Assessment",
                        [
                            ("Verdict", cpk_interp),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": cpk_interp,
                "body": (
                    f"Process capability analysis for {col_name} against spec limits "
                    f"[{lsl:.4f}, {usl:.4f}]. {cap.interpretation} "
                    f"The process operates at a {cap.sigma_level:.1f} sigma level "
                    f"with {cap.dpmo:.0f} DPMO and {cap.yield_percent:.2f}% yield."
                ),
                "next_steps": (
                    "Process is capable. Monitor with control charts to maintain performance."
                    if cap.cpk >= 1.33
                    else "Process needs improvement. "
                    + ("Center the process (mean is off-target)." if cap.cp > cap.cpk * 1.3 else "")
                    + (" Reduce variation." if cap.cp < 1.33 else "")
                ),
                "chart_guidance": (
                    "Sixpack layout: I chart, MR chart, last 25 observations, "
                    "capability histogram, normal probability plot, and capability summary."
                ),
            },
            "assumptions": {
                "normality": {
                    "pass": True,
                    "detail": "Capability indices assume normally distributed data. Check the normal probability plot.",
                },
                "stability": {
                    "pass": True,
                    "detail": "Capability analysis assumes a stable process. Verify with control charts first.",
                },
            },
            "diagnostics": diagnostics,
            "bayesian_shadow": bayes_cap,
            "guide_observation": (
                f"Capability for {col_name}: Cpk = {cap.cpk:.3f}, "
                f"sigma level = {cap.sigma_level:.1f}, "
                f"yield = {cap.yield_percent:.2f}%. {cpk_interp}."
            ),
        }
    )


# =============================================================================
# Gage R&R (Crossed)
# =============================================================================


def forge_gage_rr(df, config):
    """Gage R&R crossed study via ANOVA method."""
    from forgespc.gage import gage_rr_crossed
    from forgeviz.charts.gage import (
        gage_rr_by_operator,
        gage_rr_by_part,
        gage_rr_components,
    )

    meas_col = config.get("measurement")
    part_col = config.get("part")
    op_col = config.get("operator")

    if not all([meas_col, part_col, op_col]):
        raise ValueError("Config must include 'measurement', 'part', and 'operator' column names")

    # Validate columns exist
    for c in [meas_col, part_col, op_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in dataset")

    # Clean: drop rows where measurement is not numeric
    clean_df = df[[part_col, op_col, meas_col]].copy()
    clean_df[meas_col] = pd.to_numeric(clean_df[meas_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[meas_col])

    if len(clean_df) < 4:
        raise ValueError("Need at least 4 valid measurements for Gage R&R")

    parts = clean_df[part_col].tolist()
    operators = clean_df[op_col].tolist()
    measurements = clean_df[meas_col].tolist()

    tolerance = config.get("tolerance")
    if tolerance is not None:
        tolerance = float(tolerance)

    result = gage_rr_crossed(
        parts=parts,
        operators=operators,
        measurements=measurements,
        tolerance=tolerance,
    )

    # Build charts
    plots = []

    # 1. Components chart
    comp_spec = gage_rr_components(result.pct_contribution)
    plots.append(_to_chart(comp_spec))

    # 2. By Part chart
    part_measurements = {}
    for p, m in zip(parts, measurements):
        part_measurements.setdefault(str(p), []).append(m)
    unique_parts = sorted(part_measurements.keys())
    by_part_spec = gage_rr_by_part(unique_parts, part_measurements)
    plots.append(_to_chart(by_part_spec))

    # 3. By Operator chart
    op_measurements = {}
    for o, m in zip(operators, measurements):
        op_measurements.setdefault(str(o), []).append(m)
    unique_ops = sorted(op_measurements.keys())
    by_op_spec = gage_rr_by_operator(unique_ops, op_measurements)
    plots.append(_to_chart(by_op_spec))

    grr_pct = result.grr_percent
    grr_interp = _grr_assessment(grr_pct)

    diagnostics = [
        {
            "level": "info" if grr_pct < 10 else "warning" if grr_pct < 30 else "error",
            "title": grr_interp,
            "detail": (
                f"Repeatability: {result.pct_study_var.get('repeatability', 0):.1f}% of study variation. "
                f"Reproducibility: {result.pct_study_var.get('reproducibility', 0):.1f}% of study variation. "
                f"NDC (number of distinct categories): {result.ndc}."
            ),
        }
    ]

    if result.ndc < 5:
        diagnostics.append(
            {
                "level": "warning",
                "title": f"Low NDC ({result.ndc})",
                "detail": "AIAG recommends NDC >= 5. The measurement system cannot adequately distinguish between parts.",
            }
        )

    if result.interaction_significant:
        diagnostics.append(
            {
                "level": "info",
                "title": "Operator x Part interaction is significant",
                "detail": "Some operators measure certain parts differently. Consider operator training.",
            }
        )

    # ANOVA table for statistics
    anova_summary = {}
    for row in result.anova_table:
        src = row.get("source", "")
        anova_summary[f"anova_{src}_F"] = row.get("f")
        anova_summary[f"anova_{src}_p"] = row.get("p")

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_parts": result.n_parts,
                "n_operators": result.n_operators,
                "n_replicates": result.n_replicates,
                "n_total": result.n_total,
                "grr_percent": round(grr_pct, 2),
                "ndc": result.ndc,
                "assessment": result.assessment,
                "var_repeatability": round(result.var_repeatability, 6),
                "var_reproducibility": round(result.var_reproducibility, 6),
                "var_part": round(result.var_part, 6),
                "var_total": round(result.var_total, 6),
                "pct_contribution": result.pct_contribution,
                "pct_study_var": result.pct_study_var,
                "interaction_significant": result.interaction_significant,
                "interaction_pooled": result.interaction_pooled,
                **anova_summary,
            },
            "summary": _rich_summary(
                "GAGE R&R STUDY (CROSSED - ANOVA)",
                [
                    (
                        "Study Design",
                        [
                            ("Parts", str(result.n_parts)),
                            ("Operators", str(result.n_operators)),
                            ("Replicates", str(result.n_replicates)),
                            ("Total Measurements", str(result.n_total)),
                        ],
                    ),
                    (
                        "Variance Components (%Contribution)",
                        [
                            ("Gage R&R", f"{result.pct_contribution.get('gage_rr', 0):.2f}%"),
                            ("  Repeatability", f"{result.pct_contribution.get('repeatability', 0):.2f}%"),
                            ("  Reproducibility", f"{result.pct_contribution.get('reproducibility', 0):.2f}%"),
                            ("Part-to-Part", f"{result.pct_contribution.get('part_to_part', 0):.2f}%"),
                        ],
                    ),
                    (
                        "%Study Var",
                        [
                            ("Gage R&R", f"{result.pct_study_var.get('gage_rr', 0):.2f}%"),
                            ("  Repeatability", f"{result.pct_study_var.get('repeatability', 0):.2f}%"),
                            ("  Reproducibility", f"{result.pct_study_var.get('reproducibility', 0):.2f}%"),
                            ("Part-to-Part", f"{result.pct_study_var.get('part_to_part', 0):.2f}%"),
                        ],
                    ),
                    (
                        "Key Metrics",
                        [
                            ("%GRR (Study Var)", f"{grr_pct:.2f}%"),
                            ("NDC", str(result.ndc)),
                            ("Assessment", result.assessment),
                        ],
                    ),
                    (
                        "Interaction",
                        [
                            ("Operator x Part", "Significant" if result.interaction_significant else "Not significant"),
                            ("Pooled", "Yes" if result.interaction_pooled else "No"),
                        ],
                    ),
                ],
            ),
            "narrative": {
                "verdict": grr_interp,
                "body": (
                    f"Gage R&R study with {result.n_parts} parts, {result.n_operators} operators, "
                    f"{result.n_replicates} replicates ({result.n_total} total measurements). "
                    f"%GRR = {grr_pct:.1f}% of study variation. "
                    f"NDC = {result.ndc} distinct categories. "
                    f"Assessment: {result.assessment}."
                ),
                "next_steps": (
                    "Measurement system is acceptable. Proceed with process studies."
                    if grr_pct < 10
                    else "Consider measurement system improvement before drawing process conclusions. "
                    + (
                        "Focus on repeatability (equipment precision)."
                        if result.var_repeatability > result.var_reproducibility
                        else "Focus on reproducibility (operator training)."
                    )
                    if grr_pct < 30
                    else "Measurement system is unacceptable. "
                    + (
                        "Equipment needs repair or replacement (repeatability dominant)."
                        if result.var_repeatability > result.var_reproducibility
                        else "Operator training required (reproducibility dominant)."
                    )
                ),
                "chart_guidance": (
                    "Components chart shows variance breakdown. By-Part shows part-to-part variation. "
                    "By-Operator shows reproducibility. Look for patterns in operator differences."
                ),
            },
            "assumptions": {
                "crossed_design": {
                    "pass": True,
                    "detail": "All operators measured all parts. For destructive testing, use nested Gage R&R.",
                },
                "repeatability": {
                    "pass": True,
                    "detail": "Multiple measurements per operator-part combination enable repeatability estimation.",
                },
            },
            "diagnostics": diagnostics,
            "guide_observation": (
                f"Gage R&R: %GRR = {grr_pct:.1f}%, NDC = {result.ndc}, assessment = {result.assessment}."
            ),
        }
    )


# =============================================================================
# Handler Registry
# =============================================================================


# =============================================================================
# np Chart
# =============================================================================


def forge_np_chart(df, config):
    """np chart for defective counts (constant sample size)."""
    from forgespc.charts import np_chart
    from forgeviz.charts.control import from_spc_result

    defectives, def_col = _int_col(df, config, "column")
    sample_size = int(config.get("sample_size", config.get("n", max(defectives) + 1 if len(defectives) > 0 else 50)))

    result = np_chart(defectives, sample_size=sample_size)

    spec = from_spc_result(result, title=f"np Chart: {def_col}")
    plots = [_to_chart(spec)]

    status = _control_status(result)
    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    np_bar = float(result.limits.cl)

    return _jsonify(
        {
            "plots": plots,
            "statistics": {
                "n_samples": len(defectives),
                "sample_size": sample_size,
                "np_bar": round(np_bar, 4),
                "UCL": round(float(result.limits.ucl), 4),
                "LCL": round(float(result.limits.lcl), 4),
                "OOC_count": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "NP CHART (DEFECTIVE COUNT)",
                [
                    (
                        "Control Limits",
                        [
                            ("np-bar", f"{np_bar:.4f}"),
                            ("UCL", f"{float(result.limits.ucl):.4f}"),
                            ("LCL", f"{float(result.limits.lcl):.4f}"),
                            ("Sample size", str(sample_size)),
                        ],
                    ),
                    ("Status", [("Process", status), ("OOC Points", str(n_ooc))]),
                ],
            ),
            "narrative": {
                "verdict": f"np chart: process is {status}",
                "body": f"np chart monitors defective counts with constant sample size n={sample_size}. np-bar = {np_bar:.4f}.",
                "next_steps": "Investigate OOC points for special causes." if n_ooc > 0 else "Process is stable.",
                "chart_guidance": "Each point is the count of defectives in a sample of constant size.",
            },
            "assumptions": {"binomial": {"pass": True, "detail": "Constant sample size, binomial model."}},
            "diagnostics": _nelson_diagnostics(result),
            "guide_observation": f"np chart: np-bar={np_bar:.4f}, {n_ooc} OOC, {status}.",
        }
    )


# =============================================================================
# Conformal / Entropy SPC
# =============================================================================


def forge_conformal_control(df, config):
    """Conformal prediction control chart via forgespc."""
    from forgespc.conformal import conformal_control

    data, col_name = _col(df, config, "column", "var1")
    alpha = float(config.get("alpha", 0.05))
    cal_frac = float(config.get("calibration_fraction", 0.5))

    result = conformal_control(data.tolist(), alpha=alpha, calibration_fraction=cal_frac)

    n_ooc = len(result.ooc_indices) if result.ooc_indices else 0
    status = "IN CONTROL" if result.in_control else "OUT OF CONTROL"

    return _jsonify(
        {
            "plots": [],
            "statistics": {
                "n": len(data),
                "n_calibration": result.n_calibration,
                "n_monitoring": result.n_monitoring,
                "threshold": round(result.threshold, 4),
                "n_ooc": n_ooc,
                "alpha": alpha,
                "status": status,
            },
            "summary": _rich_summary(
                "CONFORMAL CONTROL CHART",
                [
                    (
                        "Setup",
                        [
                            ("Calibration samples", str(result.n_calibration)),
                            ("Monitoring samples", str(result.n_monitoring)),
                            ("Threshold", f"{result.threshold:.4f}"),
                        ],
                    ),
                    ("Status", [("Process", status), ("OOC Points", str(n_ooc))]),
                ],
            ),
            "narrative": {
                "verdict": f"Conformal control: {status}",
                "body": f"Distribution-free control chart using conformal prediction. {n_ooc} nonconforming points detected.",
                "next_steps": "Conformal charts require no distributional assumptions — robust for non-normal data.",
                "chart_guidance": "Points above threshold are nonconforming — possible process shift.",
            },
            "assumptions": {},
            "diagnostics": [],
            "guide_observation": f"Conformal: {n_ooc} OOC of {result.n_monitoring}, {status}.",
        }
    )


def forge_entropy_spc(df, config):
    """Entropy-based SPC chart via forgespc."""
    from forgespc.conformal import entropy_spc

    data, col_name = _col(df, config, "column", "var1")
    window = int(config.get("window_size", 20))
    n_bins = int(config.get("n_bins", 10))
    alpha = float(config.get("alpha", 0.05))

    result = entropy_spc(data.tolist(), window_size=window, n_bins=n_bins, alpha=alpha)

    n_ooc = len(result.ooc_indices) if result.ooc_indices else 0
    status = "IN CONTROL" if result.in_control else "OUT OF CONTROL"

    return _jsonify(
        {
            "plots": [],
            "statistics": {
                "n": result.n,
                "window_size": window,
                "baseline_entropy": round(result.baseline_entropy, 4),
                "ucl": round(result.ucl, 4),
                "lcl": round(result.lcl, 4),
                "n_ooc": n_ooc,
                "status": status,
            },
            "summary": _rich_summary(
                "ENTROPY SPC CHART",
                [
                    (
                        "Parameters",
                        [
                            ("Window size", str(window)),
                            ("Baseline entropy", f"{result.baseline_entropy:.4f}"),
                            ("UCL", f"{result.ucl:.4f}"),
                            ("LCL", f"{result.lcl:.4f}"),
                        ],
                    ),
                    ("Status", [("Process", status), ("OOC Points", str(n_ooc))]),
                ],
            ),
            "narrative": {
                "verdict": f"Entropy SPC: {status}",
                "body": f"Information-theoretic control chart using Shannon entropy over sliding windows. Baseline entropy = {result.baseline_entropy:.4f}.",
                "next_steps": "Entropy changes indicate distributional shifts — not just mean/variance.",
                "chart_guidance": "Entropy values outside limits suggest the data distribution has changed.",
            },
            "assumptions": {},
            "diagnostics": [],
            "guide_observation": f"Entropy SPC: baseline={result.baseline_entropy:.4f}, {n_ooc} OOC, {status}.",
        }
    )


# =============================================================================
# Handler Registry
# =============================================================================


FORGE_SPC_HANDLERS = {
    "imr": forge_imr,
    "xbar_r": forge_xbar_r,
    "xbar_s": forge_xbar_s,
    "p_chart": forge_p_chart,
    "c_chart": forge_c_chart,
    "u_chart": forge_u_chart,
    "cusum": forge_cusum,
    "ewma": forge_ewma,
    "capability": forge_capability,
    "gage_rr": forge_gage_rr,
    "np_chart": forge_np_chart,
    "conformal_control": forge_conformal_control,
    "entropy_spc": forge_entropy_spc,
}


# =============================================================================
# Entry Point
# =============================================================================


def run_forge_spc(analysis_id, df, config):
    """Run a forge-backed SPC analysis.

    Returns the result dict, or None if analysis_id is not yet ported to forge.
    Automatically enriches every result with education content.
    """
    handler = FORGE_SPC_HANDLERS.get(analysis_id)
    if handler is None:
        return None
    try:
        result = handler(df, config)
    except Exception:
        logger.exception(f"Forge SPC handler failed for {analysis_id}, falling back to legacy")
        return None

    # Enrich: education (applies to ALL handlers)
    if "education" not in result or result["education"] is None:
        result["education"] = _education("spc", analysis_id)

    # Wrap plain summary with COLOR markup if not already present
    summary = result.get("summary", "")
    if summary and "<<COLOR:" not in summary:
        title = analysis_id.replace("_", " ").upper()
        result["summary"] = _rich_summary(title, [("Result", [("Output", summary)])])

    result.setdefault("bayesian_shadow", None)
    result.setdefault("diagnostics", [])

    return result
