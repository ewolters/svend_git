"""SPC Shewhart control charts — IMR, Xbar-R, Xbar-S, P, NP, C, U."""

import numpy as np

from ..common import _narrative
from .helpers import _spc_add_ooc_markers, _spc_build_point_rules, _spc_nelson_rules


def run_imr(df, config):
    """Individual-Moving Range chart."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    n = len(data)
    mr = np.abs(np.diff(data))
    mr_bar = np.mean(mr)

    x_bar = np.mean(data)
    ucl = x_bar + 2.66 * mr_bar
    lcl = x_bar - 2.66 * mr_bar

    mr_ucl = 3.267 * mr_bar

    # Nelson rules check
    ooc_indices, rule_violations = _spc_nelson_rules(data, x_bar, ucl, lcl)
    point_rules = _spc_build_point_rules(data, x_bar, ucl, lcl, ooc_indices)

    # I Chart with OOC markers
    i_chart_data = [
        {
            "type": "scatter",
            "y": data.tolist(),
            "mode": "lines+markers",
            "name": "Value",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [x_bar] * n,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [ucl] * n,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [lcl] * n,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(i_chart_data, data, ooc_indices, point_rules=point_rules)
    result["plots"].append(
        {
            "title": "I Chart (Individuals)",
            "data": i_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # MR Chart with OOC markers
    mr_ooc = [i for i in range(len(mr)) if mr[i] > mr_ucl]
    mr_chart_data = [
        {
            "type": "scatter",
            "y": mr.tolist(),
            "mode": "lines+markers",
            "name": "MR",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [mr_bar] * (n - 1),
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [mr_ucl] * (n - 1),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(mr_chart_data, mr, mr_ooc)
    result["plots"].append(
        {
            "title": "MR Chart (Moving Range)",
            "data": mr_chart_data,
            "layout": {
                "height": 290,
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    ooc = len(ooc_indices)
    violations_text = ""
    if rule_violations:
        violations_text = "\n\nNelson Rule Violations:\n" + "\n".join(f"  {v}" for v in rule_violations)
    result["summary"] = (
        f"I-MR Chart Analysis\n\nMean: {x_bar:.4f}\nUCL: {ucl:.4f}\nLCL: {lcl:.4f}\nMR-bar: {mr_bar:.4f}\n\nOut-of-control points: {ooc}{violations_text}"
    )

    result["guide_observation"] = f"Control chart shows {ooc} out-of-control points." + (
        " Process appears stable." if ooc == 0 else " Investigation recommended."
    )
    result["statistics"] = {
        "grand_mean": float(x_bar),
        "ucl": float(ucl),
        "lcl": float(lcl),
        "mr_bar": float(mr_bar),
        "sigma": float(mr_bar / 1.128),
        "n": n,
        "n_ooc": ooc,
    }

    # Narrative
    if ooc == 0:
        _cc_verdict = "Process is in statistical control"
        _cc_body = "No special cause variation detected. Process is stable and predictable."
        _cc_next = "Process is stable &mdash; capability analysis is valid."
    else:
        _rule_summary = "; ".join(rule_violations[:3]) if rule_violations else f"{ooc} points outside control limits"
        _cc_verdict = f"Process is out of control &mdash; {ooc} signal{'s' if ooc > 1 else ''} detected"
        _cc_body = f"Found: {_rule_summary}."
        _cc_next = "Investigate special causes at the flagged points. Check timestamps against process logs for assignable causes."
    result["narrative"] = _narrative(
        _cc_verdict,
        _cc_body,
        next_steps=_cc_next,
        chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift. Two of three points beyond 2\u03c3 suggest a trend.",
    )

    if ooc_indices:
        result["what_if_data"] = {
            "type": "spc_intervention",
            "values": data.tolist(),
            "center": float(x_bar),
            "ucl": float(ucl),
            "lcl": float(lcl),
            "sigma": float(mr_bar / 1.128),
            "ooc_indices": [int(i) for i in ooc_indices],
            "first_ooc": int(min(ooc_indices)),
        }

    return result


def run_xbar_r(df, config):
    """Xbar-R Chart for subgrouped data."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    subgroup_col = config.get("subgroup")
    subgroup_size = int(config.get("subgroup_size", 5))

    if subgroup_col:
        # Group by subgroup column
        groups = df.groupby(subgroup_col)[measurement].apply(list).values
    else:
        # Create subgroups from sequential data
        groups = [data[i : i + subgroup_size] for i in range(0, len(data), subgroup_size)]
        groups = [g for g in groups if len(g) == subgroup_size]

    groups = np.array([g for g in groups if len(g) >= 2])
    n_subgroups = len(groups)

    x_bars = np.array([np.mean(g) for g in groups])
    ranges = np.array([np.max(g) - np.min(g) for g in groups])

    x_double_bar = np.mean(x_bars)
    r_bar = np.mean(ranges)

    # Control chart constants (for subgroup size 2-10)
    _d2_table = {
        2: 1.128,
        3: 1.693,
        4: 2.059,
        5: 2.326,
        6: 2.534,
        7: 2.704,
        8: 2.847,
        9: 2.970,
        10: 3.078,
    }
    _d3_table = {
        2: 0.853,
        3: 0.888,
        4: 0.880,
        5: 0.864,
        6: 0.848,
        7: 0.833,
        8: 0.820,
        9: 0.808,
        10: 0.797,
    }
    A2_table = {
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
    D3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
    D4_table = {
        2: 3.267,
        3: 2.574,
        4: 2.282,
        5: 2.114,
        6: 2.004,
        7: 1.924,
        8: 1.864,
        9: 1.816,
        10: 1.777,
    }

    n = min(subgroup_size, 10)
    A2 = A2_table.get(n, 0.577)
    D3 = D3_table.get(n, 0)
    D4 = D4_table.get(n, 2.114)

    # Xbar limits
    xbar_ucl = x_double_bar + A2 * r_bar
    xbar_lcl = x_double_bar - A2 * r_bar

    # R limits
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar

    # Nelson rules for X-bar
    xbar_ooc, xbar_violations = _spc_nelson_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl)
    xbar_point_rules = _spc_build_point_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl, xbar_ooc)
    # Nelson rules for R
    r_ooc, r_violations = _spc_nelson_rules(ranges, r_bar, r_ucl, r_lcl)
    r_point_rules = _spc_build_point_rules(ranges, r_bar, r_ucl, r_lcl, r_ooc)

    # Xbar Chart with OOC markers
    xbar_chart_data = [
        {
            "type": "scatter",
            "y": x_bars.tolist(),
            "mode": "lines+markers",
            "name": "X\u0304",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [x_double_bar] * n_subgroups,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [xbar_ucl] * n_subgroups,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [xbar_lcl] * n_subgroups,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc, point_rules=xbar_point_rules)
    result["plots"].append(
        {
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # R Chart with OOC markers
    r_chart_data = [
        {
            "type": "scatter",
            "y": ranges.tolist(),
            "mode": "lines+markers",
            "name": "R",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [r_bar] * n_subgroups,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [r_ucl] * n_subgroups,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [r_lcl] * n_subgroups,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(r_chart_data, ranges, r_ooc, point_rules=r_point_rules)
    result["plots"].append(
        {
            "title": "R Chart",
            "data": r_chart_data,
            "layout": {
                "height": 290,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    violations_text = ""
    if xbar_violations or r_violations:
        violations_text = "\n\nNelson Rule Violations:"
        for v in xbar_violations:
            violations_text += f"\n  X\u0304: {v}"
        for v in r_violations:
            violations_text += f"\n  R: {v}"

    result["summary"] = (
        f"Xbar-R Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX\u0304 Chart:\n  X\u033f: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nR Chart:\n  R\u0304: {r_bar:.4f}\n  UCL: {r_ucl:.4f}\n  LCL: {r_lcl:.4f}\n  OOC points: {len(r_ooc)}{violations_text}"
    )
    result["statistics"] = {
        "grand_mean": float(x_double_bar),
        "xbar_ucl": float(xbar_ucl),
        "xbar_lcl": float(xbar_lcl),
        "r_bar": float(r_bar),
        "r_ucl": float(r_ucl),
        "r_lcl": float(r_lcl),
        "n_subgroups": n_subgroups,
        "subgroup_size": n,
        "n_ooc_xbar": len(xbar_ooc),
        "n_ooc_r": len(r_ooc),
    }

    _xr_ooc = len(xbar_ooc) + len(r_ooc)
    if _xr_ooc == 0:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No out-of-control points in either the X\u0304 or R chart across {n_subgroups} subgroups. Process is stable and predictable.",
            next_steps="Process is stable \u2014 capability analysis is valid.",
            chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift.",
        )
    else:
        _xr_rules = (xbar_violations + r_violations)[:3]
        result["narrative"] = _narrative(
            f"Process is out of control \u2014 {_xr_ooc} signal{'s' if _xr_ooc > 1 else ''} detected",
            f"X\u0304 chart: {len(xbar_ooc)} OOC points. R chart: {len(r_ooc)} OOC points."
            + (f" Violations: {'; '.join(_xr_rules)}." if _xr_rules else ""),
            next_steps="Investigate special causes at flagged points. Check timestamps against process logs.",
            chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift.",
        )

    if xbar_ooc or r_ooc:
        _sigma_est = r_bar / (2.326 if n == 5 else 1.128 if n == 2 else 2.0)
        result["what_if_data"] = {
            "type": "spc_intervention",
            "values": x_bars.tolist(),
            "center": float(x_double_bar),
            "ucl": float(xbar_ucl),
            "lcl": float(xbar_lcl),
            "sigma": float(_sigma_est),
            "ooc_indices": sorted(set(list(xbar_ooc) + list(r_ooc))),
            "first_ooc": int(min(list(xbar_ooc) + list(r_ooc))),
        }

    return result


def run_xbar_s(df, config):
    """Xbar-S Chart (using standard deviation instead of range)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    subgroup_col = config.get("subgroup")
    subgroup_size = int(config.get("subgroup_size", 5))

    if subgroup_col:
        groups = df.groupby(subgroup_col)[measurement].apply(list).values
    else:
        groups = [data[i : i + subgroup_size] for i in range(0, len(data), subgroup_size)]
        groups = [g for g in groups if len(g) == subgroup_size]

    groups = np.array([g for g in groups if len(g) >= 2])
    n_subgroups = len(groups)

    x_bars = np.array([np.mean(g) for g in groups])
    stds = np.array([np.std(g, ddof=1) for g in groups])

    x_double_bar = np.mean(x_bars)
    s_bar = np.mean(stds)

    # Control chart constants for S chart
    B3_table = {
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0.030,
        7: 0.118,
        8: 0.185,
        9: 0.239,
        10: 0.284,
    }
    B4_table = {
        2: 3.267,
        3: 2.568,
        4: 2.266,
        5: 2.089,
        6: 1.970,
        7: 1.882,
        8: 1.815,
        9: 1.761,
        10: 1.716,
    }
    A3_table = {
        2: 2.659,
        3: 1.954,
        4: 1.628,
        5: 1.427,
        6: 1.287,
        7: 1.182,
        8: 1.099,
        9: 1.032,
        10: 0.975,
    }

    n = min(subgroup_size, 10)
    A3 = A3_table.get(n, 1.427)
    B3 = B3_table.get(n, 0)
    B4 = B4_table.get(n, 2.089)

    xbar_ucl = x_double_bar + A3 * s_bar
    xbar_lcl = x_double_bar - A3 * s_bar
    s_ucl = B4 * s_bar
    s_lcl = B3 * s_bar

    # Nelson rules for X-bar and S
    xbar_ooc, xbar_violations = _spc_nelson_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl)
    xbar_point_rules = _spc_build_point_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl, xbar_ooc)
    s_ooc, s_violations = _spc_nelson_rules(stds, s_bar, s_ucl, s_lcl)
    s_point_rules = _spc_build_point_rules(stds, s_bar, s_ucl, s_lcl, s_ooc)

    # Xbar Chart with OOC markers
    xbar_chart_data = [
        {
            "type": "scatter",
            "y": x_bars.tolist(),
            "mode": "lines+markers",
            "name": "X\u0304",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [x_double_bar] * n_subgroups,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [xbar_ucl] * n_subgroups,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [xbar_lcl] * n_subgroups,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc, point_rules=xbar_point_rules)
    result["plots"].append(
        {
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # S Chart with OOC markers
    s_chart_data = [
        {
            "type": "scatter",
            "y": stds.tolist(),
            "mode": "lines+markers",
            "name": "S",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [s_bar] * n_subgroups,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [s_ucl] * n_subgroups,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [s_lcl] * n_subgroups,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(s_chart_data, stds, s_ooc, point_rules=s_point_rules)
    result["plots"].append(
        {
            "title": "S Chart",
            "data": s_chart_data,
            "layout": {
                "height": 290,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    violations_text = ""
    if xbar_violations or s_violations:
        violations_text = "\n\nNelson Rule Violations:"
        for v in xbar_violations:
            violations_text += f"\n  X\u0304: {v}"
        for v in s_violations:
            violations_text += f"\n  S: {v}"

    result["summary"] = (
        f"Xbar-S Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX\u0304 Chart:\n  X\u033f: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nS Chart:\n  S\u0304: {s_bar:.4f}\n  UCL: {s_ucl:.4f}\n  LCL: {s_lcl:.4f}\n  OOC points: {len(s_ooc)}{violations_text}"
    )
    result["statistics"] = {
        "grand_mean": float(x_double_bar),
        "xbar_ucl": float(xbar_ucl),
        "xbar_lcl": float(xbar_lcl),
        "s_bar": float(s_bar),
        "s_ucl": float(s_ucl),
        "s_lcl": float(s_lcl),
        "n_subgroups": n_subgroups,
        "subgroup_size": n,
        "n_ooc_xbar": len(xbar_ooc),
        "n_ooc_s": len(s_ooc),
    }

    _xs_ooc = len(xbar_ooc) + len(s_ooc)
    if _xs_ooc == 0:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No out-of-control points across {n_subgroups} subgroups.",
            next_steps="Process is stable \u2014 capability analysis is valid.",
            chart_guidance="Points above UCL or below LCL are out-of-control signals.",
        )
    else:
        _xs_rules = (xbar_violations + s_violations)[:3]
        result["narrative"] = _narrative(
            f"Process is out of control \u2014 {_xs_ooc} signal{'s' if _xs_ooc > 1 else ''} detected",
            f"X\u0304 chart: {len(xbar_ooc)} OOC. S chart: {len(s_ooc)} OOC."
            + (f" {'; '.join(_xs_rules)}." if _xs_rules else ""),
            next_steps="Investigate special causes at flagged points.",
            chart_guidance="Points above UCL or below LCL are out-of-control signals.",
        )

    if xbar_ooc or s_ooc:
        result["what_if_data"] = {
            "type": "spc_intervention",
            "values": x_bars.tolist(),
            "center": float(x_double_bar),
            "ucl": float(xbar_ucl),
            "lcl": float(xbar_lcl),
            "sigma": float(s_bar),
            "ooc_indices": sorted(set(list(xbar_ooc) + list(s_ooc))),
            "first_ooc": int(min(list(xbar_ooc) + list(s_ooc))),
        }

    return result


def run_p_chart(df, config):
    """P Chart for proportion defective."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defectives = config.get("defectives")
    sample_size = config.get("sample_size")

    d = df[defectives].dropna().values
    n = df[sample_size].dropna().values
    p = d / n

    p_bar = np.sum(d) / np.sum(n)
    k = len(p)

    # Control limits (variable since sample size may vary)
    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
    lcl = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

    # OOC detection for variable-limit chart
    ooc_indices = [i for i in range(k) if p[i] > ucl[i] or p[i] < lcl[i]]

    p_chart_data = [
        {
            "type": "scatter",
            "y": p.tolist(),
            "mode": "lines+markers",
            "name": "p",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [p_bar] * k,
            "mode": "lines",
            "name": "p\u0304",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(p_chart_data, p, ooc_indices)
    result["plots"].append(
        {
            "title": "P Chart (Proportion Defective)",
            "data": p_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Proportion"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    result["summary"] = (
        f"P Chart Analysis\n\np\u0304: {p_bar:.4f} ({p_bar * 100:.2f}%)\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}"
    )
    result["guide_observation"] = (
        f"P chart: {len(ooc_indices)} out-of-control points. p\u0304 = {p_bar * 100:.2f}%."
        + (" Process is stable." if len(ooc_indices) == 0 else " Investigation recommended.")
    )
    result["statistics"] = {
        "p_bar": float(p_bar),
        "n_samples": k,
        "n_ooc": len(ooc_indices),
    }

    n_ooc = len(ooc_indices)
    if n_ooc == 0:
        verdict = f"P Chart \u2014 Process in control (p\u0304 = {p_bar * 100:.2f}%)"
        body = f"All {k} samples fall within control limits. The average defective rate is {p_bar * 100:.2f}%."
        nxt = "Monitor ongoing. If p\u0304 is too high, investigate systemic causes rather than individual points."
    else:
        verdict = f"P Chart \u2014 {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
        body = (
            f"{n_ooc} of {k} samples ({n_ooc / k * 100:.1f}%) exceed control limits. "
            f"Average defective rate p\u0304 = {p_bar * 100:.2f}%. Investigate these subgroups for assignable causes."
        )
        nxt = "Identify what changed during OOC subgroups (material, operator, machine). Address root causes before tightening limits."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="Points outside the dashed red limits are out of control. Variable limits reflect differing sample sizes.",
    )

    return result


def run_np_chart(df, config):
    """NP Chart - Number defective (constant sample size)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defectives = config.get("defectives")
    sample_size = int(config.get("sample_size", 50))

    d = df[defectives].dropna().values
    n = sample_size
    k = len(d)

    np_bar = np.mean(d)
    p_bar = np_bar / n

    # Control limits
    ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
    lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

    np_ooc, np_violations = _spc_nelson_rules(d, np_bar, ucl, lcl)
    np_point_rules = _spc_build_point_rules(d, np_bar, ucl, lcl, np_ooc)

    np_chart_data = [
        {
            "type": "scatter",
            "y": d.tolist(),
            "mode": "lines+markers",
            "name": "np",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [np_bar] * k,
            "mode": "lines",
            "name": "n\u0304p",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [ucl] * k,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [lcl] * k,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(np_chart_data, d, np_ooc, point_rules=np_point_rules)
    result["plots"].append(
        {
            "title": "NP Chart (Number Defective)",
            "data": np_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Number Defective"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    violations_text = ""
    if np_violations:
        violations_text = "\n\nNelson Rule Violations:"
        for v in np_violations:
            violations_text += f"\n  {v}"

    result["summary"] = (
        f"NP Chart Analysis\n\nn\u0304p: {np_bar:.2f}\nSample size: {n}\np\u0304: {p_bar:.4f}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(np_ooc)}{violations_text}"
    )

    n_ooc = len(np_ooc)
    viol_note = f" Nelson rule violations: {', '.join(np_violations[:3])}." if np_violations else ""
    if n_ooc == 0:
        verdict = f"NP Chart \u2014 Process in control (n\u0304p = {np_bar:.1f})"
        body = f"All {k} samples within limits. Average {np_bar:.1f} defectives per sample of {n}.{viol_note}"
        nxt = "Continue monitoring. To reduce defective count, investigate the process, not individual samples."
    else:
        verdict = f"NP Chart \u2014 {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
        body = (
            f"{n_ooc} of {k} samples exceed limits. Average defectives n\u0304p = {np_bar:.1f} "
            f"(p\u0304 = {p_bar * 100:.2f}%).{viol_note}"
        )
        nxt = "Investigate OOC subgroups for assignable causes. Check for material batches, shift changes, or equipment issues."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="Points outside limits signal unusual defective counts. Runs or trends may indicate gradual shifts.",
    )

    return result


def run_c_chart(df, config):
    """C Chart - Count of defects per unit (constant opportunity)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defects = config.get("defects")

    c = df[defects].dropna().values
    k = len(c)

    c_bar = np.mean(c)

    # Control limits (Poisson-based)
    ucl = c_bar + 3 * np.sqrt(c_bar)
    lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

    c_ooc, c_violations = _spc_nelson_rules(c, c_bar, ucl, lcl)
    c_point_rules = _spc_build_point_rules(c, c_bar, ucl, lcl, c_ooc)

    c_chart_data = [
        {
            "type": "scatter",
            "y": c.tolist(),
            "mode": "lines+markers",
            "name": "c",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [c_bar] * k,
            "mode": "lines",
            "name": "c\u0304",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": [ucl] * k,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [lcl] * k,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(c_chart_data, c, c_ooc, point_rules=c_point_rules)
    result["plots"].append(
        {
            "title": "C Chart (Defects per Unit)",
            "data": c_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Defects"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    violations_text = ""
    if c_violations:
        violations_text = "\n\nNelson Rule Violations:"
        for v in c_violations:
            violations_text += f"\n  {v}"

    result["summary"] = (
        f"C Chart Analysis\n\nc\u0304: {c_bar:.2f}\nSamples: {k}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(c_ooc)}{violations_text}"
    )

    n_ooc = len(c_ooc)
    viol_note = f" Nelson rule violations: {', '.join(c_violations[:3])}." if c_violations else ""
    if n_ooc == 0:
        verdict = f"C Chart \u2014 Process in control (c\u0304 = {c_bar:.1f})"
        body = f"All {k} samples within Poisson-based limits. Average defect count c\u0304 = {c_bar:.1f}.{viol_note}"
        nxt = "Stable process. To reduce defect count, apply Pareto analysis to identify top defect categories."
    else:
        verdict = f"C Chart \u2014 {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
        body = (
            f"{n_ooc} of {k} samples exceed limits. Average defects c\u0304 = {c_bar:.1f}.{viol_note} "
            f"The process is unstable \u2014 address special causes before process improvement."
        )
        nxt = "Investigate OOC points chronologically. Look for environmental, material, or procedural changes."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="Limits are Poisson-based (\u00b13\u221ac\u0304). Points outside = unusual defect counts for constant-opportunity inspection.",
    )

    return result


def run_u_chart(df, config):
    """U Chart - Defects per unit (variable sample size)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defects = config.get("defects")
    units = config.get("units")

    c = df[defects].dropna().values
    n = df[units].dropna().values
    u = c / n
    k = len(u)

    u_bar = np.sum(c) / np.sum(n)

    # Variable control limits
    ucl = u_bar + 3 * np.sqrt(u_bar / n)
    lcl = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n))

    # OOC detection for variable-limit chart
    u_ooc_indices = [i for i in range(k) if u[i] > ucl[i] or u[i] < lcl[i]]

    u_chart_data = [
        {
            "type": "scatter",
            "y": u.tolist(),
            "mode": "lines+markers",
            "name": "u",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [u_bar] * k,
            "mode": "lines",
            "name": "\u016b",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(u_chart_data, u, u_ooc_indices)
    result["plots"].append(
        {
            "title": "U Chart (Defects per Unit)",
            "data": u_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Defects per Unit"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    result["summary"] = (
        f"U Chart Analysis\n\n\u016b: {u_bar:.4f}\nSamples: {k}\n\nOut-of-control points: {len(u_ooc_indices)}"
    )

    n_ooc = len(u_ooc_indices)
    if n_ooc == 0:
        verdict = f"U Chart \u2014 Process in control (\u016b = {u_bar:.4f})"
        body = f"All {k} samples within variable control limits. Average defect rate \u016b = {u_bar:.4f} per unit."
        nxt = "Stable process. Variable limits account for differing inspection sizes \u2014 focus on reducing the overall rate."
    else:
        verdict = f"U Chart \u2014 {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
        body = (
            f"{n_ooc} of {k} samples ({n_ooc / k * 100:.1f}%) exceed control limits. "
            f"Average defect rate \u016b = {u_bar:.4f}. Investigate these subgroups for assignable causes."
        )
        nxt = "Identify what changed during OOC subgroups. Variable limits mean OOC points are truly unusual, not just from smaller samples."
    result["narrative"] = _narrative(
        verdict,
        body,
        next_steps=nxt,
        chart_guidance="Variable limits reflect differing inspection unit sizes. Points outside limits = genuinely unusual defect rates.",
    )

    return result
