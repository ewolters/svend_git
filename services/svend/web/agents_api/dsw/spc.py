"""DSW Statistical Process Control — SPC chart analysis blocks."""

import logging
import numpy as np
import pandas as pd
from scipy import stats

from .common import _effect_magnitude, _practical_block, _fit_best_distribution

logger = logging.getLogger(__name__)


def _spc_nelson_rules(data, cl, ucl, lcl):
    """Check all 8 Nelson rules and return OOC indices + rule violations."""
    import numpy as np
    n = len(data)
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    one_sigma_up = cl + sigma
    one_sigma_dn = cl - sigma
    two_sigma_up = cl + 2 * sigma
    two_sigma_dn = cl - 2 * sigma
    ooc_indices = set()
    violations = []

    # Rule 1: Point beyond 3σ (beyond control limits)
    for i in range(n):
        if data[i] > ucl or data[i] < lcl:
            ooc_indices.add(i)

    # Rule 2: 9 consecutive points same side of CL
    for i in range(8, n):
        window = data[i-8:i+1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            ooc_indices.update(range(i-8, i+1))
            violations.append(f"Rule 2: 9 same side at {i-8+1}-{i+1}")
            break

    # Rule 3: 6 consecutive points trending (all increasing or all decreasing)
    for i in range(5, n):
        window = data[i-5:i+1]
        diffs = [window[j+1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            ooc_indices.update(range(i-5, i+1))
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            violations.append(f"Rule 3: 6 {direction} at {i-5+1}-{i+1}")
            break

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i-13:i+1]
            diffs = [window[j+1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j+1] < 0 for j in range(12)):
                ooc_indices.update(range(i-13, i+1))
                violations.append(f"Rule 4: 14 alternating at {i-13+1}-{i+1}")
                break

    # Rule 5: 2 of 3 beyond 2σ (same side)
    for i in range(2, n):
        w = data[i-2:i+1]
        if sum(1 for v in w if v > two_sigma_up) >= 2:
            ooc_indices.update(range(i-2, i+1))
        if sum(1 for v in w if v < two_sigma_dn) >= 2:
            ooc_indices.update(range(i-2, i+1))

    # Rule 6: 4 of 5 beyond 1σ (same side)
    for i in range(4, n):
        w = data[i-4:i+1]
        if sum(1 for v in w if v > one_sigma_up) >= 4:
            ooc_indices.update(range(i-4, i+1))
        if sum(1 for v in w if v < one_sigma_dn) >= 4:
            ooc_indices.update(range(i-4, i+1))

    # Rule 7: 15 consecutive within 1σ (stratification — too little variation)
    if n >= 15:
        for i in range(14, n):
            window = data[i-14:i+1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                ooc_indices.update(range(i-14, i+1))
                violations.append(f"Rule 7: 15 within 1σ at {i-14+1}-{i+1}")
                break

    # Rule 8: 8 consecutive beyond 1σ on both sides (mixture pattern)
    if n >= 8:
        for i in range(7, n):
            window = data[i-7:i+1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                ooc_indices.update(range(i-7, i+1))
                violations.append(f"Rule 8: 8 beyond 1σ (mixture) at {i-7+1}-{i+1}")
                break

    return list(sorted(ooc_indices)), violations


def _spc_add_ooc_markers(plot_data, data, ooc_indices):
    """Add red markers for OOC points to a Plotly chart trace list."""
    if not ooc_indices:
        return
    import numpy as np
    ooc_x = ooc_indices
    ooc_y = [float(data[i]) for i in ooc_indices]
    plot_data.append({
        "type": "scatter", "x": ooc_x, "y": ooc_y,
        "mode": "markers", "name": "Out of Control",
        "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
        "showlegend": True
    })


def run_spc_analysis(df, analysis_id, config):
    """Run SPC analysis."""
    import numpy as np

    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        # Multivariate analyses (mewma, generalized_variance) don't use single measurement
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    if analysis_id == "imr":
        # Individual-Moving Range chart
        n = len(data)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr)

        x_bar = np.mean(data)
        ucl = x_bar + 2.66 * mr_bar
        lcl = x_bar - 2.66 * mr_bar

        mr_ucl = 3.267 * mr_bar

        # Nelson rules check
        ooc_indices, rule_violations = _spc_nelson_rules(data, x_bar, ucl, lcl)

        # I Chart with OOC markers
        i_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "lines+markers", "name": "Value", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_bar]*n, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(i_chart_data, data, ooc_indices)
        result["plots"].append({
            "title": "I Chart (Individuals)",
            "data": i_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True}
        })

        # MR Chart with OOC markers
        mr_ooc = [i for i in range(len(mr)) if mr[i] > mr_ucl]
        mr_chart_data = [
            {"type": "scatter", "y": mr.tolist(), "mode": "lines+markers", "name": "MR", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [mr_bar]*(n-1), "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [mr_ucl]*(n-1), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(mr_chart_data, mr, mr_ooc)
        result["plots"].append({
            "title": "MR Chart (Moving Range)",
            "data": mr_chart_data,
            "layout": {"template": "plotly_dark", "height": 250}
        })

        ooc = len(ooc_indices)
        violations_text = ""
        if rule_violations:
            violations_text = "\n\nNelson Rule Violations:\n" + "\n".join(f"  {v}" for v in rule_violations)
        result["summary"] = f"I-MR Chart Analysis\n\nMean: {x_bar:.4f}\nUCL: {ucl:.4f}\nLCL: {lcl:.4f}\nMR-bar: {mr_bar:.4f}\n\nOut-of-control points: {ooc}{violations_text}"

        result["guide_observation"] = f"Control chart shows {ooc} out-of-control points." + (" Process appears stable." if ooc == 0 else " Investigation recommended.")

    elif analysis_id == "capability":
        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None
        target = float(config.get("target")) if config.get("target") else None

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary = f"Capability Analysis\n\nMean: {mean:.4f}\nStd Dev: {std:.4f}\n"

        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * std)
            if target:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            else:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

            summary += f"\nCp: {cp:.3f}\nCpk: {cpk:.3f}"

            if cpk >= 1.33:
                summary += "\n\nProcess is capable (Cpk >= 1.33)"
            elif cpk >= 1.0:
                summary += "\n\nProcess is marginally capable (1.0 <= Cpk < 1.33)"
            else:
                summary += "\n\nProcess is NOT capable (Cpk < 1.0)"

            result["guide_observation"] = f"Process capability Cpk = {cpk:.2f}. " + ("Capable." if cpk >= 1.33 else "Needs improvement.")

        # Histogram with spec limits - Svend theme (pale green fill, bright green border)
        hist_data = [{
            "type": "histogram",
            "x": data.tolist(),
            "name": "Data",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",  # Pale green fill
                "line": {"color": "#4a9f6e", "width": 1.5}  # Bright green border
            }
        }]

        # Add LSL/USL as dashed vertical lines (matching anomaly threshold style)
        shapes = []
        annotations = []

        if lsl:
            shapes.append({
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        if usl:
            shapes.append({
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        result["plots"].append({
            "title": "Capability Histogram",
            "data": hist_data,
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "shapes": shapes,
                "annotations": annotations,
                "margin": {"t": 40}
            }
        })

        result["summary"] = summary

        # What-If data for client-side interactive exploration
        result["what_if_data"] = {
            "type": "capability",
            "mean": float(mean),
            "std": float(std),
            "n": int(len(data)),
            "current_lsl": float(lsl) if lsl else None,
            "current_usl": float(usl) if usl else None,
            "data_values": data.tolist() if len(data) <= 5000 else data[:5000].tolist(),
        }

    elif analysis_id == "xbar_r":
        # Xbar-R Chart for subgrouped data
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            # Group by subgroup column
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            # Create subgroups from sequential data
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        ranges = np.array([np.max(g) - np.min(g) for g in groups])

        x_double_bar = np.mean(x_bars)
        r_bar = np.mean(ranges)

        # Control chart constants (for subgroup size 2-10)
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d3_table = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}
        A2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_table = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

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
        # Nelson rules for R
        r_ooc, r_violations = _spc_nelson_rules(ranges, r_bar, r_ucl, r_lcl)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # R Chart with OOC markers
        r_chart_data = [
            {"type": "scatter", "y": ranges.tolist(), "mode": "lines+markers", "name": "R", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [r_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [r_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [r_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(r_chart_data, ranges, r_ooc)
        result["plots"].append({
            "title": "R Chart",
            "data": r_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        violations_text = ""
        if xbar_violations or r_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in r_violations: violations_text += f"\n  R: {v}"

        result["summary"] = f"Xbar-R Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nR Chart:\n  R̄: {r_bar:.4f}\n  UCL: {r_ucl:.4f}\n  LCL: {r_lcl:.4f}\n  OOC points: {len(r_ooc)}{violations_text}"

    elif analysis_id == "xbar_s":
        # Xbar-S Chart (using standard deviation instead of range)
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        stds = np.array([np.std(g, ddof=1) for g in groups])

        x_double_bar = np.mean(x_bars)
        s_bar = np.mean(stds)

        # Control chart constants for S chart
        c4_table = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}
        B3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
        B4_table = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
        A3_table = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}

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
        s_ooc, s_violations = _spc_nelson_rules(stds, s_bar, s_ucl, s_lcl)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # S Chart with OOC markers
        s_chart_data = [
            {"type": "scatter", "y": stds.tolist(), "mode": "lines+markers", "name": "S", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [s_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [s_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [s_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(s_chart_data, stds, s_ooc)
        result["plots"].append({
            "title": "S Chart",
            "data": s_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        violations_text = ""
        if xbar_violations or s_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in s_violations: violations_text += f"\n  S: {v}"

        result["summary"] = f"Xbar-S Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nS Chart:\n  S̄: {s_bar:.4f}\n  UCL: {s_ucl:.4f}\n  LCL: {s_lcl:.4f}\n  OOC points: {len(s_ooc)}{violations_text}"

    elif analysis_id == "p_chart":
        # P Chart for proportion defective
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
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(p_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "P Chart (Proportion Defective)",
            "data": p_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Proportion"}}
        })

        result["summary"] = f"P Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}"

    elif analysis_id == "np_chart":
        """
        NP Chart - Number defective (constant sample size).
        """
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

        np_chart_data = [
            {"type": "scatter", "y": d.tolist(), "mode": "lines+markers", "name": "np", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [np_bar]*k, "mode": "lines", "name": "n̄p", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(np_chart_data, d, np_ooc)
        result["plots"].append({
            "title": "NP Chart (Number Defective)",
            "data": np_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Number Defective"}}
        })

        violations_text = ""
        if np_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in np_violations: violations_text += f"\n  {v}"

        result["summary"] = f"NP Chart Analysis\n\nn̄p: {np_bar:.2f}\nSample size: {n}\np̄: {p_bar:.4f}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(np_ooc)}{violations_text}"

    elif analysis_id == "c_chart":
        """
        C Chart - Count of defects per unit (constant opportunity).
        """
        defects = config.get("defects")

        c = df[defects].dropna().values
        k = len(c)

        c_bar = np.mean(c)

        # Control limits (Poisson-based)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

        c_ooc, c_violations = _spc_nelson_rules(c, c_bar, ucl, lcl)

        c_chart_data = [
            {"type": "scatter", "y": c.tolist(), "mode": "lines+markers", "name": "c", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [c_bar]*k, "mode": "lines", "name": "c̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(c_chart_data, c, c_ooc)
        result["plots"].append({
            "title": "C Chart (Defects per Unit)",
            "data": c_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects"}}
        })

        violations_text = ""
        if c_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in c_violations: violations_text += f"\n  {v}"

        result["summary"] = f"C Chart Analysis\n\nc̄: {c_bar:.2f}\nSamples: {k}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(c_ooc)}{violations_text}"

    elif analysis_id == "u_chart":
        """
        U Chart - Defects per unit (variable sample size).
        """
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
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(u_chart_data, u, u_ooc_indices)
        result["plots"].append({
            "title": "U Chart (Defects per Unit)",
            "data": u_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects per Unit"}}
        })

        result["summary"] = f"U Chart Analysis\n\nū: {u_bar:.4f}\nSamples: {k}\n\nOut-of-control points: {len(u_ooc_indices)}"

    elif analysis_id == "cusum":
        """
        CUSUM Chart - Cumulative Sum for detecting small shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))  # Target value
        k_param = float(config.get("k", 0.5))  # Slack value (typically 0.5)
        h_param = float(config.get("h", 5))  # Decision interval

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        # Estimate standard deviation
        sigma = np.std(data, ddof=1)

        # Standardize
        z = (data - target) / sigma

        # Calculate CUSUM
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_pos[i] = max(0, z[i] - k_param)
                cusum_neg[i] = max(0, -z[i] - k_param)
            else:
                cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - k_param)
                cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - k_param)

        # Detect signals
        signals_pos = np.where(cusum_pos > h_param)[0]
        signals_neg = np.where(cusum_neg > h_param)[0]

        cusum_chart_data = [
            {"type": "scatter", "y": cusum_pos.tolist(), "mode": "lines", "name": "CUSUM+", "line": {"color": "#4a9f6e", "width": 2}},
            {"type": "scatter", "y": (-cusum_neg).tolist(), "mode": "lines", "name": "CUSUM-", "line": {"color": "#47a5e8", "width": 2}},
            {"type": "scatter", "y": [h_param]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [-h_param]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        # OOC markers for positive signals
        if len(signals_pos) > 0:
            cusum_chart_data.append({
                "type": "scatter", "x": signals_pos.tolist(), "y": cusum_pos[signals_pos].tolist(),
                "mode": "markers", "name": "Signal (up)",
                "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                "showlegend": True
            })
        # OOC markers for negative signals
        if len(signals_neg) > 0:
            cusum_chart_data.append({
                "type": "scatter", "x": signals_neg.tolist(), "y": (-cusum_neg[signals_neg]).tolist(),
                "mode": "markers", "name": "Signal (down)",
                "marker": {"color": "#e89547", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                "showlegend": True
            })
        result["plots"].append({
            "title": "CUSUM Chart",
            "data": cusum_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "CUSUM"}}
        })

        result["summary"] = f"CUSUM Chart Analysis\n\nTarget: {target:.4f}\nσ estimate: {sigma:.4f}\nk (slack): {k_param}\nh (decision): {h_param}\n\nUpward shift signals: {len(signals_pos)} at points {list(signals_pos[:5])}{'...' if len(signals_pos) > 5 else ''}\nDownward shift signals: {len(signals_neg)} at points {list(signals_neg[:5])}{'...' if len(signals_neg) > 5 else ''}"

    elif analysis_id == "ewma":
        """
        EWMA Chart - Exponentially Weighted Moving Average.
        Good for detecting small sustained shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))
        lambda_param = float(config.get("lambda", 0.2))  # Smoothing parameter
        L = float(config.get("L", 3))  # Control limit width

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        sigma = np.std(data, ddof=1)

        # Calculate EWMA
        ewma = np.zeros(n)
        ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target

        for i in range(1, n):
            ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]

        # Control limits (they vary with time, approaching steady state)
        factor = lambda_param / (2 - lambda_param)
        ucl = target + L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))
        lcl = target - L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))

        # Steady-state limits
        ucl_ss = target + L * sigma * np.sqrt(factor)
        lcl_ss = target - L * sigma * np.sqrt(factor)

        # OOC detection for variable-limit EWMA
        ewma_ooc = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

        ewma_chart_data = [
            {"type": "scatter", "y": ewma.tolist(), "mode": "lines+markers", "name": "EWMA", "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
            {"type": "scatter", "y": [target]*n, "mode": "lines", "name": "Target", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(ewma_chart_data, ewma, ewma_ooc)
        result["plots"].append({
            "title": "EWMA Chart",
            "data": ewma_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "EWMA"}}
        })

        result["summary"] = f"EWMA Chart Analysis\n\nTarget: {target:.4f}\nλ (smoothing): {lambda_param}\nL (sigma width): {L}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nOut-of-control points: {len(ewma_ooc)}"

    elif analysis_id == "laney_p":
        """
        Laney P' Chart - P chart adjusted for overdispersion.
        Uses sigma_z correction factor to account for extra-binomial variation.
        """
        defectives = config.get("defectives")
        sample_size_col = config.get("sample_size")

        d = df[defectives].dropna().values
        n = df[sample_size_col].dropna().values
        p = d / n
        k = len(p)

        p_bar = np.sum(d) / np.sum(n)

        # Standard p-chart z-values
        z = (p - p_bar) / np.sqrt(p_bar * (1 - p_bar) / n)

        # Moving range of z-values for sigma_z
        mr_z = np.abs(np.diff(z))
        sigma_z = np.mean(mr_z) / 1.128  # d2 for n=2

        # Laney-adjusted limits
        ucl = p_bar + 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n)
        lcl = np.maximum(0, p_bar - 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n))

        ooc_indices = [i for i in range(k) if p[i] > ucl[i] or p[i] < lcl[i]]

        lp_chart_data = [
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lp_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "Laney P' Chart",
            "data": lp_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Proportion"}}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney P' Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard P chart would give too many false alarms."

    elif analysis_id == "laney_u":
        """
        Laney U' Chart - U chart adjusted for overdispersion.
        """
        defects = config.get("defects")
        units = config.get("units")

        c = df[defects].dropna().values
        n = df[units].dropna().values
        u = c / n
        k = len(u)

        u_bar = np.sum(c) / np.sum(n)

        # Standard u-chart z-values
        z = (u - u_bar) / np.sqrt(u_bar / n)

        # Moving range of z-values for sigma_z
        mr_z = np.abs(np.diff(z))
        sigma_z = np.mean(mr_z) / 1.128

        # Laney-adjusted limits
        ucl = u_bar + 3 * sigma_z * np.sqrt(u_bar / n)
        lcl = np.maximum(0, u_bar - 3 * sigma_z * np.sqrt(u_bar / n))

        ooc_indices = [i for i in range(k) if u[i] > ucl[i] or u[i] < lcl[i]]

        lu_chart_data = [
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lu_chart_data, u, ooc_indices)
        result["plots"].append({
            "title": "Laney U' Chart",
            "data": lu_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects per Unit"}}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney U' Chart Analysis\n\nū: {u_bar:.4f}\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard U chart would give too many false alarms."

    elif analysis_id == "between_within":
        """
        Between/Within Capability - Nested variance components analysis.
        Separates total variation into between-subgroup and within-subgroup components.
        """
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))
        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None

        if subgroup_col:
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = [np.array(g) for g in groups if len(g) >= 2]
        k = len(groups)

        # Within-subgroup variance (pooled)
        within_vars = [np.var(g, ddof=1) for g in groups]
        sigma_within = np.sqrt(np.mean(within_vars))

        # Between-subgroup variance
        group_means = np.array([np.mean(g) for g in groups])
        grand_mean = np.mean(data)
        n_avg = np.mean([len(g) for g in groups])

        sigma_between_sq = np.var(group_means, ddof=1) - sigma_within**2 / n_avg
        sigma_between = np.sqrt(max(0, sigma_between_sq))

        # Total (overall)
        sigma_total = np.std(data, ddof=1)

        # Between/Within combined
        sigma_bw = np.sqrt(sigma_between**2 + sigma_within**2)

        summary = f"Between/Within Capability Analysis\n\nSubgroups: {k}\n\nVariance Components:\n  σ Within: {sigma_within:.4f}\n  σ Between: {sigma_between:.4f}\n  σ B/W: {sigma_bw:.4f}\n  σ Overall: {sigma_total:.4f}\n\n% of Total Variance:\n  Within: {(sigma_within**2 / sigma_total**2 * 100):.1f}%\n  Between: {(sigma_between**2 / sigma_total**2 * 100):.1f}%\n"

        if lsl is not None and usl is not None:
            # Within capability
            cp_within = (usl - lsl) / (6 * sigma_within)
            cpk_within = min((usl - grand_mean) / (3 * sigma_within), (grand_mean - lsl) / (3 * sigma_within))

            # B/W capability
            cp_bw = (usl - lsl) / (6 * sigma_bw)
            cpk_bw = min((usl - grand_mean) / (3 * sigma_bw), (grand_mean - lsl) / (3 * sigma_bw))

            # Overall capability
            pp = (usl - lsl) / (6 * sigma_total)
            ppk = min((usl - grand_mean) / (3 * sigma_total), (grand_mean - lsl) / (3 * sigma_total))

            summary += f"\nWithin Capability:\n  Cp: {cp_within:.3f}\n  Cpk: {cpk_within:.3f}\n\nBetween/Within Capability:\n  Cp (B/W): {cp_bw:.3f}\n  Cpk (B/W): {cpk_bw:.3f}\n\nOverall Capability:\n  Pp: {pp:.3f}\n  Ppk: {ppk:.3f}"

        # Variance components bar chart
        result["plots"].append({
            "title": "Variance Components",
            "data": [{
                "type": "bar",
                "x": ["Within", "Between", "B/W Combined", "Overall"],
                "y": [sigma_within, sigma_between, sigma_bw, sigma_total],
                "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a"]},
                "text": [f"{sigma_within:.4f}", f"{sigma_between:.4f}", f"{sigma_bw:.4f}", f"{sigma_total:.4f}"],
                "textposition": "outside"
            }],
            "layout": {"template": "plotly_dark", "height": 280, "yaxis": {"title": "Std Dev (σ)"}}
        })

        # Histogram with within vs overall fits
        from scipy import stats as sp_stats
        x_range = np.linspace(min(data), max(data), 200)
        hist_data = [
            {"type": "histogram", "x": data.tolist(), "name": "Data", "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}}, "histnorm": "probability density"},
            {"type": "scatter", "x": x_range.tolist(), "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_within).tolist(), "mode": "lines", "name": f"Within (σ={sigma_within:.3f})", "line": {"color": "#4a90d9", "width": 2}},
            {"type": "scatter", "x": x_range.tolist(), "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_total).tolist(), "mode": "lines", "name": f"Overall (σ={sigma_total:.3f})", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
        ]

        layout = {"template": "plotly_dark", "height": 300, "showlegend": True, "shapes": [], "annotations": []}
        if lsl is not None:
            layout["shapes"].append({"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": lsl, "y": 1.05, "yref": "paper", "text": "LSL", "showarrow": False, "font": {"color": "#e85747"}})
        if usl is not None:
            layout["shapes"].append({"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": usl, "y": 1.05, "yref": "paper", "text": "USL", "showarrow": False, "font": {"color": "#e85747"}})

        result["plots"].append({
            "title": "Within vs Overall Distribution",
            "data": hist_data,
            "layout": layout
        })

        result["summary"] = summary

    elif analysis_id == "nonnormal_capability":
        """
        Non-Normal Capability Analysis.
        Fits Normal, Lognormal, Weibull, and Exponential distributions,
        selects best fit, and computes equivalent Pp/Ppk.
        """
        from scipy import stats as sp_stats

        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None

        pos_data = data[data > 0]  # needed for lognormal/weibull

        # Fit distributions
        fits = {}

        # Normal
        mu_n, sigma_n = sp_stats.norm.fit(data)
        fits["Normal"] = {"params": (mu_n, sigma_n), "dist": sp_stats.norm, "args": (mu_n, sigma_n),
                          "ks": sp_stats.kstest(data, "norm", args=(mu_n, sigma_n))}

        # Lognormal (needs positive data)
        if len(pos_data) > 10:
            shape_ln, loc_ln, scale_ln = sp_stats.lognorm.fit(pos_data, floc=0)
            fits["Lognormal"] = {"params": (shape_ln, 0, scale_ln), "dist": sp_stats.lognorm, "args": (shape_ln, 0, scale_ln),
                                 "ks": sp_stats.kstest(pos_data, "lognorm", args=(shape_ln, 0, scale_ln))}

        # Weibull (needs positive data)
        if len(pos_data) > 10:
            shape_w, loc_w, scale_w = sp_stats.weibull_min.fit(pos_data, floc=0)
            fits["Weibull"] = {"params": (shape_w, 0, scale_w), "dist": sp_stats.weibull_min, "args": (shape_w, 0, scale_w),
                               "ks": sp_stats.kstest(pos_data, "weibull_min", args=(shape_w, 0, scale_w))}

        # Exponential (needs positive data)
        if len(pos_data) > 10:
            loc_e, scale_e = sp_stats.expon.fit(pos_data)
            fits["Exponential"] = {"params": (loc_e, scale_e), "dist": sp_stats.expon, "args": (loc_e, scale_e),
                                   "ks": sp_stats.kstest(pos_data, "expon", args=(loc_e, scale_e))}

        # Select best fit by KS p-value (highest = best fit)
        best_name = max(fits, key=lambda k: fits[k]["ks"].pvalue)
        best = fits[best_name]
        best_dist = best["dist"]
        best_args = best["args"]

        summary = f"Non-Normal Capability Analysis\n\nBest Fit Distribution: {best_name}\n\n"
        summary += f"Distribution Fit Comparison (Anderson-Darling / KS test):\n"
        summary += f"  {'Distribution':<15} {'KS Stat':>10} {'p-value':>10} {'Fit':>6}\n"
        summary += f"  {'-'*45}\n"
        for name, info in fits.items():
            marker = " <--" if name == best_name else ""
            summary += f"  {name:<15} {info['ks'].statistic:>10.4f} {info['ks'].pvalue:>10.4f} {marker}\n"

        # Compute Pp/Ppk using the fitted distribution
        if lsl is not None and usl is not None:
            p_lsl = best_dist.cdf(lsl, *best_args)
            p_usl = 1 - best_dist.cdf(usl, *best_args)

            # Equivalent Pp from total proportion out of spec
            from scipy.stats import norm as sp_norm
            total_ppm = (p_lsl + p_usl) * 1e6
            # Z-equivalent
            z_lsl = sp_norm.ppf(1 - p_lsl) if p_lsl < 1 else 0
            z_usl = sp_norm.ppf(1 - p_usl) if p_usl < 1 else 0
            ppk_equiv = min(z_lsl, z_usl) / 3 if (z_lsl > 0 and z_usl > 0) else 0

            # Pp from spec width vs distribution spread (0.135% to 99.865%)
            q_low = best_dist.ppf(0.00135, *best_args)
            q_high = best_dist.ppf(0.99865, *best_args)
            spread_6sigma = q_high - q_low
            pp_equiv = (usl - lsl) / spread_6sigma if spread_6sigma > 0 else 0

            summary += f"\nCapability Indices ({best_name} fit):\n"
            summary += f"  Pp (equivalent): {pp_equiv:.3f}\n"
            summary += f"  Ppk (equivalent): {ppk_equiv:.3f}\n"
            summary += f"  P(below LSL): {p_lsl*100:.4f}%\n"
            summary += f"  P(above USL): {p_usl*100:.4f}%\n"
            summary += f"  Total PPM: {total_ppm:.0f}\n"

        # Histogram with best-fit overlay
        x_range = np.linspace(min(data), max(data), 200)
        pdf_vals = best_dist.pdf(x_range, *best_args)

        hist_data = [
            {"type": "histogram", "x": data.tolist(), "name": "Data",
             "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}},
             "histnorm": "probability density"},
            {"type": "scatter", "x": x_range.tolist(), "y": pdf_vals.tolist(), "mode": "lines",
             "name": f"{best_name} Fit", "line": {"color": "#4a90d9", "width": 2}},
        ]

        layout = {"template": "plotly_dark", "height": 300, "showlegend": True, "shapes": [], "annotations": []}
        if lsl is not None:
            layout["shapes"].append({"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": lsl, "y": 1.05, "yref": "paper", "text": "LSL", "showarrow": False, "font": {"color": "#e85747"}})
        if usl is not None:
            layout["shapes"].append({"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": usl, "y": 1.05, "yref": "paper", "text": "USL", "showarrow": False, "font": {"color": "#e85747"}})

        result["plots"].append({
            "title": f"Non-Normal Capability ({best_name} Fit)",
            "data": hist_data,
            "layout": layout
        })

        # Probability plot for best fit
        sorted_d = np.sort(pos_data if best_name != "Normal" else data)
        n_pts = len(sorted_d)
        median_ranks = (np.arange(1, n_pts+1) - 0.3) / (n_pts + 0.4)
        theoretical_q = best_dist.ppf(median_ranks, *best_args)

        result["plots"].append({
            "title": f"Probability Plot ({best_name})",
            "data": [
                {"type": "scatter", "x": theoretical_q.tolist(), "y": sorted_d.tolist(),
                 "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 5}},
                {"type": "scatter", "x": [min(theoretical_q), max(theoretical_q)],
                 "y": [min(theoretical_q), max(theoretical_q)],
                 "mode": "lines", "name": "Reference", "line": {"color": "#d94a4a", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": f"Theoretical ({best_name})"}, "yaxis": {"title": "Observed"}}
        })

        result["summary"] = summary
