"""SPC helper functions — Nelson rules, point rules, OOC markers, rare event charts."""


def _spc_nelson_rules(data, cl, ucl, lcl):
    """Check all 8 Nelson rules and return OOC indices + rule violations."""
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
        window = data[i - 8 : i + 1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            ooc_indices.update(range(i - 8, i + 1))
            violations.append(f"Rule 2: 9 same side at {i - 8 + 1}-{i + 1}")
            break

    # Rule 3: 6 consecutive points trending (all increasing or all decreasing)
    for i in range(5, n):
        window = data[i - 5 : i + 1]
        diffs = [window[j + 1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            ooc_indices.update(range(i - 5, i + 1))
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            violations.append(f"Rule 3: 6 {direction} at {i - 5 + 1}-{i + 1}")
            break

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i - 13 : i + 1]
            diffs = [window[j + 1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j + 1] < 0 for j in range(12)):
                ooc_indices.update(range(i - 13, i + 1))
                violations.append(f"Rule 4: 14 alternating at {i - 13 + 1}-{i + 1}")
                break

    # Rule 5: 2 of 3 beyond 2σ (same side)
    for i in range(2, n):
        w = data[i - 2 : i + 1]
        if sum(1 for v in w if v > two_sigma_up) >= 2:
            ooc_indices.update(range(i - 2, i + 1))
        if sum(1 for v in w if v < two_sigma_dn) >= 2:
            ooc_indices.update(range(i - 2, i + 1))

    # Rule 6: 4 of 5 beyond 1σ (same side)
    for i in range(4, n):
        w = data[i - 4 : i + 1]
        if sum(1 for v in w if v > one_sigma_up) >= 4:
            ooc_indices.update(range(i - 4, i + 1))
        if sum(1 for v in w if v < one_sigma_dn) >= 4:
            ooc_indices.update(range(i - 4, i + 1))

    # Rule 7: 15 consecutive within 1σ (stratification — too little variation)
    if n >= 15:
        for i in range(14, n):
            window = data[i - 14 : i + 1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                ooc_indices.update(range(i - 14, i + 1))
                violations.append(f"Rule 7: 15 within 1σ at {i - 14 + 1}-{i + 1}")
                break

    # Rule 8: 8 consecutive beyond 1σ on both sides (mixture pattern)
    if n >= 8:
        for i in range(7, n):
            window = data[i - 7 : i + 1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                ooc_indices.update(range(i - 7, i + 1))
                violations.append(f"Rule 8: 8 beyond 1σ (mixture) at {i - 7 + 1}-{i + 1}")
                break

    return list(sorted(ooc_indices)), violations


def _spc_build_point_rules(data, cl, ucl, lcl, ooc_indices):
    """Build per-point Nelson rule annotations for OOC points.

    Returns dict {index: ["Rule 1: Beyond 3σ", ...]} for each OOC index.
    """
    n = len(data)
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    one_sigma_up = cl + sigma
    one_sigma_dn = cl - sigma
    two_sigma_up = cl + 2 * sigma
    two_sigma_dn = cl - 2 * sigma
    ooc_set = set(ooc_indices)
    rules = {i: [] for i in ooc_indices}

    # Rule 1: Beyond 3σ
    for i in ooc_set:
        if data[i] > ucl or data[i] < lcl:
            rules[i].append("Rule 1: Beyond 3\u03c3")

    # Rule 2: 9 consecutive same side
    for i in range(8, n):
        window = data[i - 8 : i + 1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            for j in range(i - 8, i + 1):
                if j in ooc_set:
                    rules[j].append("Rule 2: 9 same side")
            break

    # Rule 3: 6 consecutive trending
    for i in range(5, n):
        window = data[i - 5 : i + 1]
        diffs = [window[j + 1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            for j in range(i - 5, i + 1):
                if j in ooc_set:
                    rules[j].append(f"Rule 3: 6 {direction}")
            break

    # Rule 4: 14 alternating
    if n >= 14:
        for i in range(13, n):
            window = data[i - 13 : i + 1]
            diffs = [window[j + 1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j + 1] < 0 for j in range(12)):
                for j in range(i - 13, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 4: 14 alternating")
                break

    # Rule 5: 2 of 3 beyond 2σ
    for i in range(2, n):
        w = data[i - 2 : i + 1]
        if sum(1 for v in w if v > two_sigma_up) >= 2 or sum(1 for v in w if v < two_sigma_dn) >= 2:
            for j in range(i - 2, i + 1):
                if j in ooc_set and "Rule 5: 2/3 beyond 2\u03c3" not in rules[j]:
                    rules[j].append("Rule 5: 2/3 beyond 2\u03c3")

    # Rule 6: 4 of 5 beyond 1σ
    for i in range(4, n):
        w = data[i - 4 : i + 1]
        if sum(1 for v in w if v > one_sigma_up) >= 4 or sum(1 for v in w if v < one_sigma_dn) >= 4:
            for j in range(i - 4, i + 1):
                if j in ooc_set and "Rule 6: 4/5 beyond 1\u03c3" not in rules[j]:
                    rules[j].append("Rule 6: 4/5 beyond 1\u03c3")

    # Rule 7: 15 within 1σ (stratification)
    if n >= 15:
        for i in range(14, n):
            window = data[i - 14 : i + 1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                for j in range(i - 14, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 7: 15 within 1\u03c3")
                break

    # Rule 8: 8 beyond 1σ both sides (mixture)
    if n >= 8:
        for i in range(7, n):
            window = data[i - 7 : i + 1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                for j in range(i - 7, i + 1):
                    if j in ooc_set:
                        rules[j].append("Rule 8: Mixture pattern")
                break

    return rules


def _spc_add_ooc_markers(plot_data, data, ooc_indices, point_rules=None):
    """Add red markers for OOC points and customdata to main trace for click-to-inspect."""
    n = len(data) if hasattr(data, "__len__") else 0
    ooc_set = set(ooc_indices) if ooc_indices else set()

    # Tag the first (main data) trace with customdata so every point is clickable
    if plot_data and n > 0:
        main_trace = plot_data[0]
        if "customdata" not in main_trace:
            main_trace["customdata"] = [
                [i, "; ".join(point_rules.get(i, [])) if point_rules and i in ooc_set else ""] for i in range(n)
            ]
            main_trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>"

    if not ooc_indices:
        return
    ooc_x = ooc_indices
    ooc_y = [float(data[i]) for i in ooc_indices]
    trace = {
        "type": "scatter",
        "x": ooc_x,
        "y": ooc_y,
        "mode": "markers",
        "name": "Out of Control",
        "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
        "showlegend": True,
    }
    # Add customdata for click-to-inspect
    if point_rules is not None:
        trace["customdata"] = [[i, "; ".join(point_rules.get(i, []))] for i in ooc_indices]
        trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<br>%{customdata[1]}<extra>OOC</extra>"
    else:
        trace["customdata"] = [[i, ""] for i in ooc_indices]
        trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra>OOC</extra>"
    plot_data.append(trace)


def run_g_t_chart(df, config):
    """Rare Events Charts — G Chart / T Chart."""
    import numpy as np
    from scipy import stats
    from scipy.stats import weibull_min

    from ..common import _narrative

    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    chart_type = config.get("chart_type")  # "g" or "t" — auto-detect if not given

    col = df[var].dropna()
    values = col.values.astype(float)
    n = len(values)

    if n < 3:
        result["summary"] = "Need at least 3 data points for a rare events chart."
        return result

    # Auto-detect chart type
    if chart_type is None:
        if col.dtype in ["int64", "int32"] and (values == values.astype(int)).all():
            chart_type = "g"
        else:
            chart_type = "t"

    if chart_type == "g":
        # G Chart — geometric distribution
        g_bar = float(np.mean(values))
        p_est = 1 / (g_bar + 1) if g_bar > 0 else 0.5
        alpha_chart = 0.0027  # 3-sigma equivalent

        cl = g_bar
        if p_est > 0 and p_est < 1:
            ucl = float(stats.geom.ppf(1 - alpha_chart / 2, p_est) - 1)
            lcl = max(0, float(stats.geom.ppf(alpha_chart / 2, p_est) - 1))
        else:
            ucl = g_bar * 3
            lcl = 0

        chart_label = "G Chart (Opportunities Between Events)"
        y_label = "Count Between Events"
        shape = None
        scale = None

    else:
        # T Chart — time between events, Weibull/exponential transform
        try:
            shape, loc, scale = weibull_min.fit(values, floc=0)
        except Exception:
            shape, scale = 1.0, float(np.mean(values))

        mean_t = float(np.mean(values))
        cl = mean_t

        ucl = float(weibull_min.ppf(0.99865, shape, 0, scale))
        lcl = max(0, float(weibull_min.ppf(0.00135, shape, 0, scale)))

        chart_label = "T Chart (Time Between Events)"
        y_label = "Time Between Events"

    # Detect OOC points
    ooc_indices = []
    for i in range(n):
        if values[i] > ucl or values[i] < lcl:
            ooc_indices.append(i)

    x_axis = list(range(1, n + 1))
    colors = ["#e85747" if i in ooc_indices else "#4a9f6e" for i in range(n)]

    result["plots"].append(
        {
            "title": chart_label,
            "data": [
                {
                    "type": "scatter",
                    "x": x_axis,
                    "y": values.tolist(),
                    "mode": "lines+markers",
                    "line": {"color": "#4a9f6e"},
                    "marker": {"color": colors, "size": 6},
                    "name": var,
                    "customdata": [[i, "OOC" if i in ooc_indices else ""] for i in range(n)],
                    "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [cl, cl],
                    "mode": "lines",
                    "line": {"color": "#e8c547", "width": 1},
                    "name": f"CL = {cl:.2f}",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [ucl, ucl],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"UCL = {ucl:.2f}",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [lcl, lcl],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"LCL = {lcl:.2f}",
                },
            ],
            "layout": {
                "height": 390,
                "xaxis": {"title": "Observation", "rangeslider": {"visible": True, "thickness": 0.12}},
                "yaxis": {"title": y_label},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    summary = f"{chart_label}\n\n"
    summary += f"Variable: {var}\n"
    summary += f"Observations: {n}\n"
    summary += f"CL: {cl:.4f}\n"
    summary += f"UCL: {ucl:.4f}\n"
    summary += f"LCL: {lcl:.4f}\n"
    summary += f"Out-of-control: {len(ooc_indices)}\n"
    if ooc_indices:
        summary += f"OOC observations: {', '.join(str(i + 1) for i in ooc_indices)}\n"
    if chart_type == "t" and shape is not None:
        summary += f"\nWeibull shape: {shape:.3f} (1.0 = exponential)\n"
        summary += f"Weibull scale: {scale:.3f}\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"{'G' if chart_type == 'g' else 'T'} chart: CL={cl:.2f}, {len(ooc_indices)} OOC points out of {n}."
    )
    result["statistics"] = {
        "chart_type": chart_type,
        "n": n,
        "cl": cl,
        "ucl": ucl,
        "lcl": lcl,
        "ooc_count": len(ooc_indices),
        "ooc_indices": [i + 1 for i in ooc_indices],
    }

    # Narrative
    _gt_label = "G Chart (count between events)" if chart_type == "g" else "T Chart (time between events)"
    _gt_n_ooc = len(ooc_indices)
    if _gt_n_ooc == 0:
        _gt_verdict = f"{_gt_label} \u2014 process in control"
        _gt_body = (
            f"All {n} observations fall within control limits (CL = {cl:.2f}). The rate of rare events appears stable."
        )
    else:
        _gt_verdict = f"{_gt_label} \u2014 {_gt_n_ooc} out-of-control point{'s' if _gt_n_ooc > 1 else ''}"
        _gt_body = f"{_gt_n_ooc} of {n} observations exceed control limits, suggesting the event rate has shifted."
    result["narrative"] = _narrative(
        _gt_verdict,
        _gt_body,
        next_steps="Investigate OOC points for assignable causes. A cluster of short intervals suggests a worsening event rate."
        if _gt_n_ooc > 0
        else "Continue monitoring. Consider adding process improvement to reduce the baseline event rate.",
        chart_guidance="Points above UCL indicate unusually long gaps between events (improvement). Points below LCL indicate unusually short gaps (deterioration).",
    )

    return result
