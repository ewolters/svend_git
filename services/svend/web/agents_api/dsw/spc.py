"""DSW Statistical Process Control — control charts, capability, Bayesian SPC."""

import numpy as np
from scipy import stats as sp_stats

from .common import _narrative, SVEND_COLORS, COLOR_GOOD, COLOR_BAD, COLOR_WARNING, COLOR_INFO, COLOR_NEUTRAL, COLOR_REFERENCE, _rgba


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
        window = data[i-8:i+1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            for j in range(i-8, i+1):
                if j in ooc_set:
                    rules[j].append("Rule 2: 9 same side")
            break

    # Rule 3: 6 consecutive trending
    for i in range(5, n):
        window = data[i-5:i+1]
        diffs = [window[j+1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            for j in range(i-5, i+1):
                if j in ooc_set:
                    rules[j].append(f"Rule 3: 6 {direction}")
            break

    # Rule 4: 14 alternating
    if n >= 14:
        for i in range(13, n):
            window = data[i-13:i+1]
            diffs = [window[j+1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j+1] < 0 for j in range(12)):
                for j in range(i-13, i+1):
                    if j in ooc_set:
                        rules[j].append("Rule 4: 14 alternating")
                break

    # Rule 5: 2 of 3 beyond 2σ
    for i in range(2, n):
        w = data[i-2:i+1]
        if sum(1 for v in w if v > two_sigma_up) >= 2 or sum(1 for v in w if v < two_sigma_dn) >= 2:
            for j in range(i-2, i+1):
                if j in ooc_set and "Rule 5: 2/3 beyond 2\u03c3" not in rules[j]:
                    rules[j].append("Rule 5: 2/3 beyond 2\u03c3")

    # Rule 6: 4 of 5 beyond 1σ
    for i in range(4, n):
        w = data[i-4:i+1]
        if sum(1 for v in w if v > one_sigma_up) >= 4 or sum(1 for v in w if v < one_sigma_dn) >= 4:
            for j in range(i-4, i+1):
                if j in ooc_set and "Rule 6: 4/5 beyond 1\u03c3" not in rules[j]:
                    rules[j].append("Rule 6: 4/5 beyond 1\u03c3")

    # Rule 7: 15 within 1σ (stratification)
    if n >= 15:
        for i in range(14, n):
            window = data[i-14:i+1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                for j in range(i-14, i+1):
                    if j in ooc_set:
                        rules[j].append("Rule 7: 15 within 1\u03c3")
                break

    # Rule 8: 8 beyond 1σ both sides (mixture)
    if n >= 8:
        for i in range(7, n):
            window = data[i-7:i+1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                for j in range(i-7, i+1):
                    if j in ooc_set:
                        rules[j].append("Rule 8: Mixture pattern")
                break

    return rules


def _spc_add_ooc_markers(plot_data, data, ooc_indices, point_rules=None):
    """Add red markers for OOC points and customdata to main trace for click-to-inspect."""
    import numpy as np
    n = len(data) if hasattr(data, '__len__') else 0
    ooc_set = set(ooc_indices) if ooc_indices else set()

    # Tag the first (main data) trace with customdata so every point is clickable
    if plot_data and n > 0:
        main_trace = plot_data[0]
        if "customdata" not in main_trace:
            main_trace["customdata"] = [
                [i, "; ".join(point_rules.get(i, [])) if point_rules and i in ooc_set else ""]
                for i in range(n)
            ]
            main_trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>"

    if not ooc_indices:
        return
    ooc_x = ooc_indices
    ooc_y = [float(data[i]) for i in ooc_indices]
    trace = {
        "type": "scatter", "x": ooc_x, "y": ooc_y,
        "mode": "markers", "name": "Out of Control",
        "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
        "showlegend": True
    }
    # Add customdata for click-to-inspect
    if point_rules is not None:
        trace["customdata"] = [[i, "; ".join(point_rules.get(i, []))] for i in ooc_indices]
        trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<br>%{customdata[1]}<extra>OOC</extra>"
    else:
        trace["customdata"] = [[i, ""] for i in ooc_indices]
        trace["hovertemplate"] = "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra>OOC</extra>"
    plot_data.append(trace)




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
        point_rules = _spc_build_point_rules(data, x_bar, ucl, lcl, ooc_indices)

        # I Chart with OOC markers
        i_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "lines+markers", "name": "Value", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_bar]*n, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(i_chart_data, data, ooc_indices, point_rules=point_rules)
        result["plots"].append({
            "title": "I Chart (Individuals)",
            "data": i_chart_data,
            "layout": {"height": 290, "showlegend": True,
                        "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
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
            "layout": {"height": 290,
                        "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        ooc = len(ooc_indices)
        violations_text = ""
        if rule_violations:
            violations_text = "\n\nNelson Rule Violations:\n" + "\n".join(f"  {v}" for v in rule_violations)
        result["summary"] = f"I-MR Chart Analysis\n\nMean: {x_bar:.4f}\nUCL: {ucl:.4f}\nLCL: {lcl:.4f}\nMR-bar: {mr_bar:.4f}\n\nOut-of-control points: {ooc}{violations_text}"

        result["guide_observation"] = f"Control chart shows {ooc} out-of-control points." + (" Process appears stable." if ooc == 0 else " Investigation recommended.")

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
            _cc_verdict, _cc_body,
            next_steps=_cc_next,
            chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift. Two of three points beyond 2\u03c3 suggest a trend."
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

    elif analysis_id == "capability":
        from scipy import stats as sp_stats

        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None
        target = float(config.get("target")) if config.get("target") else None

        n = len(data)
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))

        # ── Capability indices ──────────────────────────────────────
        cp = cpk = pp = ppk = cpm = None
        ppm_below = ppm_above = ppm_total = 0.0
        yield_pct = 100.0
        sigma_level = 0.0

        if lsl is not None and usl is not None and std > 0:
            # Pp/Ppk use overall std (long-term)
            pp = (usl - lsl) / (6 * std)
            ppk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            # Cp/Cpk use within-subgroup sigma (MR-bar/d2 for individuals)
            mr = np.abs(np.diff(data))
            mr_bar = np.mean(mr) if len(mr) > 0 else std
            d2 = 1.128  # d2 constant for n=2 (moving range of 2)
            sigma_within = mr_bar / d2 if mr_bar > 0 else std
            cp = (usl - lsl) / (6 * sigma_within)
            cpk = min((usl - mean) / (3 * sigma_within), (mean - lsl) / (3 * sigma_within))
            if target is not None:
                cpm = (usl - lsl) / (6 * np.sqrt(std**2 + (mean - target)**2))

            # Expected defects
            z_lower = (mean - lsl) / std
            z_upper = (usl - mean) / std
            ppm_below = float(sp_stats.norm.cdf(-z_lower) * 1e6)
            ppm_above = float(sp_stats.norm.cdf(-z_upper) * 1e6)
            ppm_total = ppm_below + ppm_above
            yield_pct = (1 - ppm_total / 1e6) * 100
            sigma_level = float(sp_stats.norm.ppf(1 - ppm_total / 1e6) + 1.5) if ppm_total < 1e6 else 0

        elif lsl is not None and std > 0:
            cpk = (mean - lsl) / (3 * std)
            z_lower = (mean - lsl) / std
            ppm_below = float(sp_stats.norm.cdf(-z_lower) * 1e6)
            ppm_total = ppm_below
            yield_pct = (1 - ppm_total / 1e6) * 100

        elif usl is not None and std > 0:
            cpk = (usl - mean) / (3 * std)
            z_upper = (usl - mean) / std
            ppm_above = float(sp_stats.norm.cdf(-z_upper) * 1e6)
            ppm_total = ppm_above
            yield_pct = (1 - ppm_total / 1e6) * 100

        # ── Summary text ────────────────────────────────────────────
        summary = f"Capability Analysis  (n = {n})\n\n"
        summary += f"  Mean:      {mean:.4f}\n"
        summary += f"  Std Dev:   {std:.4f}\n"
        summary += f"  Min:       {float(np.min(data)):.4f}\n"
        summary += f"  Max:       {float(np.max(data)):.4f}\n"

        if cp is not None:
            summary += f"\nCapability Indices:\n"
            summary += f"  Cp:   {cp:.3f}     Pp:   {pp:.3f}\n"
            summary += f"  Cpk:  {cpk:.3f}     Ppk:  {ppk:.3f}\n"
            if cpm is not None:
                summary += f"  Cpm:  {cpm:.3f}   (Taguchi, target = {target})\n"
        elif cpk is not None:
            summary += f"\nCapability (one-sided):\n"
            summary += f"  Cpk:  {cpk:.3f}\n"

        if lsl is not None or usl is not None:
            summary += f"\nExpected Performance:\n"
            if lsl is not None:
                summary += f"  PPM < LSL:  {ppm_below:,.0f}\n"
            if usl is not None:
                summary += f"  PPM > USL:  {ppm_above:,.0f}\n"
            summary += f"  PPM Total:  {ppm_total:,.0f}\n"
            summary += f"  Yield:      {yield_pct:.4f}%\n"
            if cp is not None:
                summary += f"  Sigma:      {sigma_level:.2f}\n"

        if cpk is not None:
            if cpk >= 1.33:
                summary += f"\nProcess is capable (Cpk = {cpk:.3f} >= 1.33)"
            elif cpk >= 1.0:
                summary += f"\nProcess is marginally capable (1.0 <= Cpk = {cpk:.3f} < 1.33)"
            else:
                summary += f"\nProcess is NOT capable (Cpk = {cpk:.3f} < 1.0)"
            result["guide_observation"] = f"Process capability Cpk = {cpk:.2f}. " + ("Capable." if cpk >= 1.33 else "Needs improvement.")

        # ── Plot 1: Histogram with normal curve ─────────────────────
        x_range = np.linspace(float(np.min(data)) - 2 * std, float(np.max(data)) + 2 * std, 300)
        pdf_vals = sp_stats.norm.pdf(x_range, mean, std)

        hist_traces = [
            {
                "type": "histogram", "x": data.tolist(), "name": "Observed",
                "histnorm": "probability density",
                "marker": {"color": "rgba(74, 159, 110, 0.35)", "line": {"color": "#4a9f6e", "width": 1}},
            },
            {
                "type": "scatter", "x": x_range.tolist(), "y": pdf_vals.tolist(),
                "mode": "lines", "name": "Normal Fit",
                "line": {"color": "#4a90d9", "width": 2.5},
            },
        ]

        shapes_h = []
        annotations_h = []

        # Mean line
        shapes_h.append({
            "type": "line", "x0": mean, "x1": mean, "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "#00b894", "width": 2},
        })
        annotations_h.append({
            "x": mean, "y": 1.06, "yref": "paper", "text": "Mean",
            "showarrow": False, "font": {"color": "#00b894", "size": 10},
        })

        # Target line
        if target is not None:
            shapes_h.append({
                "type": "line", "x0": target, "x1": target, "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
            })
            annotations_h.append({
                "x": target, "y": 1.06, "yref": "paper", "text": "Target",
                "showarrow": False, "font": {"color": "#e8c547", "size": 10},
            })

        # LSL / USL lines
        if lsl is not None:
            shapes_h.append({
                "type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            })
            annotations_h.append({
                "x": lsl, "y": 1.06, "yref": "paper", "text": "LSL",
                "showarrow": False, "font": {"color": "#e85747", "size": 11},
            })
        if usl is not None:
            shapes_h.append({
                "type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            })
            annotations_h.append({
                "x": usl, "y": 1.06, "yref": "paper", "text": "USL",
                "showarrow": False, "font": {"color": "#e85747", "size": 11},
            })

        result["plots"].append({
            "title": "Capability Histogram",
            "data": hist_traces,
            "layout": {
                "height": 320,
                "shapes": shapes_h, "annotations": annotations_h,
                "showlegend": True,
                "legend": {"x": 1, "xanchor": "right", "y": 1, "bgcolor": "rgba(0,0,0,0)"},
                "margin": {"t": 40, "r": 20},
                "xaxis": {"title": measurement},
                "yaxis": {"title": "Density"},
            },
        })

        # ── Plot 2: Process spread vs specs ─────────────────────────
        if lsl is not None and usl is not None:
            spread_lo = mean - 3 * std
            spread_hi = mean + 3 * std
            pad = (usl - lsl) * 0.15

            spread_traces = [
                # Spec range
                {
                    "type": "bar", "y": [""], "x": [usl - lsl], "base": [lsl],
                    "orientation": "h", "name": "Spec Range",
                    "marker": {"color": "rgba(232, 87, 71, 0.15)", "line": {"color": "#e85747", "width": 1.5}},
                    "width": [0.5],
                },
                # Process spread (±3σ)
                {
                    "type": "bar", "y": [""], "x": [spread_hi - spread_lo], "base": [spread_lo],
                    "orientation": "h", "name": "Process ±3\u03c3",
                    "marker": {"color": "rgba(74, 159, 110, 0.25)", "line": {"color": "#4a9f6e", "width": 1.5}},
                    "width": [0.3],
                },
            ]

            spread_shapes = []
            spread_annot = []

            # Mean marker
            spread_shapes.append({
                "type": "line", "x0": mean, "x1": mean, "y0": -0.3, "y1": 0.3,
                "line": {"color": "#00b894", "width": 2.5},
            })
            spread_annot.append({
                "x": mean, "y": 0.35, "text": f"\u03bc={mean:.2f}",
                "showarrow": False, "font": {"color": "#00b894", "size": 10},
            })

            # LSL / USL labels on the bar
            spread_annot.append({
                "x": lsl, "y": -0.35, "text": f"LSL={lsl}",
                "showarrow": False, "font": {"color": "#e85747", "size": 10},
            })
            spread_annot.append({
                "x": usl, "y": -0.35, "text": f"USL={usl}",
                "showarrow": False, "font": {"color": "#e85747", "size": 10},
            })

            # ±3σ labels
            spread_annot.append({
                "x": spread_lo, "y": 0.35, "text": f"-3\u03c3",
                "showarrow": False, "font": {"color": "#4a9f6e", "size": 9},
            })
            spread_annot.append({
                "x": spread_hi, "y": 0.35, "text": f"+3\u03c3",
                "showarrow": False, "font": {"color": "#4a9f6e", "size": 9},
            })

            # Target marker
            if target is not None:
                spread_shapes.append({
                    "type": "line", "x0": target, "x1": target, "y0": -0.3, "y1": 0.3,
                    "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
                })
                spread_annot.append({
                    "x": target, "y": -0.35, "text": f"T={target}",
                    "showarrow": False, "font": {"color": "#e8c547", "size": 10},
                })

            result["plots"].append({
                "title": "Process Spread vs Specification",
                "data": spread_traces,
                "layout": {
                    "height": 180, "barmode": "overlay",
                    "shapes": spread_shapes, "annotations": spread_annot,
                    "showlegend": True,
                    "legend": {"x": 1, "xanchor": "right", "y": 1, "bgcolor": "rgba(0,0,0,0)"},
                    "xaxis": {"range": [min(lsl, spread_lo) - pad, max(usl, spread_hi) + pad], "title": measurement},
                    "yaxis": {"visible": False, "range": [-0.5, 0.5]},
                    "margin": {"t": 35, "b": 45, "l": 20, "r": 20},
                },
            })

        # ── Plot 3: Normal probability plot (Q-Q) ──────────────────
        sorted_data = np.sort(data)
        n_pts = len(sorted_data)
        probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
        theoretical_q = sp_stats.norm.ppf(probs, mean, std)

        # Shapiro-Wilk test (limited to 5000 samples)
        sw_data = data[:5000] if n > 5000 else data
        sw_stat, sw_p = sp_stats.shapiro(sw_data)
        normality_note = f"Shapiro-Wilk p = {sw_p:.4f}" + (" (normal)" if sw_p >= 0.05 else " (non-normal)")

        result["plots"].append({
            "title": f"Normal Probability Plot  ({normality_note})",
            "data": [
                {
                    "type": "scatter", "x": theoretical_q.tolist(), "y": sorted_data.tolist(),
                    "mode": "markers", "name": "Data",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.7},
                },
                {
                    "type": "scatter",
                    "x": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "y": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "mode": "lines", "name": "Reference",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Observed"},
                "showlegend": False,
                "margin": {"t": 35},
            },
        })

        result["summary"] = summary

        # What-If data for client-side interactive exploration
        result["what_if_data"] = {
            "type": "capability",
            "mean": float(mean),
            "std": float(std),
            "n": int(n),
            "current_lsl": float(lsl) if lsl else None,
            "current_usl": float(usl) if usl else None,
            "data_values": data.tolist() if n <= 5000 else data[:5000].tolist(),
        }

        # Narrative
        if cpk is not None:
            _tol_pct = ((6 * std) / (usl - lsl) * 100) if lsl is not None and usl is not None and (usl - lsl) > 0 else None
            if cpk >= 1.33:
                _cap_verdict = f"Process is capable (Cpk = {cpk:.3f})"
                _cap_next = "Process is capable. Monitor with control charts to maintain."
            elif cpk >= 1.0:
                _centering = "centering (adjust mean)" if cp is not None and cp > cpk + 0.1 else "spread (reduce variation)"
                _cap_verdict = f"Process is marginally capable (Cpk = {cpk:.3f})"
                _cap_next = f"Process is marginal. The dominant issue is {_centering}."
            else:
                _cap_verdict = f"Process is NOT capable (Cpk = {cpk:.3f})"
                _cap_next = "Process is not capable. Run a Gage R&R to confirm measurement isn't inflating variation, then investigate root causes with multi-vari or DOE."
            _cap_body = f"Cpk = {cpk:.3f}"
            if _tol_pct is not None:
                _cap_body += f" &mdash; the process uses {_tol_pct:.0f}% of the tolerance"
            _cap_body += f". Estimated <strong>{ppm_total:,.0f}</strong> defects per million ({yield_pct:.2f}% yield)."
            if cp is not None and ppk is not None and cpk < ppk - 0.05:
                _cap_body += " Short-term capability exceeds long-term &mdash; the process has shifts or drifts not captured in subgroups."
            result["narrative"] = _narrative(
                _cap_verdict, _cap_body,
                next_steps=_cap_next,
                chart_guidance="The histogram shows data vs spec limits (red lines). The normal curve is the fitted distribution. Data outside the spec lines are predicted defects."
            )

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
        xbar_point_rules = _spc_build_point_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl, xbar_ooc)
        # Nelson rules for R
        r_ooc, r_violations = _spc_nelson_rules(ranges, r_bar, r_ucl, r_lcl)
        r_point_rules = _spc_build_point_rules(ranges, r_bar, r_ucl, r_lcl, r_ooc)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc, point_rules=xbar_point_rules)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"height": 290, "showlegend": True, "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        # R Chart with OOC markers
        r_chart_data = [
            {"type": "scatter", "y": ranges.tolist(), "mode": "lines+markers", "name": "R", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [r_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [r_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [r_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(r_chart_data, ranges, r_ooc, point_rules=r_point_rules)
        result["plots"].append({
            "title": "R Chart",
            "data": r_chart_data,
            "layout": {"height": 290, "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        violations_text = ""
        if xbar_violations or r_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in r_violations: violations_text += f"\n  R: {v}"

        result["summary"] = f"Xbar-R Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nR Chart:\n  R̄: {r_bar:.4f}\n  UCL: {r_ucl:.4f}\n  LCL: {r_lcl:.4f}\n  OOC points: {len(r_ooc)}{violations_text}"

        _xr_ooc = len(xbar_ooc) + len(r_ooc)
        if _xr_ooc == 0:
            result["narrative"] = _narrative("Process is in statistical control", f"No out-of-control points in either the X\u0304 or R chart across {n_subgroups} subgroups. Process is stable and predictable.",
                next_steps="Process is stable \u2014 capability analysis is valid.", chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift.")
        else:
            _xr_rules = (xbar_violations + r_violations)[:3]
            result["narrative"] = _narrative(f"Process is out of control \u2014 {_xr_ooc} signal{'s' if _xr_ooc > 1 else ''} detected",
                f"X\u0304 chart: {len(xbar_ooc)} OOC points. R chart: {len(r_ooc)} OOC points." + (f" Violations: {'; '.join(_xr_rules)}." if _xr_rules else ""),
                next_steps="Investigate special causes at flagged points. Check timestamps against process logs.", chart_guidance="Points above UCL or below LCL are out-of-control signals. Runs of 7+ on one side of center suggest a shift.")

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
        xbar_point_rules = _spc_build_point_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl, xbar_ooc)
        s_ooc, s_violations = _spc_nelson_rules(stds, s_bar, s_ucl, s_lcl)
        s_point_rules = _spc_build_point_rules(stds, s_bar, s_ucl, s_lcl, s_ooc)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc, point_rules=xbar_point_rules)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"height": 290, "showlegend": True, "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        # S Chart with OOC markers
        s_chart_data = [
            {"type": "scatter", "y": stds.tolist(), "mode": "lines+markers", "name": "S", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [s_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [s_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [s_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(s_chart_data, stds, s_ooc, point_rules=s_point_rules)
        result["plots"].append({
            "title": "S Chart",
            "data": s_chart_data,
            "layout": {"height": 290, "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        violations_text = ""
        if xbar_violations or s_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in s_violations: violations_text += f"\n  S: {v}"

        result["summary"] = f"Xbar-S Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nS Chart:\n  S̄: {s_bar:.4f}\n  UCL: {s_ucl:.4f}\n  LCL: {s_lcl:.4f}\n  OOC points: {len(s_ooc)}{violations_text}"

        _xs_ooc = len(xbar_ooc) + len(s_ooc)
        if _xs_ooc == 0:
            result["narrative"] = _narrative("Process is in statistical control", f"No out-of-control points across {n_subgroups} subgroups.",
                next_steps="Process is stable \u2014 capability analysis is valid.", chart_guidance="Points above UCL or below LCL are out-of-control signals.")
        else:
            _xs_rules = (xbar_violations + s_violations)[:3]
            result["narrative"] = _narrative(f"Process is out of control \u2014 {_xs_ooc} signal{'s' if _xs_ooc > 1 else ''} detected",
                f"X\u0304 chart: {len(xbar_ooc)} OOC. S chart: {len(s_ooc)} OOC." + (f" {'; '.join(_xs_rules)}." if _xs_rules else ""),
                next_steps="Investigate special causes at flagged points.", chart_guidance="Points above UCL or below LCL are out-of-control signals.")

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
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(p_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "P Chart (Proportion Defective)",
            "data": p_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Proportion"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        result["summary"] = f"P Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}"
        result["guide_observation"] = f"P chart: {len(ooc_indices)} out-of-control points. p\u0304 = {p_bar*100:.2f}%." + (" Process is stable." if len(ooc_indices) == 0 else " Investigation recommended.")

        n_ooc = len(ooc_indices)
        if n_ooc == 0:
            verdict = f"P Chart — Process in control (p\u0304 = {p_bar*100:.2f}%)"
            body = f"All {k} samples fall within control limits. The average defective rate is {p_bar*100:.2f}%."
            nxt = "Monitor ongoing. If p\u0304 is too high, investigate systemic causes rather than individual points."
        else:
            verdict = f"P Chart — {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
            body = (f"{n_ooc} of {k} samples ({n_ooc/k*100:.1f}%) exceed control limits. "
                    f"Average defective rate p\u0304 = {p_bar*100:.2f}%. Investigate these subgroups for assignable causes.")
            nxt = "Identify what changed during OOC subgroups (material, operator, machine). Address root causes before tightening limits."
        result["narrative"] = _narrative(verdict, body, next_steps=nxt,
            chart_guidance="Points outside the dashed red limits are out of control. Variable limits reflect differing sample sizes.")

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
        np_point_rules = _spc_build_point_rules(d, np_bar, ucl, lcl, np_ooc)

        np_chart_data = [
            {"type": "scatter", "y": d.tolist(), "mode": "lines+markers", "name": "np", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [np_bar]*k, "mode": "lines", "name": "n̄p", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(np_chart_data, d, np_ooc, point_rules=np_point_rules)
        result["plots"].append({
            "title": "NP Chart (Number Defective)",
            "data": np_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Number Defective"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        violations_text = ""
        if np_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in np_violations: violations_text += f"\n  {v}"

        result["summary"] = f"NP Chart Analysis\n\nn̄p: {np_bar:.2f}\nSample size: {n}\np̄: {p_bar:.4f}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(np_ooc)}{violations_text}"

        n_ooc = len(np_ooc)
        viol_note = f" Nelson rule violations: {', '.join(np_violations[:3])}." if np_violations else ""
        if n_ooc == 0:
            verdict = f"NP Chart — Process in control (n\u0304p = {np_bar:.1f})"
            body = f"All {k} samples within limits. Average {np_bar:.1f} defectives per sample of {n}.{viol_note}"
            nxt = "Continue monitoring. To reduce defective count, investigate the process, not individual samples."
        else:
            verdict = f"NP Chart — {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
            body = (f"{n_ooc} of {k} samples exceed limits. Average defectives n\u0304p = {np_bar:.1f} "
                    f"(p\u0304 = {p_bar*100:.2f}%).{viol_note}")
            nxt = "Investigate OOC subgroups for assignable causes. Check for material batches, shift changes, or equipment issues."
        result["narrative"] = _narrative(verdict, body, next_steps=nxt,
            chart_guidance="Points outside limits signal unusual defective counts. Runs or trends may indicate gradual shifts.")

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
        c_point_rules = _spc_build_point_rules(c, c_bar, ucl, lcl, c_ooc)

        c_chart_data = [
            {"type": "scatter", "y": c.tolist(), "mode": "lines+markers", "name": "c", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [c_bar]*k, "mode": "lines", "name": "c̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(c_chart_data, c, c_ooc, point_rules=c_point_rules)
        result["plots"].append({
            "title": "C Chart (Defects per Unit)",
            "data": c_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Defects"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        violations_text = ""
        if c_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in c_violations: violations_text += f"\n  {v}"

        result["summary"] = f"C Chart Analysis\n\nc̄: {c_bar:.2f}\nSamples: {k}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(c_ooc)}{violations_text}"

        n_ooc = len(c_ooc)
        viol_note = f" Nelson rule violations: {', '.join(c_violations[:3])}." if c_violations else ""
        if n_ooc == 0:
            verdict = f"C Chart — Process in control (c\u0304 = {c_bar:.1f})"
            body = f"All {k} samples within Poisson-based limits. Average defect count c\u0304 = {c_bar:.1f}.{viol_note}"
            nxt = "Stable process. To reduce defect count, apply Pareto analysis to identify top defect categories."
        else:
            verdict = f"C Chart — {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
            body = (f"{n_ooc} of {k} samples exceed limits. Average defects c\u0304 = {c_bar:.1f}.{viol_note} "
                    f"The process is unstable — address special causes before process improvement.")
            nxt = "Investigate OOC points chronologically. Look for environmental, material, or procedural changes."
        result["narrative"] = _narrative(verdict, body, next_steps=nxt,
            chart_guidance="Limits are Poisson-based (\u00b13\u221ac\u0304). Points outside = unusual defect counts for constant-opportunity inspection.")

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
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(u_chart_data, u, u_ooc_indices)
        result["plots"].append({
            "title": "U Chart (Defects per Unit)",
            "data": u_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Defects per Unit"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        result["summary"] = f"U Chart Analysis\n\nū: {u_bar:.4f}\nSamples: {k}\n\nOut-of-control points: {len(u_ooc_indices)}"

        n_ooc = len(u_ooc_indices)
        if n_ooc == 0:
            verdict = f"U Chart — Process in control (\u016b = {u_bar:.4f})"
            body = f"All {k} samples within variable control limits. Average defect rate \u016b = {u_bar:.4f} per unit."
            nxt = "Stable process. Variable limits account for differing inspection sizes — focus on reducing the overall rate."
        else:
            verdict = f"U Chart — {n_ooc} out-of-control point{'s' if n_ooc > 1 else ''}"
            body = (f"{n_ooc} of {k} samples ({n_ooc/k*100:.1f}%) exceed control limits. "
                    f"Average defect rate \u016b = {u_bar:.4f}. Investigate these subgroups for assignable causes.")
            nxt = "Identify what changed during OOC subgroups. Variable limits mean OOC points are truly unusual, not just from smaller samples."
        result["narrative"] = _narrative(verdict, body, next_steps=nxt,
            chart_guidance="Variable limits reflect differing inspection unit sizes. Points outside limits = genuinely unusual defect rates.")

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
            {"type": "scatter", "y": cusum_pos.tolist(), "mode": "lines", "name": "CUSUM+", "line": {"color": "#4a9f6e", "width": 2},
             "customdata": [[i, ""] for i in range(n)], "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM+: %{y:.4f}<extra></extra>"},
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
                "showlegend": True,
                "customdata": [[int(i), "Upward shift signal"] for i in signals_pos],
                "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM+: %{y:.4f}<br>%{customdata[1]}<extra>Signal (up)</extra>"
            })
        # OOC markers for negative signals
        if len(signals_neg) > 0:
            cusum_chart_data.append({
                "type": "scatter", "x": signals_neg.tolist(), "y": (-cusum_neg[signals_neg]).tolist(),
                "mode": "markers", "name": "Signal (down)",
                "marker": {"color": "#e89547", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                "showlegend": True,
                "customdata": [[int(i), "Downward shift signal"] for i in signals_neg],
                "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM-: %{y:.4f}<br>%{customdata[1]}<extra>Signal (down)</extra>"
            })
        result["plots"].append({
            "title": "CUSUM Chart",
            "data": cusum_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "CUSUM"},
                        "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        result["summary"] = f"CUSUM Chart Analysis\n\nTarget: {target:.4f}\nσ estimate: {sigma:.4f}\nk (slack): {k_param}\nh (decision): {h_param}\n\nUpward shift signals: {len(signals_pos)} at points {list(signals_pos[:5])}{'...' if len(signals_pos) > 5 else ''}\nDownward shift signals: {len(signals_neg)} at points {list(signals_neg[:5])}{'...' if len(signals_neg) > 5 else ''}"

        _n_up = len(signals_pos)
        _n_dn = len(signals_neg)
        _cusum_total = _n_up + _n_dn
        result["guide_observation"] = (f"CUSUM chart: {_cusum_total} shift signal{'s' if _cusum_total != 1 else ''}."
                                       + (" Process appears stable." if _cusum_total == 0 else " Investigation recommended."))
        if _cusum_total == 0:
            result["narrative"] = _narrative(
                "Process is in statistical control", f"No cumulative shift signals detected (k={k_param}, h={h_param}). Process mean is stable around target ({target:.4f}).",
                next_steps="Process is stable \u2014 CUSUM is sensitive to small sustained shifts; absence of signals is strong evidence of stability.",
                chart_guidance="CUSUM+ (green) tracks upward drift; CUSUM\u2212 (blue) tracks downward drift. Crossing the red decision interval (h) signals a shift.")
        else:
            _shift_desc = []
            if _n_up > 0: _shift_desc.append(f"{_n_up} upward")
            if _n_dn > 0: _shift_desc.append(f"{_n_dn} downward")
            result["narrative"] = _narrative(
                f"CUSUM signals detected \u2014 {' and '.join(_shift_desc)} shift{'s' if _cusum_total > 1 else ''}",
                f"The cumulative sum crossed the decision interval (h={h_param}), indicating a sustained shift from target ({target:.4f}).",
                next_steps="Identify when the shift began (first signal point) and investigate process changes at that time.",
                chart_guidance="CUSUM+ (green) tracks upward drift; CUSUM\u2212 (blue) tracks downward drift. Diamond markers show where the decision interval was breached.")

        _cusum_ooc = sorted(set(list(signals_pos) + list(signals_neg)))
        if _cusum_ooc:
            result["what_if_data"] = {
                "type": "spc_intervention",
                "values": data.tolist(),
                "center": float(target),
                "ucl": float(target + h_param * sigma),
                "lcl": float(target - h_param * sigma),
                "sigma": float(sigma),
                "ooc_indices": [int(i) for i in _cusum_ooc],
                "first_ooc": int(min(_cusum_ooc)),
                "cusum_pos": cusum_pos.tolist(),
                "cusum_neg": cusum_neg.tolist(),
                "h_param": float(h_param),
                "k_param": float(k_param),
                "target": float(target),
            }

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
            {"type": "scatter", "y": ewma.tolist(), "mode": "lines+markers", "name": "EWMA", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}, "line": {"color": "#4a9f6e"}},
            {"type": "scatter", "y": [target]*n, "mode": "lines", "name": "Target", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(ewma_chart_data, ewma, ewma_ooc)
        result["plots"].append({
            "title": "EWMA Chart",
            "data": ewma_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "EWMA"},
                        "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        result["summary"] = f"EWMA Chart Analysis\n\nTarget: {target:.4f}\nλ (smoothing): {lambda_param}\nL (sigma width): {L}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nOut-of-control points: {len(ewma_ooc)}"

        _ewma_n_ooc = len(ewma_ooc)
        result["guide_observation"] = (f"EWMA chart: {_ewma_n_ooc} out-of-control point{'s' if _ewma_n_ooc != 1 else ''}."
                                       + (" Process appears stable." if _ewma_n_ooc == 0 else " Investigation recommended."))
        if _ewma_n_ooc == 0:
            result["narrative"] = _narrative(
                "Process is in statistical control", f"No EWMA points exceed control limits (\u03bb={lambda_param}, L={L}). The smoothed process mean is stable around target ({target:.4f}).",
                next_steps="Process is stable \u2014 EWMA is sensitive to small sustained shifts; absence of signals is strong evidence of stability.",
                chart_guidance="The EWMA line smooths out noise to reveal underlying trends. Limits widen from zero to steady state as the filter initialises.")
        else:
            result["narrative"] = _narrative(
                f"EWMA signals detected \u2014 {_ewma_n_ooc} point{'s' if _ewma_n_ooc > 1 else ''} out of control",
                f"The smoothed mean has drifted outside the \u00b1{L}\u03c3 control limits, indicating a sustained shift from target ({target:.4f}).",
                next_steps="Identify when the drift began (first OOC point) and correlate with process changes. EWMA detects gradual shifts that Shewhart charts miss.",
                chart_guidance="The EWMA line smooths observations \u2014 OOC points (red diamonds) indicate the smoothed mean has shifted. Variable limits reflect the filter warm-up period.")

        if ewma_ooc:
            result["what_if_data"] = {
                "type": "spc_intervention",
                "values": data.tolist(),
                "center": float(target),
                "ucl": float(ucl_ss),
                "lcl": float(lcl_ss),
                "sigma": float(sigma),
                "ooc_indices": [int(i) for i in ewma_ooc],
                "first_ooc": int(min(ewma_ooc)),
                "ewma": ewma.tolist(),
                "lambda_param": float(lambda_param),
                "target": float(target),
            }

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
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lp_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "Laney P' Chart",
            "data": lp_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Proportion"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney P' Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard P chart would give too many false alarms."

        _lp_n_ooc = len(ooc_indices)
        result["guide_observation"] = f"Laney P' chart: {_lp_n_ooc} out-of-control points. \u03c3z = {sigma_z:.3f} ({disp})." + (" Process is stable." if _lp_n_ooc == 0 else " Investigation recommended.")
        if _lp_n_ooc == 0:
            result["narrative"] = _narrative(
                f"Laney P' \u2014 Process in control (p\u0304 = {p_bar*100:.2f}%, \u03c3z = {sigma_z:.3f})",
                f"All {k} samples within overdispersion-adjusted limits. {disp} detected (\u03c3z = {sigma_z:.3f})." + (" Standard P chart limits would be too tight." if sigma_z > 1 else ""),
                next_steps="Stable process. Laney adjustment accounts for extra-binomial variation that inflates false alarms on standard P charts.",
                chart_guidance="The adjusted limits (P') are wider than standard P chart limits when \u03c3z > 1, reducing false alarms.")
        else:
            result["narrative"] = _narrative(
                f"Laney P' \u2014 {_lp_n_ooc} out-of-control point{'s' if _lp_n_ooc > 1 else ''}",
                f"{_lp_n_ooc} of {k} samples exceed the overdispersion-adjusted limits (\u03c3z = {sigma_z:.3f}). These are genuine signals even after accounting for {disp.lower()}.",
                next_steps="Investigate OOC subgroups. Because Laney P' already accounts for overdispersion, these signals are more trustworthy than standard P chart flags.",
                chart_guidance="Laney-adjusted limits are wider when overdispersion exists (\u03c3z > 1). Points still outside these wider limits are strong signals.")

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
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lu_chart_data, u, ooc_indices)
        result["plots"].append({
            "title": "Laney U' Chart",
            "data": lu_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": "Defects per Unit"}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney U' Chart Analysis\n\nū: {u_bar:.4f}\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard U chart would give too many false alarms."

        _lu_n_ooc = len(ooc_indices)
        result["guide_observation"] = f"Laney U' chart: {_lu_n_ooc} out-of-control points. \u03c3z = {sigma_z:.3f} ({disp})." + (" Process is stable." if _lu_n_ooc == 0 else " Investigation recommended.")
        if _lu_n_ooc == 0:
            result["narrative"] = _narrative(
                f"Laney U' \u2014 Process in control (\u016b = {u_bar:.4f}, \u03c3z = {sigma_z:.3f})",
                f"All {k} samples within overdispersion-adjusted limits. {disp} detected (\u03c3z = {sigma_z:.3f}).",
                next_steps="Stable process. Laney adjustment accounts for extra-Poisson variation.",
                chart_guidance="Adjusted limits are wider than standard U chart when \u03c3z > 1.")
        else:
            result["narrative"] = _narrative(
                f"Laney U' \u2014 {_lu_n_ooc} out-of-control point{'s' if _lu_n_ooc > 1 else ''}",
                f"{_lu_n_ooc} of {k} samples exceed overdispersion-adjusted limits (\u03c3z = {sigma_z:.3f}). These are genuine signals even after accounting for {disp.lower()}.",
                next_steps="Investigate OOC subgroups. Laney-adjusted signals are more trustworthy than standard U chart flags.",
                chart_guidance="Points outside the wider Laney limits are strong signals of genuine process change.")

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

        from scipy import stats as sp_stats

        # ---- Plot 1: Xbar Chart (subgroup means with control limits) ----
        x_idx = list(range(1, k + 1))
        x_labels = [str(i) for i in x_idx]
        if subgroup_col:
            sg_keys = list(df.groupby(subgroup_col).groups.keys())
            x_labels = [str(s) for s in sg_keys[:k]]

        # Control limits for Xbar using sigma_between + sigma_within
        ucl_xbar = grand_mean + 3 * sigma_bw / np.sqrt(n_avg)
        lcl_xbar = grand_mean - 3 * sigma_bw / np.sqrt(n_avg)

        # Flag out-of-control points
        ooc_xbar = [i for i, m in enumerate(group_means) if m > ucl_xbar or m < lcl_xbar]

        xbar_traces = [
            {"type": "scatter", "x": x_idx, "y": group_means.tolist(), "mode": "lines+markers",
             "name": "Subgroup Mean", "marker": {"color": "#4a90d9", "size": 6},
             "line": {"color": "#4a90d9", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [grand_mean] * k, "mode": "lines",
             "name": f"X̄ = {grand_mean:.4f}", "line": {"color": "#4a9f6e", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [ucl_xbar] * k, "mode": "lines",
             "name": f"UCL = {ucl_xbar:.4f}", "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [lcl_xbar] * k, "mode": "lines",
             "name": f"LCL = {lcl_xbar:.4f}", "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5}},
        ]
        if ooc_xbar:
            xbar_traces.append({
                "type": "scatter", "x": [x_idx[i] for i in ooc_xbar],
                "y": [group_means[i] for i in ooc_xbar], "mode": "markers",
                "name": "Out of Control", "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"},
            })

        result["plots"].append({
            "title": "X̄ Chart — Subgroup Means",
            "data": xbar_traces,
            "layout": {"height": 320,
                        "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}, "yaxis": {"title": measurement},
                        "showlegend": True,
                        "legend": {"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center",
                                   "font": {"size": 9, "color": "#b0b0b0"},
                                   "bgcolor": "rgba(0,0,0,0)"}},
            "group": "Control Charts",
        })

        # ---- Plot 2: R Chart (within-subgroup ranges) ----
        group_ranges = np.array([np.max(g) - np.min(g) for g in groups])
        r_bar = np.mean(group_ranges)
        # d3/D3/D4 constants for subgroup size (approximation for variable n)
        n_int = int(round(n_avg))
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d3_table = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}
        d2 = d2_table.get(n_int, 2.326)
        d3 = d3_table.get(n_int, 0.864)
        D3 = max(0, 1 - 3 * d3 / d2)
        D4 = 1 + 3 * d3 / d2
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar

        ooc_r = [i for i, r in enumerate(group_ranges) if r > ucl_r or r < lcl_r]

        r_traces = [
            {"type": "scatter", "x": x_idx, "y": group_ranges.tolist(), "mode": "lines+markers",
             "name": "Range", "marker": {"color": "#e89547", "size": 6},
             "line": {"color": "#e89547", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [r_bar] * k, "mode": "lines",
             "name": f"R̄ = {r_bar:.4f}", "line": {"color": "#4a9f6e", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [ucl_r] * k, "mode": "lines",
             "name": f"UCL = {ucl_r:.4f}", "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5}},
            {"type": "scatter", "x": x_idx, "y": [lcl_r] * k, "mode": "lines",
             "name": f"LCL = {lcl_r:.4f}", "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5}},
        ]
        if ooc_r:
            r_traces.append({
                "type": "scatter", "x": [x_idx[i] for i in ooc_r],
                "y": [group_ranges[i] for i in ooc_r], "mode": "markers",
                "name": "Out of Control", "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"},
            })

        result["plots"].append({
            "title": "R Chart — Within-Subgroup Ranges",
            "data": r_traces,
            "layout": {"height": 320,
                        "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}, "yaxis": {"title": "Range"},
                        "showlegend": True,
                        "legend": {"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center",
                                   "font": {"size": 9, "color": "#b0b0b0"},
                                   "bgcolor": "rgba(0,0,0,0)"}},
            "group": "Control Charts",
        })

        # ---- Plot 3: Individual Values by Subgroup (box + strip) ----
        box_traces = []
        for i, g in enumerate(groups):
            label = x_labels[i] if i < len(x_labels) else str(i + 1)
            box_traces.append({
                "type": "box", "y": g.tolist(), "name": label,
                "boxpoints": "all", "jitter": 0.4, "pointpos": 0,
                "marker": {"color": "#4a90d9", "size": 3, "opacity": 0.6},
                "line": {"color": "#4a90d9", "width": 1},
                "fillcolor": "rgba(74, 144, 217, 0.15)",
            })

        box_layout = {"height": 300,
                      "xaxis": {"title": "Subgroup"}, "yaxis": {"title": measurement},
                      "showlegend": False, "shapes": [], "annotations": []}
        # Grand mean reference line
        box_layout["shapes"].append({
            "type": "line", "x0": -0.5, "x1": k - 0.5, "y0": grand_mean, "y1": grand_mean,
            "line": {"color": "#4a9f6e", "width": 1.5, "dash": "dash"}
        })
        box_layout["annotations"].append({
            "x": k - 0.5, "y": grand_mean, "text": f"X̄={grand_mean:.3f}",
            "showarrow": False, "xanchor": "left", "font": {"color": "#4a9f6e", "size": 10}
        })
        if lsl is not None:
            box_layout["shapes"].append({"type": "line", "x0": -0.5, "x1": k - 0.5, "y0": lsl, "y1": lsl,
                                         "line": {"color": "#e85747", "dash": "dot", "width": 1.5}})
            box_layout["annotations"].append({"x": k - 0.5, "y": lsl, "text": "LSL", "showarrow": False,
                                              "xanchor": "left", "font": {"color": "#e85747", "size": 10}})
        if usl is not None:
            box_layout["shapes"].append({"type": "line", "x0": -0.5, "x1": k - 0.5, "y0": usl, "y1": usl,
                                         "line": {"color": "#e85747", "dash": "dot", "width": 1.5}})
            box_layout["annotations"].append({"x": k - 0.5, "y": usl, "text": "USL", "showarrow": False,
                                              "xanchor": "left", "font": {"color": "#e85747", "size": 10}})

        # Cap at 30 subgroups for readability; summarize if more
        if k <= 30:
            result["plots"].append({
                "title": "Individual Values by Subgroup",
                "data": box_traces,
                "layout": box_layout,
                "group": "Control Charts",
            })
        else:
            # Show first 15 and last 15 for large datasets
            subset = box_traces[:15] + box_traces[-15:]
            box_layout["annotations"].insert(0, {
                "x": 0.5, "y": 1.08, "xref": "paper", "yref": "paper",
                "text": f"Showing 30 of {k} subgroups (first 15 + last 15)",
                "showarrow": False, "font": {"color": "rgba(255,255,255,0.5)", "size": 10}
            })
            result["plots"].append({
                "title": "Individual Values by Subgroup",
                "data": subset,
                "layout": box_layout,
                "group": "Control Charts",
            })

        # ---- Plot 4: Variance Components bar chart ----
        result["plots"].append({
            "title": "Variance Components (σ)",
            "data": [{
                "type": "bar",
                "x": ["Within", "Between", "B/W Combined", "Overall"],
                "y": [sigma_within, sigma_between, sigma_bw, sigma_total],
                "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a"]},
                "text": [f"{sigma_within:.4f}", f"{sigma_between:.4f}", f"{sigma_bw:.4f}", f"{sigma_total:.4f}"],
                "textposition": "outside", "textfont": {"color": "#b0b0b0"},
            }],
            "layout": {"height": 280, "yaxis": {"title": "Std Dev (σ)"}},
            "group": "Variance",
        })

        # ---- Plot 5: % Variance Contribution (donut) ----
        within_pct = sigma_within**2 / sigma_total**2 * 100 if sigma_total > 0 else 0
        between_pct = sigma_between**2 / sigma_total**2 * 100 if sigma_total > 0 else 0
        residual_pct = max(0, 100 - within_pct - between_pct)

        donut_labels = ["Within", "Between"]
        donut_vals = [within_pct, between_pct]
        donut_colors = ["#4a9f6e", "#4a90d9"]
        if residual_pct > 0.5:
            donut_labels.append("Residual")
            donut_vals.append(residual_pct)
            donut_colors.append("rgba(255,255,255,0.15)")

        result["plots"].append({
            "title": "% Contribution to Total Variance",
            "data": [{
                "type": "pie", "labels": donut_labels, "values": donut_vals,
                "hole": 0.45, "marker": {"colors": donut_colors, "line": {"color": "rgba(0,0,0,0.3)", "width": 1}},
                "textinfo": "label+percent", "textfont": {"size": 12, "color": "#e0e0e0"},
                "hoverinfo": "label+percent+value",
            }],
            "layout": {"height": 280, "showlegend": False,
                        "annotations": [{"text": "Variance<br>Split", "x": 0.5, "y": 0.5,
                                          "font": {"size": 13, "color": "rgba(255,255,255,0.6)"},
                                          "showarrow": False}]},
            "group": "Variance",
        })

        # ---- Plot 6: Within vs Overall Distribution (histogram + fits) ----
        x_range = np.linspace(min(data), max(data), 200)
        hist_data = [
            {"type": "histogram", "x": data.tolist(), "name": "Data",
             "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}},
             "histnorm": "probability density"},
            {"type": "scatter", "x": x_range.tolist(),
             "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_within).tolist(),
             "mode": "lines", "name": f"Within (σ={sigma_within:.3f})",
             "line": {"color": "#4a90d9", "width": 2}},
            {"type": "scatter", "x": x_range.tolist(),
             "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_bw).tolist(),
             "mode": "lines", "name": f"B/W (σ={sigma_bw:.3f})",
             "line": {"color": "#e89547", "width": 2, "dash": "dot"}},
            {"type": "scatter", "x": x_range.tolist(),
             "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_total).tolist(),
             "mode": "lines", "name": f"Overall (σ={sigma_total:.3f})",
             "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
        ]

        dist_layout = {"height": 300, "showlegend": True,
                       "shapes": [], "annotations": [],
                       "legend": {"font": {"size": 9, "color": "#b0b0b0"}, "x": 0.98, "xanchor": "right", "y": 0.98,
                                  "bgcolor": "rgba(20,20,30,0.7)", "bordercolor": "rgba(255,255,255,0.1)", "borderwidth": 1}}
        if lsl is not None:
            dist_layout["shapes"].append({"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper",
                                          "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            dist_layout["annotations"].append({"x": lsl, "y": 1.05, "yref": "paper", "text": "LSL",
                                               "showarrow": False, "font": {"color": "#e85747"}})
        if usl is not None:
            dist_layout["shapes"].append({"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper",
                                          "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            dist_layout["annotations"].append({"x": usl, "y": 1.05, "yref": "paper", "text": "USL",
                                               "showarrow": False, "font": {"color": "#e85747"}})

        result["plots"].append({
            "title": "Within vs B/W vs Overall Distribution",
            "data": hist_data,
            "layout": dist_layout,
            "group": "Capability",
        })

        # ---- Plot 7: Capability Index Comparison (when specs provided) ----
        if lsl is not None and usl is not None:
            cap_categories = ["Cp / Pp", "Cpk / Ppk"]
            result["plots"].append({
                "title": "Capability Index Comparison",
                "data": [
                    {"type": "bar", "name": "Within", "x": cap_categories,
                     "y": [cp_within, cpk_within],
                     "marker": {"color": "#4a9f6e"}, "text": [f"{cp_within:.3f}", f"{cpk_within:.3f}"],
                     "textposition": "outside", "textfont": {"color": "#b0b0b0"}},
                    {"type": "bar", "name": "Between/Within", "x": cap_categories,
                     "y": [cp_bw, cpk_bw],
                     "marker": {"color": "#e89547"}, "text": [f"{cp_bw:.3f}", f"{cpk_bw:.3f}"],
                     "textposition": "outside", "textfont": {"color": "#b0b0b0"}},
                    {"type": "bar", "name": "Overall", "x": cap_categories,
                     "y": [pp, ppk],
                     "marker": {"color": "#d94a4a"}, "text": [f"{pp:.3f}", f"{ppk:.3f}"],
                     "textposition": "outside", "textfont": {"color": "#b0b0b0"}},
                ],
                "layout": {"height": 300, "barmode": "group",
                            "yaxis": {"title": "Index Value"},
                            "legend": {"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center",
                                       "font": {"size": 10, "color": "#b0b0b0"}, "bgcolor": "rgba(0,0,0,0)"},
                            "shapes": [{"type": "line", "x0": -0.5, "x1": 1.5, "y0": 1.33, "y1": 1.33,
                                         "line": {"color": "rgba(74,159,110,0.5)", "dash": "dash", "width": 1.5}},
                                        {"type": "line", "x0": -0.5, "x1": 1.5, "y0": 1.0, "y1": 1.0,
                                         "line": {"color": "rgba(232,87,71,0.5)", "dash": "dot", "width": 1.5}}],
                            "annotations": [{"x": 1.5, "y": 1.33, "text": "Target (1.33)", "showarrow": False,
                                              "xanchor": "left", "font": {"color": "rgba(74,159,110,0.7)", "size": 10}},
                                             {"x": 1.5, "y": 1.0, "text": "Minimum (1.0)", "showarrow": False,
                                              "xanchor": "left", "font": {"color": "rgba(232,87,71,0.7)", "size": 10}}]},
                "group": "Capability",
            })

        # ---- Plot 8: Normal Probability Plot ----
        sorted_data = np.sort(data)
        n_pts = len(sorted_data)
        theoretical_q = sp_stats.norm.ppf((np.arange(1, n_pts + 1) - 0.375) / (n_pts + 0.25))

        result["plots"].append({
            "title": "Normal Probability Plot",
            "data": [
                {"type": "scatter", "x": theoretical_q.tolist(), "y": sorted_data.tolist(),
                 "mode": "markers", "name": "Data",
                 "marker": {"color": "#4a90d9", "size": 3, "opacity": 0.7}},
                {"type": "scatter",
                 "x": [theoretical_q[0], theoretical_q[-1]],
                 "y": [grand_mean + sigma_total * theoretical_q[0], grand_mean + sigma_total * theoretical_q[-1]],
                 "mode": "lines", "name": "Normal Fit",
                 "line": {"color": "#d94a4a", "width": 1.5}},
            ],
            "layout": {"height": 280,
                        "xaxis": {"title": "Theoretical Quantiles"},
                        "yaxis": {"title": measurement},
                        "showlegend": True,
                        "legend": {"font": {"size": 9, "color": "#b0b0b0"}, "x": 0.02, "y": 0.98,
                                   "bgcolor": "rgba(20,20,30,0.7)", "bordercolor": "rgba(255,255,255,0.1)", "borderwidth": 1}},
            "group": "Capability",
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

        layout = {"height": 300, "showlegend": True, "shapes": [], "annotations": []}
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
            "layout": {"height": 280, "xaxis": {"title": f"Theoretical ({best_name})"}, "yaxis": {"title": "Observed"}}
        })

        result["summary"] = summary

    elif analysis_id == "moving_average":
        """
        Moving Average (MA) Chart.
        Smooths individual observations with a moving window.
        Good for detecting sustained shifts when short-term noise is high.
        """
        measurement = config.get("measurement")
        if not measurement:
            measurement = df.select_dtypes(include=[np.number]).columns[0]
        span = int(config.get("span", 5))

        data = df[measurement].dropna().values
        n = len(data)

        x_bar = np.mean(data)
        sigma = np.std(data, ddof=1)

        # Moving averages
        ma = []
        for i in range(n):
            start = max(0, i - span + 1)
            window = data[start:i + 1]
            ma.append(np.mean(window))
        ma = np.array(ma)

        # Control limits for moving average (tighten as window fills)
        ucl_arr = []
        lcl_arr = []
        for i in range(n):
            w = min(i + 1, span)
            ucl_arr.append(x_bar + 3 * sigma / np.sqrt(w))
            lcl_arr.append(x_bar - 3 * sigma / np.sqrt(w))

        # OOC detection
        ma_ooc = [i for i in range(n) if ma[i] > ucl_arr[i] or ma[i] < lcl_arr[i]]

        ma_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "markers", "name": "Individual", "marker": {"size": 4, "color": "rgba(74,159,110,0.3)"}},
            {"type": "scatter", "y": ma.tolist(), "mode": "lines+markers", "name": f"MA({span})", "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e", "width": 2}},
            {"type": "scatter", "y": [x_bar] * n, "mode": "lines", "name": "CL", "line": {"color": "#00b894", "dash": "dash"}},
            {"type": "scatter", "y": ucl_arr, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl_arr, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(ma_chart_data, ma.tolist(), ma_ooc)
        result["plots"].append({
            "title": f"Moving Average Chart (span={span})",
            "data": ma_chart_data,
            "layout": {"height": 290, "showlegend": True, "yaxis": {"title": measurement}, "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        # Steady-state limits
        ucl_ss = x_bar + 3 * sigma / np.sqrt(span)
        lcl_ss = x_bar - 3 * sigma / np.sqrt(span)

        result["summary"] = f"Moving Average Chart\n\nSpan (window size): {span}\nCenter Line: {x_bar:.4f}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nSamples: {n}\nOut-of-control points: {len(ma_ooc)}\n\nThe MA chart smooths short-term noise. With span={span}, it is effective at detecting sustained shifts of {3/np.sqrt(span):.2f}\u03c3 or larger."

        _ma_n_ooc = len(ma_ooc)
        result["guide_observation"] = f"MA chart (span={span}): {_ma_n_ooc} out-of-control points." + (" Process appears stable." if _ma_n_ooc == 0 else " Investigation recommended.")
        if _ma_n_ooc == 0:
            result["narrative"] = _narrative(
                "Process is in statistical control", f"No out-of-control points on the moving average chart (span={span}). Process mean is stable.",
                next_steps="Process is stable. The MA chart smooths noise to reveal underlying shifts.",
                chart_guidance="Faded dots are individual values; the solid line is the moving average. Limits tighten as the window fills to span={span}.")
        else:
            result["narrative"] = _narrative(
                f"MA chart \u2014 {_ma_n_ooc} out-of-control point{'s' if _ma_n_ooc > 1 else ''} detected",
                f"The smoothed moving average (span={span}) exceeds control limits at {_ma_n_ooc} point{'s' if _ma_n_ooc > 1 else ''}, indicating a sustained shift.",
                next_steps="Identify when the shift began and correlate with process changes.",
                chart_guidance="Faded dots are individual values; the solid line is the moving average. OOC points on the smoothed line indicate sustained (not transient) shifts.")

    elif analysis_id == "zone_chart":
        """
        Zone Chart — assigns zone scores based on Western Electric zones.
        Signals when cumulative score reaches 8 (equivalent to a zone rule violation).
        Color-coded A/B/C zones for visual pattern detection.
        """
        measurement = config.get("measurement")
        if not measurement:
            measurement = df.select_dtypes(include=[np.number]).columns[0]

        data = df[measurement].dropna().values
        n = len(data)

        x_bar = np.mean(data)
        mr = np.abs(np.diff(data))
        sigma = np.mean(mr) / 1.128 if len(mr) > 0 else np.std(data, ddof=1)

        # Zone boundaries
        zone_1s = sigma
        zone_2s = 2 * sigma
        zone_3s = 3 * sigma

        # Zone scoring
        scores = []
        cum_scores = []
        cum_score = 0
        signals = []
        side = 0  # +1 above CL, -1 below CL

        for i in range(n):
            z = (data[i] - x_bar) / sigma if sigma > 0 else 0
            current_side = 1 if z >= 0 else -1

            # Zone score: A=8, B=4, C=2, center=0
            abs_z = abs(z)
            if abs_z >= 3:
                score = 8  # Beyond Zone A — instant signal
            elif abs_z >= 2:
                score = 4  # Zone A
            elif abs_z >= 1:
                score = 2  # Zone B
            else:
                score = 0  # Zone C (reset)
                cum_score = 0

            # Reset on side change
            if i > 0 and current_side != side:
                cum_score = 0
            side = current_side

            cum_score += score
            scores.append(score)
            cum_scores.append(cum_score)

            if cum_score >= 8:
                signals.append(i)
                cum_score = 0  # Reset after signal

        # Plot with zone bands
        zone_shapes = [
            # Zone C (green) - within 1 sigma
            {"type": "rect", "y0": x_bar - zone_1s, "y1": x_bar + zone_1s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(74,159,110,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone B upper (yellow) - 1 to 2 sigma
            {"type": "rect", "y0": x_bar + zone_1s, "y1": x_bar + zone_2s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(243,156,18,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone B lower (yellow)
            {"type": "rect", "y0": x_bar - zone_2s, "y1": x_bar - zone_1s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(243,156,18,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone A upper (red) - 2 to 3 sigma
            {"type": "rect", "y0": x_bar + zone_2s, "y1": x_bar + zone_3s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(231,76,60,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone A lower (red)
            {"type": "rect", "y0": x_bar - zone_3s, "y1": x_bar - zone_2s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(231,76,60,0.12)", "line": {"width": 0}, "layer": "below"},
        ]

        # Color data points by zone
        colors = []
        for i in range(n):
            abs_z = abs((data[i] - x_bar) / sigma) if sigma > 0 else 0
            if abs_z >= 3:
                colors.append("#e74c3c")
            elif abs_z >= 2:
                colors.append("#f39c12")
            elif abs_z >= 1:
                colors.append("#fdcb6e")
            else:
                colors.append("#4a9f6e")

        zone_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "lines+markers",
             "name": measurement, "marker": {"size": 7, "color": colors}, "line": {"color": "rgba(200,200,200,0.3)", "width": 1},
             "customdata": [[i, ""] for i in range(n)], "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>"},
            {"type": "scatter", "y": [x_bar] * n, "mode": "lines", "name": "CL", "line": {"color": "#00b894", "width": 1.5}},
            {"type": "scatter", "y": [x_bar + zone_3s] * n, "mode": "lines", "name": "UCL (3σ)", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [x_bar - zone_3s] * n, "mode": "lines", "name": "LCL (3σ)", "line": {"color": "#d63031", "dash": "dash"}},
        ]

        # Signal markers
        if signals:
            zone_chart_data.append({
                "type": "scatter", "x": signals, "y": [data[i] for i in signals],
                "mode": "markers", "name": "Signal (score\u22658)",
                "marker": {"size": 12, "color": "#e74c3c", "symbol": "diamond", "line": {"color": "white", "width": 1.5}},
                "customdata": [[i, "Zone signal: cumulative score \u22658"] for i in signals],
                "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<br>%{customdata[1]}<extra>Signal</extra>"
            })

        result["plots"].append({
            "title": "Zone Chart",
            "data": zone_chart_data,
            "layout": {
                "height": 390, "showlegend": True,
                "yaxis": {"title": measurement},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
                "shapes": zone_shapes,
                "annotations": [
                    {"x": n - 1, "y": x_bar + zone_1s, "text": "C", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#4a9f6e"}},
                    {"x": n - 1, "y": x_bar + zone_2s, "text": "B", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#f39c12"}},
                    {"x": n - 1, "y": x_bar + zone_3s, "text": "A", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#e74c3c"}},
                ]
            },
            "interactive": {"type": "spc_inspect"}
        })

        # Cumulative score chart
        result["plots"].append({
            "title": "Cumulative Zone Score",
            "data": [
                {"type": "scatter", "y": cum_scores, "mode": "lines+markers", "name": "Cum. Score",
                 "marker": {"size": 4, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "y": [8] * n, "mode": "lines", "name": "Signal Threshold",
                 "line": {"color": "#e74c3c", "dash": "dash"}},
            ],
            "layout": {"height": 240, "showlegend": True,
                        "yaxis": {"title": "Score"}, "xaxis": {"title": "Sample", "rangeslider": {"visible": True, "thickness": 0.12}}}
        })

        result["summary"] = f"Zone Chart Analysis\n\nCenter Line: {x_bar:.4f}\nEstimated σ: {sigma:.4f}\n\nZone Boundaries:\n  C (green): ±1σ = [{x_bar - zone_1s:.4f}, {x_bar + zone_1s:.4f}]\n  B (yellow): ±2σ = [{x_bar - zone_2s:.4f}, {x_bar + zone_2s:.4f}]\n  A (red): ±3σ = [{x_bar - zone_3s:.4f}, {x_bar + zone_3s:.4f}]\n\nScoring: A=8, B=4, C=2. Signal when cumulative ≥ 8.\nSignals detected: {len(signals)}"

    elif analysis_id == "mewma":
        """
        MEWMA — Multivariate Exponentially Weighted Moving Average.
        Extends EWMA to multiple correlated quality characteristics.
        Good for detecting small sustained multivariate shifts.
        """
        from scipy.stats import chi2

        vars_list = config.get("variables", [])
        lambda_param = float(config.get("lambda", 0.1))

        if not vars_list or len(vars_list) < 2:
            # Auto-select first 2-4 numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            vars_list = num_cols[:min(4, len(num_cols))]

        X = df[vars_list].dropna().values
        n, p = X.shape

        if n < 10 or p < 2:
            result["summary"] = "MEWMA requires at least 2 variables and 10 observations."
            return result

        # Mean vector and covariance
        mu = X.mean(axis=0)
        Sigma = np.cov(X, rowvar=False, ddof=1)

        # Regularize if near-singular
        if np.linalg.cond(Sigma) > 1e10:
            Sigma += np.eye(p) * 1e-6

        # MEWMA vectors
        Z = np.zeros((n, p))
        Z[0] = lambda_param * X[0] + (1 - lambda_param) * mu
        for i in range(1, n):
            Z[i] = lambda_param * X[i] + (1 - lambda_param) * Z[i - 1]

        # T2 statistic for each MEWMA vector
        t2_values = []
        for i in range(n):
            factor = (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param) ** (2 * (i + 1)))
            Sigma_Z = factor * Sigma
            try:
                Sigma_Z_inv = np.linalg.inv(Sigma_Z)
            except np.linalg.LinAlgError:
                Sigma_Z_inv = np.linalg.pinv(Sigma_Z)
            diff = Z[i] - mu
            t2 = float(diff @ Sigma_Z_inv @ diff)
            t2_values.append(max(0, t2))

        # UCL: chi-squared approximation (asymptotic)
        ucl = chi2.ppf(1 - 0.0027, p)  # 3-sigma equivalent ARL

        # OOC
        ooc = [i for i, t2 in enumerate(t2_values) if t2 > ucl]

        mewma_chart_data = [
            {"type": "scatter", "y": t2_values, "mode": "lines+markers", "name": "MEWMA T²",
             "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
            {"type": "scatter", "y": [ucl] * n, "mode": "lines", "name": f"UCL ({ucl:.2f})",
             "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(mewma_chart_data, t2_values, ooc)

        result["plots"].append({
            "title": "MEWMA Chart",
            "data": mewma_chart_data,
            "layout": {"height": 290, "showlegend": True,
                        "yaxis": {"title": "T² Statistic"},
                        "xaxis": {"title": "Observation", "rangeslider": {"visible": True, "thickness": 0.12}}},
            "interactive": {"type": "spc_inspect"}
        })

        # Variable contribution at OOC points
        if ooc:
            first_ooc = ooc[0]
            diff = Z[first_ooc] - mu
            contributions = diff ** 2
            total_contrib = contributions.sum()
            if total_contrib > 0:
                pct_contrib = (contributions / total_contrib * 100).tolist()
            else:
                pct_contrib = [0] * p

            result["plots"].append({
                "title": f"Variable Contribution at First OOC (obs {first_ooc})",
                "data": [{"type": "bar", "x": vars_list, "y": pct_contrib,
                          "marker": {"color": "#4a9f6e"}}],
                "layout": {"height": 250,
                            "yaxis": {"title": "% Contribution"}, "xaxis": {"title": "Variable"}}
            })

        result["summary"] = f"MEWMA Chart Analysis\n\nVariables: {', '.join(vars_list)} (p={p})\nλ (smoothing): {lambda_param}\nUCL (χ²): {ucl:.4f}\n\nObservations: {n}\nOut-of-control points: {len(ooc)}\n\nNote: Smaller λ increases sensitivity to small sustained shifts but also increases false alarm rate. Typical range: 0.05–0.25."

    elif analysis_id in ("g_chart", "t_chart"):
        """
        Rare Events Charts — for processes with very low defect rates.
        G Chart: count of opportunities (items) between events (geometric distribution).
        T Chart: time between events (Weibull/exponential, data transformed for normality).
        Auto-detect: integer data → G chart, float/continuous → T chart.
        """
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
            # CL = mean, UCL/LCL based on geometric probability
            g_bar = float(np.mean(values))
            # Exact limits: Pr(G > UCL) = α/2 where G ~ Geom(p), p = 1/(g_bar+1)
            p_est = 1 / (g_bar + 1) if g_bar > 0 else 0.5
            alpha_chart = 0.0027  # 3-sigma equivalent

            cl = g_bar
            # UCL: solve for x where P(G ≤ x) ≥ 1 - α/2
            if p_est > 0 and p_est < 1:
                ucl = float(stats.geom.ppf(1 - alpha_chart / 2, p_est) - 1)
                lcl = max(0, float(stats.geom.ppf(alpha_chart / 2, p_est) - 1))
            else:
                ucl = g_bar * 3
                lcl = 0

            chart_label = "G Chart (Opportunities Between Events)"
            y_label = "Count Between Events"

        else:
            # T Chart — time between events, Weibull/exponential transform
            # Transform using Weibull: fit shape & scale, then transform to ~normal
            # If shape ≈ 1, data is exponential
            from scipy.stats import weibull_min

            # Fit Weibull
            try:
                shape, loc, scale = weibull_min.fit(values, floc=0)
            except Exception:
                shape, scale = 1.0, float(np.mean(values))

            # Transform to normal: Z = ((x/scale)^shape)
            # Use Weibull-based control limits on original scale
            mean_t = float(np.mean(values))
            cl = mean_t

            # 3-sigma limits on transformed scale → back-transform
            # Use percentiles of fitted Weibull as control limits
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

        result["plots"].append({
            "title": chart_label,
            "data": [
                {"type": "scatter", "x": x_axis, "y": values.tolist(), "mode": "lines+markers",
                 "line": {"color": "#4a9f6e"}, "marker": {"color": colors, "size": 6},
                 "name": var,
                 "customdata": [[i, "OOC" if i in ooc_indices else ""] for i in range(n)],
                 "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>"},
                {"type": "scatter", "x": [1, n], "y": [cl, cl], "mode": "lines",
                 "line": {"color": "#e8c547", "width": 1}, "name": f"CL = {cl:.2f}"},
                {"type": "scatter", "x": [1, n], "y": [ucl, ucl], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"UCL = {ucl:.2f}"},
                {"type": "scatter", "x": [1, n], "y": [lcl, lcl], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 1}, "name": f"LCL = {lcl:.2f}"},
            ],
            "layout": {
                "height": 390,
                "xaxis": {"title": "Observation", "rangeslider": {"visible": True, "thickness": 0.12}}, "yaxis": {"title": y_label},
            },
            "interactive": {"type": "spc_inspect"}
        })

        summary = f"{chart_label}\n\n"
        summary += f"Variable: {var}\n"
        summary += f"Observations: {n}\n"
        summary += f"CL: {cl:.4f}\n"
        summary += f"UCL: {ucl:.4f}\n"
        summary += f"LCL: {lcl:.4f}\n"
        summary += f"Out-of-control: {len(ooc_indices)}\n"
        if ooc_indices:
            summary += f"OOC observations: {', '.join(str(i + 1) for i in ooc_indices)}\n"
        if chart_type == "t":
            summary += f"\nWeibull shape: {shape:.3f} (1.0 = exponential)\n"
            summary += f"Weibull scale: {scale:.3f}\n"

        result["summary"] = summary
        result["guide_observation"] = f"{'G' if chart_type == 'g' else 'T'} chart: CL={cl:.2f}, {len(ooc_indices)} OOC points out of {n}."
        result["statistics"] = {
            "chart_type": chart_type, "n": n,
            "cl": cl, "ucl": ucl, "lcl": lcl,
            "ooc_count": len(ooc_indices),
            "ooc_indices": [i + 1 for i in ooc_indices],
        }

        # Narrative
        _gt_label = "G Chart (count between events)" if chart_type == "g" else "T Chart (time between events)"
        _gt_n_ooc = len(ooc_indices)
        if _gt_n_ooc == 0:
            _gt_verdict = f"{_gt_label} — process in control"
            _gt_body = f"All {n} observations fall within control limits (CL = {cl:.2f}). The rate of rare events appears stable."
        else:
            _gt_verdict = f"{_gt_label} — {_gt_n_ooc} out-of-control point{'s' if _gt_n_ooc > 1 else ''}"
            _gt_body = f"{_gt_n_ooc} of {n} observations exceed control limits, suggesting the event rate has shifted."
        result["narrative"] = _narrative(_gt_verdict, _gt_body,
            next_steps="Investigate OOC points for assignable causes. A cluster of short intervals suggests a worsening event rate." if _gt_n_ooc > 0 else "Continue monitoring. Consider adding process improvement to reduce the baseline event rate.",
            chart_guidance="Points above UCL indicate unusually long gaps between events (improvement). Points below LCL indicate unusually short gaps (deterioration)."
        )

    # =====================================================================
    # Generalized Variance Chart
    # =====================================================================
    elif analysis_id == "generalized_variance":
        """
        Generalized Variance (|S|) Chart — monitors the determinant of the
        covariance matrix for multivariate process variability.
        Each subgroup produces |S_i|, plotted against control limits derived
        from the expected distribution of |S| under normality.
        """
        from scipy import stats as gv_stats

        variables_gv = config.get("variables") or config.get("columns", [])
        subgroup_col = config.get("subgroup") or config.get("group")
        subgroup_size_gv = int(config.get("subgroup_size", 5))

        if not variables_gv or len(variables_gv) < 2:
            result["summary"] = "Need at least 2 variables for generalized variance chart."
            return result

        data_gv = df[variables_gv + ([subgroup_col] if subgroup_col else [])].dropna()
        p_gv = len(variables_gv)

        # Create subgroups
        if subgroup_col:
            subgroups = [grp[variables_gv].values for _, grp in data_gv.groupby(subgroup_col) if len(grp) >= 2]
            subgroup_labels = [str(name) for name, grp in data_gv.groupby(subgroup_col) if len(grp) >= 2]
        else:
            n_obs = len(data_gv)
            subgroups = [data_gv[variables_gv].values[i:i+subgroup_size_gv]
                         for i in range(0, n_obs - subgroup_size_gv + 1, subgroup_size_gv)]
            subgroup_labels = [str(i + 1) for i in range(len(subgroups))]

        if len(subgroups) < 3:
            result["summary"] = "Need at least 3 subgroups for generalized variance chart."
            return result

        # Compute |S_i| for each subgroup
        det_values = []
        ns_gv = []
        for sg in subgroups:
            n_sg = len(sg)
            ns_gv.append(n_sg)
            if n_sg < p_gv:
                det_values.append(0.0)
            else:
                cov_sg = np.cov(sg.T, ddof=1)
                det_values.append(float(np.linalg.det(cov_sg)))
        det_values = np.array(det_values)

        # Pooled covariance determinant
        mean_det = float(np.mean(det_values))

        # Control limits for |S|
        # E[|S|] = |Σ| * b1, Var[|S|] = |Σ|^2 * b2
        # For subgroup size n, p variables:
        # b1 = prod((n-i)/(n-1) for i in 1..p) approximately
        # Use asymptotic approximation: UCL/LCL = mean_det ± 3*SE
        # Better: use chi-squared based limits
        n_avg = int(np.mean(ns_gv))

        # Compute b1 and b2 coefficients
        b1 = 1.0
        for i in range(1, p_gv + 1):
            b1 *= (n_avg - i) / (n_avg - 1)

        # Variance coefficient (simplified)
        b2 = b1 ** 2
        for i in range(1, p_gv + 1):
            b2 *= ((n_avg - i + 2) / (n_avg - i)) if (n_avg - i) > 0 else 1
        b2 -= b1 ** 2

        # |Σ| estimate
        sigma_det = mean_det / b1 if b1 > 0 else mean_det
        se_det = sigma_det * np.sqrt(b2) if b2 > 0 else mean_det * 0.1

        cl_gv = mean_det
        ucl_gv = cl_gv + 3 * se_det
        lcl_gv = max(0, cl_gv - 3 * se_det)

        # OOC detection
        ooc_gv = []
        for i, val in enumerate(det_values):
            if val > ucl_gv or val < lcl_gv:
                ooc_gv.append(i)

        # Chart
        in_control_x = [subgroup_labels[i] for i in range(len(det_values)) if i not in ooc_gv]
        in_control_y = [det_values[i] for i in range(len(det_values)) if i not in ooc_gv]
        ooc_x = [subgroup_labels[i] for i in ooc_gv]
        ooc_y = [det_values[i] for i in ooc_gv]

        chart_traces = [
            {"x": subgroup_labels, "y": det_values.tolist(), "mode": "lines+markers",
             "name": "|S|", "marker": {"color": "#4a9f6e", "size": 6}, "line": {"color": "#4a9f6e", "width": 1}},
        ]
        if ooc_x:
            chart_traces.append({
                "x": ooc_x, "y": ooc_y, "mode": "markers", "name": "OOC",
                "marker": {"color": "#d94a4a", "size": 10, "symbol": "x"},
            })
        # Control limit lines
        chart_traces.extend([
            {"x": [subgroup_labels[0], subgroup_labels[-1]], "y": [ucl_gv, ucl_gv],
             "mode": "lines", "name": f"UCL ({ucl_gv:.4f})", "line": {"color": "#d94a4a", "dash": "dash"}},
            {"x": [subgroup_labels[0], subgroup_labels[-1]], "y": [cl_gv, cl_gv],
             "mode": "lines", "name": f"CL ({cl_gv:.4f})", "line": {"color": "#4a90d9", "dash": "dot"}},
            {"x": [subgroup_labels[0], subgroup_labels[-1]], "y": [lcl_gv, lcl_gv],
             "mode": "lines", "name": f"LCL ({lcl_gv:.4f})", "line": {"color": "#d94a4a", "dash": "dash"}},
        ])

        result["plots"].append({
            "title": "Generalized Variance |S| Chart",
            "data": chart_traces,
            "layout": {"height": 440, "xaxis": {"title": "Subgroup", "rangeslider": {"visible": True, "thickness": 0.12}}, "yaxis": {"title": "|S| (Determinant)"},
                       }
        })

        summary_gv = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_gv += f"<<COLOR:title>>GENERALIZED VARIANCE CHART<</COLOR>>\n"
        summary_gv += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_gv += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables_gv)}\n"
        summary_gv += f"<<COLOR:highlight>>Subgroups:<</COLOR>> {len(subgroups)}  (avg size = {n_avg})\n\n"
        summary_gv += f"<<COLOR:text>>Control Limits:<</COLOR>>\n"
        summary_gv += f"  UCL = {ucl_gv:.6f}\n"
        summary_gv += f"  CL  = {cl_gv:.6f}\n"
        summary_gv += f"  LCL = {lcl_gv:.6f}\n\n"
        if ooc_gv:
            summary_gv += f"<<COLOR:warning>>Out-of-control points: {len(ooc_gv)}<</COLOR>>\n"
            for idx_ooc in ooc_gv:
                summary_gv += f"  Subgroup {subgroup_labels[idx_ooc]}: |S| = {det_values[idx_ooc]:.6f}\n"
        else:
            summary_gv += f"<<COLOR:good>>Process variability in control — no OOC points<</COLOR>>\n"

        result["summary"] = summary_gv
        result["guide_observation"] = f"Generalized variance chart: {len(ooc_gv)} OOC points out of {len(subgroups)} subgroups."
        result["statistics"] = {
            "cl": cl_gv, "ucl": ucl_gv, "lcl": lcl_gv,
            "det_values": det_values.tolist(), "n_subgroups": len(subgroups),
            "ooc_count": len(ooc_gv), "p": p_gv,
        }

        # Narrative
        _gv_n_ooc = len(ooc_gv)
        if _gv_n_ooc == 0:
            _gv_verdict = f"Generalized Variance — multivariate spread in control"
            _gv_body = f"All {len(subgroups)} subgroups have covariance determinants within limits. Joint variability of {p_gv} variables is stable."
        else:
            _gv_verdict = f"Generalized Variance — {_gv_n_ooc} OOC subgroup{'s' if _gv_n_ooc > 1 else ''}"
            _gv_body = f"{_gv_n_ooc} of {len(subgroups)} subgroups show unusual joint variability across {p_gv} variables. The covariance structure has shifted."
        result["narrative"] = _narrative(_gv_verdict, _gv_body,
            next_steps="Pair with a Hotelling T² chart to distinguish mean shifts from variability shifts." if _gv_n_ooc > 0 else "Continue monitoring. Process variability is stable across all measured dimensions.",
            chart_guidance="Each point is the determinant |S| of the subgroup covariance matrix. Higher values mean more joint spread."
        )

    # ══════════════════════════════════════════════════════════════════════
    # CONFORMAL PREDICTION SPC (Burger et al., Dec 2025 — arXiv:2512.23602)
    # First commercial implementation. Distribution-free, model-agnostic.
    # ══════════════════════════════════════════════════════════════════════

    elif analysis_id == "conformal_control":
        """
        Conformal-Enhanced Control Chart — distribution-free alternative to Shewhart.
        No normality assumption. Guaranteed false alarm rate.
        Adaptive prediction intervals + uncertainty spike detection.

        Reference: Burger et al. (2025) "Distribution-Free Process Monitoring
        with Conformal Prediction", arXiv:2512.23602
        """
        n = len(data)
        alpha_conf = float(config.get("alpha", 0.05))  # False alarm rate
        cal_fraction = float(config.get("calibration_fraction", 0.5))
        spike_threshold = float(config.get("spike_threshold", 2.0))  # Multiple of median width for spike detection
        chart_type = config.get("chart_type", "individuals")  # individuals, subgroup_mean, subgroup_range
        subgroup_size = int(config.get("subgroup_size", 5))

        if n < 20:
            result["summary"] = "Need at least 20 observations for conformal control chart (calibration + monitoring)."
            return result

        # Phase I / Phase II split
        n_cal = max(10, int(n * cal_fraction))
        cal_data = data[:n_cal]
        mon_data = data[n_cal:]
        n_mon = len(mon_data)

        if chart_type == "subgroup_mean" and n_cal >= subgroup_size * 2:
            # Subgroup means
            n_sg_cal = n_cal // subgroup_size
            cal_subgroups = [cal_data[i*subgroup_size:(i+1)*subgroup_size] for i in range(n_sg_cal)]
            cal_values = np.array([np.mean(sg) for sg in cal_subgroups])
            center = np.median(cal_values)

            n_sg_mon = n_mon // subgroup_size
            mon_values = np.array([np.mean(mon_data[i*subgroup_size:(i+1)*subgroup_size]) for i in range(n_sg_mon)])
            x_labels = [f"SG {i+1}" for i in range(n_sg_cal + n_sg_mon)]
            all_values = np.concatenate([cal_values, mon_values])
            chart_label = f"Subgroup Mean (n={subgroup_size})"
        elif chart_type == "subgroup_range" and n_cal >= subgroup_size * 2:
            # Subgroup ranges
            n_sg_cal = n_cal // subgroup_size
            cal_subgroups = [cal_data[i*subgroup_size:(i+1)*subgroup_size] for i in range(n_sg_cal)]
            cal_values = np.array([np.ptp(sg) for sg in cal_subgroups])
            center = np.median(cal_values)

            n_sg_mon = n_mon // subgroup_size
            mon_values = np.array([np.ptp(mon_data[i*subgroup_size:(i+1)*subgroup_size]) for i in range(n_sg_mon)])
            x_labels = [f"SG {i+1}" for i in range(n_sg_cal + n_sg_mon)]
            all_values = np.concatenate([cal_values, mon_values])
            chart_label = f"Subgroup Range (n={subgroup_size})"
        else:
            # Individual observations
            cal_values = cal_data
            center = np.median(cal_values)
            mon_values = mon_data
            x_labels = list(range(n))
            all_values = data
            chart_label = "Individual Values"

        # Nonconformity scores (calibration): |Xi - median| (robust to outliers)
        cal_scores = np.abs(cal_values - center)

        # Conformal threshold: ⌈(1-α)(n_cal+1)⌉-th smallest score
        sorted_scores = np.sort(cal_scores)
        q_index = int(np.ceil((1 - alpha_conf) * (len(cal_scores) + 1))) - 1
        q_index = min(q_index, len(sorted_scores) - 1)
        q = float(sorted_scores[q_index])

        # Prediction interval (distribution-free)
        pi_lower = center - q
        pi_upper = center + q

        # Phase II: compute scores and flag OOC
        mon_scores = np.abs(mon_values - center)
        ooc_mask = mon_scores > q
        ooc_indices = np.where(ooc_mask)[0]
        n_ooc = len(ooc_indices)

        # All scores for plotting
        all_scores = np.abs(all_values - center)

        # ── Adaptive Prediction Intervals (model-based variant) ──
        # Use a simple rolling local model: for each monitoring point,
        # fit prediction based on recent window → normalized residuals → adaptive width
        window = min(20, n_cal)
        adaptive_widths = np.zeros(n)
        adaptive_lower = np.zeros(n)
        adaptive_upper = np.zeros(n)

        for i in range(n):
            if i < window:
                # Use calibration statistics for early points
                local_std = np.std(data[:max(i + 1, 2)], ddof=1) if i > 0 else np.std(cal_data, ddof=1)
                local_center = np.median(data[:max(i + 1, 2)])
            else:
                local_window = data[i - window:i]
                local_std = np.std(local_window, ddof=1)
                local_center = np.median(local_window)

            # Normalized score: accounts for local variability
            adaptive_widths[i] = q * max(local_std / np.std(cal_data, ddof=1), 0.5) if np.std(cal_data, ddof=1) > 0 else q
            adaptive_lower[i] = local_center - adaptive_widths[i]
            adaptive_upper[i] = local_center + adaptive_widths[i]

        # ── Uncertainty Spike Detection ──
        # Spike = adaptive width exceeds spike_threshold × median(adaptive width over calibration)
        median_width = np.median(adaptive_widths[:n_cal])
        spike_mask = adaptive_widths > spike_threshold * median_width
        spike_indices = np.where(spike_mask)[0]
        # Only count spikes in monitoring phase
        spike_indices_mon = spike_indices[spike_indices >= n_cal]

        # For Shewhart comparison
        shewhart_mean = np.mean(cal_data)
        shewhart_std = np.std(cal_data, ddof=1)
        shewhart_ucl = shewhart_mean + 3 * shewhart_std
        shewhart_lcl = shewhart_mean - 3 * shewhart_std

        # Summary
        summary_cc = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_cc += f"<<COLOR:title>>CONFORMAL-ENHANCED CONTROL CHART<</COLOR>>\n"
        summary_cc += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_cc += f"<<COLOR:dim>>Burger et al. (2025) — Distribution-free, guaranteed coverage<</COLOR>>\n\n"
        summary_cc += f"<<COLOR:highlight>>Variable:<</COLOR>> {measurement}\n"
        summary_cc += f"<<COLOR:highlight>>Chart type:<</COLOR>> {chart_label}\n"
        summary_cc += f"<<COLOR:highlight>>N:<</COLOR>> {n} ({n_cal} calibration + {n_mon} monitoring)\n"
        summary_cc += f"<<COLOR:highlight>>Significance level:<</COLOR>> α = {alpha_conf} (guaranteed ≤ {alpha_conf*100:.0f}% false alarm rate)\n\n"

        summary_cc += f"<<COLOR:text>>Conformal Control Limits (distribution-free):<</COLOR>>\n"
        summary_cc += f"  Center (median): {center:.4f}\n"
        summary_cc += f"  Conformal threshold (q): {q:.4f}\n"
        summary_cc += f"  Prediction interval: [{pi_lower:.4f}, {pi_upper:.4f}]\n\n"

        summary_cc += f"<<COLOR:text>>Traditional Shewhart Limits (for comparison):<</COLOR>>\n"
        summary_cc += f"  Mean ± 3σ: [{shewhart_lcl:.4f}, {shewhart_ucl:.4f}]\n\n"

        # Compare the two approaches
        conformal_width = pi_upper - pi_lower
        shewhart_width = shewhart_ucl - shewhart_lcl
        if conformal_width < shewhart_width:
            summary_cc += f"<<COLOR:good>>Conformal limits are {(1 - conformal_width/shewhart_width)*100:.0f}% tighter than Shewhart — more sensitive to shifts.<</COLOR>>\n\n"
        elif conformal_width > shewhart_width * 1.1:
            summary_cc += f"<<COLOR:text>>Conformal limits are wider than Shewhart — data may be non-normal (heavy tails). This is the correct adjustment.<</COLOR>>\n\n"
        else:
            summary_cc += f"<<COLOR:text>>Conformal and Shewhart limits are similar — data is approximately normal.<</COLOR>>\n\n"

        if n_ooc > 0:
            summary_cc += f"<<COLOR:warning>>⚠ {n_ooc} out-of-control point(s) in monitoring phase<</COLOR>>\n"
            if n_ooc <= 10:
                for idx in ooc_indices:
                    summary_cc += f"  Observation {n_cal + idx}: value = {mon_values[idx]:.4f}, score = {mon_scores[idx]:.4f} > q = {q:.4f}\n"
        else:
            summary_cc += f"<<COLOR:good>>No out-of-control points in monitoring phase<</COLOR>>\n"

        if len(spike_indices_mon) > 0:
            summary_cc += f"\n<<COLOR:warning>>⚠ {len(spike_indices_mon)} uncertainty spike(s) detected — leading indicators of instability<</COLOR>>\n"
            summary_cc += f"  Spike threshold: {spike_threshold}× median interval width\n"
            if len(spike_indices_mon) <= 10:
                for idx in spike_indices_mon:
                    summary_cc += f"  Observation {idx}: width = {adaptive_widths[idx]:.4f} ({adaptive_widths[idx]/median_width:.1f}× normal)\n"
        else:
            summary_cc += f"\n<<COLOR:good>>No uncertainty spikes detected<</COLOR>>\n"

        summary_cc += f"\n<<COLOR:text>>Key advantages:<</COLOR>>\n"
        summary_cc += f"  • Distribution-free: no normality assumption required\n"
        summary_cc += f"  • Guaranteed false alarm rate ≤ α = {alpha_conf}\n"
        summary_cc += f"  • Uncertainty spikes provide early warning before limits are breached\n"
        summary_cc += f"  • Adaptive intervals respond to changing process conditions\n"

        result["summary"] = summary_cc

        # Plot 1: Conformal control chart (main)
        n_cal_plot = len(cal_values)  # subgroup count for subgroup charts, n_cal for individuals
        n_total = len(all_values)
        point_colors = []
        for i in range(n_total):
            if i < len(cal_values):
                point_colors.append("#4a9f6e")  # Calibration = green
            elif all_scores[i] > q:
                point_colors.append("#e85747")  # OOC = red
            else:
                point_colors.append("#4a90d9")  # In-control monitoring = blue

        result["plots"].append({
            "title": "Conformal-Enhanced Control Chart",
            "data": [
                {"type": "scatter", "x": list(range(n_total)), "y": all_values.tolist(), "mode": "lines+markers",
                 "marker": {"size": 5, "color": point_colors}, "line": {"color": "rgba(74, 159, 110, 0.3)", "width": 1}, "name": measurement},
                {"type": "scatter", "x": list(range(n_total)), "y": [center] * n_total, "mode": "lines",
                 "line": {"color": "#00b894", "width": 1.5}, "name": f"Center (median={center:.3f})"},
                {"type": "scatter", "x": list(range(n_total)), "y": [pi_upper] * n_total, "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 1.5}, "name": f"Conformal UCL ({pi_upper:.3f})"},
                {"type": "scatter", "x": list(range(n_total)), "y": [pi_lower] * n_total, "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 1.5}, "name": f"Conformal LCL ({pi_lower:.3f})"},
                {"type": "scatter", "x": list(range(n_total)), "y": [shewhart_ucl] * n_total, "mode": "lines",
                 "line": {"color": "#9aaa9a", "dash": "dot", "width": 1}, "name": f"Shewhart ±3σ", "showlegend": True},
                {"type": "scatter", "x": list(range(n_total)), "y": [shewhart_lcl] * n_total, "mode": "lines",
                 "line": {"color": "#9aaa9a", "dash": "dot", "width": 1}, "showlegend": False},
            ],
            "layout": {
                "height": 360,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": measurement},
                "shapes": [{"type": "line", "x0": n_cal_plot, "x1": n_cal_plot, "y0": float(np.min(all_values)) - 0.1 * float(np.ptp(all_values)),
                            "y1": float(np.max(all_values)) + 0.1 * float(np.ptp(all_values)),
                            "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5}}],
                "annotations": [{"x": n_cal_plot, "y": float(np.max(all_values)), "text": "← Cal | Mon →",
                                 "showarrow": False, "font": {"color": "#e8c547", "size": 10}}],
            }
        })

        # Plot 2: Adaptive prediction interval (ribbon)
        result["plots"].append({
            "title": "Adaptive Prediction Interval (uncertainty-aware)",
            "data": [
                {"type": "scatter", "x": list(range(n)) + list(range(n - 1, -1, -1)),
                 "y": adaptive_upper.tolist() + adaptive_lower[::-1].tolist(),
                 "fill": "toself", "fillcolor": "rgba(74, 144, 217, 0.15)", "line": {"color": "transparent"},
                 "name": "Adaptive PI"},
                {"type": "scatter", "x": list(range(n)), "y": data.tolist(), "mode": "lines+markers",
                 "marker": {"size": 3, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e", "width": 1}, "name": measurement},
            ] + ([
                {"type": "scatter", "x": spike_indices_mon.tolist(),
                 "y": data[spike_indices_mon].tolist() if len(spike_indices_mon) > 0 else [],
                 "mode": "markers", "marker": {"size": 10, "color": "#e8c547", "symbol": "triangle-up", "line": {"color": "#e89547", "width": 1.5}},
                 "name": "Uncertainty Spike"}
            ] if len(spike_indices_mon) > 0 else []),
            "layout": {"height": 360, "xaxis": {"title": "Observation"}, "yaxis": {"title": measurement}, }
        })

        # Plot 3: Nonconformity score chart
        score_colors = ["#e85747" if s > q else "#4a9f6e" for s in all_scores]
        result["plots"].append({
            "title": "Nonconformity Scores vs Threshold",
            "data": [
                {"type": "bar", "x": list(range(n_total)), "y": all_scores.tolist(), "marker": {"color": score_colors}, "name": "Score"},
                {"type": "scatter", "x": list(range(n_total)), "y": [q] * n_total, "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 2}, "name": f"Threshold q={q:.3f}"},
            ],
            "layout": {"height": 320, "xaxis": {"title": "Observation"}, "yaxis": {"title": "|X - median|"}, }
        })

        # Plot 4: Interval width over time (spike detection)
        width_colors = ["#e8c547" if spike_mask[i] else "#4a90d9" for i in range(n)]
        result["plots"].append({
            "title": "Prediction Interval Width (spike = leading indicator)",
            "data": [
                {"type": "bar", "x": list(range(n)), "y": (adaptive_widths * 2).tolist(), "marker": {"color": width_colors, "opacity": 0.7}, "name": "Width"},
                {"type": "scatter", "x": list(range(n)), "y": [spike_threshold * median_width * 2] * n, "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 1.5}, "name": f"Spike threshold ({spike_threshold}× median)"},
            ],
            "layout": {"height": 320, "xaxis": {"title": "Observation"}, "yaxis": {"title": "Interval Width"}, }
        })

        result["guide_observation"] = f"Conformal control chart (distribution-free): {n_ooc} OOC, {len(spike_indices_mon)} uncertainty spikes in {n_mon} monitoring observations."
        result["statistics"] = {
            "n": n, "n_calibration": n_cal, "n_monitoring": n_mon,
            "center": float(center), "conformal_threshold": float(q),
            "pi_lower": float(pi_lower), "pi_upper": float(pi_upper),
            "shewhart_ucl": float(shewhart_ucl), "shewhart_lcl": float(shewhart_lcl),
            "n_ooc": n_ooc, "ooc_indices": [int(n_cal + i) for i in ooc_indices],
            "n_spikes": len(spike_indices_mon),
            "spike_indices": spike_indices_mon.tolist(),
            "alpha": alpha_conf,
            "method": "Burger et al. (2025) arXiv:2512.23602",
        }

        # Narrative
        _cc_n_spikes = len(spike_indices_mon)
        if n_ooc == 0 and _cc_n_spikes == 0:
            _cc_verdict = "Conformal Control Chart — process in control (distribution-free)"
            _cc_body = f"No out-of-control points or uncertainty spikes in {n_mon} monitoring observations. Guaranteed false alarm rate = {alpha_conf*100:.1f}% without normality assumption."
        else:
            _cc_verdict = f"Conformal Control Chart — {n_ooc} OOC" + (f", {_cc_n_spikes} uncertainty spike{'s' if _cc_n_spikes > 1 else ''}" if _cc_n_spikes > 0 else "")
            _cc_body = f"{n_ooc} out-of-control points detected in {n_mon} monitoring observations (calibrated on {n_cal} points). " + (f"Additionally, {_cc_n_spikes} uncertainty spikes serve as early warnings of instability." if _cc_n_spikes > 0 else "")
        result["narrative"] = _narrative(_cc_verdict, _cc_body,
            next_steps="Investigate OOC points for assignable causes. Uncertainty spikes often precede full OOC events." if n_ooc > 0 or _cc_n_spikes > 0 else "Process is stable under distribution-free monitoring.",
            chart_guidance="Blue band = adaptive prediction interval. Yellow triangles = uncertainty spikes (interval widening). Red points = out-of-control observations exceeding the conformal threshold."
        )

    elif analysis_id == "conformal_monitor":
        """
        Conformal P-Value Chart — multivariate process monitoring.
        Model-agnostic anomaly detection with conformal p-values.
        Distribution-free guaranteed false alarm rate.

        Reference: Burger et al. (2025) "Distribution-Free Process Monitoring
        with Conformal Prediction", arXiv:2512.23602
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        alpha_conf = float(config.get("alpha", 0.05))
        cal_fraction = float(config.get("calibration_fraction", 0.5))
        model_type = config.get("model", "isolation_forest")  # isolation_forest, mahalanobis

        # Get numeric columns
        variables = config.get("variables", [])
        if not variables:
            variables = df.select_dtypes(include="number").columns.tolist()
        if len(variables) < 2:
            result["summary"] = "Conformal monitoring requires at least 2 numeric variables for multivariate analysis."
            return result

        X = df[variables].dropna().values
        n = len(X)

        if n < 30:
            result["summary"] = "Need at least 30 observations for conformal multivariate monitoring."
            return result

        # Phase I / Phase II split
        n_cal = max(15, int(n * cal_fraction))
        X_cal = X[:n_cal]
        X_mon = X[n_cal:]
        n_mon = len(X_mon)

        # Standardize using calibration data
        scaler = StandardScaler()
        X_cal_scaled = scaler.fit_transform(X_cal)
        X_mon_scaled = scaler.transform(X_mon) if n_mon > 0 else np.array([]).reshape(0, len(variables))

        # Compute nonconformity scores
        if model_type == "mahalanobis":
            # Mahalanobis distance from calibration centroid
            cov_cal = np.cov(X_cal_scaled.T)
            try:
                cov_inv = np.linalg.inv(cov_cal)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_cal)
            mean_cal = np.mean(X_cal_scaled, axis=0)

            cal_scores = np.array([np.sqrt((x - mean_cal) @ cov_inv @ (x - mean_cal)) for x in X_cal_scaled])
            mon_scores = np.array([np.sqrt((x - mean_cal) @ cov_inv @ (x - mean_cal)) for x in X_mon_scaled]) if n_mon > 0 else np.array([])
            model_label = "Mahalanobis Distance"
        else:
            # Isolation Forest anomaly scores
            iso = IsolationForest(random_state=42, contamination="auto")
            iso.fit(X_cal_scaled)

            # Score = -decision_function (higher = more anomalous)
            cal_scores = -iso.decision_function(X_cal_scaled)
            mon_scores = -iso.decision_function(X_mon_scaled) if n_mon > 0 else np.array([])
            model_label = "Isolation Forest"

        all_scores = np.concatenate([cal_scores, mon_scores])

        # Conformal p-values for monitoring observations
        # p-value = (# calibration scores >= current score + 1) / (n_cal + 1)
        sorted_cal = np.sort(cal_scores)
        cal_p_values = np.array([(np.sum(cal_scores >= s) + 1) / (n_cal + 1) for s in cal_scores])
        mon_p_values = np.array([(np.sum(cal_scores >= s) + 1) / (n_cal + 1) for s in mon_scores]) if n_mon > 0 else np.array([])
        all_p_values = np.concatenate([cal_p_values, mon_p_values])

        # Anomalies: p-value < alpha
        if n_mon > 0:
            anomaly_mask = mon_p_values < alpha_conf
            anomaly_indices = np.where(anomaly_mask)[0]
            n_anomalies = len(anomaly_indices)
        else:
            anomaly_indices = np.array([])
            n_anomalies = 0

        # Variable contributions for flagged points (which variable drives the anomaly?)
        contributions = []
        if n_anomalies > 0 and n_anomalies <= 20:
            mean_cal_raw = np.mean(X_cal, axis=0)
            std_cal_raw = np.std(X_cal, axis=0, ddof=1)
            std_cal_raw[std_cal_raw == 0] = 1
            for idx in anomaly_indices[:10]:
                z_scores = np.abs((X_mon[idx] - mean_cal_raw) / std_cal_raw)
                top_var = variables[np.argmax(z_scores)]
                contributions.append((int(idx), top_var, float(np.max(z_scores))))

        # Summary
        summary_cm = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_cm += f"<<COLOR:title>>CONFORMAL P-VALUE CHART (Multivariate Monitor)<</COLOR>>\n"
        summary_cm += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_cm += f"<<COLOR:dim>>Burger et al. (2025) — Distribution-free anomaly detection<</COLOR>>\n\n"
        summary_cm += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables)} ({len(variables)} dimensions)\n"
        summary_cm += f"<<COLOR:highlight>>N:<</COLOR>> {n} ({n_cal} calibration + {n_mon} monitoring)\n"
        summary_cm += f"<<COLOR:highlight>>Anomaly model:<</COLOR>> {model_label}\n"
        summary_cm += f"<<COLOR:highlight>>Significance level:<</COLOR>> α = {alpha_conf}\n\n"

        summary_cm += f"<<COLOR:text>>Conformal P-Value Distribution (monitoring):<</COLOR>>\n"
        if n_mon > 0:
            summary_cm += f"  Mean p-value: {np.mean(mon_p_values):.4f}\n"
            summary_cm += f"  Min p-value: {np.min(mon_p_values):.4f} at observation {n_cal + int(np.argmin(mon_p_values))}\n"
            summary_cm += f"  % below α: {np.mean(mon_p_values < alpha_conf)*100:.1f}%\n\n"

        if n_anomalies > 0:
            summary_cm += f"<<COLOR:warning>>⚠ {n_anomalies} anomalous observation(s) detected (p < {alpha_conf})<</COLOR>>\n"
            if contributions:
                summary_cm += f"\n  <<COLOR:text>>Top contributing variables:<</COLOR>>\n"
                for idx, var, z in contributions:
                    summary_cm += f"    Obs {n_cal + idx}: driven by '{var}' (z = {z:.2f}σ from calibration mean)\n"
        else:
            summary_cm += f"<<COLOR:good>>No anomalies detected — all p-values ≥ {alpha_conf}<</COLOR>>\n"

        summary_cm += f"\n<<COLOR:text>>Key advantages:<</COLOR>>\n"
        summary_cm += f"  • Monitors {len(variables)} variables simultaneously\n"
        summary_cm += f"  • No distributional assumptions on joint variable behavior\n"
        summary_cm += f"  • Guaranteed false alarm rate ≤ α = {alpha_conf}\n"
        summary_cm += f"  • Intuitive p-value scale (0 = most anomalous, 1 = most normal)\n"

        result["summary"] = summary_cm

        # Plot 1: P-value chart (the main chart)
        p_colors = []
        for i, p in enumerate(all_p_values):
            if i < n_cal:
                p_colors.append("#4a9f6e")  # Calibration
            elif p < alpha_conf:
                p_colors.append("#e85747")  # Anomaly
            else:
                p_colors.append("#4a90d9")  # Normal monitoring

        result["plots"].append({
            "title": "Conformal P-Value Chart",
            "data": [
                {"type": "scatter", "x": list(range(n)), "y": all_p_values.tolist(), "mode": "markers",
                 "marker": {"size": 6, "color": p_colors}, "name": "Conformal p-value"},
                {"type": "scatter", "x": list(range(n)), "y": [alpha_conf] * n, "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 2}, "name": f"α = {alpha_conf}"},
            ],
            "layout": {
                "height": 290,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Conformal p-value", "range": [0, 1]},
                "shapes": [
                    {"type": "line", "x0": n_cal, "x1": n_cal, "y0": 0, "y1": 1,
                     "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5}},
                    {"type": "rect", "x0": 0, "x1": n, "y0": 0, "y1": alpha_conf,
                     "fillcolor": "rgba(232, 87, 71, 0.05)", "line": {"width": 0}},
                ],
                "annotations": [{"x": n_cal, "y": 0.95, "text": "← Cal | Mon →",
                                 "showarrow": False, "font": {"color": "#e8c547", "size": 10}}],
            }
        })

        # Plot 2: Anomaly scores over time
        score_colors_2 = ["#e85747" if (i >= n_cal and all_p_values[i] < alpha_conf) else "#4a9f6e" for i in range(n)]
        result["plots"].append({
            "title": f"Nonconformity Scores ({model_label})",
            "data": [
                {"type": "bar", "x": list(range(n)), "y": all_scores.tolist(),
                 "marker": {"color": score_colors_2, "opacity": 0.7}, "name": "Score"},
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Anomaly Score"},
                "shapes": [{"type": "line", "x0": n_cal, "x1": n_cal, "y0": 0, "y1": float(np.max(all_scores)),
                            "line": {"color": "#e8c547", "dash": "dashdot", "width": 1}}],
            }
        })

        # Plot 3: Variable-level view (parallel coordinates or heatmap of z-scores)
        if n_mon > 0:
            mean_cal_raw = np.mean(X_cal, axis=0)
            std_cal_raw = np.std(X_cal, axis=0, ddof=1)
            std_cal_raw[std_cal_raw == 0] = 1
            z_matrix = np.abs((X_mon - mean_cal_raw) / std_cal_raw)

            # Show z-scores for monitoring observations (heatmap)
            result["plots"].append({
                "title": "Variable Contribution Heatmap (|z-score| from calibration)",
                "data": [{
                    "type": "heatmap",
                    "z": z_matrix.T.tolist(),
                    "x": [f"{n_cal + i}" for i in range(n_mon)],
                    "y": variables,
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]],
                    "colorbar": {"title": "|z|"},
                }],
                "layout": {"height": max(200, 40 * len(variables)), "xaxis": {"title": "Observation"}, }
            })

        result["guide_observation"] = f"Conformal multivariate monitor: {n_anomalies} anomalies in {n_mon} monitoring observations across {len(variables)} variables."
        result["statistics"] = {
            "n": n, "n_calibration": n_cal, "n_monitoring": n_mon,
            "n_variables": len(variables), "variables": variables,
            "model": model_label, "alpha": alpha_conf,
            "n_anomalies": n_anomalies,
            "anomaly_indices": [int(n_cal + i) for i in anomaly_indices],
            "mean_p_value": float(np.mean(mon_p_values)) if n_mon > 0 else None,
            "min_p_value": float(np.min(mon_p_values)) if n_mon > 0 else None,
            "method": "Burger et al. (2025) arXiv:2512.23602",
        }

        # Narrative
        if n_anomalies == 0:
            _cm_verdict = f"Conformal Monitor — no anomalies ({len(variables)} variables)"
            _cm_body = f"All {n_mon} monitoring observations are within normal bounds across {len(variables)} variables using {model_label}. False alarm rate controlled at {alpha_conf*100:.1f}%."
        else:
            _cm_top_var = ""
            if contributions:
                from collections import Counter
                _cm_var_counts = Counter(v for _, v, _ in contributions)
                _cm_top_var = f" Top contributing variable: <strong>{_cm_var_counts.most_common(1)[0][0]}</strong>."
            _cm_verdict = f"Conformal Monitor — {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} detected"
            _cm_body = f"{n_anomalies} of {n_mon} monitoring observations flagged as anomalous across {len(variables)} variables.{_cm_top_var}"
        result["narrative"] = _narrative(_cm_verdict, _cm_body,
            next_steps="Check the variable contribution heatmap to identify which dimensions are driving anomalies." if n_anomalies > 0 else "Process is multivariate-stable. Continue monitoring.",
            chart_guidance="The p-value chart shows conformal p-values per observation — points below the red line are anomalous. The heatmap reveals which variables contribute most to each flagged observation."
        )

    # ── Bridge: Bayesian SPC suite lives in run_visualization ──
    elif analysis_id.startswith("bayes_spc_"):
        from .viz import run_visualization
        return run_visualization(df, analysis_id, config)

    # ── Bridge: Bayesian DOE suite ──
    elif analysis_id.startswith("bayes_doe_"):
        from ..bayes_doe import run_bayesian_doe
        return run_bayesian_doe(df, analysis_id, config)

    elif analysis_id == "entropy_spc":
        # Information-Theoretic SPC — Shannon entropy control chart
        col = config.get("var") or config.get("measurement") or config.get("column")
        n_bins = int(config.get("bins", config.get("n_bins", 10)))
        window = int(config.get("window", 30))

        if not col or col not in df.columns:
            result["summary"] = "Error: Specify a numeric column."
            return result

        data = df[col].dropna().values.astype(float)
        n = len(data)

        if n < window * 2:
            result["summary"] = f"Error: Need at least {window*2} observations for window={window}."
            return result

        # Compute rolling Shannon entropy on binned data
        global_min, global_max = float(np.min(data)), float(np.max(data))
        bin_edges = np.linspace(global_min - 1e-10, global_max + 1e-10, n_bins + 1)

        entropies = []
        for i in range(window, n + 1):
            segment = data[i - window:i]
            counts, _ = np.histogram(segment, bins=bin_edges)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = float(-np.sum(probs * np.log2(probs)))
            entropies.append(h)

        entropies = np.array(entropies)
        x_idx = list(range(window, n + 1))

        # Phase I limits (first half as baseline)
        n_phase1 = len(entropies) // 2
        h_mean = float(np.mean(entropies[:n_phase1]))
        h_std = float(np.std(entropies[:n_phase1], ddof=1))
        ucl = h_mean + 3 * h_std
        lcl = max(0, h_mean - 3 * h_std)

        # Detect OOC points
        ooc_idx = [i for i, h in enumerate(entropies) if h > ucl or h < lcl]
        ooc_pct = len(ooc_idx) / len(entropies) * 100

        # Detect entropy drops (distribution narrowing / mode collapse)
        drops = [i for i, h in enumerate(entropies) if h < lcl]
        # Detect entropy spikes (distribution spreading / bimodality forming)
        spikes = [i for i, h in enumerate(entropies) if h > ucl]

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ENTROPY SPC (INFORMATION-THEORETIC CONTROL CHART)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Variable:<</COLOR>> {col}\n"
        summary += f"<<COLOR:text>>Window:<</COLOR>> {window}    Bins: {n_bins}    N: {n}\n\n"
        summary += f"<<COLOR:accent>>\u2500\u2500 Entropy Statistics \u2500\u2500<</COLOR>>\n"
        summary += f"  Baseline mean: {h_mean:.4f} bits\n"
        summary += f"  UCL: {ucl:.4f}    LCL: {lcl:.4f}\n"
        summary += f"  OOC points: {len(ooc_idx)} ({ooc_pct:.1f}%)\n"
        if spikes:
            summary += f"  Entropy spikes (distribution spreading): {len(spikes)}\n"
        if drops:
            summary += f"  Entropy drops (distribution narrowing): {len(drops)}\n"

        result["summary"] = summary
        result["statistics"] = {
            "h_mean": h_mean, "h_std": h_std, "ucl": ucl, "lcl": lcl,
            "n_ooc": len(ooc_idx), "n_spikes": len(spikes), "n_drops": len(drops),
        }

        _es_status = "in control" if not ooc_idx else f"{len(ooc_idx)} out-of-control signals"
        _es_detail = ""
        if spikes and drops:
            _es_detail = " Both entropy spikes (distributional spreading) and drops (narrowing) detected."
        elif spikes:
            _es_detail = " Entropy spikes indicate distributional spreading \u2014 possible bimodality or increased variation."
        elif drops:
            _es_detail = " Entropy drops indicate distribution narrowing \u2014 possible mode collapse or reduced variation."

        result["guide_observation"] = f"Entropy SPC: {_es_status}. Mean entropy = {h_mean:.3f} bits."
        result["narrative"] = _narrative(
            f"Entropy SPC \u2014 {_es_status}",
            f"Rolling Shannon entropy (window={window}, {n_bins} bins) has baseline mean {h_mean:.4f} bits. "
            f"{len(ooc_idx)} points ({ooc_pct:.1f}%) fall outside 3\u03c3 limits [{lcl:.3f}, {ucl:.3f}].{_es_detail}",
            next_steps="Entropy catches distributional shifts that Shewhart charts miss \u2014 bimodality forming inside normal control limits, "
                       "shape changes without mean shift. Investigate OOC entropy points for root cause.",
            chart_guidance="Entropy above UCL = distribution is spreading or splitting. Below LCL = distribution is collapsing. "
                          "Compare with the Xbar chart \u2014 entropy often signals before the mean moves."
        )

        # Plot: entropy control chart
        h_colors = ["#dc5050" if i in ooc_idx else "#4a9f6e" for i in range(len(entropies))]
        result["plots"].append({
            "title": f"Entropy Control Chart ({col})",
            "data": [
                {"type": "scatter", "x": x_idx, "y": entropies.tolist(),
                 "mode": "lines+markers", "marker": {"size": 4, "color": h_colors},
                 "line": {"color": "#4a9f6e", "width": 1}, "name": "Entropy"},
            ],
            "layout": {
                "height": 290,
                "xaxis": {"title": "Observation", "rangeslider": {"visible": True, "thickness": 0.12}},
                "yaxis": {"title": "Shannon Entropy (bits)"},
                "shapes": [
                    {"type": "line", "x0": x_idx[0], "x1": x_idx[-1], "y0": h_mean, "y1": h_mean,
                     "line": {"color": "#4a90d9", "width": 1}},
                    {"type": "line", "x0": x_idx[0], "x1": x_idx[-1], "y0": ucl, "y1": ucl,
                     "line": {"color": "#dc5050", "dash": "dash", "width": 1}},
                    {"type": "line", "x0": x_idx[0], "x1": x_idx[-1], "y0": lcl, "y1": lcl,
                     "line": {"color": "#dc5050", "dash": "dash", "width": 1}},
                ],
            },
        })

    elif analysis_id == "degradation_capability":
        # Degradation-Aware Capability — Cpk as a function of time/usage
        meas_col = config.get("var") or config.get("measurement") or config.get("column")
        time_col = config.get("time_column") or config.get("time")
        usl = config.get("usl")
        lsl = config.get("lsl")
        window = int(config.get("window", 50))
        target_cpk = float(config.get("target_cpk", 1.33))

        if not meas_col or meas_col not in df.columns:
            result["summary"] = "Error: Specify a measurement column."
            return result
        if usl is None and lsl is None:
            result["summary"] = "Error: Specify at least one spec limit (USL or LSL)."
            return result

        usl = float(usl) if usl is not None else None
        lsl = float(lsl) if lsl is not None else None

        data = df[meas_col].dropna().values.astype(float)
        if time_col and time_col in df.columns:
            t_data = df[time_col].loc[df[meas_col].notna()].values.astype(float)
        else:
            t_data = np.arange(len(data), dtype=float)

        n = len(data)
        if n < window:
            result["summary"] = f"Error: Need at least {window} observations."
            return result

        # Rolling Cpk
        cpk_values = []
        t_centers = []
        for i in range(0, n - window + 1, max(1, window // 4)):
            seg = data[i:i + window]
            mu = np.mean(seg)
            sigma = np.std(seg, ddof=1)
            if sigma < 1e-12:
                sigma = 1e-12
            if usl is not None and lsl is not None:
                cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
            elif usl is not None:
                cpk = (usl - mu) / (3 * sigma)
            else:
                cpk = (mu - lsl) / (3 * sigma)
            cpk_values.append(float(cpk))
            t_centers.append(float(np.mean(t_data[i:i + window])))

        cpk_arr = np.array(cpk_values)
        t_arr = np.array(t_centers)

        # Fit linear trend to Cpk vs time
        slope, intercept, r_val, p_val, se = sp_stats.linregress(t_arr, cpk_arr)
        cpk_trend = intercept + slope * t_arr

        # Find crossover point where Cpk drops below target
        crossover = None
        if slope < 0 and intercept > target_cpk:
            crossover = (target_cpk - intercept) / slope
            if crossover < t_arr[0] or crossover > t_arr[-1] * 3:
                crossover = None

        cpk_start = float(cpk_trend[0])
        cpk_end = float(cpk_trend[-1])
        cpk_overall = float(np.mean(cpk_arr))
        degrading = slope < 0 and p_val < 0.05

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>DEGRADATION-AWARE CAPABILITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Variable:<</COLOR>> {meas_col}\n"
        summary += f"<<COLOR:text>>Spec limits:<</COLOR>> "
        if lsl is not None:
            summary += f"LSL={lsl}"
        if usl is not None:
            summary += f"{', ' if lsl is not None else ''}USL={usl}"
        summary += f"\n<<COLOR:text>>Window:<</COLOR>> {window}    Points: {len(cpk_values)}\n\n"
        summary += f"<<COLOR:accent>>\u2500\u2500 Cpk Trend \u2500\u2500<</COLOR>>\n"
        summary += f"  Starting Cpk: {cpk_start:.3f}\n"
        summary += f"  Ending Cpk:   {cpk_end:.3f}\n"
        summary += f"  Overall mean: {cpk_overall:.3f}\n"
        summary += f"  Slope: {slope:.6f} per unit time (p={p_val:.4f})\n"
        if degrading:
            summary += f"  <<COLOR:warning>>Significant degradation detected<</COLOR>>\n"
        if crossover is not None:
            summary += f"  Cpk reaches {target_cpk} at t = {crossover:.1f}\n"

        result["summary"] = summary
        result["statistics"] = {
            "cpk_start": cpk_start, "cpk_end": cpk_end, "cpk_overall": cpk_overall,
            "slope": float(slope), "p_value": float(p_val), "r_squared": float(r_val**2),
            "crossover_time": crossover, "degrading": degrading,
        }

        _dc_status = "degrading" if degrading else "stable"
        result["guide_observation"] = f"Degradation Cpk: {_dc_status}. Start={cpk_start:.3f}, end={cpk_end:.3f}, slope={slope:.6f}/unit."

        _dc_cross = ""
        if crossover is not None:
            _dc_cross = f" At this rate, Cpk drops below {target_cpk} at t = {crossover:.0f}."
        result["narrative"] = _narrative(
            f"Degradation Capability \u2014 Cpk is {'degrading' if degrading else 'stable'}",
            f"Rolling Cpk starts at {cpk_start:.3f} and {'decays to' if degrading else 'remains near'} {cpk_end:.3f} "
            f"(slope = {slope:.6f}/unit, p = {p_val:.4f}).{_dc_cross}"
            + (" Classical Cpk assumes stationarity and would report {:.3f} \u2014 masking the degradation.".format(cpk_overall) if degrading else ""),
            next_steps="If degrading, schedule tool changes or maintenance before Cpk crosses the target. "
                       + (f"Recommended action point: t = {crossover:.0f} to maintain Cpk > {target_cpk}." if crossover else ""),
            chart_guidance="The rolling Cpk curve shows capability over time. The trend line reveals whether the process is degrading. "
                          "The dashed red line is the target Cpk."
        )

        # Plot: Cpk degradation curve
        result["plots"].append({
            "title": f"Capability Degradation ({meas_col})",
            "data": [
                {"type": "scatter", "x": t_arr.tolist(), "y": cpk_arr.tolist(),
                 "mode": "markers", "marker": {"size": 5, "color": ["#dc5050" if c < target_cpk else "#4a9f6e" for c in cpk_arr]},
                 "name": "Rolling Cpk"},
                {"type": "scatter", "x": t_arr.tolist(), "y": cpk_trend.tolist(),
                 "mode": "lines", "line": {"color": "#4a90d9", "width": 2, "dash": "solid"}, "name": "Trend"},
            ],
            "layout": {
                "height": 290,
                "xaxis": {"title": "Time / Sequence", "rangeslider": {"visible": True, "thickness": 0.12}},
                "yaxis": {"title": "Cpk"},
                "shapes": [
                    {"type": "line", "x0": float(t_arr[0]), "x1": float(t_arr[-1]), "y0": target_cpk, "y1": target_cpk,
                     "line": {"color": "#dc5050", "dash": "dash", "width": 1}},
                ],
            },
        })

    return result


