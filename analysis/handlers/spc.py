"""SPC handler — control charts and capability via forgespc + forgeviz."""

import logging

import pandas as pd
from forgeviz.charts.control import from_spc_result, from_spc_result_pair

logger = logging.getLogger(__name__)

_CHART_FUNCS = {
    "imr": ("forgespc.charts", "individuals_moving_range_chart"),
    "xbar_r": ("forgespc.charts", "xbar_r_chart"),
    "xbar_s": ("forgespc.advanced", "xbar_s_chart"),
    "p_chart": ("forgespc.charts", "p_chart"),
    "np_chart": ("forgespc.charts", "np_chart"),
    "c_chart": ("forgespc.charts", "c_chart"),
    "u_chart": ("forgespc.charts", "u_chart"),
    "cusum": ("forgespc.advanced", "cusum_chart"),
    "ewma": ("forgespc.advanced", "ewma_chart"),
    "laney_p": ("forgespc.charts", "laney_p_chart"),
    "laney_u": ("forgespc.charts", "laney_u_chart"),
    "mewma": ("forgespc.advanced", "mewma_chart"),
    "moving_average": ("forgespc.charts", "moving_average_chart"),
    "zone_chart": ("forgespc.charts", "zone_chart"),
    "generalized_variance": ("forgespc.advanced", "generalized_variance_chart"),
}

_CAPABILITY = {"capability", "nonnormal_capability", "between_within", "capability_sixpack"}


def run(df, analysis_id, config):
    """Run SPC analysis via forgespc, return ForgeViz ChartSpecs."""
    import importlib

    if analysis_id in _CAPABILITY:
        return _capability(df, config)

    if analysis_id in ("conformal_control", "conformal_monitor"):
        return _conformal(df, analysis_id, config)

    if analysis_id in ("entropy_spc",):
        return _entropy(df, analysis_id, config)

    if analysis_id == "degradation_capability":
        return _degradation(df, config)

    entry = _CHART_FUNCS.get(analysis_id)

    if not entry:
        return {
            "summary": f"SPC analysis '{analysis_id}' not yet in forge-native dispatch.",
            "charts": [],
            "statistics": {},
        }

    module_path, func_name = entry
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        return {"summary": f"forgespc function not available: {e}", "charts": [], "statistics": {}}

    # Multivariate analyses use 'variables' list, not single column
    multivariate = analysis_id in ("mewma", "generalized_variance")
    if multivariate:
        variables = config.get("variables") or []
        if not variables or len(variables) < 2:
            return {"summary": "Error: Select 2+ measurement variables.", "charts": [], "statistics": {}}
        return _multivariate_spc(df, analysis_id, config, variables)

    col = config.get("measurement") or config.get("column") or config.get("var")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}

    try:
        kwargs = _build_kwargs(df, analysis_id, config, col)
        result = func(**kwargs)
        # MEWMA and GenVar return custom types — convert to ControlChartResult
        if hasattr(result, "to_chart_result") and not hasattr(result, "limits"):
            result = result.to_chart_result()
    except Exception as e:
        logger.exception("forgespc call failed: %s", analysis_id)
        return {"summary": f"SPC error: {e}", "charts": [], "statistics": {}}

    out = _convert_spc(result, analysis_id)
    out["_config"] = config
    return out


def _build_kwargs(df, analysis_id, config, col):

    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    kwargs = {"data": data}

    if analysis_id in ("p_chart", "np_chart", "laney_p"):
        int_data = [int(round(x)) for x in data]
        ss_col = config.get("sample_size")
        if ss_col and ss_col in df.columns:
            sizes = pd.to_numeric(df[ss_col], errors="coerce").dropna().tolist()
        else:
            sizes = [100] * len(int_data)
        if analysis_id == "laney_p":
            kwargs = {"defectives": int_data, "sample_sizes": [int(s) for s in sizes]}
        elif analysis_id == "np_chart":
            kwargs = {"defective_counts": int_data, "sample_size": int(sizes[0]) if sizes else 100}
        else:
            kwargs = {"defectives": int_data, "sample_sizes": [int(s) for s in sizes]}

    elif analysis_id in ("c_chart", "u_chart", "laney_u"):
        int_data = [int(round(x)) for x in data]
        if analysis_id in ("u_chart", "laney_u"):
            units_col = config.get("units") or config.get("inspection_units")
            if units_col and units_col in df.columns:
                units = pd.to_numeric(df[units_col], errors="coerce").dropna().tolist()
            else:
                units = [1.0] * len(int_data)
            kwargs = {"defect_counts": int_data, "inspection_units": units}
        else:
            kwargs = {"defect_counts": int_data}

    elif analysis_id in ("xbar_r", "xbar_s"):
        sg_size = int(config.get("subgroup_size", 5))
        subgroups = [data[i : i + sg_size] for i in range(0, len(data) - sg_size + 1, sg_size)]
        if len(subgroups) < 2:
            subgroups = [data[:sg_size], data[sg_size : 2 * sg_size]] if len(data) >= 2 * sg_size else [data]
        kwargs = {"subgroups": subgroups}

    elif analysis_id == "moving_average":
        window = int(config.get("window", config.get("span", 5)))
        kwargs = {"data": data, "window": window}

    elif analysis_id == "zone_chart":
        kwargs = {"data": data}

    elif analysis_id == "mewma":
        # Multivariate — collect all numeric columns or vars list
        cols = config.get("vars") or df.select_dtypes(include="number").columns.tolist()[:5]
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            cols = [col]
        mv_data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().values.tolist()
        lam = float(config.get("lambda_param", config.get("lambda", 0.2)))
        kwargs = {"data": mv_data, "lambda_param": lam}

    elif analysis_id == "generalized_variance":
        cols = config.get("vars") or df.select_dtypes(include="number").columns.tolist()[:5]
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            cols = [col]
        mv_data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().values
        sg_size = int(config.get("subgroup_size", 5))
        subgroups = [mv_data[i : i + sg_size].tolist() for i in range(0, len(mv_data) - sg_size + 1, sg_size)]
        kwargs = {"subgroups": subgroups}

    else:
        # Default: pass data + optional spec limits
        usl = config.get("usl") or config.get("USL")
        lsl = config.get("lsl") or config.get("LSL")
        if usl:
            kwargs["usl"] = float(usl)
        if lsl:
            kwargs["lsl"] = float(lsl)

    # Historical limits for Phase 2 monitoring
    hist_mean = config.get("historical_mean")
    hist_sigma = config.get("historical_sigma")
    if hist_mean and "historical_mean" in _get_func_params(analysis_id):
        kwargs["historical_mean"] = float(hist_mean)
    if hist_sigma and "historical_sigma" in _get_func_params(analysis_id):
        kwargs["historical_sigma"] = float(hist_sigma)

    return kwargs


def _get_func_params(analysis_id):
    """Get parameter names for a chart function (for safe kwarg passing)."""
    import inspect

    entry = _CHART_FUNCS.get(analysis_id)
    if not entry:
        return set()
    try:
        import importlib

        mod = importlib.import_module(entry[0])
        func = getattr(mod, entry[1])
        return set(inspect.signature(func).parameters.keys())
    except Exception:
        return set()


def _degradation(df, config):
    """Degradation capability — track Cpk over time."""
    from forgespc.capability import degradation_capability

    col = config.get("measurement") or config.get("column") or config.get("var")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    usl = float(config.get("usl") or config.get("USL") or 0)
    lsl = float(config.get("lsl") or config.get("LSL") or 0)
    window = int(config.get("window_size", 20))

    try:
        result = degradation_capability(data, usl=usl, lsl=lsl, window_size=window)
    except Exception as e:
        return {"summary": f"Degradation error: {e}", "charts": [], "statistics": {}}

    from forgeviz.core.spec import ChartSpec

    spec = ChartSpec(
        title="Capability Degradation",
        x_axis={"label": "Window"},
        y_axis={"label": "Cpk"},
    )
    spec.add_trace(result["time_indices"], result["cpk_values"], name="Cpk", color="#4a9f6e", width=2)
    spec.add_reference_line(1.33, color="#4a9f6e", dash="dashed", label="Cpk=1.33")
    spec.add_reference_line(1.0, color="#d94a4a", dash="dashed", label="Cpk=1.0")

    stats = {k: v for k, v in result.items() if k not in ("cpk_values", "time_indices", "summary")}
    return {"charts": [spec], "statistics": stats, "summary": result["summary"]}


def _convert_spc(result, analysis_id):
    """Convert forgespc ControlChartResult to handler output."""
    charts = []
    try:
        if hasattr(result, "secondary_chart") and result.secondary_chart:
            charts = [from_spc_result_pair(result, title=analysis_id.upper().replace("_", " "))]
            if isinstance(charts[0], list):
                charts = charts[0]
        else:
            charts = [from_spc_result(result)]
    except Exception:
        logger.debug("ForgeViz SPC chart conversion failed", exc_info=True)

    ooc = getattr(result, "out_of_control", [])
    violations = getattr(result, "run_violations", [])

    stats = {
        "in_control": getattr(result, "in_control", True),
        "n_ooc": len(ooc),
        "n_run_violations": len(violations),
        "out_of_control": ooc[:20],  # Cap for JSON size
        "run_violations": violations[:20],
    }
    limits = getattr(result, "limits", None)
    if limits:
        for k in ("ucl", "cl", "lcl"):
            v = getattr(limits, k, None)
            if v is not None:
                stats[k] = round(float(v), 4)

    summary = (
        getattr(result, "summary", "") or f"SPC: {'In control' if stats['in_control'] else f'{len(ooc)} OOC points'}"
    )
    if violations:
        rules_hit = sorted(set(v.get("rule", 0) for v in violations))
        summary += f" Nelson rules triggered: {rules_hit}."

    return {
        "charts": charts,
        "statistics": stats,
        "summary": summary,
    }


def _capability(df, config):
    col = config.get("measurement") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}

    usl = config.get("usl") or config.get("USL")
    lsl = config.get("lsl") or config.get("LSL")
    if usl is None or lsl is None:
        return {"summary": "Error: USL and LSL required for capability.", "charts": [], "statistics": {}}

    usl, lsl = float(usl), float(lsl)
    if usl <= lsl:
        return {"summary": f"Error: USL ({usl}) must be > LSL ({lsl}).", "charts": [], "statistics": {}}

    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

    try:
        from forgespc.capability import calculate_capability

        result = calculate_capability(data, usl=usl, lsl=lsl)

        from forgeviz.charts.capability import capability_histogram

        chart = capability_histogram(
            data, usl=usl, lsl=lsl, cpk=getattr(result, "cpk", None), cp=getattr(result, "cp", None)
        )

        stats = {}
        for k in ("cp", "cpk", "pp", "ppk", "cpm", "percent_out"):
            v = getattr(result, k, None)
            if v is not None:
                stats[k] = round(float(v), 4)

        return {
            "charts": [chart],
            "statistics": stats,
            "summary": f"Capability: Cpk = {stats.get('cpk', 'N/A')}, Ppk = {stats.get('ppk', 'N/A')}",
        }
    except Exception as e:
        return {"summary": f"Capability error: {e}", "charts": [], "statistics": {}}


def _conformal(df, analysis_id, config):
    """Conformal SPC — forge-native via forgespc + forgeviz."""
    if analysis_id == "conformal_monitor":
        return _conformal_monitor(df, config)

    col = config.get("measurement") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

    if len(data) < 20:
        return {"summary": "Need at least 20 observations for conformal control chart.", "charts": [], "statistics": {}}

    try:
        from forgespc.conformal import conformal_control
        from forgeviz.charts.control import from_conformal_result

        alpha = float(config.get("alpha", 0.05))
        result = conformal_control(data, alpha=alpha) if "alpha" in config else conformal_control(data)
        charts = from_conformal_result(result, title=f"Conformal Control — {col}")

        stats = {
            "in_control": result.in_control,
            "n_ooc": result.n_ooc,
            "threshold": round(result.threshold, 4),
            "n_calibration": result.n_calibration,
            "n_monitoring": result.n_monitoring,
            "alpha": result.alpha,
        }

        n_ooc = result.n_ooc
        summary = f"{'In control' if result.in_control else f'{n_ooc} OOC point(s) detected'} — conformal prediction intervals (α={result.alpha})."

        return {"charts": charts, "statistics": stats, "summary": summary}
    except Exception as e:
        logger.exception("Conformal control error")
        return {"summary": f"Conformal SPC error: {e}", "charts": [], "statistics": {}}


def _conformal_monitor(df, config):
    """Conformal multivariate monitor — delegates to legacy (complex multi-chart output)."""
    # Conformal monitor has complex multi-plot output (p-values, scores, heatmap)
    # that forgespc doesn't yet produce natively. Delegate to legacy + convert.
    try:
        from agents_api.analysis.spc.conformal import run_conformal_monitor

        result = run_conformal_monitor(df, config)
        result["charts"] = _plotly_to_chartspec(result.get("plots", []))
        result.pop("plots", None)
        return result
    except Exception as e:
        logger.exception("Conformal monitor error")
        return {"summary": f"Conformal monitor error: {e}", "charts": [], "statistics": {}}


def _multivariate_spc(df, analysis_id, config, variables):
    """MEWMA and Generalized Variance — forge-native via forgespc + forgeviz."""
    try:
        valid_cols = [v for v in variables if v in df.columns]
        if len(valid_cols) < 2:
            return {"summary": "Error: Need 2+ valid numeric columns.", "charts": [], "statistics": {}}

        data_arrays = [pd.to_numeric(df[c], errors="coerce").dropna().tolist() for c in valid_cols]
        # Align lengths
        min_len = min(len(a) for a in data_arrays)
        data_arrays = [a[:min_len] for a in data_arrays]

        if analysis_id == "mewma":
            from forgespc.advanced import mewma_chart
            from forgeviz.charts.control import from_mewma_result

            result = mewma_chart(data_arrays)
            chart = from_mewma_result(result, title=f"MEWMA — {', '.join(valid_cols[:3])}")

            stats = {
                "in_control": result.in_control,
                "n_ooc": len(result.out_of_control_indices or []),
                "ucl": round(result.ucl, 4),
                "lambda": result.lambda_param,
                "n_vars": result.n_vars,
                "n": result.n,
            }
            summary = f"MEWMA: {'In control' if result.in_control else f'{len(result.out_of_control_indices)} OOC'} — {result.n_vars} variables, λ={result.lambda_param}."
            return {"charts": [chart], "statistics": stats, "summary": summary}

        elif analysis_id == "generalized_variance":
            import numpy as np
            from forgespc.advanced import generalized_variance_chart

            # GenVar expects list of subgroups (each subgroup is n×p matrix)
            subgroup_size = int(config.get("subgroup_size", 5))
            combined = np.column_stack(data_arrays)
            n_obs = len(combined)
            n_subgroups = n_obs // subgroup_size
            if n_subgroups < 5:
                return {
                    "summary": f"Need at least {subgroup_size * 5} observations for subgroup_size={subgroup_size}.",
                    "charts": [],
                    "statistics": {},
                }

            subgroups = [combined[i * subgroup_size : (i + 1) * subgroup_size].tolist() for i in range(n_subgroups)]
            result = generalized_variance_chart(subgroups)

            # GenVar returns GeneralizedVarianceResult — has det_values, ucl, lcl, cl
            from forgeviz.core.colors import STATUS_GREEN, STATUS_RED, get_color
            from forgeviz.core.spec import ChartSpec

            spec = ChartSpec(
                title=f"Generalized Variance |S| — {', '.join(valid_cols[:3])}",
                chart_type="control_chart",
                x_axis={"label": "Subgroup"},
                y_axis={"label": "|S|"},
            )
            x = list(range(1, len(result.gv_values) + 1))
            spec.add_trace(x, result.gv_values, name="|S|", color=get_color(0), width=1.5, marker_size=4)
            spec.add_reference_line(result.ucl, color=STATUS_RED, dash="dashed", label="UCL")
            spec.add_reference_line(result.cl, color=STATUS_GREEN, label="CL")
            if result.lcl > 0:
                spec.add_reference_line(result.lcl, color=STATUS_RED, dash="dashed", label="LCL")

            ooc_idx = [i for i, v in enumerate(result.gv_values) if v > result.ucl or v < result.lcl]
            if ooc_idx:
                spec.add_marker(ooc_idx, color=STATUS_RED, size=8, symbol="circle", label="OOC")

            stats = {
                "in_control": len(ooc_idx) == 0,
                "n_ooc": len(ooc_idx),
                "ucl": round(result.ucl, 4),
                "cl": round(result.cl, 4),
                "n_subgroups": n_subgroups,
                "n_vars": len(valid_cols),
            }
            summary = f"Gen. Variance: {'In control' if not ooc_idx else f'{len(ooc_idx)} OOC'} — {n_subgroups} subgroups, {len(valid_cols)} variables."
            return {"charts": [spec], "statistics": stats, "summary": summary}

    except Exception as e:
        logger.exception("Multivariate SPC error: %s", analysis_id)
        return {"summary": f"Multivariate SPC error: {e}", "charts": [], "statistics": {}}


def _plotly_to_chartspec(plotly_plots):
    """Convert legacy Plotly plot dicts to ForgeViz ChartSpec objects."""
    from forgeviz.core.spec import ChartSpec

    charts = []
    colors = ["#4a9f6e", "#4a90d9", "#e8c547", "#d94a4a", "#a78bfa", "#f472b6", "#9aaa9a"]

    for plot in plotly_plots:
        if not plot or not plot.get("data"):
            continue
        title = plot.get("title", "")
        layout = plot.get("layout", {})
        x_title = (layout.get("xaxis") or {}).get("title", "")
        y_title = (layout.get("yaxis") or {}).get("title", "")

        spec = ChartSpec(
            title=title,
            x_axis={"label": x_title},
            y_axis={"label": y_title},
        )

        for i, trace in enumerate(plot["data"]):
            trace_type = trace.get("type", "scatter")
            mode = trace.get("mode", "")
            name = trace.get("name", "")
            x = trace.get("x", [])
            y = trace.get("y", [])

            # Skip invisible legend-only traces
            if x == [None] or y == [None]:
                continue

            # Determine color
            marker = trace.get("marker", {})
            line = trace.get("line", {})
            color = None
            if isinstance(marker.get("color"), str):
                color = marker["color"]
            elif isinstance(line.get("color"), str) and line["color"] != "transparent":
                color = line["color"]
            if not color:
                color = colors[i % len(colors)]

            # Determine dash
            dash = line.get("dash", "")

            # Determine trace rendering
            if trace_type == "bar":
                spec.add_trace(x, y, name=name, trace_type="bar", color=color)
            elif trace_type == "heatmap":
                # Heatmap — build ChartSpec directly (forgeviz.heatmap has arg conflicts)
                z = trace.get("z", [])
                h_x = trace.get("x", [])
                h_y = trace.get("y", [])
                h_spec = ChartSpec(
                    title=title,
                    chart_type="heatmap",
                    x_axis={"label": x_title, "categories": h_x},
                    y_axis={"label": "", "categories": h_y},
                )
                h_spec.traces = [
                    {
                        "z": z,
                        "x": h_x,
                        "y": h_y,
                        "trace_type": "heatmap",
                        "colorscale": trace.get("colorscale", [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]]),
                    }
                ]
                charts.append(h_spec)
                spec = None
                break
            elif trace.get("fill") == "toself":
                # Filled band — add as two boundary lines
                # The x/y arrays are doubled (upper then lower reversed)
                n_half = len(x) // 2
                if n_half > 0:
                    spec.add_trace(
                        x[:n_half], y[:n_half], name=name + " (upper)", color="#4a90d9", dash="dotted", width=1
                    )
                    spec.add_trace(
                        x[:n_half], y[n_half:], name=name + " (lower)", color="#4a90d9", dash="dotted", width=1
                    )
            elif "markers" in mode and "lines" not in mode:
                marker_size = marker.get("size", 5)
                spec.add_trace(x, y, name=name, trace_type="scatter", color=color, marker_size=marker_size)
            else:
                width = line.get("width", 1.5)
                spec.add_trace(x, y, name=name, color=color, dash=dash, width=width)

        if spec is not None:
            # Add reference lines from shapes
            for shape in layout.get("shapes", []):
                if shape.get("type") == "line" and shape.get("y0") == shape.get("y1"):
                    spec.add_reference_line(
                        float(shape["y0"]),
                        color=shape.get("line", {}).get("color", "#9aaa9a"),
                        dash=shape.get("line", {}).get("dash", ""),
                    )
            charts.append(spec)

    return charts


def _entropy(df, analysis_id, config):
    """Delegate to legacy entropy SPC handler and convert charts."""
    try:
        from agents_api.analysis.spc.conformal import run_entropy_spc

        result = run_entropy_spc(df, config)
        if result and result.get("plots"):
            result["charts"] = _plotly_to_chartspec(result.get("plots", []))
            result.pop("plots", None)
        return result
    except Exception as e:
        logger.exception("Entropy SPC error")
        return {"summary": f"Entropy SPC error: {e}", "charts": [], "statistics": {}}
