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
        return _entropy(df, config)

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

    return _convert_spc(result, analysis_id)


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

    stats = {
        "in_control": getattr(result, "in_control", True),
        "n_ooc": len(getattr(result, "out_of_control", [])),
    }
    limits = getattr(result, "limits", None)
    if limits:
        for k in ("ucl", "cl", "lcl"):
            v = getattr(limits, k, None)
            if v is not None:
                stats[k] = round(float(v), 4)

    ooc = getattr(result, "out_of_control", [])
    summary = (
        getattr(result, "summary", "") or f"SPC: {'In control' if stats['in_control'] else f'{len(ooc)} OOC points'}"
    )

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
            data, usl=usl, lsl=lsl, cpk=getattr(result, "cpk", None), ppk=getattr(result, "ppk", None)
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
    col = config.get("measurement") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

    try:
        from forgespc.conformal import conformal_control

        result = conformal_control(data)
        return _convert_spc(result, analysis_id)
    except Exception as e:
        return {"summary": f"Conformal SPC error: {e}", "charts": [], "statistics": {}}


def _entropy(df, config):
    col = config.get("measurement") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

    try:
        from forgespc.conformal import entropy_spc

        result = entropy_spc(data)
        return _convert_spc(result, analysis_id="entropy_spc")
    except Exception as e:
        return {"summary": f"Entropy SPC error: {e}", "charts": [], "statistics": {}}
