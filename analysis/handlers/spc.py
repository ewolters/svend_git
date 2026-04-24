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
    except Exception as e:
        logger.exception("forgespc call failed: %s", analysis_id)
        return {"summary": f"SPC error: {e}", "charts": [], "statistics": {}}

    return _convert_spc(result, analysis_id)


def _build_kwargs(df, analysis_id, config, col):
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    kwargs = {"data": data}

    if analysis_id in ("p_chart", "np_chart"):
        kwargs = {"defectives": data}
        ss_col = config.get("sample_size")
        if ss_col and ss_col in df.columns:
            kwargs["sample_sizes"] = pd.to_numeric(df[ss_col], errors="coerce").dropna().tolist()
    elif analysis_id in ("c_chart", "u_chart"):
        kwargs = {"defects": data}
        if analysis_id == "u_chart":
            units_col = config.get("units")
            if units_col and units_col in df.columns:
                kwargs["units"] = pd.to_numeric(df[units_col], errors="coerce").dropna().tolist()
    elif analysis_id in ("xbar_r", "xbar_s"):
        kwargs["subgroup_size"] = int(config.get("subgroup_size", 5))

    hist_mean = config.get("historical_mean")
    hist_sigma = config.get("historical_sigma")
    if hist_mean:
        kwargs["center"] = float(hist_mean)
    if hist_sigma:
        kwargs["sigma"] = float(hist_sigma)

    return kwargs


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
