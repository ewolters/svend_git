"""Chart rendering engine — builds Plotly charts from declarative specs.

Consolidates the repetitive pattern: read columns → group by → build traces
→ set layout → return result. Standard charts become ~8-line spec dicts
instead of 40-80 lines of Plotly boilerplate.

Delegates to chart_defaults.py trace builders for consistent styling.

CR: 3c0d0e53
"""

from ..chart_defaults import (
    bar_trace,
    boxplot_trace,
    heatmap_trace,
    histogram_trace,
    line_trace,
    scatter_trace,
)

# ── Trace builder dispatch ────────────────────────────────────────────────

_TRACE_BUILDERS = {
    "histogram": lambda data, spec, ci, **kw: histogram_trace(
        x=data["x"],
        name=kw.get("name"),
        color_index=kw.get("color_index", 0),
        **(spec.get("trace_kw", {})),
    ),
    "box": lambda data, spec, ci, **kw: boxplot_trace(
        y=data["y"],
        name=kw.get("name"),
        color_index=kw.get("color_index", 0),
    ),
    "scatter": lambda data, spec, ci, **kw: scatter_trace(
        x=data["x"],
        y=data["y"],
        name=kw.get("name"),
        color_index=kw.get("color_index", 0),
        mode=spec.get("mode", "markers"),
    ),
    "line": lambda data, spec, ci, **kw: line_trace(
        x=data["x"],
        y=data["y"],
        name=kw.get("name"),
        color_index=kw.get("color_index", 0),
    ),
    "bar": lambda data, spec, ci, **kw: bar_trace(
        x=data["x"],
        y=data["y"],
        name=kw.get("name"),
        color_index=kw.get("color_index", 0),
    ),
    "heatmap": lambda data, spec, ci, **kw: heatmap_trace(
        z=data["z"],
        x=data.get("x_labels"),
        y=data.get("y_labels"),
        colorscale=spec.get("colorscale", "RdBu"),
        **({"zmid": 0} if spec.get("zmid") else {}),
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────


def _resolve_groupby(config, key):
    """Resolve groupby column, treating empty/None/"None" as no grouping."""
    if not key:
        return None
    val = config.get(key)
    if val and val != "" and val != "None":
        return val
    return None


def _extract_data(df, spec, config, group_mask=None):
    """Extract x/y/z data arrays from DataFrame per spec."""
    sub = df if group_mask is None else df.loc[group_mask]
    data = {}

    x_key = spec.get("x")
    y_key = spec.get("y")

    if x_key:
        col = config.get(x_key)
        if col and col in sub.columns:
            data["x"] = sub[col].dropna().tolist()

    if y_key:
        col = config.get(y_key)
        if col and col in sub.columns:
            data["y"] = sub[col].dropna().tolist()

    return data


def _format_title(template, config, spec, group_col=None):
    """Format title template with resolved column names."""
    replacements = {}
    if spec.get("x"):
        replacements["x"] = config.get(spec["x"], "")
    if spec.get("y"):
        replacements["y"] = config.get(spec["y"], "")
    if group_col:
        replacements["group"] = group_col
    # Also support y_cols for timeseries
    if "y_cols" in replacements:
        pass
    try:
        return template.format(**replacements)
    except KeyError:
        return template


# ── Main engine ───────────────────────────────────────────────────────────


def render_chart(df, config, spec, hooks=None):
    """Render a chart from a declarative ChartSpec.

    Args:
        df: pandas DataFrame with data
        config: analysis config dict from frontend
        spec: ChartSpec dict (from specs.py)
        hooks: dict of hook_name → callable (from hooks.py)

    Returns:
        Result dict: {"plots": [...], "summary": str, ...}
    """
    result = {"plots": [], "summary": ""}
    hooks = hooks or {}

    # 1. Pre-compute hook (transforms data, returns extra traces/config)
    extra_traces = []
    pre_hook = spec.get("pre_compute")
    if pre_hook and pre_hook in hooks:
        hook_result = hooks[pre_hook](df, config)
        if hook_result:
            df = hook_result.get("df", df)
            extra_traces = hook_result.get("extra_traces", [])
            config = {**config, **hook_result.get("extra_config", {})}
            # Allow hook to set result directly (for heatmap, etc.)
            if "result" in hook_result:
                return hook_result["result"]

    # 2. Resolve groupby
    group_col = _resolve_groupby(config, spec.get("groupby"))
    trace_type = spec.get("trace_type", "scatter")
    builder = _TRACE_BUILDERS.get(trace_type)

    if not builder:
        result["summary"] = f"Unsupported trace type: {trace_type}"
        return result

    # 3. Build traces
    traces = []
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        for i, group in enumerate(groups):
            mask = df[group_col] == group
            data = _extract_data(df, spec, config, mask)
            if not data:
                continue
            trace = builder(data, spec, config, name=str(group), color_index=i)
            traces.append(trace)
    else:
        data = _extract_data(df, spec, config)
        if data:
            trace = builder(data, spec, config, color_index=0)
            traces.append(trace)

    # 4. Post-compute hook (can append extra traces, e.g. trendline)
    post_hook = spec.get("post_compute")
    if post_hook and post_hook in hooks:
        post_result = hooks[post_hook](df, config, traces)
        if post_result:
            traces.extend(post_result.get("extra_traces", []))
            if "summary" in post_result:
                result["summary"] = post_result["summary"]

    # Add any pre-hook extra traces
    traces.extend(extra_traces)

    # 5. Assemble plot
    if traces:
        title_template = spec.get("title_grouped", spec.get("title", "")) if group_col else spec.get("title", "")
        title = _format_title(title_template, config, spec, group_col)

        layout = {"height": 300}
        layout.update(spec.get("layout", {}))
        if group_col:
            layout["showlegend"] = True
        if spec.get("x") and config.get(spec["x"]):
            layout.setdefault("xaxis", {})["title"] = config.get(spec["x"])
        if spec.get("y") and config.get(spec["y"]):
            layout.setdefault("yaxis", {})["title"] = config.get(spec["y"])

        result["plots"].append({"title": title, "data": traces, "layout": layout})

    # 6. Summary hook
    summary_hook = spec.get("summary_fn")
    if summary_hook and summary_hook in hooks:
        result["summary"] = hooks[summary_hook](df, config, result)

    return result
