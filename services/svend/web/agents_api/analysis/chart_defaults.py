"""DSW Chart Defaults — canonical chart styling for all DSW analyses.

Every Plotly plot dict returned by DSW analyses passes through
apply_chart_defaults() in the post-processor to enforce consistent
height, margins, legend placement, colors, and transparency.

Trace builder functions produce pre-styled Plotly trace dicts that
analysis code can use directly.

CR: 5528303a — INIT-009 / E9-003
"""

from .common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
    _rgba,
)

# ── Allowed palette (for validation) ──────────────────────────────────────

ALLOWED_HEX = {c.lower() for c in SVEND_COLORS} | {
    c.lower()
    for c in [
        COLOR_GOOD,
        COLOR_BAD,
        COLOR_WARNING,
        COLOR_INFO,
        COLOR_NEUTRAL,
        COLOR_REFERENCE,
        COLOR_GOLD,
    ]
}

# ── Off-palette → palette normalization map (VIS-001 §4.2) ───────────────
# Maps commonly used off-palette hex colors to their nearest palette equivalent.
# Applied by _normalize_color() in the post-processing pipeline.

_COLOR_NORMALIZE = {
    # Green variants → COLOR_GOOD (#4a9f6e)
    "#00b894": COLOR_GOOD,
    "#2ecc71": COLOR_GOOD,
    "#27ae60": COLOR_GOOD,
    "#5ad4a0": COLOR_GOOD,
    "#3d8a5c": COLOR_GOOD,
    "#2c5f2d": COLOR_GOOD,
    "#2a6b3a": COLOR_GOOD,
    # Red variants → COLOR_BAD (#d06060)
    "#d63031": COLOR_BAD,
    "#e85747": COLOR_BAD,
    "#d94a4a": COLOR_BAD,
    "#dc5050": COLOR_BAD,
    "#e74c3c": COLOR_BAD,
    "#ff7675": COLOR_BAD,
    "#9f4a4a": COLOR_BAD,
    "#e84747": COLOR_BAD,
    # Blue variants → COLOR_INFO (#4a9faf) or SVEND_COLORS[9] (#5b8bd6)
    "#47a5e8": COLOR_INFO,
    "#4a90d9": "#5b8bd6",
    "#5a9fd4": "#5b8bd6",
    "#5b9bd5": "#5b8bd6",
    "#3498db": "#5b8bd6",
    "#6ab7d4": COLOR_INFO,
    # Teal variants → SVEND_COLORS[6] (#4ac9c0)
    "#1abc9c": "#4ac9c0",
    "#47e8c4": "#4ac9c0",
    # Purple variants → SVEND_COLORS[4] (#8a7fbf)
    "#6c5ce7": "#8a7fbf",
    "#9b59b6": "#8a7fbf",
    "#7a5fb8": "#8a7fbf",
    "#7d4ad9": "#8a7fbf",
    "#7a6a9a": "#8a7fbf",
    "#6a7fff": "#8a7fbf",
    # Pink variants → SVEND_COLORS[5] (#d66b9f)
    "#ff7eb9": "#d66b9f",
    "#e84393": "#d66b9f",
    "#d94a99": "#d66b9f",
    # Orange/warning variants → COLOR_WARNING (#e89547)
    "#f39c12": COLOR_WARNING,
    "#e67e22": COLOR_WARNING,
    "#e8953f": COLOR_WARNING,
    # Gold variants → COLOR_GOLD (#e8c547)
    "#fdcb6e": COLOR_GOLD,
    "#c4e847": COLOR_GOLD,
    # Brown variants → SVEND_COLORS[7] (#c97a4a)
    "#c75a3a": "#c97a4a",
    # Neutral/gray variants → COLOR_NEUTRAL (#888888)
    "#9aaa9a": COLOR_NEUTRAL,
    "#5a6a5a": COLOR_NEUTRAL,
    "#b0b0b0": COLOR_NEUTRAL,
    "#7a8f7a": COLOR_NEUTRAL,
    # Reference variants → COLOR_REFERENCE (#d4a24a)
    "#d9a04a": COLOR_REFERENCE,
    "#d4a05a": COLOR_REFERENCE,
}

# ── Standard layout constants ──────────────────────────────────────────────

CHART_HEIGHT = 300
CHART_HEIGHT_MULTI = 350  # multi-panel / subplots
CHART_MARGINS = {"l": 60, "r": 20, "t": 40, "b": 60}
CHART_FONT = {"family": "Inter, system-ui, sans-serif", "size": 12}
LEGEND_DEFAULTS = {
    "orientation": "h",
    "yanchor": "bottom",
    "y": -0.25,
    "xanchor": "left",
    "x": 0,
}


def _normalize_color(color):
    """Normalize an off-palette color to its nearest palette equivalent (VIS-001 §4.2).

    Returns the normalized color string. Palette colors and rgba() of palette
    colors pass through unchanged. None/empty pass through unchanged.
    """
    if not color or not isinstance(color, str):
        return color

    s = color.strip().lower()

    # Direct hex match
    if s.startswith("#") and len(s) == 7:
        return _COLOR_NORMALIZE.get(s, color)

    # rgba() — normalize the base RGB if off-palette
    if s.startswith("rgba(") or s.startswith("rgb("):
        import re

        m = re.match(
            r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([0-9.]+))?\s*\)", s
        )
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            base_hex = f"#{r:02x}{g:02x}{b:02x}"
            replacement = _COLOR_NORMALIZE.get(base_hex)
            if replacement:
                alpha = m.group(4)
                if alpha is not None:
                    return _rgba(replacement, float(alpha))
                return replacement
    return color


def _normalize_trace_colors(trace):
    """Normalize all color fields in a trace dict to palette equivalents."""
    if not isinstance(trace, dict):
        return

    # marker.color
    marker = trace.get("marker")
    if isinstance(marker, dict):
        mc = marker.get("color")
        if isinstance(mc, str):
            marker["color"] = _normalize_color(mc)
        # marker.line.color
        ml = marker.get("line")
        if isinstance(ml, dict) and isinstance(ml.get("color"), str):
            ml["color"] = _normalize_color(ml["color"])

    # line.color
    line = trace.get("line")
    if isinstance(line, dict) and isinstance(line.get("color"), str):
        line["color"] = _normalize_color(line["color"])

    # fillcolor
    fc = trace.get("fillcolor")
    if isinstance(fc, str):
        trace["fillcolor"] = _normalize_color(fc)


def apply_chart_defaults(plot_dict):
    """Apply standard styling to a Plotly plot dict (in-place, returns dict).

    Handles both full figure dicts (with layout key) and bare-layout dicts.
    Enforces VIS-001: colors, legend, backgrounds, dimensions, font.
    """
    if not isinstance(plot_dict, dict):
        return plot_dict

    layout = plot_dict.get("layout", plot_dict)

    # Height — only set if not explicitly specified
    if "height" not in layout:
        # Check if subplots (multiple axes)
        has_subplots = any(k.startswith("yaxis") and k != "yaxis" for k in layout)
        layout["height"] = CHART_HEIGHT_MULTI if has_subplots else CHART_HEIGHT

    # Margins — merge (keep explicit overrides)
    if "margin" not in layout:
        layout["margin"] = dict(CHART_MARGINS)
    else:
        for k, v in CHART_MARGINS.items():
            layout["margin"].setdefault(k, v)

    # Font
    layout.setdefault("font", dict(CHART_FONT))

    # Transparent background — OVERRIDE non-transparent (VIS-001 §6.1)
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"
    layout["plot_bgcolor"] = "rgba(0,0,0,0)"

    # Legend enforcement (VIS-001 §5)
    legend = layout.get("legend")
    if legend is None:
        layout["legend"] = dict(LEGEND_DEFAULTS)
    elif isinstance(legend, dict):
        # Fix inside-chart legends (VIS-001 §5.2)
        y = legend.get("y")
        orientation = legend.get("orientation", "v")
        if y is not None and 0 < y < 1 and orientation != "h":
            # Legend is inside chart body — replace with standard position
            layout["legend"] = dict(LEGEND_DEFAULTS)

    # Grid styling
    for axis_key in [k for k in layout if k.startswith(("xaxis", "yaxis"))]:
        axis = layout[axis_key]
        if isinstance(axis, dict):
            axis.setdefault("gridcolor", "rgba(128,128,128,0.15)")
            axis.setdefault("zerolinecolor", "rgba(128,128,128,0.25)")

    # Default axes (if no xaxis2/yaxis2 etc.)
    for ax in ("xaxis", "yaxis"):
        if ax not in layout:
            layout[ax] = {}
        layout[ax].setdefault("gridcolor", "rgba(128,128,128,0.15)")
        layout[ax].setdefault("zerolinecolor", "rgba(128,128,128,0.25)")

    # Enforce SVEND_COLORS on traces + normalize off-palette colors (VIS-001 §4)
    data = plot_dict.get("data", [])
    for i, trace in enumerate(data):
        if isinstance(trace, dict):
            _normalize_trace_colors(trace)
            _apply_trace_colors(trace, i)

    return plot_dict


def _apply_trace_colors(trace, index):
    """Apply SVEND_COLORS to a trace if it has no explicit color."""
    color = SVEND_COLORS[index % len(SVEND_COLORS)]
    trace_type = trace.get("type", "scatter")

    # Skip if color already explicitly set
    if trace_type in ("scatter", "scattergl"):
        marker = trace.get("marker", {})
        line = trace.get("line", {})
        if not marker.get("color") and not line.get("color"):
            trace.setdefault("marker", {})["color"] = color
            if trace.get("mode", "").startswith("lines"):
                trace.setdefault("line", {})["color"] = color
    elif trace_type == "bar":
        marker = trace.get("marker", {})
        if not marker.get("color"):
            trace.setdefault("marker", {})["color"] = color
    elif trace_type == "histogram":
        marker = trace.get("marker", {})
        if not marker.get("color"):
            trace.setdefault("marker", {})["color"] = _rgba(color, 0.7)
            trace["marker"].setdefault("line", {"color": color, "width": 1})
    elif trace_type == "box":
        marker = trace.get("marker", {})
        if not marker.get("color"):
            trace.setdefault("marker", {})["color"] = color
            trace.setdefault("line", {})["color"] = color
            trace.setdefault("fillcolor", _rgba(color, 0.3))


# ── Trace builders ─────────────────────────────────────────────────────────


def histogram_trace(x, name=None, color_index=0, **kwargs):
    """Pre-styled histogram trace."""
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    trace = {
        "type": "histogram",
        "x": x if isinstance(x, list) else list(x),
        "marker": {
            "color": _rgba(c, 0.7),
            "line": {"color": c, "width": 1},
        },
        "opacity": 0.85,
    }
    if name:
        trace["name"] = name
    trace.update(kwargs)
    return trace


def boxplot_trace(y, name=None, color_index=0, **kwargs):
    """Pre-styled boxplot trace."""
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    trace = {
        "type": "box",
        "y": y if isinstance(y, list) else list(y),
        "marker": {"color": c},
        "line": {"color": c},
        "fillcolor": _rgba(c, 0.3),
    }
    if name:
        trace["name"] = name
    trace.update(kwargs)
    return trace


def scatter_trace(x, y, name=None, color_index=0, mode="markers", **kwargs):
    """Pre-styled scatter trace."""
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    trace = {
        "type": "scatter",
        "x": x if isinstance(x, list) else list(x),
        "y": y if isinstance(y, list) else list(y),
        "mode": mode,
        "marker": {"color": c, "size": 6},
    }
    if "lines" in mode:
        trace["line"] = {"color": c, "width": 2}
    if name:
        trace["name"] = name
    trace.update(kwargs)
    return trace


def line_trace(x, y, name=None, color_index=0, dash=None, **kwargs):
    """Pre-styled line trace."""
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    line_style = {"color": c, "width": 2}
    if dash:
        line_style["dash"] = dash
    trace = {
        "type": "scatter",
        "x": x if isinstance(x, list) else list(x),
        "y": y if isinstance(y, list) else list(y),
        "mode": "lines",
        "line": line_style,
    }
    if name:
        trace["name"] = name
    trace.update(kwargs)
    return trace


def bar_trace(x, y, name=None, color_index=0, **kwargs):
    """Pre-styled bar trace."""
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    trace = {
        "type": "bar",
        "x": x if isinstance(x, list) else list(x),
        "y": y if isinstance(y, list) else list(y),
        "marker": {"color": c},
    }
    if name:
        trace["name"] = name
    trace.update(kwargs)
    return trace


def heatmap_trace(z, x=None, y=None, colorscale=None, **kwargs):
    """Pre-styled heatmap trace."""
    if colorscale is None:
        # SVEND green-to-red diverging
        colorscale = [
            [0, COLOR_GOOD],
            [0.5, COLOR_NEUTRAL],
            [1, COLOR_BAD],
        ]
    trace = {
        "type": "heatmap",
        "z": z if isinstance(z, list) else [list(row) for row in z],
        "colorscale": colorscale,
    }
    if x is not None:
        trace["x"] = x if isinstance(x, list) else list(x)
    if y is not None:
        trace["y"] = y if isinstance(y, list) else list(y)
    trace.update(kwargs)
    return trace


def control_chart_trace(
    x, y, ucl, lcl, cl=None, name="Observations", color_index=0, **kwargs
):
    """Pre-styled control chart (observation line + control limits).

    Returns a list of traces: [observations, UCL, LCL, optional CL].
    """
    c = SVEND_COLORS[color_index % len(SVEND_COLORS)]
    x_list = x if isinstance(x, list) else list(x)
    traces = [
        {
            "type": "scatter",
            "x": x_list,
            "y": y if isinstance(y, list) else list(y),
            "mode": "lines+markers",
            "name": name,
            "marker": {"color": c, "size": 5},
            "line": {"color": c, "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_list,
            "y": ucl if isinstance(ucl, list) else [ucl] * len(x_list),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
        },
        {
            "type": "scatter",
            "x": x_list,
            "y": lcl if isinstance(lcl, list) else [lcl] * len(x_list),
            "mode": "lines",
            "name": "LCL",
            "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
        },
    ]
    if cl is not None:
        traces.append(
            {
                "type": "scatter",
                "x": x_list,
                "y": cl if isinstance(cl, list) else [cl] * len(x_list),
                "mode": "lines",
                "name": "CL",
                "line": {"color": COLOR_REFERENCE, "width": 1, "dash": "dot"},
            }
        )
    return traces


def reference_line(value, orientation="h", name="Reference", **kwargs):
    """Horizontal or vertical reference line shape dict for layout.shapes."""
    if orientation == "h":
        shape = {
            "type": "line",
            "x0": 0,
            "x1": 1,
            "xref": "paper",
            "y0": value,
            "y1": value,
            "line": {"color": COLOR_REFERENCE, "width": 1.5, "dash": "dash"},
        }
    else:
        shape = {
            "type": "line",
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "x0": value,
            "x1": value,
            "line": {"color": COLOR_REFERENCE, "width": 1.5, "dash": "dash"},
        }
    shape.update(kwargs)
    return shape
