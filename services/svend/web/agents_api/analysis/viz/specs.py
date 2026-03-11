"""Declarative chart specifications — data-driven Plotly chart rendering.

Each entry in _CHART_SPECS is a dict describing how to render a chart type.
The engine (engine.py) interprets these specs to build Plotly traces.

Adding a new standard chart = adding one dict entry here.

CR: 3c0d0e53
"""

_CHART_SPECS = {
    # ── Tier 1: Pure specs (engine handles 100%) ──────────────────────────
    "histogram": {
        "trace_type": "histogram",
        "x": "var",
        "groupby": "by",
        "title": "Histogram of {x}",
        "title_grouped": "Histogram of {x} by {group}",
        "layout": {"barmode": "overlay"},
        "trace_kw": {"opacity": 0.7},
    },
    "boxplot": {
        "trace_type": "box",
        "y": "var",
        "groupby": "by",
        "title": "Box Plot of {y}",
        "title_grouped": "Box Plot of {y} by {group}",
    },
    "heatmap": {
        "trace_type": "heatmap",
        "title": "Correlation Heatmap",
        "layout": {"height": 400},
        "colorscale": "RdBu",
        "zmid": True,
        "pre_compute": "heatmap_correlation",
    },
    "timeseries": {
        "trace_type": "line",
        "title": "Time Series",
        "layout": {"height": 350},
        "pre_compute": "timeseries_multi_y",
    },
    "bubble": {
        "trace_type": "scatter",
        "x": "x",
        "y": "y",
        "groupby": "color",
        "title": "{y} vs {x}",
        "title_grouped": "{y} vs {x} (by {group})",
        "layout": {"height": 400},
        "pre_compute": "bubble_sizes",
        "summary_fn": "bubble_summary",
    },
    # ── Tier 2: Specs + hooks (pre/post computation) ─────────────────────
    "scatter": {
        "trace_type": "scatter",
        "x": "x",
        "y": "y",
        "groupby": "color",
        "title": "{y} vs {x}",
        "title_grouped": "{y} vs {x} (by {group})",
        "post_compute": "scatter_trendline",
    },
    "pareto": {
        "trace_type": "bar",
        "title": "Pareto Chart",
        "pre_compute": "pareto_compute",
    },
    "probability": {
        "trace_type": "scatter",
        "title": "Probability Plot",
        "layout": {"height": 350},
        "pre_compute": "probability_compute",
    },
    "dotplot": {
        "trace_type": "scatter",
        "title": "Dotplot",
        "pre_compute": "dotplot_compute",
    },
    "individual_value_plot": {
        "trace_type": "scatter",
        "title": "Individual Value Plot",
        "layout": {"height": 350},
        "pre_compute": "individual_value_compute",
    },
    "interval_plot": {
        "trace_type": "scatter",
        "title": "Interval Plot",
        "layout": {"height": 350},
        "pre_compute": "interval_compute",
    },
    "contour": {
        "trace_type": "scatter",  # engine won't render — hook returns result directly
        "title": "Contour Plot",
        "pre_compute": "contour_compute",
    },
    "surface_3d": {
        "trace_type": "scatter",
        "title": "3D Surface",
        "pre_compute": "surface_3d_compute",
    },
    "contour_overlay": {
        "trace_type": "scatter",
        "title": "Contour Overlay",
        "pre_compute": "contour_overlay_compute",
    },
}
