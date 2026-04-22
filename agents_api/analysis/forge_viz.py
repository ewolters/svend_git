"""Forge-native visualization handlers — ForgeViz ChartSpec output.

Replaces the Plotly-based viz engine for the new workbench.
Each handler reads columns from the DataFrame and returns a result dict
with ForgeViz ChartSpec dicts in the 'plots' key.

CR: 87bf578a (workbench migration)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _col(df, config, key, fallback=None):
    """Extract a numeric column, coercing and dropping NaN."""
    name = config.get(key) or config.get(fallback or "")
    if not name or name not in df.columns:
        return None, name
    s = pd.to_numeric(df[name], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column '{name}' has no valid numeric data")
    return s, name


def _group_col(df, config):
    """Get group column name if present."""
    return config.get("group") or config.get("factor")


def _result(plots, summary=""):
    return {"plots": plots, "summary": summary, "statistics": {}, "narrative": {}}


# =============================================================================
# Chart handlers
# =============================================================================


def viz_histogram(df, config):
    from forgeviz.charts.distribution import histogram

    col, name = _col(df, config, "column", "var1")
    group = _group_col(df, config)
    if group and group in df.columns:
        specs = []
        for i, (g, sub) in enumerate(df.groupby(group)):
            vals = pd.to_numeric(sub[name], errors="coerce").dropna().tolist()
            if vals:
                specs.append(histogram(vals, title=f"{name} — {g}").to_dict())
        return _result(specs, f"Histogram of {name} by {group}")
    spec = histogram(col.tolist(), title=name)
    return _result([spec.to_dict()], f"Histogram of {name}")


def viz_boxplot(df, config):
    from forgeviz.charts.distribution import box_plot

    col, name = _col(df, config, "column", "var1")
    group = _group_col(df, config)
    if group and group in df.columns:
        groups = {}
        for g, sub in df.groupby(group):
            vals = pd.to_numeric(sub[name], errors="coerce").dropna().tolist()
            if vals:
                groups[str(g)] = vals
        spec = box_plot(groups, title=f"{name} by {group}")
    else:
        spec = box_plot({name: col.tolist()}, title=name)
    return _result([spec.to_dict()], f"Box plot of {name}")


def viz_scatter(df, config):
    from forgeviz.charts.scatter import scatter as scatter_chart

    x_col = config.get("x")
    y_col = config.get("y")
    if not x_col or not y_col:
        raise ValueError("Scatter requires x and y columns")
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index)
    spec = scatter_chart(x.loc[idx].tolist(), y.loc[idx].tolist(), title=f"{y_col} vs {x_col}")
    return _result([spec.to_dict()], f"Scatter plot: {y_col} vs {x_col}")


def viz_dotplot(df, config):
    from forgeviz.charts.statistical import dotplot

    col, name = _col(df, config, "column", "var1")
    group = _group_col(df, config)
    if group and group in df.columns:
        cats, vals = [], []
        for g, sub in df.groupby(group):
            v = pd.to_numeric(sub[name], errors="coerce").dropna().tolist()
            cats.extend([str(g)] * len(v))
            vals.extend(v)
        spec = dotplot(cats, vals, title=f"{name} by {group}")
    else:
        spec = dotplot([name] * len(col), col.tolist(), title=name)
    return _result([spec.to_dict()], f"Dotplot of {name}")


def viz_pareto(df, config):
    from forgeviz.charts.generic import bar

    col = config.get("column")
    if not col or col not in df.columns:
        raise ValueError("Pareto requires a categorical column")
    counts = df[col].value_counts().sort_values(ascending=False)
    spec = bar(
        categories=counts.index.tolist(),
        values=counts.values.tolist(),
        title=f"Pareto: {col}",
    )
    return _result([spec.to_dict()], f"Pareto chart of {col}")


def viz_heatmap(df, config):
    from forgeviz.charts.statistical import heatmap

    cols = config.get("columns", [])
    if len(cols) < 2:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:5]
    corr = df[cols].corr()
    spec = heatmap(
        x_labels=corr.columns.tolist(),
        y_labels=corr.index.tolist(),
        z_matrix=corr.values.tolist(),
        title="Correlation Heatmap",
    )
    return _result([spec.to_dict()], "Correlation heatmap")


def viz_timeseries(df, config):
    from forgeviz.charts.generic import line

    col, name = _col(df, config, "column")
    time_col = config.get("time")
    if time_col and time_col in df.columns:
        x = df[time_col].tolist()
    else:
        x = list(range(len(col)))
    spec = line(x=x, y=col.tolist(), title=f"Time Series: {name}")
    return _result([spec.to_dict()], f"Time series of {name}")


def viz_individual_value_plot(df, config):
    from forgeviz.charts.statistical import individual_value_plot

    col, name = _col(df, config, "column", "var1")
    group = _group_col(df, config)
    if group and group in df.columns:
        groups = {}
        for g, sub in df.groupby(group):
            vals = pd.to_numeric(sub[name], errors="coerce").dropna().tolist()
            if vals:
                groups[str(g)] = vals
    else:
        groups = {name: col.tolist()}
    spec = individual_value_plot(groups, title=f"Individual Value Plot: {name}")
    return _result([spec.to_dict()], f"Individual value plot of {name}")


def viz_interval_plot(df, config):
    from forgeviz.charts.statistical import interval_plot

    col, name = _col(df, config, "column", "var1")
    group = _group_col(df, config)
    if group and group in df.columns:
        groups = {}
        for g, sub in df.groupby(group):
            vals = pd.to_numeric(sub[name], errors="coerce").dropna().tolist()
            if vals:
                groups[str(g)] = vals
    else:
        groups = {name: col.tolist()}
    spec = interval_plot(groups, title=f"Interval Plot: {name}")
    return _result([spec.to_dict()], f"Interval plot of {name}")


def viz_probability(df, config):

    col, name = _col(df, config, "column")
    # Probability plot approximated as sorted values vs theoretical quantiles
    import statistics

    vals = sorted(col.tolist())
    n = len(vals)
    probs = [(i + 0.5) / n for i in range(n)]
    # Normal quantiles
    z = []
    for p in probs:
        try:
            z.append(statistics.NormalDist().inv_cdf(p))
        except (ValueError, statistics.StatisticsError):
            z.append(0)
    from forgeviz.charts.scatter import scatter as scatter_chart

    spec = scatter_chart(z, vals, title=f"Normal Probability Plot: {name}")
    return _result([spec.to_dict()], f"Probability plot of {name}")


def viz_bubble(df, config):
    from forgeviz.charts.statistical import bubble

    x_col = config.get("x")
    y_col = config.get("y")
    size_col = config.get("size") or config.get("column")
    if not x_col or not y_col:
        raise ValueError("Bubble requires x and y columns")
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index)
    sizes = None
    if size_col and size_col in df.columns:
        s = pd.to_numeric(df[size_col], errors="coerce")
        sizes = s.loc[idx].fillna(1).tolist()
    spec = bubble(
        x.loc[idx].tolist(), y.loc[idx].tolist(), sizes=sizes or [5] * len(idx), title=f"Bubble: {y_col} vs {x_col}"
    )
    return _result([spec.to_dict()], f"Bubble chart: {y_col} vs {x_col}")


def viz_matrix(df, config):
    from forgeviz.charts.statistical import scatter_matrix

    cols = config.get("columns", [])
    if not cols:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:4]
    data = {c: pd.to_numeric(df[c], errors="coerce").dropna().tolist() for c in cols}
    specs = scatter_matrix(data, title="Scatter Matrix")
    # scatter_matrix returns list[ChartSpec]
    if isinstance(specs, list):
        return _result([s.to_dict() if hasattr(s, "to_dict") else s for s in specs], f"Matrix plot: {', '.join(cols)}")
    return _result([specs.to_dict()], f"Matrix plot: {', '.join(cols)}")


def viz_mosaic(df, config):
    from forgeviz.charts.statistical import mosaic

    var1 = config.get("var1")
    var2 = config.get("var2")
    if not var1 or not var2:
        raise ValueError("Mosaic requires var1 and var2")
    # mosaic expects contingency dict: {row: {col: count}}
    ct = pd.crosstab(df[var1], df[var2])
    contingency = {str(r): {str(c): int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}
    spec = mosaic(contingency, title=f"Mosaic: {var1} x {var2}")
    return _result([spec.to_dict()], f"Mosaic plot: {var1} x {var2}")


def viz_parallel_coordinates(df, config):
    from forgeviz.charts.statistical import parallel_coordinates

    cols = config.get("columns", [])
    if not cols:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:5]
    data = {c: pd.to_numeric(df[c], errors="coerce").fillna(0).tolist() for c in cols}
    group = _group_col(df, config)
    highlight = None
    if group and group in df.columns:
        # Highlight first group
        first_val = df[group].iloc[0]
        highlight = [i for i, v in enumerate(df[group]) if v == first_val]
    spec = parallel_coordinates(data, title="Parallel Coordinates", highlight_idx=highlight)
    return _result([spec.to_dict()], f"Parallel coordinates: {', '.join(cols)}")


def viz_contour(df, config):
    from forgeviz.charts.surface import contour_plot

    x_col = config.get("x")
    y_col = config.get("y")
    z_col = config.get("z") or config.get("column")
    if not x_col or not y_col or not z_col:
        raise ValueError("Contour requires x, y, z columns")
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    z = pd.to_numeric(df[z_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index).intersection(z.index)
    spec = contour_plot(x.loc[idx].tolist(), y.loc[idx].tolist(), z.loc[idx].tolist(), title=f"Contour: {z_col}")
    return _result([spec.to_dict()], f"Contour plot: {z_col}")


def viz_surface_3d(df, config):

    x_col = config.get("x")
    y_col = config.get("y")
    z_col = config.get("z") or config.get("column")
    if not x_col or not y_col or not z_col:
        raise ValueError("Surface 3D requires x, y, z columns")
    # Use contour_plot as fallback — forgeviz surface requires a model, not raw data
    from forgeviz.charts.surface import contour_plot

    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    z = pd.to_numeric(df[z_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index).intersection(z.index)
    spec = contour_plot(x.loc[idx].tolist(), y.loc[idx].tolist(), z.loc[idx].tolist(), title=f"Surface: {z_col}")
    return _result([spec.to_dict()], f"3D surface: {z_col}")


# =============================================================================
# Dispatch
# =============================================================================

FORGE_VIZ_HANDLERS = {
    "histogram": viz_histogram,
    "boxplot": viz_boxplot,
    "scatter": viz_scatter,
    "dotplot": viz_dotplot,
    "pareto": viz_pareto,
    "heatmap": viz_heatmap,
    "timeseries": viz_timeseries,
    "individual_value_plot": viz_individual_value_plot,
    "interval_plot": viz_interval_plot,
    "probability": viz_probability,
    "bubble": viz_bubble,
    "matrix": viz_matrix,
    "mosaic": viz_mosaic,
    "parallel_coordinates": viz_parallel_coordinates,
    "contour": viz_contour,
    "contour_overlay": viz_contour,  # same as contour for now
    "surface_3d": viz_surface_3d,
}


def run_forge_viz(analysis_id, df, config):
    """ForgeViz-native viz handler. Returns ChartSpec dicts, not Plotly."""
    handler = FORGE_VIZ_HANDLERS.get(analysis_id)
    if handler:
        return handler(df, config)

    # Fall back to legacy viz (Bayesian SPC etc) — these return Plotly
    from agents_api.analysis.viz import run_visualization

    return run_visualization(df, analysis_id, config)
