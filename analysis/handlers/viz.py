"""Viz handler — pure chart rendering via forgeviz."""

import logging

import pandas as pd
from forgeviz.charts.distribution import box_plot, histogram
from forgeviz.charts.scatter import pareto, scatter
from forgeviz.charts.statistical import bubble, dotplot, heatmap, mosaic, parallel_coordinates

logger = logging.getLogger(__name__)


def run(df, analysis_id, config):
    """Build ForgeViz chart directly from data."""
    dispatch = {
        "histogram": _histogram,
        "boxplot": _boxplot,
        "scatter": _scatter,
        "pareto": _pareto,
        "heatmap": _heatmap,
        "dotplot": _dotplot,
        "individual_value_plot": _dotplot,
        "bubble": _bubble,
        "parallel_coordinates": _parallel,
        "mosaic": _mosaic,
    }

    fn = dispatch.get(analysis_id)
    if fn:
        try:
            return fn(df, config)
        except Exception as e:
            logger.exception("Viz error: %s", analysis_id)
            return {"summary": f"Chart error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"Viz type '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


def _histogram(df, config):
    col = config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    chart = histogram(data, title=f"Histogram of {col}")
    return {"charts": [chart], "statistics": {"n": len(data)}, "summary": f"Histogram of {col} ({len(data)} values)."}


def _boxplot(df, config):
    col = config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    chart = box_plot(data, title=f"Box Plot of {col}")
    return {"charts": [chart], "statistics": {"n": len(data)}, "summary": f"Box plot of {col}."}


def _scatter(df, config):
    x_col = config.get("x")
    y_col = config.get("y")
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index)
    chart = scatter(x.loc[idx].tolist(), y.loc[idx].tolist(), x_label=x_col, y_label=y_col, title=f"{x_col} vs {y_col}")
    return {"charts": [chart], "statistics": {"n": len(idx)}, "summary": f"Scatter: {x_col} vs {y_col}."}


def _pareto(df, config):
    cat_col = config.get("category")
    val_col = config.get("value")
    if val_col:
        counts = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False)
    else:
        counts = df[cat_col].value_counts()
    chart = pareto(counts.index.tolist(), counts.values.tolist(), title=f"Pareto of {cat_col}")
    return {"charts": [chart], "statistics": {"categories": len(counts)}, "summary": f"Pareto chart of {cat_col}."}


def _heatmap(df, config):
    cols = config.get("vars") or df.select_dtypes(include="number").columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    corr = df[cols].corr()
    chart = heatmap(corr.values.tolist(), x_labels=cols, y_labels=cols, title="Correlation Heatmap")
    return {"charts": [chart], "statistics": {}, "summary": "Correlation heatmap."}


def _dotplot(df, config):
    col = config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
    chart = dotplot(data, title=f"Dotplot of {col}")
    return {"charts": [chart], "statistics": {"n": len(data)}, "summary": f"Dotplot of {col}."}


def _bubble(df, config):
    x_col, y_col = config.get("x"), config.get("y")
    size_col = config.get("size")
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    s = pd.to_numeric(df[size_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index).intersection(s.index)
    chart = bubble(
        x.loc[idx].tolist(),
        y.loc[idx].tolist(),
        s.loc[idx].tolist(),
        x_label=x_col,
        y_label=y_col,
        title=f"Bubble: {x_col} vs {y_col}",
    )
    return {"charts": [chart], "statistics": {"n": len(idx)}, "summary": "Bubble chart."}


def _parallel(df, config):
    dims = config.get("dimensions") or df.select_dtypes(include="number").columns.tolist()[:6]
    if isinstance(dims, str):
        dims = [dims]
    data = df[dims].apply(pd.to_numeric, errors="coerce").dropna()
    chart = parallel_coordinates(data.values.tolist(), dimension_labels=dims, title="Parallel Coordinates")
    return {
        "charts": [chart],
        "statistics": {"dimensions": len(dims)},
        "summary": f"Parallel coordinates ({len(dims)} dims).",
    }


def _mosaic(df, config):
    row_var = config.get("row_var")
    col_var = config.get("col_var")
    ct = pd.crosstab(df[row_var], df[col_var])
    chart = mosaic(
        ct.values.tolist(), row_labels=ct.index.tolist(), col_labels=ct.columns.tolist(), title=f"{row_var} × {col_var}"
    )
    return {"charts": [chart], "statistics": {}, "summary": f"Mosaic: {row_var} × {col_var}."}
