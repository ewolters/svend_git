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
        "matrix": _matrix,
        "interval_plot": _interval,
        "timeseries": _timeseries,
        "probability": _probability,
        "contour": _contour,
        "contour_overlay": _contour,
        "surface_3d": _contour,
    }

    # Bayesian SPC viz — delegate to bayesian handler
    if analysis_id.startswith("bayes_spc_"):
        return _bayes_spc_viz(df, analysis_id, config)

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


def _matrix(df, config):
    cols = config.get("vars") or df.select_dtypes(include="number").columns.tolist()[:6]
    if isinstance(cols, str):
        cols = [cols]
    from forgeviz.charts.statistical import scatter_matrix

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    chart = scatter_matrix(data.values.tolist(), labels=cols, title="Matrix Plot")
    return {"charts": [chart], "statistics": {"variables": len(cols)}, "summary": f"Matrix plot ({len(cols)} vars)."}


def _interval(df, config):
    col = config.get("var") or config.get("column")
    factor = config.get("factor") or config.get("group")
    from forgeviz.charts.statistical import interval_plot

    data = pd.to_numeric(df[col], errors="coerce").dropna()
    if factor and factor in df.columns:
        groups = df.groupby(factor)[col].apply(lambda g: pd.to_numeric(g, errors="coerce").dropna().tolist()).to_dict()
        chart = interval_plot(groups, title=f"Interval Plot: {col} by {factor}")
    else:
        chart = interval_plot({col: data.tolist()}, title=f"Interval Plot: {col}")
    return {"charts": [chart], "statistics": {"n": len(data)}, "summary": f"Interval plot of {col}."}


def _timeseries(df, config):
    x_col = config.get("x") or df.columns[0]
    y_cols = config.get("y") or [df.select_dtypes(include="number").columns[0]]
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    from forgeviz.core.spec import ChartSpec

    spec = ChartSpec(title="Time Series", x_axis={"label": x_col}, y_axis={"label": "Value"})
    for i, yc in enumerate(y_cols):
        if yc in df.columns:
            y_data = pd.to_numeric(df[yc], errors="coerce").dropna()
            x_data = list(range(len(y_data)))
            colors = ["#4a9f6e", "#e8c547", "#4dc9c0", "#a78bfa", "#f472b6"]
            spec.add_trace(x_data, y_data.tolist(), name=yc, color=colors[i % len(colors)], width=1.5)
    return {"charts": [spec], "statistics": {}, "summary": f"Time series: {', '.join(y_cols)}."}


def _probability(df, config):
    col = config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    import numpy as np
    from scipy import stats as sp

    # Q-Q plot against normal
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical = [float(sp.norm.ppf((i + 0.5) / n)) for i in range(n)]
    from forgeviz.core.spec import ChartSpec

    spec = ChartSpec(
        title=f"Probability Plot: {col}",
        chart_type="probability",
        x_axis={"label": "Theoretical Quantiles"},
        y_axis={"label": "Sample Quantiles"},
    )
    spec.add_trace(theoretical, sorted_data.tolist(), name="Data", trace_type="scatter", color="#4a9f6e", marker_size=4)
    # Reference line
    mean, std = float(np.mean(data)), float(np.std(data, ddof=1))
    ref_x = [theoretical[0], theoretical[-1]]
    ref_y = [mean + std * theoretical[0], mean + std * theoretical[-1]]
    spec.add_trace(ref_x, ref_y, name="Reference", color="#d94a4a", dash="dashed", width=1)
    return {
        "charts": [spec],
        "statistics": {"n": n, "mean": round(mean, 4), "std": round(std, 4)},
        "summary": f"Probability plot of {col}.",
    }


def _contour(df, config):
    x_col = config.get("x")
    y_col = config.get("y")
    z_col = config.get("z")
    from forgeviz.charts.surface import contour_plot

    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    z = pd.to_numeric(df[z_col], errors="coerce").dropna()
    idx = x.index.intersection(y.index).intersection(z.index)
    chart = contour_plot(
        x.loc[idx].tolist(),
        y.loc[idx].tolist(),
        z.loc[idx].tolist(),
        x_label=x_col,
        y_label=y_col,
        title=f"Contour: {z_col}",
    )
    return {"charts": [chart], "statistics": {"n": len(idx)}, "summary": f"Contour plot: {z_col} vs {x_col}, {y_col}."}


def _bayes_spc_viz(df, analysis_id, config):
    """Bayesian SPC visualization — uses forgepbs + forgeviz.charts.bayesian."""
    col = config.get("measurement") or config.get("var")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}
    data = pd.to_numeric(df[col], errors="coerce").dropna().values

    try:
        import numpy as np
        from forgepbs.charts.adaptive import AdaptiveControlLimits
        from forgepbs.core.posterior import NormalGammaPosterior

        prior = NormalGammaPosterior(
            mu=float(np.mean(data[:10])), kappa=1, alpha=2, beta=max(float(np.var(data[:10])), 1e-10)
        )
        acl = AdaptiveControlLimits(prior=prior.copy())
        points = acl.process_batch(data)

        from forgeviz.charts.bayesian import bayesian_control_chart

        chart = bayesian_control_chart(
            list(data),
            [p.ucl for p in points],
            [p.cl for p in points],
            [p.lcl for p in points],
            title=f"Bayesian SPC — {analysis_id.replace('bayes_spc_', '').title()}",
        )

        usl = config.get("usl") or config.get("USL")
        lsl = config.get("lsl") or config.get("LSL")
        if usl:
            chart.add_reference_line(float(usl), color="#d94a4a", dash="dashed", label=f"USL={usl}")
        if lsl:
            chart.add_reference_line(float(lsl), color="#d94a4a", dash="dashed", label=f"LSL={lsl}")

        return {"charts": [chart], "statistics": {"n": len(data)}, "summary": f"Bayesian SPC: {analysis_id}."}
    except Exception as e:
        return {"summary": f"Bayesian SPC viz error: {e}", "charts": [], "statistics": {}}
