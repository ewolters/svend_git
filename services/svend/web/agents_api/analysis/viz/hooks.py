"""Pre/post-compute hooks for Tier 2 chart specs.

Each hook is called by the engine at the appropriate phase:
- pre_compute hooks: (df, config) → {"df", "extra_traces", "extra_config", "result"}
- post_compute hooks: (df, config, traces) → {"extra_traces", "summary"}
- summary_fn hooks: (df, config, result) → str

Hooks that return "result" in their dict short-circuit the engine — the hook
builds the entire result itself (used for complex charts like contour).

CR: 3c0d0e53
"""

from collections import Counter

import numpy as np

from ..common import SVEND_COLORS


def _resolve_groupby(config, key):
    """Check if a groupby key has a valid value."""
    val = config.get(key)
    return val if val and val != "" and val != "None" else None


# ── Pre-compute hooks ────────────────────────────────────────────────────


def heatmap_correlation(df, config):
    """Pre-compute correlation matrix for heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": "Correlation Heatmap",
            "data": [
                {
                    "type": "heatmap",
                    "z": corr.values.tolist(),
                    "x": numeric_cols,
                    "y": numeric_cols,
                    "colorscale": "RdBu",
                    "zmid": 0,
                }
            ],
            "layout": {"height": 400},
        }
    )
    return {"result": result}


def timeseries_multi_y(df, config):
    """Pre-compute multi-y timeseries traces."""
    x_col = config.get("x")
    y_cols = config.get("y", [])
    show_markers = config.get("markers", False)

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    traces = []
    for i, y_col in enumerate(y_cols):
        trace = {
            "type": "scatter",
            "x": df[x_col].astype(str).tolist(),
            "y": df[y_col].tolist(),
            "mode": "lines+markers" if show_markers else "lines",
            "name": y_col,
            "line": {"color": SVEND_COLORS[i % len(SVEND_COLORS)]},
        }
        traces.append(trace)

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Time Series: {', '.join(y_cols)}",
            "data": traces,
            "layout": {
                "height": 350,
                "xaxis": {"title": x_col},
                "yaxis": {"title": "Value"},
                "showlegend": len(y_cols) > 1,
            },
        }
    )
    return {"result": result}


def bubble_sizes(df, config):
    """Pre-compute bubble sizes from size column."""
    size_var = config.get("size")
    if not size_var or size_var not in df.columns:
        return None

    size_vals = df[size_var].dropna().values.astype(float)
    s_min, s_max = float(size_vals.min()), float(size_vals.max())
    s_range = s_max - s_min if s_max != s_min else 1.0

    x_var = config.get("x")
    y_var = config.get("y")
    color_var = _resolve_groupby(config, "color")

    traces = []
    if color_var and color_var in df.columns:
        for i, group in enumerate(df[color_var].dropna().unique()):
            sub = df.loc[df[color_var] == group]
            sizes = (
                (sub[size_var].fillna(s_min).astype(float) - s_min) / s_range * 35 + 5
            ).tolist()
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": sub[x_var].tolist(),
                    "y": sub[y_var].tolist(),
                    "marker": {
                        "size": sizes,
                        "sizemode": "diameter",
                        "color": SVEND_COLORS[i % len(SVEND_COLORS)],
                        "opacity": 0.7,
                    },
                    "name": str(group),
                }
            )
    else:
        sizes = (
            (df[size_var].fillna(s_min).astype(float) - s_min) / s_range * 35 + 5
        ).tolist()
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": df[x_var].tolist(),
                "y": df[y_var].tolist(),
                "marker": {
                    "size": sizes,
                    "sizemode": "diameter",
                    "color": "rgba(74,159,110,0.6)",
                    "line": {"color": "#4a9f6e", "width": 1},
                },
                "name": size_var,
            }
        )

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Bubble Chart: {y_var} vs {x_var} (size: {size_var})",
            "data": traces,
            "layout": {
                "height": 400,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
                "showlegend": bool(color_var),
            },
        }
    )
    return {"result": result}


def bubble_summary(df, config, result):
    """Generate bubble chart summary."""
    x_var = config.get("x")
    y_var = config.get("y")
    size_var = config.get("size")
    color_var = _resolve_groupby(config, "color")
    summary = f"Bubble Chart\n\nX: {x_var}, Y: {y_var}, Size: {size_var}"
    if color_var:
        summary += f", Color: {color_var}"
    summary += f"\nObservations: {len(df)}"
    return summary


def pareto_compute(df, config):
    """Pre-compute Pareto chart (bars + cumulative line)."""
    category = config.get("category") or config.get("var")
    value_col = config.get("value")

    if value_col and value_col != "":
        counts = df.groupby(category)[value_col].sum().sort_values(ascending=False)
    else:
        counts = df[category].value_counts()

    cumulative = counts.cumsum() / counts.sum() * 100

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Pareto Chart: {category}",
            "data": [
                {
                    "type": "bar",
                    "x": [str(c) for c in counts.index],
                    "y": counts.values.tolist(),
                    "marker": {
                        "color": "rgba(74, 159, 110, 0.7)",
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                    "name": "Count",
                },
                {
                    "type": "scatter",
                    "x": [str(c) for c in cumulative.index],
                    "y": cumulative.values.tolist(),
                    "mode": "lines+markers",
                    "line": {"color": "#e89547", "width": 2},
                    "marker": {"size": 6},
                    "name": "Cumulative %",
                    "yaxis": "y2",
                },
            ],
            "layout": {
                "height": 350,
                "xaxis": {"title": category},
                "yaxis": {"title": "Count"},
                "yaxis2": {
                    "title": "Cumulative %",
                    "overlaying": "y",
                    "side": "right",
                    "range": [0, 105],
                },
                "showlegend": True,
            },
        }
    )
    result["summary"] = (
        f"Pareto Analysis: {category}\n\nCategories: {len(counts)}\nTop category: {counts.index[0]} ({counts.iloc[0]})"
    )
    return {"result": result}


def probability_compute(df, config):
    """Pre-compute probability plot (quantiles + fit line)."""
    from scipy import stats

    var = config.get("var")
    dist = config.get("dist", "norm")

    x = df[var].dropna().values
    x_sorted = np.sort(x)
    n = len(x_sorted)

    pp = (np.arange(1, n + 1) - 0.5) / n

    if dist == "norm":
        theoretical = stats.norm.ppf(pp)
        dist_name = "Normal"
    elif dist == "lognorm":
        theoretical = stats.lognorm.ppf(pp, s=1)
        dist_name = "Lognormal"
        x_sorted = np.log(x_sorted[x_sorted > 0])
        theoretical = stats.norm.ppf(pp[: len(x_sorted)])
    elif dist == "expon":
        theoretical = stats.expon.ppf(pp)
        dist_name = "Exponential"
    elif dist == "weibull":
        theoretical = stats.weibull_min.ppf(pp, c=2)
        dist_name = "Weibull"
    else:
        theoretical = stats.norm.ppf(pp)
        dist_name = "Normal"

    slope, intercept = np.polyfit(theoretical, x_sorted, 1)
    fit_line = slope * theoretical + intercept

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Probability Plot ({dist_name}): {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": x_sorted.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": "rgba(74, 159, 110, 0.5)",
                        "size": 6,
                        "line": {"color": "#4a9f6e", "width": 1},
                    },
                    "name": "Data",
                },
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": fit_line.tolist(),
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "Fit Line",
                },
            ],
            "layout": {
                "height": 350,
                "xaxis": {"title": f"Theoretical Quantiles ({dist_name})"},
                "yaxis": {"title": var},
            },
        }
    )

    if dist == "norm":
        ad_stat = stats.anderson(x)
        result["summary"] = (
            f"Probability Plot ({dist_name})\n\nAnderson-Darling: {ad_stat.statistic:.4f}\n"
            f"Critical values (15%, 10%, 5%, 2.5%, 1%):\n{ad_stat.critical_values}"
        )

    return {"result": result}


def dotplot_compute(df, config):
    """Pre-compute dotplot with stacking."""
    var = config.get("var")
    groupby = _resolve_groupby(config, "group")

    result = {"plots": [], "summary": ""}

    if groupby and groupby in df.columns:
        groups = df[groupby].dropna().unique()
        traces = []
        for i, grp in enumerate(groups):
            vals = np.sort(df[df[groupby] == grp][var].dropna().values)
            color = SVEND_COLORS[i % len(SVEND_COLORS)]
            if len(vals) == 0:
                continue
            rng = float(np.ptp(vals)) if np.ptp(vals) > 0 else 1.0
            bin_width = (
                rng / max(min(int(np.sqrt(len(vals))), 30), 5) if len(vals) > 1 else rng
            )
            if bin_width == 0:
                bin_width = 1.0
            binned = np.round(vals / bin_width) * bin_width
            x_out, y_out = [], []
            counts = Counter()
            for bv, rv in zip(binned, vals):
                counts[bv] += 1
                x_out.append(float(rv))
                y_out.append(i + (counts[bv] - 1) * 0.08)
            traces.append(
                {
                    "type": "scatter",
                    "x": x_out,
                    "y": y_out,
                    "mode": "markers",
                    "marker": {"color": color, "size": 7},
                    "name": str(grp),
                }
            )
        result["plots"].append(
            {
                "title": f"Dotplot: {var} by {groupby}",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": var},
                    "yaxis": {
                        "tickvals": list(range(len(groups))),
                        "ticktext": [str(g) for g in groups],
                        "title": groupby,
                    },
                },
            }
        )
    else:
        vals = np.sort(df[var].dropna().values)
        if len(vals) > 0:
            rng = float(np.ptp(vals)) if np.ptp(vals) > 0 else 1.0
            bin_width = (
                rng / max(min(int(np.sqrt(len(vals))), 30), 5) if len(vals) > 1 else rng
            )
            if bin_width == 0:
                bin_width = 1.0
            binned = np.round(vals / bin_width) * bin_width
            x_out, y_out = [], []
            counts = Counter()
            for bv, rv in zip(binned, vals):
                counts[bv] += 1
                x_out.append(float(rv))
                y_out.append(counts[bv])
            result["plots"].append(
                {
                    "title": f"Dotplot: {var}",
                    "data": [
                        {
                            "type": "scatter",
                            "x": x_out,
                            "y": y_out,
                            "mode": "markers",
                            "marker": {"color": "#4a9f6e", "size": 7},
                            "name": var,
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": var},
                        "yaxis": {"title": "Count"},
                    },
                }
            )

    return {"result": result}


def individual_value_compute(df, config):
    """Pre-compute individual value plot with jitter, means, and CIs."""
    from scipy import stats

    var = config.get("var")
    groupby = _resolve_groupby(config, "group")
    show_mean = config.get("show_mean", True)
    show_ci = config.get("show_ci", True)
    conf = config.get("confidence", 0.95)

    result = {"plots": [], "summary": ""}
    traces = []

    if groupby and groupby in df.columns:
        groups = df[groupby].dropna().unique()
        for i, grp in enumerate(groups):
            vals = df[df[groupby] == grp][var].dropna().values
            color = SVEND_COLORS[i % len(SVEND_COLORS)]
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            traces.append(
                {
                    "type": "scatter",
                    "x": (np.full(len(vals), i) + jitter).tolist(),
                    "y": vals.tolist(),
                    "mode": "markers",
                    "marker": {"color": color, "size": 6, "opacity": 0.6},
                    "name": str(grp),
                    "showlegend": True,
                }
            )
            if show_mean and len(vals) > 0:
                mean_val = float(np.mean(vals))
                traces.append(
                    {
                        "type": "scatter",
                        "x": [i],
                        "y": [mean_val],
                        "mode": "markers",
                        "marker": {
                            "color": color,
                            "size": 14,
                            "symbol": "diamond",
                            "line": {"color": "#fff", "width": 1},
                        },
                        "showlegend": False,
                    }
                )
            if show_ci and len(vals) > 1:
                mean_val = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ci = stats.t.interval(conf, df=len(vals) - 1, loc=mean_val, scale=se)
                traces.append(
                    {
                        "type": "scatter",
                        "x": [i, i],
                        "y": [ci[0], ci[1]],
                        "mode": "lines",
                        "line": {"color": color, "width": 2},
                        "showlegend": False,
                    }
                )

        result["plots"].append(
            {
                "title": f"Individual Value Plot: {var} by {groupby}",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {
                        "tickvals": list(range(len(groups))),
                        "ticktext": [str(g) for g in groups],
                        "title": groupby,
                    },
                    "yaxis": {"title": var},
                    "showlegend": True,
                },
            }
        )
    else:
        vals = df[var].dropna().values
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        traces.append(
            {
                "type": "scatter",
                "x": jitter.tolist(),
                "y": vals.tolist(),
                "mode": "markers",
                "marker": {"color": "rgba(74,159,110,0.6)", "size": 6},
                "name": var,
                "showlegend": False,
            }
        )
        if show_mean and len(vals) > 0:
            mean_val = float(np.mean(vals))
            traces.append(
                {
                    "type": "scatter",
                    "x": [0],
                    "y": [mean_val],
                    "mode": "markers",
                    "marker": {"color": "#e89547", "size": 14, "symbol": "diamond"},
                    "showlegend": False,
                }
            )
        result["plots"].append(
            {
                "title": f"Individual Value Plot: {var}",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"showticklabels": False},
                    "yaxis": {"title": var},
                },
            }
        )

    return {"result": result}


def interval_compute(df, config):
    """Pre-compute interval plot (means + CIs per group)."""
    from scipy import stats

    var = config.get("var")
    groupby = _resolve_groupby(config, "group")
    conf = config.get("confidence", 0.95)

    result = {"plots": [], "summary": ""}

    if not groupby or groupby not in df.columns:
        result["summary"] = "Interval plot requires a grouping variable."
        return {"result": result}

    groups = df[groupby].dropna().unique()
    means = []
    ci_lo = []
    ci_hi = []
    labels = []

    for grp in groups:
        vals = df[df[groupby] == grp][var].dropna().values
        if len(vals) < 2:
            continue
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ci = stats.t.interval(conf, df=len(vals) - 1, loc=m, scale=se)
        means.append(m)
        ci_lo.append(m - ci[0])
        ci_hi.append(ci[1] - m)
        labels.append(str(grp))

    overall_mean = float(df[var].dropna().mean())

    traces = [
        {
            "type": "scatter",
            "x": labels,
            "y": means,
            "mode": "markers",
            "marker": {"color": "#4a9f6e", "size": 10, "symbol": "diamond"},
            "error_y": {
                "type": "data",
                "symmetric": False,
                "array": ci_hi,
                "arrayminus": ci_lo,
                "color": "#4a9f6e",
                "thickness": 2,
                "width": 8,
            },
            "name": f"Mean \u00b1 {conf * 100:.0f}% CI",
        },
        {
            "type": "scatter",
            "x": [labels[0], labels[-1]] if labels else [],
            "y": [overall_mean, overall_mean] if labels else [],
            "mode": "lines",
            "line": {"color": "#e89547", "dash": "dash", "width": 1.5},
            "name": f"Overall Mean ({overall_mean:.4g})",
        },
    ]

    result["plots"].append(
        {
            "title": f"Interval Plot: {var} by {groupby} ({conf * 100:.0f}% CI)",
            "data": traces,
            "layout": {
                "height": 350,
                "xaxis": {"title": groupby},
                "yaxis": {"title": var},
                "showlegend": True,
            },
        }
    )
    return {"result": result}


def contour_compute(df, config):
    """Pre-compute contour plot via griddata interpolation."""
    from scipy.interpolate import griddata

    x_var = config.get("x")
    y_var = config.get("y")
    z_var = config.get("z")

    x = df[x_var].dropna().values.astype(float)
    y = df[y_var].dropna().values.astype(float)
    z = df[z_var].dropna().values.astype(float)

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi_grid, yi_grid), method="cubic")

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Contour: {z_var} vs ({x_var}, {y_var})",
            "data": [
                {
                    "type": "contour",
                    "x": xi.tolist(),
                    "y": yi.tolist(),
                    "z": np.nan_to_num(zi, nan=0.0).tolist(),
                    "colorscale": "Viridis",
                    "contours": {"showlabels": True},
                    "colorbar": {"title": z_var},
                },
                {
                    "type": "scatter",
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": "#fff",
                        "size": 4,
                        "line": {"color": "#333", "width": 1},
                    },
                    "name": "Data",
                    "showlegend": False,
                },
            ],
            "layout": {
                "height": 400,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
            },
        }
    )
    return {"result": result}


def surface_3d_compute(df, config):
    """Pre-compute 3D surface via griddata interpolation."""
    from scipy.interpolate import griddata

    x_var = config.get("x")
    y_var = config.get("y")
    z_var = config.get("z")

    x = df[x_var].dropna().values.astype(float)
    y = df[y_var].dropna().values.astype(float)
    z = df[z_var].dropna().values.astype(float)

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi_grid, yi_grid), method="cubic")

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"3D Surface: {z_var}",
            "data": [
                {
                    "type": "surface",
                    "x": xi.tolist(),
                    "y": yi.tolist(),
                    "z": np.nan_to_num(zi, nan=0.0).tolist(),
                    "colorscale": "Viridis",
                    "colorbar": {"title": z_var},
                }
            ],
            "layout": {
                "height": 500,
                "scene": {
                    "xaxis": {"title": x_var},
                    "yaxis": {"title": y_var},
                    "zaxis": {"title": z_var},
                },
            },
        }
    )
    return {"result": result}


def contour_overlay_compute(df, config):
    """Pre-compute multi-response contour overlay."""
    from scipy.interpolate import griddata

    x_var = config.get("x")
    y_var = config.get("y")
    responses = config.get("responses", [])

    if isinstance(responses, str):
        responses = [responses]
    if not responses:
        z_var = config.get("z")
        responses = [z_var] if z_var else []

    x = df[x_var].dropna().values.astype(float)
    y = df[y_var].dropna().values.astype(float)
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    traces = []
    for i, resp in enumerate(responses):
        z = df[resp].dropna().values.astype(float)
        zi = griddata((x[: len(z)], y[: len(z)]), z, (xi_grid, yi_grid), method="cubic")
        color = SVEND_COLORS[i % len(SVEND_COLORS)]
        traces.append(
            {
                "type": "contour",
                "x": xi.tolist(),
                "y": yi.tolist(),
                "z": np.nan_to_num(zi, nan=0.0).tolist(),
                "contours": {"showlabels": True, "coloring": "lines"},
                "line": {"color": color, "width": 2},
                "name": resp,
                "showscale": False,
            }
        )

    # Scatter overlay of data points
    traces.append(
        {
            "type": "scatter",
            "x": x.tolist(),
            "y": y.tolist(),
            "mode": "markers",
            "marker": {"color": "#333", "size": 5},
            "name": "Data",
        }
    )

    result = {"plots": [], "summary": ""}
    result["plots"].append(
        {
            "title": f"Contour Overlay: {', '.join(responses)}",
            "data": traces,
            "layout": {
                "height": 450,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
                "showlegend": True,
            },
        }
    )
    return {"result": result}


# ── Post-compute hooks ───────────────────────────────────────────────────


def scatter_trendline(df, config, traces):
    """Post-compute: add trendline to scatter if configured."""
    if not config.get("trendline", False):
        return None

    x_var = config.get("x")
    y_var = config.get("y")

    x = df[x_var].dropna()
    y = df[y_var].loc[x.index].dropna()
    common_idx = x.index.intersection(y.index)
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    if len(x) <= 1:
        return None

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return {
        "extra_traces": [
            {
                "type": "scatter",
                "x": [float(x.min()), float(x.max())],
                "y": [float(p(x.min())), float(p(x.max()))],
                "mode": "lines",
                "line": {"color": "#e8c547", "dash": "dash"},
                "name": "Trendline",
            }
        ]
    }


# ── Hook registry ────────────────────────────────────────────────────────

HOOKS = {
    # Pre-compute
    "heatmap_correlation": heatmap_correlation,
    "timeseries_multi_y": timeseries_multi_y,
    "bubble_sizes": bubble_sizes,
    "pareto_compute": pareto_compute,
    "probability_compute": probability_compute,
    "dotplot_compute": dotplot_compute,
    "individual_value_compute": individual_value_compute,
    "interval_compute": interval_compute,
    "contour_compute": contour_compute,
    "surface_3d_compute": surface_3d_compute,
    "contour_overlay_compute": contour_overlay_compute,
    # Post-compute
    "scatter_trendline": scatter_trendline,
    # Summary
    "bubble_summary": bubble_summary,
}
