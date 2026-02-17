"""DSW Visualization — chart and plot analysis methods."""

import json
import logging
import uuid
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

from accounts.permissions import gated, require_auth, require_enterprise

from .common import _effect_magnitude, _practical_block, _fit_best_distribution

logger = logging.getLogger(__name__)


def _nig_posterior_update(data, mu0, nu0, alpha0, beta0):
    """Normal-Inverse-Gamma conjugate posterior update."""
    import numpy as np
    n = len(data)
    x_bar = np.mean(data)
    nu_n = nu0 + n
    mu_n = (nu0 * mu0 + n * x_bar) / nu_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * np.sum((data - x_bar) ** 2) + (n * nu0 * (x_bar - mu0) ** 2) / (2.0 * nu_n)
    return mu_n, nu_n, alpha_n, beta_n


def _nig_sample(mu_n, nu_n, alpha_n, beta_n, n_samples=10000):
    """Draw (mu, sigma) samples from NIG posterior."""
    import numpy as np
    from scipy.stats import invgamma
    rng = np.random.default_rng(42)
    sigma2_samples = invgamma.rvs(a=alpha_n, scale=beta_n, size=n_samples, random_state=rng)
    mu_samples = rng.normal(loc=mu_n, scale=np.sqrt(sigma2_samples / nu_n))
    sigma_samples = np.sqrt(sigma2_samples)
    return mu_samples, sigma_samples


def _cpk_from_params(mu, sigma, usl=None, lsl=None):
    """Vectorized Cpk from arrays of mu and sigma. Supports one-sided specs."""
    import numpy as np
    if usl is not None and lsl is not None:
        cpu = (usl - mu) / (3.0 * sigma)
        cpl = (mu - lsl) / (3.0 * sigma)
        return np.minimum(cpu, cpl)
    elif usl is not None:
        return (usl - mu) / (3.0 * sigma)
    elif lsl is not None:
        return (mu - lsl) / (3.0 * sigma)
    else:
        return np.zeros_like(mu)


def run_visualization(df, analysis_id, config):
    """Create visualizations."""
    import numpy as np
    import pandas as pd

    result = {"plots": [], "summary": ""}

    # SVEND theme colors
    theme_colors = ['#4a9f6e', '#4a9f6e', '#e89547', '#9f4a4a', '#e8c547', '#7a6a9a']

    if analysis_id == "histogram":
        var = config.get("var")
        bins = int(config.get("bins", 20))
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "histogram",
                    "x": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "opacity": 0.7,
                    "nbinsx": bins,
                    "marker": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Histogram of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "barmode": "overlay", "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Histogram of {var}",
                "data": [{"type": "histogram", "x": df[var].dropna().tolist(), "nbinsx": bins, "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "boxplot":
        var = config.get("var")
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            # Create separate box for each group with different colors
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "box",
                    "y": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "marker": {"color": theme_colors[i % len(theme_colors)]},
                    "line": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Box Plot of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Box Plot of {var}",
                "data": [{"type": "box", "y": df[var].dropna().tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}, "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "scatter":
        x_var = config.get("x")
        y_var = config.get("y")
        color_var = config.get("color")
        trendline = config.get("trendline", False)

        # Distinct colors for groups (fill, border) - semi-transparent fill with solid border
        group_colors = [
            ('rgba(74, 159, 110, 0.5)', '#4a9f6e'),   # Green
            ('rgba(232, 149, 71, 0.5)', '#e89547'),   # Orange
            ('rgba(159, 74, 74, 0.5)', '#9f4a4a'),    # Red
            ('rgba(71, 165, 232, 0.5)', '#47a5e8'),   # Blue
            ('rgba(232, 197, 71, 0.5)', '#e8c547'),   # Yellow
            ('rgba(122, 106, 154, 0.5)', '#7a6a9a'), # Purple
        ]

        data = []

        if color_var and color_var != "" and color_var != "None":
            # Create separate trace for each group
            groups = df[color_var].dropna().unique()
            for i, group in enumerate(groups):
                fill_color, border_color = group_colors[i % len(group_colors)]
                mask = df[color_var] == group
                data.append({
                    "type": "scatter",
                    "x": df.loc[mask, x_var].tolist(),
                    "y": df.loc[mask, y_var].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": fill_color,
                        "size": 8,
                        "line": {"color": border_color, "width": 1.5}
                    },
                    "name": str(group)
                })
        else:
            data.append({
                "type": "scatter",
                "x": df[x_var].tolist(),
                "y": df[y_var].tolist(),
                "mode": "markers",
                "marker": {
                    "color": "rgba(74, 159, 110, 0.5)",
                    "size": 8,
                    "line": {"color": "#4a9f6e", "width": 1.5}
                },
                "name": y_var
            })

        if trendline:
            import numpy as np
            x = df[x_var].dropna()
            y = df[y_var].loc[x.index].dropna()
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                data.append({
                    "type": "scatter",
                    "x": [float(x.min()), float(x.max())],
                    "y": [float(p(x.min())), float(p(x.max()))],
                    "mode": "lines",
                    "line": {"color": "#e8c547", "dash": "dash"},
                    "name": "Trendline"
                })

        title = f"{y_var} vs {x_var}"
        if color_var and color_var != "" and color_var != "None":
            title += f" (by {color_var})"

        result["plots"].append({
            "title": title,
            "data": data,
            "layout": {
                "height": 300,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
                "showlegend": len(data) > 1
            }
        })

    elif analysis_id == "heatmap":
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()

        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{"type": "heatmap", "z": corr.values.tolist(), "x": numeric_cols, "y": numeric_cols, "colorscale": "RdBu", "zmid": 0}],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "pareto":
        category = config.get("category") or config.get("var")
        value_col = config.get("value")

        if value_col and value_col != "":
            # Sum values by category
            counts = df.groupby(category)[value_col].sum().sort_values(ascending=False)
        else:
            # Count occurrences
            counts = df[category].value_counts()

        cumulative = counts.cumsum() / counts.sum() * 100

        result["plots"].append({
            "title": f"Pareto Chart - {category}",
            "data": [
                {"type": "bar", "x": [str(x) for x in counts.index.tolist()], "y": counts.values.tolist(), "name": "Count", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "x": [str(x) for x in counts.index.tolist()], "y": cumulative.values.tolist(), "name": "Cumulative %", "yaxis": "y2", "line": {"color": "#fdcb6e"}}
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "yaxis2": {"overlaying": "y", "side": "right", "range": [0, 100], "title": "Cumulative %"}
            }
        })

        # Find 80% cutoff
        cutoff_idx = (cumulative >= 80).idxmax() if (cumulative >= 80).any() else counts.index[-1]
        vital_few = counts.loc[:cutoff_idx]
        result["summary"] = f"Pareto Analysis\n\nTotal categories: {len(counts)}\nVital few (80%): {len(vital_few)} categories\n\nTop contributors:\n"
        for cat, val in counts.head(5).items():
            pct = val / counts.sum() * 100
            result["summary"] += f"  {cat}: {val} ({pct:.1f}%)\n"

    elif analysis_id == "matrix":
        import numpy as np
        vars_list = config.get("vars", [])
        color_var = config.get("color")

        if len(vars_list) < 2:
            result["summary"] = "Please select at least 2 variables for matrix plot."
            return result

        # Create scatter matrix
        n_vars = len(vars_list)
        fig_data = []

        for i, y_var in enumerate(vars_list):
            for j, x_var in enumerate(vars_list):
                row = n_vars - i
                col = j + 1

                if i == j:
                    # Diagonal: histogram
                    fig_data.append({
                        "type": "histogram",
                        "x": df[x_var].dropna().tolist(),
                        "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })
                else:
                    # Off-diagonal: scatter
                    fig_data.append({
                        "type": "scatter",
                        "x": df[x_var].tolist(),
                        "y": df[y_var].tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 3},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })

        # Build layout with subplots
        layout = {
            "template": "plotly_dark",
            "height": 100 + n_vars * 120,
            "showlegend": False,
        }

        # Create axis layout
        for i in range(n_vars):
            col = i + 1
            row = n_vars - i
            x_key = f"xaxis{col if col > 1 else ''}"
            y_key = f"yaxis{row if row > 1 else ''}"

            layout[x_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[i] if row == 1 else "",
                "showticklabels": row == 1
            }
            layout[y_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[n_vars - 1 - i] if col == 1 else "",
                "showticklabels": col == 1
            }

        result["plots"].append({
            "title": "Matrix Plot",
            "data": fig_data,
            "layout": layout
        })

    elif analysis_id == "timeseries":
        import numpy as np
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
                "line": {"color": theme_colors[i % len(theme_colors)]}
            }
            traces.append(trace)

        result["plots"].append({
            "title": f"Time Series: {', '.join(y_cols)}",
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": x_col},
                "yaxis": {"title": "Value"},
                "showlegend": len(y_cols) > 1
            }
        })

    elif analysis_id == "probability":
        import numpy as np
        from scipy import stats

        var = config.get("var")
        dist = config.get("dist", "norm")

        x = df[var].dropna().values
        x_sorted = np.sort(x)
        n = len(x_sorted)

        # Calculate plotting positions (Hazen)
        pp = (np.arange(1, n + 1) - 0.5) / n

        # Get theoretical quantiles for chosen distribution
        if dist == "norm":
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"
        elif dist == "lognorm":
            theoretical = stats.lognorm.ppf(pp, s=1)
            dist_name = "Lognormal"
            x_sorted = np.log(x_sorted[x_sorted > 0])
            theoretical = stats.norm.ppf(pp[:len(x_sorted)])
        elif dist == "expon":
            theoretical = stats.expon.ppf(pp)
            dist_name = "Exponential"
        elif dist == "weibull":
            # Weibull with shape=2
            theoretical = stats.weibull_min.ppf(pp, c=2)
            dist_name = "Weibull"
        else:
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"

        # Fit line
        slope, intercept = np.polyfit(theoretical, x_sorted, 1)
        fit_line = slope * theoretical + intercept

        result["plots"].append({
            "title": f"Probability Plot ({dist_name}): {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": x_sorted.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": fit_line.tolist(),
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "Fit Line"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": f"Theoretical Quantiles ({dist_name})"},
                "yaxis": {"title": var}
            }
        })

        # Anderson-Darling test for normality
        if dist == "norm":
            ad_stat = stats.anderson(x)
            result["summary"] = f"Probability Plot ({dist_name})\n\nAnderson-Darling: {ad_stat.statistic:.4f}\nCritical values (15%, 10%, 5%, 2.5%, 1%):\n{ad_stat.critical_values}"

    elif analysis_id == "individual_value_plot":
        import numpy as np
        from scipy import stats

        var = config.get("var")
        groupby = config.get("group")
        show_mean = config.get("show_mean", True)
        show_ci = config.get("show_ci", True)
        conf = config.get("confidence", 0.95)

        traces = []
        has_groups = groupby and groupby != "" and groupby != "None"

        if has_groups:
            groups = df[groupby].dropna().unique()
            for i, grp in enumerate(groups):
                vals = df[df[groupby] == grp][var].dropna().values
                color = theme_colors[i % len(theme_colors)]
                jitter = np.random.uniform(-0.15, 0.15, len(vals))
                traces.append({
                    "type": "scatter", "x": (np.full(len(vals), i) + jitter).tolist(),
                    "y": vals.tolist(), "mode": "markers",
                    "marker": {"color": color, "size": 6, "opacity": 0.6},
                    "name": str(grp), "showlegend": True
                })
                if show_mean and len(vals) > 0:
                    mean_val = float(np.mean(vals))
                    traces.append({
                        "type": "scatter", "x": [i], "y": [mean_val],
                        "mode": "markers", "marker": {"color": color, "size": 14, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                        "showlegend": False
                    })
                if show_ci and len(vals) > 1:
                    mean_val = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    ci = stats.t.interval(conf, df=len(vals)-1, loc=mean_val, scale=se)
                    traces.append({
                        "type": "scatter", "x": [i, i], "y": [ci[0], ci[1]],
                        "mode": "lines", "line": {"color": color, "width": 2},
                        "showlegend": False
                    })

            result["plots"].append({
                "title": f"Individual Value Plot: {var} by {groupby}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"tickvals": list(range(len(groups))), "ticktext": [str(g) for g in groups], "title": groupby}, "yaxis": {"title": var}, "showlegend": True}
            })
        else:
            vals = df[var].dropna().values
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            traces.append({
                "type": "scatter", "x": jitter.tolist(), "y": vals.tolist(),
                "mode": "markers", "marker": {"color": "rgba(74,159,110,0.6)", "size": 6},
                "name": var, "showlegend": False
            })
            if show_mean and len(vals) > 0:
                mean_val = float(np.mean(vals))
                traces.append({
                    "type": "scatter", "x": [0], "y": [mean_val],
                    "mode": "markers", "marker": {"color": "#e89547", "size": 14, "symbol": "diamond"},
                    "showlegend": False
                })
            result["plots"].append({
                "title": f"Individual Value Plot: {var}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"showticklabels": False}, "yaxis": {"title": var}}
            })

    elif analysis_id == "interval_plot":
        import numpy as np
        from scipy import stats

        var = config.get("var")
        groupby = config.get("group")
        conf = config.get("confidence", 0.95)

        if not groupby or groupby == "" or groupby == "None":
            result["summary"] = "Interval plot requires a grouping variable."
        else:
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
                ci = stats.t.interval(conf, df=len(vals)-1, loc=m, scale=se)
                means.append(m)
                ci_lo.append(m - ci[0])
                ci_hi.append(ci[1] - m)
                labels.append(str(grp))

            overall_mean = float(df[var].dropna().mean())

            traces = [{
                "type": "scatter", "x": labels, "y": means,
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 10, "symbol": "diamond"},
                "error_y": {"type": "data", "symmetric": False, "array": ci_hi, "arrayminus": ci_lo, "color": "#4a9f6e", "thickness": 2, "width": 8},
                "name": f"Mean \u00b1 {conf*100:.0f}% CI"
            }, {
                "type": "scatter", "x": [labels[0], labels[-1]], "y": [overall_mean, overall_mean],
                "mode": "lines", "line": {"color": "#e89547", "dash": "dash", "width": 1.5},
                "name": f"Overall Mean ({overall_mean:.4g})"
            }]

            result["plots"].append({
                "title": f"Interval Plot: {var} by {groupby} ({conf*100:.0f}% CI)",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"title": groupby}, "yaxis": {"title": var}, "showlegend": True}
            })

    elif analysis_id == "dotplot":
        import numpy as np

        var = config.get("var")
        groupby = config.get("group")

        has_groups = groupby and groupby != "" and groupby != "None"

        if has_groups:
            groups = df[groupby].dropna().unique()
            traces = []
            for i, grp in enumerate(groups):
                vals = np.sort(df[df[groupby] == grp][var].dropna().values)
                color = theme_colors[i % len(theme_colors)]
                # Stack dots: for repeated values, offset y
                if len(vals) == 0:
                    continue
                rng = float(np.ptp(vals)) if np.ptp(vals) > 0 else 1.0
                bin_width = rng / max(min(int(np.sqrt(len(vals))), 30), 5) if len(vals) > 1 else rng
                if bin_width == 0:
                    bin_width = 1.0
                binned = np.round(vals / bin_width) * bin_width
                x_out, y_out = [], []
                from collections import Counter
                counts = Counter()
                for bv, rv in zip(binned, vals):
                    counts[bv] += 1
                    x_out.append(float(rv))
                    y_out.append(i + (counts[bv] - 1) * 0.08)
                traces.append({
                    "type": "scatter", "x": x_out, "y": y_out,
                    "mode": "markers", "marker": {"color": color, "size": 7},
                    "name": str(grp)
                })
            result["plots"].append({
                "title": f"Dotplot: {var} by {groupby}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"title": var}, "yaxis": {"tickvals": list(range(len(groups))), "ticktext": [str(g) for g in groups], "title": groupby}}
            })
        else:
            vals = np.sort(df[var].dropna().values)
            if len(vals) > 0:
                rng = float(np.ptp(vals)) if np.ptp(vals) > 0 else 1.0
                bin_width = rng / max(min(int(np.sqrt(len(vals))), 30), 5) if len(vals) > 1 else rng
                if bin_width == 0:
                    bin_width = 1.0
                binned = np.round(vals / bin_width) * bin_width
                x_out, y_out = [], []
                from collections import Counter
                counts = Counter()
                for bv, rv in zip(binned, vals):
                    counts[bv] += 1
                    x_out.append(float(rv))
                    y_out.append(counts[bv])
                result["plots"].append({
                    "title": f"Dotplot: {var}",
                    "data": [{"type": "scatter", "x": x_out, "y": y_out, "mode": "markers", "marker": {"color": "#4a9f6e", "size": 7}, "name": var}],
                    "layout": {"height": 300, "xaxis": {"title": var}, "yaxis": {"title": "Count"}}
                })

    # =====================================================================
    # Bubble Chart (backend)
    # =====================================================================
    elif analysis_id == "bubble":
        x_var = config.get("x")
        y_var = config.get("y")
        size_var = config.get("size")
        color_var = config.get("color")

        size_vals = df[size_var].dropna().values.astype(float)
        s_min, s_max = float(size_vals.min()), float(size_vals.max())
        s_range = s_max - s_min if s_max != s_min else 1.0

        data = []
        if color_var and color_var not in ("", "None"):
            for i, group in enumerate(df[color_var].dropna().unique()):
                sub = df.loc[df[color_var] == group]
                sizes = ((sub[size_var].fillna(s_min).astype(float) - s_min) / s_range * 35 + 5).tolist()
                data.append({
                    "type": "scatter", "mode": "markers",
                    "x": sub[x_var].tolist(), "y": sub[y_var].tolist(),
                    "marker": {"size": sizes, "sizemode": "diameter",
                               "color": theme_colors[i % len(theme_colors)], "opacity": 0.7},
                    "name": str(group),
                })
        else:
            sizes = ((df[size_var].fillna(s_min).astype(float) - s_min) / s_range * 35 + 5).tolist()
            data.append({
                "type": "scatter", "mode": "markers",
                "x": df[x_var].tolist(), "y": df[y_var].tolist(),
                "marker": {"size": sizes, "sizemode": "diameter",
                           "color": "rgba(74,159,110,0.6)",
                           "line": {"color": "#4a9f6e", "width": 1}},
                "name": size_var,
            })

        result["plots"].append({
            "title": f"Bubble Chart: {y_var} vs {x_var} (size: {size_var})",
            "data": data,
            "layout": {"height": 400, "xaxis": {"title": x_var}, "yaxis": {"title": y_var},
                       "showlegend": bool(color_var and color_var not in ("", "None"))},
        })
        result["summary"] = (
            f"Bubble Chart\n\nX: {x_var}, Y: {y_var}, Size: {size_var}"
            + (f", Color: {color_var}" if color_var and color_var not in ("", "None") else "")
            + f"\nObservations: {len(df)}"
        )

    # =====================================================================
    # Parallel Coordinates
    # =====================================================================
    elif analysis_id == "parallel_coordinates":
        dims = config.get("dimensions", [])
        color_col = config.get("color")

        if len(dims) < 2:
            result["summary"] = "Select at least 2 dimensions for parallel coordinates."
            return result

        dimensions = []
        for col in dims:
            if np.issubdtype(df[col].dtype, np.number):
                dimensions.append({
                    "label": col,
                    "values": df[col].fillna(df[col].median()).tolist(),
                    "range": [float(df[col].min()), float(df[col].max())],
                })
            else:
                cats = df[col].dropna().unique().tolist()
                cat_map = {c: i for i, c in enumerate(cats)}
                dimensions.append({
                    "label": col,
                    "values": df[col].map(cat_map).fillna(-1).astype(int).tolist(),
                    "tickvals": list(range(len(cats))),
                    "ticktext": [str(c) for c in cats],
                })

        trace = {"type": "parcoords", "dimensions": dimensions}
        if color_col and color_col not in ("", "None"):
            if np.issubdtype(df[color_col].dtype, np.number):
                trace["line"] = {
                    "color": df[color_col].fillna(0).tolist(),
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#d94a4a"]],
                    "showscale": True, "colorbar": {"title": color_col},
                }
            else:
                cats = df[color_col].dropna().unique().tolist()
                cat_map = {c: i for i, c in enumerate(cats)}
                trace["line"] = {
                    "color": df[color_col].map(cat_map).fillna(0).tolist(),
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#4a90d9"], [1, "#d94a4a"]],
                    "showscale": True,
                }
        else:
            trace["line"] = {"color": "#4a9f6e"}

        result["plots"].append({
            "title": f"Parallel Coordinates ({len(dims)} dimensions)",
            "data": [trace],
            "layout": {"height": 450},
        })
        result["summary"] = (
            f"Parallel Coordinates Plot\n\n"
            f"Dimensions: {len(dims)}\nObservations: {len(df)}\n\n"
            f"Drag axis ranges to filter. Reorder axes by dragging labels."
        )

    # =====================================================================
    # Contour Plot
    # =====================================================================
    elif analysis_id == "contour":
        from scipy.interpolate import griddata as scipy_griddata

        x_col = config.get("x")
        y_col = config.get("y")
        z_col = config.get("z")

        common = df[[x_col, y_col, z_col]].dropna()
        if len(common) < 4:
            result["summary"] = "Need at least 4 non-missing data points for contour interpolation."
            return result

        x, y, z = common[x_col].values.astype(float), common[y_col].values.astype(float), common[z_col].values.astype(float)

        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi_grid = scipy_griddata((x, y), z, (xi_grid, yi_grid), method="cubic")
        nan_mask = np.isnan(zi_grid)
        if nan_mask.any():
            zi_linear = scipy_griddata((x, y), z, (xi_grid, yi_grid), method="linear")
            zi_grid[nan_mask] = zi_linear[nan_mask]

        result["plots"].append({
            "title": f"Contour: {z_col}",
            "data": [
                {
                    "type": "contour",
                    "x": xi.tolist(), "y": yi.tolist(),
                    "z": np.where(np.isnan(zi_grid), None, zi_grid).tolist(),
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#d94a4a"]],
                    "contours": {"showlabels": True, "labelfont": {"size": 10, "color": "#fff"}},
                    "colorbar": {"title": z_col},
                },
                {
                    "type": "scatter", "mode": "markers",
                    "x": x.tolist(), "y": y.tolist(),
                    "marker": {"color": "#fff", "size": 3, "opacity": 0.5},
                    "showlegend": False,
                },
            ],
            "layout": {"height": 450, "xaxis": {"title": x_col}, "yaxis": {"title": y_col}},
        })
        result["summary"] = (
            f"Contour Plot\n\nX: {x_col}, Y: {y_col}, Z: {z_col}\n"
            f"Data points: {len(x)}\nGrid: 50x50, interpolation: cubic\n\n"
            f"Contour lines show estimated iso-response levels. White dots show data locations."
        )

    # =====================================================================
    # 3D Surface Plot
    # =====================================================================
    elif analysis_id == "surface_3d":
        from scipy.interpolate import griddata as scipy_griddata

        x_col = config.get("x")
        y_col = config.get("y")
        z_col = config.get("z")

        common = df[[x_col, y_col, z_col]].dropna()
        if len(common) < 4:
            result["summary"] = "Need at least 4 non-missing data points for surface interpolation."
            return result

        x, y, z = common[x_col].values.astype(float), common[y_col].values.astype(float), common[z_col].values.astype(float)

        xi = np.linspace(x.min(), x.max(), 40)
        yi = np.linspace(y.min(), y.max(), 40)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi_grid = scipy_griddata((x, y), z, (xi_grid, yi_grid), method="cubic")
        nan_mask = np.isnan(zi_grid)
        if nan_mask.any():
            zi_linear = scipy_griddata((x, y), z, (xi_grid, yi_grid), method="linear")
            zi_grid[nan_mask] = zi_linear[nan_mask]

        result["plots"].append({
            "title": f"3D Surface: {z_col}",
            "data": [{
                "type": "surface",
                "x": xi.tolist(), "y": yi.tolist(),
                "z": np.where(np.isnan(zi_grid), None, zi_grid).tolist(),
                "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#d94a4a"]],
                "colorbar": {"title": z_col},
            }],
            "layout": {"height": 500,
                       "scene": {"xaxis": {"title": x_col}, "yaxis": {"title": y_col}, "zaxis": {"title": z_col}}},
        })
        result["summary"] = (
            f"3D Surface Plot\n\nX: {x_col}, Y: {y_col}, Z: {z_col}\n"
            f"Data points: {len(x)}\nGrid: 40x40\n\n"
            f"Drag to rotate. Scroll to zoom."
        )

    # =====================================================================
    # Contour Plot Overlay
    # =====================================================================
    elif analysis_id == "contour_overlay":
        """
        Contour Plot Overlay -- overlay contour lines from multiple responses
        on a single plot. Useful for DOE optimization (finding regions that
        satisfy multiple response targets simultaneously).
        """
        from scipy.interpolate import griddata as scipy_griddata

        x_col_co = config.get("x")
        y_col_co = config.get("y")
        z_cols_co = config.get("z_columns") or config.get("responses", [])

        if isinstance(z_cols_co, str):
            z_cols_co = [z_cols_co]

        if len(z_cols_co) < 2:
            result["summary"] = "Need at least 2 response variables for contour overlay."
            return result

        all_cols_co = [x_col_co, y_col_co] + z_cols_co
        common_co = df[all_cols_co].dropna()

        if len(common_co) < 4:
            result["summary"] = "Need at least 4 data points for contour interpolation."
            return result

        x_co = common_co[x_col_co].values.astype(float)
        y_co = common_co[y_col_co].values.astype(float)

        xi_co = np.linspace(x_co.min(), x_co.max(), 50)
        yi_co = np.linspace(y_co.min(), y_co.max(), 50)
        xi_grid_co, yi_grid_co = np.meshgrid(xi_co, yi_co)

        overlay_colors = [
            [[0, "rgba(74,159,110,0.1)"], [1, "rgba(74,159,110,0.6)"]],
            [[0, "rgba(74,144,217,0.1)"], [1, "rgba(74,144,217,0.6)"]],
            [[0, "rgba(217,74,74,0.1)"], [1, "rgba(217,74,74,0.6)"]],
            [[0, "rgba(232,197,71,0.1)"], [1, "rgba(232,197,71,0.6)"]],
            [[0, "rgba(122,106,154,0.1)"], [1, "rgba(122,106,154,0.6)"]],
        ]
        line_colors = ["#4a9f6e", "#4a90d9", "#d94a4a", "#e8c547", "#7a6a9a"]

        overlay_traces = []
        summary_lines = []

        for zi, z_col_name in enumerate(z_cols_co):
            z_vals = common_co[z_col_name].values.astype(float)
            zi_grid = scipy_griddata((x_co, y_co), z_vals, (xi_grid_co, yi_grid_co), method="cubic")
            nan_mask = np.isnan(zi_grid)
            if nan_mask.any():
                zi_linear = scipy_griddata((x_co, y_co), z_vals, (xi_grid_co, yi_grid_co), method="linear")
                zi_grid[nan_mask] = zi_linear[nan_mask]

            color_idx = zi % len(line_colors)
            overlay_traces.append({
                "type": "contour",
                "x": xi_co.tolist(), "y": yi_co.tolist(),
                "z": np.where(np.isnan(zi_grid), None, zi_grid).tolist(),
                "name": z_col_name,
                "contours": {"showlabels": True, "labelfont": {"size": 9, "color": line_colors[color_idx]},
                             "coloring": "lines"},
                "line": {"color": line_colors[color_idx], "width": 2},
                "showscale": False,
            })
            summary_lines.append(f"  {z_col_name}: range [{np.nanmin(z_vals):.3f}, {np.nanmax(z_vals):.3f}]")

        # Add data points
        overlay_traces.append({
            "type": "scatter", "mode": "markers",
            "x": x_co.tolist(), "y": y_co.tolist(),
            "marker": {"color": "#fff", "size": 4, "opacity": 0.6, "line": {"color": "#333", "width": 1}},
            "name": "Data points", "showlegend": True,
        })

        result["plots"].append({
            "title": f"Contour Overlay: {', '.join(z_cols_co)}",
            "data": overlay_traces,
            "layout": {"height": 500, "xaxis": {"title": x_col_co}, "yaxis": {"title": y_col_co}}
        })

        result["summary"] = (
            f"Contour Plot Overlay\n\n"
            f"X: {x_col_co}, Y: {y_col_co}\n"
            f"Responses overlaid ({len(z_cols_co)}):\n" +
            "\n".join(summary_lines) +
            f"\n\nData points: {len(x_co)}\n"
            f"Each response shown with distinct contour line color.\n"
            f"Use this to identify regions satisfying multiple targets."
        )

    # =====================================================================
    # Mosaic Plot
    # =====================================================================
    elif analysis_id == "mosaic":
        row_var = config.get("row_var")
        col_var = config.get("col_var")

        # Cap levels
        if df[row_var].nunique() > 15:
            top_rows = df[row_var].value_counts().head(15).index
            df = df[df[row_var].isin(top_rows)]
        if df[col_var].nunique() > 15:
            top_cols = df[col_var].value_counts().head(15).index
            df = df[df[col_var].isin(top_cols)]

        ct = pd.crosstab(df[row_var], df[col_var])
        row_totals = ct.sum(axis=1)
        grand_total = int(ct.values.sum())
        col_names = ct.columns.tolist()
        row_names = ct.index.tolist()

        shapes = []
        annotations = []
        x_cursor = 0.0

        for ri, row_name in enumerate(row_names):
            row_width = float(row_totals[row_name]) / grand_total
            y_cursor = 0.0
            col_total = float(row_totals[row_name])

            for ci, col_name in enumerate(col_names):
                cell_val = float(ct.loc[row_name, col_name])
                cell_height = cell_val / col_total if col_total > 0 else 0

                shapes.append({
                    "type": "rect",
                    "x0": x_cursor, "x1": x_cursor + row_width,
                    "y0": y_cursor, "y1": y_cursor + cell_height,
                    "fillcolor": theme_colors[ci % len(theme_colors)],
                    "opacity": 0.7,
                    "line": {"color": "#1a1a2e", "width": 1},
                })
                if cell_height > 0.06 and row_width > 0.06:
                    annotations.append({
                        "x": x_cursor + row_width / 2,
                        "y": y_cursor + cell_height / 2,
                        "text": str(int(cell_val)),
                        "showarrow": False,
                        "font": {"color": "#fff", "size": 10},
                    })
                y_cursor += cell_height

            annotations.append({
                "x": x_cursor + row_width / 2, "y": -0.04,
                "text": str(row_name), "showarrow": False,
                "font": {"color": "#b0b0b0", "size": 9},
            })
            x_cursor += row_width

        # Legend traces for column categories
        legend_traces = [
            {
                "type": "scatter", "x": [None], "y": [None], "mode": "markers",
                "marker": {"color": theme_colors[ci % len(theme_colors)], "size": 10},
                "name": str(col_name), "showlegend": True,
            }
            for ci, col_name in enumerate(col_names)
        ]

        result["plots"].append({
            "title": f"Mosaic: {row_var} x {col_var}",
            "data": legend_traces,
            "layout": {
                "height": 400,
                "shapes": shapes, "annotations": annotations,
                "xaxis": {"range": [0, 1], "title": row_var, "showticklabels": False},
                "yaxis": {"range": [-0.08, 1], "title": col_var, "showticklabels": False},
                "showlegend": True,
            },
        })
        result["summary"] = (
            f"Mosaic Plot\n\nRow: {row_var} ({len(row_names)} levels)\n"
            f"Column: {col_var} ({len(col_names)} levels)\n"
            f"Total: {grand_total} observations\n\n"
            f"Tile widths proportional to row totals. Heights proportional to column distribution within each row."
        )

    # -- Bayesian SPC Suite ------------------------------------------------

    elif analysis_id == "bayes_spc_capability":
        from scipy.stats import t as tdist
        col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
        data = df[col].dropna().values.astype(float)
        usl_raw = config.get("usl")
        lsl_raw = config.get("lsl")
        usl = float(usl_raw) if usl_raw not in (None, "", "null") else None
        lsl = float(lsl_raw) if lsl_raw not in (None, "", "null") else None
        if usl is None and lsl is None:
            result["summary"] = "<<COLOR:error>>At least one spec limit (USL or LSL) is required<</COLOR>>"
            return result
        if usl is not None and lsl is not None and usl <= lsl:
            result["summary"] = "<<COLOR:error>>USL must be greater than LSL<</COLOR>>"
            return result
        target = config.get("target")
        if target not in (None, "", "null"):
            target = float(target)
        elif usl is not None and lsl is not None:
            target = (usl + lsl) / 2.0
        elif usl is not None:
            target = usl
        else:
            target = lsl
        spec_label = f"USL={usl}" if lsl is None else (f"LSL={lsl}" if usl is None else f"USL={usl}, LSL={lsl}")
        n_mc = int(config.get("n_mc", 10000))
        prior_type = config.get("prior_type", "weakly_informative")

        n = len(data)
        x_bar = float(np.mean(data))
        s = float(np.std(data, ddof=1)) if n > 1 else 0.01

        if prior_type == "informative":
            pp = config.get("prior_params", {})
            mu0 = float(pp.get("mu0", x_bar))
            nu0 = float(pp.get("nu0", 5))
            alpha0 = float(pp.get("alpha0", 3))
            beta0 = float(pp.get("beta0", s**2 * 2))
        elif prior_type == "historical":
            pp = config.get("prior_params", {})
            hist_mean = float(pp.get("hist_mean", x_bar))
            hist_std = float(pp.get("hist_std", s))
            hist_n = int(pp.get("hist_n", 30))
            mu0, nu0, alpha0, beta0 = hist_mean, float(hist_n), hist_n / 2.0, hist_n / 2.0 * hist_std**2
        else:
            s2 = float(np.var(data, ddof=1)) if n > 1 else 1.0
            mu0, nu0, alpha0, beta0 = x_bar, 1.0, 2.0, max(s2, 1e-10)

        mu_n, nu_n, alpha_n, beta_n = _nig_posterior_update(data, mu0, nu0, alpha0, beta0)
        mu_samples, sigma_samples = _nig_sample(mu_n, nu_n, alpha_n, beta_n, n_mc)
        cpk_samples = _cpk_from_params(mu_samples, sigma_samples, usl, lsl)

        cpk_median = float(np.median(cpk_samples))
        cpk_ci = (float(np.percentile(cpk_samples, 2.5)), float(np.percentile(cpk_samples, 97.5)))
        p_gt_1 = float(np.mean(cpk_samples > 1.0))
        p_gt_133 = float(np.mean(cpk_samples > 1.33))
        p_gt_167 = float(np.mean(cpk_samples > 1.67))
        p_gt_2 = float(np.mean(cpk_samples > 2.0))

        if s > 0:
            if usl is not None and lsl is not None:
                cpk_freq = float(min((usl - x_bar) / (3 * s), (x_bar - lsl) / (3 * s)))
            elif usl is not None:
                cpk_freq = float((usl - x_bar) / (3 * s))
            else:
                cpk_freq = float((x_bar - lsl) / (3 * s))
        else:
            cpk_freq = 0.0

        rng_pp = np.random.default_rng(123)
        x_pred = rng_pp.normal(loc=mu_samples, scale=sigma_samples)
        oos_mask = np.zeros(len(x_pred), dtype=bool)
        if lsl is not None:
            oos_mask |= (x_pred < lsl)
        if usl is not None:
            oos_mask |= (x_pred > usl)
        p_oos = float(np.mean(oos_mask))

        df_t = 2 * alpha_n
        loc_t = mu_n
        scale_t = float(np.sqrt(beta_n * (nu_n + 1) / (alpha_n * nu_n)))
        pp_dist = tdist(df=df_t, loc=loc_t, scale=scale_t)
        dpmo = p_oos * 1e6

        sigma_99 = float(np.percentile(sigma_samples, 99))
        sigma_iqr = float(np.percentile(sigma_samples, 75) - np.percentile(sigma_samples, 25))
        sigma_warning = ""
        if sigma_iqr > 0 and sigma_99 > 5 * sigma_iqr + float(np.median(sigma_samples)):
            sigma_warning = "Data may be non-normal, from mixed processes, or contain outliers. Consider transformations or a mixture model."

        if p_gt_133 >= 0.95:
            verdict_color, verdict = "success", "CAPABLE — P(Cpk > 1.33) >= 95%"
        elif p_gt_133 >= 0.80:
            verdict_color, verdict = "highlight", "MARGINAL — P(Cpk > 1.33) between 80-95%"
        else:
            verdict_color, verdict = "error", "NOT CAPABLE — P(Cpk > 1.33) < 80%"

        sep70 = '=' * 70
        sep40 = '-' * 40
        summary = f"<<COLOR:accent>>{sep70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN PROCESS CAPABILITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{sep70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    <<COLOR:highlight>>Spec:<</COLOR>> {spec_label}    <<COLOR:highlight>>Target:<</COLOR>> {target}\n\n"
        summary += f"<<COLOR:accent>>{sep40}<</COLOR>>\n"
        summary += f"<<COLOR:title>>Posterior Cpk<</COLOR>>\n"
        summary += f"  Median: <<COLOR:highlight>>{cpk_median:.4f}<</COLOR>>    95% CI: [{cpk_ci[0]:.4f}, {cpk_ci[1]:.4f}]\n"
        summary += f"  Frequentist Cpk (point estimate): {cpk_freq:.4f}\n\n"
        summary += f"<<COLOR:accent>>{sep40}<</COLOR>>\n"
        summary += f"<<COLOR:title>>Probability Table<</COLOR>>\n"
        summary += f"  P(Cpk > 1.00) = <<COLOR:{'success' if p_gt_1 > 0.9 else 'error'}>>{p_gt_1:.1%}<</COLOR>>\n"
        summary += f"  P(Cpk > 1.33) = <<COLOR:{'success' if p_gt_133 > 0.9 else 'error'}>>{p_gt_133:.1%}<</COLOR>>\n"
        summary += f"  P(Cpk > 1.67) = <<COLOR:{'success' if p_gt_167 > 0.9 else 'text'}>>{p_gt_167:.1%}<</COLOR>>\n"
        summary += f"  P(Cpk > 2.00) = {p_gt_2:.1%}\n\n"
        summary += f"<<COLOR:accent>>{sep40}<</COLOR>>\n"
        summary += f"<<COLOR:title>>Posterior Predictive<</COLOR>>\n"
        summary += f"  P(out of spec) = {p_oos:.6f}    DPMO = <<COLOR:highlight>>{dpmo:.0f}<</COLOR>>\n"
        summary += "  (No 1.5-sigma shift assumption — uncertainty is a first-class citizen)\n\n"
        summary += f"<<COLOR:{verdict_color}>>{verdict}<</COLOR>>\n\n"
        if sigma_warning:
            summary += f"<<COLOR:error>>Warning: {sigma_warning}<</COLOR>>\n\n"
        summary += "<<COLOR:text>>Accounts for parameter uncertainty; no long-term shift assumption needed.\n"
        summary += "When n is small, uncertainty is large — this is reflected in the credible interval.<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {
            "cpk_median": cpk_median, "cpk_ci_lower": cpk_ci[0], "cpk_ci_upper": cpk_ci[1],
            "cpk_frequentist": cpk_freq, "p_cpk_gt_1": p_gt_1, "p_cpk_gt_133": p_gt_133, "p_cpk_gt_167": p_gt_167,
            "p_cpk_gt_2": p_gt_2, "dpmo": dpmo, "p_out_of_spec": p_oos,
        }

        cpk_hist_vals, cpk_hist_edges = np.histogram(cpk_samples, bins=80)
        cpk_hist_centers = (cpk_hist_edges[:-1] + cpk_hist_edges[1:]) / 2
        ci_mask = (cpk_hist_centers >= cpk_ci[0]) & (cpk_hist_centers <= cpk_ci[1])
        cpk_ymax = int(max(cpk_hist_vals))
        result["plots"].append({
            "title": "Posterior Distribution of Cpk",
            "data": [
                {"type": "bar", "x": cpk_hist_centers.tolist(), "y": cpk_hist_vals.tolist(),
                 "marker": {"color": ["rgba(74,159,110,0.7)" if m else "rgba(74,159,110,0.2)" for m in ci_mask]},
                 "name": "Posterior", "showlegend": False},
                {"type": "scatter", "x": [1.0, 1.0], "y": [0, cpk_ymax], "mode": "lines",
                 "line": {"color": "#e89547", "dash": "dash", "width": 2}, "name": "Cpk = 1.0"},
                {"type": "scatter", "x": [1.33, 1.33], "y": [0, cpk_ymax], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash", "width": 2}, "name": "Cpk = 1.33"},
                {"type": "scatter", "x": [cpk_freq, cpk_freq], "y": [0, cpk_ymax], "mode": "lines",
                 "line": {"color": "#5b9bd5", "width": 2}, "name": "Frequentist"},
            ],
            "layout": {"height": 300, "xaxis": {"title": "Cpk"}, "yaxis": {"title": "Count"},
                        "annotations": [{"x": cpk_median, "y": cpk_ymax * 0.9,
                                         "text": f"Median: {cpk_median:.3f}", "showarrow": True,
                                         "arrowhead": 2, "font": {"color": "#4a9f6e"}}]}
        })

        lo_bound = (lsl - 3 * s) if lsl is not None else (data.min() - 3 * s)
        hi_bound = (usl + 3 * s) if usl is not None else (data.max() + 3 * s)
        x_range = np.linspace(min(lo_bound, data.min()), max(hi_bound, data.max()), 300)
        pp_pdf = pp_dist.pdf(x_range)
        from scipy.stats import norm
        norm_pdf = norm.pdf(x_range, loc=x_bar, scale=s) if s > 0 else np.zeros_like(x_range)
        data_hist_vals, data_hist_edges = np.histogram(data, bins=40, density=True)
        data_hist_centers = (data_hist_edges[:-1] + data_hist_edges[1:]) / 2
        peak_y = max(max(pp_pdf), max(norm_pdf)) if len(pp_pdf) > 0 else 1
        pred_traces = [
            {"type": "bar", "x": data_hist_centers.tolist(), "y": data_hist_vals.tolist(),
             "marker": {"color": "rgba(74,159,110,0.3)"}, "name": "Data", "showlegend": True},
            {"type": "scatter", "x": x_range.tolist(), "y": pp_pdf.tolist(), "mode": "lines",
             "line": {"color": "#e89547", "width": 2}, "name": "Predictive (Student-t)"},
            {"type": "scatter", "x": x_range.tolist(), "y": norm_pdf.tolist(), "mode": "lines",
             "line": {"color": "#5b9bd5", "dash": "dash", "width": 1.5}, "name": "Normal fit"},
        ]
        if lsl is not None:
            pred_traces.append({"type": "scatter", "x": [lsl, lsl], "y": [0, peak_y], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dot", "width": 2}, "name": "LSL"})
        if usl is not None:
            pred_traces.append({"type": "scatter", "x": [usl, usl], "y": [0, peak_y], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dot", "width": 2}, "name": "USL"})
        ann_x = (usl + lsl) / 2 if (usl is not None and lsl is not None) else (usl if usl is not None else lsl)
        result["plots"].append({
            "title": "Posterior Predictive vs Spec Limits",
            "data": pred_traces,
            "layout": {"height": 300, "xaxis": {"title": col},
                        "annotations": [{"x": ann_x, "y": max(pp_pdf) * 0.95,
                                         "text": f"DPMO: {dpmo:.0f}", "showarrow": False,
                                         "font": {"color": "#e89547", "size": 13}}]}
        })

        thresholds = np.linspace(0.5, 3.0, 100)
        p_above = [float(np.mean(cpk_samples > t)) for t in thresholds]
        result["plots"].append({
            "title": "P(Cpk > Threshold)",
            "data": [
                {"type": "scatter", "x": thresholds.tolist(), "y": p_above, "mode": "lines",
                 "line": {"color": "#4a9f6e", "width": 2.5}, "name": "P(Cpk > threshold)"},
                {"type": "scatter", "x": [0.5, 3.0], "y": [0.95, 0.95], "mode": "lines",
                 "line": {"color": "#e89547", "dash": "dash"}, "name": "95% confidence"},
            ],
            "layout": {"height": 250, "xaxis": {"title": "Threshold"}, "yaxis": {"title": "Probability", "range": [0, 1.05]}}
        })

        overlay_traces = [
            {"type": "histogram", "x": data.tolist(), "nbinsx": 40, "histnorm": "probability density",
             "marker": {"color": "rgba(74,159,110,0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
            {"type": "scatter", "x": x_range.tolist(), "y": pp_pdf.tolist(), "mode": "lines",
             "line": {"color": "#e89547", "width": 2.5}, "name": "Posterior Predictive"},
            {"type": "scatter", "x": x_range.tolist(), "y": norm_pdf.tolist(), "mode": "lines",
             "line": {"color": "#5b9bd5", "dash": "dash", "width": 1.5}, "name": "Normal Fit"},
        ]
        if lsl is not None:
            overlay_traces.append({"type": "scatter", "x": [lsl, lsl], "y": [0, peak_y], "mode": "lines",
                 "line": {"color": "#e85747", "width": 2}, "name": "LSL", "showlegend": False})
        if usl is not None:
            overlay_traces.append({"type": "scatter", "x": [usl, usl], "y": [0, peak_y], "mode": "lines",
                 "line": {"color": "#e85747", "width": 2}, "name": "USL", "showlegend": False})
        result["plots"].append({
            "title": "Data vs Predictive Distribution",
            "data": overlay_traces,
            "layout": {"height": 300, "xaxis": {"title": col}, "yaxis": {"title": "Density"}}
        })

        result["guide_observation"] = f"Bayesian capability: Cpk median {cpk_median:.3f} [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}], P(Cpk>1.33)={p_gt_133:.1%}, DPMO={dpmo:.0f}"

    elif analysis_id == "bayes_spc_changepoint":
        from scipy.special import logsumexp, gammaln
        col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
        data = df[col].dropna().values.astype(float)
        hazard_rate = float(config.get("hazard_rate", 0.01))
        min_seg = int(config.get("min_segment_length", 5))

        n = len(data)
        max_rl = min(500, n)
        log_H = np.log(hazard_rate)
        log_1mH = np.log(1 - hazard_rate)

        n_cal = min(50, max(10, n // 5))
        cal_data = data[:n_cal]
        s2_cal = float(np.var(cal_data, ddof=1)) if n_cal > 1 else 1.0
        mu0_cp = float(np.mean(cal_data))
        nu0_cp = 1.0
        alpha0_cp = 2.0
        beta0_cp = max(s2_cal * alpha0_cp, 1e-10)

        log_R = -np.inf * np.ones((n + 1, max_rl + 1))
        log_R[0, 0] = 0.0

        ss_n = np.zeros(max_rl + 1)
        ss_sum = np.zeros(max_rl + 1)
        ss_sum2 = np.zeros(max_rl + 1)

        cp_prob = np.zeros(n)
        shift_prob = np.zeros(n)

        for t in range(n):
            x = data[t]
            rl_range = min(t + 1, max_rl)
            log_pred = np.full(rl_range + 1, -np.inf)

            for r in range(rl_range + 1):
                nn = ss_n[r]
                if nn == 0:
                    nu_r = nu0_cp
                    alpha_r = alpha0_cp
                    mu_r = mu0_cp
                    beta_r = beta0_cp
                else:
                    xbar_r = ss_sum[r] / nn
                    nu_r = nu0_cp + nn
                    mu_r = (nu0_cp * mu0_cp + nn * xbar_r) / nu_r
                    alpha_r = alpha0_cp + nn / 2.0
                    beta_r = beta0_cp + 0.5 * (ss_sum2[r] - nn * xbar_r**2) + \
                             (nn * nu0_cp * (xbar_r - mu0_cp)**2) / (2.0 * nu_r)
                    beta_r = max(beta_r, 1e-10)

                df_r = 2 * alpha_r
                scale_r = np.sqrt(beta_r * (nu_r + 1) / (alpha_r * nu_r))
                scale_r = max(scale_r, 1e-10)
                z = (x - mu_r) / scale_r
                log_pred[r] = (gammaln((df_r + 1) / 2) - gammaln(df_r / 2) -
                               0.5 * np.log(df_r * np.pi) - np.log(scale_r) -
                               ((df_r + 1) / 2) * np.log(1 + z**2 / df_r))

            log_growth = np.full(max_rl + 1, -np.inf)
            for r in range(rl_range + 1):
                if r < max_rl and log_R[t, r] > -1e300:
                    log_growth[r + 1] = log_R[t, r] + log_pred[r] + log_1mH

            log_cp_terms = []
            for r in range(rl_range + 1):
                if log_R[t, r] > -1e300:
                    log_cp_terms.append(log_R[t, r] + log_pred[r] + log_H)
            log_cp = logsumexp(log_cp_terms) if log_cp_terms else -np.inf

            log_R[t + 1, 0] = log_cp
            for r in range(1, max_rl + 1):
                log_R[t + 1, r] = log_growth[r]

            log_evidence = logsumexp(log_R[t + 1, :max_rl + 1])
            log_R[t + 1, :max_rl + 1] -= log_evidence

            cp_prob[t] = np.exp(log_R[t + 1, 0])

            if t + 1 <= max_rl:
                shift_prob[t] = float(np.clip(1.0 - np.exp(log_R[t + 1, t + 1]), 0.0, 1.0))
            else:
                shift_prob[t] = 1.0

            new_ss_n = np.zeros(max_rl + 1)
            new_ss_sum = np.zeros(max_rl + 1)
            new_ss_sum2 = np.zeros(max_rl + 1)
            for r in range(1, min(t + 2, max_rl + 1)):
                new_ss_n[r] = ss_n[r - 1] + 1
                new_ss_sum[r] = ss_sum[r - 1] + x
                new_ss_sum2[r] = ss_sum2[r - 1] + x**2
            new_ss_n[0] = 0
            new_ss_sum[0] = 0
            new_ss_sum2[0] = 0
            ss_n, ss_sum, ss_sum2 = new_ss_n, new_ss_sum, new_ss_sum2

        changepoints = []
        for t in range(min_seg, n - min_seg):
            if shift_prob[t] > 0.5 and (t == 0 or shift_prob[t - 1] <= 0.5):
                if not changepoints or (t - changepoints[-1]) >= min_seg:
                    changepoints.append(t)

        boundaries = [0] + changepoints + [n]
        segments = []
        for i in range(len(boundaries) - 1):
            seg_data = data[boundaries[i]:boundaries[i + 1]]
            segments.append({
                "start": int(boundaries[i]), "end": int(boundaries[i + 1]),
                "mean": float(np.mean(seg_data)), "std": float(np.std(seg_data, ddof=1)) if len(seg_data) > 1 else 0,
                "n": len(seg_data)
            })

        sep70 = '=' * 70
        summary = f"<<COLOR:accent>>{sep70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION (BOCPD)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{sep70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    <<COLOR:highlight>>Hazard rate:<</COLOR>> {hazard_rate}\n\n"

        if changepoints:
            summary += f"<<COLOR:success>>Detected {len(changepoints)} change point(s):<</COLOR>>\n\n"
            for i, cp in enumerate(changepoints):
                seg_before = segments[i]
                seg_after = segments[i + 1]
                summary += f"  <<COLOR:highlight>>Change {i+1}:<</COLOR>> observation {cp}, P(shifted) = {shift_prob[cp]:.3f}\n"
                summary += f"    Before: mean = {seg_before['mean']:.4f}, std = {seg_before['std']:.4f} (n={seg_before['n']})\n"
                summary += f"    After:  mean = {seg_after['mean']:.4f}, std = {seg_after['std']:.4f} (n={seg_after['n']})\n\n"
        else:
            summary += f"<<COLOR:text>>No significant change points detected (threshold: P > 0.5)<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"n_changepoints": len(changepoints), "changepoints": changepoints, "segments": segments}

        rl_display = min(max_rl, 100)
        heatmap_data = np.exp(log_R[1:n + 1, :rl_display]).T
        result["plots"].append({
            "title": "Run-Length Posterior",
            "data": [{"type": "heatmap", "z": heatmap_data.tolist(),
                       "colorscale": "Viridis", "showscale": True,
                       "colorbar": {"title": "P(r)"}}],
            "layout": {"height": 300, "xaxis": {"title": "Observation"}, "yaxis": {"title": "Run Length"}}
        })

        colors = [f"rgba({int(255*p)},{int(255*(1-p))},80,0.8)" for p in shift_prob]
        result["plots"].append({
            "title": "Shift Probability — P(process has changed)",
            "data": [
                {"type": "scatter", "y": shift_prob.tolist(), "mode": "lines",
                 "line": {"color": "#d94a4a", "width": 2}, "name": "P(shifted)"},
                {"type": "scatter", "x": [0, n], "y": [0.5, 0.5], "mode": "lines",
                 "line": {"color": "#e89547", "dash": "dash"}, "name": "Threshold (50%)"},
                {"type": "scatter", "x": [0, n], "y": [0.95, 0.95], "mode": "lines",
                 "line": {"color": "#d94a4a", "dash": "dot", "width": 1}, "name": "Alarm (95%)"},
            ],
            "layout": {"height": 250, "xaxis": {"title": "Observation"},
                        "yaxis": {"title": "P(shifted)", "range": [0, 1.05]}}
        })

        proc_data = [{"type": "scatter", "y": data.tolist(), "mode": "lines+markers",
                       "marker": {"size": 4, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}, "name": col}]
        for i, cp in enumerate(changepoints):
            proc_data.append({"type": "scatter", "x": [cp, cp], "y": [float(data.min()), float(data.max())],
                              "mode": "lines", "line": {"color": "#e85747", "width": 2, "dash": "dash"}, "name": f"Change {i+1}"})
        for seg in segments:
            proc_data.append({"type": "scatter", "x": [seg["start"], seg["end"] - 1],
                              "y": [seg["mean"], seg["mean"]], "mode": "lines",
                              "line": {"color": "#e89547", "width": 2}, "name": f"mean={seg['mean']:.2f}", "showlegend": False})
        result["plots"].append({
            "title": "Process Data with Change Points",
            "data": proc_data,
            "layout": {"height": 350, "xaxis": {"title": "Observation"}, "yaxis": {"title": col}}
        })

        result["guide_observation"] = f"BOCPD detected {len(changepoints)} change point(s) in {n} observations"

    elif analysis_id == "bayes_spc_control":
        col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
        data = df[col].dropna().values.astype(float)
        ref_mean = config.get("reference_mean")
        ref_std = config.get("reference_std")
        shift_size = float(config.get("shift_size", 1.5))
        trans_prob = float(config.get("transition_prob", 0.01))

        n = len(data)
        n_ref = min(20, n)
        if ref_mean is None or ref_mean == "" or ref_mean == "null":
            ref_mean = float(np.mean(data[:n_ref]))
        else:
            ref_mean = float(ref_mean)
        if ref_std is None or ref_std == "" or ref_std == "null":
            ref_std = float(np.std(data[:n_ref], ddof=1)) if n_ref > 1 else float(np.std(data, ddof=1))
        else:
            ref_std = float(ref_std)
        ref_std = max(ref_std, 1e-10)

        delta = shift_size * ref_std
        p_recover = 0.05

        from scipy.stats import norm

        log_p_ic = np.zeros(n)
        log_alpha_ic = 0.0
        log_alpha_sh = np.log(1e-10)

        seq_mu = np.zeros(n)
        seq_ci_lo = np.zeros(n)
        seq_ci_hi = np.zeros(n)
        from scipy.stats import t as tdist_sc
        mu0_s, nu0_s, alpha0_s, beta0_s = ref_mean, 1.0, 2.0, max(ref_std**2, 1e-10)

        for t in range(n):
            x = data[t]

            ll_ic = norm.logpdf(x, loc=ref_mean, scale=ref_std)
            ll_sh_plus = norm.logpdf(x, loc=ref_mean + delta, scale=ref_std)
            ll_sh_minus = norm.logpdf(x, loc=ref_mean - delta, scale=ref_std)
            ll_sh = np.logaddexp(ll_sh_plus, ll_sh_minus) - np.log(2)

            log_t_ic_ic = np.log(1 - trans_prob)
            log_t_ic_sh = np.log(trans_prob)
            log_t_sh_ic = np.log(p_recover)
            log_t_sh_sh = np.log(1 - p_recover)

            from scipy.special import logsumexp as _lse
            new_log_alpha_ic = _lse([log_alpha_ic + log_t_ic_ic, log_alpha_sh + log_t_sh_ic]) + ll_ic
            new_log_alpha_sh = _lse([log_alpha_ic + log_t_ic_sh, log_alpha_sh + log_t_sh_sh]) + ll_sh

            log_evidence = _lse([new_log_alpha_ic, new_log_alpha_sh])
            log_alpha_ic = new_log_alpha_ic - log_evidence
            log_alpha_sh = new_log_alpha_sh - log_evidence

            log_p_ic[t] = log_alpha_ic
            p_shifted = 1.0 - np.exp(log_alpha_ic)

            seg_data = data[:t + 1]
            mu_n_s, nu_n_s, alpha_n_s, beta_n_s = _nig_posterior_update(seg_data, mu0_s, nu0_s, alpha0_s, beta0_s)
            seq_mu[t] = mu_n_s
            if alpha_n_s > 0 and nu_n_s > 0 and beta_n_s > 0:
                scale_s = np.sqrt(beta_n_s / (alpha_n_s * nu_n_s))
                ci_half = tdist_sc.ppf(0.975, df=2 * alpha_n_s) * scale_s
                seq_ci_lo[t] = mu_n_s - ci_half
                seq_ci_hi[t] = mu_n_s + ci_half
            else:
                seq_ci_lo[t] = mu_n_s
                seq_ci_hi[t] = mu_n_s

        p_shifted_arr = 1.0 - np.exp(log_p_ic)
        n_alarms = int(np.sum(p_shifted_arr > 0.5))

        sep70 = '=' * 70
        summary = f"<<COLOR:accent>>{sep70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CONTROL CHART<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{sep70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}    <<COLOR:highlight>>Ref mean:<</COLOR>> {ref_mean:.4f}    <<COLOR:highlight>>Ref std:<</COLOR>> {ref_std:.4f}\n"
        summary += f"<<COLOR:highlight>>Shift size:<</COLOR>> {shift_size}-sigma    <<COLOR:highlight>>P(shift):<</COLOR>> {trans_prob}\n\n"

        if n_alarms > 0:
            first_alarm = int(np.argmax(p_shifted_arr > 0.5))
            summary += f"<<COLOR:error>>ALERT: {n_alarms} observations with P(shifted) > 0.5<</COLOR>>\n"
            summary += f"<<COLOR:highlight>>First alarm at observation {first_alarm}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Final posterior mean:<</COLOR>> {seq_mu[-1]:.4f} [{seq_ci_lo[-1]:.4f}, {seq_ci_hi[-1]:.4f}]\n"
        else:
            summary += f"<<COLOR:success>>Process appears in control — no observations with P(shifted) > 0.5<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Final posterior mean:<</COLOR>> {seq_mu[-1]:.4f} [{seq_ci_lo[-1]:.4f}, {seq_ci_hi[-1]:.4f}]\n"

        result["summary"] = summary
        result["statistics"] = {"n_alarms": n_alarms, "ref_mean": ref_mean, "ref_std": ref_std,
                                 "final_mu": float(seq_mu[-1]), "final_ci": [float(seq_ci_lo[-1]), float(seq_ci_hi[-1])]}

        colors = [f"rgb({int(255*p)},{int(255*(1-p))},80)" for p in p_shifted_arr]
        result["plots"].append({
            "title": "Process Data — Colored by P(shifted)",
            "data": [
                {"type": "scatter", "y": data.tolist(), "mode": "markers",
                 "marker": {"color": colors, "size": 6, "line": {"color": "#333", "width": 0.5}},
                 "name": col, "showlegend": False},
                {"type": "scatter", "y": data.tolist(), "mode": "lines",
                 "line": {"color": "rgba(150,150,150,0.3)", "width": 1}, "showlegend": False},
                {"type": "scatter", "x": [0, n - 1], "y": [ref_mean, ref_mean], "mode": "lines",
                 "line": {"color": "#5b9bd5", "dash": "dash"}, "name": "Reference mean"},
            ],
            "layout": {"height": 300, "xaxis": {"title": "Observation"}, "yaxis": {"title": col}}
        })

        x_idx = list(range(n))
        result["plots"].append({
            "title": "Sequential Posterior for mean",
            "data": [
                {"type": "scatter", "x": x_idx, "y": seq_ci_hi.tolist(), "mode": "lines",
                 "line": {"color": "rgba(74,159,110,0.2)", "width": 0}, "showlegend": False},
                {"type": "scatter", "x": x_idx, "y": seq_ci_lo.tolist(), "mode": "lines",
                 "line": {"color": "rgba(74,159,110,0.2)", "width": 0}, "fill": "tonexty",
                 "fillcolor": "rgba(74,159,110,0.15)", "name": "95% CI"},
                {"type": "scatter", "x": x_idx, "y": seq_mu.tolist(), "mode": "lines",
                 "line": {"color": "#4a9f6e", "width": 2}, "name": "Posterior mean"},
                {"type": "scatter", "x": [0, n - 1], "y": [ref_mean, ref_mean], "mode": "lines",
                 "line": {"color": "#5b9bd5", "dash": "dash"}, "name": "Reference"},
            ],
            "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": "mean"}}
        })

        alarm_colors = [f"rgba({int(255*p)},{int(255*(1-p))},80,0.8)" for p in p_shifted_arr]
        result["plots"].append({
            "title": "Shift Probability Timeline",
            "data": [
                {"type": "bar", "y": p_shifted_arr.tolist(), "marker": {"color": alarm_colors}, "name": "P(shifted)", "showlegend": False},
                {"type": "scatter", "x": [0, n - 1], "y": [0.5, 0.5], "mode": "lines",
                 "line": {"color": "#e89547", "dash": "dash", "width": 2}, "name": "Alarm threshold"},
            ],
            "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": "P(shifted)", "range": [0, 1.05]}}
        })

        result["guide_observation"] = f"Bayesian control chart: {n_alarms} alarms in {n} observations (shift={shift_size}-sigma)"

    elif analysis_id == "bayes_spc_acceptance":
        from scipy.stats import beta as betadist
        col = config.get("measurement") or df.select_dtypes(include="number").columns[0]
        data = df[col].dropna().values.astype(float)

        aql = float(config.get("aql", 0.01))
        threshold = float(config.get("acceptance_threshold", 0.95))
        prior_alpha = float(config.get("prior_alpha", 1))
        prior_beta = float(config.get("prior_beta", 1))

        manual_defectives = config.get("defectives")
        manual_sample = config.get("sample_size")
        if manual_defectives is not None and manual_sample is not None:
            k = int(manual_defectives)
            n_total = int(manual_sample)
        else:
            usl_a = config.get("usl")
            lsl_a = config.get("lsl")
            n_total = len(data)
            k = 0
            if usl_a is not None and usl_a != "" and usl_a != "null":
                k += int(np.sum(data > float(usl_a)))
            if lsl_a is not None and lsl_a != "" and lsl_a != "null":
                k += int(np.sum(data < float(lsl_a)))

        post_alpha = prior_alpha + k
        post_beta_param = prior_beta + n_total - k
        p_accept = float(betadist.cdf(aql, post_alpha, post_beta_param))
        post_mean = post_alpha / (post_alpha + post_beta_param)
        post_ci = (float(betadist.ppf(0.025, post_alpha, post_beta_param)),
                   float(betadist.ppf(0.975, post_alpha, post_beta_param)))

        if p_accept >= threshold:
            decision = "ACCEPT"
            decision_color = "success"
        elif p_accept <= (1 - threshold):
            decision = "REJECT"
            decision_color = "error"
        else:
            decision = "CONTINUE SAMPLING"
            decision_color = "highlight"

        seq_p_accept = []
        seq_k = 0
        earliest_accept = None
        earliest_reject = None
        for i in range(n_total):
            if manual_defectives is not None:
                seq_k_i = int(round(k * (i + 1) / n_total))
            else:
                val = data[i] if i < len(data) else data[-1]
                is_def = False
                usl_a = config.get("usl")
                lsl_a = config.get("lsl")
                if usl_a is not None and usl_a != "" and usl_a != "null" and val > float(usl_a):
                    is_def = True
                if lsl_a is not None and lsl_a != "" and lsl_a != "null" and val < float(lsl_a):
                    is_def = True
                seq_k += int(is_def)
                seq_k_i = seq_k

            pa_i = float(betadist.cdf(aql, prior_alpha + seq_k_i, prior_beta + (i + 1) - seq_k_i))
            seq_p_accept.append(pa_i)

            if earliest_accept is None and pa_i >= threshold:
                earliest_accept = i + 1
            if earliest_reject is None and pa_i <= (1 - threshold):
                earliest_reject = i + 1

        boundary_n = list(range(1, n_total + 1))
        accept_boundary = []
        reject_boundary = []
        for ni in boundary_n:
            max_k_accept = -1
            min_k_reject = ni + 1
            for ki in range(ni + 1):
                pa = betadist.cdf(aql, prior_alpha + ki, prior_beta + ni - ki)
                if pa >= threshold:
                    max_k_accept = ki
                if pa <= (1 - threshold) and ki < min_k_reject:
                    min_k_reject = ki
            accept_boundary.append(max_k_accept if max_k_accept >= 0 else None)
            reject_boundary.append(min_k_reject if min_k_reject <= ni else None)

        sep70 = '=' * 70
        sep40 = '-' * 40
        summary = f"<<COLOR:accent>>{sep70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN ACCEPTANCE SAMPLING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{sep70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n_total}    <<COLOR:highlight>>Defectives:<</COLOR>> {k}    <<COLOR:highlight>>AQL:<</COLOR>> {aql}\n"
        summary += f"<<COLOR:highlight>>Prior:<</COLOR>> Beta({prior_alpha}, {prior_beta})    <<COLOR:highlight>>Threshold:<</COLOR>> {threshold}\n\n"
        summary += f"<<COLOR:accent>>{sep40}<</COLOR>>\n"
        summary += f"<<COLOR:title>>Posterior for Defect Rate<</COLOR>>\n"
        summary += f"  Mean: <<COLOR:highlight>>{post_mean:.6f}<</COLOR>>    95% CI: [{post_ci[0]:.6f}, {post_ci[1]:.6f}]\n"
        summary += f"  P(p < AQL) = <<COLOR:{'success' if p_accept > threshold else 'error'}>>{p_accept:.4f}<</COLOR>>\n\n"
        summary += f"<<COLOR:{decision_color}>>Decision: {decision}<</COLOR>>\n\n"

        if earliest_accept:
            summary += f"<<COLOR:success>>Earliest acceptance possible at n = {earliest_accept}<</COLOR>>\n"
        if earliest_reject:
            summary += f"<<COLOR:error>>Earliest rejection at n = {earliest_reject}<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {
            "defectives": k, "sample_size": n_total, "defect_rate_mean": post_mean,
            "defect_rate_ci": list(post_ci), "p_accept": p_accept, "decision": decision,
            "earliest_accept": earliest_accept, "earliest_reject": earliest_reject,
        }

        x_range = np.linspace(0, min(max(post_ci[1] * 3, aql * 5), 1.0), 300)
        post_pdf = betadist.pdf(x_range, post_alpha, post_beta_param)
        prior_pdf = betadist.pdf(x_range, prior_alpha, prior_beta) if prior_alpha > 0 and prior_beta > 0 else np.zeros_like(x_range)
        result["plots"].append({
            "title": "Posterior for Defect Rate",
            "data": [
                {"type": "scatter", "x": x_range.tolist(), "y": post_pdf.tolist(), "mode": "lines",
                 "fill": "tozeroy", "fillcolor": "rgba(74,159,110,0.2)",
                 "line": {"color": "#4a9f6e", "width": 2}, "name": "Posterior"},
                {"type": "scatter", "x": x_range.tolist(), "y": prior_pdf.tolist(), "mode": "lines",
                 "line": {"color": "#888", "dash": "dash", "width": 1.5}, "name": "Prior"},
                {"type": "scatter", "x": [aql, aql], "y": [0, max(post_pdf) if len(post_pdf) > 0 else 1],
                 "mode": "lines", "line": {"color": "#e85747", "width": 2}, "name": f"AQL = {aql}"},
            ],
            "layout": {"height": 300, "xaxis": {"title": "Defect Rate (p)"}, "yaxis": {"title": "Density"},
                        "annotations": [{"x": post_mean, "y": max(post_pdf) * 0.9 if len(post_pdf) > 0 else 0.5,
                                         "text": f"P(p<AQL)={p_accept:.3f}", "showarrow": True,
                                         "font": {"color": "#4a9f6e"}}]}
        })

        result["plots"].append({
            "title": "Sequential P(p < AQL) — Earliest Stopping",
            "data": [
                {"type": "scatter", "x": list(range(1, n_total + 1)), "y": seq_p_accept, "mode": "lines",
                 "line": {"color": "#4a9f6e", "width": 2}, "name": "P(p < AQL)"},
                {"type": "scatter", "x": [1, n_total], "y": [threshold, threshold], "mode": "lines",
                 "line": {"color": "#4a9f6e", "dash": "dash"}, "name": f"Accept ({threshold})"},
                {"type": "scatter", "x": [1, n_total], "y": [1 - threshold, 1 - threshold], "mode": "lines",
                 "line": {"color": "#e85747", "dash": "dash"}, "name": f"Reject ({1-threshold:.2f})"},
            ],
            "layout": {"height": 250, "xaxis": {"title": "Items Inspected"}, "yaxis": {"title": "P(p < AQL)", "range": [0, 1.05]},
                        "annotations": ([{"x": earliest_accept, "y": threshold, "text": f"Accept @ n={earliest_accept}",
                                          "showarrow": True, "font": {"color": "#4a9f6e"}}] if earliest_accept else [])}
        })

        accept_y = [b if b is not None else None for b in accept_boundary]
        reject_y = [b if b is not None else None for b in reject_boundary]
        boundary_plots = [
            {"type": "scatter", "x": boundary_n, "y": accept_y, "mode": "lines",
             "line": {"color": "#4a9f6e", "width": 2}, "name": "Accept boundary", "connectgaps": False},
            {"type": "scatter", "x": boundary_n, "y": reject_y, "mode": "lines",
             "line": {"color": "#e85747", "width": 2}, "name": "Reject boundary", "connectgaps": False},
        ]
        boundary_plots.append({"type": "scatter", "x": [n_total], "y": [k], "mode": "markers",
                                "marker": {"color": "#e89547", "size": 12, "symbol": "star"},
                                "name": f"Observed ({n_total}, {k})"})
        result["plots"].append({
            "title": "Decision Boundaries",
            "data": boundary_plots,
            "layout": {"height": 300, "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Defectives (k)"}}
        })

        result["guide_observation"] = f"Bayesian acceptance: {k}/{n_total} defectives, P(p<AQL)={p_accept:.3f}, decision={decision}"

    return result
