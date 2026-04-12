"""SPC analysis package — drop-in replacement for the monolithic spc.run_spc_analysis.

Dispatches analysis_id to the appropriate sub-module function.
Each function takes (df, config) and returns result dict.
"""

from .advanced import (
    run_cusum,
    run_ewma,
    run_laney_p,
    run_laney_u,
    run_moving_average,
    run_zone_chart,
)
from .capability import (
    run_between_within,
    run_capability,
    run_degradation_capability,
    run_nonnormal_capability,
)
from .conformal import (
    run_conformal_control,
    run_conformal_monitor,
    run_entropy_spc,
)
from .multivariate import (
    run_generalized_variance,
    run_mewma,
)
from .shewhart import (
    run_c_chart,
    run_imr,
    run_np_chart,
    run_p_chart,
    run_u_chart,
    run_xbar_r,
    run_xbar_s,
)

# analysis_id -> handler function
_DISPATCH = {
    "imr": run_imr,
    "xbar_r": run_xbar_r,
    "xbar_s": run_xbar_s,
    "p_chart": run_p_chart,
    "np_chart": run_np_chart,
    "c_chart": run_c_chart,
    "u_chart": run_u_chart,
    "cusum": run_cusum,
    "ewma": run_ewma,
    "laney_p": run_laney_p,
    "laney_u": run_laney_u,
    "moving_average": run_moving_average,
    "zone_chart": run_zone_chart,
    "capability": run_capability,
    "nonnormal_capability": run_nonnormal_capability,
    "degradation_capability": run_degradation_capability,
    "between_within": run_between_within,
    "mewma": run_mewma,
    "generalized_variance": run_generalized_variance,
    "conformal_control": run_conformal_control,
    "conformal_monitor": run_conformal_monitor,
    "entropy_spc": run_entropy_spc,
}


def run_spc_analysis(df, analysis_id, config):
    """Run SPC analysis — drop-in replacement for the monolithic dispatcher.

    Handles all 22 native analysis IDs plus bridge dispatches to:
    - g_chart / t_chart (rare events, handled in the original monolith)
    - bayes_spc_* (bridges to viz module)
    - bayes_doe_* (bridges to bayes_doe module)
    """
    # ── Bridge: Bayesian SPC suite lives in run_visualization ──
    if analysis_id.startswith("bayes_spc_"):
        from ..viz import run_visualization

        return run_visualization(df, analysis_id, config)

    # ── Bridge: Bayesian DOE suite ──
    if analysis_id.startswith("bayes_doe_"):
        from ...bayes_doe import run_bayesian_doe

        return run_bayesian_doe(df, analysis_id, config)

    # ── G/T chart (rare events) — kept inline as in the monolith ──
    if analysis_id in ("g_chart", "t_chart"):
        return _run_g_t_chart(df, config)

    # ── Standard dispatch ──
    handler = _DISPATCH.get(analysis_id)
    if handler is not None:
        return handler(df, config)

    # Unknown analysis_id — return empty result
    return {
        "plots": [],
        "summary": f"Unknown SPC analysis_id: {analysis_id}",
        "guide_observation": "",
    }


def _run_g_t_chart(df, config):
    """Rare Events Charts — G Chart / T Chart."""
    import numpy as np
    from scipy import stats
    from scipy.stats import weibull_min

    from ..common import _narrative

    result = {"plots": [], "summary": "", "guide_observation": ""}

    var = config.get("var") or config.get("var1")
    chart_type = config.get("chart_type")  # "g" or "t" — auto-detect if not given

    col = df[var].dropna()
    values = col.values.astype(float)
    n = len(values)

    if n < 3:
        result["summary"] = "Need at least 3 data points for a rare events chart."
        return result

    # Auto-detect chart type
    if chart_type is None:
        if col.dtype in ["int64", "int32"] and (values == values.astype(int)).all():
            chart_type = "g"
        else:
            chart_type = "t"

    if chart_type == "g":
        # G Chart — geometric distribution
        g_bar = float(np.mean(values))
        p_est = 1 / (g_bar + 1) if g_bar > 0 else 0.5
        alpha_chart = 0.0027  # 3-sigma equivalent

        cl = g_bar
        if p_est > 0 and p_est < 1:
            ucl = float(stats.geom.ppf(1 - alpha_chart / 2, p_est) - 1)
            lcl = max(0, float(stats.geom.ppf(alpha_chart / 2, p_est) - 1))
        else:
            ucl = g_bar * 3
            lcl = 0

        chart_label = "G Chart (Opportunities Between Events)"
        y_label = "Count Between Events"
        shape = None
        scale = None

    else:
        # T Chart — time between events, Weibull/exponential transform
        try:
            shape, loc, scale = weibull_min.fit(values, floc=0)
        except Exception:
            shape, scale = 1.0, float(np.mean(values))

        mean_t = float(np.mean(values))
        cl = mean_t

        ucl = float(weibull_min.ppf(0.99865, shape, 0, scale))
        lcl = max(0, float(weibull_min.ppf(0.00135, shape, 0, scale)))

        chart_label = "T Chart (Time Between Events)"
        y_label = "Time Between Events"

    # Detect OOC points
    ooc_indices = []
    for i in range(n):
        if values[i] > ucl or values[i] < lcl:
            ooc_indices.append(i)

    x_axis = list(range(1, n + 1))
    colors = ["#e85747" if i in ooc_indices else "#4a9f6e" for i in range(n)]

    result["plots"].append(
        {
            "title": chart_label,
            "data": [
                {
                    "type": "scatter",
                    "x": x_axis,
                    "y": values.tolist(),
                    "mode": "lines+markers",
                    "line": {"color": "#4a9f6e"},
                    "marker": {"color": colors, "size": 6},
                    "name": var,
                    "customdata": [[i, "OOC" if i in ooc_indices else ""] for i in range(n)],
                    "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [cl, cl],
                    "mode": "lines",
                    "line": {"color": "#e8c547", "width": 1},
                    "name": f"CL = {cl:.2f}",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [ucl, ucl],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"UCL = {ucl:.2f}",
                },
                {
                    "type": "scatter",
                    "x": [1, n],
                    "y": [lcl, lcl],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1},
                    "name": f"LCL = {lcl:.2f}",
                },
            ],
            "layout": {
                "height": 390,
                "xaxis": {
                    "title": "Observation",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": y_label},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    summary = f"{chart_label}\n\n"
    summary += f"Variable: {var}\n"
    summary += f"Observations: {n}\n"
    summary += f"CL: {cl:.4f}\n"
    summary += f"UCL: {ucl:.4f}\n"
    summary += f"LCL: {lcl:.4f}\n"
    summary += f"Out-of-control: {len(ooc_indices)}\n"
    if ooc_indices:
        summary += f"OOC observations: {', '.join(str(i + 1) for i in ooc_indices)}\n"
    if chart_type == "t" and shape is not None:
        summary += f"\nWeibull shape: {shape:.3f} (1.0 = exponential)\n"
        summary += f"Weibull scale: {scale:.3f}\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"{'G' if chart_type == 'g' else 'T'} chart: CL={cl:.2f}, {len(ooc_indices)} OOC points out of {n}."
    )
    result["statistics"] = {
        "chart_type": chart_type,
        "n": n,
        "cl": cl,
        "ucl": ucl,
        "lcl": lcl,
        "ooc_count": len(ooc_indices),
        "ooc_indices": [i + 1 for i in ooc_indices],
    }

    # Narrative
    _gt_label = "G Chart (count between events)" if chart_type == "g" else "T Chart (time between events)"
    _gt_n_ooc = len(ooc_indices)
    if _gt_n_ooc == 0:
        _gt_verdict = f"{_gt_label} \u2014 process in control"
        _gt_body = (
            f"All {n} observations fall within control limits (CL = {cl:.2f}). The rate of rare events appears stable."
        )
    else:
        _gt_verdict = f"{_gt_label} \u2014 {_gt_n_ooc} out-of-control point{'s' if _gt_n_ooc > 1 else ''}"
        _gt_body = f"{_gt_n_ooc} of {n} observations exceed control limits, suggesting the event rate has shifted."
    result["narrative"] = _narrative(
        _gt_verdict,
        _gt_body,
        next_steps=(
            "Investigate OOC points for assignable causes. A cluster of short intervals suggests a worsening event rate."
            if _gt_n_ooc > 0
            else "Continue monitoring. Consider adding process improvement to reduce the baseline event rate."
        ),
        chart_guidance="Points above UCL indicate unusually long gaps between events (improvement). Points below LCL indicate unusually short gaps (deterioration).",
    )

    return result
