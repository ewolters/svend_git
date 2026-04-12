"""SPC advanced charts — CUSUM, EWMA, Laney P'/U', Moving Average, Zone Chart."""

import numpy as np

from ..common import _narrative
from .helpers import _spc_add_ooc_markers


def run_cusum(df, config):
    """CUSUM Chart - Cumulative Sum for detecting small shifts."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    target = float(config.get("target", 0))  # Target value
    k_param = float(config.get("k", 0.5))  # Slack value (typically 0.5)
    h_param = float(config.get("h", 5))  # Decision interval

    data = df[measurement].dropna().values
    n = len(data)

    if target == 0:
        target = np.mean(data)

    # Estimate standard deviation
    sigma = np.std(data, ddof=1)

    # Standardize
    z = (data - target) / sigma

    # Calculate CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    for i in range(n):
        if i == 0:
            cusum_pos[i] = max(0, z[i] - k_param)
            cusum_neg[i] = max(0, -z[i] - k_param)
        else:
            cusum_pos[i] = max(0, cusum_pos[i - 1] + z[i] - k_param)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - z[i] - k_param)

    # Detect signals
    signals_pos = np.where(cusum_pos > h_param)[0]
    signals_neg = np.where(cusum_neg > h_param)[0]

    cusum_chart_data = [
        {
            "type": "scatter",
            "y": cusum_pos.tolist(),
            "mode": "lines",
            "name": "CUSUM+",
            "line": {"color": "#4a9f6e", "width": 2},
            "customdata": [[i, ""] for i in range(n)],
            "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM+: %{y:.4f}<extra></extra>",
        },
        {
            "type": "scatter",
            "y": (-cusum_neg).tolist(),
            "mode": "lines",
            "name": "CUSUM-",
            "line": {"color": "#47a5e8", "width": 2},
        },
        {
            "type": "scatter",
            "y": [h_param] * n,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [-h_param] * n,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    # OOC markers for positive signals
    if len(signals_pos) > 0:
        cusum_chart_data.append(
            {
                "type": "scatter",
                "x": signals_pos.tolist(),
                "y": cusum_pos[signals_pos].tolist(),
                "mode": "markers",
                "name": "Signal (up)",
                "marker": {
                    "color": "#d94a4a",
                    "size": 9,
                    "symbol": "diamond",
                    "line": {"color": "#fff", "width": 1},
                },
                "showlegend": True,
                "customdata": [[int(i), "Upward shift signal"] for i in signals_pos],
                "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM+: %{y:.4f}<br>%{customdata[1]}<extra>Signal (up)</extra>",
            }
        )
    # OOC markers for negative signals
    if len(signals_neg) > 0:
        cusum_chart_data.append(
            {
                "type": "scatter",
                "x": signals_neg.tolist(),
                "y": (-cusum_neg[signals_neg]).tolist(),
                "mode": "markers",
                "name": "Signal (down)",
                "marker": {
                    "color": "#e89547",
                    "size": 9,
                    "symbol": "diamond",
                    "line": {"color": "#fff", "width": 1},
                },
                "showlegend": True,
                "customdata": [[int(i), "Downward shift signal"] for i in signals_neg],
                "hovertemplate": "Obs #%{customdata[0]}<br>CUSUM-: %{y:.4f}<br>%{customdata[1]}<extra>Signal (down)</extra>",
            }
        )
    result["plots"].append(
        {
            "title": "CUSUM Chart",
            "data": cusum_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "CUSUM"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    result["summary"] = (
        f"CUSUM Chart Analysis\n\nTarget: {target:.4f}\n\u03c3 estimate: {sigma:.4f}\nk (slack): {k_param}\nh (decision): {h_param}\n\nUpward shift signals: {len(signals_pos)} at points {list(signals_pos[:5])}{'...' if len(signals_pos) > 5 else ''}\nDownward shift signals: {len(signals_neg)} at points {list(signals_neg[:5])}{'...' if len(signals_neg) > 5 else ''}"
    )

    _n_up = len(signals_pos)
    _n_dn = len(signals_neg)
    _cusum_total = _n_up + _n_dn
    result["guide_observation"] = f"CUSUM chart: {_cusum_total} shift signal{'s' if _cusum_total != 1 else ''}." + (
        " Process appears stable." if _cusum_total == 0 else " Investigation recommended."
    )
    result["statistics"] = {
        "target": float(target),
        "sigma": float(sigma),
        "k_param": float(k_param),
        "h_param": float(h_param),
        "n": n,
        "n_signals_up": _n_up,
        "n_signals_down": _n_dn,
        "n_ooc": _cusum_total,
    }
    if _cusum_total == 0:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No cumulative shift signals detected (k={k_param}, h={h_param}). Process mean is stable around target ({target:.4f}).",
            next_steps="Process is stable \u2014 CUSUM is sensitive to small sustained shifts; absence of signals is strong evidence of stability.",
            chart_guidance="CUSUM+ (green) tracks upward drift; CUSUM\u2212 (blue) tracks downward drift. Crossing the red decision interval (h) signals a shift.",
        )
    else:
        _shift_desc = []
        if _n_up > 0:
            _shift_desc.append(f"{_n_up} upward")
        if _n_dn > 0:
            _shift_desc.append(f"{_n_dn} downward")
        result["narrative"] = _narrative(
            f"CUSUM signals detected \u2014 {' and '.join(_shift_desc)} shift{'s' if _cusum_total > 1 else ''}",
            f"The cumulative sum crossed the decision interval (h={h_param}), indicating a sustained shift from target ({target:.4f}).",
            next_steps="Identify when the shift began (first signal point) and investigate process changes at that time.",
            chart_guidance="CUSUM+ (green) tracks upward drift; CUSUM\u2212 (blue) tracks downward drift. Diamond markers show where the decision interval was breached.",
        )

    _cusum_ooc = sorted(set(list(signals_pos) + list(signals_neg)))
    if _cusum_ooc:
        result["what_if_data"] = {
            "type": "spc_intervention",
            "values": data.tolist(),
            "center": float(target),
            "ucl": float(target + h_param * sigma),
            "lcl": float(target - h_param * sigma),
            "sigma": float(sigma),
            "ooc_indices": [int(i) for i in _cusum_ooc],
            "first_ooc": int(min(_cusum_ooc)),
            "cusum_pos": cusum_pos.tolist(),
            "cusum_neg": cusum_neg.tolist(),
            "h_param": float(h_param),
            "k_param": float(k_param),
            "target": float(target),
        }

    return result


def run_ewma(df, config):
    """EWMA Chart - Exponentially Weighted Moving Average."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    target = float(config.get("target", 0))
    lambda_param = float(config.get("lambda", 0.2))  # Smoothing parameter
    L = float(config.get("L", 3))  # Control limit width

    data = df[measurement].dropna().values
    n = len(data)

    if target == 0:
        target = np.mean(data)

    sigma = np.std(data, ddof=1)

    # Calculate EWMA
    ewma = np.zeros(n)
    ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target

    for i in range(1, n):
        ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i - 1]

    # Control limits (they vary with time, approaching steady state)
    factor = lambda_param / (2 - lambda_param)
    ucl = target + L * sigma * np.sqrt(factor * (1 - (1 - lambda_param) ** (2 * np.arange(1, n + 1))))
    lcl = target - L * sigma * np.sqrt(factor * (1 - (1 - lambda_param) ** (2 * np.arange(1, n + 1))))

    # Steady-state limits
    ucl_ss = target + L * sigma * np.sqrt(factor)
    lcl_ss = target - L * sigma * np.sqrt(factor)

    # OOC detection for variable-limit EWMA
    ewma_ooc = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

    ewma_chart_data = [
        {
            "type": "scatter",
            "y": ewma.tolist(),
            "mode": "lines+markers",
            "name": "EWMA",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
            "line": {"color": "#4a9f6e"},
        },
        {
            "type": "scatter",
            "y": [target] * n,
            "mode": "lines",
            "name": "Target",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(ewma_chart_data, ewma, ewma_ooc)
    result["plots"].append(
        {
            "title": "EWMA Chart",
            "data": ewma_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "EWMA"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    result["summary"] = (
        f"EWMA Chart Analysis\n\nTarget: {target:.4f}\n\u03bb (smoothing): {lambda_param}\nL (sigma width): {L}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nOut-of-control points: {len(ewma_ooc)}"
    )

    _ewma_n_ooc = len(ewma_ooc)
    result["guide_observation"] = (
        f"EWMA chart: {_ewma_n_ooc} out-of-control point{'s' if _ewma_n_ooc != 1 else ''}."
        + (" Process appears stable." if _ewma_n_ooc == 0 else " Investigation recommended.")
    )
    result["statistics"] = {
        "target": float(target),
        "sigma": float(sigma),
        "lambda_param": float(lambda_param),
        "L": float(L),
        "ucl_ss": float(ucl_ss),
        "lcl_ss": float(lcl_ss),
        "n": n,
        "n_ooc": _ewma_n_ooc,
    }
    if _ewma_n_ooc == 0:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No EWMA points exceed control limits (\u03bb={lambda_param}, L={L}). The smoothed process mean is stable around target ({target:.4f}).",
            next_steps="Process is stable \u2014 EWMA is sensitive to small sustained shifts; absence of signals is strong evidence of stability.",
            chart_guidance="The EWMA line smooths out noise to reveal underlying trends. Limits widen from zero to steady state as the filter initialises.",
        )
    else:
        result["narrative"] = _narrative(
            f"EWMA signals detected \u2014 {_ewma_n_ooc} point{'s' if _ewma_n_ooc > 1 else ''} out of control",
            f"The smoothed mean has drifted outside the \u00b1{L}\u03c3 control limits, indicating a sustained shift from target ({target:.4f}).",
            next_steps="Identify when the drift began (first OOC point) and correlate with process changes. EWMA detects gradual shifts that Shewhart charts miss.",
            chart_guidance="The EWMA line smooths observations \u2014 OOC points (red diamonds) indicate the smoothed mean has shifted. Variable limits reflect the filter warm-up period.",
        )

    if ewma_ooc:
        result["what_if_data"] = {
            "type": "spc_intervention",
            "values": data.tolist(),
            "center": float(target),
            "ucl": float(ucl_ss),
            "lcl": float(lcl_ss),
            "sigma": float(sigma),
            "ooc_indices": [int(i) for i in ewma_ooc],
            "first_ooc": int(min(ewma_ooc)),
            "ewma": ewma.tolist(),
            "lambda_param": float(lambda_param),
            "target": float(target),
        }

    return result


def run_laney_p(df, config):
    """Laney P' Chart - P chart adjusted for overdispersion."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defectives = config.get("defectives")
    sample_size_col = config.get("sample_size")

    d = df[defectives].dropna().values
    n = df[sample_size_col].dropna().values
    p = d / n
    k = len(p)

    p_bar = np.sum(d) / np.sum(n)

    # Standard p-chart z-values
    z = (p - p_bar) / np.sqrt(p_bar * (1 - p_bar) / n)

    # Moving range of z-values for sigma_z
    mr_z = np.abs(np.diff(z))
    sigma_z = np.mean(mr_z) / 1.128  # d2 for n=2

    # Laney-adjusted limits
    ucl = p_bar + 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n)
    lcl = np.maximum(0, p_bar - 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n))

    ooc_indices = [i for i in range(k) if p[i] > ucl[i] or p[i] < lcl[i]]

    lp_chart_data = [
        {
            "type": "scatter",
            "y": p.tolist(),
            "mode": "lines+markers",
            "name": "p",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [p_bar] * k,
            "mode": "lines",
            "name": "p\u0304",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL'",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL'",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(lp_chart_data, p, ooc_indices)
    result["plots"].append(
        {
            "title": "Laney P' Chart",
            "data": lp_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Proportion"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
    result["summary"] = (
        f"Laney P' Chart Analysis\n\np\u0304: {p_bar:.4f} ({p_bar * 100:.2f}%)\n\u03c3z: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: \u03c3z > 1 indicates overdispersion \u2014 standard P chart would give too many false alarms."
    )

    _lp_n_ooc = len(ooc_indices)
    result["guide_observation"] = (
        f"Laney P' chart: {_lp_n_ooc} out-of-control points. \u03c3z = {sigma_z:.3f} ({disp})."
        + (" Process is stable." if _lp_n_ooc == 0 else " Investigation recommended.")
    )
    if _lp_n_ooc == 0:
        result["narrative"] = _narrative(
            f"Laney P' \u2014 Process in control (p\u0304 = {p_bar * 100:.2f}%, \u03c3z = {sigma_z:.3f})",
            f"All {k} samples within overdispersion-adjusted limits. {disp} detected (\u03c3z = {sigma_z:.3f})."
            + (" Standard P chart limits would be too tight." if sigma_z > 1 else ""),
            next_steps="Stable process. Laney adjustment accounts for extra-binomial variation that inflates false alarms on standard P charts.",
            chart_guidance="The adjusted limits (P') are wider than standard P chart limits when \u03c3z > 1, reducing false alarms.",
        )
    else:
        result["narrative"] = _narrative(
            f"Laney P' \u2014 {_lp_n_ooc} out-of-control point{'s' if _lp_n_ooc > 1 else ''}",
            f"{_lp_n_ooc} of {k} samples exceed the overdispersion-adjusted limits (\u03c3z = {sigma_z:.3f}). These are genuine signals even after accounting for {disp.lower()}.",
            next_steps="Investigate OOC subgroups. Because Laney P' already accounts for overdispersion, these signals are more trustworthy than standard P chart flags.",
            chart_guidance="Laney-adjusted limits are wider when overdispersion exists (\u03c3z > 1). Points still outside these wider limits are strong signals.",
        )

    return result


def run_laney_u(df, config):
    """Laney U' Chart - U chart adjusted for overdispersion."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    defects = config.get("defects")
    units = config.get("units")

    c = df[defects].dropna().values
    n = df[units].dropna().values
    u = c / n
    k = len(u)

    u_bar = np.sum(c) / np.sum(n)

    # Standard u-chart z-values
    z = (u - u_bar) / np.sqrt(u_bar / n)

    # Moving range of z-values for sigma_z
    mr_z = np.abs(np.diff(z))
    sigma_z = np.mean(mr_z) / 1.128

    # Laney-adjusted limits
    ucl = u_bar + 3 * sigma_z * np.sqrt(u_bar / n)
    lcl = np.maximum(0, u_bar - 3 * sigma_z * np.sqrt(u_bar / n))

    ooc_indices = [i for i in range(k) if u[i] > ucl[i] or u[i] < lcl[i]]

    lu_chart_data = [
        {
            "type": "scatter",
            "y": u.tolist(),
            "mode": "lines+markers",
            "name": "u",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",
                "size": 5,
                "line": {"color": "#4a9f6e", "width": 1.5},
            },
        },
        {
            "type": "scatter",
            "y": [u_bar] * k,
            "mode": "lines",
            "name": "\u016b",
            "line": {"color": "#00b894"},
        },
        {
            "type": "scatter",
            "y": ucl.tolist(),
            "mode": "lines",
            "name": "UCL'",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl.tolist(),
            "mode": "lines",
            "name": "LCL'",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(lu_chart_data, u, ooc_indices)
    result["plots"].append(
        {
            "title": "Laney U' Chart",
            "data": lu_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "Defects per Unit"},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
    result["summary"] = (
        f"Laney U' Chart Analysis\n\n\u016b: {u_bar:.4f}\n\u03c3z: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: \u03c3z > 1 indicates overdispersion \u2014 standard U chart would give too many false alarms."
    )

    _lu_n_ooc = len(ooc_indices)
    result["guide_observation"] = (
        f"Laney U' chart: {_lu_n_ooc} out-of-control points. \u03c3z = {sigma_z:.3f} ({disp})."
        + (" Process is stable." if _lu_n_ooc == 0 else " Investigation recommended.")
    )
    if _lu_n_ooc == 0:
        result["narrative"] = _narrative(
            f"Laney U' \u2014 Process in control (\u016b = {u_bar:.4f}, \u03c3z = {sigma_z:.3f})",
            f"All {k} samples within overdispersion-adjusted limits. {disp} detected (\u03c3z = {sigma_z:.3f}).",
            next_steps="Stable process. Laney adjustment accounts for extra-Poisson variation.",
            chart_guidance="Adjusted limits are wider than standard U chart when \u03c3z > 1.",
        )
    else:
        result["narrative"] = _narrative(
            f"Laney U' \u2014 {_lu_n_ooc} out-of-control point{'s' if _lu_n_ooc > 1 else ''}",
            f"{_lu_n_ooc} of {k} samples exceed overdispersion-adjusted limits (\u03c3z = {sigma_z:.3f}). These are genuine signals even after accounting for {disp.lower()}.",
            next_steps="Investigate OOC subgroups. Laney-adjusted signals are more trustworthy than standard U chart flags.",
            chart_guidance="Points outside the wider Laney limits are strong signals of genuine process change.",
        )

    return result


def run_moving_average(df, config):
    """Moving Average (MA) Chart."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if not measurement:
        measurement = df.select_dtypes(include=[np.number]).columns[0]
    span = int(config.get("span", 5))

    data = df[measurement].dropna().values
    n = len(data)

    x_bar = np.mean(data)
    sigma = np.std(data, ddof=1)

    # Moving averages
    ma = []
    for i in range(n):
        start = max(0, i - span + 1)
        window = data[start : i + 1]
        ma.append(np.mean(window))
    ma = np.array(ma)

    # Control limits for moving average (tighten as window fills)
    ucl_arr = []
    lcl_arr = []
    for i in range(n):
        w = min(i + 1, span)
        ucl_arr.append(x_bar + 3 * sigma / np.sqrt(w))
        lcl_arr.append(x_bar - 3 * sigma / np.sqrt(w))

    # OOC detection
    ma_ooc = [i for i in range(n) if ma[i] > ucl_arr[i] or ma[i] < lcl_arr[i]]

    ma_chart_data = [
        {
            "type": "scatter",
            "y": data.tolist(),
            "mode": "markers",
            "name": "Individual",
            "marker": {"size": 4, "color": "rgba(74,159,110,0.3)"},
        },
        {
            "type": "scatter",
            "y": ma.tolist(),
            "mode": "lines+markers",
            "name": f"MA({span})",
            "marker": {"size": 5, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e", "width": 2},
        },
        {
            "type": "scatter",
            "y": [x_bar] * n,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": ucl_arr,
            "mode": "lines",
            "name": "UCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": lcl_arr,
            "mode": "lines",
            "name": "LCL",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(ma_chart_data, ma.tolist(), ma_ooc)
    result["plots"].append(
        {
            "title": f"Moving Average Chart (span={span})",
            "data": ma_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": measurement},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # Steady-state limits
    ucl_ss = x_bar + 3 * sigma / np.sqrt(span)
    lcl_ss = x_bar - 3 * sigma / np.sqrt(span)

    result["summary"] = (
        f"Moving Average Chart\n\nSpan (window size): {span}\nCenter Line: {x_bar:.4f}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nSamples: {n}\nOut-of-control points: {len(ma_ooc)}\n\nThe MA chart smooths short-term noise. With span={span}, it is effective at detecting sustained shifts of {3 / np.sqrt(span):.2f}\u03c3 or larger."
    )

    _ma_n_ooc = len(ma_ooc)
    result["guide_observation"] = f"MA chart (span={span}): {_ma_n_ooc} out-of-control points." + (
        " Process appears stable." if _ma_n_ooc == 0 else " Investigation recommended."
    )
    if _ma_n_ooc == 0:
        result["narrative"] = _narrative(
            "Process is in statistical control",
            f"No out-of-control points on the moving average chart (span={span}). Process mean is stable.",
            next_steps="Process is stable. The MA chart smooths noise to reveal underlying shifts.",
            chart_guidance="Faded dots are individual values; the solid line is the moving average. Limits tighten as the window fills to span={span}.",
        )
    else:
        result["narrative"] = _narrative(
            f"MA chart \u2014 {_ma_n_ooc} out-of-control point{'s' if _ma_n_ooc > 1 else ''} detected",
            f"The smoothed moving average (span={span}) exceeds control limits at {_ma_n_ooc} point{'s' if _ma_n_ooc > 1 else ''}, indicating a sustained shift.",
            next_steps="Identify when the shift began and correlate with process changes.",
            chart_guidance="Faded dots are individual values; the solid line is the moving average. OOC points on the smoothed line indicate sustained (not transient) shifts.",
        )

    return result


def run_zone_chart(df, config):
    """Zone Chart -- assigns zone scores based on Western Electric zones."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if not measurement:
        measurement = df.select_dtypes(include=[np.number]).columns[0]

    data = df[measurement].dropna().values
    n = len(data)

    x_bar = np.mean(data)
    mr = np.abs(np.diff(data))
    sigma = np.mean(mr) / 1.128 if len(mr) > 0 else np.std(data, ddof=1)

    # Zone boundaries
    zone_1s = sigma
    zone_2s = 2 * sigma
    zone_3s = 3 * sigma

    # Zone scoring
    scores = []
    cum_scores = []
    cum_score = 0
    signals = []
    side = 0  # +1 above CL, -1 below CL

    for i in range(n):
        z = (data[i] - x_bar) / sigma if sigma > 0 else 0
        current_side = 1 if z >= 0 else -1

        # Zone score: A=8, B=4, C=2, center=0
        abs_z = abs(z)
        if abs_z >= 3:
            score = 8  # Beyond Zone A — instant signal
        elif abs_z >= 2:
            score = 4  # Zone A
        elif abs_z >= 1:
            score = 2  # Zone B
        else:
            score = 0  # Zone C (reset)
            cum_score = 0

        # Reset on side change
        if i > 0 and current_side != side:
            cum_score = 0
        side = current_side

        cum_score += score
        scores.append(score)
        cum_scores.append(cum_score)

        if cum_score >= 8:
            signals.append(i)
            cum_score = 0  # Reset after signal

    # Plot with zone bands
    zone_shapes = [
        # Zone C (green) - within 1 sigma
        {
            "type": "rect",
            "y0": x_bar - zone_1s,
            "y1": x_bar + zone_1s,
            "x0": 0,
            "x1": n - 1,
            "fillcolor": "rgba(74,159,110,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
        # Zone B upper (yellow) - 1 to 2 sigma
        {
            "type": "rect",
            "y0": x_bar + zone_1s,
            "y1": x_bar + zone_2s,
            "x0": 0,
            "x1": n - 1,
            "fillcolor": "rgba(243,156,18,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
        # Zone B lower (yellow)
        {
            "type": "rect",
            "y0": x_bar - zone_2s,
            "y1": x_bar - zone_1s,
            "x0": 0,
            "x1": n - 1,
            "fillcolor": "rgba(243,156,18,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
        # Zone A upper (red) - 2 to 3 sigma
        {
            "type": "rect",
            "y0": x_bar + zone_2s,
            "y1": x_bar + zone_3s,
            "x0": 0,
            "x1": n - 1,
            "fillcolor": "rgba(231,76,60,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
        # Zone A lower (red)
        {
            "type": "rect",
            "y0": x_bar - zone_3s,
            "y1": x_bar - zone_2s,
            "x0": 0,
            "x1": n - 1,
            "fillcolor": "rgba(231,76,60,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    # Color data points by zone
    colors = []
    for i in range(n):
        abs_z = abs((data[i] - x_bar) / sigma) if sigma > 0 else 0
        if abs_z >= 3:
            colors.append("#e74c3c")
        elif abs_z >= 2:
            colors.append("#f39c12")
        elif abs_z >= 1:
            colors.append("#fdcb6e")
        else:
            colors.append("#4a9f6e")

    zone_chart_data = [
        {
            "type": "scatter",
            "y": data.tolist(),
            "mode": "lines+markers",
            "name": measurement,
            "marker": {"size": 7, "color": colors},
            "line": {"color": "rgba(200,200,200,0.3)", "width": 1},
            "customdata": [[i, ""] for i in range(n)],
            "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<extra></extra>",
        },
        {
            "type": "scatter",
            "y": [x_bar] * n,
            "mode": "lines",
            "name": "CL",
            "line": {"color": "#00b894", "width": 1.5},
        },
        {
            "type": "scatter",
            "y": [x_bar + zone_3s] * n,
            "mode": "lines",
            "name": "UCL (3\u03c3)",
            "line": {"color": "#d63031", "dash": "dash"},
        },
        {
            "type": "scatter",
            "y": [x_bar - zone_3s] * n,
            "mode": "lines",
            "name": "LCL (3\u03c3)",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]

    # Signal markers
    if signals:
        zone_chart_data.append(
            {
                "type": "scatter",
                "x": signals,
                "y": [data[i] for i in signals],
                "mode": "markers",
                "name": "Signal (score\u22658)",
                "marker": {
                    "size": 12,
                    "color": "#e74c3c",
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 1.5},
                },
                "customdata": [[i, "Zone signal: cumulative score \u22658"] for i in signals],
                "hovertemplate": "Obs #%{customdata[0]}<br>Value: %{y:.4f}<br>%{customdata[1]}<extra>Signal</extra>",
            }
        )

    result["plots"].append(
        {
            "title": "Zone Chart",
            "data": zone_chart_data,
            "layout": {
                "height": 390,
                "showlegend": True,
                "yaxis": {"title": measurement},
                "xaxis": {"rangeslider": {"visible": True, "thickness": 0.12}},
                "shapes": zone_shapes,
                "annotations": [
                    {
                        "x": n - 1,
                        "y": x_bar + zone_1s,
                        "text": "C",
                        "showarrow": False,
                        "xanchor": "right",
                        "font": {"size": 10, "color": "#4a9f6e"},
                    },
                    {
                        "x": n - 1,
                        "y": x_bar + zone_2s,
                        "text": "B",
                        "showarrow": False,
                        "xanchor": "right",
                        "font": {"size": 10, "color": "#f39c12"},
                    },
                    {
                        "x": n - 1,
                        "y": x_bar + zone_3s,
                        "text": "A",
                        "showarrow": False,
                        "xanchor": "right",
                        "font": {"size": 10, "color": "#e74c3c"},
                    },
                ],
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # Cumulative score chart
    result["plots"].append(
        {
            "title": "Cumulative Zone Score",
            "data": [
                {
                    "type": "scatter",
                    "y": cum_scores,
                    "mode": "lines+markers",
                    "name": "Cum. Score",
                    "marker": {"size": 4, "color": "#4a9f6e"},
                    "line": {"color": "#4a9f6e"},
                },
                {
                    "type": "scatter",
                    "y": [8] * n,
                    "mode": "lines",
                    "name": "Signal Threshold",
                    "line": {"color": "#e74c3c", "dash": "dash"},
                },
            ],
            "layout": {
                "height": 240,
                "showlegend": True,
                "yaxis": {"title": "Score"},
                "xaxis": {
                    "title": "Sample",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
        }
    )

    result["summary"] = (
        f"Zone Chart Analysis\n\nCenter Line: {x_bar:.4f}\nEstimated \u03c3: {sigma:.4f}\n\nZone Boundaries:\n  C (green): \u00b11\u03c3 = [{x_bar - zone_1s:.4f}, {x_bar + zone_1s:.4f}]\n  B (yellow): \u00b12\u03c3 = [{x_bar - zone_2s:.4f}, {x_bar + zone_2s:.4f}]\n  A (red): \u00b13\u03c3 = [{x_bar - zone_3s:.4f}, {x_bar + zone_3s:.4f}]\n\nScoring: A=8, B=4, C=2. Signal when cumulative \u2265 8.\nSignals detected: {len(signals)}"
    )

    return result
