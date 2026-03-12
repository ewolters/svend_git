"""D-Chart — factor divergence over rolling time windows.

CR: 3c0d0e53
"""

import logging

import numpy as np
import pandas as pd

from ..common import COLOR_BAD, COLOR_GOOD, COLOR_NEUTRAL, COLOR_REFERENCE, SVEND_COLORS, _rgba
from .helpers import _build_grid, _d_chart_body, _d_chart_nextsteps, _d_narrative, _jsd, _kde_density, _noise_floor

logger = logging.getLogger(__name__)


def run_d_chart(df, config):
    """D-Chart — track per-factor JSD vs pooled distribution over rolling time windows.

    Config:
        variable:     numeric column to analyze
        factor:       categorical column (e.g. Shift, Machine, Operator)
        time_col:     datetime or ordinal column for windowing
        window_size:  number of observations per window (default 50)
        step_size:    window step (default = window_size // 2)
    """

    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = config.get("variable")
    factor = config.get("factor")
    time_col = config.get("time_col")
    window_size = int(config.get("window_size", 50))
    step_size = int(config.get("step_size", 0)) or window_size // 2

    if not variable or not factor:
        result["summary"] = "<<COLOR:danger>>Error: variable and factor are required.<</COLOR>>"
        return result

    # Prepare data
    df = df.dropna(subset=[variable, factor])
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        time_col = None

    values = df[variable].astype(float).values
    factors = df[factor].astype(str).values
    unique_factors = sorted(set(factors))
    n = len(df)

    if n < window_size:
        result["summary"] = f"<<COLOR:warning>>Need at least {window_size} observations (have {n}).<</COLOR>>"
        return result

    if len(unique_factors) < 2:
        result["summary"] = "<<COLOR:warning>>Need at least 2 factor levels for divergence analysis.<</COLOR>>"
        return result

    # Build KDE grid from full data
    grid = _build_grid(values)

    # Per-factor baseline density from the first third of data.
    # D-Chart answers "which factor CHANGED?" not "which factor is different?"
    # Comparing each factor to its own baseline avoids the pooled-contamination
    # problem where a stable factor appears divergent because others shifted the pool.
    n_baseline_rows = n // 3
    baseline_df = df.iloc[:n_baseline_rows]
    factor_baselines = {}
    for fval in unique_factors:
        mask = baseline_df[factor].astype(str) == fval
        bdata = baseline_df.loc[mask, variable].astype(float).values
        if len(bdata) >= 5:
            factor_baselines[fval] = _kde_density(bdata, grid)
        else:
            factor_baselines[fval] = None

    # Compute noise floor (expected JSD from random sampling within a factor)
    # Use the largest factor's baseline data for the noise estimate
    biggest_factor = max(unique_factors, key=lambda f: (factors == f).sum())
    biggest_data = values[factors == biggest_factor]
    noise = _noise_floor(biggest_data, len(biggest_data) // 2, grid, B=200)

    # Rolling windows
    windows = []
    starts = list(range(0, n - window_size + 1, step_size))
    if not starts:
        starts = [0]

    for start in starts:
        end = start + window_size
        w_df = df.iloc[start:end]
        w_values = w_df[variable].astype(float).values

        if time_col:
            tc = w_df[time_col]
            if pd.api.types.is_datetime64_any_dtype(tc):
                midpoint = tc.iloc[len(tc) // 2]
                label = str(midpoint.date()) if hasattr(midpoint, "date") else str(midpoint)
            elif pd.api.types.is_numeric_dtype(tc):
                midpoint = tc.iloc[len(tc) // 2]
                label = str(midpoint)
            else:
                try:
                    ts = pd.to_datetime(tc)
                    midpoint = ts.iloc[len(ts) // 2]
                    label = str(midpoint.date()) if hasattr(midpoint, "date") else str(midpoint)
                except Exception:
                    label = f"{start}-{end}"
                    midpoint = start + window_size // 2
        else:
            label = f"{start}-{end}"
            midpoint = start + window_size // 2

        # Per-factor JSD: compare factor's current window to its OWN baseline.
        # This measures temporal drift, not cross-sectional difference.
        factor_jsds = {}
        for fval in unique_factors:
            mask = w_df[factor].astype(str) == fval
            fdata = w_values[mask.values]
            if len(fdata) >= 5 and factor_baselines[fval] is not None:
                f_density = _kde_density(fdata, grid)
                factor_jsds[fval] = _jsd(f_density, factor_baselines[fval], grid)
            else:
                factor_jsds[fval] = 0.0

        windows.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "midpoint": midpoint,
                "factor_jsds": factor_jsds,
                "max_jsd": max(factor_jsds.values()) if factor_jsds else 0.0,
            }
        )

    # Cumulative information score with exponential recency weighting
    lam = 0.05  # decay rate
    T = len(windows)
    noise = float(noise)  # ensure plain float for JSON
    info_scores = {}
    for fval in unique_factors:
        weighted_sum = 0.0
        weight_total = 0.0
        for t, w in enumerate(windows):
            weight = np.exp(-lam * (T - 1 - t))
            excess = max(0, float(w["factor_jsds"].get(fval, 0)) - noise)
            weighted_sum += weight * excess
            weight_total += weight
        info_scores[fval] = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Build Plotly chart — line chart, one trace per factor + noise floor
    traces = []
    x_labels = [w["label"] for w in windows]

    for i, fval in enumerate(unique_factors):
        y_vals = [w["factor_jsds"].get(fval, 0) for w in windows]
        traces.append(
            {
                "type": "scatter",
                "x": x_labels,
                "y": y_vals,
                "mode": "lines+markers",
                "name": str(fval),
                "line": {"color": SVEND_COLORS[i % len(SVEND_COLORS)], "width": 2},
                "marker": {"size": 5},
            }
        )

    # Noise floor line
    traces.append(
        {
            "type": "scatter",
            "x": x_labels,
            "y": [noise] * len(x_labels),
            "mode": "lines",
            "name": f"Noise Floor ({noise:.4f})",
            "line": {"color": COLOR_NEUTRAL, "dash": "dash", "width": 1.5},
        }
    )

    result["plots"].append(
        {
            "title": f"D-Chart: {factor} Divergence Over Time ({variable})",
            "data": traces,
            "layout": {
                "height": 400,
                "xaxis": {"title": time_col or "Window"},
                "yaxis": {"title": "JSD (bits)", "rangemode": "tozero"},
                "legend": {"orientation": "h", "y": -0.2},
            },
        }
    )

    # Information score bar chart
    sorted_factors = sorted(info_scores.items(), key=lambda x: x[1], reverse=True)
    bar_colors = [SVEND_COLORS[unique_factors.index(f) % len(SVEND_COLORS)] for f, _ in sorted_factors]
    result["plots"].append(
        {
            "title": f"Cumulative Information Score by {factor}",
            "data": [
                {
                    "type": "bar",
                    "x": [f for f, _ in sorted_factors],
                    "y": [round(s, 4) for _, s in sorted_factors],
                    "marker": {"color": bar_colors},
                }
            ],
            "layout": {
                "height": 400,
                "xaxis": {"title": factor},
                "yaxis": {"title": "Weighted Excess JSD", "rangemode": "tozero"},
            },
        }
    )

    # ── Heatmap: factor × time window ──
    z_data = []
    for fval in unique_factors:
        row = [round(float(w["factor_jsds"].get(fval, 0)), 4) for w in windows]
        z_data.append(row)
    result["plots"].append(
        {
            "title": f"Divergence Heatmap: {factor} × Time",
            "data": [
                {
                    "type": "heatmap",
                    "z": z_data,
                    "x": x_labels,
                    "y": list(unique_factors),
                    "colorscale": [[0, _rgba(COLOR_GOOD, 0.1)], [0.5, COLOR_REFERENCE], [1, COLOR_BAD]],
                    "colorbar": {"title": "JSD", "len": 0.8},
                    "hovertemplate": "%{y} @ %{x}<br>JSD = %{z:.4f}<extra></extra>",
                }
            ],
            "layout": {
                "height": 400,
                "xaxis": {"title": time_col or "Window"},
                "yaxis": {"title": factor},
            },
        }
    )

    # ── KDE overlay: most divergent factor's baseline vs peak window ──
    max_factor_name = sorted_factors[0][0] if sorted_factors else unique_factors[0]
    # Find the window with peak divergence for this factor
    peak_w = max(windows, key=lambda w: w["factor_jsds"].get(max_factor_name, 0))
    peak_df = df.iloc[peak_w["start"] : peak_w["end"]]
    peak_vals = peak_df[variable].astype(float).values
    peak_mask = peak_df[factor].astype(str) == max_factor_name
    peak_factor_vals = peak_vals[peak_mask.values]

    kde_traces = []
    # Baseline density for this factor
    if factor_baselines[max_factor_name] is not None:
        kde_traces.append(
            {
                "type": "scatter",
                "x": grid.tolist(),
                "y": factor_baselines[max_factor_name].tolist(),
                "mode": "lines",
                "fill": "tozeroy",
                "fillcolor": _rgba(COLOR_GOOD, 0.15),
                "line": {"color": COLOR_GOOD, "width": 2},
                "name": f"{max_factor_name} (baseline)",
            }
        )
    # Current (peak window) density for this factor
    if len(peak_factor_vals) >= 5:
        d1 = _kde_density(peak_factor_vals, grid)
        kde_traces.append(
            {
                "type": "scatter",
                "x": grid.tolist(),
                "y": d1.tolist(),
                "mode": "lines",
                "fill": "tozeroy",
                "fillcolor": _rgba(COLOR_BAD, 0.15),
                "line": {"color": COLOR_BAD, "width": 2},
                "name": f"{max_factor_name} (peak window {peak_w['label']})",
            }
        )
    result["plots"].append(
        {
            "title": f"Distribution Shift: {max_factor_name} — Baseline vs Peak",
            "data": kde_traces,
            "layout": {
                "height": 400,
                "xaxis": {"title": variable},
                "yaxis": {"title": "Density"},
                "legend": {"orientation": "h", "y": -0.2},
            },
        }
    )

    # ── Phase detection: when did divergence start? ──
    # Compare each factor's JSD to its own early baseline (first third of windows).
    # Flag onset when 3 consecutive windows exceed baseline_p75 + 3 × baseline_iqr.
    onset_info = {}
    n_baseline = max(len(windows) // 3, 3)
    for fval, score in sorted_factors:
        if score < noise * 3:
            continue  # skip factors with weak cumulative signal
        jsd_series = [float(w["factor_jsds"].get(fval, 0)) for w in windows]
        baseline = sorted(jsd_series[:n_baseline])
        if len(baseline) < 3:
            continue
        bp75 = np.percentile(baseline, 75)
        biqr = np.percentile(baseline, 75) - np.percentile(baseline, 25)
        threshold = bp75 + 3 * max(biqr, noise)
        onset_idx = None
        for i in range(n_baseline, len(jsd_series) - 2):
            if all(jsd_series[i + k] > threshold for k in range(3)):
                onset_idx = i
                break
        if onset_idx is not None:
            onset_info[fval] = {"window_idx": onset_idx, "label": windows[onset_idx]["label"]}

    # Summary
    max_factor = sorted_factors[0][0] if sorted_factors else "N/A"
    max_score = sorted_factors[0][1] if sorted_factors else 0
    any_above_noise = any(w["max_jsd"] > noise for w in windows)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>D-CHART: FACTOR DIVERGENCE ANALYSIS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {variable}\n"
    summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(unique_factors)} levels)\n"
    summary += f"<<COLOR:highlight>>Windows:<</COLOR>> {len(windows)} (size={window_size}, step={step_size})\n"
    summary += f"<<COLOR:highlight>>Noise Floor:<</COLOR>> {noise:.4f} bits (95th percentile of null)\n\n"

    summary += "<<COLOR:accent>>── Information Scores (recency-weighted excess JSD) ──<</COLOR>>\n"
    for fval, score in sorted_factors:
        flag = " <<COLOR:warning>>▲<</COLOR>>" if score > noise * 2 else ""
        summary += f"  {fval}: {score:.4f}{flag}\n"

    if onset_info:
        summary += "\n<<COLOR:accent>>── Onset Detection ──<</COLOR>>\n"
        for fval, info in onset_info.items():
            summary += f"  {fval}: divergence sustained from window {info['label']}\n"

    summary += "\n<<COLOR:accent>>── Assessment ──<</COLOR>>\n"
    if any_above_noise and max_score > noise * 2:
        summary += f"<<COLOR:danger>>Factor '{max_factor}' shows systematic divergence from the pooled distribution.<</COLOR>>\n"
        summary += "<<COLOR:warning>>This factor is contributing non-random variation to the process.<</COLOR>>\n"
    elif any_above_noise:
        summary += "<<COLOR:warning>>Some windows show divergence above noise floor, but the cumulative pattern is moderate.<</COLOR>>\n"
    else:
        summary += (
            "<<COLOR:good>>All factor levels behave consistently — no systematic divergence detected.<</COLOR>>\n"
        )

    result["summary"] = summary

    onset_str = ""
    if onset_info:
        onset_str = " Onset: " + ", ".join(f"{f} from {i['label']}" for f, i in onset_info.items()) + "."

    result["guide_observation"] = (
        f"D-Chart: {factor} on {variable}, {len(windows)} windows. "
        f"Top divergent factor: {max_factor} (info score={max_score:.4f}). "
        f"Noise floor: {noise:.4f}. "
        + (
            "Systematic divergence detected."
            if any_above_noise and max_score > noise * 2
            else "No systematic divergence."
        )
        + onset_str
    )

    result["statistics"] = {
        "noise_floor": noise,
        "n_windows": len(windows),
        "n_factors": len(unique_factors),
        "info_scores": {f: round(s, 6) for f, s in sorted_factors},
        "max_factor": max_factor,
        "max_info_score": round(max_score, 6),
        "onset": {f: i["label"] for f, i in onset_info.items()},
    }

    result["narrative"] = _d_narrative(
        f"D-Chart: {factor} divergence on {variable}",
        _d_chart_body(sorted_factors, noise, any_above_noise, variable, factor),
        _d_chart_nextsteps(sorted_factors, noise, any_above_noise, factor),
        "The top chart tracks JSD per factor over time windows. Points above the dashed noise floor indicate non-random divergence. The bar chart ranks factors by cumulative recency-weighted excess divergence.",
    )

    result["education"] = {
        "title": "Understanding the D-Chart",
        "content": (
            "<dl>"
            "<dt>What is a D-Chart?</dt>"
            "<dd>A Divergence Chart monitors how much each factor level's distribution "
            "differs from the overall process distribution over time. Unlike traditional "
            "control charts that track means or ranges, it tracks <em>distributional shape</em> "
            "changes — catching shifts in spread, skew, or tails that mean/range charts miss.</dd>"
            "<dt>What is JSD (Jensen-Shannon Divergence)?</dt>"
            "<dd>A symmetric, bounded measure of how different two probability distributions "
            "are. JSD = 0 means identical; JSD = 1 (in bits) means completely different. "
            "It is the information-theoretic gold standard for comparing distributions.</dd>"
            "<dt>What is the Noise Floor?</dt>"
            "<dd>The expected JSD from random sampling alone (no real factor effect). Computed "
            "via bootstrap permutation. Points above the noise floor indicate real, "
            "non-random divergence — not just sampling noise.</dd>"
            "<dt>What is the Information Score?</dt>"
            "<dd>A cumulative, recency-weighted sum of excess JSD (above noise floor) across "
            "time windows. Higher scores mean the factor consistently produces a different "
            "distribution. Recent windows are weighted more heavily.</dd>"
            "<dt>How to interpret</dt>"
            "<dd><strong>All below noise floor</strong>: Factor has no effect on the distribution — "
            "the process is factor-invariant. <strong>Sporadic exceedances</strong>: Possible "
            "intermittent factor effect — investigate timing. <strong>Sustained above noise</strong>: "
            "Systematic divergence — this factor genuinely changes the process output.</dd>"
            "</dl>"
        ),
    }

    return result
