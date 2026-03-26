"""D-Sig — distributional signal detection.

CR: 3c0d0e53
"""

import logging

import numpy as np
import pandas as pd

from ..common import (
    COLOR_REFERENCE,
    SVEND_COLORS,
    _rgba,
)
from .helpers import _d_narrative, _jsd, _kde_density

logger = logging.getLogger(__name__)


def run_d_sig(df, config):
    """Process signature comparison via functional JSD.

    Compares time-series profiles across groups by computing windowed JSD
    at each time point to identify where and how process signatures diverge.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = (
        config.get("variable") or config.get("measurement") or config.get("profile")
    )
    time_col = config.get("time_col") or config.get("time")
    group_col = config.get("group") or config.get("factor")

    if not variable or variable not in df.columns:
        result["summary"] = (
            "<<COLOR:danger>>Please select a valid profile variable.<</COLOR>>"
        )
        return result
    if not time_col or time_col not in df.columns:
        result["summary"] = (
            "<<COLOR:danger>>Please select a valid time/sequence column.<</COLOR>>"
        )
        return result
    if not group_col or group_col not in df.columns:
        result["summary"] = (
            "<<COLOR:danger>>Please select a valid group column.<</COLOR>>"
        )
        return result

    work = df[[variable, time_col, group_col]].dropna()
    work[variable] = work[variable].astype(float)
    work[group_col] = work[group_col].astype(str)
    try:
        work[time_col] = work[time_col].astype(float)
    except (ValueError, TypeError):
        # If time_col is datetime-like, convert to numeric
        work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])

    groups = work[group_col].unique()
    if len(groups) < 2:
        result["summary"] = (
            "<<COLOR:danger>>Need at least 2 groups for signature comparison.<</COLOR>>"
        )
        return result

    # Build per-group profiles sorted by time
    group_profiles = {}
    for g in groups:
        gdf = work[work[group_col] == g].sort_values(time_col)
        if len(gdf) < 5:
            continue
        group_profiles[g] = {
            "time": gdf[time_col].values.astype(float),
            "values": gdf[variable].values.astype(float),
        }

    if len(group_profiles) < 2:
        result["summary"] = (
            "<<COLOR:danger>>Not enough groups with ≥5 observations.<</COLOR>>"
        )
        return result

    group_names = list(group_profiles.keys())

    # Choose reference group (largest)
    ref_group = max(group_names, key=lambda g: len(group_profiles[g]["values"]))

    # Common time grid
    all_times = np.concatenate([gp["time"] for gp in group_profiles.values()])
    t_min, t_max = float(all_times.min()), float(all_times.max())
    n_grid_pts = min(100, max(20, len(all_times) // len(group_profiles)))
    common_time = np.linspace(t_min, t_max, n_grid_pts)

    # Interpolate each group to common grid
    interp_profiles = {}
    for g, gp in group_profiles.items():
        interp_profiles[g] = np.interp(common_time, gp["time"], gp["values"])

    ref_profile = interp_profiles[ref_group]

    # Compute pointwise divergence via windowed comparison
    # For each time point, compare the value distributions in a window around it
    window_half = max(3, n_grid_pts // 10)
    test_groups = [g for g in group_names if g != ref_group]

    group_divergences = {}
    pointwise_jsd = {}

    for g in test_groups:
        g_profile = interp_profiles[g]
        pw_jsd = np.zeros(n_grid_pts)

        for t_idx in range(n_grid_pts):
            lo = max(0, t_idx - window_half)
            hi = min(n_grid_pts, t_idx + window_half + 1)

            ref_window = ref_profile[lo:hi]
            g_window = g_profile[lo:hi]

            if len(ref_window) >= 3 and len(g_window) >= 3:
                # Use local value distribution comparison
                all_vals = np.concatenate([ref_window, g_window])
                local_grid = np.linspace(
                    all_vals.min() - 0.1 * (np.ptp(all_vals) or 1),
                    all_vals.max() + 0.1 * (np.ptp(all_vals) or 1),
                    100,
                )
                ref_dens = _kde_density(ref_window, local_grid)
                g_dens = _kde_density(g_window, local_grid)
                pw_jsd[t_idx] = _jsd(ref_dens, g_dens, local_grid)
            else:
                pw_jsd[t_idx] = 0.0

        pointwise_jsd[g] = pw_jsd
        group_divergences[g] = {
            "mean_jsd": float(np.mean(pw_jsd)),
            "max_jsd": float(np.max(pw_jsd)),
            "peak_time": float(common_time[np.argmax(pw_jsd)]),
            "rmse": float(np.sqrt(np.mean((g_profile - ref_profile) ** 2))),
        }

    # Sort by mean JSD
    sorted_groups = sorted(
        test_groups, key=lambda g: group_divergences[g]["mean_jsd"], reverse=True
    )

    # --- Plot 1: Profile overlay ---
    profile_traces = []
    profile_traces.append(
        {
            "type": "scatter",
            "x": common_time.tolist(),
            "y": ref_profile.tolist(),
            "mode": "lines",
            "name": f"{ref_group} (ref)",
            "line": {"color": COLOR_REFERENCE, "width": 3},
        }
    )
    for i, g in enumerate(sorted_groups):
        profile_traces.append(
            {
                "type": "scatter",
                "x": common_time.tolist(),
                "y": interp_profiles[g].tolist(),
                "mode": "lines",
                "name": g,
                "line": {"color": SVEND_COLORS[i % len(SVEND_COLORS)], "width": 1.5},
            }
        )
    result["plots"].append(
        {
            "title": "Process Signatures",
            "data": profile_traces,
            "layout": {
                "height": 340,
                "xaxis": {"title": time_col},
                "yaxis": {"title": variable},
                "showlegend": True,
            },
        }
    )

    # --- Plot 2: Pointwise JSD timeline ---
    jsd_traces = []
    for i, g in enumerate(sorted_groups):
        jsd_traces.append(
            {
                "type": "scatter",
                "x": common_time.tolist(),
                "y": pointwise_jsd[g].tolist(),
                "mode": "lines",
                "name": g,
                "fill": "tozeroy",
                "line": {"color": SVEND_COLORS[i % len(SVEND_COLORS)], "width": 1.5},
                "fillcolor": _rgba(SVEND_COLORS[i % len(SVEND_COLORS)], 0.1),
            }
        )
    result["plots"].append(
        {
            "title": "Pointwise JSD vs Reference",
            "data": jsd_traces,
            "layout": {
                "height": 300,
                "xaxis": {"title": time_col},
                "yaxis": {"title": "JSD (bits)"},
                "showlegend": True,
            },
        }
    )

    # --- Plot 3: Divergence summary bar chart ---
    bar_names = [g for g in sorted_groups]
    bar_means = [group_divergences[g]["mean_jsd"] for g in sorted_groups]
    bar_maxes = [group_divergences[g]["max_jsd"] for g in sorted_groups]
    bar_colors_mean = [
        SVEND_COLORS[i % len(SVEND_COLORS)] for i in range(len(sorted_groups))
    ]

    result["plots"].append(
        {
            "title": "Signature Divergence Summary",
            "data": [
                {
                    "type": "bar",
                    "x": bar_names,
                    "y": bar_means,
                    "name": "Mean JSD",
                    "marker": {"color": bar_colors_mean},
                    "text": [f"{v:.4f}" for v in bar_means],
                    "textposition": "outside",
                },
                {
                    "type": "bar",
                    "x": bar_names,
                    "y": bar_maxes,
                    "name": "Peak JSD",
                    "marker": {"color": [_rgba(c, 0.4) for c in bar_colors_mean]},
                    "text": [f"{v:.4f}" for v in bar_maxes],
                    "textposition": "outside",
                },
            ],
            "layout": {
                "height": 300,
                "barmode": "group",
                "yaxis": {"title": "JSD (bits)"},
                "showlegend": True,
            },
        }
    )

    # --- Summary ---
    summary = "<<COLOR:title>>D-SIG — PROCESS SIGNATURE COMPARISON<</COLOR>>\n\n"
    summary += f"<<COLOR:header>>Reference:<</COLOR>> '{ref_group}' (n={len(group_profiles[ref_group]['values'])})\n"
    summary += f"<<COLOR:header>>Variable:<</COLOR>> {variable} over {time_col}\n"
    summary += f"<<COLOR:header>>Window size:<</COLOR>> ±{window_half} points\n\n"

    summary += "<<COLOR:header>>Divergence Rankings:<</COLOR>>\n"
    summary += f"  {'Group':<15} {'Mean JSD':>10} {'Peak JSD':>10} {'Peak At':>10} {'RMSE':>10}\n"
    summary += f"  {'-' * 58}\n"
    for g in sorted_groups:
        d = group_divergences[g]
        summary += f"  {g:<15} {d['mean_jsd']:>10.4f} {d['max_jsd']:>10.4f} {d['peak_time']:>10.1f} {d['rmse']:>10.3f}\n"

    most_div = sorted_groups[0] if sorted_groups else None
    if most_div:
        d = group_divergences[most_div]
        summary += f"\n<<COLOR:highlight>>Most divergent:<</COLOR>> '{most_div}' (mean JSD = {d['mean_jsd']:.4f})\n"
        summary += f"  Peak divergence at {time_col} = {d['peak_time']:.1f} (JSD = {d['max_jsd']:.4f})\n"

    result["summary"] = summary
    result["guide_observation"] = (
        f"D-Sig: Most divergent group '{most_div}' (mean JSD={group_divergences[most_div]['mean_jsd']:.4f}) "
        f"vs ref '{ref_group}'"
        if most_div
        else "D-Sig: No divergent groups found"
    )
    result["statistics"] = {
        "reference": ref_group,
        "n_groups": len(group_profiles),
        "group_divergences": {g: group_divergences[g] for g in sorted_groups},
    }

    # --- narrative ---
    if most_div:
        d = group_divergences[most_div]
        verdict = f"Most divergent: {most_div} (mean JSD = {d['mean_jsd']:.4f})"
        body = (
            f"Group '<strong>{most_div}</strong>' shows the highest signature divergence "
            f"from reference '<strong>{ref_group}</strong>', with mean JSD = {d['mean_jsd']:.4f} "
            f"and peak divergence at {time_col} = {d['peak_time']:.1f} "
            f"(JSD = {d['max_jsd']:.4f})."
        )
        if d["mean_jsd"] > 0.01:
            body += " This is a meaningful divergence — the process profile differs substantially."
            nxt = f"Investigate what differs about '{most_div}' at the peak divergence time point."
        else:
            body += " Divergence is small — profiles are broadly similar."
            nxt = "Profiles are consistent — continue monitoring."
    else:
        verdict = "No divergent groups found"
        body = "All groups show similar process signatures compared to the reference."
        nxt = "Profiles are consistent — no action needed."
    result["narrative"] = _d_narrative(
        f"D-Sig: {verdict}",
        body,
        nxt,
        "The profile overlay shows each group's time-series signature. "
        "The JSD-over-time chart shows where signatures diverge — peaks indicate "
        "time points where a group's behavior differs most from the reference.",
    )

    result["education"] = {
        "title": "Understanding Process Signatures (D-Sig)",
        "content": (
            "<dl>"
            "<dt>What is a Process Signature?</dt>"
            "<dd>A time-ordered profile of a measurement variable — e.g., temperature over "
            "a batch cycle, force during a press stroke, or voltage across a test sequence. "
            "D-Sig compares these profiles across groups to find where and how they diverge.</dd>"
            "<dt>How does it work?</dt>"
            "<dd>At each time point, it computes the JSD between each group's windowed "
            "distribution and the reference group's. This produces a 'divergence-over-time' "
            "trace that shows <em>when</em> during the process the signatures differ.</dd>"
            "<dt>What is the Peak Divergence?</dt>"
            "<dd>The time point where a group's distribution differs most from the reference. "
            "This is where you should focus your investigation — it's the moment in the "
            "process where something changes.</dd>"
            "<dt>How to interpret</dt>"
            "<dd><strong>Flat, low JSD trace</strong>: The group's signature matches the reference "
            "throughout. <strong>Spike at specific time</strong>: A localized divergence — something "
            "happens at that point. <strong>Sustained elevation</strong>: The entire profile "
            "differs — a fundamentally different process mode.</dd>"
            "</dl>"
        ),
    }

    return result
