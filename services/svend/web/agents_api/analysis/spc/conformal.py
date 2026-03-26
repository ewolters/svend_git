"""SPC conformal prediction charts — Conformal Control, Conformal Monitor, Entropy SPC."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..common import _narrative


def run_conformal_control(df, config):
    """Conformal-Enhanced Control Chart -- distribution-free alternative to Shewhart."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement") or config.get("var") or config.get("column")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    n = len(data)
    alpha_conf = float(config.get("alpha", 0.05))  # False alarm rate
    cal_fraction = float(config.get("calibration_fraction", 0.5))
    spike_threshold = float(
        config.get("spike_threshold", 2.0)
    )  # Multiple of median width for spike detection
    chart_type = config.get(
        "chart_type", "individuals"
    )  # individuals, subgroup_mean, subgroup_range
    subgroup_size = int(config.get("subgroup_size", 5))

    if n < 20:
        result["summary"] = (
            "Need at least 20 observations for conformal control chart (calibration + monitoring)."
        )
        return result

    # Phase I / Phase II split
    n_cal = max(10, int(n * cal_fraction))
    cal_data = data[:n_cal]
    mon_data = data[n_cal:]
    n_mon = len(mon_data)

    if chart_type == "subgroup_mean" and n_cal >= subgroup_size * 2:
        # Subgroup means
        n_sg_cal = n_cal // subgroup_size
        cal_subgroups = [
            cal_data[i * subgroup_size : (i + 1) * subgroup_size]
            for i in range(n_sg_cal)
        ]
        cal_values = np.array([np.mean(sg) for sg in cal_subgroups])
        center = np.median(cal_values)

        n_sg_mon = n_mon // subgroup_size
        mon_values = np.array(
            [
                np.mean(mon_data[i * subgroup_size : (i + 1) * subgroup_size])
                for i in range(n_sg_mon)
            ]
        )
        x_labels = [f"SG {i + 1}" for i in range(n_sg_cal + n_sg_mon)]
        all_values = np.concatenate([cal_values, mon_values])
        chart_label = f"Subgroup Mean (n={subgroup_size})"
    elif chart_type == "subgroup_range" and n_cal >= subgroup_size * 2:
        # Subgroup ranges
        n_sg_cal = n_cal // subgroup_size
        cal_subgroups = [
            cal_data[i * subgroup_size : (i + 1) * subgroup_size]
            for i in range(n_sg_cal)
        ]
        cal_values = np.array([np.ptp(sg) for sg in cal_subgroups])
        center = np.median(cal_values)

        n_sg_mon = n_mon // subgroup_size
        mon_values = np.array(
            [
                np.ptp(mon_data[i * subgroup_size : (i + 1) * subgroup_size])
                for i in range(n_sg_mon)
            ]
        )
        all_values = np.concatenate([cal_values, mon_values])
        chart_label = f"Subgroup Range (n={subgroup_size})"
    else:
        # Individual observations
        cal_values = cal_data
        center = np.median(cal_values)
        mon_values = mon_data
        _x_labels = list(range(n))
        all_values = data
        chart_label = "Individual Values"

    # Nonconformity scores (calibration): |Xi - median| (robust to outliers)
    cal_scores = np.abs(cal_values - center)

    # Conformal threshold: ceil((1-alpha)(n_cal+1))-th smallest score
    sorted_scores = np.sort(cal_scores)
    q_index = int(np.ceil((1 - alpha_conf) * (len(cal_scores) + 1))) - 1
    q_index = min(q_index, len(sorted_scores) - 1)
    q = float(sorted_scores[q_index])

    # Prediction interval (distribution-free)
    pi_lower = center - q
    pi_upper = center + q

    # Phase II: compute scores and flag OOC
    mon_scores = np.abs(mon_values - center)
    ooc_mask = mon_scores > q
    ooc_indices = np.where(ooc_mask)[0]
    n_ooc = len(ooc_indices)

    # All scores for plotting
    all_scores = np.abs(all_values - center)

    # ── Adaptive Prediction Intervals (model-based variant) ──
    window = min(20, n_cal)
    adaptive_widths = np.zeros(n)
    adaptive_lower = np.zeros(n)
    adaptive_upper = np.zeros(n)

    for i in range(n):
        if i < window:
            local_std = (
                np.std(data[: max(i + 1, 2)], ddof=1)
                if i > 0
                else np.std(cal_data, ddof=1)
            )
            local_center = np.median(data[: max(i + 1, 2)])
        else:
            local_window = data[i - window : i]
            local_std = np.std(local_window, ddof=1)
            local_center = np.median(local_window)

        adaptive_widths[i] = (
            q * max(local_std / np.std(cal_data, ddof=1), 0.5)
            if np.std(cal_data, ddof=1) > 0
            else q
        )
        adaptive_lower[i] = local_center - adaptive_widths[i]
        adaptive_upper[i] = local_center + adaptive_widths[i]

    # ── Uncertainty Spike Detection ──
    median_width = np.median(adaptive_widths[:n_cal])
    spike_mask = adaptive_widths > spike_threshold * median_width
    spike_indices = np.where(spike_mask)[0]
    spike_indices_mon = spike_indices[spike_indices >= n_cal]

    # For Shewhart comparison
    shewhart_mean = np.mean(cal_data)
    shewhart_std = np.std(cal_data, ddof=1)
    shewhart_ucl = shewhart_mean + 3 * shewhart_std
    shewhart_lcl = shewhart_mean - 3 * shewhart_std

    # Summary
    summary_cc = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary_cc += "<<COLOR:title>>CONFORMAL-ENHANCED CONTROL CHART<</COLOR>>\n"
    summary_cc += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary_cc += "<<COLOR:dim>>Burger et al. (2025) \u2014 Distribution-free, guaranteed coverage<</COLOR>>\n\n"
    summary_cc += f"<<COLOR:highlight>>Variable:<</COLOR>> {measurement}\n"
    summary_cc += f"<<COLOR:highlight>>Chart type:<</COLOR>> {chart_label}\n"
    summary_cc += f"<<COLOR:highlight>>N:<</COLOR>> {n} ({n_cal} calibration + {n_mon} monitoring)\n"
    summary_cc += f"<<COLOR:highlight>>Significance level:<</COLOR>> \u03b1 = {alpha_conf} (guaranteed \u2264 {alpha_conf * 100:.0f}% false alarm rate)\n\n"

    summary_cc += (
        "<<COLOR:text>>Conformal Control Limits (distribution-free):<</COLOR>>\n"
    )
    summary_cc += f"  Center (median): {center:.4f}\n"
    summary_cc += f"  Conformal threshold (q): {q:.4f}\n"
    summary_cc += f"  Prediction interval: [{pi_lower:.4f}, {pi_upper:.4f}]\n\n"

    summary_cc += (
        "<<COLOR:text>>Traditional Shewhart Limits (for comparison):<</COLOR>>\n"
    )
    summary_cc += f"  Mean \u00b1 3\u03c3: [{shewhart_lcl:.4f}, {shewhart_ucl:.4f}]\n\n"

    # Compare the two approaches
    conformal_width = pi_upper - pi_lower
    shewhart_width = shewhart_ucl - shewhart_lcl
    if conformal_width < shewhart_width:
        summary_cc += f"<<COLOR:good>>Conformal limits are {(1 - conformal_width / shewhart_width) * 100:.0f}% tighter than Shewhart \u2014 more sensitive to shifts.<</COLOR>>\n\n"
    elif conformal_width > shewhart_width * 1.1:
        summary_cc += "<<COLOR:text>>Conformal limits are wider than Shewhart \u2014 data may be non-normal (heavy tails). This is the correct adjustment.<</COLOR>>\n\n"
    else:
        summary_cc += "<<COLOR:text>>Conformal and Shewhart limits are similar \u2014 data is approximately normal.<</COLOR>>\n\n"

    if n_ooc > 0:
        summary_cc += f"<<COLOR:warning>>\u26a0 {n_ooc} out-of-control point(s) in monitoring phase<</COLOR>>\n"
        if n_ooc <= 10:
            for idx in ooc_indices:
                summary_cc += f"  Observation {n_cal + idx}: value = {mon_values[idx]:.4f}, score = {mon_scores[idx]:.4f} > q = {q:.4f}\n"
    else:
        summary_cc += (
            "<<COLOR:good>>No out-of-control points in monitoring phase<</COLOR>>\n"
        )

    if len(spike_indices_mon) > 0:
        summary_cc += f"\n<<COLOR:warning>>\u26a0 {len(spike_indices_mon)} uncertainty spike(s) detected \u2014 leading indicators of instability<</COLOR>>\n"
        summary_cc += (
            f"  Spike threshold: {spike_threshold}\u00d7 median interval width\n"
        )
        if len(spike_indices_mon) <= 10:
            for idx in spike_indices_mon:
                summary_cc += f"  Observation {idx}: width = {adaptive_widths[idx]:.4f} ({adaptive_widths[idx] / median_width:.1f}\u00d7 normal)\n"
    else:
        summary_cc += "\n<<COLOR:good>>No uncertainty spikes detected<</COLOR>>\n"

    summary_cc += "\n<<COLOR:text>>Key advantages:<</COLOR>>\n"
    summary_cc += "  \u2022 Distribution-free: no normality assumption required\n"
    summary_cc += f"  \u2022 Guaranteed false alarm rate \u2264 \u03b1 = {alpha_conf}\n"
    summary_cc += (
        "  \u2022 Uncertainty spikes provide early warning before limits are breached\n"
    )
    summary_cc += "  \u2022 Adaptive intervals respond to changing process conditions\n"

    result["summary"] = summary_cc

    # Plot 1: Conformal control chart (main)
    n_cal_plot = len(cal_values)
    n_total = len(all_values)
    point_colors = []
    for i in range(n_total):
        if i < len(cal_values):
            point_colors.append("#4a9f6e")
        elif all_scores[i] > q:
            point_colors.append("#e85747")
        else:
            point_colors.append("#4a90d9")

    result["plots"].append(
        {
            "title": "Conformal-Enhanced Control Chart",
            "data": [
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": all_values.tolist(),
                    "mode": "lines+markers",
                    "marker": {"size": 5, "color": point_colors},
                    "line": {"color": "rgba(74, 159, 110, 0.3)", "width": 1},
                    "name": measurement,
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [center] * n_total,
                    "mode": "lines",
                    "line": {"color": "#00b894", "width": 1.5},
                    "name": f"Center (median={center:.3f})",
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [pi_upper] * n_total,
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                    "name": f"Conformal UCL ({pi_upper:.3f})",
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [pi_lower] * n_total,
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                    "name": f"Conformal LCL ({pi_lower:.3f})",
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [shewhart_ucl] * n_total,
                    "mode": "lines",
                    "line": {"color": "#9aaa9a", "dash": "dot", "width": 1},
                    "name": "Shewhart \u00b13\u03c3",
                    "showlegend": True,
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [shewhart_lcl] * n_total,
                    "mode": "lines",
                    "line": {"color": "#9aaa9a", "dash": "dot", "width": 1},
                    "showlegend": False,
                },
            ],
            "layout": {
                "height": 360,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": measurement},
                "shapes": [
                    {
                        "type": "line",
                        "x0": n_cal_plot,
                        "x1": n_cal_plot,
                        "y0": float(np.min(all_values))
                        - 0.1 * float(np.ptp(all_values)),
                        "y1": float(np.max(all_values))
                        + 0.1 * float(np.ptp(all_values)),
                        "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
                    }
                ],
                "annotations": [
                    {
                        "x": n_cal_plot,
                        "y": float(np.max(all_values)),
                        "text": "\u2190 Cal | Mon \u2192",
                        "showarrow": False,
                        "font": {"color": "#e8c547", "size": 10},
                    }
                ],
            },
        }
    )

    # Plot 2: Adaptive prediction interval (ribbon)
    result["plots"].append(
        {
            "title": "Adaptive Prediction Interval (uncertainty-aware)",
            "data": [
                {
                    "type": "scatter",
                    "x": list(range(n)) + list(range(n - 1, -1, -1)),
                    "y": adaptive_upper.tolist() + adaptive_lower[::-1].tolist(),
                    "fill": "toself",
                    "fillcolor": "rgba(74, 144, 217, 0.15)",
                    "line": {"color": "transparent"},
                    "name": "Adaptive PI",
                },
                {
                    "type": "scatter",
                    "x": list(range(n)),
                    "y": data.tolist(),
                    "mode": "lines+markers",
                    "marker": {"size": 3, "color": "#4a9f6e"},
                    "line": {"color": "#4a9f6e", "width": 1},
                    "name": measurement,
                },
            ]
            + (
                [
                    {
                        "type": "scatter",
                        "x": spike_indices_mon.tolist(),
                        "y": (
                            data[spike_indices_mon].tolist()
                            if len(spike_indices_mon) > 0
                            else []
                        ),
                        "mode": "markers",
                        "marker": {
                            "size": 10,
                            "color": "#e8c547",
                            "symbol": "triangle-up",
                            "line": {"color": "#e89547", "width": 1.5},
                        },
                        "name": "Uncertainty Spike",
                    }
                ]
                if len(spike_indices_mon) > 0
                else []
            ),
            "layout": {
                "height": 360,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": measurement},
            },
        }
    )

    # Plot 3: Nonconformity score chart
    score_colors = ["#e85747" if s > q else "#4a9f6e" for s in all_scores]
    result["plots"].append(
        {
            "title": "Nonconformity Scores vs Threshold",
            "data": [
                {
                    "type": "bar",
                    "x": list(range(n_total)),
                    "y": all_scores.tolist(),
                    "marker": {"color": score_colors},
                    "name": "Score",
                },
                {
                    "type": "scatter",
                    "x": list(range(n_total)),
                    "y": [q] * n_total,
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": f"Threshold q={q:.3f}",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "|X - median|"},
            },
        }
    )

    # Plot 4: Interval width over time (spike detection)
    width_colors = ["#e8c547" if spike_mask[i] else "#4a90d9" for i in range(n)]
    result["plots"].append(
        {
            "title": "Prediction Interval Width (spike = leading indicator)",
            "data": [
                {
                    "type": "bar",
                    "x": list(range(n)),
                    "y": (adaptive_widths * 2).tolist(),
                    "marker": {"color": width_colors, "opacity": 0.7},
                    "name": "Width",
                },
                {
                    "type": "scatter",
                    "x": list(range(n)),
                    "y": [spike_threshold * median_width * 2] * n,
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                    "name": f"Spike threshold ({spike_threshold}\u00d7 median)",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Interval Width"},
            },
        }
    )

    result["guide_observation"] = (
        f"Conformal control chart (distribution-free): {n_ooc} OOC, {len(spike_indices_mon)} uncertainty spikes in {n_mon} monitoring observations."
    )
    result["statistics"] = {
        "n": n,
        "n_calibration": n_cal,
        "n_monitoring": n_mon,
        "center": float(center),
        "conformal_threshold": float(q),
        "pi_lower": float(pi_lower),
        "pi_upper": float(pi_upper),
        "shewhart_ucl": float(shewhart_ucl),
        "shewhart_lcl": float(shewhart_lcl),
        "n_ooc": n_ooc,
        "ooc_indices": [int(n_cal + i) for i in ooc_indices],
        "n_spikes": len(spike_indices_mon),
        "spike_indices": spike_indices_mon.tolist(),
        "alpha": alpha_conf,
        "method": "Burger et al. (2025) arXiv:2512.23602",
    }

    # Narrative
    _cc_n_spikes = len(spike_indices_mon)
    if n_ooc == 0 and _cc_n_spikes == 0:
        _cc_verdict = (
            "Conformal Control Chart \u2014 process in control (distribution-free)"
        )
        _cc_body = f"No out-of-control points or uncertainty spikes in {n_mon} monitoring observations. Guaranteed false alarm rate = {alpha_conf * 100:.1f}% without normality assumption."
    else:
        _cc_verdict = f"Conformal Control Chart \u2014 {n_ooc} OOC" + (
            f", {_cc_n_spikes} uncertainty spike{'s' if _cc_n_spikes > 1 else ''}"
            if _cc_n_spikes > 0
            else ""
        )
        _cc_body = (
            f"{n_ooc} out-of-control points detected in {n_mon} monitoring observations (calibrated on {n_cal} points). "
            + (
                f"Additionally, {_cc_n_spikes} uncertainty spikes serve as early warnings of instability."
                if _cc_n_spikes > 0
                else ""
            )
        )
    result["narrative"] = _narrative(
        _cc_verdict,
        _cc_body,
        next_steps=(
            "Investigate OOC points for assignable causes. Uncertainty spikes often precede full OOC events."
            if n_ooc > 0 or _cc_n_spikes > 0
            else "Process is stable under distribution-free monitoring."
        ),
        chart_guidance="Blue band = adaptive prediction interval. Yellow triangles = uncertainty spikes (interval widening). Red points = out-of-control observations exceeding the conformal threshold.",
    )

    return result


def run_conformal_monitor(df, config):
    """Conformal P-Value Chart -- multivariate process monitoring."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    alpha_conf = float(config.get("alpha", 0.05))
    cal_fraction = float(config.get("calibration_fraction", 0.5))
    model_type = config.get(
        "model", "isolation_forest"
    )  # isolation_forest, mahalanobis

    # Get numeric columns
    variables = config.get("variables", [])
    if not variables:
        # Fallback: if single var provided, use all numeric columns
        variables = df.select_dtypes(include="number").columns.tolist()
    if len(variables) < 2:
        result["summary"] = (
            "Conformal monitoring requires at least 2 numeric variables for multivariate analysis."
        )
        return result

    X = df[variables].dropna().values
    n = len(X)

    if n < 30:
        result["summary"] = (
            "Need at least 30 observations for conformal multivariate monitoring."
        )
        return result

    # Phase I / Phase II split
    n_cal = max(15, int(n * cal_fraction))
    X_cal = X[:n_cal]
    X_mon = X[n_cal:]
    n_mon = len(X_mon)

    # Standardize using calibration data
    scaler = StandardScaler()
    X_cal_scaled = scaler.fit_transform(X_cal)
    X_mon_scaled = (
        scaler.transform(X_mon)
        if n_mon > 0
        else np.array([]).reshape(0, len(variables))
    )

    # Compute nonconformity scores
    if model_type == "mahalanobis":
        # Mahalanobis distance from calibration centroid
        cov_cal = np.cov(X_cal_scaled.T)
        try:
            cov_inv = np.linalg.inv(cov_cal)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_cal)
        mean_cal = np.mean(X_cal_scaled, axis=0)

        cal_scores = np.array(
            [np.sqrt((x - mean_cal) @ cov_inv @ (x - mean_cal)) for x in X_cal_scaled]
        )
        mon_scores = (
            np.array(
                [
                    np.sqrt((x - mean_cal) @ cov_inv @ (x - mean_cal))
                    for x in X_mon_scaled
                ]
            )
            if n_mon > 0
            else np.array([])
        )
        model_label = "Mahalanobis Distance"
    else:
        # Isolation Forest anomaly scores
        iso = IsolationForest(random_state=42, contamination="auto")
        iso.fit(X_cal_scaled)

        # Score = -decision_function (higher = more anomalous)
        cal_scores = -iso.decision_function(X_cal_scaled)
        mon_scores = -iso.decision_function(X_mon_scaled) if n_mon > 0 else np.array([])
        model_label = "Isolation Forest"

    all_scores = np.concatenate([cal_scores, mon_scores])

    # Conformal p-values for monitoring observations
    cal_p_values = np.array(
        [(np.sum(cal_scores >= s) + 1) / (n_cal + 1) for s in cal_scores]
    )
    mon_p_values = (
        np.array([(np.sum(cal_scores >= s) + 1) / (n_cal + 1) for s in mon_scores])
        if n_mon > 0
        else np.array([])
    )
    all_p_values = np.concatenate([cal_p_values, mon_p_values])

    # Anomalies: p-value < alpha
    if n_mon > 0:
        anomaly_mask = mon_p_values < alpha_conf
        anomaly_indices = np.where(anomaly_mask)[0]
        n_anomalies = len(anomaly_indices)
    else:
        anomaly_indices = np.array([])
        n_anomalies = 0

    # Variable contributions for flagged points
    contributions = []
    if n_anomalies > 0 and n_anomalies <= 20:
        mean_cal_raw = np.mean(X_cal, axis=0)
        std_cal_raw = np.std(X_cal, axis=0, ddof=1)
        std_cal_raw[std_cal_raw == 0] = 1
        for idx in anomaly_indices[:10]:
            z_scores = np.abs((X_mon[idx] - mean_cal_raw) / std_cal_raw)
            top_var = variables[np.argmax(z_scores)]
            contributions.append((int(idx), top_var, float(np.max(z_scores))))

    # Summary
    summary_cm = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary_cm += (
        "<<COLOR:title>>CONFORMAL P-VALUE CHART (Multivariate Monitor)<</COLOR>>\n"
    )
    summary_cm += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary_cm += "<<COLOR:dim>>Burger et al. (2025) \u2014 Distribution-free anomaly detection<</COLOR>>\n\n"
    summary_cm += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables)} ({len(variables)} dimensions)\n"
    summary_cm += f"<<COLOR:highlight>>N:<</COLOR>> {n} ({n_cal} calibration + {n_mon} monitoring)\n"
    summary_cm += f"<<COLOR:highlight>>Anomaly model:<</COLOR>> {model_label}\n"
    summary_cm += (
        f"<<COLOR:highlight>>Significance level:<</COLOR>> \u03b1 = {alpha_conf}\n\n"
    )

    summary_cm += (
        "<<COLOR:text>>Conformal P-Value Distribution (monitoring):<</COLOR>>\n"
    )
    if n_mon > 0:
        summary_cm += f"  Mean p-value: {np.mean(mon_p_values):.4f}\n"
        summary_cm += f"  Min p-value: {np.min(mon_p_values):.4f} at observation {n_cal + int(np.argmin(mon_p_values))}\n"
        summary_cm += (
            f"  % below \u03b1: {np.mean(mon_p_values < alpha_conf) * 100:.1f}%\n\n"
        )

    if n_anomalies > 0:
        summary_cm += f"<<COLOR:warning>>\u26a0 {n_anomalies} anomalous observation(s) detected (p < {alpha_conf})<</COLOR>>\n"
        if contributions:
            summary_cm += "\n  <<COLOR:text>>Top contributing variables:<</COLOR>>\n"
            for idx, var, z in contributions:
                summary_cm += f"    Obs {n_cal + idx}: driven by '{var}' (z = {z:.2f}\u03c3 from calibration mean)\n"
    else:
        summary_cm += f"<<COLOR:good>>No anomalies detected \u2014 all p-values \u2265 {alpha_conf}<</COLOR>>\n"

    summary_cm += "\n<<COLOR:text>>Key advantages:<</COLOR>>\n"
    summary_cm += f"  \u2022 Monitors {len(variables)} variables simultaneously\n"
    summary_cm += "  \u2022 No distributional assumptions on joint variable behavior\n"
    summary_cm += f"  \u2022 Guaranteed false alarm rate \u2264 \u03b1 = {alpha_conf}\n"
    summary_cm += (
        "  \u2022 Intuitive p-value scale (0 = most anomalous, 1 = most normal)\n"
    )

    result["summary"] = summary_cm

    # Plot 1: P-value chart (the main chart)
    p_colors = []
    for i, p in enumerate(all_p_values):
        if i < n_cal:
            p_colors.append("#4a9f6e")
        elif p < alpha_conf:
            p_colors.append("#e85747")
        else:
            p_colors.append("#4a90d9")

    result["plots"].append(
        {
            "title": "Conformal P-Value Chart",
            "data": [
                {
                    "type": "scatter",
                    "x": list(range(n)),
                    "y": all_p_values.tolist(),
                    "mode": "markers",
                    "marker": {"size": 6, "color": p_colors},
                    "name": "Conformal p-value",
                },
                {
                    "type": "scatter",
                    "x": list(range(n)),
                    "y": [alpha_conf] * n,
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": f"\u03b1 = {alpha_conf}",
                },
            ],
            "layout": {
                "height": 290,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Conformal p-value", "range": [0, 1]},
                "shapes": [
                    {
                        "type": "line",
                        "x0": n_cal,
                        "x1": n_cal,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
                    },
                    {
                        "type": "rect",
                        "x0": 0,
                        "x1": n,
                        "y0": 0,
                        "y1": alpha_conf,
                        "fillcolor": "rgba(232, 87, 71, 0.05)",
                        "line": {"width": 0},
                    },
                ],
                "annotations": [
                    {
                        "x": n_cal,
                        "y": 0.95,
                        "text": "\u2190 Cal | Mon \u2192",
                        "showarrow": False,
                        "font": {"color": "#e8c547", "size": 10},
                    }
                ],
            },
        }
    )

    # Plot 2: Anomaly scores over time
    score_colors_2 = [
        "#e85747" if (i >= n_cal and all_p_values[i] < alpha_conf) else "#4a9f6e"
        for i in range(n)
    ]
    result["plots"].append(
        {
            "title": f"Nonconformity Scores ({model_label})",
            "data": [
                {
                    "type": "bar",
                    "x": list(range(n)),
                    "y": all_scores.tolist(),
                    "marker": {"color": score_colors_2, "opacity": 0.7},
                    "name": "Score",
                },
            ],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "Anomaly Score"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": n_cal,
                        "x1": n_cal,
                        "y0": 0,
                        "y1": float(np.max(all_scores)),
                        "line": {"color": "#e8c547", "dash": "dashdot", "width": 1},
                    }
                ],
            },
        }
    )

    # Plot 3: Variable-level view (heatmap of z-scores)
    if n_mon > 0:
        mean_cal_raw = np.mean(X_cal, axis=0)
        std_cal_raw = np.std(X_cal, axis=0, ddof=1)
        std_cal_raw[std_cal_raw == 0] = 1
        z_matrix = np.abs((X_mon - mean_cal_raw) / std_cal_raw)

        result["plots"].append(
            {
                "title": "Variable Contribution Heatmap (|z-score| from calibration)",
                "data": [
                    {
                        "type": "heatmap",
                        "z": z_matrix.T.tolist(),
                        "x": [f"{n_cal + i}" for i in range(n_mon)],
                        "y": variables,
                        "colorscale": [
                            [0, "#4a9f6e"],
                            [0.5, "#e8c547"],
                            [1, "#e85747"],
                        ],
                        "colorbar": {"title": "|z|"},
                    }
                ],
                "layout": {
                    "height": max(200, 40 * len(variables)),
                    "xaxis": {"title": "Observation"},
                },
            }
        )

    result["guide_observation"] = (
        f"Conformal multivariate monitor: {n_anomalies} anomalies in {n_mon} monitoring observations across {len(variables)} variables."
    )
    result["statistics"] = {
        "n": n,
        "n_calibration": n_cal,
        "n_monitoring": n_mon,
        "n_variables": len(variables),
        "variables": variables,
        "model": model_label,
        "alpha": alpha_conf,
        "n_anomalies": n_anomalies,
        "anomaly_indices": [int(n_cal + i) for i in anomaly_indices],
        "mean_p_value": float(np.mean(mon_p_values)) if n_mon > 0 else None,
        "min_p_value": float(np.min(mon_p_values)) if n_mon > 0 else None,
        "method": "Burger et al. (2025) arXiv:2512.23602",
    }

    # Narrative
    if n_anomalies == 0:
        _cm_verdict = (
            f"Conformal Monitor \u2014 no anomalies ({len(variables)} variables)"
        )
        _cm_body = f"All {n_mon} monitoring observations are within normal bounds across {len(variables)} variables using {model_label}. False alarm rate controlled at {alpha_conf * 100:.1f}%."
    else:
        _cm_top_var = ""
        if contributions:
            from collections import Counter

            _cm_var_counts = Counter(v for _, v, _ in contributions)
            _cm_top_var = f" Top contributing variable: <strong>{_cm_var_counts.most_common(1)[0][0]}</strong>."
        _cm_verdict = f"Conformal Monitor \u2014 {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} detected"
        _cm_body = f"{n_anomalies} of {n_mon} monitoring observations flagged as anomalous across {len(variables)} variables.{_cm_top_var}"
    result["narrative"] = _narrative(
        _cm_verdict,
        _cm_body,
        next_steps=(
            "Check the variable contribution heatmap to identify which dimensions are driving anomalies."
            if n_anomalies > 0
            else "Process is multivariate-stable. Continue monitoring."
        ),
        chart_guidance="The p-value chart shows conformal p-values per observation \u2014 points below the red line are anomalous. The heatmap reveals which variables contribute most to each flagged observation.",
    )

    return result


def run_entropy_spc(df, config):
    """Information-Theoretic SPC -- Shannon entropy control chart."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    col = config.get("var") or config.get("measurement") or config.get("column")
    n_bins = int(config.get("bins", config.get("n_bins", 10)))
    window = int(config.get("window", 30))

    if not col or col not in df.columns:
        result["summary"] = "Error: Specify a numeric column."
        return result

    data = df[col].dropna().values.astype(float)
    n = len(data)

    if n < window * 2:
        result["summary"] = (
            f"Error: Need at least {window * 2} observations for window={window}."
        )
        return result

    # Compute rolling Shannon entropy on binned data
    global_min, global_max = float(np.min(data)), float(np.max(data))
    bin_edges = np.linspace(global_min - 1e-10, global_max + 1e-10, n_bins + 1)

    entropies = []
    for i in range(window, n + 1):
        segment = data[i - window : i]
        counts, _ = np.histogram(segment, bins=bin_edges)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = float(-np.sum(probs * np.log2(probs)))
        entropies.append(h)

    entropies = np.array(entropies)
    x_idx = list(range(window, n + 1))

    # Phase I limits (first half as baseline)
    n_phase1 = len(entropies) // 2
    h_mean = float(np.mean(entropies[:n_phase1]))
    h_std = float(np.std(entropies[:n_phase1], ddof=1))
    ucl = h_mean + 3 * h_std
    lcl = max(0, h_mean - 3 * h_std)

    # Detect OOC points
    ooc_idx = [i for i, h in enumerate(entropies) if h > ucl or h < lcl]
    ooc_pct = len(ooc_idx) / len(entropies) * 100

    # Detect entropy drops (distribution narrowing / mode collapse)
    drops = [i for i, h in enumerate(entropies) if h < lcl]
    # Detect entropy spikes (distribution spreading / bimodality forming)
    spikes = [i for i, h in enumerate(entropies) if h > ucl]

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += (
        "<<COLOR:title>>ENTROPY SPC (INFORMATION-THEORETIC CONTROL CHART)<</COLOR>>\n"
    )
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Variable:<</COLOR>> {col}\n"
    summary += (
        f"<<COLOR:text>>Window:<</COLOR>> {window}    Bins: {n_bins}    N: {n}\n\n"
    )
    summary += (
        "<<COLOR:accent>>\u2500\u2500 Entropy Statistics \u2500\u2500<</COLOR>>\n"
    )
    summary += f"  Baseline mean: {h_mean:.4f} bits\n"
    summary += f"  UCL: {ucl:.4f}    LCL: {lcl:.4f}\n"
    summary += f"  OOC points: {len(ooc_idx)} ({ooc_pct:.1f}%)\n"
    if spikes:
        summary += f"  Entropy spikes (distribution spreading): {len(spikes)}\n"
    if drops:
        summary += f"  Entropy drops (distribution narrowing): {len(drops)}\n"

    result["summary"] = summary
    result["statistics"] = {
        "h_mean": h_mean,
        "h_std": h_std,
        "ucl": ucl,
        "lcl": lcl,
        "n_ooc": len(ooc_idx),
        "n_spikes": len(spikes),
        "n_drops": len(drops),
    }

    _es_status = (
        "in control" if not ooc_idx else f"{len(ooc_idx)} out-of-control signals"
    )
    _es_detail = ""
    if spikes and drops:
        _es_detail = " Both entropy spikes (distributional spreading) and drops (narrowing) detected."
    elif spikes:
        _es_detail = " Entropy spikes indicate distributional spreading \u2014 possible bimodality or increased variation."
    elif drops:
        _es_detail = " Entropy drops indicate distribution narrowing \u2014 possible mode collapse or reduced variation."

    result["guide_observation"] = (
        f"Entropy SPC: {_es_status}. Mean entropy = {h_mean:.3f} bits."
    )
    result["narrative"] = _narrative(
        f"Entropy SPC \u2014 {_es_status}",
        f"Rolling Shannon entropy (window={window}, {n_bins} bins) has baseline mean {h_mean:.4f} bits. "
        f"{len(ooc_idx)} points ({ooc_pct:.1f}%) fall outside 3\u03c3 limits [{lcl:.3f}, {ucl:.3f}].{_es_detail}",
        next_steps="Entropy catches distributional shifts that Shewhart charts miss \u2014 bimodality forming inside normal control limits, "
        "shape changes without mean shift. Investigate OOC entropy points for root cause.",
        chart_guidance="Entropy above UCL = distribution is spreading or splitting. Below LCL = distribution is collapsing. "
        "Compare with the Xbar chart \u2014 entropy often signals before the mean moves.",
    )

    # Plot: entropy control chart
    h_colors = ["#dc5050" if i in ooc_idx else "#4a9f6e" for i in range(len(entropies))]
    result["plots"].append(
        {
            "title": f"Entropy Control Chart ({col})",
            "data": [
                {
                    "type": "scatter",
                    "x": x_idx,
                    "y": entropies.tolist(),
                    "mode": "lines+markers",
                    "marker": {"size": 4, "color": h_colors},
                    "line": {"color": "#4a9f6e", "width": 1},
                    "name": "Entropy",
                },
            ],
            "layout": {
                "height": 290,
                "xaxis": {
                    "title": "Observation",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": "Shannon Entropy (bits)"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": x_idx[0],
                        "x1": x_idx[-1],
                        "y0": h_mean,
                        "y1": h_mean,
                        "line": {"color": "#4a90d9", "width": 1},
                    },
                    {
                        "type": "line",
                        "x0": x_idx[0],
                        "x1": x_idx[-1],
                        "y0": ucl,
                        "y1": ucl,
                        "line": {"color": "#dc5050", "dash": "dash", "width": 1},
                    },
                    {
                        "type": "line",
                        "x0": x_idx[0],
                        "x1": x_idx[-1],
                        "y0": lcl,
                        "y1": lcl,
                        "line": {"color": "#dc5050", "dash": "dash", "width": 1},
                    },
                ],
            },
        }
    )

    return result
