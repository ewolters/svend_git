"""SPC capability analyses — Capability, Non-Normal, Degradation, Between/Within."""

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm

from ..common import _narrative


def run_capability(df, config):
    """Standard capability analysis with Cp, Cpk, Pp, Ppk."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    lsl = float(config.get("lsl")) if config.get("lsl") else None
    usl = float(config.get("usl")) if config.get("usl") else None
    target = float(config.get("target")) if config.get("target") else None

    n = len(data)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))

    # ── Capability indices ──────────────────────────────────────
    cp = cpk = pp = ppk = cpm = None
    ppm_below = ppm_above = ppm_total = 0.0
    yield_pct = 100.0
    sigma_level = 0.0

    if lsl is not None and usl is not None and std > 0:
        # Pp/Ppk use overall std (long-term)
        pp = (usl - lsl) / (6 * std)
        ppk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        # Cp/Cpk use within-subgroup sigma (MR-bar/d2 for individuals)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr) if len(mr) > 0 else std
        d2 = 1.128  # d2 constant for n=2 (moving range of 2)
        sigma_within = mr_bar / d2 if mr_bar > 0 else std
        cp = (usl - lsl) / (6 * sigma_within)
        cpk = min((usl - mean) / (3 * sigma_within), (mean - lsl) / (3 * sigma_within))
        if target is not None:
            cpm = (usl - lsl) / (6 * np.sqrt(std**2 + (mean - target) ** 2))

        # Expected defects
        z_lower = (mean - lsl) / std
        z_upper = (usl - mean) / std
        ppm_below = float(sp_stats.norm.cdf(-z_lower) * 1e6)
        ppm_above = float(sp_stats.norm.cdf(-z_upper) * 1e6)
        ppm_total = ppm_below + ppm_above
        yield_pct = (1 - ppm_total / 1e6) * 100
        if ppm_total > 0 and ppm_total < 1e6:
            sigma_level = float(sp_stats.norm.ppf(1 - ppm_total / 1e6) + 1.5)
        elif ppm_total == 0:
            sigma_level = 6.0  # Cap at 6 sigma when zero defects predicted
        else:
            sigma_level = 0.0

    elif lsl is not None and std > 0:
        cpk = (mean - lsl) / (3 * std)
        z_lower = (mean - lsl) / std
        ppm_below = float(sp_stats.norm.cdf(-z_lower) * 1e6)
        ppm_total = ppm_below
        yield_pct = (1 - ppm_total / 1e6) * 100

    elif usl is not None and std > 0:
        cpk = (usl - mean) / (3 * std)
        z_upper = (usl - mean) / std
        ppm_above = float(sp_stats.norm.cdf(-z_upper) * 1e6)
        ppm_total = ppm_above
        yield_pct = (1 - ppm_total / 1e6) * 100

    # ── Summary text ────────────────────────────────────────────
    summary = f"Capability Analysis  (n = {n})\n\n"
    summary += f"  Mean:      {mean:.4f}\n"
    summary += f"  Std Dev:   {std:.4f}\n"
    summary += f"  Min:       {float(np.min(data)):.4f}\n"
    summary += f"  Max:       {float(np.max(data)):.4f}\n"

    if cp is not None:
        summary += "\nCapability Indices:\n"
        summary += f"  Cp:   {cp:.3f}     Pp:   {pp:.3f}\n"
        summary += f"  Cpk:  {cpk:.3f}     Ppk:  {ppk:.3f}\n"
        if cpm is not None:
            summary += f"  Cpm:  {cpm:.3f}   (Taguchi, target = {target})\n"
    elif cpk is not None:
        summary += "\nCapability (one-sided):\n"
        summary += f"  Cpk:  {cpk:.3f}\n"

    if lsl is not None or usl is not None:
        summary += "\nExpected Performance:\n"
        if lsl is not None:
            summary += f"  PPM < LSL:  {ppm_below:,.0f}\n"
        if usl is not None:
            summary += f"  PPM > USL:  {ppm_above:,.0f}\n"
        summary += f"  PPM Total:  {ppm_total:,.0f}\n"
        summary += f"  Yield:      {yield_pct:.4f}%\n"
        if cp is not None:
            summary += f"  Sigma:      {sigma_level:.2f}\n"

    if cpk is not None:
        if cpk >= 1.33:
            summary += f"\nProcess is capable (Cpk = {cpk:.3f} >= 1.33)"
        elif cpk >= 1.0:
            summary += f"\nProcess is marginally capable (1.0 <= Cpk = {cpk:.3f} < 1.33)"
        else:
            summary += f"\nProcess is NOT capable (Cpk = {cpk:.3f} < 1.0)"
        result["guide_observation"] = f"Process capability Cpk = {cpk:.2f}. " + (
            "Capable." if cpk >= 1.33 else "Needs improvement."
        )

    result["statistics"] = {
        "mean": float(mean),
        "std": float(std),
        "n": n,
        "cp": float(cp) if cp is not None else None,
        "cpk": float(cpk) if cpk is not None else None,
        "pp": float(pp) if pp is not None else None,
        "ppk": float(ppk) if ppk is not None else None,
        "sigma_level": float(sigma_level),
        "ppm_total": float(ppm_total),
        "yield_pct": float(yield_pct),
    }

    # ── Plot 1: Histogram with normal curve ─────────────────────
    x_range = np.linspace(float(np.min(data)) - 2 * std, float(np.max(data)) + 2 * std, 300)
    pdf_vals = sp_stats.norm.pdf(x_range, mean, std)

    hist_traces = [
        {
            "type": "histogram",
            "x": data.tolist(),
            "name": "Observed",
            "histnorm": "probability density",
            "marker": {
                "color": "rgba(74, 159, 110, 0.35)",
                "line": {"color": "#4a9f6e", "width": 1},
            },
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": pdf_vals.tolist(),
            "mode": "lines",
            "name": "Normal Fit",
            "line": {"color": "#4a90d9", "width": 2.5},
        },
    ]

    shapes_h = []
    annotations_h = []

    # Mean line
    shapes_h.append(
        {
            "type": "line",
            "x0": mean,
            "x1": mean,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#00b894", "width": 2},
        }
    )
    annotations_h.append(
        {
            "x": mean,
            "y": 1.06,
            "yref": "paper",
            "text": "Mean",
            "showarrow": False,
            "font": {"color": "#00b894", "size": 10},
        }
    )

    # Target line
    if target is not None:
        shapes_h.append(
            {
                "type": "line",
                "x0": target,
                "x1": target,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
            }
        )
        annotations_h.append(
            {
                "x": target,
                "y": 1.06,
                "yref": "paper",
                "text": "Target",
                "showarrow": False,
                "font": {"color": "#e8c547", "size": 10},
            }
        )

    # LSL / USL lines
    if lsl is not None:
        shapes_h.append(
            {
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        annotations_h.append(
            {
                "x": lsl,
                "y": 1.06,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11},
            }
        )
    if usl is not None:
        shapes_h.append(
            {
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        annotations_h.append(
            {
                "x": usl,
                "y": 1.06,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11},
            }
        )

    result["plots"].append(
        {
            "title": "Capability Histogram",
            "data": hist_traces,
            "layout": {
                "height": 320,
                "shapes": shapes_h,
                "annotations": annotations_h,
                "showlegend": True,
                "legend": {
                    "x": 1,
                    "xanchor": "right",
                    "y": 1,
                    "bgcolor": "rgba(0,0,0,0)",
                },
                "margin": {"t": 40, "r": 20},
                "xaxis": {"title": measurement},
                "yaxis": {"title": "Density"},
            },
        }
    )

    # ── Plot 2: Process spread vs specs ─────────────────────────
    if lsl is not None and usl is not None:
        spread_lo = mean - 3 * std
        spread_hi = mean + 3 * std
        pad = (usl - lsl) * 0.15

        spread_traces = [
            # Spec range
            {
                "type": "bar",
                "y": [""],
                "x": [usl - lsl],
                "base": [lsl],
                "orientation": "h",
                "name": "Spec Range",
                "marker": {
                    "color": "rgba(232, 87, 71, 0.15)",
                    "line": {"color": "#e85747", "width": 1.5},
                },
                "width": [0.5],
            },
            # Process spread (±3σ)
            {
                "type": "bar",
                "y": [""],
                "x": [spread_hi - spread_lo],
                "base": [spread_lo],
                "orientation": "h",
                "name": "Process \u00b13\u03c3",
                "marker": {
                    "color": "rgba(74, 159, 110, 0.25)",
                    "line": {"color": "#4a9f6e", "width": 1.5},
                },
                "width": [0.3],
            },
        ]

        spread_shapes = []
        spread_annot = []

        # Mean marker
        spread_shapes.append(
            {
                "type": "line",
                "x0": mean,
                "x1": mean,
                "y0": -0.3,
                "y1": 0.3,
                "line": {"color": "#00b894", "width": 2.5},
            }
        )
        spread_annot.append(
            {
                "x": mean,
                "y": 0.35,
                "text": f"\u03bc={mean:.2f}",
                "showarrow": False,
                "font": {"color": "#00b894", "size": 10},
            }
        )

        # LSL / USL labels on the bar
        spread_annot.append(
            {
                "x": lsl,
                "y": -0.35,
                "text": f"LSL={lsl}",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 10},
            }
        )
        spread_annot.append(
            {
                "x": usl,
                "y": -0.35,
                "text": f"USL={usl}",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 10},
            }
        )

        # ±3σ labels
        spread_annot.append(
            {
                "x": spread_lo,
                "y": 0.35,
                "text": "-3\u03c3",
                "showarrow": False,
                "font": {"color": "#4a9f6e", "size": 9},
            }
        )
        spread_annot.append(
            {
                "x": spread_hi,
                "y": 0.35,
                "text": "+3\u03c3",
                "showarrow": False,
                "font": {"color": "#4a9f6e", "size": 9},
            }
        )

        # Target marker
        if target is not None:
            spread_shapes.append(
                {
                    "type": "line",
                    "x0": target,
                    "x1": target,
                    "y0": -0.3,
                    "y1": 0.3,
                    "line": {"color": "#e8c547", "dash": "dashdot", "width": 1.5},
                }
            )
            spread_annot.append(
                {
                    "x": target,
                    "y": -0.35,
                    "text": f"T={target}",
                    "showarrow": False,
                    "font": {"color": "#e8c547", "size": 10},
                }
            )

        result["plots"].append(
            {
                "title": "Process Spread vs Specification",
                "data": spread_traces,
                "layout": {
                    "height": 180,
                    "barmode": "overlay",
                    "shapes": spread_shapes,
                    "annotations": spread_annot,
                    "showlegend": True,
                    "legend": {
                        "x": 1,
                        "xanchor": "right",
                        "y": 1,
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                    "xaxis": {
                        "range": [min(lsl, spread_lo) - pad, max(usl, spread_hi) + pad],
                        "title": measurement,
                    },
                    "yaxis": {"visible": False, "range": [-0.5, 0.5]},
                    "margin": {"t": 35, "b": 45, "l": 20, "r": 20},
                },
            }
        )

    # ── Plot 3: Normal probability plot (Q-Q) ──────────────────
    sorted_data = np.sort(data)
    n_pts = len(sorted_data)
    probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
    theoretical_q = sp_stats.norm.ppf(probs, mean, std)

    # Shapiro-Wilk test (limited to 5000 samples)
    sw_data = data[:5000] if n > 5000 else data
    sw_stat, sw_p = sp_stats.shapiro(sw_data)
    normality_note = f"Shapiro-Wilk p = {sw_p:.4f}" + (" (normal)" if sw_p >= 0.05 else " (non-normal)")

    result["plots"].append(
        {
            "title": f"Normal Probability Plot  ({normality_note})",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "name": "Data",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.7},
                },
                {
                    "type": "scatter",
                    "x": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "y": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "mode": "lines",
                    "name": "Reference",
                    "line": {"color": "#e85747", "dash": "dash", "width": 1.5},
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Observed"},
                "showlegend": False,
                "margin": {"t": 35},
            },
        }
    )

    result["summary"] = summary

    # What-If data for client-side interactive exploration
    result["what_if_data"] = {
        "type": "capability",
        "mean": float(mean),
        "std": float(std),
        "n": int(n),
        "current_lsl": float(lsl) if lsl else None,
        "current_usl": float(usl) if usl else None,
        "data_values": data.tolist() if n <= 5000 else data[:5000].tolist(),
    }

    # Narrative
    if cpk is not None:
        _tol_pct = ((6 * std) / (usl - lsl) * 100) if lsl is not None and usl is not None and (usl - lsl) > 0 else None
        if cpk >= 1.33:
            _cap_verdict = f"Process is capable (Cpk = {cpk:.3f})"
            _cap_next = "Process is capable. Monitor with control charts to maintain."
        elif cpk >= 1.0:
            _centering = "centering (adjust mean)" if cp is not None and cp > cpk + 0.1 else "spread (reduce variation)"
            _cap_verdict = f"Process is marginally capable (Cpk = {cpk:.3f})"
            _cap_next = f"Process is marginal. The dominant issue is {_centering}."
        else:
            _cap_verdict = f"Process is NOT capable (Cpk = {cpk:.3f})"
            _cap_next = "Process is not capable. Run a Gage R&R to confirm measurement isn't inflating variation, then investigate root causes with multi-vari or DOE."
        _cap_body = f"Cpk = {cpk:.3f}"
        if _tol_pct is not None:
            _cap_body += f" -- the process uses {_tol_pct:.0f}% of the tolerance"
        _cap_body += f". Estimated {ppm_total:,.0f} defects per million ({yield_pct:.2f}% yield)."
        if cp is not None and ppk is not None and cpk < ppk - 0.05:
            _cap_body += " Short-term capability exceeds long-term -- the process has shifts or drifts not captured in subgroups."
        result["narrative"] = _narrative(
            _cap_verdict,
            _cap_body,
            next_steps=_cap_next,
            chart_guidance="The histogram shows data vs spec limits (red lines). The normal curve is the fitted distribution. Data outside the spec lines are predicted defects.",
        )

    return result


def run_nonnormal_capability(df, config):
    """Non-Normal Capability Analysis. Fits Normal, Lognormal, Weibull, Exponential."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    lsl = float(config.get("lsl")) if config.get("lsl") else None
    usl = float(config.get("usl")) if config.get("usl") else None

    pos_data = data[data > 0]  # needed for lognormal/weibull

    # Fit distributions
    fits = {}

    # Normal
    mu_n, sigma_n = sp_stats.norm.fit(data)
    fits["Normal"] = {
        "params": (mu_n, sigma_n),
        "dist": sp_stats.norm,
        "args": (mu_n, sigma_n),
        "ks": sp_stats.kstest(data, "norm", args=(mu_n, sigma_n)),
    }

    # Lognormal (needs positive data)
    if len(pos_data) > 10:
        shape_ln, loc_ln, scale_ln = sp_stats.lognorm.fit(pos_data, floc=0)
        fits["Lognormal"] = {
            "params": (shape_ln, 0, scale_ln),
            "dist": sp_stats.lognorm,
            "args": (shape_ln, 0, scale_ln),
            "ks": sp_stats.kstest(pos_data, "lognorm", args=(shape_ln, 0, scale_ln)),
        }

    # Weibull (needs positive data)
    if len(pos_data) > 10:
        shape_w, loc_w, scale_w = sp_stats.weibull_min.fit(pos_data, floc=0)
        fits["Weibull"] = {
            "params": (shape_w, 0, scale_w),
            "dist": sp_stats.weibull_min,
            "args": (shape_w, 0, scale_w),
            "ks": sp_stats.kstest(pos_data, "weibull_min", args=(shape_w, 0, scale_w)),
        }

    # Exponential (needs positive data)
    if len(pos_data) > 10:
        loc_e, scale_e = sp_stats.expon.fit(pos_data)
        fits["Exponential"] = {
            "params": (loc_e, scale_e),
            "dist": sp_stats.expon,
            "args": (loc_e, scale_e),
            "ks": sp_stats.kstest(pos_data, "expon", args=(loc_e, scale_e)),
        }

    # Select best fit by KS p-value (highest = best fit)
    best_name = max(fits, key=lambda k: fits[k]["ks"].pvalue)
    best = fits[best_name]
    best_dist = best["dist"]
    best_args = best["args"]

    summary = f"Non-Normal Capability Analysis\n\nBest Fit Distribution: {best_name}\n\n"
    summary += "Distribution Fit Comparison (Anderson-Darling / KS test):\n"
    summary += f"  {'Distribution':<15} {'KS Stat':>10} {'p-value':>10} {'Fit':>6}\n"
    summary += f"  {'-' * 45}\n"
    for name, info in fits.items():
        marker = " <--" if name == best_name else ""
        summary += f"  {name:<15} {info['ks'].statistic:>10.4f} {info['ks'].pvalue:>10.4f} {marker}\n"

    # Compute Pp/Ppk using the fitted distribution
    if lsl is not None and usl is not None:
        p_lsl = best_dist.cdf(lsl, *best_args)
        p_usl = 1 - best_dist.cdf(usl, *best_args)

        # Equivalent Pp from total proportion out of spec
        total_ppm = (p_lsl + p_usl) * 1e6
        # Z-equivalent
        z_lsl = sp_norm.ppf(1 - p_lsl) if p_lsl < 1 else 0
        z_usl = sp_norm.ppf(1 - p_usl) if p_usl < 1 else 0
        ppk_equiv = min(z_lsl, z_usl) / 3 if (z_lsl > 0 and z_usl > 0) else 0

        # Pp from spec width vs distribution spread (0.135% to 99.865%)
        q_low = best_dist.ppf(0.00135, *best_args)
        q_high = best_dist.ppf(0.99865, *best_args)
        spread_6sigma = q_high - q_low
        pp_equiv = (usl - lsl) / spread_6sigma if spread_6sigma > 0 else 0

        summary += f"\nCapability Indices ({best_name} fit):\n"
        summary += f"  Pp (equivalent): {pp_equiv:.3f}\n"
        summary += f"  Ppk (equivalent): {ppk_equiv:.3f}\n"
        summary += f"  P(below LSL): {p_lsl * 100:.4f}%\n"
        summary += f"  P(above USL): {p_usl * 100:.4f}%\n"
        summary += f"  Total PPM: {total_ppm:.0f}\n"

    # Histogram with best-fit overlay
    x_range = np.linspace(min(data), max(data), 200)
    pdf_vals = best_dist.pdf(x_range, *best_args)

    hist_data = [
        {
            "type": "histogram",
            "x": data.tolist(),
            "name": "Data",
            "marker": {
                "color": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e", "width": 1},
            },
            "histnorm": "probability density",
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": pdf_vals.tolist(),
            "mode": "lines",
            "name": f"{best_name} Fit",
            "line": {"color": "#4a90d9", "width": 2},
        },
    ]

    layout = {"height": 300, "showlegend": True, "shapes": [], "annotations": []}
    if lsl is not None:
        layout["shapes"].append(
            {
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        layout["annotations"].append(
            {
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747"},
            }
        )
    if usl is not None:
        layout["shapes"].append(
            {
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        layout["annotations"].append(
            {
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747"},
            }
        )

    result["plots"].append(
        {
            "title": f"Non-Normal Capability ({best_name} Fit)",
            "data": hist_data,
            "layout": layout,
        }
    )

    # Probability plot for best fit
    sorted_d = np.sort(pos_data if best_name != "Normal" else data)
    n_pts = len(sorted_d)
    median_ranks = (np.arange(1, n_pts + 1) - 0.3) / (n_pts + 0.4)
    theoretical_q = best_dist.ppf(median_ranks, *best_args)

    result["plots"].append(
        {
            "title": f"Probability Plot ({best_name})",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_d.tolist(),
                    "mode": "markers",
                    "name": "Data",
                    "marker": {"color": "#4a9f6e", "size": 5},
                },
                {
                    "type": "scatter",
                    "x": [min(theoretical_q), max(theoretical_q)],
                    "y": [min(theoretical_q), max(theoretical_q)],
                    "mode": "lines",
                    "name": "Reference",
                    "line": {"color": "#d94a4a", "dash": "dash"},
                },
            ],
            "layout": {
                "height": 280,
                "xaxis": {"title": f"Theoretical ({best_name})"},
                "yaxis": {"title": "Observed"},
            },
        }
    )

    result["summary"] = summary

    return result


def run_degradation_capability(df, config):
    """Degradation-Aware Capability -- Cpk as a function of time/usage."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    meas_col = config.get("var") or config.get("measurement") or config.get("column")
    time_col = config.get("time_column") or config.get("time")
    usl = config.get("usl")
    lsl = config.get("lsl")
    window = int(config.get("window", 50))
    target_cpk = float(config.get("target_cpk", 1.33))

    if not meas_col or meas_col not in df.columns:
        result["summary"] = "Error: Specify a measurement column."
        return result
    if usl is None and lsl is None:
        result["summary"] = "Error: Specify at least one spec limit (USL or LSL)."
        return result

    usl = float(usl) if usl is not None else None
    lsl = float(lsl) if lsl is not None else None

    data = df[meas_col].dropna().values.astype(float)
    if time_col and time_col in df.columns:
        t_data = df[time_col].loc[df[meas_col].notna()].values.astype(float)
    else:
        t_data = np.arange(len(data), dtype=float)

    n = len(data)
    if n < window:
        result["summary"] = f"Error: Need at least {window} observations."
        return result

    # Rolling Cpk
    cpk_values = []
    t_centers = []
    for i in range(0, n - window + 1, max(1, window // 4)):
        seg = data[i : i + window]
        mu = np.mean(seg)
        sigma = np.std(seg, ddof=1)
        if sigma < 1e-12:
            sigma = 1e-12
        if usl is not None and lsl is not None:
            cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
        elif usl is not None:
            cpk = (usl - mu) / (3 * sigma)
        else:
            cpk = (mu - lsl) / (3 * sigma)
        cpk_values.append(float(cpk))
        t_centers.append(float(np.mean(t_data[i : i + window])))

    cpk_arr = np.array(cpk_values)
    t_arr = np.array(t_centers)

    # Fit linear trend to Cpk vs time
    slope, intercept, r_val, p_val, se = sp_stats.linregress(t_arr, cpk_arr)
    cpk_trend = intercept + slope * t_arr

    # Find crossover point where Cpk drops below target
    crossover = None
    if slope < 0 and intercept > target_cpk:
        crossover = (target_cpk - intercept) / slope
        if crossover < t_arr[0] or crossover > t_arr[-1] * 3:
            crossover = None

    cpk_start = float(cpk_trend[0])
    cpk_end = float(cpk_trend[-1])
    cpk_overall = float(np.mean(cpk_arr))
    degrading = slope < 0 and p_val < 0.05

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>DEGRADATION-AWARE CAPABILITY<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Variable:<</COLOR>> {meas_col}\n"
    summary += "<<COLOR:text>>Spec limits:<</COLOR>> "
    if lsl is not None:
        summary += f"LSL={lsl}"
    if usl is not None:
        summary += f"{', ' if lsl is not None else ''}USL={usl}"
    summary += f"\n<<COLOR:text>>Window:<</COLOR>> {window}    Points: {len(cpk_values)}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Cpk Trend \u2500\u2500<</COLOR>>\n"
    summary += f"  Starting Cpk: {cpk_start:.3f}\n"
    summary += f"  Ending Cpk:   {cpk_end:.3f}\n"
    summary += f"  Overall mean: {cpk_overall:.3f}\n"
    summary += f"  Slope: {slope:.6f} per unit time (p={p_val:.4f})\n"
    if degrading:
        summary += "  <<COLOR:warning>>Significant degradation detected<</COLOR>>\n"
    if crossover is not None:
        summary += f"  Cpk reaches {target_cpk} at t = {crossover:.1f}\n"

    result["summary"] = summary
    result["statistics"] = {
        "cpk_start": cpk_start,
        "cpk_end": cpk_end,
        "cpk_overall": cpk_overall,
        "slope": float(slope),
        "p_value": float(p_val),
        "r_squared": float(r_val**2),
        "crossover_time": crossover,
        "degrading": degrading,
    }

    _dc_status = "degrading" if degrading else "stable"
    result["guide_observation"] = (
        f"Degradation Cpk: {_dc_status}. Start={cpk_start:.3f}, end={cpk_end:.3f}, slope={slope:.6f}/unit."
    )

    _dc_cross = ""
    if crossover is not None:
        _dc_cross = f" At this rate, Cpk drops below {target_cpk} at t = {crossover:.0f}."
    result["narrative"] = _narrative(
        f"Degradation Capability \u2014 Cpk is {'degrading' if degrading else 'stable'}",
        f"Rolling Cpk starts at {cpk_start:.3f} and {'decays to' if degrading else 'remains near'} {cpk_end:.3f} "
        f"(slope = {slope:.6f}/unit, p = {p_val:.4f}).{_dc_cross}"
        + (
            f" Classical Cpk assumes stationarity and would report {cpk_overall:.3f} \u2014 masking the degradation."
            if degrading
            else ""
        ),
        next_steps="If degrading, schedule tool changes or maintenance before Cpk crosses the target. "
        + (f"Recommended action point: t = {crossover:.0f} to maintain Cpk > {target_cpk}." if crossover else ""),
        chart_guidance="The rolling Cpk curve shows capability over time. The trend line reveals whether the process is degrading. "
        "The dashed red line is the target Cpk.",
    )

    # Plot: Cpk degradation curve
    result["plots"].append(
        {
            "title": f"Capability Degradation ({meas_col})",
            "data": [
                {
                    "type": "scatter",
                    "x": t_arr.tolist(),
                    "y": cpk_arr.tolist(),
                    "mode": "markers",
                    "marker": {
                        "size": 5,
                        "color": ["#dc5050" if c < target_cpk else "#4a9f6e" for c in cpk_arr],
                    },
                    "name": "Rolling Cpk",
                },
                {
                    "type": "scatter",
                    "x": t_arr.tolist(),
                    "y": cpk_trend.tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a90d9", "width": 2, "dash": "solid"},
                    "name": "Trend",
                },
            ],
            "layout": {
                "height": 290,
                "xaxis": {
                    "title": "Time / Sequence",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": "Cpk"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": float(t_arr[0]),
                        "x1": float(t_arr[-1]),
                        "y0": target_cpk,
                        "y1": target_cpk,
                        "line": {"color": "#dc5050", "dash": "dash", "width": 1},
                    },
                ],
            },
        }
    )

    return result


def run_between_within(df, config):
    """Between/Within Capability - Nested variance components analysis."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    if measurement and measurement in df.columns:
        data = df[measurement].dropna().values
    else:
        num_cols = df.select_dtypes(include="number").columns
        measurement = num_cols[0] if len(num_cols) > 0 else df.columns[0]
        data = df[measurement].dropna().values

    subgroup_col = config.get("subgroup")
    subgroup_size = int(config.get("subgroup_size", 5))
    lsl = float(config.get("lsl")) if config.get("lsl") else None
    usl = float(config.get("usl")) if config.get("usl") else None

    if subgroup_col:
        groups = df.groupby(subgroup_col)[measurement].apply(list).values
    else:
        groups = [data[i : i + subgroup_size] for i in range(0, len(data), subgroup_size)]
        groups = [g for g in groups if len(g) == subgroup_size]

    groups = [np.array(g) for g in groups if len(g) >= 2]
    k = len(groups)

    # Within-subgroup variance (pooled)
    within_vars = [np.var(g, ddof=1) for g in groups]
    sigma_within = np.sqrt(np.mean(within_vars))

    # Between-subgroup variance
    group_means = np.array([np.mean(g) for g in groups])
    grand_mean = np.mean(data)
    n_avg = np.mean([len(g) for g in groups])

    sigma_between_sq = np.var(group_means, ddof=1) - sigma_within**2 / n_avg
    sigma_between = np.sqrt(max(0, sigma_between_sq))

    # Total (overall)
    sigma_total = np.std(data, ddof=1)

    # Between/Within combined
    sigma_bw = np.sqrt(sigma_between**2 + sigma_within**2)

    summary = f"Between/Within Capability Analysis\n\nSubgroups: {k}\n\nVariance Components:\n  \u03c3 Within: {sigma_within:.4f}\n  \u03c3 Between: {sigma_between:.4f}\n  \u03c3 B/W: {sigma_bw:.4f}\n  \u03c3 Overall: {sigma_total:.4f}\n\n% of Total Variance:\n  Within: {(sigma_within**2 / sigma_total**2 * 100):.1f}%\n  Between: {(sigma_between**2 / sigma_total**2 * 100):.1f}%\n"

    if lsl is not None and usl is not None:
        # Within capability
        cp_within = (usl - lsl) / (6 * sigma_within)
        cpk_within = min(
            (usl - grand_mean) / (3 * sigma_within),
            (grand_mean - lsl) / (3 * sigma_within),
        )

        # B/W capability
        cp_bw = (usl - lsl) / (6 * sigma_bw)
        cpk_bw = min((usl - grand_mean) / (3 * sigma_bw), (grand_mean - lsl) / (3 * sigma_bw))

        # Overall capability
        pp = (usl - lsl) / (6 * sigma_total)
        ppk = min(
            (usl - grand_mean) / (3 * sigma_total),
            (grand_mean - lsl) / (3 * sigma_total),
        )

        summary += f"\nWithin Capability:\n  Cp: {cp_within:.3f}\n  Cpk: {cpk_within:.3f}\n\nBetween/Within Capability:\n  Cp (B/W): {cp_bw:.3f}\n  Cpk (B/W): {cpk_bw:.3f}\n\nOverall Capability:\n  Pp: {pp:.3f}\n  Ppk: {ppk:.3f}"

    # ---- Plot 1: Xbar Chart (subgroup means with control limits) ----
    x_idx = list(range(1, k + 1))
    x_labels = [str(i) for i in x_idx]
    if subgroup_col:
        sg_keys = list(df.groupby(subgroup_col).groups.keys())
        x_labels = [str(s) for s in sg_keys[:k]]

    # Control limits for Xbar using sigma_between + sigma_within
    ucl_xbar = grand_mean + 3 * sigma_bw / np.sqrt(n_avg)
    lcl_xbar = grand_mean - 3 * sigma_bw / np.sqrt(n_avg)

    # Flag out-of-control points
    ooc_xbar = [i for i, m in enumerate(group_means) if m > ucl_xbar or m < lcl_xbar]

    xbar_traces = [
        {
            "type": "scatter",
            "x": x_idx,
            "y": group_means.tolist(),
            "mode": "lines+markers",
            "name": "Subgroup Mean",
            "marker": {"color": "#4a90d9", "size": 6},
            "line": {"color": "#4a90d9", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [grand_mean] * k,
            "mode": "lines",
            "name": f"X\u0304 = {grand_mean:.4f}",
            "line": {"color": "#4a9f6e", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [ucl_xbar] * k,
            "mode": "lines",
            "name": f"UCL = {ucl_xbar:.4f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [lcl_xbar] * k,
            "mode": "lines",
            "name": f"LCL = {lcl_xbar:.4f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
        },
    ]
    if ooc_xbar:
        xbar_traces.append(
            {
                "type": "scatter",
                "x": [x_idx[i] for i in ooc_xbar],
                "y": [group_means[i] for i in ooc_xbar],
                "mode": "markers",
                "name": "Out of Control",
                "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"},
            }
        )

    result["plots"].append(
        {
            "title": "X\u0304 Chart \u2014 Subgroup Means",
            "data": xbar_traces,
            "layout": {
                "height": 320,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": measurement},
                "showlegend": True,
                "legend": {
                    "orientation": "h",
                    "y": 1.15,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 9, "color": "#b0b0b0"},
                    "bgcolor": "rgba(0,0,0,0)",
                },
            },
            "group": "Control Charts",
        }
    )

    # ---- Plot 2: R Chart (within-subgroup ranges) ----
    group_ranges = np.array([np.max(g) - np.min(g) for g in groups])
    r_bar = np.mean(group_ranges)
    # d3/D3/D4 constants for subgroup size (approximation for variable n)
    n_int = int(round(n_avg))
    d2_table = {
        2: 1.128,
        3: 1.693,
        4: 2.059,
        5: 2.326,
        6: 2.534,
        7: 2.704,
        8: 2.847,
        9: 2.970,
        10: 3.078,
    }
    d3_table = {
        2: 0.853,
        3: 0.888,
        4: 0.880,
        5: 0.864,
        6: 0.848,
        7: 0.833,
        8: 0.820,
        9: 0.808,
        10: 0.797,
    }
    d2 = d2_table.get(n_int, 2.326)
    d3 = d3_table.get(n_int, 0.864)
    D3 = max(0, 1 - 3 * d3 / d2)
    D4 = 1 + 3 * d3 / d2
    ucl_r = D4 * r_bar
    lcl_r = D3 * r_bar

    ooc_r = [i for i, r in enumerate(group_ranges) if r > ucl_r or r < lcl_r]

    r_traces = [
        {
            "type": "scatter",
            "x": x_idx,
            "y": group_ranges.tolist(),
            "mode": "lines+markers",
            "name": "Range",
            "marker": {"color": "#e89547", "size": 6},
            "line": {"color": "#e89547", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [r_bar] * k,
            "mode": "lines",
            "name": f"R\u0304 = {r_bar:.4f}",
            "line": {"color": "#4a9f6e", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [ucl_r] * k,
            "mode": "lines",
            "name": f"UCL = {ucl_r:.4f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
        },
        {
            "type": "scatter",
            "x": x_idx,
            "y": [lcl_r] * k,
            "mode": "lines",
            "name": f"LCL = {lcl_r:.4f}",
            "line": {"color": "#d94a4a", "dash": "dash", "width": 1.5},
        },
    ]
    if ooc_r:
        r_traces.append(
            {
                "type": "scatter",
                "x": [x_idx[i] for i in ooc_r],
                "y": [group_ranges[i] for i in ooc_r],
                "mode": "markers",
                "name": "Out of Control",
                "marker": {"color": "#e85747", "size": 10, "symbol": "diamond"},
            }
        )

    result["plots"].append(
        {
            "title": "R Chart \u2014 Within-Subgroup Ranges",
            "data": r_traces,
            "layout": {
                "height": 320,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": "Range"},
                "showlegend": True,
                "legend": {
                    "orientation": "h",
                    "y": 1.15,
                    "x": 0.5,
                    "xanchor": "center",
                    "font": {"size": 9, "color": "#b0b0b0"},
                    "bgcolor": "rgba(0,0,0,0)",
                },
            },
            "group": "Control Charts",
        }
    )

    # ---- Plot 3: Individual Values by Subgroup (box + strip) ----
    box_traces = []
    for i, g in enumerate(groups):
        label = x_labels[i] if i < len(x_labels) else str(i + 1)
        box_traces.append(
            {
                "type": "box",
                "y": g.tolist(),
                "name": label,
                "boxpoints": "all",
                "jitter": 0.4,
                "pointpos": 0,
                "marker": {"color": "#4a90d9", "size": 3, "opacity": 0.6},
                "line": {"color": "#4a90d9", "width": 1},
                "fillcolor": "rgba(74, 144, 217, 0.15)",
            }
        )

    box_layout = {
        "height": 300,
        "xaxis": {"title": "Subgroup"},
        "yaxis": {"title": measurement},
        "showlegend": False,
        "shapes": [],
        "annotations": [],
    }
    # Grand mean reference line
    box_layout["shapes"].append(
        {
            "type": "line",
            "x0": -0.5,
            "x1": k - 0.5,
            "y0": grand_mean,
            "y1": grand_mean,
            "line": {"color": "#4a9f6e", "width": 1.5, "dash": "dash"},
        }
    )
    box_layout["annotations"].append(
        {
            "x": k - 0.5,
            "y": grand_mean,
            "text": f"X\u0304={grand_mean:.3f}",
            "showarrow": False,
            "xanchor": "left",
            "font": {"color": "#4a9f6e", "size": 10},
        }
    )
    if lsl is not None:
        box_layout["shapes"].append(
            {
                "type": "line",
                "x0": -0.5,
                "x1": k - 0.5,
                "y0": lsl,
                "y1": lsl,
                "line": {"color": "#e85747", "dash": "dot", "width": 1.5},
            }
        )
        box_layout["annotations"].append(
            {
                "x": k - 0.5,
                "y": lsl,
                "text": "LSL",
                "showarrow": False,
                "xanchor": "left",
                "font": {"color": "#e85747", "size": 10},
            }
        )
    if usl is not None:
        box_layout["shapes"].append(
            {
                "type": "line",
                "x0": -0.5,
                "x1": k - 0.5,
                "y0": usl,
                "y1": usl,
                "line": {"color": "#e85747", "dash": "dot", "width": 1.5},
            }
        )
        box_layout["annotations"].append(
            {
                "x": k - 0.5,
                "y": usl,
                "text": "USL",
                "showarrow": False,
                "xanchor": "left",
                "font": {"color": "#e85747", "size": 10},
            }
        )

    # Cap at 30 subgroups for readability; summarize if more
    if k <= 30:
        result["plots"].append(
            {
                "title": "Individual Values by Subgroup",
                "data": box_traces,
                "layout": box_layout,
                "group": "Control Charts",
            }
        )
    else:
        # Show first 15 and last 15 for large datasets
        subset = box_traces[:15] + box_traces[-15:]
        box_layout["annotations"].insert(
            0,
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": f"Showing 30 of {k} subgroups (first 15 + last 15)",
                "showarrow": False,
                "font": {"color": "rgba(255,255,255,0.5)", "size": 10},
            },
        )
        result["plots"].append(
            {
                "title": "Individual Values by Subgroup",
                "data": subset,
                "layout": box_layout,
                "group": "Control Charts",
            }
        )

    # ---- Plot 4: Variance Components bar chart ----
    result["plots"].append(
        {
            "title": "Variance Components (\u03c3)",
            "data": [
                {
                    "type": "bar",
                    "x": ["Within", "Between", "B/W Combined", "Overall"],
                    "y": [sigma_within, sigma_between, sigma_bw, sigma_total],
                    "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a"]},
                    "text": [
                        f"{sigma_within:.4f}",
                        f"{sigma_between:.4f}",
                        f"{sigma_bw:.4f}",
                        f"{sigma_total:.4f}",
                    ],
                    "textposition": "outside",
                    "textfont": {"color": "#b0b0b0"},
                }
            ],
            "layout": {"height": 280, "yaxis": {"title": "Std Dev (\u03c3)"}},
            "group": "Variance",
        }
    )

    # ---- Plot 5: % Variance Contribution (donut) ----
    within_pct = sigma_within**2 / sigma_total**2 * 100 if sigma_total > 0 else 0
    between_pct = sigma_between**2 / sigma_total**2 * 100 if sigma_total > 0 else 0
    residual_pct = max(0, 100 - within_pct - between_pct)

    donut_labels = ["Within", "Between"]
    donut_vals = [within_pct, between_pct]
    donut_colors = ["#4a9f6e", "#4a90d9"]
    if residual_pct > 0.5:
        donut_labels.append("Residual")
        donut_vals.append(residual_pct)
        donut_colors.append("rgba(255,255,255,0.15)")

    result["plots"].append(
        {
            "title": "% Contribution to Total Variance",
            "data": [
                {
                    "type": "pie",
                    "labels": donut_labels,
                    "values": donut_vals,
                    "hole": 0.45,
                    "marker": {
                        "colors": donut_colors,
                        "line": {"color": "rgba(0,0,0,0.3)", "width": 1},
                    },
                    "textinfo": "label+percent",
                    "textfont": {"size": 12, "color": "#e0e0e0"},
                    "hoverinfo": "label+percent+value",
                }
            ],
            "layout": {
                "height": 280,
                "showlegend": False,
                "annotations": [
                    {
                        "text": "Variance<br>Split",
                        "x": 0.5,
                        "y": 0.5,
                        "font": {"size": 13, "color": "rgba(255,255,255,0.6)"},
                        "showarrow": False,
                    }
                ],
            },
            "group": "Variance",
        }
    )

    # ---- Plot 6: Within vs Overall Distribution (histogram + fits) ----
    x_range = np.linspace(min(data), max(data), 200)
    hist_data = [
        {
            "type": "histogram",
            "x": data.tolist(),
            "name": "Data",
            "marker": {
                "color": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e", "width": 1},
            },
            "histnorm": "probability density",
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_within).tolist(),
            "mode": "lines",
            "name": f"Within (\u03c3={sigma_within:.3f})",
            "line": {"color": "#4a90d9", "width": 2},
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_bw).tolist(),
            "mode": "lines",
            "name": f"B/W (\u03c3={sigma_bw:.3f})",
            "line": {"color": "#e89547", "width": 2, "dash": "dot"},
        },
        {
            "type": "scatter",
            "x": x_range.tolist(),
            "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_total).tolist(),
            "mode": "lines",
            "name": f"Overall (\u03c3={sigma_total:.3f})",
            "line": {"color": "#d94a4a", "width": 2, "dash": "dash"},
        },
    ]

    dist_layout = {
        "height": 300,
        "showlegend": True,
        "shapes": [],
        "annotations": [],
        "legend": {
            "font": {"size": 9, "color": "#b0b0b0"},
            "x": 0.98,
            "xanchor": "right",
            "y": 0.98,
            "bgcolor": "rgba(20,20,30,0.7)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "borderwidth": 1,
        },
    }
    if lsl is not None:
        dist_layout["shapes"].append(
            {
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        dist_layout["annotations"].append(
            {
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747"},
            }
        )
    if usl is not None:
        dist_layout["shapes"].append(
            {
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
            }
        )
        dist_layout["annotations"].append(
            {
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747"},
            }
        )

    result["plots"].append(
        {
            "title": "Within vs B/W vs Overall Distribution",
            "data": hist_data,
            "layout": dist_layout,
            "group": "Capability",
        }
    )

    # ---- Plot 7: Capability Index Comparison (when specs provided) ----
    if lsl is not None and usl is not None:
        cap_categories = ["Cp / Pp", "Cpk / Ppk"]
        result["plots"].append(
            {
                "title": "Capability Index Comparison",
                "data": [
                    {
                        "type": "bar",
                        "name": "Within",
                        "x": cap_categories,
                        "y": [cp_within, cpk_within],
                        "marker": {"color": "#4a9f6e"},
                        "text": [f"{cp_within:.3f}", f"{cpk_within:.3f}"],
                        "textposition": "outside",
                        "textfont": {"color": "#b0b0b0"},
                    },
                    {
                        "type": "bar",
                        "name": "Between/Within",
                        "x": cap_categories,
                        "y": [cp_bw, cpk_bw],
                        "marker": {"color": "#e89547"},
                        "text": [f"{cp_bw:.3f}", f"{cpk_bw:.3f}"],
                        "textposition": "outside",
                        "textfont": {"color": "#b0b0b0"},
                    },
                    {
                        "type": "bar",
                        "name": "Overall",
                        "x": cap_categories,
                        "y": [pp, ppk],
                        "marker": {"color": "#d94a4a"},
                        "text": [f"{pp:.3f}", f"{ppk:.3f}"],
                        "textposition": "outside",
                        "textfont": {"color": "#b0b0b0"},
                    },
                ],
                "layout": {
                    "height": 300,
                    "barmode": "group",
                    "yaxis": {"title": "Index Value"},
                    "legend": {
                        "orientation": "h",
                        "y": 1.12,
                        "x": 0.5,
                        "xanchor": "center",
                        "font": {"size": 10, "color": "#b0b0b0"},
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                    "shapes": [
                        {
                            "type": "line",
                            "x0": -0.5,
                            "x1": 1.5,
                            "y0": 1.33,
                            "y1": 1.33,
                            "line": {
                                "color": "rgba(74,159,110,0.5)",
                                "dash": "dash",
                                "width": 1.5,
                            },
                        },
                        {
                            "type": "line",
                            "x0": -0.5,
                            "x1": 1.5,
                            "y0": 1.0,
                            "y1": 1.0,
                            "line": {
                                "color": "rgba(232,87,71,0.5)",
                                "dash": "dot",
                                "width": 1.5,
                            },
                        },
                    ],
                    "annotations": [
                        {
                            "x": 1.5,
                            "y": 1.33,
                            "text": "Target (1.33)",
                            "showarrow": False,
                            "xanchor": "left",
                            "font": {"color": "rgba(74,159,110,0.7)", "size": 10},
                        },
                        {
                            "x": 1.5,
                            "y": 1.0,
                            "text": "Minimum (1.0)",
                            "showarrow": False,
                            "xanchor": "left",
                            "font": {"color": "rgba(232,87,71,0.7)", "size": 10},
                        },
                    ],
                },
                "group": "Capability",
            }
        )

    # ---- Plot 8: Normal Probability Plot ----
    sorted_data = np.sort(data)
    n_pts = len(sorted_data)
    theoretical_q = sp_stats.norm.ppf((np.arange(1, n_pts + 1) - 0.375) / (n_pts + 0.25))

    result["plots"].append(
        {
            "title": "Normal Probability Plot",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "name": "Data",
                    "marker": {"color": "#4a90d9", "size": 3, "opacity": 0.7},
                },
                {
                    "type": "scatter",
                    "x": [theoretical_q[0], theoretical_q[-1]],
                    "y": [
                        grand_mean + sigma_total * theoretical_q[0],
                        grand_mean + sigma_total * theoretical_q[-1],
                    ],
                    "mode": "lines",
                    "name": "Normal Fit",
                    "line": {"color": "#d94a4a", "width": 1.5},
                },
            ],
            "layout": {
                "height": 280,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": measurement},
                "showlegend": True,
                "legend": {
                    "font": {"size": 9, "color": "#b0b0b0"},
                    "x": 0.02,
                    "y": 0.98,
                    "bgcolor": "rgba(20,20,30,0.7)",
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "borderwidth": 1,
                },
            },
            "group": "Capability",
        }
    )

    result["summary"] = summary

    return result
