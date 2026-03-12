"""D-NonNorm — KDE-based non-normal capability analysis.

CR: 3c0d0e53
"""

import logging

import numpy as np

from ..common import (
    COLOR_BAD,
    COLOR_GOLD,
    SVEND_COLORS,
    _fit_best_distribution,
    _rgba,
)
from .helpers import _d_narrative, _kde_density

logger = logging.getLogger(__name__)


def run_d_nonnorm(df, config):
    """Non-normal capability analysis using KDE density estimation.

    Computes Pp/Ppk equivalents directly from the empirical KDE density,
    compares against normal-assumption values, and quantifies the normality
    penalty.
    """
    from scipy.stats import norm as sp_norm

    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = config.get("variable") or config.get("measurement")
    if not variable or variable not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid measurement variable.<</COLOR>>"
        return result

    lsl = float(config["lsl"]) if config.get("lsl") is not None else None
    usl = float(config["usl"]) if config.get("usl") is not None else None
    if lsl is None and usl is None:
        result["summary"] = "<<COLOR:danger>>At least one spec limit (LSL or USL) is required.<</COLOR>>"
        return result

    data = df[variable].dropna().astype(float).values
    data = data[np.isfinite(data)]
    if len(data) < 10:
        result["summary"] = "<<COLOR:danger>>Need at least 10 observations.<</COLOR>>"
        return result

    # Spec limit validation
    data_lo, data_hi = float(data.min()), float(data.max())
    spec_lo = lsl if lsl is not None else data_lo
    spec_hi = usl if usl is not None else data_hi
    if spec_hi < data_lo or spec_lo > data_hi:
        result["summary"] = (
            f"<<COLOR:danger>>Spec limits [{spec_lo}, {spec_hi}] do not overlap with data range "
            f"[{data_lo:.4f}, {data_hi:.4f}]. Check your spec limits.<</COLOR>>"
        )
        return result

    n = len(data)
    mu, sigma = float(np.mean(data)), float(np.std(data, ddof=1))

    # Build KDE on fine grid
    margin = (data_hi - data_lo) * 0.3
    grid = np.linspace(data_lo - margin, data_hi + margin, 1000)
    dx = grid[1] - grid[0]
    kde_density = _kde_density(data, grid)

    # Normalize to proper PDF
    total_area = np.trapz(kde_density, grid)
    if total_area > 0:
        kde_density = kde_density / total_area

    # KDE CDF
    kde_cdf = np.cumsum(kde_density) * dx

    # Tail probabilities from KDE
    p_below_lsl = float(np.interp(lsl, grid, kde_cdf)) if lsl is not None else 0.0
    p_above_usl = float(1.0 - np.interp(usl, grid, kde_cdf)) if usl is not None else 0.0
    p_below_lsl = max(1e-10, min(1.0 - 1e-10, p_below_lsl))
    p_above_usl = max(1e-10, min(1.0 - 1e-10, p_above_usl))

    # Z-equivalents from KDE
    z_lower_kde = float(sp_norm.ppf(1.0 - p_below_lsl)) if lsl is not None else 999.0
    z_upper_kde = float(sp_norm.ppf(1.0 - p_above_usl)) if usl is not None else 999.0
    ppk_kde = min(z_lower_kde, z_upper_kde) / 3.0

    # KDE quantiles for Pp
    q_low_kde = float(np.interp(0.00135, kde_cdf, grid))
    q_high_kde = float(np.interp(0.99865, kde_cdf, grid))
    spread_kde = q_high_kde - q_low_kde
    if lsl is not None and usl is not None and spread_kde > 0:
        pp_kde = (usl - lsl) / spread_kde
    else:
        pp_kde = None

    # Normal-assumption Cpk
    cpk_lower = (mu - lsl) / (3 * sigma) if lsl is not None and sigma > 0 else 999.0
    cpk_upper = (usl - mu) / (3 * sigma) if usl is not None and sigma > 0 else 999.0
    cpk_normal = min(cpk_lower, cpk_upper)
    if lsl is not None and usl is not None and sigma > 0:
        cp_normal = (usl - lsl) / (6 * sigma)
    else:
        cp_normal = None

    # PPM estimates
    ppm_kde = (p_below_lsl + p_above_usl) * 1e6
    p_norm_below = float(sp_norm.cdf(lsl, mu, sigma)) if lsl is not None else 0.0
    p_norm_above = float(1.0 - sp_norm.cdf(usl, mu, sigma)) if usl is not None else 0.0
    ppm_normal = (p_norm_below + p_norm_above) * 1e6

    # Best-fit parametric
    dist_name, dist_obj, dist_args, dist_pval = _fit_best_distribution(data)
    p_fit_below = float(dist_obj.cdf(lsl, *dist_args)) if lsl is not None else 0.0
    p_fit_above = float(1.0 - dist_obj.cdf(usl, *dist_args)) if usl is not None else 0.0
    p_fit_below = max(1e-10, min(1.0 - 1e-10, p_fit_below))
    p_fit_above = max(1e-10, min(1.0 - 1e-10, p_fit_above))
    z_fit_lower = float(sp_norm.ppf(1.0 - p_fit_below)) if lsl is not None else 999.0
    z_fit_upper = float(sp_norm.ppf(1.0 - p_fit_above)) if usl is not None else 999.0
    ppk_fit = min(z_fit_lower, z_fit_upper) / 3.0
    ppm_fit = (p_fit_below + p_fit_above) * 1e6

    # Normality penalty
    penalty = cpk_normal - ppk_kde

    # --- Plot 1: KDE density with spec limits and tail shading ---
    kde_trace = {
        "type": "scatter",
        "x": grid.tolist(),
        "y": kde_density.tolist(),
        "mode": "lines",
        "name": "KDE Density",
        "line": {"color": SVEND_COLORS[0], "width": 2.5},
        "fill": "tozeroy",
        "fillcolor": _rgba(SVEND_COLORS[0], 0.15),
    }
    normal_pdf = sp_norm.pdf(grid, mu, sigma)
    normal_trace = {
        "type": "scatter",
        "x": grid.tolist(),
        "y": normal_pdf.tolist(),
        "mode": "lines",
        "name": "Normal Assumption",
        "line": {"color": SVEND_COLORS[1], "width": 1.5, "dash": "dash"},
    }
    shapes_p1, annotations_p1 = [], []
    if lsl is not None:
        # Shade tail below LSL
        tail_mask = grid <= lsl
        if np.any(tail_mask):
            grid[tail_mask].tolist()
            kde_density[tail_mask].tolist()
            result["plots"].append(None)  # placeholder
        shapes_p1.append(
            {
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": COLOR_BAD, "dash": "dash", "width": 2},
            }
        )
        annotations_p1.append(
            {
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": f"LSL={lsl}",
                "showarrow": False,
                "font": {"color": COLOR_BAD, "size": 10},
            }
        )
    if usl is not None:
        shapes_p1.append(
            {
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": COLOR_BAD, "dash": "dash", "width": 2},
            }
        )
        annotations_p1.append(
            {
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": f"USL={usl}",
                "showarrow": False,
                "font": {"color": COLOR_BAD, "size": 10},
            }
        )

    plot1_traces = [kde_trace, normal_trace]

    # Add tail fill traces
    if lsl is not None:
        tail_mask = grid <= lsl
        if np.any(tail_mask):
            plot1_traces.append(
                {
                    "type": "scatter",
                    "x": grid[tail_mask].tolist(),
                    "y": kde_density[tail_mask].tolist(),
                    "mode": "lines",
                    "fill": "tozeroy",
                    "fillcolor": _rgba(COLOR_BAD, 0.3),
                    "line": {"color": "rgba(0,0,0,0)"},
                    "name": f"Below LSL ({p_below_lsl * 1e6:.0f} PPM)",
                    "showlegend": True,
                }
            )
    if usl is not None:
        tail_mask = grid >= usl
        if np.any(tail_mask):
            plot1_traces.append(
                {
                    "type": "scatter",
                    "x": grid[tail_mask].tolist(),
                    "y": kde_density[tail_mask].tolist(),
                    "mode": "lines",
                    "fill": "tozeroy",
                    "fillcolor": _rgba(COLOR_BAD, 0.3),
                    "line": {"color": "rgba(0,0,0,0)"},
                    "name": f"Above USL ({p_above_usl * 1e6:.0f} PPM)",
                    "showlegend": True,
                }
            )

    # Remove the placeholder
    result["plots"] = []
    result["plots"].append(
        {
            "title": "KDE Capability Density",
            "data": plot1_traces,
            "layout": {
                "height": 340,
                "shapes": shapes_p1,
                "annotations": annotations_p1,
                "xaxis": {"title": variable},
                "yaxis": {"title": "Density"},
                "showlegend": True,
            },
        }
    )

    # --- Plot 2: Method comparison bar chart ---
    methods = ["KDE", "Normal", dist_name.title()]
    ppk_vals = [ppk_kde, cpk_normal, ppk_fit]
    bar_colors = [SVEND_COLORS[0], SVEND_COLORS[1], SVEND_COLORS[2]]

    result["plots"].append(
        {
            "title": "Capability Method Comparison",
            "data": [
                {
                    "type": "bar",
                    "x": methods,
                    "y": ppk_vals,
                    "name": "Ppk Equivalent",
                    "marker": {"color": bar_colors},
                    "text": [f"{v:.3f}" for v in ppk_vals],
                    "textposition": "outside",
                },
            ],
            "layout": {
                "height": 300,
                "yaxis": {"title": "Ppk"},
                "showlegend": False,
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": 2.5,
                        "y0": 1.33,
                        "y1": 1.33,
                        "line": {"color": COLOR_GOLD, "dash": "dash", "width": 1.5},
                    }
                ],
                "annotations": [
                    {
                        "x": 2.5,
                        "y": 1.33,
                        "text": "Target 1.33",
                        "showarrow": False,
                        "font": {"color": COLOR_GOLD, "size": 10},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 3: QQ plot vs normal ---
    sorted_data = np.sort(data)
    theoretical_q = sp_norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    result["plots"].append(
        {
            "title": "Normal Q-Q Plot",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "name": "Data",
                    "marker": {"color": SVEND_COLORS[0], "size": 4, "opacity": 0.6},
                },
                {
                    "type": "scatter",
                    "x": [theoretical_q[0], theoretical_q[-1]],
                    "y": [mu + sigma * theoretical_q[0], mu + sigma * theoretical_q[-1]],
                    "mode": "lines",
                    "name": "Normal Reference",
                    "line": {"color": COLOR_BAD, "dash": "dash", "width": 1.5},
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": variable},
                "showlegend": True,
            },
        }
    )

    # --- Summary ---
    summary = "<<COLOR:title>>D-NONNORM — KDE-BASED CAPABILITY<</COLOR>>\n\n"
    summary += f"<<COLOR:header>>Data:<</COLOR>> {n} observations of '{variable}'\n"
    summary += f"  Mean: {mu:.4f}  |  Std Dev: {sigma:.4f}\n"
    if lsl is not None:
        summary += f"  LSL: {lsl}\n"
    if usl is not None:
        summary += f"  USL: {usl}\n"
    summary += "\n<<COLOR:header>>Capability Comparison:<</COLOR>>\n"
    summary += f"  {'Method':<18} {'Ppk':>8} {'PPM':>10}\n"
    summary += f"  {'-' * 38}\n"
    summary += f"  {'KDE (empirical)':<18} {ppk_kde:>8.3f} {ppm_kde:>10,.0f}\n"
    summary += f"  {'Normal assumption':<18} {cpk_normal:>8.3f} {ppm_normal:>10,.0f}\n"
    summary += f"  {dist_name.title() + ' fit':<18} {ppk_fit:>8.3f} {ppm_fit:>10,.0f}\n"
    if pp_kde is not None:
        summary += f"\n  Pp (KDE): {pp_kde:.3f}"
        if cp_normal is not None:
            summary += f"  |  Cp (Normal): {cp_normal:.3f}"
        summary += "\n"

    summary += "\n<<COLOR:header>>Normality Penalty:<</COLOR>> "
    if abs(penalty) < 0.01:
        summary += f"<<COLOR:success>>Negligible ({penalty:+.3f})<</COLOR>> — normal assumption is adequate.\n"
    elif penalty > 0:
        summary += f"<<COLOR:warning>>Normal overestimates capability by {penalty:.3f}<</COLOR>>\n"
        summary += f"  The normal assumption gives {penalty:.3f} higher Ppk than the actual KDE density.\n"
        summary += f"  Use KDE-based Ppk ({ppk_kde:.3f}) for decisions.\n"
    else:
        summary += f"<<COLOR:success>>Normal underestimates capability by {abs(penalty):.3f}<</COLOR>>\n"
        summary += f"  KDE shows the process is actually {abs(penalty):.3f} better than normal assumes.\n"

    summary += f"\n<<COLOR:header>>Best-Fit Distribution:<</COLOR>> {dist_name.title()} (KS p={dist_pval:.4f})\n"

    if ppk_kde >= 1.33:
        summary += f"\n<<COLOR:success>>Process is capable (Ppk = {ppk_kde:.3f} ≥ 1.33).<</COLOR>>"
    elif ppk_kde >= 1.0:
        summary += f"\n<<COLOR:warning>>Process is marginally capable (Ppk = {ppk_kde:.3f}).<</COLOR>>"
    else:
        summary += f"\n<<COLOR:danger>>Process is NOT capable (Ppk = {ppk_kde:.3f} < 1.0).<</COLOR>>"

    result["summary"] = summary
    result["guide_observation"] = (
        f"D-NonNorm: Ppk(KDE)={ppk_kde:.3f}, Ppk(Normal)={cpk_normal:.3f}, "
        f"normality penalty={penalty:+.3f}, PPM(KDE)={ppm_kde:.0f}"
    )
    result["statistics"] = {
        "ppk_kde": round(ppk_kde, 4),
        "ppk_normal": round(cpk_normal, 4),
        "ppk_fit": round(ppk_fit, 4),
        "normality_penalty": round(penalty, 4),
        "ppm_kde": round(ppm_kde, 1),
        "ppm_normal": round(ppm_normal, 1),
        "best_fit": dist_name,
        "n": n,
    }

    # --- narrative ---
    if ppk_kde >= 1.33:
        verdict = f"Capable — Ppk(KDE) = {ppk_kde:.3f}"
        body = (
            f"Using kernel density estimation (no normality assumption), the process "
            f"achieves Ppk = <strong>{ppk_kde:.3f}</strong>, comfortably above the 1.33 threshold. "
            f"Estimated defect rate: {ppm_kde:.0f} PPM."
        )
    elif ppk_kde >= 1.0:
        verdict = f"Marginally Capable — Ppk(KDE) = {ppk_kde:.3f}"
        body = (
            f"KDE-based capability is <strong>{ppk_kde:.3f}</strong> — above 1.0 but below the "
            f"1.33 target. Estimated defect rate: {ppm_kde:.0f} PPM."
        )
    else:
        verdict = f"Not Capable — Ppk(KDE) = {ppk_kde:.3f}"
        body = (
            f"KDE-based capability is <strong>{ppk_kde:.3f}</strong>, below 1.0. "
            f"Estimated defect rate: {ppm_kde:.0f} PPM — improvement needed."
        )
    if abs(penalty) > 0.05:
        direction = "overstates" if penalty > 0 else "understates"
        body += (
            f" The normality assumption {direction} capability by "
            f"<strong>{abs(penalty):.3f}</strong> (Normal Ppk = {cpk_normal:.3f})."
        )
    body += f" Best-fit distribution: {dist_name.title()}."
    result["narrative"] = _d_narrative(
        f"D-NonNorm: {verdict}",
        body,
        (
            "Process is capable — monitor for drift."
            if ppk_kde >= 1.33
            else "Investigate sources of variation to improve capability."
            if ppk_kde >= 1.0
            else "Prioritise variation reduction; consider the D-Chart to identify contributing factors."
        ),
        "The top plot overlays KDE (actual shape) vs normal assumption. "
        "The middle plot shows the best-fit parametric distribution. "
        "Shaded tail areas show where defects occur under each model.",
    )

    result["education"] = {
        "title": "Understanding Non-Normal Capability (D-NonNorm)",
        "content": (
            "<dl>"
            "<dt>Why non-normal capability?</dt>"
            "<dd>Traditional Cpk/Ppk assumes data follows a normal (bell-curve) distribution. "
            "Many real processes are skewed, heavy-tailed, or bounded. When normality doesn't hold, "
            "normal-based Ppk can dramatically over- or under-estimate true capability.</dd>"
            "<dt>What is KDE-based Ppk?</dt>"
            "<dd>Kernel Density Estimation fits a smooth curve to the actual data shape — "
            "no distributional assumption needed. Ppk is then computed from the true tail "
            "probabilities, giving an honest capability estimate.</dd>"
            "<dt>What is the Normality Penalty?</dt>"
            "<dd>The difference between normal-assumption Ppk and KDE-based Ppk. "
            "A positive penalty means normal <em>overstates</em> capability (the real "
            "tails are heavier). A negative penalty means normal <em>understates</em> it "
            "(the real distribution is tighter than a bell curve).</dd>"
            "<dt>How to interpret</dt>"
            "<dd><strong>Ppk(KDE) ≥ 1.33</strong>: Capable regardless of distribution shape. "
            "<strong>Large normality penalty (> 0.1)</strong>: The normal assumption is misleading — "
            "always use the KDE value. <strong>PPM comparison</strong>: Compare KDE vs Normal PPM "
            "to see the real-world defect rate difference.</dd>"
            "</dl>"
        ),
    }

    return result
