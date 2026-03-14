"""D-Cpk — factor-attributed capability divergence.

CR: 3c0d0e53
"""

import logging

from ..common import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_REFERENCE,
    SVEND_COLORS,
    _rgba,
)
from .helpers import (
    _build_grid,
    _compute_cpk,
    _d_cpk_body,
    _d_cpk_nextsteps,
    _d_narrative,
    _decompose_divergence,
    _jsd,
    _jsd_tail,
    _kde_density,
    _noise_floor,
    _p_within_spec,
)

logger = logging.getLogger(__name__)


def run_d_cpk(df, config):
    """D-Cpk — factor-attributed capability divergence.

    For each factor level, computes how much the capability (P(within spec))
    diverges from the pooled capability using JSD of Bernoulli distributions.
    Includes counterfactual analysis: "Cpk if we removed this factor."

    Config:
        variable:   numeric column
        factor:     categorical column
        lsl:        lower spec limit (optional)
        usl:        upper spec limit (optional)
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = config.get("variable") or config.get("measurement")
    factor = config.get("factor")
    lsl = config.get("lsl")
    usl = config.get("usl")

    if not variable or not factor:
        result["summary"] = "<<COLOR:danger>>Error: variable and factor are required.<</COLOR>>"
        return result

    if lsl is None and usl is None:
        result["summary"] = "<<COLOR:danger>>Error: at least one spec limit (LSL or USL) is required.<</COLOR>>"
        return result

    lsl = float(lsl) if lsl is not None else None
    usl = float(usl) if usl is not None else None

    # Prepare data
    df = df.dropna(subset=[variable, factor])
    values = df[variable].astype(float).values
    factors = df[factor].astype(str).values
    unique_factors = sorted(set(factors))

    if len(unique_factors) < 2:
        result["summary"] = "<<COLOR:warning>>Need at least 2 factor levels.<</COLOR>>"
        return result

    # Sanity check: do spec limits overlap with the data?
    data_lo, data_hi = float(values.min()), float(values.max())
    spec_lo = lsl if lsl is not None else data_lo
    spec_hi = usl if usl is not None else data_hi
    if spec_hi < data_lo or spec_lo > data_hi:
        result["summary"] = (
            f"<<COLOR:danger>>Spec limits [{spec_lo}, {spec_hi}] do not overlap with the data range "
            f"[{data_lo:.2f}, {data_hi:.2f}].<</COLOR>>\n\n"
            f"<<COLOR:warning>>The entire dataset falls outside your specification window. "
            f"Please check that LSL and USL are correct for this variable.<</COLOR>>"
        )
        return result

    grid = _build_grid(values)

    # Pooled density and capability
    pooled_density = _kde_density(values, grid)
    pooled_pws = _p_within_spec(pooled_density, grid, lsl, usl)
    pooled_cpk = _compute_cpk(values, lsl, usl)

    # Noise floor — uses full density JSD (same method as D-Chart)
    noise = _noise_floor(values, len(values) // len(unique_factors), grid, B=200)

    # Per-factor analysis
    factor_results = []
    for fval in unique_factors:
        mask = factors == fval
        fdata = values[mask]
        n_f = len(fdata)

        if n_f < 5:
            factor_results.append(
                {
                    "factor": fval,
                    "n": n_f,
                    "jsd": 0.0,
                    "pws": 0.0,
                    "direction": 0,
                    "cpk": 0.0,
                    "cpk_without": 0.0,
                }
            )
            continue

        f_density = _kde_density(fdata, grid)
        f_pws = _p_within_spec(f_density, grid, lsl, usl)
        f_cpk = _compute_cpk(fdata, lsl, usl)

        # Full density JSD vs complement (consistent with D-Chart)
        # Uses complement (all data except this factor) instead of pooled
        # to avoid the "pooled contamination" problem.
        other_data = values[~mask]
        complement_density = _kde_density(other_data, grid) if len(other_data) >= 5 else pooled_density
        jsd_full = _jsd(f_density, complement_density, grid)

        # Tail-only JSD: divergence in the out-of-spec regions only
        jsd_tails = _jsd_tail(f_density, complement_density, grid, lsl, usl)

        # Defect efficiency: what fraction of total divergence produces defects
        defect_eff = jsd_tails / jsd_full if jsd_full > 0 else 0.0

        # Location vs scale decomposition
        loc_pct, scale_pct = _decompose_divergence(fdata, other_data)

        # Signed direction: positive = factor WORSENS capability
        direction = 1 if f_cpk < pooled_cpk else -1

        # Counterfactual: Cpk without this factor
        cpk_without = _compute_cpk(other_data, lsl, usl) if len(other_data) >= 5 else pooled_cpk

        # PPM defect rate (computed here so we can sort by it)
        from scipy import stats as sp_stats

        mu_f = fdata.mean()
        sigma_f = fdata.std(ddof=1)
        ppm = 0.0
        if sigma_f > 0:
            defect_rate = 0.0
            if lsl is not None:
                defect_rate += sp_stats.norm.cdf(lsl, mu_f, sigma_f)
            if usl is not None:
                defect_rate += 1 - sp_stats.norm.cdf(usl, mu_f, sigma_f)
            ppm = round(defect_rate * 1_000_000, 1)

        factor_results.append(
            {
                "factor": fval,
                "n": n_f,
                "jsd": float(jsd_full),
                "jsd_tail": float(jsd_tails),
                "defect_eff": float(defect_eff),
                "loc_pct": float(loc_pct),
                "scale_pct": float(scale_pct),
                "pws": f_pws,
                "cpk": f_cpk,
                "direction": direction,
                "cpk_without": cpk_without,
                "delta_cpk": cpk_without - pooled_cpk,
                "ppm": ppm,
            }
        )

    # Sort: highest defect rate first (PPM descending) — this is what practitioners
    # care about. Ties broken by JSD.
    factor_results.sort(key=lambda x: (-x["ppm"], -x["jsd"]))

    # Build charts

    # Plot 1: Cpk comparison per factor (direct, no signed JSD confusion)
    # Sort by Cpk ascending (worst first) for this chart
    fr_by_cpk = sorted(factor_results, key=lambda x: x["cpk"])
    cpk_bar_colors = []
    for fr in fr_by_cpk:
        if fr["jsd"] < noise:
            cpk_bar_colors.append(_rgba(COLOR_NEUTRAL, 0.6))
        elif fr["cpk"] < pooled_cpk:
            cpk_bar_colors.append(COLOR_BAD)
        else:
            cpk_bar_colors.append(COLOR_GOOD)

    result["plots"].append(
        {
            "title": f"D-Cpk: {factor} Capability Comparison ({variable})",
            "data": [
                {
                    "type": "bar",
                    "x": [fr["factor"] for fr in fr_by_cpk],
                    "y": [round(fr["cpk"], 3) for fr in fr_by_cpk],
                    "marker": {"color": cpk_bar_colors},
                    "text": [
                        f"Cpk={fr['cpk']:.3f}<br>JSD={fr['jsd']:.4f}<br>PPM={fr.get('ppm', 'N/A')}" for fr in fr_by_cpk
                    ],
                    "hoverinfo": "text+x",
                    "textposition": "outside",
                    "name": "Factor Cpk",
                },
                {
                    "type": "scatter",
                    "x": [fr_by_cpk[0]["factor"], fr_by_cpk[-1]["factor"]],
                    "y": [pooled_cpk, pooled_cpk],
                    "mode": "lines",
                    "name": f"Pooled Cpk ({pooled_cpk:.3f})",
                    "line": {"color": COLOR_REFERENCE, "dash": "dash", "width": 2},
                },
            ],
            "layout": {
                "height": 400,
                "xaxis": {"title": factor},
                "yaxis": {"title": "Cpk", "rangemode": "tozero"},
            },
        }
    )

    # ── Distribution overlay with spec limits ──
    dist_traces = []
    for i, fr in enumerate(factor_results):
        if fr["n"] < 5:
            continue
        mask = factors == fr["factor"]
        fdata = values[mask]
        fd = _kde_density(fdata, grid)
        c = SVEND_COLORS[i % len(SVEND_COLORS)]
        dist_traces.append(
            {
                "type": "scatter",
                "x": grid.tolist(),
                "y": fd.tolist(),
                "mode": "lines",
                "fill": "tozeroy",
                "fillcolor": _rgba(c, 0.12),
                "line": {"color": c, "width": 2},
                "name": f"{fr['factor']} (Cpk={fr['cpk']:.2f})",
            }
        )
    # Spec limit lines
    y_max = max((max(t["y"]) for t in dist_traces if t.get("y")), default=0.5)
    if lsl is not None:
        dist_traces.append(
            {
                "type": "scatter",
                "x": [lsl, lsl],
                "y": [0, y_max * 1.1],
                "mode": "lines",
                "line": {"color": COLOR_BAD, "width": 2, "dash": "dash"},
                "name": f"LSL ({lsl})",
                "showlegend": True,
            }
        )
    if usl is not None:
        dist_traces.append(
            {
                "type": "scatter",
                "x": [usl, usl],
                "y": [0, y_max * 1.1],
                "mode": "lines",
                "line": {"color": COLOR_BAD, "width": 2, "dash": "dash"},
                "name": f"USL ({usl})",
                "showlegend": True,
            }
        )
    result["plots"].append(
        {
            "title": f"Distribution Overlay by {factor} with Spec Limits",
            "data": dist_traces,
            "layout": {
                "height": 400,
                "xaxis": {"title": variable},
                "yaxis": {"title": "Density"},
                "legend": {"orientation": "h", "y": -0.2},
            },
        }
    )

    # Counterfactual Cpk chart
    result["plots"].append(
        {
            "title": f"Counterfactual: Cpk Without Each {factor} Level",
            "data": [
                {
                    "type": "bar",
                    "x": [fr["factor"] for fr in factor_results],
                    "y": [round(fr["cpk_without"], 3) for fr in factor_results],
                    "marker": {"color": [COLOR_BAD if fr["delta_cpk"] > 0.05 else COLOR_INFO for fr in factor_results]},
                    "name": "Cpk without factor",
                },
                {
                    "type": "scatter",
                    "x": [factor_results[0]["factor"], factor_results[-1]["factor"]],
                    "y": [pooled_cpk, pooled_cpk],
                    "mode": "lines",
                    "name": f"Pooled Cpk ({pooled_cpk:.3f})",
                    "line": {"color": COLOR_REFERENCE, "dash": "dash", "width": 2},
                },
            ],
            "layout": {
                "height": 400,
                "xaxis": {"title": factor},
                "yaxis": {"title": "Cpk", "rangemode": "tozero"},
            },
        }
    )

    # ── PPM / Yield impact (pre-computed in factor loop) ──
    ppm_factors = [fr for fr in factor_results if fr["n"] >= 5 and fr["ppm"] > 0]
    if ppm_factors:
        # Sort PPM chart by PPM descending (worst first, matches factor_results order)
        ppm_sorted = sorted(ppm_factors, key=lambda x: -x["ppm"])
        result["plots"].append(
            {
                "title": f"Estimated Defect Rate (PPM) by {factor}",
                "data": [
                    {
                        "type": "bar",
                        "x": [fr["factor"] for fr in ppm_sorted],
                        "y": [fr["ppm"] for fr in ppm_sorted],
                        "marker": {
                            "color": [
                                COLOR_BAD if fr["jsd"] > noise else _rgba(COLOR_NEUTRAL, 0.6) for fr in ppm_sorted
                            ]
                        },
                        "text": [f"{fr['ppm']:,.0f}" for fr in ppm_sorted],
                        "textposition": "outside",
                    }
                ],
                "layout": {
                    "height": 400,
                    "xaxis": {"title": factor},
                    "yaxis": {"title": "PPM (parts per million)", "rangemode": "tozero"},
                },
            }
        )

    # ── Divergence Profile: bridges D-Chart (total JSD) and D-Cpk (defect impact) ──
    # Stacked bar: total JSD split into tail (defect-producing) vs body (within-spec)
    profile_factors = [fr for fr in factor_results if fr["n"] >= 5 and fr["jsd"] > 0]
    if profile_factors:
        p_sorted = sorted(profile_factors, key=lambda x: -x["jsd"])
        result["plots"].append(
            {
                "title": f"Divergence Profile: Total vs Defect-Producing ({factor})",
                "data": [
                    {
                        "type": "bar",
                        "x": [fr["factor"] for fr in p_sorted],
                        "y": [round(fr["jsd_tail"], 6) for fr in p_sorted],
                        "name": "Tail JSD (defect-producing)",
                        "marker": {"color": COLOR_BAD},
                    },
                    {
                        "type": "bar",
                        "x": [fr["factor"] for fr in p_sorted],
                        "y": [round(fr["jsd"] - fr["jsd_tail"], 6) for fr in p_sorted],
                        "name": "Body JSD (within-spec)",
                        "marker": {"color": COLOR_INFO},
                    },
                ],
                "layout": {
                    "height": 400,
                    "barmode": "stack",
                    "xaxis": {"title": factor},
                    "yaxis": {"title": "JSD (bits)", "rangemode": "tozero"},
                    "legend": {"orientation": "h", "y": -0.2},
                    "annotations": [
                        {
                            "x": fr["factor"],
                            "y": fr["jsd"] + 0.002,
                            "text": f"{fr['defect_eff']:.0%}",
                            "showarrow": False,
                            "font": {"size": 10, "color": COLOR_BAD},
                        }
                        for fr in p_sorted
                        if fr["defect_eff"] > 0.01
                    ],
                },
            }
        )

    # Summary — factor_results is sorted by PPM descending (worst defect producer first)
    significant = [fr for fr in factor_results if fr["jsd"] > noise]
    degraders = [fr for fr in significant if fr["direction"] > 0]
    worst = degraders[0] if degraders else (factor_results[0] if factor_results else None)

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>D-Cpk: FACTOR-ATTRIBUTED CAPABILITY DIVERGENCE<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {variable}\n"
    summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(unique_factors)} levels)\n"
    spec_str = f"LSL={lsl}" if lsl is not None else ""
    if usl is not None:
        spec_str += (", " if spec_str else "") + f"USL={usl}"
    summary += f"<<COLOR:highlight>>Spec Limits:<</COLOR>> {spec_str}\n"
    summary += f"<<COLOR:highlight>>Pooled Cpk:<</COLOR>> {pooled_cpk:.3f}\n"
    summary += f"<<COLOR:highlight>>Noise Floor:<</COLOR>> {noise:.4f} bits\n\n"

    summary += "<<COLOR:accent>>── Factor Attribution (sorted by defect rate) ──<</COLOR>>\n"
    for fr in factor_results:
        dir_sym = "▲" if fr["direction"] > 0 else "▼"
        sig = " <<COLOR:danger>>***<</COLOR>>" if fr["jsd"] > noise else ""
        summary += f"  {fr['factor']}: PPM={fr['ppm']:,.0f} | Cpk={fr['cpk']:.3f} | JSD={fr['jsd']:.4f} {dir_sym} | Cpk_without={fr['cpk_without']:.3f}{sig}\n"

    # Divergence profile: bridges D-Chart (total JSD) and D-Cpk (defect impact)
    profile_frs = [fr for fr in factor_results if fr.get("jsd", 0) > 0 and fr["n"] >= 5]
    if profile_frs:
        summary += "\n<<COLOR:accent>>── Divergence Profile ──<</COLOR>>\n"
        for fr in profile_frs:
            loc_label = "location" if fr["loc_pct"] > fr["scale_pct"] else "scale"
            loc_dom = max(fr["loc_pct"], fr["scale_pct"])
            summary += (
                f"  {fr['factor']}: Total JSD={fr['jsd']:.4f} | "
                f"Tail JSD={fr['jsd_tail']:.4f} | "
                f"Defect efficiency={fr['defect_eff']:.0%} | "
                f"Driver: {loc_label} ({loc_dom:.0%})\n"
            )

    summary += "\n<<COLOR:accent>>── Assessment ──<</COLOR>>\n"
    if degraders:
        pooled_ppm = sum(fr["ppm"] * fr["n"] for fr in factor_results) / sum(fr["n"] for fr in factor_results)
        summary += f"<<COLOR:danger>>Factor '{worst['factor']}' is the largest defect contributor at {worst['ppm']:,.0f} PPM (pooled avg: {pooled_ppm:,.0f} PPM).<</COLOR>>\n"
        summary += f"<<COLOR:warning>>Cpk for {worst['factor']}: {worst['cpk']:.3f} vs pooled {pooled_cpk:.3f}. Removing it would raise Cpk to {worst['cpk_without']:.3f}.<</COLOR>>\n"
        # Explain the D-Chart vs D-Cpk ranking difference if it exists
        jsd_top = max(degraders, key=lambda x: x["jsd"])
        ppm_top = degraders[0]  # already sorted by PPM
        if jsd_top["factor"] != ppm_top["factor"]:
            summary += (
                f"<<COLOR:highlight>>Note: D-Chart ranks {jsd_top['factor']} highest (largest distributional change) "
                f"while D-Cpk ranks {ppm_top['factor']} highest (most defects). "
                f"{ppm_top['factor']}'s divergence is {ppm_top['defect_eff']:.0%} defect-efficient "
                f"({int(ppm_top['scale_pct'] * 100)}% scale) vs {jsd_top['factor']}'s "
                f"{jsd_top['defect_eff']:.0%} ({int(jsd_top['loc_pct'] * 100)}% location).<</COLOR>>\n"
            )
        if len(degraders) > 1:
            others = ", ".join(f"{fr['factor']} ({fr['ppm']:,.0f} PPM)" for fr in degraders[1:])
            summary += f"<<COLOR:highlight>>Also degrading: {others}<</COLOR>>\n"
    elif significant:
        summary += "<<COLOR:warning>>Significant divergence detected but factors are performing BETTER than pooled.<</COLOR>>\n"
    else:
        summary += "<<COLOR:good>>No significant factor-attributed capability divergence. All levels perform consistently.<</COLOR>>\n"

    result["summary"] = summary

    obs = (
        f"D-Cpk: {factor} on {variable} ({spec_str}). Pooled Cpk={pooled_cpk:.3f}. "
        f"{len(significant)} of {len(unique_factors)} factors show significant divergence. "
    )
    if worst and worst.get("ppm", 0) > 0 and worst["jsd"] > noise:
        obs += f"Worst: {worst['factor']} ({worst['ppm']:,.0f} PPM, Cpk={worst['cpk']:.3f}, counterfactual Cpk={worst['cpk_without']:.3f})."
    else:
        obs += "No significant factor effects on capability."
    result["guide_observation"] = obs

    result["statistics"] = {
        "pooled_cpk": round(pooled_cpk, 4),
        "pooled_pws": round(pooled_pws, 4),
        "noise_floor": round(noise, 6),
        "n_significant": len(significant),
        "factors": {
            fr["factor"]: {
                "jsd": round(fr["jsd"], 6),
                "jsd_tail": round(fr.get("jsd_tail", 0), 6),
                "defect_efficiency": round(fr.get("defect_eff", 0), 4),
                "location_pct": round(fr.get("loc_pct", 0.5), 4),
                "scale_pct": round(fr.get("scale_pct", 0.5), 4),
                "cpk": round(fr["cpk"], 4),
                "pws": round(fr["pws"], 4),
                "direction": fr["direction"],
                "cpk_without": round(fr["cpk_without"], 4),
                "delta_cpk": round(fr.get("delta_cpk", 0), 4),
                "ppm": fr.get("ppm"),
            }
            for fr in factor_results
        },
    }

    result["narrative"] = _d_narrative(
        f"D-Cpk: {factor} capability attribution on {variable}",
        _d_cpk_body(factor_results, noise, pooled_cpk, variable, factor, spec_str),
        _d_cpk_nextsteps(factor_results, noise, factor),
        "The Cpk comparison chart shows each factor's capability vs pooled. "
        "The divergence profile (stacked bar) decomposes total JSD into defect-producing "
        "(tail, red) vs within-spec (body, blue). Percentages above bars show defect efficiency — "
        "the fraction of divergence that translates to defects.",
    )

    result["education"] = {
        "title": "Understanding D-Cpk",
        "content": (
            "<dl>"
            "<dt>What is D-Cpk?</dt>"
            "<dd>D-Cpk attributes capability differences to specific factor levels. "
            "Standard Cpk tells you <em>if</em> the process is capable. D-Cpk tells you "
            "<em>which factors are dragging capability down</em> and by how much.</dd>"
            "<dt>How does it work?</dt>"
            "<dd>For each factor level it computes a KDE-based Cpk (no normality assumption), "
            "then measures how much that level's distribution diverges from the pooled distribution "
            "using JSD. Divergence is decomposed into tail (defect-producing) vs body (within-spec) "
            "components.</dd>"
            "<dt>What is Defect Efficiency?</dt>"
            "<dd>The fraction of a factor's total divergence that occurs in the spec tails "
            "(where defects happen). High defect efficiency (e.g., 80%) means the factor "
            "mostly affects the tails — it is directly creating defects. Low efficiency "
            "means the factor shifts the distribution but mostly within spec.</dd>"
            "<dt>What is the Counterfactual Cpk?</dt>"
            "<dd>The Cpk the process <em>would have</em> if that factor level were removed. "
            "A large gap between actual Cpk and counterfactual Cpk means fixing that factor "
            "would significantly improve capability. This directly prioritizes improvement actions.</dd>"
            "<dt>How to interpret</dt>"
            "<dd><strong>All factors below noise floor</strong>: Capability is uniform across "
            "factor levels — look elsewhere for improvement. <strong>Factor above noise with "
            "high defect efficiency</strong>: Priority target — this factor is directly causing "
            "defects. <strong>Factor above noise with low defect efficiency</strong>: The factor "
            "changes the distribution but not the defect rate — may be acceptable.</dd>"
            "</dl>"
        ),
    }

    return result
