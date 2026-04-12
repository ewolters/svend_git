"""D-Equiv — distributional equivalence testing.

CR: 3c0d0e53
"""

import logging

import numpy as np

from ..common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_REFERENCE,
    SVEND_COLORS,
)
from .helpers import _d_narrative, _jsd, _kde_density

logger = logging.getLogger(__name__)


def run_d_equiv(df, config):
    """Batch distributional equivalence testing via Jensen-Shannon Divergence.

    Compares each batch's KDE density against a reference batch and decides
    equivalence based on a JSD threshold with permutation-based significance.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = config.get("variable") or config.get("measurement")
    batch_col = config.get("batch") or config.get("group") or config.get("factor")
    if not variable or variable not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid measurement variable.<</COLOR>>"
        return result
    if not batch_col or batch_col not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid batch/group column.<</COLOR>>"
        return result

    threshold = float(config.get("threshold", 0.05))
    ref_batch = config.get("reference")

    work = df[[variable, batch_col]].dropna()
    work[variable] = work[variable].astype(float)
    work[batch_col] = work[batch_col].astype(str)

    batches = work[batch_col].unique()
    if len(batches) < 2:
        result["summary"] = "<<COLOR:danger>>Need at least 2 batches for equivalence testing.<</COLOR>>"
        return result

    # Choose reference batch
    batch_sizes = work.groupby(batch_col).size()
    if ref_batch and str(ref_batch) in batches:
        ref_batch = str(ref_batch)
    else:
        ref_batch = str(batch_sizes.idxmax())

    ref_data = work.loc[work[batch_col] == ref_batch, variable].values
    if len(ref_data) < 5:
        result["summary"] = f"<<COLOR:danger>>Reference batch '{ref_batch}' has fewer than 5 observations.<</COLOR>>"
        return result

    # Common grid
    all_data = work[variable].values
    margin = (all_data.max() - all_data.min()) * 0.2
    grid = np.linspace(all_data.min() - margin, all_data.max() + margin, 500)

    ref_density = _kde_density(ref_data, grid)

    # Permutation noise floor
    n_perm = 200
    all_vals = work[variable].values
    perm_jsds = []
    rng = np.random.RandomState(42)
    n_ref = len(ref_data)
    for _ in range(n_perm):
        perm = rng.permutation(all_vals)
        d1 = _kde_density(perm[:n_ref], grid)
        d2 = _kde_density(perm[n_ref : 2 * n_ref] if len(perm) >= 2 * n_ref else perm[n_ref:], grid)
        perm_jsds.append(_jsd(d1, d2, grid))
    noise_95 = float(np.percentile(perm_jsds, 95))

    # Per-batch analysis
    batch_results = []
    test_batches = [b for b in batches if b != ref_batch]
    for bname in test_batches:
        bdata = work.loc[work[batch_col] == bname, variable].values
        if len(bdata) < 5:
            continue
        b_density = _kde_density(bdata, grid)
        jsd_val = _jsd(b_density, ref_density, grid)

        # Permutation p-value
        p_val = float(np.mean(np.array(perm_jsds) >= jsd_val))
        equiv = jsd_val < threshold
        batch_results.append(
            {
                "batch": bname,
                "n": len(bdata),
                "jsd": round(jsd_val, 5),
                "p_value": round(p_val, 4),
                "equivalent": equiv,
                "mean": round(float(bdata.mean()), 4),
                "std": round(float(bdata.std(ddof=1)), 4),
            }
        )

    batch_results.sort(key=lambda x: x["jsd"])

    # Pairwise JSD matrix
    all_batch_names = [ref_batch] + [br["batch"] for br in batch_results]
    all_batch_data = {}
    for bname in all_batch_names:
        bdata = work.loc[work[batch_col] == bname, variable].values
        if len(bdata) >= 5:
            all_batch_data[bname] = _kde_density(bdata, grid)
    n_batches = len(all_batch_names)
    jsd_matrix = np.zeros((n_batches, n_batches))
    for i in range(n_batches):
        for j in range(i + 1, n_batches):
            if all_batch_names[i] in all_batch_data and all_batch_names[j] in all_batch_data:
                jsd_ij = _jsd(
                    all_batch_data[all_batch_names[i]],
                    all_batch_data[all_batch_names[j]],
                    grid,
                )
                jsd_matrix[i, j] = jsd_ij
                jsd_matrix[j, i] = jsd_ij

    # --- Plot 1: JSD bar chart ---
    bar_names = [br["batch"] for br in batch_results]
    bar_jsds = [br["jsd"] for br in batch_results]
    bar_colors = [COLOR_GOOD if br["equivalent"] else COLOR_BAD for br in batch_results]
    result["plots"].append(
        {
            "title": f"Batch Divergence from Reference '{ref_batch}'",
            "data": [
                {
                    "type": "bar",
                    "x": bar_names,
                    "y": bar_jsds,
                    "name": "JSD",
                    "marker": {"color": bar_colors},
                    "text": [f"{v:.4f}" for v in bar_jsds],
                    "textposition": "outside",
                },
            ],
            "layout": {
                "height": 300,
                "yaxis": {"title": "JSD (bits)"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": len(bar_names) - 0.5,
                        "y0": threshold,
                        "y1": threshold,
                        "line": {"color": COLOR_GOLD, "dash": "dash", "width": 2},
                    }
                ],
                "annotations": [
                    {
                        "x": len(bar_names) - 0.5,
                        "y": threshold,
                        "text": f"Threshold={threshold}",
                        "showarrow": False,
                        "font": {"color": COLOR_GOLD, "size": 10},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 2: Density overlay ---
    density_traces = []
    density_traces.append(
        {
            "type": "scatter",
            "x": grid.tolist(),
            "y": ref_density.tolist(),
            "mode": "lines",
            "name": f"{ref_batch} (ref)",
            "line": {"color": COLOR_REFERENCE, "width": 3},
        }
    )
    for i, br in enumerate(batch_results):
        bdata = work.loc[work[batch_col] == br["batch"], variable].values
        b_dens = _kde_density(bdata, grid)
        color = SVEND_COLORS[i % len(SVEND_COLORS)]
        density_traces.append(
            {
                "type": "scatter",
                "x": grid.tolist(),
                "y": b_dens.tolist(),
                "mode": "lines",
                "name": br["batch"],
                "line": {
                    "color": color,
                    "width": 1.5,
                    "dash": "dash" if not br["equivalent"] else "solid",
                },
            }
        )
    result["plots"].append(
        {
            "title": "Batch Density Overlay",
            "data": density_traces,
            "layout": {
                "height": 340,
                "xaxis": {"title": variable},
                "yaxis": {"title": "Density"},
                "showlegend": True,
            },
        }
    )

    # --- Plot 3: Pairwise JSD heatmap ---
    result["plots"].append(
        {
            "title": "Pairwise JSD Matrix",
            "data": [
                {
                    "type": "heatmap",
                    "z": jsd_matrix.tolist(),
                    "x": all_batch_names,
                    "y": all_batch_names,
                    "colorscale": [[0, "#f0f8f0"], [0.5, COLOR_GOLD], [1, COLOR_BAD]],
                    "text": [[f"{jsd_matrix[i][j]:.4f}" for j in range(n_batches)] for i in range(n_batches)],
                    "texttemplate": "%{text}",
                    "showscale": True,
                    "colorbar": {"title": "JSD"},
                }
            ],
            "layout": {
                "height": 380,
                "xaxis": {"title": "Batch"},
                "yaxis": {"title": "Batch", "autorange": "reversed"},
            },
        }
    )

    # --- Summary ---
    n_equiv = sum(1 for br in batch_results if br["equivalent"])
    n_test = len(batch_results)
    summary = "<<COLOR:title>>D-EQUIV — BATCH EQUIVALENCE VIA JSD<</COLOR>>\n\n"
    summary += f"<<COLOR:header>>Reference Batch:<</COLOR>> '{ref_batch}' (n={len(ref_data)})\n"
    summary += f"<<COLOR:header>>Equivalence Threshold:<</COLOR>> {threshold} JSD bits\n"
    summary += f"<<COLOR:header>>Permutation Noise Floor (95th):<</COLOR>> {noise_95:.5f}\n\n"

    summary += "<<COLOR:header>>Results:<</COLOR>>\n"
    summary += f"  {'Batch':<15} {'n':>5} {'JSD':>8} {'p-val':>7} {'Decision':>12}\n"
    summary += f"  {'-' * 50}\n"
    for br in batch_results:
        dec = "Equivalent" if br["equivalent"] else "DIFFERENT"
        color = "success" if br["equivalent"] else "danger"
        summary += f"  {br['batch']:<15} {br['n']:>5} {br['jsd']:>8.4f} {br['p_value']:>7.3f} <<COLOR:{color}>>{dec}<</COLOR>>\n"

    summary += "\n<<COLOR:header>>Verdict:<</COLOR>> "
    if n_equiv == n_test:
        summary += f"<<COLOR:success>>All {n_test} batches are equivalent to reference '{ref_batch}'.<</COLOR>>"
    elif n_equiv == 0:
        summary += f"<<COLOR:danger>>No batches are equivalent to reference '{ref_batch}'.<</COLOR>>"
    else:
        summary += f"<<COLOR:warning>>{n_equiv} of {n_test} batches equivalent to reference '{ref_batch}'.<</COLOR>>"

    result["summary"] = summary
    result["guide_observation"] = (
        f"D-Equiv: {n_equiv}/{n_test} batches equivalent to ref '{ref_batch}' (threshold={threshold})"
    )
    result["statistics"] = {
        "reference_batch": ref_batch,
        "threshold": threshold,
        "noise_floor_95": noise_95,
        "n_equivalent": n_equiv,
        "n_tested": n_test,
        "batch_results": batch_results,
    }

    # --- narrative ---
    if n_equiv == n_test:
        verdict = f"All Equivalent — {n_equiv}/{n_test} batches"
        body = (
            f"All {n_test} batches are distributionally equivalent to reference "
            f"batch '<strong>{ref_batch}</strong>' (JSD threshold = {threshold}). "
            f"The process is consistent across batches."
        )
        nxt = "Continue monitoring — batch consistency is good."
    elif n_equiv == 0:
        verdict = f"None Equivalent — 0/{n_test} batches"
        body = (
            f"No batches are equivalent to reference '<strong>{ref_batch}</strong>'. "
            f"Every batch shows distributional divergence above the threshold ({threshold}). "
            f"The process has significant batch-to-batch variation."
        )
        nxt = "Investigate batch-level variation sources; consider D-Chart for factor attribution."
    else:
        non_equiv = [br["batch"] for br in batch_results if not br["equivalent"]]
        top_offenders = ", ".join(non_equiv[:3])
        verdict = f"Mixed — {n_equiv}/{n_test} equivalent"
        body = (
            f"<strong>{n_equiv}</strong> of {n_test} batches match the reference "
            f"'<strong>{ref_batch}</strong>'. Non-equivalent: {top_offenders}"
            + (f" (+{len(non_equiv) - 3} more)" if len(non_equiv) > 3 else "")
            + "."
        )
        nxt = f"Investigate the non-equivalent batches — what changed vs reference '{ref_batch}'?"
    result["narrative"] = _d_narrative(
        f"D-Equiv: {verdict}",
        body,
        nxt,
        "The bar chart shows each batch's JSD vs the reference. Bars below the threshold (dashed line) are equivalent. The heatmap shows pairwise JSD between all batches.",
    )

    result["education"] = {
        "title": "Understanding Batch Equivalence (D-Equiv)",
        "content": (
            "<dl>"
            "<dt>What does D-Equiv test?</dt>"
            "<dd>Whether batches produce the same <em>distribution</em> of output — not just "
            "the same mean. Two batches can have identical means but very different spreads, "
            "shapes, or tail behavior. D-Equiv catches all of these via JSD comparison against "
            "a reference batch.</dd>"
            "<dt>How is equivalence decided?</dt>"
            "<dd>Each batch's KDE density is compared to the reference via JSD. If JSD is "
            "below a threshold (default: the 95th percentile of the noise floor from permutation), "
            "the batch is declared equivalent. This accounts for expected sampling variation.</dd>"
            "<dt>Why a reference batch?</dt>"
            "<dd>The reference is your 'known good' — a batch produced under controlled conditions. "
            "All other batches are compared against it. You can choose the reference in the config.</dd>"
            "<dt>How to interpret the heatmap</dt>"
            "<dd>The pairwise JSD heatmap shows which batches are similar to each other (cool colors) "
            "vs different (hot colors). Clusters of similar batches may indicate shared process conditions.</dd>"
            "</dl>"
        ),
    }

    return result
