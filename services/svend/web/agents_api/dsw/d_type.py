"""D-Type Process Intelligence — factor divergence analysis via Jensen-Shannon Divergence.

Two tools:
  - D-Chart: factor divergence tracked over rolling time windows
  - D-Cpk:  factor-attributed capability divergence with counterfactual analysis

Mathematical core uses Gaussian KDE with ISJ bandwidth selection and JSD (base-2)
bounded [0, 1] in bits. Noise floor estimated via bootstrap permutation.
"""

import logging

import numpy as np
import pandas as pd

from .common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_NEUTRAL,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
    _rgba,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mathematical core
# ---------------------------------------------------------------------------


def _kde_density(x, grid, bandwidth=None):
    """Gaussian KDE with ISJ bandwidth via KDEpy FFTKDE (fast).

    Falls back to scipy gaussian_kde with Silverman if KDEpy unavailable.
    Returns density array evaluated at `grid` points.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.ones_like(grid) / (grid[-1] - grid[0])

    try:
        from KDEpy import FFTKDE

        bw = bandwidth or "ISJ"
        # FFTKDE evaluates on its own grid; we interpolate to ours
        _grid, density = FFTKDE(bw=bw, kernel="gaussian").fit(x).evaluate(len(grid))
        # Interpolate to our grid
        density = np.interp(grid, _grid, density)
    except Exception:
        from scipy.stats import gaussian_kde

        bw = bandwidth or "silverman"
        try:
            kde = gaussian_kde(x, bw_method=float(bw) if isinstance(bw, (int, float)) else bw)
            density = kde(grid)
        except Exception:
            kde = gaussian_kde(x, bw_method="silverman")
            density = kde(grid)

    # Clamp and normalize
    density = np.maximum(density, 0)
    total = np.trapz(density, grid)
    if total > 0:
        density = density / total
    return density


def _jsd(p, q, grid):
    """Jensen-Shannon Divergence in bits (base 2), bounded [0, 1].

    Uses scipy's jensenshannon (which returns the *distance*, i.e. sqrt(JSD)).
    We square it to get the actual divergence.
    """
    from scipy.spatial.distance import jensenshannon

    # Normalize to proper PMFs over grid with epsilon floor
    # (scipy's jensenshannon uses rel_entr internally which gives inf when q=0, p>0)
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    dx = np.diff(grid)
    dx = np.append(dx, dx[-1])
    p_pmf = p * dx + 1e-300
    q_pmf = q * dx + 1e-300
    p_pmf = p_pmf / p_pmf.sum()
    q_pmf = q_pmf / q_pmf.sum()

    js_dist = jensenshannon(p_pmf, q_pmf, base=2)
    if not np.isfinite(js_dist):
        return 0.0
    return float(js_dist**2)  # divergence = distance²


def _jsd_tail(p, q, grid, lsl=None, usl=None):
    """Tail contribution to total JSD — the portion of divergence in out-of-spec regions.

    Uses the same PMF normalization as the full JSD, then sums only the
    element-wise divergence terms that fall outside spec limits. This gives a
    proper decomposition: tail_contribution + body_contribution = total JSD.
    """
    tail_mask = np.zeros_like(grid, dtype=bool)
    if lsl is not None:
        tail_mask |= grid < lsl
    if usl is not None:
        tail_mask |= grid > usl

    if not tail_mask.any():
        return 0.0

    # Build PMFs with same normalization as _jsd
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    dx = np.diff(grid)
    dx = np.append(dx, dx[-1])
    p_pmf = p * dx + 1e-300
    q_pmf = q * dx + 1e-300
    p_pmf = p_pmf / p_pmf.sum()
    q_pmf = q_pmf / q_pmf.sum()

    # M = midpoint distribution
    m_pmf = 0.5 * (p_pmf + q_pmf)

    # Element-wise JSD contribution: 0.5 * [p*log(p/m) + q*log(q/m)]
    # Only sum terms in the tail region
    eps = 1e-300
    tail_jsd = 0.0
    for i in np.where(tail_mask)[0]:
        pi, qi, mi = p_pmf[i], q_pmf[i], m_pmf[i]
        if mi > eps:
            if pi > eps:
                tail_jsd += 0.5 * pi * np.log2(pi / mi)
            if qi > eps:
                tail_jsd += 0.5 * qi * np.log2(qi / mi)

    return max(0.0, float(tail_jsd))


def _decompose_divergence(fdata, ref_data):
    """Decompose distributional divergence into location and scale components.

    Returns (location_pct, scale_pct) — percentage of divergence attributable
    to mean shift vs variance change. Uses squared z-score decomposition.
    """
    mu_f, mu_r = fdata.mean(), ref_data.mean()
    sd_f, sd_r = fdata.std(ddof=1), ref_data.std(ddof=1)

    if sd_r == 0 or sd_f == 0:
        return 0.5, 0.5  # can't decompose

    # Squared standardized effects
    location_effect = ((mu_f - mu_r) / sd_r) ** 2
    scale_effect = (sd_f / sd_r - 1) ** 2

    total = location_effect + scale_effect
    if total == 0:
        return 0.5, 0.5
    return location_effect / total, scale_effect / total


def _noise_floor(pooled, n_per_group, grid, B=200, quantile=0.95, rng=None):
    """Bootstrap noise floor: expected JSD from random splits of pooled data.

    Splits pooled array into two halves B times, computes JSD each time,
    returns the `quantile`-th percentile as the noise floor.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pooled = np.asarray(pooled, dtype=float)
    pooled = pooled[np.isfinite(pooled)]
    n = len(pooled)
    if n < 10:
        return 0.0

    half = max(n // 2, 5)
    jsds = []
    for _ in range(B):
        idx = rng.permutation(n)
        a = pooled[idx[:half]]
        b = pooled[idx[half : half * 2]]
        pa = _kde_density(a, grid)
        pb = _kde_density(b, grid)
        jsds.append(_jsd(pa, pb, grid))

    return float(np.percentile(jsds, quantile * 100))


def _build_grid(data, n_points=512):
    """Build evaluation grid spanning the data range with 10% padding."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * 0.1 if hi > lo else 1.0
    return np.linspace(lo - margin, hi + margin, n_points)


# ---------------------------------------------------------------------------
# D-Chart: Factor Divergence Over Time
# ---------------------------------------------------------------------------


def run_d_chart(df, config):
    """D-Chart — track per-factor JSD vs pooled distribution over rolling time windows.

    Config:
        variable:     numeric column to analyze
        factor:       categorical column (e.g. Shift, Machine, Operator)
        time_col:     datetime or ordinal column for windowing
        window_size:  number of observations per window (default 50)
        step_size:    window step (default = window_size // 2)
    """
    import pandas as pd

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


# ---------------------------------------------------------------------------
# D-Cpk: Factor-Attributed Capability Divergence
# ---------------------------------------------------------------------------


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

    variable = config.get("variable")
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


# ---------------------------------------------------------------------------
# Capability helpers
# ---------------------------------------------------------------------------


def _p_within_spec(density, grid, lsl, usl):
    """Probability of being within spec limits given a density over grid."""
    mask = np.ones_like(grid, dtype=bool)
    if lsl is not None:
        mask &= grid >= lsl
    if usl is not None:
        mask &= grid <= usl
    return float(np.trapz(density[mask], grid[mask]))


def _compute_cpk(data, lsl, usl):
    """Classical Cpk from data."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return 0.0
    mu = data.mean()
    sigma = data.std(ddof=1)
    if sigma == 0:
        return 0.0
    cpks = []
    if usl is not None:
        cpks.append((usl - mu) / (3 * sigma))
    if lsl is not None:
        cpks.append((mu - lsl) / (3 * sigma))
    return min(cpks) if cpks else 0.0


def _bernoulli_jsd(p1, p2):
    """JSD between two Bernoulli distributions Bernoulli(p1) and Bernoulli(p2)."""
    from scipy.spatial.distance import jensenshannon

    # Clamp to avoid log(0)
    eps = 1e-12
    p1 = np.clip(p1, eps, 1 - eps)
    p2 = np.clip(p2, eps, 1 - eps)
    dist1 = np.array([p1, 1 - p1])
    dist2 = np.array([p2, 1 - p2])
    js_dist = jensenshannon(dist1, dist2, base=2)
    return float(js_dist**2)


def _cpk_noise_floor(values, n_factors, grid, lsl, usl, B=200):
    """Bootstrap noise floor for capability JSD.

    Randomly assigns factor labels and computes JSD of resulting Bernoulli capabilities.
    """
    rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 10:
        return 0.0

    pooled_density = _kde_density(values, grid)
    pooled_pws = _p_within_spec(pooled_density, grid, lsl, usl)

    jsds = []
    group_size = max(n // n_factors, 10)
    for _ in range(B):
        idx = rng.permutation(n)
        subset = values[idx[:group_size]]
        sub_density = _kde_density(subset, grid)
        sub_pws = _p_within_spec(sub_density, grid, lsl, usl)
        jsds.append(_bernoulli_jsd(sub_pws, pooled_pws))

    return float(np.percentile(jsds, 95))


# ---------------------------------------------------------------------------
# Narrative helpers
# ---------------------------------------------------------------------------


def _d_narrative(title, body, next_steps, chart_guidance):
    """Build HTML narrative string matching DSW standard format."""
    parts = [f'<div class="dsw-verdict">{title}</div>', f"<p>{body}</p>"]
    if chart_guidance:
        parts.append(f"<p><strong>In the chart:</strong> {chart_guidance}</p>")
    if next_steps:
        parts.append(f'<div class="dsw-next"><strong>Next &rarr;</strong> {next_steps}</div>')
    return "\n".join(parts)


def _d_chart_body(sorted_factors, noise, any_above_noise, variable, factor):
    top = sorted_factors[0] if sorted_factors else ("N/A", 0)
    if any_above_noise and top[1] > noise * 2:
        return (
            f"The divergence analysis reveals that <strong>{top[0]}</strong> is the most divergent "
            f"level of {factor}, with a cumulative information score of {top[1]:.4f} — "
            f"well above the noise floor of {noise:.4f}. This means {top[0]} produces a "
            f"systematically different distribution of {variable} compared to the overall process."
        )
    elif any_above_noise:
        return (
            f"Some windows show factor divergence above the noise floor ({noise:.4f}), "
            f"but the cumulative pattern is moderate. The top factor is {top[0]} "
            f"with an information score of {top[1]:.4f}."
        )
    return (
        f"All levels of {factor} produce distributions of {variable} that are "
        f"statistically indistinguishable from the pooled distribution (all below "
        f"noise floor of {noise:.4f}). The process is factor-invariant."
    )


def _d_chart_nextsteps(sorted_factors, noise, any_above_noise, factor):
    top = sorted_factors[0] if sorted_factors else ("N/A", 0)
    if any_above_noise and top[1] > noise * 2:
        return (
            f"Investigate what is different about {top[0]}. "
            f"Run a targeted comparison (B-tTest or B-ANOVA) between {top[0]} and other levels. "
            f"Consider a D-Cpk analysis to quantify capability impact."
        )
    return "No action required — factor divergence is within expected random variation."


def _d_cpk_body(factor_results, noise, pooled_cpk, variable, factor, spec_str):
    significant = [fr for fr in factor_results if fr["jsd"] > noise]
    if not significant:
        return (
            f"No factor level shows significant capability divergence ({spec_str}). "
            f"The pooled Cpk of {pooled_cpk:.3f} is attributable to common-cause variation "
            f"across all levels of {factor}."
        )
    worst = significant[0]
    return (
        f"<strong>{worst['factor']}</strong> is the primary driver of capability divergence. "
        f"Its Cpk ({worst['cpk']:.3f}) differs significantly from the pooled value ({pooled_cpk:.3f}). "
        f"Counterfactual analysis shows removing {worst['factor']} would change Cpk to "
        f"{worst['cpk_without']:.3f} (Δ = {worst.get('delta_cpk', 0):+.3f})."
    )


def _d_cpk_nextsteps(factor_results, noise, factor):
    significant = [fr for fr in factor_results if fr["jsd"] > noise]
    if not significant:
        return "No action required — capability is consistent across all factor levels."
    worst = significant[0]
    if worst["direction"] > 0:
        return (
            f"Focus improvement on {worst['factor']}. "
            f"Investigate root cause of degraded capability — consider an RCA session. "
            f"Run a D-Chart to track whether divergence is chronic or intermittent."
        )
    return (
        f"Factor {worst['factor']} actually performs BETTER than pooled. "
        f"Study what makes this level effective and transfer the practice to other levels."
    )


# ---------------------------------------------------------------------------
# D-NonNorm: KDE-based Non-Normal Capability
# ---------------------------------------------------------------------------


def run_d_nonnorm(df, config):
    """Non-normal capability analysis using KDE density estimation.

    Computes Pp/Ppk equivalents directly from the empirical KDE density,
    compares against normal-assumption values, and quantifies the normality
    penalty.
    """
    from scipy.stats import norm as sp_norm

    from .common import _fit_best_distribution

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


# ---------------------------------------------------------------------------
# D-Equiv: Batch Equivalence via JSD
# ---------------------------------------------------------------------------


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
                jsd_ij = _jsd(all_batch_data[all_batch_names[i]], all_batch_data[all_batch_names[j]], grid)
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
                "line": {"color": color, "width": 1.5, "dash": "dash" if not br["equivalent"] else "solid"},
            }
        )
    result["plots"].append(
        {
            "title": "Batch Density Overlay",
            "data": density_traces,
            "layout": {"height": 340, "xaxis": {"title": variable}, "yaxis": {"title": "Density"}, "showlegend": True},
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


# ---------------------------------------------------------------------------
# D-Sig: Process Signature Comparison
# ---------------------------------------------------------------------------


def run_d_sig(df, config):
    """Process signature comparison via functional JSD.

    Compares time-series profiles across groups by computing windowed JSD
    at each time point to identify where and how process signatures diverge.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variable = config.get("variable") or config.get("profile")
    time_col = config.get("time_col") or config.get("time")
    group_col = config.get("group") or config.get("factor")

    if not variable or variable not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid profile variable.<</COLOR>>"
        return result
    if not time_col or time_col not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid time/sequence column.<</COLOR>>"
        return result
    if not group_col or group_col not in df.columns:
        result["summary"] = "<<COLOR:danger>>Please select a valid group column.<</COLOR>>"
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
        result["summary"] = "<<COLOR:danger>>Need at least 2 groups for signature comparison.<</COLOR>>"
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
        result["summary"] = "<<COLOR:danger>>Not enough groups with ≥5 observations.<</COLOR>>"
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
                    all_vals.min() - 0.1 * (np.ptp(all_vals) or 1), all_vals.max() + 0.1 * (np.ptp(all_vals) or 1), 100
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
    sorted_groups = sorted(test_groups, key=lambda g: group_divergences[g]["mean_jsd"], reverse=True)

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
            "layout": {"height": 340, "xaxis": {"title": time_col}, "yaxis": {"title": variable}, "showlegend": True},
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
    bar_colors_mean = [SVEND_COLORS[i % len(SVEND_COLORS)] for i in range(len(sorted_groups))]

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
            "layout": {"height": 300, "barmode": "group", "yaxis": {"title": "JSD (bits)"}, "showlegend": True},
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
        summary += (
            f"  {g:<15} {d['mean_jsd']:>10.4f} {d['max_jsd']:>10.4f} {d['peak_time']:>10.1f} {d['rmse']:>10.3f}\n"
        )

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


# ---------------------------------------------------------------------------
# D-Multi: Multivariate Capability via PCA + T²
# ---------------------------------------------------------------------------


def run_d_multi(df, config):
    """Multivariate capability analysis via PCA and Hotelling's T².

    Reduces correlated quality characteristics to principal components,
    computes KDE-based capability on each, and uses T² for joint OOC detection.
    """
    from scipy.stats import f as f_dist

    result = {"plots": [], "summary": "", "guide_observation": ""}

    variables = config.get("variables") or config.get("columns", [])
    if not variables or len(variables) < 2:
        result["summary"] = "<<COLOR:danger>>Select at least 2 numeric variables.<</COLOR>>"
        return result

    missing = [v for v in variables if v not in df.columns]
    if missing:
        result["summary"] = f"<<COLOR:danger>>Columns not found: {', '.join(missing)}<</COLOR>>"
        return result

    tolerance_pct = config.get("tolerance_pct")

    work = df[variables].dropna().astype(float)
    n, p = work.shape
    if n < p + 5:
        result["summary"] = f"<<COLOR:danger>>Need at least {p + 5} observations (have {n}).<</COLOR>>"
        return result

    data_matrix = work.values
    means = data_matrix.mean(axis=0)
    stds = data_matrix.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0

    # Standardize
    Z = (data_matrix - means) / stds

    # PCA
    try:
        from sklearn.decomposition import PCA

        pca = PCA()
        scores = pca.fit_transform(Z)
        explained = pca.explained_variance_ratio_
        loadings = pca.components_  # shape (n_components, p)
    except ImportError:
        # Fallback: manual PCA via eigendecomposition
        cov_matrix = np.cov(Z, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained = eigenvalues / eigenvalues.sum()
        scores = Z @ eigenvectors
        loadings = eigenvectors.T

    # Retain components explaining ≥95% variance
    cumvar = np.cumsum(explained)
    k = int(np.searchsorted(cumvar, 0.95) + 1)
    k = min(k, p)

    scores_k = scores[:, :k]
    explained[:k]

    # Hotelling's T² using all p variables
    cov_inv = np.linalg.pinv(np.cov(Z, rowvar=False))
    T2 = np.array([float(z @ cov_inv @ z) for z in Z])

    # UCL for T² (F-distribution based)
    T2_ucl = (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_dist.ppf(0.9973, p, n - p)
    ooc_mask = T2 > T2_ucl
    n_ooc = int(ooc_mask.sum())

    # Per-component capability (using ±3 as natural spec limits for standardized scores)
    component_cpk = []
    for j in range(k):
        comp_scores = scores_k[:, j]
        comp_mean = float(comp_scores.mean())
        comp_std = float(comp_scores.std(ddof=1))
        if comp_std > 0:
            # For PCA components, spec = ±3 standardized units (natural process limits)
            cpk_lo = (comp_mean - (-3)) / (3 * comp_std)
            cpk_hi = (3 - comp_mean) / (3 * comp_std)
            cpk_j = min(cpk_lo, cpk_hi)
        else:
            cpk_j = 999.0
        component_cpk.append(round(float(cpk_j), 3))

    mcpk = min(component_cpk) if component_cpk else 0.0

    # If user provided tolerance, compute per-variable capability too
    var_cpk = []
    if tolerance_pct:
        tol = float(tolerance_pct) / 100.0
        for i_v, v in enumerate(variables):
            v_mean = means[i_v]
            v_std = stds[i_v]
            v_range = np.ptp(data_matrix[:, i_v])
            v_lsl = v_mean - tol * v_range
            v_usl = v_mean + tol * v_range
            if v_std > 0:
                cpk_lo = (v_mean - v_lsl) / (3 * v_std)
                cpk_hi = (v_usl - v_mean) / (3 * v_std)
                var_cpk.append(
                    {
                        "variable": v,
                        "cpk": round(min(cpk_lo, cpk_hi), 3),
                        "lsl": round(v_lsl, 4),
                        "usl": round(v_usl, 4),
                    }
                )
            else:
                var_cpk.append({"variable": v, "cpk": 999.0, "lsl": round(v_lsl, 4), "usl": round(v_usl, 4)})

    # T² capability: proportion within UCL
    t2_capability = float(1.0 - ooc_mask.mean())

    # --- Plot 1: PCA biplot (PC1 vs PC2) ---
    pc1, pc2 = scores[:, 0], scores[:, 1]
    # T² ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    # Eigenvalue scaling for ellipse
    ev1 = explained[0] * p  # variance on PC1
    ev2 = explained[1] * p if p > 1 else 1
    ellipse_r = np.sqrt(T2_ucl)
    ex = ellipse_r * np.sqrt(ev1) * np.cos(theta)
    ey = ellipse_r * np.sqrt(ev2) * np.sin(theta)

    biplot_traces = [
        {
            "type": "scatter",
            "x": pc1[~ooc_mask].tolist(),
            "y": pc2[~ooc_mask].tolist(),
            "mode": "markers",
            "name": "In Control",
            "marker": {"color": SVEND_COLORS[0], "size": 4, "opacity": 0.5},
        },
        {
            "type": "scatter",
            "x": pc1[ooc_mask].tolist(),
            "y": pc2[ooc_mask].tolist(),
            "mode": "markers",
            "name": f"OOC ({n_ooc})",
            "marker": {"color": COLOR_BAD, "size": 6, "symbol": "x"},
        },
        {
            "type": "scatter",
            "x": ex.tolist(),
            "y": ey.tolist(),
            "mode": "lines",
            "name": "T² UCL",
            "line": {"color": COLOR_BAD, "dash": "dash", "width": 1.5},
        },
    ]
    # Loading arrows
    arrow_annotations = []
    scale_factor = max(abs(pc1).max(), abs(pc2).max()) * 0.8
    for i_v, v in enumerate(variables):
        lx = loadings[0, i_v] * scale_factor
        ly = loadings[1, i_v] * scale_factor
        arrow_annotations.append(
            {
                "x": lx,
                "y": ly,
                "ax": 0,
                "ay": 0,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1.5,
                "arrowcolor": COLOR_INFO,
                "text": v,
                "font": {"color": COLOR_INFO, "size": 9},
            }
        )

    result["plots"].append(
        {
            "title": f"PCA Biplot (PC1: {explained[0] * 100:.1f}%, PC2: {explained[1] * 100:.1f}%)",
            "data": biplot_traces,
            "layout": {
                "height": 380,
                "xaxis": {"title": f"PC1 ({explained[0] * 100:.1f}%)"},
                "yaxis": {"title": f"PC2 ({explained[1] * 100:.1f}%)", "scaleanchor": "x"},
                "showlegend": True,
                "annotations": arrow_annotations,
            },
        }
    )

    # --- Plot 2: T² chart ---
    obs_idx = list(range(1, n + 1))
    result["plots"].append(
        {
            "title": "Hotelling's T² Chart",
            "data": [
                {
                    "type": "scatter",
                    "x": obs_idx,
                    "y": T2.tolist(),
                    "mode": "lines+markers",
                    "name": "T²",
                    "marker": {"color": [COLOR_BAD if ooc else SVEND_COLORS[0] for ooc in ooc_mask], "size": 4},
                    "line": {"color": SVEND_COLORS[0], "width": 1},
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "T²"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 1,
                        "x1": n,
                        "y0": T2_ucl,
                        "y1": T2_ucl,
                        "line": {"color": COLOR_BAD, "dash": "dash", "width": 2},
                    }
                ],
                "annotations": [
                    {
                        "x": n,
                        "y": T2_ucl,
                        "text": f"UCL={T2_ucl:.1f}",
                        "showarrow": False,
                        "font": {"color": COLOR_BAD, "size": 10},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 3: Component capability bars ---
    pc_labels = [f"PC{j + 1}" for j in range(k)]
    result["plots"].append(
        {
            "title": "Per-Component Capability",
            "data": [
                {
                    "type": "bar",
                    "x": pc_labels,
                    "y": component_cpk,
                    "name": "Cpk",
                    "marker": {
                        "color": [
                            COLOR_GOOD if c >= 1.33 else (COLOR_WARNING if c >= 1.0 else COLOR_BAD)
                            for c in component_cpk
                        ]
                    },
                    "text": [f"{c:.2f}" for c in component_cpk],
                    "textposition": "outside",
                },
            ],
            "layout": {
                "height": 280,
                "yaxis": {"title": "Cpk"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": k - 0.5,
                        "y0": 1.33,
                        "y1": 1.33,
                        "line": {"color": COLOR_GOLD, "dash": "dash", "width": 1.5},
                    }
                ],
                "annotations": [
                    {
                        "x": k - 0.5,
                        "y": 1.33,
                        "text": "1.33",
                        "showarrow": False,
                        "font": {"color": COLOR_GOLD},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 4: Correlation heatmap ---
    corr = np.corrcoef(data_matrix, rowvar=False)
    result["plots"].append(
        {
            "title": "Variable Correlation Matrix",
            "data": [
                {
                    "type": "heatmap",
                    "z": corr.tolist(),
                    "x": variables,
                    "y": variables,
                    "colorscale": [[0, "#d06060"], [0.5, "#ffffff"], [1, "#4a9f6e"]],
                    "zmin": -1,
                    "zmax": 1,
                    "text": [[f"{corr[i][j]:.2f}" for j in range(p)] for i in range(p)],
                    "texttemplate": "%{text}",
                    "showscale": True,
                    "colorbar": {"title": "r"},
                }
            ],
            "layout": {"height": 360, "yaxis": {"autorange": "reversed"}},
        }
    )

    # --- Summary ---
    summary = "<<COLOR:title>>D-MULTI — MULTIVARIATE CAPABILITY<</COLOR>>\n\n"
    summary += f"<<COLOR:header>>Data:<</COLOR>> {n} observations, {p} variables\n"
    summary += f"<<COLOR:header>>Variables:<</COLOR>> {', '.join(variables)}\n\n"

    summary += "<<COLOR:header>>PCA Decomposition:<</COLOR>>\n"
    summary += f"  Components retained: {k} of {p} (≥95% variance)\n"
    for j in range(k):
        summary += f"  PC{j + 1}: {explained[j] * 100:.1f}% variance, Cpk = {component_cpk[j]:.3f}\n"
    summary += f"  Cumulative: {cumvar[k - 1] * 100:.1f}%\n"

    summary += "\n<<COLOR:header>>Joint Capability:<</COLOR>>\n"
    summary += f"  MCpk (min component Cpk): {mcpk:.3f}\n"
    summary += f"  T² Capability: {t2_capability * 100:.1f}% within UCL\n"
    summary += f"  T² UCL: {T2_ucl:.2f} (0.27% false alarm rate)\n"
    summary += f"  OOC observations: {n_ooc} of {n} ({n_ooc / n * 100:.1f}%)\n"

    if var_cpk:
        summary += f"\n<<COLOR:header>>Per-Variable Capability (±{tolerance_pct}% tolerance):<</COLOR>>\n"
        for vc in var_cpk:
            summary += f"  {vc['variable']}: Cpk = {vc['cpk']:.3f}\n"

    if mcpk >= 1.33 and n_ooc == 0:
        summary += "\n<<COLOR:success>>Multivariate process is capable and in control.<</COLOR>>"
    elif mcpk >= 1.0:
        summary += f"\n<<COLOR:warning>>Multivariate process is marginally capable (MCpk = {mcpk:.3f}).<</COLOR>>"
    else:
        summary += f"\n<<COLOR:danger>>Multivariate process is NOT capable (MCpk = {mcpk:.3f}).<</COLOR>>"

    if n_ooc > 0:
        summary += (
            f"\n<<COLOR:warning>>{n_ooc} observations exceed T² UCL — investigate multivariate outliers.<</COLOR>>"
        )

    result["summary"] = summary
    result["guide_observation"] = (
        f"D-Multi: MCpk={mcpk:.3f}, T² OOC={n_ooc}/{n}, {k} PCs retain {cumvar[k - 1] * 100:.1f}% variance"
    )
    result["statistics"] = {
        "n": n,
        "p": p,
        "k_components": k,
        "explained_variance": [round(float(e), 4) for e in explained[:k]],
        "component_cpk": component_cpk,
        "mcpk": round(mcpk, 4),
        "t2_ucl": round(T2_ucl, 2),
        "n_ooc": n_ooc,
        "t2_capability": round(t2_capability, 4),
    }

    # --- narrative ---
    if mcpk >= 1.33 and n_ooc == 0:
        verdict = f"Capable & In Control — MCpk = {mcpk:.3f}"
        body = (
            f"The multivariate process is capable (<strong>MCpk = {mcpk:.3f}</strong>) "
            f"with no T² outliers. {k} principal components retain "
            f"{cumvar[k - 1] * 100:.1f}% of the variance across {p} variables."
        )
        nxt = "Process is healthy — continue monitoring."
    elif mcpk >= 1.0:
        verdict = f"Marginally Capable — MCpk = {mcpk:.3f}"
        body = (
            f"MCpk = <strong>{mcpk:.3f}</strong> — above 1.0 but below target. "
            f"{n_ooc} T² outlier{'s' if n_ooc != 1 else ''} detected out of {n} observations. "
            f"{k} PCs retain {cumvar[k - 1] * 100:.1f}% variance."
        )
        nxt = "Identify which variables contribute most to the weakest principal component."
    else:
        verdict = f"Not Capable — MCpk = {mcpk:.3f}"
        body = (
            f"MCpk = <strong>{mcpk:.3f}</strong> — the process is not jointly capable "
            f"across the {p} variables. {n_ooc} T² outlier{'s' if n_ooc != 1 else ''} detected."
        )
        nxt = "Run individual capability studies to identify the weakest variables, then address jointly."
    if n_ooc > 0:
        body += f" <strong>{n_ooc}</strong> observations exceed the T² upper control limit — investigate these multivariate outliers."
    result["narrative"] = _d_narrative(
        f"D-Multi: {verdict}",
        body,
        nxt,
        "The T² chart shows Hotelling's T² statistic per observation — points above "
        "the UCL are multivariate outliers. The component Cpk chart shows capability "
        "on each principal component (the joint minimum is MCpk).",
    )

    result["education"] = {
        "title": "Understanding Multivariate Capability (D-Multi)",
        "content": (
            "<dl>"
            "<dt>Why multivariate?</dt>"
            "<dd>When you have multiple correlated quality characteristics (e.g., length, width, "
            "weight), checking each separately misses the joint picture. A part can pass every "
            "individual spec but still be out of spec <em>jointly</em> because the variables "
            "interact.</dd>"
            "<dt>What is PCA doing here?</dt>"
            "<dd>Principal Component Analysis rotates correlated variables into uncorrelated "
            "'principal components'. We keep enough PCs to explain most of the variance, then "
            "compute KDE-based capability on each — the minimum across PCs is MCpk.</dd>"
            "<dt>What is Hotelling's T²?</dt>"
            "<dd>A multivariate distance measure — how far each observation is from the "
            "centre in all dimensions simultaneously. Points above the UCL are multivariate "
            "outliers, even if they look normal on any single variable.</dd>"
            "<dt>How to interpret MCpk</dt>"
            "<dd><strong>≥ 1.33</strong>: Jointly capable across all dimensions. "
            "<strong>1.0–1.33</strong>: Marginally capable — check which PC is weakest and "
            "trace back to the original variables. <strong>&lt; 1.0</strong>: Not jointly "
            "capable — one or more correlated variable combinations are out of spec.</dd>"
            "</dl>"
        ),
    }

    return result


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


def run_d_type(df, analysis_id, config):
    """Dispatcher for D-Type analyses."""
    if analysis_id == "d_chart":
        return run_d_chart(df, config)
    elif analysis_id == "d_cpk":
        return run_d_cpk(df, config)
    elif analysis_id == "d_nonnorm":
        return run_d_nonnorm(df, config)
    elif analysis_id == "d_equiv":
        return run_d_equiv(df, config)
    elif analysis_id == "d_sig":
        return run_d_sig(df, config)
    elif analysis_id == "d_multi":
        return run_d_multi(df, config)
    else:
        return {
            "plots": [],
            "summary": f"<<COLOR:danger>>Unknown D-Type analysis: {analysis_id}<</COLOR>>",
            "guide_observation": "",
        }
