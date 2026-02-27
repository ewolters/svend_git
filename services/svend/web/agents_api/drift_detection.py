"""
Concept Drift Detection — three-lane diagnostic suite for ML model health.

Lanes:
    A) Data drift      — PSI per feature (+ optional ADWIN on key features)
    B) Prediction drift — PSI on predicted scores, ADWIN on score mean
    C) Error drift      — ADWIN on loss stream, Page-Hinkley on loss stream

Detectors:
    ADWIN       — Adaptive windowing, detects changes in expectation of a bounded stream
    Page-Hinkley — Cumulative deviation from running mean, good for sustained shifts
    PSI          — Population Stability Index, discretized KL-like divergence

Dependencies: numpy, scipy
"""

import logging
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PSI_BINS = 10
_PSI_EPSILON = 1e-4  # smoothing to avoid log(0)
_PSI_LOW = 0.10
_PSI_MODERATE = 0.20
_PSI_HIGH = 0.25
_ADWIN_DELTA = 0.002  # confidence parameter (lower = fewer false alarms)
_PH_DELTA = 0.005     # tolerance for Page-Hinkley (minimum change to detect)
_PH_LAMBDA = 50       # threshold for Page-Hinkley alarm


# ===========================================================================
# Main entry point
# ===========================================================================
def run_drift_detection(df, analysis_id, config):
    """
    Dispatch to the appropriate drift analysis.

    analysis_id:
        drift_report  — full 3-lane diagnostic
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "drift_report":
        return _run_drift_report(df, config, result)
    else:
        result["summary"] = f"Error: Unknown drift analysis '{analysis_id}'."
        return result


def _run_drift_report(df, config, result):
    """Full 3-lane drift diagnostic."""

    features = config.get("features", [])
    target = config.get("target", "")
    prediction_col = config.get("prediction_col", "")
    split_point = config.get("split_point", None)
    split_pct = float(config.get("split_pct", 50))

    # Determine reference / current windows
    if not features:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target and prediction from feature list
        features = [f for f in features if f != target and f != prediction_col]

    n = len(df)
    if split_point is not None:
        split_idx = int(split_point)
    else:
        split_idx = int(n * split_pct / 100)

    if split_idx < 10 or split_idx > n - 10:
        result["summary"] = (
            f"Error: Split point {split_idx} leaves too few observations in one window. "
            f"Need at least 10 in each window (n={n})."
        )
        return result

    ref_df = df.iloc[:split_idx]
    cur_df = df.iloc[split_idx:]

    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append(f"<<COLOR:title>>CONCEPT DRIFT DIAGNOSTIC<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")
    lines.append(f"<<COLOR:highlight>>Total observations:<</COLOR>> {n}")
    lines.append(f"<<COLOR:highlight>>Reference window:<</COLOR>> rows 0–{split_idx-1} (n={split_idx})")
    lines.append(f"<<COLOR:highlight>>Current window:<</COLOR>> rows {split_idx}–{n-1} (n={n - split_idx})")
    lines.append(f"<<COLOR:highlight>>Features monitored:<</COLOR>> {len(features)}")

    any_drift = False
    drift_banners = []

    # ─── LANE A: Data Drift (features) ───
    lines.append(f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>")
    lines.append(f"<<COLOR:title>>LANE A — DATA DRIFT (Feature Distributions)<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n")

    psi_results = []
    for feat in features:
        ref_vals = ref_df[feat].dropna().values.astype(float)
        cur_vals = cur_df[feat].dropna().values.astype(float)
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue
        psi_val, bin_contributions = _compute_psi(ref_vals, cur_vals)
        severity = _psi_severity(psi_val)
        psi_results.append({
            "feature": feat, "psi": psi_val, "severity": severity,
            "ref_mean": float(np.mean(ref_vals)), "cur_mean": float(np.mean(cur_vals)),
            "ref_std": float(np.std(ref_vals)), "cur_std": float(np.std(cur_vals)),
            "bin_contributions": bin_contributions,
        })

    # Sort by PSI descending
    psi_results.sort(key=lambda x: x["psi"], reverse=True)

    lines.append(f"{'Feature':<20} {'PSI':>8} {'Severity':<10} {'Ref μ':>10} {'Cur μ':>10} {'Δμ':>10}")
    lines.append(f"{'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for pr in psi_results:
        delta_mu = pr["cur_mean"] - pr["ref_mean"]
        icon = "●" if pr["severity"] == "high" else "◐" if pr["severity"] == "moderate" else "○"
        lines.append(
            f"{icon} {pr['feature']:<18} {pr['psi']:>8.4f} {pr['severity']:<10} "
            f"{pr['ref_mean']:>10.3f} {pr['cur_mean']:>10.3f} {delta_mu:>+10.3f}"
        )
        if pr["severity"] in ("high", "moderate"):
            any_drift = True
            drift_banners.append({
                "detector": "PSI", "stream": f"feature:{pr['feature']}",
                "severity": pr["severity"],
                "detail": f"PSI={pr['psi']:.4f}, Δμ={delta_mu:+.3f}",
            })

    # ADWIN on top-drifting features
    adwin_feat_results = []
    for pr in psi_results[:3]:  # top 3 by PSI
        feat = pr["feature"]
        vals = df[feat].dropna().values.astype(float)
        adwin_result = _adwin_detect(vals, stream_name=f"feature:{feat}")
        adwin_feat_results.append(adwin_result)
        if adwin_result["detected"]:
            any_drift = True

    if adwin_feat_results:
        lines.append(f"\n<<COLOR:highlight>>ADWIN on top features:<</COLOR>>")
        for ar in adwin_feat_results:
            status = "DRIFT" if ar["detected"] else "stable"
            lines.append(
                f"  {ar['stream']}: {status}"
                + (f" — change at obs {ar['change_idx']}, "
                   f"μ_before={ar['mean_before']:.3f}, μ_after={ar['mean_after']:.3f}, "
                   f"Δ={ar['shift_magnitude']:+.3f}" if ar["detected"] else "")
            )

    # PSI bar chart
    if psi_results:
        result["plots"].append(_build_psi_bar(psi_results, "Data Drift — PSI per Feature"))

    # ─── LANE B: Prediction Drift ───
    if prediction_col and prediction_col in df.columns:
        lines.append(f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>")
        lines.append(f"<<COLOR:title>>LANE B — PREDICTION DRIFT (Model Output Distribution)<</COLOR>>")
        lines.append(f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n")

        ref_preds = ref_df[prediction_col].dropna().values.astype(float)
        cur_preds = cur_df[prediction_col].dropna().values.astype(float)

        if len(ref_preds) >= 5 and len(cur_preds) >= 5:
            pred_psi, pred_bins = _compute_psi(ref_preds, cur_preds)
            pred_severity = _psi_severity(pred_psi)
            lines.append(f"<<COLOR:highlight>>PSI on predictions:<</COLOR>> {pred_psi:.4f} ({pred_severity})")
            lines.append(f"  Ref mean: {np.mean(ref_preds):.4f}, Cur mean: {np.mean(cur_preds):.4f}")

            if pred_severity in ("high", "moderate"):
                any_drift = True
                drift_banners.append({
                    "detector": "PSI", "stream": "prediction",
                    "severity": pred_severity,
                    "detail": f"PSI={pred_psi:.4f}",
                })

            # ADWIN on prediction stream
            all_preds = df[prediction_col].dropna().values.astype(float)
            pred_adwin = _adwin_detect(all_preds, stream_name="prediction")
            if pred_adwin["detected"]:
                any_drift = True
                lines.append(
                    f"<<COLOR:warning>>ADWIN: prediction mean shifted at obs {pred_adwin['change_idx']}<</COLOR>>"
                    f" — μ_before={pred_adwin['mean_before']:.4f}, "
                    f"μ_after={pred_adwin['mean_after']:.4f}"
                )
                drift_banners.append({
                    "detector": "ADWIN", "stream": "prediction",
                    "severity": "high" if abs(pred_adwin["shift_magnitude"]) > 0.5 else "moderate",
                    "detail": f"change at obs {pred_adwin['change_idx']}, Δ={pred_adwin['shift_magnitude']:+.4f}",
                })
            else:
                lines.append(f"<<COLOR:good>>ADWIN: no significant prediction drift detected<</COLOR>>")

            # KDE comparison plot
            result["plots"].append(_build_distribution_comparison(
                ref_preds, cur_preds, prediction_col,
                f"Prediction Drift — {prediction_col}"
            ))
    else:
        lines.append(f"\n<<COLOR:text>>LANE B (Prediction Drift): skipped — no prediction column specified<</COLOR>>")

    # ─── LANE C: Error Drift ───
    has_error = target and prediction_col and target in df.columns and prediction_col in df.columns
    if has_error:
        lines.append(f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>")
        lines.append(f"<<COLOR:title>>LANE C — ERROR DRIFT (Model Performance Over Time)<</COLOR>>")
        lines.append(f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n")

        actual = df[target].values.astype(float)
        predicted = df[prediction_col].values.astype(float)
        residuals = actual - predicted
        loss = residuals ** 2  # squared error

        # ADWIN on loss stream
        loss_adwin = _adwin_detect(loss, stream_name="squared_error")
        if loss_adwin["detected"]:
            any_drift = True
            lines.append(
                f"<<COLOR:warning>>ADWIN on loss: error level shifted at obs {loss_adwin['change_idx']}<</COLOR>>\n"
                f"  MSE_before={loss_adwin['mean_before']:.4f}, MSE_after={loss_adwin['mean_after']:.4f}, "
                f"Δ={loss_adwin['shift_magnitude']:+.4f}"
            )
            drift_banners.append({
                "detector": "ADWIN", "stream": "squared_error",
                "severity": "high",
                "detail": f"MSE shifted at obs {loss_adwin['change_idx']}, "
                          f"Δ={loss_adwin['shift_magnitude']:+.4f}",
            })
        else:
            lines.append(f"<<COLOR:good>>ADWIN on loss: no significant error drift<</COLOR>>")

        # Page-Hinkley on loss stream
        ph_up = _page_hinkley_detect(loss, direction="up", stream_name="squared_error")
        ph_down = _page_hinkley_detect(loss, direction="down", stream_name="squared_error")

        for ph in [ph_up, ph_down]:
            if ph["detected"]:
                any_drift = True
                lines.append(
                    f"<<COLOR:warning>>Page-Hinkley ({ph['direction']}): sustained error shift "
                    f"starting ~obs {ph['change_idx']}<</COLOR>>\n"
                    f"  Cumulative stat: {ph['cumulative_value']:.4f}, "
                    f"threshold λ={ph['lambda']:.1f}"
                )
                drift_banners.append({
                    "detector": "Page-Hinkley", "stream": f"squared_error ({ph['direction']})",
                    "severity": "high",
                    "detail": f"sustained {ph['direction']} drift from obs {ph['change_idx']}",
                })
            else:
                lines.append(
                    f"<<COLOR:good>>Page-Hinkley ({ph['direction']}): no sustained error shift<</COLOR>>"
                )

        # Ref vs current error stats
        ref_loss = loss[:split_idx]
        cur_loss = loss[split_idx:]
        lines.append(f"\n<<COLOR:highlight>>Error summary:<</COLOR>>")
        lines.append(f"  Ref MSE: {np.mean(ref_loss):.4f} ± {np.std(ref_loss):.4f}")
        lines.append(f"  Cur MSE: {np.mean(cur_loss):.4f} ± {np.std(cur_loss):.4f}")
        lines.append(f"  Ref RMSE: {np.sqrt(np.mean(ref_loss)):.4f}")
        lines.append(f"  Cur RMSE: {np.sqrt(np.mean(cur_loss)):.4f}")

        # Rolling loss plot
        result["plots"].append(_build_rolling_loss_plot(
            loss, split_idx, loss_adwin, ph_up, ph_down
        ))
    else:
        lines.append(f"\n<<COLOR:text>>LANE C (Error Drift): skipped — need both target and prediction columns<</COLOR>>")

    # ─── DRIFT BANNERS ───
    if drift_banners:
        banner_lines = [f"\n<<COLOR:accent>>{'═' * 70}<</COLOR>>"]
        banner_lines.append(f"<<COLOR:warning>>DRIFT DETECTED — {len(drift_banners)} signal(s)<</COLOR>>")
        banner_lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")
        for db in drift_banners:
            banner_lines.append(
                f"  [{db['detector']}] {db['stream']} — {db['severity'].upper()} — {db['detail']}"
            )
        # Prepend banners to top of summary
        lines = banner_lines + ["\n"] + lines
    else:
        no_drift = [f"\n<<COLOR:good>>No significant drift detected across all lanes.<</COLOR>>\n"]
        lines = no_drift + lines

    result["summary"] = "\n".join(lines)

    # Statistics
    result["statistics"] = {
        "n_total": n,
        "n_reference": split_idx,
        "n_current": n - split_idx,
        "n_features": len(features),
        "drift_detected": any_drift,
        "n_drift_signals": len(drift_banners),
        "drift_banners": drift_banners,
        "psi_per_feature": {
            pr["feature"]: {"psi": round(pr["psi"], 4), "severity": pr["severity"]}
            for pr in psi_results
        },
    }

    n_high = sum(1 for db in drift_banners if db["severity"] == "high")
    n_mod = sum(1 for db in drift_banners if db["severity"] == "moderate")
    result["guide_observation"] = (
        f"Drift diagnostic: {len(drift_banners)} signals "
        f"({n_high} high, {n_mod} moderate) across {len(features)} features. "
        + ("Model may need retraining." if n_high > 0 else
           "Minor drift — monitor closely." if n_mod > 0 else
           "No evidence of drift — model appears healthy.")
    )

    return result


# ===========================================================================
# ADWIN — Adaptive Windowing
# ===========================================================================
def _adwin_detect(stream, delta=_ADWIN_DELTA, stream_name=""):
    """
    Simplified ADWIN: detect change in expectation of a bounded stream.

    Scans for the split point that maximizes evidence of mean change.
    Uses Hoeffding-style bound: ε = sqrt(1/(2m) * ln(4n/δ)) where m = min(n1,n2).

    Input stream is normalized to [0,1] for bounded guarantees.
    """
    x = np.asarray(stream, dtype=float)
    n = len(x)

    if n < 20:
        return {"detected": False, "stream": stream_name,
                "change_idx": None, "mean_before": None, "mean_after": None,
                "shift_magnitude": None, "window_length": n}

    # Normalize to [0,1] for Hoeffding bound validity
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_range = x_max - x_min
    if x_range < 1e-12:
        return {"detected": False, "stream": stream_name,
                "change_idx": None, "mean_before": None, "mean_after": None,
                "shift_magnitude": None, "window_length": n}

    x_norm = (x - x_min) / x_range

    # Cumulative sums for O(n) mean computation
    cumsum = np.cumsum(x_norm)

    best_cut = None
    best_evidence = 0.0

    # Check possible cut points (skip edges)
    min_window = max(10, n // 20)
    for t in range(min_window, n - min_window):
        n1 = t
        n2 = n - t
        mean1 = cumsum[t - 1] / n1
        mean2 = (cumsum[-1] - cumsum[t - 1]) / n2

        # Hoeffding bound
        m = min(n1, n2)
        epsilon = np.sqrt(0.5 / m * np.log(4 * n / delta))

        diff = abs(mean1 - mean2)
        evidence = diff - epsilon

        if evidence > best_evidence:
            best_evidence = evidence
            best_cut = t

    if best_cut is not None and best_evidence > 0:
        mean_before_raw = float(np.mean(x[:best_cut]))
        mean_after_raw = float(np.mean(x[best_cut:]))
        return {
            "detected": True,
            "stream": stream_name,
            "change_idx": int(best_cut),
            "mean_before": round(mean_before_raw, 6),
            "mean_after": round(mean_after_raw, 6),
            "shift_magnitude": round(mean_after_raw - mean_before_raw, 6),
            "window_length": n,
            "epsilon": round(float(np.sqrt(0.5 / min(best_cut, n - best_cut) * np.log(4 * n / delta))), 6),
        }
    else:
        return {
            "detected": False,
            "stream": stream_name,
            "change_idx": None,
            "mean_before": round(float(np.mean(x)), 6),
            "mean_after": None,
            "shift_magnitude": None,
            "window_length": n,
        }


# ===========================================================================
# Page-Hinkley — Cumulative deviation detector
# ===========================================================================
def _page_hinkley_detect(stream, direction="up", ph_delta=_PH_DELTA,
                         ph_lambda=_PH_LAMBDA, stream_name=""):
    """
    Page-Hinkley test for sustained mean shift.

    For upward drift: track Σ(x_t - x̄_t - δ), alarm when max - current > λ
    For downward drift: track Σ(x̄_t - x_t - δ), alarm when max - current > λ

    Returns change index (first time alarm triggered).
    """
    x = np.asarray(stream, dtype=float)
    n = len(x)

    if n < 20:
        return {"detected": False, "direction": direction, "stream": stream_name,
                "change_idx": None, "cumulative_value": 0,
                "delta": ph_delta, "lambda": ph_lambda}

    # Standardize for comparable thresholds
    x_std = (x - np.mean(x)) / (np.std(x) + 1e-12)

    cumulative = 0.0
    min_cumulative = 0.0
    running_sum = 0.0
    change_idx = None

    for t in range(n):
        running_sum += x_std[t]
        running_mean = running_sum / (t + 1)

        if direction == "up":
            cumulative += x_std[t] - running_mean - ph_delta
        else:
            cumulative += running_mean - x_std[t] - ph_delta

        min_cumulative = min(min_cumulative, cumulative)

        if cumulative - min_cumulative > ph_lambda:
            change_idx = t
            break

    return {
        "detected": change_idx is not None,
        "direction": direction,
        "stream": stream_name,
        "change_idx": int(change_idx) if change_idx is not None else None,
        "cumulative_value": round(float(cumulative), 4),
        "min_cumulative": round(float(min_cumulative), 4),
        "delta": ph_delta,
        "lambda": ph_lambda,
    }


# ===========================================================================
# PSI — Population Stability Index
# ===========================================================================
def _compute_psi(reference, current, n_bins=_PSI_BINS):
    """
    Compute PSI between reference and current distributions.

    Binning: quantile-based from reference distribution.
    Smoothing: epsilon to avoid log(0).

    Returns (psi_value, bin_contributions).
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Quantile-based bins from reference
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        # Fallback to equal-width bins
        bin_edges = np.linspace(ref.min(), ref.max(), n_bins + 1)

    # Compute proportions
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_props = ref_counts / len(ref) + _PSI_EPSILON
    cur_props = cur_counts / len(cur) + _PSI_EPSILON

    # Normalize after epsilon
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    # PSI = Σ (p_i - q_i) * ln(p_i / q_i)
    bin_psi = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi = float(np.sum(bin_psi))

    bin_contributions = [
        {"bin_low": float(bin_edges[i]), "bin_high": float(bin_edges[i + 1]),
         "ref_prop": float(ref_props[i]), "cur_prop": float(cur_props[i]),
         "psi_contribution": float(bin_psi[i])}
        for i in range(len(bin_psi))
    ]

    return psi, bin_contributions


def _psi_severity(psi_val):
    if psi_val >= _PSI_HIGH:
        return "high"
    elif psi_val >= _PSI_MODERATE:
        return "moderate"
    elif psi_val >= _PSI_LOW:
        return "low"
    return "negligible"


# ===========================================================================
# Plotting
# ===========================================================================
def _build_psi_bar(psi_results, title):
    """Horizontal bar chart of PSI per feature."""
    names = [pr["feature"] for pr in psi_results]
    values = [pr["psi"] for pr in psi_results]
    colors = [
        "rgba(208,96,96,0.7)" if pr["severity"] == "high" else
        "rgba(200,170,60,0.7)" if pr["severity"] == "moderate" else
        "rgba(150,150,150,0.5)" if pr["severity"] == "low" else
        "rgba(74,159,110,0.5)"
        for pr in psi_results
    ]

    return {
        "title": title,
        "data": [{
            "type": "bar", "orientation": "h",
            "x": values, "y": names,
            "marker": {"color": colors},
            "hovertemplate": "%{y}: PSI=%{x:.4f}<extra></extra>",
        }],
        "layout": {
            "template": "plotly_dark",
            "height": max(200, len(names) * 25),
            "xaxis": {"title": "Population Stability Index"},
            "yaxis": {"automargin": True},
            "shapes": [
                {"type": "line", "x0": _PSI_LOW, "x1": _PSI_LOW,
                 "y0": -0.5, "y1": len(names) - 0.5,
                 "line": {"color": "rgba(200,170,60,0.4)", "dash": "dot", "width": 1}},
                {"type": "line", "x0": _PSI_HIGH, "x1": _PSI_HIGH,
                 "y0": -0.5, "y1": len(names) - 0.5,
                 "line": {"color": "rgba(208,96,96,0.4)", "dash": "dot", "width": 1}},
            ],
            "annotations": [
                {"x": _PSI_LOW, "y": len(names) - 0.5, "text": "low", "showarrow": False,
                 "yshift": 10, "font": {"size": 9, "color": "#aaa"}},
                {"x": _PSI_HIGH, "y": len(names) - 0.5, "text": "high", "showarrow": False,
                 "yshift": 10, "font": {"size": 9, "color": "#d96060"}},
            ],
        },
    }


def _build_distribution_comparison(ref_vals, cur_vals, col_name, title):
    """Overlaid histograms comparing reference vs current distributions."""
    return {
        "title": title,
        "data": [
            {"type": "histogram", "x": ref_vals.tolist(), "name": "Reference",
             "opacity": 0.6, "marker": {"color": "rgba(74,159,110,0.6)"},
             "nbinsx": 30},
            {"type": "histogram", "x": cur_vals.tolist(), "name": "Current",
             "opacity": 0.6, "marker": {"color": "rgba(208,96,96,0.6)"},
             "nbinsx": 30},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "barmode": "overlay",
            "xaxis": {"title": col_name},
            "yaxis": {"title": "Count"},
        },
    }


def _build_rolling_loss_plot(loss, split_idx, adwin_result, ph_up, ph_down):
    """Rolling loss plot with split line and detected change points."""
    n = len(loss)
    window = max(10, n // 50)
    # Rolling mean of loss
    rolling = np.convolve(loss, np.ones(window) / window, mode='valid')
    x_rolling = list(range(window - 1, n))

    traces = [
        {"type": "scatter", "x": x_rolling, "y": rolling.tolist(),
         "mode": "lines", "line": {"color": "#4a9f6e", "width": 1.5},
         "name": f"Rolling MSE (w={window})"},
    ]

    shapes = [
        {"type": "line", "x0": split_idx, "x1": split_idx,
         "y0": 0, "y1": float(np.max(rolling)) * 1.1,
         "line": {"color": "rgba(255,255,255,0.3)", "dash": "dash"},
        }
    ]

    annotations = [
        {"x": split_idx, "y": float(np.max(rolling)) * 1.05,
         "text": "ref|cur", "showarrow": False,
         "font": {"size": 10, "color": "#aaa"}},
    ]

    # Mark ADWIN change point
    if adwin_result["detected"]:
        shapes.append({
            "type": "line",
            "x0": adwin_result["change_idx"], "x1": adwin_result["change_idx"],
            "y0": 0, "y1": float(np.max(rolling)) * 1.1,
            "line": {"color": "#d94a4a", "width": 2},
        })
        annotations.append({
            "x": adwin_result["change_idx"],
            "y": float(np.max(rolling)) * 0.95,
            "text": "ADWIN", "showarrow": True, "arrowhead": 2,
            "font": {"size": 10, "color": "#d94a4a"},
        })

    # Mark Page-Hinkley change points
    for ph, color in [(ph_up, "#e67e22"), (ph_down, "#3498db")]:
        if ph["detected"]:
            shapes.append({
                "type": "line",
                "x0": ph["change_idx"], "x1": ph["change_idx"],
                "y0": 0, "y1": float(np.max(rolling)) * 1.1,
                "line": {"color": color, "width": 1.5, "dash": "dashdot"},
            })
            annotations.append({
                "x": ph["change_idx"],
                "y": float(np.max(rolling)) * 0.85,
                "text": f"PH-{ph['direction']}", "showarrow": True, "arrowhead": 2,
                "font": {"size": 9, "color": color},
            })

    return {
        "title": "Error Drift — Rolling Loss Over Time",
        "data": traces,
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": "Observation"},
            "yaxis": {"title": "Squared Error (rolling mean)"},
            "shapes": shapes,
            "annotations": annotations,
        },
    }
