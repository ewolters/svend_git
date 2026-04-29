"""PBS handler — Process Belief System analyses via forgepbs + forgeviz.

Routes pbs_belief, pbs_cpk, pbs_full, etc. to forgepbs computation classes,
then builds ForgeViz ChartSpecs from the results.

No inline chart construction. No Plotly. Forge packages do the work.
"""

import math

import numpy as np
import pandas as pd
from forgepbs.capability.bayesian_cpk import bayesian_cpk
from forgepbs.charts.adaptive import AdaptiveControlLimits
from forgepbs.charts.belief import BeliefChart
from forgepbs.charts.edetector import EDetector
from forgepbs.core.forecast import forecast as pbs_forecast
from forgepbs.core.posterior import NormalGammaPosterior
from forgepbs.health.fusion import multi_stream_health
from forgeviz.charts.bayesian import (
    bayesian_capability,
    bayesian_control_chart,
)
from forgeviz.core.spec import ChartSpec

_PBS_IDS = {
    "pbs_belief",
    "pbs_cpk",
    "pbs_cpk_traj",
    "pbs_edetector",
    "pbs_health",
    "pbs_full",
    "pbs_evidence",
    "pbs_adaptive",
    "pbs_predictive",
}


def run(df, analysis_id, config):
    """Dispatch PBS analysis. Returns raw result dict for chain.assemble()."""
    if analysis_id not in _PBS_IDS:
        return {"summary": f"Unknown PBS analysis: {analysis_id}", "charts": [], "statistics": {}}

    col = config.get("column", "")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}

    df[col] = pd.to_numeric(df[col], errors="coerce")
    y = df[col].dropna().values.astype(float)
    n = len(y)
    if n < 5:
        return {"summary": "Error: Need at least 5 observations.", "charts": [], "statistics": {}}

    USL = _safe_float(config.get("USL"))
    LSL = _safe_float(config.get("LSL"))
    target = _safe_float(config.get("target"))

    if USL is not None and LSL is not None and USL <= LSL:
        return {
            "summary": f"Error: USL ({USL}) must be greater than LSL ({LSL}). Check your spec limits.",
            "charts": [],
            "statistics": {},
        }

    # Calibration
    n_cal = min(50, max(10, n // 5))
    cal = y[:n_cal]
    sigma_cal = float(np.std(cal, ddof=1)) if n_cal > 1 else 0.01

    if target is not None:
        mu_0 = target
    elif USL is not None and LSL is not None:
        mu_0 = (USL + LSL) / 2.0
    else:
        mu_0 = float(np.mean(cal))

    prior = NormalGammaPosterior(mu=mu_0, kappa=1.0, alpha=2.0, beta=max(sigma_cal**2 * 2.0, 1e-10))

    hazard_raw = config.get("hazard_lambda")
    hazard_val = _safe_float(hazard_raw)
    if hazard_val is not None and hazard_raw != "auto":
        hazard_lambda = hazard_val
    else:
        hazard_lambda = max(20.0, min(200.0, n / 4.0))

    beta_rob = _safe_float(config.get("beta_robustness")) or 0.0

    dispatch = {
        "pbs_full": lambda: _full(y, prior, USL, LSL, target, hazard_lambda, beta_rob, sigma_cal, mu_0, config),
        "pbs_belief": lambda: _belief(y, prior, hazard_lambda, beta_rob),
        "pbs_edetector": lambda: _edetector(y, mu_0, USL, LSL, sigma_cal, config),
        "pbs_adaptive": lambda: _adaptive(y, prior),
        "pbs_cpk": lambda: _cpk(y, prior, USL, LSL),
        "pbs_cpk_traj": lambda: _cpk_traj(y, prior, USL, LSL),
        "pbs_predictive": lambda: _predictive(y, USL, LSL),
        "pbs_evidence": lambda: _evidence(y, prior, mu_0, sigma_cal),
        "pbs_health": lambda: _health(y, prior, USL, LSL, mu_0, hazard_lambda, beta_rob, sigma_cal),
    }
    return dispatch[analysis_id]()


# ── Individual analyses ─────────────────────────────────────────────────


def _belief(y, prior, hazard_lambda, beta_rob):
    bc = BeliefChart(hazard_rate=1.0 / hazard_lambda, prior=prior.copy(), beta_robustness=beta_rob)
    points = bc.process_batch(y)

    shift_probs = [p.shift_probability for p in points]
    ts = list(range(len(y)))

    spec = ChartSpec(
        title="Belief Chart — P(Process Shifted)",
        chart_type="belief_chart",
        x_axis={"label": "Observation"},
        y_axis={"label": "P(shift)", "min_val": 0, "max_val": 1.05},
    )
    spec.add_trace(ts, shift_probs, name="P(shift)", color="#d94a4a", width=2)
    spec.add_reference_line(0.5, color="#d4a24a", dash="dotted", label="Watch (50%)")
    spec.add_reference_line(0.95, color="#d94a4a", dash="dashed", label="Alarm (95%)")

    ooc = [i for i, p in enumerate(shift_probs) if p >= 0.95]
    if ooc:
        spec.add_marker(ooc, color="#d94a4a", size=6, symbol="circle", label="Alarm")

    last_sp = shift_probs[-1] if shift_probs else 0
    status = "ALARM" if last_sp >= 0.95 else "CAUTION" if last_sp >= 0.5 else "STABLE"

    return {
        "charts": [spec],
        "statistics": {
            "shift_probability": round(last_sp, 4),
            "status": status,
            "n_observations": len(y),
            "hazard_lambda": hazard_lambda,
        },
        "summary": f"Process belief: {status}. Current P(shift) = {last_sp:.3f} at observation {len(y)}.",
    }


def _edetector(y, mu_0, USL, LSL, sigma_cal, config):
    if USL is not None and LSL is not None:
        bounds = (float(LSL), float(USL))
    else:
        bounds = (mu_0 - 4.0 * max(sigma_cal, 1e-6), mu_0 + 4.0 * max(sigma_cal, 1e-6))

    alpha = float(config.get("edetector_alpha", 0.05))
    ed = EDetector(mu_0=mu_0, bounds=bounds, alpha=alpha)
    points = ed.process_batch(y)

    ts = list(range(len(y)))
    # Combined log evidence = max of upper and lower
    log_Ns = [max(p.log_e_upper, p.log_e_lower) for p in points]
    threshold = ed.log_threshold

    spec = ChartSpec(
        title="E-Detector — Distribution-Free Changepoint",
        chart_type="e_detector",
        x_axis={"label": "Observation"},
        y_axis={"label": "log(E)"},
    )
    spec.add_trace(ts, log_Ns, name="log(E)", color="#4a9f6e", width=2)
    spec.add_reference_line(threshold, color="#d94a4a", dash="dashed", label=f"Threshold (1/α = {1 / alpha:.0f})")
    spec.add_reference_line(0, color="#666666", dash="dotted", label="Baseline")

    alarms = [i for i, p in enumerate(points) if p.alert]
    if alarms:
        spec.add_marker(alarms, color="#d94a4a", size=8, symbol="square", label="Alarm")

    peak_idx = max(range(len(log_Ns)), key=lambda i: log_Ns[i]) if log_Ns else 0
    has_alarm = bool(alarms)

    return {
        "charts": [spec],
        "statistics": {
            "peak_log_E": round(log_Ns[peak_idx], 3) if log_Ns else 0,
            "peak_observation": peak_idx,
            "threshold": round(threshold, 3),
            "alarm_detected": has_alarm,
            "n_alarms": len(alarms),
        },
        "summary": f"E-Detector: {'ALARM — changepoint detected' if has_alarm else 'No changepoint detected'}. Peak log(E) = {log_Ns[peak_idx]:.1f}.",
    }


def _adaptive(y, prior):
    acl = AdaptiveControlLimits(prior=prior.copy())
    points = acl.process_batch(y)

    data_pts = [p.value for p in points]
    cls = [p.cl for p in points]
    ucls = [p.ucl for p in points]
    lcls = [p.lcl for p in points]

    chart = bayesian_control_chart(
        data=data_pts,
        posterior_ucl=ucls,
        posterior_cl=cls,
        posterior_lcl=lcls,
        title="Adaptive Control Limits",
    )

    ooc = [i for i in range(len(y)) if y[i] > ucls[i] or y[i] < lcls[i]]

    return {
        "charts": [chart],
        "statistics": {
            "final_cl": round(cls[-1], 4) if cls else 0,
            "final_ucl": round(ucls[-1], 4) if ucls else 0,
            "final_lcl": round(lcls[-1], 4) if lcls else 0,
            "n_ooc": len(ooc),
            "ooc_rate": round(len(ooc) / len(y), 4) if len(y) else 0,
        },
        "summary": f"Adaptive limits: CL={cls[-1]:.3f}, {len(ooc)} OOC in {len(y)} observations.",
    }


def _cpk(y, prior, USL, LSL):
    if USL is None or LSL is None:
        return {"summary": "Error: USL and LSL required for capability.", "charts": [], "statistics": {}}

    post = prior.copy()
    post.update(y)
    result = bayesian_cpk(post, lsl=LSL, usl=USL)

    chart = bayesian_capability(
        cpk_samples=result.samples,
        cpk_mean=result.mean,
        cpk_ci_lower=result.ci_lower,
        cpk_ci_upper=result.ci_upper,
        target_cpk=1.33,
        title="Bayesian Cpk Posterior",
    )

    return {
        "charts": [chart],
        "statistics": {
            "cpk_mean": round(result.mean, 4),
            "cpk_median": round(result.median, 4),
            "cpk_ci_lower": round(result.ci_lower, 4),
            "cpk_ci_upper": round(result.ci_upper, 4),
            "p_above_133": round(result.p_above_133, 4),
            "p_above_1": round(result.p_above_1, 4),
        },
        "summary": f"Bayesian Cpk = {result.mean:.3f} [{result.ci_lower:.3f}, {result.ci_upper:.3f}]. "
        f"P(Cpk ≥ 1.33) = {result.p_above_133:.1%}.",
    }


def _cpk_traj(y, prior, USL, LSL):
    if USL is None or LSL is None:
        return {"summary": "Error: USL and LSL required.", "charts": [], "statistics": {}}

    step = max(1, len(y) // 50)
    cpk_means, cpk_lows, cpk_highs, ts = [], [], [], []

    for i in range(step, len(y) + 1, step):
        p = prior.copy()
        p.update(y[:i])
        r = bayesian_cpk(p, lsl=LSL, usl=USL, n_samples=2000, seed=42 + i)
        cpk_means.append(r.mean)
        cpk_lows.append(r.ci_lower)
        cpk_highs.append(r.ci_upper)
        ts.append(i)

    spec = ChartSpec(
        title="Cpk Trajectory",
        chart_type="cpk_trajectory",
        x_axis={"label": "Observations"},
        y_axis={"label": "Cpk"},
    )
    spec.add_trace(ts, cpk_means, name="Cpk (mean)", color="#4a9f6e", width=2)
    spec.add_trace(ts, cpk_highs, name="95% CI upper", color="#4a9f6e", width=1, dash="dotted", opacity=0.5)
    spec.add_trace(ts, cpk_lows, name="95% CI lower", color="#4a9f6e", width=1, dash="dotted", opacity=0.5)
    spec.add_reference_line(1.33, color="#d4a24a", dash="dashed", label="Target (1.33)")

    return {
        "charts": [spec],
        "statistics": {
            "final_cpk": round(cpk_means[-1], 4) if cpk_means else 0,
            "ci_width_initial": round(cpk_highs[0] - cpk_lows[0], 4) if cpk_highs else 0,
            "ci_width_final": round(cpk_highs[-1] - cpk_lows[-1], 4) if cpk_highs else 0,
        },
        "summary": f"Cpk trajectory: {cpk_means[-1]:.3f} after {len(y)} obs. CI narrowed from "
        f"{cpk_highs[0] - cpk_lows[0]:.3f} to {cpk_highs[-1] - cpk_lows[-1]:.3f}.",
    }


def _predictive(y, USL, LSL):
    if len(y) < 10:
        return {"summary": "Error: Need ≥10 observations for prediction.", "charts": [], "statistics": {}}

    # Build posterior from data, then forecast
    post = NormalGammaPosterior(
        mu=float(np.mean(y[:10])), kappa=1.0, alpha=2.0, beta=max(float(np.var(y[:10])) * 2, 1e-10)
    )
    post.update(y)

    horizon = min(25, len(y))
    envelope = pbs_forecast(post, horizon=horizon, level=0.90)

    spec = ChartSpec(
        title="Predictive Chart",
        chart_type="predictive",
        x_axis={"label": "Observation"},
        y_axis={"label": "Value"},
    )
    spec.add_trace(list(range(len(y))), list(y), name="Data", color="#4a9f6e", width=1.5, marker_size=3)

    if envelope.points:
        fan_x = [len(y) - 1 + p.horizon for p in envelope.points]
        spec.add_trace(
            fan_x, [p.mean for p in envelope.points], name="Predicted", color="#e8c547", width=1.5, dash="dashed"
        )
        spec.add_trace(
            fan_x,
            [p.upper for p in envelope.points],
            name="90% upper",
            color="#d94a4a",
            width=1,
            dash="dotted",
            opacity=0.6,
        )
        spec.add_trace(
            fan_x,
            [p.lower for p in envelope.points],
            name="90% lower",
            color="#d94a4a",
            width=1,
            dash="dotted",
            opacity=0.6,
        )

    if USL is not None:
        spec.add_reference_line(USL, color="#d94a4a", dash="dashed", label=f"USL={USL}")
    if LSL is not None:
        spec.add_reference_line(LSL, color="#d94a4a", dash="dashed", label=f"LSL={LSL}")

    # Check if prediction bounds exceed spec
    exceeds = False
    if envelope.points and (USL is not None or LSL is not None):
        for p in envelope.points:
            if USL is not None and p.upper > USL:
                exceeds = True
            if LSL is not None and p.lower < LSL:
                exceeds = True

    return {
        "charts": [spec],
        "statistics": {
            "posterior_mean": round(envelope.posterior_mean, 4),
            "posterior_std": round(envelope.posterior_std, 4),
            "prediction_exceeds_spec": exceeds,
            "horizon": horizon,
        },
        "summary": f"Predictive: {len(y)} obs, {horizon}-step forecast. "
        f"{'Prediction bounds exceed spec limits.' if exceeds else 'Within predicted bounds.'}",
    }


def _evidence(y, prior, mu_0, sigma_cal):
    from forgepbs.charts.belief import EvidenceAccumulation

    ea = EvidenceAccumulation(mu_0=mu_0, sigma_ref=sigma_cal)
    points = ea.process_batch(y)

    ts = list(range(len(y)))
    log_es = [p.log_e_accumulated for p in points]

    spec = ChartSpec(
        title="Evidence Accumulation",
        chart_type="evidence_accumulation",
        x_axis={"label": "Observation"},
        y_axis={"label": "log(E-value)"},
    )
    spec.add_trace(ts, log_es, name="log(E)", color="#6ab7d4", width=2)
    spec.add_reference_line(math.log(20), color="#4a9f6e", dash="dashed", label="Strong (20:1)")
    spec.add_reference_line(math.log(3), color="#d4a24a", dash="dotted", label="Moderate (3:1)")

    last_pt = points[-1] if points else None
    level = last_pt.evidence_level if last_pt else "none"
    peak_e = last_pt.e_value if last_pt else 1

    return {
        "charts": [spec],
        "statistics": {
            "peak_e_value": round(peak_e, 3),
            "peak_log_e": round(log_es[-1], 3) if log_es else 0,
            "evidence_level": level,
        },
        "summary": f"Evidence: E = {peak_e:.1f} ({level}). "
        f"{'Process shifted from baseline.' if peak_e >= 20 else 'No strong evidence of shift.'}",
    }


def _health(y, prior, USL, LSL, mu_0, hazard_lambda, beta_rob, sigma_cal):
    belief = _belief(y, prior, hazard_lambda, beta_rob)
    edet = _edetector(y, mu_0, USL, LSL, sigma_cal, {})

    h_spc = 1 - belief["statistics"]["shift_probability"]
    h_edet = 0.0 if edet["statistics"]["alarm_detected"] else 1.0
    h_cpk = 0.5
    if USL is not None and LSL is not None:
        cpk_r = _cpk(y, prior, USL, LSL)
        h_cpk = cpk_r["statistics"].get("p_above_133", 0.5)

    streams = {"spc": h_spc, "edetector": h_edet, "cpk": h_cpk}
    health = multi_stream_health(streams)
    score = health.score if hasattr(health, "score") else float(health)

    spec = ChartSpec(
        title="Process Health",
        chart_type="health",
        x_axis={"label": "Stream"},
        y_axis={"label": "Health", "min_val": 0, "max_val": 1},
    )
    names = list(streams.keys())
    values = [streams[k] for k in names]
    spec.add_trace(names, values, name="Health", trace_type="bar", color="#4a9f6e")
    spec.add_reference_line(0.5, color="#d4a24a", dash="dashed", label="Watch")

    return {
        "charts": [spec],
        "statistics": {"overall_health": round(score, 4), **{f"{k}_health": round(v, 4) for k, v in streams.items()}},
        "summary": f"Process health: {score:.1%}. SPC={h_spc:.2f}, E-Det={h_edet:.2f}, Cpk={h_cpk:.2f}.",
    }


def _full(y, prior, USL, LSL, target, hazard_lambda, beta_rob, sigma_cal, mu_0, config):
    charts = []
    all_stats = {}

    for name, fn in [
        ("belief", lambda: _belief(y, prior, hazard_lambda, beta_rob)),
        ("edetector", lambda: _edetector(y, mu_0, USL, LSL, sigma_cal, config)),
        ("adaptive", lambda: _adaptive(y, prior)),
    ]:
        r = fn()
        charts.extend(r["charts"])
        all_stats.update({f"{name}_{k}": v for k, v in r["statistics"].items()})

    if USL is not None and LSL is not None:
        for name, fn in [
            ("cpk", lambda: _cpk(y, prior, USL, LSL)),
            ("cpk_traj", lambda: _cpk_traj(y, prior, USL, LSL)),
        ]:
            r = fn()
            charts.extend(r["charts"])
            all_stats.update({f"{name}_{k}": v for k, v in r["statistics"].items()})

    if len(y) >= 10:
        r = _predictive(y, USL, LSL)
        charts.extend(r["charts"])

    r = _evidence(y, prior, mu_0, sigma_cal)
    charts.extend(r["charts"])
    all_stats.update({f"evidence_{k}": v for k, v in r["statistics"].items()})

    sp = all_stats.get("belief_shift_probability", 0)
    status = all_stats.get("belief_status", "UNKNOWN")

    return {
        "charts": charts,
        "statistics": all_stats,
        "_layout": {"mode": "compose"},
        "summary": f"PBS Full: {status}. P(shift) = {sp:.3f}, {len(y)} observations.",
    }


def _safe_float(val):
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
