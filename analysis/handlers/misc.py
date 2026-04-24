"""Misc handlers — thin wrappers for smaller analysis types.

Covers: simulation, drift, anytime, quality_econ, ishap, bayes_msa,
d_type, reliability, siop. Each is a small function calling the
respective forge package.
"""

import logging

import numpy as np
import pandas as pd
from forgeviz.core.spec import ChartSpec

logger = logging.getLogger(__name__)


# ── Reliability ──────────────────────────────────────────────────────────


def run_reliability(df, analysis_id, config):
    col = config.get("time") or config.get("var") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a time/measurement column.", "charts": [], "statistics": {}}

    data = pd.to_numeric(df[col], errors="coerce").dropna().values

    try:
        if analysis_id == "weibull":
            from forgestat.reliability.distributions import weibull_fit

            result = weibull_fit(data)
            from forgeviz.charts.reliability import weibull_probability_plot

            chart = weibull_probability_plot(data, shape=result.shape, scale=result.scale)
            return {
                "charts": [chart],
                "statistics": {
                    "shape": round(result.shape, 4),
                    "scale": round(result.scale, 4),
                    "mttf": round(result.mttf, 2) if hasattr(result, "mttf") else None,
                },
                "summary": f"Weibull: β={result.shape:.3f}, η={result.scale:.1f}",
            }

        elif analysis_id == "kaplan_meier":
            from forgestat.reliability.survival import kaplan_meier

            event_col = config.get("event")
            events = (
                pd.to_numeric(df[event_col], errors="coerce").values
                if event_col and event_col in df.columns
                else np.ones(len(data))
            )
            result = kaplan_meier(data, events)
            from forgeviz.charts.reliability import survival_curve

            chart = survival_curve(result.times, result.survival, title="Kaplan-Meier Survival")
            return {
                "charts": [chart],
                "statistics": {"median_survival": round(result.median, 2) if hasattr(result, "median") else None},
                "summary": "Kaplan-Meier survival analysis.",
            }
    except Exception as e:
        logger.exception("Reliability error: %s", analysis_id)
        return {"summary": f"Reliability error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"Reliability '{analysis_id}' not yet migrated.", "charts": [], "statistics": {}}


# ── Quality Economics ────────────────────────────────────────────────────


def run_quality_econ(df, analysis_id, config):
    try:
        from forgestat.quality.economics import taguchi_loss

        if analysis_id == "taguchi_loss":
            col = config.get("column") or config.get("var")
            data = (
                pd.to_numeric(df[col], errors="coerce").dropna().values if col and col in df.columns else np.array([])
            )
            target = float(config.get("target", 0))
            delta0 = float(config.get("delta0", 1))
            cost = float(config.get("cost_at_limit", 100))
            result = taguchi_loss(data, target=target, delta0=delta0, cost_at_limit=cost)
            stats = (
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
                if hasattr(result, "__dict__")
                else {"loss": result}
            )
            return {"charts": [], "statistics": stats, "summary": f"Taguchi loss: {stats}"}

        elif analysis_id == "cost_of_quality":
            prevention = float(config.get("prevention", 0))
            appraisal = float(config.get("appraisal", 0))
            internal = float(config.get("internal_failure", 0))
            external = float(config.get("external_failure", 0))
            revenue = float(config.get("revenue", 1))
            total = prevention + appraisal + internal + external
            pct = total / revenue * 100 if revenue > 0 else 0
            stats = {
                "prevention": prevention,
                "appraisal": appraisal,
                "internal_failure": internal,
                "external_failure": external,
                "total_coq": round(total, 2),
                "coq_pct_revenue": round(pct, 2),
            }
            spec = ChartSpec(
                title="Cost of Quality", chart_type="coq", x_axis={"label": "Category"}, y_axis={"label": "$"}
            )
            spec.add_trace(
                ["Prevention", "Appraisal", "Internal", "External"],
                [prevention, appraisal, internal, external],
                name="COQ",
                trace_type="bar",
                color="#4a9f6e",
            )
            return {"charts": [spec], "statistics": stats, "summary": f"COQ: ${total:,.0f} ({pct:.1f}% of revenue)"}

    except Exception as e:
        logger.exception("Quality econ error: %s", analysis_id)
        return {"summary": f"Quality economics error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"Quality econ '{analysis_id}' not yet migrated.", "charts": [], "statistics": {}}


# ── Simulation ───────────────────────────────────────────────────────────


def run_simulation(df, analysis_id, config):
    return {"summary": f"Simulation '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


# ── Drift ────────────────────────────────────────────────────────────────


def run_drift(df, analysis_id, config):
    return {"summary": f"Drift '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


# ── Anytime ──────────────────────────────────────────────────────────────


def run_anytime(df, analysis_id, config):
    return {"summary": f"Anytime '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


# ── iSHAP ────────────────────────────────────────────────────────────────


def run_ishap(df, analysis_id, config):
    return {"summary": f"iSHAP '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


# ── Bayesian MSA ─────────────────────────────────────────────────────────


def run_bayes_msa(df, analysis_id, config):
    return {
        "summary": f"Bayesian MSA '{analysis_id}' not yet in forge-native dispatch.",
        "charts": [],
        "statistics": {},
    }


# ── D-Type ───────────────────────────────────────────────────────────────


def run_d_type(df, analysis_id, config):
    return {"summary": f"D-Type '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}


# ── SIOP ─────────────────────────────────────────────────────────────────


def run_siop(df, analysis_id, config):
    return {"summary": f"SIOP '{analysis_id}' not yet in forge-native dispatch.", "charts": [], "statistics": {}}
