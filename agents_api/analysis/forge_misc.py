"""Forge-backed miscellaneous analysis handlers.

Wraps legacy handlers for: simulation, causal, drift, d_type,
anytime, quality_econ, pbs, ishap, bayes_msa, viz, reliability.

Object 271 — Analysis Workbench migration.
"""

import logging

logger = logging.getLogger(__name__)


def _wrap_legacy(module_path, func_name, df, analysis_id, config, **kwargs):
    """Call a legacy handler by module path and normalize output."""
    import importlib

    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Cannot import {module_path}.{func_name}: {e}")
        return None

    result = func(df, analysis_id, config, **kwargs)
    if result is None:
        return None

    result.setdefault("plots", [])
    result.setdefault("statistics", {})
    result.setdefault("summary", "")
    result.setdefault(
        "narrative",
        {
            "verdict": result.get("guide_observation", ""),
            "body": result.get("summary", ""),
            "next_steps": "",
            "chart_guidance": "",
        },
    )
    result.setdefault("assumptions", {})
    result.setdefault("diagnostics", [])
    result.setdefault("guide_observation", "")
    return result


# =============================================================================
# D-Type (divergence-based metrics)
# =============================================================================

_D_TYPE_IDS = ["d_chart", "d_cpk", "d_nonnorm", "d_equiv", "d_sig", "d_multi"]


def run_forge_d_type(analysis_id, df, config):
    if analysis_id not in _D_TYPE_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.d_type", "run_d_type", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge d_type failed for {analysis_id}")
        return None


# =============================================================================
# Simulation
# =============================================================================

_SIMULATION_IDS = ["monte_carlo", "tolerance_stackup", "variance_propagation"]


def run_forge_simulation(analysis_id, df, config):
    if analysis_id not in _SIMULATION_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.simulation", "run_simulation", df, analysis_id, config, user=None)
    except Exception:
        logger.exception(f"Forge simulation failed for {analysis_id}")
        return None


# =============================================================================
# Causal
# =============================================================================

_CAUSAL_IDS = ["causal_pc", "causal_lingam"]


def run_forge_causal(analysis_id, df, config):
    if analysis_id not in _CAUSAL_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.causal", "run_causal_discovery", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge causal failed for {analysis_id}")
        return None


# =============================================================================
# Drift
# =============================================================================

_DRIFT_IDS = ["drift_report"]


def run_forge_drift(analysis_id, df, config):
    if analysis_id not in _DRIFT_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.drift", "run_drift_detection", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge drift failed for {analysis_id}")
        return None


# =============================================================================
# Anytime-valid
# =============================================================================

_ANYTIME_IDS = ["anytime_valid", "anytime_onesample", "anytime_ab"]


def run_forge_anytime(analysis_id, df, config):
    if analysis_id not in _ANYTIME_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.anytime", "run_anytime_valid", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge anytime failed for {analysis_id}")
        return None


# =============================================================================
# Quality Economics
# =============================================================================

_QUALITY_ECON_IDS = [
    "taguchi_loss",
    "process_decision",
    "acceptance_decision",
    "cost_of_quality",
    "quality_econ",
    "lot_sentencing",
]


def run_forge_quality_econ(analysis_id, df, config):
    if analysis_id not in _QUALITY_ECON_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.quality_econ", "run_quality_econ", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge quality_econ failed for {analysis_id}")
        return None


# =============================================================================
# PBS (Process Belief System)
# =============================================================================

_PBS_IDS = [
    "pbs",
    "process_belief",
    "belief_chart",
    "pbs_belief",
    "pbs_cpk",
    "pbs_cpk_traj",
    "pbs_edetector",
    "pbs_evidence",
    "pbs_full",
    "pbs_health",
    "pbs_adaptive",
    "pbs_predictive",
]


def run_forge_pbs(analysis_id, df, config):
    if analysis_id not in _PBS_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.pbs", "run_pbs", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge pbs failed for {analysis_id}")
        return None


# =============================================================================
# Interventional SHAP
# =============================================================================

_ISHAP_IDS = ["interventional_shap", "ishap"]


def run_forge_ishap(analysis_id, df, config):
    if analysis_id not in _ISHAP_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.ishap", "run_interventional_shap", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge ishap failed for {analysis_id}")
        return None


# =============================================================================
# Bayesian MSA
# =============================================================================

_BAYES_MSA_IDS = ["bayes_msa", "msa_gage_rr"]


def run_forge_bayes_msa(analysis_id, df, config):
    if analysis_id not in _BAYES_MSA_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.msa", "run_bayes_msa", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge bayes_msa failed for {analysis_id}")
        return None


# =============================================================================
# Reliability (standalone — not survival in advanced.py)
# =============================================================================

_RELIABILITY_IDS = [
    "reliability_weibull",
    "weibull",
    "kaplan_meier",
    "lognormal",
    "exponential",
    "accelerated_life",
    "competing_risks",
    "repairable_systems",
    "distribution_id",
    "warranty",
    "reliability_test_plan",
]


def run_forge_reliability(analysis_id, df, config):
    if analysis_id not in _RELIABILITY_IDS:
        return None
    try:
        return _wrap_legacy("agents_api.analysis.reliability", "run_reliability_analysis", df, analysis_id, config)
    except Exception:
        logger.exception(f"Forge reliability failed for {analysis_id}")
        return None
