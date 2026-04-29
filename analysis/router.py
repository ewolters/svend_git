"""Analysis router — maps (type, analysis_id) to handler functions.

Each handler: f(df, analysis_id, config) → raw result dict
The raw result is then passed through chain.assemble() for the 10-key contract.

Handlers live in analysis/handlers/<type>.py. Each module exposes a `run()`
function or a dict of {analysis_id: callable} for fine-grained routing.
"""

import importlib
import logging

from . import chain

logger = logging.getLogger(__name__)

# Maps analysis type → module path under analysis.handlers
# Each module must expose: run(df, analysis_id, config) → dict
_HANDLER_REGISTRY = {
    "pbs": "analysis.handlers.pbs",
    "stats": "analysis.handlers.stats",
    "spc": "analysis.handlers.spc",
    "bayesian": "analysis.handlers.bayesian",
    "viz": "analysis.handlers.viz",
    "causal": "analysis.handlers.causal",
    "reliability": ("analysis.handlers.misc", "run_reliability"),
    "quality_econ": ("analysis.handlers.misc", "run_quality_econ"),
    "simulation": ("analysis.handlers.misc", "run_simulation"),
    "drift": ("analysis.handlers.misc", "run_drift"),
    "anytime": ("analysis.handlers.misc", "run_anytime"),
    "ishap": ("analysis.handlers.misc", "run_ishap"),
    "bayes_msa": ("analysis.handlers.misc", "run_bayes_msa"),
    "d_type": ("analysis.handlers.misc", "run_d_type"),
    "siop": ("analysis.handlers.misc", "run_siop"),
}


def dispatch(analysis_type, analysis_id, df, config):
    """Route an analysis request to the correct handler + chain.

    Args:
        analysis_type: e.g. "pbs", "stats", "spc"
        analysis_id: e.g. "pbs_belief", "ttest", "capability"
        df: pandas DataFrame with the data
        config: dict of user configuration

    Returns:
        Dict conforming to the 10-key contract, or None if type not registered.

    Raises:
        None — returns None for unregistered types so caller can fall back.
    """
    entry = _HANDLER_REGISTRY.get(analysis_type)
    if not entry:
        return None  # not yet migrated — caller falls back to old dispatch

    try:
        if isinstance(entry, tuple):
            module_path, func_name = entry
            mod = importlib.import_module(module_path)
            handler = getattr(mod, func_name)
        else:
            mod = importlib.import_module(entry)
            handler = mod.run
        raw = handler(df, analysis_id, config)
        raw["_config"] = config
        return chain.assemble(raw, analysis_type, analysis_id)
    except Exception:
        logger.exception("Handler failed: %s/%s", analysis_type, analysis_id)
        return {
            "summary": f"Analysis failed: {analysis_type}/{analysis_id}",
            "plots": [],
            "statistics": {},
            "narrative": {"verdict": "Analysis failed.", "body": "", "next_steps": "", "chart_guidance": ""},
            "education": None,
            "diagnostics": [],
            "assumptions": {},
            "evidence_grade": None,
            "bayesian_shadow": None,
            "guide_observation": "",
            "what_if": None,
            "_analysis_type": analysis_type,
            "_analysis_id": analysis_id,
        }


def is_registered(analysis_type):
    """Check if a type has been migrated to the new router."""
    return analysis_type in _HANDLER_REGISTRY
