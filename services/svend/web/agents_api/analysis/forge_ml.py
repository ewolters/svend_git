"""Forge-backed ML analysis handlers.

Wraps legacy ML handlers (sklearn pipelines, boosting, etc.) and
normalizes output to forge result schema.

Object 271 — Analysis Workbench migration.
"""

import logging

logger = logging.getLogger(__name__)

# ML analysis IDs from legacy dispatch
_ML_IDS = [
    "classification",
    "regression_ml",
    "xgboost",
    "lightgbm",
    "model_compare",
    "shap_explain",
    "hyperparameter_tune",
    "clustering",
    "pca",
    "feature",
    "isolation_forest",
    "factor_analysis",
    "bayesian_regression",
    "gam",
    "gaussian_process",
    "pls",
    "regularized_regression",
    "sem",
    "discriminant_analysis",
    "correspondence_analysis",
    "item_analysis",
]


def _wrap_legacy_ml(analysis_id, df, config):
    """Call legacy ML handler, normalize output to forge schema."""
    from .ml import run_ml_analysis

    # Legacy ML handler requires user arg — pass None for forge context
    result = run_ml_analysis(df, analysis_id, config, user=None)
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


# Generate handler functions dynamically
def _make_handler(aid):
    def handler(df, config):
        return _wrap_legacy_ml(aid, df, config)

    handler.__name__ = f"forge_{aid}"
    handler.__doc__ = f"ML handler for {aid} (legacy-wrapped)."
    return handler


FORGE_ML_HANDLERS = {aid: _make_handler(aid) for aid in _ML_IDS}


def run_forge_ml(analysis_id, df, config):
    """Run a forge-backed ML analysis.

    Returns the result dict, or None if not recognized.
    """
    handler = FORGE_ML_HANDLERS.get(analysis_id)
    if handler is None:
        return None
    try:
        return handler(df, config)
    except Exception:
        logger.exception(f"Forge ML handler failed for {analysis_id}, falling back to legacy")
        return None
