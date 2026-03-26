"""ML analysis dispatcher.

Routes analysis_id to the appropriate sub-module.

CR: 3c0d0e53
"""

import logging

from .advanced_regression import _run_advanced_regression
from .boosting import _run_boosting
from .model_tools import _run_model_tools
from .multivariate import _run_multivariate
from .supervised import _run_supervised
from .unsupervised import _run_unsupervised

logger = logging.getLogger(__name__)

_SUPERVISED = {"classification", "regression_ml"}
_BOOSTING = {"xgboost", "lightgbm"}
_MODEL_TOOLS = {"model_compare", "shap_explain", "hyperparameter_tune"}
_UNSUPERVISED = {"clustering", "pca", "feature", "isolation_forest", "factor_analysis"}
_ADVANCED_REG = {
    "bayesian_regression",
    "gam",
    "gaussian_process",
    "pls",
    "regularized_regression",
}
_MULTIVARIATE = {
    "sem",
    "discriminant_analysis",
    "correspondence_analysis",
    "item_analysis",
}


def run_ml_analysis(df, analysis_id, config, user):
    """Run ML analysis — dispatches to sub-module by analysis_id."""
    if analysis_id in _SUPERVISED:
        return _run_supervised(df, analysis_id, config, user)
    if analysis_id in _BOOSTING:
        return _run_boosting(df, analysis_id, config, user)
    if analysis_id in _MODEL_TOOLS:
        return _run_model_tools(df, analysis_id, config, user)
    if analysis_id in _UNSUPERVISED:
        return _run_unsupervised(df, analysis_id, config, user)
    if analysis_id in _ADVANCED_REG:
        return _run_advanced_regression(df, analysis_id, config, user)
    if analysis_id in _MULTIVARIATE:
        return _run_multivariate(df, analysis_id, config, user)

    logger.warning(f"Unknown ML analysis_id: {analysis_id}")
    return {
        "plots": [],
        "summary": f"Unknown ML analysis: {analysis_id}",
        "guide_observation": "",
    }
