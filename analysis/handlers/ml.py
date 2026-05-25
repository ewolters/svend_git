"""ML handler — 21 machine learning analyses via forgeml + forgeviz.

Routes all ml/* registry entries to forgeml functions.
No sklearn imports here — forgeml owns the computation.
"""

import logging

import numpy as np
import pandas as pd
from forgeviz.charts.generic import bar
from forgeviz.charts.scatter import scatter

logger = logging.getLogger(__name__)


def run(df, analysis_id, config):
    """Dispatch ML analysis to forgeml, return raw result for chain.assemble()."""
    dispatch = {
        "classification": _classification,
        "regression_ml": _regression_ml,
        "model_compare": _model_compare,
        "xgboost": _xgboost,
        "lightgbm": _lightgbm,
        "shap_explain": _shap_explain,
        "hyperparameter_tune": _hyperparameter_tune,
        "clustering": _clustering,
        "pca": _pca,
        "feature": _feature_importance,
        "bayesian_regression": _bayesian_regression,
        "gam": _gam,
        "isolation_forest": _isolation_forest,
        "gaussian_process": _gaussian_process,
        "pls": _pls,
        "sem": _sem,
        "regularized_regression": _regularized_regression,
        "discriminant_analysis": _discriminant_analysis,
        "factor_analysis": _factor_analysis,
        "correspondence_analysis": _correspondence_analysis,
        "item_analysis": _item_analysis,
    }

    fn = dispatch.get(analysis_id)
    if fn:
        return fn(df, config)

    return {"summary": f"ML analysis '{analysis_id}' not recognized.", "charts": [], "statistics": {}}


# ── Helpers ─────────────────────────────────────────────────────────────


def _get_X_y(df, config):
    """Extract feature matrix X and target y from config."""
    target = config.get("target")
    features = config.get("features", [])
    if isinstance(features, str):
        features = [features]
    if not features:
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()
    y = df[target].loc[X.index] if target else None
    return X, y, features


def _get_X(df, config):
    """Extract feature matrix X for unsupervised analyses."""
    features = config.get("features", [])
    if isinstance(features, str):
        features = [features]
    if not features:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()
    return X, features


def _importance_chart(importance, title="Feature Importance"):
    """Build a bar chart from a feature importance dict."""
    if not importance:
        return None
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]
    names = [x[0] for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]
    return bar(names, vals, title=title)


def _err(msg):
    return {"summary": f"Error: {msg}", "charts": [], "statistics": {}}


# ── Supervised ──────────────────────────────────────────────────────────


def _classification(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.supervised import classify

    result = classify(X, y, algorithm=config.get("algorithm", "rf"))
    charts = []
    chart = _importance_chart(result.feature_importance, "Classification — Feature Importance")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": {**result.statistics, "algorithm": result.algorithm},
        "summary": f"Classification ({result.algorithm}): accuracy={result.statistics.get('accuracy', '?')}",
    }


def _regression_ml(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.supervised import regress

    result = regress(X, y, algorithm=config.get("algorithm", "rf"))
    charts = []
    chart = _importance_chart(result.feature_importance, "Regression — Feature Importance")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": {**result.statistics, "algorithm": result.algorithm},
        "summary": f"ML Regression ({result.algorithm}): R²={result.statistics.get('r_squared', '?')}, RMSE={result.statistics.get('rmse', '?')}",
    }


def _discriminant_analysis(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.supervised import discriminant_analysis

    method = config.get("method", "lda")
    result = discriminant_analysis(X, y, method=method)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Discriminant Analysis ({method.upper()}): accuracy={result.statistics.get('accuracy', '?')}",
    }


# ── Boosting ────────────────────────────────────────────────────────────


def _xgboost(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    try:
        from forgeml.boosting import xgboost_analysis
    except ImportError:
        return _err("xgboost not installed.")
    result = xgboost_analysis(X, y, task_type=config.get("task_type", "auto"))
    charts = []
    chart = _importance_chart(result.feature_importance, "XGBoost — Feature Importance")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": result.statistics,
        "summary": f"XGBoost: {result.statistics}",
    }


def _lightgbm(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    try:
        from forgeml.boosting import lightgbm_analysis
    except ImportError:
        return _err("lightgbm not installed.")
    result = lightgbm_analysis(X, y, task_type=config.get("task_type", "auto"))
    charts = []
    chart = _importance_chart(result.feature_importance, "LightGBM — Feature Importance")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": result.statistics,
        "summary": f"LightGBM: {result.statistics}",
    }


# ── Explanation ─────────────────────────────────────────────────────────


def _shap_explain(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.explanation import shap_explain

    result = shap_explain(X, y, task_type=config.get("task_type", "auto"))
    charts = []
    chart = _importance_chart(result.feature_importance, "SHAP Feature Importance")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": {**result.statistics, "feature_importance": result.feature_importance},
        "summary": f"SHAP ({result.algorithm}): top feature = {list(result.feature_importance.keys())[0] if result.feature_importance else 'none'}",
    }


def _model_compare(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.explanation import model_compare

    result = model_compare(X, y, task_type=config.get("task_type", "auto"), cv_folds=int(config.get("cv_folds", 5)))
    charts = []
    names = [r["model"] for r in result.results]
    scores = [r["mean_score"] for r in result.results]
    charts.append(bar(names, scores, title=f"Model Comparison ({result.metric})"))
    return {
        "charts": charts,
        "statistics": {
            "results": result.results,
            "best_model": result.best_model,
            "best_score": result.best_score,
            "metric": result.metric,
        },
        "summary": f"Best model: {result.best_model} ({result.metric}={result.best_score})",
    }


def _hyperparameter_tune(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import hyperparameter_tune

    result = hyperparameter_tune(
        X,
        y,
        task_type=config.get("task_type", "auto"),
        n_iter=int(config.get("n_iter", 20)),
        cv=int(config.get("cv_folds", 5)),
    )
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Best score: {result.statistics.get('best_score', '?')} with {result.statistics.get('best_params', {})}",
    }


# ── Unsupervised ────────────────────────────────────────────────────────


def _clustering(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import cluster

    n_clusters = int(config.get("k") or config.get("n_clusters") or 3)
    result = cluster(X, n_clusters=n_clusters)
    sil = result.statistics.get("silhouette", 0)
    charts = []
    if len(features) >= 2:
        x_vals = X[features[0]].tolist()
        y_vals = X[features[1]].tolist()
        charts.append(
            scatter(
                x_vals,
                y_vals,
                x_label=features[0],
                y_label=features[1],
                title=f"Clusters (k={n_clusters}, silhouette={sil:.3f})",
            )
        )
    return {
        "charts": charts,
        "statistics": result.statistics,
        "summary": f"K-Means: {n_clusters} clusters, silhouette={sil:.3f}",
    }


def _pca(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import pca

    n_comp = config.get("n_components")
    result = pca(X, n_components=int(n_comp) if n_comp else None)
    evr = result.statistics.get("explained_variance_ratio", [])
    charts = []
    if evr:
        charts.append(bar([f"PC{i + 1}" for i in range(len(evr))], evr, title="PCA — Explained Variance"))
    return {
        "charts": charts,
        "statistics": result.statistics,
        "summary": f"PCA: {result.statistics.get('n_components')} components, cumulative={result.statistics.get('cumulative_variance', '?')}",
    }


def _feature_importance(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.unsupervised import feature_importance

    result = feature_importance(X, y, n_top=int(config.get("n_top", 10)))
    charts = []
    chart = _importance_chart(result.feature_importance, "Feature Importance (Random Forest)")
    if chart:
        charts.append(chart)
    return {
        "charts": charts,
        "statistics": {**result.statistics, "importance": result.feature_importance},
        "summary": f"Top feature: {list(result.feature_importance.keys())[0] if result.feature_importance else 'none'}",
    }


def _isolation_forest(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import isolation_forest

    contamination = float(config.get("contamination", 0.1))
    result = isolation_forest(X, contamination=contamination)
    n_out = result.statistics.get("n_outliers", 0)
    charts = []
    if len(features) >= 2:
        x_vals = X[features[0]].tolist()
        y_vals = X[features[1]].tolist()
        charts.append(
            scatter(
                x_vals, y_vals, x_label=features[0], y_label=features[1], title=f"Isolation Forest — {n_out} outliers"
            )
        )
    return {
        "charts": charts,
        "statistics": result.statistics,
        "summary": f"Isolation Forest: {n_out}/{result.statistics.get('n_total', '?')} outliers ({result.statistics.get('outlier_fraction', 0) * 100:.1f}%)",
    }


def _factor_analysis(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import factor_analysis

    n_factors = config.get("n_factors")
    result = factor_analysis(X, n_factors=int(n_factors) if n_factors else None)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Factor Analysis: {result.statistics.get('n_factors')} factors extracted",
    }


def _correspondence_analysis(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import correspondence_analysis

    result = correspondence_analysis(X)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Correspondence Analysis: inertia={result.statistics.get('total_inertia', '?')}, dim1={result.statistics.get('dim1_pct', '?')}%",
    }


def _item_analysis(df, config):
    X, features = _get_X(df, config)
    from forgeml.unsupervised import item_analysis

    result = item_analysis(X)
    alpha = result.statistics.get("cronbach_alpha", 0)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Item Analysis: Cronbach's α={alpha:.3f}, {result.statistics.get('n_items')} items",
    }


# ── Advanced Regression ─────────────────────────────────────────────────


def _bayesian_regression(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import bayesian_regression

    result = bayesian_regression(X, y)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"Bayesian Ridge: R²={result.statistics.get('r_squared', '?')}",
    }


def _gam(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import gam

    result = gam(X, y)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"GAM: R²={result.statistics.get('r_squared', '?')}, RMSE={result.statistics.get('rmse', '?')}",
    }


def _gaussian_process(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import gaussian_process

    result = gaussian_process(X, y)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"GP Regression: R²={result.statistics.get('r_squared', '?')}, kernel={result.statistics.get('kernel', '?')}",
    }


def _pls(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import pls

    n_comp = config.get("n_components")
    result = pls(X, y, n_components=int(n_comp) if n_comp else None)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"PLS: R²={result.statistics.get('r_squared', '?')}, {result.statistics.get('n_components')} components",
    }


def _regularized_regression(df, config):
    X, y, features = _get_X_y(df, config)
    if y is None:
        return _err("Select a target variable.")
    from forgeml.tuning import regularized_regression

    method = config.get("method", "lasso")
    alpha = float(config.get("alpha", 1.0))
    result = regularized_regression(X, y, method=method, alpha=alpha)
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"{method.title()}: R²={result.statistics.get('r_squared', '?')}, {result.statistics.get('n_nonzero_coefficients', '?')} non-zero coefficients",
    }


def _sem(df, config):
    outcome = config.get("outcome") or config.get("target")
    predictors = config.get("predictors") or config.get("features", [])
    mediator = config.get("mediator")
    model_type = config.get("model_type", "path")
    if not outcome or not predictors:
        return _err("Select outcome and predictors.")
    from forgeml.tuning import sem_analysis

    result = sem_analysis(df, outcome, predictors, mediator=mediator, model_type=model_type)
    if result.diagnostics:
        return {"charts": [], "statistics": result.statistics, "summary": result.diagnostics[0]}
    return {
        "charts": [],
        "statistics": result.statistics,
        "summary": f"SEM ({model_type}): model fitted with {len(predictors)} predictors → {outcome}",
    }
