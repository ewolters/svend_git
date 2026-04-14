"""Gradient boosting — XGBoost and LightGBM handlers.

CR: 3c0d0e53
"""

import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from ..common import (
    _build_ml_diagnostics,
    _classification_reliability,
    _regression_reliability,
    _stratified_split_3way,
    cache_model,
)

logger = logging.getLogger(__name__)


def _run_boosting(df, analysis_id, config, user):
    """Run XGBoost or LightGBM analysis."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])

    if not target or not features:
        result["summary"] = "Error: target and features are required for boosting models."
        return result

    X = df[features].dropna()
    y = df[target].loc[X.index]

    if analysis_id == "xgboost":
        import xgboost as xgb

        from accounts.constants import can_use_ml
        from agents_api.gpu_manager import GPUTrainingContext

        if not can_use_ml(getattr(user, "tier", "free")):
            result["summary"] = "Error: XGBoost requires a paid tier with ML access."
            return result

        # User-configurable params
        n_estimators = int(config.get("n_estimators", 100))
        max_depth = int(config.get("max_depth", 6))
        learning_rate = float(config.get("learning_rate", 0.1))
        subsample = float(config.get("subsample", 0.8))
        colsample = float(config.get("colsample_bytree", 0.8))

        # Auto-detect task
        task_type = config.get("task_type", "auto")
        if task_type == "auto":
            task_type = (
                "classification"
                if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10))
                else "regression"
            )

        # Encode target
        label_map = None
        y_work = y.copy()
        if task_type == "classification" and (y_work.dtype == object or y_work.dtype.name == "category"):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_work = pd.Series(le.fit_transform(y_work), index=y_work.index)
            label_map = {i: c for i, c in enumerate(le.classes_)}

        # Encode categorical features
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            X_enc[col] = pd.Categorical(X_enc[col]).codes.astype(int)
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

        # 3-way split for conformal prediction
        if task_type == "classification":
            X_train, X_cal, X_test, y_train, y_cal, y_test = _stratified_split_3way(X_enc, y_work)
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X_enc, y_work, test_size=0.30, random_state=42)
            X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

        with GPUTrainingContext() as gpu:
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "random_state": 42,
                "verbosity": 0,
                **gpu.xgb_params(),
            }
            gpu_used = gpu.available

            _use_sample_weight = False
            if task_type == "classification":
                params["use_label_encoder"] = False
                params["eval_metric"] = "logloss"
                # Auto class weighting for imbalanced data
                from collections import Counter as _Counter

                _counts = _Counter(y_work)
                _majority_pct = max(_counts.values()) / len(y_work)
                if _majority_pct > 0.75 and len(_counts) == 2:
                    _neg, _pos = sorted(_counts.values())
                    params["scale_pos_weight"] = _neg / _pos
                elif _majority_pct > 0.75:
                    from sklearn.utils.class_weight import compute_sample_weight

                    _sample_weights = compute_sample_weight("balanced", y_train)
                    _use_sample_weight = True
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)

            if _use_sample_weight:
                model.fit(X_train, y_train, sample_weight=_sample_weights)
            else:
                model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            result["summary"] = (
                f"XGBoost Classification {'(GPU)' if gpu_used else '(CPU)'}\n\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Params: depth={max_depth}, lr={learning_rate}, n_est={n_estimators}\n\n"
                f"{classification_report(y_test, y_pred)}"
            )
            metrics_dict = {"accuracy": float(accuracy)}
            _classification_reliability(y_work, y_test, y_pred, metrics_dict)
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            result["summary"] = (
                f"XGBoost Regression {'(GPU)' if gpu_used else '(CPU)'}\n\n"
                f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\n"
                f"Params: depth={max_depth}, lr={learning_rate}, n_est={n_estimators}"
            )
            metrics_dict = {"r2": float(r2), "rmse": rmse}
            _regression_reliability(y_work, y_test, y_pred, metrics_dict)

        # Full diagnostic suite
        result["plots"].extend(
            _build_ml_diagnostics(
                model,
                X_test,
                y_test,
                y_pred,
                features,
                task_type,
                label_map=label_map,
                model_name=f"XGBoost ({'GPU' if gpu_used else 'CPU'})",
            )
        )

        # Conformal prediction
        conformal_state = None
        try:
            from forgestat.conformal import compute_conformal

            cf = compute_conformal(model, X_cal, y_cal, task_type=task_type)
            conformal_state = cf.get_state()

            if task_type == "regression":
                qhat_90 = cf.qhats.get("0.1", 0)
                y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
                covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
                emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                result["summary"] += (
                    f"\n\nConformal 90% interval: ±{qhat_90:.4f} (empirical coverage: {emp_coverage:.1%}, n_cal={cf.n_cal})"
                )
            else:
                if hasattr(model, "predict_proba"):
                    proba_test = model.predict_proba(X_test)
                    pred_sets, meta = cf.predict_sets(proba_test, alpha=0.10)
                    covered = sum(1 for i, ps in enumerate(pred_sets) if int(y_test.iloc[i]) in ps)
                    emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                    avg_ss = float(np.mean([len(ps) for ps in pred_sets]))
                    result["summary"] += (
                        f"\n\nConformal 90% prediction sets: avg size={avg_ss:.2f}, coverage={emp_coverage:.1%}, n_cal={cf.n_cal}"
                    )

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = conformal_state
        except Exception as e:
            logger.warning(f"Conformal prediction failed for XGBoost: {e}")

        # Cache model
        if user and user.is_authenticated:
            model_key = str(uuid.uuid4())
            cache_meta = {
                "model_type": f"XGBoost ({'GPU' if gpu_used else 'CPU'})",
                "features": features,
                "target": target,
                "metrics": metrics_dict,
            }
            if conformal_state:
                cache_meta["conformal_state"] = conformal_state
                cache_meta["split_seed"] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

        result["guide_observation"] = (
            f"XGBoost {'(GPU)' if gpu_used else '(CPU)'}: {list(metrics_dict.values())[0]:.4f} ({list(metrics_dict.keys())[0]})."
        )

    elif analysis_id == "lightgbm":
        import lightgbm as lgb

        from accounts.constants import can_use_ml
        from agents_api.gpu_manager import GPUTrainingContext

        if not can_use_ml(getattr(user, "tier", "free")):
            result["summary"] = "Error: LightGBM requires a paid tier with ML access."
            return result

        n_estimators = int(config.get("n_estimators", 100))
        num_leaves = int(config.get("num_leaves", 31))
        learning_rate = float(config.get("learning_rate", 0.1))
        subsample = float(config.get("subsample", 0.8))

        task_type = config.get("task_type", "auto")
        if task_type == "auto":
            task_type = (
                "classification"
                if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10))
                else "regression"
            )

        label_map = None
        y_work = y.copy()
        if task_type == "classification" and (y_work.dtype == object or y_work.dtype.name == "category"):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_work = pd.Series(le.fit_transform(y_work), index=y_work.index)
            label_map = {i: c for i, c in enumerate(le.classes_)}

        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            X_enc[col] = pd.Categorical(X_enc[col]).codes.astype(int)
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

        # 3-way split for conformal prediction
        if task_type == "classification":
            X_train, X_cal, X_test, y_train, y_cal, y_test = _stratified_split_3way(X_enc, y_work)
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X_enc, y_work, test_size=0.30, random_state=42)
            X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

        with GPUTrainingContext() as gpu:
            params = {
                "n_estimators": n_estimators,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "random_state": 42,
                "verbosity": -1,
                **gpu.lgb_params(),
            }
            gpu_used = gpu.available

            if task_type == "classification":
                # Auto class weighting for imbalanced data
                from collections import Counter as _Counter

                _counts = _Counter(y_work)
                _majority_pct = max(_counts.values()) / len(y_work)
                if _majority_pct > 0.75:
                    params["is_unbalance"] = True
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            result["summary"] = (
                f"LightGBM Classification {'(GPU)' if gpu_used else '(CPU)'}\n\n"
                f"Accuracy: {accuracy:.4f}\n"
                f"Params: leaves={num_leaves}, lr={learning_rate}, n_est={n_estimators}\n\n"
                f"{classification_report(y_test, y_pred)}"
            )
            metrics_dict = {"accuracy": float(accuracy)}
            _classification_reliability(y_work, y_test, y_pred, metrics_dict)
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            result["summary"] = (
                f"LightGBM Regression {'(GPU)' if gpu_used else '(CPU)'}\n\n"
                f"R²: {r2:.4f}\nRMSE: {rmse:.4f}\n"
                f"Params: leaves={num_leaves}, lr={learning_rate}, n_est={n_estimators}"
            )
            metrics_dict = {"r2": float(r2), "rmse": rmse}
            _regression_reliability(y_work, y_test, y_pred, metrics_dict)

        # Full diagnostic suite
        result["plots"].extend(
            _build_ml_diagnostics(
                model,
                X_test,
                y_test,
                y_pred,
                features,
                task_type,
                label_map=label_map,
                model_name=f"LightGBM ({'GPU' if gpu_used else 'CPU'})",
            )
        )

        # Conformal prediction
        conformal_state = None
        try:
            from forgestat.conformal import compute_conformal

            cf = compute_conformal(model, X_cal, y_cal, task_type=task_type)
            conformal_state = cf.get_state()

            if task_type == "regression":
                qhat_90 = cf.qhats.get("0.1", 0)
                y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
                covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
                emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                result["summary"] += (
                    f"\n\nConformal 90% interval: ±{qhat_90:.4f} (empirical coverage: {emp_coverage:.1%}, n_cal={cf.n_cal})"
                )
            else:
                if hasattr(model, "predict_proba"):
                    proba_test = model.predict_proba(X_test)
                    pred_sets, meta = cf.predict_sets(proba_test, alpha=0.10)
                    covered = sum(1 for i, ps in enumerate(pred_sets) if int(y_test.iloc[i]) in ps)
                    emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                    avg_ss = float(np.mean([len(ps) for ps in pred_sets]))
                    result["summary"] += (
                        f"\n\nConformal 90% prediction sets: avg size={avg_ss:.2f}, coverage={emp_coverage:.1%}, n_cal={cf.n_cal}"
                    )

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = conformal_state
        except Exception as e:
            logger.warning(f"Conformal prediction failed for LightGBM: {e}")

        if user and user.is_authenticated:
            model_key = str(uuid.uuid4())
            cache_meta = {
                "model_type": f"LightGBM ({'GPU' if gpu_used else 'CPU'})",
                "features": features,
                "target": target,
                "metrics": metrics_dict,
            }
            if conformal_state:
                cache_meta["conformal_state"] = conformal_state
                cache_meta["split_seed"] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

        result["guide_observation"] = (
            f"LightGBM {'(GPU)' if gpu_used else '(CPU)'}: {list(metrics_dict.values())[0]:.4f} ({list(metrics_dict.keys())[0]})."
        )

    return result
