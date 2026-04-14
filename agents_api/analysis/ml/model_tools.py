"""Model tools — comparison, SHAP explainability, hyperparameter tuning.

CR: 3c0d0e53
"""

import logging
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from ..common import _stratified_split_3way, cache_model, get_cached_model

logger = logging.getLogger(__name__)


def _run_model_tools(df, analysis_id, config, user):
    """Run model tool analysis (model_compare, shap_explain, hyperparameter_tune)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])

    if not target:
        result["summary"] = "Error: Please select a target variable."
        return result
    if not features:
        result["summary"] = "Error: Please select at least one feature."
        return result

    X = df[features].dropna()
    y = df[target].loc[X.index]

    # 3-way split for conformal prediction: train 70% / calibration 15% / test 15%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    if analysis_id == "model_compare":
        import time as _time

        from sklearn.model_selection import cross_validate as sk_cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from accounts.constants import can_use_ml

        if not can_use_ml(getattr(user, "tier", "free")):
            result["summary"] = "Error: Model comparison requires a paid tier with ML access."
            return result

        cv_folds = int(config.get("cv_folds", 5))
        task_type = config.get("task_type", "auto")

        # Auto-detect task type
        if task_type == "auto":
            if y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10):
                task_type = "classification"
            else:
                task_type = "regression"

        # Encode categorical target for classification
        if task_type == "classification" and (y.dtype == object or y.dtype.name == "category"):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_enc = pd.Series(le.fit_transform(y), index=y.index)
        else:
            y_enc = y

        # Encode categorical features
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            X_enc[col] = pd.Categorical(X_enc[col]).codes.astype(int)

        # Fill NaN
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

        # Build model roster
        models = []
        if task_type == "classification":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB

            models = [
                (
                    "Random Forest",
                    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                ),
                (
                    "Logistic Regression",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=1000)),
                        ]
                    ),
                ),
                ("LDA", LinearDiscriminantAnalysis()),
                ("Naive Bayes", GaussianNB()),
            ]
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
            }
            primary_metric = "accuracy"
        else:
            from sklearn.linear_model import (
                BayesianRidge,
                ElasticNet,
                Lasso,
                LinearRegression,
                Ridge,
            )

            models = [
                (
                    "Random Forest",
                    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                ),
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge(alpha=1.0)),
                ("LASSO", Lasso(alpha=0.1)),
                ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                ("Bayesian Ridge", BayesianRidge()),
            ]
            scoring = {
                "r2": "r2",
                "neg_rmse": "neg_root_mean_squared_error",
                "neg_mae": "neg_mean_absolute_error",
            }
            primary_metric = "r2"

        # Try adding XGBoost / LightGBM if installed
        try:
            import xgboost as xgb

            if task_type == "classification":
                models.append(
                    (
                        "XGBoost",
                        xgb.XGBClassifier(
                            n_estimators=100,
                            use_label_encoder=False,
                            eval_metric="logloss",
                            random_state=42,
                            verbosity=0,
                        ),
                    )
                )
            else:
                models.append(
                    (
                        "XGBoost",
                        xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                    )
                )
        except ImportError:
            pass

        try:
            import lightgbm as lgb

            if task_type == "classification":
                models.append(
                    (
                        "LightGBM",
                        lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1),
                    )
                )
            else:
                models.append(
                    (
                        "LightGBM",
                        lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1),
                    )
                )
        except ImportError:
            pass

        # Cross-validate each model
        comparison = []
        best_score = -999
        best_model_name = ""
        best_model_obj = None

        for name, mdl in models:
            try:
                t0 = _time.time()
                cv_results = sk_cross_validate(
                    mdl,
                    X_enc,
                    y_enc,
                    cv=cv_folds,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                )
                fit_time = _time.time() - t0

                row = {"model": name, "fit_time": round(fit_time, 2)}
                for metric_name, scorer_key in scoring.items():
                    test_key = f"test_{metric_name}"
                    train_key = f"train_{metric_name}"
                    vals = cv_results.get(test_key, [])
                    train_vals = cv_results.get(train_key, [])
                    mean_val = float(np.mean(vals)) if len(vals) > 0 else 0
                    std_val = float(np.std(vals)) if len(vals) > 0 else 0

                    # Handle negative metrics
                    if "neg_" in scorer_key:
                        mean_val = -mean_val
                        train_vals = -np.array(train_vals) if len(train_vals) > 0 else train_vals

                    row[metric_name] = round(mean_val, 4)
                    row[f"{metric_name}_std"] = round(std_val, 4)
                    row[f"{metric_name}_train"] = round(float(np.mean(train_vals)), 4) if len(train_vals) > 0 else None

                comparison.append(row)

                # Track best
                score_val = row.get(primary_metric, 0)
                if score_val > best_score:
                    best_score = score_val
                    best_model_name = name
                    best_model_obj = mdl
            except Exception as e_model:
                logger.warning(f"Model compare: {name} failed: {e_model}")
                comparison.append({"model": name, "error": str(e_model)[:100]})

        # Summary text
        summary_lines = [f"Model Comparison — {task_type.title()} ({cv_folds}-fold CV)\n"]
        summary_lines.append(f"{'Model':<22} {'Score':>8} {'± Std':>8} {'Train':>8} {'Time':>6}")
        summary_lines.append("-" * 55)
        for row in comparison:
            if "error" in row:
                summary_lines.append(f"{row['model']:<22} ERROR: {row['error'][:30]}")
                continue
            score = row.get(primary_metric, 0)
            std = row.get(f"{primary_metric}_std", 0)
            train = row.get(f"{primary_metric}_train", 0)
            marker = " *" if row["model"] == best_model_name else ""
            summary_lines.append(
                f"{row['model']:<22} {score:>8.4f} {std:>8.4f} {train:>8.4f} {row['fit_time']:>5.1f}s{marker}"
            )
        summary_lines.append(f"\n* Best: {best_model_name} ({primary_metric} = {best_score:.4f})")

        # --- MODEL ASSESSMENT ---
        assess = f"\n{'─' * 55}\nMODEL ASSESSMENT\n{'─' * 55}\n"
        valid_rows = [r for r in comparison if "error" not in r]

        # Overfitting diagnosis
        overfit_warnings = []
        for row in valid_rows:
            train_score = row.get(f"{primary_metric}_train", 0)
            test_score = row.get(primary_metric, 0)
            if train_score and test_score and train_score > 0:
                gap = train_score - test_score
                gap_pct = gap / train_score * 100 if train_score != 0 else 0
                if gap_pct > 15:
                    overfit_warnings.append(
                        f"  ⚠ {row['model']}: train={train_score:.4f} vs test={test_score:.4f} (gap: {gap_pct:.0f}%) — likely overfit"
                    )
                elif gap_pct > 8:
                    overfit_warnings.append(
                        f"  ◐ {row['model']}: train-test gap of {gap_pct:.0f}% — monitor for overfitting"
                    )

        if overfit_warnings:
            assess += "Overfitting check:\n" + "\n".join(overfit_warnings) + "\n\n"
        else:
            assess += "Overfitting check: ✓ No significant train-test gaps detected.\n\n"

        # Is the best model meaningfully better than others?
        if len(valid_rows) >= 2:
            scores_sorted = sorted(valid_rows, key=lambda r: r.get(primary_metric, 0), reverse=True)
            best_r = scores_sorted[0]
            second_r = scores_sorted[1]
            margin = best_r.get(primary_metric, 0) - second_r.get(primary_metric, 0)
            best_std = best_r.get(f"{primary_metric}_std", 0)
            if margin < best_std:
                assess += f"Winner margin: {best_r['model']} beats {second_r['model']} by {margin:.4f} — within 1 std ({best_std:.4f}).\n"
                assess += (
                    f"  → The difference may be noise. Consider {second_r['model']} if it's simpler or faster.\n\n"
                )
            else:
                assess += f"Winner margin: {best_r['model']} beats {second_r['model']} by {margin:.4f} (> 1 std). Solid winner.\n\n"

        # Baseline comparison
        if task_type == "classification":
            from collections import Counter

            majority_pct = max(Counter(y_enc).values()) / len(y_enc)
            assess += f"Baseline (majority class): {majority_pct:.4f}\n"
            if best_score < majority_pct + 0.02:
                assess += "⚠ Best model is barely better than always guessing the majority class.\n"
            else:
                assess += f"Lift over baseline: {best_score - majority_pct:+.4f}\n"
        else:
            assess += "Baseline (predict mean): R²=0.000\n"
            assess += f"Best model lift: R²={best_score:.4f}\n"

        # Recommendation
        assess += "\nRecommendation: "
        if task_type == "classification" and best_score >= 0.90:
            assess += f"Deploy {best_model_name}. High accuracy — suitable for automated decision support."
        elif task_type == "classification" and best_score >= 0.80:
            assess += f"{best_model_name} is solid for screening. Review misclassifications before high-stakes use."
        elif task_type == "regression" and best_score >= 0.80:
            assess += f"Deploy {best_model_name}. Strong R² — suitable for forecasting."
        elif task_type == "regression" and best_score >= 0.50:
            assess += f"{best_model_name} captures the main trends. Consider adding features or engineering interactions to improve."
        else:
            assess += "Model performance is limited. Consider: (1) more/better features, (2) data quality issues, (3) whether the target is predictable from these features."

        summary_lines.append(assess)
        result["summary"] = "\n".join(summary_lines)

        # Comparison bar chart
        model_names = [r["model"] for r in comparison if "error" not in r]
        scores = [r.get(primary_metric, 0) for r in comparison if "error" not in r]
        stds = [r.get(f"{primary_metric}_std", 0) for r in comparison if "error" not in r]

        result["plots"].append(
            {
                "title": f"Model Comparison — {primary_metric.upper()}",
                "data": [
                    {
                        "type": "bar",
                        "x": model_names,
                        "y": scores,
                        "error_y": {"type": "data", "array": stds, "visible": True},
                        "marker": {
                            "color": [
                                ("rgba(74,159,110,0.85)" if n == best_model_name else "rgba(74,159,110,0.4)")
                                for n in model_names
                            ],
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                    }
                ],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": primary_metric.upper()},
                    "xaxis": {"title": "Model"},
                },
            }
        )

        # Multi-metric heatmap (all metrics x all models)
        metric_keys = [k for k in scoring.keys() if not k.startswith("neg_")]
        # Also include the neg_ metrics with corrected names
        for k in scoring.keys():
            if k.startswith("neg_"):
                clean_name = k.replace("neg_", "")
                if clean_name not in metric_keys:
                    metric_keys.append(clean_name)

        valid_models = [r for r in comparison if "error" not in r]
        if len(valid_models) > 1 and len(metric_keys) > 1:
            heatmap_z = []
            for mk in metric_keys:
                row = []
                for r in valid_models:
                    row.append(round(r.get(mk, 0), 4))
                heatmap_z.append(row)
            heatmap_names = [r["model"] for r in valid_models]

            result["plots"].append(
                {
                    "title": "Multi-Metric Comparison",
                    "data": [
                        {
                            "type": "heatmap",
                            "z": heatmap_z,
                            "x": heatmap_names,
                            "y": [m.upper() for m in metric_keys],
                            "colorscale": [
                                [0, "#0d120d"],
                                [0.5, "#2a6b3a"],
                                [1, "#4a9f6e"],
                            ],
                            "showscale": True,
                            "text": [[f"{v:.4f}" for v in row] for row in heatmap_z],
                            "texttemplate": "%{text}",
                            "hovertemplate": "%{y}: %{z:.4f}<br>%{x}<extra></extra>",
                        }
                    ],
                    "layout": {
                        "height": max(200, len(metric_keys) * 50 + 80),
                        "xaxis": {"title": ""},
                        "yaxis": {"autorange": "reversed"},
                    },
                }
            )

        # Fit time comparison
        if valid_models:
            fit_times = [r.get("fit_time", 0) for r in valid_models]
            fit_names = [r["model"] for r in valid_models]
            result["plots"].append(
                {
                    "title": "Training Time (seconds)",
                    "data": [
                        {
                            "type": "bar",
                            "x": fit_names,
                            "y": fit_times,
                            "marker": {
                                "color": "rgba(232,149,71,0.6)",
                                "line": {"color": "#e89547", "width": 1},
                            },
                            "text": [f"{t:.1f}s" for t in fit_times],
                            "textposition": "auto",
                        }
                    ],
                    "layout": {"height": 250, "yaxis": {"title": "Seconds"}},
                }
            )

        # ROC curves (classification, binary)
        if task_type == "classification" and y_enc.nunique() == 2:
            from sklearn.metrics import auc as sk_auc
            from sklearn.metrics import roc_curve

            roc_traces = []
            for name, mdl in models:
                try:
                    mdl_fitted = mdl.__class__(**mdl.get_params()) if not isinstance(mdl, Pipeline) else mdl
                    mdl_fitted.fit(X_enc.values, y_enc.values)
                    if hasattr(mdl_fitted, "predict_proba"):
                        y_prob = mdl_fitted.predict_proba(X_enc.values)[:, 1]
                        fpr, tpr, _ = roc_curve(y_enc.values, y_prob)
                        roc_auc = sk_auc(fpr, tpr)
                        roc_traces.append(
                            {
                                "type": "scatter",
                                "x": fpr.tolist(),
                                "y": tpr.tolist(),
                                "mode": "lines",
                                "name": f"{name} (AUC={roc_auc:.3f})",
                            }
                        )
                except Exception:
                    pass
            if roc_traces:
                roc_traces.append(
                    {
                        "type": "scatter",
                        "x": [0, 1],
                        "y": [0, 1],
                        "mode": "lines",
                        "line": {"dash": "dash", "color": "#555"},
                        "name": "Random",
                    }
                )
                result["plots"].append(
                    {
                        "title": "ROC Curves",
                        "data": roc_traces,
                        "layout": {
                            "height": 350,
                            "xaxis": {"title": "False Positive Rate"},
                            "yaxis": {"title": "True Positive Rate"},
                        },
                    }
                )

        # Actual vs Predicted overlay (regression)
        if task_type == "regression":
            avp_traces = []
            colors = ["#4a9f6e", "#e89547", "#6a7fff", "#e8c547", "#ff7eb9", "#4a9faf"]
            for i, (name, mdl) in enumerate(models[:4]):
                try:
                    mdl_fitted = mdl.__class__(**mdl.get_params()) if not isinstance(mdl, Pipeline) else mdl
                    mdl_fitted.fit(
                        X_train.values if hasattr(X_train, "values") else X_train,
                        y_train.values if hasattr(y_train, "values") else y_train,
                    )
                    y_pred_cmp = mdl_fitted.predict(X_test.values if hasattr(X_test, "values") else X_test)
                    avp_traces.append(
                        {
                            "type": "scatter",
                            "x": y_test.tolist(),
                            "y": y_pred_cmp.tolist(),
                            "mode": "markers",
                            "marker": {
                                "color": colors[i % len(colors)],
                                "size": 4,
                                "opacity": 0.6,
                            },
                            "name": name,
                        }
                    )
                except Exception:
                    pass
            if avp_traces:
                y_range = [float(y_test.min()), float(y_test.max())]
                avp_traces.append(
                    {
                        "type": "scatter",
                        "x": y_range,
                        "y": y_range,
                        "mode": "lines",
                        "line": {"dash": "dash", "color": "#d94a4a"},
                        "name": "Perfect",
                    }
                )
                result["plots"].append(
                    {
                        "title": "Actual vs Predicted (Top Models)",
                        "data": avp_traces,
                        "layout": {
                            "height": 350,
                            "xaxis": {"title": "Actual"},
                            "yaxis": {"title": "Predicted"},
                        },
                    }
                )

        # Train best model with conformal calibration (70/15/15 split)
        if best_model_obj is not None and user and user.is_authenticated:
            try:
                # Split for conformal: train on 70%, calibrate on 15%, evaluate on 15%
                if task_type == "classification":
                    Xc_train, Xc_cal, Xc_test, yc_train, yc_cal, yc_test = _stratified_split_3way(
                        pd.DataFrame(X_enc.values, columns=X_enc.columns),
                        (
                            pd.Series(y_enc.values, name=y_enc.name)
                            if hasattr(y_enc, "name")
                            else pd.Series(y_enc.values)
                        ),
                    )
                else:
                    Xc_train, Xc_temp, yc_train, yc_temp = train_test_split(
                        X_enc, y_enc, test_size=0.30, random_state=42
                    )
                    Xc_cal, Xc_test, yc_cal, yc_test = train_test_split(
                        Xc_temp, yc_temp, test_size=0.50, random_state=42
                    )

                best_clone = (
                    best_model_obj.__class__(**best_model_obj.get_params())
                    if not isinstance(best_model_obj, Pipeline)
                    else best_model_obj
                )
                best_clone.fit(
                    Xc_train.values if hasattr(Xc_train, "values") else Xc_train,
                    yc_train.values if hasattr(yc_train, "values") else yc_train,
                )

                conformal_state = None
                try:
                    from forgestat.conformal import compute_conformal

                    cf = compute_conformal(best_clone, Xc_cal, yc_cal, task_type=task_type)
                    conformal_state = cf.get_state()
                except Exception:
                    pass

                model_key = str(uuid.uuid4())
                cache_meta = {
                    "model_type": f"Best: {best_model_name}",
                    "features": features,
                    "target": target,
                    "metrics": {primary_metric: best_score},
                }
                if conformal_state:
                    cache_meta["conformal_state"] = conformal_state
                    cache_meta["split_seed"] = 42
                cache_model(user.id, model_key, best_clone, cache_meta)
                result["model_key"] = model_key
                result["can_save"] = True
            except Exception:
                pass

        result["comparison"] = comparison
        result["best_model"] = best_model_name
        overfit_models = [
            r["model"]
            for r in valid_rows
            if r.get(f"{primary_metric}_train", 0)
            and r.get(primary_metric, 0)
            and (r[f"{primary_metric}_train"] - r[primary_metric]) / r[f"{primary_metric}_train"] * 100 > 15
        ]
        obs = f"Compared {len(comparison)} models ({task_type}, {cv_folds}-fold CV). Best: {best_model_name} ({primary_metric}={best_score:.4f})."
        if overfit_models:
            obs += f" Overfitting risk: {', '.join(overfit_models)}."
        result["guide_observation"] = obs

    elif analysis_id == "shap_explain":
        import shap

        from accounts.constants import can_use_ml

        if not can_use_ml(getattr(user, "tier", "free")):
            result["summary"] = "Error: SHAP explainability requires a paid tier with ML access."
            return result

        # Get model from cache
        model_key = config.get("model_key", "")
        if not model_key:
            result["summary"] = "Error: No model selected. Train a model first, then explain it."
            return result

        cached = get_cached_model(user.id if user else 0, model_key)
        if not cached:
            result["summary"] = "Error: Model not found in cache. It may have expired — retrain and try again."
            return result

        model_obj = cached["model"]
        model_features = cached.get("meta", {}).get("features", features)
        explain_mode = config.get("mode", "global")  # global or single
        sample_idx = int(config.get("sample_index", 0))

        # Prepare data
        X_explain = X[model_features] if all(f in X.columns for f in model_features) else X
        # Encode categoricals
        for col in X_explain.select_dtypes(include=["object", "category"]).columns:
            X_explain[col] = pd.Categorical(X_explain[col]).codes.astype(int)
        X_explain = X_explain.fillna(X_explain.median(numeric_only=True))

        # Cap background data for KernelExplainer
        max_bg = 100
        if len(X_explain) > max_bg:
            bg_data = X_explain.sample(max_bg, random_state=42)
        else:
            bg_data = X_explain

        # Choose explainer
        is_tree = hasattr(model_obj, "feature_importances_") and type(model_obj).__name__ in (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
        )

        if is_tree:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(bg_data)
        else:
            explainer = shap.KernelExplainer(model_obj.predict, bg_data.values[:50])
            shap_values = explainer.shap_values(bg_data.values[:max_bg])

        # Handle multi-class shap_values (list of arrays)
        if isinstance(shap_values, list):
            # Take the first class for binary or max-variance class for multi
            if len(shap_values) == 2:
                sv = shap_values[1]
            else:
                variances = [np.var(s) for s in shap_values]
                sv = shap_values[np.argmax(variances)]
        else:
            sv = shap_values

        feature_names_explain = list(X_explain.columns) if hasattr(X_explain, "columns") else model_features

        # Global: mean absolute SHAP values (feature importance bar)
        mean_abs = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        result["plots"].append(
            {
                "title": "SHAP Feature Importance (mean |SHAP|)",
                "data": [
                    {
                        "type": "bar",
                        "orientation": "h",
                        "x": mean_abs[sorted_idx].tolist(),
                        "y": [feature_names_explain[i] for i in sorted_idx],
                        "marker": {
                            "color": "rgba(74,159,110,0.6)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                    }
                ],
                "layout": {
                    "height": max(250, len(feature_names_explain) * 25),
                    "xaxis": {"title": "mean |SHAP value|"},
                },
            }
        )

        # Beeswarm plot (scatter of SHAP values per feature)
        beeswarm_traces = []
        for i in sorted_idx[-10:]:  # Top 10
            feat_name = feature_names_explain[i]
            vals = sv[:, i]
            feat_vals = bg_data.iloc[:, i].values if hasattr(bg_data, "iloc") else bg_data[:, i]
            # Normalize feature values for color
            fmin, fmax = float(np.min(feat_vals)), float(np.max(feat_vals))
            colors = ((feat_vals - fmin) / (fmax - fmin + 1e-10)).tolist()
            jitter = np.random.normal(0, 0.15, len(vals))
            beeswarm_traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": vals.tolist(),
                    "y": (jitter + len(sorted_idx) - 1 - np.where(sorted_idx == i)[0][0]).tolist(),
                    "marker": {
                        "size": 3,
                        "color": colors,
                        "colorscale": [[0, "#4a9faf"], [1, "#d94a4a"]],
                        "opacity": 0.7,
                    },
                    "name": feat_name,
                    "showlegend": False,
                    "hovertemplate": f"{feat_name}<br>SHAP: %{{x:.3f}}<br>Value: %{{text}}<extra></extra>",
                    "text": [f"{v:.2f}" for v in feat_vals],
                }
            )
        if beeswarm_traces:
            tick_labels = [feature_names_explain[i] for i in sorted_idx[-10:]][::-1]
            result["plots"].append(
                {
                    "title": "SHAP Beeswarm (Top 10 Features)",
                    "data": beeswarm_traces,
                    "layout": {
                        "height": 350,
                        "xaxis": {
                            "title": "SHAP value (impact on prediction)",
                            "zeroline": True,
                            "zerolinecolor": "#555",
                        },
                        "yaxis": {
                            "tickvals": list(range(len(tick_labels))),
                            "ticktext": tick_labels,
                        },
                        "showlegend": False,
                    },
                }
            )

        # Single prediction waterfall (if requested or always show first sample)
        if explain_mode == "single" or True:
            idx = min(sample_idx, len(sv) - 1)
            single_sv = sv[idx]
            base_val = (
                float(explainer.expected_value)
                if not isinstance(explainer.expected_value, (list, np.ndarray))
                else float(
                    explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
                )
            )

            # Waterfall as horizontal bar
            sorted_single = np.argsort(np.abs(single_sv))[-10:]
            wf_names = [feature_names_explain[i] for i in sorted_single]
            wf_vals = [float(single_sv[i]) for i in sorted_single]
            wf_colors = ["rgba(74,159,110,0.7)" if v >= 0 else "rgba(208,96,96,0.7)" for v in wf_vals]

            result["plots"].append(
                {
                    "title": f"SHAP Waterfall (Sample #{idx})",
                    "data": [
                        {
                            "type": "bar",
                            "orientation": "h",
                            "x": wf_vals,
                            "y": wf_names,
                            "marker": {"color": wf_colors},
                        }
                    ],
                    "layout": {
                        "height": max(200, len(wf_names) * 25),
                        "xaxis": {"title": "SHAP value"},
                        "annotations": [
                            {
                                "x": 0,
                                "y": -0.15,
                                "xref": "paper",
                                "yref": "paper",
                                "text": f"Base value: {base_val:.3f}",
                                "showarrow": False,
                                "font": {"size": 11, "color": "#9aaa9a"},
                            }
                        ],
                    },
                }
            )

        # Dependence plot (top feature vs SHAP)
        top_feat_idx = sorted_idx[-1]
        dep_x = bg_data.iloc[:, top_feat_idx].values if hasattr(bg_data, "iloc") else bg_data[:, top_feat_idx]
        dep_y = sv[:, top_feat_idx]
        result["plots"].append(
            {
                "title": f"SHAP Dependence: {feature_names_explain[top_feat_idx]}",
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": dep_x.tolist(),
                        "y": dep_y.tolist(),
                        "marker": {"color": "rgba(74,159,110,0.5)", "size": 4},
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": feature_names_explain[top_feat_idx]},
                    "yaxis": {"title": "SHAP value"},
                },
            }
        )

        # Summary text
        top_3 = [(feature_names_explain[i], float(mean_abs[i])) for i in sorted_idx[-3:]][::-1]
        result["summary"] = (
            f"SHAP Explainability Analysis\n\n"
            f"Explainer: {'TreeExplainer' if is_tree else 'KernelExplainer'}\n"
            f"Background samples: {len(bg_data)}\n\n"
            f"Top 3 features by mean |SHAP|:\n"
            + "\n".join(f"  {i + 1}. {name}: {val:.4f}" for i, (name, val) in enumerate(top_3))
        )

        result["guide_observation"] = f"SHAP: top feature is {top_3[0][0]} (mean |SHAP| = {top_3[0][1]:.4f})."

    elif analysis_id == "hyperparameter_tune":
        import time as _time

        import optuna

        from accounts.constants import can_use_ml

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not can_use_ml(getattr(user, "tier", "free")):
            result["summary"] = "Error: Hyperparameter tuning requires a paid tier with ML access."
            return result

        model_type = config.get("model_type", "rf")
        n_trials = min(int(config.get("n_trials", 30)), 50)
        task_type = config.get("task_type", "auto")
        cv_folds = int(config.get("cv_folds", 3))

        if task_type == "auto":
            task_type = (
                "classification"
                if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10))
                else "regression"
            )

        # Encode
        y_work = y.copy()
        if task_type == "classification" and (y_work.dtype == object or y_work.dtype.name == "category"):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_work = pd.Series(le.fit_transform(y_work), index=y_work.index)

        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            X_enc[col] = pd.Categorical(X_enc[col]).codes.astype(int)
        X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

        from sklearn.model_selection import cross_val_score

        def objective(trial):
            if model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": 42,
                    "n_jobs": -1,
                }
                mdl = (
                    RandomForestClassifier(**params)
                    if task_type == "classification"
                    else RandomForestRegressor(**params)
                )
            elif model_type == "xgboost":
                import xgboost as xgb

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "random_state": 42,
                    "verbosity": 0,
                }
                mdl = (
                    xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                    if task_type == "classification"
                    else xgb.XGBRegressor(**params)
                )
            elif model_type == "lightgbm":
                import lightgbm as lgb

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 127),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "random_state": 42,
                    "verbosity": -1,
                }
                mdl = lgb.LGBMClassifier(**params) if task_type == "classification" else lgb.LGBMRegressor(**params)
            elif model_type == "ridge":
                from sklearn.linear_model import LogisticRegression, Ridge

                alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
                if task_type == "classification":
                    mdl = LogisticRegression(C=1.0 / alpha, max_iter=1000, penalty="l2")
                else:
                    mdl = Ridge(alpha=alpha)
            elif model_type == "lasso":
                from sklearn.linear_model import Lasso, LogisticRegression

                alpha = trial.suggest_float("alpha", 0.001, 10.0, log=True)
                if task_type == "classification":
                    mdl = LogisticRegression(C=1.0 / alpha, max_iter=1000, penalty="l1", solver="saga")
                else:
                    mdl = Lasso(alpha=alpha)
            else:
                # Default: random forest
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "random_state": 42,
                    "n_jobs": -1,
                }
                mdl = (
                    RandomForestClassifier(**params)
                    if task_type == "classification"
                    else RandomForestRegressor(**params)
                )

            scoring = "accuracy" if task_type == "classification" else "r2"
            scores = cross_val_score(mdl, X_enc, y_work, cv=cv_folds, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))

        # Run optimization with timeout
        study = optuna.create_study(direction="maximize")
        t0 = _time.time()
        study.optimize(objective, n_trials=n_trials, timeout=120)
        elapsed = _time.time() - t0

        best = study.best_trial
        best_params = best.params
        best_score = best.value

        # Summary
        summary_lines = [
            f"Hyperparameter Tuning — {model_type.upper()} ({task_type})\n",
            f"Trials: {len(study.trials)} / {n_trials} ({elapsed:.1f}s)",
            f"CV Folds: {cv_folds}",
            f"Best Score: {best_score:.4f}",
            "\nBest Parameters:",
        ]
        for k, v in best_params.items():
            summary_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        result["summary"] = "\n".join(summary_lines)

        # Optimization history plot
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        best_so_far = []
        running_best = -999
        for v in trial_values:
            running_best = max(running_best, v)
            best_so_far.append(running_best)

        result["plots"].append(
            {
                "title": "Optimization History",
                "data": [
                    {
                        "type": "scatter",
                        "x": trial_numbers,
                        "y": trial_values,
                        "mode": "markers",
                        "marker": {"color": "rgba(74,159,110,0.4)", "size": 5},
                        "name": "Trial Score",
                    },
                    {
                        "type": "scatter",
                        "x": trial_numbers,
                        "y": best_so_far,
                        "mode": "lines",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "name": "Best So Far",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Trial"},
                    "yaxis": {"title": "Score"},
                },
            }
        )

        # Parameter importance (if enough trials)
        if len(study.trials) >= 5:
            try:
                param_importance = optuna.importance.get_param_importances(study)
                p_names = list(param_importance.keys())[::-1]
                p_vals = [param_importance[k] for k in p_names]
                result["plots"].append(
                    {
                        "title": "Hyperparameter Importance",
                        "data": [
                            {
                                "type": "bar",
                                "orientation": "h",
                                "x": p_vals,
                                "y": p_names,
                                "marker": {
                                    "color": "rgba(74,159,110,0.6)",
                                    "line": {"color": "#4a9f6e", "width": 1},
                                },
                            }
                        ],
                        "layout": {"height": max(200, len(p_names) * 30)},
                    }
                )
            except Exception:
                pass

        # Train final model with best params and cache
        if user and user.is_authenticated:
            try:
                # Rebuild model with best params
                type(
                    "FakeTrial",
                    (),
                    {
                        "suggest_int": lambda s, n, *a, **kw: best_params[n],
                        "suggest_float": lambda s, n, *a, **kw: best_params[n],
                    },
                )()
                # Simpler: just train directly
                if model_type == "rf":
                    final_model = (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(
                        **{k: v for k, v in best_params.items()},
                        random_state=42,
                        n_jobs=-1,
                    )
                elif model_type == "xgboost":
                    import xgboost as xgb

                    final_model = (xgb.XGBClassifier if task_type == "classification" else xgb.XGBRegressor)(
                        **{k: v for k, v in best_params.items()},
                        random_state=42,
                        verbosity=0,
                    )
                elif model_type == "lightgbm":
                    import lightgbm as lgb

                    final_model = (lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor)(
                        **{k: v for k, v in best_params.items()},
                        random_state=42,
                        verbosity=-1,
                    )
                elif model_type in ("ridge", "lasso"):
                    from sklearn.linear_model import Lasso, LogisticRegression, Ridge

                    alpha = best_params.get("alpha", 1.0)
                    if task_type == "classification":
                        penalty = "l1" if model_type == "lasso" else "l2"
                        solver = "saga" if model_type == "lasso" else "lbfgs"
                        final_model = LogisticRegression(C=1.0 / alpha, max_iter=1000, penalty=penalty, solver=solver)
                    else:
                        final_model = (Lasso if model_type == "lasso" else Ridge)(alpha=alpha)
                else:
                    final_model = (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(
                        **{k: v for k, v in best_params.items()},
                        random_state=42,
                        n_jobs=-1,
                    )

                final_model.fit(X_enc, y_work)
                model_key = str(uuid.uuid4())
                primary = "accuracy" if task_type == "classification" else "r2"
                cache_model(
                    user.id,
                    model_key,
                    final_model,
                    {
                        "model_type": f"Tuned {model_type.upper()}",
                        "features": features,
                        "target": target,
                        "metrics": {primary: best_score},
                    },
                )
                result["model_key"] = model_key
                result["can_save"] = True
            except Exception as e_tune:
                logger.warning(f"Could not cache tuned model: {e_tune}")

        result["best_params"] = best_params
        result["guide_observation"] = (
            f"Tuned {model_type.upper()}: best score {best_score:.4f} in {len(study.trials)} trials."
        )

    return result
