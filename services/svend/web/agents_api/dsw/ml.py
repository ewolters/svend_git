"""DSW Machine Learning Analysis — ML model training and evaluation."""

import json
import logging
import uuid
import numpy as np
import pandas as pd
from scipy import stats

from .common import (
    _effect_magnitude, _practical_block, _ml_interpretation,
    _build_ml_diagnostics, _clean_for_ml, _stratified_split,
    _stratified_split_3way, _classification_reliability, _auto_train,
    _claude_generate_schema, _generate_data_from_schema,
    _claude_interpret_results, cache_model, _create_ml_evidence,
    save_model_to_disk, get_cached_model,
)

logger = logging.getLogger(__name__)


def run_ml_analysis(df, analysis_id, config, user):
    """Run ML analysis."""
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])
    algorithm = config.get("algorithm", "rf")
    split_val = float(config.get("split", 20))
    test_split = split_val if split_val < 1 else split_val / 100

    # Analyses that don't require a target variable or handle their own data prep
    unsupervised_analyses = ["pca", "clustering", "isolation_forest", "regularized_regression", "discriminant_analysis", "factor_analysis"]

    if analysis_id in unsupervised_analyses:
        # For unsupervised methods, only need features
        if not features:
            # Use all numeric columns if no features specified
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[features].dropna()
        y = None
        X_train = X_test = X
        y_train = y_test = None
    else:
        # Supervised methods need target and features
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

    if analysis_id == "classification":
        if algorithm == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CLASSIFICATION RESULTS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Algorithm:<</COLOR>> {algorithm.upper()}\n"
        summary += f"<<COLOR:highlight>>Accuracy:<</COLOR>> {accuracy:.4f}\n\n"
        summary += f"<<COLOR:text>>{report}<</COLOR>>"
        summary += _ml_interpretation("classification", {"accuracy": accuracy}, y_test, y_pred, features, target, model)
        result["summary"] = summary

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            result["plots"].append({
                "title": "Feature Importance",
                "data": [{
                    "type": "bar",
                    "x": importances[sorted_idx].tolist(),
                    "y": [features[i] for i in sorted_idx],
                    "orientation": "h",
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": max(200, len(features) * 25)}
            })

        from collections import Counter
        majority_pct = max(Counter(y_test).values()) / len(y_test)
        lift = accuracy - majority_pct
        obs = f"Classification ({algorithm.upper()}): accuracy={accuracy:.1%}, baseline={majority_pct:.1%}, lift={lift:+.1%}."
        if lift < 0.02:
            obs += " Barely better than guessing — needs improvement."
        elif accuracy >= 0.85:
            obs += " Strong model suitable for decision support."
        else:
            obs += " Moderate model — useful for screening."
        result["guide_observation"] = obs

        # Confusion matrix heatmap
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y_test.tolist()) | set(y_pred.tolist())))
        str_labels = [str(l) for l in labels]
        result["plots"].append({
            "title": "Confusion Matrix",
            "data": [{
                "type": "heatmap",
                "z": cm.tolist(),
                "x": str_labels,
                "y": str_labels,
                "colorscale": [[0, "#1a1a2e"], [1, "#4a9f6e"]],
                "showscale": True,
                "text": cm.tolist(),
                "texttemplate": "%{text}",
                "hovertemplate": "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
            }],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Actual", "autorange": "reversed"}}
        })

        # ROC curve (binary or one-vs-rest for multiclass)
        try:
            from sklearn.metrics import roc_curve, auc
            classes = sorted(list(set(y_test)))
            if len(classes) == 2:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=classes[1])
                    roc_auc = auc(fpr, tpr)
                    result["plots"].append({
                        "title": f"ROC Curve (AUC = {roc_auc:.3f})",
                        "data": [
                            {"type": "scatter", "x": fpr.tolist(), "y": tpr.tolist(), "mode": "lines", "line": {"color": "#4a9f6e", "width": 2}, "name": f"ROC (AUC={roc_auc:.3f})"},
                            {"type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "lines", "line": {"color": "#d94a4a", "dash": "dash"}, "name": "Random"}
                        ],
                        "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "False Positive Rate"}, "yaxis": {"title": "True Positive Rate"}}
                    })
        except Exception:
            pass

        # Conformal prediction sets
        conformal_state = None
        try:
            from ..conformal import compute_conformal
            # Need integer-encoded y for conformal classifier
            y_cal_int = y_cal.copy()
            if y_cal_int.dtype == object or y_cal_int.dtype.name == "category":
                from sklearn.preprocessing import LabelEncoder
                _le_cf = LabelEncoder()
                _le_cf.fit(y)
                y_cal_int = _le_cf.transform(y_cal)
                y_test_int = _le_cf.transform(y_test)
            else:
                y_cal_int = np.asarray(y_cal).astype(int)
                y_test_int = np.asarray(y_test).astype(int)

            if hasattr(model, 'predict_proba'):
                cf = compute_conformal(model, X_cal, y_cal_int, task_type="classification")
                conformal_state = cf.get_state()

                # Empirical coverage on test set
                proba_test = model.predict_proba(X_test)
                pred_sets, meta = cf.predict_sets(proba_test, alpha=0.10)
                covered = sum(1 for i, ps in enumerate(pred_sets) if int(y_test_int[i]) in ps)
                emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                set_sizes = [len(ps) for ps in pred_sets]
                avg_set_size = float(np.mean(set_sizes))
                single_class_pct = sum(1 for s in set_sizes if s == 1) / len(set_sizes) if set_sizes else 0

                result["summary"] += f"\n\n<<COLOR:accent>>── Conformal Prediction Sets (90% nominal) ──<</COLOR>>\n"
                result["summary"] += f"<<COLOR:highlight>>Average set size:<</COLOR>> {avg_set_size:.2f} classes\n"
                result["summary"] += f"<<COLOR:highlight>>Empirical test coverage:<</COLOR>> {emp_coverage:.1%}\n"
                result["summary"] += f"<<COLOR:highlight>>Single-class predictions:<</COLOR>> {single_class_pct:.1%}\n"
                result["summary"] += f"<<COLOR:highlight>>Calibration set:<</COLOR>> {cf.n_cal} observations\n"
                result["summary"] += f"<<COLOR:text>>Nominal coverage: 90% (finite-sample marginal guarantee under exchangeability)<</COLOR>>\n"
                result["summary"] += f"<<COLOR:text>>When the model is confident, you get 1 class. When uncertain, you get 2+.<</COLOR>>"

                # Prediction set size histogram
                max_size = max(set_sizes) if set_sizes else 1
                result["plots"].append({
                    "title": f"Conformal Prediction Set Size (coverage: {emp_coverage:.1%})",
                    "data": [{
                        "type": "histogram",
                        "x": set_sizes,
                        "marker": {"color": "rgba(74, 159, 110, 0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                        "name": "Set Size",
                    }],
                    "layout": {"template": "plotly_dark", "height": 250,
                               "xaxis": {"title": "Number of Classes in Set", "dtick": 1},
                               "yaxis": {"title": "Count"}},
                })

                result["statistics"] = result.get("statistics", {})
                result["statistics"]["conformal"] = {
                    "alpha": 0.10, "nominal_coverage": 0.90,
                    "empirical_coverage": round(float(emp_coverage), 4),
                    "avg_set_size": round(avg_set_size, 4),
                    "single_class_pct": round(single_class_pct, 4),
                    "n_calibration": cf.n_cal, "split_seed": 42,
                }
        except Exception as e:
            logger.warning(f"Conformal prediction failed for classification: {e}")

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_meta = {
                'model_type': f"Classification ({algorithm.upper()})",
                'features': features,
                'target': target,
                'metrics': {'accuracy': float(accuracy)},
            }
            if conformal_state:
                cache_meta['conformal_state'] = conformal_state
                cache_meta['split_seed'] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "regression_ml":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>REGRESSION RESULTS (RANDOM FOREST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n"
        summary += f"<<COLOR:highlight>>RMSE:<</COLOR>> {rmse:.4f}\n"
        summary += _ml_interpretation("regression", {"r2": r2, "rmse": rmse}, y_test, y_pred, features, target)
        result["summary"] = summary

        r2_label, _ = _effect_magnitude(r2, "r_squared")
        y_range_val = float(np.ptp(y_test)) if len(y_test) > 0 else 1
        rmse_pct = rmse / y_range_val * 100 if y_range_val > 0 else 0
        result["guide_observation"] = f"Random Forest Regression: R²={r2:.3f} ({r2_label}), RMSE={rmse:.2f} ({rmse_pct:.0f}% of range). " + (
            "Strong predictive model." if r2 >= 0.7 else "Moderate model — useful for trends." if r2 >= 0.4 else "Weak model — features may not predict target well.")

        # Actual vs Predicted
        result["plots"].append({
            "title": "Actual vs Predicted",
            "data": [
                {"type": "scatter", "x": y_test.tolist(), "y": y_pred.tolist(), "mode": "markers", "marker": {"color": "#6c5ce7", "size": 5}},
                {"type": "scatter", "x": [y_test.min(), y_test.max()], "y": [y_test.min(), y_test.max()], "mode": "lines", "line": {"color": "#ff7675", "dash": "dash"}}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"}}
        })

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            result["plots"].append({
                "title": "Feature Importance",
                "data": [{
                    "type": "bar",
                    "x": importances[sorted_idx].tolist(),
                    "y": [features[i] for i in sorted_idx],
                    "orientation": "h",
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": max(200, len(features) * 25)}
            })

        # Residual plot
        residuals = (y_test.values - y_pred).tolist()
        result["plots"].append({
            "title": "Residuals vs Predicted",
            "data": [
                {"type": "scatter", "x": y_pred.tolist(), "y": residuals, "mode": "markers", "marker": {"color": "rgba(74, 144, 217, 0.5)", "size": 5, "line": {"color": "#4a90d9", "width": 1}}, "name": "Residuals"},
                {"type": "scatter", "x": [float(y_pred.min()), float(y_pred.max())], "y": [0, 0], "mode": "lines", "line": {"color": "#d94a4a", "dash": "dash"}, "name": "Zero"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Residual"}}
        })

        # Conformal prediction intervals
        conformal_state = None
        try:
            from ..conformal import compute_conformal
            cf = compute_conformal(model, X_cal, y_cal, task_type="regression")
            conformal_state = cf.get_state()
            qhat_90 = cf.qhats.get("0.1", 0)
            qhat_95 = cf.qhats.get("0.05", 0)

            # Empirical coverage on test set
            y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
            covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
            emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0

            result["summary"] += f"\n\n<<COLOR:accent>>── Conformal Prediction Intervals (90% nominal) ──<</COLOR>>\n"
            result["summary"] += f"<<COLOR:highlight>>Interval half-width:<</COLOR>> ±{qhat_90:.4f}\n"
            result["summary"] += f"<<COLOR:highlight>>Empirical test coverage:<</COLOR>> {emp_coverage:.1%}\n"
            result["summary"] += f"<<COLOR:highlight>>Calibration set:<</COLOR>> {cf.n_cal} observations\n"
            result["summary"] += f"<<COLOR:highlight>>95% interval half-width:<</COLOR>> ±{qhat_95:.4f}\n"
            result["summary"] += f"<<COLOR:text>>Nominal coverage: 90% (finite-sample marginal guarantee under exchangeability)<</COLOR>>"

            # Conformal interval plot
            sort_idx = np.argsort(y_pred)
            y_pred_s = y_pred[sort_idx]
            y_test_s = y_test.values[sort_idx]
            y_lo_s = y_lo[sort_idx]
            y_hi_s = y_hi[sort_idx]
            inside = ((y_test_s >= y_lo_s) & (y_test_s <= y_hi_s))
            result["plots"].append({
                "title": f"Conformal Prediction Intervals (coverage: {emp_coverage:.1%})",
                "data": [
                    {"type": "scatter", "x": list(range(len(y_pred_s))), "y": y_hi_s.tolist(), "mode": "lines", "line": {"width": 0}, "showlegend": False},
                    {"type": "scatter", "x": list(range(len(y_pred_s))), "y": y_lo_s.tolist(), "mode": "lines", "fill": "tonexty", "fillcolor": "rgba(74, 159, 110, 0.15)", "line": {"width": 0}, "name": "90% interval"},
                    {"type": "scatter", "x": list(range(len(y_pred_s))), "y": y_pred_s.tolist(), "mode": "lines", "line": {"color": "#4a9f6e", "width": 1.5}, "name": "Predicted"},
                    {"type": "scatter", "x": [i for i, v in enumerate(inside) if v], "y": [y_test_s[i] for i, v in enumerate(inside) if v], "mode": "markers", "marker": {"color": "#2ecc71", "size": 5}, "name": "Inside"},
                    {"type": "scatter", "x": [i for i, v in enumerate(inside) if not v], "y": [y_test_s[i] for i, v in enumerate(inside) if not v], "mode": "markers", "marker": {"color": "#e74c3c", "size": 7, "symbol": "x"}, "name": "Outside"},
                ],
                "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Observation (sorted by prediction)"}, "yaxis": {"title": target}},
            })

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = {
                "alpha": 0.10, "nominal_coverage": 0.90,
                "empirical_coverage": round(float(emp_coverage), 4),
                "qhat_90": round(qhat_90, 4), "qhat_95": round(qhat_95, 4),
                "median_width": round(2 * qhat_90, 4),
                "n_calibration": cf.n_cal, "split_seed": 42,
            }
        except Exception as e:
            logger.warning(f"Conformal prediction failed for regression: {e}")

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_meta = {
                'model_type': "Random Forest Regressor",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2), 'rmse': float(rmse)},
            }
            if conformal_state:
                cache_meta['conformal_state'] = conformal_state
                cache_meta['split_seed'] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "model_compare":
        from sklearn.model_selection import cross_validate as sk_cross_validate
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from accounts.constants import can_use_ml
        import time as _time

        if not can_use_ml(getattr(user, 'tier', 'free')):
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
            from sklearn.linear_model import LogisticRegression
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.naive_bayes import GaussianNB

            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ("Logistic Regression", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])),
                ("LDA", LinearDiscriminantAnalysis()),
                ("Naive Bayes", GaussianNB()),
            ]
            scoring = {"accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted", "recall": "recall_weighted"}
            primary_metric = "accuracy"
        else:
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge

            models = [
                ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ("Linear Regression", LinearRegression()),
                ("Ridge", Ridge(alpha=1.0)),
                ("LASSO", Lasso(alpha=0.1)),
                ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
                ("Bayesian Ridge", BayesianRidge()),
            ]
            scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}
            primary_metric = "r2"

        # Try adding XGBoost / LightGBM if installed
        try:
            import xgboost as xgb
            if task_type == "classification":
                models.append(("XGBoost", xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0)))
            else:
                models.append(("XGBoost", xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
        except ImportError:
            pass

        try:
            import lightgbm as lgb
            if task_type == "classification":
                models.append(("LightGBM", lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)))
            else:
                models.append(("LightGBM", lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)))
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
                    mdl, X_enc, y_enc, cv=cv_folds, scoring=scoring,
                    return_train_score=True, n_jobs=-1,
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
            summary_lines.append(f"{row['model']:<22} {score:>8.4f} {std:>8.4f} {train:>8.4f} {row['fit_time']:>5.1f}s{marker}")
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
                    overfit_warnings.append(f"  ⚠ {row['model']}: train={train_score:.4f} vs test={test_score:.4f} (gap: {gap_pct:.0f}%) — likely overfit")
                elif gap_pct > 8:
                    overfit_warnings.append(f"  ◐ {row['model']}: train-test gap of {gap_pct:.0f}% — monitor for overfitting")

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
                assess += f"  → The difference may be noise. Consider {second_r['model']} if it's simpler or faster.\n\n"
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
            assess += f"Baseline (predict mean): R²=0.000\n"
            assess += f"Best model lift: R²={best_score:.4f}\n"

        # Recommendation
        assess += f"\nRecommendation: "
        if task_type == "classification" and best_score >= 0.90:
            assess += f"Deploy {best_model_name}. High accuracy — suitable for automated decision support."
        elif task_type == "classification" and best_score >= 0.80:
            assess += f"{best_model_name} is solid for screening. Review misclassifications before high-stakes use."
        elif task_type == "regression" and best_score >= 0.80:
            assess += f"Deploy {best_model_name}. Strong R² — suitable for forecasting."
        elif task_type == "regression" and best_score >= 0.50:
            assess += f"{best_model_name} captures the main trends. Consider adding features or engineering interactions to improve."
        else:
            assess += f"Model performance is limited. Consider: (1) more/better features, (2) data quality issues, (3) whether the target is predictable from these features."

        summary_lines.append(assess)
        result["summary"] = "\n".join(summary_lines)

        # Comparison bar chart
        model_names = [r["model"] for r in comparison if "error" not in r]
        scores = [r.get(primary_metric, 0) for r in comparison if "error" not in r]
        stds = [r.get(f"{primary_metric}_std", 0) for r in comparison if "error" not in r]

        result["plots"].append({
            "title": f"Model Comparison — {primary_metric.upper()}",
            "data": [{
                "type": "bar",
                "x": model_names,
                "y": scores,
                "error_y": {"type": "data", "array": stds, "visible": True},
                "marker": {
                    "color": ["rgba(74,159,110,0.85)" if n == best_model_name else "rgba(74,159,110,0.4)" for n in model_names],
                    "line": {"color": "#4a9f6e", "width": 1}
                },
            }],
            "layout": {
                "template": "plotly_dark", "height": 300,
                "yaxis": {"title": primary_metric.upper()},
                "xaxis": {"title": "Model"},
            }
        })

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

            result["plots"].append({
                "title": "Multi-Metric Comparison",
                "data": [{
                    "type": "heatmap",
                    "z": heatmap_z,
                    "x": heatmap_names,
                    "y": [m.upper() for m in metric_keys],
                    "colorscale": [[0, "#0d120d"], [0.5, "#2a6b3a"], [1, "#4a9f6e"]],
                    "showscale": True,
                    "text": [[f"{v:.4f}" for v in row] for row in heatmap_z],
                    "texttemplate": "%{text}",
                    "hovertemplate": "%{y}: %{z:.4f}<br>%{x}<extra></extra>",
                }],
                "layout": {
                    "template": "plotly_dark", "height": max(200, len(metric_keys) * 50 + 80),
                    "xaxis": {"title": ""},
                    "yaxis": {"autorange": "reversed"},
                },
            })

        # Fit time comparison
        if valid_models:
            fit_times = [r.get("fit_time", 0) for r in valid_models]
            fit_names = [r["model"] for r in valid_models]
            result["plots"].append({
                "title": "Training Time (seconds)",
                "data": [{
                    "type": "bar", "x": fit_names, "y": fit_times,
                    "marker": {"color": "rgba(232,149,71,0.6)", "line": {"color": "#e89547", "width": 1}},
                    "text": [f"{t:.1f}s" for t in fit_times], "textposition": "auto",
                }],
                "layout": {"template": "plotly_dark", "height": 250, "yaxis": {"title": "Seconds"}},
            })

        # ROC curves (classification, binary)
        if task_type == "classification" and y_enc.nunique() == 2:
            from sklearn.metrics import roc_curve, auc as sk_auc
            roc_traces = []
            for name, mdl in models:
                try:
                    mdl_fitted = mdl.__class__(**mdl.get_params()) if not isinstance(mdl, Pipeline) else mdl
                    mdl_fitted.fit(X_enc.values, y_enc.values)
                    if hasattr(mdl_fitted, 'predict_proba'):
                        y_prob = mdl_fitted.predict_proba(X_enc.values)[:, 1]
                        fpr, tpr, _ = roc_curve(y_enc.values, y_prob)
                        roc_auc = sk_auc(fpr, tpr)
                        roc_traces.append({
                            "type": "scatter", "x": fpr.tolist(), "y": tpr.tolist(),
                            "mode": "lines", "name": f"{name} (AUC={roc_auc:.3f})",
                        })
                except Exception:
                    pass
            if roc_traces:
                roc_traces.append({
                    "type": "scatter", "x": [0, 1], "y": [0, 1],
                    "mode": "lines", "line": {"dash": "dash", "color": "#555"}, "name": "Random",
                })
                result["plots"].append({
                    "title": "ROC Curves",
                    "data": roc_traces,
                    "layout": {
                        "template": "plotly_dark", "height": 350,
                        "xaxis": {"title": "False Positive Rate"},
                        "yaxis": {"title": "True Positive Rate"},
                    }
                })

        # Actual vs Predicted overlay (regression)
        if task_type == "regression":
            avp_traces = []
            colors = ["#4a9f6e", "#e89547", "#6a7fff", "#e8c547", "#ff7eb9", "#4a9faf"]
            for i, (name, mdl) in enumerate(models[:4]):
                try:
                    mdl_fitted = mdl.__class__(**mdl.get_params()) if not isinstance(mdl, Pipeline) else mdl
                    mdl_fitted.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train.values if hasattr(y_train, 'values') else y_train)
                    y_pred_cmp = mdl_fitted.predict(X_test.values if hasattr(X_test, 'values') else X_test)
                    avp_traces.append({
                        "type": "scatter", "x": y_test.tolist(), "y": y_pred_cmp.tolist(),
                        "mode": "markers", "marker": {"color": colors[i % len(colors)], "size": 4, "opacity": 0.6},
                        "name": name,
                    })
                except Exception:
                    pass
            if avp_traces:
                y_range = [float(y_test.min()), float(y_test.max())]
                avp_traces.append({
                    "type": "scatter", "x": y_range, "y": y_range,
                    "mode": "lines", "line": {"dash": "dash", "color": "#d94a4a"}, "name": "Perfect",
                })
                result["plots"].append({
                    "title": "Actual vs Predicted (Top Models)",
                    "data": avp_traces,
                    "layout": {
                        "template": "plotly_dark", "height": 350,
                        "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"},
                    }
                })

        # Train best model with conformal calibration (70/15/15 split)
        if best_model_obj is not None and user and user.is_authenticated:
            try:
                # Split for conformal: train on 70%, calibrate on 15%, evaluate on 15%
                if task_type == "classification":
                    Xc_train, Xc_cal, Xc_test, yc_train, yc_cal, yc_test = _stratified_split_3way(
                        pd.DataFrame(X_enc.values, columns=X_enc.columns),
                        pd.Series(y_enc.values, name=y_enc.name) if hasattr(y_enc, 'name') else pd.Series(y_enc.values),
                    )
                else:
                    Xc_train, Xc_temp, yc_train, yc_temp = train_test_split(X_enc, y_enc, test_size=0.30, random_state=42)
                    Xc_cal, Xc_test, yc_cal, yc_test = train_test_split(Xc_temp, yc_temp, test_size=0.50, random_state=42)

                best_clone = best_model_obj.__class__(**best_model_obj.get_params()) if not isinstance(best_model_obj, Pipeline) else best_model_obj
                best_clone.fit(Xc_train.values if hasattr(Xc_train, 'values') else Xc_train,
                               yc_train.values if hasattr(yc_train, 'values') else yc_train)

                conformal_state = None
                try:
                    from ..conformal import compute_conformal
                    cf = compute_conformal(best_clone, Xc_cal, yc_cal, task_type=task_type)
                    conformal_state = cf.get_state()
                except Exception:
                    pass

                model_key = str(uuid.uuid4())
                cache_meta = {
                    'model_type': f"Best: {best_model_name}",
                    'features': features,
                    'target': target,
                    'metrics': {primary_metric: best_score},
                }
                if conformal_state:
                    cache_meta['conformal_state'] = conformal_state
                    cache_meta['split_seed'] = 42
                cache_model(user.id, model_key, best_clone, cache_meta)
                result["model_key"] = model_key
                result["can_save"] = True
            except Exception:
                pass

        result["comparison"] = comparison
        result["best_model"] = best_model_name
        overfit_models = [r["model"] for r in valid_rows if r.get(f"{primary_metric}_train", 0) and r.get(primary_metric, 0) and (r[f"{primary_metric}_train"] - r[primary_metric]) / r[f"{primary_metric}_train"] * 100 > 15]
        obs = f"Compared {len(comparison)} models ({task_type}, {cv_folds}-fold CV). Best: {best_model_name} ({primary_metric}={best_score:.4f})."
        if overfit_models:
            obs += f" Overfitting risk: {', '.join(overfit_models)}."
        result["guide_observation"] = obs

    elif analysis_id == "xgboost":
        import xgboost as xgb
        from agents_api.gpu_manager import GPUTrainingContext
        from accounts.constants import can_use_ml

        if not can_use_ml(getattr(user, 'tier', 'free')):
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
            task_type = "classification" if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10)) else "regression"

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
                "n_estimators": n_estimators, "max_depth": max_depth,
                "learning_rate": learning_rate, "subsample": subsample,
                "colsample_bytree": colsample, "random_state": 42, "verbosity": 0,
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

        # Full diagnostic suite
        result["plots"].extend(_build_ml_diagnostics(
            model, X_test, y_test, y_pred, features, task_type,
            label_map=label_map, model_name=f"XGBoost ({'GPU' if gpu_used else 'CPU'})",
        ))

        # Conformal prediction
        conformal_state = None
        try:
            from ..conformal import compute_conformal
            cf = compute_conformal(model, X_cal, y_cal, task_type=task_type)
            conformal_state = cf.get_state()

            if task_type == "regression":
                qhat_90 = cf.qhats.get("0.1", 0)
                y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
                covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
                emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                result["summary"] += f"\n\nConformal 90% interval: ±{qhat_90:.4f} (empirical coverage: {emp_coverage:.1%}, n_cal={cf.n_cal})"
            else:
                if hasattr(model, 'predict_proba'):
                    proba_test = model.predict_proba(X_test)
                    pred_sets, meta = cf.predict_sets(proba_test, alpha=0.10)
                    covered = sum(1 for i, ps in enumerate(pred_sets) if int(y_test.iloc[i]) in ps)
                    emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                    avg_ss = float(np.mean([len(ps) for ps in pred_sets]))
                    result["summary"] += f"\n\nConformal 90% prediction sets: avg size={avg_ss:.2f}, coverage={emp_coverage:.1%}, n_cal={cf.n_cal}"

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = conformal_state
        except Exception as e:
            logger.warning(f"Conformal prediction failed for XGBoost: {e}")

        # Cache model
        if user and user.is_authenticated:
            model_key = str(uuid.uuid4())
            cache_meta = {
                'model_type': f"XGBoost ({'GPU' if gpu_used else 'CPU'})",
                'features': features, 'target': target, 'metrics': metrics_dict,
            }
            if conformal_state:
                cache_meta['conformal_state'] = conformal_state
                cache_meta['split_seed'] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

        result["guide_observation"] = f"XGBoost {'(GPU)' if gpu_used else '(CPU)'}: {list(metrics_dict.values())[0]:.4f} ({list(metrics_dict.keys())[0]})."

    elif analysis_id == "lightgbm":
        import lightgbm as lgb
        from agents_api.gpu_manager import GPUTrainingContext
        from accounts.constants import can_use_ml

        if not can_use_ml(getattr(user, 'tier', 'free')):
            result["summary"] = "Error: LightGBM requires a paid tier with ML access."
            return result

        n_estimators = int(config.get("n_estimators", 100))
        num_leaves = int(config.get("num_leaves", 31))
        learning_rate = float(config.get("learning_rate", 0.1))
        subsample = float(config.get("subsample", 0.8))

        task_type = config.get("task_type", "auto")
        if task_type == "auto":
            task_type = "classification" if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10)) else "regression"

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
                "n_estimators": n_estimators, "num_leaves": num_leaves,
                "learning_rate": learning_rate, "subsample": subsample,
                "random_state": 42, "verbosity": -1,
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

        # Full diagnostic suite
        result["plots"].extend(_build_ml_diagnostics(
            model, X_test, y_test, y_pred, features, task_type,
            label_map=label_map, model_name=f"LightGBM ({'GPU' if gpu_used else 'CPU'})",
        ))

        # Conformal prediction
        conformal_state = None
        try:
            from ..conformal import compute_conformal
            cf = compute_conformal(model, X_cal, y_cal, task_type=task_type)
            conformal_state = cf.get_state()

            if task_type == "regression":
                qhat_90 = cf.qhats.get("0.1", 0)
                y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
                covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
                emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                result["summary"] += f"\n\nConformal 90% interval: ±{qhat_90:.4f} (empirical coverage: {emp_coverage:.1%}, n_cal={cf.n_cal})"
            else:
                if hasattr(model, 'predict_proba'):
                    proba_test = model.predict_proba(X_test)
                    pred_sets, meta = cf.predict_sets(proba_test, alpha=0.10)
                    covered = sum(1 for i, ps in enumerate(pred_sets) if int(y_test.iloc[i]) in ps)
                    emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0
                    avg_ss = float(np.mean([len(ps) for ps in pred_sets]))
                    result["summary"] += f"\n\nConformal 90% prediction sets: avg size={avg_ss:.2f}, coverage={emp_coverage:.1%}, n_cal={cf.n_cal}"

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = conformal_state
        except Exception as e:
            logger.warning(f"Conformal prediction failed for LightGBM: {e}")

        if user and user.is_authenticated:
            model_key = str(uuid.uuid4())
            cache_meta = {
                'model_type': f"LightGBM ({'GPU' if gpu_used else 'CPU'})",
                'features': features, 'target': target, 'metrics': metrics_dict,
            }
            if conformal_state:
                cache_meta['conformal_state'] = conformal_state
                cache_meta['split_seed'] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

        result["guide_observation"] = f"LightGBM {'(GPU)' if gpu_used else '(CPU)'}: {list(metrics_dict.values())[0]:.4f} ({list(metrics_dict.keys())[0]})."

    elif analysis_id == "shap_explain":
        import shap
        from accounts.constants import can_use_ml

        if not can_use_ml(getattr(user, 'tier', 'free')):
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
        is_tree = hasattr(model_obj, 'feature_importances_') and type(model_obj).__name__ in (
            'RandomForestClassifier', 'RandomForestRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
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

        feature_names_explain = list(X_explain.columns) if hasattr(X_explain, 'columns') else model_features

        # Global: mean absolute SHAP values (feature importance bar)
        mean_abs = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        result["plots"].append({
            "title": "SHAP Feature Importance (mean |SHAP|)",
            "data": [{
                "type": "bar", "orientation": "h",
                "x": mean_abs[sorted_idx].tolist(),
                "y": [feature_names_explain[i] for i in sorted_idx],
                "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
            }],
            "layout": {"template": "plotly_dark", "height": max(250, len(feature_names_explain) * 25), "xaxis": {"title": "mean |SHAP value|"}},
        })

        # Beeswarm plot (scatter of SHAP values per feature)
        beeswarm_traces = []
        for i in sorted_idx[-10:]:  # Top 10
            feat_name = feature_names_explain[i]
            vals = sv[:, i]
            feat_vals = bg_data.iloc[:, i].values if hasattr(bg_data, 'iloc') else bg_data[:, i]
            # Normalize feature values for color
            fmin, fmax = float(np.min(feat_vals)), float(np.max(feat_vals))
            colors = ((feat_vals - fmin) / (fmax - fmin + 1e-10)).tolist()
            jitter = np.random.normal(0, 0.15, len(vals))
            beeswarm_traces.append({
                "type": "scatter", "mode": "markers",
                "x": vals.tolist(),
                "y": (jitter + len(sorted_idx) - 1 - np.where(sorted_idx == i)[0][0]).tolist(),
                "marker": {"size": 3, "color": colors, "colorscale": [[0, "#4a9faf"], [1, "#d94a4a"]], "opacity": 0.7},
                "name": feat_name, "showlegend": False,
                "hovertemplate": f"{feat_name}<br>SHAP: %{{x:.3f}}<br>Value: %{{text}}<extra></extra>",
                "text": [f"{v:.2f}" for v in feat_vals],
            })
        if beeswarm_traces:
            tick_labels = [feature_names_explain[i] for i in sorted_idx[-10:]][::-1]
            result["plots"].append({
                "title": "SHAP Beeswarm (Top 10 Features)",
                "data": beeswarm_traces,
                "layout": {
                    "template": "plotly_dark", "height": 350,
                    "xaxis": {"title": "SHAP value (impact on prediction)", "zeroline": True, "zerolinecolor": "#555"},
                    "yaxis": {"tickvals": list(range(len(tick_labels))), "ticktext": tick_labels},
                    "showlegend": False,
                },
            })

        # Single prediction waterfall (if requested or always show first sample)
        if explain_mode == "single" or True:
            idx = min(sample_idx, len(sv) - 1)
            single_sv = sv[idx]
            base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0])

            # Waterfall as horizontal bar
            sorted_single = np.argsort(np.abs(single_sv))[-10:]
            wf_names = [feature_names_explain[i] for i in sorted_single]
            wf_vals = [float(single_sv[i]) for i in sorted_single]
            wf_colors = ["rgba(74,159,110,0.7)" if v >= 0 else "rgba(208,96,96,0.7)" for v in wf_vals]

            result["plots"].append({
                "title": f"SHAP Waterfall (Sample #{idx})",
                "data": [{
                    "type": "bar", "orientation": "h",
                    "x": wf_vals, "y": wf_names,
                    "marker": {"color": wf_colors},
                }],
                "layout": {
                    "template": "plotly_dark", "height": max(200, len(wf_names) * 25),
                    "xaxis": {"title": "SHAP value"},
                    "annotations": [{"x": 0, "y": -0.15, "xref": "paper", "yref": "paper",
                                     "text": f"Base value: {base_val:.3f}", "showarrow": False,
                                     "font": {"size": 11, "color": "#9aaa9a"}}],
                },
            })

        # Dependence plot (top feature vs SHAP)
        top_feat_idx = sorted_idx[-1]
        dep_x = bg_data.iloc[:, top_feat_idx].values if hasattr(bg_data, 'iloc') else bg_data[:, top_feat_idx]
        dep_y = sv[:, top_feat_idx]
        result["plots"].append({
            "title": f"SHAP Dependence: {feature_names_explain[top_feat_idx]}",
            "data": [{
                "type": "scatter", "mode": "markers",
                "x": dep_x.tolist(), "y": dep_y.tolist(),
                "marker": {"color": "rgba(74,159,110,0.5)", "size": 4},
            }],
            "layout": {
                "template": "plotly_dark", "height": 300,
                "xaxis": {"title": feature_names_explain[top_feat_idx]},
                "yaxis": {"title": "SHAP value"},
            },
        })

        # Summary text
        top_3 = [(feature_names_explain[i], float(mean_abs[i])) for i in sorted_idx[-3:]][::-1]
        result["summary"] = (
            f"SHAP Explainability Analysis\n\n"
            f"Explainer: {'TreeExplainer' if is_tree else 'KernelExplainer'}\n"
            f"Background samples: {len(bg_data)}\n\n"
            f"Top 3 features by mean |SHAP|:\n"
            + "\n".join(f"  {i+1}. {name}: {val:.4f}" for i, (name, val) in enumerate(top_3))
        )

        result["guide_observation"] = f"SHAP: top feature is {top_3[0][0]} (mean |SHAP| = {top_3[0][1]:.4f})."

    elif analysis_id == "hyperparameter_tune":
        import optuna
        from accounts.constants import can_use_ml
        import time as _time

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if not can_use_ml(getattr(user, 'tier', 'free')):
            result["summary"] = "Error: Hyperparameter tuning requires a paid tier with ML access."
            return result

        model_type = config.get("model_type", "rf")
        n_trials = min(int(config.get("n_trials", 30)), 50)
        task_type = config.get("task_type", "auto")
        cv_folds = int(config.get("cv_folds", 3))

        if task_type == "auto":
            task_type = "classification" if (y.nunique() <= 20 and (y.dtype == object or y.dtype.name == "category" or y.nunique() <= 10)) else "regression"

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
                    "random_state": 42, "n_jobs": -1,
                }
                mdl = RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)
            elif model_type == "xgboost":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "random_state": 42, "verbosity": 0,
                }
                mdl = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss") if task_type == "classification" else xgb.XGBRegressor(**params)
            elif model_type == "lightgbm":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 127),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "random_state": 42, "verbosity": -1,
                }
                mdl = lgb.LGBMClassifier(**params) if task_type == "classification" else lgb.LGBMRegressor(**params)
            elif model_type == "ridge":
                from sklearn.linear_model import Ridge, LogisticRegression
                alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
                if task_type == "classification":
                    mdl = LogisticRegression(C=1.0/alpha, max_iter=1000, penalty="l2")
                else:
                    mdl = Ridge(alpha=alpha)
            elif model_type == "lasso":
                from sklearn.linear_model import Lasso, LogisticRegression
                alpha = trial.suggest_float("alpha", 0.001, 10.0, log=True)
                if task_type == "classification":
                    mdl = LogisticRegression(C=1.0/alpha, max_iter=1000, penalty="l1", solver="saga")
                else:
                    mdl = Lasso(alpha=alpha)
            else:
                # Default: random forest
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "random_state": 42, "n_jobs": -1,
                }
                mdl = RandomForestClassifier(**params) if task_type == "classification" else RandomForestRegressor(**params)

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
            f"\nBest Parameters:",
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

        result["plots"].append({
            "title": "Optimization History",
            "data": [
                {"type": "scatter", "x": trial_numbers, "y": trial_values, "mode": "markers",
                 "marker": {"color": "rgba(74,159,110,0.4)", "size": 5}, "name": "Trial Score"},
                {"type": "scatter", "x": trial_numbers, "y": best_so_far, "mode": "lines",
                 "line": {"color": "#4a9f6e", "width": 2}, "name": "Best So Far"},
            ],
            "layout": {
                "template": "plotly_dark", "height": 300,
                "xaxis": {"title": "Trial"}, "yaxis": {"title": "Score"},
            },
        })

        # Parameter importance (if enough trials)
        if len(study.trials) >= 5:
            try:
                param_importance = optuna.importance.get_param_importances(study)
                p_names = list(param_importance.keys())[::-1]
                p_vals = [param_importance[k] for k in p_names]
                result["plots"].append({
                    "title": "Hyperparameter Importance",
                    "data": [{
                        "type": "bar", "orientation": "h",
                        "x": p_vals, "y": p_names,
                        "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                    }],
                    "layout": {"template": "plotly_dark", "height": max(200, len(p_names) * 30)},
                })
            except Exception:
                pass

        # Train final model with best params and cache
        if user and user.is_authenticated:
            try:
                # Rebuild model with best params
                obj_trial = type('FakeTrial', (), {'suggest_int': lambda s, n, *a, **kw: best_params[n],
                                                    'suggest_float': lambda s, n, *a, **kw: best_params[n]})()
                # Simpler: just train directly
                if model_type == "rf":
                    final_model = (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(
                        **{k: v for k, v in best_params.items()}, random_state=42, n_jobs=-1)
                elif model_type == "xgboost":
                    import xgboost as xgb
                    final_model = (xgb.XGBClassifier if task_type == "classification" else xgb.XGBRegressor)(
                        **{k: v for k, v in best_params.items()}, random_state=42, verbosity=0)
                elif model_type == "lightgbm":
                    import lightgbm as lgb
                    final_model = (lgb.LGBMClassifier if task_type == "classification" else lgb.LGBMRegressor)(
                        **{k: v for k, v in best_params.items()}, random_state=42, verbosity=-1)
                elif model_type in ("ridge", "lasso"):
                    from sklearn.linear_model import Ridge, Lasso, LogisticRegression
                    alpha = best_params.get("alpha", 1.0)
                    if task_type == "classification":
                        penalty = "l1" if model_type == "lasso" else "l2"
                        solver = "saga" if model_type == "lasso" else "lbfgs"
                        final_model = LogisticRegression(C=1.0/alpha, max_iter=1000, penalty=penalty, solver=solver)
                    else:
                        final_model = (Lasso if model_type == "lasso" else Ridge)(alpha=alpha)
                else:
                    final_model = (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(
                        **{k: v for k, v in best_params.items()}, random_state=42, n_jobs=-1)

                final_model.fit(X_enc, y_work)
                model_key = str(uuid.uuid4())
                primary = "accuracy" if task_type == "classification" else "r2"
                cache_model(user.id, model_key, final_model, {
                    'model_type': f"Tuned {model_type.upper()}",
                    'features': features, 'target': target,
                    'metrics': {primary: best_score},
                })
                result["model_key"] = model_key
                result["can_save"] = True
            except Exception as e_tune:
                logger.warning(f"Could not cache tuned model: {e_tune}")

        result["best_params"] = best_params
        result["guide_observation"] = f"Tuned {model_type.upper()}: best score {best_score:.4f} in {len(study.trials)} trials."

    elif analysis_id == "clustering":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = int(config.get("n_clusters", 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Silhouette for selected k
        from sklearn.metrics import silhouette_score as _sil_score
        sil = _sil_score(X_scaled, clusters) if n_clusters > 1 and n_clusters < len(X_scaled) else 0.0

        # Cluster size distribution
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>K-MEANS CLUSTERING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Clusters:<</COLOR>> {n_clusters}\n"
        summary += f"<<COLOR:highlight>>Inertia:<</COLOR>> {kmeans.inertia_:.2f}\n"
        summary += f"<<COLOR:highlight>>Silhouette Score:<</COLOR>> {sil:.3f}\n\n"

        summary += f"<<COLOR:text>>Cluster Sizes:<</COLOR>>\n"
        for c_id, c_size in cluster_sizes.items():
            summary += f"  Cluster {c_id}: {c_size} observations ({c_size/len(clusters)*100:.0f}%)\n"

        summary += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CLUSTER QUALITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"

        if sil >= 0.7:
            summary += f"<<COLOR:good>>Strong cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += f"<<COLOR:text>>Clusters are well-separated and internally cohesive. This grouping reflects real structure in the data.<</COLOR>>"
        elif sil >= 0.5:
            summary += f"<<COLOR:good>>Reasonable cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += f"<<COLOR:text>>Clusters have meaningful separation. Some overlap exists but the grouping is useful.<</COLOR>>"
        elif sil >= 0.25:
            summary += f"<<COLOR:warn>>Weak cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += f"<<COLOR:text>>Clusters overlap substantially. Try different k values or different features. The data may not have clear natural groups.<</COLOR>>"
        else:
            summary += f"<<COLOR:danger>>No meaningful cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += f"<<COLOR:text>>Clusters are essentially arbitrary. The data does not separate into distinct groups with these features.<</COLOR>>"

        # Size imbalance warning
        max_size = cluster_sizes.max()
        min_size = cluster_sizes.min()
        if max_size > 5 * min_size:
            summary += f"\n\n<<COLOR:warn>>Imbalanced clusters:<</COLOR>> largest is {max_size/min_size:.0f}x the smallest. One cluster may be catching 'everything else.' Consider increasing k."

        result["summary"] = summary

        # Scatter plot of first two features colored by cluster
        if len(features) >= 2:
            result["plots"].append({
                "title": f"Clusters ({features[0]} vs {features[1]})",
                "data": [{
                    "type": "scatter",
                    "x": X[features[0]].tolist(),
                    "y": X[features[1]].tolist(),
                    "mode": "markers",
                    "marker": {"color": clusters.tolist(), "colorscale": "Viridis", "size": 6}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })

        # Elbow plot with silhouette scores
        max_k = min(10, len(X_scaled) - 1)
        if max_k >= 2:
            from sklearn.metrics import silhouette_score
            k_range = range(2, max_k + 1)
            inertias = []
            silhouettes = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lab = km.fit_predict(X_scaled)
                inertias.append(float(km.inertia_))
                silhouettes.append(float(silhouette_score(X_scaled, lab)))
            best_k = list(k_range)[np.argmax(silhouettes)]
            result["plots"].append({
                "title": "Elbow Plot & Silhouette Score",
                "data": [
                    {"type": "scatter", "x": list(k_range), "y": inertias, "mode": "lines+markers", "marker": {"color": "#4a9f6e", "size": 7}, "line": {"color": "#4a9f6e"}, "name": "Inertia", "yaxis": "y"},
                    {"type": "scatter", "x": list(k_range), "y": silhouettes, "mode": "lines+markers", "marker": {"color": "#e89547", "size": 7}, "line": {"color": "#e89547"}, "name": "Silhouette", "yaxis": "y2"},
                    {"type": "scatter", "x": [best_k], "y": [max(silhouettes)], "mode": "markers", "marker": {"color": "#d94a4a", "size": 12, "symbol": "star"}, "name": f"Best k={best_k}", "yaxis": "y2"}
                ],
                "layout": {
                    "template": "plotly_dark", "height": 300,
                    "xaxis": {"title": "Number of Clusters (k)"},
                    "yaxis": {"title": "Inertia", "side": "left"},
                    "yaxis2": {"title": "Silhouette Score", "side": "right", "overlaying": "y"},
                    "legend": {"x": 0.5, "y": 1.15, "orientation": "h"}
                }
            })

        sil_quality = "strong" if sil >= 0.5 else "weak" if sil >= 0.25 else "no meaningful"
        result["guide_observation"] = f"K-Means: {n_clusters} clusters, silhouette={sil:.3f} ({sil_quality} structure). " + (
            f"Optimal k by silhouette: {best_k}." if max_k >= 2 else "")

    elif analysis_id == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        n_components = int(config.get("n_components", 2))
        color_by = config.get("color")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PRINCIPAL COMPONENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {pca.n_components_}\n\n"

        summary += f"<<COLOR:text>>Explained Variance:<</COLOR>>\n"
        cumulative = 0
        for i, (var, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
            cumulative += ratio * 100
            summary += f"  PC{i+1}: {ratio*100:.1f}% (cumulative: {cumulative:.1f}%)\n"

        summary += f"\n<<COLOR:text>>Loadings (feature weights):<</COLOR>>\n"
        for i, component in enumerate(pca.components_[:3]):  # Show first 3 PCs max
            summary += f"\n  PC{i+1}:\n"
            sorted_idx = np.argsort(np.abs(component))[::-1]
            for j in sorted_idx[:5]:  # Top 5 loadings
                summary += f"    {features[j]}: {component[j]:.3f}\n"

        result["summary"] = summary

        # Biplot (first 2 components)
        if pca.n_components_ >= 2:
            color_values = None
            if color_by and color_by in df.columns:
                color_values = df[color_by].loc[X.index].astype(str).tolist()

            scatter_data = {
                "type": "scatter",
                "x": X_pca[:, 0].tolist(),
                "y": X_pca[:, 1].tolist(),
                "mode": "markers",
                "marker": {"size": 6},
                "name": "Observations"
            }
            if color_values:
                scatter_data["text"] = color_values
                scatter_data["marker"]["color"] = [hash(v) % 10 for v in color_values]
                scatter_data["marker"]["colorscale"] = "Viridis"

            result["plots"].append({
                "title": "PCA Biplot",
                "data": [scatter_data],
                "layout": {
                    "template": "plotly_dark",
                    "height": 400,
                    "xaxis": {"title": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"},
                    "yaxis": {"title": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"}
                }
            })

        # Scree plot
        result["plots"].append({
            "title": "Scree Plot",
            "data": [{
                "type": "bar",
                "x": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "y": (pca.explained_variance_ratio_ * 100).tolist(),
                "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
            }],
            "layout": {
                "template": "plotly_dark",
                "height": 250,
                "yaxis": {"title": "Variance Explained (%)"}
            }
        })

    elif analysis_id == "feature":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Determine if classification or regression
        y_unique = y.nunique()
        is_classification = y.dtype == 'object' or y_unique < 10

        if is_classification:
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y_encoded)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>FEATURE IMPORTANCE (Random Forest)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Task:<</COLOR>> {'Classification' if is_classification else 'Regression'}\n\n"

        summary += f"<<COLOR:text>>Feature Rankings:<</COLOR>>\n"
        for rank, idx in enumerate(indices, 1):
            bar_len = int(importances[idx] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            summary += f"  {rank}. {features[idx]:<20} {bar} {importances[idx]:.3f}\n"

        result["summary"] = summary

        # Horizontal bar chart
        sorted_features = [features[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]

        result["plots"].append({
            "title": "Feature Importance",
            "data": [{
                "type": "bar",
                "x": sorted_importances[::-1],
                "y": sorted_features[::-1],
                "orientation": "h",
                "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
            }],
            "layout": {
                "template": "plotly_dark",
                "height": max(250, len(features) * 25),
                "xaxis": {"title": "Importance"},
                "margin": {"l": 150}
            }
        })

    # =========================================================================
    # PHASE 1: CAUSAL LENS TOOLKIT
    # =========================================================================

    elif analysis_id == "bayesian_regression":
        """
        Bayesian Linear Regression - native uncertainty, posterior over coefficients.
        Directly feeds the belief engine with credible intervals.
        """
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        # Scale features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Bayesian Ridge Regression
        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        # Get predictions with uncertainty
        y_pred, y_std = model.predict(X_scaled, return_std=True)

        # Coefficient posteriors (mean and std)
        coef_mean = model.coef_
        # Approximate coefficient std from the model's alpha/lambda
        coef_std = np.sqrt(1.0 / model.lambda_) * np.ones_like(coef_mean)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN LINEAR REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n\n"

        # Model fit
        r2 = model.score(X_scaled, y)
        summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
        summary += f"  R² = {r2:.4f}\n"
        summary += f"  α (noise precision) = {model.alpha_:.4f}\n"
        summary += f"  λ (weight precision) = {model.lambda_:.4f}\n\n"

        # Coefficient posteriors with credible intervals
        summary += f"<<COLOR:text>>Coefficient Posteriors (95% Credible Intervals):<</COLOR>>\n"
        summary += f"<<COLOR:text>>These intervals feed directly into Synara edge weights<</COLOR>>\n\n"

        for i, feat in enumerate(features):
            mean = coef_mean[i]
            std = coef_std[i]
            ci_low = mean - 1.96 * std
            ci_high = mean + 1.96 * std

            # Determine significance (CI doesn't include 0)
            sig = "***" if ci_low > 0 or ci_high < 0 else ""

            summary += f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"

        summary += f"\n  Intercept: {model.intercept_:.4f}\n"

        # Synara integration note
        summary += f"\n<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += f"  Coefficients with CIs not crossing zero indicate strong causal evidence.\n"
        summary += f"  CI width indicates uncertainty in edge weight.\n"

        result["summary"] = summary

        # Plot 1: Coefficient posteriors with credible intervals
        result["plots"].append({
            "title": "Coefficient Posteriors (95% CI)",
            "data": [{
                "type": "scatter",
                "x": coef_mean.tolist(),
                "y": features,
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 10},
                "error_x": {
                    "type": "data",
                    "array": (1.96 * coef_std).tolist(),
                    "color": "#4a9f6e",
                    "thickness": 2,
                    "width": 6
                },
                "name": "Coefficient ± 95% CI"
            }, {
                "type": "scatter",
                "x": [0],
                "y": [features[len(features)//2]],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 1},
                "showlegend": False
            }],
            "layout": {
                "height": max(300, len(features) * 30),
                "xaxis": {"title": "Coefficient Value", "zeroline": True, "zerolinecolor": "#e85747"},
                "margin": {"l": 150},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(features)-0.5, "line": {"color": "#e85747", "dash": "dash"}}]
            }
        })

        # Plot 2: Predictions with uncertainty bands
        sorted_idx = np.argsort(y.values)
        result["plots"].append({
            "title": "Predictions with Uncertainty",
            "data": [{
                "type": "scatter",
                "x": list(range(len(y))),
                "y": y.values[sorted_idx].tolist(),
                "mode": "markers",
                "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 5, "line": {"color": "#4a9f6e", "width": 1}},
                "name": "Actual"
            }, {
                "type": "scatter",
                "x": list(range(len(y))),
                "y": y_pred[sorted_idx].tolist(),
                "mode": "lines",
                "line": {"color": "#e89547"},
                "name": "Predicted"
            }, {
                "type": "scatter",
                "x": list(range(len(y))) + list(range(len(y)))[::-1],
                "y": (y_pred[sorted_idx] + 1.96*y_std[sorted_idx]).tolist() + (y_pred[sorted_idx] - 1.96*y_std[sorted_idx])[::-1].tolist(),
                "fill": "toself",
                "fillcolor": "rgba(232, 149, 71, 0.2)",
                "line": {"color": "transparent"},
                "name": "95% CI"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Observation (sorted)"}, "yaxis": {"title": target}}
        })

        result["guide_observation"] = f"Bayesian regression R²={r2:.3f}. Coefficients with CIs not crossing zero are significant causal candidates."

        # Include coefficient data for Synara integration
        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {
                    "feature": feat,
                    "mean": float(coef_mean[i]),
                    "ci_low": float(coef_mean[i] - 1.96 * coef_std[i]),
                    "ci_high": float(coef_mean[i] + 1.96 * coef_std[i])
                }
                for i, feat in enumerate(features)
            ]
        }

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'model': model, 'scaler': scaler}, {
                'model_type': "Bayesian Ridge Regression",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "gam":
        """
        Generalized Additive Model - human-readable spline curves per feature.
        Shows HOW each variable bends the response (interpretable nonlinearity).
        """
        try:
            from pygam import LinearGAM, s
        except ImportError:
            result["summary"] = "Error: pygam not installed. Run: pip install pygam"
            return result

        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build GAM with spline terms for each feature
        # s(i) creates a spline term for feature i
        terms = s(0)
        for i in range(1, X_scaled.shape[1]):
            terms += s(i)

        gam = LinearGAM(terms)
        gam.fit(X_scaled, y)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GENERALIZED ADDITIVE MODEL (GAM)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> y = Σ f_i(x_i) + ε  (smooth splines)\n\n"

        # Model statistics
        summary += f"<<COLOR:text>>Model Statistics:<</COLOR>>\n"
        summary += f"  Pseudo R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}\n"
        summary += f"  GCV Score = {gam.statistics_['GCV']:.4f}\n"
        summary += f"  Effective DF = {gam.statistics_['edof']:.1f}\n\n"

        # Feature significance (approximate p-values)
        summary += f"<<COLOR:text>>Feature Effects (Spline Significance):<</COLOR>>\n"
        summary += f"<<COLOR:text>>Each curve shows HOW the feature affects {target}<</COLOR>>\n\n"

        p_values = gam.statistics_['p_values']
        for i, feat in enumerate(features):
            p = p_values[i] if i < len(p_values) else 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary += f"  {feat:<20} p = {p:.4f} {sig}\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  The partial dependence plots show the SHAPE of each effect.\n"
        summary += f"  Non-linear curves reveal complex causal relationships.\n"

        result["summary"] = summary

        # Generate partial dependence plots for each feature
        for i, feat in enumerate(features):
            try:
                XX = gam.generate_X_grid(term=i, n=100)
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                # Convert back to original scale for x-axis
                x_grid = XX[:, i] * scaler.scale_[i] + scaler.mean_[i]

                result["plots"].append({
                    "title": f"Effect of {feat}",
                    "data": [{
                        "type": "scatter",
                        "x": x_grid.tolist(),
                        "y": pdep.tolist(),
                        "mode": "lines",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "name": "Effect"
                    }, {
                        "type": "scatter",
                        "x": x_grid.tolist() + x_grid[::-1].tolist(),
                        "y": confi[:, 0].tolist() + confi[::-1, 1].tolist(),
                        "fill": "toself",
                        "fillcolor": "rgba(74, 159, 110, 0.2)",
                        "line": {"color": "transparent"},
                        "name": "95% CI"
                    }],
                    "layout": {
                        "height": 250,
                        "xaxis": {"title": feat},
                        "yaxis": {"title": f"Effect on {target}"},
                        "showlegend": False
                    }
                })
            except Exception:
                logger.warning(f"GAM: partial dependence failed for feature '{feat}', skipping plot")

        result["guide_observation"] = f"GAM shows smooth effect curves for each feature. Non-linear patterns indicate complex causal relationships."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'gam': gam, 'scaler': scaler}, {
                'model_type': "Generalized Additive Model",
                'features': features,
                'target': target,
                'metrics': {'pseudo_r2': float(gam.statistics_['pseudo_r2']['explained_deviance'])}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "isolation_forest":
        """
        Isolation Forest - anomaly detection as 'missing cause' signal.
        Points that don't fit trigger causal expansion in Synara.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        contamination = float(config.get("contamination", 0.05))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        # Identify anomalies
        anomalies = predictions == -1
        n_anomalies = anomalies.sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ISOLATION FOREST (Anomaly Detection)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Contamination:<</COLOR>> {contamination:.1%}\n\n"

        summary += f"<<COLOR:text>>Results:<</COLOR>>\n"
        summary += f"  Total observations: {len(predictions)}\n"
        summary += f"  Anomalies detected: {n_anomalies} ({anomaly_pct:.1f}%)\n"
        summary += f"  Normal observations: {len(predictions) - n_anomalies}\n\n"

        # Anomaly score statistics
        summary += f"<<COLOR:text>>Anomaly Scores:<</COLOR>>\n"
        summary += f"  Mean score: {scores.mean():.4f}\n"
        summary += f"  Std score: {scores.std():.4f}\n"
        summary += f"  Threshold: ~0.0 (negative = anomaly)\n\n"

        # Show most anomalous observations
        if n_anomalies > 0:
            summary += f"<<COLOR:warning>>MOST ANOMALOUS OBSERVATIONS:<</COLOR>>\n"
            anomaly_idx = np.where(anomalies)[0]
            sorted_anomalies = anomaly_idx[np.argsort(scores[anomaly_idx])][:10]

            for idx in sorted_anomalies:
                summary += f"  Row {idx}: score={scores[idx]:.4f}\n"
                for feat in features[:3]:
                    summary += f"    {feat}={X[feat].iloc[idx]:.2f}\n"
                summary += "\n"

        summary += f"<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += f"  Anomalies are observations that don't fit the current model.\n"
        summary += f"  This signals MISSING CAUSES - trigger causal expansion.\n"
        summary += f"  Investigate what makes these points different.\n"

        result["summary"] = summary

        # Plot 1: Anomaly scores distribution
        result["plots"].append({
            "title": "Anomaly Score Distribution",
            "data": [{
                "type": "histogram",
                "x": scores.tolist(),
                "nbinsx": 50,
                "marker": {"color": "rgba(74, 159, 110, 0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                "name": "Scores"
            }, {
                "type": "scatter",
                "x": [0, 0],
                "y": [0, len(scores) / 10],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
                "name": "Threshold"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Anomaly Score (negative = anomaly)"},
                "yaxis": {"title": "Count"},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash"}}]
            }
        })

        # Plot 2: Scatter with anomalies highlighted (first 2 features)
        if len(features) >= 2:
            colors = ['#e85747' if a else 'rgba(74, 159, 110, 0.5)' for a in anomalies]
            sizes = [12 if a else 6 for a in anomalies]

            result["plots"].append({
                "title": f"Anomalies: {features[0]} vs {features[1]}",
                "data": [{
                    "type": "scatter",
                    "x": X[features[0]].tolist(),
                    "y": X[features[1]].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": colors,
                        "size": sizes,
                        "line": {"color": "#e85747", "width": [1 if a else 0 for a in anomalies]}
                    },
                    "text": [f"Score: {s:.3f}" for s in scores],
                    "hoverinfo": "text+x+y"
                }],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": features[0]},
                    "yaxis": {"title": features[1]}
                }
            })

        # Store anomaly data for potential export
        result["anomalies"] = {
            "indices": np.where(anomalies)[0].tolist(),
            "scores": scores[anomalies].tolist(),
            "count": int(n_anomalies)
        }

        result["guide_observation"] = f"Isolation Forest detected {n_anomalies} anomalies ({anomaly_pct:.1f}%). These are 'missing cause' signals - investigate what makes them different."

    elif analysis_id == "gaussian_process":
        """
        Gaussian Process Regression - uncertainty quantification over subsets.
        Shows mean prediction with confidence bands.
        GP is O(n³) — subsample large datasets to avoid freezing.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        # GP is O(n³) — cap at 500 rows to prevent freezing
        MAX_GP_ROWS = 500
        n_rows = len(X)
        subsampled = False
        if n_rows > MAX_GP_ROWS:
            idx = np.random.RandomState(42).choice(n_rows, MAX_GP_ROWS, replace=False)
            idx.sort()
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            subsampled = True

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define kernel: constant * RBF + noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

        # Fit GP — reduce restarts for larger datasets
        n_restarts = 2 if len(X) > 300 else 5
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
        gp.fit(X_scaled, y)

        # Predictions with uncertainty
        y_pred, y_std = gp.predict(X_scaled, return_std=True)

        # For visualization, use first feature
        feature_for_plot = features[0]
        x_plot = X[feature_for_plot].values
        sort_idx = np.argsort(x_plot)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAUSSIAN PROCESS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        if subsampled:
            summary += f"<<COLOR:warning>>Note: Subsampled to {MAX_GP_ROWS} rows (from {n_rows}) — GP is O(n³)<</COLOR>>\n"
        summary += f"\n"

        summary += f"<<COLOR:text>>Kernel:<</COLOR>>\n"
        summary += f"  {gp.kernel_}\n\n"

        summary += f"<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += f"  Log-Marginal-Likelihood: {gp.log_marginal_likelihood_value_:.4f}\n"

        # Residual analysis
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean())**2)
        summary += f"  RMSE: {rmse:.4f}\n"
        summary += f"  R²: {r2:.4f}\n\n"

        summary += f"<<COLOR:text>>Uncertainty Statistics:<</COLOR>>\n"
        summary += f"  Mean std: {y_std.mean():.4f}\n"
        summary += f"  Max std: {y_std.max():.4f}\n"
        summary += f"  Min std: {y_std.min():.4f}\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  GP provides full uncertainty quantification.\n"
        summary += f"  Wide bands = less confident predictions.\n"
        summary += f"  Useful for detecting extrapolation risk.\n"

        result["summary"] = summary

        # Plot 1: Predictions with uncertainty bands
        result["plots"].append({
            "title": f"GP Fit: {target} vs {feature_for_plot}",
            "data": [
                {
                    "type": "scatter",
                    "x": x_plot[sort_idx].tolist(),
                    "y": y.values[sort_idx].tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6},
                    "name": "Observed"
                },
                {
                    "type": "scatter",
                    "x": x_plot[sort_idx].tolist(),
                    "y": y_pred[sort_idx].tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "GP Mean"
                },
                {
                    "type": "scatter",
                    "x": np.concatenate([x_plot[sort_idx], x_plot[sort_idx][::-1]]).tolist(),
                    "y": np.concatenate([
                        (y_pred[sort_idx] + 1.96 * y_std[sort_idx]),
                        (y_pred[sort_idx][::-1] - 1.96 * y_std[sort_idx][::-1])
                    ]).tolist(),
                    "fill": "toself",
                    "fillcolor": "rgba(74, 159, 110, 0.2)",
                    "line": {"color": "rgba(74, 159, 110, 0)"},
                    "name": "95% CI"
                }
            ],
            "layout": {
                "height": 350,
                "xaxis": {"title": feature_for_plot},
                "yaxis": {"title": target}
            }
        })

        # Plot 2: Uncertainty map
        result["plots"].append({
            "title": "Prediction Uncertainty",
            "data": [{
                "type": "scatter",
                "x": x_plot.tolist(),
                "y": y_std.tolist(),
                "mode": "markers",
                "marker": {
                    "color": y_std.tolist(),
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]],
                    "size": 8,
                    "showscale": True,
                    "colorbar": {"title": "Std"}
                },
                "name": "Uncertainty"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": feature_for_plot},
                "yaxis": {"title": "Prediction Std Dev"}
            }
        })

        result["guide_observation"] = f"Gaussian Process fit with R²={r2:.3f}. Mean uncertainty: {y_std.mean():.3f}. Check wide uncertainty bands for extrapolation risk."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'gp': gp, 'scaler': scaler}, {
                'model_type': "Gaussian Process Regressor",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "pls":
        """
        Partial Least Squares - handles collinearity in process data.
        Projects to latent space before regression.
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        n_components = int(config.get("n_components", min(3, len(features))))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PLS
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_scaled, y)

        # Get scores and loadings
        X_scores = pls.x_scores_  # T scores
        Y_scores = pls.y_scores_  # U scores
        X_loadings = pls.x_loadings_  # P loadings
        Y_loadings = pls.y_loadings_  # Q loadings
        X_weights = pls.x_weights_  # W weights

        # Predictions
        y_pred = pls.predict(X_scaled).flatten()
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean())**2)

        # Explained variance per component
        # PLS doesn't provide this directly, estimate from scores
        total_var_x = np.var(X_scaled, axis=0).sum()
        explained_var = []
        for i in range(n_components):
            comp_var = np.var(X_scores[:, i]) * np.sum(X_loadings[:, i]**2)
            explained_var.append(comp_var / total_var_x * 100)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PARTIAL LEAST SQUARES (PLS) REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {n_components}\n\n"

        summary += f"<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += f"  R²: {r2:.4f}\n"
        summary += f"  RMSE: {rmse:.4f}\n\n"

        summary += f"<<COLOR:text>>Component Details:<</COLOR>>\n"
        for i in range(n_components):
            summary += f"  Component {i+1}: ~{explained_var[i]:.1f}% X variance\n"
        summary += "\n"

        summary += f"<<COLOR:text>>Feature Weights (Importance):<</COLOR>>\n"
        # VIP scores (Variable Importance in Projection)
        vip = np.sqrt(len(features) * np.sum(X_weights**2 * np.sum(Y_scores**2, axis=0), axis=1) / np.sum(Y_scores**2))
        for feat, v in sorted(zip(features, vip), key=lambda x: -x[1]):
            marker = "★" if v > 1.0 else " "
            summary += f"  {marker} {feat}: VIP={v:.3f}\n"
        summary += "\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  PLS handles collinear features by projecting to latent space.\n"
        summary += f"  VIP > 1.0 indicates important predictors.\n"
        summary += f"  Score plots reveal sample groupings.\n"

        result["summary"] = summary

        # Plot 1: Score plot (T1 vs T2)
        if n_components >= 2:
            result["plots"].append({
                "title": "PLS Score Plot (T1 vs T2)",
                "data": [{
                    "type": "scatter",
                    "x": X_scores[:, 0].tolist(),
                    "y": X_scores[:, 1].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": y.values.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]],
                        "size": 8,
                        "showscale": True,
                        "colorbar": {"title": target}
                    },
                    "text": [f"{target}={v:.2f}" for v in y.values],
                    "hoverinfo": "text+x+y"
                }],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "T1 (Component 1)"},
                    "yaxis": {"title": "T2 (Component 2)"}
                }
            })

        # Plot 2: Loading plot (feature contributions)
        result["plots"].append({
            "title": "PLS Loadings (Component 1)",
            "data": [{
                "type": "bar",
                "x": features,
                "y": X_loadings[:, 0].tolist(),
                "marker": {
                    "color": ["#4a9f6e" if v > 0 else "#e85747" for v in X_loadings[:, 0]]
                }
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Feature"},
                "yaxis": {"title": "Loading on Component 1"}
            }
        })

        # Plot 3: VIP scores
        vip_sorted = sorted(zip(features, vip), key=lambda x: -x[1])
        result["plots"].append({
            "title": "Variable Importance (VIP)",
            "data": [{
                "type": "bar",
                "x": [f[0] for f in vip_sorted],
                "y": [f[1] for f in vip_sorted],
                "marker": {
                    "color": ["#4a9f6e" if v > 1.0 else "rgba(74, 159, 110, 0.4)" for _, v in vip_sorted]
                }
            }, {
                "type": "scatter",
                "x": [f[0] for f in vip_sorted],
                "y": [1.0] * len(features),
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash"},
                "name": "VIP=1 threshold"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Feature"},
                "yaxis": {"title": "VIP Score"},
                "showlegend": False
            }
        })

        result["vip_scores"] = {feat: float(v) for feat, v in zip(features, vip)}
        result["guide_observation"] = f"PLS with {n_components} components achieved R²={r2:.3f}. Top features by VIP: {', '.join([f[0] for f in vip_sorted[:3]])}."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'pls': pls, 'scaler': scaler}, {
                'model_type': "PLS Regression",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2), 'n_components': n_components}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "sem":
        """
        Structural Equation Modeling - causal path analysis.
        Supports path models, mediation, and latent variables.
        """
        try:
            from semopy import Model
            from semopy.stats import calc_stats
        except ImportError:
            result["summary"] = "Error: semopy not installed. Run: pip install semopy"
            return result

        model_type = config.get("model_type", "path")
        outcome = config.get("outcome")
        predictors = config.get("predictors", [])
        mediator = config.get("mediator")

        if not outcome or not predictors:
            result["summary"] = "Error: Please select outcome and predictors."
            return result

        # Build model specification based on type
        if model_type == "mediation" and mediator:
            # Classic mediation: X → M → Y (and X → Y for direct effect)
            predictor = predictors[0] if predictors else None
            if not predictor:
                result["summary"] = "Error: Mediation requires at least one predictor."
                return result

            model_spec = f"""
            # Direct effect
            {outcome} ~ c*{predictor}
            # Path a: predictor to mediator
            {mediator} ~ a*{predictor}
            # Path b: mediator to outcome
            {outcome} ~ b*{mediator}
            # Indirect effect
            indirect := a*b
            # Total effect
            total := c + a*b
            """
        else:
            # Path model: multiple predictors → outcome
            predictor_terms = " + ".join(predictors)
            model_spec = f"{outcome} ~ {predictor_terms}"

        # Prepare data - only keep relevant columns
        relevant_cols = [outcome] + predictors
        if mediator:
            relevant_cols.append(mediator)
        relevant_cols = list(set(relevant_cols))

        model_df = df[relevant_cols].dropna()

        if len(model_df) < 30:
            result["summary"] = f"Warning: Only {len(model_df)} complete cases. SEM typically needs n > 200 for stable estimates."

        # Fit model
        try:
            mod = Model(model_spec)
            mod.fit(model_df)
        except Exception as e:
            result["summary"] = f"Error fitting SEM model: {str(e)}"
            return result

        # Get estimates
        estimates = mod.inspect()

        # Get fit statistics
        try:
            stats = calc_stats(mod)
            chi2 = stats.get("chi2", [None])[0]
            dof = stats.get("dof", [None])[0]
            pvalue = stats.get("chi2 p-value", [None])[0]
            cfi = stats.get("CFI", [None])[0]
            tli = stats.get("TLI", [None])[0]
            rmsea = stats.get("RMSEA", [None])[0]
            srmr = stats.get("SRMR", [None])[0]
        except Exception:
            chi2 = dof = pvalue = cfi = tli = rmsea = srmr = None

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>STRUCTURAL EQUATION MODEL<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

        summary += f"<<COLOR:highlight>>Model Type:<</COLOR>> {model_type.title()}\n"
        summary += f"<<COLOR:highlight>>Outcome:<</COLOR>> {outcome}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        if mediator:
            summary += f"<<COLOR:highlight>>Mediator:<</COLOR>> {mediator}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(model_df)}\n\n"

        # Fit indices
        summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
        if chi2 is not None:
            summary += f"  χ² = {chi2:.3f}, df = {dof}, p = {pvalue:.4f}\n"
        if cfi is not None:
            cfi_ok = "✓" if cfi > 0.95 else "⚠" if cfi > 0.90 else "✗"
            summary += f"  CFI = {cfi:.3f} {cfi_ok} (>.95 good)\n"
        if tli is not None:
            tli_ok = "✓" if tli > 0.95 else "⚠" if tli > 0.90 else "✗"
            summary += f"  TLI = {tli:.3f} {tli_ok} (>.95 good)\n"
        if rmsea is not None:
            rmsea_ok = "✓" if rmsea < 0.05 else "⚠" if rmsea < 0.08 else "✗"
            summary += f"  RMSEA = {rmsea:.3f} {rmsea_ok} (<.05 good)\n"
        if srmr is not None:
            srmr_ok = "✓" if srmr < 0.08 else "⚠" if srmr < 0.10 else "✗"
            summary += f"  SRMR = {srmr:.3f} {srmr_ok} (<.08 good)\n"
        summary += "\n"

        # Path estimates
        summary += f"<<COLOR:text>>Path Estimates:<</COLOR>>\n"
        for _, row in estimates.iterrows():
            lval = row.get('lval', '')
            op = row.get('op', '')
            rval = row.get('rval', '')
            est = row.get('Estimate', 0)
            se = row.get('Std. Err', 0)
            pval = row.get('p-value', 1)

            if op == '~':  # Regression
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                summary += f"  {lval} ← {rval}: β = {est:.3f} (SE={se:.3f}) {sig}\n"
            elif op == ':=':  # Defined parameter
                summary += f"  <<COLOR:accent>>{lval}<</COLOR>>: {est:.3f} (SE={se:.3f})\n"
        summary += "\n"

        # Interpretation for mediation
        if model_type == "mediation" and mediator:
            indirect_row = estimates[estimates['lval'] == 'indirect']
            if not indirect_row.empty:
                indirect_est = indirect_row['Estimate'].values[0]
                indirect_p = indirect_row.get('p-value', pd.Series([1])).values[0]

                summary += f"<<COLOR:success>>MEDIATION ANALYSIS:<</COLOR>>\n"
                if indirect_p < 0.05:
                    summary += f"  Significant indirect effect: {indirect_est:.3f}\n"
                    summary += f"  {mediator} mediates the {predictors[0]} → {outcome} relationship.\n"
                else:
                    summary += f"  No significant mediation (indirect = {indirect_est:.3f}, p > .05)\n"

        result["summary"] = summary

        # Plot 1: Path diagram (simplified as coefficient bar chart)
        path_data = estimates[estimates['op'] == '~'].copy()
        if not path_data.empty:
            labels = [f"{row['rval']} → {row['lval']}" for _, row in path_data.iterrows()]
            coefs = path_data['Estimate'].tolist()
            colors = ["#4a9f6e" if c > 0 else "#e85747" for c in coefs]

            result["plots"].append({
                "title": "Path Coefficients",
                "data": [{
                    "type": "bar",
                    "x": coefs,
                    "y": labels,
                    "orientation": "h",
                    "marker": {"color": colors}
                }],
                "layout": {
                    "height": max(200, len(labels) * 40),
                    "xaxis": {"title": "Standardized Coefficient"},
                    "yaxis": {"automargin": True},
                    "margin": {"l": 150}
                }
            })

        # Plot 2: Residuals vs fitted (if possible)
        try:
            fitted = mod.predict(model_df)
            if outcome in fitted.columns:
                y_fitted = fitted[outcome].values
                y_actual = model_df[outcome].values
                residuals = y_actual - y_fitted

                result["plots"].append({
                    "title": f"Residuals: {outcome}",
                    "data": [{
                        "type": "scatter",
                        "x": y_fitted.tolist(),
                        "y": residuals.tolist(),
                        "mode": "markers",
                        "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6}
                    }, {
                        "type": "scatter",
                        "x": [min(y_fitted), max(y_fitted)],
                        "y": [0, 0],
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"}
                    }],
                    "layout": {
                        "height": 250,
                        "xaxis": {"title": "Fitted"},
                        "yaxis": {"title": "Residual"},
                        "showlegend": False
                    }
                })
        except Exception:
            pass

        fit_quality = "good" if (cfi and cfi > 0.95 and rmsea and rmsea < 0.05) else "acceptable" if (cfi and cfi > 0.90) else "poor"
        result["guide_observation"] = f"SEM {model_type} model with {fit_quality} fit. Check path coefficients for significant relationships."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, mod, {
                'model_type': f"SEM ({model_type.title()})",
                'features': predictors,
                'target': outcome,
                'metrics': {'cfi': float(cfi) if cfi else None, 'rmsea': float(rmsea) if rmsea else None}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "regularized_regression":
        """
        Ridge, LASSO, and Elastic Net regression with cross-validated alpha selection.
        Handles multicollinearity (Ridge), feature selection (LASSO), or both (Elastic Net).
        """
        from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import r2_score, mean_squared_error

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "elastic_net")  # ridge, lasso, elastic_net
        alpha = 1 - config.get("conf", 95) / 100

        data = df[predictors + [response]].dropna()
        X = data[predictors].values
        y = data[response].values
        n, p_vars = X.shape

        # Standardize for regularization
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Cross-validated alpha selection
        alphas = np.logspace(-4, 4, 100)

        if method == "ridge":
            cv_model = RidgeCV(alphas=alphas, cv=min(5, n))
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = Ridge(alpha=best_alpha).fit(X_scaled, y)
            method_name = "Ridge"
        elif method == "lasso":
            cv_model = LassoCV(alphas=alphas, cv=min(5, n), max_iter=10000)
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = Lasso(alpha=best_alpha, max_iter=10000).fit(X_scaled, y)
            method_name = "LASSO"
        else:  # elastic_net
            l1_ratio = float(config.get("l1_ratio", 0.5))
            cv_model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratio, cv=min(5, n), max_iter=10000)
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=10000).fit(X_scaled, y)
            method_name = f"Elastic Net (L1 ratio={l1_ratio})"

        y_pred = final_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        coefs = final_model.coef_
        intercept = final_model.intercept_

        # Cross-validation R²
        cv_scores = cross_val_score(final_model, X_scaled, y, cv=min(5, n), scoring='r2')

        # Count non-zero coefficients (for LASSO/elastic net)
        n_nonzero = np.sum(np.abs(coefs) > 1e-10)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>{method_name.upper()} REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Optimal α (CV):<</COLOR>> {best_alpha:.6f}\n\n"

        summary += f"<<COLOR:text>>Model Performance:<</COLOR>>\n"
        summary += f"  R²: {r2:.4f}\n"
        summary += f"  RMSE: {rmse:.4f}\n"
        summary += f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n"
        if method != "ridge":
            summary += f"  Non-zero coefficients: {n_nonzero}/{p_vars}\n"
        summary += f"  Intercept: {intercept:.4f}\n\n"

        summary += f"<<COLOR:text>>Standardized Coefficients:<</COLOR>>\n"
        summary += f"{'Predictor':<25} {'Coefficient':>12} {'|Coef|':>10} {'Status':>10}\n"
        summary += f"{'─' * 60}\n"

        # Sort by absolute coefficient
        coef_order = np.argsort(-np.abs(coefs))
        for idx in coef_order:
            status = "<<COLOR:good>>selected<</COLOR>>" if abs(coefs[idx]) > 1e-10 else "<<COLOR:text>>dropped<</COLOR>>"
            summary += f"{predictors[idx]:<25} {coefs[idx]:>12.6f} {abs(coefs[idx]):>10.6f} {status:>10}\n"

        if method != "ridge" and n_nonzero < p_vars:
            dropped = [predictors[i] for i in range(p_vars) if abs(coefs[i]) <= 1e-10]
            summary += f"\n<<COLOR:text>>Dropped features ({len(dropped)}): {', '.join(dropped)}<</COLOR>>\n"

        result["summary"] = summary

        # Coefficient bar plot
        sorted_idx = coef_order
        result["plots"].append({
            "title": f"{method_name} Coefficients (α={best_alpha:.4f})",
            "data": [{
                "type": "bar",
                "x": [predictors[i] for i in sorted_idx],
                "y": [float(coefs[i]) for i in sorted_idx],
                "marker": {"color": ["#4a9f6e" if abs(coefs[i]) > 1e-10 else "rgba(90,106,90,0.3)" for i in sorted_idx]}
            }],
            "layout": {"height": 300, "xaxis": {"tickangle": -45}, "yaxis": {"title": "Standardized Coefficient"}}
        })

        # Actual vs Predicted
        result["plots"].append({
            "title": "Actual vs Predicted",
            "data": [
                {"type": "scatter", "x": y.tolist(), "y": y_pred.tolist(), "mode": "markers",
                 "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}, "name": "Data"},
                {"type": "scatter", "x": [float(y.min()), float(y.max())], "y": [float(y.min()), float(y.max())],
                 "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": "Perfect Fit"}
            ],
            "layout": {"height": 300, "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"}}
        })

        # Residuals vs fitted
        residuals_rr = y - y_pred
        result["plots"].append({
            "title": "Residuals vs Fitted",
            "data": [{
                "type": "scatter", "x": y_pred.tolist(), "y": residuals_rr.tolist(),
                "mode": "markers", "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}
            }],
            "layout": {
                "height": 280, "xaxis": {"title": "Fitted"}, "yaxis": {"title": "Residual"},
                "shapes": [{"type": "line", "x0": float(y_pred.min()), "x1": float(y_pred.max()),
                            "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"{method_name}: R²={r2:.3f}, CV R²={cv_scores.mean():.3f}, α={best_alpha:.4f}. {n_nonzero}/{p_vars} features selected."
        result["statistics"] = {
            "r2": float(r2), "rmse": float(rmse),
            "cv_r2_mean": float(cv_scores.mean()), "cv_r2_std": float(cv_scores.std()),
            "best_alpha": float(best_alpha), "n_nonzero": int(n_nonzero),
            "coefficients": {predictors[i]: float(coefs[i]) for i in range(p_vars)},
            "intercept": float(intercept), "method": method
        }

    elif analysis_id == "discriminant_analysis":
        """
        Discriminant Analysis — LDA and QDA for classification and dimensionality reduction.
        LDA finds linear boundaries; QDA allows quadratic (class-specific covariances).
        Reports classification accuracy, prior probabilities, discriminant coefficients.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import LabelEncoder

        response = config.get("response") or config.get("target")
        predictors = config.get("predictors") or config.get("features", [])
        method = config.get("method", "lda")  # lda or qda

        if not response:
            result["summary"] = "Error: Please select a target (group) variable."
            return result
        if not predictors:
            predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != response]
        if not predictors:
            result["summary"] = "Error: No numeric predictor variables available."
            return result

        data = df[[response] + predictors].dropna()
        le = LabelEncoder()
        y = le.fit_transform(data[response])
        X = data[predictors].values.astype(float)
        classes = le.classes_

        # Split for evaluation
        from sklearn.model_selection import cross_val_score as cvs
        split_val = float(config.get("split", 20))
        test_frac = split_val if split_val < 1 else split_val / 100

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)

        if method == "qda":
            model = QuadraticDiscriminantAnalysis()
            model_name = "Quadratic Discriminant Analysis (QDA)"
        else:
            n_components = min(len(classes) - 1, len(predictors))
            model = LinearDiscriminantAnalysis(n_components=n_components)
            model_name = "Linear Discriminant Analysis (LDA)"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        cv_scores = cvs(model, X, y, cv=min(5, len(classes)), scoring='accuracy')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        result["plots"].append({
            "data": [{
                "z": cm.tolist(),
                "x": [str(c) for c in classes],
                "y": [str(c) for c in classes],
                "type": "heatmap",
                "colorscale": [[0, "#f0f4f0"], [1, "#2c5f2d"]],
                "text": [[str(v) for v in row] for row in cm],
                "texttemplate": "%{text}",
                "showscale": True
            }],
            "layout": {
                "title": f"{model_name} — Confusion Matrix",
                "xaxis": {"title": "Predicted"},
                "yaxis": {"title": "Actual", "autorange": "reversed"},
                "template": "plotly_white"
            }
        })

        # LDA: scatter plot in discriminant space
        if method != "qda" and hasattr(model, 'transform') and n_components >= 1:
            X_proj = model.transform(X)
            if X_proj.shape[1] >= 2:
                traces = []
                colors = ['#2c5f2d', '#4a90d9', '#d94a4a', '#d9a04a', '#7d4ad9', '#d94a99']
                for i, cls in enumerate(classes):
                    mask = y == i
                    traces.append({
                        "x": X_proj[mask, 0].tolist(),
                        "y": X_proj[mask, 1].tolist(),
                        "mode": "markers",
                        "name": str(cls),
                        "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.7},
                        "type": "scatter"
                    })
                result["plots"].append({
                    "data": traces,
                    "layout": {
                        "title": "LDA — Discriminant Space Projection",
                        "xaxis": {"title": "LD1"},
                        "yaxis": {"title": "LD2"},
                        "template": "plotly_white"
                    }
                })
            else:
                # 1D discriminant: histogram
                traces = []
                colors = ['#2c5f2d', '#4a90d9', '#d94a4a', '#d9a04a']
                for i, cls in enumerate(classes):
                    mask = y == i
                    traces.append({
                        "x": X_proj[mask, 0].tolist(),
                        "type": "histogram",
                        "name": str(cls),
                        "opacity": 0.6,
                        "marker": {"color": colors[i % len(colors)]}
                    })
                result["plots"].append({
                    "data": traces,
                    "layout": {
                        "title": "LDA — Discriminant Score Distribution",
                        "xaxis": {"title": "LD1 Score"},
                        "yaxis": {"title": "Count"},
                        "barmode": "overlay",
                        "template": "plotly_white"
                    }
                })

        # Coefficient importance (LDA only)
        coef_info = ""
        if method != "qda" and hasattr(model, 'coef_'):
            coef_magnitudes = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            sorted_idx = np.argsort(coef_magnitudes)[::-1]
            coef_rows = []
            for idx in sorted_idx:
                coef_val = model.coef_[0][idx] if model.coef_.ndim > 1 else model.coef_[idx]
                coef_rows.append(f"| {predictors[idx]} | {coef_val:.4f} |")
            coef_info = "\n\n**Discriminant Coefficients (LD1):**\n| Predictor | Coefficient |\n|---|---|\n" + "\n".join(coef_rows)

        # Classification report
        cr = classification_report(y_test, y_pred, target_names=[str(c) for c in classes])

        result["summary"] = f"**{model_name}**\n\nClasses: {', '.join(str(c) for c in classes)} ({len(classes)} groups)\nTraining accuracy: {train_acc:.3f}\nTest accuracy: {test_acc:.3f}\nCV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\nPrior probabilities: {', '.join(f'{c}: {p:.3f}' for c, p in zip(classes, model.priors_))}{coef_info}\n\n**Classification Report:**\n```\n{cr}\n```"
        result["guide_observation"] = f"{method.upper()}: test accuracy={test_acc:.3f}, CV accuracy={cv_scores.mean():.3f}. {len(classes)} classes, {len(predictors)} predictors."
        result["statistics"] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_classes": len(classes),
            "n_predictors": len(predictors),
            "priors": {str(c): float(p) for c, p in zip(classes, model.priors_)},
            "method": method
        }

    elif analysis_id == "factor_analysis":
        """
        Exploratory Factor Analysis — identifies latent factors underlying observed variables.
        Supports varimax, promax, and no rotation. Includes scree plot, loading heatmap,
        communalities table.
        """
        variables = config.get("variables", [])
        n_factors = config.get("n_factors", None)
        rotation = config.get("rotation", "varimax")

        try:
            from sklearn.decomposition import FactorAnalysis
            from scipy import stats as fstats

            if not variables:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()

            data = df[variables].dropna()
            N = len(data)
            p = len(variables)

            # Standardize
            X = data.values.astype(float)
            X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

            # Determine number of factors via eigenvalues > 1 (Kaiser criterion) if not specified
            corr_matrix = np.corrcoef(X_std.T)
            eigvals = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]

            if n_factors is None:
                n_factors = max(1, int(np.sum(eigvals > 1)))
            n_factors = min(n_factors, p)

            # Fit factor analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            scores = fa.fit_transform(X_std)
            loadings = fa.components_.T  # shape: (p, n_factors)

            # Apply rotation
            if rotation == "varimax" and n_factors > 1:
                # Varimax rotation
                rotated = loadings.copy()
                for _ in range(100):
                    old = rotated.copy()
                    for j in range(n_factors):
                        for k in range(j + 1, n_factors):
                            u = rotated[:, j]**2 - rotated[:, k]**2
                            v = 2 * rotated[:, j] * rotated[:, k]
                            A = np.sum(u)
                            B = np.sum(v)
                            C = np.sum(u**2 - v**2)
                            D = 2 * np.sum(u * v)
                            num = D - 2 * A * B / p
                            den = C - (A**2 - B**2) / p
                            angle = 0.25 * np.arctan2(num, den)
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            rotated[:, [j, k]] = rotated[:, [j, k]] @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                    if np.allclose(rotated, old, atol=1e-6):
                        break
                loadings = rotated

            # Communalities
            communalities = np.sum(loadings**2, axis=1)

            # Variance explained
            var_explained = np.sum(loadings**2, axis=0)
            pct_var = var_explained / p * 100

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>EXPLORATORY FACTOR ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables)}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n"
            summary_text += f"<<COLOR:highlight>>Factors extracted:<</COLOR>> {n_factors}\n"
            summary_text += f"<<COLOR:highlight>>Rotation:<</COLOR>> {rotation}\n\n"

            # Factor loadings table
            summary_text += f"<<COLOR:text>>Factor Loadings" + (f" ({rotation} rotated)" if rotation != "none" else "") + ":<</COLOR>>\n"
            header = f"{'Variable':<20}" + "".join([f"{'F' + str(i+1):>10}" for i in range(n_factors)]) + f"{'Communality':>12}\n"
            summary_text += header
            summary_text += f"{'─' * (20 + 10 * n_factors + 12)}\n"
            for vi, var_name in enumerate(variables):
                row = f"{var_name:<20}"
                for fi in range(n_factors):
                    val = loadings[vi, fi]
                    if abs(val) >= 0.4:
                        row += f"<<COLOR:good>>{val:>10.3f}<</COLOR>>"
                    else:
                        row += f"{val:>10.3f}"
                row += f"{communalities[vi]:>12.3f}\n"
                summary_text += row

            summary_text += f"\n<<COLOR:text>>Variance Explained:<</COLOR>>\n"
            for fi in range(n_factors):
                summary_text += f"  Factor {fi+1}: {var_explained[fi]:.3f} ({pct_var[fi]:.1f}%)\n"
            summary_text += f"  Total: {np.sum(var_explained):.3f} ({np.sum(pct_var):.1f}%)\n"

            result["summary"] = summary_text

            # Scree plot
            result["plots"].append({
                "title": "Scree Plot",
                "data": [
                    {"x": list(range(1, len(eigvals) + 1)), "y": eigvals.tolist(),
                     "mode": "lines+markers", "name": "Eigenvalues",
                     "marker": {"color": "#4a90d9", "size": 8}, "line": {"color": "#4a90d9", "width": 2}},
                    {"x": [1, len(eigvals)], "y": [1, 1],
                     "mode": "lines", "name": "Kaiser Criterion",
                     "line": {"color": "#d94a4a", "dash": "dash"}}
                ],
                "layout": {"height": 280, "xaxis": {"title": "Factor Number"}, "yaxis": {"title": "Eigenvalue"}, "template": "plotly_white"}
            })

            # Loading heatmap
            result["plots"].append({
                "title": f"Factor Loadings Heatmap ({rotation})",
                "data": [{
                    "type": "heatmap",
                    "z": loadings.tolist(),
                    "x": [f"Factor {i+1}" for i in range(n_factors)],
                    "y": variables,
                    "colorscale": "RdBu", "zmid": 0,
                    "text": [[f"{loadings[vi, fi]:.3f}" for fi in range(n_factors)] for vi in range(p)],
                    "texttemplate": "%{text}",
                    "showscale": True
                }],
                "layout": {"height": max(250, p * 25), "template": "plotly_white"}
            })

            result["statistics"] = {
                "n_factors": n_factors, "rotation": rotation, "n": N,
                "eigenvalues": eigvals.tolist(),
                "variance_explained": var_explained.tolist(),
                "pct_variance": pct_var.tolist(),
                "communalities": {v: float(communalities[i]) for i, v in enumerate(variables)},
                "total_variance_explained": float(np.sum(pct_var))
            }
            result["guide_observation"] = f"Factor analysis: {n_factors} factors extracted ({rotation}), explaining {np.sum(pct_var):.1f}% of variance."

        except Exception as e:
            result["summary"] = f"Factor analysis error: {str(e)}"

    # =====================================================================
    # Correspondence Analysis
    # =====================================================================
    elif analysis_id == "correspondence_analysis":
        """
        Correspondence Analysis — visualizes associations in a contingency table
        as a biplot. Decomposes chi-squared structure into orthogonal dimensions.
        Shows row and column profiles in shared low-dimensional space.
        """
        row_var_ca = config.get("row_var") or config.get("rows")
        col_var_ca = config.get("col_var") or config.get("columns")

        data_ca = df[[row_var_ca, col_var_ca]].dropna()

        try:
            ct_ca = pd.crosstab(data_ca[row_var_ca], data_ca[col_var_ca])
            if ct_ca.shape[0] < 2 or ct_ca.shape[1] < 2:
                result["summary"] = "Need at least 2 rows and 2 columns for correspondence analysis."
                return result

            # Total, row/col profiles
            N_ca = ct_ca.values.sum()
            P_ca = ct_ca.values / N_ca  # correspondence matrix
            r_ca = P_ca.sum(axis=1)  # row masses
            c_ca = P_ca.sum(axis=0)  # column masses

            # Standardized residuals
            Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r_ca))
            Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c_ca))
            S_ca = Dr_inv_sqrt @ (P_ca - np.outer(r_ca, c_ca)) @ Dc_inv_sqrt

            # SVD
            U_ca, sigma_ca, Vt_ca = np.linalg.svd(S_ca, full_matrices=False)

            # Number of dimensions (min of rows-1, cols-1)
            n_dims = min(ct_ca.shape[0] - 1, ct_ca.shape[1] - 1, 2)
            if n_dims < 1:
                result["summary"] = "Not enough dimensions for correspondence analysis."
                return result

            # Inertia (eigenvalues = sigma^2)
            inertia = sigma_ca ** 2
            total_inertia = float(np.sum(inertia))
            pct_inertia = inertia / total_inertia * 100 if total_inertia > 0 else inertia * 0

            # Row and column coordinates (principal coordinates)
            row_coords = Dr_inv_sqrt @ U_ca[:, :n_dims] * sigma_ca[:n_dims]
            col_coords = Dc_inv_sqrt @ Vt_ca[:n_dims, :].T * sigma_ca[:n_dims]

            row_labels_ca = [str(x) for x in ct_ca.index]
            col_labels_ca = [str(x) for x in ct_ca.columns]

            # Chi-squared test of independence
            from scipy import stats as ca_stats
            chi2_ca, p_chi2_ca, dof_ca, _ = ca_stats.chi2_contingency(ct_ca.values)

            # Summary
            summary_ca = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_ca += f"<<COLOR:title>>CORRESPONDENCE ANALYSIS<</COLOR>>\n"
            summary_ca += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_ca += f"<<COLOR:highlight>>Row variable:<</COLOR>> {row_var_ca} ({ct_ca.shape[0]} levels)\n"
            summary_ca += f"<<COLOR:highlight>>Column variable:<</COLOR>> {col_var_ca} ({ct_ca.shape[1]} levels)\n"
            summary_ca += f"<<COLOR:highlight>>Total N:<</COLOR>> {int(N_ca)}\n\n"

            summary_ca += f"<<COLOR:text>>Chi-squared test of independence:<</COLOR>>\n"
            summary_ca += f"  χ² = {chi2_ca:.2f},  df = {dof_ca},  p = {p_chi2_ca:.4f}"
            if p_chi2_ca < 0.05:
                summary_ca += f"  <<COLOR:good>>Significant association<</COLOR>>"
            summary_ca += "\n\n"

            summary_ca += f"<<COLOR:text>>Inertia (variance explained by dimensions):<</COLOR>>\n"
            for d in range(min(len(inertia), 5)):
                summary_ca += f"  Dim {d+1}: {inertia[d]:.4f} ({pct_inertia[d]:.1f}%)\n"
            summary_ca += f"  Total: {total_inertia:.4f}\n"

            if n_dims >= 2:
                summary_ca += f"\n<<COLOR:text>>First 2 dimensions explain {pct_inertia[0]+pct_inertia[1]:.1f}% of inertia.<</COLOR>>\n"

            # Row contributions
            summary_ca += f"\n<<COLOR:text>>Row Coordinates:<</COLOR>>\n"
            summary_ca += f"{'Level':<20}" + "".join([f"{'Dim '+str(d+1):>10}" for d in range(n_dims)]) + "\n"
            summary_ca += f"{'─' * (20 + 10 * n_dims)}\n"
            for ri, rl in enumerate(row_labels_ca):
                row_str = f"{rl:<20}"
                for d in range(n_dims):
                    row_str += f"{row_coords[ri, d]:>10.4f}"
                summary_ca += row_str + "\n"

            summary_ca += f"\n<<COLOR:text>>Column Coordinates:<</COLOR>>\n"
            summary_ca += f"{'Level':<20}" + "".join([f"{'Dim '+str(d+1):>10}" for d in range(n_dims)]) + "\n"
            summary_ca += f"{'─' * (20 + 10 * n_dims)}\n"
            for ci, cl in enumerate(col_labels_ca):
                col_str = f"{cl:<20}"
                for d in range(n_dims):
                    col_str += f"{col_coords[ci, d]:>10.4f}"
                summary_ca += col_str + "\n"

            result["summary"] = summary_ca

            # Biplot (if 2D)
            if n_dims >= 2:
                traces_ca = [
                    {"type": "scatter", "mode": "markers+text",
                     "x": row_coords[:, 0].tolist(), "y": row_coords[:, 1].tolist(),
                     "text": row_labels_ca, "textposition": "top center",
                     "name": f"{row_var_ca} (rows)",
                     "marker": {"color": "#4a9f6e", "size": 10, "symbol": "circle"}},
                    {"type": "scatter", "mode": "markers+text",
                     "x": col_coords[:, 0].tolist(), "y": col_coords[:, 1].tolist(),
                     "text": col_labels_ca, "textposition": "bottom center",
                     "name": f"{col_var_ca} (columns)",
                     "marker": {"color": "#4a90d9", "size": 10, "symbol": "diamond"}},
                ]
                result["plots"].append({
                    "title": f"Correspondence Analysis Biplot ({pct_inertia[0]:.1f}% + {pct_inertia[1]:.1f}% = {pct_inertia[0]+pct_inertia[1]:.1f}%)",
                    "data": traces_ca,
                    "layout": {
                        "height": 450,
                        "xaxis": {"title": f"Dimension 1 ({pct_inertia[0]:.1f}%)", "zeroline": True, "zerolinecolor": "#5a6a5a"},
                        "yaxis": {"title": f"Dimension 2 ({pct_inertia[1]:.1f}%)", "zeroline": True, "zerolinecolor": "#5a6a5a"},
                    }
                })

            # Scree plot of inertia
            result["plots"].append({
                "title": "Inertia Scree Plot",
                "data": [{
                    "x": list(range(1, len(inertia) + 1)),
                    "y": pct_inertia[:len(inertia)].tolist(),
                    "mode": "lines+markers", "name": "% Inertia",
                    "marker": {"color": "#4a9f6e", "size": 8},
                    "line": {"color": "#4a9f6e", "width": 2},
                }],
                "layout": {"height": 280, "xaxis": {"title": "Dimension"}, "yaxis": {"title": "% of Inertia"}}
            })

            result["guide_observation"] = f"Correspondence analysis: χ²={chi2_ca:.1f} (p={p_chi2_ca:.4f}), {n_dims} dimensions explain {sum(pct_inertia[:n_dims]):.1f}% of inertia."
            result["statistics"] = {
                "chi2": chi2_ca, "p_value": p_chi2_ca, "total_inertia": total_inertia,
                "n_dims": n_dims, "inertia": inertia[:n_dims].tolist(),
                "pct_inertia": pct_inertia[:n_dims].tolist(),
                "row_coords": {rl: row_coords[ri, :n_dims].tolist() for ri, rl in enumerate(row_labels_ca)},
                "col_coords": {cl: col_coords[ci, :n_dims].tolist() for ci, cl in enumerate(col_labels_ca)},
            }

        except Exception as e:
            result["summary"] = f"Correspondence analysis error: {str(e)}"

    # =====================================================================
    # Item Analysis (Cronbach's Alpha)
    # =====================================================================
    elif analysis_id == "item_analysis":
        """
        Item Analysis — reliability assessment for multi-item scales/questionnaires.
        Computes Cronbach's alpha (overall and if-item-deleted), item-total correlations,
        inter-item correlation matrix. Standard tool for survey/psychometric validation.
        """
        items_ia = config.get("items") or config.get("variables", [])

        if not items_ia:
            items_ia = df.select_dtypes(include=[np.number]).columns.tolist()

        data_ia = df[items_ia].dropna()
        n_ia = len(data_ia)
        k_ia = len(items_ia)

        if k_ia < 2:
            result["summary"] = "Need at least 2 items for reliability analysis."
            return result

        try:
            X_ia = data_ia.values.astype(float)

            # Cronbach's alpha
            item_vars = np.var(X_ia, axis=0, ddof=1)
            total_var = np.var(X_ia.sum(axis=1), ddof=1)
            alpha_overall = (k_ia / (k_ia - 1)) * (1 - np.sum(item_vars) / total_var) if total_var > 0 else 0

            # Item statistics
            total_scores = X_ia.sum(axis=1)
            item_stats = []
            for i, item_name in enumerate(items_ia):
                item_mean = float(np.mean(X_ia[:, i]))
                item_std = float(np.std(X_ia[:, i], ddof=1))
                # Corrected item-total correlation (correlation with total minus this item)
                rest_total = total_scores - X_ia[:, i]
                corr_it = float(np.corrcoef(X_ia[:, i], rest_total)[0, 1]) if item_std > 0 else 0

                # Alpha if item deleted
                if k_ia > 2:
                    remaining = np.delete(X_ia, i, axis=1)
                    rem_item_vars = np.var(remaining, axis=0, ddof=1)
                    rem_total_var = np.var(remaining.sum(axis=1), ddof=1)
                    k_rem = k_ia - 1
                    alpha_deleted = (k_rem / (k_rem - 1)) * (1 - np.sum(rem_item_vars) / rem_total_var) if rem_total_var > 0 else 0
                else:
                    alpha_deleted = 0

                item_stats.append({
                    "item": item_name, "mean": item_mean, "std": item_std,
                    "corrected_item_total": corr_it, "alpha_if_deleted": float(alpha_deleted),
                })

            # Inter-item correlation matrix
            corr_matrix_ia = np.corrcoef(X_ia.T)
            # Average inter-item correlation (off-diagonal)
            off_diag = corr_matrix_ia[np.triu_indices(k_ia, k=1)]
            avg_inter_item = float(np.mean(off_diag)) if len(off_diag) > 0 else 0

            # Standardized alpha (based on average inter-item correlation)
            std_alpha = (k_ia * avg_inter_item) / (1 + (k_ia - 1) * avg_inter_item) if (1 + (k_ia - 1) * avg_inter_item) > 0 else 0

            # Summary
            summary_ia = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_ia += f"<<COLOR:title>>ITEM ANALYSIS (RELIABILITY)<</COLOR>>\n"
            summary_ia += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_ia += f"<<COLOR:highlight>>Items:<</COLOR>> {k_ia}\n"
            summary_ia += f"<<COLOR:highlight>>N (complete cases):<</COLOR>> {n_ia}\n\n"

            summary_ia += f"<<COLOR:text>>Overall Reliability:<</COLOR>>\n"
            alpha_color = "good" if alpha_overall >= 0.7 else ("warning" if alpha_overall >= 0.5 else "accent")
            summary_ia += f"  <<COLOR:{alpha_color}>>Cronbach's α = {alpha_overall:.4f}<</COLOR>>\n"
            summary_ia += f"  Standardized α = {std_alpha:.4f}\n"
            summary_ia += f"  Average inter-item correlation = {avg_inter_item:.4f}\n\n"

            if alpha_overall >= 0.9:
                summary_ia += f"  <<COLOR:good>>Excellent reliability<</COLOR>>\n"
            elif alpha_overall >= 0.8:
                summary_ia += f"  <<COLOR:good>>Good reliability<</COLOR>>\n"
            elif alpha_overall >= 0.7:
                summary_ia += f"  <<COLOR:good>>Acceptable reliability<</COLOR>>\n"
            elif alpha_overall >= 0.6:
                summary_ia += f"  <<COLOR:warning>>Questionable reliability<</COLOR>>\n"
            elif alpha_overall >= 0.5:
                summary_ia += f"  <<COLOR:warning>>Poor reliability<</COLOR>>\n"
            else:
                summary_ia += f"  <<COLOR:accent>>Unacceptable reliability<</COLOR>>\n"

            summary_ia += f"\n<<COLOR:text>>Item Statistics:<</COLOR>>\n"
            summary_ia += f"{'Item':<25} {'Mean':>8} {'SD':>8} {'r(item-total)':>14} {'α if deleted':>12}\n"
            summary_ia += f"{'─' * 72}\n"
            for s in item_stats:
                flag = " <<COLOR:warning>>↑<</COLOR>>" if s["alpha_if_deleted"] > alpha_overall + 0.01 else ""
                summary_ia += f"{s['item']:<25} {s['mean']:>8.3f} {s['std']:>8.3f} {s['corrected_item_total']:>14.4f} {s['alpha_if_deleted']:>12.4f}{flag}\n"

            summary_ia += f"\n<<COLOR:text>>↑ = removing this item would improve α<</COLOR>>\n"

            result["summary"] = summary_ia

            # Item-total correlation bar chart
            result["plots"].append({
                "title": "Corrected Item-Total Correlations",
                "data": [{
                    "type": "bar",
                    "x": [s["item"] for s in item_stats],
                    "y": [s["corrected_item_total"] for s in item_stats],
                    "marker": {"color": ["#4a9f6e" if s["corrected_item_total"] >= 0.3 else "#d94a4a" for s in item_stats]},
                }],
                "layout": {"height": 300, "xaxis": {"tickangle": -45}, "yaxis": {"title": "Corrected Item-Total r"},
                           "shapes": [{"type": "line", "x0": -0.5, "x1": k_ia - 0.5,
                                       "y0": 0.3, "y1": 0.3, "line": {"color": "#e89547", "dash": "dash"}}]}
            })

            # Alpha-if-deleted plot
            result["plots"].append({
                "title": "Cronbach's α if Item Deleted",
                "data": [
                    {"type": "bar",
                     "x": [s["item"] for s in item_stats],
                     "y": [s["alpha_if_deleted"] for s in item_stats],
                     "marker": {"color": ["#d94a4a" if s["alpha_if_deleted"] > alpha_overall else "#4a9f6e" for s in item_stats]}},
                    {"type": "scatter", "mode": "lines", "name": f"Current α ({alpha_overall:.3f})",
                     "x": [items_ia[0], items_ia[-1]], "y": [alpha_overall, alpha_overall],
                     "line": {"color": "#e89547", "dash": "dash"}},
                ],
                "layout": {"height": 300, "xaxis": {"tickangle": -45}, "yaxis": {"title": "Cronbach's α"}}
            })

            # Inter-item correlation heatmap
            result["plots"].append({
                "title": "Inter-Item Correlation Matrix",
                "data": [{
                    "type": "heatmap",
                    "z": corr_matrix_ia.tolist(),
                    "x": items_ia, "y": items_ia,
                    "colorscale": "RdBu", "zmid": 0,
                    "text": [[f"{corr_matrix_ia[i, j]:.2f}" for j in range(k_ia)] for i in range(k_ia)],
                    "texttemplate": "%{text}", "showscale": True,
                }],
                "layout": {"height": max(300, k_ia * 25)}
            })

            n_weak = sum(1 for s in item_stats if s["corrected_item_total"] < 0.3)
            result["guide_observation"] = f"Item analysis: α={alpha_overall:.3f} ({k_ia} items). {n_weak} items with weak item-total correlation (<0.3)."
            result["statistics"] = {
                "cronbach_alpha": alpha_overall, "standardized_alpha": std_alpha,
                "avg_inter_item_correlation": avg_inter_item,
                "n_items": k_ia, "n_cases": n_ia,
                "item_stats": item_stats,
            }

        except Exception as e:
            result["summary"] = f"Item analysis error: {str(e)}"

    return result


