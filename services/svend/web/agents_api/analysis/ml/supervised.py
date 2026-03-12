"""Supervised ML — classification and regression (Random Forest).

CR: 3c0d0e53
"""

import logging
import uuid
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from ..common import (
    _effect_magnitude,
    _ml_interpretation,
    cache_model,
)

logger = logging.getLogger(__name__)


def _run_supervised(df, analysis_id, config, user):
    """Run supervised ML analysis (classification or regression_ml)."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])
    algorithm = config.get("algorithm", "rf")
    split_val = float(config.get("split", 20))
    split_val if split_val < 1 else split_val / 100

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
        summary += "<<COLOR:title>>CLASSIFICATION RESULTS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Algorithm:<</COLOR>> {algorithm.upper()}\n"
        summary += f"<<COLOR:highlight>>Accuracy:<</COLOR>> {accuracy:.4f}\n\n"
        summary += f"<<COLOR:text>>{report}<</COLOR>>"
        summary += _ml_interpretation("classification", {"accuracy": accuracy}, y_test, y_pred, features, target, model)
        result["summary"] = summary

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            result["plots"].append(
                {
                    "title": "Feature Importance",
                    "data": [
                        {
                            "type": "bar",
                            "x": importances[sorted_idx].tolist(),
                            "y": [features[i] for i in sorted_idx],
                            "orientation": "h",
                            "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        }
                    ],
                    "layout": {"height": max(200, len(features) * 25)},
                }
            )

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
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(list(set(y_test.tolist()) | set(y_pred.tolist())))
        str_labels = [str(lbl) for lbl in labels]
        result["plots"].append(
            {
                "title": "Confusion Matrix",
                "data": [
                    {
                        "type": "heatmap",
                        "z": cm.tolist(),
                        "x": str_labels,
                        "y": str_labels,
                        "colorscale": [[0, "#1a1a2e"], [1, "#4a9f6e"]],
                        "showscale": True,
                        "text": cm.tolist(),
                        "texttemplate": "%{text}",
                        "hovertemplate": "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Predicted"},
                    "yaxis": {"title": "Actual", "autorange": "reversed"},
                },
            }
        )

        # ROC curve (binary or one-vs-rest for multiclass)
        try:
            classes = sorted(list(set(y_test)))
            if len(classes) == 2:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=classes[1])
                    roc_auc = auc(fpr, tpr)
                    result["plots"].append(
                        {
                            "title": f"ROC Curve (AUC = {roc_auc:.3f})",
                            "data": [
                                {
                                    "type": "scatter",
                                    "x": fpr.tolist(),
                                    "y": tpr.tolist(),
                                    "mode": "lines",
                                    "line": {"color": "#4a9f6e", "width": 2},
                                    "name": f"ROC (AUC={roc_auc:.3f})",
                                },
                                {
                                    "type": "scatter",
                                    "x": [0, 1],
                                    "y": [0, 1],
                                    "mode": "lines",
                                    "line": {"color": "#d94a4a", "dash": "dash"},
                                    "name": "Random",
                                },
                            ],
                            "layout": {
                                "height": 300,
                                "xaxis": {"title": "False Positive Rate"},
                                "yaxis": {"title": "True Positive Rate"},
                            },
                        }
                    )
        except Exception:
            pass

        # Conformal prediction sets
        conformal_state = None
        try:
            from agents_api.conformal import compute_conformal

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

            if hasattr(model, "predict_proba"):
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

                result["summary"] += "\n\n<<COLOR:accent>>── Conformal Prediction Sets (90% nominal) ──<</COLOR>>\n"
                result["summary"] += f"<<COLOR:highlight>>Average set size:<</COLOR>> {avg_set_size:.2f} classes\n"
                result["summary"] += f"<<COLOR:highlight>>Empirical test coverage:<</COLOR>> {emp_coverage:.1%}\n"
                result["summary"] += f"<<COLOR:highlight>>Single-class predictions:<</COLOR>> {single_class_pct:.1%}\n"
                result["summary"] += f"<<COLOR:highlight>>Calibration set:<</COLOR>> {cf.n_cal} observations\n"
                result["summary"] += (
                    "<<COLOR:text>>Nominal coverage: 90% (finite-sample marginal guarantee under exchangeability)<</COLOR>>\n"
                )
                result["summary"] += (
                    "<<COLOR:text>>When the model is confident, you get 1 class. When uncertain, you get 2+.<</COLOR>>"
                )

                # Prediction set size histogram
                result["plots"].append(
                    {
                        "title": f"Conformal Prediction Set Size (coverage: {emp_coverage:.1%})",
                        "data": [
                            {
                                "type": "histogram",
                                "x": set_sizes,
                                "marker": {
                                    "color": "rgba(74, 159, 110, 0.6)",
                                    "line": {"color": "#4a9f6e", "width": 1},
                                },
                                "name": "Set Size",
                            }
                        ],
                        "layout": {
                            "height": 250,
                            "xaxis": {"title": "Number of Classes in Set", "dtick": 1},
                            "yaxis": {"title": "Count"},
                        },
                    }
                )

                result["statistics"] = result.get("statistics", {})
                result["statistics"]["conformal"] = {
                    "alpha": 0.10,
                    "nominal_coverage": 0.90,
                    "empirical_coverage": round(float(emp_coverage), 4),
                    "avg_set_size": round(avg_set_size, 4),
                    "single_class_pct": round(single_class_pct, 4),
                    "n_calibration": cf.n_cal,
                    "split_seed": 42,
                }
        except Exception as e:
            logger.warning(f"Conformal prediction failed for classification: {e}")

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_meta = {
                "model_type": f"Classification ({algorithm.upper()})",
                "features": features,
                "target": target,
                "metrics": {"accuracy": float(accuracy)},
            }
            if conformal_state:
                cache_meta["conformal_state"] = conformal_state
                cache_meta["split_seed"] = 42
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
        summary += "<<COLOR:title>>REGRESSION RESULTS (RANDOM FOREST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n"
        summary += f"<<COLOR:highlight>>RMSE:<</COLOR>> {rmse:.4f}\n"
        summary += _ml_interpretation("regression", {"r2": r2, "rmse": rmse}, y_test, y_pred, features, target)
        result["summary"] = summary

        r2_label, _ = _effect_magnitude(r2, "r_squared")
        y_range_val = float(np.ptp(y_test)) if len(y_test) > 0 else 1
        rmse_pct = rmse / y_range_val * 100 if y_range_val > 0 else 0
        result["guide_observation"] = (
            f"Random Forest Regression: R²={r2:.3f} ({r2_label}), RMSE={rmse:.2f} ({rmse_pct:.0f}% of range). "
            + (
                "Strong predictive model."
                if r2 >= 0.7
                else "Moderate model — useful for trends."
                if r2 >= 0.4
                else "Weak model — features may not predict target well."
            )
        )

        # Actual vs Predicted
        result["plots"].append(
            {
                "title": "Actual vs Predicted",
                "data": [
                    {
                        "type": "scatter",
                        "x": y_test.tolist(),
                        "y": y_pred.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#6c5ce7", "size": 5},
                    },
                    {
                        "type": "scatter",
                        "x": [y_test.min(), y_test.max()],
                        "y": [y_test.min(), y_test.max()],
                        "mode": "lines",
                        "line": {"color": "#ff7675", "dash": "dash"},
                    },
                ],
                "layout": {"height": 300, "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"}},
            }
        )

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            result["plots"].append(
                {
                    "title": "Feature Importance",
                    "data": [
                        {
                            "type": "bar",
                            "x": importances[sorted_idx].tolist(),
                            "y": [features[i] for i in sorted_idx],
                            "orientation": "h",
                            "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        }
                    ],
                    "layout": {"height": max(200, len(features) * 25)},
                }
            )

        # Residual plot
        residuals = (y_test.values - y_pred).tolist()
        result["plots"].append(
            {
                "title": "Residuals vs Predicted",
                "data": [
                    {
                        "type": "scatter",
                        "x": y_pred.tolist(),
                        "y": residuals,
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 144, 217, 0.5)",
                            "size": 5,
                            "line": {"color": "#4a90d9", "width": 1},
                        },
                        "name": "Residuals",
                    },
                    {
                        "type": "scatter",
                        "x": [float(y_pred.min()), float(y_pred.max())],
                        "y": [0, 0],
                        "mode": "lines",
                        "line": {"color": "#d94a4a", "dash": "dash"},
                        "name": "Zero",
                    },
                ],
                "layout": {"height": 300, "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Residual"}},
            }
        )

        # Conformal prediction intervals
        conformal_state = None
        try:
            from agents_api.conformal import compute_conformal

            cf = compute_conformal(model, X_cal, y_cal, task_type="regression")
            conformal_state = cf.get_state()
            qhat_90 = cf.qhats.get("0.1", 0)
            qhat_95 = cf.qhats.get("0.05", 0)

            # Empirical coverage on test set
            y_lo, y_hi = cf.predict_interval(y_pred, alpha=0.10)
            covered = np.sum((y_test.values >= y_lo) & (y_test.values <= y_hi))
            emp_coverage = covered / len(y_test) if len(y_test) > 0 else 0

            result["summary"] += "\n\n<<COLOR:accent>>── Conformal Prediction Intervals (90% nominal) ──<</COLOR>>\n"
            result["summary"] += f"<<COLOR:highlight>>Interval half-width:<</COLOR>> ±{qhat_90:.4f}\n"
            result["summary"] += f"<<COLOR:highlight>>Empirical test coverage:<</COLOR>> {emp_coverage:.1%}\n"
            result["summary"] += f"<<COLOR:highlight>>Calibration set:<</COLOR>> {cf.n_cal} observations\n"
            result["summary"] += f"<<COLOR:highlight>>95% interval half-width:<</COLOR>> ±{qhat_95:.4f}\n"
            result["summary"] += (
                "<<COLOR:text>>Nominal coverage: 90% (finite-sample marginal guarantee under exchangeability)<</COLOR>>"
            )

            # Conformal interval plot
            sort_idx = np.argsort(y_pred)
            y_pred_s = y_pred[sort_idx]
            y_test_s = y_test.values[sort_idx]
            y_lo_s = y_lo[sort_idx]
            y_hi_s = y_hi[sort_idx]
            inside = (y_test_s >= y_lo_s) & (y_test_s <= y_hi_s)
            result["plots"].append(
                {
                    "title": f"Conformal Prediction Intervals (coverage: {emp_coverage:.1%})",
                    "data": [
                        {
                            "type": "scatter",
                            "x": list(range(len(y_pred_s))),
                            "y": y_hi_s.tolist(),
                            "mode": "lines",
                            "line": {"width": 0},
                            "showlegend": False,
                        },
                        {
                            "type": "scatter",
                            "x": list(range(len(y_pred_s))),
                            "y": y_lo_s.tolist(),
                            "mode": "lines",
                            "fill": "tonexty",
                            "fillcolor": "rgba(74, 159, 110, 0.15)",
                            "line": {"width": 0},
                            "name": "90% interval",
                        },
                        {
                            "type": "scatter",
                            "x": list(range(len(y_pred_s))),
                            "y": y_pred_s.tolist(),
                            "mode": "lines",
                            "line": {"color": "#4a9f6e", "width": 1.5},
                            "name": "Predicted",
                        },
                        {
                            "type": "scatter",
                            "x": [i for i, v in enumerate(inside) if v],
                            "y": [y_test_s[i] for i, v in enumerate(inside) if v],
                            "mode": "markers",
                            "marker": {"color": "#2ecc71", "size": 5},
                            "name": "Inside",
                        },
                        {
                            "type": "scatter",
                            "x": [i for i, v in enumerate(inside) if not v],
                            "y": [y_test_s[i] for i, v in enumerate(inside) if not v],
                            "mode": "markers",
                            "marker": {"color": "#e74c3c", "size": 7, "symbol": "x"},
                            "name": "Outside",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Observation (sorted by prediction)"},
                        "yaxis": {"title": target},
                    },
                }
            )

            result["statistics"] = result.get("statistics", {})
            result["statistics"]["conformal"] = {
                "alpha": 0.10,
                "nominal_coverage": 0.90,
                "empirical_coverage": round(float(emp_coverage), 4),
                "qhat_90": round(qhat_90, 4),
                "qhat_95": round(qhat_95, 4),
                "median_width": round(2 * qhat_90, 4),
                "n_calibration": cf.n_cal,
                "split_seed": 42,
            }
        except Exception as e:
            logger.warning(f"Conformal prediction failed for regression: {e}")

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_meta = {
                "model_type": "Random Forest Regressor",
                "features": features,
                "target": target,
                "metrics": {"r2": float(r2), "rmse": float(rmse)},
            }
            if conformal_state:
                cache_meta["conformal_state"] = conformal_state
                cache_meta["split_seed"] = 42
            cache_model(user.id, model_key, model, cache_meta)
            result["model_key"] = model_key
            result["can_save"] = True

    return result
