"""DSW common utilities — shared across all DSW sub-modules.

Extracted from dsw_views.py: model cache, logging, ML helpers,
Claude schema/interpret integration, diagnostic plots, and data splitting.
"""

import json
import logging
import math
import time
import uuid
import tempfile
from pathlib import Path
from collections import OrderedDict
from threading import Lock

from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django.conf import settings
from ..models import DSWResult, AgentLog, SavedModel
from accounts.permissions import gated, gated_paid, require_auth, require_enterprise

logger = logging.getLogger(__name__)

# Temporary cache for trained models (max 100 per user, expires after 1 hour)
# Structure: {user_id: {model_key: {'model': model, 'metadata': {...}, 'timestamp': time}}}
_model_cache = {}
_model_cache_lock = Lock()
MODEL_CACHE_MAX_SIZE = 100
MODEL_CACHE_EXPIRY = 3600  # 1 hour


def cache_model(user_id, model_key, model, metadata):
    """Cache a trained model temporarily for later saving."""
    with _model_cache_lock:
        if user_id not in _model_cache:
            _model_cache[user_id] = OrderedDict()

        # Clean expired entries
        now = time.time()
        expired = [k for k, v in _model_cache[user_id].items() if now - v['timestamp'] > MODEL_CACHE_EXPIRY]
        for k in expired:
            del _model_cache[user_id][k]

        # Limit cache size per user
        while len(_model_cache[user_id]) >= MODEL_CACHE_MAX_SIZE:
            _model_cache[user_id].popitem(last=False)

        _model_cache[user_id][model_key] = {
            'model': model,
            'metadata': metadata,
            'timestamp': now
        }


def get_cached_model(user_id, model_key):
    """Retrieve a cached model."""
    with _model_cache_lock:
        if user_id in _model_cache and model_key in _model_cache[user_id]:
            entry = _model_cache[user_id][model_key]
            if time.time() - entry['timestamp'] < MODEL_CACHE_EXPIRY:
                return entry
            else:
                del _model_cache[user_id][model_key]
    return None


def log_agent_action(user, agent, action, latency_ms=None, success=True, error_message="", metadata=None):
    """Log an agent action to the database."""
    try:
        AgentLog.objects.create(
            user=user if user.is_authenticated else None,
            agent=agent,
            action=action,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=json.dumps(metadata) if metadata else "",
        )
    except Exception as e:
        logger.warning(f"Failed to log agent action: {e}")


def _preload_llm_background():
    """Start loading the LLM in a background thread if not already loaded."""
    import threading
    try:
        # Quick CUDA check before attempting load
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available - skipping LLM preload")
                return
        except ImportError:
            logger.warning("PyTorch not available - skipping LLM preload")
            return

        from .. import views as agent_views
        if not agent_views._shared_llm_loaded:
            logger.info("Preloading LLM in background (data loaded)")
            def load():
                try:
                    agent_views.get_shared_llm()
                    logger.info("LLM preload completed")
                except Exception as e:
                    logger.error(f"LLM preload failed: {e}")
            threading.Thread(target=load, daemon=True).start()
    except Exception as e:
        logger.warning(f"Could not trigger LLM preload: {e}")


def save_model_to_disk(user, model, model_type, dsw_result_id, name=None,
                       metrics=None, features=None, target=None,
                       project_id=None, training_config=None,
                       data_lineage=None, parent_model_id=None):
    """Save a trained model to disk and create database record.

    Returns SavedModel instance or None if failed.
    """
    import pickle

    try:
        # Create user's model directory
        models_dir = Path(settings.MEDIA_ROOT) / "models" / str(user.id)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        model_id = uuid.uuid4()
        model_filename = f"{model_id}.pkl"
        model_path = models_dir / model_filename

        # Save model with pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Resolve optional project link
        project = None
        if project_id:
            try:
                from core.models import Project
                project = Project.objects.get(id=project_id)
            except Exception:
                logger.warning(f"Could not link model to project {project_id}")

        # Resolve parent model for versioning
        version = 1
        parent_model = None
        if parent_model_id:
            try:
                parent_model = SavedModel.objects.get(id=parent_model_id, user=user)
                version = parent_model.version + 1
            except SavedModel.DoesNotExist:
                pass

        # Create database record
        saved_model = SavedModel.objects.create(
            id=model_id,
            user=user,
            name=name or f"{model_type} - {dsw_result_id[:8]}",
            model_type=model_type,
            model_path=str(model_path),
            dsw_result_id=dsw_result_id,
            metrics=json.dumps(metrics) if metrics else "",
            feature_names=json.dumps(features) if features else "",
            target_name=target or "",
            project=project,
            training_config=training_config or {},
            data_lineage=data_lineage or {},
            version=version,
            parent_model=parent_model,
        )

        logger.info(f"Saved model {model_id} v{version} for user {user.username}")
        return saved_model

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


# ─── Synara Evidence Bridge ───────────────────────────────────────────────────
# Auto-creates core.Evidence when ML models are trained with a project link.


def _create_ml_evidence(user, project_id, model_type, metrics, importances, task, target):
    """Create core.Evidence from ML model results, linked to a project.

    Called automatically when a model is trained with a project_id.
    Returns the Evidence instance or None on failure.
    """
    try:
        from core.models import Project
        from core.models.hypothesis import Evidence as CoreEvidence

        project = Project.objects.get(id=project_id)

        top_features = [f["feature"] for f in (importances or [])[:3]]
        primary_metric = metrics.get("accuracy") or metrics.get("r2")
        metric_name = "Accuracy" if task == "classification" else "R²"

        metric_str = f"{primary_metric:.4f}" if isinstance(primary_metric, (int, float)) else str(primary_metric)
        features_str = ", ".join(top_features) if top_features else "N/A"

        summary = (
            f"ML Model ({model_type}) trained to predict '{target}'. "
            f"{metric_name}: {metric_str}. "
            f"Top predictive factors: {features_str}."
        )

        confidence = 0.5
        if isinstance(primary_metric, (int, float)):
            confidence = min(max(float(primary_metric), 0.1), 0.95)

        evidence = CoreEvidence.objects.create(
            project=project,
            summary=summary,
            source_type="analysis",
            source_description=f"DSW ML Pipeline ({model_type})",
            result_type="statistical",
            confidence=confidence,
            created_by=user,
            raw_output={
                "model_type": model_type,
                "metrics": metrics,
                "top_features": top_features,
                "task": task,
                "target": target,
            },
        )

        logger.info(f"Created ML evidence {evidence.id} for project {project_id}")
        return evidence

    except Exception as e:
        logger.warning(f"Could not create ML evidence for project {project_id}: {e}")
        return None


# ─── ML Diagnostic Engine ──────────────────────────────────────────────────────
# Shared by From Intent, From Data, XGBoost, LightGBM, Model Compare.
# Generates a comprehensive suite of Plotly charts for any trained model.

def _build_ml_diagnostics(model, X_test, y_test, y_pred, features, task,
                          label_map=None, model_name="Model"):
    """Build comprehensive diagnostic plots for a trained ML model.

    Returns list of Plotly plot dicts (same format as DSW analysis plots).

    Classification (6 plots): confusion matrix, ROC, precision-recall,
    feature importance, probability distribution, calibration curve.

    Regression (6 plots): actual-vs-predicted, residuals, residual histogram,
    Q-Q plot, feature importance, scale-location.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    plots = []

    if task == "classification":
        plots.extend(_diag_classification(model, X_test, y_test, y_pred, features, label_map, model_name))
    else:
        plots.extend(_diag_regression(model, X_test, y_test, y_pred, features, model_name))

    return plots


def _diag_classification(model, X_test, y_test, y_pred, features, label_map, model_name):
    """Classification diagnostic suite."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

    plots = []
    classes = sorted(list(set(y_test.tolist()) | set(y_pred.tolist())))
    str_labels = [str(label_map.get(c, c) if label_map else c) for c in classes]

    # 1. Confusion Matrix (annotated with counts + percentages)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100)
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append({
                "x": str_labels[j], "y": str_labels[i],
                "text": f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)",
                "showarrow": False, "font": {"color": "#fff" if cm[i][j] > cm.max() * 0.5 else "#aaa", "size": 12},
            })
    plots.append({
        "title": "Confusion Matrix",
        "data": [{
            "type": "heatmap", "z": cm.tolist(), "x": str_labels, "y": str_labels,
            "colorscale": [[0, "#0d120d"], [0.5, "#2a6b3a"], [1, "#4a9f6e"]],
            "showscale": True, "hoverongaps": False,
            "hovertemplate": "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        }],
        "layout": {
            "template": "plotly_dark", "height": 320,
            "xaxis": {"title": "Predicted", "side": "bottom"},
            "yaxis": {"title": "Actual", "autorange": "reversed"},
            "annotations": annotations,
        },
    })

    # 2. ROC Curve
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            if len(classes) == 2:
                # Binary ROC
                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=classes[1])
                roc_auc = auc(fpr, tpr)
                # Find optimal threshold (Youden's J)
                j_scores = tpr - fpr
                opt_idx = np.argmax(j_scores)
                plots.append({
                    "title": f"ROC Curve (AUC = {roc_auc:.3f})",
                    "data": [
                        {"type": "scatter", "x": fpr.tolist(), "y": tpr.tolist(), "mode": "lines",
                         "line": {"color": "#4a9f6e", "width": 2.5}, "name": f"ROC (AUC={roc_auc:.3f})",
                         "fill": "tozeroy", "fillcolor": "rgba(74,159,110,0.1)"},
                        {"type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "lines",
                         "line": {"dash": "dash", "color": "#555", "width": 1}, "name": "Random", "showlegend": True},
                        {"type": "scatter", "x": [float(fpr[opt_idx])], "y": [float(tpr[opt_idx])],
                         "mode": "markers", "marker": {"color": "#e89547", "size": 10, "symbol": "star"},
                         "name": f"Optimal (J={j_scores[opt_idx]:.2f})", "showlegend": True},
                    ],
                    "layout": {
                        "template": "plotly_dark", "height": 320,
                        "xaxis": {"title": "False Positive Rate", "range": [0, 1]},
                        "yaxis": {"title": "True Positive Rate", "range": [0, 1.05]},
                    },
                })
            else:
                # Multiclass one-vs-rest
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=classes)
                roc_traces = []
                colors = ["#4a9f6e", "#e89547", "#6a7fff", "#e8c547", "#ff7eb9", "#4a9faf", "#d06060", "#8a7fbf"]
                for i, cls in enumerate(classes):
                    if y_bin.shape[1] > i:
                        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                        auc_i = auc(fpr_i, tpr_i)
                        lbl = str(label_map.get(cls, cls) if label_map else cls)
                        roc_traces.append({
                            "type": "scatter", "x": fpr_i.tolist(), "y": tpr_i.tolist(),
                            "mode": "lines", "line": {"color": colors[i % len(colors)], "width": 2},
                            "name": f"{lbl} (AUC={auc_i:.3f})",
                        })
                roc_traces.append({
                    "type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "lines",
                    "line": {"dash": "dash", "color": "#555"}, "name": "Random",
                })
                plots.append({
                    "title": "ROC Curves (One-vs-Rest)",
                    "data": roc_traces,
                    "layout": {"template": "plotly_dark", "height": 350, "xaxis": {"title": "FPR"}, "yaxis": {"title": "TPR"}},
                })
        except Exception:
            pass

    # 3. Precision-Recall Curve
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            if len(classes) == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label=classes[1])
                ap = average_precision_score(y_test, y_prob[:, 1], pos_label=classes[1])
                plots.append({
                    "title": f"Precision-Recall Curve (AP = {ap:.3f})",
                    "data": [
                        {"type": "scatter", "x": recall.tolist(), "y": precision.tolist(), "mode": "lines",
                         "line": {"color": "#4a9faf", "width": 2.5}, "name": f"PR (AP={ap:.3f})",
                         "fill": "tozeroy", "fillcolor": "rgba(74,159,175,0.1)"},
                    ],
                    "layout": {
                        "template": "plotly_dark", "height": 320,
                        "xaxis": {"title": "Recall", "range": [0, 1]},
                        "yaxis": {"title": "Precision", "range": [0, 1.05]},
                    },
                })
            else:
                # Multiclass: per-class PR curves
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=classes)
                pr_traces = []
                colors = ["#4a9faf", "#e89547", "#6a7fff", "#e8c547", "#ff7eb9", "#4a9f6e"]
                for i, cls in enumerate(classes):
                    if y_bin.shape[1] > i:
                        p_i, r_i, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                        ap_i = average_precision_score(y_bin[:, i], y_prob[:, i])
                        lbl = str(label_map.get(cls, cls) if label_map else cls)
                        pr_traces.append({
                            "type": "scatter", "x": r_i.tolist(), "y": p_i.tolist(),
                            "mode": "lines", "line": {"color": colors[i % len(colors)], "width": 2},
                            "name": f"{lbl} (AP={ap_i:.3f})",
                        })
                plots.append({
                    "title": "Precision-Recall Curves (Per Class)",
                    "data": pr_traces,
                    "layout": {"template": "plotly_dark", "height": 350, "xaxis": {"title": "Recall"}, "yaxis": {"title": "Precision"}},
                })
        except Exception:
            pass

    # 4. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plots.append({
            "title": "Feature Importance",
            "data": [{
                "type": "bar", "orientation": "h",
                "x": importances[sorted_idx].tolist(),
                "y": [features[i] if i < len(features) else f"feat_{i}" for i in sorted_idx],
                "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
            }],
            "layout": {"template": "plotly_dark", "height": max(220, len(features) * 22)},
        })

    # 5. Predicted Probability Distribution
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            prob_traces = []
            colors = ["#4a9f6e", "#d06060", "#4a9faf", "#e8c547", "#8a7fbf", "#e89547"]
            for i, cls in enumerate(classes):
                mask = (np.array(y_test) == cls)
                if mask.any() and y_prob.shape[1] > 1:
                    col_idx = min(1, y_prob.shape[1] - 1) if len(classes) == 2 else i
                    lbl = str(label_map.get(cls, cls) if label_map else cls)
                    prob_traces.append({
                        "type": "histogram", "x": y_prob[mask, col_idx].tolist(),
                        "name": f"Actual: {lbl}", "opacity": 0.6,
                        "marker": {"color": colors[i % len(colors)]},
                    })
            if prob_traces:
                plots.append({
                    "title": "Predicted Probability Distribution",
                    "data": prob_traces,
                    "layout": {
                        "template": "plotly_dark", "height": 280, "barmode": "overlay",
                        "xaxis": {"title": "Predicted Probability"}, "yaxis": {"title": "Count"},
                    },
                })
        except Exception:
            pass

    # 6. Calibration Curve (reliability diagram)
    if hasattr(model, 'predict_proba') and len(classes) == 2:
        try:
            from sklearn.calibration import calibration_curve
            y_prob = model.predict_proba(X_test)[:, 1]
            y_binary = (np.array(y_test) == classes[1]).astype(int)
            n_bins = min(10, max(3, len(y_test) // 20))
            fraction_pos, mean_pred = calibration_curve(y_binary, y_prob, n_bins=n_bins)
            plots.append({
                "title": "Calibration Curve",
                "data": [
                    {"type": "scatter", "x": mean_pred.tolist(), "y": fraction_pos.tolist(),
                     "mode": "lines+markers", "line": {"color": "#4a9f6e", "width": 2},
                     "marker": {"size": 8, "color": "#4a9f6e"}, "name": model_name},
                    {"type": "scatter", "x": [0, 1], "y": [0, 1], "mode": "lines",
                     "line": {"dash": "dash", "color": "#555"}, "name": "Perfectly Calibrated"},
                ],
                "layout": {
                    "template": "plotly_dark", "height": 300,
                    "xaxis": {"title": "Mean Predicted Probability", "range": [0, 1]},
                    "yaxis": {"title": "Fraction of Positives", "range": [0, 1]},
                },
            })
        except Exception:
            pass

    return plots


def _diag_regression(model, X_test, y_test, y_pred, features, model_name):
    """Regression diagnostic suite."""
    import numpy as np
    from scipy import stats

    plots = []
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    residuals = y_test_arr - y_pred_arr

    # 1. Actual vs Predicted (with R² annotation and prediction band)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_arr, y_pred_arr)
    y_min, y_max = float(y_test_arr.min()), float(y_test_arr.max())
    plots.append({
        "title": f"Actual vs Predicted (R² = {r2:.4f})",
        "data": [
            {"type": "scatter", "x": y_test_arr.tolist(), "y": y_pred_arr.tolist(),
             "mode": "markers", "marker": {"color": "rgba(74,159,110,0.5)", "size": 5,
             "line": {"color": "#4a9f6e", "width": 0.5}}, "name": "Predictions"},
            {"type": "scatter", "x": [y_min, y_max], "y": [y_min, y_max],
             "mode": "lines", "line": {"color": "#d06060", "dash": "dash", "width": 2}, "name": "Perfect Fit"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 320,
            "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"},
            "annotations": [{"x": 0.05, "y": 0.95, "xref": "paper", "yref": "paper",
                             "text": f"R² = {r2:.4f}", "showarrow": False,
                             "font": {"size": 14, "color": "#4a9f6e"}, "bgcolor": "rgba(0,0,0,0.5)"}],
        },
    })

    # 2. Residuals vs Predicted (with lowess trend)
    # Color by magnitude
    abs_res = np.abs(residuals)
    res_norm = (abs_res - abs_res.min()) / (abs_res.max() - abs_res.min() + 1e-10)
    plots.append({
        "title": "Residuals vs Predicted",
        "data": [
            {"type": "scatter", "x": y_pred_arr.tolist(), "y": residuals.tolist(),
             "mode": "markers", "marker": {"color": res_norm.tolist(), "colorscale": [[0, "#4a9f6e"], [1, "#d06060"]],
             "size": 5, "showscale": True, "colorbar": {"title": "|Resid|", "thickness": 10, "len": 0.6}},
             "name": "Residuals",
             "hovertemplate": "Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"},
            {"type": "scatter", "x": [float(y_pred_arr.min()), float(y_pred_arr.max())], "y": [0, 0],
             "mode": "lines", "line": {"color": "#d06060", "dash": "dash", "width": 1.5}, "name": "Zero"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 320,
            "xaxis": {"title": "Predicted"}, "yaxis": {"title": "Residual"},
        },
    })

    # 3. Residual Distribution (histogram + fitted normal curve)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, res_mean, res_std)
    # Scale normal curve to match histogram
    bin_width = (residuals.max() - residuals.min()) / 30
    y_norm_scaled = y_norm * len(residuals) * bin_width

    # Shapiro-Wilk test for normality (on sample if large)
    n_sw = min(len(residuals), 5000)
    sw_stat, sw_p = stats.shapiro(residuals[:n_sw])

    plots.append({
        "title": f"Residual Distribution (Shapiro-Wilk p = {sw_p:.4f})",
        "data": [
            {"type": "histogram", "x": residuals.tolist(), "nbinsx": 30,
             "marker": {"color": "rgba(74,159,110,0.5)", "line": {"color": "#4a9f6e", "width": 1}},
             "name": "Residuals"},
            {"type": "scatter", "x": x_norm.tolist(), "y": y_norm_scaled.tolist(),
             "mode": "lines", "line": {"color": "#e89547", "width": 2.5}, "name": "Normal Fit"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300, "barmode": "overlay",
            "xaxis": {"title": "Residual"}, "yaxis": {"title": "Frequency"},
        },
    })

    # 4. Q-Q Plot (Normal Probability Plot)
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical_q = stats.norm.ppf(np.linspace(0.5/n, 1 - 0.5/n, n))
    # Reference line
    slope, intercept, _, _, _ = stats.linregress(theoretical_q, sorted_res)

    plots.append({
        "title": "Normal Q-Q Plot",
        "data": [
            {"type": "scatter", "x": theoretical_q.tolist(), "y": sorted_res.tolist(),
             "mode": "markers", "marker": {"color": "rgba(74,159,175,0.6)", "size": 4}, "name": "Residuals"},
            {"type": "scatter", "x": [float(theoretical_q.min()), float(theoretical_q.max())],
             "y": [float(slope * theoretical_q.min() + intercept), float(slope * theoretical_q.max() + intercept)],
             "mode": "lines", "line": {"color": "#d06060", "dash": "dash", "width": 2}, "name": "Reference Line"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 320,
            "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Sample Quantiles"},
        },
    })

    # 5. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plots.append({
            "title": "Feature Importance",
            "data": [{
                "type": "bar", "orientation": "h",
                "x": importances[sorted_idx].tolist(),
                "y": [features[i] if i < len(features) else f"feat_{i}" for i in sorted_idx],
                "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
            }],
            "layout": {"template": "plotly_dark", "height": max(220, len(features) * 22)},
        })

    # 6. Scale-Location Plot (sqrt|standardized residuals| vs fitted)
    std_res = residuals / (res_std if res_std > 0 else 1)
    sqrt_abs_std_res = np.sqrt(np.abs(std_res))
    plots.append({
        "title": "Scale-Location (Homoscedasticity Check)",
        "data": [
            {"type": "scatter", "x": y_pred_arr.tolist(), "y": sqrt_abs_std_res.tolist(),
             "mode": "markers", "marker": {"color": "rgba(138,127,191,0.5)", "size": 5}, "name": "\u221a|Std. Residual|"},
        ],
        "layout": {
            "template": "plotly_dark", "height": 300,
            "xaxis": {"title": "Fitted Values"}, "yaxis": {"title": "\u221a|Standardized Residual|"},
        },
    })

    return plots


# ─── ML Lab Helpers (Claude-powered From Intent / From Data) ───────────────────

_SCHEMA_SYSTEM_PROMPT = """You are a data scientist designing a realistic dataset for ML training.
Given the user's intent and optional domain, design a schema that would produce meaningful data.
Return ONLY valid JSON with this exact structure:
{
  "name": "descriptive dataset name",
  "target": "target_column_name",
  "task": "classification" or "regression",
  "features": [
    {"name": "column_name", "type": "numeric", "distribution": "normal", "params": {"mean": 50, "std": 10}},
    {"name": "column_name", "type": "numeric", "distribution": "uniform", "params": {"low": 0, "high": 100}},
    {"name": "column_name", "type": "categorical", "categories": ["A", "B", "C"], "probabilities": [0.5, 0.3, 0.2]}
  ],
  "target_spec": {
    "type": "categorical" or "numeric",
    "categories": ["pass", "fail"],
    "feature_weights": {"column_name": 0.6, "column_name": -0.3}
  }
}
Design 4-8 features. For target_spec.feature_weights, specify how each feature correlates with the target (-1 to 1).
For classification targets, use 2-4 categories. For regression, omit "categories".
Make feature names domain-appropriate and realistic."""

_INTERPRET_SYSTEM_PROMPT = """You are a quality engineer's data science advisor.
Interpret ML results concisely and practically.
In 3-4 sentences: What did the model learn? Which features drive predictions and why?
What should the engineer investigate or do next? Be specific to the domain."""


def _claude_generate_schema(user, intent, domain, n_records):
    """Ask Claude to design a dataset schema from natural language intent."""
    from agents_api.llm_manager import LLMManager

    prompt = f"Intent: {intent}"
    if domain:
        prompt += f"\nDomain: {domain}"
    prompt += f"\nTarget dataset size: {n_records} records"

    result = LLMManager.chat(
        user=user,
        messages=[{"role": "user", "content": prompt}],
        system=_SCHEMA_SYSTEM_PROMPT,
        max_tokens=2048,
        temperature=0.4,
    )

    if not result or not result.get("content"):
        logger.error("Claude schema generation returned no content")
        return None

    # Parse JSON from response (handle markdown code blocks)
    text = result["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.error(f"Could not parse schema JSON from Claude response: {text[:200]}")
        return None


def _generate_data_from_schema(schema, n_records):
    """Generate synthetic data from a Claude-designed schema."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    data = {}

    # Generate features
    for feat in schema.get("features", []):
        name = feat["name"]
        ftype = feat.get("type", "numeric")

        if ftype == "numeric":
            dist = feat.get("distribution", "normal")
            params = feat.get("params", {})
            if dist == "normal":
                data[name] = np.random.normal(params.get("mean", 0), params.get("std", 1), n_records)
            elif dist == "uniform":
                data[name] = np.random.uniform(params.get("low", 0), params.get("high", 100), n_records)
            elif dist == "exponential":
                data[name] = np.random.exponential(params.get("scale", 1), n_records)
            elif dist == "poisson":
                data[name] = np.random.poisson(params.get("lam", 5), n_records).astype(float)
            else:
                data[name] = np.random.normal(0, 1, n_records)
        elif ftype == "categorical":
            cats = feat.get("categories", ["A", "B"])
            probs = feat.get("probabilities", None)
            if probs and len(probs) == len(cats):
                # Normalize probabilities
                total = sum(probs)
                probs = [p / total for p in probs]
            else:
                probs = None
            data[name] = np.random.choice(cats, n_records, p=probs)

    df = pd.DataFrame(data)

    # Generate target based on feature weights
    target_spec = schema.get("target_spec", {})
    target_name = schema.get("target", "target")
    weights = target_spec.get("feature_weights", {})
    task = schema.get("task", "classification")

    # Build linear combination from numeric features
    score = np.zeros(n_records)
    for feat_name, weight in weights.items():
        if feat_name in df.columns:
            col = df[feat_name]
            if col.dtype == object:
                # Categorical: encode as integers
                col = pd.Categorical(col).codes.astype(float)
            # Standardize before weighting
            std = col.std()
            if std > 0:
                score += weight * ((col - col.mean()) / std)

    # Add noise
    score += np.random.normal(0, 0.3, n_records)

    if task == "classification":
        categories = target_spec.get("categories", ["class_0", "class_1"])
        if len(categories) == 2:
            # Binary: logistic threshold
            prob = 1 / (1 + np.exp(-score))
            df[target_name] = np.where(prob > 0.5, categories[1], categories[0])
        else:
            # Multi-class: bucket the score
            thresholds = np.linspace(score.min(), score.max(), len(categories) + 1)
            labels = np.digitize(score, thresholds[1:-1])
            labels = np.clip(labels, 0, len(categories) - 1)
            df[target_name] = [categories[i] for i in labels]
    else:
        # Regression: use score directly, scaled to reasonable range
        target_mean = target_spec.get("mean", 50)
        target_std = target_spec.get("std", 10)
        df[target_name] = score * target_std + target_mean

    return df


def _clean_for_ml(df, target):
    """Clean a DataFrame for ML: encode categoricals, handle missing values."""
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # Separate target
    y = df[target]
    X = df.drop(columns=[target])

    # Encode categorical features — cast to int to avoid CategoricalDtype errors in numpy/sklearn
    for col in X.select_dtypes(include=["object", "category"]).columns:
        codes = pd.Categorical(X[col]).codes.astype(int)
        X[col] = codes.astype(np.int32)

    # Encode categorical target for classification
    label_map = None
    if hasattr(y.dtype, 'categories') or y.dtype == object or y.dtype.name == "category":
        cats = y.unique().tolist()
        label_map = {cat: i for i, cat in enumerate(sorted(str(c) for c in cats))}
        y = y.map(label_map).astype(np.int32)

    # Fill remaining NaN with median for numeric
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Ensure all columns are numeric (final safety cast)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X, y, label_map


def _stratified_split(X, y, test_size=0.2, base_seed=42, max_retries=10):
    """Stratified train/test split with retry to ensure all classes appear in test set.

    Uses StratifiedShuffleSplit and retries with different seeds if any class
    is missing from the test set. Falls back to plain stratified split, then
    unstratified split if stratification is impossible (e.g. class with 1 sample).
    """
    from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

    all_classes = set(y.unique())

    for seed in range(base_seed, base_seed + max_retries):
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(sss.split(X, y))
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if set(y_te.unique()) == all_classes:
                return X_tr, X_te, y_tr, y_te
        except ValueError:
            pass

    # Fallback: plain stratified split
    try:
        return train_test_split(X, y, test_size=test_size, random_state=base_seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=base_seed)


def _stratified_split_3way(X, y, base_seed=42):
    """Stratified 3-way split: train 70% / calibration 15% / test 15%.

    Used for conformal prediction — calibration set is never seen during training.
    """
    from sklearn.model_selection import train_test_split

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=base_seed, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=base_seed, stratify=y_temp
        )
    except ValueError:
        # Fallback: unstratified if classes too small
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=base_seed
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=base_seed
        )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def _classification_reliability(y_full, y_test, y_pred, metrics):
    """Compute reliability warnings and enriched metrics for a classification result.

    Mutates `metrics` in place: adds balanced_accuracy, f1_macro, recall_macro,
    baseline_accuracy, class_balance, per_class, reliability_warnings.
    For binary tasks with probabilities, call separately for average_precision.
    """
    from collections import Counter
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, recall_score,
        classification_report,
    )

    counts = Counter(y_full)
    majority_pct = max(counts.values()) / len(y_full)
    all_classes = set(y_full.unique())

    # Enriched metrics
    metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_test, y_pred), 4)
    metrics["f1_macro"] = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
    metrics["recall_macro"] = round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4)
    metrics["baseline_accuracy"] = round(majority_pct, 4)
    metrics["class_balance"] = {str(k): int(v) for k, v in counts.items()}

    # Per-class breakdown
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["per_class"] = {}
    for k, vals in report.items():
        if k not in ("accuracy", "macro avg", "weighted avg") and isinstance(vals, dict):
            metrics["per_class"][str(k)] = {
                m: round(float(v), 4) for m, v in vals.items()
            }

    # Reliability warnings
    warnings = []
    test_classes = set(y_test.unique()) if hasattr(y_test, 'unique') else set(y_test)
    if test_classes != all_classes:
        missing = all_classes - test_classes
        warnings.append({"level": "high", "msg": f"Test split is missing classes: {missing}. Metrics are unreliable — all scores reflect majority-class performance only."})

    acc = metrics.get("accuracy", 0)

    if acc >= 0.99:
        warnings.append({"level": "high", "msg": "Perfect or near-perfect accuracy — check for data leakage or target-derived features."})

    if abs(acc - majority_pct) < 0.01:
        warnings.append({"level": "high", "msg": f"Model accuracy ({acc:.1%}) matches the majority baseline ({majority_pct:.1%}). The model is not learning from features."})
    elif abs(acc - majority_pct) < 0.02:
        warnings.append({"level": "high", "msg": f"Model accuracy ({acc:.1%}) is within 2% of majority baseline ({majority_pct:.1%}). Lift is negligible."})

    if majority_pct > 0.80:
        warnings.append({"level": "medium", "msg": f"Severe class imbalance ({majority_pct:.0%} majority). Balanced accuracy ({metrics['balanced_accuracy']:.1%}) is more reliable than standard accuracy."})

    if metrics["balanced_accuracy"] < 0.55 and acc > 0.80:
        warnings.append({"level": "high", "msg": f"High accuracy ({acc:.1%}) but low balanced accuracy ({metrics['balanced_accuracy']:.1%}) — model is biased toward the majority class."})

    # Minority class recall check
    for cls_key, cls_metrics in metrics["per_class"].items():
        try:
            cls_count = counts.get(int(cls_key), counts.get(cls_key, 0))
        except (ValueError, TypeError):
            cls_count = counts.get(cls_key, 0)
        if cls_count / len(y_full) < 0.20 and cls_metrics.get("recall", 1) < 0.50:
            warnings.append({"level": "high", "msg": f"Minority class '{cls_key}' recall is {cls_metrics['recall']:.0%} — the model fails to detect most instances of this class."})

    metrics["reliability_warnings"] = warnings
    return metrics


def _auto_train(X, y, task=None):
    """Auto-detect task type and train the best available model.

    Returns (model, metrics_dict, feature_importances_list, task,
             X_test, y_test, y_pred) — last 3 for diagnostic plots.
    """
    import numpy as np
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        average_precision_score,
        r2_score, mean_squared_error, mean_absolute_error,
    )

    # Auto-detect task type
    if task is None:
        is_cat = y.dtype == object or y.dtype.name == "category" or hasattr(y.dtype, 'categories')
        if y.nunique() <= 20 and (is_cat or y.nunique() <= 10):
            task = "classification"
        else:
            task = "regression"

    if task == "classification":
        # Stratified split with retry for missing classes
        X_train, X_test, y_train, y_test = _stratified_split(X, y)

        # Auto class weighting when imbalanced
        counts = Counter(y)
        majority_pct = max(counts.values()) / len(y)
        use_balanced = majority_pct > 0.75

        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced" if use_balanced else None,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }

        # Binary: add average_precision (PR AUC)
        n_classes = y.nunique()
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["average_precision"] = round(average_precision_score(y_test, y_proba), 4)
            except Exception:
                pass

        # Reliability: balanced metrics, per-class, warnings
        _classification_reliability(y, y_test, y_pred, metrics)

    else:
        # --- Regression (unchanged split) ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "r2": round(r2_score(y_test, y_pred), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }

    # Feature importance
    importances = []
    if hasattr(model, "feature_importances_"):
        for i, col in enumerate(X.columns):
            importances.append({
                "feature": col,
                "importance": round(float(model.feature_importances_[i]), 4),
            })
        importances.sort(key=lambda x: x["importance"], reverse=True)

    return model, metrics, importances, task, X_test, y_test, y_pred


def _claude_interpret_results(user, context, metrics, importances, task=None):
    """Ask Claude to interpret ML results in plain English."""
    from agents_api.llm_manager import LLMManager

    top_features = importances[:5] if importances else []
    prompt = f"""Context: {context}
Task: {task or 'auto-detected'}
Metrics: {json.dumps(metrics)}
Top features: {json.dumps(top_features)}"""

    result = LLMManager.chat(
        user=user,
        messages=[{"role": "user", "content": prompt}],
        system=_INTERPRET_SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.5,
    )

    if result and result.get("content"):
        return result["content"]
    return None


# ─── Shared Statistical Helpers ──────────────────────────────────────────────
# Used by stats.py, ml.py, spc.py, viz.py, dispatch.py


def _effect_magnitude(value, effect_type):
    """Classify effect size magnitude. Returns (label, is_meaningful)."""
    av = abs(value)
    if effect_type == "cohens_d":
        if av < 0.2: return "negligible", False
        if av < 0.5: return "small", False
        if av < 0.8: return "medium", True
        return "large", True
    elif effect_type == "eta_squared":
        if av < 0.01: return "negligible", False
        if av < 0.06: return "small", False
        if av < 0.14: return "medium", True
        return "large", True
    elif effect_type == "cramers_v":
        if av < 0.1: return "negligible", False
        if av < 0.3: return "small", False
        if av < 0.5: return "medium", True
        return "large", True
    elif effect_type == "r_squared":
        if av < 0.02: return "negligible", False
        if av < 0.13: return "small", False
        if av < 0.26: return "medium", True
        return "large", True
    return "unknown", False


def _practical_block(effect_name, effect_val, effect_type, pval, alpha=0.05, context=""):
    """Build practical significance interpretation block for analysis summaries."""
    label, meaningful = _effect_magnitude(effect_val, effect_type)

    b = f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
    b += f"<<COLOR:title>>PRACTICAL SIGNIFICANCE<</COLOR>>\n"
    b += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"
    b += f"<<COLOR:highlight>>{effect_name}:<</COLOR>> {abs(effect_val):.3f} ({label} effect)\n\n"

    if pval < alpha and meaningful:
        b += f"<<COLOR:good>>Both statistically and practically significant.<</COLOR>>\n"
        b += f"<<COLOR:text>>The difference is real and large enough to act on.{' ' + context if context else ''}<</COLOR>>"
    elif pval < alpha and label == "small":
        b += f"<<COLOR:warn>>Statistically significant but small effect.<</COLOR>>\n"
        b += f"<<COLOR:text>>The difference is real but may be too small to justify action. Consider whether the cost of change is worth this magnitude.{' ' + context if context else ''}<</COLOR>>"
    elif pval < alpha:
        b += f"<<COLOR:warn>>Statistically significant but negligible effect.<</COLOR>>\n"
        b += f"<<COLOR:text>>With enough data, even trivial differences reach significance. This difference is too small to act on.<</COLOR>>"
    elif pval >= alpha and meaningful:
        b += f"<<COLOR:warn>>Not statistically significant, but the effect size is {label}.<</COLOR>>\n"
        b += f"<<COLOR:text>>This may indicate insufficient sample size rather than no real effect. Consider collecting more data before concluding there is no difference.<</COLOR>>"
    else:
        b += f"<<COLOR:text>>Not statistically significant, and the effect is {label}.<</COLOR>>\n"
        b += f"<<COLOR:text>>No evidence of a meaningful difference.<</COLOR>>"

    return b


def _ml_interpretation(task, metrics, y_test=None, y_pred=None, features=None, target=None, model=None):
    """Build practical interpretation block for ML models."""
    import numpy as np
    b = f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
    b += f"<<COLOR:title>>MODEL ASSESSMENT<</COLOR>>\n"
    b += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"

    if task == "classification" and y_test is not None:
        acc = metrics.get("accuracy", 0)
        from collections import Counter
        class_counts = Counter(y_test)
        majority_pct = max(class_counts.values()) / len(y_test)
        n_classes = len(class_counts)
        lift = acc - majority_pct

        b += f"<<COLOR:highlight>>Accuracy:<</COLOR>> {acc:.1%}\n"
        b += f"<<COLOR:highlight>>Baseline (always predict majority class):<</COLOR>> {majority_pct:.1%}\n"
        b += f"<<COLOR:highlight>>Lift over baseline:<</COLOR>> {lift:+.1%}\n\n"

        if lift < 0.02:
            b += f"<<COLOR:danger>>Model is barely better than always guessing '{class_counts.most_common(1)[0][0]}'.<</COLOR>>\n"
            b += f"<<COLOR:text>>Consider: more features, better features, or more data. The model is not learning meaningful patterns.<</COLOR>>"
        elif lift < 0.10:
            b += f"<<COLOR:warn>>Model is modestly better than baseline ({lift:+.1%}).<</COLOR>>\n"
            b += f"<<COLOR:text>>Useful for screening but not for high-stakes decisions. Consider feature engineering or more data.<</COLOR>>"
        elif acc >= 0.95:
            b += f"<<COLOR:good>>Excellent performance.<</COLOR>>\n"
            b += f"<<COLOR:text>>Verify this isn't due to data leakage (a feature that perfectly predicts the target because it's derived from it).<</COLOR>>"
        elif acc >= 0.85:
            b += f"<<COLOR:good>>Strong model — suitable for decision support.<</COLOR>>\n"
            b += f"<<COLOR:text>>Review the confusion matrix to check if errors are concentrated in specific classes.<</COLOR>>"
        else:
            b += f"<<COLOR:text>>Moderate performance.<</COLOR>>\n"
            b += f"<<COLOR:text>>The model learned real patterns ({lift:+.1%} over baseline). Consider whether this accuracy is sufficient for your use case.<</COLOR>>"

        if majority_pct > 0.8:
            minority = class_counts.most_common()[-1]
            b += f"\n\n<<COLOR:warn>>Class imbalance detected:<</COLOR>> '{minority[0]}' has only {minority[1]} samples ({minority[1]/len(y_test):.1%}).\n"
            b += f"<<COLOR:text>>Accuracy may be misleading. Check precision and recall for the minority class — F1 score is more informative here.<</COLOR>>"

        if y_pred is not None and n_classes <= 10:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(list(set(y_test)))
            np.fill_diagonal(cm, 0)
            if cm.max() > 0:
                worst_i, worst_j = np.unravel_index(cm.argmax(), cm.shape)
                b += f"\n\n<<COLOR:text>>Most confused pair:<</COLOR>> '{labels[worst_i]}' misclassified as '{labels[worst_j]}' ({cm[worst_i, worst_j]} times)."

    elif task == "regression":
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)

        b += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n"
        b += f"<<COLOR:highlight>>RMSE:<</COLOR>> {rmse:.4f}\n"

        if y_test is not None and len(y_test) > 0:
            y_range = float(np.ptp(y_test))
            y_mean = float(np.mean(y_test))
            rmse_pct = (rmse / y_range * 100) if y_range > 0 else 0
            cv_rmse = (rmse / abs(y_mean) * 100) if abs(y_mean) > 0 else 0
            b += f"<<COLOR:highlight>>RMSE as % of range:<</COLOR>> {rmse_pct:.1f}%\n"
            b += f"<<COLOR:highlight>>CV(RMSE):<</COLOR>> {cv_rmse:.1f}% of mean\n\n"

            if r2 < 0.1:
                b += f"<<COLOR:danger>>Model explains very little variation.<</COLOR>>\n"
                b += f"<<COLOR:text>>The features selected may not predict '{target}'. Try different features or check data quality.<</COLOR>>"
            elif r2 < 0.5:
                b += f"<<COLOR:warn>>Moderate fit — predictions will have substantial error.<</COLOR>>\n"
                b += f"<<COLOR:text>>On average, predictions are off by {rmse:.2f} ({rmse_pct:.0f}% of the data range). Useful for trend identification but not precision control.<</COLOR>>"
            elif r2 < 0.8:
                b += f"<<COLOR:text>>Good fit — the model captures most of the pattern.<</COLOR>>\n"
                b += f"<<COLOR:text>>Predictions are off by +-{rmse:.2f} on average. Suitable for forecasting and decision support.<</COLOR>>"
            else:
                b += f"<<COLOR:good>>Strong fit.<</COLOR>>\n"
                b += f"<<COLOR:text>>Model explains {r2*100:.0f}% of variation. Predictions are off by +-{rmse:.2f}. Check for data leakage if R² > 0.95.<</COLOR>>"
        else:
            r2_label, _ = _effect_magnitude(r2, "r_squared")
            b += f"\n<<COLOR:text>>R² is {r2_label}. Model explains {r2*100:.0f}% of variation in '{target}'.<</COLOR>>"

    return b


def _fit_best_distribution(data):
    """Fit multiple distributions and return the best by KS p-value.
    Returns (dist_name, scipy_dist, fit_args, ks_pvalue)."""
    from scipy import stats as sp_stats
    import numpy as np

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        mu, sigma = float(np.mean(data)), max(float(np.std(data)), 1e-6)
        return "normal", sp_stats.norm, (mu, sigma), 1.0

    candidates = []

    # Normal
    try:
        mu, sigma = sp_stats.norm.fit(data)
        ks = sp_stats.kstest(data, "norm", args=(mu, sigma))
        candidates.append(("normal", sp_stats.norm, (mu, sigma), ks.pvalue))
    except Exception:
        pass

    # Lognormal (only positive data)
    if np.all(data > 0):
        try:
            s, loc, scale = sp_stats.lognorm.fit(data, floc=0)
            ks = sp_stats.kstest(data, "lognorm", args=(s, 0, scale))
            candidates.append(("lognormal", sp_stats.lognorm, (s, 0, scale), ks.pvalue))
        except Exception:
            pass

    # Weibull (only positive data)
    if np.all(data > 0):
        try:
            c, loc, scale = sp_stats.weibull_min.fit(data, floc=0)
            ks = sp_stats.kstest(data, "weibull_min", args=(c, 0, scale))
            candidates.append(("weibull", sp_stats.weibull_min, (c, 0, scale), ks.pvalue))
        except Exception:
            pass

    # Exponential (only positive data)
    if np.all(data > 0):
        try:
            loc, scale = sp_stats.expon.fit(data)
            ks = sp_stats.kstest(data, "expon", args=(loc, scale))
            candidates.append(("exponential", sp_stats.expon, (loc, scale), ks.pvalue))
        except Exception:
            pass

    # Gamma (only positive data)
    if np.all(data > 0):
        try:
            a, loc, scale = sp_stats.gamma.fit(data, floc=0)
            ks = sp_stats.kstest(data, "gamma", args=(a, 0, scale))
            candidates.append(("gamma", sp_stats.gamma, (a, 0, scale), ks.pvalue))
        except Exception:
            pass

    # Uniform
    try:
        loc, scale = sp_stats.uniform.fit(data)
        ks = sp_stats.kstest(data, "uniform", args=(loc, scale))
        candidates.append(("uniform", sp_stats.uniform, (loc, scale), ks.pvalue))
    except Exception:
        pass

    if not candidates:
        mu, sigma = float(np.mean(data)), max(float(np.std(data)), 1e-6)
        return "normal", sp_stats.norm, (mu, sigma), 0.0

    best = max(candidates, key=lambda x: x[3])
    return best
