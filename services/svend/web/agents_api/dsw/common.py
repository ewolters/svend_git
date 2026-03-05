"""DSW common utilities — shared across all DSW sub-modules.

Extracted from dsw_views.py: model cache, logging, ML helpers,
Claude schema/interpret integration, diagnostic plots, and data splitting.
"""

import json
import logging
import math
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np
from django.conf import settings
from django.http import JsonResponse

from ..models import AgentLog, SavedModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVEND theme colors — canonical palette
# Matches CSS custom properties in base_app.html. All DSW chart modules
# should import from here instead of defining their own palettes.
# ---------------------------------------------------------------------------

# Primary categorical palette (10 colors, for multi-series charts)
SVEND_COLORS = [
    "#4a9f6e",  # accent-primary (green)
    "#4a9faf",  # accent-blue (teal)
    "#e89547",  # accent-orange
    "#e8c547",  # accent-gold
    "#8a7fbf",  # accent-purple
    "#d66b9f",  # magenta
    "#4ac9c0",  # cyan
    "#c97a4a",  # brown
    "#6b9f4a",  # olive
    "#5b8bd6",  # blue
]

# Semantic colors — use for consistent meaning across all charts
COLOR_GOOD = "#4a9f6e"  # success, in-control, healthy
COLOR_BAD = "#d06060"  # error, out-of-control, defect
COLOR_WARNING = "#e89547"  # caution, moderate
COLOR_INFO = "#4a9faf"  # informational, secondary
COLOR_NEUTRAL = "#888888"  # noise floor, baseline, inactive
COLOR_REFERENCE = "#d4a24a"  # reference lines, pooled values
COLOR_GOLD = "#e8c547"  # tertiary emphasis, threshold


# Alpha variants for fills and bands
def _rgba(hex_color, alpha=0.15):
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# JSON-safe serialization (handles numpy NaN/Inf/types)
# ---------------------------------------------------------------------------


def sanitize_for_json(obj):
    """Recursively replace NaN/Infinity/numpy types with JSON-safe values."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _strip_non_serializable(obj):
    """Recursively convert non-JSON-serializable objects to strings."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _strip_non_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_non_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Catch-all for sklearn objects, scalers, etc.
    return str(type(obj).__name__)


def safe_json_response(data, status=200):
    """JsonResponse that safely handles numpy types and NaN/Infinity."""
    return JsonResponse(sanitize_for_json(data), status=status)


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
        expired = [k for k, v in _model_cache[user_id].items() if now - v["timestamp"] > MODEL_CACHE_EXPIRY]
        for k in expired:
            del _model_cache[user_id][k]

        # Limit cache size per user
        while len(_model_cache[user_id]) >= MODEL_CACHE_MAX_SIZE:
            _model_cache[user_id].popitem(last=False)

        _model_cache[user_id][model_key] = {"model": model, "metadata": metadata, "timestamp": now}


def get_cached_model(user_id, model_key):
    """Retrieve a cached model."""
    with _model_cache_lock:
        if user_id in _model_cache and model_key in _model_cache[user_id]:
            entry = _model_cache[user_id][model_key]
            if time.time() - entry["timestamp"] < MODEL_CACHE_EXPIRY:
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
            is_success=success,
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


def save_model_to_disk(
    user,
    model,
    model_type,
    dsw_result_id,
    name=None,
    metrics=None,
    features=None,
    target=None,
    project_id=None,
    training_config=None,
    data_lineage=None,
    parent_model_id=None,
):
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

        # Sanitize training_config — strip non-serializable objects (sklearn scalers, etc.)
        safe_config = _strip_non_serializable(training_config or {})

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
            training_config=safe_config,
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


def _build_ml_diagnostics(model, X_test, y_test, y_pred, features, task, label_map=None, model_name="Model"):
    """Build comprehensive diagnostic plots for a trained ML model.

    Returns list of Plotly plot dicts (same format as DSW analysis plots).

    Classification (6 plots): confusion matrix, ROC, precision-recall,
    feature importance, probability distribution, calibration curve.

    Regression (6 plots): actual-vs-predicted, residuals, residual histogram,
    Q-Q plot, feature importance, scale-location.
    """

    plots = []

    if task == "classification":
        plots.extend(_diag_classification(model, X_test, y_test, y_pred, features, label_map, model_name))
    else:
        plots.extend(_diag_regression(model, X_test, y_test, y_pred, features, model_name))

    return plots


def _diag_classification(model, X_test, y_test, y_pred, features, label_map, model_name):
    """Classification diagnostic suite."""
    import numpy as np
    from sklearn.metrics import (
        auc,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    plots = []
    classes = sorted(list(set(y_test.tolist()) | set(y_pred.tolist())))
    str_labels = [str(label_map.get(c, c) if label_map else c) for c in classes]

    # 1. Confusion Matrix (annotated with counts + percentages)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(
                {
                    "x": str_labels[j],
                    "y": str_labels[i],
                    "text": f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)",
                    "showarrow": False,
                    "font": {"color": "#fff" if cm[i][j] > cm.max() * 0.5 else "#aaa", "size": 12},
                }
            )
    plots.append(
        {
            "title": "Confusion Matrix",
            "data": [
                {
                    "type": "heatmap",
                    "z": cm.tolist(),
                    "x": str_labels,
                    "y": str_labels,
                    "colorscale": [[0, "#0d120d"], [0.5, "#2a6b3a"], [1, "#4a9f6e"]],
                    "showscale": True,
                    "hoverongaps": False,
                    "hovertemplate": "Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
                }
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Predicted", "side": "bottom"},
                "yaxis": {"title": "Actual", "autorange": "reversed"},
                "annotations": annotations,
            },
        }
    )

    # 2. ROC Curve
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if len(classes) == 2:
                # Binary ROC
                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1], pos_label=classes[1])
                roc_auc = auc(fpr, tpr)
                # Find optimal threshold (Youden's J)
                j_scores = tpr - fpr
                opt_idx = np.argmax(j_scores)
                plots.append(
                    {
                        "title": f"ROC Curve (AUC = {roc_auc:.3f})",
                        "data": [
                            {
                                "type": "scatter",
                                "x": fpr.tolist(),
                                "y": tpr.tolist(),
                                "mode": "lines",
                                "line": {"color": "#4a9f6e", "width": 2.5},
                                "name": f"ROC (AUC={roc_auc:.3f})",
                                "fill": "tozeroy",
                                "fillcolor": "rgba(74,159,110,0.1)",
                            },
                            {
                                "type": "scatter",
                                "x": [0, 1],
                                "y": [0, 1],
                                "mode": "lines",
                                "line": {"dash": "dash", "color": "#555", "width": 1},
                                "name": "Random",
                                "showlegend": True,
                            },
                            {
                                "type": "scatter",
                                "x": [float(fpr[opt_idx])],
                                "y": [float(tpr[opt_idx])],
                                "mode": "markers",
                                "marker": {"color": "#e89547", "size": 10, "symbol": "star"},
                                "name": f"Optimal (J={j_scores[opt_idx]:.2f})",
                                "showlegend": True,
                            },
                        ],
                        "layout": {
                            "height": 320,
                            "xaxis": {"title": "False Positive Rate", "range": [0, 1]},
                            "yaxis": {"title": "True Positive Rate", "range": [0, 1.05]},
                        },
                    }
                )
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
                        roc_traces.append(
                            {
                                "type": "scatter",
                                "x": fpr_i.tolist(),
                                "y": tpr_i.tolist(),
                                "mode": "lines",
                                "line": {"color": colors[i % len(colors)], "width": 2},
                                "name": f"{lbl} (AUC={auc_i:.3f})",
                            }
                        )
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
                plots.append(
                    {
                        "title": "ROC Curves (One-vs-Rest)",
                        "data": roc_traces,
                        "layout": {"height": 350, "xaxis": {"title": "FPR"}, "yaxis": {"title": "TPR"}},
                    }
                )
        except Exception:
            pass

    # 3. Precision-Recall Curve
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if len(classes) == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label=classes[1])
                ap = average_precision_score(y_test, y_prob[:, 1], pos_label=classes[1])
                plots.append(
                    {
                        "title": f"Precision-Recall Curve (AP = {ap:.3f})",
                        "data": [
                            {
                                "type": "scatter",
                                "x": recall.tolist(),
                                "y": precision.tolist(),
                                "mode": "lines",
                                "line": {"color": "#4a9faf", "width": 2.5},
                                "name": f"PR (AP={ap:.3f})",
                                "fill": "tozeroy",
                                "fillcolor": "rgba(74,159,175,0.1)",
                            },
                        ],
                        "layout": {
                            "height": 320,
                            "xaxis": {"title": "Recall", "range": [0, 1]},
                            "yaxis": {"title": "Precision", "range": [0, 1.05]},
                        },
                    }
                )
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
                        pr_traces.append(
                            {
                                "type": "scatter",
                                "x": r_i.tolist(),
                                "y": p_i.tolist(),
                                "mode": "lines",
                                "line": {"color": colors[i % len(colors)], "width": 2},
                                "name": f"{lbl} (AP={ap_i:.3f})",
                            }
                        )
                plots.append(
                    {
                        "title": "Precision-Recall Curves (Per Class)",
                        "data": pr_traces,
                        "layout": {"height": 350, "xaxis": {"title": "Recall"}, "yaxis": {"title": "Precision"}},
                    }
                )
        except Exception:
            pass

    # 4. Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plots.append(
            {
                "title": "Feature Importance",
                "data": [
                    {
                        "type": "bar",
                        "orientation": "h",
                        "x": importances[sorted_idx].tolist(),
                        "y": [features[i] if i < len(features) else f"feat_{i}" for i in sorted_idx],
                        "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                    }
                ],
                "layout": {"height": max(220, len(features) * 22)},
            }
        )

    # 5. Predicted Probability Distribution
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            prob_traces = []
            colors = ["#4a9f6e", "#d06060", "#4a9faf", "#e8c547", "#8a7fbf", "#e89547"]
            for i, cls in enumerate(classes):
                mask = np.array(y_test) == cls
                if mask.any() and y_prob.shape[1] > 1:
                    col_idx = min(1, y_prob.shape[1] - 1) if len(classes) == 2 else i
                    lbl = str(label_map.get(cls, cls) if label_map else cls)
                    prob_traces.append(
                        {
                            "type": "histogram",
                            "x": y_prob[mask, col_idx].tolist(),
                            "name": f"Actual: {lbl}",
                            "opacity": 0.6,
                            "marker": {"color": colors[i % len(colors)]},
                        }
                    )
            if prob_traces:
                plots.append(
                    {
                        "title": "Predicted Probability Distribution",
                        "data": prob_traces,
                        "layout": {
                            "height": 280,
                            "barmode": "overlay",
                            "xaxis": {"title": "Predicted Probability"},
                            "yaxis": {"title": "Count"},
                        },
                    }
                )
        except Exception:
            pass

    # 6. Calibration Curve (reliability diagram)
    if hasattr(model, "predict_proba") and len(classes) == 2:
        try:
            from sklearn.calibration import calibration_curve

            y_prob = model.predict_proba(X_test)[:, 1]
            y_binary = (np.array(y_test) == classes[1]).astype(int)
            n_bins = min(10, max(3, len(y_test) // 20))
            fraction_pos, mean_pred = calibration_curve(y_binary, y_prob, n_bins=n_bins)
            plots.append(
                {
                    "title": "Calibration Curve",
                    "data": [
                        {
                            "type": "scatter",
                            "x": mean_pred.tolist(),
                            "y": fraction_pos.tolist(),
                            "mode": "lines+markers",
                            "line": {"color": "#4a9f6e", "width": 2},
                            "marker": {"size": 8, "color": "#4a9f6e"},
                            "name": model_name,
                        },
                        {
                            "type": "scatter",
                            "x": [0, 1],
                            "y": [0, 1],
                            "mode": "lines",
                            "line": {"dash": "dash", "color": "#555"},
                            "name": "Perfectly Calibrated",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Mean Predicted Probability", "range": [0, 1]},
                        "yaxis": {"title": "Fraction of Positives", "range": [0, 1]},
                    },
                }
            )
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
    plots.append(
        {
            "title": f"Actual vs Predicted (R² = {r2:.4f})",
            "data": [
                {
                    "type": "scatter",
                    "x": y_test_arr.tolist(),
                    "y": y_pred_arr.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74,159,110,0.5)", "size": 5, "line": {"color": "#4a9f6e", "width": 0.5}},
                    "name": "Predictions",
                },
                {
                    "type": "scatter",
                    "x": [y_min, y_max],
                    "y": [y_min, y_max],
                    "mode": "lines",
                    "line": {"color": "#d06060", "dash": "dash", "width": 2},
                    "name": "Perfect Fit",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Actual"},
                "yaxis": {"title": "Predicted"},
                "annotations": [
                    {
                        "x": 0.05,
                        "y": 0.95,
                        "xref": "paper",
                        "yref": "paper",
                        "text": f"R² = {r2:.4f}",
                        "showarrow": False,
                        "font": {"size": 14, "color": "#4a9f6e"},
                        "bgcolor": "rgba(0,0,0,0.5)",
                    }
                ],
            },
        }
    )

    # 2. Residuals vs Predicted (with lowess trend)
    # Color by magnitude
    abs_res = np.abs(residuals)
    res_norm = (abs_res - abs_res.min()) / (abs_res.max() - abs_res.min() + 1e-10)
    plots.append(
        {
            "title": "Residuals vs Predicted",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred_arr.tolist(),
                    "y": residuals.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": res_norm.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [1, "#d06060"]],
                        "size": 5,
                        "showscale": True,
                        "colorbar": {"title": "|Resid|", "thickness": 10, "len": 0.6},
                    },
                    "name": "Residuals",
                    "hovertemplate": "Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
                },
                {
                    "type": "scatter",
                    "x": [float(y_pred_arr.min()), float(y_pred_arr.max())],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#d06060", "dash": "dash", "width": 1.5},
                    "name": "Zero",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Predicted"},
                "yaxis": {"title": "Residual"},
            },
        }
    )

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

    plots.append(
        {
            "title": f"Residual Distribution (Shapiro-Wilk p = {sw_p:.4f})",
            "data": [
                {
                    "type": "histogram",
                    "x": residuals.tolist(),
                    "nbinsx": 30,
                    "marker": {"color": "rgba(74,159,110,0.5)", "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals",
                },
                {
                    "type": "scatter",
                    "x": x_norm.tolist(),
                    "y": y_norm_scaled.tolist(),
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2.5},
                    "name": "Normal Fit",
                },
            ],
            "layout": {
                "height": 300,
                "barmode": "overlay",
                "xaxis": {"title": "Residual"},
                "yaxis": {"title": "Frequency"},
            },
        }
    )

    # 4. Q-Q Plot (Normal Probability Plot)
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical_q = stats.norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
    # Reference line
    slope, intercept, _, _, _ = stats.linregress(theoretical_q, sorted_res)

    plots.append(
        {
            "title": "Normal Q-Q Plot",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_res.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74,159,175,0.6)", "size": 4},
                    "name": "Residuals",
                },
                {
                    "type": "scatter",
                    "x": [float(theoretical_q.min()), float(theoretical_q.max())],
                    "y": [
                        float(slope * theoretical_q.min() + intercept),
                        float(slope * theoretical_q.max() + intercept),
                    ],
                    "mode": "lines",
                    "line": {"color": "#d06060", "dash": "dash", "width": 2},
                    "name": "Reference Line",
                },
            ],
            "layout": {
                "height": 320,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Sample Quantiles"},
            },
        }
    )

    # 5. Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        plots.append(
            {
                "title": "Feature Importance",
                "data": [
                    {
                        "type": "bar",
                        "orientation": "h",
                        "x": importances[sorted_idx].tolist(),
                        "y": [features[i] if i < len(features) else f"feat_{i}" for i in sorted_idx],
                        "marker": {"color": "rgba(74,159,110,0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                    }
                ],
                "layout": {"height": max(220, len(features) * 22)},
            }
        )

    # 6. Scale-Location Plot (sqrt|standardized residuals| vs fitted)
    std_res = residuals / (res_std if res_std > 0 else 1)
    sqrt_abs_std_res = np.sqrt(np.abs(std_res))
    plots.append(
        {
            "title": "Scale-Location (Homoscedasticity Check)",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred_arr.tolist(),
                    "y": sqrt_abs_std_res.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(138,127,191,0.5)", "size": 5},
                    "name": "\u221a|Std. Residual|",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Fitted Values"},
                "yaxis": {"title": "\u221a|Standardized Residual|"},
            },
        }
    )

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
What should the engineer investigate or do next? Be specific to the domain.
If reliability warnings are provided, address the most critical ones directly.
Do NOT just list warnings — explain what they mean for this specific use case."""


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

        match = re.search(r"\{[\s\S]*\}", text)
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
    import numpy as np
    import pandas as pd

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
    if hasattr(y.dtype, "categories") or y.dtype == object or y.dtype.name == "category":
        cats = y.unique().tolist()
        label_map = {cat: i for i, cat in enumerate(sorted(str(c) for c in cats))}
        y = y.map(label_map).astype(np.int32)

    # Fill remaining NaN with median for numeric
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Ensure all columns are numeric (final safety cast)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, label_map


def _stratified_split(X, y, test_size=0.2, base_seed=42, max_retries=10):
    """Stratified train/test split with retry to ensure all classes appear in test set.

    Uses StratifiedShuffleSplit and retries with different seeds if any class
    is missing from the test set. Falls back to plain stratified split, then
    unstratified split if stratification is impossible (e.g. class with 1 sample).
    """
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=base_seed, stratify=y)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=base_seed, stratify=y_temp
        )
    except ValueError:
        # Fallback: unstratified if classes too small
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=base_seed)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=base_seed)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def _classification_reliability(y_full, y_test, y_pred, metrics):
    """Compute reliability warnings and enriched metrics for a classification result.

    Mutates `metrics` in place: adds balanced_accuracy, f1_macro, recall_macro,
    baseline_accuracy, class_balance, per_class, reliability_warnings.
    For binary tasks with probabilities, call separately for average_precision.
    """
    from collections import Counter

    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        f1_score,
        recall_score,
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
            metrics["per_class"][str(k)] = {m: round(float(v), 4) for m, v in vals.items()}

    # Reliability warnings
    warnings = []
    test_classes = set(y_test.unique()) if hasattr(y_test, "unique") else set(y_test)
    if test_classes != all_classes:
        missing = all_classes - test_classes
        warnings.append(
            {
                "level": "high",
                "msg": f"Test split is missing classes: {missing}. Metrics are unreliable — all scores reflect majority-class performance only.",
            }
        )

    acc = metrics.get("accuracy", 0)

    if acc >= 0.99:
        warnings.append(
            {
                "level": "critical",
                "msg": "Perfect or near-perfect accuracy — check for data leakage or target-derived features.",
            }
        )

    if abs(acc - majority_pct) < 0.01:
        warnings.append(
            {
                "level": "high",
                "msg": f"Model accuracy ({acc:.1%}) matches the majority baseline ({majority_pct:.1%}). The model is not learning from features.",
            }
        )
    elif abs(acc - majority_pct) < 0.02:
        warnings.append(
            {
                "level": "high",
                "msg": f"Model accuracy ({acc:.1%}) is within 2% of majority baseline ({majority_pct:.1%}). Lift is negligible.",
            }
        )

    if majority_pct > 0.80:
        warnings.append(
            {
                "level": "medium",
                "msg": f"Severe class imbalance ({majority_pct:.0%} majority). Balanced accuracy ({metrics['balanced_accuracy']:.1%}) is more reliable than standard accuracy.",
            }
        )

    if metrics["balanced_accuracy"] < 0.55 and acc > 0.80:
        warnings.append(
            {
                "level": "high",
                "msg": f"High accuracy ({acc:.1%}) but low balanced accuracy ({metrics['balanced_accuracy']:.1%}) — model is biased toward the majority class.",
            }
        )

    # Minority class recall check
    for cls_key, cls_metrics in metrics["per_class"].items():
        try:
            cls_count = counts.get(int(cls_key), counts.get(cls_key, 0))
        except (ValueError, TypeError):
            cls_count = counts.get(cls_key, 0)
        if cls_count / len(y_full) < 0.20 and cls_metrics.get("recall", 1) < 0.50:
            warnings.append(
                {
                    "level": "high",
                    "msg": f"Minority class '{cls_key}' recall is {cls_metrics['recall']:.0%} — the model fails to detect most instances of this class.",
                }
            )

    metrics["reliability_warnings"] = warnings
    return metrics


def _regression_reliability(y_full, y_test, y_pred, metrics):
    """Compute reliability warnings for regression results.

    Mutates `metrics` in place: adds reliability_warnings.
    """
    import numpy as np

    warnings = []
    r2 = metrics.get("r2", 0)
    rmse = metrics.get("rmse", 0)

    if r2 < 0:
        warnings.append(
            {
                "level": "critical",
                "msg": f"Negative R\u00b2 ({r2:.4f}) \u2014 the model is worse than predicting the mean. Features may be noise.",
            }
        )

    if r2 >= 0.99:
        warnings.append(
            {
                "level": "critical",
                "msg": "R\u00b2 \u2265 0.99 \u2014 check for data leakage or target-derived features.",
            }
        )

    if 0 <= r2 < 0.10:
        warnings.append(
            {
                "level": "high",
                "msg": f"R\u00b2 = {r2:.4f} \u2014 the model explains less than 10% of variance. Features may not predict target.",
            }
        )

    if y_test is not None and len(y_test) > 0:
        y_range = float(np.ptp(y_test))
        y_std = float(np.std(y_test))
        if y_range > 0:
            rmse_pct = rmse / y_range * 100
            if rmse_pct > 50:
                warnings.append(
                    {
                        "level": "high",
                        "msg": f"RMSE is {rmse_pct:.0f}% of the target range \u2014 predictions are very imprecise.",
                    }
                )
            elif rmse_pct > 25:
                warnings.append(
                    {
                        "level": "medium",
                        "msg": f"RMSE is {rmse_pct:.0f}% of the target range \u2014 moderate prediction error.",
                    }
                )

        if y_pred is not None and y_std > 0:
            residuals = np.array(y_test) - np.array(y_pred)
            mean_resid = float(np.mean(residuals))
            if abs(mean_resid) > 0.1 * y_std:
                direction = "over" if mean_resid < 0 else "under"
                warnings.append(
                    {
                        "level": "medium",
                        "msg": f"Model systematically {direction}-predicts (mean residual: {mean_resid:.3f}).",
                    }
                )

    metrics["reliability_warnings"] = warnings
    return metrics


def _data_skepticism(X, y, importances=None):
    """Data-level skepticism checks applicable to both classification and regression.

    Returns list of warning dicts: [{"level": "critical"|"high"|"medium", "msg": "..."}].
    """
    import numpy as np

    warnings = []
    n_rows, n_cols = X.shape

    # Dimensionality
    if n_cols > n_rows * 0.5:
        warnings.append(
            {
                "level": "critical",
                "msg": f"Very high dimensionality: {n_cols} features for {n_rows} rows. Model is likely memorizing noise.",
            }
        )
    elif n_cols > n_rows * 0.2:
        warnings.append(
            {
                "level": "high",
                "msg": f"High dimensionality: {n_cols} features for {n_rows} rows (ratio {n_cols / n_rows:.2f}). Risk of overfitting.",
            }
        )

    # Small dataset
    if n_rows < 50:
        warnings.append(
            {
                "level": "high",
                "msg": f"Very small dataset ({n_rows} rows). Metrics are unreliable \u2014 consider collecting more data.",
            }
        )
    elif n_rows < 100:
        warnings.append(
            {"level": "medium", "msg": f"Small dataset ({n_rows} rows). Cross-validation variance will be high."}
        )

    # Feature importance concentration
    if importances and len(importances) >= 3:
        top_imp = importances[0].get("importance", 0)
        total_imp = sum(f.get("importance", 0) for f in importances)
        if total_imp > 0 and top_imp / total_imp > 0.70:
            warnings.append(
                {
                    "level": "medium",
                    "msg": f"Feature '{importances[0]['feature']}' dominates ({top_imp / total_imp:.0%} of total importance). "
                    f"Model is essentially a single-feature predictor.",
                }
            )

    # Multicollinearity (fast correlation check)
    try:
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] >= 2:
            corr = numeric_X.corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_corr = corr.max().max()
            if max_corr > 0.95:
                pair = corr.stack().idxmax()
                warnings.append(
                    {
                        "level": "high",
                        "msg": f"Near-perfect collinearity ({max_corr:.2f}) between '{pair[0]}' and '{pair[1]}'. "
                        f"One may be redundant or derived from the other.",
                    }
                )
            elif max_corr > 0.85:
                pair = corr.stack().idxmax()
                warnings.append(
                    {
                        "level": "medium",
                        "msg": f"High correlation ({max_corr:.2f}) between '{pair[0]}' and '{pair[1]}'. "
                        f"Consider removing one to improve interpretability.",
                    }
                )
    except Exception:
        pass

    # Leakage-suspect feature names
    if importances and len(importances) > 0:
        top_feat = importances[0]
        suspect_patterns = ["_id", "index", "row_num", "timestamp", "date_created"]
        feat_lower = top_feat["feature"].lower()
        for pattern in suspect_patterns:
            if pattern in feat_lower and top_feat.get("importance", 0) > 0.15:
                warnings.append(
                    {
                        "level": "high",
                        "msg": f"Top feature '{top_feat['feature']}' looks like an identifier or timestamp. "
                        f"This may indicate data leakage.",
                    }
                )
                break

    return warnings


# ─── Bayesian Model Confidence ────────────────────────────────────────────────


def _concern_sigmoid(x, center, steepness):
    """Map metric x to concern probability [0, 1] via logistic sigmoid."""
    import math

    z = steepness * (x - center)
    z = max(-20, min(20, z))  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-z))


# Concern weights for log-linear fusion (matches PBS MultiStreamHealth pattern)
_CONCERN_WEIGHTS = {
    # Critical — model may be fundamentally broken
    "leakage": 1.0,
    "not_learning": 1.0,
    "random_signal": 1.0,
    "accuracy_illusion": 0.9,
    "duplicate_contamination": 0.9,
    # Structural — model is limited but may still be useful
    "class_imbalance": 0.7,
    "minority_blindness": 0.7,
    "imprecision": 0.7,
    "overfit_risk": 0.7,
    "unstable_performance": 0.7,
    # Advisory — worth knowing, shouldn't crater confidence alone
    "collinearity": 0.4,
    "single_feature": 0.4,
    "small_sample": 0.5,
    "bias": 0.5,
}


def _permutation_reality_test(model, X, y, task, cv=None):
    """Run permutation test to compute P(real signal > random | observed lift).

    Shuffles y N times, retrains, records metric distribution.
    Returns empirical p-value: proportion of permuted scores >= real score.

    Scoring: PR-AUC (binary classification), balanced_accuracy (multiclass),
    R² (regression). If caller provides a cv splitter (GroupKFold,
    TimeSeriesSplit), the permutation respects that split regime.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, permutation_test_score

    n_rows = X.shape[0]

    # Cap dataset size for speed — subsample large datasets
    max_rows = 2000
    if n_rows > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_rows, max_rows, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True) if hasattr(y, "iloc") else np.array(y)[idx]
        n_rows = max_rows

    if n_rows < 500:
        n_perms = 20
    elif n_rows <= 2000:
        n_perms = 15
    else:
        n_perms = 10

    # Pick the honest metric for this task
    n_classes = len(set(y))
    if task == "classification":
        if n_classes == 2 and hasattr(model, "predict_proba"):
            scoring = "average_precision"  # PR-AUC — best for rare events
        else:
            scoring = "balanced_accuracy"
    else:
        scoring = "r2"

    # Default to StratifiedKFold(3) if no cv splitter provided
    if cv is None:
        if task == "classification":
            cv = StratifiedKFold(n_splits=min(3, n_rows), shuffle=True, random_state=42)
        else:
            cv = min(3, n_rows)

    estimator = clone(model)
    if hasattr(estimator, "n_estimators"):
        estimator.set_params(n_estimators=min(getattr(estimator, "n_estimators", 100), 30))
    # Keep n_jobs=1 inside the estimator — permutation_test_score already
    # parallelises across permutations; nesting parallelism causes OOM on
    # production (single gunicorn worker, limited RAM).
    if hasattr(estimator, "n_jobs"):
        estimator.set_params(n_jobs=1)

    real_score, perm_scores, p_value = permutation_test_score(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_permutations=n_perms,
        n_jobs=1,
        random_state=42,
    )

    # Compute baseline for plot annotation
    if scoring == "average_precision":
        from collections import Counter

        counts = Counter(y)
        prevalence = min(counts.values()) / len(y) if len(y) > 0 else 0.5
        baseline = prevalence  # random PR-AUC ≈ prevalence
    elif scoring == "balanced_accuracy":
        baseline = 1.0 / max(n_classes, 2)  # chance level
    else:
        baseline = 0.0  # random R² ≈ 0

    def _safe_float(v, default=0.0):
        v = float(v)
        return default if (np.isnan(v) or np.isinf(v)) else v

    return {
        "p_value": _safe_float(p_value, 1.0),
        "real_score": _safe_float(real_score),
        "perm_scores": [_safe_float(s) for s in perm_scores],
        "n_permutations": n_perms,
        "scoring": scoring,
        "baseline": _safe_float(baseline),
    }


def _duplicate_audit(X, y):
    """Check for exact/near duplicates, ID-like columns, and perfect separability."""
    import numpy as np

    n_rows = len(X)

    # Exact duplicates
    n_exact_dups = int(X.duplicated().sum())
    n_exact_dups / n_rows if n_rows > 0 else 0

    # Near-duplicates: round numeric features to 3 decimals, re-check
    n_near_dups = 0
    try:
        X_rounded = X.copy()
        numeric_cols = X_rounded.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_rounded[numeric_cols] = X_rounded[numeric_cols].round(3)
            n_near_dups = int(X_rounded.duplicated().sum()) - n_exact_dups
            n_near_dups / n_rows if n_rows > 0 else 0
    except Exception:
        pass

    dup_rate = (n_exact_dups + max(n_near_dups, 0)) / n_rows if n_rows > 0 else 0

    # ID-like column detection
    id_columns = []
    for col in X.columns:
        col_lower = str(col).lower()
        series = X[col]
        n_unique = series.nunique()

        name_match = any(kw in col_lower for kw in ("_id", "index", "_key", "row_num", "record_id"))
        is_monotonic = False
        if series.dtype in (np.int32, np.int64, np.float64):
            try:
                is_monotonic = bool(series.is_monotonic_increasing or series.is_monotonic_decreasing)
            except Exception:
                pass
        high_cardinality = n_unique > 0.9 * n_rows and n_rows > 20

        if name_match or (is_monotonic and high_cardinality):
            id_columns.append(str(col))

    # Near-perfect single-feature separability via univariate AUC (classification only)
    perfect_separators = []
    n_unique_y = len(set(y)) if hasattr(y, "__len__") else 0
    if 2 <= n_unique_y <= 20:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        check_cols = [c for c in X.columns if str(c) not in id_columns][:30]
        classes = sorted(set(y))
        is_binary = len(classes) == 2
        for col in check_cols:
            try:
                vals = X[col]
                if not np.issubdtype(vals.dtype, np.number):
                    continue
                if vals.nunique() < 2:
                    continue
                if is_binary:
                    auc = roc_auc_score(y, vals)
                    auc = max(auc, 1.0 - auc)  # direction-agnostic
                else:
                    y_bin = label_binarize(y, classes=classes)
                    auc = roc_auc_score(
                        y_bin, np.column_stack([vals] * len(classes)), multi_class="ovr", average="macro"
                    )
                if auc > 0.995:
                    perfect_separators.append(str(col))
            except Exception:
                pass

    return {
        "duplicate_rate": round(dup_rate, 4),
        "n_exact_duplicates": n_exact_dups,
        "n_near_duplicates": max(n_near_dups, 0),
        "id_columns": id_columns,
        "perfect_separators": perfect_separators,
    }


def _build_permutation_histogram(real_score, perm_scores, p_value, scoring, baseline=None):
    """Build Plotly spec for permutation test null distribution histogram."""
    signal_prob = 1.0 - p_value
    label = scoring.replace("_", " ").title()

    import numpy as np

    counts, _ = np.histogram(perm_scores, bins="auto")
    max_count = int(counts.max()) if len(counts) else len(perm_scores) // 3

    traces = [
        {
            "type": "histogram",
            "x": perm_scores,
            "marker": {
                "color": "rgba(128,128,128,0.45)",
                "line": {"color": "rgba(128,128,128,0.7)", "width": 1},
            },
            "name": "Permuted (null)",
            "nbinsx": min(len(perm_scores), 20),
            "hovertemplate": "%{x:.3f}<extra>permuted</extra>",
        },
        {
            "type": "scatter",
            "x": [real_score, real_score],
            "y": [0, max_count * 1.1],
            "mode": "lines",
            "line": {"color": "#4a9f6e", "width": 3},
            "name": f"Your model ({real_score:.3f})",
            "hoverinfo": "skip",
        },
    ]

    # Baseline line (prevalence for PR-AUC, chance for balanced_accuracy, 0 for R²)
    if baseline is not None:
        traces.append(
            {
                "type": "scatter",
                "x": [baseline, baseline],
                "y": [0, max_count * 1.1],
                "mode": "lines",
                "line": {"color": "#d4a24a", "width": 2, "dash": "dash"},
                "name": f"Random baseline ({baseline:.3f})",
                "hoverinfo": "skip",
            }
        )

    annotations = [
        {
            "x": real_score,
            "y": max_count * 0.9,
            "text": f"p = {p_value:.3f}",
            "showarrow": True,
            "arrowhead": 2,
            "arrowcolor": "#4a9f6e",
            "font": {"color": "#4a9f6e", "size": 11, "family": "monospace"},
            "ax": 40 if real_score > np.mean(perm_scores) else -40,
            "ay": -20,
        }
    ]

    return {
        "data": traces,
        "layout": {
            "height": 220,
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "margin": {"t": 45, "b": 40, "l": 45, "r": 20},
            "title": {
                "text": f"Permutation Reality Test — p = {p_value:.3f} (1 − p = {signal_prob:.0%})",
                "font": {"size": 12, "color": "#b0b0b0"},
            },
            "xaxis": {
                "title": label,
                "gridcolor": "rgba(128,128,128,0.1)",
            },
            "yaxis": {
                "title": "Count",
                "gridcolor": "rgba(128,128,128,0.1)",
            },
            "annotations": annotations,
            "showlegend": True,
            "legend": {"orientation": "h", "y": 1.15, "font": {"size": 10}},
        },
    }


def _bayesian_model_beliefs(metrics, X, y, importances, task, *, model=None, cv_std=None):
    """Compute calibrated Bayesian beliefs about model trustworthiness.

    Each concern is mapped to a probability via sigmoid. Overall model
    confidence is computed via weighted log-linear fusion (same pattern as
    PBS MultiStreamHealth). Returns deterministic narrative and Plotly gauge.

    Returns dict with:
        model_confidence: float (0-1)
        beliefs: list of {concern, probability, narrative, evidence}
        narrative: str
        gauge_plot: dict (Plotly spec)
    """
    import math
    from collections import Counter

    import numpy as np

    beliefs = []
    n_rows, n_cols = X.shape

    if task == "classification":
        acc = metrics.get("accuracy", 0)
        balanced_acc = metrics.get("balanced_accuracy", acc)
        metrics.get("f1_macro", metrics.get("f1", 0))

        counts = Counter(y)
        majority_pct = max(counts.values()) / len(y) if len(y) > 0 else 0.5
        baseline = majority_pct
        lift = acc - baseline

        # Class imbalance
        p = _concern_sigmoid(majority_pct, center=0.80, steepness=15)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "class_imbalance",
                    "probability": round(p, 3),
                    "narrative": (
                        f"{majority_pct:.0%} of data is the majority class. "
                        f"Balanced accuracy ({balanced_acc:.3f}) is more reliable than standard accuracy ({acc:.3f})."
                    ),
                    "evidence": {"majority_pct": round(majority_pct, 4), "balanced_accuracy": round(balanced_acc, 4)},
                }
            )

        # Accuracy illusion (accuracy looks good but balanced accuracy tells a different story)
        gap = acc - balanced_acc
        p = _concern_sigmoid(gap, center=0.10, steepness=25)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "accuracy_illusion",
                    "probability": round(p, 3),
                    "narrative": (
                        f"Accuracy ({acc:.3f}) minus balanced accuracy ({balanced_acc:.3f}) = {gap:.3f} gap. "
                        f"The model performs well on the majority class but poorly on minority classes."
                    ),
                    "evidence": {
                        "accuracy": round(acc, 4),
                        "balanced_accuracy": round(balanced_acc, 4),
                        "gap": round(gap, 4),
                    },
                }
            )

        # Leakage
        p = _concern_sigmoid(acc, center=0.97, steepness=50)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "leakage",
                    "probability": round(p, 3),
                    "narrative": (
                        f"Accuracy is {acc:.3f}. At this level, check whether any feature is derived from "
                        f"or strongly correlated with the target."
                    ),
                    "evidence": {"accuracy": round(acc, 4)},
                }
            )

        # Not learning (small lift over baseline)
        p = _concern_sigmoid(lift, center=0.05, steepness=-40)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "not_learning",
                    "probability": round(p, 3),
                    "narrative": (
                        f"Accuracy ({acc:.3f}) is only {lift:+.3f} above the majority baseline ({baseline:.3f}). "
                        f"The remaining {1 - acc:.3f} error is {'consistent with noise.' if lift < 0.02 else 'modest lift.'}"
                    ),
                    "evidence": {"accuracy": round(acc, 4), "baseline": round(baseline, 4), "lift": round(lift, 4)},
                }
            )

        # Minority blindness
        per_class = metrics.get("per_class", {})
        if per_class:
            min_recall = min((v.get("recall", 1.0) for v in per_class.values()), default=1.0)
            p = _concern_sigmoid(1 - min_recall, center=0.50, steepness=8)
            if p > 0.05:
                beliefs.append(
                    {
                        "concern": "minority_blindness",
                        "probability": round(p, 3),
                        "narrative": (
                            f"Worst per-class recall is {min_recall:.3f}. "
                            f"The model misses {1 - min_recall:.0%} of instances in its weakest class."
                        ),
                        "evidence": {"min_class_recall": round(min_recall, 4)},
                    }
                )

    else:  # regression
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)

        # Leakage
        p = _concern_sigmoid(r2, center=0.97, steepness=50)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "leakage",
                    "probability": round(p, 3),
                    "narrative": (
                        f"R\u00b2 = {r2:.4f}. Near-perfect fit \u2014 check for target-derived features or data leakage."
                    ),
                    "evidence": {"r2": round(r2, 4)},
                }
            )

        # Not learning
        p = _concern_sigmoid(r2, center=0.15, steepness=-15)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "not_learning",
                    "probability": round(p, 3),
                    "narrative": (
                        f"R\u00b2 = {r2:.4f} \u2014 the model explains {r2:.0%} of variance. "
                        f"The remaining {1 - r2:.0%} is unexplained."
                    ),
                    "evidence": {"r2": round(r2, 4)},
                }
            )

        # Imprecision
        y_range = float(np.ptp(y)) if len(y) > 0 else 1.0
        rmse_frac = rmse / y_range if y_range > 0 else 0
        p = _concern_sigmoid(rmse_frac, center=0.35, steepness=8)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "imprecision",
                    "probability": round(p, 3),
                    "narrative": (
                        f"RMSE ({rmse:.4f}) is {rmse_frac:.0%} of the target range ({y_range:.4g}). "
                        f"Predictions deviate substantially from actuals."
                    ),
                    "evidence": {
                        "rmse": round(rmse, 4),
                        "target_range": round(y_range, 4),
                        "rmse_fraction": round(rmse_frac, 4),
                    },
                }
            )

        # Bias
        y_std = float(np.std(y)) if len(y) > 0 else 1.0
        # bias check requires y_pred which we don't have here — skip if not available in metrics
        mean_resid = metrics.get("_mean_residual")
        if mean_resid is not None and y_std > 0:
            bias_frac = abs(mean_resid) / y_std
            p = _concern_sigmoid(bias_frac, center=0.10, steepness=15)
            if p > 0.05:
                direction = "over" if mean_resid < 0 else "under"
                beliefs.append(
                    {
                        "concern": "bias",
                        "probability": round(p, 3),
                        "narrative": (
                            f"Model systematically {direction}-predicts (mean residual: {mean_resid:.3f}, "
                            f"{bias_frac:.0%} of target std)."
                        ),
                        "evidence": {"mean_residual": round(mean_resid, 4), "bias_fraction": round(bias_frac, 4)},
                    }
                )

    # ── Data-level beliefs (both tasks) ──

    # Overfit risk (feature/row ratio)
    ratio = n_cols / n_rows if n_rows > 0 else 0
    p = _concern_sigmoid(ratio, center=0.20, steepness=12)
    if p > 0.05:
        beliefs.append(
            {
                "concern": "overfit_risk",
                "probability": round(p, 3),
                "narrative": (
                    f"{n_cols} features for {n_rows} rows (ratio {ratio:.3f}). "
                    f"{'High risk of memorizing noise.' if ratio > 0.3 else 'Moderate overfitting risk.'}"
                ),
                "evidence": {"n_features": n_cols, "n_rows": n_rows, "ratio": round(ratio, 4)},
            }
        )

    # Small sample
    log_n = -math.log10(max(n_rows, 1))
    log_threshold = -math.log10(100)  # center at 100 rows
    p = _concern_sigmoid(log_n, center=log_threshold, steepness=5)
    if p > 0.05:
        beliefs.append(
            {
                "concern": "small_sample",
                "probability": round(p, 3),
                "narrative": f"{n_rows} rows. {'Metrics are unreliable.' if n_rows < 50 else 'Cross-validation variance will be elevated.'}",
                "evidence": {"n_rows": n_rows},
            }
        )

    # Collinearity
    try:
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] >= 2:
            corr = numeric_X.corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_corr = float(corr.max().max())
            p = _concern_sigmoid(max_corr, center=0.85, steepness=15)
            if p > 0.05:
                pair = corr.stack().idxmax()
                beliefs.append(
                    {
                        "concern": "collinearity",
                        "probability": round(p, 3),
                        "narrative": (
                            f"Correlation of {max_corr:.2f} between '{pair[0]}' and '{pair[1]}'. One may be redundant."
                        ),
                        "evidence": {
                            "max_correlation": round(max_corr, 4),
                            "feature_a": str(pair[0]),
                            "feature_b": str(pair[1]),
                        },
                    }
                )
    except Exception:
        pass

    # Single-feature dominance
    if importances and len(importances) >= 3:
        top_imp = importances[0].get("importance", 0)
        total_imp = sum(f.get("importance", 0) for f in importances)
        frac = top_imp / total_imp if total_imp > 0 else 0
        p = _concern_sigmoid(frac, center=0.60, steepness=8)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "single_feature",
                    "probability": round(p, 3),
                    "narrative": (
                        f"'{importances[0]['feature']}' accounts for {frac:.0%} of total importance. "
                        f"Model is essentially a single-feature predictor."
                    ),
                    "evidence": {"top_feature": importances[0]["feature"], "importance_fraction": round(frac, 4)},
                }
            )

    # ── Permutation reality test (empirical, not heuristic) ──
    permutation_result = None
    permutation_plot = None
    if model is not None:
        try:
            permutation_result = _permutation_reality_test(model, X, y, task)
            p_val = permutation_result["p_value"]
            if p_val > 0.05:
                beliefs.append(
                    {
                        "concern": "random_signal",
                        "probability": round(p_val, 3),
                        "narrative": (
                            f"Permutation p-value = {p_val:.3f} (risk proxy: {p_val:.0%} chance observed "
                            f"performance is random luck). "
                            f"{'Indistinguishable from shuffled labels.' if p_val > 0.10 else 'Marginal — real signal not confirmed.'} "
                            f"[{permutation_result['n_permutations']} permutations, {permutation_result['scoring']}]"
                        ),
                        "evidence": {
                            "p_value": round(p_val, 4),
                            "real_score": round(permutation_result["real_score"], 4),
                            "n_permutations": permutation_result["n_permutations"],
                            "scoring": permutation_result["scoring"],
                        },
                    }
                )
            permutation_plot = _build_permutation_histogram(
                permutation_result["real_score"],
                permutation_result["perm_scores"],
                p_val,
                permutation_result["scoring"],
                baseline=permutation_result.get("baseline"),
            )
        except Exception:
            pass

    # ── Duplicate contamination audit ──
    try:
        dup_result = _duplicate_audit(X, y)
        dup_rate = dup_result["duplicate_rate"]
        id_cols = dup_result["id_columns"]
        separators = dup_result["perfect_separators"]
        dup_concern = max(
            _concern_sigmoid(dup_rate, center=0.10, steepness=20),
            0.8 if id_cols else 0.0,
            0.9 if separators else 0.0,
        )
        if dup_concern > 0.05:
            parts = []
            n_exact = dup_result["n_exact_duplicates"]
            n_near = dup_result["n_near_duplicates"]
            if n_exact > 0:
                parts.append(f"{n_exact} exact duplicate rows")
            if n_near > 0:
                parts.append(f"{n_near} near-duplicate rows (within rounding)")
            if n_exact > 0 or n_near > 0:
                parts.append(f"({dup_rate:.1%} total)")
            if id_cols:
                parts.append(f"ID-like columns: {', '.join(id_cols)}")
            if separators:
                parts.append(f"Near-perfect separators (AUC > 0.995): {', '.join(separators)}")
            beliefs.append(
                {
                    "concern": "duplicate_contamination",
                    "probability": round(dup_concern, 3),
                    "narrative": ". ".join(parts) + ". These may inflate metrics if they span train/test.",
                    "evidence": {
                        "duplicate_rate": round(dup_rate, 4),
                        "n_exact_duplicates": n_exact,
                        "n_near_duplicates": n_near,
                        "id_columns": id_cols,
                        "perfect_separators": separators,
                    },
                }
            )
    except Exception:
        pass

    # ── Stability belief (from CV std, full pipeline only) ──
    if cv_std is not None and cv_std > 0:
        cv_mean = metrics.get("accuracy", metrics.get("r2", 0.5))
        cov = cv_std / max(abs(cv_mean), 0.01)
        p = _concern_sigmoid(cov, center=0.10, steepness=15)
        if p > 0.05:
            beliefs.append(
                {
                    "concern": "unstable_performance",
                    "probability": round(p, 3),
                    "narrative": (
                        f"Cross-validation std = {cv_std:.4f} (CoV = {cov:.2f}). "
                        f"{'Performance varies significantly across folds.' if cov > 0.15 else 'Moderate fold-to-fold variation.'}"
                    ),
                    "evidence": {"cv_std": round(cv_std, 4), "cv_cov": round(cov, 4)},
                }
            )

    # ── Weighted log-linear fusion ──
    active = [(b, _CONCERN_WEIGHTS.get(b["concern"], 0.5)) for b in beliefs if b["probability"] > 0.1]
    if active:
        total_weight = sum(w for _, w in active)
        log_trust = sum(w * math.log(1.0 - min(b["probability"], 0.95)) for b, w in active) / total_weight
        model_confidence = math.exp(log_trust)
    else:
        model_confidence = 0.95

    model_confidence = round(max(0.01, min(0.99, model_confidence)), 3)

    # ── Deterministic narrative ──
    narrative = f"Model Confidence: {model_confidence:.0%}\n"
    if task == "classification":
        acc = metrics.get("accuracy", 0)
        narrative += f"\nAccuracy: {acc:.3f}"
        majority_pct = max(Counter(y).values()) / len(y) if len(y) > 0 else 0.5
        if majority_pct > 0.70:
            narrative += f" \u2014 {majority_pct:.0%} of data is one class."
            balanced_acc = metrics.get("balanced_accuracy", acc)
            narrative += f"\nBalanced accuracy: {balanced_acc:.3f}"
            if balanced_acc < 0.55:
                narrative += " \u2014 barely above chance (0.500)."
            narrative += "\n"
    else:
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)
        narrative += f"\nR\u00b2: {r2:.4f} \u2014 explains {r2:.0%} of variance, {1 - r2:.0%} unexplained."
        narrative += f"\nRMSE: {rmse:.4f}\n"

    if beliefs:
        sorted_beliefs = sorted(beliefs, key=lambda b: b["probability"], reverse=True)
        active_beliefs = [b for b in sorted_beliefs if b["probability"] > 0.15]
        if active_beliefs:
            narrative += "\nConcerns:\n"
            for b in active_beliefs:
                narrative += f"  P({b['concern'].replace('_', ' ')}) = {b['probability']:.2f} \u2014 {b['narrative']}\n"
    else:
        narrative += "\nNo significant concerns detected."

    # ── Gauge plot (PBS pattern) ──
    if model_confidence >= 0.70:
        gauge_color = "#4a9f6e"
    elif model_confidence >= 0.40:
        gauge_color = "#d4a24a"
    else:
        gauge_color = "#d94a4a"

    gauge_plot = {
        "data": [
            {
                "type": "indicator",
                "mode": "gauge+number",
                "value": round(model_confidence * 100, 1),
                "number": {"suffix": "%", "font": {"size": 28, "color": gauge_color}},
                "title": {"text": "Model Confidence", "font": {"size": 13, "color": "#b0b0b0"}},
                "gauge": {
                    "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": "#666"}},
                    "bar": {"color": gauge_color, "thickness": 0.75},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(217,74,74,0.12)"},
                        {"range": [40, 70], "color": "rgba(212,162,74,0.12)"},
                        {"range": [70, 100], "color": "rgba(74,159,110,0.12)"},
                    ],
                },
            }
        ],
        "layout": {
            "height": 180,
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
            "margin": {"t": 50, "b": 10, "l": 30, "r": 30},
        },
    }

    return {
        "model_confidence": model_confidence,
        "beliefs": beliefs,
        "narrative": narrative,
        "gauge_plot": gauge_plot,
        "permutation_plot": permutation_plot,
    }


def _auto_train(X, y, task=None):
    """Auto-detect task type and train the best available model.

    Returns (model, metrics_dict, feature_importances_list, task,
             X_test, y_test, y_pred) — last 3 for diagnostic plots.
    """
    from collections import Counter

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split

    # Auto-detect task type
    if task is None:
        is_cat = y.dtype == object or y.dtype.name == "category" or hasattr(y.dtype, "categories")
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
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
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

        # MCC — best single metric under imbalance (all classification)
        try:
            metrics["mcc"] = round(matthews_corrcoef(y_test, y_pred), 4)
        except Exception:
            pass

        # Binary: add average_precision (PR AUC) and Brier score
        n_classes = y.nunique()
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["average_precision"] = round(average_precision_score(y_test, y_proba), 4)
                metrics["brier_score"] = round(brier_score_loss(y_test, y_proba), 4)
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
        _regression_reliability(y, y_test, y_pred, metrics)

    # Feature importance
    importances = []
    if hasattr(model, "feature_importances_"):
        for i, col in enumerate(X.columns):
            importances.append(
                {
                    "feature": col,
                    "importance": round(float(model.feature_importances_[i]), 4),
                }
            )
        importances.sort(key=lambda x: x["importance"], reverse=True)

    return model, metrics, importances, task, X_test, y_test, y_pred


def _claude_interpret_results(user, context, metrics, importances, task=None, warnings=None):
    """Ask Claude to interpret ML results in plain English."""
    from agents_api.llm_manager import LLMManager

    top_features = importances[:5] if importances else []
    prompt = f"""Context: {context}
Task: {task or "auto-detected"}
Metrics: {json.dumps(metrics)}
Top features: {json.dumps(top_features)}"""

    if warnings:
        critical = [w for w in warnings if w.get("level") in ("critical", "high")]
        if critical:
            prompt += f"\nReliability warnings: {json.dumps(critical)}"

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
        if av < 0.2:
            return "negligible", False
        if av < 0.5:
            return "small", False
        if av < 0.8:
            return "medium", True
        return "large", True
    elif effect_type == "eta_squared":
        if av < 0.01:
            return "negligible", False
        if av < 0.06:
            return "small", False
        if av < 0.14:
            return "medium", True
        return "large", True
    elif effect_type == "cramers_v":
        if av < 0.1:
            return "negligible", False
        if av < 0.3:
            return "small", False
        if av < 0.5:
            return "medium", True
        return "large", True
    elif effect_type == "r_squared":
        if av < 0.02:
            return "negligible", False
        if av < 0.13:
            return "small", False
        if av < 0.26:
            return "medium", True
        return "large", True
    return "unknown", False


# ---------------------------------------------------------------------------
# Assumption checking helpers (for automatic diagnostics)
# ---------------------------------------------------------------------------


def _check_normality(data, label="Data", alpha=0.05):
    """Check normality of data. Returns a diagnostic dict or None."""
    from scipy import stats as sp_stats

    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 8:
        return None  # too few for meaningful test
    try:
        if len(data) <= 5000:
            stat, p = sp_stats.shapiro(data[:5000])
            test_name = "Shapiro-Wilk"
        else:
            stat, p = sp_stats.normaltest(data)
            test_name = "D'Agostino-Pearson"
    except Exception:
        return None

    if p < alpha:
        return {
            "level": "warning",
            "title": f"Normality: {label} departs from normal ({test_name} p = {p:.4f})",
            "detail": "This test assumes normally distributed data. Results may be unreliable for non-normal data.",
            "_p": p,
            "_test": test_name,
        }
    return None  # passes — no warning needed


def _check_equal_variance(*groups, labels=None, alpha=0.05):
    """Check equal variances via Levene's test. Returns a diagnostic dict or None."""
    from scipy import stats as sp_stats

    groups = [np.asarray(g)[~np.isnan(np.asarray(g))] for g in groups]
    if any(len(g) < 3 for g in groups):
        return None
    try:
        stat, p = sp_stats.levene(*groups)
    except Exception:
        return None

    if p < alpha:
        labels_str = " vs ".join(labels) if labels else "groups"
        return {
            "level": "warning",
            "title": f"Unequal Variances (Levene p = {p:.4f})",
            "detail": f"Variances differ between {labels_str}. Using Welch's correction is recommended.",
            "_p": p,
        }
    return None


def _check_outliers(data, label="Data", method="iqr"):
    """Flag outliers via IQR method. Returns diagnostic dict or None."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 10:
        return None
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = int(np.sum((data < lower) | (data > upper)))
    pct = n_outliers / len(data) * 100
    if n_outliers > 0 and pct > 1:
        return {
            "level": "warning" if pct < 5 else "error",
            "title": f"Outliers: {n_outliers} values ({pct:.1f}%) in {label} exceed 1.5\u00d7IQR",
            "detail": f"Range [{lower:.4f}, {upper:.4f}]. Outliers can inflate variance and distort means.",
            "_n_outliers": n_outliers,
            "_pct": pct,
        }
    return None


def _cross_validate(primary_p, alt_p, primary_name, alt_name, alpha=0.05, normality_failed=False):
    """Compare two test p-values and return agreement/contradiction diagnostic."""
    primary_sig = primary_p < alpha
    alt_sig = alt_p < alpha
    if primary_sig != alt_sig:
        explanation = ""
        if normality_failed:
            explanation = " Non-normality may be affecting the parametric test."
        elif abs(primary_p - alpha) < 0.02 or abs(alt_p - alpha) < 0.02:
            explanation = " One p-value is near the threshold \u2014 borderline result."
        return {
            "level": "contradiction",
            "title": f"Contradiction: {primary_name} and {alt_name} disagree",
            "detail": (
                f"{primary_name} p = {primary_p:.4f} ({'significant' if primary_sig else 'not significant'}), "
                f"{alt_name} p = {alt_p:.4f} ({'significant' if alt_sig else 'not significant'}).{explanation}"
            ),
        }
    else:
        return {
            "level": "info",
            "title": f"Cross-check: {alt_name} agrees (p = {alt_p:.4f})",
            "detail": "Both parametric and non-parametric tests reach the same conclusion.",
        }


def _narrative(verdict, body, next_steps=None, chart_guidance=None):
    """Build HTML narrative block for charts-first output.

    Returns HTML string rendered in the .dsw-narrative div (prose, not monospace).
    """
    parts = [f'<div class="dsw-verdict">{verdict}</div>', f"<p>{body}</p>"]
    if chart_guidance:
        parts.append(f"<p><strong>In the chart:</strong> {chart_guidance}</p>")
    if next_steps:
        parts.append(f'<div class="dsw-next"><strong>Next &rarr;</strong> {next_steps}</div>')
    return "\n".join(parts)


def _practical_block(effect_name, effect_val, effect_type, pval, alpha=0.05, context=""):
    """Build practical significance interpretation block for analysis summaries."""
    label, meaningful = _effect_magnitude(effect_val, effect_type)

    b = f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
    b += "<<COLOR:title>>PRACTICAL SIGNIFICANCE<</COLOR>>\n"
    b += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"
    b += f"<<COLOR:highlight>>{effect_name}:<</COLOR>> {abs(effect_val):.3f} ({label} effect)\n\n"

    if pval < alpha and meaningful:
        b += "<<COLOR:good>>Both statistically and practically significant.<</COLOR>>\n"
        b += f"<<COLOR:text>>The difference is real and large enough to act on.{' ' + context if context else ''}<</COLOR>>"
    elif pval < alpha and label == "small":
        b += "<<COLOR:warn>>Statistically significant but small effect.<</COLOR>>\n"
        b += f"<<COLOR:text>>The difference is real but may be too small to justify action. Consider whether the cost of change is worth this magnitude.{' ' + context if context else ''}<</COLOR>>"
    elif pval < alpha:
        b += "<<COLOR:warn>>Statistically significant but negligible effect.<</COLOR>>\n"
        b += "<<COLOR:text>>With enough data, even trivial differences reach significance. This difference is too small to act on.<</COLOR>>"
    elif pval >= alpha and meaningful:
        b += f"<<COLOR:warn>>Not statistically significant, but the effect size is {label}.<</COLOR>>\n"
        b += "<<COLOR:text>>This may indicate insufficient sample size rather than no real effect. Consider collecting more data before concluding there is no difference.<</COLOR>>"
    else:
        b += f"<<COLOR:text>>Not statistically significant, and the effect is {label}.<</COLOR>>\n"
        b += "<<COLOR:text>>No evidence of a meaningful difference.<</COLOR>>"

    return b


# ---------------------------------------------------------------------------
# Bayesian Insurance — shadow computation + evidence grading
# ---------------------------------------------------------------------------


def _bayesian_shadow(shadow_type, **kwargs):
    """Compute Bayesian shadow for a frequentist test.

    Returns dict with bf10, bf_label, credible_interval, interpretation, shadow_type.
    Returns None if computation fails.

    All math extracted from bayesian.py — operates on pre-extracted arrays,
    not column names, so it works regardless of input format (2-column or factor).
    """
    from scipy import stats as sp_stats
    from scipy.integrate import quad

    try:
        bf10 = None
        ci_dict = None
        interp_parts = []

        if shadow_type in ("ttest_1samp", "ttest_2samp", "ttest_paired"):
            # JZS Bayes Factor (Rouder et al. 2009)
            r_scale = 0.707  # √2/2, standard JZS default

            if shadow_type == "ttest_1samp":
                x = np.asarray(kwargs["x"], dtype=float)
                x = x[~np.isnan(x)]
                mu = float(kwargs.get("mu", 0))
                if len(x) < 3:
                    return None
                t_stat, _ = sp_stats.ttest_1samp(x, mu)
                n_eff = len(x)
                v = len(x) - 1
                d = (np.mean(x) - mu) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0
                se_d = np.sqrt(1 / len(x) + d**2 / (2 * len(x)))

            elif shadow_type == "ttest_2samp":
                x = np.asarray(kwargs["x"], dtype=float)
                y = np.asarray(kwargs["y"], dtype=float)
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                if len(x) < 2 or len(y) < 2:
                    return None
                t_stat, _ = sp_stats.ttest_ind(x, y)
                n1, n2 = len(x), len(y)
                n_eff = n1 * n2 / (n1 + n2)
                v = n1 + n2 - 2
                pooled_std = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / v)
                d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
                se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

            else:  # ttest_paired
                x = np.asarray(kwargs["x"], dtype=float)
                y = np.asarray(kwargs["y"], dtype=float)
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                if len(x) < 3:
                    return None
                diff = x - y
                t_stat, _ = sp_stats.ttest_rel(x, y)
                n_eff = len(diff)
                v = len(diff) - 1
                d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
                se_d = np.sqrt(1 / len(diff) + d**2 / (2 * len(diff)))

            # JZS integrand (Rouder et al. 2009, Eq. 2)
            def _jzs_integrand(g):
                nrg = n_eff * r_scale**2 * g
                return (
                    (1 + nrg) ** (-0.5)
                    * (1 + t_stat**2 / ((1 + nrg) * v)) ** (-(v + 1) / 2)
                    / (1 + t_stat**2 / v) ** (-(v + 1) / 2)
                    * (2 * np.pi) ** (-0.5)
                    * g ** (-1.5)
                    * np.exp(-1 / (2 * g))
                )

            bf10, _ = quad(_jzs_integrand, 1e-10, np.inf)
            bf10 = max(bf10, 1e-10)

            # 95% credible interval on Cohen's d
            z95 = 1.96
            ci_dict = {
                "param": "Cohen's d",
                "estimate": round(float(d), 4),
                "ci_low": round(float(d - z95 * se_d), 4),
                "ci_high": round(float(d + z95 * se_d), 4),
                "level": 0.95,
            }
            ci_excludes_zero = ci_dict["ci_low"] > 0 or ci_dict["ci_high"] < 0
            interp_parts.append(
                f"95% CrI for d: [{ci_dict['ci_low']:.3f}, {ci_dict['ci_high']:.3f}]"
                + (" (excludes zero)" if ci_excludes_zero else " (includes zero)")
            )

        elif shadow_type == "anova":
            # BIC-approximated BF (Wagenmakers 2007)
            groups = [np.asarray(g, dtype=float) for g in kwargs["groups"]]
            groups = [g[~np.isnan(g)] for g in groups]
            if len(groups) < 2 or any(len(g) < 2 for g in groups):
                return None
            all_data = np.concatenate(groups)
            n_total = len(all_data)
            k = len(groups)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
            ss_total = ss_between + ss_within
            if ss_within <= 0 or n_total <= k:
                return None
            bic_h0 = n_total * np.log(ss_total / n_total) + 1 * np.log(n_total)
            bic_h1 = n_total * np.log(ss_within / n_total) + k * np.log(n_total)
            bf10 = np.exp((bic_h0 - bic_h1) / 2)
            bf10 = min(float(bf10), 1e10)

        elif shadow_type == "correlation":
            # BF via integral under uniform prior on ρ (Ly et al. 2016)
            x = np.asarray(kwargs["x"], dtype=float)
            y = np.asarray(kwargs["y"], dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]
            n = len(x)
            if n < 4:
                return None
            r, _ = sp_stats.pearsonr(x, y)

            def _corr_bf_integrand(rho):
                if abs(rho) >= 1:
                    return 0.0
                log_term = ((n - 2) / 2) * np.log(1 - rho**2) - ((n - 1) / 2) * np.log(1 - r * rho)
                return np.exp(log_term)

            bf_integral, _ = quad(_corr_bf_integrand, -1 + 1e-10, 1 - 1e-10)
            bf10 = bf_integral / 2.0 if bf_integral > 0 else 1e-10

            # Fisher z CI on r
            z_r = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
            se_z = 1 / np.sqrt(n - 3)
            z95 = 1.96
            r_low = np.tanh(z_r - z95 * se_z)
            r_high = np.tanh(z_r + z95 * se_z)
            ci_dict = {
                "param": "r",
                "estimate": round(float(r), 4),
                "ci_low": round(float(r_low), 4),
                "ci_high": round(float(r_high), 4),
                "level": 0.95,
            }
            interp_parts.append(f"95% CrI for r: [{r_low:.3f}, {r_high:.3f}]")

        elif shadow_type == "proportion":
            # Savage-Dickey BF under Beta(1,1) prior
            successes = int(kwargs["x"])
            n = int(kwargs["n"])
            p0 = float(kwargs.get("p0", 0.5))
            if n < 1:
                return None
            a_post = 1 + successes
            b_post = 1 + n - successes
            posterior_at_p0 = sp_stats.beta.pdf(p0, a_post, b_post)
            bf10 = 1.0 / posterior_at_p0 if posterior_at_p0 > 0 else 1e10
            # Beta posterior CI
            ci_low = float(sp_stats.beta.ppf(0.025, a_post, b_post))
            ci_high = float(sp_stats.beta.ppf(0.975, a_post, b_post))
            p_hat = successes / n
            ci_dict = {
                "param": "proportion",
                "estimate": round(p_hat, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "level": 0.95,
            }
            interp_parts.append(f"95% CrI for proportion: [{ci_low:.3f}, {ci_high:.3f}]")

        elif shadow_type == "chi2":
            # BIC-approximated BF (Wagenmakers 2007)
            chi2_stat = float(kwargs["chi2_stat"])
            dof = int(kwargs["dof"])
            n_obs = int(kwargs["n_obs"])
            if n_obs < 2 or dof < 1:
                return None
            log_bf10 = 0.5 * (chi2_stat - dof * np.log(n_obs))
            bf10 = float(np.exp(np.clip(log_bf10, -500, 500)))

        elif shadow_type == "regression":
            # BIC-approximated BF for regression (null vs full model)
            r_squared = float(kwargs["r_squared"])
            n_obs = int(kwargs["n_obs"])
            k_predictors = int(kwargs["k_predictors"])
            if n_obs <= k_predictors + 1 or r_squared < 0:
                return None
            ss_total = float(kwargs.get("ss_total", 1.0))
            ss_res = ss_total * (1 - r_squared)
            if ss_res <= 0:
                ss_res = 1e-10
            bic_h0 = n_obs * np.log(ss_total / n_obs) + 1 * np.log(n_obs)
            bic_h1 = n_obs * np.log(ss_res / n_obs) + (k_predictors + 1) * np.log(n_obs)
            bf10 = float(np.exp(np.clip((bic_h0 - bic_h1) / 2, -500, 500)))
            ci_dict = {
                "param": "R²",
                "estimate": round(r_squared, 4),
                "ci_low": None,
                "ci_high": None,
                "level": 0.95,
            }
            interp_parts.append(f"R² = {r_squared:.3f} with {k_predictors} predictors")

        elif shadow_type == "variance":
            # BIC-based BF for F-test (variance ratio)
            f_stat = float(kwargs["f_stat"])
            df1 = int(kwargs["df1"])
            df2 = int(kwargs["df2"])
            n_obs = int(kwargs.get("n_obs", df1 + df2 + 2))
            if df1 < 1 or df2 < 1:
                return None
            # Chi2 approximation: F ~ chi2/df ratio
            chi2_approx = f_stat * df1
            log_bf10 = 0.5 * (chi2_approx - df1 * np.log(n_obs))
            bf10 = float(np.exp(np.clip(log_bf10, -500, 500)))
            interp_parts.append(f"F({df1},{df2}) = {f_stat:.2f}")

        elif shadow_type == "nonparametric":
            # Effect-size conversion BF for nonparametric tests
            # Converts r or eta² to approximate BF via normal approx
            effect_r = float(kwargs.get("effect_r", 0))
            n_obs = int(kwargs["n_obs"])
            if n_obs < 4:
                return None
            # Fisher z-transform for r
            z_r = 0.5 * np.log((1 + abs(effect_r)) / (1 - min(abs(effect_r), 0.999)))
            se_z = 1 / np.sqrt(n_obs - 3) if n_obs > 3 else 1.0
            # Approximate BF from z-score (Wetzels & Wagenmakers 2012)
            z_score = z_r / se_z
            # BF10 ≈ exp(z²/2) / sqrt(n) for moderate effects
            log_bf10 = 0.5 * z_score**2 - 0.5 * np.log(n_obs)
            bf10 = float(np.exp(np.clip(log_bf10, -500, 500)))
            ci_dict = {
                "param": "effect r",
                "estimate": round(effect_r, 4),
                "ci_low": round(float(np.tanh(z_r - 1.96 * se_z)), 4),
                "ci_high": round(float(np.tanh(z_r + 1.96 * se_z)), 4),
                "level": 0.95,
            }
            interp_parts.append(f"95% CrI for r: [{ci_dict['ci_low']:.3f}, {ci_dict['ci_high']:.3f}]")

        else:
            return None

        if bf10 is None:
            return None

        bf10 = float(bf10)

        # BF interpretation label
        if bf10 > 100:
            bf_label = "extreme"
        elif bf10 > 30:
            bf_label = "very strong"
        elif bf10 > 10:
            bf_label = "strong"
        elif bf10 > 3:
            bf_label = "moderate"
        elif bf10 > 1:
            bf_label = "weak"
        elif bf10 > 1 / 3:
            bf_label = "weak (for H\u2080)"
        elif bf10 > 1 / 10:
            bf_label = "moderate (for H\u2080)"
        else:
            bf_label = "strong (for H\u2080)"

        # Build interpretation string
        if bf10 >= 1:
            interp = f"BF\u2081\u2080 = {bf10:.1f} \u2014 the data are {bf10:.1f}\u00d7 more likely under H\u2081 than H\u2080 ({bf_label} evidence)."
        else:
            bf01 = 1 / bf10 if bf10 > 0 else float("inf")
            interp = (
                f"BF\u2081\u2080 = {bf10:.2f} (BF\u2080\u2081 = {bf01:.1f}) \u2014 the data favor H\u2080 ({bf_label})."
            )
        if interp_parts:
            interp += " " + ". ".join(interp_parts) + "."

        return {
            "shadow_type": shadow_type,
            "bf10": round(bf10, 4),
            "bf_label": bf_label,
            "credible_interval": ci_dict,
            "interpretation": interp,
        }

    except Exception:
        return None


def _evidence_grade(p_value, bf10=None, effect_magnitude=None, cross_val_agrees=None):
    """Synthesize evidence grade from available metrics.

    Returns dict: {"grade": "Strong", "rationale": "...", "components": [...]}
    Returns None if p_value is None.
    """
    if p_value is None:
        return None
    try:
        score = 0
        components = []

        # P-value contribution
        if p_value < 0.001:
            score += 3
            components.append("p < 0.001")
        elif p_value < 0.01:
            score += 2
            components.append(f"p = {p_value:.4f}")
        elif p_value < 0.05:
            score += 1
            components.append(f"p = {p_value:.3f}")
        else:
            components.append(f"p = {p_value:.3f} (n.s.)")

        # BF contribution
        if bf10 is not None:
            if bf10 > 10:
                score += 3
                components.append(f"BF\u2081\u2080 = {bf10:.1f}")
            elif bf10 > 3:
                score += 2
                components.append(f"BF\u2081\u2080 = {bf10:.1f}")
            elif bf10 > 1:
                score += 1
                components.append(f"BF\u2081\u2080 = {bf10:.1f}")
            else:
                components.append(f"BF\u2081\u2080 = {bf10:.2f}")

        # Effect magnitude contribution
        if effect_magnitude is not None:
            if effect_magnitude == "large":
                score += 2
                components.append("large effect")
            elif effect_magnitude == "medium":
                score += 1
                components.append("medium effect")
            else:
                components.append(f"{effect_magnitude} effect")

        # Cross-validation contribution
        if cross_val_agrees is True:
            score += 1
            components.append("cross-check agrees")
        elif cross_val_agrees is False:
            score -= 1
            components.append("cross-check disagrees")

        # Grade mapping
        if score >= 8:
            grade = "Strong"
        elif score >= 5:
            grade = "Moderate"
        elif score >= 2:
            grade = "Weak"
        else:
            grade = "Inconclusive"

        return {
            "grade": grade,
            "rationale": ", ".join(components),
            "components": components,
        }
    except Exception:
        return None


def _ml_interpretation(task, metrics, y_test=None, y_pred=None, features=None, target=None, model=None):
    """Build practical interpretation block for ML models."""
    import numpy as np

    b = f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
    b += "<<COLOR:title>>MODEL ASSESSMENT<</COLOR>>\n"
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
            b += "<<COLOR:text>>Consider: more features, better features, or more data. The model is not learning meaningful patterns.<</COLOR>>"
        elif lift < 0.10:
            b += f"<<COLOR:warn>>Model is modestly better than baseline ({lift:+.1%}).<</COLOR>>\n"
            b += "<<COLOR:text>>Useful for screening but not for high-stakes decisions. Consider feature engineering or more data.<</COLOR>>"
        elif acc >= 0.95:
            b += "<<COLOR:good>>Excellent performance.<</COLOR>>\n"
            b += "<<COLOR:text>>Verify this isn't due to data leakage (a feature that perfectly predicts the target because it's derived from it).<</COLOR>>"
        elif acc >= 0.85:
            b += "<<COLOR:good>>Strong model — suitable for decision support.<</COLOR>>\n"
            b += "<<COLOR:text>>Review the confusion matrix to check if errors are concentrated in specific classes.<</COLOR>>"
        else:
            b += "<<COLOR:text>>Moderate performance.<</COLOR>>\n"
            b += f"<<COLOR:text>>The model learned real patterns ({lift:+.1%} over baseline). Consider whether this accuracy is sufficient for your use case.<</COLOR>>"

        if majority_pct > 0.8:
            minority = class_counts.most_common()[-1]
            b += f"\n\n<<COLOR:warn>>Class imbalance detected:<</COLOR>> '{minority[0]}' has only {minority[1]} samples ({minority[1] / len(y_test):.1%}).\n"
            b += "<<COLOR:text>>Accuracy may be misleading. Check precision and recall for the minority class — F1 score is more informative here.<</COLOR>>"

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
                b += "<<COLOR:danger>>Model explains very little variation.<</COLOR>>\n"
                b += f"<<COLOR:text>>The features selected may not predict '{target}'. Try different features or check data quality.<</COLOR>>"
            elif r2 < 0.5:
                b += "<<COLOR:warn>>Moderate fit — predictions will have substantial error.<</COLOR>>\n"
                b += f"<<COLOR:text>>On average, predictions are off by {rmse:.2f} ({rmse_pct:.0f}% of the data range). Useful for trend identification but not precision control.<</COLOR>>"
            elif r2 < 0.8:
                b += "<<COLOR:text>>Good fit — the model captures most of the pattern.<</COLOR>>\n"
                b += f"<<COLOR:text>>Predictions are off by +-{rmse:.2f} on average. Suitable for forecasting and decision support.<</COLOR>>"
            else:
                b += "<<COLOR:good>>Strong fit.<</COLOR>>\n"
                b += f"<<COLOR:text>>Model explains {r2 * 100:.0f}% of variation. Predictions are off by +-{rmse:.2f}. Check for data leakage if R² > 0.95.<</COLOR>>"
        else:
            r2_label, _ = _effect_magnitude(r2, "r_squared")
            b += f"\n<<COLOR:text>>R² is {r2_label}. Model explains {r2 * 100:.0f}% of variation in '{target}'.<</COLOR>>"

    return b


def _fit_best_distribution(data):
    """Fit multiple distributions and return the best by KS p-value.
    Returns (dist_name, scipy_dist, fit_args, ks_pvalue)."""
    import numpy as np
    from scipy import stats as sp_stats

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
