"""Autopilot ML Pipeline views.

Three autopilot modes plus retrain:
1. Clean+Train: Triage clean → auto-train → diagnostics
2. Full Pipeline: Triage → Compare → Best → Explain → Tune
3. Augment+Train: Forge synthetic rows → merge → train
4. Retrain: Replay a saved training recipe on new data

All require paid tier (@gated_paid).
"""

import json
import logging
import time
import uuid

import numpy as np
import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid, require_auth
from .dsw.common import (
    _auto_train,
    _build_ml_diagnostics,
    _clean_for_ml,
    _create_ml_evidence,
    cache_model,
    save_model_to_disk,
)
from .ml_pipeline import forge_augment_df, train_with_recipe, triage_clean_df
from .models import DSWResult, SavedModel

logger = logging.getLogger(__name__)


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return super().default(obj)


def _json_response(data, status=200):
    """JsonResponse that safely handles numpy types."""
    return HttpResponse(
        json.dumps(data, cls=_NumpySafeEncoder),
        content_type="application/json",
        status=status,
    )


def _compute_feature_info(df, feature_names):
    """Compute feature ranges/categories for the Prediction Profiler."""
    info = {}
    for col in feature_names:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            info[col] = {
                "type": "numeric",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            info[col] = {
                "type": "categorical",
                "categories": [str(c) for c in df[col].dropna().unique().tolist()[:50]],
            }
    return info


def _compute_feature_stats(X_train, feature_names, feature_info):
    """Compute training data distribution stats for optimization feasibility checks.

    Stores means, stds, and covariance matrix for numeric features.
    This enables Mahalanobis distance (how far is a point from training data?)
    and correlation-aware optimization constraints.
    """
    numeric_cols = [f for f in feature_names
                    if f in X_train.columns and feature_info.get(f, {}).get("type") == "numeric"]

    if len(numeric_cols) < 2:
        return {}

    X_num = X_train[numeric_cols].dropna()
    if len(X_num) < 10:
        return {}

    means = X_num.mean().to_dict()
    stds = X_num.std().to_dict()

    # Covariance matrix for Mahalanobis distance
    cov = X_num.cov()
    # Store as nested dict (JSON-serializable)
    cov_dict = {c: {r: float(cov.loc[r, c]) for r in numeric_cols} for c in numeric_cols}

    # Feature correlations (for detecting implausible combos)
    corr = X_num.corr()
    # Store strong correlations (|r| > 0.5) as warnings
    strong_corrs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            r = float(corr.loc[c1, c2])
            if abs(r) > 0.5:
                strong_corrs.append({"f1": c1, "f2": c2, "r": round(r, 3)})

    return {
        "numeric_features": numeric_cols,
        "means": {k: round(float(v), 6) for k, v in means.items()},
        "stds": {k: round(float(v), 6) for k, v in stds.items()},
        "covariance": cov_dict,
        "strong_correlations": strong_corrs,
        "n_train": len(X_num),
    }


def _build_training_interpretation(task, metrics, model_type, y_test, y_pred,
                                    top_features, target, original_shape, clean_shape):
    """Build a plain-language interpretation of ML training results."""
    lines = []

    # Reliability header — prepend when HIGH warnings exist
    warnings = metrics.get("reliability_warnings", [])
    high_warnings = [w for w in warnings if w.get("level") == "high"]
    if high_warnings:
        lines.append("⚠ Reliability: LOW — metrics likely inflated by imbalance, split issues, or leakage.")
        lines.append("")

    if task == "classification":
        acc = metrics.get("accuracy", 0)
        bal_acc = metrics.get("balanced_accuracy")
        f1_macro = metrics.get("f1_macro")

        # Baseline comparison
        baseline_acc = metrics.get("baseline_accuracy")
        if baseline_acc is None:
            from collections import Counter
            class_counts = Counter(y_test)
            baseline_acc = max(class_counts.values()) / len(y_test) if len(y_test) > 0 else 0
        else:
            from collections import Counter
            class_counts = Counter(y_test)

        majority_pct = baseline_acc
        lift = acc - majority_pct
        n_classes = len(class_counts)

        lines.append(f"Your {model_type} predicts '{target}' with {acc:.1%} accuracy ({n_classes} classes).")
        lines.append(f"Baseline (always guess majority class): {majority_pct:.1%}. Model lift: {lift:+.1%}.")

        # Balanced accuracy context when imbalanced
        if bal_acc is not None and majority_pct > 0.70:
            lines.append(f"Balanced accuracy (mean per-class recall): {bal_acc:.1%} — this is the more honest metric for imbalanced data.")
        if f1_macro is not None and majority_pct > 0.70:
            lines.append(f"F1 macro: {f1_macro:.1%} (punishes minority neglect).")

        if lift < 0.02:
            lines.append("⚠ Model is barely better than guessing. Consider different features or more data.")
        elif acc >= 0.99 and majority_pct > 0.80:
            lines.append("⚠ Perfect scores on imbalanced data usually mean the model ignores the minority class, or the dataset contains target-derived features.")
        elif acc >= 0.95:
            lines.append("✓ Excellent. Verify no data leakage (feature derived from target).")
        elif acc >= 0.85:
            lines.append("✓ Strong model — suitable for decision support.")
        elif acc >= 0.70:
            lines.append("Moderate performance. Useful for screening, not for high-stakes automated decisions.")
        else:
            lines.append("Limited performance. The target may be hard to predict with these features.")

        # Class imbalance — minority class recall
        if majority_pct > 0.8:
            minority = class_counts.most_common()[-1]
            lines.append(f"⚠ Class imbalance: '{minority[0]}' is only {minority[1]/len(y_test):.0%} of the data.")
            per_class = metrics.get("per_class", {})
            for cls_key, cls_m in per_class.items():
                # Find minority classes
                try:
                    cls_count = class_counts.get(int(cls_key), class_counts.get(cls_key, 0))
                except (ValueError, TypeError):
                    cls_count = class_counts.get(cls_key, 0)
                if cls_count / len(y_test) < 0.20:
                    r = cls_m.get("recall", 0)
                    lines.append(f"  → Class '{cls_key}' recall: {r:.0%}" + (" ⚠ model misses most of this class" if r < 0.5 else ""))

    elif task == "regression":
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)

        y_range = float(np.ptp(y_test)) if len(y_test) > 0 else 1
        y_mean = float(np.mean(y_test)) if len(y_test) > 0 else 0
        rmse_pct = rmse / y_range * 100 if y_range > 0 else 0

        lines.append(f"Your {model_type} predicts '{target}' with R²={r2:.3f} (explains {r2*100:.0f}% of variation).")
        lines.append(f"Average prediction error: ±{rmse:.4f} ({rmse_pct:.0f}% of data range).")

        if r2 >= 0.8:
            lines.append("✓ Strong model — suitable for forecasting and process optimization.")
        elif r2 >= 0.5:
            lines.append("Moderate fit. Useful for identifying trends, but predictions have notable uncertainty.")
        elif r2 >= 0.2:
            lines.append("Weak fit. The model captures some patterns but misses most variation. Try adding more features.")
        else:
            lines.append("⚠ Very weak fit. These features may not meaningfully predict the target.")

    # Data quality impact
    orig_rows, clean_rows = original_shape[0], clean_shape[0]
    if orig_rows > clean_rows:
        removed = orig_rows - clean_rows
        pct = removed / orig_rows * 100
        if pct > 20:
            lines.append(f"⚠ Triage removed {removed} rows ({pct:.0f}%). This may bias the model if removed rows differ systematically.")
        else:
            lines.append(f"Triage cleaned {removed} rows ({pct:.0f}% of data).")

    # Top features
    if top_features:
        feat_names = [f[0] if isinstance(f, (list, tuple)) else str(f) for f in top_features[:3]]
        lines.append(f"Top drivers: {', '.join(feat_names)}.")

    # Next steps
    lines.append("")
    lines.append("Next steps:")
    lines.append("• Use the Profiler to explore how each feature affects predictions")
    lines.append("• Upload new data to test predictions on unseen samples")
    if task == "classification" and metrics.get("accuracy", 0) < 0.85:
        lines.append("• Try Full Pipeline mode to compare multiple algorithms")
    if task == "regression" and metrics.get("r2", 0) < 0.7:
        lines.append("• Consider adding interaction features or trying Full Pipeline mode")

    return "\n".join(lines)


def _build_retrain_interpretation(task, metrics, old_metrics, comparison, model_type, target):
    """Build interpretation for retrained model comparing to previous version."""
    lines = []

    if task == "classification":
        old_acc = old_metrics.get("accuracy", 0)
        new_acc = metrics.get("accuracy", 0)
        delta = new_acc - old_acc
        lines.append(f"Retrained {model_type} for '{target}': accuracy {old_acc:.1%} → {new_acc:.1%} ({delta:+.1%}).")
        if delta > 0.02:
            lines.append("✓ Model improved. The new data helped.")
        elif delta > -0.02:
            lines.append("Model performance is stable. The new data is consistent with previous training data.")
        else:
            lines.append("⚠ Model degraded. Possible causes: data distribution shift, data quality issues, or the new data introduces new patterns the model hasn't seen.")
    else:
        old_r2 = old_metrics.get("r2", 0)
        new_r2 = metrics.get("r2", 0)
        delta = new_r2 - old_r2
        lines.append(f"Retrained {model_type} for '{target}': R² {old_r2:.3f} → {new_r2:.3f} ({delta:+.3f}).")
        if delta > 0.02:
            lines.append("✓ Model improved with new data.")
        elif delta > -0.02:
            lines.append("Performance stable — model generalizes well to new data.")
        else:
            lines.append("⚠ Model degraded. Check if the new data has different characteristics.")

    # Metric-by-metric changes
    notable = []
    for k, v in comparison.items():
        if abs(v.get("change", 0)) > 0.01:
            direction = "↑" if v["change"] > 0 else "↓"
            notable.append(f"{k}: {v['previous']:.4f} → {v['current']:.4f} {direction}")
    if notable:
        lines.append("Changes: " + ", ".join(notable))

    return "\n".join(lines)


def _compute_subgroup_diagnostics(X_test, y_test, y_pred, feature_info, task):
    """Slice test set by categorical features and report per-segment metrics."""
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
    results = []
    if not feature_info:
        return results

    overall_metric = None
    metric_name = None
    if task == "classification":
        metric_name = "accuracy"
        overall_metric = accuracy_score(y_test, y_pred)
    else:
        metric_name = "r2"
        overall_metric = r2_score(y_test, y_pred) if len(y_test) > 1 else 0

    for col, info in feature_info.items():
        if info.get("type") != "categorical" or col not in X_test.columns:
            continue
        cats = info.get("categories", [])
        if not cats or len(cats) > 10:
            continue

        sorted_cats = sorted(cats)
        segments = []
        for code, cat_name in enumerate(sorted_cats):
            mask = X_test[col] == code
            n = int(mask.sum())
            if n < 5:
                continue
            yt = y_test[mask]
            yp = y_pred[mask] if hasattr(y_pred, '__getitem__') else np.array(y_pred)[mask]
            if task == "classification":
                val = float(accuracy_score(yt, yp))
            else:
                val = float(r2_score(yt, yp)) if n > 1 else 0.0
            flag = "warning" if overall_metric and val < overall_metric * 0.85 else "ok"
            segments.append({
                "value": str(cat_name), "n": n,
                "metric": metric_name, "score": round(val, 4), "flag": flag,
            })

        if segments:
            results.append({
                "feature": col,
                "overall": round(overall_metric, 4) if overall_metric else 0,
                "metric": metric_name,
                "segments": segments,
            })

    return results


def _compute_threshold_analysis(y_test, model, X_test, class_names=None):
    """Sweep classification thresholds and compute metrics at each."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    if not hasattr(model, "predict_proba"):
        return None

    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        return None

    n_classes = y_prob.shape[1]
    if n_classes != 2:
        return None  # Binary only

    pos_probs = y_prob[:, 1]
    y_true = np.array(y_test)

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95
    rows = []
    for t in thresholds:
        y_hat = (pos_probs >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
        })

    # Find optimal thresholds
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])
    # Youden's J = sensitivity + specificity - 1 = recall + (tn/(tn+fp)) - 1
    best_youden = max(rows, key=lambda r: r["recall"] + (r["tn"] / (r["tn"] + r["fp"]) if (r["tn"] + r["fp"]) > 0 else 0) - 1)

    names = class_names or ["0", "1"]
    return {
        "thresholds": rows,
        "optimal": {
            "f1": best_f1["threshold"],
            "accuracy": best_acc["threshold"],
            "youden": best_youden["threshold"],
        },
        "class_names": [str(n) for n in names],
    }


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def autopilot_clean_train(request):
    """Clean data with Triage, then auto-train best model.

    POST /api/dsw/autopilot/clean-train/
    - file: CSV (multipart)
    - target: target column name
    - project_id: optional project UUID
    - triage_config: optional JSON cleaning config
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    target = request.POST.get("target", "").strip()
    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    project_id = request.POST.get("project_id")
    triage_config = {}
    try:
        triage_config = json.loads(request.POST.get("triage_config", "{}"))
    except json.JSONDecodeError:
        pass

    try:
        start = time.time()
        df = pd.read_csv(request.FILES["file"])

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found'}, status=400)

        if len(df) < 10:
            return JsonResponse({"error": "Dataset too small — need at least 10 rows"}, status=400)

        original_shape = list(df.shape)
        steps = []

        # Step 1: Triage clean
        df_clean, cleaning_summary = triage_clean_df(df, triage_config)
        steps.append({
            "name": "Triage Clean",
            "status": "completed",
            "summary": cleaning_summary,
        })

        # Step 2: Train model with recipe
        model, metrics, importances, task, X_test, y_test, y_pred, recipe = \
            train_with_recipe(df_clean, target)

        steps.append({
            "name": "Auto-Train",
            "status": "completed",
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
        })

        # Step 3: Diagnostic plots
        feature_names = recipe["features"]
        label_map = recipe.get("label_map")
        diag_plots = _build_ml_diagnostics(
            model, X_test, y_test, y_pred, feature_names, task,
            label_map=label_map, model_name=type(model).__name__,
        )

        steps.append({
            "name": "Diagnostics",
            "status": "completed",
            "plot_count": len(diag_plots),
        })

        # Save model
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        data_lineage = {
            "source_type": "upload",
            "original_file": request.FILES["file"].name,
            "original_shape": original_shape,
            "cleaned_shape": list(df_clean.shape),
            "triage_applied": True,
            "triage_config": triage_config,
            "pipeline": "autopilot_clean_train",
        }

        recipe["feature_info"] = _compute_feature_info(df_clean, feature_names)
        recipe["feature_stats"] = _compute_feature_stats(df_clean, feature_names, recipe["feature_info"])

        # Store threshold analysis in recipe for profiler access
        if task == "classification":
            label_map_ct = recipe.get("label_map", {})
            inv_map_ct = {int(v): k for k, v in label_map_ct.items()} if label_map_ct else None
            class_names_ct = [inv_map_ct[i] for i in sorted(inv_map_ct)] if inv_map_ct else None
            ta_ct = _compute_threshold_analysis(y_test, model, X_test, class_names_ct)
            if ta_ct:
                recipe["threshold_analysis"] = ta_ct

        model_key = str(uuid.uuid4())
        cache_model(request.user.id, model_key, model, {
            "model_type": type(model).__name__,
            "metrics": metrics,
            "features": feature_names,
            "target": target,
            "training_config": recipe,
            "data_lineage": data_lineage,
        })

        saved_model = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=type(model).__name__,
            dsw_result_id=result_id,
            name=f"Autopilot: {target} ({type(model).__name__})",
            metrics=metrics,
            features=feature_names,
            target=target,
            project_id=project_id,
            training_config=recipe,
            data_lineage=data_lineage,
        )

        # Synara evidence
        if project_id:
            _create_ml_evidence(
                request.user, project_id, type(model).__name__,
                metrics, importances, task, target,
            )

        elapsed = time.time() - start

        primary_metric = metrics.get("accuracy") or metrics.get("r2", "N/A")
        metric_name = "Accuracy" if task == "classification" else "R²"

        result_data = {
            "pipeline": "clean_train",
            "steps": steps,
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "feature_importance": importances[:10],
            "plots": diag_plots,
            "model_key": model_key,
            "training_config": recipe,
            "data_lineage": data_lineage,
            "elapsed_seconds": round(elapsed, 2),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="autopilot_clean_train",
            data=json.dumps(result_data, cls=_NumpySafeEncoder),
        )

        response = {
            "result_id": result_id,
            "pipeline_stages": [
                {"name": s["name"], "success": s.get("status") == "completed"}
                for s in steps
            ],
            "cleaning": cleaning_summary,
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "plots": diag_plots,
            "recipe": recipe,
            "saved_model_id": str(saved_model.id) if saved_model else None,
            "interpretation": _build_training_interpretation(
                task, metrics, type(model).__name__, y_test, y_pred,
                importances[:5] if importances else [], target,
                original_shape, list(df_clean.shape),
            ),
            "subgroup_diagnostics": _compute_subgroup_diagnostics(
                X_test, y_test, y_pred, recipe.get("feature_info", {}), task,
            ),
        }

        if task == "classification":
            label_map = recipe.get("label_map", {})
            inv_map = {int(v): k for k, v in label_map.items()} if label_map else None
            class_names = [inv_map[i] for i in sorted(inv_map)] if inv_map else None
            ta = _compute_threshold_analysis(y_test, model, X_test, class_names)
            if ta:
                response["threshold_analysis"] = ta

        request.user.increment_queries()
        return _json_response(response)

    except Exception as e:
        logger.error(f"Autopilot clean+train failed: {e}", exc_info=True)
        return _json_response({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def autopilot_full_pipeline(request):
    """Full ML pipeline: Clean → Compare → Best → Explain → Tune.

    POST /api/dsw/autopilot/full-pipeline/
    - file: CSV (multipart)
    - target: target column name
    - project_id: optional
    - triage_config: optional JSON
    - cv_folds: 3/5/10 (default 5)
    - n_trials: Optuna trials (default 30)
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    target = request.POST.get("target", "").strip()
    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    project_id = request.POST.get("project_id")
    cv_folds = int(request.POST.get("cv_folds", "5"))
    n_trials = min(int(request.POST.get("n_trials", "30")), 50)

    triage_config = {}
    try:
        triage_config = json.loads(request.POST.get("triage_config", "{}"))
    except json.JSONDecodeError:
        pass

    try:
        start = time.time()
        df = pd.read_csv(request.FILES["file"])

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found'}, status=400)

        if len(df) < 10:
            return JsonResponse({"error": "Dataset too small — need at least 10 rows"}, status=400)

        original_shape = list(df.shape)
        steps = []
        all_plots = []

        # Step 1: Triage clean
        df_clean, cleaning_summary = triage_clean_df(df, triage_config)
        steps.append({
            "name": "Triage Clean",
            "status": "completed",
            "summary": cleaning_summary,
        })

        # Step 2: Model comparison
        feature_names = [c for c in df_clean.columns if c != target]
        X, y, label_map = _clean_for_ml(df_clean, target)

        from sklearn.model_selection import cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Auto-detect task
        n_unique = len(set(y))
        task_type = "classification" if n_unique <= 20 and n_unique / len(y) < 0.05 else "regression"

        if task_type == "classification":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB

            roster = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=42))]),
                "LDA": LinearDiscriminantAnalysis(),
                "GaussianNB": GaussianNB(),
            }
            scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
            primary_score = "accuracy"
        else:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import (
                BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge,
            )

            roster = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Linear": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "LASSO": Lasso(alpha=0.01),
                "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
                "BayesianRidge": BayesianRidge(),
            }
            scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}
            primary_score = "r2"

        # Try adding XGBoost/LightGBM
        try:
            import xgboost as xgb
            if task_type == "classification":
                roster["XGBoost"] = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
            else:
                roster["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            pass
        try:
            import lightgbm as lgb
            if task_type == "classification":
                roster["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
            else:
                roster["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        except ImportError:
            pass

        # Run cross-validation for each model
        comparison = []
        for name, model_obj in roster.items():
            try:
                cv_results = cross_validate(
                    model_obj, X, y, cv=min(cv_folds, len(X)),
                    scoring=scoring, return_train_score=True,
                )
                row = {"model": name}
                for metric_key in scoring:
                    row[f"test_{metric_key}_mean"] = float(cv_results[f"test_{metric_key}"].mean())
                    row[f"test_{metric_key}_std"] = float(cv_results[f"test_{metric_key}"].std())
                comparison.append(row)
            except Exception as e:
                logger.warning(f"Model {name} failed in comparison: {e}")

        # Find best model
        if not comparison:
            return JsonResponse({"error": "All models failed during comparison"}, status=500)

        best_row = max(comparison, key=lambda r: r.get(f"test_{primary_score}_mean", -999))
        best_name = best_row["model"]
        best_model_obj = roster[best_name]

        steps.append({
            "name": "Model Compare",
            "status": "completed",
            "models_tested": len(comparison),
            "best_model": best_name,
            "comparison": comparison,
        })

        # Step 3: Train best model on full train set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        best_model_obj.fit(X_train, y_train)
        y_pred = best_model_obj.predict(X_test)

        # Compute metrics
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
        else:
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }

        # Feature importance
        importances = []
        if hasattr(best_model_obj, "feature_importances_"):
            for i, imp in enumerate(best_model_obj.feature_importances_):
                importances.append({"feature": feature_names[i] if i < len(feature_names) else f"f{i}", "importance": float(imp)})
            importances.sort(key=lambda x: x["importance"], reverse=True)

        # Diagnostic plots
        diag_plots = _build_ml_diagnostics(
            best_model_obj, X_test, y_test, y_pred, feature_names, task_type,
            label_map=label_map, model_name=best_name,
        )
        all_plots.extend(diag_plots)

        # Step 4: SHAP explanation (if available)
        shap_plots = []
        try:
            import shap
            if hasattr(best_model_obj, "predict"):
                try:
                    explainer = shap.TreeExplainer(best_model_obj)
                except Exception:
                    X_bg = X_test[:100] if len(X_test) > 100 else X_test
                    explainer = shap.KernelExplainer(best_model_obj.predict, X_bg)

                shap_values = explainer.shap_values(X_test[:100])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

                # SHAP importance bar chart
                mean_abs = np.abs(shap_values).mean(axis=0)
                sorted_idx = np.argsort(mean_abs)[::-1][:15]
                shap_plots.append({
                    "title": "SHAP Feature Importance",
                    "data": [{
                        "type": "bar",
                        "x": [float(mean_abs[i]) for i in sorted_idx],
                        "y": [feature_names[i] if i < len(feature_names) else f"f{i}" for i in sorted_idx],
                        "orientation": "h",
                        "marker": {"color": "rgba(74, 159, 110, 0.7)"},
                    }],
                    "layout": {"yaxis": {"autorange": "reversed"}, "xaxis": {"title": "Mean |SHAP|"}},
                })

                all_plots.extend(shap_plots)

                steps.append({
                    "name": "SHAP Explain",
                    "status": "completed",
                    "plot_count": len(shap_plots),
                })
        except Exception as e:
            steps.append({
                "name": "SHAP Explain",
                "status": "skipped",
                "reason": str(e),
            })

        # Step 5: Optuna tuning
        tuning_result = None
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                if "RandomForest" in best_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 3, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    }
                    if task_type == "classification":
                        m = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                    else:
                        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                elif "XGBoost" in best_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    }
                    if task_type == "classification":
                        m = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
                    else:
                        m = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                elif "LightGBM" in best_name:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    }
                    if task_type == "classification":
                        m = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
                    else:
                        m = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
                else:
                    # For models without tunable params, return baseline
                    return best_row.get(f"test_{primary_score}_mean", 0)

                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(m, X, y, cv=min(cv_folds, len(X)), scoring=primary_score)
                return float(scores.mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=120)

            tuning_result = {
                "best_params": study.best_params,
                "best_score": float(study.best_value),
                "n_trials": len(study.trials),
            }

            # Retrain with best params if tuning improved
            if study.best_value > best_row.get(f"test_{primary_score}_mean", 0):
                if "RandomForest" in best_name:
                    if task_type == "classification":
                        best_model_obj = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
                    else:
                        best_model_obj = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
                elif "XGBoost" in best_name:
                    if task_type == "classification":
                        best_model_obj = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0)
                    else:
                        best_model_obj = xgb.XGBRegressor(**study.best_params, random_state=42, verbosity=0)
                elif "LightGBM" in best_name:
                    if task_type == "classification":
                        best_model_obj = lgb.LGBMClassifier(**study.best_params, random_state=42, verbosity=-1)
                    else:
                        best_model_obj = lgb.LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)

                best_model_obj.fit(X_train, y_train)
                y_pred = best_model_obj.predict(X_test)

                if task_type == "classification":
                    metrics = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                    }
                else:
                    metrics = {
                        "r2": float(r2_score(y_test, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    }

                tuning_result["tuned_metrics"] = metrics

            steps.append({
                "name": "Hyperparameter Tune",
                "status": "completed",
                "result": tuning_result,
            })

        except Exception as e:
            steps.append({
                "name": "Hyperparameter Tune",
                "status": "skipped",
                "reason": str(e),
            })

        # Save final model
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        # Convert label_map keys to strings for JSON serialization
        json_label_map = None
        if label_map:
            json_label_map = {str(k): int(v) if isinstance(v, (int, np.integer)) else v
                              for k, v in label_map.items()}

        training_config = {
            "features": feature_names,
            "target": target,
            "task_type": task_type,
            "model_class": type(best_model_obj).__name__,
            "hyperparams": best_model_obj.get_params() if hasattr(best_model_obj, "get_params") else {},
            "cv_folds": cv_folds,
            "comparison_models": [r["model"] for r in comparison],
            "tuning_trials": n_trials,
            "source": "autopilot_full_pipeline",
            "label_map": json_label_map,
        }
        fi_fp = _compute_feature_info(df_clean, feature_names)
        training_config["feature_info"] = fi_fp
        training_config["feature_stats"] = _compute_feature_stats(df_clean, feature_names, fi_fp)
        data_lineage = {
            "source_type": "upload",
            "original_file": request.FILES["file"].name,
            "original_shape": original_shape,
            "cleaned_shape": list(df_clean.shape),
            "triage_applied": True,
            "pipeline": "autopilot_full_pipeline",
        }

        # Store threshold analysis for profiler
        if task_type == "classification":
            inv_map_fp = {int(v): k for k, v in json_label_map.items()} if json_label_map else None
            cn_fp = [inv_map_fp[i] for i in sorted(inv_map_fp)] if inv_map_fp else None
            ta_fp = _compute_threshold_analysis(y_test, best_model_obj, X_test, cn_fp)
            if ta_fp:
                training_config["threshold_analysis"] = ta_fp

        model_key = str(uuid.uuid4())
        cache_model(request.user.id, model_key, best_model_obj, {
            "model_type": type(best_model_obj).__name__,
            "metrics": metrics,
            "features": feature_names,
            "target": target,
            "training_config": training_config,
            "data_lineage": data_lineage,
        })

        saved_model = save_model_to_disk(
            user=request.user,
            model=best_model_obj,
            model_type=type(best_model_obj).__name__,
            dsw_result_id=result_id,
            name=f"Pipeline: {target} ({type(best_model_obj).__name__})",
            metrics=metrics,
            features=feature_names,
            target=target,
            project_id=project_id,
            training_config=training_config,
            data_lineage=data_lineage,
        )

        if project_id:
            _create_ml_evidence(
                request.user, project_id, type(best_model_obj).__name__,
                metrics, importances, task_type, target,
            )

        elapsed = time.time() - start

        primary_metric = metrics.get("accuracy") or metrics.get("r2", "N/A")
        metric_name = "Accuracy" if task_type == "classification" else "R²"

        result_data = {
            "pipeline": "full_pipeline",
            "steps": steps,
            "model_type": type(best_model_obj).__name__,
            "task": task_type,
            "metrics": metrics,
            "feature_importance": importances[:10],
            "plots": all_plots,
            "model_key": model_key,
            "comparison": comparison,
            "tuning": tuning_result,
            "training_config": training_config,
            "data_lineage": data_lineage,
            "elapsed_seconds": round(elapsed, 2),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="autopilot_full_pipeline",
            data=json.dumps(result_data, cls=_NumpySafeEncoder),
        )

        response = {
            "result_id": result_id,
            "pipeline_stages": [
                {"name": s["name"], "success": s.get("status") == "completed"}
                for s in steps
            ],
            "cleaning": cleaning_summary,
            "comparison": comparison,
            "tuning": tuning_result,
            "model_type": type(best_model_obj).__name__,
            "task": task_type,
            "metrics": metrics,
            "shap_plots": [p for p in all_plots if "SHAP" in (p.get("title") or p.get("layout", {}).get("title", {}).get("text", "") if isinstance(p.get("layout", {}).get("title"), dict) else p.get("layout", {}).get("title", ""))],
            "plots": [p for p in all_plots if "SHAP" not in (p.get("title") or p.get("layout", {}).get("title", {}).get("text", "") if isinstance(p.get("layout", {}).get("title"), dict) else p.get("layout", {}).get("title", ""))],
            "recipe": training_config,
            "saved_model_id": str(saved_model.id) if saved_model else None,
            "interpretation": _build_training_interpretation(
                task_type, metrics, type(best_model_obj).__name__, y_test, y_pred,
                importances[:5] if importances else [], target,
                original_shape, list(df_clean.shape),
            ),
            "subgroup_diagnostics": _compute_subgroup_diagnostics(
                X_test, y_test, y_pred, training_config.get("feature_info", {}), task_type,
            ),
        }

        if task_type == "classification":
            inv_map = {int(v): k for k, v in json_label_map.items()} if json_label_map else None
            class_names = [inv_map[i] for i in sorted(inv_map)] if inv_map else None
            ta = _compute_threshold_analysis(y_test, best_model_obj, X_test, class_names)
            if ta:
                response["threshold_analysis"] = ta

        request.user.increment_queries()
        return _json_response(response)

    except Exception as e:
        logger.error(f"Autopilot full pipeline failed: {e}", exc_info=True)
        return _json_response({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def autopilot_augment_train(request):
    """Augment small dataset with Forge synthetic data, then train.

    POST /api/dsw/autopilot/augment-train/
    - file: CSV (multipart)
    - target: target column name
    - n_synthetic: rows to generate (default: match original size)
    - project_id: optional
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    target = request.POST.get("target", "").strip()
    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    project_id = request.POST.get("project_id")

    try:
        start = time.time()
        df = pd.read_csv(request.FILES["file"])

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found'}, status=400)

        if len(df) < 10:
            return JsonResponse({"error": "Dataset too small — need at least 10 rows"}, status=400)

        original_shape = list(df.shape)
        n_synthetic = int(request.POST.get("n_synthetic", str(len(df))))
        n_synthetic = max(10, min(n_synthetic, 10000))

        steps = []

        # Step 1: Forge augment
        augmented_df, forge_report = forge_augment_df(df, n_synthetic)
        steps.append({
            "name": "Forge Augment",
            "status": "completed",
            "report": forge_report,
        })

        # Step 2: Train on augmented data
        model, metrics, importances, task, X_test, y_test, y_pred, recipe = \
            train_with_recipe(augmented_df, target)

        steps.append({
            "name": "Auto-Train",
            "status": "completed",
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
        })

        # Step 3: Diagnostics
        feature_names = recipe["features"]
        label_map = recipe.get("label_map")
        diag_plots = _build_ml_diagnostics(
            model, X_test, y_test, y_pred, feature_names, task,
            label_map=label_map, model_name=type(model).__name__,
        )

        steps.append({
            "name": "Diagnostics",
            "status": "completed",
            "plot_count": len(diag_plots),
        })

        # Save model
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        data_lineage = {
            "source_type": "augmented",
            "original_file": request.FILES["file"].name,
            "original_shape": original_shape,
            "synthetic_rows": n_synthetic,
            "augmented_shape": list(augmented_df.shape),
            "forge_applied": True,
            "pipeline": "autopilot_augment_train",
        }

        recipe["source"] = "autopilot_augment_train"
        recipe["forge_config"] = {"n_synthetic": n_synthetic}
        recipe["feature_info"] = _compute_feature_info(augmented_df, feature_names)
        recipe["feature_stats"] = _compute_feature_stats(augmented_df, feature_names, recipe["feature_info"])

        if task == "classification":
            label_map_at = recipe.get("label_map", {})
            inv_map_at = {int(v): k for k, v in label_map_at.items()} if label_map_at else None
            cn_at = [inv_map_at[i] for i in sorted(inv_map_at)] if inv_map_at else None
            ta_at = _compute_threshold_analysis(y_test, model, X_test, cn_at)
            if ta_at:
                recipe["threshold_analysis"] = ta_at

        model_key = str(uuid.uuid4())
        cache_model(request.user.id, model_key, model, {
            "model_type": type(model).__name__,
            "metrics": metrics,
            "features": feature_names,
            "target": target,
            "training_config": recipe,
            "data_lineage": data_lineage,
        })

        saved_model = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=type(model).__name__,
            dsw_result_id=result_id,
            name=f"Augmented: {target} ({type(model).__name__})",
            metrics=metrics,
            features=feature_names,
            target=target,
            project_id=project_id,
            training_config=recipe,
            data_lineage=data_lineage,
        )

        if project_id:
            _create_ml_evidence(
                request.user, project_id, type(model).__name__,
                metrics, importances, task, target,
            )

        elapsed = time.time() - start

        primary_metric = metrics.get("accuracy") or metrics.get("r2", "N/A")
        metric_name = "Accuracy" if task == "classification" else "R²"

        result_data = {
            "pipeline": "augment_train",
            "steps": steps,
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "feature_importance": importances[:10],
            "plots": diag_plots,
            "model_key": model_key,
            "forge_report": forge_report,
            "training_config": recipe,
            "data_lineage": data_lineage,
            "elapsed_seconds": round(elapsed, 2),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="autopilot_augment_train",
            data=json.dumps(result_data, cls=_NumpySafeEncoder),
        )

        response = {
            "result_id": result_id,
            "pipeline_stages": [
                {"name": s["name"], "success": s.get("status") == "completed"}
                for s in steps
            ],
            "augmentation": forge_report,
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "plots": diag_plots,
            "recipe": recipe,
            "saved_model_id": str(saved_model.id) if saved_model else None,
            "interpretation": _build_training_interpretation(
                task, metrics, type(model).__name__, y_test, y_pred,
                importances[:5] if importances else [], target,
                original_shape, list(augmented_df.shape),
            ),
            "subgroup_diagnostics": _compute_subgroup_diagnostics(
                X_test, y_test, y_pred, recipe.get("feature_info", {}), task,
            ),
        }

        if task == "classification":
            label_map = recipe.get("label_map", {})
            inv_map = {int(v): k for k, v in label_map.items()} if label_map else None
            class_names = [inv_map[i] for i in sorted(inv_map)] if inv_map else None
            ta = _compute_threshold_analysis(y_test, model, X_test, class_names)
            if ta:
                response["threshold_analysis"] = ta

        request.user.increment_queries()
        return _json_response(response)

    except Exception as e:
        logger.error(f"Autopilot augment+train failed: {e}", exc_info=True)
        return _json_response({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def retrain_model(request, model_id):
    """Retrain an existing model using its stored recipe.

    POST /api/dsw/models/<uuid>/retrain/
    - file: optional new data CSV (uses stored data path if not provided)
    """
    try:
        saved = SavedModel.objects.get(id=model_id, user=request.user)
    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)

    recipe = saved.training_config
    if not recipe or not recipe.get("features"):
        return JsonResponse({"error": "No training recipe found for this model"}, status=400)

    try:
        import pickle
        from pathlib import Path

        start = time.time()

        # Load data — from upload or from stored path
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
            data_source = request.FILES["file"].name
        elif saved.data_lineage and saved.data_lineage.get("data_path"):
            data_path = Path(saved.data_lineage["data_path"])
            if not data_path.exists():
                return JsonResponse({"error": "Original training data no longer available. Upload new data."}, status=400)
            df = pd.read_csv(data_path)
            data_source = str(data_path)
        else:
            return JsonResponse({"error": "No data available. Upload a CSV file."}, status=400)

        target = recipe["target"]
        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found in data'}, status=400)

        # Re-apply triage if original recipe used it
        triage_applied = False
        if saved.data_lineage and saved.data_lineage.get("triage_applied"):
            triage_config = saved.data_lineage.get("triage_config", {})
            df, _ = triage_clean_df(df, triage_config)
            triage_applied = True

        # Re-apply forge augmentation if original recipe used it
        forge_applied = False
        if recipe.get("forge_config"):
            n_synthetic = recipe["forge_config"].get("n_synthetic", len(df))
            df, _ = forge_augment_df(df, n_synthetic)
            forge_applied = True

        # Train
        model, metrics, importances, task, X_test, y_test, y_pred, new_recipe = \
            train_with_recipe(df, target, config={"task_type": recipe.get("task_type")})

        # Diagnostic plots
        feature_names = new_recipe["features"]
        label_map = new_recipe.get("label_map")
        diag_plots = _build_ml_diagnostics(
            model, X_test, y_test, y_pred, feature_names, task,
            label_map=label_map, model_name=type(model).__name__,
        )

        # Save as new version
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        data_lineage = {
            "source_type": "retrain",
            "data_source": data_source,
            "original_model_id": str(saved.id),
            "triage_applied": triage_applied,
            "forge_applied": forge_applied,
            "data_shape": list(df.shape),
            "pipeline": "retrain",
        }

        new_recipe["source"] = "retrain"
        new_recipe["feature_info"] = _compute_feature_info(df, feature_names)
        new_recipe["feature_stats"] = _compute_feature_stats(df, feature_names, new_recipe["feature_info"])

        if task == "classification":
            label_map_rt = new_recipe.get("label_map", {})
            inv_map_rt = {int(v): k for k, v in label_map_rt.items()} if label_map_rt else None
            cn_rt = [inv_map_rt[i] for i in sorted(inv_map_rt)] if inv_map_rt else None
            ta_rt = _compute_threshold_analysis(y_test, model, X_test, cn_rt)
            if ta_rt:
                new_recipe["threshold_analysis"] = ta_rt

        project_id = saved.project_id or request.POST.get("project_id")

        model_key = str(uuid.uuid4())
        cache_model(request.user.id, model_key, model, {
            "model_type": type(model).__name__,
            "metrics": metrics,
            "features": feature_names,
            "target": target,
            "training_config": new_recipe,
            "data_lineage": data_lineage,
        })

        new_saved = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=type(model).__name__,
            dsw_result_id=result_id,
            name=saved.name,
            metrics=metrics,
            features=feature_names,
            target=target,
            project_id=str(project_id) if project_id else None,
            training_config=new_recipe,
            data_lineage=data_lineage,
            parent_model_id=str(saved.id),
        )

        if project_id:
            _create_ml_evidence(
                request.user, str(project_id), type(model).__name__,
                metrics, importances, task, target,
            )

        elapsed = time.time() - start

        # Compare with previous version
        old_metrics = json.loads(saved.metrics) if saved.metrics else {}
        comparison = {}
        for key in set(list(metrics.keys()) + list(old_metrics.keys())):
            old_val = old_metrics.get(key)
            new_val = metrics.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                comparison[key] = {
                    "previous": old_val,
                    "current": new_val,
                    "change": new_val - old_val,
                }

        result_data = {
            "pipeline": "retrain",
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "previous_metrics": old_metrics,
            "comparison": comparison,
            "feature_importance": importances[:10],
            "plots": diag_plots,
            "model_key": model_key,
            "version": new_saved.version if new_saved else saved.version + 1,
            "elapsed_seconds": round(elapsed, 2),
            "summary": (
                f"Retrained {saved.name} v{saved.version} → v{new_saved.version if new_saved else saved.version + 1}. "
                + " ".join(
                    f"{k}: {v['previous']:.4f} → {v['current']:.4f} ({'+' if v['change'] >= 0 else ''}{v['change']:.4f})"
                    for k, v in comparison.items()
                )
            ),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="retrain",
            data=json.dumps(result_data, cls=_NumpySafeEncoder),
        )

        response = {
            "result_id": result_id,
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "comparison": comparison,
            "plots": diag_plots,
            "version": new_saved.version if new_saved else saved.version + 1,
            "saved_model_id": str(new_saved.id) if new_saved else None,
            "interpretation": _build_retrain_interpretation(task, metrics, old_metrics, comparison, type(model).__name__, target),
            "subgroup_diagnostics": _compute_subgroup_diagnostics(
                X_test, y_test, y_pred, new_recipe.get("feature_info", {}), task,
            ),
        }

        if task == "classification":
            label_map = new_recipe.get("label_map", {})
            inv_map = {int(v): k for k, v in label_map.items()} if label_map else None
            class_names = [inv_map[i] for i in sorted(inv_map)] if inv_map else None
            ta = _compute_threshold_analysis(y_test, model, X_test, class_names)
            if ta:
                response["threshold_analysis"] = ta

        request.user.increment_queries()
        return _json_response(response)

    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return _json_response({"error": str(e)}, status=500)
