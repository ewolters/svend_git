"""DSW ML Lab endpoints -- from-intent, from-data, model management, scrub."""

import json
import logging
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, gated_paid, require_auth

from ..models import DSWResult, SavedModel
from .common import (
    _auto_train,
    _bayesian_model_beliefs,
    _build_ml_diagnostics,
    _claude_generate_schema,
    _clean_for_ml,
    _create_ml_evidence,
    _generate_data_from_schema,
    cache_model,
    get_cached_model,
    log_agent_action,
    sanitize_for_json,
    save_model_to_disk,
)

logger = logging.getLogger(__name__)


def _ml_connect_investigation(request, investigation_id, event_description, sample_size=None):
    """CANON-002 §12 — connect ML results to investigation graph."""
    from core.models import MeasurementSystem

    from ..investigation_bridge import InferenceSpec, connect_tool

    try:
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name="ML Model",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        spec = InferenceSpec(event_description=event_description, sample_size=sample_size)
        connect_tool(
            investigation_id=investigation_id,
            tool_output=tool_output,
            tool_type="ml",
            user=request.user,
            spec=spec,
        )
    except Exception:
        logger.exception("ML investigation bridge error")


def _read_csv_safe(file_or_path):
    """Read CSV with encoding fallback: UTF-8 → latin-1."""
    import io

    import pandas as pd

    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_or_path, encoding="latin-1")


@require_http_methods(["POST"])
@gated_paid
def dsw_from_intent(request):
    """Generate schema + synthetic data + model from natural language intent."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    intent = data.get("intent", "").strip()
    domain = data.get("domain")
    n_records = min(int(data.get("n_records", 500)), 5000)
    data.get("priority", "balanced")

    if not intent:
        return JsonResponse({"error": "Intent is required"}, status=400)

    try:
        schema = _claude_generate_schema(request.user, intent, domain, n_records)
        if not schema:
            return JsonResponse(
                {
                    "error": "Could not generate a dataset schema from your intent. Try rephrasing.",
                    "suggestion": "Be specific about what you want to predict and what factors might influence it.",
                },
                status=500,
            )

        df = _generate_data_from_schema(schema, n_records)
        target_name = schema.get("target", "target")
        feature_names = [c for c in df.columns if c != target_name]

        X, y, label_map = _clean_for_ml(df, target_name)
        model, metrics, importances, task, X_test, y_test, y_pred = _auto_train(X, y)

        diag_plots = _build_ml_diagnostics(
            model,
            X_test,
            y_test,
            y_pred,
            feature_names,
            task,
            label_map=label_map,
            model_name=type(model).__name__,
        )

        # Pop non-scalar entries before belief computation
        metrics.pop("reliability_warnings", None)
        metrics.pop("per_class", None)
        metrics.pop("class_balance", None)
        belief_result = _bayesian_model_beliefs(metrics, X, y, importances, task, model=model)

        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        data_dir = Path(settings.MEDIA_ROOT) / "dsw_data" / str(request.user.id)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{result_id}.csv"
        df.to_csv(data_path, index=False)

        training_config = {
            "features": feature_names,
            "target": target_name,
            "task_type": task,
            "model_class": type(model).__name__,
            "hyperparams": model.get_params() if hasattr(model, "get_params") else {},
            "source": "from_intent",
            "intent": intent,
            "n_records": n_records,
        }
        data_lineage_info = {
            "source_type": "synthetic",
            "intent": intent,
            "domain": domain or "auto-detected",
            "original_shape": list(df.shape),
            "data_path": str(data_path),
        }

        project_id = data.get("project_id")
        model_key = str(uuid.uuid4())
        cache_model(
            request.user.id,
            model_key,
            model,
            {
                "model_type": type(model).__name__,
                "metrics": metrics,
                "features": feature_names,
                "target": target_name,
                "training_config": training_config,
                "data_lineage": data_lineage_info,
            },
        )

        result_data = {
            "schema": {
                "name": schema.get("name", f"Schema for: {intent[:50]}"),
                "intent": intent,
                "domain": domain or "auto-detected",
                "features": feature_names,
                "target": target_name,
                "task": task,
            },
            "data_shape": list(df.shape),
            "data_path": str(data_path),
            "model_type": type(model).__name__,
            "metrics": metrics,
            "feature_importance": importances[:10],
            "pipeline": [
                "schema_generation",
                "data_synthesis",
                "data_cleaning",
                "model_training",
                "diagnostics",
                "belief_assessment",
            ],
            "warnings": [
                {
                    "level": "critical" if b["probability"] > 0.7 else "warning" if b["probability"] > 0.4 else "info",
                    "msg": b["narrative"],
                }
                for b in belief_result["beliefs"]
                if b["probability"] > 0.3
            ],
            "model_confidence": belief_result["model_confidence"],
            "beliefs": belief_result["beliefs"],
            "confidence_narrative": belief_result["narrative"],
            "confidence_gauge": belief_result["gauge_plot"],
            "permutation_plot": belief_result.get("permutation_plot"),
            "summary": (
                f"Generated {n_records} synthetic records based on intent: {intent}. "
                f"Trained {type(model).__name__} ({task}). "
                + (
                    f"Accuracy: {metrics.get('accuracy', 'N/A')}"
                    if task == "classification"
                    else f"R\u00b2: {metrics.get('r2', 'N/A')}"
                )
            ),
            "model_key": model_key,
            "plots": diag_plots,
        }

        DSWResult.objects.create(
            id=result_id, user=request.user, result_type="from_intent", data=json.dumps(result_data)
        )

        saved_model = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=type(model).__name__,
            dsw_result_id=result_id,
            name=f"Intent: {intent[:50]}",
            metrics=metrics,
            features=feature_names,
            target=target_name,
            project_id=project_id,
            training_config=training_config,
            data_lineage=data_lineage_info,
        )

        if project_id:
            _create_ml_evidence(request.user, project_id, type(model).__name__, metrics, importances, task, target_name)

        # CANON-002 §12 — investigation bridge (dual-write)
        investigation_id = data.get("investigation_id")
        if investigation_id:
            _ml_connect_investigation(
                request,
                investigation_id,
                f"ML model ({type(model).__name__}) for {target_name}: "
                + (
                    f"accuracy={metrics.get('accuracy', 'N/A')}"
                    if task == "classification"
                    else f"R²={metrics.get('r2', 'N/A')}"
                ),
                sample_size=n_records,
            )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent",
            success=True,
            metadata={"result_id": result_id, "task": task, "n_records": n_records},
        )

        response_data = {"result_id": result_id, **result_data}
        if saved_model:
            response_data["model_id"] = str(saved_model.id)
        return JsonResponse(sanitize_for_json(response_data))

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"DSW from_intent failed: {e}")
        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent",
            success=False,
            error_message=str(e)[:500],
            metadata={"intent": intent[:100], "domain": domain},
        )
        return JsonResponse(
            {
                "error": f"Pipeline failed: {str(e)[:200]}",
                "suggestion": "Try rephrasing your intent or use a different domain.",
            },
            status=500,
        )


@require_http_methods(["POST"])
@gated_paid
def dsw_from_data(request):
    """Auto-ML from uploaded data with Claude interpretation."""
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file = request.FILES["file"]
    target = request.POST.get("target", "").strip()
    intent = request.POST.get("intent", "")

    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    try:
        df = _read_csv_safe(file)

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found in data'}, status=400)
        if len(df) < 10:
            return JsonResponse({"error": "Dataset too small \u2014 need at least 10 rows."}, status=400)

        original_shape = list(df.shape)
        feature_names = [c for c in df.columns if c != target]
        X, y, label_map = _clean_for_ml(df, target)
        model, metrics, importances, task, X_test, y_test, y_pred = _auto_train(X, y)

        diag_plots = _build_ml_diagnostics(
            model,
            X_test,
            y_test,
            y_pred,
            feature_names,
            task,
            label_map=label_map,
            model_name=type(model).__name__,
        )

        # Pop non-scalar entries before belief computation
        metrics.pop("reliability_warnings", None)
        metrics.pop("per_class", None)
        metrics.pop("class_balance", None)
        belief_result = _bayesian_model_beliefs(metrics, X, y, importances, task, model=model)

        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        data_dir = Path(settings.MEDIA_ROOT) / "dsw_data" / str(request.user.id)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{result_id}.csv"
        df.to_csv(data_path, index=False)

        training_config = {
            "features": feature_names,
            "target": target,
            "task_type": task,
            "model_class": type(model).__name__,
            "hyperparams": model.get_params() if hasattr(model, "get_params") else {},
            "source": "from_data",
        }
        data_lineage_info = {
            "source_type": "upload",
            "original_file": file.name,
            "original_shape": original_shape,
            "cleaned_shape": [X.shape[0], X.shape[1]],
            "data_path": str(data_path),
        }

        project_id = request.POST.get("project_id")
        model_key = str(uuid.uuid4())
        cache_model(
            request.user.id,
            model_key,
            model,
            {
                "model_type": type(model).__name__,
                "metrics": metrics,
                "features": feature_names,
                "target": target,
                "training_config": training_config,
                "data_lineage": data_lineage_info,
            },
        )

        result_data = {
            "original_shape": original_shape,
            "cleaned_shape": [X.shape[0], X.shape[1]],
            "data_path": str(data_path),
            "model_type": type(model).__name__,
            "task": task,
            "metrics": metrics,
            "feature_importance": importances[:10],
            "pipeline": ["data_loading", "data_cleaning", "model_training", "diagnostics", "belief_assessment"],
            "warnings": [
                {
                    "level": "critical" if b["probability"] > 0.7 else "warning" if b["probability"] > 0.4 else "info",
                    "msg": b["narrative"],
                }
                for b in belief_result["beliefs"]
                if b["probability"] > 0.3
            ],
            "model_confidence": belief_result["model_confidence"],
            "beliefs": belief_result["beliefs"],
            "confidence_narrative": belief_result["narrative"],
            "confidence_gauge": belief_result["gauge_plot"],
            "permutation_plot": belief_result.get("permutation_plot"),
            "summary": (
                f"Processed {original_shape[0]} rows with {original_shape[1]} columns. Target: {target}. "
                f"Trained {type(model).__name__} ({task}). "
                + (
                    f"Accuracy: {metrics.get('accuracy', 'N/A')}"
                    if task == "classification"
                    else f"R²: {metrics.get('r2', 'N/A')}"
                )
            ),
            "model_key": model_key,
            "plots": diag_plots,
        }

        DSWResult.objects.create(id=result_id, user=request.user, result_type="from_data", data=json.dumps(result_data))

        saved_model = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=type(model).__name__,
            dsw_result_id=result_id,
            name=f"Data: {target} prediction",
            metrics=metrics,
            features=feature_names,
            target=target,
            project_id=project_id,
            training_config=training_config,
            data_lineage=data_lineage_info,
        )

        if project_id:
            _create_ml_evidence(request.user, project_id, type(model).__name__, metrics, importances, task, target)

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data",
            success=True,
            metadata={"result_id": result_id, "task": task, "rows": original_shape[0]},
        )

        response_data = {"result_id": result_id, **result_data}
        if saved_model:
            response_data["model_id"] = str(saved_model.id)

        # CANON-002 §12 — investigation bridge (dual-write)
        inv_id = request.POST.get("investigation_id", "")
        if inv_id:
            _ml_connect_investigation(
                request,
                inv_id,
                f"ML model ({type(model).__name__}) for {target}: "
                + (
                    f"accuracy={metrics.get('accuracy', 'N/A')}"
                    if task == "classification"
                    else f"R²={metrics.get('r2', 'N/A')}"
                ),
                sample_size=original_shape[0],
            )

        return JsonResponse(sanitize_for_json(response_data))

    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"DSW from_data failed: {e}")
        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data",
            success=False,
            error_message=str(e)[:500],
            metadata={"target": target, "intent": intent[:100] if intent else None},
        )
        return JsonResponse(
            {
                "error": f"Pipeline failed: {str(e)[:200]}",
                "suggestion": "Check that your data is properly formatted and the target column exists.",
            },
            status=500,
        )


@require_http_methods(["GET"])
@require_auth
def dsw_download(request, result_id, file_type):
    """Download DSW result files."""
    try:
        db_result = DSWResult.objects.get(id=result_id, user=request.user)
        result = json.loads(db_result.data)
    except DSWResult.DoesNotExist:
        return JsonResponse({"error": "Result not found"}, status=404)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if file_type == "report":
                if hasattr(result, "full_report"):
                    content = result.full_report()
                elif hasattr(result, "report"):
                    content = result.report.to_markdown()
                elif isinstance(result, dict) and "summary" in result:
                    content = f"# DSW Report\n\n{result['summary']}\n\n## Metrics\n\n"
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            content += f"- {k}: {v}\n"
                else:
                    return JsonResponse({"error": "No report available"}, status=400)
                path = tmpdir / "report.md"
                path.write_text(content)
                return FileResponse(open(path, "rb"), as_attachment=True, filename="report.md")

            elif file_type == "code":
                if hasattr(result, "deployment_code"):
                    code = result.deployment_code
                elif hasattr(result, "code"):
                    code = result.code
                else:
                    code = "# Generated by SVEND DSW Pipeline\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\n\ndf = pd.read_csv('data.csv')\nX = df.drop('target', axis=1)\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\naccuracy = model.score(X_test, y_test)\nprint(f'Accuracy: {accuracy:.3f}')\n"
                path = tmpdir / "train.py"
                path.write_text(code)
                return FileResponse(open(path, "rb"), as_attachment=True, filename="train.py")

            elif file_type == "data":
                data_path = result.get("data_path")
                if data_path and Path(data_path).exists():
                    return FileResponse(open(data_path, "rb"), as_attachment=True, filename="data.csv")
                else:
                    return JsonResponse({"error": "No data available"}, status=400)

            elif file_type == "model":
                try:
                    saved_model = SavedModel.objects.get(dsw_result_id=result_id, user=request.user)
                    model_path = Path(saved_model.model_path)
                    if model_path.exists():
                        return FileResponse(
                            open(model_path, "rb"),
                            as_attachment=True,
                            filename=f"{saved_model.name.replace(' ', '_')}.pkl",
                        )
                    else:
                        return JsonResponse({"error": "Model file not found"}, status=404)
                except SavedModel.DoesNotExist:
                    return JsonResponse({"error": "No saved model for this result"}, status=404)
            else:
                return JsonResponse({"error": f"Unknown file type: {file_type}"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
@require_auth
def list_models(request):
    """List user's saved models."""
    models_qs = SavedModel.objects.filter(user=request.user).select_related("project")
    project_id = request.GET.get("project_id")
    if project_id:
        models_qs = models_qs.filter(project_id=project_id)

    return JsonResponse(
        {
            "models": [
                {
                    "id": str(m.id),
                    "name": m.name,
                    "description": m.description,
                    "model_type": m.model_type,
                    "dsw_result_id": m.dsw_result_id,
                    "metrics": json.loads(m.metrics) if m.metrics else None,
                    "features": json.loads(m.feature_names) if m.feature_names else None,
                    "target": m.target_name,
                    "created_at": m.created_at.isoformat(),
                    "project_id": str(m.project_id) if m.project_id else None,
                    "project_title": m.project.title if m.project else None,
                    "training_config": m.training_config or {},
                    "data_lineage": m.data_lineage or {},
                    "version": m.version,
                    "parent_model_id": str(m.parent_model_id) if m.parent_model_id else None,
                }
                for m in models_qs
            ]
        }
    )


@require_http_methods(["POST"])
@require_auth
def save_model_from_cache(request):
    """Save a trained model from the cache to permanent storage."""
    try:
        data = json.loads(request.body)
        model_key = data.get("model_key")
        name = data.get("name", "Untitled Model")
        description = data.get("description", "")

        if not model_key:
            return JsonResponse({"error": "model_key is required"}, status=400)

        cached = get_cached_model(request.user.id, model_key)
        if not cached:
            return JsonResponse({"error": "Model not found in cache. It may have expired."}, status=404)

        model = cached["model"]
        metadata = cached["metadata"]

        saved = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=metadata.get("model_type", "unknown"),
            dsw_result_id=model_key,
            name=name,
            metrics=metadata.get("metrics"),
            features=metadata.get("features"),
            target=metadata.get("target"),
            project_id=data.get("project_id"),
            training_config=metadata.get("training_config"),
            data_lineage=metadata.get("data_lineage"),
        )

        if saved:
            if description:
                saved.description = description
                saved.save()
            return JsonResponse(
                {
                    "success": True,
                    "model_id": str(saved.id),
                    "name": saved.name,
                    "message": f"Model '{name}' saved successfully",
                }
            )
        else:
            return JsonResponse({"error": "Failed to save model"}, status=500)

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
@require_auth
def download_model(request, model_id):
    """Download a saved model by ID."""
    try:
        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)
        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)
        return FileResponse(
            open(model_path, "rb"), as_attachment=True, filename=f"{saved_model.name.replace(' ', '_')}.pkl"
        )
    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@require_http_methods(["DELETE"])
@require_auth
def delete_model(request, model_id):
    """Delete a saved model."""
    try:
        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)
        if model_path.exists():
            model_path.unlink()
        saved_model.delete()
        return JsonResponse({"success": True})
    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@require_http_methods(["POST"])
@gated
def run_model(request, model_id):
    """Run inference with a saved model."""
    try:
        import pickle

        import numpy as np
        import pandas as pd

        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)
        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if "file" in request.FILES:
            df = _read_csv_safe(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No input data provided"}, status=400)
        else:
            return JsonResponse({"error": "No input data provided"}, status=400)

        features = json.loads(saved_model.feature_names) if saved_model.feature_names else list(df.columns)
        X = df[features] if all(f in df.columns for f in features) else df

        training_config = saved_model.training_config or {}
        feature_info = training_config.get("feature_info", {})

        for col in X.select_dtypes(include=["object", "category"]).columns:
            if col in feature_info and feature_info[col].get("categories"):
                train_cats = sorted(feature_info[col]["categories"])
                X[col] = pd.Categorical(X[col], categories=train_cats).codes.astype(int)
            else:
                X[col] = pd.Categorical(X[col]).codes.astype(int)

        X = X.fillna(0)
        X = X.replace(-1, 0)
        predictions = model.predict(X)

        label_map = training_config.get("label_map")
        if label_map:
            inv_map = {int(v): k for k, v in label_map.items()}
            try:
                predictions = [inv_map.get(int(p), p) for p in predictions]
            except (ValueError, TypeError):
                predictions = predictions.tolist()

        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X).tolist()
            except Exception:
                pass

        intervals = None
        prediction_sets = None
        conformal_state = training_config.get("conformal_state")
        if conformal_state:
            try:
                from ..conformal import ConformalClassifier, ConformalRegressor

                if conformal_state["type"] == "regression":
                    cf = ConformalRegressor.from_state(conformal_state)
                    pred_arr = np.array(
                        predictions if isinstance(predictions, list) else predictions.tolist(), dtype=np.float64
                    )
                    lo, hi = cf.predict_interval(pred_arr, alpha=0.10)
                    intervals = [[round(float(lo[i]), 4), round(float(hi[i]), 4)] for i in range(len(lo))]
                elif conformal_state["type"] == "classification" and probabilities is not None:
                    cf = ConformalClassifier.from_state(conformal_state)
                    sets, meta = cf.predict_sets(np.array(probabilities), alpha=0.10)
                    prediction_sets = sets
            except Exception:
                pass

        if intervals is None:
            want_intervals = False
            if request.body:
                try:
                    body_data = json.loads(request.body)
                    want_intervals = body_data.get("intervals", False)
                except (json.JSONDecodeError, AttributeError):
                    pass
            if want_intervals and hasattr(model, "estimators_"):
                try:
                    tree_preds = np.array([t.predict(X) for t in model.estimators_])
                    lo = np.percentile(tree_preds, 5, axis=0)
                    hi = np.percentile(tree_preds, 95, axis=0)
                    intervals = [[round(float(lo[i]), 4), round(float(hi[i]), 4)] for i in range(len(lo))]
                except Exception:
                    pass

        request.user.increment_queries()

        pred_list = predictions if isinstance(predictions, list) else predictions.tolist()
        result = {"predictions": pred_list, "probabilities": probabilities, "count": len(pred_list)}
        if intervals:
            result["intervals"] = intervals
            if conformal_state:
                result["interval_coverage"] = "90% nominal (split conformal)"
        if prediction_sets:
            result["prediction_sets"] = prediction_sets
            result["conformal_info"] = "90% nominal coverage (split conformal)"
        return JsonResponse(result)

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Exception as e:
        logger.exception("Model inference error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@require_auth
def optimize_model(request, model_id):
    """Find optimal input values with density awareness, constraints, costs, and diminishing returns."""
    try:
        import pickle

        import numpy as np
        import pandas as pd
        from scipy.optimize import differential_evolution

        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)
        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        data = json.loads(request.body)
        goal = data.get("goal", "maximize")
        target_value = data.get("target_value", 0)
        current_values = data.get("current_values", {})
        user_bounds = data.get("feature_bounds", {})
        sum_constraints = data.get("sum_constraints", [])
        cost_weights = data.get("cost_weights", {})
        use_density = data.get("density_penalty", True)

        features = json.loads(saved_model.feature_names) if saved_model.feature_names else []
        training_config = saved_model.training_config or {}
        feature_info = training_config.get("feature_info", {})
        feature_stats = training_config.get("feature_stats", {})

        numeric_feats, cat_feats, bounds, ranges = [], [], [], []
        for feat in features:
            info = feature_info.get(feat, {})
            if info.get("categories"):
                cat_feats.append(feat)
            else:
                numeric_feats.append(feat)
                lo, hi = info.get("min", 0), info.get("max", 100)
                ub = user_bounds.get(feat, {})
                if ub.get("min") is not None:
                    lo = max(lo, float(ub["min"]))
                if ub.get("max") is not None:
                    hi = min(hi, float(ub["max"]))
                if lo >= hi:
                    hi = lo + 1e-4
                bounds.append((lo, hi))
                ranges.append(hi - lo if hi > lo else 1)

        if not numeric_feats:
            return JsonResponse({"error": "No numeric features to optimize"}, status=400)

        num_idx = {f: i for i, f in enumerate(numeric_feats)}
        current_numeric = np.array(
            [float(current_values.get(f, (bounds[i][0] + bounds[i][1]) / 2)) for i, f in enumerate(numeric_feats)]
        )
        ranges_arr = np.array(ranges)
        cost_arr = np.array([float(cost_weights.get(f, 1)) for f in numeric_feats])
        has_costs = any(cost_weights.get(f) and float(cost_weights[f]) != 1 for f in numeric_feats)

        cov_inv, means_arr, density_threshold = None, None, 1.0
        if use_density and feature_stats.get("covariance") and feature_stats.get("means"):
            try:
                stat_feats = feature_stats["numeric_features"]
                means_arr = np.array([feature_stats["means"].get(f, 0) for f in stat_feats])
                cov_matrix = np.array([[feature_stats["covariance"][c][r] for c in stat_feats] for r in stat_feats])
                cov_matrix += np.eye(len(stat_feats)) * 1e-6
                cov_inv = np.linalg.inv(cov_matrix)
                density_threshold = np.sqrt(len(stat_feats)) * 1.5
            except Exception:
                cov_inv = None

        baseline_pred = abs(
            _predict_numeric(
                model,
                current_numeric,
                {f: current_values.get(f, "") for f in cat_feats},
                numeric_feats,
                cat_feats,
                features,
                feature_info,
                current_values,
            )
        )
        pred_scale = max(baseline_pred, 1.0)

        def _predict_row(numeric_vals, cat_vals):
            return _predict_numeric(
                model, numeric_vals, cat_vals, numeric_feats, cat_feats, features, feature_info, current_values
            )

        cat_combos = [current_values]
        if cat_feats:
            from itertools import product

            cat_options = []
            for feat in cat_feats:
                info = feature_info.get(feat, {})
                cats = info.get("categories", [current_values.get(feat, "")])
                cat_options.append([(feat, c) for c in cats[:5]])
            total_combos = 1
            for opts in cat_options:
                total_combos *= len(opts)
            if total_combos <= 20:
                cat_combos = [dict(combo) for combo in product(*cat_options)]

        best_val, best_inputs = None, None

        for cat_vals in cat_combos:

            def objective(x, _cv=cat_vals):
                pred = _predict_row(x, _cv)
                obj = -pred if goal == "maximize" else (pred if goal == "minimize" else abs(pred - target_value))
                if cov_inv is not None:
                    sf = feature_stats["numeric_features"]
                    x_full = np.array(
                        [float(x[num_idx[f]]) if f in num_idx else means_arr[i] for i, f in enumerate(sf)]
                    )
                    diff = x_full - means_arr
                    mahal = float(np.sqrt(max(0, diff @ cov_inv @ diff)))
                    if mahal > density_threshold:
                        obj += 0.05 * pred_scale * ((mahal - density_threshold) / density_threshold) ** 2
                if has_costs:
                    deltas = np.abs(x - current_numeric) / ranges_arr
                    obj += 0.03 * pred_scale * float(np.sum(cost_arr * deltas))
                for sc in sum_constraints:
                    sc_feats, sc_op, sc_limit = (
                        sc.get("features", []),
                        sc.get("operator", "<="),
                        float(sc.get("limit", 0)),
                    )
                    feat_sum = sum(float(x[num_idx[f]]) for f in sc_feats if f in num_idx)
                    if sc_op == "<=" and feat_sum > sc_limit:
                        obj += 10 * pred_scale * ((feat_sum - sc_limit) / max(sc_limit, 1)) ** 2
                    elif sc_op == ">=" and feat_sum < sc_limit:
                        obj += 10 * pred_scale * ((sc_limit - feat_sum) / max(sc_limit, 1)) ** 2
                return obj

            result = differential_evolution(
                objective, bounds, maxiter=30, tol=1e-4, seed=42, polish=True, init="latinhypercube", popsize=10
            )
            pred = _predict_row(result.x, cat_vals)
            if (
                best_val is None
                or (goal == "maximize" and pred > best_val)
                or (goal == "minimize" and pred < best_val)
                or (goal == "target" and abs(pred - target_value) < abs(best_val - target_value))
            ):
                best_val = pred
                best_inputs = {}
                for i, feat in enumerate(numeric_feats):
                    best_inputs[feat] = round(float(result.x[i]), 4)
                for feat in cat_feats:
                    best_inputs[feat] = cat_vals.get(feat, current_values.get(feat, ""))

        response = {
            "optimal_inputs": best_inputs,
            "predicted_value": round(best_val, 4) if best_val is not None else None,
            "goal": goal,
        }
        if not best_inputs:
            return JsonResponse(response)

        best_cat = {f: best_inputs.get(f, current_values.get(f, "")) for f in cat_feats}
        best_num = np.array([float(best_inputs.get(f, 0)) for f in numeric_feats])

        # Prescription
        prescription = []
        for feat in features:
            curr, opt = current_values.get(feat), best_inputs.get(feat)
            if curr is None or opt is None:
                continue
            info = feature_info.get(feat, {})
            if info.get("categories"):
                if str(curr) != str(opt):
                    prescription.append({"feature": feat, "from": str(curr), "to": str(opt), "type": "switch"})
            else:
                try:
                    delta = float(opt) - float(curr)
                    if abs(delta) < 1e-6:
                        prescription.append({"feature": feat, "action": "hold", "value": round(float(opt), 4)})
                    else:
                        feat_range = (info.get("max", 100) - info.get("min", 0)) or 1
                        prescription.append(
                            {
                                "feature": feat,
                                "from": round(float(curr), 4),
                                "to": round(float(opt), 4),
                                "delta": round(delta, 4),
                                "direction": "increase" if delta > 0 else "decrease",
                                "magnitude_pct": round(abs(delta) / feat_range * 100, 1),
                                "type": "adjust",
                                "cost": float(cost_weights.get(feat, 1)),
                            }
                        )
                except (ValueError, TypeError):
                    pass
        response["prescription"] = prescription

        # Edge warnings
        edge_warnings = []
        for feat in numeric_feats:
            info = feature_info.get(feat, {})
            lo, hi = info.get("min", 0), info.get("max", 100)
            val = best_inputs.get(feat)
            if val is None:
                continue
            feat_range = (hi - lo) or 1
            if (val - lo) / feat_range < 0.05:
                edge_warnings.append({"feature": feat, "edge": "lower", "value": round(val, 4), "bound": round(lo, 4)})
            elif (hi - val) / feat_range < 0.05:
                edge_warnings.append({"feature": feat, "edge": "upper", "value": round(val, 4), "bound": round(hi, 4)})
        response["edge_warnings"] = edge_warnings

        # Feasibility
        feasibility = {"score": 1.0, "label": "feasible", "details": ""}
        if cov_inv is not None:
            try:
                stat_feats = feature_stats["numeric_features"]
                opt_vals = np.array([float(best_inputs.get(f, feature_stats["means"].get(f, 0))) for f in stat_feats])
                diff = opt_vals - means_arr
                mahal_dist = float(np.sqrt(max(0, diff @ cov_inv @ diff)))
                p = len(stat_feats)
                threshold_ok, threshold_warn = np.sqrt(p) * 1.5, np.sqrt(p) * 2.5
                if mahal_dist <= threshold_ok:
                    feasibility = {
                        "score": round(1.0 - mahal_dist / (threshold_warn * 1.5), 2),
                        "label": "high",
                        "mahalanobis": round(mahal_dist, 2),
                        "details": "This combination is well within your observed data -- prediction is reliable.",
                    }
                elif mahal_dist <= threshold_warn:
                    feasibility = {
                        "score": round(0.5 - (mahal_dist - threshold_ok) / (threshold_warn * 2), 2),
                        "label": "moderate",
                        "mahalanobis": round(mahal_dist, 2),
                        "details": "This combination is at the edge of your data. Prediction has moderate uncertainty.",
                    }
                else:
                    feasibility = {
                        "score": max(0.0, round(0.2 - mahal_dist / (threshold_warn * 5), 2)),
                        "label": "low",
                        "mahalanobis": round(mahal_dist, 2),
                        "details": "This combination is far from any training data -- the model is extrapolating. Collect data near these values before acting.",
                    }
                violated = []
                for corr in feature_stats.get("strong_correlations", []):
                    f1, f2, r = corr["f1"], corr["f2"], corr["r"]
                    if f1 in best_inputs and f2 in best_inputs:
                        v1 = (float(best_inputs[f1]) - feature_stats["means"].get(f1, 0)) / max(
                            feature_stats["stds"].get(f1, 1), 1e-6
                        )
                        v2 = (float(best_inputs[f2]) - feature_stats["means"].get(f2, 0)) / max(
                            feature_stats["stds"].get(f2, 1), 1e-6
                        )
                        if r > 0 and v1 * v2 < -1:
                            violated.append(
                                f"{f1} and {f2} are positively correlated (r={r}) but the optimum pushes them apart"
                            )
                        elif r < 0 and v1 * v2 > 1:
                            violated.append(
                                f"{f1} and {f2} are negatively correlated (r={r}) but the optimum pushes them together"
                            )
                if violated:
                    feasibility["correlation_warnings"] = violated
            except Exception:
                pass
        response["feasibility"] = feasibility

        # Sensitivity
        sensitivity = []
        for i, feat in enumerate(numeric_feats):
            info = feature_info.get(feat, {})
            lo, hi = info.get("min", 0), info.get("max", 100)
            step = (hi - lo) * 0.01 if (hi - lo) > 0 else 0.01
            opt_val = float(best_inputs.get(feat, 0))
            x_up, x_down = list(best_num), list(best_num)
            x_up[i] = min(opt_val + step, hi)
            x_down[i] = max(opt_val - step, lo)
            pred_up, pred_down = _predict_row(x_up, best_cat), _predict_row(x_down, best_cat)
            gradient = (pred_up - pred_down) / (2 * step) if step > 0 else 0
            sensitivity.append(
                {"feature": feat, "gradient": round(gradient, 6), "impact": round(abs(gradient) * (hi - lo), 4)}
            )
        sensitivity.sort(key=lambda s: abs(s["impact"]), reverse=True)
        response["sensitivity"] = sensitivity

        # Diminishing returns
        diminishing = []
        for i, feat in enumerate(numeric_feats):
            curr_val = float(current_values.get(feat, current_numeric[i]))
            opt_val = float(best_inputs.get(feat, curr_val))
            if abs(opt_val - curr_val) < 1e-6:
                continue
            steps = np.linspace(curr_val, opt_val, 11)
            preds = [
                _predict_row([sv if j == i else best_num[j] for j in range(len(numeric_feats))], best_cat)
                for sv in steps
            ]
            total_gain = abs(preds[-1] - preds[0])
            if total_gain < 1e-8:
                continue
            knee_pct, knee_val = 100, opt_val
            for k in range(1, len(preds)):
                if abs(preds[k] - preds[0]) / total_gain >= 0.8:
                    knee_pct = round(k / (len(preds) - 1) * 100)
                    knee_val = round(float(steps[k]), 4)
                    break
            diminishing.append(
                {
                    "feature": feat,
                    "knee_pct": knee_pct,
                    "knee_value": knee_val,
                    "total_gain": round(total_gain, 4),
                    "current": round(curr_val, 4),
                    "optimal": round(opt_val, 4),
                }
            )
        response["diminishing_returns"] = diminishing

        # Constraint satisfaction
        constraint_status = []
        for sc in sum_constraints:
            sc_feats, sc_op, sc_limit = sc.get("features", []), sc.get("operator", "<="), float(sc.get("limit", 0))
            feat_sum = sum(float(best_inputs.get(f, 0)) for f in sc_feats if f in best_inputs)
            satisfied = (sc_op == "<=" and feat_sum <= sc_limit + 1e-6) or (
                sc_op == ">=" and feat_sum >= sc_limit - 1e-6
            )
            constraint_status.append(
                {
                    "features": sc_feats,
                    "operator": sc_op,
                    "limit": sc_limit,
                    "actual": round(feat_sum, 4),
                    "satisfied": satisfied,
                }
            )
        if constraint_status:
            response["constraints"] = constraint_status

        # Prediction interval at optimal point
        if hasattr(model, "estimators_"):
            try:
                opt_row = {f: best_inputs.get(f, current_values.get(f, "")) for f in features}
                opt_df = pd.DataFrame([opt_row])
                for col in opt_df.select_dtypes(include=["object", "category"]).columns:
                    if col in feature_info and feature_info[col].get("categories"):
                        opt_df[col] = pd.Categorical(
                            opt_df[col], categories=sorted(feature_info[col]["categories"])
                        ).codes.astype(int)
                    else:
                        opt_df[col] = pd.Categorical(opt_df[col]).codes.astype(int)
                opt_df = opt_df.fillna(0).replace(-1, 0)
                X_opt = opt_df[features] if all(f in opt_df.columns for f in features) else opt_df
                tree_preds = np.array([t.predict(X_opt)[0] for t in model.estimators_])
                response["interval"] = {
                    "lower": round(float(np.percentile(tree_preds, 5)), 4),
                    "upper": round(float(np.percentile(tree_preds, 95)), 4),
                    "std": round(float(np.std(tree_preds)), 4),
                }
            except Exception:
                pass

        return JsonResponse(response)

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Exception as e:
        logger.exception("Model optimization error")
        return JsonResponse({"error": str(e)}, status=500)


def _predict_numeric(model, numeric_vals, cat_vals, numeric_feats, cat_feats, features, feature_info, current_values):
    """Shared prediction helper for optimize -- builds DataFrame from numeric array + cat dict."""
    import pandas as pd

    row = {}
    for i, feat in enumerate(numeric_feats):
        row[feat] = numeric_vals[i]
    for feat in cat_feats:
        row[feat] = cat_vals.get(feat, current_values.get(feat, ""))
    df = pd.DataFrame([row])
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in feature_info and feature_info[col].get("categories"):
            df[col] = pd.Categorical(df[col], categories=sorted(feature_info[col]["categories"])).codes.astype(int)
        else:
            df[col] = pd.Categorical(df[col]).codes.astype(int)
    df = df.fillna(0).replace(-1, 0)
    X = df[features] if all(f in df.columns for f in features) else df
    return float(model.predict(X)[0])


@require_http_methods(["GET"])
@require_auth
def models_summary(request):
    """Summary stats for ML Hub dashboard."""
    from datetime import timedelta

    from django.utils import timezone

    models_qs = SavedModel.objects.filter(user=request.user).select_related("project")
    week_ago = timezone.now() - timedelta(days=7)

    projects, best_model, best_metric = {}, None, -1
    for m in models_qs:
        key = str(m.project_id) if m.project_id else "ungrouped"
        if key not in projects:
            projects[key] = {
                "project_id": str(m.project_id) if m.project_id else None,
                "project_title": m.project.title if m.project else "Ungrouped",
                "model_count": 0,
            }
        projects[key]["model_count"] += 1
        try:
            metrics = json.loads(m.metrics) if m.metrics else {}
            primary = metrics.get("accuracy") or metrics.get("r2", 0)
            if isinstance(primary, (int, float)) and primary > best_metric:
                best_metric = primary
                best_model = {"id": str(m.id), "name": m.name, "metric": primary}
        except (json.JSONDecodeError, TypeError):
            pass

    return JsonResponse(
        {
            "total_models": models_qs.count(),
            "recent_models": models_qs.filter(created_at__gte=week_ago).count(),
            "project_groups": list(projects.values()),
            "best_model": best_model,
        }
    )


@require_http_methods(["GET"])
@require_auth
def model_versions(request, model_id):
    """Get version history for a model (walks parent chain)."""
    try:
        model = SavedModel.objects.get(id=model_id, user=request.user)
    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)

    versions, current, seen = [], model, set()
    while current and current.id not in seen:
        seen.add(current.id)
        versions.append(
            {
                "id": str(current.id),
                "name": current.name,
                "version": current.version,
                "model_type": current.model_type,
                "metrics": json.loads(current.metrics) if current.metrics else {},
                "created_at": current.created_at.isoformat(),
            }
        )
        current = current.parent_model
    return JsonResponse({"versions": versions})


@require_http_methods(["GET"])
@require_auth
def model_report(request, model_id):
    """Fetch the full training report for a saved model."""
    try:
        model = SavedModel.objects.get(id=model_id, user=request.user)
    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)

    if not model.dsw_result_id:
        return JsonResponse({"error": "No training report stored for this model"}, status=404)

    try:
        result = DSWResult.objects.get(id=model.dsw_result_id, user=request.user)
    except DSWResult.DoesNotExist:
        return JsonResponse({"error": "Training report not found"}, status=404)

    try:
        result_data = json.loads(result.data)
    except (json.JSONDecodeError, TypeError):
        return JsonResponse({"error": "Invalid report data"}, status=500)

    steps = result_data.get("steps", [])
    plots = result_data.get("plots", [])
    data_lineage = result_data.get("data_lineage", {})

    cleaning = None
    if data_lineage.get("triage_applied"):
        cleaning = {
            "original_shape": data_lineage.get("original_shape"),
            "cleaned_shape": data_lineage.get("cleaned_shape"),
        }

    augmentation = None
    if data_lineage.get("forge_applied"):
        augmentation = {
            "original_rows": (data_lineage.get("original_shape") or [0])[0],
            "synthetic_rows": data_lineage.get("synthetic_rows"),
            "total_rows": (data_lineage.get("augmented_shape") or [0])[0],
        }

    def _is_shap(p):
        title = p.get("title") or ""
        if not title:
            layout = p.get("layout", {})
            t = layout.get("title", "")
            title = t.get("text", "") if isinstance(t, dict) else (t or "")
        return "SHAP" in title

    shap_plots = [p for p in plots if _is_shap(p)]
    other_plots = [p for p in plots if not _is_shap(p)]

    return JsonResponse(
        {
            "pipeline_stages": [{"name": s["name"], "success": s.get("status") == "completed"} for s in steps],
            "cleaning": cleaning,
            "augmentation": augmentation,
            "comparison": result_data.get("comparison"),
            "tuning": result_data.get("tuning"),
            "model_type": result_data.get("model_type"),
            "task": result_data.get("task"),
            "metrics": result_data.get("metrics"),
            "shap_plots": shap_plots,
            "plots": other_plots,
            "recipe": result_data.get("training_config"),
            "elapsed_seconds": result_data.get("elapsed_seconds"),
        }
    )


# =============================================================================
# Scrub Endpoints - Standalone Data Cleaning
# =============================================================================


@require_http_methods(["POST"])
@gated
def scrub_data(request):
    """Clean uploaded data using Scrub. POST with file upload or JSON data."""
    import pandas as pd

    try:
        from scrub import CleaningConfig, DataCleaner

        if "file" in request.FILES:
            df = _read_csv_safe(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No data provided"}, status=400)
        else:
            return JsonResponse({"error": "No data provided"}, status=400)

        config_data = {}
        if request.body:
            try:
                body = json.loads(request.body)
                config_data = body.get("config", {})
            except Exception:
                pass

        config = CleaningConfig(
            detect_outliers=config_data.get("detect_outliers", True),
            handle_missing=config_data.get("handle_missing", True),
            normalize_factors=config_data.get("normalize_factors", True),
            correct_types=config_data.get("correct_types", True),
            domain_rules=config_data.get("domain_rules", {}),
        )
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df, config)
        request.user.increment_queries()

        return JsonResponse(
            {
                "success": True,
                "original_shape": list(result.original_shape),
                "cleaned_shape": list(result.cleaned_shape),
                "outliers_flagged": result.outliers.count if result.outliers else 0,
                "missing_filled": result.missing.total_filled if result.missing else 0,
                "normalizations": result.normalization.total_changes if result.normalization else 0,
                "warnings": result.warnings,
                "report_markdown": result.to_markdown(),
                "report_summary": result.summary(),
                "data": df_clean.to_dict(orient="records"),
            }
        )
    except Exception as e:
        logger.exception("Scrub error")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
@gated
def scrub_analyze(request):
    """Analyze data without modifying it. Returns analysis of what would be cleaned + potential biases."""
    import pandas as pd

    try:
        from scrub import DataCleaner

        if "file" in request.FILES:
            df = _read_csv_safe(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No data provided"}, status=400)
        else:
            return JsonResponse({"error": "No data provided"}, status=400)

        cleaner = DataCleaner()
        analysis = cleaner.analyze(df)
        analysis["bias_warnings"] = _detect_data_biases(df)
        return JsonResponse({"success": True, "analysis": analysis})
    except Exception as e:
        logger.exception("Scrub analyze error")
        return JsonResponse({"error": str(e)}, status=500)


def _detect_data_biases(df) -> list:
    """Detect potential biases in dataset."""
    warnings = []

    sensitive_patterns = [
        "gender",
        "sex",
        "race",
        "ethnicity",
        "religion",
        "age",
        "disability",
        "marital",
        "nationality",
        "origin",
        "zip",
        "zipcode",
    ]

    for col in df.columns:
        col_lower = col.lower()
        for pattern in sensitive_patterns:
            if pattern in col_lower:
                warnings.append(
                    {
                        "type": "sensitive_feature",
                        "column": col,
                        "message": f"Column '{col}' may contain sensitive/protected information. Consider whether it should be used in modeling.",
                        "severity": "high",
                    }
                )
                break

    for col in df.columns:
        if df[col].dtype in ["object", "bool", "category"] or df[col].nunique() <= 10:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) >= 2:
                max_ratio, min_ratio = value_counts.iloc[0], value_counts.iloc[-1]
                if max_ratio > 0.9:
                    warnings.append(
                        {
                            "type": "class_imbalance",
                            "column": col,
                            "message": f"Column '{col}' has severe class imbalance ({max_ratio:.1%} vs {min_ratio:.1%}). Consider resampling strategies.",
                            "severity": "medium",
                        }
                    )

    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) > 1:
        for i, col1 in enumerate(missing_cols):
            for col2 in missing_cols[i + 1 :]:
                both_missing = (df[col1].isnull() & df[col2].isnull()).mean()
                col1_missing, col2_missing = df[col1].isnull().mean(), df[col2].isnull().mean()
                expected = col1_missing * col2_missing
                if both_missing > 2 * expected and both_missing > 0.05:
                    warnings.append(
                        {
                            "type": "correlated_missing",
                            "columns": [col1, col2],
                            "message": f"Missing data in '{col1}' and '{col2}' are correlated. This could indicate systematic bias in data collection.",
                            "severity": "medium",
                        }
                    )
    return warnings
