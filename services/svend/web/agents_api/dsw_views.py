"""DSW (Decision Science Workbench) API views."""

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
from .models import DSWResult, AgentLog, SavedModel
from accounts.permissions import gated, require_auth, require_enterprise

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

        from . import views as agent_views
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


def save_model_to_disk(user, model, model_type, dsw_result_id, name=None, metrics=None, features=None, target=None):
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
        )

        logger.info(f"Saved model {model_id} for user {user.username}")
        return saved_model

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


@csrf_exempt
@require_http_methods(["POST"])
@gated
def dsw_from_intent(request):
    """Run DSW pipeline from intent (zero data)."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    intent = data.get("intent", "").strip()
    domain = data.get("domain")
    n_records = int(data.get("n_records", 500))
    priority = data.get("priority", "balanced")

    if not intent:
        return JsonResponse({"error": "Intent is required"}, status=400)

    try:
        from dsw import DecisionScienceWorkbench

        dsw = DecisionScienceWorkbench()
        result = dsw.from_intent(
            intent=intent,
            domain=domain if domain else None,
            n_records=n_records,
            priority=priority,
        )

        # Store result in database
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        # Save synthetic data to file
        data_dir = Path(settings.MEDIA_ROOT) / "dsw_data" / str(request.user.id)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{result_id}.csv"
        result.synthetic_data.to_csv(data_path, index=False)

        result_data = {
            "schema": {
                "name": result.schema.name,
                "intent": result.schema.intent,
                "domain": result.schema.domain,
                "features": [f.name for f in result.schema.features],
                "target": result.schema.target_name,
            },
            "data_shape": list(result.synthetic_data.shape),
            "data_path": str(data_path),
            "model_type": result.model_type,
            "metrics": result.metrics,
            "pipeline": result.pipeline_steps,
            "warnings": result.warnings,
            "summary": result.summary(),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_intent",
            data=json.dumps(result_data),
        )

        # Save trained model
        saved_model = None
        if result.model:
            saved_model = save_model_to_disk(
                user=request.user,
                model=result.model,
                model_type=result.model_type,
                dsw_result_id=result_id,
                name=f"Model: {intent[:50]}",
                metrics=result.metrics,
                features=[f.name for f in result.schema.features],
                target=result.schema.target_name,
            )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent",
            success=True,
            metadata={"result_id": result_id, "domain": domain},
        )

        # Track usage
        request.user.increment_queries()

        response_data = {
            "result_id": result_id,
            **result_data,
        }
        if saved_model:
            response_data["model_id"] = str(saved_model.id)

        return JsonResponse(response_data)

    except ImportError as e:
        # DSW module not available - return mock result for testing
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        mock_result = {
            "schema": {
                "name": f"Schema for: {intent[:50]}",
                "intent": intent,
                "domain": domain or "auto-detected",
                "features": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
                "target": "target",
            },
            "data_shape": [n_records, 6],
            "model_type": "RandomForest",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1": 0.85,
                "auc_roc": 0.91,
            },
            "pipeline": ["schema_generation", "data_synthesis", "data_cleaning", "model_training"],
            "warnings": ["This is a mock result - DSW module not available"],
            "summary": f"Generated {n_records} synthetic records based on intent: {intent}. Model achieved 85% accuracy.",
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_intent_mock",
            data=json.dumps(mock_result),
        )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent_mock",
            success=True,
            metadata={"result_id": result_id, "mock": True},
        )

        # Track usage
        request.user.increment_queries()

        return JsonResponse({"result_id": result_id, **mock_result})

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

        return JsonResponse({
            "error": f"Pipeline failed: {str(e)[:200]}",
            "suggestion": "Try rephrasing your intent or use a different domain.",
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def dsw_from_data(request):
    """Run DSW pipeline from uploaded data."""
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file = request.FILES["file"]
    target = request.POST.get("target", "").strip()
    intent = request.POST.get("intent", "")
    priority = request.POST.get("priority", "balanced")

    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    try:
        import pandas as pd

        # Read uploaded CSV
        df = pd.read_csv(file)

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found in data'}, status=400)

        from dsw import DecisionScienceWorkbench

        dsw = DecisionScienceWorkbench()
        result = dsw.from_data(
            data=df,
            target=target,
            intent=intent,
            priority=priority,
        )

        # Store result in database
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        result_data = {
            "original_shape": list(result.original_data.shape),
            "cleaned_shape": list(result.cleaned_data.shape),
            "model_type": result.model_type,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance[:10],
            "pipeline": result.pipeline_steps,
            "warnings": result.warnings,
            "summary": result.summary(),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_data",
            data=json.dumps(result_data),
        )

        # Save trained model
        saved_model = None
        if result.model:
            features = [c for c in df.columns if c != target]
            saved_model = save_model_to_disk(
                user=request.user,
                model=result.model,
                model_type=result.model_type,
                dsw_result_id=result_id,
                name=f"Model: {target} prediction",
                metrics=result.metrics,
                features=features,
                target=target,
            )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data",
            success=True,
            metadata={"result_id": result_id, "rows": df.shape[0]},
        )

        # Track usage
        request.user.increment_queries()

        response_data = {"result_id": result_id, **result_data}
        if saved_model:
            response_data["model_id"] = str(saved_model.id)

        # Link results as evidence to a Problem (if problem_id provided)
        problem_id = request.POST.get("problem_id", "")
        if problem_id:
            try:
                from .views import add_finding_to_problem
                metrics = result_data.get("metrics", {})
                summary = (
                    f"Built {result_data.get('model_type', 'model')} on {target}. "
                    f"Accuracy: {metrics.get('accuracy', 'N/A')}, "
                    f"Top features: {', '.join(f['feature'] for f in result_data.get('feature_importance', [])[:3])}"
                )
                evidence = add_finding_to_problem(
                    user=request.user,
                    problem_id=problem_id,
                    summary=summary,
                    evidence_type="data_analysis",
                    source="DSW (from-data)",
                )
                if evidence:
                    response_data["problem_updated"] = True
                    response_data["evidence_id"] = evidence["id"]
            except Exception as e_prob:
                logger.warning(f"Could not link DSW from-data to problem {problem_id}: {e_prob}")

        return JsonResponse(response_data)

    except ImportError as e:
        # DSW module not available - return mock result
        import pandas as pd

        df = pd.read_csv(file)
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        mock_result = {
            "original_shape": list(df.shape),
            "cleaned_shape": [int(df.shape[0] * 0.95), df.shape[1]],
            "model_type": "GradientBoosting",
            "metrics": {
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.84,
                "f1": 0.82,
            },
            "feature_importance": [
                {"feature": col, "importance": 1.0 / len(df.columns)}
                for col in df.columns[:10] if col != target
            ],
            "pipeline": ["data_loading", "data_cleaning", "feature_engineering", "model_training"],
            "warnings": ["This is a mock result - DSW module not available"],
            "summary": f"Processed {df.shape[0]} rows with {df.shape[1]} columns. Target: {target}. Model achieved 82% accuracy.",
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_data_mock",
            data=json.dumps(mock_result),
        )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data_mock",
            success=True,
            metadata={"result_id": result_id, "mock": True},
        )

        # Track usage
        request.user.increment_queries()

        return JsonResponse({"result_id": result_id, **mock_result})

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

        return JsonResponse({
            "error": f"Pipeline failed: {str(e)[:200]}",
            "suggestion": "Check that your data is properly formatted and the target column exists.",
        }, status=500)


@require_http_methods(["GET"])
@require_auth
def dsw_download(request, result_id, file_type):
    """Download DSW result files."""
    # Try to get from database
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
                    # Generate template code
                    code = """# Generated by SVEND DSW Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.3f}')
"""

                path = tmpdir / "train.py"
                path.write_text(code)
                return FileResponse(open(path, "rb"), as_attachment=True, filename="train.py")

            elif file_type == "data":
                # Check for saved data file path
                data_path = result.get("data_path")
                if data_path and Path(data_path).exists():
                    return FileResponse(
                        open(data_path, "rb"),
                        as_attachment=True,
                        filename="data.csv"
                    )
                else:
                    return JsonResponse({"error": "No data available"}, status=400)

            elif file_type == "model":
                # Find saved model for this DSW result
                try:
                    saved_model = SavedModel.objects.get(dsw_result_id=result_id, user=request.user)
                    model_path = Path(saved_model.model_path)
                    if model_path.exists():
                        return FileResponse(
                            open(model_path, "rb"),
                            as_attachment=True,
                            filename=f"{saved_model.name.replace(' ', '_')}.pkl"
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
    models = SavedModel.objects.filter(user=request.user)
    return JsonResponse({
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
            }
            for m in models
        ]
    })


@csrf_exempt
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

        # Get model from cache
        cached = get_cached_model(request.user.id, model_key)
        if not cached:
            return JsonResponse({"error": "Model not found in cache. It may have expired."}, status=404)

        model = cached['model']
        metadata = cached['metadata']

        # Save to disk
        saved = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=metadata.get('model_type', 'unknown'),
            dsw_result_id=model_key,
            name=name,
            metrics=metadata.get('metrics'),
            features=metadata.get('features'),
            target=metadata.get('target')
        )

        if saved:
            # Update description
            if description:
                saved.description = description
                saved.save()

            return JsonResponse({
                "success": True,
                "model_id": str(saved.id),
                "name": saved.name,
                "message": f"Model '{name}' saved successfully"
            })
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
            open(model_path, "rb"),
            as_attachment=True,
            filename=f"{saved_model.name.replace(' ', '_')}.pkl"
        )

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@csrf_exempt
@require_http_methods(["DELETE"])
@require_auth
def delete_model(request, model_id):
    """Delete a saved model."""
    try:
        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)

        # Delete file
        if model_path.exists():
            model_path.unlink()

        # Delete record
        saved_model.delete()

        return JsonResponse({"success": True})

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def run_model(request, model_id):
    """Run inference with a saved model."""
    try:
        import pickle
        import pandas as pd

        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)

        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Get input data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No input data provided"}, status=400)
        else:
            return JsonResponse({"error": "No input data provided"}, status=400)

        # Run prediction
        features = json.loads(saved_model.feature_names) if saved_model.feature_names else list(df.columns)
        X = df[features] if all(f in df.columns for f in features) else df

        predictions = model.predict(X)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X).tolist()
            except Exception:
                pass

        # Track query usage
        request.user.increment_queries()

        return JsonResponse({
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "count": len(predictions),
        })

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Exception as e:
        logger.exception("Model inference error")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Scrub Endpoints - Standalone Data Cleaning
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def scrub_data(request):
    """
    Clean uploaded data using Scrub.

    POST with file upload or JSON data.
    Returns cleaned data + detailed report.
    """
    import pandas as pd

    try:
        from scrub import DataCleaner, CleaningConfig

        # Get data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No data provided"}, status=400)
        else:
            return JsonResponse({"error": "No data provided"}, status=400)

        # Get config from request
        config_data = {}
        if request.body:
            try:
                body = json.loads(request.body)
                config_data = body.get("config", {})
            except:
                pass

        config = CleaningConfig(
            detect_outliers=config_data.get("detect_outliers", True),
            handle_missing=config_data.get("handle_missing", True),
            normalize_factors=config_data.get("normalize_factors", True),
            correct_types=config_data.get("correct_types", True),
            domain_rules=config_data.get("domain_rules", {}),
        )

        # Clean
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df, config)

        # Track query usage
        request.user.increment_queries()

        return JsonResponse({
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
        })

    except Exception as e:
        logger.exception("Scrub error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def scrub_analyze(request):
    """
    Analyze data without modifying it.

    Returns analysis of what would be cleaned + potential biases.
    """
    import pandas as pd

    try:
        from scrub import DataCleaner

        # Get data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
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

        # Add bias detection
        bias_warnings = _detect_data_biases(df)
        analysis["bias_warnings"] = bias_warnings

        return JsonResponse({
            "success": True,
            "analysis": analysis,
        })

    except Exception as e:
        logger.exception("Scrub analyze error")
        return JsonResponse({"error": str(e)}, status=500)


def _detect_data_biases(df) -> list:
    """
    Detect potential biases in dataset.

    Checks for:
    - Proxy variables for protected classes
    - Class imbalance
    - Missing data patterns that correlate with sensitive features
    """
    import pandas as pd

    warnings = []

    # Protected/sensitive column patterns
    sensitive_patterns = [
        "gender", "sex", "race", "ethnicity", "religion", "age",
        "disability", "marital", "nationality", "origin", "zip", "zipcode",
    ]

    # Check for sensitive columns
    for col in df.columns:
        col_lower = col.lower()
        for pattern in sensitive_patterns:
            if pattern in col_lower:
                warnings.append({
                    "type": "sensitive_feature",
                    "column": col,
                    "message": f"Column '{col}' may contain sensitive/protected information. "
                               f"Consider whether it should be used in modeling.",
                    "severity": "high",
                })
                break

    # Check for class imbalance in likely target columns
    for col in df.columns:
        if df[col].dtype in ['object', 'bool', 'category'] or df[col].nunique() <= 10:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) >= 2:
                max_ratio = value_counts.iloc[0]
                min_ratio = value_counts.iloc[-1]
                if max_ratio > 0.9:
                    warnings.append({
                        "type": "class_imbalance",
                        "column": col,
                        "message": f"Column '{col}' has severe class imbalance "
                                   f"({max_ratio:.1%} vs {min_ratio:.1%}). "
                                   f"Consider resampling strategies.",
                        "severity": "medium",
                    })

    # Check for correlated missing data
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) > 1:
        for i, col1 in enumerate(missing_cols):
            for col2 in missing_cols[i+1:]:
                # Check if missing in one correlates with missing in other
                both_missing = (df[col1].isnull() & df[col2].isnull()).mean()
                col1_missing = df[col1].isnull().mean()
                col2_missing = df[col2].isnull().mean()
                expected = col1_missing * col2_missing

                if both_missing > 2 * expected and both_missing > 0.05:
                    warnings.append({
                        "type": "correlated_missing",
                        "columns": [col1, col2],
                        "message": f"Missing data in '{col1}' and '{col2}' are correlated. "
                                   f"This could indicate systematic bias in data collection.",
                        "severity": "medium",
                    })

    return warnings


# =============================================================================
# ANALYSIS WORKBENCH ENDPOINTS
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def run_analysis(request):
    """
    Run a statistical/ML analysis.

    Request body:
    {
        "type": "stats" | "ml" | "spc" | "viz",
        "analysis": "anova" | "regression" | etc,
        "config": {...},
        "data_id": "...",
        "problem_id": "..." (optional - links results as evidence to a Problem),
        "project_id": "..." (optional - links result to a Project for A3 import),
        "title": "..." (optional - human-readable title for the result),
        "save_result": true (optional - persist result for later import)
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})
    data_id = body.get("data_id")
    project_id = body.get("project_id")
    result_title = body.get("title", "")
    save_result = body.get("save_result", False)

    start_time = time.time()

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        df = None

        # Source 0: Inline data (from learning module or API callers)
        inline_data = body.get("data")
        if inline_data and isinstance(inline_data, dict):
            try:
                df = pd.DataFrame(inline_data)
                if len(df) > 10000:
                    return JsonResponse({"error": "Inline data limited to 10,000 rows"}, status=400)
            except Exception as e:
                return JsonResponse({"error": f"Invalid inline data: {e}"}, status=400)

        # Source 1: Uploaded via upload_data endpoint (data_xxx format)
        if df is None and data_id and data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            # Fallback to temp directory
            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        # Source 2: Triage cleaned dataset
        if df is None and data_id:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "No data loaded. Please load a dataset first."}, status=400)

        result = {"plots": [], "summary": "", "guide_observation": ""}

        # Route to appropriate analysis
        if analysis_type == "stats":
            result = run_statistical_analysis(df, analysis_id, config)
        elif analysis_type == "ml":
            result = run_ml_analysis(df, analysis_id, config, request.user)
        elif analysis_type == "spc":
            result = run_spc_analysis(df, analysis_id, config)
        elif analysis_type == "viz":
            result = run_visualization(df, analysis_id, config)
        elif analysis_type == "bayesian":
            result = run_bayesian_analysis(df, analysis_id, config)
        elif analysis_type == "reliability":
            result = run_reliability_analysis(df, analysis_id, config)
        else:
            return JsonResponse({"error": f"Unknown analysis type: {analysis_type}"}, status=400)

        latency = int((time.time() - start_time) * 1000)
        log_agent_action(request.user, "analysis", analysis_id, latency_ms=latency)

        logger.info(f"Analysis complete: {analysis_id}, plots: {len(result.get('plots', []))}, summary length: {len(result.get('summary', ''))}")

        # Link results as evidence to a Problem (if problem_id provided)
        problem_id = body.get("problem_id")
        if problem_id:
            try:
                from .views import add_finding_to_problem

                # Use guide_observation if set, otherwise build from analysis_id + summary
                observation = result.get("guide_observation", "")
                if not observation:
                    summary_text = result.get("summary", "")
                    # Strip color tags and truncate for evidence summary
                    import re
                    clean = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary_text)
                    observation = f"DSW {analysis_id}: {clean[:200]}" if clean else f"DSW analysis: {analysis_id}"

                evidence_type = {
                    "stats": "data_analysis",
                    "ml": "data_analysis",
                    "bayesian": "data_analysis",
                    "spc": "data_analysis",
                    "viz": "observation",
                }.get(analysis_type, "data_analysis")

                evidence = add_finding_to_problem(
                    user=request.user,
                    problem_id=problem_id,
                    summary=observation,
                    evidence_type=evidence_type,
                    source=f"DSW ({analysis_id})",
                )
                if evidence:
                    result["problem_updated"] = True
                    result["evidence_id"] = evidence["id"]
            except Exception as e:
                logger.warning(f"Could not link DSW result to problem {problem_id}: {e}")

        # Save result to database for A3/method import if requested
        if save_result or project_id:
            try:
                result_id = f"dsw_{uuid.uuid4().hex[:8]}"
                from core.models import Project

                project = None
                if project_id:
                    try:
                        project = Project.objects.get(id=project_id, user=request.user)
                    except Project.DoesNotExist:
                        pass

                DSWResult.objects.create(
                    id=result_id,
                    user=request.user,
                    result_type=f"{analysis_type}_{analysis_id}",
                    data=json.dumps({
                        "analysis_type": analysis_type,
                        "analysis_id": analysis_id,
                        "config": config,
                        "summary": result.get("summary", ""),
                        "guide_observation": result.get("guide_observation", ""),
                        "plots_count": len(result.get("plots", [])),
                    }),
                    project=project,
                    title=result_title or f"{analysis_id.replace('_', ' ').title()} Analysis",
                )
                result["result_id"] = result_id
            except Exception as e:
                logger.warning(f"Could not save DSW result: {e}")

        return JsonResponse(result)

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        log_agent_action(request.user, "analysis", analysis_id, success=False, error_message=str(e))
        return JsonResponse({"error": str(e)}, status=500)


def run_statistical_analysis(df, analysis_id, config):
    """Run statistical analysis."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "descriptive":
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Get selected vars from config, fall back to all numeric
        vars_from_config = config.get("vars", [])
        if isinstance(vars_from_config, list) and len(vars_from_config) > 0:
            vars_to_analyze = [v for v in vars_from_config if v in df.columns]
        else:
            vars_to_analyze = numeric_cols

        if not vars_to_analyze:
            result["summary"] = "No numeric variables found to analyze."
            return result

        desc = df[vars_to_analyze].describe().to_string()
        result["summary"] = f"Descriptive Statistics:\n\n{desc}"

        # Add explicit statistics for Synara integration
        result["statistics"] = {}
        for var in vars_to_analyze:
            col = df[var].dropna()
            result["statistics"][f"mean({var})"] = float(col.mean())
            result["statistics"][f"std({var})"] = float(col.std())
            result["statistics"][f"min({var})"] = float(col.min())
            result["statistics"][f"max({var})"] = float(col.max())
            result["statistics"][f"median({var})"] = float(col.median())
            result["statistics"][f"n({var})"] = int(len(col))

        # Add histogram for each variable
        for var in vars_to_analyze:
            try:
                data = df[var].dropna().tolist()
                if len(data) > 0:
                    result["plots"].append({
                        "title": f"Distribution of {var}",
                        "data": [{
                            "type": "histogram",
                            "x": data,
                            "name": var,
                            "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                        }],
                        "layout": {"height": 200}
                    })
            except Exception as plot_err:
                logger.warning(f"Could not create histogram for {var}: {plot_err}")

    elif analysis_id == "ttest":
        # One-sample t-test
        var1 = config.get("var1")
        mu = float(config.get("mu", 0))
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        n = len(x)
        stat, pval = stats.ttest_1samp(x, mu)

        # Confidence interval
        se = x.std() / np.sqrt(n)
        ci = stats.t.interval(conf/100, n-1, loc=x.mean(), scale=se)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Hypothesized mean:<</COLOR>> {mu}\n\n"
        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {x.mean():.4f}\n"
        summary += f"  Std Dev: {x.std():.4f}\n"
        summary += f"  SE Mean: {se:.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  {conf}% CI: ({ci[0]:.4f}, {ci[1]:.4f})\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Reject H₀: Mean differs significantly from {mu} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>Fail to reject H₀ (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"One-sample t-test with p-value {pval:.4f}. " + ("Significant." if pval < alpha else "Not significant.")

        # Explicit statistics for Synara
        result["statistics"] = {
            f"mean({var1})": float(x.mean()),
            f"std({var1})": float(x.std()),
            f"n({var1})": int(n),
            "t_statistic": float(stat),
            "p_value": float(pval),
            f"ci_lower({var1})": float(ci[0]),
            f"ci_upper({var1})": float(ci[1]),
        }

        # Histogram with mean line and CI band
        result["plots"].append({
            "title": f"Distribution of {var1} with Mean & {conf}% CI",
            "data": [
                {"type": "histogram", "x": x.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": var1},
                {"type": "scatter", "x": [float(x.mean()), float(x.mean())], "y": [0, n/5], "mode": "lines", "line": {"color": "#4a90d9", "width": 2}, "name": f"Mean ({x.mean():.3f})"},
                {"type": "scatter", "x": [mu, mu], "y": [0, n/5], "mode": "lines", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}, "name": f"H₀ μ = {mu}"},
                {"type": "scatter", "x": [float(ci[0]), float(ci[1]), float(ci[1]), float(ci[0]), float(ci[0])], "y": [0, 0, n/5, n/5, 0], "fill": "toself", "fillcolor": "rgba(74, 144, 217, 0.15)", "line": {"color": "rgba(74, 144, 217, 0.3)"}, "name": f"{conf}% CI"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": var1}, "yaxis": {"title": "Count"}, "barmode": "overlay"}
        })

    elif analysis_id == "ttest2":
        # Two-sample t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        y = df[var2].dropna()
        stat, pval = stats.ttest_ind(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TWO-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1:<</COLOR>> {var1} (n = {len(x)}, mean = {x.mean():.4f}, std = {x.std():.4f})\n"
        summary += f"<<COLOR:text>>Sample 2:<</COLOR>> {var2} (n = {len(y)}, mean = {y.mean():.4f}, std = {y.std():.4f})\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Difference of means: {x.mean() - y.mean():.4f}\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Means are significantly different (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Two-sample t-test with p-value {pval:.4f}."
        result["statistics"] = {
            f"mean({var1})": float(x.mean()),
            f"mean({var2})": float(y.mean()),
            "difference": float(x.mean() - y.mean()),
            "t_statistic": float(stat),
            "p_value": float(pval),
        }

        # Side-by-side box plots
        result["plots"].append({
            "title": f"Comparison: {var1} vs {var2}",
            "data": [
                {"type": "box", "y": x.tolist(), "name": var1, "marker": {"color": "#4a9f6e"}, "boxpoints": "outliers"},
                {"type": "box", "y": y.tolist(), "name": var2, "marker": {"color": "#4a90d9"}, "boxpoints": "outliers"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "yaxis": {"title": "Value"}}
        })

    elif analysis_id == "paired_t":
        # Paired t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        y = df[var2].dropna()
        # Align samples
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        diff = x - y
        stat, pval = stats.ttest_rel(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PAIRED T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1 (Before):<</COLOR>> {var1}\n"
        summary += f"<<COLOR:text>>Sample 2 (After):<</COLOR>> {var2}\n"
        summary += f"<<COLOR:text>>Pairs:<</COLOR>> {len(x)}\n\n"
        summary += f"<<COLOR:text>>Difference Statistics:<</COLOR>>\n"
        summary += f"  Mean difference: {diff.mean():.4f}\n"
        summary += f"  Std of differences: {diff.std():.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Significant difference between paired observations (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Paired t-test with p-value {pval:.4f}."
        result["statistics"] = {
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std()),
            "n_pairs": int(len(x)),
            "t_statistic": float(stat),
            "p_value": float(pval),
        }

        # Histogram of differences
        result["plots"].append({
            "title": f"Distribution of Differences ({var1} − {var2})",
            "data": [
                {"type": "histogram", "x": diff.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Differences"},
                {"type": "scatter", "x": [float(diff.mean()), float(diff.mean())], "y": [0, len(x)/5], "mode": "lines", "line": {"color": "#4a90d9", "width": 2}, "name": f"Mean diff ({diff.mean():.3f})"},
                {"type": "scatter", "x": [0, 0], "y": [0, len(x)/5], "mode": "lines", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}, "name": "Zero (no diff)"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Difference"}, "yaxis": {"title": "Count"}}
        })

    elif analysis_id == "anova":
        response = config.get("response")
        # Support both 'factor' (single) and 'factors' (list) for backwards compatibility
        factor = config.get("factor")
        factors = config.get("factors", [])
        if factor and not factors:
            factors = [factor]

        if len(factors) >= 1:
            # One-way ANOVA
            factor_col = factors[0]
            groups = [df[df[factor_col] == level][response].dropna() for level in df[factor_col].unique()]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups

            stat, pval = stats.f_oneway(*groups)

            # Calculate group statistics
            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>ONE-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"

            summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
            for level in df[factor_col].unique():
                grp = df[df[factor_col] == level][response].dropna()
                summary += f"  {level}: n={len(grp)}, mean={grp.mean():.4f}, std={grp.std():.4f}\n"

            summary += f"\n<<COLOR:text>>ANOVA Results:<</COLOR>>\n"
            summary += f"  F-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n\n"

            if pval < 0.05:
                summary += f"<<COLOR:good>>Significant difference between groups (p < 0.05)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Run post-hoc tests (Tukey HSD, Games-Howell, or Dunnett) to identify which groups differ.<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>"

            result["summary"] = summary

            # Box plot
            result["plots"].append({
                "title": f"{response} by {factor_col}",
                "data": [{
                    "type": "box",
                    "y": df[response].tolist(),
                    "x": df[factor_col].astype(str).tolist(),
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })
        else:
            result["summary"] = "Please select a factor column for ANOVA."

    elif analysis_id == "anova2":
        # Two-way ANOVA
        response = config.get("response")
        factor_a = config.get("factor_a")
        factor_b = config.get("factor_b")

        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            # Build formula
            formula = f'{response} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>TWO-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor A:<</COLOR>> {factor_a}\n"
            summary += f"<<COLOR:highlight>>Factor B:<</COLOR>> {factor_b}\n\n"

            summary += f"<<COLOR:text>>ANOVA Table:<</COLOR>>\n"
            summary += anova_table.to_string() + "\n\n"

            # Interpret results
            for idx in anova_table.index:
                if 'PR(>F)' in anova_table.columns:
                    p = anova_table.loc[idx, 'PR(>F)']
                    if not np.isnan(p):
                        sig = "<<COLOR:good>>*<</COLOR>>" if p < 0.05 else ""
                        summary += f"{idx}: p = {p:.4f} {sig}\n"

            result["summary"] = summary

            # Interaction plot
            means = df.groupby([factor_a, factor_b])[response].mean().unstack()
            traces = []
            for col in means.columns:
                traces.append({
                    "type": "scatter",
                    "x": means.index.astype(str).tolist(),
                    "y": means[col].tolist(),
                    "mode": "lines+markers",
                    "name": str(col)
                })

            result["plots"].append({
                "title": "Interaction Plot",
                "data": traces,
                "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": factor_a}}
            })

        except ImportError:
            result["summary"] = "Two-way ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Two-way ANOVA error: {str(e)}"

    elif analysis_id == "regression":
        response = config.get("response")
        predictors = config.get("predictors", [])
        degree = int(config.get("degree", 1))
        interactions = config.get("interactions", "none")

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error

        X_raw = df[predictors].dropna()
        y = df[response].loc[X_raw.index]
        n = len(y)

        # Build feature names and transform data
        feature_names = list(predictors)

        if degree > 1 or interactions == "all":
            # Use PolynomialFeatures for polynomial and/or interaction terms
            include_interaction = interactions == "all"
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=(degree == 1 and include_interaction)
            )
            X = poly.fit_transform(X_raw)

            # Generate readable feature names
            feature_names = []
            for powers in poly.powers_:
                parts = []
                for i, power in enumerate(powers):
                    if power == 0:
                        continue
                    elif power == 1:
                        parts.append(predictors[i])
                    else:
                        parts.append(f"{predictors[i]}^{power}")
                if parts:
                    feature_names.append("·".join(parts) if len(parts) > 1 else parts[0])
        else:
            X = X_raw.values

        p = X.shape[1]  # Number of features after transformation

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate comprehensive statistics
        residuals = y.values - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.values - np.mean(y.values))**2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = ss_res / (n - p - 1) if n > p + 1 else 1e-10
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate standard errors and t-statistics
        X_with_const = np.column_stack([np.ones(n), X])
        try:
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(var_coef)
            coefs = np.concatenate([[model.intercept_], model.coef_])
            t_stats = coefs / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        except:
            se = np.zeros(p + 1)
            t_stats = np.zeros(p + 1)
            p_values = np.ones(p + 1)
            coefs = np.concatenate([[model.intercept_], model.coef_])

        # F-statistic
        f_stat = (ss_tot - ss_res) / p / mse if p > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1) if p > 0 else 1

        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals)**2) / ss_res

        # Build colored summary output
        model_type = "Linear" if degree == 1 else "Quadratic" if degree == 2 else "Cubic"
        if interactions == "all":
            model_type += " + Interactions"

        summary = "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                          {model_type.upper()} REGRESSION RESULTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n\n"

        summary += f"<<COLOR:dim>>Dep. Variable:<</COLOR>>    <<COLOR:text>>{response}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Observations:<</COLOR>> <<COLOR:text>>{n}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Features:<</COLOR>>     <<COLOR:text>>{p}<</COLOR>> (from {len(predictors)} predictors)\n"
        summary += f"<<COLOR:dim>>Model:<</COLOR>>            <<COLOR:text>>OLS - {model_type}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                               MODEL FIT<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        r2_color = "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger"
        summary += f"<<COLOR:dim>>Residual std. error:<</COLOR>> <<COLOR:text>>{np.sqrt(mse):.4f}<</COLOR>> on <<COLOR:text>>{n - p - 1}<</COLOR>> degrees of freedom\n"
        summary += f"<<COLOR:dim>>Multiple R-squared:<</COLOR>>  <<COLOR:{r2_color}>>{r2:.4f}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Adjusted R-squared:<</COLOR>> <<COLOR:{r2_color}>>{adj_r2:.4f}<</COLOR>>\n"

        f_color = "success" if f_pvalue < 0.05 else "warning" if f_pvalue < 0.1 else "danger"
        summary += f"<<COLOR:dim>>F-statistic:<</COLOR>>        <<COLOR:text>>{f_stat:.2f}<</COLOR>> on <<COLOR:text>>{p}<</COLOR>> and <<COLOR:text>>{n - p - 1}<</COLOR>> DF\n"
        summary += f"<<COLOR:dim>>p-value:<</COLOR>>            <<COLOR:{f_color}>>{f_pvalue:.4e}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Durbin-Watson:<</COLOR>>      <<COLOR:text>>{dw:.3f}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += "<<COLOR:accent>>                              COEFFICIENTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += "<<COLOR:dim>>                            Estimate    Std.Err    t value    Pr(>|t|)     <</COLOR>>\n"

        names = ["(Intercept)"] + feature_names
        non_sig_predictors = []
        for i, name in enumerate(names):
            pv = p_values[i]
            sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else ".  " if pv < 0.1 else "   "
            p_color = "success" if pv < 0.05 else "warning" if pv < 0.1 else "dim"
            summary += f"<<COLOR:text>>{name:<24}<</COLOR>> {coefs[i]:>10.4f}   {se[i]:>9.4f}   {t_stats[i]:>8.3f}    <<COLOR:{p_color}>>{pv:>9.4f}  {sig}<</COLOR>>\n"
            if i > 0 and pv >= 0.1:  # Track non-significant predictors (excluding intercept)
                non_sig_predictors.append(name)

        summary += "<<COLOR:dim>>---<</COLOR>>\n"
        summary += "<<COLOR:dim>>Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1<</COLOR>>\n\n"

        # Diagnostics interpretation
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                            DIAGNOSTICS SUMMARY<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        # Interpret R²
        if r2 > 0.7:
            summary += "<<COLOR:success>>✓ Good fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        elif r2 > 0.4:
            summary += "<<COLOR:warning>>◐ Moderate fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        else:
            summary += "<<COLOR:danger>>✗ Poor fit:<</COLOR>> Model explains only {:.1f}% of variance\n".format(r2*100)

        # Interpret Durbin-Watson
        if 1.5 < dw < 2.5:
            summary += "<<COLOR:success>>✓ No autocorrelation:<</COLOR>> Durbin-Watson ≈ 2\n"
        else:
            summary += "<<COLOR:warning>>◐ Possible autocorrelation:<</COLOR>> Durbin-Watson = {:.2f}\n".format(dw)

        # Interpret F-test
        if f_pvalue < 0.05:
            summary += "<<COLOR:success>>✓ Model significant:<</COLOR>> F-test p < 0.05\n"
        else:
            summary += "<<COLOR:danger>>✗ Model not significant:<</COLOR>> F-test p = {:.3f}\n".format(f_pvalue)

        # Calculate diagnostic values first (needed for suggestions and plots)
        std_residuals = residuals / np.std(residuals)

        # Leverage (hat values)
        try:
            H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
            leverage = np.diag(H)
        except:
            leverage = np.ones(n) / n

        # Cook's distance
        cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage + 1e-10)**2)

        # Square root of standardized residuals for scale-location
        sqrt_std_resid = np.sqrt(np.abs(std_residuals))

        # Model improvement suggestions
        suggestions = []

        # Low R² suggestions
        if r2 < 0.4:
            suggestions.append("Add more predictors or interaction terms (X1*X2)")
            suggestions.append("Try polynomial terms (X², X³) for non-linear relationships")
            suggestions.append("Check for outliers that may be distorting the fit")
        elif r2 < 0.7:
            suggestions.append("Consider adding interaction terms or polynomial features")

        # Non-significant predictors
        if non_sig_predictors:
            if len(non_sig_predictors) <= 3:
                suggestions.append(f"Consider removing non-significant: {', '.join(non_sig_predictors)}")
            else:
                suggestions.append(f"Consider removing {len(non_sig_predictors)} non-significant predictors")

        # Autocorrelation
        if dw < 1.5:
            suggestions.append("Positive autocorrelation detected - consider time series methods or lag terms")
        elif dw > 2.5:
            suggestions.append("Negative autocorrelation detected - check data ordering")

        # Model not significant
        if f_pvalue >= 0.05:
            suggestions.append("Model not significant - try different predictors or check data quality")

        # High leverage points
        high_leverage = int(np.sum(leverage > 2 * (p + 1) / n)) if n > 0 else 0
        if high_leverage > 0:
            suggestions.append(f"{high_leverage} high-leverage points detected - check for influential outliers")

        # Large Cook's distance
        high_cooks = int(np.sum(cooks_d > 4 / n)) if n > 0 else 0
        if high_cooks > 0:
            suggestions.append(f"{high_cooks} influential observations (Cook's D) - consider robust regression")

        if suggestions:
            summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += "<<COLOR:accent>>                          IMPROVEMENT SUGGESTIONS<</COLOR>>\n"
            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            for i, sug in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                summary += f"<<COLOR:warning>>{i}.<</COLOR>> <<COLOR:text>>{sug}<</COLOR>>\n"

        result["summary"] = summary

        # Create 4-panel diagnostic plots

        # 1. Residuals vs Fitted
        result["plots"].append({
            "title": "1. Residuals vs Fitted",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": residuals.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [min(y_pred), max(y_pred)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Zero line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "Residuals"}}
        })

        # 2. Normal Q-Q Plot
        sorted_std_resid = np.sort(std_residuals)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_std_resid)))
        result["plots"].append({
            "title": "2. Normal Q-Q",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [-3, 3],
                    "y": [-3, 3],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Normal line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # 3. Scale-Location
        result["plots"].append({
            "title": "3. Scale-Location",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": sqrt_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "√|Std. Residuals|"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "√|Standardized residuals|"}}
        })

        # 4. Residuals vs Leverage
        result["plots"].append({
            "title": "4. Residuals vs Leverage",
            "data": [
                {
                    "type": "scatter",
                    "x": leverage.tolist(),
                    "y": std_residuals.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": cooks_d.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [0.5, "#e89547"], [1, "#9f4a4a"]],
                        "size": 6,
                        "colorbar": {"title": "Cook's D", "len": 0.5}
                    },
                    "name": "Observations"
                },
                {
                    "type": "scatter",
                    "x": [0, max(leverage)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "showlegend": False
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Leverage"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # Explicit statistics for Synara integration
        result["statistics"] = {
            "R²": float(r2),
            "Adj_R²": float(adj_r2),
            "RMSE": float(rmse),
            "F_statistic": float(f_stat),
            "F_p_value": float(f_pvalue),
            "n": int(n),
            "predictors": int(p),
        }
        # Add coefficients as statistics
        for i, feat in enumerate(feature_names):
            result["statistics"][f"coef({feat})"] = float(model.coef_[i])
            if i < len(p_values) - 1:
                result["statistics"][f"p_value({feat})"] = float(p_values[i + 1])

    elif analysis_id == "correlation":
        vars_list = config.get("vars", [])
        method = config.get("method", "pearson")

        if vars_list:
            numeric_cols = [v for v in vars_list if v in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[numeric_cols].corr(method=method)

        # Build formatted output
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CORRELATION ANALYSIS ({method.upper()})<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(numeric_cols)}\n\n"
        summary += "<<COLOR:text>>Correlation Matrix:<</COLOR>>\n"
        summary += corr_matrix.to_string() + "\n"

        result["summary"] = summary

        # Heatmap
        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{
                "type": "heatmap",
                "z": corr_matrix.values.tolist(),
                "x": numeric_cols,
                "y": numeric_cols,
                "colorscale": "RdBu",
                "zmid": 0,
                "text": [[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                "texttemplate": "%{text}",
                "textfont": {"size": 10}
            }],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "normality":
        var = config.get("var")
        test_type = config.get("test", "anderson")

        x = df[var].dropna()
        n = len(x)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NORMALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n\n"

        if test_type == "anderson":
            stat_result = stats.anderson(x)
            summary += f"<<COLOR:text>>Anderson-Darling Test<</COLOR>>\n"
            summary += f"  Statistic: {stat_result.statistic:.4f}\n"
            summary += f"  Critical Values:\n"
            for cv, sl in zip(stat_result.critical_values, stat_result.significance_level):
                marker = "<<COLOR:good>>✓<</COLOR>>" if stat_result.statistic < cv else "<<COLOR:bad>>✗<</COLOR>>"
                summary += f"    {marker} {sl}%: {cv:.4f}\n"
        elif test_type == "shapiro":
            stat, pval = stats.shapiro(x)
            summary += f"<<COLOR:text>>Shapiro-Wilk Test<</COLOR>>\n"
            summary += f"  W-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"
        elif test_type == "ks":
            stat, pval = stats.kstest(x, 'norm', args=(x.mean(), x.std()))
            summary += f"<<COLOR:text>>Kolmogorov-Smirnov Test<</COLOR>>\n"
            summary += f"  D-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Q-Q plot
        sorted_data = np.sort(x)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(x)))

        result["plots"].append({
            "title": f"Normal Q-Q Plot: {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_quantiles.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": [theoretical_quantiles.min(), theoretical_quantiles.max()],
                    "y": [sorted_data.min(), sorted_data.max()],
                    "mode": "lines",
                    "line": {"color": "#ff7675", "dash": "dash"},
                    "name": "Reference"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Sample Quantiles"}
            }
        })

        # Histogram with normal curve overlay
        x_range = np.linspace(float(x.min()), float(x.max()), 100)
        normal_pdf = stats.norm.pdf(x_range, x.mean(), x.std())
        bin_width = (x.max() - x.min()) / min(30, max(5, int(np.sqrt(n))))
        normal_scaled = normal_pdf * n * bin_width

        result["plots"].append({
            "title": f"Histogram with Normal Curve: {var}",
            "data": [
                {"type": "histogram", "x": x.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
                {"type": "scatter", "x": x_range.tolist(), "y": normal_scaled.tolist(), "mode": "lines", "line": {"color": "#d94a4a", "width": 2}, "name": "Normal fit"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": var}, "yaxis": {"title": "Count"}, "barmode": "overlay"}
        })

    elif analysis_id == "chi2":
        row_var = config.get("row_var") or config.get("var1") or config.get("var")
        col_var = config.get("col_var") or config.get("var2") or config.get("group_var")

        # Create contingency table
        contingency = pd.crosstab(df[row_var], df[col_var])
        chi2, pval, dof, expected = stats.chi2_contingency(contingency)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHI-SQUARE TEST FOR INDEPENDENCE<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Row Variable:<</COLOR>> {row_var}\n"
        summary += f"<<COLOR:highlight>>Column Variable:<</COLOR>> {col_var}\n\n"

        summary += f"<<COLOR:text>>Contingency Table (Observed):<</COLOR>>\n"
        summary += contingency.to_string() + "\n\n"

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Chi-square statistic: {chi2:.4f}\n"
        summary += f"  Degrees of freedom: {dof}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>Variables are significantly associated (p < 0.05)<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant association found (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Heatmap of observed counts
        result["plots"].append({
            "title": f"Contingency Table: {row_var} × {col_var}",
            "data": [{
                "type": "heatmap",
                "z": contingency.values.tolist(),
                "x": contingency.columns.astype(str).tolist(),
                "y": contingency.index.astype(str).tolist(),
                "colorscale": "Blues",
                "text": contingency.values.tolist(),
                "texttemplate": "%{text}",
                "textfont": {"size": 12}
            }],
            "layout": {"template": "plotly_dark", "height": 300}
        })

    elif analysis_id == "prop_1sample":
        """
        One-Proportion Z-Test — test if an observed proportion equals a hypothesized value.
        Uses normal approximation to the binomial; reports Z, p-value, and Wilson CI.
        """
        var = config.get("var") or config.get("var1")
        event = config.get("event")  # value to count as success
        p0 = float(config.get("p0", 0.5))  # hypothesized proportion
        alt = config.get("alternative", "two-sided")  # two-sided, greater, less
        alpha = 1 - float(config.get("conf", 95)) / 100

        col = df[var].dropna()
        n = len(col)
        if event is not None and str(event) != "":
            x = int((col.astype(str) == str(event)).sum())
        else:
            # If binary 0/1, count 1s
            x = int((col == 1).sum()) if col.dtype in ['int64', 'float64'] else int(col.value_counts().iloc[0])
        p_hat = x / n if n > 0 else 0

        # Z-test
        se0 = np.sqrt(p0 * (1 - p0) / n) if n > 0 else 1
        z_stat = (p_hat - p0) / se0 if se0 > 0 else 0

        if alt == "greater":
            p_val = float(1 - stats.norm.cdf(z_stat))
        elif alt == "less":
            p_val = float(stats.norm.cdf(z_stat))
        else:
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Wilson confidence interval
        z_crit = stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z_crit**2 / n
        center = (p_hat + z_crit**2 / (2 * n)) / denom
        margin = z_crit * np.sqrt((p_hat * (1 - p_hat) + z_crit**2 / (4 * n)) / n) / denom
        ci_lo, ci_hi = max(0, center - margin), min(1, center + margin)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-PROPORTION Z-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        if event is not None and str(event) != "":
            summary += f"<<COLOR:highlight>>Event:<</COLOR>> {event}\n"
        summary += f"<<COLOR:highlight>>H₀:<</COLOR>> p = {p0}\n"
        summary += f"<<COLOR:highlight>>H₁:<</COLOR>> p {'≠' if alt == 'two-sided' else '>' if alt == 'greater' else '<'} {p0}\n\n"
        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  N: {n}\n"
        summary += f"  Successes: {x}\n"
        summary += f"  p̂: {p_hat:.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Z-statistic: {z_stat:.4f}\n"
        summary += f"  p-value: {p_val:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI (Wilson): ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Proportion differs significantly from {p0} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference from {p0} (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Proportion bar with CI and reference line
        result["plots"].append({
            "data": [
                {"type": "bar", "x": ["Observed"], "y": [p_hat], "marker": {"color": "#4a9f6e"},
                 "error_y": {"type": "data", "symmetric": False, "array": [ci_hi - p_hat], "arrayminus": [p_hat - ci_lo], "color": "#5a6a5a"},
                 "name": f"p̂ = {p_hat:.4f}"}
            ],
            "layout": {
                "title": "Observed Proportion vs Hypothesized",
                "yaxis": {"title": "Proportion", "range": [0, min(1.05, max(ci_hi + 0.1, p0 + 0.2))]},
                "shapes": [{"type": "line", "x0": -0.5, "x1": 0.5, "y0": p0, "y1": p0,
                            "line": {"color": "#e89547", "dash": "dash", "width": 2}}],
                "annotations": [{"x": 0.5, "y": p0, "text": f"H₀: p={p0}", "showarrow": False, "xanchor": "left", "font": {"color": "#e89547"}}],
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"1-prop Z-test: p̂={p_hat:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "n": n, "successes": x, "p_hat": p_hat, "p0": p0,
            "z_statistic": float(z_stat), "p_value": p_val,
            "ci_lower": ci_lo, "ci_upper": ci_hi, "alternative": alt
        }

    elif analysis_id == "prop_2sample":
        """
        Two-Proportion Z-Test — compare proportions between two groups.
        Tests H₀: p₁ = p₂. Reports pooled Z, individual CIs, and difference CI.
        """
        var = config.get("var") or config.get("var1")
        group_var = config.get("group_var") or config.get("var2") or config.get("factor")
        event = config.get("event")
        alt = config.get("alternative", "two-sided")
        alpha = 1 - float(config.get("conf", 95)) / 100

        data = df[[var, group_var]].dropna()
        groups = sorted(data[group_var].unique().tolist(), key=str)
        if len(groups) != 2:
            result["summary"] = f"Two-proportion test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = data[data[group_var] == groups[0]][var]
        g2 = data[data[group_var] == groups[1]][var]
        n1, n2 = len(g1), len(g2)

        if event is not None and str(event) != "":
            x1 = int((g1.astype(str) == str(event)).sum())
            x2 = int((g2.astype(str) == str(event)).sum())
        else:
            x1 = int((g1 == 1).sum()) if g1.dtype in ['int64', 'float64'] else int(g1.value_counts().iloc[0])
            x2 = int((g2 == 1).sum()) if g2.dtype in ['int64', 'float64'] else int(g2.value_counts().iloc[0])

        p1 = x1 / n1 if n1 > 0 else 0
        p2 = x2 / n2 if n2 > 0 else 0
        p_pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0

        se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) if (n1 > 0 and n2 > 0) else 1
        z_stat = (p1 - p2) / se_pooled if se_pooled > 0 else 0

        if alt == "greater":
            p_val = float(1 - stats.norm.cdf(z_stat))
        elif alt == "less":
            p_val = float(stats.norm.cdf(z_stat))
        else:
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        # Difference CI (unpooled SE)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) if (n1 > 0 and n2 > 0) else 0
        diff = p1 - p2
        ci_lo = diff - z_crit * se_diff
        ci_hi = diff + z_crit * se_diff

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TWO-PROPORTION Z-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {group_var}\n\n"

        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  {'Group':<15} {'N':>6} {'Events':>8} {'Proportion':>12}\n"
        summary += f"  {'─' * 45}\n"
        summary += f"  {str(groups[0]):<15} {n1:>6} {x1:>8} {p1:>12.4f}\n"
        summary += f"  {str(groups[1]):<15} {n2:>6} {x2:>8} {p2:>12.4f}\n\n"
        summary += f"<<COLOR:text>>Difference (p₁ − p₂):<</COLOR>> {diff:.4f}\n"
        summary += f"<<COLOR:text>>{100*(1-alpha):.0f}% CI for difference:<</COLOR>> ({ci_lo:.4f}, {ci_hi:.4f})\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Z-statistic: {z_stat:.4f}\n"
        summary += f"  p-value: {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Proportions differ significantly (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in proportions (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Side-by-side bar chart
        result["plots"].append({
            "data": [
                {"type": "bar", "x": [str(groups[0]), str(groups[1])], "y": [p1, p2],
                 "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                 "text": [f"{p1:.3f}", f"{p2:.3f}"], "textposition": "outside"}
            ],
            "layout": {
                "title": "Proportions by Group",
                "yaxis": {"title": "Proportion", "range": [0, max(p1, p2) * 1.3 + 0.05]},
                "template": "plotly_white"
            }
        })

        # Difference CI plot
        result["plots"].append({
            "data": [{
                "type": "scatter", "x": [diff], "y": ["p₁ − p₂"], "mode": "markers",
                "marker": {"size": 12, "color": "#4a9f6e"},
                "error_x": {"type": "data", "symmetric": False, "array": [ci_hi - diff], "arrayminus": [diff - ci_lo], "color": "#5a6a5a"}
            }],
            "layout": {
                "title": f"Difference in Proportions ({100*(1-alpha):.0f}% CI)",
                "xaxis": {"title": "p₁ − p₂", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": 0.5, "line": {"color": "#e89547", "dash": "dash"}}],
                "height": 200, "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"2-prop Z-test: p₁={p1:.4f} vs p₂={p2:.4f}, Z={z_stat:.3f}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "n1": n1, "n2": n2, "x1": x1, "x2": x2, "p1": p1, "p2": p2,
            "difference": diff, "z_statistic": float(z_stat), "p_value": p_val,
            "ci_lower": ci_lo, "ci_upper": ci_hi, "alternative": alt
        }

    elif analysis_id == "fisher_exact":
        """
        Fisher's Exact Test — exact test for 2×2 contingency tables.
        Preferred over chi-square when expected cell counts are small (<5).
        Reports odds ratio, exact p-value, and odds ratio CI.
        """
        var1 = config.get("var") or config.get("var1") or config.get("row_var")
        var2 = config.get("var2") or config.get("group_var") or config.get("col_var")
        alt = config.get("alternative", "two-sided")
        alpha = 1 - float(config.get("conf", 95)) / 100

        ct = pd.crosstab(df[var1], df[var2])
        if ct.shape != (2, 2):
            result["summary"] = f"Fisher's exact test requires a 2×2 table. Got {ct.shape[0]}×{ct.shape[1]}. Ensure both variables have exactly 2 levels."
            return result

        table = ct.values
        odds_ratio, p_val = stats.fisher_exact(table, alternative=alt)

        # Odds ratio CI via log method
        a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
        if all(v > 0 for v in [a, b, c, d]):
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
            z_crit = stats.norm.ppf(1 - alpha / 2)
            or_ci_lo = np.exp(log_or - z_crit * se_log_or)
            or_ci_hi = np.exp(log_or + z_crit * se_log_or)
        else:
            or_ci_lo, or_ci_hi = 0, float('inf')

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>FISHER'S EXACT TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Row variable:<</COLOR>> {var1}\n"
        summary += f"<<COLOR:highlight>>Column variable:<</COLOR>> {var2}\n\n"

        summary += f"<<COLOR:text>>2×2 Contingency Table:<</COLOR>>\n"
        summary += f"  {'':>15} {str(ct.columns[0]):>10} {str(ct.columns[1]):>10}\n"
        summary += f"  {str(ct.index[0]):>15} {a:>10} {b:>10}\n"
        summary += f"  {str(ct.index[1]):>15} {c:>10} {d:>10}\n\n"
        summary += f"<<COLOR:text>>Results:<</COLOR>>\n"
        summary += f"  Odds Ratio: {odds_ratio:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI: ({or_ci_lo:.4f}, {or_ci_hi:.4f})\n"
        summary += f"  p-value (exact): {p_val:.4f}\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Significant association (p < {alpha}). Odds ratio ≠ 1.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant association (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        # Mosaic-like stacked bar
        col_labels = [str(c) for c in ct.columns]
        row_labels = [str(r) for r in ct.index]
        result["plots"].append({
            "data": [
                {"type": "bar", "x": col_labels, "y": [int(table[0, 0]), int(table[0, 1])], "name": row_labels[0], "marker": {"color": "#4a9f6e"}},
                {"type": "bar", "x": col_labels, "y": [int(table[1, 0]), int(table[1, 1])], "name": row_labels[1], "marker": {"color": "#4a90d9"}}
            ],
            "layout": {"title": "Contingency Table", "barmode": "stack", "yaxis": {"title": "Count"}, "template": "plotly_white"}
        })

        # Odds ratio forest-style plot
        if odds_ratio > 0 and or_ci_hi < 1e6:
            result["plots"].append({
                "data": [{
                    "type": "scatter", "x": [odds_ratio], "y": ["OR"], "mode": "markers",
                    "marker": {"size": 12, "color": "#4a9f6e"},
                    "error_x": {"type": "data", "symmetric": False, "array": [or_ci_hi - odds_ratio], "arrayminus": [odds_ratio - or_ci_lo], "color": "#5a6a5a"}
                }],
                "layout": {
                    "title": f"Odds Ratio ({100*(1-alpha):.0f}% CI)",
                    "xaxis": {"title": "Odds Ratio", "type": "log"},
                    "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": 0.5, "line": {"color": "#e89547", "dash": "dash"}}],
                    "height": 180, "template": "plotly_white"
                }
            })

        result["guide_observation"] = f"Fisher's exact: OR={odds_ratio:.3f}, p={p_val:.4f}. " + ("Significant association." if p_val < alpha else "No association.")
        result["statistics"] = {
            "odds_ratio": float(odds_ratio), "p_value": float(p_val),
            "or_ci_lower": float(or_ci_lo), "or_ci_upper": float(or_ci_hi),
            "table": table.tolist(), "alternative": alt
        }

    elif analysis_id == "poisson_1sample":
        """
        One-Sample Poisson Rate Test — test if an observed event rate equals a hypothesized rate.
        Uses exact Poisson test (conditional) or normal approximation for large counts.
        """
        var = config.get("var") or config.get("var1")
        rate0 = float(config.get("rate0", 1.0))  # hypothesized rate
        exposure = float(config.get("exposure", 1.0))  # time/area/units of exposure
        alt = config.get("alternative", "two-sided")
        alpha = 1 - float(config.get("conf", 95)) / 100

        col = df[var].dropna()
        total_count = float(col.sum())
        n = len(col)
        observed_rate = total_count / exposure if exposure > 0 else 0
        expected_count = rate0 * exposure

        # Exact Poisson test
        if alt == "greater":
            p_val = float(1 - stats.poisson.cdf(int(total_count) - 1, expected_count))
        elif alt == "less":
            p_val = float(stats.poisson.cdf(int(total_count), expected_count))
        else:
            # Two-sided: 2 * min(left, right)
            p_left = stats.poisson.cdf(int(total_count), expected_count)
            p_right = 1 - stats.poisson.cdf(int(total_count) - 1, expected_count)
            p_val = float(min(1.0, 2 * min(p_left, p_right)))

        # Exact Poisson CI for rate
        z_crit = stats.norm.ppf(1 - alpha / 2)
        if total_count > 0:
            ci_lo = stats.chi2.ppf(alpha / 2, 2 * total_count) / (2 * exposure)
            ci_hi = stats.chi2.ppf(1 - alpha / 2, 2 * (total_count + 1)) / (2 * exposure)
        else:
            ci_lo = 0
            ci_hi = stats.chi2.ppf(1 - alpha / 2, 2) / (2 * exposure)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-SAMPLE POISSON RATE TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>H₀:<</COLOR>> rate = {rate0}\n"
        summary += f"<<COLOR:highlight>>Exposure:<</COLOR>> {exposure}\n\n"
        summary += f"<<COLOR:text>>Sample Results:<</COLOR>>\n"
        summary += f"  Total count: {total_count:.0f}\n"
        summary += f"  Observed rate: {observed_rate:.4f}\n"
        summary += f"  Expected count (under H₀): {expected_count:.1f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  p-value (exact): {p_val:.4f}\n"
        summary += f"  {100*(1-alpha):.0f}% CI for rate: ({ci_lo:.4f}, {ci_hi:.4f})\n\n"

        if p_val < alpha:
            summary += f"<<COLOR:good>>Rate differs significantly from {rate0} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference from {rate0} (p ≥ {alpha})<</COLOR>>"

        result["summary"] = summary

        result["plots"].append({
            "data": [
                {"type": "bar", "x": ["Observed"], "y": [observed_rate], "marker": {"color": "#4a9f6e"},
                 "error_y": {"type": "data", "symmetric": False, "array": [ci_hi - observed_rate], "arrayminus": [observed_rate - ci_lo], "color": "#5a6a5a"},
                 "name": f"Rate = {observed_rate:.4f}"}
            ],
            "layout": {
                "title": "Observed Rate vs Hypothesized",
                "yaxis": {"title": "Rate"},
                "shapes": [{"type": "line", "x0": -0.5, "x1": 0.5, "y0": rate0, "y1": rate0,
                            "line": {"color": "#e89547", "dash": "dash", "width": 2}}],
                "annotations": [{"x": 0.5, "y": rate0, "text": f"H₀: λ={rate0}", "showarrow": False, "xanchor": "left", "font": {"color": "#e89547"}}],
                "template": "plotly_white"
            }
        })

        # Distribution plot
        x_range = list(range(max(0, int(total_count) - 15), int(total_count) + 16))
        pmf_vals = [float(stats.poisson.pmf(k, expected_count)) for k in x_range]
        result["plots"].append({
            "data": [
                {"type": "bar", "x": x_range, "y": pmf_vals, "name": f"Poisson(λ={expected_count:.1f})",
                 "marker": {"color": ["#d94a4a" if k == int(total_count) else "#4a9f6e" for k in x_range], "opacity": 0.7}}
            ],
            "layout": {
                "title": f"Poisson Distribution under H₀ (observed = {int(total_count)})",
                "xaxis": {"title": "Count"}, "yaxis": {"title": "Probability"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Poisson rate test: observed rate={observed_rate:.4f}, H₀ rate={rate0}, p={p_val:.4f}. " + ("Significant." if p_val < alpha else "Not significant.")
        result["statistics"] = {
            "total_count": total_count, "exposure": exposure,
            "observed_rate": observed_rate, "hypothesized_rate": rate0,
            "p_value": p_val, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "alternative": alt
        }

    # ── Power & Sample Size Calculators ──────────────────────────────────────

    elif analysis_id == "power_z":
        """
        Power / sample size for 1-sample Z-test.
        Given delta, sigma, alpha, and power → required n. Also produces a power curve.
        """
        delta = float(config.get("delta", 0.5))
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        d = abs(delta) / sigma  # standardised effect
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)
        n_req = math.ceil(((z_a + z_b) / d) ** 2) if d > 0 else 9999

        # Power curve over n
        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            ncp = d * math.sqrt(nn)
            if alt == "two-sided":
                pw = 1 - norm.cdf(z_a - ncp) + norm.cdf(-z_a - ncp)
            else:
                pw = 1 - norm.cdf(z_a - ncp)
            powers.append(pw)

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Sample Z (δ={delta}, σ={sigma})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Sample Z-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>Effect: δ = {delta}, σ = {sigma} → Cohen's d = {d:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, alternative = {alt}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-sample Z power: need n={n_req} for d={d:.3f} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "effect_size_d": d, "alpha": alpha, "power": target_power, "delta": delta, "sigma": sigma}

    elif analysis_id == "power_1prop":
        """
        Power / sample size for 1-proportion test.
        Given p0 (null), pa (alt), alpha, power → n.
        """
        p0 = float(config.get("p0", 0.5))
        pa = float(config.get("pa", 0.6))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)

        # Fleiss formula
        n_req = math.ceil(((z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(pa * (1 - pa))) / (pa - p0)) ** 2) if pa != p0 else 9999

        # Cohen's h
        h = abs(2 * (math.asin(math.sqrt(pa)) - math.asin(math.sqrt(p0))))

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            se0 = math.sqrt(p0 * (1 - p0) / nn)
            sea = math.sqrt(pa * (1 - pa) / nn)
            if alt == "two-sided":
                z_crit = z_a * se0
                pw = 1 - norm.cdf((p0 + z_crit - pa) / sea) + norm.cdf((p0 - z_crit - pa) / sea)
            elif alt == "greater":
                z_crit = z_a * se0
                pw = 1 - norm.cdf((p0 + z_crit - pa) / sea)
            else:
                z_crit = z_a * se0
                pw = norm.cdf((p0 - z_crit - pa) / sea)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Proportion (p₀={p0}, pₐ={pa})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: p = {p0}, H₁: p = {pa} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-prop power: need n={n_req} to detect p={pa} vs p₀={p0} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "p0": p0, "pa": pa, "cohens_h": h, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_2prop":
        """
        Power / sample size for 2-proportion test.
        Given p1, p2, alpha, power → n per group.
        """
        p1 = float(config.get("p1", 0.5))
        p2 = float(config.get("p2", 0.6))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        ratio = float(config.get("ratio", 1.0))  # n2/n1
        alt = config.get("alternative", "two-sided")

        from scipy.stats import norm
        if alt == "two-sided":
            z_a = norm.ppf(1 - alpha / 2)
        else:
            z_a = norm.ppf(1 - alpha)
        z_b = norm.ppf(target_power)
        p_bar = (p1 + ratio * p2) / (1 + ratio)
        h = abs(2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2))))

        numer = z_a * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
        n1 = math.ceil((numer / (p1 - p2)) ** 2) if p1 != p2 else 9999
        n2 = math.ceil(n1 * ratio)

        ns = list(range(2, max(n1 * 3, 50)))
        powers = []
        for nn in ns:
            nn2 = max(2, int(nn * ratio))
            se = math.sqrt(p1 * (1 - p1) / nn + p2 * (1 - p2) / nn2)
            if se > 0:
                z_stat = abs(p1 - p2) / se
                if alt == "two-sided":
                    pw = 1 - norm.cdf(z_a - z_stat) + norm.cdf(-z_a - z_stat)
                else:
                    pw = 1 - norm.cdf(z_a - z_stat)
            else:
                pw = 1.0
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n1], "y": [target_power], "mode": "markers", "name": f"n₁={n1}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 2-Proportion (p₁={p1}, p₂={p2})", "xaxis": {"title": "Sample Size per Group (n₁)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Proportion Test<</COLOR>>\n\n"
            f"<<COLOR:text>>p₁ = {p1}, p₂ = {p2} → Cohen's h = {h:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, ratio n₂/n₁ = {ratio}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n1}, n₂ = {n2} (total = {n1 + n2})<</COLOR>>\n"
        )
        result["guide_observation"] = f"2-prop power: need n₁={n1}, n₂={n2} for |Δp|={abs(p1 - p2):.3f} at power={target_power}."
        result["statistics"] = {"n1": n1, "n2": n2, "total_n": n1 + n2, "p1": p1, "p2": p2, "cohens_h": h, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_1variance":
        """
        Power / sample size for 1-variance chi-square test.
        Tests H₀: σ² = σ₀² vs H₁: σ² = σ₁².
        """
        sigma0 = float(config.get("sigma0", 1.0))
        sigma1 = float(config.get("sigma1", 1.5))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        alt = config.get("alternative", "two-sided")

        from scipy.stats import chi2 as chi2_dist
        ratio_sq = (sigma1 / sigma0) ** 2  # variance ratio σ₁²/σ₀²

        # Iterative search for n
        n_req = 2
        for n_try in range(2, 10000):
            df_val = n_try - 1
            if alt == "two-sided":
                lo = chi2_dist.ppf(alpha / 2, df_val)
                hi = chi2_dist.ppf(1 - alpha / 2, df_val)
                pw = chi2_dist.cdf(lo / ratio_sq, df_val) + 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
            elif alt == "greater":
                hi = chi2_dist.ppf(1 - alpha, df_val)
                pw = 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
            else:
                lo = chi2_dist.ppf(alpha, df_val)
                pw = chi2_dist.cdf(lo / ratio_sq, df_val)
            if pw >= target_power:
                n_req = n_try
                break
        else:
            n_req = 10000

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            df_val = nn - 1
            if alt == "two-sided":
                lo = chi2_dist.ppf(alpha / 2, df_val)
                hi = chi2_dist.ppf(1 - alpha / 2, df_val)
                pw = chi2_dist.cdf(lo / ratio_sq, df_val) + 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
            elif alt == "greater":
                hi = chi2_dist.ppf(1 - alpha, df_val)
                pw = 1 - chi2_dist.cdf(hi / ratio_sq, df_val)
            else:
                lo = chi2_dist.ppf(alpha, df_val)
                pw = chi2_dist.cdf(lo / ratio_sq, df_val)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 1-Variance (σ₀={sigma0}, σ₁={sigma1})", "xaxis": {"title": "Sample Size (n)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 1-Variance Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ = {sigma0}, H₁: σ = {sigma1} → ratio σ₁²/σ₀² = {ratio_sq:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"1-variance power: need n={n_req} to detect σ₁={sigma1} vs σ₀={sigma0} at power={target_power}."
        result["statistics"] = {"required_n": n_req, "sigma0": sigma0, "sigma1": sigma1, "variance_ratio": ratio_sq, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_2variance":
        """
        Power / sample size for 2-variance F-test.
        Tests H₀: σ₁² = σ₂² vs H₁: σ₁²/σ₂² = ratio.
        """
        var_ratio = float(config.get("variance_ratio", 2.0))  # σ₁²/σ₂²
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))
        ratio_n = float(config.get("ratio", 1.0))  # n2/n1
        alt = config.get("alternative", "two-sided")

        from scipy.stats import f as f_dist
        n_req = 2
        for n_try in range(2, 10000):
            n2_try = max(2, int(n_try * ratio_n))
            df1 = n_try - 1
            df2 = n2_try - 1
            if alt == "two-sided":
                f_lo = f_dist.ppf(alpha / 2, df1, df2)
                f_hi = f_dist.ppf(1 - alpha / 2, df1, df2)
                pw = f_dist.cdf(f_lo / var_ratio, df1, df2) + 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            else:
                f_hi = f_dist.ppf(1 - alpha, df1, df2)
                pw = 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            if pw >= target_power:
                n_req = n_try
                break
        else:
            n_req = 10000
        n2_req = max(2, int(n_req * ratio_n))

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            nn2 = max(2, int(nn * ratio_n))
            df1 = nn - 1
            df2 = nn2 - 1
            if alt == "two-sided":
                f_lo = f_dist.ppf(alpha / 2, df1, df2)
                f_hi = f_dist.ppf(1 - alpha / 2, df1, df2)
                pw = f_dist.cdf(f_lo / var_ratio, df1, df2) + 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            else:
                f_hi = f_dist.ppf(1 - alpha, df1, df2)
                pw = 1 - f_dist.cdf(f_hi / var_ratio, df1, df2)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n₁={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — 2-Variance F-Test (ratio={var_ratio})", "xaxis": {"title": "Sample Size per Group (n₁)"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2-Variance F-Test<</COLOR>>\n\n"
            f"<<COLOR:text>>H₀: σ₁²/σ₂² = 1, H₁: σ₁²/σ₂² = {var_ratio}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}, n₂/n₁ = {ratio_n}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n₁ = {n_req}, n₂ = {n2_req} (total = {n_req + n2_req})<</COLOR>>\n"
        )
        result["guide_observation"] = f"2-variance power: need n₁={n_req}, n₂={n2_req} for ratio={var_ratio} at power={target_power}."
        result["statistics"] = {"n1": n_req, "n2": n2_req, "total_n": n_req + n2_req, "variance_ratio": var_ratio, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_equivalence":
        """
        Power / sample size for equivalence test (TOST).
        Two one-sided tests to establish equivalence within ±margin.
        """
        delta = float(config.get("delta", 0.0))  # true difference
        margin = float(config.get("margin", 0.5))  # equivalence margin
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))

        from scipy.stats import norm
        z_a = norm.ppf(1 - alpha)

        # For TOST, power = P(reject both one-sided tests)
        n_req = 2
        for n_try in range(2, 10000):
            se = sigma * math.sqrt(2 / n_try)
            # Power of TOST ≈ Φ((margin - |delta|)/se - z_a)
            pw = norm.cdf((margin - abs(delta)) / se - z_a)
            if pw >= target_power:
                n_req = n_try
                break
        else:
            n_req = 10000

        ns = list(range(2, max(n_req * 3, 50)))
        powers = []
        for nn in ns:
            se = sigma * math.sqrt(2 / nn)
            pw = norm.cdf((margin - abs(delta)) / se - z_a)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": ns, "y": powers, "mode": "lines", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [n_req], "y": [target_power], "mode": "markers", "name": f"n/group={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Power Curve — Equivalence (TOST, margin=±{margin})", "xaxis": {"title": "Sample Size per Group"}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — Equivalence Test (TOST)<</COLOR>>\n\n"
            f"<<COLOR:text>>Equivalence margin: ±{margin}, true difference: {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required: n = {n_req} per group (total = {2 * n_req})<</COLOR>>\n"
        )
        result["guide_observation"] = f"Equivalence power: need n={n_req}/group for margin=±{margin}, δ={delta} at power={target_power}."
        result["statistics"] = {"n_per_group": n_req, "total_n": 2 * n_req, "margin": margin, "delta": delta, "sigma": sigma, "alpha": alpha, "power": target_power}

    elif analysis_id == "power_doe":
        """
        Power / sample size for 2-level factorial DOE.
        Given number of factors, effect size, sigma, reps → power.
        Or: given target power → required replicates.
        """
        n_factors = int(config.get("factors", 3))
        delta = float(config.get("delta", 1.0))  # minimum detectable effect
        sigma = float(config.get("sigma", 1.0))
        alpha = float(config.get("alpha", 0.05))
        target_power = float(config.get("power", 0.80))

        from scipy.stats import norm, t as t_dist
        n_runs_base = 2 ** n_factors  # full factorial runs

        # Find required replicates
        req_reps = 1
        for reps in range(1, 100):
            n_total = n_runs_base * reps
            df_error = n_total - n_runs_base  # error df = N - p
            if df_error < 1:
                continue
            # Each effect estimated from N/2 + vs N/2 - observations
            se = sigma * math.sqrt(4.0 / n_total)  # SE of effect estimate
            if se <= 0:
                continue
            t_crit = t_dist.ppf(1 - alpha / 2, df_error)
            ncp = abs(delta) / se
            pw = 1 - t_dist.cdf(t_crit - ncp, df_error) + t_dist.cdf(-t_crit - ncp, df_error)
            if pw >= target_power:
                req_reps = reps
                break
        else:
            req_reps = 100

        # Power curve over replicates
        reps_range = list(range(1, max(req_reps * 3, 10)))
        powers = []
        for reps in reps_range:
            n_total = n_runs_base * reps
            df_error = n_total - n_runs_base
            if df_error < 1:
                powers.append(0.0)
                continue
            se = sigma * math.sqrt(4.0 / n_total)
            t_crit = t_dist.ppf(1 - alpha / 2, df_error)
            ncp = abs(delta) / se
            pw = 1 - t_dist.cdf(t_crit - ncp, df_error) + t_dist.cdf(-t_crit - ncp, df_error)
            powers.append(max(0, min(1, pw)))

        result["plots"].append({
            "data": [
                {"x": reps_range, "y": powers, "mode": "lines+markers", "name": "Power", "line": {"color": "#4a90d9", "width": 2}},
                {"x": [req_reps], "y": [target_power], "mode": "markers", "name": f"reps={req_reps}", "marker": {"size": 12, "color": "#d94a4a", "symbol": "star"}}
            ],
            "layout": {"title": f"DOE Power — 2^{n_factors} Factorial (Δ={delta}, σ={sigma})", "xaxis": {"title": "Number of Replicates", "dtick": 1}, "yaxis": {"title": "Power", "range": [0, 1]}, "template": "plotly_white"}
        })

        n_total_req = n_runs_base * req_reps
        result["summary"] = (
            f"<<COLOR:header>>Power Analysis — 2^{n_factors} Factorial DOE<</COLOR>>\n\n"
            f"<<COLOR:text>>Factors: {n_factors}, base runs: {n_runs_base}<</COLOR>>\n"
            f"<<COLOR:text>>Minimum detectable effect: Δ = {delta}, σ = {sigma}<</COLOR>>\n"
            f"<<COLOR:text>>α = {alpha}, desired power = {target_power}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required replicates: {req_reps} → total runs = {n_total_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"DOE power: 2^{n_factors} design needs {req_reps} reps ({n_total_req} runs) to detect Δ={delta} at power={target_power}."
        result["statistics"] = {"factors": n_factors, "base_runs": n_runs_base, "required_reps": req_reps, "total_runs": n_total_req, "delta": delta, "sigma": sigma, "alpha": alpha, "power": target_power}

    elif analysis_id == "sample_size_ci":
        """
        Sample size for estimation — determine n needed for a target CI half-width.
        Supports mean (Z or t) and proportion.
        """
        est_type = config.get("type", "mean")  # mean or proportion
        conf_level = float(config.get("conf", 95)) / 100 if float(config.get("conf", 95)) > 1 else float(config.get("conf", 95))
        target_width = float(config.get("half_width", 1.0))  # desired CI half-width (margin of error)

        from scipy.stats import norm
        z = norm.ppf(1 - (1 - conf_level) / 2)

        if est_type == "proportion":
            p_est = float(config.get("p_est", 0.5))  # expected proportion
            n_req = math.ceil((z ** 2 * p_est * (1 - p_est)) / (target_width ** 2))

            # Width curve
            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * math.sqrt(p_est * (1 - p_est) / nn) for nn in ns]

            result["plots"].append({
                "data": [
                    {"x": ns, "y": widths, "mode": "lines", "name": "CI Half-Width", "line": {"color": "#4a90d9", "width": 2}},
                    {"x": [n_req], "y": [target_width], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}},
                    {"x": ns, "y": [target_width] * len(ns), "mode": "lines", "name": "Target", "line": {"color": "#d94a4a", "dash": "dash", "width": 1}}
                ],
                "layout": {"title": f"CI Half-Width vs n — Proportion (p≈{p_est})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "Half-Width"}, "template": "plotly_white"}
            })

            extra_text = f"<<COLOR:text>>Expected proportion: p ≈ {p_est}<</COLOR>>\n"
        else:
            sigma = float(config.get("sigma", 1.0))
            n_req = math.ceil((z * sigma / target_width) ** 2)

            ns = list(range(2, max(n_req * 3, 50)))
            widths = [z * sigma / math.sqrt(nn) for nn in ns]

            result["plots"].append({
                "data": [
                    {"x": ns, "y": widths, "mode": "lines", "name": "CI Half-Width", "line": {"color": "#4a90d9", "width": 2}},
                    {"x": [n_req], "y": [target_width], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}},
                    {"x": ns, "y": [target_width] * len(ns), "mode": "lines", "name": "Target", "line": {"color": "#d94a4a", "dash": "dash", "width": 1}}
                ],
                "layout": {"title": f"CI Half-Width vs n — Mean (σ={sigma})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "Half-Width"}, "template": "plotly_white"}
            })
            extra_text = f"<<COLOR:text>>Population σ = {sigma}<</COLOR>>\n"

        n_req = max(n_req, 2)
        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Estimation ({est_type.title()})<</COLOR>>\n\n"
            f"<<COLOR:text>>Confidence level: {conf_level:.0%}, target half-width (margin of error): {target_width}<</COLOR>>\n"
            f"{extra_text}\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
        )
        result["guide_observation"] = f"Sample size for {conf_level:.0%} CI with half-width={target_width}: n={n_req}."
        result["statistics"] = {"required_n": n_req, "type": est_type, "confidence": conf_level, "half_width": target_width}

    elif analysis_id == "sample_size_tolerance":
        """
        Sample size for tolerance intervals.
        Given coverage (e.g. 99% of population), confidence (e.g. 95%), → n.
        Uses the Howe (1969) / Odeh-Owen factor approach.
        """
        coverage = float(config.get("coverage", 0.99))
        confidence = float(config.get("confidence", 0.95))
        interval_type = config.get("type", "two-sided")  # two-sided or one-sided

        from scipy.stats import norm, chi2 as chi2_dist
        z_p = norm.ppf((1 + coverage) / 2) if interval_type == "two-sided" else norm.ppf(coverage)

        # Iterative: find n so that k*sigma covers coverage proportion at given confidence
        # Using the non-central chi-square approach
        n_req = 2
        for n_try in range(2, 10000):
            df = n_try - 1
            # k factor: k = z_p * sqrt(n * (df) / chi2_lower)
            chi2_lo = chi2_dist.ppf(1 - confidence, df)
            k = z_p * math.sqrt(n_try * df / chi2_lo) / math.sqrt(df) if chi2_lo > 0 else 999
            # The sample k factor shrinks as n grows; we need k computable
            # Simplified: n must satisfy z_p * sqrt(1 + 1/n) * sqrt(df/chi2_lo) ≤ achievable
            # Actually for tolerance intervals: n needed so k-factor is finite
            # Howe's approach: n ≈ (z_p / E)^2 * (1 + 1/(2*df)) ... but let's use a direct criterion:
            # Check if the tolerance factor k satisfies coverage at confidence level
            chi2_lo_check = chi2_dist.ppf(1 - confidence, df)
            if chi2_lo_check > 0:
                k_factor = z_p * math.sqrt(n_try / chi2_lo_check)
                # Tolerance interval width: k_factor * s; guaranteed coverage at confidence
                # We want k_factor to be ≤ a reasonable bound (converges as n → ∞)
                # k_factor → z_p as n → ∞. We need n large enough that k_factor is close to z_p
                # Criterion: k_factor < z_p * 1.1 (within 10%)
                if k_factor <= z_p * 1.10:
                    n_req = n_try
                    break
        else:
            n_req = 10000

        # k-factor curve
        ns = list(range(2, max(n_req * 3, 100)))
        k_factors = []
        for nn in ns:
            df = nn - 1
            chi2_lo_val = chi2_dist.ppf(1 - confidence, df)
            if chi2_lo_val > 0:
                k_factors.append(z_p * math.sqrt(nn / chi2_lo_val))
            else:
                k_factors.append(float('nan'))

        result["plots"].append({
            "data": [
                {"x": ns, "y": k_factors, "mode": "lines", "name": "k-factor", "line": {"color": "#4a90d9", "width": 2}},
                {"x": ns, "y": [z_p] * len(ns), "mode": "lines", "name": f"z_p={z_p:.3f} (asymptote)", "line": {"color": "#4a9f6e", "dash": "dash", "width": 1}},
                {"x": [n_req], "y": [z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1)) if chi2_dist.ppf(1 - confidence, n_req - 1) > 0 else z_p], "mode": "markers", "name": f"n={n_req}", "marker": {"size": 10, "color": "#d94a4a"}}
            ],
            "layout": {"title": f"Tolerance k-Factor vs n ({coverage:.0%}/{confidence:.0%})", "xaxis": {"title": "Sample Size"}, "yaxis": {"title": "k-Factor"}, "template": "plotly_white"}
        })

        result["summary"] = (
            f"<<COLOR:header>>Sample Size for Tolerance Interval<</COLOR>>\n\n"
            f"<<COLOR:text>>Coverage: {coverage:.1%} of population<</COLOR>>\n"
            f"<<COLOR:text>>Confidence: {confidence:.1%}<</COLOR>>\n"
            f"<<COLOR:text>>Type: {interval_type}<</COLOR>>\n\n"
            f"<<COLOR:green>>Required sample size: n = {n_req}<</COLOR>>\n"
            f"<<COLOR:text>>At this n, the tolerance k-factor ≈ {z_p * math.sqrt(n_req / chi2_dist.ppf(1 - confidence, n_req - 1)):.3f} (asymptote = {z_p:.3f})<</COLOR>>\n"
        )
        result["guide_observation"] = f"Tolerance interval ({coverage:.0%}/{confidence:.0%}): need n={n_req}."
        result["statistics"] = {"required_n": n_req, "coverage": coverage, "confidence": confidence, "type": interval_type, "z_p": z_p}

    elif analysis_id == "granger":
        """
        Granger Causality Test - does X help predict Y?
        Tests if past values of X improve prediction of Y beyond Y's own past.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        var_x = config.get("var_x")
        var_y = config.get("var_y")
        max_lag = int(config.get("max_lag", 4))

        # Prepare data - must be stationary ideally
        x = df[var_x].dropna().values
        y = df[var_y].dropna().values

        # Align lengths
        min_len = min(len(x), len(y))
        data_matrix = np.column_stack([y[:min_len], x[:min_len]])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GRANGER CAUSALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Question:<</COLOR>> Does {var_x} Granger-cause {var_y}?\n"
        summary += f"<<COLOR:highlight>>Max Lags:<</COLOR>> {max_lag}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {min_len}\n\n"

        summary += f"<<COLOR:text>>Interpretation:<</COLOR>>\n"
        summary += f"  If p < 0.05, past values of {var_x} help predict {var_y}\n"
        summary += f"  beyond what {var_y}'s own past provides.\n\n"

        try:
            # Run Granger test (suppress output)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            gc_results = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)
            sys.stdout = old_stdout

            summary += f"<<COLOR:text>>Results by Lag:<</COLOR>>\n"
            summary += f"{'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significant':<12}\n"
            summary += f"{'-'*42}\n"

            significant_lags = []
            p_values = []
            f_stats = []

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    test_result = gc_results[lag][0]['ssr_ftest']
                    f_stat = test_result[0]
                    p_val = test_result[1]
                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

                    f_stats.append(f_stat)
                    p_values.append(p_val)

                    if p_val < 0.05:
                        significant_lags.append(lag)

                    summary += f"{lag:<6} {f_stat:<12.4f} {p_val:<12.4f} {sig}\n"

            summary += f"\n<<COLOR:accent>>────────────────────────────────────────<</COLOR>>\n"
            if significant_lags:
                summary += f"<<COLOR:good>>CONCLUSION: {var_x} Granger-causes {var_y}<</COLOR>>\n"
                summary += f"<<COLOR:text>>Significant at lags: {significant_lags}<</COLOR>>\n"
                summary += f"<<COLOR:text>>This suggests {var_x} has predictive power for {var_y}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>CONCLUSION: No Granger causality detected<</COLOR>>\n"
                summary += f"<<COLOR:text>>{var_x} does not significantly improve prediction of {var_y}.<</COLOR>>\n"

            result["summary"] = summary

            # Plot: p-values by lag
            result["plots"].append({
                "title": f"Granger Causality: {var_x} → {var_y}",
                "data": [{
                    "type": "bar",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": p_values,
                    "marker": {
                        "color": ["#e85747" if p < 0.05 else "rgba(74, 159, 110, 0.4)" for p in p_values],
                        "line": {"color": "#4a9f6e", "width": 1.5}
                    }
                }, {
                    "type": "scatter",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": [0.05] * len(p_values),
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": "α = 0.05"
                }],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": "p-value", "range": [0, max(0.2, max(p_values) * 1.1)]},
                    "xaxis": {"title": "Lag"}
                }
            })

            # Time series plot
            result["plots"].append({
                "title": f"Time Series: {var_x} and {var_y}",
                "data": [{
                    "type": "scatter",
                    "y": x[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_x,
                    "line": {"color": "#4a9f6e"}
                }, {
                    "type": "scatter",
                    "y": y[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_y,
                    "yaxis": "y2",
                    "line": {"color": "#47a5e8"}
                }],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": var_x, "side": "left"},
                    "yaxis2": {"title": var_y, "side": "right", "overlaying": "y"},
                    "xaxis": {"title": "Observation"}
                }
            })

            result["guide_observation"] = f"Granger causality: {var_x} → {var_y} " + (f"significant at lags {significant_lags}." if significant_lags else "not significant.")

            # Statistics for Synara
            result["statistics"] = {
                f"granger_min_pvalue": float(min(p_values)),
                f"granger_significant_lags": len(significant_lags),
                f"granger_causal": 1 if significant_lags else 0
            }

        except Exception as e:
            result["summary"] = f"Granger causality test failed: {str(e)}\n\nEnsure both variables are numeric and have sufficient observations."

    elif analysis_id == "changepoint":
        """
        Change Point Detection - when did the process shift?
        Uses PELT algorithm to find optimal change points.
        """
        var = config.get("var")
        penalty = config.get("penalty", "bic")
        min_size = int(config.get("min_size", 10))

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> PELT with {penalty.upper()} penalty\n\n"

        try:
            import ruptures as rpt

            # Use PELT algorithm (optimal for multiple change points)
            model = rpt.Pelt(model="rbf", min_size=min_size).fit(data)

            # Determine penalty value
            if penalty == "bic":
                pen = np.log(n) * np.var(data)
            elif penalty == "aic":
                pen = 2 * np.var(data)
            else:
                pen = float(penalty) if penalty.replace('.','').isdigit() else np.log(n) * np.var(data)

            change_points = model.predict(pen=pen)

            # Remove the last point (always equals n)
            if change_points and change_points[-1] == n:
                change_points = change_points[:-1]

            summary += f"<<COLOR:text>>Change Points Detected:<</COLOR>> {len(change_points)}\n\n"

            if change_points:
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
                summary += f"<<COLOR:text>>{'Index':<10} {'Value at CP':<15} {'Segment Mean':<15}<</COLOR>>\n"
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"

                segments = [0] + change_points + [n]
                for i, cp in enumerate(change_points):
                    seg_start = segments[i]
                    seg_end = cp
                    seg_mean = np.mean(data[seg_start:seg_end])
                    summary += f"{cp:<10} {data[cp-1]:<15.4f} {seg_mean:<15.4f}\n"

                # Final segment
                final_mean = np.mean(data[change_points[-1]:])
                summary += f"\n<<COLOR:text>>Final segment mean: {final_mean:.4f}<</COLOR>>\n"

                summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
                summary += f"  These points mark where the process behavior shifted.\n"
                summary += f"  Investigate what changed at these times.\n"
            else:
                summary += f"<<COLOR:text>>No significant change points detected.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The process appears stable over this period.<</COLOR>>\n"

            result["summary"] = summary

            # Plot with change points
            plot_data = [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines",
                "name": var,
                "line": {"color": "#4a9f6e", "width": 1.5}
            }]

            # Add vertical lines for change points
            shapes = []
            for cp in change_points:
                shapes.append({
                    "type": "line",
                    "x0": cp,
                    "x1": cp,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2}
                })

            # Add segment means as horizontal lines
            segments = [0] + list(change_points) + [n]
            for i in range(len(segments) - 1):
                seg_mean = np.mean(data[segments[i]:segments[i+1]])
                plot_data.append({
                    "type": "scatter",
                    "x": [segments[i], segments[i+1]-1],
                    "y": [seg_mean, seg_mean],
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2},
                    "name": f"Segment {i+1} mean" if i == 0 else None,
                    "showlegend": i == 0
                })

            result["plots"].append({
                "title": f"Change Points: {var}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "shapes": shapes,
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": var}
                }
            })

            result["guide_observation"] = f"Detected {len(change_points)} change point(s) in {var}." + (" Investigate what caused these shifts." if change_points else " Process appears stable.")

            # Statistics for Synara
            result["statistics"] = {
                "change_points_count": len(change_points),
                "change_point_indices": change_points if change_points else []
            }

        except ImportError:
            result["summary"] = "Change point detection requires the 'ruptures' package.\n\nInstall with: pip install ruptures"
        except Exception as e:
            result["summary"] = f"Change point detection failed: {str(e)}"

    elif analysis_id == "mann_whitney":
        """
        Mann-Whitney U Test - non-parametric alternative to 2-sample t-test.
        Tests if two groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Mann-Whitney U requires exactly 2 groups. Found {len(groups)}: {list(groups)}"
            return result

        group1 = df[df[group_var] == groups[0]][var].dropna()
        group2 = df[df[group_var] == groups[1]][var].dropna()

        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MANN-WHITNEY U TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        summary += f"  {groups[0]}: median = {group1.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[:n1].mean():.1f}\n"
        summary += f"  {groups[1]}: median = {group2.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[n1:].mean():.1f}\n\n"

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  U statistic: {stat:.2f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  Effect size (r): {effect_size:.3f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>Groups differ significantly (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots
        result["plots"].append({
            "title": f"Mann-Whitney: {var} by {group_var}",
            "data": [
                {"type": "box", "y": group1.tolist(), "name": str(groups[0]), "marker": {"color": "#4a9f6e"}, "fillcolor": "rgba(74, 159, 110, 0.3)"},
                {"type": "box", "y": group2.tolist(), "name": str(groups[1]), "marker": {"color": "#47a5e8"}, "fillcolor": "rgba(71, 165, 232, 0.3)"}
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Mann-Whitney U test p = {pval:.4f}. " + ("Groups differ significantly." if pval < 0.05 else "No significant difference.")
        result["statistics"] = {"U_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_size)}

    elif analysis_id == "kruskal":
        """
        Kruskal-Wallis H Test - non-parametric alternative to one-way ANOVA.
        Tests if multiple groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        group_data = [df[df[group_var] == g][var].dropna().values for g in groups]

        stat, pval = stats.kruskal(*group_data)

        # Effect size (epsilon squared)
        n_total = sum(len(g) for g in group_data)
        epsilon_sq = (stat - len(groups) + 1) / (n_total - len(groups))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>KRUSKAL-WALLIS H TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(groups)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        for g, data in zip(groups, group_data):
            summary += f"  {g}: n={len(data)}, median={np.median(data):.4f}\n"

        summary += f"\n<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  H statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  ε² (effect size): {epsilon_sq:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>At least one group differs significantly (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider post-hoc tests (Dunn's test) to identify which groups differ.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference among groups (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for all groups
        result["plots"].append({
            "title": f"Kruskal-Wallis: {var} by {group_var}",
            "data": [
                {"type": "box", "y": data.tolist(), "name": str(g),
                 "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"][i % 5]},
                 "fillcolor": ["rgba(74,159,110,0.3)", "rgba(71,165,232,0.3)", "rgba(232,149,71,0.3)", "rgba(159,74,74,0.3)", "rgba(108,92,231,0.3)"][i % 5]}
                for i, (g, data) in enumerate(zip(groups, group_data))
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Kruskal-Wallis H = {stat:.2f}, p = {pval:.4f}. " + ("Groups differ." if pval < 0.05 else "No difference.")
        result["statistics"] = {"H_statistic": float(stat), "p_value": float(pval), "epsilon_squared": float(epsilon_sq)}

    elif analysis_id == "wilcoxon":
        """
        Wilcoxon Signed-Rank Test - non-parametric alternative to paired t-test.
        Tests if paired differences are symmetrically distributed around zero.
        """
        var1 = config.get("var1")
        var2 = config.get("var2")

        sample1 = df[var1].dropna()
        sample2 = df[var2].dropna()
        # Align lengths for paired data
        min_len = min(len(sample1), len(sample2))
        sample1 = sample1.iloc[:min_len]
        sample2 = sample2.iloc[:min_len]

        if min_len < 6:
            result["summary"] = f"Wilcoxon signed-rank requires at least 6 paired observations. Got {min_len}."
            return result

        diffs = sample1.values - sample2.values
        stat, pval = stats.wilcoxon(sample1, sample2)

        # Effect size: r = Z / sqrt(N)
        z_score = stats.norm.ppf(pval / 2)
        effect_r = abs(z_score) / np.sqrt(min_len)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>WILCOXON SIGNED-RANK TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Pair:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N pairs:<</COLOR>> {min_len}\n\n"
        summary += f"<<COLOR:text>>Differences (var1 - var2):<</COLOR>>\n"
        summary += f"  Median diff: {np.median(diffs):.4f}\n"
        summary += f"  Mean diff:   {np.mean(diffs):.4f}\n"
        summary += f"  Std diff:    {np.std(diffs, ddof=1):.4f}\n\n"
        summary += f"<<COLOR:highlight>>Test Statistic (W):<</COLOR>> {stat:.2f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>Effect Size (r):<</COLOR>> {effect_r:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:accent>>Significant difference between paired samples (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference between paired samples (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Histogram of differences
        result["plots"].append({
            "title": f"Paired Differences: {var1} - {var2}",
            "data": [
                {"type": "histogram", "x": diffs.tolist(), "name": "Differences",
                 "marker": {"color": "rgba(71,165,232,0.7)", "line": {"color": "#47a5e8", "width": 1}}},
                {"type": "scatter", "x": [0, 0], "y": [0, min_len // 3], "mode": "lines",
                 "name": "Zero", "line": {"color": "#e89547", "dash": "dash", "width": 2}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "Difference"}, "yaxis": {"title": "Count"}}
        })

        result["guide_observation"] = f"Wilcoxon signed-rank W = {stat:.2f}, p = {pval:.4f}. " + ("Paired samples differ." if pval < 0.05 else "No paired difference.")
        result["statistics"] = {"W_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_r), "median_diff": float(np.median(diffs)), "n_pairs": int(min_len)}

    elif analysis_id == "friedman":
        """
        Friedman Test - non-parametric alternative to repeated measures ANOVA.
        Tests if k related samples have different distributions.
        Requires multiple measurement columns (repeated measures).
        """
        vars_list = config.get("vars", [])
        if not vars_list:
            # Fallback: use var1 and var2 as minimum
            var1 = config.get("var1")
            var2 = config.get("var2")
            if var1 and var2:
                vars_list = [var1, var2]

        if len(vars_list) < 3:
            result["summary"] = f"Friedman test requires at least 3 related samples (repeated measures). Got {len(vars_list)}.\n\nSelect 3+ measurement columns (e.g., Time1, Time2, Time3)."
            return result

        # Drop rows with any missing values across all vars
        clean_df = df[vars_list].dropna()
        n_subjects = len(clean_df)

        if n_subjects < 5:
            result["summary"] = f"Friedman test requires at least 5 complete observations. Got {n_subjects}."
            return result

        groups = [clean_df[v].values for v in vars_list]
        stat, pval = stats.friedmanchisquare(*groups)

        # Effect size: Kendall's W = chi2 / (N * (k - 1))
        k = len(vars_list)
        kendall_w = stat / (n_subjects * (k - 1))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>FRIEDMAN TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Repeated Measures:<</COLOR>> {', '.join(vars_list)}\n"
        summary += f"<<COLOR:highlight>>N subjects:<</COLOR>> {n_subjects}\n"
        summary += f"<<COLOR:highlight>>k conditions:<</COLOR>> {k}\n\n"

        for v in vars_list:
            col = clean_df[v]
            summary += f"  {v}: median={col.median():.4f}, mean={col.mean():.4f}, sd={col.std():.4f}\n"
        summary += "\n"

        summary += f"<<COLOR:highlight>>Chi-square:<</COLOR>> {stat:.4f}\n"
        summary += f"<<COLOR:highlight>>df:<</COLOR>> {k - 1}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>Kendall's W:<</COLOR>> {kendall_w:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:accent>>Significant difference across conditions (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider Wilcoxon signed-rank tests for pairwise comparisons (with Bonferroni correction).<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference across conditions (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for each condition
        result["plots"].append({
            "title": "Friedman: Repeated Measures Comparison",
            "data": [
                {"type": "box", "y": clean_df[v].tolist(), "name": v,
                 "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84747", "#47e8c4", "#c4e847"][i % 8]},
                 "fillcolor": ["rgba(74,159,110,0.3)", "rgba(71,165,232,0.3)", "rgba(232,149,71,0.3)", "rgba(159,74,74,0.3)", "rgba(108,92,231,0.3)", "rgba(232,71,71,0.3)", "rgba(71,232,196,0.3)", "rgba(196,232,71,0.3)"][i % 8]}
                for i, v in enumerate(vars_list)
            ],
            "layout": {"height": 300, "yaxis": {"title": "Value"}}
        })

        result["guide_observation"] = f"Friedman chi2 = {stat:.2f}, p = {pval:.4f}, W = {kendall_w:.3f}. " + ("Conditions differ." if pval < 0.05 else "No difference.")
        result["statistics"] = {"chi2_statistic": float(stat), "p_value": float(pval), "kendall_w": float(kendall_w), "df": int(k - 1), "n_subjects": int(n_subjects)}

    elif analysis_id == "spearman":
        """
        Spearman Rank Correlation - non-parametric measure of monotonic association.
        Returns rho, p-value, and confidence interval (unlike matrix-only correlation).
        """
        var1 = config.get("var1", config.get("var"))
        var2 = config.get("var2", config.get("group_var"))

        x = df[var1].dropna()
        y = df[var2].dropna()
        # Align
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        n = len(x)

        if n < 5:
            result["summary"] = f"Spearman correlation requires at least 5 observations. Got {n}."
            return result

        rho, pval = stats.spearmanr(x, y)

        # Fisher z-transform CI
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3) if n > 3 else float('inf')
        ci_low = np.tanh(z - 1.96 * se)
        ci_high = np.tanh(z + 1.96 * se)

        # Interpret strength
        abs_rho = abs(rho)
        if abs_rho >= 0.7:
            strength = "strong"
        elif abs_rho >= 0.4:
            strength = "moderate"
        elif abs_rho >= 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        direction = "positive" if rho > 0 else "negative"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>SPEARMAN RANK CORRELATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {var1} vs {var2}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:highlight>>Spearman rho:<</COLOR>> {rho:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {pval:.6f}\n"
        summary += f"<<COLOR:highlight>>95% CI:<</COLOR>> [{ci_low:.4f}, {ci_high:.4f}]\n\n"
        summary += f"<<COLOR:text>>Interpretation: {strength.capitalize()} {direction} monotonic association<</COLOR>>\n"

        if pval < 0.05:
            summary += f"<<COLOR:accent>>Statistically significant (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Not statistically significant (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Scatter with rank overlay
        result["plots"].append({
            "title": f"Spearman: {var1} vs {var2} (rho={rho:.3f})",
            "data": [{
                "type": "scatter", "mode": "markers",
                "x": x.tolist(), "y": y.tolist(),
                "marker": {"color": "#47a5e8", "size": 6, "opacity": 0.7},
                "name": "Data"
            }],
            "layout": {"height": 300, "xaxis": {"title": var1}, "yaxis": {"title": var2}}
        })

        result["guide_observation"] = f"Spearman rho = {rho:.3f}, p = {pval:.4f}. {strength.capitalize()} {direction} monotonic association."
        result["statistics"] = {"spearman_rho": float(rho), "p_value": float(pval), "ci_lower": float(ci_low), "ci_upper": float(ci_high), "n": int(n)}

    elif analysis_id == "main_effects":
        """
        Main Effects Plot - shows how each factor affects the response.
        Essential for DOE analysis.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if not factors:
            result["summary"] = "Please select at least one factor."
            return result

        y = df[response].dropna()

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MAIN EFFECTS PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
        summary += f"<<COLOR:highlight>>Grand Mean:<</COLOR>> {y.mean():.4f}\n\n"

        summary += f"<<COLOR:text>>Effect Sizes (deviation from grand mean):<</COLOR>>\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

        for i, factor in enumerate(factors):
            # Calculate means for each level
            factor_means = df.groupby(factor)[response].mean()

            summary += f"\n  <<COLOR:accent>>{factor}:<</COLOR>>\n"
            for level, mean_val in factor_means.items():
                effect = mean_val - y.mean()
                direction = "+" if effect > 0 else ""
                summary += f"    {level}: {mean_val:.4f} (effect: {direction}{effect:.4f})\n"

            # Create main effects plot
            levels = [str(l) for l in factor_means.index.tolist()]
            means = factor_means.values.tolist()

            result["plots"].append({
                "title": f"Main Effect: {factor}",
                "data": [{
                    "type": "scatter",
                    "x": levels,
                    "y": means,
                    "mode": "lines+markers",
                    "marker": {"color": colors[i % len(colors)], "size": 10},
                    "line": {"color": colors[i % len(colors)], "width": 2}
                }, {
                    "type": "scatter",
                    "x": levels,
                    "y": [y.mean()] * len(levels),
                    "mode": "lines",
                    "line": {"color": "#9aaa9a", "dash": "dash"},
                    "name": "Grand Mean"
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": factor},
                    "yaxis": {"title": f"Mean of {response}"}
                }
            })

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  Steeper slopes indicate stronger effects.\n"
        summary += f"  Flat lines suggest the factor has little impact.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Main effects plot for {len(factors)} factor(s). Check for steep slopes indicating strong effects."

    elif analysis_id == "interaction":
        """
        Interaction Plot - shows how factors interact.
        Non-parallel lines indicate interactions.
        """
        response = config.get("response")
        factor1 = config.get("factor1")
        factor2 = config.get("factor2")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>INTERACTION PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>X-axis factor:<</COLOR>> {factor1}\n"
        summary += f"<<COLOR:highlight>>Trace factor:<</COLOR>> {factor2}\n\n"

        # Calculate interaction means
        interaction_means = df.groupby([factor1, factor2])[response].mean().unstack()

        summary += f"<<COLOR:text>>Cell Means:<</COLOR>>\n"
        summary += interaction_means.to_string() + "\n\n"

        # Check for interaction (compare slopes)
        levels1 = interaction_means.index.tolist()
        levels2 = interaction_means.columns.tolist()

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7', '#e8c547']

        plot_data = []
        for i, lev2 in enumerate(levels2):
            means = interaction_means[lev2].tolist()
            plot_data.append({
                "type": "scatter",
                "x": [str(l) for l in levels1],
                "y": means,
                "mode": "lines+markers",
                "name": str(lev2),
                "marker": {"color": colors[i % len(colors)], "size": 8},
                "line": {"color": colors[i % len(colors)], "width": 2}
            })

        result["plots"].append({
            "title": f"Interaction: {factor1} × {factor2}",
            "data": plot_data,
            "layout": {
                "height": 350,
                "xaxis": {"title": factor1},
                "yaxis": {"title": f"Mean of {response}"},
                "legend": {"title": {"text": factor2}}
            }
        })

        # Simple interaction detection (check if lines are parallel)
        if len(levels2) >= 2 and len(levels1) >= 2:
            slopes = []
            for lev2 in levels2:
                vals = interaction_means[lev2].values
                slope = (vals[-1] - vals[0]) / (len(vals) - 1) if len(vals) > 1 else 0
                slopes.append(slope)

            slope_diff = max(slopes) - min(slopes)
            has_interaction = slope_diff > 0.1 * abs(np.mean(slopes)) if np.mean(slopes) != 0 else slope_diff > 0.1

            summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
            if has_interaction:
                summary += f"<<COLOR:warning>>POTENTIAL INTERACTION DETECTED<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines are not parallel, suggesting {factor1} and {factor2} interact.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The effect of {factor1} depends on the level of {factor2}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:good>>NO STRONG INTERACTION<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines appear roughly parallel.<</COLOR>>\n"
                summary += f"<<COLOR:text>>Factors may act independently.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Interaction plot for {factor1} × {factor2}. " + ("Non-parallel lines suggest interaction." if 'has_interaction' in dir() and has_interaction else "Check for parallel lines.")

    elif analysis_id == "logistic":
        """
        Logistic Regression - for binary outcomes.
        Returns odds ratios, confidence intervals, and ROC curve.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

        response = config.get("response")
        predictors = config.get("predictors", [])

        if not predictors:
            result["summary"] = "Please select at least one predictor."
            return result

        # Prepare data
        X = df[predictors].dropna()
        y = df[response].loc[X.index]

        # Encode target if needed
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            result["summary"] = f"Logistic regression requires binary outcome. Found {len(unique_vals)} unique values."
            return result

        # Map to 0/1
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = (y == unique_vals[1]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Coefficients and odds ratios
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>AUC (ROC):<</COLOR>> {roc_auc:.4f}\n\n"

        summary += f"<<COLOR:text>>Coefficients & Odds Ratios:<</COLOR>>\n"
        summary += f"  {'Predictor':<20} {'Coef':>10} {'Odds Ratio':>12}\n"
        summary += f"  {'-'*44}\n"
        for pred, coef, odds in zip(predictors, coefs, odds_ratios):
            summary += f"  {pred:<20} {coef:>10.4f} {odds:>12.4f}\n"

        summary += f"\n<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
        summary += f"  Predicted:    0      1\n"
        summary += f"  Actual 0:  {cm[0,0]:>4}   {cm[0,1]:>4}\n"
        summary += f"  Actual 1:  {cm[1,0]:>4}   {cm[1,1]:>4}\n"

        # ROC curve plot
        result["plots"].append({
            "title": "ROC Curve",
            "data": [{
                "type": "scatter",
                "x": fpr.tolist(),
                "y": tpr.tolist(),
                "mode": "lines",
                "name": f"AUC = {roc_auc:.3f}",
                "line": {"color": "#4a9f6e", "width": 2}
            }, {
                "type": "scatter",
                "x": [0, 1],
                "y": [0, 1],
                "mode": "lines",
                "name": "Random",
                "line": {"color": "#9aaa9a", "dash": "dash"}
            }],
            "layout": {
                "height": 300,
                "xaxis": {"title": "False Positive Rate"},
                "yaxis": {"title": "True Positive Rate"}
            }
        })

        # Odds ratio forest plot
        result["plots"].append({
            "title": "Odds Ratios (Log Scale)",
            "data": [{
                "type": "bar",
                "x": odds_ratios.tolist(),
                "y": predictors,
                "orientation": "h",
                "marker": {"color": ["#4a9f6e" if o > 1 else "#e85747" for o in odds_ratios]}
            }],
            "layout": {
                "height": 250,
                "xaxis": {"type": "log", "title": "Odds Ratio"},
                "shapes": [{
                    "type": "line",
                    "x0": 1, "x1": 1,
                    "y0": -0.5, "y1": len(predictors) - 0.5,
                    "line": {"color": "#9aaa9a", "dash": "dash"}
                }]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Logistic regression with AUC = {roc_auc:.3f}. Odds ratios > 1 increase probability of outcome."
        result["statistics"] = {"AUC": float(roc_auc), "accuracy": float((y_pred == y_test).mean())}

    elif analysis_id == "f_test":
        """
        F-test for equality of variances between two groups.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"F-test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        var1, var2 = g1.var(), g2.var()
        n1, n2 = len(g1), len(g2)

        # F statistic (larger variance / smaller variance)
        if var1 >= var2:
            F = var1 / var2
            df1, df2 = n1 - 1, n2 - 1
        else:
            F = var2 / var1
            df1, df2 = n2 - 1, n1 - 1

        from scipy import stats
        p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>F-TEST FOR EQUALITY OF VARIANCES<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n\n"
        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        summary += f"  {groups[0]}: n={n1}, variance={var1:.4f}, StDev={np.sqrt(var1):.4f}\n"
        summary += f"  {groups[1]}: n={n2}, variance={var2:.4f}, StDev={np.sqrt(var2):.4f}\n\n"
        summary += f"<<COLOR:highlight>>F statistic:<</COLOR>> {F:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>Variances are SIGNIFICANTLY DIFFERENT (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider using Welch's t-test or non-parametric alternatives.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>No significant difference in variances (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Equal variance assumption is reasonable.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"F = {F:.2f}, p = {p_value:.4f}. " + ("Variances differ significantly." if p_value < 0.05 else "Variances are similar.")
        result["statistics"] = {"F_statistic": float(F), "p_value": float(p_value), "variance_ratio": float(max(var1,var2)/min(var1,var2))}

        # Variance comparison bar chart + side-by-side box plots
        result["plots"].append({
            "title": f"Variance Comparison: {groups[0]} vs {groups[1]}",
            "data": [
                {"type": "bar", "x": [str(groups[0]), str(groups[1])], "y": [float(var1), float(var2)], "marker": {"color": ["#4a9f6e", "#4a90d9"]}, "name": "Variance"},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "yaxis": {"title": "Variance"}, "xaxis": {"title": group_var}}
        })
        result["plots"].append({
            "title": f"Distribution by Group: {var}",
            "data": [
                {"type": "box", "y": g1.tolist(), "name": str(groups[0]), "marker": {"color": "#4a9f6e"}, "boxpoints": "outliers"},
                {"type": "box", "y": g2.tolist(), "name": str(groups[1]), "marker": {"color": "#4a90d9"}, "boxpoints": "outliers"}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "yaxis": {"title": var}}
        })

    elif analysis_id == "equivalence":
        """
        TOST (Two One-Sided Tests) for equivalence testing.
        Tests whether two means are equivalent within a specified margin.
        """
        var = config.get("var")
        group_var = config.get("group_var")
        margin = float(config.get("margin", 0.5))  # Equivalence margin

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Equivalence test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        from scipy import stats

        mean1, mean2 = g1.mean(), g2.mean()
        std1, std2 = g1.std(), g2.std()
        n1, n2 = len(g1), len(g2)

        diff = mean1 - mean2
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        df_val = n1 + n2 - 2

        # TOST: Two one-sided tests
        t_lower = (diff - (-margin)) / se
        t_upper = (diff - margin) / se

        p_lower = 1 - stats.t.cdf(t_lower, df_val)
        p_upper = stats.t.cdf(t_upper, df_val)
        p_tost = max(p_lower, p_upper)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>EQUIVALENCE TEST (TOST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n"
        summary += f"<<COLOR:highlight>>Equivalence margin:<</COLOR>> ±{margin}\n\n"
        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        summary += f"  {groups[0]}: {mean1:.4f} (n={n1})\n"
        summary += f"  {groups[1]}: {mean2:.4f} (n={n2})\n"
        summary += f"  Difference: {diff:.4f}\n\n"
        summary += f"<<COLOR:highlight>>TOST p-value:<</COLOR>> {p_tost:.4f}\n\n"

        if p_tost < 0.05:
            summary += f"<<COLOR:good>>EQUIVALENT within ±{margin} (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>The difference is small enough to be considered equivalent.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:warning>>NOT EQUIVALENT (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Cannot conclude equivalence within the specified margin.<</COLOR>>\n"

        # Equivalence plot
        result["plots"].append({
            "title": "Equivalence Plot",
            "data": [{
                "type": "scatter",
                "x": [diff],
                "y": [0.5],
                "mode": "markers",
                "marker": {"size": 15, "color": "#4a9f6e" if p_tost < 0.05 else "#e85747"},
                "error_x": {"type": "constant", "value": 1.96 * se, "color": "#4a9f6e"},
                "name": "Mean Difference"
            }],
            "layout": {
                "height": 200,
                "xaxis": {"title": "Difference", "zeroline": True},
                "yaxis": {"visible": False, "range": [0, 1]},
                "shapes": [
                    {"type": "line", "x0": -margin, "x1": -margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "line", "x0": margin, "x1": margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "rect", "x0": -margin, "x1": margin, "y0": 0, "y1": 1, "fillcolor": "rgba(74,159,110,0.1)", "line": {"width": 0}}
                ]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"TOST p = {p_tost:.4f}. " + (f"Groups equivalent within ±{margin}." if p_tost < 0.05 else "Cannot confirm equivalence.")
        result["statistics"] = {"TOST_p_value": float(p_tost), "mean_difference": float(diff), "margin": float(margin)}

    elif analysis_id == "runs_test":
        """
        Runs test for randomness in a sequence.
        Tests whether the sequence is random or has patterns.
        """
        var = config.get("var")
        data = df[var].dropna().values

        from scipy import stats

        # Convert to binary (above/below median)
        median = np.median(data)
        binary = (data > median).astype(int)

        # Count runs
        n_runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                n_runs += 1

        n_pos = np.sum(binary)
        n_neg = len(binary) - n_pos
        n = len(binary)

        # Expected runs and variance
        expected_runs = (2 * n_pos * n_neg / n) + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        # Z-score
        z = (n_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>RUNS TEST FOR RANDOMNESS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:text>>Run Statistics:<</COLOR>>\n"
        summary += f"  Observed runs: {n_runs}\n"
        summary += f"  Expected runs: {expected_runs:.2f}\n"
        summary += f"  Above median: {n_pos}\n"
        summary += f"  Below median: {n_neg}\n\n"
        summary += f"<<COLOR:highlight>>Z-statistic:<</COLOR>> {z:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            if n_runs < expected_runs:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too few runs (clustering)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values tend to cluster together, suggesting trends or patterns.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too many runs (oscillation)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values alternate too frequently, suggesting negative autocorrelation.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>SEQUENCE APPEARS RANDOM (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>No evidence of patterns or trends in the data.<</COLOR>>\n"

        # Plot the sequence with run coloring
        colors = []
        current_color = "#4a9f6e"
        for i in range(len(binary)):
            if i > 0 and binary[i] != binary[i-1]:
                current_color = "#47a5e8" if current_color == "#4a9f6e" else "#4a9f6e"
            colors.append(current_color)

        result["plots"].append({
            "title": f"Sequence Plot ({n_runs} runs)",
            "data": [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines+markers",
                "marker": {"color": colors, "size": 6},
                "line": {"color": "rgba(74,159,110,0.3)"}
            }, {
                "type": "scatter",
                "y": [median] * len(data),
                "mode": "lines",
                "name": "Median",
                "line": {"color": "#e89547", "dash": "dash"}
            }],
            "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Runs test: {n_runs} runs, p = {p_value:.4f}. " + ("Non-random pattern detected." if p_value < 0.05 else "Sequence appears random.")
        result["statistics"] = {"runs": int(n_runs), "expected_runs": float(expected_runs), "Z_statistic": float(z), "p_value": float(p_value)}

    elif analysis_id == "sign_test":
        """
        One-sample sign test for median.
        Non-parametric alternative to one-sample t-test.
        """
        var = config.get("var")
        h0_median = float(config.get("hypothesized_median", 0))
        data = df[var].dropna().values

        from scipy import stats

        # Count values above and below hypothesized median
        above = np.sum(data > h0_median)
        below = np.sum(data < h0_median)
        ties = np.sum(data == h0_median)
        n_used = above + below  # ties excluded

        # Two-sided binomial test
        k = min(above, below)
        p_value = 2 * stats.binom.cdf(k, n_used, 0.5) if n_used > 0 else 1.0
        p_value = min(p_value, 1.0)

        sample_median = np.median(data)

        # Confidence interval on median (binomial-based)
        sorted_data = np.sort(data)
        n_total = len(data)
        ci_idx = int(stats.binom.ppf(0.025, n_total, 0.5))
        ci_lower = sorted_data[ci_idx] if ci_idx < n_total else sorted_data[0]
        ci_upper = sorted_data[n_total - 1 - ci_idx] if ci_idx < n_total else sorted_data[-1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>SIGN TEST FOR MEDIAN<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>H₀ Median:<</COLOR>> {h0_median}\n\n"
        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  N: {len(data)}\n"
        summary += f"  Sample median: {sample_median:.4f}\n"
        summary += f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
        summary += f"<<COLOR:text>>Sign Counts:<</COLOR>>\n"
        summary += f"  Above H₀: {above}\n"
        summary += f"  Below H₀: {below}\n"
        summary += f"  Ties (excluded): {ties}\n\n"
        summary += f"<<COLOR:highlight>>p-value (two-sided):<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>REJECT H₀ — Median differs from {h0_median}<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>FAIL TO REJECT H₀ — No evidence median differs from {h0_median}<</COLOR>>\n"

        # Dot plot with median lines
        result["plots"].append({
            "title": f"Sign Test: {var}",
            "data": [
                {"type": "scatter", "y": data.tolist(), "mode": "markers", "name": "Data",
                 "marker": {"color": ["#4a9f6e" if v > h0_median else "#d94a4a" if v < h0_median else "#e89547" for v in data], "size": 6}},
                {"type": "scatter", "y": [sample_median]*len(data), "mode": "lines", "name": f"Sample Median ({sample_median:.2f})",
                 "line": {"color": "#4a9f6e", "dash": "dash"}},
                {"type": "scatter", "y": [h0_median]*len(data), "mode": "lines", "name": f"H₀ ({h0_median})",
                 "line": {"color": "#d94a4a", "dash": "dot"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Sign test p = {p_value:.4f}. " + (f"Median differs from {h0_median}." if p_value < 0.05 else f"No evidence median differs from {h0_median}.")
        result["statistics"] = {"sample_median": float(sample_median), "above": int(above), "below": int(below), "p_value": float(p_value)}

    elif analysis_id == "mood_median":
        """
        Mood's Median Test - k-sample test comparing medians.
        Non-parametric alternative to one-way ANOVA for medians.
        """
        var = config.get("var")
        group_col = config.get("group") or config.get("group_var") or config.get("factor")

        from scipy import stats

        groups = df[group_col].dropna().unique()
        data_by_group = [df[df[group_col] == g][var].dropna().values for g in groups]
        all_data = np.concatenate(data_by_group)
        grand_median = np.median(all_data)

        # Contingency table: above/below grand median per group
        table = np.zeros((2, len(groups)), dtype=int)
        for j, grp_data in enumerate(data_by_group):
            table[0, j] = np.sum(grp_data > grand_median)  # above
            table[1, j] = np.sum(grp_data <= grand_median)  # at or below

        # Chi-squared test on contingency table
        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        group_medians = {str(g): float(np.median(d)) for g, d in zip(groups, data_by_group)}

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MOOD'S MEDIAN TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {group_col}\n"
        summary += f"<<COLOR:highlight>>Grand Median:<</COLOR>> {grand_median:.4f}\n\n"
        summary += f"<<COLOR:text>>Group Medians:<</COLOR>>\n"
        for g, m in group_medians.items():
            summary += f"  {g}: {m:.4f}\n"
        summary += f"\n<<COLOR:text>>Contingency Table (above/at-or-below grand median):<</COLOR>>\n"
        summary += f"  {'Group':<15} {'Above':>8} {'At/Below':>10} {'N':>6}\n"
        summary += f"  {'-'*42}\n"
        for j, g in enumerate(groups):
            summary += f"  {str(g):<15} {table[0,j]:>8} {table[1,j]:>10} {table[0,j]+table[1,j]:>6}\n"
        summary += f"\n<<COLOR:highlight>>Chi-squared:<</COLOR>> {chi2:.4f}\n"
        summary += f"<<COLOR:highlight>>df:<</COLOR>> {dof}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>SIGNIFICANT — At least one group median differs<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>NOT SIGNIFICANT — No evidence of median differences<</COLOR>>\n"

        # Box plots by group
        traces = []
        theme_colors = ['#4a9f6e', '#4a90d9', '#e89547', '#d94a4a', '#9f4a4a', '#7a6a9a']
        for i, (g, d) in enumerate(zip(groups, data_by_group)):
            traces.append({"type": "box", "y": d.tolist(), "name": str(g),
                          "marker": {"color": theme_colors[i % len(theme_colors)]}})
        traces.append({"type": "scatter", "x": [str(g) for g in groups],
                       "y": [grand_median]*len(groups), "mode": "lines",
                       "name": f"Grand Median ({grand_median:.2f})",
                       "line": {"color": "#e89547", "dash": "dash", "width": 2}})
        result["plots"].append({
            "title": f"Mood's Median Test: {var} by {group_col}",
            "data": traces,
            "layout": {"template": "plotly_dark", "height": 280, "showlegend": True}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Mood's median test: χ² = {chi2:.2f}, p = {p_value:.4f}. " + ("Medians differ." if p_value < 0.05 else "No evidence of median differences.")
        result["statistics"] = {"chi_squared": float(chi2), "df": int(dof), "p_value": float(p_value), "grand_median": float(grand_median)}

    elif analysis_id == "multi_vari":
        """
        Multi-Vari Chart - shows variation across multiple factors.
        Essential for understanding sources of variation.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if len(factors) < 1:
            result["summary"] = "Please select at least one factor."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MULTI-VARI CHART<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

        if len(factors) == 1:
            # Single factor multi-vari
            factor = factors[0]
            groups = df.groupby(factor)[response]

            plot_data = []
            x_positions = []
            x_labels = []

            for i, (level, group_data) in enumerate(groups):
                vals = group_data.dropna().values
                x_pos = i
                x_positions.append(x_pos)
                x_labels.append(str(level))

                # Individual points
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos + np.random.uniform(-0.1, 0.1) for _ in vals],
                    "y": vals.tolist(),
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6},
                    "showlegend": False
                })

                # Mean marker
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos],
                    "y": [vals.mean()],
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 12, "symbol": "diamond"},
                    "showlegend": False
                })

                summary += f"  {level}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n"

            # Connect means
            means = [groups.get_group(l).mean() for l in df[factor].dropna().unique()]
            plot_data.append({
                "type": "scatter",
                "x": x_positions,
                "y": means,
                "mode": "lines",
                "line": {"color": "#e89547", "width": 2},
                "showlegend": False
            })

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor}",
                "data": plot_data,
                "layout": {
                    "height": 300,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor},
                    "yaxis": {"title": response}
                }
            })

        else:
            # Two or more factors - nested multi-vari
            factor1, factor2 = factors[0], factors[1]
            levels1 = df[factor1].dropna().unique()
            levels2 = df[factor2].dropna().unique()

            plot_data = []
            x_positions = []
            x_labels = []
            pos = 0

            for i, lev1 in enumerate(levels1):
                group_means = []
                for j, lev2 in enumerate(levels2):
                    mask = (df[factor1] == lev1) & (df[factor2] == lev2)
                    vals = df.loc[mask, response].dropna().values

                    if len(vals) > 0:
                        x_positions.append(pos)
                        x_labels.append(f"{lev2}")

                        # Individual points
                        plot_data.append({
                            "type": "scatter",
                            "x": [pos + np.random.uniform(-0.15, 0.15) for _ in vals],
                            "y": vals.tolist(),
                            "mode": "markers",
                            "marker": {"color": colors[i % len(colors)], "size": 5, "opacity": 0.5},
                            "showlegend": False
                        })

                        # Mean
                        group_means.append((pos, vals.mean()))
                        pos += 1

                # Connect means within factor1 level
                if group_means:
                    plot_data.append({
                        "type": "scatter",
                        "x": [g[0] for g in group_means],
                        "y": [g[1] for g in group_means],
                        "mode": "lines+markers",
                        "marker": {"color": colors[i % len(colors)], "size": 10, "symbol": "diamond"},
                        "line": {"color": colors[i % len(colors)], "width": 2},
                        "name": str(lev1)
                    })

                pos += 0.5  # Gap between factor1 levels

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor1}/{factor2}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor2},
                    "yaxis": {"title": response},
                    "showlegend": True
                }
            })

            summary += f"<<COLOR:text>>Nested structure: {factor1} (colors) → {factor2} (x-axis)<</COLOR>>\n\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  • Vertical spread within groups = within-group variation\n"
        summary += f"  • Differences between group means = between-group variation\n"
        summary += f"  • Compare spreads to identify dominant sources of variation\n"

        result["summary"] = summary
        result["guide_observation"] = f"Multi-vari chart showing variation in {response} across {len(factors)} factor(s)."

    elif analysis_id == "arima":
        """
        ARIMA Time Series Analysis - fit and forecast.
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        var = config.get("var")
        p = int(config.get("p", 1))  # AR order
        d = int(config.get("d", 1))  # Differencing
        q = int(config.get("q", 1))  # MA order
        forecast_periods = int(config.get("forecast", 10))

        data = df[var].dropna().values

        # Stationarity test
        adf_result = adfuller(data)
        adf_stat, adf_pval = adf_result[0], adf_result[1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ARIMA TIME SERIES ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> ARIMA({p},{d},{q})\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Stationarity Test (ADF):<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()

            # Model summary
            summary += f"<<COLOR:text>>Model Parameters:<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            summary += f"<<COLOR:text>>Forecast ({forecast_periods} periods):<</COLOR>>\n"
            for i in range(min(5, forecast_periods)):
                summary += f"  Period {i+1}: {fc_mean.iloc[i]:.4f} [{fc_ci.iloc[i, 0]:.4f}, {fc_ci.iloc[i, 1]:.4f}]\n"
            if forecast_periods > 5:
                summary += f"  ... ({forecast_periods - 5} more periods)\n"

            # Plot
            x_hist = list(range(len(data)))
            x_fc = list(range(len(data), len(data) + forecast_periods))

            result["plots"].append({
                "title": f"ARIMA({p},{d},{q}) Forecast",
                "data": [
                    {"type": "scatter", "x": x_hist, "y": data.tolist(), "mode": "lines", "name": "Historical", "line": {"color": "#4a9f6e"}},
                    {"type": "scatter", "x": x_fc, "y": fc_mean.tolist(), "mode": "lines", "name": "Forecast", "line": {"color": "#e89547"}},
                    {"type": "scatter", "x": x_fc + x_fc[::-1], "y": fc_ci.iloc[:, 1].tolist() + fc_ci.iloc[::-1, 0].tolist(),
                     "fill": "toself", "fillcolor": "rgba(232,149,71,0.2)", "line": {"color": "transparent"}, "name": "95% CI"}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}}
            })

            # Residual diagnostics
            residuals = fitted.resid

            result["plots"].append({
                "title": "Residuals",
                "data": [{"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}}
            })

            result["statistics"] = {"AIC": float(fitted.aic), "BIC": float(fitted.bic), "ADF_pvalue": float(adf_pval)}

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += f"<<COLOR:text>>Try different p, d, q values or check data for issues.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"ARIMA({p},{d},{q}) model. {'Stationary data.' if adf_pval < 0.05 else 'Consider differencing.'}"

    elif analysis_id == "sarima":
        """
        SARIMA — Seasonal ARIMA. Extends ARIMA with seasonal (P,D,Q,m) parameters.
        Uses statsmodels SARIMAX.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller

        var = config.get("var")
        p = int(config.get("p", 1))
        d = int(config.get("d", 1))
        q = int(config.get("q", 1))
        P = int(config.get("P", 1))
        D = int(config.get("D", 1))
        Q = int(config.get("Q", 1))
        m = int(config.get("m", 12))  # Seasonal period
        forecast_periods = int(config.get("forecast", 24))

        ts_data = df[var].dropna().values

        # Stationarity test
        adf_result = adfuller(ts_data)
        adf_stat, adf_pval = adf_result[0], adf_result[1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>SARIMA SEASONAL FORECASTING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> SARIMA({p},{d},{q})({P},{D},{Q})[{m}]\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(ts_data)}\n"
        summary += f"<<COLOR:highlight>>Seasonal period:<</COLOR>> {m}\n\n"

        summary += f"<<COLOR:text>>Stationarity Test (ADF):<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = SARIMAX(ts_data, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)

            summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n"
            summary += f"  Log-likelihood: {fitted.llf:.2f}\n\n"

            # Parameter summary
            summary += f"<<COLOR:text>>Parameters:<</COLOR>>\n"
            param_names = fitted.param_names if hasattr(fitted, 'param_names') else [f"param_{i}" for i in range(len(fitted.params))]
            params = fitted.params if hasattr(fitted.params, '__len__') else [fitted.params]
            bse = fitted.bse if hasattr(fitted, 'bse') else [None] * len(params)
            pvals = fitted.pvalues if hasattr(fitted, 'pvalues') else [None] * len(params)
            for i, name in enumerate(param_names):
                val = float(params[i])
                se = float(bse[i]) if bse is not None and i < len(bse) else None
                pval = float(pvals[i]) if pvals is not None and i < len(pvals) else None
                sig = "<<COLOR:good>>*<</COLOR>>" if pval is not None and pval < 0.05 else ""
                se_str = f"{se:.4f}" if se is not None else "N/A"
                p_str = f"{pval:.4f}" if pval is not None else "N/A"
                summary += f"  {name:<20} {val:>10.4f}  (SE={se_str}, p={p_str}) {sig}\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            # Convert to lists for uniform handling
            fc_mean_list = fc_mean.tolist() if hasattr(fc_mean, 'tolist') else list(fc_mean)
            if hasattr(fc_ci, 'iloc'):
                fc_lower = fc_ci.iloc[:, 0].tolist()
                fc_upper = fc_ci.iloc[:, 1].tolist()
            else:
                fc_lower = fc_ci[:, 0].tolist()
                fc_upper = fc_ci[:, 1].tolist()

            summary += f"\n<<COLOR:text>>Forecast ({forecast_periods} periods):<</COLOR>>\n"
            for i in range(min(6, forecast_periods)):
                summary += f"  Period {i+1}: {fc_mean_list[i]:.4f} [{fc_lower[i]:.4f}, {fc_upper[i]:.4f}]\n"
            if forecast_periods > 6:
                summary += f"  ... ({forecast_periods - 6} more periods)\n"

            # Ljung-Box test on residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(fitted.resid, lags=[min(10, len(ts_data) // 5)], return_df=True)
            lb_p = float(lb['lb_pvalue'].iloc[0])
            summary += f"\n<<COLOR:text>>Ljung-Box Test (residual autocorrelation):<</COLOR>>\n"
            summary += f"  p-value: {lb_p:.4f}\n"
            if lb_p > 0.05:
                summary += f"  <<COLOR:good>>Residuals appear uncorrelated — good model fit.<</COLOR>>\n"
            else:
                summary += f"  <<COLOR:warning>>Residuals show autocorrelation — consider different orders.<</COLOR>>\n"

            # Plot
            x_hist = list(range(len(ts_data)))
            x_fc = list(range(len(ts_data), len(ts_data) + forecast_periods))

            result["plots"].append({
                "title": f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}] Forecast",
                "data": [
                    {"type": "scatter", "x": x_hist, "y": ts_data.tolist(), "mode": "lines", "name": "Historical", "line": {"color": "#4a9f6e"}},
                    {"type": "scatter", "x": x_fc, "y": fc_mean_list, "mode": "lines", "name": "Forecast", "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "scatter", "x": x_fc + x_fc[::-1], "y": fc_upper + fc_lower[::-1],
                     "fill": "toself", "fillcolor": "rgba(232,149,71,0.2)", "line": {"color": "transparent"}, "name": "95% CI"}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}}
            })

            # Residual plot
            residuals = fitted.resid
            result["plots"].append({
                "title": "Residuals",
                "data": [{"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}}
            })

            # ACF of residuals
            from statsmodels.tsa.stattools import acf
            resid_clean = residuals[~np.isnan(residuals)] if isinstance(residuals, np.ndarray) else residuals.dropna()
            n_lags = min(30, len(resid_clean) // 3)
            acf_vals = acf(resid_clean, nlags=n_lags, fft=True)
            ci_bound = 1.96 / np.sqrt(len(residuals))
            result["plots"].append({
                "title": "Residual ACF",
                "data": [
                    {"type": "bar", "x": list(range(n_lags + 1)), "y": acf_vals.tolist(),
                     "marker": {"color": ["#d94a4a" if abs(v) > ci_bound and i > 0 else "#4a9f6e" for i, v in enumerate(acf_vals)]}}
                ],
                "layout": {
                    "height": 200, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]},
                    "shapes": [
                        {"type": "line", "x0": 0, "x1": n_lags, "y0": ci_bound, "y1": ci_bound, "line": {"color": "#e89547", "dash": "dash"}},
                        {"type": "line", "x0": 0, "x1": n_lags, "y0": -ci_bound, "y1": -ci_bound, "line": {"color": "#e89547", "dash": "dash"}}
                    ],
                    "template": "plotly_white"
                }
            })

            result["statistics"] = {
                "AIC": float(fitted.aic), "BIC": float(fitted.bic),
                "ADF_pvalue": float(adf_pval), "ljung_box_p": lb_p,
                "order": [p, d, q], "seasonal_order": [P, D, Q, m]
            }
            result["guide_observation"] = f"SARIMA({p},{d},{q})({P},{D},{Q})[{m}]: AIC={fitted.aic:.1f}. " + ("Good residuals." if lb_p > 0.05 else "Check residuals.")

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += f"<<COLOR:text>>Try different (p,d,q)(P,D,Q)[m] values. Ensure enough data for seasonal period m={m}.<</COLOR>>\n"

        result["summary"] = summary

    elif analysis_id == "decomposition":
        """
        Time Series Decomposition - trend, seasonal, residual.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        var = config.get("var")
        period = int(config.get("period", 12))
        model_type = config.get("model", "additive")  # additive or multiplicative

        data = df[var].dropna()

        if len(data) < 2 * period:
            result["summary"] = f"Need at least {2 * period} observations for period={period}. Have {len(data)}."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TIME SERIES DECOMPOSITION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type.capitalize()}\n"
        summary += f"<<COLOR:highlight>>Period:<</COLOR>> {period}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        decomp = seasonal_decompose(data, model=model_type, period=period)

        # Trend statistics
        trend_clean = decomp.trend.dropna()
        summary += f"<<COLOR:text>>Trend:<</COLOR>>\n"
        summary += f"  Start: {trend_clean.iloc[0]:.4f}\n"
        summary += f"  End: {trend_clean.iloc[-1]:.4f}\n"
        summary += f"  Change: {trend_clean.iloc[-1] - trend_clean.iloc[0]:.4f}\n\n"

        # Seasonal strength
        seasonal_var = decomp.seasonal.var()
        resid_var = decomp.resid.dropna().var()
        seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0

        summary += f"<<COLOR:text>>Seasonal Strength:<</COLOR>> {seasonal_strength:.2%}\n"
        if seasonal_strength > 0.6:
            summary += f"  <<COLOR:accent>>Strong seasonality detected<</COLOR>>\n"
        elif seasonal_strength > 0.3:
            summary += f"  Moderate seasonality\n"
        else:
            summary += f"  Weak or no seasonality\n"

        x_vals = list(range(len(data)))

        # Original
        result["plots"].append({
            "title": "Original Series",
            "data": [{"type": "scatter", "x": x_vals, "y": data.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
            "layout": {"height": 150}
        })

        # Trend
        result["plots"].append({
            "title": "Trend",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.trend.tolist(), "mode": "lines", "line": {"color": "#47a5e8"}}],
            "layout": {"height": 150}
        })

        # Seasonal
        result["plots"].append({
            "title": "Seasonal",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.seasonal.tolist(), "mode": "lines", "line": {"color": "#e89547"}}],
            "layout": {"height": 150}
        })

        # Residual
        result["plots"].append({
            "title": "Residual",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.resid.tolist(), "mode": "lines", "line": {"color": "#9aaa9a"}}],
            "layout": {"height": 150}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Decomposition with {seasonal_strength:.0%} seasonal strength."
        result["statistics"] = {"seasonal_strength": float(seasonal_strength), "trend_change": float(trend_clean.iloc[-1] - trend_clean.iloc[0])}

    elif analysis_id == "acf_pacf":
        """
        Autocorrelation and Partial Autocorrelation plots.
        Essential for ARIMA model identification.
        """
        from statsmodels.tsa.stattools import acf, pacf

        var = config.get("var")
        max_lags = int(config.get("lags", 20))

        data = df[var].dropna().values

        acf_vals = acf(data, nlags=max_lags)
        pacf_vals = pacf(data, nlags=max_lags)

        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(len(data))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ACF / PACF ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        summary += f"<<COLOR:highlight>>Lags shown:<</COLOR>> {max_lags}\n\n"

        # Find significant lags
        sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
        sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

        summary += f"<<COLOR:text>>Significant ACF lags:<</COLOR>> {sig_acf[:5] if sig_acf else 'None'}\n"
        summary += f"<<COLOR:text>>Significant PACF lags:<</COLOR>> {sig_pacf[:5] if sig_pacf else 'None'}\n\n"

        # ARIMA order suggestions
        summary += f"<<COLOR:success>>ARIMA ORDER SUGGESTIONS:<</COLOR>>\n"
        if len(sig_pacf) > 0 and (len(sig_acf) == 0 or sig_acf[0] > sig_pacf[-1]):
            summary += f"  PACF cuts off at lag {max(sig_pacf)}: Try AR({max(sig_pacf)})\n"
        if len(sig_acf) > 0 and (len(sig_pacf) == 0 or sig_pacf[0] > sig_acf[-1]):
            summary += f"  ACF cuts off at lag {max(sig_acf)}: Try MA({max(sig_acf)})\n"
        if len(sig_acf) > 0 and len(sig_pacf) > 0:
            summary += f"  Both taper: Try ARMA({min(3, max(sig_pacf))},{min(3, max(sig_acf))})\n"

        lags = list(range(max_lags + 1))

        # ACF plot
        result["plots"].append({
            "title": "Autocorrelation Function (ACF)",
            "data": [
                {"type": "bar", "x": lags, "y": acf_vals.tolist(), "marker": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]}}
        })

        # PACF plot
        result["plots"].append({
            "title": "Partial Autocorrelation Function (PACF)",
            "data": [
                {"type": "bar", "x": lags, "y": pacf_vals.tolist(), "marker": {"color": "#47a5e8"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF", "range": [-1, 1]}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"ACF/PACF analysis. Significant lags help identify ARIMA orders."

    elif analysis_id == "weibull":
        """
        Weibull Reliability Analysis - life data analysis.
        """
        from scipy import stats
        from scipy.optimize import minimize

        var = config.get("var")  # Time to failure
        censored = config.get("censored")  # Optional censoring indicator

        data = df[var].dropna().values
        data = data[data > 0]  # Weibull requires positive values

        if len(data) < 3:
            result["summary"] = "Need at least 3 data points for Weibull analysis."
            return result

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>WEIBULL RELIABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (time to failure)\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Weibull Parameters:<</COLOR>>\n"
        summary += f"  Shape (β): {shape:.4f}\n"
        summary += f"  Scale (η): {scale:.4f}\n\n"

        # Interpret shape parameter
        summary += f"<<COLOR:text>>Shape Interpretation:<</COLOR>>\n"
        if shape < 1:
            summary += f"  β < 1: <<COLOR:warning>>Infant mortality / early failures<</COLOR>>\n"
            summary += f"  Failure rate DECREASES over time (burn-in period)\n"
        elif shape == 1:
            summary += f"  β ≈ 1: Random failures (exponential distribution)\n"
            summary += f"  Failure rate is CONSTANT\n"
        else:
            summary += f"  β > 1: <<COLOR:accent>>Wear-out failures<</COLOR>>\n"
            summary += f"  Failure rate INCREASES over time\n"

        # Reliability metrics
        mean_life = scale * np.exp(np.log(np.exp(1)) / shape) if shape > 0 else scale
        b10 = stats.weibull_min.ppf(0.10, shape, 0, scale)  # 10% failure life
        b50 = stats.weibull_min.ppf(0.50, shape, 0, scale)  # Median life

        summary += f"\n<<COLOR:text>>Reliability Metrics:<</COLOR>>\n"
        summary += f"  Mean Life (MTTF): {mean_life:.2f}\n"
        summary += f"  B10 Life (10% fail): {b10:.2f}\n"
        summary += f"  B50 Life (median): {b50:.2f}\n"

        # Reliability at specific times
        summary += f"\n<<COLOR:text>>Reliability R(t):<</COLOR>>\n"
        for t in [b10, b50, scale, 2*scale]:
            r = 1 - stats.weibull_min.cdf(t, shape, 0, scale)
            summary += f"  R({t:.1f}) = {r:.2%}\n"

        # Weibull probability plot
        sorted_data = np.sort(data)
        n = len(sorted_data)
        rank = np.arange(1, n + 1)
        median_rank = (rank - 0.3) / (n + 0.4)  # Median rank approximation

        # Linearized Weibull
        x_plot = np.log(sorted_data)
        y_plot = np.log(-np.log(1 - median_rank))

        # Fitted line
        x_fit = np.linspace(x_plot.min(), x_plot.max(), 100)
        y_fit = shape * (x_fit - np.log(scale))

        result["plots"].append({
            "title": "Weibull Probability Plot",
            "data": [
                {"type": "scatter", "x": x_plot.tolist(), "y": y_plot.tolist(), "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 8}},
                {"type": "scatter", "x": x_fit.tolist(), "y": y_fit.tolist(), "mode": "lines", "name": "Fit", "line": {"color": "#e89547"}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "ln(-ln(1-F))"}}
        })

        # Reliability curve
        t_range = np.linspace(0, 2 * scale, 100)
        reliability = 1 - stats.weibull_min.cdf(t_range, shape, 0, scale)
        hazard = (shape / scale) * (t_range / scale) ** (shape - 1)

        result["plots"].append({
            "title": "Reliability & Hazard Functions",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": reliability.tolist(), "mode": "lines", "name": "R(t)", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": t_range.tolist(), "y": (hazard / hazard.max()).tolist(), "mode": "lines", "name": "h(t) scaled", "line": {"color": "#e85747", "dash": "dash"}, "yaxis": "y2"}
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Reliability", "range": [0, 1]},
                "yaxis2": {"title": "Hazard (scaled)", "overlaying": "y", "side": "right"}
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Weibull β={shape:.2f} ({'wear-out' if shape > 1 else 'early failure' if shape < 1 else 'random'}), η={scale:.2f}."
        result["statistics"] = {"shape_beta": float(shape), "scale_eta": float(scale), "MTTF": float(mean_life), "B10": float(b10)}

    elif analysis_id == "kaplan_meier":
        """
        Kaplan-Meier Survival Analysis — non-parametric survival curve estimation.
        Estimates the survival function S(t) = P(T > t) from censored data.
        Optional grouping for comparing survival across strata (log-rank test).
        """
        from scipy import stats as scipy_stats

        # Support both old (time/event/group) and new (time_col/event_col/group_col) config keys
        time_col = config.get("time_col") or config.get("time") or config.get("var1")
        event_col = config.get("event_col") or config.get("event") or config.get("var2")
        group_col = config.get("group_col") or config.get("group")
        alpha = 1 - float(config.get("conf", 95)) / 100

        if not time_col or time_col not in df.columns:
            result["summary"] = "Error: Please select a valid time variable."
            return result

        try:
            cols_needed = [c for c in [time_col, event_col, group_col] if c and c in df.columns]
            data = df[cols_needed].dropna()
            times = data[time_col].values.astype(float)
            events = data[event_col].values.astype(int) if event_col and event_col in data.columns else np.ones(len(times), dtype=int)

            def km_estimate(t, e):
                """Product-limit estimator with Greenwood CIs."""
                unique_times = np.sort(np.unique(t[e == 1]))
                n_risk = []
                n_event = []
                survival = []
                s = 1.0
                var_sum = 0.0
                ci_lower = []
                ci_upper = []
                z = scipy_stats.norm.ppf(1 - alpha / 2)

                for ti in unique_times:
                    ni = np.sum(t >= ti)
                    di = np.sum((t == ti) & (e == 1))
                    n_risk.append(int(ni))
                    n_event.append(int(di))
                    if ni > 0:
                        s *= (1 - di / ni)
                        if ni > di:
                            var_sum += di / (ni * (ni - di))
                    survival.append(s)
                    se = s * np.sqrt(var_sum) if var_sum > 0 else 0
                    ci_lower.append(max(0, s - z * se))
                    ci_upper.append(min(1, s + z * se))

                return unique_times, survival, n_risk, n_event, ci_lower, ci_upper

            if group_col and group_col in data.columns and group_col != "" and group_col != "None":
                groups = sorted(data[group_col].unique())
                colors = ['#2c5f2d', '#4a90d9', '#d94a4a', '#d9a04a', '#7d4ad9']

                traces = []
                summary_parts = []
                median_survivals = {}

                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(t_g, e_g)

                    color = colors[i % len(colors)]
                    traces.append({
                        "x": [0] + ut.tolist(), "y": [1.0] + surv,
                        "mode": "lines", "name": f"{grp} (n={len(t_g)})",
                        "line": {"shape": "hv", "color": color, "width": 2}
                    })
                    traces.append({
                        "x": [0] + ut.tolist() + ut.tolist()[::-1] + [0],
                        "y": [1.0] + ci_hi + ci_lo[::-1] + [1.0],
                        "fill": "toself", "line": {"width": 0},
                        "showlegend": False, "name": f"{grp} CI", "opacity": 0.2
                    })

                    median = None
                    for j, s_val in enumerate(surv):
                        if s_val <= 0.5:
                            median = float(ut[j])
                            break
                    median_survivals[str(grp)] = median
                    summary_parts.append(f"**{grp}** (n={len(t_g)}): Median survival = {median if median else 'not reached'}")

                result["plots"].append({
                    "data": traces,
                    "layout": {
                        "title": "Kaplan-Meier Survival Curves by Group",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        "template": "plotly_white"
                    }
                })

                # Cumulative hazard plot: H(t) = -ln(S(t))
                ch_traces = []
                for i, grp in enumerate(groups):
                    mask = data[group_col] == grp
                    t_g = times[mask]
                    e_g = events[mask]
                    ut_ch, surv_ch, _, _, _, _ = km_estimate(t_g, e_g)
                    cum_haz = [-np.log(max(s, 1e-10)) for s in surv_ch]
                    ch_traces.append({
                        "x": [0] + ut_ch.tolist(), "y": [0.0] + cum_haz,
                        "mode": "lines", "name": f"{grp}",
                        "line": {"shape": "hv", "color": colors[i % len(colors)], "width": 2}
                    })
                result["plots"].append({
                    "data": ch_traces,
                    "layout": {
                        "title": "Cumulative Hazard by Group",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "H(t) = −ln S(t)"},
                        "template": "plotly_white"
                    }
                })

                # Log-rank test (Mantel-Haenszel)
                all_event_times = np.sort(np.unique(times[events == 1]))
                observed = {str(g): 0.0 for g in groups}
                expected = {str(g): 0.0 for g in groups}

                for ti in all_event_times:
                    total_at_risk = np.sum(times >= ti)
                    total_events = np.sum((times == ti) & (events == 1))
                    for g in groups:
                        mask_g = data[group_col] == g
                        t_g = times[mask_g]
                        e_g = events[mask_g]
                        n_g = np.sum(t_g >= ti)
                        d_g = np.sum((t_g == ti) & (e_g == 1))
                        observed[str(g)] += d_g
                        if total_at_risk > 0:
                            expected[str(g)] += n_g * total_events / total_at_risk

                chi2_stat = sum((observed[str(g)] - expected[str(g)])**2 / expected[str(g)]
                                for g in groups if expected[str(g)] > 0)
                df_lr = len(groups) - 1
                p_logrank = 1 - scipy_stats.chi2.cdf(chi2_stat, df_lr)

                summary_parts.insert(0, f"**Log-rank test**: chi2={chi2_stat:.3f}, df={df_lr}, p={p_logrank:.4f} {'(significant)' if p_logrank < alpha else '(not significant)'}\n")
                result["summary"] = "\n".join(summary_parts)
                result["guide_observation"] = f"KM analysis: {len(groups)} groups, log-rank p={p_logrank:.4f}. {'Survival differs significantly.' if p_logrank < alpha else 'No significant difference.'}"
                result["statistics"] = {
                    "log_rank_chi2": float(chi2_stat), "log_rank_p": float(p_logrank),
                    "df": df_lr, "n_total": len(data), "n_events": int(events.sum()),
                    "median_survival": median_survivals
                }
            else:
                ut, surv, nr, ne, ci_lo, ci_hi = km_estimate(times, events)

                result["plots"].append({
                    "data": [
                        {"x": [0] + ut.tolist(), "y": [1.0] + surv, "mode": "lines", "name": "Survival",
                         "line": {"shape": "hv", "color": "#2c5f2d", "width": 2}},
                        {"x": [0] + ut.tolist(), "y": [1.0] + ci_hi, "mode": "lines", "name": "Upper CI",
                         "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1}, "showlegend": False},
                        {"x": [0] + ut.tolist(), "y": [1.0] + ci_lo, "mode": "lines", "name": "Lower CI",
                         "line": {"shape": "hv", "dash": "dash", "color": "#2c5f2d", "width": 1},
                         "fill": "tonexty", "fillcolor": "rgba(44,95,45,0.15)", "showlegend": False}
                    ],
                    "layout": {
                        "title": "Kaplan-Meier Survival Curve",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                        "template": "plotly_white"
                    }
                })

                # Cumulative hazard plot: H(t) = -ln(S(t))
                cum_haz = [-np.log(max(s, 1e-10)) for s in surv]
                result["plots"].append({
                    "data": [
                        {"x": [0] + ut.tolist(), "y": [0.0] + cum_haz, "mode": "lines", "name": "Cumulative Hazard",
                         "line": {"shape": "hv", "color": "#4a90d9", "width": 2}}
                    ],
                    "layout": {
                        "title": "Cumulative Hazard Function",
                        "xaxis": {"title": f"Time ({time_col})"},
                        "yaxis": {"title": "H(t) = −ln S(t)"},
                        "template": "plotly_white"
                    }
                })

                median = None
                for j, s_val in enumerate(surv):
                    if s_val <= 0.5:
                        median = float(ut[j])
                        break

                result["summary"] = f"**Kaplan-Meier Survival Analysis**\n\nN = {len(data)}, Events = {int(events.sum())}, Censored = {int((events == 0).sum())}\nMedian survival time: {median if median else 'not reached'}"
                result["guide_observation"] = f"KM curve: n={len(data)}, {int(events.sum())} events. Median survival = {median if median else 'not reached'}."
                result["statistics"] = {
                    "n_total": len(data), "n_events": int(events.sum()),
                    "n_censored": int((events == 0).sum()), "median_survival": median
                }

        except Exception as e:
            result["summary"] = f"Kaplan-Meier error: {str(e)}"

    elif analysis_id == "cox_ph":
        """
        Cox Proportional Hazards Regression — semi-parametric survival model.
        Estimates hazard ratios for covariates without assuming a baseline hazard distribution.
        Reports coefficients, hazard ratios, 95% CIs, concordance index.
        """
        import numpy as np
        from scipy import stats as scipy_stats

        time_col = config.get("time_col") or config.get("time") or config.get("var1")
        event_col = config.get("event_col") or config.get("event") or config.get("var2")
        covariates = config.get("covariates", [])
        alpha = 1 - float(config.get("conf", 95)) / 100

        if not time_col or not event_col:
            result["summary"] = "Error: Please select a time variable and an event/censor variable."
            return result
        if not covariates:
            result["summary"] = "Error: Please select at least one covariate."
            return result

        try:
            from statsmodels.duration.hazard_regression import PHReg

            all_cols = [time_col, event_col] + covariates
            data = df[[c for c in all_cols if c in df.columns]].dropna()

            # Build covariate matrix (handle categorical columns)
            X_parts = []
            covariate_names = []
            for cov in covariates:
                if cov not in data.columns:
                    continue
                if data[cov].dtype == 'object' or data[cov].nunique() < 6:
                    dummies = pd.get_dummies(data[cov], prefix=cov, drop_first=True)
                    X_parts.append(dummies)
                    covariate_names.extend(dummies.columns.tolist())
                else:
                    X_parts.append(data[[cov]])
                    covariate_names.append(cov)

            X = pd.concat(X_parts, axis=1).values.astype(float)
            times_arr = data[time_col].values.astype(float)
            events_arr = data[event_col].values.astype(int)

            model = PHReg(times_arr, X, events_arr, ties="breslow")
            fit = model.fit()

            coefs = fit.params
            se = fit.bse
            z_vals = coefs / se
            p_vals = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_vals)))
            hr = np.exp(coefs)
            z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
            hr_lower = np.exp(coefs - z_crit * se)
            hr_upper = np.exp(coefs + z_crit * se)

            # Summary table
            rows = []
            for i, name in enumerate(covariate_names):
                rows.append(f"| {name} | {coefs[i]:.4f} | {se[i]:.4f} | {z_vals[i]:.3f} | {p_vals[i]:.4f} | {hr[i]:.3f} | ({hr_lower[i]:.3f}, {hr_upper[i]:.3f}) |")

            table = "| Covariate | Coef | SE | z | p-value | Hazard Ratio | 95% CI |\n"
            table += "|---|---|---|---|---|---|---|\n"
            table += "\n".join(rows)

            # Concordance index (Harrell's C)
            lp = X @ coefs
            concordant = 0
            discordant = 0
            for i in range(len(times_arr)):
                for j in range(i + 1, len(times_arr)):
                    if events_arr[i] == 1 and times_arr[i] < times_arr[j]:
                        if lp[i] > lp[j]:
                            concordant += 1
                        elif lp[i] < lp[j]:
                            discordant += 1
                    elif events_arr[j] == 1 and times_arr[j] < times_arr[i]:
                        if lp[j] > lp[i]:
                            concordant += 1
                        elif lp[j] < lp[i]:
                            discordant += 1
            c_index = concordant / (concordant + discordant) if (concordant + discordant) > 0 else 0.5

            # Forest plot of hazard ratios
            result["plots"].append({
                "data": [
                    {
                        "x": hr.tolist(),
                        "y": covariate_names,
                        "mode": "markers",
                        "marker": {"size": 10, "color": "#2c5f2d"},
                        "error_x": {
                            "type": "data", "symmetric": False,
                            "array": (hr_upper - hr).tolist(),
                            "arrayminus": (hr - hr_lower).tolist(),
                            "color": "#2c5f2d"
                        },
                        "type": "scatter",
                        "name": "Hazard Ratio"
                    }
                ],
                "layout": {
                    "title": "Cox PH — Hazard Ratios (Forest Plot)",
                    "xaxis": {"title": "Hazard Ratio", "type": "log"},
                    "yaxis": {"title": ""},
                    "shapes": [{"type": "line", "x0": 1, "x1": 1, "y0": -0.5, "y1": len(covariate_names) - 0.5, "line": {"dash": "dash", "color": "red"}}],
                    "template": "plotly_white"
                }
            })

            # Risk score distribution by event status
            result["plots"].append({
                "data": [
                    {"x": lp[events_arr == 1].tolist(), "type": "histogram", "name": "Events",
                     "opacity": 0.7, "marker": {"color": "#d94a4a"}},
                    {"x": lp[events_arr == 0].tolist(), "type": "histogram", "name": "Censored",
                     "opacity": 0.7, "marker": {"color": "#4a90d9"}}
                ],
                "layout": {
                    "title": "Linear Predictor Distribution by Event Status",
                    "xaxis": {"title": "Linear Predictor (Xβ)"},
                    "yaxis": {"title": "Count"},
                    "barmode": "overlay", "template": "plotly_white"
                }
            })

            # Martingale-like residuals vs linear predictor
            # Martingale residual ≈ event_i - expected events (approximated)
            baseline_cumhaz = np.zeros(len(times_arr))
            sorted_idx = np.argsort(times_arr)
            risk_scores = np.exp(lp)
            for ii in range(len(sorted_idx)):
                idx_i = sorted_idx[ii]
                at_risk = risk_scores[sorted_idx[ii:]].sum()
                if at_risk > 0 and events_arr[idx_i] == 1:
                    baseline_cumhaz[sorted_idx[ii:]] += 1.0 / at_risk
            mart_resid = events_arr - risk_scores * baseline_cumhaz
            result["plots"].append({
                "data": [{
                    "x": lp.tolist(), "y": mart_resid.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": ["#d94a4a" if e == 1 else "#4a90d9" for e in events_arr], "size": 5, "opacity": 0.6},
                    "name": "Residuals"
                }],
                "layout": {
                    "title": "Martingale Residuals vs Linear Predictor",
                    "xaxis": {"title": "Linear Predictor (Xβ)"},
                    "yaxis": {"title": "Martingale Residual"},
                    "shapes": [{"type": "line", "x0": float(lp.min()), "x1": float(lp.max()), "y0": 0, "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            n_events = int(events_arr.sum())
            ll = float(fit.llf) if hasattr(fit, 'llf') else None

            result["summary"] = f"**Cox Proportional Hazards Regression**\n\nN = {len(data)}, Events = {n_events}, Censored = {len(data) - n_events}\nConcordance index (C) = {c_index:.3f}\n{'Log-likelihood = ' + f'{ll:.2f}' if ll else ''}\n\n{table}"
            result["guide_observation"] = f"Cox PH: n={len(data)}, {n_events} events, C-index={c_index:.3f}. " + ", ".join(f"{covariate_names[i]}: HR={hr[i]:.2f} (p={p_vals[i]:.4f})" for i in range(len(covariate_names)))

            result["statistics"] = {
                "n_total": len(data),
                "n_events": n_events,
                "concordance": float(c_index),
                "log_likelihood": ll,
                "coefficients": {covariate_names[i]: {
                    "coef": float(coefs[i]), "se": float(se[i]),
                    "z": float(z_vals[i]), "p": float(p_vals[i]),
                    "hazard_ratio": float(hr[i]),
                    "hr_ci_lower": float(hr_lower[i]), "hr_ci_upper": float(hr_upper[i])
                } for i in range(len(covariate_names))}
            }

        except ImportError:
            result["summary"] = "Cox PH requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Cox PH error: {str(e)}"

    elif analysis_id == "gage_rr":
        """
        Gage R&R (Repeatability and Reproducibility) Study.
        Measurement System Analysis for continuous data.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        operator = config.get("operator")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAGE R&R STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part}\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        # Get data
        data = df[[measurement, part, operator]].dropna()

        n_parts = data[part].nunique()
        n_operators = data[operator].nunique()
        n_replicates = len(data) // (n_parts * n_operators)

        summary += f"<<COLOR:text>>Study Design:<</COLOR>>\n"
        summary += f"  Parts: {n_parts}\n"
        summary += f"  Operators: {n_operators}\n"
        summary += f"  Replicates: {n_replicates}\n\n"

        # Calculate variance components using ANOVA
        from scipy import stats

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Part variation
        part_means = data.groupby(part)[measurement].mean()
        part_var = part_means.var() * n_operators * n_replicates

        # Operator variation
        op_means = data.groupby(operator)[measurement].mean()
        op_var = op_means.var() * n_parts * n_replicates

        # Repeatability (within operator-part)
        repeatability_var = data.groupby([part, operator])[measurement].var().mean()

        # Reproducibility (between operators)
        reproducibility_var = max(0, op_var - repeatability_var / (n_parts * n_replicates))

        # Gage R&R
        gage_rr_var = repeatability_var + reproducibility_var
        total_variation = gage_rr_var + part_var

        # Percentages
        pct_repeatability = 100 * np.sqrt(repeatability_var / total_variation) if total_variation > 0 else 0
        pct_reproducibility = 100 * np.sqrt(reproducibility_var / total_variation) if total_variation > 0 else 0
        pct_gage_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
        summary += f"  {'Source':<20} {'Variance':>12} {'%Contribution':>14}\n"
        summary += f"  {'-'*48}\n"
        summary += f"  {'Total Gage R&R':<20} {gage_rr_var:>12.4f} {pct_gage_rr:>13.1f}%\n"
        summary += f"    {'Repeatability':<18} {repeatability_var:>12.4f} {pct_repeatability:>13.1f}%\n"
        summary += f"    {'Reproducibility':<18} {reproducibility_var:>12.4f} {pct_reproducibility:>13.1f}%\n"
        summary += f"  {'Part-to-Part':<20} {part_var:>12.4f} {pct_part:>13.1f}%\n\n"

        # Assessment
        summary += f"<<COLOR:success>>ASSESSMENT:<</COLOR>>\n"
        if pct_gage_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
            summary += f"  Measurement system is acceptable.\n"
        elif pct_gage_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
            summary += f"  May be acceptable depending on application.\n"
        else:
            summary += f"  <<COLOR:bad>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
            summary += f"  Measurement system needs improvement.\n"

        # Number of distinct categories
        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"
        summary += f"  (Should be >= 5 for adequate discrimination)\n"

        # Plots
        # By Part
        result["plots"].append({
            "title": f"{measurement} by {part}",
            "data": [{
                "type": "box",
                "x": data[part].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#4a9f6e"}
            }],
            "layout": {"height": 250, "xaxis": {"title": part}}
        })

        # By Operator
        result["plots"].append({
            "title": f"{measurement} by {operator}",
            "data": [{
                "type": "box",
                "x": data[operator].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#47a5e8"}
            }],
            "layout": {"height": 250, "xaxis": {"title": operator}}
        })

        # Components of variation bar chart
        result["plots"].append({
            "title": "Components of Variation",
            "data": [{
                "type": "bar",
                "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                "y": [pct_gage_rr, pct_repeatability, pct_reproducibility, pct_part],
                "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]}
            }],
            "layout": {"height": 250, "yaxis": {"title": "% of Total Variation"}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Gage R&R = {pct_gage_rr:.1f}%. " + ("Acceptable." if pct_gage_rr < 30 else "Needs improvement.")
        result["statistics"] = {"gage_rr_pct": float(pct_gage_rr), "repeatability_pct": float(pct_repeatability), "reproducibility_pct": float(pct_reproducibility), "ndc": int(ndc)}

    # ── MSA (Measurement System Analysis) Expansion ─────────────────────────

    elif analysis_id == "gage_rr_nested":
        """
        Nested Gage R&R — when operators measure *different* parts (destructive testing).
        Variance components from nested ANOVA: part(operator), repeatability.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        operator = config.get("operator")

        data = df[[measurement, part, operator]].dropna()
        n_operators = data[operator].nunique()
        operators = sorted(data[operator].unique(), key=str)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NESTED GAGE R&R (DESTRUCTIVE)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part} (nested within {operator})\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Operator means
        op_means = data.groupby(operator)[measurement].mean()
        parts_per_op = data.groupby(operator)[part].nunique().mean()
        reps_per_part = len(data) / (data.groupby([operator, part]).ngroups) if data.groupby([operator, part]).ngroups > 0 else 1

        # Between-operator variance
        ss_operator = sum(len(data[data[operator] == o]) * (op_means[o] - grand_mean) ** 2 for o in operators)
        df_operator = n_operators - 1
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0

        # Between-part(operator) variance
        ss_part = 0
        df_part = 0
        for op in operators:
            op_data = data[data[operator] == op]
            op_mean = op_data[measurement].mean()
            for p in op_data[part].unique():
                part_data = op_data[op_data[part] == p]
                ss_part += len(part_data) * (part_data[measurement].mean() - op_mean) ** 2
                df_part += 1
        df_part -= n_operators  # df for part(operator)
        ms_part = ss_part / df_part if df_part > 0 else 0

        # Repeatability (within)
        ss_error = sum((data[measurement] - data.groupby([operator, part])[measurement].transform('mean')) ** 2)
        df_error = len(data) - data.groupby([operator, part]).ngroups
        ms_error = float(ss_error / df_error) if df_error > 0 else 0

        # Variance components
        repeat_var = ms_error
        r = reps_per_part
        part_var = max(0, (ms_part - ms_error) / r) if r > 0 else 0
        n_p = parts_per_op
        reprod_var = max(0, (ms_operator - ms_part) / (n_p * r)) if n_p * r > 0 else 0
        gage_rr_var = repeat_var + reprod_var
        total_variation = gage_rr_var + part_var

        pct_gage_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_repeat = 100 * np.sqrt(repeat_var / total_variation) if total_variation > 0 else 0
        pct_reprod = 100 * np.sqrt(reprod_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components (Nested ANOVA):<</COLOR>>\n"
        summary += f"  {'Source':<25} {'Variance':>12} {'%Study Var':>12}\n"
        summary += f"  {'-'*52}\n"
        summary += f"  {'Total Gage R&R':<25} {gage_rr_var:>12.4f} {pct_gage_rr:>11.1f}%\n"
        summary += f"    {'Repeatability':<23} {repeat_var:>12.4f} {pct_repeat:>11.1f}%\n"
        summary += f"    {'Reproducibility':<23} {reprod_var:>12.4f} {pct_reprod:>11.1f}%\n"
        summary += f"  {'Part-to-Part':<25} {part_var:>12.4f} {pct_part:>11.1f}%\n\n"

        if pct_gage_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT — Gage R&R < 10%<</COLOR>>\n"
        elif pct_gage_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL — Gage R&R 10-30%<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:bad>>UNACCEPTABLE — Gage R&R > 30%<</COLOR>>\n"

        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"

        # Plots
        result["plots"].append({
            "data": [{"type": "box", "x": data[operator].astype(str).tolist(), "y": data[measurement].tolist(), "marker": {"color": "#4a9f6e"}}],
            "layout": {"title": f"{measurement} by {operator}", "xaxis": {"title": operator}, "template": "plotly_white"}
        })
        result["plots"].append({
            "data": [{"type": "bar", "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                      "y": [pct_gage_rr, pct_repeat, pct_reprod, pct_part],
                      "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]}}],
            "layout": {"title": "Components of Variation", "yaxis": {"title": "% Study Var"}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Nested Gage R&R = {pct_gage_rr:.1f}%, NDC={ndc}. " + ("Acceptable." if pct_gage_rr < 30 else "Needs improvement.")
        result["statistics"] = {"gage_rr_pct": float(pct_gage_rr), "repeatability_pct": float(pct_repeat), "reproducibility_pct": float(pct_reprod), "part_pct": float(pct_part), "ndc": ndc}

    elif analysis_id == "gage_linearity_bias":
        """
        Gage Linearity & Bias Study — measures how bias changes across the measurement range.
        Bias = average measured − reference. Linearity = slope of bias vs reference.
        """
        measurement = config.get("measurement")
        reference = config.get("reference")  # known/reference values

        data = df[[measurement, reference]].dropna()
        bias = data[measurement] - data[reference]
        data_with_bias = data.copy()
        data_with_bias["bias"] = bias

        # Linearity regression: bias = a + b * reference
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(data[reference], bias)

        ref_mean = data[reference].mean()
        overall_bias = bias.mean()
        bias_pct = 100 * abs(overall_bias) / data[reference].std() if data[reference].std() > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAGE LINEARITY & BIAS STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference:<</COLOR>> {reference}\n"
        summary += f"<<COLOR:highlight>>N observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>BIAS:<</COLOR>>\n"
        summary += f"  Overall Bias = {overall_bias:.4f}\n"
        summary += f"  Bias as % of Process Variation ≈ {bias_pct:.1f}%\n"
        summary += f"  {'Bias is significant' if abs(overall_bias) / (bias.std() / np.sqrt(len(bias))) > 2 else 'Bias is not significant'}\n\n"

        summary += f"<<COLOR:text>>LINEARITY:<</COLOR>>\n"
        summary += f"  Bias = {intercept:.4f} + {slope:.4f} × Reference\n"
        summary += f"  Slope = {slope:.4f} (p = {p_value:.4f})\n"
        summary += f"  R² = {r_value**2:.4f}\n"
        summary += f"  {'Linearity is significant (bias changes across range)' if p_value < 0.05 else 'Linearity is not significant (bias is constant)'}\n"

        # Scatter: bias vs reference
        ref_range = np.linspace(data[reference].min(), data[reference].max(), 100)
        fit_line = intercept + slope * ref_range

        result["plots"].append({
            "data": [
                {"x": data[reference].tolist(), "y": bias.tolist(), "mode": "markers", "name": "Bias", "marker": {"color": "#4a90d9", "size": 6}},
                {"x": ref_range.tolist(), "y": fit_line.tolist(), "mode": "lines", "name": f"Fit (slope={slope:.4f})", "line": {"color": "#d94a4a", "width": 2}},
                {"x": ref_range.tolist(), "y": [0] * len(ref_range), "mode": "lines", "name": "Zero bias", "line": {"color": "#4a9f6e", "dash": "dash", "width": 1}}
            ],
            "layout": {"title": "Gage Linearity (Bias vs Reference)", "xaxis": {"title": "Reference Value"}, "yaxis": {"title": "Bias (Measured − Reference)"}, "template": "plotly_white"}
        })

        # Bias by reference level (grouped)
        ref_groups = pd.qcut(data[reference], min(5, data[reference].nunique()), duplicates='drop')
        grouped_bias = data_with_bias.groupby(ref_groups, observed=False)["bias"].agg(["mean", "std", "count"])

        result["plots"].append({
            "data": [{"type": "bar", "x": [str(g) for g in grouped_bias.index], "y": grouped_bias["mean"].tolist(),
                      "error_y": {"type": "data", "array": (grouped_bias["std"] / np.sqrt(grouped_bias["count"])).tolist(), "visible": True},
                      "marker": {"color": "#e89547"}}],
            "layout": {"title": "Average Bias by Reference Level", "xaxis": {"title": "Reference Range"}, "yaxis": {"title": "Average Bias"}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Linearity: slope={slope:.4f} (p={p_value:.4f}), overall bias={overall_bias:.4f}. " + ("Linearity issue detected." if p_value < 0.05 else "No linearity issue.")
        result["statistics"] = {"bias": float(overall_bias), "slope": float(slope), "intercept": float(intercept), "r_squared": float(r_value**2), "p_value": float(p_value), "bias_pct": float(bias_pct)}

    elif analysis_id == "gage_type1":
        """
        Type 1 Gage Study — single part measured repeatedly by one operator.
        Assesses basic repeatability and bias against a known reference value.
        Cg and Cgk indices.
        """
        measurement = config.get("measurement")
        ref_value = float(config.get("reference", 0))
        tolerance = float(config.get("tolerance", 1.0))  # total tolerance = USL - LSL

        data = df[measurement].dropna()
        n = len(data)
        mean_val = data.mean()
        std_val = data.std(ddof=1)
        bias = mean_val - ref_value
        t_stat = bias / (std_val / np.sqrt(n)) if std_val > 0 else 0
        from scipy.stats import t as t_dist
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), n - 1))

        # Cg and Cgk indices
        k = 0.2  # 20% of tolerance (industry standard)
        cg = (k * tolerance) / (6 * std_val) if std_val > 0 else 0
        cgk = (k * tolerance / 2 - abs(bias)) / (3 * std_val) if std_val > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TYPE 1 GAGE STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Reference value:<</COLOR>> {ref_value}\n"
        summary += f"<<COLOR:highlight>>Tolerance:<</COLOR>> {tolerance}\n"
        summary += f"<<COLOR:highlight>>N measurements:<</COLOR>> {n}\n\n"

        summary += f"<<COLOR:text>>Descriptive Statistics:<</COLOR>>\n"
        summary += f"  Mean = {mean_val:.4f}\n"
        summary += f"  Std Dev = {std_val:.4f}\n"
        summary += f"  Bias = {bias:.4f} (Mean − Ref)\n"
        summary += f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}\n"
        summary += f"  {'Bias is significant' if p_val < 0.05 else 'Bias is not significant'}\n\n"

        summary += f"<<COLOR:text>>Capability Indices:<</COLOR>>\n"
        summary += f"  Cg  = {cg:.3f} {'(≥ 1.33 required)' if cg < 1.33 else '✓'}\n"
        summary += f"  Cgk = {cgk:.3f} {'(≥ 1.33 required)' if cgk < 1.33 else '✓'}\n\n"

        if cg >= 1.33 and cgk >= 1.33:
            summary += f"  <<COLOR:good>>ACCEPTABLE — both Cg and Cgk ≥ 1.33<</COLOR>>\n"
        else:
            summary += f"  <<COLOR:bad>>NOT ACCEPTABLE — improve repeatability or reduce bias<</COLOR>>\n"

        # Histogram with reference line
        result["plots"].append({
            "data": [
                {"type": "histogram", "x": data.tolist(), "marker": {"color": "#4a90d9", "opacity": 0.7}, "name": "Measurements"},
                {"type": "scatter", "x": [ref_value, ref_value], "y": [0, n * 0.3], "mode": "lines", "name": f"Ref = {ref_value}", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}}
            ],
            "layout": {"title": "Type 1 Gage Study — Measurement Distribution", "xaxis": {"title": measurement}, "yaxis": {"title": "Count"}, "template": "plotly_white"}
        })

        # Run chart
        result["plots"].append({
            "data": [
                {"x": list(range(1, n + 1)), "y": data.tolist(), "mode": "lines+markers", "name": "Measurements", "line": {"color": "#4a90d9", "width": 1}, "marker": {"size": 4}},
                {"x": [1, n], "y": [ref_value, ref_value], "mode": "lines", "name": "Reference", "line": {"color": "#d94a4a", "dash": "dash", "width": 2}},
                {"x": [1, n], "y": [mean_val, mean_val], "mode": "lines", "name": f"Mean = {mean_val:.4f}", "line": {"color": "#4a9f6e", "width": 1}}
            ],
            "layout": {"title": "Run Chart", "xaxis": {"title": "Observation"}, "yaxis": {"title": measurement}, "template": "plotly_white"}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Type 1 Gage: Cg={cg:.3f}, Cgk={cgk:.3f}, bias={bias:.4f} (p={p_val:.4f}). " + ("Acceptable." if cg >= 1.33 and cgk >= 1.33 else "Not acceptable.")
        result["statistics"] = {"mean": float(mean_val), "std": float(std_val), "bias": float(bias), "p_value": float(p_val), "cg": float(cg), "cgk": float(cgk)}

    elif analysis_id == "attribute_gage":
        """
        Attribute Gage Study (Binary) — evaluate an attribute measurement system.
        Each appraiser classifies parts as pass/fail. Compare to known reference.
        Reports % agreement, % effectiveness (detection rate), false alarm rate.
        """
        result_col = config.get("result") or config.get("measurement")  # appraiser's call
        reference_col = config.get("reference")  # known good/bad
        appraiser_col = config.get("appraiser") or config.get("operator")

        data = df[[result_col, reference_col]].dropna()
        if appraiser_col and appraiser_col in df.columns:
            data = df[[result_col, reference_col, appraiser_col]].dropna()
        else:
            appraiser_col = None

        # Ensure binary
        result_vals = data[result_col].astype(str)
        ref_vals = data[reference_col].astype(str)
        unique_vals = sorted(set(result_vals.unique()) | set(ref_vals.unique()), key=str)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ATTRIBUTE GAGE STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Appraisal:<</COLOR>> {result_col}\n"
        summary += f"<<COLOR:highlight>>Reference:<</COLOR>> {reference_col}\n"
        summary += f"<<COLOR:highlight>>Categories:<</COLOR>> {unique_vals}\n"
        summary += f"<<COLOR:highlight>>N assessments:<</COLOR>> {len(data)}\n\n"

        # Overall agreement
        agree = (result_vals == ref_vals).sum()
        total = len(data)
        pct_agree = 100 * agree / total if total > 0 else 0

        # If binary (2 categories): compute sensitivity/specificity
        if len(unique_vals) == 2:
            pos_label = unique_vals[1]  # assume second value is "positive/fail/defective"
            tp = ((result_vals == pos_label) & (ref_vals == pos_label)).sum()
            tn = ((result_vals != pos_label) & (ref_vals != pos_label)).sum()
            fp = ((result_vals == pos_label) & (ref_vals != pos_label)).sum()
            fn = ((result_vals != pos_label) & (ref_vals == pos_label)).sum()

            sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
            false_alarm = 100 * fp / (fp + tn) if (fp + tn) > 0 else 0
            miss_rate = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0

            summary += f"<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
            summary += f"  {'':>15} {'Ref: ' + str(unique_vals[0]):>15} {'Ref: ' + str(unique_vals[1]):>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[0]):>15} {tn:>15} {fn:>15}\n"
            summary += f"  {'Call: ' + str(unique_vals[1]):>15} {fp:>15} {tp:>15}\n\n"

            summary += f"<<COLOR:text>>Effectiveness Metrics:<</COLOR>>\n"
            summary += f"  Overall Agreement:  {pct_agree:.1f}%\n"
            summary += f"  Sensitivity (detect): {sensitivity:.1f}%\n"
            summary += f"  Specificity (accept): {specificity:.1f}%\n"
            summary += f"  False Alarm Rate:   {false_alarm:.1f}%\n"
            summary += f"  Miss Rate:          {miss_rate:.1f}%\n\n"

            if pct_agree >= 90:
                summary += f"  <<COLOR:good>>ACCEPTABLE — agreement ≥ 90%<</COLOR>>\n"
            else:
                summary += f"  <<COLOR:bad>>NOT ACCEPTABLE — agreement < 90%<</COLOR>>\n"

            # Confusion matrix heatmap
            result["plots"].append({
                "data": [{"type": "heatmap", "z": [[tn, fn], [fp, tp]], "x": [str(unique_vals[0]), str(unique_vals[1])],
                          "y": [str(unique_vals[0]), str(unique_vals[1])], "text": [[str(tn), str(fn)], [str(fp), str(tp)]],
                          "texttemplate": "%{text}", "colorscale": [[0, "#f0f0f0"], [1, "#4a90d9"]], "showscale": False}],
                "layout": {"title": "Confusion Matrix", "xaxis": {"title": "Reference"}, "yaxis": {"title": "Appraiser Call"}, "template": "plotly_white"}
            })

            result["statistics"] = {"agreement_pct": float(pct_agree), "sensitivity": float(sensitivity), "specificity": float(specificity), "false_alarm_rate": float(false_alarm), "miss_rate": float(miss_rate)}
        else:
            summary += f"<<COLOR:text>>Overall Agreement: {pct_agree:.1f}% ({agree}/{total})<</COLOR>>\n"
            result["statistics"] = {"agreement_pct": float(pct_agree)}

        # By appraiser if available
        if appraiser_col:
            appraisers = sorted(data[appraiser_col].unique(), key=str)
            app_agree = []
            for app in appraisers:
                mask = data[appraiser_col] == app
                a = (data.loc[mask, result_col].astype(str) == data.loc[mask, reference_col].astype(str)).mean() * 100
                app_agree.append(a)
            result["plots"].append({
                "data": [{"type": "bar", "x": [str(a) for a in appraisers], "y": app_agree,
                          "marker": {"color": ["#4a9f6e" if a >= 90 else "#d94a4a" for a in app_agree]}}],
                "layout": {"title": "Agreement % by Appraiser", "xaxis": {"title": "Appraiser"}, "yaxis": {"title": "% Agreement", "range": [0, 100]}, "template": "plotly_white"}
            })

        result["summary"] = summary
        result["guide_observation"] = f"Attribute gage: {pct_agree:.1f}% overall agreement. " + ("Acceptable." if pct_agree >= 90 else "Needs improvement.")

    elif analysis_id == "attribute_agreement":
        """
        Attribute Agreement Analysis — Kappa and Kendall statistics for multiple appraisers.
        Cohen's Kappa (2 raters) or Fleiss' Kappa (3+ raters).
        """
        appraiser_col = config.get("appraiser") or config.get("operator")
        part_col = config.get("part") or config.get("item")
        rating_col = config.get("rating") or config.get("measurement")

        data = df[[appraiser_col, part_col, rating_col]].dropna()
        appraisers = sorted(data[appraiser_col].unique(), key=str)
        parts = sorted(data[part_col].unique(), key=str)
        categories = sorted(data[rating_col].unique(), key=str)

        n_appraisers = len(appraisers)
        n_parts = len(parts)
        n_categories = len(categories)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ATTRIBUTE AGREEMENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Appraisers:<</COLOR>> {n_appraisers} ({', '.join(str(a) for a in appraisers)})\n"
        summary += f"<<COLOR:highlight>>Parts:<</COLOR>> {n_parts}\n"
        summary += f"<<COLOR:highlight>>Categories:<</COLOR>> {categories}\n\n"

        # Build rating matrix: rows = parts, columns = appraisers
        pivot = data.pivot_table(index=part_col, columns=appraiser_col, values=rating_col, aggfunc='first')

        # Within-appraiser agreement (if repeated trials)
        trial_counts = data.groupby([appraiser_col, part_col]).size()
        has_repeats = (trial_counts > 1).any()

        # Between-appraiser agreement
        pairwise_kappas = []
        if n_appraisers == 2:
            # Cohen's Kappa
            r1 = pivot.iloc[:, 0].astype(str).values
            r2 = pivot.iloc[:, 1].astype(str).values
            mask = ~pd.isna(r1) & ~pd.isna(r2)
            r1, r2 = r1[mask], r2[mask]
            n = len(r1)
            po = (r1 == r2).mean()
            pe = sum((r1 == c).mean() * (r2 == c).mean() for c in categories)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
            pairwise_kappas.append(("All", kappa))

            summary += f"<<COLOR:text>>Cohen's Kappa (2 raters):<</COLOR>>\n"
            summary += f"  κ = {kappa:.4f}\n"
            summary += f"  Observed agreement: {po:.1%}\n"
            summary += f"  Expected agreement: {pe:.1%}\n\n"
        else:
            # Fleiss' Kappa for 3+ raters
            # Build category count matrix
            cat_to_idx = {c: i for i, c in enumerate(categories)}
            n_matrix = np.zeros((n_parts, n_categories))
            for i, p in enumerate(parts):
                part_data = data[data[part_col] == p][rating_col].astype(str)
                for val in part_data:
                    if val in cat_to_idx:
                        n_matrix[i, cat_to_idx[val]] += 1

            N = n_parts
            n_raters = n_matrix.sum(axis=1).mean()  # average raters per item
            p_j = n_matrix.sum(axis=0) / (N * n_raters)  # proportion in each category
            Pe = (p_j ** 2).sum()

            Pi = (n_matrix ** 2).sum(axis=1) - n_raters
            Pi = Pi / (n_raters * (n_raters - 1)) if n_raters > 1 else Pi
            Po = Pi.mean()

            kappa = (Po - Pe) / (1 - Pe) if Pe < 1 else 1.0

            summary += f"<<COLOR:text>>Fleiss' Kappa ({n_appraisers} raters):<</COLOR>>\n"
            summary += f"  κ = {kappa:.4f}\n"
            summary += f"  Observed agreement: {Po:.1%}\n"
            summary += f"  Expected agreement: {Pe:.1%}\n\n"

            # Also compute pairwise Cohens
            for i in range(n_appraisers):
                for j in range(i + 1, n_appraisers):
                    r1 = pivot.iloc[:, i].astype(str).values
                    r2 = pivot.iloc[:, j].astype(str).values
                    mask = ~pd.isna(r1) & ~pd.isna(r2)
                    if mask.sum() > 0:
                        r1m, r2m = r1[mask], r2[mask]
                        po_pair = (r1m == r2m).mean()
                        pe_pair = sum((r1m == c).mean() * (r2m == c).mean() for c in categories)
                        k_pair = (po_pair - pe_pair) / (1 - pe_pair) if pe_pair < 1 else 1.0
                        pairwise_kappas.append((f"{appraisers[i]} vs {appraisers[j]}", k_pair))

        # Interpret kappa
        kappa_val = kappa if 'kappa' in dir() else (pairwise_kappas[0][1] if pairwise_kappas else 0)
        if kappa_val >= 0.81:
            interp = "Almost perfect"
        elif kappa_val >= 0.61:
            interp = "Substantial"
        elif kappa_val >= 0.41:
            interp = "Moderate"
        elif kappa_val >= 0.21:
            interp = "Fair"
        else:
            interp = "Slight/Poor"
        summary += f"<<COLOR:text>>Interpretation:<</COLOR>> {interp} agreement (Landis & Koch)\n"

        # Agreement by appraiser
        app_agreements = []
        for app in appraisers:
            app_data = data[data[appraiser_col] == app]
            if has_repeats:
                # Within-appraiser: across trials for same part
                within_agree = app_data.groupby(part_col)[rating_col].apply(lambda g: g.astype(str).nunique() == 1).mean() * 100
            else:
                within_agree = 100.0  # single trial = always agrees with self
            app_agreements.append(within_agree)

        result["plots"].append({
            "data": [{"type": "bar", "x": [str(a) for a in appraisers], "y": app_agreements,
                      "marker": {"color": "#4a90d9"}}],
            "layout": {"title": "Within-Appraiser Agreement %", "xaxis": {"title": "Appraiser"}, "yaxis": {"title": "% Self-Consistent", "range": [0, 100]}, "template": "plotly_white"}
        })

        if pairwise_kappas:
            result["plots"].append({
                "data": [{"type": "bar", "x": [p[0] for p in pairwise_kappas], "y": [p[1] for p in pairwise_kappas],
                          "marker": {"color": ["#4a9f6e" if p[1] >= 0.6 else "#e89547" if p[1] >= 0.4 else "#d94a4a" for p in pairwise_kappas]}}],
                "layout": {"title": "Pairwise Cohen's Kappa", "xaxis": {"title": "Pair"}, "yaxis": {"title": "κ", "range": [-0.2, 1]}, "template": "plotly_white"}
            })

        result["summary"] = summary
        result["guide_observation"] = f"Attribute agreement: κ={kappa_val:.3f} ({interp}). " + ("Good agreement." if kappa_val >= 0.6 else "Agreement needs improvement.")
        result["statistics"] = {"kappa": float(kappa_val), "interpretation": interp, "n_appraisers": n_appraisers, "n_parts": n_parts}

    elif analysis_id == "stepwise":
        """
        Stepwise Regression - automatic variable selection.
        Uses forward selection, backward elimination, or both.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import statsmodels.api as sm

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "both")  # forward, backward, both
        alpha_enter = float(config.get("alpha_enter", 0.05))
        alpha_remove = float(config.get("alpha_remove", 0.10))

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>STEPWISE REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {len(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method}\n"
        summary += f"<<COLOR:highlight>>α to enter:<</COLOR>> {alpha_enter}\n"
        summary += f"<<COLOR:highlight>>α to remove:<</COLOR>> {alpha_remove}\n\n"

        selected = []
        remaining = list(predictors)
        step_history = []

        def get_pvalue(X_subset, y_data, var_name):
            """Get p-value for adding a variable."""
            X_with_const = sm.add_constant(X_subset)
            try:
                model = sm.OLS(y_data, X_with_const).fit()
                idx = list(X_subset.columns).index(var_name) + 1  # +1 for constant
                return model.pvalues.iloc[idx]
            except:
                return 1.0

        # Stepwise selection
        step = 0
        while True:
            step += 1
            changed = False

            # Forward step: try adding variables
            if method in ["forward", "both"] and remaining:
                best_pval = 1.0
                best_var = None

                for var in remaining:
                    if selected:
                        X_test = X_full[selected + [var]]
                    else:
                        X_test = X_full[[var]]

                    pval = get_pvalue(X_test, y, var)
                    if pval < best_pval:
                        best_pval = pval
                        best_var = var

                if best_var and best_pval < alpha_enter:
                    selected.append(best_var)
                    remaining.remove(best_var)
                    step_history.append(f"Step {step}: ADD {best_var} (p={best_pval:.4f})")
                    changed = True

            # Backward step: try removing variables
            if method in ["backward", "both"] and selected:
                worst_pval = 0.0
                worst_var = None

                X_current = X_full[selected]
                X_with_const = sm.add_constant(X_current)
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    for i, var in enumerate(selected):
                        pval = model.pvalues.iloc[i + 1]  # +1 for constant
                        if pval > worst_pval:
                            worst_pval = pval
                            worst_var = var
                except:
                    pass

                if worst_var and worst_pval > alpha_remove:
                    selected.remove(worst_var)
                    remaining.append(worst_var)
                    step_history.append(f"Step {step}: REMOVE {worst_var} (p={worst_pval:.4f})")
                    changed = True

            if not changed:
                break

            if step > 50:  # Safety limit
                break

        summary += f"<<COLOR:text>>Selection History:<</COLOR>>\n"
        for hist in step_history:
            summary += f"  {hist}\n"
        summary += "\n"

        # Final model
        if selected:
            X_final = X_full[selected]
            X_with_const = sm.add_constant(X_final)
            final_model = sm.OLS(y, X_with_const).fit()

            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:accent>>                              FINAL MODEL<</COLOR>>\n"
            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:highlight>>Selected Variables:<</COLOR>> {', '.join(selected)}\n"
            summary += f"<<COLOR:highlight>>R²:<</COLOR>> {final_model.rsquared:.4f}\n"
            summary += f"<<COLOR:highlight>>Adjusted R²:<</COLOR>> {final_model.rsquared_adj:.4f}\n"
            summary += f"<<COLOR:highlight>>F-statistic:<</COLOR>> {final_model.fvalue:.4f} (p={final_model.f_pvalue:.4e})\n\n"

            summary += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}\n"
            summary += f"  {'-'*66}\n"
            for i, name in enumerate(['const'] + selected):
                summary += f"  {name:<20} {final_model.params.iloc[i]:>12.4f} {final_model.bse.iloc[i]:>12.4f} {final_model.tvalues.iloc[i]:>10.3f} {final_model.pvalues.iloc[i]:>10.4f}\n"

            result["statistics"] = {
                "R²": float(final_model.rsquared),
                "Adj_R²": float(final_model.rsquared_adj),
                "n_selected": len(selected),
                "selected_vars": selected
            }

            # Coefficient plot
            result["plots"].append({
                "title": "Stepwise Selected Coefficients",
                "data": [{
                    "type": "bar",
                    "x": selected,
                    "y": [float(final_model.params.iloc[i+1]) for i in range(len(selected))],
                    "marker": {"color": "#4a9f6e"}
                }],
                "layout": {"height": 250, "yaxis": {"title": "Coefficient"}}
            })
        else:
            summary += f"<<COLOR:warning>>No variables met the selection criteria.<</COLOR>>\n"
            result["statistics"] = {"n_selected": 0, "selected_vars": []}

        result["summary"] = summary
        result["guide_observation"] = f"Stepwise regression selected {len(selected)} of {len(predictors)} predictors."

    elif analysis_id == "best_subsets":
        """
        Best Subsets Regression - evaluate all possible models.
        Compares models by R², Adjusted R², Cp, BIC.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from itertools import combinations

        response = config.get("response")
        predictors = config.get("predictors", [])
        max_predictors = min(int(config.get("max_predictors", 8)), len(predictors), 8)

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)
        p_full = len(predictors)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BEST SUBSETS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {p_full}\n"
        summary += f"<<COLOR:highlight>>Max subset size:<</COLOR>> {max_predictors}\n\n"

        # Calculate full model MSE for Cp
        model_full = LinearRegression().fit(X_full, y)
        mse_full = np.mean((y - model_full.predict(X_full))**2)

        results_list = []

        # Evaluate all subsets up to max_predictors
        for k in range(1, max_predictors + 1):
            for combo in combinations(predictors, k):
                X_sub = X_full[list(combo)]
                model = LinearRegression().fit(X_sub, y)
                y_pred = model.predict(X_sub)

                sse = np.sum((y - y_pred)**2)
                r2 = 1 - sse / np.sum((y - y.mean())**2)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                # Mallows' Cp
                cp = sse / mse_full - n + 2 * (k + 1)

                # BIC
                bic = n * np.log(sse / n) + k * np.log(n)

                results_list.append({
                    "vars": combo,
                    "k": k,
                    "r2": r2,
                    "adj_r2": adj_r2,
                    "cp": cp,
                    "bic": bic
                })

        # Sort by different criteria
        by_r2 = sorted(results_list, key=lambda x: -x["r2"])
        by_adj_r2 = sorted(results_list, key=lambda x: -x["adj_r2"])
        by_cp = sorted(results_list, key=lambda x: x["cp"])
        by_bic = sorted(results_list, key=lambda x: x["bic"])

        summary += f"<<COLOR:text>>Best Models by Criterion:<</COLOR>>\n\n"

        summary += f"<<COLOR:accent>>By Adjusted R² (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10} {'Predictors'}\n"
        summary += f"  {'-'*70}\n"
        for res in by_adj_r2[:5]:
            vars_str = ', '.join(res['vars'][:3]) + ('...' if len(res['vars']) > 3 else '')
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f} {vars_str}\n"

        summary += f"\n<<COLOR:accent>>By Mallows' Cp (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10}\n"
        summary += f"  {'-'*50}\n"
        for res in by_cp[:5]:
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f}\n"

        # Best model recommendation
        best = by_adj_r2[0]
        summary += f"\n<<COLOR:success>>RECOMMENDED MODEL:<</COLOR>>\n"
        summary += f"  Variables: {', '.join(best['vars'])}\n"
        summary += f"  R² = {best['r2']:.4f}, Adj R² = {best['adj_r2']:.4f}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Best subsets: recommended {best['k']}-variable model with Adj R² = {best['adj_r2']:.4f}"
        result["statistics"] = {
            "best_r2": float(best["r2"]),
            "best_adj_r2": float(best["adj_r2"]),
            "best_vars": list(best["vars"]),
            "models_evaluated": len(results_list)
        }

        # Plot: R² and Adj R² by number of variables
        k_values = sorted(set(r["k"] for r in results_list))
        best_r2_by_k = [max(r["r2"] for r in results_list if r["k"] == k) for k in k_values]
        best_adj_r2_by_k = [max(r["adj_r2"] for r in results_list if r["k"] == k) for k in k_values]

        result["plots"].append({
            "title": "Best Subsets: R² by Model Size",
            "data": [
                {"type": "scatter", "x": k_values, "y": best_r2_by_k, "mode": "lines+markers", "name": "R²", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": k_values, "y": best_adj_r2_by_k, "mode": "lines+markers", "name": "Adj R²", "line": {"color": "#47a5e8"}}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Number of Predictors"}, "yaxis": {"title": "R²"}}
        })

    elif analysis_id == "bootstrap_ci":
        """
        Bootstrap Confidence Intervals - non-parametric inference.
        Resampling-based confidence intervals for statistics.
        """
        var = config.get("var")
        statistic = config.get("statistic", "mean")  # mean, median, std, correlation
        var2 = config.get("var2")  # For correlation
        n_bootstrap = int(config.get("n_bootstrap", 1000))
        conf_level = float(config.get("conf", 95)) / 100

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BOOTSTRAP CONFIDENCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Statistic:<</COLOR>> {statistic}\n"
        summary += f"<<COLOR:highlight>>Bootstrap samples:<</COLOR>> {n_bootstrap}\n"
        summary += f"<<COLOR:highlight>>Confidence level:<</COLOR>> {conf_level*100:.0f}%\n\n"

        np.random.seed(42)

        # Calculate bootstrap distribution
        boot_stats = []

        if statistic == "correlation" and var2:
            data2 = df[var2].dropna().values
            min_len = min(len(data), len(data2))
            data = data[:min_len]
            data2 = data2[:min_len]
            observed = np.corrcoef(data, data2)[0, 1]

            for _ in range(n_bootstrap):
                idx = np.random.choice(min_len, min_len, replace=True)
                boot_stats.append(np.corrcoef(data[idx], data2[idx])[0, 1])
        else:
            if statistic == "mean":
                observed = np.mean(data)
                stat_func = np.mean
            elif statistic == "median":
                observed = np.median(data)
                stat_func = np.median
            elif statistic == "std":
                observed = np.std(data, ddof=1)
                stat_func = lambda x: np.std(x, ddof=1)
            elif statistic == "trimmed_mean":
                from scipy.stats import trim_mean
                observed = trim_mean(data, 0.1)
                stat_func = lambda x: trim_mean(x, 0.1)
            else:
                observed = np.mean(data)
                stat_func = np.mean

            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data, n, replace=True)
                boot_stats.append(stat_func(boot_sample))

        boot_stats = np.array(boot_stats)

        # Calculate CI using percentile method
        alpha = 1 - conf_level
        ci_lower = np.percentile(boot_stats, alpha/2 * 100)
        ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)

        # BCa (Bias-Corrected and Accelerated) interval
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < observed))

        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            if statistic == "mean":
                jackknife_stats.append(np.mean(jack_sample))
            elif statistic == "median":
                jackknife_stats.append(np.median(jack_sample))
            else:
                jackknife_stats.append(np.mean(jack_sample))

        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats)**3) / (6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5 + 1e-10)

        # BCa quantiles
        z_alpha_low = stats.norm.ppf(alpha/2)
        z_alpha_high = stats.norm.ppf(1 - alpha/2)

        bca_low_q = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        bca_high_q = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        bca_lower = np.percentile(boot_stats, bca_low_q * 100)
        bca_upper = np.percentile(boot_stats, bca_high_q * 100)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Observed {statistic}: {observed:.4f}\n"
        summary += f"  Bootstrap SE: {np.std(boot_stats):.4f}\n"
        summary += f"  Bootstrap Bias: {np.mean(boot_stats) - observed:.4f}\n\n"

        summary += f"<<COLOR:text>>Confidence Intervals:<</COLOR>>\n"
        summary += f"  Percentile: ({ci_lower:.4f}, {ci_upper:.4f})\n"
        summary += f"  BCa:        ({bca_lower:.4f}, {bca_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {conf_level*100:.0f}% confident the true {statistic}\n"
        summary += f"  lies between {bca_lower:.4f} and {bca_upper:.4f}.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Bootstrap {conf_level*100:.0f}% CI for {statistic}: ({bca_lower:.4f}, {bca_upper:.4f})"
        result["statistics"] = {
            f"observed_{statistic}": float(observed),
            "ci_lower": float(bca_lower),
            "ci_upper": float(bca_upper),
            "bootstrap_se": float(np.std(boot_stats))
        }

        # Histogram of bootstrap distribution
        result["plots"].append({
            "title": f"Bootstrap Distribution of {statistic.title()}",
            "data": [
                {"type": "histogram", "x": boot_stats.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}},
                {"type": "scatter", "x": [observed, observed], "y": [0, n_bootstrap/20], "mode": "lines", "line": {"color": "#e89547", "width": 2, "dash": "dash"}, "name": "Observed"},
                {"type": "scatter", "x": [bca_lower, bca_upper], "y": [0, 0], "mode": "markers", "marker": {"color": "#e85747", "size": 12, "symbol": "triangle-up"}, "name": "CI bounds"}
            ],
            "layout": {"height": 250, "xaxis": {"title": statistic.title()}, "yaxis": {"title": "Frequency"}}
        })

    elif analysis_id == "box_cox":
        """
        Box-Cox Transformation - find optimal power transformation.
        Transforms data to approximate normality.
        """
        var = config.get("var")

        data = df[var].dropna().values

        # Box-Cox requires positive data
        if np.any(data <= 0):
            # Shift data to be positive
            shift = -np.min(data) + 1
            data_shifted = data + shift
            shifted = True
        else:
            data_shifted = data
            shift = 0
            shifted = False

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BOX-COX TRANSFORMATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        if shifted:
            summary += f"<<COLOR:warning>>Data shifted by {shift:.4f} (original had non-positive values)<</COLOR>>\n"
        summary += "\n"

        # Find optimal lambda
        transformed, optimal_lambda = stats.boxcox(data_shifted)

        # Test common transformations
        lambdas = [-2, -1, -0.5, 0, 0.5, 1, 2]
        lambda_names = ["1/x²", "1/x", "1/√x", "ln(x)", "√x", "x (none)", "x²"]

        summary += f"<<COLOR:text>>Common Transformations (Log-Likelihood):<</COLOR>>\n"
        for lam, name in zip(lambdas, lambda_names):
            if lam == 0:
                trans = np.log(data_shifted)
            else:
                trans = (data_shifted**lam - 1) / lam
            # Calculate log-likelihood
            ll = -len(data)/2 * np.log(np.var(trans)) + (lam - 1) * np.sum(np.log(data_shifted))
            summary += f"  λ = {lam:>5} ({name:<8}): LL = {ll:.2f}\n"

        summary += f"\n<<COLOR:success>>OPTIMAL TRANSFORMATION:<</COLOR>>\n"
        summary += f"  λ = {optimal_lambda:.4f}\n"

        # Interpret lambda
        if abs(optimal_lambda) < 0.1:
            suggestion = "ln(x) - logarithmic"
        elif abs(optimal_lambda - 0.5) < 0.1:
            suggestion = "√x - square root"
        elif abs(optimal_lambda - 1) < 0.1:
            suggestion = "x - no transformation needed"
        elif abs(optimal_lambda + 1) < 0.1:
            suggestion = "1/x - reciprocal"
        elif optimal_lambda < 0:
            suggestion = f"x^{optimal_lambda:.2f} - inverse power"
        else:
            suggestion = f"x^{optimal_lambda:.2f} - power transformation"

        summary += f"  Suggested: {suggestion}\n\n"

        # Normality tests before and after
        _, p_before = stats.shapiro(data[:min(5000, len(data))])
        _, p_after = stats.shapiro(transformed[:min(5000, len(transformed))])

        summary += f"<<COLOR:text>>Normality Tests (Shapiro-Wilk):<</COLOR>>\n"
        summary += f"  Original: p = {p_before:.4f} {'(normal)' if p_before > 0.05 else '(non-normal)'}\n"
        summary += f"  Transformed: p = {p_after:.4f} {'(normal)' if p_after > 0.05 else '(non-normal)'}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Box-Cox optimal λ = {optimal_lambda:.3f}. {suggestion}."
        result["statistics"] = {
            "optimal_lambda": float(optimal_lambda),
            "p_before": float(p_before),
            "p_after": float(p_after),
            "shift_applied": float(shift)
        }

        # Plot: original vs transformed distributions
        result["plots"].append({
            "title": "Original Distribution",
            "data": [{"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(232, 87, 71, 0.4)", "line": {"color": "#e85747", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": var}}
        })

        result["plots"].append({
            "title": f"Transformed (λ = {optimal_lambda:.2f})",
            "data": [{"type": "histogram", "x": transformed.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": f"Box-Cox({var})"}}
        })

        # Lambda vs log-likelihood profile
        lambda_range = np.linspace(max(-3, optimal_lambda - 2), min(3, optimal_lambda + 2), 50)
        log_likelihoods = []
        for lam in lambda_range:
            if abs(lam) < 1e-10:
                trans = np.log(data_shifted)
            else:
                trans = (data_shifted**lam - 1) / lam
            ll = -len(data)/2 * np.log(np.var(trans)) + (lam - 1) * np.sum(np.log(data_shifted))
            log_likelihoods.append(float(ll))
        result["plots"].append({
            "title": "Lambda vs Log-Likelihood",
            "data": [
                {"type": "scatter", "x": lambda_range.tolist(), "y": log_likelihoods, "mode": "lines", "line": {"color": "#4a9f6e", "width": 2}, "name": "Log-Likelihood"},
                {"type": "scatter", "x": [float(optimal_lambda)], "y": [max(log_likelihoods)], "mode": "markers", "marker": {"color": "#d94a4a", "size": 10, "symbol": "diamond"}, "name": f"Optimal λ = {optimal_lambda:.3f}"}
            ],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Lambda (λ)"}, "yaxis": {"title": "Log-Likelihood"}}
        })

    elif analysis_id == "robust_regression":
        """
        Robust Regression - outlier-resistant regression.
        Uses Huber or MM estimators.
        """
        import statsmodels.api as sm
        from statsmodels.robust.robust_linear_model import RLM

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "huber")  # huber, bisquare, andrews

        X = df[predictors].dropna()
        y = df[response].loc[X.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ROBUST REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method.title()} M-estimator\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        # Select M-estimator
        if method == "huber":
            M_estimator = sm.robust.norms.HuberT()
            summary += f"<<COLOR:dim>>Huber's T: downweights residuals > 1.345σ<</COLOR>>\n"
        elif method == "bisquare":
            M_estimator = sm.robust.norms.TukeyBiweight()
            summary += f"<<COLOR:dim>>Tukey's Bisquare: zero weight for residuals > 4.685σ<</COLOR>>\n"
        elif method == "andrews":
            M_estimator = sm.robust.norms.AndrewWave()
            summary += f"<<COLOR:dim>>Andrew's Wave: sinusoidal downweighting<</COLOR>>\n"
        else:
            M_estimator = sm.robust.norms.HuberT()

        X_const = sm.add_constant(X)

        # Fit OLS for comparison
        ols_model = sm.OLS(y, X_const).fit()

        # Fit robust model
        robust_model = RLM(y, X_const, M=M_estimator).fit()

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                        COMPARISON: OLS vs ROBUST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'OLS Coef':>12} {'Robust Coef':>12} {'Difference':>12}\n"
        summary += f"  {'-'*58}\n"

        var_names = ['const'] + predictors
        for i, name in enumerate(var_names):
            ols_coef = ols_model.params.iloc[i]
            rob_coef = robust_model.params.iloc[i]
            diff = rob_coef - ols_coef
            diff_pct = 100 * diff / abs(ols_coef) if abs(ols_coef) > 1e-10 else 0
            flag = "<<COLOR:warning>>*<</COLOR>>" if abs(diff_pct) > 10 else ""
            summary += f"  {name:<20} {ols_coef:>12.4f} {rob_coef:>12.4f} {diff:>+12.4f} {flag}\n"

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                           ROBUST MODEL DETAILS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'z':>10} {'P>|z|':>10}\n"
        summary += f"  {'-'*66}\n"

        for i, name in enumerate(var_names):
            summary += f"  {name:<20} {robust_model.params.iloc[i]:>12.4f} {robust_model.bse.iloc[i]:>12.4f} {robust_model.tvalues.iloc[i]:>10.3f} {robust_model.pvalues.iloc[i]:>10.4f}\n"

        # Identify outliers (low weights)
        weights = robust_model.weights
        outlier_threshold = 0.5
        outliers = np.where(weights < outlier_threshold)[0]

        summary += f"\n<<COLOR:text>>Observations with low weight (<{outlier_threshold}):<</COLOR>> {len(outliers)}\n"
        if len(outliers) > 0 and len(outliers) <= 10:
            summary += f"  Indices: {list(outliers)}\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        coef_diffs = np.abs(robust_model.params.values - ols_model.params.values)
        if np.max(coef_diffs[1:]) > 0.1 * np.max(np.abs(ols_model.params.values[1:])):
            summary += f"  <<COLOR:warning>>Coefficients differ substantially - outliers are influential.<</COLOR>>\n"
            summary += f"  Robust estimates may be more reliable.\n"
        else:
            summary += f"  <<COLOR:good>>Coefficients are similar - outliers have minimal influence.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Robust regression ({method}) identified {len(outliers)} low-weight observations."
        result["statistics"] = {
            "n_outliers": len(outliers),
            "method": method,
            **{f"coef_{name}": float(robust_model.params.iloc[i]) for i, name in enumerate(var_names)}
        }

        # Plot: OLS residuals vs Robust residuals
        ols_resid = ols_model.resid
        rob_resid = robust_model.resid

        result["plots"].append({
            "title": "Residual Comparison",
            "data": [
                {"type": "scatter", "x": ols_resid.tolist(), "y": rob_resid.tolist(), "mode": "markers",
                 "marker": {"color": weights.tolist(), "colorscale": [[0, "#e85747"], [1, "#4a9f6e"]], "size": 6, "colorbar": {"title": "Weight"}}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "OLS Residuals"}, "yaxis": {"title": "Robust Residuals"}}
        })

        # Weights histogram
        result["plots"].append({
            "title": "Observation Weights",
            "data": [{"type": "histogram", "x": weights.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": "Weight", "range": [0, 1.1]}}
        })

    elif analysis_id == "tolerance_interval":
        """
        Tolerance Intervals - contain a proportion of the population.
        Both normal-based and non-parametric methods.
        """
        var = config.get("var")
        proportion = float(config.get("proportion", 0.95))  # Proportion of population
        confidence = float(config.get("confidence", 0.95))  # Confidence level
        method = config.get("method", "normal")  # normal or nonparametric

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TOLERANCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Coverage:<</COLOR>> {proportion*100:.0f}% of population\n"
        summary += f"<<COLOR:highlight>>Confidence:<</COLOR>> {confidence*100:.0f}%\n\n"

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {mean:.4f}\n"
        summary += f"  Std Dev: {std:.4f}\n\n"

        # Normal-based tolerance interval
        # k factor from tolerance interval tables (approximation)
        z_p = stats.norm.ppf((1 + proportion) / 2)
        chi2_val = stats.chi2.ppf(1 - confidence, n - 1)

        # Two-sided tolerance factor
        k_normal = z_p * np.sqrt((n - 1) * (1 + 1/n) / chi2_val)

        tol_lower_normal = mean - k_normal * std
        tol_upper_normal = mean + k_normal * std

        summary += f"<<COLOR:accent>>Normal-Based Tolerance Interval:<</COLOR>>\n"
        summary += f"  k factor: {k_normal:.4f}\n"
        summary += f"  Interval: ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})\n\n"

        # Non-parametric tolerance interval
        # Uses order statistics
        # For 95/95, need approximately n >= 59 for two-sided
        # Coverage probability for (X(r), X(n-r+1)) where r is chosen appropriately

        # Simple approach: use percentiles
        alpha = 1 - confidence
        beta = 1 - proportion

        # Find r such that P(at least proportion*100% between X(r) and X(n-r+1)) >= confidence
        # Using binomial distribution
        from scipy.special import comb

        r_found = None
        for r in range(1, n//2 + 1):
            # Probability that at least proportion of population is between order statistics
            prob = 0
            for j in range(r, n - r + 2):
                prob += comb(n, j, exact=True) * (proportion**(j)) * ((1-proportion)**(n-j))
            if prob >= confidence:
                r_found = r
                break

        if r_found:
            sorted_data = np.sort(data)
            tol_lower_np = sorted_data[r_found - 1]
            tol_upper_np = sorted_data[n - r_found]
            summary += f"<<COLOR:accent>>Non-Parametric Tolerance Interval:<</COLOR>>\n"
            summary += f"  Uses order statistics X({r_found}) and X({n - r_found + 1})\n"
            summary += f"  Interval: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"
        else:
            tol_lower_np = np.min(data)
            tol_upper_np = np.max(data)
            summary += f"<<COLOR:warning>>Non-Parametric: Sample too small for exact interval.<</COLOR>>\n"
            summary += f"  Using min/max: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"

        # Comparison with confidence interval
        se = std / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se

        summary += f"<<COLOR:dim>>For comparison - {confidence*100:.0f}% CI for mean:<</COLOR>>\n"
        summary += f"  ({ci_lower:.4f}, {ci_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {confidence*100:.0f}% confident that at least {proportion*100:.0f}%\n"
        summary += f"  of the population falls within the tolerance interval.\n"
        summary += f"\n<<COLOR:dim>>Note: Tolerance intervals are WIDER than confidence intervals<</COLOR>>\n"
        summary += f"<<COLOR:dim>>because they cover the population, not just the mean.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Tolerance interval ({proportion*100:.0f}%/{confidence*100:.0f}%): ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})"
        result["statistics"] = {
            "tol_lower_normal": float(tol_lower_normal),
            "tol_upper_normal": float(tol_upper_normal),
            "tol_lower_np": float(tol_lower_np),
            "tol_upper_np": float(tol_upper_np),
            "k_factor": float(k_normal),
            "mean": float(mean),
            "std": float(std)
        }

        # Plot showing intervals
        result["plots"].append({
            "title": "Tolerance vs Confidence Intervals",
            "data": [
                {"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": var},
                "shapes": [
                    # Tolerance interval (normal)
                    {"type": "rect", "x0": tol_lower_normal, "x1": tol_upper_normal, "y0": 0, "y1": 1, "yref": "paper",
                     "fillcolor": "rgba(232, 149, 71, 0.2)", "line": {"color": "#e89547", "width": 2}},
                    # Confidence interval
                    {"type": "rect", "x0": ci_lower, "x1": ci_upper, "y0": 0.4, "y1": 0.6, "yref": "paper",
                     "fillcolor": "rgba(71, 165, 232, 0.4)", "line": {"color": "#47a5e8", "width": 2}},
                    # Mean line
                    {"type": "line", "x0": mean, "x1": mean, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#4a9f6e", "width": 2, "dash": "dash"}}
                ],
                "annotations": [
                    {"x": (tol_lower_normal + tol_upper_normal)/2, "y": 0.95, "yref": "paper", "text": "Tolerance", "showarrow": False, "font": {"color": "#e89547"}},
                    {"x": (ci_lower + ci_upper)/2, "y": 0.5, "yref": "paper", "text": "CI", "showarrow": False, "font": {"color": "#47a5e8"}}
                ]
            }
        })

    elif analysis_id == "tukey_hsd":
        """
        Tukey's Honestly Significant Difference — pairwise comparison after ANOVA.
        Controls family-wise error rate for all pairwise comparisons.
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        data = df[[response, factor]].dropna()
        tukey = pairwise_tukeyhsd(data[response], data[factor], alpha=alpha)

        # Parse results into structured data
        pairs = []
        for i in range(len(tukey.summary().data) - 1):
            row = tukey.summary().data[i + 1]
            pairs.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": float(row[2]),
                "p_adj": float(row[3]),
                "lower": float(row[4]),
                "upper": float(row[5]),
                "reject": bool(row[6])
            })

        n_sig = sum(1 for p in pairs if p["reject"])
        n_total = len(pairs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TUKEY'S HSD POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Pairwise Comparisons:<</COLOR>>\n"
        summary += f"{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'p-adj':>8} {'Lower':>8} {'Upper':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig_mark = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['meandiff']:>8.4f} {p['p_adj']:>8.4f} {p['lower']:>8.4f} {p['upper']:>8.4f} {sig_mark:>5}\n"

        summary += f"\n<<COLOR:text>>Summary: {n_sig}/{n_total} pairs significantly different<</COLOR>>\n"
        if n_sig > 0:
            summary += f"<<COLOR:good>>Significant pairs (p < {alpha}):<</COLOR>>\n"
            for p in pairs:
                if p["reject"]:
                    summary += f"  • {p['group1']} vs {p['group2']}: diff = {p['meandiff']:.4f}\n"

        result["summary"] = summary

        # CI plot for pairwise differences
        traces = []
        y_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        traces.append({
            "type": "scatter",
            "x": [p["meandiff"] for p in pairs],
            "y": y_labels,
            "mode": "markers",
            "marker": {"color": colors, "size": 10},
            "error_x": {
                "type": "data",
                "symmetric": False,
                "array": [p["upper"] - p["meandiff"] for p in pairs],
                "arrayminus": [p["meandiff"] - p["lower"] for p in pairs],
                "color": "#5a6a5a"
            },
            "showlegend": False
        })
        result["plots"].append({
            "title": "Tukey HSD — Pairwise Differences with CIs",
            "data": traces,
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(pairs) - 0.5, "line": {"color": "#e89547", "dash": "dash"}}]
            }
        })

        # Group means with SE error bars
        group_stats = data.groupby(factor)[response].agg(['mean', 'std', 'count'])
        group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
        result["plots"].append({
            "title": f"Group Means — {response} by {factor}",
            "data": [{
                "x": [str(g) for g in group_stats.index],
                "y": group_stats['mean'].tolist(),
                "error_y": {"type": "data", "array": group_stats['se'].tolist(), "visible": True, "color": "#5a6a5a"},
                "type": "bar",
                "marker": {"color": "#4a9f6e", "opacity": 0.8}
            }],
            "layout": {
                "height": 280,
                "xaxis": {"title": factor},
                "yaxis": {"title": f"Mean {response}"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Tukey HSD: {n_sig}/{n_total} pairwise comparisons significant at α={alpha}."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig, "n_comparisons": n_total, "alpha": alpha}

    elif analysis_id == "dunnett":
        """
        Dunnett's Test — compare each treatment group to a control group.
        More powerful than Tukey when only control comparisons matter.
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        control = config.get("control")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[response, factor]].dropna()
        levels = data[factor].unique().tolist()

        if control is None or control not in levels:
            control = levels[0]

        control_data = data[data[factor] == control][response].values
        treatments = [lev for lev in levels if lev != control]

        comparisons = []
        for treat in treatments:
            treat_data = data[data[factor] == treat][response].values
            # Dunnett uses pooled variance and critical values; scipy has it from 1.11+
            try:
                from scipy.stats import dunnett as scipy_dunnett
                res = scipy_dunnett(treat_data, control=control_data)
                p_val = float(res.pvalue[0]) if hasattr(res.pvalue, '__len__') else float(res.pvalue)
                stat_val = float(res.statistic[0]) if hasattr(res.statistic, '__len__') else float(res.statistic)
            except (ImportError, AttributeError):
                # Fallback: Welch t-test with Bonferroni correction
                stat_val, p_raw = stats.ttest_ind(treat_data, control_data, equal_var=False)
                stat_val = float(stat_val)
                p_val = min(float(p_raw) * len(treatments), 1.0)

            mean_diff = float(np.mean(treat_data) - np.mean(control_data))
            comparisons.append({
                "treatment": str(treat),
                "control": str(control),
                "mean_diff": mean_diff,
                "statistic": stat_val,
                "p_value": p_val,
                "reject": p_val < alpha
            })

        n_sig = sum(1 for c in comparisons if c["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>DUNNETT'S TEST (vs Control)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Control group:<</COLOR>> {control}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Control: {control} (n={len(control_data)}, mean={np.mean(control_data):.4f})<</COLOR>>\n\n"
        summary += f"{'Treatment':<20} {'Diff':>8} {'Statistic':>10} {'p-value':>8} {'Sig':>5}\n"
        summary += f"{'─' * 55}\n"
        for c in comparisons:
            sig = "<<COLOR:good>>*<</COLOR>>" if c["reject"] else ""
            summary += f"{c['treatment']:<20} {c['mean_diff']:>8.4f} {c['statistic']:>10.4f} {c['p_value']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(comparisons)} treatments differ from control<</COLOR>>\n"

        result["summary"] = summary

        # Bar chart of differences with SE error bars
        se_bars = []
        for c in comparisons:
            treat_vals = data[data[factor] == c["treatment"]][response].values
            se = float(np.sqrt(np.var(treat_vals, ddof=1)/len(treat_vals) + np.var(control_data, ddof=1)/len(control_data)))
            se_bars.append(se)
        result["plots"].append({
            "title": f"Dunnett's Test — Difference from Control ({control})",
            "data": [{
                "type": "bar",
                "x": [c["treatment"] for c in comparisons],
                "y": [c["mean_diff"] for c in comparisons],
                "marker": {"color": ["#4a9f6e" if c["reject"] else "rgba(90,106,90,0.5)" for c in comparisons],
                           "line": {"color": "#4a9f6e", "width": 1}},
                "error_y": {"type": "data", "array": se_bars, "visible": True, "color": "rgba(200,200,200,0.7)"},
                "text": [f"p={c['p_value']:.4f}" for c in comparisons],
                "textposition": "outside"
            }],
            "layout": {"height": 300, "yaxis": {"title": f"Difference from {control}"}}
        })

        result["guide_observation"] = f"Dunnett's test vs {control}: {n_sig}/{len(comparisons)} treatments differ."
        result["statistics"] = {"comparisons": comparisons, "control": str(control)}

    elif analysis_id == "games_howell":
        """
        Games-Howell Test — post-hoc for unequal variances and/or unequal group sizes.
        Does not assume equal variances (unlike Tukey).
        """
        response = config.get("response") or config.get("var")
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[response, factor]].dropna()
        levels = sorted(data[factor].unique().tolist(), key=str)

        group_stats = {}
        for lev in levels:
            g = data[data[factor] == lev][response].values
            group_stats[lev] = {"mean": np.mean(g), "var": np.var(g, ddof=1), "n": len(g)}

        # Pairwise Games-Howell: Welch t with Studentized Range adjustment
        from itertools import combinations
        from scipy.stats import studentized_range

        pairs = []
        level_pairs = list(combinations(levels, 2))
        for g1, g2 in level_pairs:
            s1 = group_stats[g1]
            s2 = group_stats[g2]
            mean_diff = s1["mean"] - s2["mean"]
            se = np.sqrt(s1["var"] / s1["n"] + s2["var"] / s2["n"])

            # Welch-Satterthwaite degrees of freedom
            num = (s1["var"] / s1["n"] + s2["var"] / s2["n"]) ** 2
            denom = (s1["var"] / s1["n"]) ** 2 / (s1["n"] - 1) + (s2["var"] / s2["n"]) ** 2 / (s2["n"] - 1)
            df_welch = num / denom if denom > 0 else 1

            q_stat = abs(mean_diff) / se if se > 0 else 0
            k = len(levels)

            # p-value from studentized range distribution
            try:
                p_val = float(studentized_range.sf(q_stat * np.sqrt(2), k, df_welch))
            except Exception:
                # Fallback to Welch t-test with Bonferroni
                t_stat = mean_diff / se if se > 0 else 0
                p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))
                p_val = min(float(p_raw) * len(level_pairs), 1.0)

            pairs.append({
                "group1": str(g1),
                "group2": str(g2),
                "meandiff": float(mean_diff),
                "se": float(se),
                "q": float(q_stat),
                "df": float(df_welch),
                "p_value": float(p_val),
                "reject": p_val < alpha
            })

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAMES-HOWELL POST-HOC TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor}\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n"
        summary += f"<<COLOR:text>>(Does not assume equal variances)<</COLOR>>\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        for lev in levels:
            s = group_stats[lev]
            summary += f"  {lev}: n={s['n']}, mean={s['mean']:.4f}, std={np.sqrt(s['var']):.4f}\n"

        summary += f"\n{'Group 1':<15} {'Group 2':<15} {'Diff':>8} {'SE':>8} {'q':>8} {'p-val':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['meandiff']:>8.4f} {p['se']:>8.4f} {p['q']:>8.4f} {p['p_value']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(pairs)} pairs significantly different<</COLOR>>\n"

        result["summary"] = summary

        # CI-style plot with error bars (like Tukey)
        y_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        # Approximate CI half-width from SE and df
        ci_half = []
        for p in pairs:
            try:
                t_crit = stats.t.ppf(1 - alpha / 2, p["df"])
                ci_half.append(float(t_crit * p["se"]))
            except Exception:
                ci_half.append(0)
        result["plots"].append({
            "title": "Games-Howell — Pairwise Differences with CI",
            "data": [{
                "type": "scatter",
                "x": [p["meandiff"] for p in pairs],
                "y": y_labels,
                "mode": "markers",
                "marker": {"color": colors, "size": 10},
                "error_x": {"type": "data", "array": ci_half, "color": "rgba(74,159,110,0.6)", "thickness": 2},
                "showlegend": False
            }],
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Mean Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True}
            }
        })

        result["guide_observation"] = f"Games-Howell: {n_sig}/{len(pairs)} pairs significant (unequal variances assumed)."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig}

    elif analysis_id == "dunn":
        """
        Dunn's Test — non-parametric post-hoc after Kruskal-Wallis.
        Uses rank sums with Bonferroni correction.
        """
        var = config.get("var") or config.get("response")
        group_var = config.get("group_var") or config.get("factor")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[[var, group_var]].dropna()
        levels = sorted(data[group_var].unique().tolist(), key=str)

        # Assign overall ranks
        data = data.copy()
        data["_rank"] = stats.rankdata(data[var])
        n_total = len(data)

        group_info = {}
        for lev in levels:
            g = data[data[group_var] == lev]
            group_info[lev] = {"mean_rank": g["_rank"].mean(), "n": len(g), "median": g[var].median()}

        # Pairwise Dunn's test
        from itertools import combinations
        level_pairs = list(combinations(levels, 2))
        n_comparisons = len(level_pairs)

        # Tied-rank correction factor
        ranks = data["_rank"].values
        unique_ranks, counts = np.unique(ranks, return_counts=True)
        tie_correction = 1 - np.sum(counts ** 3 - counts) / (n_total ** 3 - n_total) if n_total > 1 else 1

        pairs = []
        for g1, g2 in level_pairs:
            s1 = group_info[g1]
            s2 = group_info[g2]
            diff = s1["mean_rank"] - s2["mean_rank"]

            # Standard error with tie correction
            se = np.sqrt(tie_correction * (n_total * (n_total + 1) / 12) * (1.0 / s1["n"] + 1.0 / s2["n"]))
            z = diff / se if se > 0 else 0
            p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni

            pairs.append({
                "group1": str(g1),
                "group2": str(g2),
                "rank_diff": float(diff),
                "z": float(z),
                "p_raw": float(p_raw),
                "p_adj": float(p_adj),
                "reject": p_adj < alpha
            })

        n_sig = sum(1 for p in pairs if p["reject"])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>DUNN'S TEST (Post-Hoc for Kruskal-Wallis)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(levels)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Correction:<</COLOR>> Bonferroni\n"
        summary += f"<<COLOR:highlight>>Alpha:<</COLOR>> {alpha}\n\n"

        summary += f"<<COLOR:text>>Group Rank Statistics:<</COLOR>>\n"
        for lev in levels:
            s = group_info[lev]
            summary += f"  {lev}: n={s['n']}, median={s['median']:.4f}, mean rank={s['mean_rank']:.1f}\n"

        summary += f"\n{'Group 1':<15} {'Group 2':<15} {'Rank Diff':>10} {'Z':>8} {'p-raw':>8} {'p-adj':>8} {'Sig':>5}\n"
        summary += f"{'─' * 75}\n"
        for p in pairs:
            sig = "<<COLOR:good>>*<</COLOR>>" if p["reject"] else ""
            summary += f"{p['group1']:<15} {p['group2']:<15} {p['rank_diff']:>10.2f} {p['z']:>8.4f} {p['p_raw']:>8.4f} {p['p_adj']:>8.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>{n_sig}/{len(pairs)} pairs significantly different (Bonferroni-adjusted)<</COLOR>>\n"

        result["summary"] = summary

        # Mean rank comparison plot
        result["plots"].append({
            "title": f"Dunn's Test — Mean Ranks by Group",
            "data": [{
                "type": "bar",
                "x": [str(lev) for lev in levels],
                "y": [group_info[lev]["mean_rank"] for lev in levels],
                "marker": {"color": "#4a9f6e", "line": {"color": "#3d8a5c", "width": 1}},
                "text": [f"n={group_info[lev]['n']}" for lev in levels],
                "textposition": "outside"
            }],
            "layout": {"height": 300, "yaxis": {"title": "Mean Rank"}, "xaxis": {"title": group_var}}
        })

        # Pairwise rank differences plot
        pair_labels = [f"{p['group1']} - {p['group2']}" for p in pairs]
        pair_colors = ["#4a9f6e" if p["reject"] else "#5a6a5a" for p in pairs]
        result["plots"].append({
            "title": "Dunn's Test — Pairwise Rank Differences",
            "data": [{
                "type": "scatter",
                "x": [p["rank_diff"] for p in pairs],
                "y": pair_labels,
                "mode": "markers",
                "marker": {"color": pair_colors, "size": 10},
                "text": [f"z={p['z']:.2f}, p={p['p_adj']:.4f}" for p in pairs],
                "hoverinfo": "text+x",
                "showlegend": False
            }],
            "layout": {
                "height": max(250, 40 * len(pairs)),
                "xaxis": {"title": "Rank Difference", "zeroline": True, "zerolinecolor": "#e89547", "zerolinewidth": 2},
                "yaxis": {"automargin": True}
            }
        })

        result["guide_observation"] = f"Dunn's test: {n_sig}/{len(pairs)} pairwise comparisons significant (Bonferroni)."
        result["statistics"] = {"pairs": pairs, "n_significant": n_sig, "n_comparisons": n_comparisons}

    elif analysis_id == "hotelling_t2":
        """
        Hotelling's T² Test — multivariate extension of the two-sample t-test.
        Tests whether two groups have different mean vectors across multiple response variables.
        """
        responses = config.get("responses", [])
        group_var = config.get("group_var") or config.get("factor")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[responses + [group_var]].dropna()
        groups = sorted(data[group_var].unique().tolist(), key=str)

        if len(groups) != 2:
            result["summary"] = f"Hotelling's T² requires exactly 2 groups. Found {len(groups)}: {groups}"
            return result

        g1_data = data[data[group_var] == groups[0]][responses].values
        g2_data = data[data[group_var] == groups[1]][responses].values
        n1, n2 = len(g1_data), len(g2_data)
        p = len(responses)

        mean1 = np.mean(g1_data, axis=0)
        mean2 = np.mean(g2_data, axis=0)
        diff = mean1 - mean2

        # Pooled covariance matrix
        S1 = np.cov(g1_data, rowvar=False, ddof=1)
        S2 = np.cov(g2_data, rowvar=False, ddof=1)
        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

        # Hotelling's T²
        S_inv = np.linalg.inv(S_pooled * (1.0 / n1 + 1.0 / n2))
        T2 = float(diff @ S_inv @ diff)

        # Convert to F-statistic
        df1 = p
        df2 = n1 + n2 - p - 1
        F_stat = T2 * df2 / (p * (n1 + n2 - 2))
        p_value = float(1 - stats.f.cdf(F_stat, df1, df2))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>HOTELLING'S T² TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Group variable:<</COLOR>> {group_var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        summary += f"{'Variable':<20} {str(groups[0]):>12} {str(groups[1]):>12} {'Difference':>12}\n"
        summary += f"{'─' * 58}\n"
        for i, var in enumerate(responses):
            summary += f"{var:<20} {mean1[i]:>12.4f} {mean2[i]:>12.4f} {diff[i]:>12.4f}\n"

        summary += f"\n<<COLOR:text>>Test Statistics:<</COLOR>>\n"
        summary += f"  Hotelling's T²: {T2:.4f}\n"
        summary += f"  F-statistic: {F_stat:.4f} (df1={df1}, df2={df2})\n"
        summary += f"  p-value: {p_value:.4f}\n\n"

        if p_value < alpha:
            summary += f"<<COLOR:good>>Mean vectors differ significantly (p < {alpha})<</COLOR>>\n"
            summary += f"<<COLOR:text>>The groups have different multivariate profiles across the response variables.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference in mean vectors (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Radar/profile plot of group means
        traces = []
        for idx, grp in enumerate(groups):
            grp_means = [float(mean1[i]) if idx == 0 else float(mean2[i]) for i in range(p)]
            colors = ["#4a9f6e", "#47a5e8"]
            traces.append({
                "type": "scatterpolar",
                "r": grp_means + [grp_means[0]],
                "theta": responses + [responses[0]],
                "name": str(grp),
                "fill": "toself",
                "fillcolor": f"rgba({','.join(str(int(c, 16)) for c in [colors[idx][1:3], colors[idx][3:5], colors[idx][5:7]])}, 0.15)",
                "line": {"color": colors[idx]}
            })
        result["plots"].append({
            "title": "Multivariate Profile — Group Means",
            "data": traces,
            "layout": {"height": 350, "polar": {"radialaxis": {"visible": True}}}
        })

        # Per-variable box plots by group
        box_traces = []
        grp_colors = ["#4a9f6e", "#47a5e8"]
        for gi, grp in enumerate(groups):
            grp_data = data[data[group_var] == grp]
            for vi, var in enumerate(responses):
                box_traces.append({
                    "type": "box", "y": grp_data[var].tolist(),
                    "x": [var] * len(grp_data), "name": str(grp),
                    "marker": {"color": grp_colors[gi]},
                    "legendgroup": str(grp), "showlegend": vi == 0
                })
        result["plots"].append({
            "title": "Response Distributions by Group",
            "data": box_traces,
            "layout": {
                "height": 300, "boxmode": "group",
                "xaxis": {"title": "Response Variable"},
                "yaxis": {"title": "Value"},
                "template": "plotly_white"
            }
        })

        result["guide_observation"] = f"Hotelling's T² = {T2:.2f}, F = {F_stat:.2f}, p = {p_value:.4f}. " + ("Groups differ." if p_value < alpha else "No difference.")
        result["statistics"] = {"T2": T2, "F_statistic": F_stat, "p_value": p_value, "df1": df1, "df2": df2, "mean_diff": diff.tolist()}

    elif analysis_id == "manova":
        """
        One-Way MANOVA — Multivariate Analysis of Variance.
        Tests whether group means differ across multiple response variables simultaneously.
        Reports Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Largest Root.
        """
        responses = config.get("responses", [])
        factor = config.get("factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        data = df[responses + [factor]].dropna()
        groups = sorted(data[factor].unique().tolist(), key=str)
        k = len(groups)
        p = len(responses)
        N = len(data)

        if k < 2:
            result["summary"] = f"MANOVA requires at least 2 groups. Found {k}."
            return result

        # Overall mean
        grand_mean = data[responses].values.mean(axis=0)

        # Between-groups (hypothesis) and within-groups (error) SSCP matrices
        H = np.zeros((p, p))  # Hypothesis SSCP
        E = np.zeros((p, p))  # Error SSCP

        group_means = {}
        for grp in groups:
            grp_data = data[data[factor] == grp][responses].values
            n_g = len(grp_data)
            grp_mean = grp_data.mean(axis=0)
            group_means[grp] = {"mean": grp_mean, "n": n_g}

            diff = (grp_mean - grand_mean).reshape(-1, 1)
            H += n_g * (diff @ diff.T)

            centered = grp_data - grp_mean
            E += centered.T @ centered

        # Four test statistics
        try:
            E_inv = np.linalg.inv(E)
            HE_inv = H @ E_inv
            eigenvalues = np.real(np.linalg.eigvals(HE_inv))
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]
        except np.linalg.LinAlgError:
            result["summary"] = "MANOVA: Error matrix is singular. Check for collinear responses or insufficient data."
            return result

        s = min(p, k - 1)

        # 1. Wilks' Lambda
        wilks = float(np.linalg.det(E) / np.linalg.det(E + H))

        # Wilks' Lambda → F approximation (Rao's F)
        df_h = p * (k - 1)
        df_e = N - k
        if p == 1 or k == 2:
            F_wilks = ((1 - wilks) / wilks) * (df_e / df_h)
            df1_w, df2_w = df_h, df_e
        elif p == 2 or k == 3:
            wilks_sqrt = np.sqrt(wilks) if wilks > 0 else 0
            r = df_e - (p - k + 2) / 2
            F_wilks = ((1 - wilks_sqrt) / wilks_sqrt) * (r / df_h) if wilks_sqrt > 0 else 0
            df1_w, df2_w = df_h, 2 * (r - 1) if r > 1 else 1
        else:
            # General case: Chi-square approximation
            t = np.sqrt((p**2 * (k-1)**2 - 4) / (p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5) > 0 else 1
            df1_w = p * (k - 1)
            ms = N - 1 - (p + k) / 2
            df2_w = ms * t - df1_w / 2 + 1
            wilks_t = wilks ** (1/t) if wilks > 0 and t > 0 else 0
            F_wilks = ((1 - wilks_t) / wilks_t) * (df2_w / df1_w) if wilks_t > 0 else 0

        p_wilks = float(1 - stats.f.cdf(max(F_wilks, 0), max(df1_w, 1), max(df2_w, 1)))

        # 2. Pillai's Trace
        pillai = float(np.sum(eigenvalues / (1 + eigenvalues)))
        df1_p = s * max(p, k - 1)
        df2_p = s * (N - k - p + s)
        F_pillai = (pillai / s) * (df2_p / (max(p, k-1))) / ((1 - pillai / s)) if (1 - pillai / s) > 0 else 0
        p_pillai = float(1 - stats.f.cdf(max(F_pillai, 0), max(df1_p, 1), max(df2_p, 1)))

        # 3. Hotelling-Lawley Trace
        hl_trace = float(np.sum(eigenvalues))
        df1_hl = s * max(p, k - 1)
        df2_hl = s * (N - k - p - 1) + 2
        F_hl = (hl_trace / s) * (df2_hl / max(p, k-1)) if max(p, k-1) > 0 else 0
        p_hl = float(1 - stats.f.cdf(max(F_hl, 0), max(df1_hl, 1), max(df2_hl, 1)))

        # 4. Roy's Largest Root
        roy = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
        df1_r = max(p, k - 1)
        df2_r = N - k - max(p, k-1) + 1
        F_roy = roy * df2_r / df1_r if df1_r > 0 else 0
        p_roy = float(1 - stats.f.cdf(max(F_roy, 0), max(df1_r, 1), max(df2_r, 1)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-WAY MANOVA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        header = f"{'Variable':<20}" + "".join(f"{str(g):>12}" for g in groups)
        summary += header + "\n" + "─" * len(header) + "\n"
        for i, var in enumerate(responses):
            row = f"{var:<20}" + "".join(f"{group_means[g]['mean'][i]:>12.4f}" for g in groups)
            summary += row + "\n"
        summary += "\n"

        summary += f"<<COLOR:text>>MANOVA Test Statistics:<</COLOR>>\n"
        summary += f"{'Test':<25} {'Value':>10} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
        summary += f"{'─' * 62}\n"

        tests = [
            ("Wilks' Lambda", wilks, F_wilks, p_wilks),
            ("Pillai's Trace", pillai, F_pillai, p_pillai),
            ("Hotelling-Lawley Trace", hl_trace, F_hl, p_hl),
            ("Roy's Largest Root", roy, F_roy, p_roy),
        ]
        for name, val, f_val, p_val in tests:
            sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
            summary += f"{name:<25} {val:>10.4f} {f_val:>10.4f} {p_val:>10.4f} {sig:>5}\n"

        summary += f"\n<<COLOR:text>>Eigenvalues of H·E⁻¹:<</COLOR>> {', '.join(f'{e:.4f}' for e in eigenvalues)}\n\n"

        # Overall interpretation (use Pillai's — most robust)
        if p_pillai < alpha:
            summary += f"<<COLOR:good>>Significant multivariate effect (Pillai's Trace, p < {alpha})<</COLOR>>\n"
            summary += f"<<COLOR:text>>Group means differ across the response variables considered jointly.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant multivariate effect (p >= {alpha})<</COLOR>>"

        result["summary"] = summary

        # Group centroid plot (first 2 responses, or first 2 discriminant functions)
        if p >= 2:
            traces = []
            colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84393", "#00b894", "#fdcb6e"]
            for i, grp in enumerate(groups):
                grp_data = data[data[factor] == grp]
                traces.append({
                    "type": "scatter",
                    "x": grp_data[responses[0]].tolist(),
                    "y": grp_data[responses[1]].tolist(),
                    "mode": "markers",
                    "name": str(grp),
                    "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6}
                })
                # Centroid
                traces.append({
                    "type": "scatter",
                    "x": [float(group_means[grp]["mean"][0])],
                    "y": [float(group_means[grp]["mean"][1])],
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 14, "symbol": "diamond", "line": {"color": "white", "width": 2}},
                    "showlegend": False
                })
            result["plots"].append({
                "title": f"Group Centroids — {responses[0]} vs {responses[1]}",
                "data": traces,
                "layout": {"height": 350, "xaxis": {"title": responses[0]}, "yaxis": {"title": responses[1]}}
            })

        # Per-response box plots by group
        box_traces_m = []
        m_colors = ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7", "#e84393", "#00b894", "#fdcb6e"]
        for gi, grp in enumerate(groups):
            grp_d = data[data[factor] == grp]
            for vi, var in enumerate(responses):
                box_traces_m.append({
                    "type": "box", "y": grp_d[var].tolist(),
                    "x": [var] * len(grp_d), "name": str(grp),
                    "marker": {"color": m_colors[gi % len(m_colors)]},
                    "legendgroup": str(grp), "showlegend": vi == 0
                })
        result["plots"].append({
            "title": "Response Distributions by Group",
            "data": box_traces_m,
            "layout": {"height": 300, "boxmode": "group", "xaxis": {"title": "Response"}, "yaxis": {"title": "Value"}, "template": "plotly_white"}
        })

        # Correlation heatmap of response variables
        corr_mat = data[responses].corr().values
        result["plots"].append({
            "data": [{
                "z": corr_mat.tolist(), "x": responses, "y": responses,
                "type": "heatmap", "colorscale": [[0, "#d94a4a"], [0.5, "#f0f4f0"], [1, "#2c5f2d"]],
                "zmin": -1, "zmax": 1,
                "text": [[f"{corr_mat[i][j]:.3f}" for j in range(len(responses))] for i in range(len(responses))],
                "texttemplate": "%{text}", "showscale": True
            }],
            "layout": {"title": "Response Correlation Matrix", "height": 300, "template": "plotly_white"}
        })

        result["guide_observation"] = f"MANOVA: Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}; Pillai's V = {pillai:.4f}, p = {p_pillai:.4f}. " + ("Multivariate effect detected." if p_pillai < alpha else "No multivariate effect.")
        result["statistics"] = {
            "wilks_lambda": wilks, "wilks_F": F_wilks, "wilks_p": p_wilks,
            "pillai_trace": pillai, "pillai_F": F_pillai, "pillai_p": p_pillai,
            "hotelling_lawley": hl_trace, "hl_F": F_hl, "hl_p": p_hl,
            "roys_root": roy, "roy_F": F_roy, "roy_p": p_roy,
            "eigenvalues": eigenvalues.tolist(),
            "n_groups": k, "n_responses": p, "N": N
        }

    elif analysis_id == "nested_anova":
        """
        Nested (Hierarchical) ANOVA — random effects model.
        Tests fixed factor effect while accounting for a random nesting factor.
        E.g., operators nested within machines, batches within suppliers.
        Uses linear mixed-effects model (statsmodels mixedlm).
        """
        response = config.get("response") or config.get("var")
        fixed_factor = config.get("fixed_factor") or config.get("factor")
        random_factor = config.get("random_factor") or config.get("group_var")
        alpha = 1 - config.get("conf", 95) / 100

        try:
            from statsmodels.formula.api import mixedlm

            data = df[[response, fixed_factor, random_factor]].dropna()
            N = len(data)

            # Fit mixed model: response ~ fixed_factor with random intercept for random_factor
            formula = f'{response} ~ C({fixed_factor})'
            model = mixedlm(formula, data, groups=data[random_factor])
            fit = model.fit(reml=True)

            # Extract results
            fixed_effects = {}
            for name, val in fit.fe_params.items():
                pval = float(fit.pvalues[name]) if name in fit.pvalues else None
                se = float(fit.bse[name]) if name in fit.bse else None
                fixed_effects[name] = {"coef": float(val), "se": se, "p_value": pval}

            # Variance components
            var_random = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
            var_residual = float(fit.scale)
            var_total = var_random + var_residual
            icc = var_random / var_total if var_total > 0 else 0

            # Group stats
            fixed_levels = sorted(data[fixed_factor].unique().tolist(), key=str)
            random_levels = sorted(data[random_factor].unique().tolist(), key=str)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>NESTED ANOVA (Mixed-Effects Model)<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Fixed factor:<</COLOR>> {fixed_factor} ({len(fixed_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>Random factor (nesting):<</COLOR>> {random_factor} ({len(random_levels)} levels)\n"
            summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary += f"<<COLOR:text>>Fixed Effects:<</COLOR>>\n"
            summary += f"{'Term':<30} {'Coef':>8} {'SE':>8} {'p-value':>8} {'Sig':>5}\n"
            summary += f"{'─' * 62}\n"
            for name, fe in fixed_effects.items():
                sig = "<<COLOR:good>>*<</COLOR>>" if fe["p_value"] is not None and fe["p_value"] < alpha else ""
                p_str = f"{fe['p_value']:.4f}" if fe["p_value"] is not None else "N/A"
                se_str = f"{fe['se']:.4f}" if fe["se"] is not None else "N/A"
                summary += f"{name:<30} {fe['coef']:>8.4f} {se_str:>8} {p_str:>8} {sig:>5}\n"

            summary += f"\n<<COLOR:text>>Variance Components:<</COLOR>>\n"
            summary += f"  {random_factor} (random): {var_random:.4f} ({icc*100:.1f}% of total)\n"
            summary += f"  Residual: {var_residual:.4f} ({(1-icc)*100:.1f}% of total)\n"
            summary += f"  Total: {var_total:.4f}\n"
            summary += f"  ICC (Intraclass Correlation): {icc:.4f}\n\n"

            if icc > 0.1:
                summary += f"<<COLOR:good>>ICC = {icc:.3f} — substantial variation attributed to {random_factor}.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The nesting structure accounts for {icc*100:.1f}% of the variance. Ignoring it would inflate Type I error.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:text>>ICC = {icc:.3f} — low variation from {random_factor}. A standard ANOVA may suffice.<</COLOR>>\n"

            # Check if fixed factor is significant
            sig_fixed = any(fe["p_value"] is not None and fe["p_value"] < alpha
                           for name, fe in fixed_effects.items() if name != "Intercept")
            if sig_fixed:
                summary += f"<<COLOR:good>>Fixed factor {fixed_factor} has significant effect.<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>Fixed factor {fixed_factor} not significant after accounting for {random_factor}.<</COLOR>>"

            result["summary"] = summary

            # Box plot with nesting structure
            traces = []
            for i, fl in enumerate(fixed_levels):
                subset = data[data[fixed_factor] == fl]
                traces.append({
                    "type": "box",
                    "y": subset[response].tolist(),
                    "x": [str(fl)] * len(subset),
                    "name": str(fl),
                    "boxpoints": "all",
                    "jitter": 0.3,
                    "pointpos": 0,
                    "marker": {"size": 4, "opacity": 0.5}
                })

            result["plots"].append({
                "title": f"Nested ANOVA: {response} by {fixed_factor} (nested in {random_factor})",
                "data": traces,
                "layout": {"height": 300, "yaxis": {"title": response}, "xaxis": {"title": fixed_factor}}
            })

            # Residuals vs fitted values
            fitted_vals = fit.fittedvalues
            resid_vals = fit.resid
            result["plots"].append({
                "title": "Residuals vs Fitted Values",
                "data": [{
                    "x": fitted_vals.tolist(), "y": resid_vals.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted Values"}, "yaxis": {"title": "Residuals"},
                    "shapes": [{"type": "line", "x0": float(fitted_vals.min()), "x1": float(fitted_vals.max()),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # Normal Q-Q plot of residuals
            from scipy import stats as qstats
            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)]
            result["plots"].append({
                "title": "Normal Q-Q Plot of Residuals",
                "data": [
                    {"x": theoretical_q, "y": sorted_resid.tolist(), "mode": "markers", "type": "scatter",
                     "marker": {"color": "#4a9f6e", "size": 4}, "name": "Residuals"},
                    {"x": [theoretical_q[0], theoretical_q[-1]], "y": [theoretical_q[0] * np.std(sorted_resid) + np.mean(sorted_resid),
                     theoretical_q[-1] * np.std(sorted_resid) + np.mean(sorted_resid)],
                     "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": "Reference"}
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Sample Quantiles"},
                    "template": "plotly_white"
                }
            })

            result["guide_observation"] = f"Nested ANOVA: ICC = {icc:.3f} ({icc*100:.1f}% variance from {random_factor}). " + ("Fixed effect significant." if sig_fixed else "Fixed effect not significant.")
            result["statistics"] = {
                "fixed_effects": fixed_effects,
                "var_random": var_random,
                "var_residual": var_residual,
                "icc": icc,
                "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
                "bic": float(fit.bic) if hasattr(fit, 'bic') else None,
                "converged": fit.converged if hasattr(fit, 'converged') else True
            }

        except ImportError:
            result["summary"] = "Nested ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Nested ANOVA error: {str(e)}"

    elif analysis_id == "acceptance_sampling":
        """
        Acceptance Sampling Plans — single and double sampling for lot inspection.
        Computes OC curve, AOQ curve, ATI, and determines accept/reject based on AQL and LTPD.
        Supports both attribute (defective count) and variable (normal) plans.
        """
        import numpy as np
        from scipy import stats as scipy_stats
        from scipy.special import comb as nCr

        plan_type = config.get("plan_type", "single")  # single, double
        n_sample = int(config.get("sample_size", 50))
        accept_num = int(config.get("accept_number", 2))  # Ac
        lot_size = int(config.get("lot_size", 1000))
        aql = float(config.get("aql", 0.01))  # Acceptable Quality Level
        ltpd = float(config.get("ltpd", 0.05))  # Lot Tolerance Percent Defective

        # For double sampling
        n1 = int(config.get("n1", 30))
        c1 = int(config.get("c1", 1))  # Accept on first sample
        r1 = int(config.get("r1", 4))  # Reject on first sample
        n2 = int(config.get("n2", 30))
        c2 = int(config.get("c2", 4))  # Accept on combined

        try:
            # OC curve: probability of acceptance at various defect rates
            p_range = np.linspace(0, min(0.15, ltpd * 3), 200)

            if plan_type == "double":
                # Double sampling OC curve
                pa_values = []
                for p in p_range:
                    if p == 0:
                        pa_values.append(1.0)
                        continue
                    # P(accept on 1st sample): P(d1 <= c1)
                    p_accept_1 = sum(scipy_stats.binom.pmf(d, n1, p) for d in range(c1 + 1))
                    # P(reject on 1st sample): P(d1 >= r1)
                    p_reject_1 = sum(scipy_stats.binom.pmf(d, n1, p) for d in range(r1, n1 + 1))
                    # P(go to 2nd sample): c1 < d1 < r1
                    p_second = 1 - p_accept_1 - p_reject_1
                    # P(accept on combined): P(d1+d2 <= c2) for each d1 in [c1+1, r1-1]
                    p_accept_2 = 0
                    for d1 in range(c1 + 1, r1):
                        p_d1 = scipy_stats.binom.pmf(d1, n1, p)
                        max_d2 = c2 - d1
                        if max_d2 >= 0:
                            p_accept_2 += p_d1 * sum(scipy_stats.binom.pmf(d2, n2, p) for d2 in range(max_d2 + 1))
                    pa = p_accept_1 + p_accept_2
                    pa_values.append(float(min(1.0, max(0.0, pa))))
                pa_values = np.array(pa_values)
                plan_desc = f"Double: n1={n1}, c1={c1}, r1={r1}, n2={n2}, c2={c2}"
            else:
                # Single sampling OC curve using binomial
                pa_values = np.array([float(scipy_stats.binom.cdf(accept_num, n_sample, p)) if p > 0 else 1.0 for p in p_range])
                plan_desc = f"Single: n={n_sample}, Ac={accept_num}"

            # Key probabilities
            pa_aql = float(np.interp(aql, p_range, pa_values))
            pa_ltpd = float(np.interp(ltpd, p_range, pa_values))

            # Producer's risk (alpha) = 1 - P(accept at AQL)
            alpha_risk = 1 - pa_aql
            # Consumer's risk (beta) = P(accept at LTPD)
            beta_risk = pa_ltpd

            # AOQ curve (Average Outgoing Quality)
            aoq_values = pa_values * p_range * (lot_size - n_sample) / lot_size
            aoql = float(np.max(aoq_values))
            aoql_p = float(p_range[np.argmax(aoq_values)])

            # ATI (Average Total Inspection)
            ati_values = n_sample * pa_values + lot_size * (1 - pa_values)

            # OC Curve plot
            result["plots"].append({
                "data": [
                    {
                        "x": (p_range * 100).tolist(), "y": pa_values.tolist(),
                        "mode": "lines", "name": "OC Curve",
                        "line": {"color": "#2c5f2d", "width": 2}
                    },
                    {
                        "x": [aql * 100], "y": [pa_aql],
                        "mode": "markers+text", "name": f"AQL ({aql*100:.1f}%)",
                        "marker": {"color": "#4a90d9", "size": 10},
                        "text": [f"AQL: Pa={pa_aql:.3f}"], "textposition": "top right"
                    },
                    {
                        "x": [ltpd * 100], "y": [pa_ltpd],
                        "mode": "markers+text", "name": f"LTPD ({ltpd*100:.1f}%)",
                        "marker": {"color": "#d94a4a", "size": 10},
                        "text": [f"LTPD: Pa={pa_ltpd:.3f}"], "textposition": "top left"
                    }
                ],
                "layout": {
                    "title": f"Operating Characteristic (OC) Curve — {plan_desc}",
                    "xaxis": {"title": "Lot Defect Rate (%)"},
                    "yaxis": {"title": "Probability of Acceptance", "range": [0, 1.05]},
                    "template": "plotly_white"
                }
            })

            # AOQ Curve plot
            result["plots"].append({
                "data": [
                    {
                        "x": (p_range * 100).tolist(), "y": (aoq_values * 100).tolist(),
                        "mode": "lines", "name": "AOQ Curve",
                        "line": {"color": "#d9a04a", "width": 2}
                    },
                    {
                        "x": [aoql_p * 100], "y": [aoql * 100],
                        "mode": "markers+text", "name": f"AOQL={aoql*100:.3f}%",
                        "marker": {"color": "#d94a4a", "size": 10},
                        "text": [f"AOQL={aoql*100:.3f}%"], "textposition": "top right"
                    }
                ],
                "layout": {
                    "title": "Average Outgoing Quality (AOQ) Curve",
                    "xaxis": {"title": "Incoming Defect Rate (%)"},
                    "yaxis": {"title": "Average Outgoing Quality (%)"},
                    "template": "plotly_white"
                }
            })

            result["summary"] = f"**Acceptance Sampling Plan**\n\n**Plan:** {plan_desc}\n**Lot size:** {lot_size}\n\n| Metric | Value |\n|---|---|\n| P(accept) at AQL ({aql*100:.1f}%) | {pa_aql:.4f} |\n| P(accept) at LTPD ({ltpd*100:.1f}%) | {pa_ltpd:.4f} |\n| Producer's risk (α) | {alpha_risk:.4f} ({alpha_risk*100:.1f}%) |\n| Consumer's risk (β) | {beta_risk:.4f} ({beta_risk*100:.1f}%) |\n| AOQL | {aoql*100:.4f}% at p={aoql_p*100:.2f}% |\n| ATI at AQL | {float(np.interp(aql, p_range, ati_values)):.0f} units |"

            result["guide_observation"] = f"Acceptance sampling ({plan_desc}): α={alpha_risk:.3f}, β={beta_risk:.3f}, AOQL={aoql*100:.4f}%."

            result["statistics"] = {
                "plan_type": plan_type,
                "sample_size": n_sample if plan_type == "single" else n1 + n2,
                "lot_size": lot_size,
                "aql": aql,
                "ltpd": ltpd,
                "pa_at_aql": pa_aql,
                "pa_at_ltpd": pa_ltpd,
                "producers_risk_alpha": alpha_risk,
                "consumers_risk_beta": beta_risk,
                "aoql": aoql,
                "aoql_defect_rate": aoql_p
            }

        except Exception as e:
            result["summary"] = f"Acceptance sampling error: {str(e)}"

    elif analysis_id == "glm":
        """
        General Linear Model — unified engine for ANOVA, ANCOVA, multivariate regression,
        and mixed-effects models. Handles:
        - Pure ANOVA (factors only)
        - Pure regression (covariates only)
        - ANCOVA (factors + covariates with interactions and LS-Means)
        - Mixed models (fixed + random factors)
        - Two-way+ interactions between factors and factor*covariate
        Produces Type III ANOVA table, LS-Means, effect sizes, interaction plots,
        and full residual diagnostics (4-panel).
        """
        response = config.get("response") or config.get("var")
        fixed_factors = config.get("fixed_factors", [])
        random_factors = config.get("random_factors", [])
        covariates = config.get("covariates", [])
        include_interactions = config.get("interactions", True)
        include_factor_cov_interactions = config.get("factor_covariate_interactions", False)
        alpha = 1 - config.get("conf", 95) / 100

        # Support single-value fallbacks from generic dialogs
        if not fixed_factors and config.get("factor"):
            fixed_factors = [config["factor"]]
        if not random_factors and config.get("random_factor"):
            random_factors = [config["random_factor"]]
        if not covariates and config.get("covariate"):
            covariates = [config["covariate"]]

        all_cols = [response] + fixed_factors + random_factors + covariates
        try:
            data = df[all_cols].dropna()
            N = len(data)
            from scipy import stats as qstats

            # ── Build formula ──
            terms = []
            for f in fixed_factors:
                terms.append(f"C({f})")
            for c in covariates:
                terms.append(c)

            # Factor*Factor interactions
            if include_interactions and len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        terms.append(f"C({fixed_factors[i]}):C({fixed_factors[j]})")

            # Factor*Covariate interactions (ANCOVA: test homogeneity of slopes)
            if covariates and fixed_factors:
                if include_factor_cov_interactions or len(fixed_factors) == 1:
                    # Auto-include for single-factor ANCOVA to test assumption
                    for f in fixed_factors:
                        for c in covariates:
                            terms.append(f"C({f}):{c}")

            formula = f"{response} ~ " + " + ".join(terms) if terms else f"{response} ~ 1"

            # Determine model type label
            has_factors = len(fixed_factors) > 0
            has_covariates = len(covariates) > 0
            has_random = len(random_factors) > 0
            if has_factors and has_covariates:
                model_label = "ANCOVA" if not has_random else "Mixed ANCOVA"
            elif has_factors and not has_covariates:
                model_label = "ANOVA (GLM)" if not has_random else "Mixed-Effects ANOVA"
            elif has_covariates and not has_factors:
                model_label = "Multiple Regression" if len(covariates) > 1 else "Simple Regression"
            else:
                model_label = "Intercept-Only Model"

            # ═══════════════════════════════════════════
            # MIXED MODEL (random factors present)
            # ═══════════════════════════════════════════
            if has_random:
                from statsmodels.formula.api import mixedlm
                group_var = random_factors[0]
                model = mixedlm(formula, data, groups=data[group_var])
                fit = model.fit(reml=True)

                var_random = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
                var_residual = float(fit.scale)
                var_total = var_random + var_residual
                icc = var_random / var_total if var_total > 0 else 0

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                summary_text += f"<<COLOR:highlight>>Fixed:<</COLOR>> {', '.join(fixed_factors + covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Random:<</COLOR>> {', '.join(random_factors)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

                summary_text += f"<<COLOR:text>>Fixed Effects:<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'Sig':>5}\n"
                summary_text += f"{'─' * 80}\n"
                for name in fit.fe_params.index:
                    coef = float(fit.fe_params[name])
                    se = float(fit.bse[name]) if name in fit.bse.index else None
                    pv = float(fit.pvalues[name]) if name in fit.pvalues.index else None
                    z = coef / se if se and se > 0 else 0
                    sig = "<<COLOR:good>>*<</COLOR>>" if pv is not None and pv < alpha else ""
                    p_str = f"{pv:.4f}" if pv is not None else "N/A"
                    se_str = f"{se:.4f}" if se is not None else "N/A"
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {sig:>5}\n"

                summary_text += f"\n<<COLOR:text>>Variance Components:<</COLOR>>\n"
                summary_text += f"  {group_var} (random): {var_random:.4f} ({icc*100:.1f}% of total)\n"
                summary_text += f"  Residual: {var_residual:.4f} ({(1-icc)*100:.1f}% of total)\n"
                summary_text += f"  ICC (Intraclass Correlation): {icc:.4f}\n"

                if icc > 0.1:
                    summary_text += f"\n<<COLOR:good>>ICC = {icc:.3f} — substantial clustering. Mixed model is appropriate.<</COLOR>>"
                else:
                    summary_text += f"\n<<COLOR:text>>ICC = {icc:.3f} — low clustering. A fixed-effects model may suffice.<</COLOR>>"

                fitted_vals = fit.fittedvalues
                resid_vals = fit.resid

                result["statistics"] = {
                    "model_type": "mixed", "model_label": model_label,
                    "n": N, "formula": formula,
                    "var_random": var_random, "var_residual": var_residual,
                    "icc": icc,
                    "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
                }

            # ═══════════════════════════════════════════
            # FIXED MODEL (OLS — ANOVA/ANCOVA/Regression)
            # ═══════════════════════════════════════════
            else:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols

                model = ols(formula, data=data).fit()

                # Type III ANOVA table
                try:
                    anova_table = sm.stats.anova_lm(model, typ=3)
                except Exception:
                    anova_table = sm.stats.anova_lm(model, typ=2)

                # Compute effect sizes (partial eta-squared)
                ss_residual = float(anova_table.loc['Residual', 'sum_sq']) if 'Residual' in anova_table.index else 0
                ss_total = float(anova_table['sum_sq'].sum())

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                if fixed_factors:
                    summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(fixed_factors)}\n"
                if covariates:
                    summary_text += f"<<COLOR:highlight>>Covariates:<</COLOR>> {', '.join(covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}, R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}\n\n"

                # ANOVA table with partial eta-squared
                summary_text += f"<<COLOR:text>>Analysis of Variance (Type III SS):<</COLOR>>\n"
                eta_header = " η²p" if has_factors else ""
                summary_text += f"{'Source':<30} {'DF':>5} {'Adj SS':>12} {'Adj MS':>12} {'F':>10} {'p-value':>10}{eta_header:>8}\n"
                summary_text += f"{'─' * (87 + (8 if has_factors else 0))}\n"

                for idx in anova_table.index:
                    row = anova_table.loc[idx]
                    df_val = int(row['df']) if 'df' in row else ""
                    ss = float(row['sum_sq']) if 'sum_sq' in row else 0
                    ms = float(row['mean_sq']) if 'mean_sq' in row else 0
                    f_val = float(row['F']) if 'F' in row and not np.isnan(row['F']) else None
                    pv = float(row['PR(>F)']) if 'PR(>F)' in row and not np.isnan(row['PR(>F)']) else None
                    sig = "<<COLOR:good>>*<</COLOR>>" if pv is not None and pv < alpha else ""
                    f_str = f"{f_val:.4f}" if f_val is not None else ""
                    p_str = f"{pv:.4f}" if pv is not None else ""
                    # Partial eta-squared = SS_effect / (SS_effect + SS_error)
                    if has_factors and idx != 'Residual' and idx != 'Intercept' and ss_residual > 0:
                        eta_p = ss / (ss + ss_residual)
                        eta_str = f"{eta_p:>7.3f}"
                    else:
                        eta_str = "" if has_factors else ""
                    summary_text += f"{str(idx):<30} {df_val:>5} {ss:>12.4f} {ms:>12.4f} {f_str:>10} {p_str:>10} {sig} {eta_str}\n"

                summary_text += f"\n<<COLOR:text>>Model Summary:<</COLOR>>\n"
                summary_text += f"  S (root MSE): {np.sqrt(model.mse_resid):.4f}\n"
                summary_text += f"  R²: {model.rsquared:.4f}  Adj R²: {model.rsquared_adj:.4f}\n"
                summary_text += f"  AIC: {model.aic:.1f}  BIC: {model.bic:.1f}\n"

                # Coefficients table
                summary_text += f"\n<<COLOR:text>>Coefficients:<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'t':>8} {'p-value':>10}\n"
                summary_text += f"{'─' * 75}\n"
                for name in model.params.index:
                    coef = float(model.params[name])
                    se = float(model.bse[name])
                    t = float(model.tvalues[name])
                    pv = float(model.pvalues[name])
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {pv:>10.4f}\n"

                # ── LS-Means (Adjusted Means) for ANCOVA ──
                if has_factors and has_covariates:
                    summary_text += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
                    summary_text += f"<<COLOR:text>>Least-Squares Means (Adjusted Means):<</COLOR>>\n"
                    summary_text += f"<<COLOR:text>>Covariates held at their means: " + ", ".join([f"{c}={data[c].mean():.4f}" for c in covariates]) + "<</COLOR>>\n\n"

                    for factor in fixed_factors:
                        levels = sorted(data[factor].unique().tolist(), key=str)
                        raw_means = data.groupby(factor)[response].mean()
                        raw_sds = data.groupby(factor)[response].std()
                        raw_ns = data.groupby(factor)[response].count()

                        # Compute LS-means by predicting at covariate means
                        cov_means = {c: data[c].mean() for c in covariates}
                        ls_means = {}
                        for lev in levels:
                            pred_data = data[data[factor] == lev].copy()
                            for c in covariates:
                                pred_data[c] = cov_means[c]
                            ls_means[lev] = float(model.predict(pred_data).mean())

                        summary_text += f"  {factor}:\n"
                        summary_text += f"  {'Level':<20} {'N':>6} {'Raw Mean':>10} {'Adj Mean':>10} {'Std Dev':>10}\n"
                        summary_text += f"  {'─' * 58}\n"
                        for lev in levels:
                            rm = float(raw_means[lev])
                            adj = ls_means[lev]
                            sd = float(raw_sds[lev])
                            n_lev = int(raw_ns[lev])
                            summary_text += f"  {str(lev):<20} {n_lev:>6} {rm:>10.4f} {adj:>10.4f} {sd:>10.4f}\n"

                        diff = max(ls_means.values()) - min(ls_means.values())
                        raw_diff = float(raw_means.max()) - float(raw_means.min())
                        if abs(raw_diff - diff) > 0.01 * abs(raw_diff + 0.001):
                            summary_text += f"\n  <<COLOR:highlight>>Note: Adjusted means differ from raw means — covariate adjustment matters.<</COLOR>>\n"
                            summary_text += f"  Raw range: {raw_diff:.4f} → Adjusted range: {diff:.4f}\n"

                    # Homogeneity of slopes test (factor*covariate interaction significance)
                    fxcov_terms = [f"C({f}):{c}" for f in fixed_factors for c in covariates]
                    sig_interactions = []
                    for term in fxcov_terms:
                        for idx in anova_table.index:
                            if term.replace("C(", "").replace(")", "") in str(idx) or str(idx) == term:
                                pv_term = float(anova_table.loc[idx, 'PR(>F)']) if 'PR(>F)' in anova_table.columns and not np.isnan(anova_table.loc[idx, 'PR(>F)']) else None
                                if pv_term is not None and pv_term < alpha:
                                    sig_interactions.append((str(idx), pv_term))

                    if sig_interactions:
                        summary_text += f"\n<<COLOR:bad>>⚠ Homogeneity of Slopes Violated:<</COLOR>>\n"
                        for term, pv in sig_interactions:
                            summary_text += f"  {term}: p = {pv:.4f} — slopes differ across groups. ANCOVA assumption violated.\n"
                        summary_text += f"  <<COLOR:text>>Consider: separate regressions per group, or remove the covariate.<</COLOR>>\n"
                    elif has_factors and has_covariates:
                        summary_text += f"\n<<COLOR:good>>Homogeneity of slopes OK — factor*covariate interactions are not significant.<</COLOR>>\n"

                fitted_vals = model.fittedvalues
                resid_vals = model.resid

                result["statistics"] = {
                    "model_type": "fixed", "model_label": model_label,
                    "n": N, "formula": formula,
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_pvalue": float(model.f_pvalue),
                    "aic": float(model.aic), "bic": float(model.bic),
                    "root_mse": float(np.sqrt(model.mse_resid)),
                }

            result["summary"] = summary_text

            # ═══════════════════════════════════════════
            # PLOTS — Full 4-panel residual diagnostics
            # ═══════════════════════════════════════════

            # 1. Residuals vs Fitted
            result["plots"].append({
                "title": "Residuals vs Fitted Values",
                "data": [{
                    "x": fitted_vals.tolist(), "y": resid_vals.tolist(),
                    "mode": "markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6}
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted Value"}, "yaxis": {"title": "Residual"},
                    "shapes": [{"type": "line", "x0": float(fitted_vals.min()), "x1": float(fitted_vals.max()),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # 2. Normal Q-Q Plot
            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)]
            result["plots"].append({
                "title": "Normal Probability Plot of Residuals",
                "data": [
                    {"x": theoretical_q, "y": sorted_resid.tolist(), "mode": "markers", "type": "scatter",
                     "marker": {"color": "#4a9f6e", "size": 4}, "name": "Residuals"},
                    {"x": [theoretical_q[0], theoretical_q[-1]],
                     "y": [theoretical_q[0] * np.std(sorted_resid) + np.mean(sorted_resid),
                           theoretical_q[-1] * np.std(sorted_resid) + np.mean(sorted_resid)],
                     "mode": "lines", "line": {"color": "#e89547", "dash": "dash"}, "name": "Reference"}
                ],
                "layout": {"height": 280, "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Sample Quantiles"}, "template": "plotly_white"}
            })

            # 3. Residuals Histogram
            hist_vals, bin_edges = np.histogram(resid_vals.values, bins='auto')
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append({
                "title": "Histogram of Residuals",
                "data": [{"type": "bar", "x": bin_centers, "y": hist_vals.tolist(),
                          "marker": {"color": "#4a90d9", "opacity": 0.7}}],
                "layout": {"height": 250, "xaxis": {"title": "Residual"}, "yaxis": {"title": "Frequency"}, "template": "plotly_white"}
            })

            # 4. Residuals vs Observation Order
            result["plots"].append({
                "title": "Residuals vs Observation Order",
                "data": [{
                    "x": list(range(1, len(resid_vals) + 1)), "y": resid_vals.tolist(),
                    "mode": "lines+markers", "type": "scatter",
                    "marker": {"color": "#4a9f6e", "size": 3}, "line": {"color": "#4a9f6e", "width": 1}
                }],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Observation Order"}, "yaxis": {"title": "Residual"},
                    "shapes": [{"type": "line", "x0": 1, "x1": len(resid_vals),
                                "y0": 0, "y1": 0, "line": {"color": "#e89547", "dash": "dash"}}],
                    "template": "plotly_white"
                }
            })

            # ── Main Effects Plots ──
            if fixed_factors:
                for factor in fixed_factors:
                    levels = sorted(data[factor].unique().tolist(), key=str)
                    means = [float(data[data[factor] == lev][response].mean()) for lev in levels]
                    cis = [1.96 * float(data[data[factor] == lev][response].std()) / np.sqrt(len(data[data[factor] == lev])) for lev in levels]
                    grand_mean = float(data[response].mean())
                    result["plots"].append({
                        "title": f"Main Effects Plot: {factor}",
                        "data": [{
                            "x": [str(l) for l in levels], "y": means,
                            "error_y": {"type": "data", "array": cis, "visible": True, "color": "#4a90d9"},
                            "mode": "lines+markers", "type": "scatter",
                            "marker": {"color": "#4a90d9", "size": 8},
                            "line": {"color": "#4a90d9", "width": 2}, "name": "Mean"
                        }],
                        "layout": {
                            "height": 260, "xaxis": {"title": factor}, "yaxis": {"title": f"Mean of {response}"},
                            "shapes": [{"type": "line", "x0": -0.5, "x1": len(levels) - 0.5,
                                        "y0": grand_mean, "y1": grand_mean,
                                        "line": {"color": "#999", "dash": "dash", "width": 1}}],
                            "template": "plotly_white"
                        }
                    })

            # ── Interaction Plots (Factor × Factor) ──
            if len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        f1, f2 = fixed_factors[i], fixed_factors[j]
                        traces = []
                        colors_int = ["#4a90d9", "#d94a4a", "#4a9f6e", "#d9a04a", "#9b59b6", "#e67e22"]
                        f2_levels = sorted(data[f2].unique().tolist(), key=str)
                        f1_levels = sorted(data[f1].unique().tolist(), key=str)
                        for ci, lev2 in enumerate(f2_levels):
                            sub = data[data[f2] == lev2]
                            means_int = [float(sub[sub[f1] == lev1][response].mean()) if len(sub[sub[f1] == lev1]) > 0 else None for lev1 in f1_levels]
                            traces.append({
                                "x": [str(l) for l in f1_levels], "y": means_int,
                                "mode": "lines+markers", "name": f"{f2}={lev2}",
                                "marker": {"color": colors_int[ci % len(colors_int)], "size": 7},
                                "line": {"color": colors_int[ci % len(colors_int)], "width": 2}
                            })
                        result["plots"].append({
                            "title": f"Interaction Plot: {f1} × {f2}",
                            "data": traces,
                            "layout": {"height": 280, "xaxis": {"title": f1}, "yaxis": {"title": f"Mean of {response}"}, "template": "plotly_white"}
                        })

            # ── ANCOVA: Covariate scatter by factor ──
            if has_factors and has_covariates:
                for c in covariates:
                    for f in fixed_factors:
                        traces_cov = []
                        colors_cov = ["#4a90d9", "#d94a4a", "#4a9f6e", "#d9a04a", "#9b59b6"]
                        f_levels = sorted(data[f].unique().tolist(), key=str)
                        for fi, lev in enumerate(f_levels):
                            sub = data[data[f] == lev]
                            traces_cov.append({
                                "x": sub[c].tolist(), "y": sub[response].tolist(),
                                "mode": "markers", "name": f"{f}={lev}",
                                "marker": {"color": colors_cov[fi % len(colors_cov)], "size": 5, "opacity": 0.7}
                            })
                            # Add regression line per group
                            if len(sub) > 2:
                                slope, intercept, _, _, _ = qstats.linregress(sub[c].values, sub[response].values)
                                x_line = [float(sub[c].min()), float(sub[c].max())]
                                y_line = [intercept + slope * x for x in x_line]
                                traces_cov.append({
                                    "x": x_line, "y": y_line,
                                    "mode": "lines", "name": f"{lev} fit",
                                    "line": {"color": colors_cov[fi % len(colors_cov)], "width": 1.5, "dash": "dash"},
                                    "showlegend": False
                                })
                        result["plots"].append({
                            "title": f"Covariate Plot: {response} vs {c} by {f}",
                            "data": traces_cov,
                            "layout": {"height": 300, "xaxis": {"title": c}, "yaxis": {"title": response}, "template": "plotly_white"}
                        })

            result["guide_observation"] = f"{model_label}: {response} ~ {' + '.join(fixed_factors + covariates + random_factors)}. N={N}."

        except Exception as e:
            result["summary"] = f"GLM error: {str(e)}"

    elif analysis_id == "manova":
        """
        Multivariate ANOVA — tests group differences across multiple response variables.
        Uses Pillai's trace, Wilks' lambda, Hotelling-Lawley, Roy's greatest root.
        """
        responses = config.get("responses", [])
        factor = config.get("factor") or config.get("group_var") or config.get("group")
        alpha = 1 - config.get("conf", 95) / 100

        if not responses and config.get("response"):
            responses = [config["response"]]

        try:
            all_cols = responses + [factor]
            data = df[all_cols].dropna()
            N = len(data)
            groups = sorted(data[factor].unique().tolist(), key=str)
            k = len(groups)
            p = len(responses)

            # Compute group means and overall mean
            overall_mean = data[responses].mean().values
            group_data = {g: data[data[factor] == g][responses].values for g in groups}
            group_means = {g: v.mean(axis=0) for g, v in group_data.items()}
            group_ns = {g: len(v) for g, v in group_data.items()}

            # Between-group SSCP matrix (H)
            H = np.zeros((p, p))
            for g in groups:
                diff = (group_means[g] - overall_mean).reshape(-1, 1)
                H += group_ns[g] * diff @ diff.T

            # Within-group SSCP matrix (E)
            E = np.zeros((p, p))
            for g in groups:
                centered = group_data[g] - group_means[g]
                E += centered.T @ centered

            # Test statistics
            df_h = k - 1
            df_e = N - k

            # Eigenvalues of E^-1 H
            try:
                E_inv = np.linalg.inv(E)
                eigvals = np.real(np.linalg.eigvals(E_inv @ H))
                eigvals = np.sort(eigvals)[::-1]
            except np.linalg.LinAlgError:
                eigvals = np.array([0.0] * p)

            # Pillai's trace
            pillai = np.sum(eigvals / (1 + eigvals))

            # Wilks' lambda
            wilks = np.prod(1 / (1 + eigvals))

            # Hotelling-Lawley trace
            hotelling = np.sum(eigvals)

            # Roy's greatest root
            roy = eigvals[0] if len(eigvals) > 0 else 0

            # Approximate F-test for Wilks' lambda
            s = min(p, df_h)
            m = (abs(p - df_h) - 1) / 2
            n_param = (df_e - p - 1) / 2
            if s > 0 and wilks > 0:
                r = df_e - (p - df_h + 1) / 2
                u = (p * df_h - 2) / 4
                if p**2 + df_h**2 - 5 > 0:
                    t = np.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5))
                else:
                    t = 1
                df1 = p * df_h
                df2 = r * t - 2 * u
                if df2 > 0:
                    f_wilks = ((1 - wilks**(1/t)) / (wilks**(1/t))) * (df2 / df1)
                    from scipy import stats as fstats
                    p_wilks = 1 - fstats.f.cdf(f_wilks, df1, df2)
                else:
                    f_wilks = None
                    p_wilks = None
            else:
                f_wilks = None
                p_wilks = None

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
            summary_text += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Multivariate Test Statistics:<</COLOR>>\n"
            summary_text += f"{'Test':<25} {'Value':>10} {'Approx F':>10} {'p-value':>10}\n"
            summary_text += f"{'─' * 57}\n"
            pillai_label = "Pillai's Trace"
            summary_text += f"{pillai_label:<25} {pillai:>10.4f} {'':>10} {'':>10}\n"
            wilks_f_str = f"{f_wilks:.4f}" if f_wilks is not None else "N/A"
            wilks_p_str = f"{p_wilks:.4f}" if p_wilks is not None else "N/A"
            wilks_label = "Wilks' Lambda"
            summary_text += f"{wilks_label:<25} {wilks:>10.4f} {wilks_f_str:>10} {wilks_p_str:>10}\n"
            summary_text += f"{'Hotelling-Lawley':<25} {hotelling:>10.4f} {'':>10} {'':>10}\n"
            roy_label = "Roy's Greatest Root"
            summary_text += f"{roy_label:<25} {roy:>10.4f} {'':>10} {'':>10}\n\n"

            # Univariate ANOVAs
            summary_text += f"<<COLOR:text>>Univariate ANOVA per Response:<</COLOR>>\n"
            summary_text += f"{'Response':<20} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
            summary_text += f"{'─' * 47}\n"
            from scipy import stats as fstats
            for resp in responses:
                group_vals = [data[data[factor] == g][resp].values for g in groups]
                f_stat, p_val = fstats.f_oneway(*group_vals)
                sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
                summary_text += f"{resp:<20} {f_stat:>10.4f} {p_val:>10.4f} {sig:>5}\n"

            if p_wilks is not None and p_wilks < alpha:
                summary_text += f"\n<<COLOR:good>>Significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"
            elif p_wilks is not None:
                summary_text += f"\n<<COLOR:text>>No significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"

            result["summary"] = summary_text

            # Mean profiles plot
            for resp in responses:
                means = [float(data[data[factor] == g][resp].mean()) for g in groups]
                sds = [float(data[data[factor] == g][resp].std()) for g in groups]
                result["plots"].append({
                    "title": f"Group Means: {resp} by {factor}",
                    "data": [{
                        "x": [str(g) for g in groups], "y": means,
                        "error_y": {"type": "data", "array": sds, "visible": True},
                        "type": "bar", "marker": {"color": "#4a90d9"}
                    }],
                    "layout": {"height": 250, "yaxis": {"title": resp}, "xaxis": {"title": factor}, "template": "plotly_white"}
                })

            result["statistics"] = {
                "pillai": float(pillai), "wilks_lambda": float(wilks),
                "hotelling_lawley": float(hotelling), "roys_greatest_root": float(roy),
                "f_wilks": float(f_wilks) if f_wilks else None,
                "p_wilks": float(p_wilks) if p_wilks else None,
                "n_groups": k, "n_responses": p, "n": N
            }
            result["guide_observation"] = f"MANOVA: {', '.join(responses)} by {factor}. Wilks' Λ={wilks:.4f}" + (f", p={p_wilks:.4f}" if p_wilks else "") + "."

        except Exception as e:
            result["summary"] = f"MANOVA error: {str(e)}"

    elif analysis_id == "tolerance_interval":
        """
        Tolerance Intervals — bounds containing a specified proportion of the population
        with a given confidence level. Normal-based and non-parametric methods.
        """
        var = config.get("var") or config.get("response")
        conf = config.get("conf", 95) / 100
        coverage = config.get("coverage", 95) / 100
        method = config.get("method", "normal")

        try:
            vals = df[var].dropna().values.astype(float)
            n = len(vals)
            xbar = float(np.mean(vals))
            s = float(np.std(vals, ddof=1))

            from scipy import stats as tstats

            if method == "nonparametric":
                # Non-parametric: order statistics
                sorted_vals = np.sort(vals)
                # Find r such that P(X_(r) to X_(n-r+1) covers coverage% with conf% confidence)
                from scipy.stats import beta as beta_dist
                best_r = 1
                for r in range(1, n // 2):
                    prob = beta_dist.cdf(coverage, n - 2 * r + 1, 2 * r)
                    if prob >= conf:
                        best_r = r
                        break
                lower = float(sorted_vals[best_r - 1])
                upper = float(sorted_vals[n - best_r])
                k_factor = None
                method_desc = f"Non-parametric (order statistics r={best_r})"
            else:
                # Normal-based: k-factor from chi-squared and normal quantiles
                z_p = float(tstats.norm.ppf((1 + coverage) / 2))
                chi2_val = float(tstats.chi2.ppf(1 - conf, n - 1))
                k_factor = z_p * np.sqrt((n - 1) * (1 + 1 / n) / chi2_val)
                lower = xbar - k_factor * s
                upper = xbar + k_factor * s
                method_desc = f"Normal (k={k_factor:.4f})"

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>TOLERANCE INTERVAL<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method_desc}\n\n"
            summary_text += f"<<COLOR:highlight>>Confidence:<</COLOR>> {conf*100:.0f}%\n"
            summary_text += f"<<COLOR:highlight>>Coverage:<</COLOR>> {coverage*100:.0f}%\n\n"
            summary_text += f"<<COLOR:text>>Tolerance Interval:<</COLOR>> [{lower:.4f}, {upper:.4f}]\n"
            summary_text += f"<<COLOR:text>>Mean:<</COLOR>> {xbar:.4f}\n"
            summary_text += f"<<COLOR:text>>Std Dev:<</COLOR>> {s:.4f}\n\n"
            summary_text += f"<<COLOR:highlight>>Interpretation:<</COLOR>> With {conf*100:.0f}% confidence, at least {coverage*100:.0f}% of the population falls between {lower:.4f} and {upper:.4f}."

            result["summary"] = summary_text

            # Histogram with tolerance bounds
            hist_vals, bin_edges = np.histogram(vals, bins='auto')
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append({
                "title": f"Tolerance Interval: {var}",
                "data": [
                    {"type": "bar", "x": bin_centers, "y": hist_vals.tolist(), "marker": {"color": "#4a90d9", "opacity": 0.7}, "name": "Data"},
                ],
                "layout": {
                    "height": 300, "xaxis": {"title": var}, "yaxis": {"title": "Frequency"},
                    "shapes": [
                        {"type": "line", "x0": lower, "x1": lower, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
                        {"type": "line", "x0": upper, "x1": upper, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
                        {"type": "line", "x0": xbar, "x1": xbar, "y0": 0, "y1": max(hist_vals) * 1.1, "line": {"color": "#4a9f6e", "width": 2}},
                    ],
                    "annotations": [
                        {"x": lower, "y": max(hist_vals) * 1.05, "text": f"Lower: {lower:.2f}", "showarrow": False, "font": {"color": "#d94a4a"}},
                        {"x": upper, "y": max(hist_vals) * 1.05, "text": f"Upper: {upper:.2f}", "showarrow": False, "font": {"color": "#d94a4a"}},
                    ],
                    "template": "plotly_white"
                }
            })

            result["statistics"] = {
                "mean": xbar, "std": s, "n": n,
                "lower": lower, "upper": upper,
                "confidence": conf, "coverage": coverage,
                "k_factor": k_factor, "method": method
            }
            result["guide_observation"] = f"Tolerance interval for {var}: [{lower:.4f}, {upper:.4f}] ({conf*100:.0f}% conf, {coverage*100:.0f}% coverage)."

        except Exception as e:
            result["summary"] = f"Tolerance interval error: {str(e)}"

    elif analysis_id == "variance_components":
        """
        Variance Components — decomposes total variance into components from random factors.
        Uses ANOVA-based (Type I MS) or REML method.
        """
        response = config.get("response") or config.get("var")
        factors = config.get("factors", [])
        if not factors and config.get("factor"):
            factors = [config["factor"]]
        method = config.get("method", "anova")

        try:
            all_cols = [response] + factors
            data = df[all_cols].dropna()
            N = len(data)

            components = {}
            total_var = float(data[response].var())

            if method == "reml" and len(factors) == 1:
                from statsmodels.formula.api import mixedlm
                formula = f"{response} ~ 1"
                model = mixedlm(formula, data, groups=data[factors[0]])
                fit = model.fit(reml=True)
                var_factor = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, 'iloc') else float(fit.cov_re)
                var_error = float(fit.scale)
                components[factors[0]] = var_factor
                components["Error"] = var_error
            else:
                # ANOVA method: for each factor, compute between-group MS and within-group MS
                from scipy import stats as vstats
                remaining_var = total_var
                for factor in factors:
                    groups = data.groupby(factor)[response]
                    group_means = groups.mean()
                    grand_mean = data[response].mean()
                    k_groups = len(group_means)
                    n_per = N / k_groups  # average group size
                    ms_between = float(np.sum(groups.count() * (group_means - grand_mean)**2) / (k_groups - 1))
                    ms_within = float(np.sum(groups.apply(lambda x: np.sum((x - x.mean())**2))) / (N - k_groups))
                    var_component = max(0, (ms_between - ms_within) / n_per)
                    components[factor] = var_component
                components["Error"] = float(data.groupby(factors[0])[response].apply(lambda x: x.var()).mean()) if factors else total_var

            comp_total = sum(components.values())
            pct = {k: v / comp_total * 100 if comp_total > 0 else 0 for k, v in components.items()}

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>VARIANCE COMPONENTS ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
            summary_text += f"<<COLOR:highlight>>Method:<</COLOR>> {method.upper()}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
            summary_text += f"{'Source':<20} {'Variance':>12} {'% of Total':>12} {'Std Dev':>12}\n"
            summary_text += f"{'─' * 58}\n"
            for source, var_val in components.items():
                summary_text += f"{source:<20} {var_val:>12.4f} {pct[source]:>11.1f}% {np.sqrt(var_val):>12.4f}\n"
            summary_text += f"{'─' * 58}\n"
            summary_text += f"{'Total':<20} {comp_total:>12.4f} {'100.0%':>12} {np.sqrt(comp_total):>12.4f}\n"

            result["summary"] = summary_text

            # Pie chart of variance components
            labels = list(components.keys())
            values = [components[k] for k in labels]
            colors = ["#4a90d9", "#d9a04a", "#4a9f6e", "#d94a4a", "#9b59b6", "#3498db"]
            result["plots"].append({
                "title": "Variance Components",
                "data": [{
                    "type": "pie",
                    "labels": labels,
                    "values": values,
                    "marker": {"colors": colors[:len(labels)]},
                    "textinfo": "label+percent",
                    "hole": 0.3
                }],
                "layout": {"height": 300, "template": "plotly_white"}
            })

            # Bar chart
            result["plots"].append({
                "title": "Variance Components (Bar)",
                "data": [{
                    "type": "bar", "x": labels, "y": values,
                    "marker": {"color": colors[:len(labels)]},
                    "text": [f"{pct[k]:.1f}%" for k in labels],
                    "textposition": "outside"
                }],
                "layout": {"height": 280, "yaxis": {"title": "Variance"}, "template": "plotly_white"}
            })

            result["statistics"] = {
                "components": {k: {"variance": v, "pct": pct[k], "std_dev": float(np.sqrt(v))} for k, v in components.items()},
                "total_variance": comp_total,
                "method": method, "n": N
            }
            result["guide_observation"] = f"Variance components: " + ", ".join([f"{k}={pct[k]:.1f}%" for k in components]) + "."

        except Exception as e:
            result["summary"] = f"Variance components error: {str(e)}"

    elif analysis_id == "ordinal_logistic":
        """
        Ordinal Logistic Regression — proportional odds model for ordered categorical outcomes.
        Uses statsmodels OrderedModel.
        """
        response = config.get("response") or config.get("var")
        predictors = config.get("predictors", [])
        if not predictors and config.get("predictor"):
            predictors = [config["predictor"]]

        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            all_cols = [response] + predictors
            data = df[all_cols].dropna()
            N = len(data)

            # Encode response as ordered categorical
            categories = sorted(data[response].unique().tolist(), key=str)
            cat_map = {c: i for i, c in enumerate(categories)}
            data["_y_ordinal"] = data[response].map(cat_map)

            # Fit proportional odds model
            X = data[predictors].values.astype(float)
            y = data["_y_ordinal"].values

            model = OrderedModel(y, X, distr='logit')
            fit = model.fit(method='bfgs', disp=False)

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>ORDINAL LOGISTIC REGRESSION (Proportional Odds)<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response} ({len(categories)} ordered levels)\n"
            summary_text += f"<<COLOR:highlight>>Levels:<</COLOR>> {' < '.join(str(c) for c in categories)}\n"
            summary_text += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary_text += f"{'Parameter':<25} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'OR':>10}\n"
            summary_text += f"{'─' * 75}\n"

            param_names = list(predictors) + [f"threshold_{i}" for i in range(len(categories) - 1)]
            for i, name in enumerate(param_names):
                if i < len(fit.params):
                    coef = float(fit.params[i])
                    se = float(fit.bse[i]) if i < len(fit.bse) else None
                    pv = float(fit.pvalues[i]) if i < len(fit.pvalues) else None
                    z = coef / se if se and se > 0 else 0
                    odds_ratio = np.exp(coef) if i < len(predictors) else None
                    se_str = f"{se:.4f}" if se else "N/A"
                    p_str = f"{pv:.4f}" if pv else "N/A"
                    or_str = f"{odds_ratio:.4f}" if odds_ratio else ""
                    summary_text += f"{name:<25} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {or_str:>10}\n"

            if hasattr(fit, 'llf'):
                summary_text += f"\n<<COLOR:text>>Log-Likelihood:<</COLOR>> {fit.llf:.2f}\n"
            if hasattr(fit, 'aic'):
                summary_text += f"<<COLOR:text>>AIC:<</COLOR>> {fit.aic:.2f}\n"

            result["summary"] = summary_text

            # Predicted probabilities for first predictor
            if len(predictors) >= 1:
                x_range = np.linspace(float(data[predictors[0]].min()), float(data[predictors[0]].max()), 100)
                X_pred = np.zeros((100, len(predictors)))
                X_pred[:, 0] = x_range
                for j in range(1, len(predictors)):
                    X_pred[:, j] = data[predictors[j]].mean()

                pred_probs = fit.predict(X_pred)
                traces = []
                colors_cat = ["#4a90d9", "#d9a04a", "#4a9f6e", "#d94a4a", "#9b59b6"]
                for ci, cat in enumerate(categories):
                    col_idx = ci if ci < pred_probs.shape[1] else pred_probs.shape[1] - 1
                    traces.append({
                        "x": x_range.tolist(),
                        "y": pred_probs[:, col_idx].tolist(),
                        "mode": "lines", "name": str(cat),
                        "line": {"color": colors_cat[ci % len(colors_cat)], "width": 2}
                    })
                result["plots"].append({
                    "title": f"Predicted Probabilities by {predictors[0]}",
                    "data": traces,
                    "layout": {"height": 300, "xaxis": {"title": predictors[0]}, "yaxis": {"title": "Probability"}, "template": "plotly_white"}
                })

            result["statistics"] = {
                "n": N, "n_categories": len(categories),
                "log_likelihood": float(fit.llf) if hasattr(fit, 'llf') else None,
                "aic": float(fit.aic) if hasattr(fit, 'aic') else None,
            }
            result["guide_observation"] = f"Ordinal logistic: {response} ({len(categories)} levels) ~ {', '.join(predictors)}. N={N}."

        except ImportError:
            result["summary"] = "Ordinal logistic requires statsmodels >= 0.13. Install with: pip install --upgrade statsmodels"
        except Exception as e:
            result["summary"] = f"Ordinal logistic error: {str(e)}"

    # ── Data Profile ──────────────────────────────────────────────────────
    elif analysis_id == "data_profile":
        try:
            n_rows, n_cols = df.shape
            mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

            # Summary stats table
            desc = df.describe(include="all").T
            desc["missing"] = df.isnull().sum()
            desc["missing%"] = (df.isnull().sum() / n_rows * 100).round(1)
            desc["unique"] = df.nunique()
            desc["dtype"] = df.dtypes

            tbl_rows = []
            for col in df.columns:
                r = desc.loc[col] if col in desc.index else {}
                tbl_rows.append(f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td>"
                    f"<td>{r.get('mean', ''):.4g}</td>" if col in numeric_cols else
                    f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                    f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                    f"<td>{int(r.get('unique', 0))}</td><td>-</td>"
                    f"<td>{r.get('top', '-')}</td></tr>")

            table_html = "<table class='result-table'><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing%</th><th>Unique</th><th>Mean/Top</th></tr>"
            for col in df.columns:
                miss = int(df[col].isnull().sum())
                miss_pct = miss / n_rows * 100 if n_rows > 0 else 0
                uniq = df[col].nunique()
                if col in numeric_cols:
                    val = f"{df[col].mean():.4g}"
                else:
                    top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                    val = str(top)[:20]
                table_html += f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{miss}</td><td>{miss_pct:.1f}%</td><td>{uniq}</td><td>{val}</td></tr>"
            table_html += "</table>"

            # Top correlations
            corr_text = ""
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        pairs.append((numeric_cols[i], numeric_cols[j], abs(corr_matrix.iloc[i, j]), corr_matrix.iloc[i, j]))
                pairs.sort(key=lambda x: x[2], reverse=True)
                top_pairs = pairs[:10]
                corr_lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, _, v in top_pairs]
                corr_text = "\n".join(corr_lines)

            total_missing = int(df.isnull().sum().sum())
            total_cells = n_rows * n_cols
            complete_rows = int((~df.isnull().any(axis=1)).sum())

            summary = f"""DATA PROFILE
{'='*50}
Shape: {n_rows} rows x {n_cols} columns
Memory: {mem_mb:.2f} MB

Column Types:
  Numeric:     {len(numeric_cols)}
  Categorical: {len(cat_cols)}
  Datetime:    {len(dt_cols)}

Missing Data:
  Total missing cells: {total_missing} / {total_cells} ({total_missing/total_cells*100:.1f}%)
  Complete rows: {complete_rows} / {n_rows} ({complete_rows/n_rows*100:.1f}%)

Top Correlations:
{corr_text if corr_text else '  (need 2+ numeric columns)'}"""

            result["summary"] = summary
            result["tables"] = [{"title": "Column Summary", "html": table_html}]

            # Plot 1: Correlation heatmap
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                result["plots"].append({
                    "data": [{"type": "heatmap", "z": corr_matrix.values.tolist(),
                              "x": numeric_cols, "y": numeric_cols,
                              "colorscale": "RdBu_r", "zmin": -1, "zmax": 1,
                              "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                              "texttemplate": "%{text}", "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>"}],
                    "layout": {"title": "Correlation Matrix", "height": 400, "template": "plotly_white"}
                })

            # Plot 2: Missing percentage bar chart
            miss_cols = df.columns[df.isnull().any()].tolist()
            if miss_cols:
                miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
                result["plots"].append({
                    "data": [{"type": "bar", "x": [round(p, 1) for p in miss_pcts], "y": miss_cols,
                              "orientation": "h", "marker": {"color": "rgba(232,71,71,0.6)"}}],
                    "layout": {"title": "Missing Data by Column (%)", "height": max(250, len(miss_cols) * 25),
                               "xaxis": {"title": "% Missing"}, "template": "plotly_white", "margin": {"l": 120}}
                })

            # Plot 3: Distribution grid (top 6 numeric)
            for col in numeric_cols[:6]:
                vals = df[col].dropna().tolist()
                if len(vals) > 0:
                    result["plots"].append({
                        "data": [{"type": "histogram", "x": vals, "marker": {"color": "rgba(74,144,217,0.6)"}, "nbinsx": 30}],
                        "layout": {"title": f"Distribution: {col}", "height": 250, "template": "plotly_white",
                                   "xaxis": {"title": col}, "yaxis": {"title": "Count"}}
                    })

            result["guide_observation"] = f"Data profile: {n_rows} rows, {n_cols} cols, {total_missing} missing cells, {len(numeric_cols)} numeric columns."

        except Exception as e:
            result["summary"] = f"Data profile error: {str(e)}"

    # ── Missing Data Analysis ─────────────────────────────────────────────
    elif analysis_id == "missing_data_analysis":
        try:
            n_rows, n_cols = df.shape
            miss_count = df.isnull().sum()
            miss_pct = (miss_count / n_rows * 100).round(2)
            cols_with_missing = miss_count[miss_count > 0].sort_values(ascending=False)

            # Row completeness
            row_completeness = ((~df.isnull()).sum(axis=1) / n_cols * 100)
            complete_rows = int((row_completeness == 100).sum())

            # Missing patterns
            miss_indicator = df.isnull().astype(int)
            pattern_strs = miss_indicator.apply(lambda r: "".join(str(v) for v in r), axis=1)
            pattern_counts = pattern_strs.value_counts()
            n_patterns = len(pattern_counts)

            # Pattern table
            pattern_rows = []
            for pat, cnt in pattern_counts.head(15).items():
                cols_missing = [df.columns[i] for i, v in enumerate(pat) if v == '1']
                desc = ", ".join(cols_missing) if cols_missing else "(complete)"
                pattern_rows.append(f"  {cnt:>6} rows ({cnt/n_rows*100:.1f}%): {desc}")
            pattern_text = "\n".join(pattern_rows)

            # Little's MCAR test approximation
            mcar_text = ""
            if len(cols_with_missing) >= 2 and len(cols_with_missing) < n_cols:
                try:
                    observed_counts = pattern_counts.values
                    # Expected under MCAR: each column missing independently
                    col_miss_rates = miss_count / n_rows
                    expected_probs = []
                    for pat in pattern_counts.index:
                        prob = 1.0
                        for i, v in enumerate(pat):
                            p_miss = col_miss_rates.iloc[i]
                            prob *= p_miss if v == '1' else (1 - p_miss)
                        expected_probs.append(prob)
                    expected_counts = np.array(expected_probs) * n_rows
                    # Filter out zero-expected
                    mask = expected_counts > 0.5
                    if mask.sum() >= 2:
                        from scipy.stats import chi2
                        obs = observed_counts[mask]
                        exp = expected_counts[mask]
                        chi2_stat = float(np.sum((obs - exp)**2 / exp))
                        dof = int(mask.sum() - 1)
                        p_val = float(1 - chi2.cdf(chi2_stat, dof))
                        conclusion = "Data appears MCAR (p >= 0.05)" if p_val >= 0.05 else "Data may NOT be MCAR (p < 0.05)"
                        mcar_text = f"""
MCAR Test (Chi-squared approximation):
  Chi-squared: {chi2_stat:.2f}  (df={dof})
  p-value: {p_val:.4f}
  Conclusion: {conclusion}"""
                except Exception:
                    mcar_text = "\nMCAR Test: Could not compute (insufficient patterns)"

            # Missing correlation
            miss_corr_text = ""
            if len(cols_with_missing) >= 2:
                miss_corr = miss_indicator[cols_with_missing.index].corr()
                pairs = []
                idx_list = cols_with_missing.index.tolist()
                for i in range(len(idx_list)):
                    for j in range(i+1, len(idx_list)):
                        pairs.append((idx_list[i], idx_list[j], miss_corr.loc[idx_list[i], idx_list[j]]))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, v in pairs[:5]]
                    miss_corr_text = "\nMissing Correlation (top pairs):\n" + "\n".join(lines)

            summary_lines = [f"MISSING DATA ANALYSIS", "=" * 50]
            summary_lines.append(f"Dataset: {n_rows} rows x {n_cols} columns")
            summary_lines.append(f"Total missing: {int(miss_count.sum())} / {n_rows * n_cols} ({miss_count.sum() / (n_rows * n_cols) * 100:.1f}%)")
            summary_lines.append(f"Complete rows: {complete_rows} / {n_rows} ({complete_rows/n_rows*100:.1f}%)")
            summary_lines.append(f"\nColumns with missing data ({len(cols_with_missing)}):")
            for col, cnt in cols_with_missing.items():
                summary_lines.append(f"  {col:<30} {cnt:>6} ({miss_pct[col]:.1f}%)")
            summary_lines.append(f"\nMissing Patterns ({n_patterns} unique):")
            summary_lines.append(pattern_text)
            if mcar_text:
                summary_lines.append(mcar_text)
            if miss_corr_text:
                summary_lines.append(miss_corr_text)

            result["summary"] = "\n".join(summary_lines)

            # Plot 1: Missing pattern heatmap
            if len(cols_with_missing) > 0:
                top_patterns = pattern_counts.head(20)
                z_data = []
                y_labels = []
                for pat, cnt in top_patterns.items():
                    z_data.append([int(v) for v in pat])
                    cols_miss = sum(1 for v in pat if v == '1')
                    y_labels.append(f"{cnt} rows ({cols_miss} missing)")
                result["plots"].append({
                    "data": [{"type": "heatmap", "z": z_data, "x": df.columns.tolist(), "y": y_labels,
                              "colorscale": [[0, "rgba(74,144,217,0.15)"], [1, "rgba(232,71,71,0.7)"]],
                              "showscale": False, "hovertemplate": "%{x}: %{z}<extra>0=present, 1=missing</extra>"}],
                    "layout": {"title": "Missing Data Patterns", "height": max(300, len(z_data) * 25 + 100),
                               "template": "plotly_white", "xaxis": {"tickangle": -45}, "margin": {"b": 100, "l": 150}}
                })

            # Plot 2: Row completeness histogram
            result["plots"].append({
                "data": [{"type": "histogram", "x": row_completeness.tolist(), "nbinsx": 20,
                          "marker": {"color": "rgba(74,159,110,0.6)"}}],
                "layout": {"title": "Row Completeness Distribution", "height": 280, "template": "plotly_white",
                           "xaxis": {"title": "% Complete", "range": [0, 105]}, "yaxis": {"title": "Row Count"}}
            })

            result["guide_observation"] = f"Missing data: {int(miss_count.sum())} cells across {len(cols_with_missing)} columns. {n_patterns} unique patterns."

        except Exception as e:
            result["summary"] = f"Missing data analysis error: {str(e)}"

    # ── Outlier Analysis ──────────────────────────────────────────────────
    elif analysis_id == "outlier_analysis":
        try:
            columns = config.get("columns", [])
            methods = config.get("methods", ["iqr"])
            iqr_mult = float(config.get("iqr_multiplier", 1.5))
            z_thresh = float(config.get("zscore_threshold", 3.0))

            if not columns:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

            if not columns:
                result["summary"] = "No numeric columns found for outlier analysis."
                return result

            all_results = {}
            consensus = np.zeros(len(df), dtype=int)

            for col in columns:
                vals = df[col].dropna()
                col_results = {}

                if "iqr" in methods:
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                    mask = (df[col] < lower) | (df[col] > upper)
                    col_results["IQR"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                          "bounds": f"[{lower:.4g}, {upper:.4g}]"}
                    consensus += mask.astype(int).values

                if "zscore" in methods:
                    z = np.abs((df[col] - vals.mean()) / vals.std()) if vals.std() > 0 else pd.Series(0, index=df.index)
                    mask = z > z_thresh
                    col_results["Z-score"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                              "threshold": z_thresh}
                    consensus += mask.astype(int).values

                if "modified_zscore" in methods:
                    median = vals.median()
                    mad = np.median(np.abs(vals - median))
                    if mad > 0:
                        modified_z = 0.6745 * np.abs(df[col] - median) / mad
                        mask = modified_z > 3.5
                    else:
                        mask = pd.Series(False, index=df.index)
                    col_results["Modified Z"] = {"count": int(mask.sum()), "pct": round(mask.sum() / len(df) * 100, 1),
                                                 "threshold": 3.5}
                    consensus += mask.astype(int).values

                if "mahalanobis" in methods and len(columns) >= 2:
                    try:
                        from scipy.spatial.distance import mahalanobis as mah_dist
                        sub = df[columns].dropna()
                        if len(sub) > len(columns):
                            cov = np.cov(sub.T)
                            cov_inv = np.linalg.inv(cov + np.eye(len(columns)) * 1e-6)
                            mean = sub.mean().values
                            dists = sub.apply(lambda r: mah_dist(r.values, mean, cov_inv), axis=1)
                            from scipy.stats import chi2 as chi2_dist
                            threshold = chi2_dist.ppf(0.975, df=len(columns))
                            mask_full = dists > np.sqrt(threshold)
                            col_results["Mahalanobis"] = {"count": int(mask_full.sum()),
                                                          "pct": round(mask_full.sum() / len(sub) * 100, 1),
                                                          "threshold": f"chi2(p=0.975, df={len(columns)})"}
                    except Exception:
                        col_results["Mahalanobis"] = {"count": 0, "pct": 0, "error": "Could not compute"}

                all_results[col] = col_results

            # Build summary
            summary_lines = ["OUTLIER ANALYSIS", "=" * 50]
            summary_lines.append(f"Columns: {', '.join(columns)}")
            summary_lines.append(f"Methods: {', '.join(methods)}")
            summary_lines.append(f"Rows: {len(df)}\n")

            for col, methods_res in all_results.items():
                summary_lines.append(f"{col}:")
                for method, info in methods_res.items():
                    summary_lines.append(f"  {method:<18} {info['count']:>5} outliers ({info['pct']}%)")
                summary_lines.append("")

            # Consensus
            n_methods_used = len([m for m in methods if m != "mahalanobis" or len(columns) >= 2])
            if n_methods_used >= 2:
                flagged_all = int((consensus >= n_methods_used * len(columns)).sum()) if n_methods_used > 0 else 0
                flagged_majority = int((consensus >= max(1, n_methods_used * len(columns) // 2)).sum())
                summary_lines.append(f"Consensus:")
                summary_lines.append(f"  Flagged by majority of methods: {flagged_majority} rows")

            result["summary"] = "\n".join(summary_lines)

            # Table
            table_html = "<table class='result-table'><tr><th>Column</th><th>Method</th><th>Outliers</th><th>%</th><th>Details</th></tr>"
            for col, methods_res in all_results.items():
                for method, info in methods_res.items():
                    detail = info.get("bounds", info.get("threshold", ""))
                    table_html += f"<tr><td>{col}</td><td>{method}</td><td>{info['count']}</td><td>{info['pct']}%</td><td>{detail}</td></tr>"
            table_html += "</table>"
            result["tables"] = [{"title": "Outlier Summary", "html": table_html}]

            # Plot: Boxplots
            for col in columns[:6]:
                vals = df[col].dropna().tolist()
                result["plots"].append({
                    "data": [{"type": "box", "y": vals, "name": col, "boxpoints": "outliers",
                              "marker": {"color": "rgba(74,144,217,0.6)", "outliercolor": "rgba(232,71,71,0.8)"},
                              "line": {"color": "rgba(74,144,217,0.8)"}}],
                    "layout": {"title": f"Outlier Detection: {col}", "height": 300, "template": "plotly_white",
                               "yaxis": {"title": col}}
                })

            result["guide_observation"] = f"Outlier analysis on {len(columns)} columns with {len(methods)} methods."

        except Exception as e:
            result["summary"] = f"Outlier analysis error: {str(e)}"

    # ── Duplicate Analysis ────────────────────────────────────────────────
    elif analysis_id == "duplicate_analysis":
        try:
            mode = config.get("mode", "exact")
            subset_cols = config.get("subset_columns", [])

            if mode == "subset" and subset_cols:
                check_cols = [c for c in subset_cols if c in df.columns]
            else:
                check_cols = df.columns.tolist()

            duplicated_mask = df.duplicated(subset=check_cols, keep=False)
            n_dup_rows = int(duplicated_mask.sum())
            n_dup_groups = int(df[duplicated_mask].groupby(check_cols).ngroups) if n_dup_rows > 0 else 0
            first_dup_mask = df.duplicated(subset=check_cols, keep="first")
            n_extra = int(first_dup_mask.sum())

            summary_lines = ["DUPLICATE ANALYSIS", "=" * 50]
            summary_lines.append(f"Mode: {'Exact (all columns)' if mode == 'exact' else 'Subset (' + ', '.join(check_cols) + ')'}")
            summary_lines.append(f"Total rows: {len(df)}")
            summary_lines.append(f"Unique rows: {len(df) - n_extra}")
            summary_lines.append(f"Duplicate rows: {n_dup_rows} ({n_dup_rows/len(df)*100:.1f}%)")
            summary_lines.append(f"Duplicate groups: {n_dup_groups}")
            summary_lines.append(f"Extra copies (removable): {n_extra}")

            # Show top duplicate groups
            if n_dup_rows > 0:
                dup_df = df[duplicated_mask].copy()
                group_sizes = dup_df.groupby(check_cols).size().sort_values(ascending=False)
                summary_lines.append(f"\nLargest duplicate groups:")
                for i, (vals, cnt) in enumerate(group_sizes.head(10).items()):
                    if isinstance(vals, tuple):
                        desc = ", ".join(f"{c}={v}" for c, v in zip(check_cols, vals))
                    else:
                        desc = f"{check_cols[0]}={vals}"
                    summary_lines.append(f"  Group {i+1}: {cnt} copies — {desc[:80]}")

            result["summary"] = "\n".join(summary_lines)

            # Table of sample duplicates
            if n_dup_rows > 0:
                sample = df[duplicated_mask].head(20)
                table_html = "<table class='result-table'><tr>"
                for c in sample.columns:
                    table_html += f"<th>{c}</th>"
                table_html += "<th>Row#</th></tr>"
                for idx, row in sample.iterrows():
                    table_html += "<tr>"
                    for c in sample.columns:
                        table_html += f"<td>{row[c]}</td>"
                    table_html += f"<td>{idx}</td></tr>"
                table_html += "</table>"
                result["tables"] = [{"title": "Sample Duplicate Rows", "html": table_html}]

            # Plot: duplicate group size histogram
            if n_dup_groups > 0:
                group_sizes_list = dup_df.groupby(check_cols).size().tolist()
                result["plots"].append({
                    "data": [{"type": "histogram", "x": group_sizes_list,
                              "marker": {"color": "rgba(232,149,71,0.6)"}}],
                    "layout": {"title": "Duplicate Group Sizes", "height": 280, "template": "plotly_white",
                               "xaxis": {"title": "Copies per Group"}, "yaxis": {"title": "Number of Groups"}}
                })

            result["guide_observation"] = f"Duplicate analysis ({mode}): {n_dup_rows} duplicate rows in {n_dup_groups} groups."

        except Exception as e:
            result["summary"] = f"Duplicate analysis error: {str(e)}"

    # ── Meta-Analysis ─────────────────────────────────────────────────────
    elif analysis_id == "meta_analysis":
        try:
            mode = config.get("mode", "precomputed")
            subgroup_col = config.get("subgroup_col", "")

            if mode == "precomputed":
                effect_col = config.get("effect_col", "")
                se_col = config.get("se_col", "")
                study_col = config.get("study_col", "")
                if not all([effect_col, se_col, study_col]):
                    result["summary"] = "Please specify effect size, SE, and study label columns."
                    return result
                effects = df[effect_col].dropna().values.astype(float)
                ses = df[se_col].dropna().values.astype(float)
                studies = df[study_col].values[:len(effects)]
            else:
                # Raw mode: compute Cohen's d
                m1c, s1c, n1c = config.get("mean1_col", ""), config.get("sd1_col", ""), config.get("n1_col", "")
                m2c, s2c, n2c = config.get("mean2_col", ""), config.get("sd2_col", ""), config.get("n2_col", "")
                study_col = config.get("study_col", "")
                if not all([m1c, s1c, n1c, m2c, s2c, n2c]):
                    result["summary"] = "Please specify all 6 raw data columns (mean, SD, n for each group)."
                    return result
                m1 = df[m1c].values.astype(float)
                s1 = df[s1c].values.astype(float)
                n1 = df[n1c].values.astype(float)
                m2 = df[m2c].values.astype(float)
                s2 = df[s2c].values.astype(float)
                n2 = df[n2c].values.astype(float)
                sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                effects = (m1 - m2) / sp
                ses = np.sqrt(1/n1 + 1/n2 + effects**2 / (2 * (n1 + n2)))
                studies = df[study_col].values[:len(effects)] if study_col else np.arange(1, len(effects) + 1)

            k = len(effects)
            if k < 2:
                result["summary"] = "Need at least 2 studies for meta-analysis."
                return result

            # Weights
            w = 1.0 / (ses**2)

            # Fixed effects
            fe_est = float(np.sum(w * effects) / np.sum(w))
            fe_se = float(np.sqrt(1.0 / np.sum(w)))
            fe_ci_lo = fe_est - 1.96 * fe_se
            fe_ci_hi = fe_est + 1.96 * fe_se
            fe_z = fe_est / fe_se
            from scipy.stats import norm as norm_dist
            fe_p = float(2 * (1 - norm_dist.cdf(abs(fe_z))))

            # Heterogeneity
            Q = float(np.sum(w * (effects - fe_est)**2))
            df_q = k - 1
            from scipy.stats import chi2 as chi2_dist_fn
            Q_p = float(1 - chi2_dist_fn.cdf(Q, df_q))
            I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0
            H2 = Q / df_q if df_q > 0 else 1

            # DerSimonian-Laird tau²
            C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
            tau2 = max(0, (Q - df_q) / C) if C > 0 else 0

            # Random effects
            w_re = 1.0 / (ses**2 + tau2)
            re_est = float(np.sum(w_re * effects) / np.sum(w_re))
            re_se = float(np.sqrt(1.0 / np.sum(w_re)))
            re_ci_lo = re_est - 1.96 * re_se
            re_ci_hi = re_est + 1.96 * re_se
            re_z = re_est / re_se
            re_p = float(2 * (1 - norm_dist.cdf(abs(re_z))))

            interpretation = "Low heterogeneity" if I2 < 25 else ("Moderate heterogeneity" if I2 < 75 else "High heterogeneity")
            if I2 < 40:
                interpretation += " — fixed and random effects models give similar results."
            else:
                interpretation += " — random effects model is more appropriate."

            summary = f"""META-ANALYSIS
{'='*50}
Studies: {k}

Fixed Effects Model:
  Pooled estimate: {fe_est:.4f} (95% CI: {fe_ci_lo:.4f}, {fe_ci_hi:.4f})
  z = {fe_z:.3f}, p = {fe_p:.4f}

Random Effects Model (DerSimonian-Laird):
  Pooled estimate: {re_est:.4f} (95% CI: {re_ci_lo:.4f}, {re_ci_hi:.4f})
  z = {re_z:.3f}, p = {re_p:.4f}

Heterogeneity:
  Q = {Q:.2f} (df={df_q}, p={Q_p:.4f})
  I² = {I2:.1f}% (variation due to heterogeneity)
  tau² = {tau2:.4f} (between-study variance)
  H² = {H2:.2f}

Interpretation:
  {interpretation}"""

            result["summary"] = summary

            # Table
            table_html = "<table class='result-table'><tr><th>Study</th><th>Effect</th><th>SE</th><th>95% CI</th><th>Weight (FE)</th><th>Weight (RE)</th></tr>"
            for i in range(k):
                ci_lo = effects[i] - 1.96 * ses[i]
                ci_hi = effects[i] + 1.96 * ses[i]
                w_fe_pct = w[i] / np.sum(w) * 100
                w_re_pct = w_re[i] / np.sum(w_re) * 100
                table_html += f"<tr><td>{studies[i]}</td><td>{effects[i]:.4f}</td><td>{ses[i]:.4f}</td>"
                table_html += f"<td>[{ci_lo:.4f}, {ci_hi:.4f}]</td><td>{w_fe_pct:.1f}%</td><td>{w_re_pct:.1f}%</td></tr>"
            table_html += f"<tr style='font-weight:bold;border-top:2px solid #666;'><td>Fixed Effects</td><td>{fe_est:.4f}</td><td>{fe_se:.4f}</td><td>[{fe_ci_lo:.4f}, {fe_ci_hi:.4f}]</td><td>100%</td><td>-</td></tr>"
            table_html += f"<tr style='font-weight:bold;'><td>Random Effects</td><td>{re_est:.4f}</td><td>{re_se:.4f}</td><td>[{re_ci_lo:.4f}, {re_ci_hi:.4f}]</td><td>-</td><td>100%</td></tr>"
            table_html += "</table>"
            result["tables"] = [{"title": "Study Results", "html": table_html}]

            # Forest plot
            forest_y = list(range(k, 0, -1))
            ci_lows = [effects[i] - 1.96 * ses[i] for i in range(k)]
            ci_highs = [effects[i] + 1.96 * ses[i] for i in range(k)]
            study_labels = [str(s) for s in studies]

            forest_data = [
                # Study CIs (horizontal lines)
                {"type": "scatter", "x": list(effects), "y": forest_y,
                 "mode": "markers", "name": "Studies",
                 "marker": {"size": [max(4, min(16, float(w_re[i]/np.max(w_re)*16))) for i in range(k)], "color": "rgba(74,144,217,0.8)"},
                 "error_x": {"type": "data", "symmetric": False,
                             "array": [ci_highs[i] - effects[i] for i in range(k)],
                             "arrayminus": [effects[i] - ci_lows[i] for i in range(k)],
                             "color": "rgba(74,144,217,0.5)", "thickness": 2},
                 "text": study_labels, "hovertemplate": "%{text}: %{x:.4f}<extra></extra>"},
                # Pooled diamond (RE)
                {"type": "scatter", "x": [re_ci_lo, re_est, re_ci_hi, re_est, re_ci_lo],
                 "y": [0, -0.3, 0, 0.3, 0], "mode": "lines", "fill": "toself",
                 "fillcolor": "rgba(232,71,71,0.3)", "line": {"color": "rgba(232,71,71,0.8)"},
                 "name": "Pooled (RE)", "hoverinfo": "skip"},
                # Zero line
                {"type": "scatter", "x": [0, 0], "y": [-1, k + 1], "mode": "lines",
                 "line": {"color": "gray", "dash": "dash", "width": 1}, "showlegend": False, "hoverinfo": "skip"}
            ]
            result["plots"].append({
                "data": forest_data,
                "layout": {"title": "Forest Plot", "height": max(350, k * 30 + 120), "template": "plotly_white",
                           "yaxis": {"tickvals": forest_y + [0], "ticktext": study_labels + ["Pooled (RE)"],
                                     "range": [-1, k + 1]},
                           "xaxis": {"title": "Effect Size", "zeroline": True},
                           "showlegend": False, "margin": {"l": 120}}
            })

            # Funnel plot
            result["plots"].append({
                "data": [
                    {"type": "scatter", "x": list(effects), "y": list(ses), "mode": "markers",
                     "marker": {"size": 8, "color": "rgba(74,144,217,0.7)"}, "name": "Studies",
                     "text": study_labels, "hovertemplate": "%{text}: effect=%{x:.4f}, SE=%{y:.4f}<extra></extra>"},
                    # Funnel lines
                    {"type": "scatter", "x": [re_est - 1.96 * max(ses), re_est, re_est + 1.96 * max(ses)],
                     "y": [max(ses), 0, max(ses)], "mode": "lines",
                     "line": {"color": "gray", "dash": "dash"}, "name": "95% CI", "hoverinfo": "skip"},
                    {"type": "scatter", "x": [re_est, re_est], "y": [0, max(ses) * 1.1], "mode": "lines",
                     "line": {"color": "rgba(232,71,71,0.5)", "dash": "dot"}, "name": "Pooled", "hoverinfo": "skip"}
                ],
                "layout": {"title": "Funnel Plot", "height": 350, "template": "plotly_white",
                           "xaxis": {"title": "Effect Size"}, "yaxis": {"title": "Standard Error", "autorange": "reversed"}}
            })

            result["guide_observation"] = f"Meta-analysis of {k} studies. RE pooled: {re_est:.4f}. I²={I2:.1f}%. {interpretation}"

        except Exception as e:
            result["summary"] = f"Meta-analysis error: {str(e)}"

    # ── Effect Size Calculator ────────────────────────────────────────────
    elif analysis_id == "effect_size_calculator":
        try:
            effect_type = config.get("effect_type", "cohens_d")
            results_list = []

            if effect_type in ("cohens_d", "hedges_g", "glass_delta"):
                m1 = float(config.get("mean1", 0))
                s1 = float(config.get("sd1", 1))
                n1 = int(config.get("n1", 10))
                m2 = float(config.get("mean2", 0))
                s2 = float(config.get("sd2", 1))
                n2 = int(config.get("n2", 10))

                sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

                if effect_type == "cohens_d":
                    d = (m1 - m2) / sp if sp > 0 else 0
                    se = np.sqrt(1/n1 + 1/n2 + d**2 / (2 * (n1 + n2)))
                    name = "Cohen's d"
                elif effect_type == "hedges_g":
                    d_raw = (m1 - m2) / sp if sp > 0 else 0
                    J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                    d = d_raw * J
                    se = np.sqrt(1/n1 + 1/n2 + d**2 / (2 * (n1 + n2))) * J
                    name = "Hedges' g"
                else:  # glass_delta
                    d = (m1 - m2) / s2 if s2 > 0 else 0
                    se = np.sqrt(1/n1 + d**2 / (2 * n2))
                    name = "Glass's delta"

                ci_lo = d - 1.96 * se
                ci_hi = d + 1.96 * se

                if abs(d) < 0.2:
                    magnitude = "Negligible"
                elif abs(d) < 0.5:
                    magnitude = "Small"
                elif abs(d) < 0.8:
                    magnitude = "Medium"
                else:
                    magnitude = "Large"

                summary = f"""EFFECT SIZE CALCULATOR
{'='*50}
Type: {name}

Input:
  Group 1: M={m1}, SD={s1}, n={n1}
  Group 2: M={m2}, SD={s2}, n={n2}

Result:
  {name} = {d:.4f}
  Standard Error = {se:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})

Interpretation:
  Magnitude: {magnitude} (Cohen's benchmarks: 0.2=small, 0.5=medium, 0.8=large)
  Direction: {'Group 1 > Group 2' if d > 0 else 'Group 2 > Group 1' if d < 0 else 'No difference'}"""

                # Plot
                result["plots"].append({
                    "data": [
                        {"type": "scatter", "x": [d], "y": [name], "mode": "markers",
                         "marker": {"size": 14, "color": "rgba(74,144,217,0.8)"},
                         "error_x": {"type": "data", "symmetric": False,
                                     "array": [ci_hi - d], "arrayminus": [d - ci_lo],
                                     "color": "rgba(74,144,217,0.5)", "thickness": 3}},
                        {"type": "scatter", "x": [0, 0], "y": [-0.5, 1.5], "mode": "lines",
                         "line": {"color": "gray", "dash": "dash"}, "showlegend": False}
                    ],
                    "layout": {"title": f"{name} with 95% CI", "height": 200, "template": "plotly_white",
                               "xaxis": {"title": "Effect Size"}, "showlegend": False}
                })

            elif effect_type in ("odds_ratio", "risk_ratio"):
                a = int(config.get("a", 0))
                b = int(config.get("b", 0))
                c = int(config.get("c", 0))
                dd = int(config.get("d", 0))

                if effect_type == "odds_ratio":
                    if b * c > 0:
                        es = (a * dd) / (b * c)
                        se_ln = np.sqrt(1/max(a,0.5) + 1/max(b,0.5) + 1/max(c,0.5) + 1/max(dd,0.5))
                    else:
                        es, se_ln = 0, 0
                    name = "Odds Ratio"
                else:
                    r1 = a / (a + b) if (a + b) > 0 else 0
                    r2 = c / (c + dd) if (c + dd) > 0 else 0
                    es = r1 / r2 if r2 > 0 else 0
                    se_ln = np.sqrt(1/max(a,0.5) - 1/max(a+b,1) + 1/max(c,0.5) - 1/max(c+dd,1))
                    name = "Risk Ratio"

                ci_lo = es * np.exp(-1.96 * se_ln) if es > 0 else 0
                ci_hi = es * np.exp(1.96 * se_ln) if es > 0 else 0

                summary = f"""EFFECT SIZE CALCULATOR
{'='*50}
Type: {name}

Input (2x2 table):
               Outcome+  Outcome-
  Exposed      {a:<9} {b}
  Not Exposed  {c:<9} {dd}

Result:
  {name} = {es:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})
  ln({name}) SE = {se_ln:.4f}

Interpretation:
  {'No association (= 1.0)' if abs(es - 1.0) < 0.01 else ('Positive association' if es > 1 else 'Negative association')}"""

                result["plots"].append({
                    "data": [
                        {"type": "scatter", "x": [es], "y": [name], "mode": "markers",
                         "marker": {"size": 14, "color": "rgba(232,149,71,0.8)"},
                         "error_x": {"type": "data", "symmetric": False,
                                     "array": [ci_hi - es], "arrayminus": [es - ci_lo],
                                     "color": "rgba(232,149,71,0.5)", "thickness": 3}},
                        {"type": "scatter", "x": [1, 1], "y": [-0.5, 1.5], "mode": "lines",
                         "line": {"color": "gray", "dash": "dash"}, "showlegend": False}
                    ],
                    "layout": {"title": f"{name} with 95% CI", "height": 200, "template": "plotly_white",
                               "xaxis": {"title": name}, "showlegend": False}
                })
            else:
                summary = f"Unknown effect size type: {effect_type}"

            result["summary"] = summary
            result["guide_observation"] = f"Effect size calculated: {effect_type}"

        except Exception as e:
            result["summary"] = f"Effect size error: {str(e)}"

    return result


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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

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

        result["summary"] = f"Classification Results\n\nAlgorithm: {algorithm}\nAccuracy: {accuracy:.4f}\n\n{report}"

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            result["plots"].append({
                "title": "Feature Importance",
                "data": [{
                    "type": "bar",
                    "x": importances.tolist(),
                    "y": features,
                    "orientation": "h",
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })

        result["guide_observation"] = f"Classification model achieved {accuracy:.1%} accuracy."

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

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, model, {
                'model_type': f"Classification ({algorithm.upper()})",
                'features': features,
                'target': target,
                'metrics': {'accuracy': float(accuracy)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "regression_ml":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        result["summary"] = f"Regression Results\n\nR²: {r2:.4f}\nRMSE: {rmse:.4f}"

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

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, model, {
                'model_type': "Random Forest Regressor",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2), 'rmse': float(rmse)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "clustering":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = int(config.get("n_clusters", 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        result["summary"] = f"K-Means Clustering\n\nClusters: {n_clusters}\nInertia: {kmeans.inertia_:.2f}"

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

        result["guide_observation"] = f"K-Means with {n_clusters} clusters. Inertia: {kmeans.inertia_:.2f}."

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

    return result


def run_bayesian_analysis(df, analysis_id, config):
    """Run Bayesian inference analyses - feeds Synara hypothesis testing."""
    import numpy as np
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    ci_level = float(config.get("ci", 0.95))
    z = stats.norm.ppf((1 + ci_level) / 2)

    if analysis_id == "bayes_regression":
        # Bayesian Linear Regression with credible intervals
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        target = config.get("target")
        features = config.get("features", [])

        if not target or not features:
            result["summary"] = "Error: Select target and at least one feature"
            return result

        y = df[target].dropna()
        X = df[features].loc[y.index].dropna()
        y = y.loc[X.index]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        y_pred, y_std = model.predict(X_scaled, return_std=True)
        coef_mean = model.coef_
        coef_std = np.sqrt(1.0 / model.lambda_) * np.ones_like(coef_mean)
        r2 = model.score(X_scaled, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n\n"
        summary += f"<<COLOR:text>>Coefficient Posteriors ({int(ci_level*100)}% Credible Intervals):<</COLOR>>\n\n"

        for i, feat in enumerate(features):
            mean = coef_mean[i]
            std = coef_std[i]
            ci_low = mean - z * std
            ci_high = mean + z * std
            sig = "***" if ci_low > 0 or ci_high < 0 else ""
            summary += f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"

        result["summary"] = summary
        result["plots"].append({
            "title": "Coefficient Posteriors",
            "data": [{
                "type": "scatter",
                "x": coef_mean.tolist(),
                "y": features,
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 10},
                "error_x": {"type": "data", "array": (z * coef_std).tolist(), "color": "#4a9f6e"},
                "name": f"β ± {int(ci_level*100)}% CI"
            }],
            "layout": {"height": max(300, len(features) * 30), "xaxis": {"zeroline": True}, "margin": {"l": 150}}
        })

        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {"feature": feat, "mean": float(coef_mean[i]), "ci_low": float(coef_mean[i] - z * coef_std[i]), "ci_high": float(coef_mean[i] + z * coef_std[i])}
                for i, feat in enumerate(features)
            ]
        }

    elif analysis_id == "bayes_ttest":
        # Bayesian t-test comparing two groups
        var1 = config.get("var1")
        var2 = config.get("var2")
        prior_scale = config.get("prior_scale", "medium")

        scale_map = {"small": 0.2, "medium": 0.5, "large": 0.8, "ultrawide": 1.0}
        scale = scale_map.get(prior_scale, 0.5)

        x1 = df[var1].dropna().values
        x2 = df[var2].dropna().values

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(x1)-1)*np.var(x1, ddof=1) + (len(x2)-1)*np.var(x2, ddof=1)) / (len(x1)+len(x2)-2))
        cohens_d = (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

        # Bayes Factor approximation (JZS prior)
        t_stat, p_value = stats.ttest_ind(x1, x2)
        n_eff = 2 / (1/len(x1) + 1/len(x2))
        bf10 = np.exp(0.5 * (np.log(n_eff) - np.log(2*np.pi) - (t_stat**2)/n_eff)) if abs(t_stat) < 10 else 1e6

        # Posterior on effect size (approximate)
        se_d = np.sqrt((len(x1)+len(x2))/(len(x1)*len(x2)) + cohens_d**2/(2*(len(x1)+len(x2))))
        d_ci_low = cohens_d - z * se_d
        d_ci_high = cohens_d + z * se_d

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>{var1}<</COLOR>> (n={len(x1)}, μ={np.mean(x1):.3f})\n"
        summary += f"<<COLOR:highlight>>{var2}<</COLOR>> (n={len(x2)}, μ={np.mean(x2):.3f})\n\n"
        summary += f"<<COLOR:text>>Effect Size (Cohen's d):<</COLOR>> {cohens_d:.3f} [{d_ci_low:.3f}, {d_ci_high:.3f}]\n"
        summary += f"<<COLOR:text>>Bayes Factor (BF10):<</COLOR>> {bf10:.2f}\n\n"

        if bf10 > 10:
            summary += f"<<COLOR:success>>Strong evidence for difference<</COLOR>>\n"
        elif bf10 > 3:
            summary += f"<<COLOR:warning>>Moderate evidence for difference<</COLOR>>\n"
        elif bf10 > 1:
            summary += f"<<COLOR:text>>Weak evidence for difference<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Evidence favors no difference<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"cohens_d": cohens_d, "bf10": bf10, "d_ci_low": d_ci_low, "d_ci_high": d_ci_high}

        # Posterior distribution plot
        d_range = np.linspace(cohens_d - 3*se_d, cohens_d + 3*se_d, 100)
        posterior = stats.norm.pdf(d_range, cohens_d, se_d)

        result["plots"].append({
            "title": "Posterior Distribution of Effect Size",
            "data": [{
                "type": "scatter",
                "x": d_range.tolist(),
                "y": posterior.tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": "Posterior"
            }, {
                "type": "scatter",
                "x": [0, 0],
                "y": [0, max(posterior)],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash"},
                "name": "No effect"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Cohen's d"}, "yaxis": {"title": "Density"}}
        })

    elif analysis_id == "bayes_ab":
        # Bayesian A/B test for proportions
        group_col = config.get("group")
        success_col = config.get("success")
        prior_type = config.get("prior", "uniform")

        prior_map = {"uniform": (1, 1), "jeffreys": (0.5, 0.5), "informed": (5, 5)}
        a_prior, b_prior = prior_map.get(prior_type, (1, 1))

        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            result["summary"] = "Error: Need at least 2 groups"
            return result

        g1, g2 = groups[0], groups[1]
        s1 = df[df[group_col] == g1][success_col].sum()
        n1 = len(df[df[group_col] == g1])
        s2 = df[df[group_col] == g2][success_col].sum()
        n2 = len(df[df[group_col] == g2])

        # Posterior Beta distributions
        a1, b1 = a_prior + s1, b_prior + n1 - s1
        a2, b2 = a_prior + s2, b_prior + n2 - s2

        # Monte Carlo estimation of P(p1 > p2)
        samples1 = np.random.beta(a1, b1, 10000)
        samples2 = np.random.beta(a2, b2, 10000)
        prob_better = np.mean(samples1 > samples2)

        rate1, rate2 = s1/n1, s2/n2
        lift = (rate1 - rate2) / rate2 if rate2 > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN A/B TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Group A ({g1}):<</COLOR>> {s1}/{n1} = {rate1:.1%}\n"
        summary += f"<<COLOR:highlight>>Group B ({g2}):<</COLOR>> {s2}/{n2} = {rate2:.1%}\n\n"
        summary += f"<<COLOR:text>>P({g1} > {g2}):<</COLOR>> {prob_better:.1%}\n"
        summary += f"<<COLOR:text>>Relative Lift:<</COLOR>> {lift:+.1%}\n\n"

        if prob_better > 0.95:
            summary += f"<<COLOR:success>>Strong evidence {g1} is better<</COLOR>>\n"
        elif prob_better > 0.75:
            summary += f"<<COLOR:warning>>Moderate evidence {g1} is better<</COLOR>>\n"
        elif prob_better < 0.05:
            summary += f"<<COLOR:success>>Strong evidence {g2} is better<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Inconclusive - need more data<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"prob_better": prob_better, "rate_a": rate1, "rate_b": rate2, "lift": lift}

        # Posterior distributions
        x = np.linspace(0, 1, 200)
        result["plots"].append({
            "title": "Posterior Distributions",
            "data": [{
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a1, b1).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": f"{g1}"
            }, {
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a2, b2).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(232, 149, 71, 0.3)",
                "line": {"color": "#e89547"},
                "name": f"{g2}"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Conversion Rate"}, "yaxis": {"title": "Density"}}
        })

    elif analysis_id == "bayes_correlation":
        # Bayesian correlation
        var1 = config.get("var1")
        var2 = config.get("var2")

        x = df[var1].dropna()
        y = df[var2].loc[x.index].dropna()
        x = x.loc[y.index]

        r, p = stats.pearsonr(x, y)
        n = len(x)

        # Fisher z-transformation for CI
        z_r = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
        se_z = 1 / np.sqrt(n - 3) if n > 3 else 1
        z_low = z_r - z * se_z
        z_high = z_r + z * se_z
        r_low = (np.exp(2*z_low) - 1) / (np.exp(2*z_low) + 1)
        r_high = (np.exp(2*z_high) - 1) / (np.exp(2*z_high) + 1)

        # BF approximation
        bf10 = np.sqrt((n-1)/2) * np.exp(stats.t.logpdf(r * np.sqrt(n-2) / np.sqrt(1-r**2), n-2)) if abs(r) < 0.999 else 100

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CORRELATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>{var1}<</COLOR>> vs <<COLOR:highlight>>{var2}<</COLOR>> (n={n})\n\n"
        summary += f"<<COLOR:text>>Correlation (r):<</COLOR>> {r:.3f} [{r_low:.3f}, {r_high:.3f}]\n"
        summary += f"<<COLOR:text>>Bayes Factor:<</COLOR>> {bf10:.2f}\n\n"

        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        summary += f"<<COLOR:text>>Interpretation: {strength} {direction} correlation<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"r": r, "r_ci_low": r_low, "r_ci_high": r_high, "bf10": bf10}

        result["plots"].append({
            "title": f"Scatter: {var1} vs {var2}",
            "data": [{
                "type": "scatter",
                "x": x.values.tolist(),
                "y": y.values.tolist(),
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 6, "opacity": 0.6}
            }],
            "layout": {"height": 300, "xaxis": {"title": var1}, "yaxis": {"title": var2}}
        })

    elif analysis_id == "bayes_anova":
        # Bayesian ANOVA
        response = config.get("response")
        factor = config.get("factor")

        groups = df.groupby(factor)[response].apply(list).to_dict()
        group_names = list(groups.keys())
        group_data = [np.array(groups[g]) for g in group_names]

        # F-test for BF approximation
        f_stat, p_value = stats.f_oneway(*group_data)

        # Effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - df[response].mean())**2 for g in group_data)
        ss_total = sum((df[response] - df[response].mean())**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN ANOVA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(group_names)} levels)\n\n"

        for name in group_names:
            g = groups[name]
            summary += f"  {name}: n={len(g)}, μ={np.mean(g):.3f}, σ={np.std(g):.3f}\n"

        summary += f"\n<<COLOR:text>>F-statistic:<</COLOR>> {f_stat:.3f}\n"
        summary += f"<<COLOR:text>>Effect size (η²):<</COLOR>> {eta_sq:.3f}\n"

        result["summary"] = summary
        result["statistics"] = {"f_stat": f_stat, "eta_squared": eta_sq, "p_value": p_value}

        # Box plot
        result["plots"].append({
            "title": f"{response} by {factor}",
            "data": [{
                "type": "box",
                "y": groups[name],
                "name": str(name),
                "marker": {"color": "#4a9f6e"}
            } for name in group_names],
            "layout": {"height": 350}
        })

    elif analysis_id == "bayes_changepoint":
        # Bayesian change point detection
        var = config.get("var")
        time_col = config.get("time")
        max_cp = int(config.get("max_cp", 2))

        data = df[var].dropna().values
        n = len(data)

        if time_col:
            time_idx = df[time_col].loc[df[var].dropna().index].values
        else:
            time_idx = np.arange(n)

        # Simple Bayesian change point (PELT-like with BIC)
        from scipy.signal import find_peaks

        # Cumulative sum for change detection
        cumsum = np.cumsum(data - np.mean(data))
        diff2 = np.abs(np.diff(cumsum, 2)) if n > 2 else []

        # Find peaks in second derivative (change points)
        if len(diff2) > 0:
            peaks, props = find_peaks(diff2, height=np.std(diff2), distance=max(5, n//10))
            changepoints = peaks[:max_cp] + 1 if len(peaks) > 0 else []
        else:
            changepoints = []

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        if len(changepoints) > 0:
            summary += f"<<COLOR:success>>Detected {len(changepoints)} change point(s):<</COLOR>>\n"
            for i, cp in enumerate(changepoints):
                before = data[:cp]
                after = data[cp:]
                summary += f"  Point {i+1}: index {cp}, before μ={np.mean(before):.3f}, after μ={np.mean(after):.3f}\n"
        else:
            summary += f"<<COLOR:text>>No significant change points detected<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"n_changepoints": len(changepoints), "changepoint_indices": list(changepoints)}

        # Time series plot with change points
        plot_data = [{
            "type": "scatter",
            "x": time_idx.tolist() if hasattr(time_idx, 'tolist') else list(time_idx),
            "y": data.tolist(),
            "mode": "lines+markers",
            "marker": {"size": 4, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e"},
            "name": var
        }]

        for cp in changepoints:
            plot_data.append({
                "type": "scatter",
                "x": [time_idx[cp], time_idx[cp]],
                "y": [min(data), max(data)],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
                "name": f"Change @ {cp}"
            })

        result["plots"].append({
            "title": "Time Series with Change Points",
            "data": plot_data,
            "layout": {"height": 350, "xaxis": {"title": time_col or "Index"}, "yaxis": {"title": var}}
        })

    elif analysis_id == "bayes_proportion":
        # Bayesian proportion estimation
        success_col = config.get("success")
        prior_type = config.get("prior", "uniform")

        prior_map = {"uniform": (1, 1), "jeffreys": (0.5, 0.5), "optimistic": (8, 2), "pessimistic": (2, 8)}
        a_prior, b_prior = prior_map.get(prior_type, (1, 1))

        data = df[success_col].dropna()
        successes = int(data.sum())
        n = len(data)

        # Posterior
        a_post = a_prior + successes
        b_post = b_prior + n - successes

        # Posterior mean and CI
        post_mean = a_post / (a_post + b_post)
        ci_low, ci_high = stats.beta.ppf([(1-ci_level)/2, (1+ci_level)/2], a_post, b_post)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN PROPORTION ESTIMATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observed:<</COLOR>> {successes}/{n} = {successes/n:.1%}\n"
        summary += f"<<COLOR:highlight>>Prior:<</COLOR>> Beta({a_prior}, {b_prior})\n\n"
        summary += f"<<COLOR:text>>Posterior Mean:<</COLOR>> {post_mean:.1%}\n"
        summary += f"<<COLOR:text>>{int(ci_level*100)}% Credible Interval:<</COLOR>> [{ci_low:.1%}, {ci_high:.1%}]\n"

        result["summary"] = summary
        result["statistics"] = {"proportion": post_mean, "ci_low": ci_low, "ci_high": ci_high, "n": n, "successes": successes}

        # Posterior distribution
        x = np.linspace(0, 1, 200)
        result["plots"].append({
            "title": "Posterior Distribution",
            "data": [{
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a_post, b_post).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": f"Beta({a_post:.0f}, {b_post:.0f})"
            }, {
                "type": "scatter",
                "x": [ci_low, ci_high],
                "y": [0, 0],
                "mode": "lines",
                "line": {"color": "#e89547", "width": 4},
                "name": f"{int(ci_level*100)}% CI"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Proportion"}, "yaxis": {"title": "Density"}}
        })

    return result


def run_reliability_analysis(df, analysis_id, config):
    """Run reliability/survival analysis."""
    import numpy as np
    from scipy import stats as sp_stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "weibull":
        # Weibull Distribution Analysis
        time_col = config.get("time")
        censor_col = config.get("censor")  # optional: 1=failed, 0=censored

        times = df[time_col].dropna().values
        times = times[times > 0]

        if censor_col and censor_col in df.columns:
            censor = df[censor_col].dropna().values[:len(times)]
            failed = times[censor == 1]
        else:
            failed = times

        # Fit Weibull: scipy uses (c, loc, scale) = (shape/beta, loc, scale/eta)
        shape, loc, scale = sp_stats.weibull_min.fit(failed, floc=0)

        # Probability plot data (Weibull)
        sorted_t = np.sort(failed)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)  # Bernard's approximation

        # Theoretical line
        t_range = np.linspace(sorted_t[0] * 0.8, sorted_t[-1] * 1.2, 200)
        cdf_fit = sp_stats.weibull_min.cdf(t_range, shape, 0, scale)

        # Probability plot (linearized)
        result["plots"].append({
            "title": "Weibull Probability Plot",
            "data": [
                {"type": "scatter", "x": np.log(sorted_t).tolist(), "y": np.log(-np.log(1 - median_ranks)).tolist(),
                 "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 7}},
                {"type": "scatter", "x": np.log(t_range[cdf_fit > 0]).tolist(),
                 "y": np.log(-np.log(1 - cdf_fit[cdf_fit > 0])).tolist() if np.any(cdf_fit > 0) else [],
                 "mode": "lines", "name": f"Weibull (β={shape:.2f}, η={scale:.1f})",
                 "line": {"color": "#d94a4a", "width": 2}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "ln(-ln(1-F))"}, "showlegend": True}
        })

        # Reliability curve
        result["plots"].append({
            "title": "Reliability Function",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": (1 - cdf_fit).tolist(),
                 "mode": "lines", "name": "R(t)", "line": {"color": "#4a9f6e", "width": 2}, "fill": "tozeroy",
                 "fillcolor": "rgba(74, 159, 110, 0.15)"},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": "Time"}, "yaxis": {"title": "Reliability R(t)", "range": [0, 1]}}
        })

        # B-life calculations
        b10 = sp_stats.weibull_min.ppf(0.10, shape, 0, scale)
        b50 = sp_stats.weibull_min.ppf(0.50, shape, 0, scale)
        import math
        mttf = scale * np.exp(math.lgamma(1 + 1/shape))

        summary = f"Weibull Analysis\n\nShape (β): {shape:.4f}\nScale (η): {scale:.2f}\n\nB10 Life: {b10:.2f}\nB50 Life (median): {b50:.2f}\nMTTF: {mttf:.2f}\n\n"
        if shape < 1:
            summary += "β < 1: Decreasing failure rate (infant mortality)"
        elif shape == 1:
            summary += "β ≈ 1: Constant failure rate (random failures)"
        else:
            summary += "β > 1: Increasing failure rate (wear-out)"

        result["summary"] = summary
        result["guide_observation"] = f"Weibull β={shape:.2f}. " + ("Infant mortality pattern." if shape < 1 else "Wear-out pattern." if shape > 1 else "Random failures.")

    elif analysis_id == "lognormal":
        # Lognormal Distribution Analysis
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]

        # Fit lognormal
        shape_ln, loc_ln, scale_ln = sp_stats.lognorm.fit(times, floc=0)
        mu = np.log(scale_ln)
        sigma = shape_ln

        sorted_t = np.sort(times)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)

        t_range = np.linspace(sorted_t[0] * 0.5, sorted_t[-1] * 1.5, 200)
        cdf_fit = sp_stats.lognorm.cdf(t_range, shape_ln, 0, scale_ln)

        # Probability plot (lognormal linearized)
        result["plots"].append({
            "title": "Lognormal Probability Plot",
            "data": [
                {"type": "scatter", "x": np.log(sorted_t).tolist(),
                 "y": sp_stats.norm.ppf(median_ranks).tolist(),
                 "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 7}},
                {"type": "scatter", "x": np.log(t_range[cdf_fit > 0]).tolist(),
                 "y": sp_stats.norm.ppf(np.clip(cdf_fit[cdf_fit > 0], 1e-10, 1-1e-10)).tolist(),
                 "mode": "lines", "name": f"Lognormal (μ={mu:.2f}, σ={sigma:.2f})",
                 "line": {"color": "#d94a4a", "width": 2}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "Std Normal Quantile"}, "showlegend": True}
        })

        # Reliability curve
        result["plots"].append({
            "title": "Reliability Function",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": (1 - cdf_fit).tolist(),
                 "mode": "lines", "name": "R(t)", "line": {"color": "#4a90d9", "width": 2}, "fill": "tozeroy",
                 "fillcolor": "rgba(74, 144, 217, 0.15)"},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": "Time"}, "yaxis": {"title": "Reliability R(t)", "range": [0, 1]}}
        })

        b10 = sp_stats.lognorm.ppf(0.10, shape_ln, 0, scale_ln)
        b50 = sp_stats.lognorm.ppf(0.50, shape_ln, 0, scale_ln)
        mean_life = np.exp(mu + sigma**2 / 2)

        result["summary"] = f"Lognormal Analysis\n\nμ (log mean): {mu:.4f}\nσ (log std): {sigma:.4f}\n\nB10 Life: {b10:.2f}\nB50 Life (median): {b50:.2f}\nMean Life: {mean_life:.2f}"

    elif analysis_id == "exponential":
        # Exponential Distribution Analysis
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]

        # MLE for exponential: rate = 1/mean
        mttf = np.mean(times)
        rate = 1.0 / mttf

        sorted_t = np.sort(times)
        n = len(sorted_t)
        median_ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)

        t_range = np.linspace(0, sorted_t[-1] * 1.5, 200)
        rel = np.exp(-rate * t_range)

        # Exponential probability plot (linearized: ln(1-F) vs t)
        result["plots"].append({
            "title": "Exponential Probability Plot",
            "data": [
                {"type": "scatter", "x": sorted_t.tolist(), "y": (-np.log(1 - median_ranks)).tolist(),
                 "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 7}},
                {"type": "scatter", "x": t_range.tolist(), "y": (rate * t_range).tolist(),
                 "mode": "lines", "name": f"Exp (λ={rate:.4f})",
                 "line": {"color": "#d94a4a", "width": 2}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": "-ln(1-F)"}, "showlegend": True}
        })

        # Reliability curve
        result["plots"].append({
            "title": "Reliability Function",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": rel.tolist(),
                 "mode": "lines", "name": "R(t)", "line": {"color": "#e89547", "width": 2}, "fill": "tozeroy",
                 "fillcolor": "rgba(232, 149, 71, 0.15)"},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": "Time"}, "yaxis": {"title": "Reliability R(t)", "range": [0, 1]}}
        })

        # Confidence interval on MTTF (chi-squared)
        chi2_lower = sp_stats.chi2.ppf(0.025, 2 * n)
        chi2_upper = sp_stats.chi2.ppf(0.975, 2 * n)
        mttf_lower = 2 * n * mttf / chi2_upper
        mttf_upper = 2 * n * mttf / chi2_lower

        result["summary"] = f"Exponential Analysis\n\nFailure rate (λ): {rate:.6f}\nMTTF: {mttf:.2f}\n95% CI on MTTF: [{mttf_lower:.2f}, {mttf_upper:.2f}]\n\nSample size: {n}\n\nNote: Exponential assumes constant failure rate (no wear-out)."

    elif analysis_id == "kaplan_meier":
        # Kaplan-Meier Survival Analysis
        time_col = config.get("time")
        event_col = config.get("event")  # 1=event occurred, 0=censored

        times = df[time_col].dropna().values
        if event_col and event_col in df.columns:
            events = df[event_col].dropna().values[:len(times)]
        else:
            events = np.ones(len(times))

        # Sort by time
        order = np.argsort(times)
        times = times[order]
        events = events[order]

        # KM estimator
        unique_times = np.unique(times[events == 1])
        n_at_risk = len(times)
        survival = []
        s = 1.0
        km_times = [0]
        km_survival = [1.0]
        km_ci_lower = [1.0]
        km_ci_upper = [1.0]
        var_sum = 0

        for t in unique_times:
            d = np.sum((times == t) & (events == 1))
            c = np.sum((times == t) & (events == 0))
            n = n_at_risk
            if n > 0:
                s *= (1 - d / n)
                if d > 0 and n > d:
                    var_sum += d / (n * (n - d))
            n_at_risk -= (d + c)

            # Greenwood confidence interval
            se = s * np.sqrt(var_sum) if var_sum > 0 else 0
            km_times.append(float(t))
            km_survival.append(float(s))
            km_ci_lower.append(max(0, float(s - 1.96 * se)))
            km_ci_upper.append(min(1, float(s + 1.96 * se)))

        # Censored points
        cens_t = times[events == 0]
        # Interpolate survival at censored times
        cens_s = []
        for ct in cens_t:
            idx = np.searchsorted(km_times, ct, side='right') - 1
            cens_s.append(km_survival[max(0, idx)])

        result["plots"].append({
            "title": "Kaplan-Meier Survival Curve",
            "data": [
                {"type": "scatter", "x": km_times, "y": km_survival, "mode": "lines",
                 "name": "Survival", "line": {"color": "#4a9f6e", "width": 2, "shape": "hv"}},
                {"type": "scatter", "x": km_times, "y": km_ci_upper, "mode": "lines",
                 "name": "95% CI", "line": {"color": "#4a9f6e", "width": 1, "dash": "dot", "shape": "hv"}, "showlegend": False},
                {"type": "scatter", "x": km_times, "y": km_ci_lower, "mode": "lines",
                 "name": "95% CI", "line": {"color": "#4a9f6e", "width": 1, "dash": "dot", "shape": "hv"},
                 "fill": "tonexty", "fillcolor": "rgba(74, 159, 110, 0.15)"},
            ] + ([{
                "type": "scatter", "x": cens_t.tolist(), "y": cens_s,
                "mode": "markers", "name": "Censored",
                "marker": {"color": "#4a90d9", "size": 8, "symbol": "cross"}
            }] if len(cens_t) > 0 else []),
            "layout": {"template": "plotly_dark", "height": 320, "xaxis": {"title": "Time"}, "yaxis": {"title": "Survival Probability", "range": [0, 1.05]}, "showlegend": True}
        })

        n_events = int(np.sum(events))
        n_censored = int(np.sum(events == 0))
        median_survival = "N/A"
        for i, s_val in enumerate(km_survival):
            if s_val <= 0.5:
                median_survival = f"{km_times[i]:.2f}"
                break

        result["summary"] = f"Kaplan-Meier Survival Analysis\n\nTotal observations: {len(times)}\nEvents: {n_events}\nCensored: {n_censored}\n\nMedian survival time: {median_survival}\nFinal survival probability: {km_survival[-1]:.4f}"

    elif analysis_id == "reliability_test_plan":
        # Reliability Test Planning — sample size for demonstration testing
        target_rel = float(config.get("target_reliability", 0.90))
        confidence = float(config.get("confidence", 0.95))
        test_duration = float(config.get("test_duration", 1000))
        dist = config.get("distribution", "exponential")

        if dist == "exponential":
            # For exponential: n = ln(1-C) / ln(R) where 0 failures
            n_required = int(np.ceil(np.log(1 - confidence) / np.log(target_rel)))

            # With allowed failures
            from scipy.special import comb
            results_table = []
            for failures in range(6):
                # Sum of binomial terms
                cum_prob = sum(comb(n_required + failures, k) * (1 - target_rel)**k * target_rel**(n_required + failures - k)
                               for k in range(failures + 1))
                n_for_f = n_required + failures
                # Adjust n until confidence is met
                n_adj = n_for_f
                while True:
                    cum = sum(comb(n_adj, k) * (1 - target_rel)**k * target_rel**(n_adj - k)
                             for k in range(failures + 1))
                    if 1 - cum >= confidence:
                        break
                    n_adj += 1
                    if n_adj > 10000:
                        break
                results_table.append({"failures": failures, "sample_size": n_adj})

            # Bar chart of sample sizes by allowed failures
            result["plots"].append({
                "title": "Required Sample Size vs Allowed Failures",
                "data": [{
                    "type": "bar",
                    "x": [f"{r['failures']} failures" for r in results_table],
                    "y": [r["sample_size"] for r in results_table],
                    "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a", "#9f4a4a", "#7a6a9a"]},
                    "text": [str(r["sample_size"]) for r in results_table],
                    "textposition": "outside"
                }],
                "layout": {"template": "plotly_dark", "height": 280, "yaxis": {"title": "Sample Size"}}
            })

            summary = f"Reliability Demonstration Test Plan\n\nTarget Reliability: {target_rel*100:.1f}%\nConfidence Level: {confidence*100:.1f}%\nTest Duration: {test_duration}\nDistribution: Exponential\n\nRequired Sample Sizes:\n"
            for r in results_table:
                summary += f"  {r['failures']} allowed failures: n = {r['sample_size']}\n"
            summary += f"\nZero-failure plan: Test {n_required} units for {test_duration} each with 0 failures."

            result["summary"] = summary

        elif dist == "weibull":
            beta = float(config.get("beta", 2.0))
            # For Weibull with shape beta
            n_required = int(np.ceil(np.log(1 - confidence) / np.log(target_rel)))
            af = (test_duration / test_duration) ** beta  # acceleration factor placeholder (ratio = 1 if test = use)

            result["plots"].append({
                "title": "Test Plan Parameters",
                "data": [{
                    "type": "bar",
                    "x": ["Sample Size", "Test Duration"],
                    "y": [n_required, test_duration],
                    "marker": {"color": ["#4a9f6e", "#4a90d9"]},
                    "text": [str(n_required), str(test_duration)],
                    "textposition": "outside"
                }],
                "layout": {"template": "plotly_dark", "height": 250, "yaxis": {"title": "Value"}}
            })

            result["summary"] = f"Reliability Test Plan (Weibull)\n\nTarget Reliability: {target_rel*100:.1f}%\nConfidence: {confidence*100:.1f}%\nWeibull β: {beta}\n\nZero-failure plan: Test {n_required} units for {test_duration} each."

    elif analysis_id == "distribution_id":
        """
        Distribution Identification — fits multiple distributions,
        ranks by goodness-of-fit, shows probability plots for top fits.
        """
        time_col = config.get("time")
        times = df[time_col].dropna().values
        times = times[times > 0]
        n = len(times)

        distributions = {}

        # Normal
        mu, sigma = sp_stats.norm.fit(times)
        ks = sp_stats.kstest(times, "norm", args=(mu, sigma))
        distributions["Normal"] = {"dist": sp_stats.norm, "args": (mu, sigma), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                   "params": f"μ={mu:.2f}, σ={sigma:.2f}"}

        # Lognormal
        s, loc, scale = sp_stats.lognorm.fit(times, floc=0)
        ks = sp_stats.kstest(times, "lognorm", args=(s, 0, scale))
        distributions["Lognormal"] = {"dist": sp_stats.lognorm, "args": (s, 0, scale), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                      "params": f"μ={np.log(scale):.2f}, σ={s:.2f}"}

        # Weibull
        c, loc_w, scale_w = sp_stats.weibull_min.fit(times, floc=0)
        ks = sp_stats.kstest(times, "weibull_min", args=(c, 0, scale_w))
        distributions["Weibull"] = {"dist": sp_stats.weibull_min, "args": (c, 0, scale_w), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                    "params": f"β={c:.2f}, η={scale_w:.2f}"}

        # Exponential
        loc_e, scale_e = sp_stats.expon.fit(times)
        ks = sp_stats.kstest(times, "expon", args=(loc_e, scale_e))
        distributions["Exponential"] = {"dist": sp_stats.expon, "args": (loc_e, scale_e), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                        "params": f"λ={1/scale_e:.4f}"}

        # Gamma
        a, loc_g, scale_g = sp_stats.gamma.fit(times, floc=0)
        ks = sp_stats.kstest(times, "gamma", args=(a, 0, scale_g))
        distributions["Gamma"] = {"dist": sp_stats.gamma, "args": (a, 0, scale_g), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                  "params": f"α={a:.2f}, β={scale_g:.2f}"}

        # Loglogistic (use fisk distribution in scipy)
        c_ll, loc_ll, scale_ll = sp_stats.fisk.fit(times, floc=0)
        ks = sp_stats.kstest(times, "fisk", args=(c_ll, 0, scale_ll))
        distributions["Loglogistic"] = {"dist": sp_stats.fisk, "args": (c_ll, 0, scale_ll), "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                                        "params": f"μ={np.log(scale_ll):.2f}, σ={1/c_ll:.2f}"}

        # Rank by KS p-value (higher = better fit)
        ranked = sorted(distributions.items(), key=lambda x: x[1]["ks_p"], reverse=True)

        summary = f"Distribution Identification\n\nSample size: {n}\n\n"
        summary += f"{'Distribution':<15} {'Parameters':<25} {'KS Stat':>10} {'p-value':>10}\n"
        summary += f"{'-'*65}\n"
        for i, (name, info) in enumerate(ranked):
            marker = " <-- Best" if i == 0 else ""
            summary += f"{name:<15} {info['params']:<25} {info['ks_stat']:>10.4f} {info['ks_p']:>10.4f}{marker}\n"

        best_name, best_info = ranked[0]
        summary += f"\nRecommended: {best_name} ({best_info['params']})"

        result["summary"] = summary

        # Probability plots for top 3 distributions
        sorted_t = np.sort(times)
        median_ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)
        theme_colors = ['#4a9f6e', '#4a90d9', '#e89547']

        for idx, (name, info) in enumerate(ranked[:3]):
            dist = info["dist"]
            args = info["args"]
            theoretical = dist.ppf(median_ranks, *args)

            result["plots"].append({
                "title": f"Probability Plot — {name}" + (" (Best)" if idx == 0 else ""),
                "data": [
                    {"type": "scatter", "x": theoretical.tolist(), "y": sorted_t.tolist(),
                     "mode": "markers", "name": "Data", "marker": {"color": theme_colors[idx], "size": 5}},
                    {"type": "scatter", "x": [float(min(theoretical)), float(max(theoretical))],
                     "y": [float(min(theoretical)), float(max(theoretical))],
                     "mode": "lines", "name": "Reference", "line": {"color": "#d94a4a", "dash": "dash"}},
                ],
                "layout": {"template": "plotly_dark", "height": 250, "showlegend": True,
                           "xaxis": {"title": f"Theoretical ({name})"}, "yaxis": {"title": "Observed"}}
            })

        # Overlay histogram with top 3 PDFs
        x_range = np.linspace(min(times), max(times), 200)
        hist_traces = [{"type": "histogram", "x": times.tolist(), "name": "Data",
                       "marker": {"color": "rgba(74,159,110,0.3)", "line": {"color": "#4a9f6e", "width": 1}},
                       "histnorm": "probability density"}]
        for idx, (name, info) in enumerate(ranked[:3]):
            pdf = info["dist"].pdf(x_range, *info["args"])
            hist_traces.append({"type": "scatter", "x": x_range.tolist(), "y": pdf.tolist(),
                               "mode": "lines", "name": name, "line": {"color": theme_colors[idx], "width": 2}})
        result["plots"].append({
            "title": "Distribution Comparison",
            "data": hist_traces,
            "layout": {"template": "plotly_dark", "height": 280, "showlegend": True, "xaxis": {"title": "Time"}, "yaxis": {"title": "Density"}}
        })

    elif analysis_id == "accelerated_life":
        """
        Accelerated Life Testing — fits life-stress model.
        Supports Arrhenius (temperature) and Inverse Power Law (voltage/stress).
        """
        time_col = config.get("time")
        stress_col = config.get("stress")
        model_type = config.get("model", "arrhenius")  # arrhenius or inverse_power
        use_stress = float(config.get("use_stress", 25))  # use condition

        times = df[time_col].dropna().values
        stresses = df[stress_col].dropna().values[:len(times)]

        unique_stresses = np.sort(np.unique(stresses))
        if len(unique_stresses) < 2:
            result["summary"] = "Error: Need at least 2 stress levels for ALT."
            return result

        # Fit Weibull at each stress level
        stress_results = []
        for stress in unique_stresses:
            mask = stresses == stress
            t_at_stress = times[mask]
            t_at_stress = t_at_stress[t_at_stress > 0]
            if len(t_at_stress) < 3:
                continue
            shape, _, scale = sp_stats.weibull_min.fit(t_at_stress, floc=0)
            stress_results.append({"stress": float(stress), "shape": shape, "scale": scale, "n": len(t_at_stress)})

        if len(stress_results) < 2:
            result["summary"] = "Error: Not enough data at each stress level (need 3+ per level)."
            return result

        # Common shape assumption (average)
        common_shape = np.mean([r["shape"] for r in stress_results])

        # Fit life-stress model: ln(scale) = a + b * transform(stress)
        log_scales = np.array([np.log(r["scale"]) for r in stress_results])
        stress_vals = np.array([r["stress"] for r in stress_results])

        if model_type == "arrhenius":
            # Arrhenius: ln(L) = a + b/T  (T in Kelvin)
            x_transform = 1.0 / (stress_vals + 273.15)  # assume Celsius
            x_use = 1.0 / (use_stress + 273.15)
            stress_label = "1/T (K)"
        else:
            # Inverse Power: ln(L) = a - b*ln(S)
            x_transform = np.log(stress_vals)
            x_use = np.log(use_stress)
            stress_label = "ln(Stress)"

        # Linear regression
        slope, intercept, r_value, _, _ = sp_stats.linregress(x_transform, log_scales)
        log_scale_use = intercept + slope * x_use
        scale_use = np.exp(log_scale_use)

        # Life at use conditions
        b10_use = sp_stats.weibull_min.ppf(0.10, common_shape, 0, scale_use)
        b50_use = sp_stats.weibull_min.ppf(0.50, common_shape, 0, scale_use)
        import math
        mttf_use = scale_use * np.exp(math.lgamma(1 + 1/common_shape))

        summary = f"Accelerated Life Testing\n\n"
        summary += f"Model: {'Arrhenius' if model_type == 'arrhenius' else 'Inverse Power Law'}\n"
        summary += f"Use Stress: {use_stress}\n"
        summary += f"Common Shape (β): {common_shape:.3f}\n\n"
        summary += f"Stress Level Results:\n"
        summary += f"  {'Stress':>10} {'n':>5} {'Shape':>8} {'Scale':>10}\n"
        summary += f"  {'-'*38}\n"
        for r in stress_results:
            summary += f"  {r['stress']:>10.1f} {r['n']:>5} {r['shape']:>8.3f} {r['scale']:>10.1f}\n"
        summary += f"\nLife-Stress Model: R² = {r_value**2:.4f}\n"
        summary += f"\nExtrapolated Life at Use Conditions ({use_stress}):\n"
        summary += f"  Scale (η): {scale_use:.1f}\n"
        summary += f"  B10 Life: {b10_use:.1f}\n"
        summary += f"  B50 Life: {b50_use:.1f}\n"
        summary += f"  MTTF: {mttf_use:.1f}"

        result["summary"] = summary

        # Life vs Stress plot
        x_plot = np.linspace(min(x_transform)*0.9, max(max(x_transform), x_use)*1.1, 100)
        y_plot = np.exp(intercept + slope * x_plot)

        result["plots"].append({
            "title": "Life vs Stress Relationship",
            "data": [
                {"type": "scatter", "x": x_transform.tolist(), "y": [r["scale"] for r in stress_results],
                 "mode": "markers", "name": "Test Data", "marker": {"color": "#4a9f6e", "size": 10}},
                {"type": "scatter", "x": x_plot.tolist(), "y": y_plot.tolist(),
                 "mode": "lines", "name": "Model Fit", "line": {"color": "#4a90d9", "width": 2}},
                {"type": "scatter", "x": [float(x_use)], "y": [float(scale_use)],
                 "mode": "markers", "name": f"Use ({use_stress})", "marker": {"color": "#d94a4a", "size": 12, "symbol": "star"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True,
                       "xaxis": {"title": stress_label}, "yaxis": {"title": "Characteristic Life (η)", "type": "log"}}
        })

        # Reliability at use stress
        t_range = np.linspace(0, scale_use * 2, 200)
        rel_use = 1 - sp_stats.weibull_min.cdf(t_range, common_shape, 0, scale_use)
        result["plots"].append({
            "title": f"Reliability at Use Stress ({use_stress})",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": rel_use.tolist(),
                 "mode": "lines", "name": "R(t)", "line": {"color": "#4a9f6e", "width": 2},
                 "fill": "tozeroy", "fillcolor": "rgba(74,159,110,0.15)"},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": "Time"}, "yaxis": {"title": "Reliability", "range": [0, 1.05]}}
        })

    elif analysis_id == "repairable_systems":
        """
        Repairable Systems — Crow-AMSAA (Power Law NHPP).
        Models failure intensity for repairable systems.
        """
        time_col = config.get("time")
        system_col = config.get("system")

        if system_col and system_col in df.columns:
            # Multiple systems
            systems = df[system_col].unique()
            all_events = []
            for sys in systems:
                events = np.sort(df[df[system_col] == sys][time_col].dropna().values)
                all_events.append(events)
        else:
            # Single system — all events
            all_events = [np.sort(df[time_col].dropna().values)]

        n_systems = len(all_events)
        total_events = sum(len(e) for e in all_events)

        # Pool all events for single-system analysis
        pooled = np.sort(np.concatenate(all_events))
        n = len(pooled)
        T = pooled[-1]  # total observation time

        # Fit Power Law (Crow-AMSAA): N(t) = (t/θ)^β
        # MLE: β = n / Σln(T/ti), θ = T / n^(1/β)
        log_sum = np.sum(np.log(T / pooled[pooled > 0]))
        beta_crow = n / log_sum if log_sum > 0 else 1.0
        theta_crow = T / (n ** (1 / beta_crow))

        # Laplace test for trend
        laplace_stat = (np.mean(pooled) - T/2) / (T / np.sqrt(12 * n))
        laplace_p = 2 * (1 - sp_stats.norm.cdf(abs(laplace_stat)))

        summary = f"Repairable Systems Analysis (Crow-AMSAA)\n\n"
        summary += f"Systems: {n_systems}\n"
        summary += f"Total Events: {total_events}\n"
        summary += f"Observation Period: {T:.1f}\n\n"
        summary += f"Power Law Parameters:\n"
        summary += f"  β (shape): {beta_crow:.4f}\n"
        summary += f"  θ (scale): {theta_crow:.2f}\n\n"
        summary += f"Trend Test (Laplace):\n"
        summary += f"  Statistic: {laplace_stat:.4f}\n"
        summary += f"  p-value: {laplace_p:.4f}\n"

        if laplace_p < 0.05:
            if laplace_stat > 0:
                summary += f"  Result: DETERIORATING — failure rate increasing\n"
            else:
                summary += f"  Result: IMPROVING — failure rate decreasing\n"
        else:
            summary += f"  Result: NO TREND — stable failure rate (HPP)\n"

        if beta_crow > 1:
            summary += f"\nβ > 1: System deteriorating (wear-out)"
        elif beta_crow < 1:
            summary += f"\nβ < 1: System improving (reliability growth)"
        else:
            summary += f"\nβ ≈ 1: Constant failure rate"

        result["summary"] = summary

        # MCF (Mean Cumulative Function) plot
        mcf_t = [0] + pooled.tolist()
        mcf_n = list(range(len(mcf_t)))
        # Fitted model: E[N(t)] = (t/θ)^β
        t_fit = np.linspace(0, T * 1.2, 200)
        n_fit = (t_fit / theta_crow) ** beta_crow

        result["plots"].append({
            "title": "Mean Cumulative Function (MCF)",
            "data": [
                {"type": "scatter", "x": mcf_t, "y": mcf_n, "mode": "lines",
                 "name": "Observed", "line": {"color": "#4a9f6e", "width": 2, "shape": "hv"}},
                {"type": "scatter", "x": t_fit.tolist(), "y": n_fit.tolist(), "mode": "lines",
                 "name": f"Crow-AMSAA (β={beta_crow:.2f})", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True,
                       "xaxis": {"title": "Time"}, "yaxis": {"title": "Cumulative Events"}}
        })

        # Instantaneous failure rate: λ(t) = (β/θ)(t/θ)^(β-1)
        t_rate = np.linspace(pooled[0] * 0.5, T * 1.1, 200)
        rate = (beta_crow / theta_crow) * (t_rate / theta_crow) ** (beta_crow - 1)
        result["plots"].append({
            "title": "Failure Intensity (ROCOF)",
            "data": [
                {"type": "scatter", "x": t_rate.tolist(), "y": rate.tolist(), "mode": "lines",
                 "name": "λ(t)", "line": {"color": "#e89547", "width": 2}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Time"}, "yaxis": {"title": "Failure Rate"}}
        })

    elif analysis_id == "warranty":
        """
        Warranty Prediction — forecasts future returns from field data.
        """
        time_col = config.get("time")       # time-to-return (age at return)
        warranty_period = float(config.get("warranty_period", 365))
        fleet_size = int(config.get("fleet_size", 1000))

        times = df[time_col].dropna().values
        times = times[times > 0]
        n_returns = len(times)

        # Fit Weibull to return times
        shape, _, scale = sp_stats.weibull_min.fit(times, floc=0)

        # Return rate function: F(t)
        t_range = np.linspace(0, warranty_period * 1.5, 300)
        cdf = sp_stats.weibull_min.cdf(t_range, shape, 0, scale)

        # Projected returns
        projected_in_warranty = fleet_size * sp_stats.weibull_min.cdf(warranty_period, shape, 0, scale)

        # Monthly projection
        months = int(warranty_period / 30)
        monthly_returns = []
        for m in range(months + 1):
            t = m * 30
            cum_returns = fleet_size * sp_stats.weibull_min.cdf(t, shape, 0, scale)
            monthly_returns.append({"month": m, "cumulative": cum_returns, "incremental": 0})
        for i in range(1, len(monthly_returns)):
            monthly_returns[i]["incremental"] = monthly_returns[i]["cumulative"] - monthly_returns[i-1]["cumulative"]

        summary = f"Warranty Prediction\n\n"
        summary += f"Observed Returns: {n_returns}\n"
        summary += f"Fleet Size: {fleet_size}\n"
        summary += f"Warranty Period: {warranty_period:.0f}\n\n"
        summary += f"Fitted Distribution: Weibull\n"
        summary += f"  Shape (β): {shape:.3f}\n"
        summary += f"  Scale (η): {scale:.1f}\n\n"
        summary += f"Projected Returns in Warranty: {projected_in_warranty:.0f} ({projected_in_warranty/fleet_size*100:.2f}%)\n\n"
        summary += f"Monthly Forecast (next 6 months):\n"
        summary += f"  {'Month':>6} {'Incremental':>13} {'Cumulative':>12}\n"
        summary += f"  {'-'*35}\n"
        for mr in monthly_returns[:7]:
            summary += f"  {mr['month']:>6} {mr['incremental']:>13.1f} {mr['cumulative']:>12.1f}\n"

        result["summary"] = summary

        # Cumulative return rate
        result["plots"].append({
            "title": "Cumulative Return Rate",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": (cdf * 100).tolist(), "mode": "lines",
                 "name": "Projected %", "line": {"color": "#4a9f6e", "width": 2}, "fill": "tozeroy", "fillcolor": "rgba(74,159,110,0.15)"},
                {"type": "scatter", "x": [warranty_period, warranty_period], "y": [0, float(cdf[-1]*100)],
                 "mode": "lines", "name": "Warranty End", "line": {"color": "#d94a4a", "dash": "dash", "width": 2}},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "showlegend": True,
                       "xaxis": {"title": "Age (days)"}, "yaxis": {"title": "Cumulative Return Rate (%)"}}
        })

        # Monthly incremental returns
        result["plots"].append({
            "title": "Monthly Incremental Returns",
            "data": [{
                "type": "bar",
                "x": [f"M{mr['month']}" for mr in monthly_returns[1:]],
                "y": [mr["incremental"] for mr in monthly_returns[1:]],
                "marker": {"color": "#4a90d9"},
            }],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Month"}, "yaxis": {"title": "Returns"}}
        })

    elif analysis_id == "competing_risks":
        """
        Competing Risks Analysis — estimates cumulative incidence functions (CIF)
        when multiple failure modes exist. Uses Aalen-Johansen estimator.
        """
        time_col = config.get("time") or config.get("var")
        event_col = config.get("event") or config.get("failure_mode")
        try:
            data = df[[time_col, event_col]].dropna()
            times = data[time_col].values.astype(float)
            events = data[event_col].values
            N = len(data)

            # Event types: 0 = censored, others are failure modes
            unique_events = sorted([e for e in np.unique(events) if e != 0 and str(e) != '0' and str(e).lower() != 'censored'], key=str)
            n_events = len(unique_events)

            if n_events < 1:
                result["summary"] = "No failure events found. Ensure event column has non-zero values for failure modes."
                return result

            # Sort by time
            order = np.argsort(times)
            times = times[order]
            events = events[order]

            # Unique event times
            unique_times = np.sort(np.unique(times))

            # Kaplan-Meier overall survival for denominator
            n_risk = np.zeros(len(unique_times))
            km_surv = np.ones(len(unique_times) + 1)

            # Compute CIF for each event type
            cifs = {}
            for event_type in unique_events:
                cif_vals = [0.0]
                cif_times = [0.0]
                surv_prev = 1.0

                for ti, t in enumerate(unique_times):
                    at_risk = np.sum(times >= t)
                    if at_risk == 0:
                        continue

                    # Count events of any type and specific type at this time
                    d_j = np.sum((times == t) & (events == event_type))
                    d_all = np.sum((times == t) & (events != 0) & (np.array([str(e) for e in events]) != '0') & (np.array([str(e).lower() for e in events]) != 'censored'))

                    # Cause-specific hazard
                    h_j = d_j / at_risk
                    h_all = d_all / at_risk

                    # CIF increment = S(t-) * h_j(t)
                    cif_increment = surv_prev * h_j

                    cif_vals.append(cif_vals[-1] + cif_increment)
                    cif_times.append(t)

                    # Update overall survival
                    surv_prev = surv_prev * (1 - h_all)

                cifs[str(event_type)] = {"times": cif_times, "values": cif_vals}

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += f"<<COLOR:title>>COMPETING RISKS ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Time variable:<</COLOR>> {time_col}\n"
            summary_text += f"<<COLOR:highlight>>Event variable:<</COLOR>> {event_col}\n"
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n"
            summary_text += f"<<COLOR:highlight>>Failure modes:<</COLOR>> {n_events}\n\n"

            summary_text += f"<<COLOR:text>>Cumulative Incidence at Final Time:<</COLOR>>\n"
            summary_text += f"{'Failure Mode':<20} {'Events':>8} {'CIF (final)':>12}\n"
            summary_text += f"{'─' * 42}\n"
            for event_type in unique_events:
                n_events_type = int(np.sum(events == event_type))
                final_cif = cifs[str(event_type)]["values"][-1]
                summary_text += f"{str(event_type):<20} {n_events_type:>8} {final_cif:>12.4f}\n"

            n_censored = int(np.sum((events == 0) | (np.array([str(e) for e in events]) == '0') | (np.array([str(e).lower() for e in events]) == 'censored')))
            summary_text += f"\n<<COLOR:text>>Censored observations:<</COLOR>> {n_censored}"

            result["summary"] = summary_text

            # CIF plot
            traces = []
            colors = ["#4a90d9", "#d94a4a", "#4a9f6e", "#d9a04a", "#9b59b6"]
            for ei, event_type in enumerate(unique_events):
                cif = cifs[str(event_type)]
                traces.append({
                    "x": cif["times"], "y": cif["values"],
                    "mode": "lines", "name": f"CIF: {event_type}",
                    "line": {"color": colors[ei % len(colors)], "width": 2, "shape": "hv"}
                })

            result["plots"].append({
                "title": "Cumulative Incidence Functions",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": time_col}, "yaxis": {"title": "Cumulative Incidence", "range": [0, 1.05]},
                    "template": "plotly_white"
                }
            })

            # Stacked area plot
            stacked_traces = []
            for ei, event_type in enumerate(unique_events):
                cif = cifs[str(event_type)]
                stacked_traces.append({
                    "x": cif["times"], "y": cif["values"],
                    "mode": "lines", "name": str(event_type),
                    "stackgroup": "one",
                    "line": {"color": colors[ei % len(colors)]},
                    "fillcolor": colors[ei % len(colors)] + "40"
                })
            result["plots"].append({
                "title": "Stacked Cumulative Incidence",
                "data": stacked_traces,
                "layout": {"height": 300, "xaxis": {"title": time_col}, "yaxis": {"title": "Cumulative Incidence"}, "template": "plotly_white"}
            })

            result["statistics"] = {
                "n": N, "n_censored": n_censored, "n_failure_modes": n_events,
                "cif_final": {str(et): cifs[str(et)]["values"][-1] for et in unique_events},
                "event_counts": {str(et): int(np.sum(events == et)) for et in unique_events}
            }
            result["guide_observation"] = f"Competing risks: {n_events} failure modes. " + ", ".join([f"{et}: CIF={cifs[str(et)]['values'][-1]:.3f}" for et in unique_events]) + "."

        except Exception as e:
            result["summary"] = f"Competing risks error: {str(e)}"

    return result


def _spc_nelson_rules(data, cl, ucl, lcl):
    """Check all 8 Nelson rules and return OOC indices + rule violations."""
    import numpy as np
    n = len(data)
    sigma = (ucl - cl) / 3 if ucl != cl else 1
    one_sigma_up = cl + sigma
    one_sigma_dn = cl - sigma
    two_sigma_up = cl + 2 * sigma
    two_sigma_dn = cl - 2 * sigma
    ooc_indices = set()
    violations = []

    # Rule 1: Point beyond 3σ (beyond control limits)
    for i in range(n):
        if data[i] > ucl or data[i] < lcl:
            ooc_indices.add(i)

    # Rule 2: 9 consecutive points same side of CL
    for i in range(8, n):
        window = data[i-8:i+1]
        if all(v > cl for v in window) or all(v < cl for v in window):
            ooc_indices.update(range(i-8, i+1))
            violations.append(f"Rule 2: 9 same side at {i-8+1}-{i+1}")
            break

    # Rule 3: 6 consecutive points trending (all increasing or all decreasing)
    for i in range(5, n):
        window = data[i-5:i+1]
        diffs = [window[j+1] - window[j] for j in range(5)]
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            ooc_indices.update(range(i-5, i+1))
            direction = "increasing" if diffs[0] > 0 else "decreasing"
            violations.append(f"Rule 3: 6 {direction} at {i-5+1}-{i+1}")
            break

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i-13:i+1]
            diffs = [window[j+1] - window[j] for j in range(13)]
            if all(diffs[j] * diffs[j+1] < 0 for j in range(12)):
                ooc_indices.update(range(i-13, i+1))
                violations.append(f"Rule 4: 14 alternating at {i-13+1}-{i+1}")
                break

    # Rule 5: 2 of 3 beyond 2σ (same side)
    for i in range(2, n):
        w = data[i-2:i+1]
        if sum(1 for v in w if v > two_sigma_up) >= 2:
            ooc_indices.update(range(i-2, i+1))
        if sum(1 for v in w if v < two_sigma_dn) >= 2:
            ooc_indices.update(range(i-2, i+1))

    # Rule 6: 4 of 5 beyond 1σ (same side)
    for i in range(4, n):
        w = data[i-4:i+1]
        if sum(1 for v in w if v > one_sigma_up) >= 4:
            ooc_indices.update(range(i-4, i+1))
        if sum(1 for v in w if v < one_sigma_dn) >= 4:
            ooc_indices.update(range(i-4, i+1))

    # Rule 7: 15 consecutive within 1σ (stratification — too little variation)
    if n >= 15:
        for i in range(14, n):
            window = data[i-14:i+1]
            if all(one_sigma_dn <= v <= one_sigma_up for v in window):
                ooc_indices.update(range(i-14, i+1))
                violations.append(f"Rule 7: 15 within 1σ at {i-14+1}-{i+1}")
                break

    # Rule 8: 8 consecutive beyond 1σ on both sides (mixture pattern)
    if n >= 8:
        for i in range(7, n):
            window = data[i-7:i+1]
            if all(v > one_sigma_up or v < one_sigma_dn for v in window):
                ooc_indices.update(range(i-7, i+1))
                violations.append(f"Rule 8: 8 beyond 1σ (mixture) at {i-7+1}-{i+1}")
                break

    return list(sorted(ooc_indices)), violations


def _spc_add_ooc_markers(plot_data, data, ooc_indices):
    """Add red markers for OOC points to a Plotly chart trace list."""
    if not ooc_indices:
        return
    import numpy as np
    ooc_x = ooc_indices
    ooc_y = [float(data[i]) for i in ooc_indices]
    plot_data.append({
        "type": "scatter", "x": ooc_x, "y": ooc_y,
        "mode": "markers", "name": "Out of Control",
        "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
        "showlegend": True
    })


def run_spc_analysis(df, analysis_id, config):
    """Run SPC analysis."""
    import numpy as np

    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    data = df[measurement].dropna().values

    if analysis_id == "imr":
        # Individual-Moving Range chart
        n = len(data)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr)

        x_bar = np.mean(data)
        ucl = x_bar + 2.66 * mr_bar
        lcl = x_bar - 2.66 * mr_bar

        mr_ucl = 3.267 * mr_bar

        # Nelson rules check
        ooc_indices, rule_violations = _spc_nelson_rules(data, x_bar, ucl, lcl)

        # I Chart with OOC markers
        i_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "lines+markers", "name": "Value", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_bar]*n, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(i_chart_data, data, ooc_indices)
        result["plots"].append({
            "title": "I Chart (Individuals)",
            "data": i_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True}
        })

        # MR Chart with OOC markers
        mr_ooc = [i for i in range(len(mr)) if mr[i] > mr_ucl]
        mr_chart_data = [
            {"type": "scatter", "y": mr.tolist(), "mode": "lines+markers", "name": "MR", "marker": {"color": "rgba(74, 159, 110, 0.4)", "size": 5, "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [mr_bar]*(n-1), "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [mr_ucl]*(n-1), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(mr_chart_data, mr, mr_ooc)
        result["plots"].append({
            "title": "MR Chart (Moving Range)",
            "data": mr_chart_data,
            "layout": {"template": "plotly_dark", "height": 250}
        })

        ooc = len(ooc_indices)
        violations_text = ""
        if rule_violations:
            violations_text = "\n\nNelson Rule Violations:\n" + "\n".join(f"  {v}" for v in rule_violations)
        result["summary"] = f"I-MR Chart Analysis\n\nMean: {x_bar:.4f}\nUCL: {ucl:.4f}\nLCL: {lcl:.4f}\nMR-bar: {mr_bar:.4f}\n\nOut-of-control points: {ooc}{violations_text}"

        result["guide_observation"] = f"Control chart shows {ooc} out-of-control points." + (" Process appears stable." if ooc == 0 else " Investigation recommended.")

    elif analysis_id == "capability":
        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None
        target = float(config.get("target")) if config.get("target") else None

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary = f"Capability Analysis\n\nMean: {mean:.4f}\nStd Dev: {std:.4f}\n"

        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * std)
            if target:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            else:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

            summary += f"\nCp: {cp:.3f}\nCpk: {cpk:.3f}"

            if cpk >= 1.33:
                summary += "\n\nProcess is capable (Cpk >= 1.33)"
            elif cpk >= 1.0:
                summary += "\n\nProcess is marginally capable (1.0 <= Cpk < 1.33)"
            else:
                summary += "\n\nProcess is NOT capable (Cpk < 1.0)"

            result["guide_observation"] = f"Process capability Cpk = {cpk:.2f}. " + ("Capable." if cpk >= 1.33 else "Needs improvement.")

        # Histogram with spec limits - Svend theme (pale green fill, bright green border)
        hist_data = [{
            "type": "histogram",
            "x": data.tolist(),
            "name": "Data",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",  # Pale green fill
                "line": {"color": "#4a9f6e", "width": 1.5}  # Bright green border
            }
        }]

        # Add LSL/USL as dashed vertical lines (matching anomaly threshold style)
        shapes = []
        annotations = []

        if lsl:
            shapes.append({
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        if usl:
            shapes.append({
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        result["plots"].append({
            "title": "Capability Histogram",
            "data": hist_data,
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "shapes": shapes,
                "annotations": annotations,
                "margin": {"t": 40}
            }
        })

        result["summary"] = summary

    elif analysis_id == "xbar_r":
        # Xbar-R Chart for subgrouped data
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            # Group by subgroup column
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            # Create subgroups from sequential data
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        ranges = np.array([np.max(g) - np.min(g) for g in groups])

        x_double_bar = np.mean(x_bars)
        r_bar = np.mean(ranges)

        # Control chart constants (for subgroup size 2-10)
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d3_table = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}
        A2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_table = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        n = min(subgroup_size, 10)
        A2 = A2_table.get(n, 0.577)
        D3 = D3_table.get(n, 0)
        D4 = D4_table.get(n, 2.114)

        # Xbar limits
        xbar_ucl = x_double_bar + A2 * r_bar
        xbar_lcl = x_double_bar - A2 * r_bar

        # R limits
        r_ucl = D4 * r_bar
        r_lcl = D3 * r_bar

        # Nelson rules for X-bar
        xbar_ooc, xbar_violations = _spc_nelson_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl)
        # Nelson rules for R
        r_ooc, r_violations = _spc_nelson_rules(ranges, r_bar, r_ucl, r_lcl)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # R Chart with OOC markers
        r_chart_data = [
            {"type": "scatter", "y": ranges.tolist(), "mode": "lines+markers", "name": "R", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [r_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [r_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [r_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(r_chart_data, ranges, r_ooc)
        result["plots"].append({
            "title": "R Chart",
            "data": r_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        violations_text = ""
        if xbar_violations or r_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in r_violations: violations_text += f"\n  R: {v}"

        result["summary"] = f"Xbar-R Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nR Chart:\n  R̄: {r_bar:.4f}\n  UCL: {r_ucl:.4f}\n  LCL: {r_lcl:.4f}\n  OOC points: {len(r_ooc)}{violations_text}"

    elif analysis_id == "xbar_s":
        # Xbar-S Chart (using standard deviation instead of range)
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        stds = np.array([np.std(g, ddof=1) for g in groups])

        x_double_bar = np.mean(x_bars)
        s_bar = np.mean(stds)

        # Control chart constants for S chart
        c4_table = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}
        B3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
        B4_table = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
        A3_table = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}

        n = min(subgroup_size, 10)
        A3 = A3_table.get(n, 1.427)
        B3 = B3_table.get(n, 0)
        B4 = B4_table.get(n, 2.089)

        xbar_ucl = x_double_bar + A3 * s_bar
        xbar_lcl = x_double_bar - A3 * s_bar
        s_ucl = B4 * s_bar
        s_lcl = B3 * s_bar

        # Nelson rules for X-bar and S
        xbar_ooc, xbar_violations = _spc_nelson_rules(x_bars, x_double_bar, xbar_ucl, xbar_lcl)
        s_ooc, s_violations = _spc_nelson_rules(stds, s_bar, s_ucl, s_lcl)

        # Xbar Chart with OOC markers
        xbar_chart_data = [
            {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(xbar_chart_data, x_bars, xbar_ooc)
        result["plots"].append({
            "title": "Xbar Chart",
            "data": xbar_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # S Chart with OOC markers
        s_chart_data = [
            {"type": "scatter", "y": stds.tolist(), "mode": "lines+markers", "name": "S", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [s_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [s_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [s_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(s_chart_data, stds, s_ooc)
        result["plots"].append({
            "title": "S Chart",
            "data": s_chart_data,
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        violations_text = ""
        if xbar_violations or s_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in xbar_violations: violations_text += f"\n  X̄: {v}"
            for v in s_violations: violations_text += f"\n  S: {v}"

        result["summary"] = f"Xbar-S Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {len(xbar_ooc)}\n\nS Chart:\n  S̄: {s_bar:.4f}\n  UCL: {s_ucl:.4f}\n  LCL: {s_lcl:.4f}\n  OOC points: {len(s_ooc)}{violations_text}"

    elif analysis_id == "p_chart":
        # P Chart for proportion defective
        defectives = config.get("defectives")
        sample_size = config.get("sample_size")

        d = df[defectives].dropna().values
        n = df[sample_size].dropna().values
        p = d / n

        p_bar = np.sum(d) / np.sum(n)
        k = len(p)

        # Control limits (variable since sample size may vary)
        ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
        lcl = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

        # OOC detection for variable-limit chart
        ooc_indices = [i for i in range(k) if p[i] > ucl[i] or p[i] < lcl[i]]

        p_chart_data = [
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(p_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "P Chart (Proportion Defective)",
            "data": p_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Proportion"}}
        })

        result["summary"] = f"P Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}"

    elif analysis_id == "np_chart":
        """
        NP Chart - Number defective (constant sample size).
        """
        defectives = config.get("defectives")
        sample_size = int(config.get("sample_size", 50))

        d = df[defectives].dropna().values
        n = sample_size
        k = len(d)

        np_bar = np.mean(d)
        p_bar = np_bar / n

        # Control limits
        ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

        np_ooc, np_violations = _spc_nelson_rules(d, np_bar, ucl, lcl)

        np_chart_data = [
            {"type": "scatter", "y": d.tolist(), "mode": "lines+markers", "name": "np", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [np_bar]*k, "mode": "lines", "name": "n̄p", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(np_chart_data, d, np_ooc)
        result["plots"].append({
            "title": "NP Chart (Number Defective)",
            "data": np_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Number Defective"}}
        })

        violations_text = ""
        if np_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in np_violations: violations_text += f"\n  {v}"

        result["summary"] = f"NP Chart Analysis\n\nn̄p: {np_bar:.2f}\nSample size: {n}\np̄: {p_bar:.4f}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(np_ooc)}{violations_text}"

    elif analysis_id == "c_chart":
        """
        C Chart - Count of defects per unit (constant opportunity).
        """
        defects = config.get("defects")

        c = df[defects].dropna().values
        k = len(c)

        c_bar = np.mean(c)

        # Control limits (Poisson-based)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

        c_ooc, c_violations = _spc_nelson_rules(c, c_bar, ucl, lcl)

        c_chart_data = [
            {"type": "scatter", "y": c.tolist(), "mode": "lines+markers", "name": "c", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [c_bar]*k, "mode": "lines", "name": "c̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(c_chart_data, c, c_ooc)
        result["plots"].append({
            "title": "C Chart (Defects per Unit)",
            "data": c_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects"}}
        })

        violations_text = ""
        if c_violations:
            violations_text = "\n\nNelson Rule Violations:"
            for v in c_violations: violations_text += f"\n  {v}"

        result["summary"] = f"C Chart Analysis\n\nc̄: {c_bar:.2f}\nSamples: {k}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {len(c_ooc)}{violations_text}"

    elif analysis_id == "u_chart":
        """
        U Chart - Defects per unit (variable sample size).
        """
        defects = config.get("defects")
        units = config.get("units")

        c = df[defects].dropna().values
        n = df[units].dropna().values
        u = c / n
        k = len(u)

        u_bar = np.sum(c) / np.sum(n)

        # Variable control limits
        ucl = u_bar + 3 * np.sqrt(u_bar / n)
        lcl = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n))

        # OOC detection for variable-limit chart
        u_ooc_indices = [i for i in range(k) if u[i] > ucl[i] or u[i] < lcl[i]]

        u_chart_data = [
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(u_chart_data, u, u_ooc_indices)
        result["plots"].append({
            "title": "U Chart (Defects per Unit)",
            "data": u_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects per Unit"}}
        })

        result["summary"] = f"U Chart Analysis\n\nū: {u_bar:.4f}\nSamples: {k}\n\nOut-of-control points: {len(u_ooc_indices)}"

    elif analysis_id == "cusum":
        """
        CUSUM Chart - Cumulative Sum for detecting small shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))  # Target value
        k_param = float(config.get("k", 0.5))  # Slack value (typically 0.5)
        h_param = float(config.get("h", 5))  # Decision interval

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        # Estimate standard deviation
        sigma = np.std(data, ddof=1)

        # Standardize
        z = (data - target) / sigma

        # Calculate CUSUM
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_pos[i] = max(0, z[i] - k_param)
                cusum_neg[i] = max(0, -z[i] - k_param)
            else:
                cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - k_param)
                cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - k_param)

        # Detect signals
        signals_pos = np.where(cusum_pos > h_param)[0]
        signals_neg = np.where(cusum_neg > h_param)[0]

        cusum_chart_data = [
            {"type": "scatter", "y": cusum_pos.tolist(), "mode": "lines", "name": "CUSUM+", "line": {"color": "#4a9f6e", "width": 2}},
            {"type": "scatter", "y": (-cusum_neg).tolist(), "mode": "lines", "name": "CUSUM-", "line": {"color": "#47a5e8", "width": 2}},
            {"type": "scatter", "y": [h_param]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [-h_param]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        # OOC markers for positive signals
        if len(signals_pos) > 0:
            cusum_chart_data.append({
                "type": "scatter", "x": signals_pos.tolist(), "y": cusum_pos[signals_pos].tolist(),
                "mode": "markers", "name": "Signal (up)",
                "marker": {"color": "#d94a4a", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                "showlegend": True
            })
        # OOC markers for negative signals
        if len(signals_neg) > 0:
            cusum_chart_data.append({
                "type": "scatter", "x": signals_neg.tolist(), "y": (-cusum_neg[signals_neg]).tolist(),
                "mode": "markers", "name": "Signal (down)",
                "marker": {"color": "#e89547", "size": 9, "symbol": "diamond", "line": {"color": "#fff", "width": 1}},
                "showlegend": True
            })
        result["plots"].append({
            "title": "CUSUM Chart",
            "data": cusum_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "CUSUM"}}
        })

        result["summary"] = f"CUSUM Chart Analysis\n\nTarget: {target:.4f}\nσ estimate: {sigma:.4f}\nk (slack): {k_param}\nh (decision): {h_param}\n\nUpward shift signals: {len(signals_pos)} at points {list(signals_pos[:5])}{'...' if len(signals_pos) > 5 else ''}\nDownward shift signals: {len(signals_neg)} at points {list(signals_neg[:5])}{'...' if len(signals_neg) > 5 else ''}"

    elif analysis_id == "ewma":
        """
        EWMA Chart - Exponentially Weighted Moving Average.
        Good for detecting small sustained shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))
        lambda_param = float(config.get("lambda", 0.2))  # Smoothing parameter
        L = float(config.get("L", 3))  # Control limit width

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        sigma = np.std(data, ddof=1)

        # Calculate EWMA
        ewma = np.zeros(n)
        ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target

        for i in range(1, n):
            ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]

        # Control limits (they vary with time, approaching steady state)
        factor = lambda_param / (2 - lambda_param)
        ucl = target + L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))
        lcl = target - L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))

        # Steady-state limits
        ucl_ss = target + L * sigma * np.sqrt(factor)
        lcl_ss = target - L * sigma * np.sqrt(factor)

        # OOC detection for variable-limit EWMA
        ewma_ooc = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

        ewma_chart_data = [
            {"type": "scatter", "y": ewma.tolist(), "mode": "lines+markers", "name": "EWMA", "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
            {"type": "scatter", "y": [target]*n, "mode": "lines", "name": "Target", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(ewma_chart_data, ewma, ewma_ooc)
        result["plots"].append({
            "title": "EWMA Chart",
            "data": ewma_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "EWMA"}}
        })

        result["summary"] = f"EWMA Chart Analysis\n\nTarget: {target:.4f}\nλ (smoothing): {lambda_param}\nL (sigma width): {L}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nOut-of-control points: {len(ewma_ooc)}"

    elif analysis_id == "laney_p":
        """
        Laney P' Chart - P chart adjusted for overdispersion.
        Uses sigma_z correction factor to account for extra-binomial variation.
        """
        defectives = config.get("defectives")
        sample_size_col = config.get("sample_size")

        d = df[defectives].dropna().values
        n = df[sample_size_col].dropna().values
        p = d / n
        k = len(p)

        p_bar = np.sum(d) / np.sum(n)

        # Standard p-chart z-values
        z = (p - p_bar) / np.sqrt(p_bar * (1 - p_bar) / n)

        # Moving range of z-values for sigma_z
        mr_z = np.abs(np.diff(z))
        sigma_z = np.mean(mr_z) / 1.128  # d2 for n=2

        # Laney-adjusted limits
        ucl = p_bar + 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n)
        lcl = np.maximum(0, p_bar - 3 * sigma_z * np.sqrt(p_bar * (1 - p_bar) / n))

        ooc_indices = [i for i in range(k) if p[i] > ucl[i] or p[i] < lcl[i]]

        lp_chart_data = [
            {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lp_chart_data, p, ooc_indices)
        result["plots"].append({
            "title": "Laney P' Chart",
            "data": lp_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Proportion"}}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney P' Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard P chart would give too many false alarms."

    elif analysis_id == "laney_u":
        """
        Laney U' Chart - U chart adjusted for overdispersion.
        """
        defects = config.get("defects")
        units = config.get("units")

        c = df[defects].dropna().values
        n = df[units].dropna().values
        u = c / n
        k = len(u)

        u_bar = np.sum(c) / np.sum(n)

        # Standard u-chart z-values
        z = (u - u_bar) / np.sqrt(u_bar / n)

        # Moving range of z-values for sigma_z
        mr_z = np.abs(np.diff(z))
        sigma_z = np.mean(mr_z) / 1.128

        # Laney-adjusted limits
        ucl = u_bar + 3 * sigma_z * np.sqrt(u_bar / n)
        lcl = np.maximum(0, u_bar - 3 * sigma_z * np.sqrt(u_bar / n))

        ooc_indices = [i for i in range(k) if u[i] > ucl[i] or u[i] < lcl[i]]

        lu_chart_data = [
            {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
            {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
            {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL'", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL'", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(lu_chart_data, u, ooc_indices)
        result["plots"].append({
            "title": "Laney U' Chart",
            "data": lu_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects per Unit"}}
        })

        disp = "Overdispersion" if sigma_z > 1 else "Underdispersion" if sigma_z < 1 else "None"
        result["summary"] = f"Laney U' Chart Analysis\n\nū: {u_bar:.4f}\nσz: {sigma_z:.4f} ({disp})\nSamples: {k}\n\nOut-of-control points: {len(ooc_indices)}\n\nNote: σz > 1 indicates overdispersion — standard U chart would give too many false alarms."

    elif analysis_id == "between_within":
        """
        Between/Within Capability - Nested variance components analysis.
        Separates total variation into between-subgroup and within-subgroup components.
        """
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))
        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None

        if subgroup_col:
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = [np.array(g) for g in groups if len(g) >= 2]
        k = len(groups)

        # Within-subgroup variance (pooled)
        within_vars = [np.var(g, ddof=1) for g in groups]
        sigma_within = np.sqrt(np.mean(within_vars))

        # Between-subgroup variance
        group_means = np.array([np.mean(g) for g in groups])
        grand_mean = np.mean(data)
        n_avg = np.mean([len(g) for g in groups])

        sigma_between_sq = np.var(group_means, ddof=1) - sigma_within**2 / n_avg
        sigma_between = np.sqrt(max(0, sigma_between_sq))

        # Total (overall)
        sigma_total = np.std(data, ddof=1)

        # Between/Within combined
        sigma_bw = np.sqrt(sigma_between**2 + sigma_within**2)

        summary = f"Between/Within Capability Analysis\n\nSubgroups: {k}\n\nVariance Components:\n  σ Within: {sigma_within:.4f}\n  σ Between: {sigma_between:.4f}\n  σ B/W: {sigma_bw:.4f}\n  σ Overall: {sigma_total:.4f}\n\n% of Total Variance:\n  Within: {(sigma_within**2 / sigma_total**2 * 100):.1f}%\n  Between: {(sigma_between**2 / sigma_total**2 * 100):.1f}%\n"

        if lsl is not None and usl is not None:
            # Within capability
            cp_within = (usl - lsl) / (6 * sigma_within)
            cpk_within = min((usl - grand_mean) / (3 * sigma_within), (grand_mean - lsl) / (3 * sigma_within))

            # B/W capability
            cp_bw = (usl - lsl) / (6 * sigma_bw)
            cpk_bw = min((usl - grand_mean) / (3 * sigma_bw), (grand_mean - lsl) / (3 * sigma_bw))

            # Overall capability
            pp = (usl - lsl) / (6 * sigma_total)
            ppk = min((usl - grand_mean) / (3 * sigma_total), (grand_mean - lsl) / (3 * sigma_total))

            summary += f"\nWithin Capability:\n  Cp: {cp_within:.3f}\n  Cpk: {cpk_within:.3f}\n\nBetween/Within Capability:\n  Cp (B/W): {cp_bw:.3f}\n  Cpk (B/W): {cpk_bw:.3f}\n\nOverall Capability:\n  Pp: {pp:.3f}\n  Ppk: {ppk:.3f}"

        # Variance components bar chart
        result["plots"].append({
            "title": "Variance Components",
            "data": [{
                "type": "bar",
                "x": ["Within", "Between", "B/W Combined", "Overall"],
                "y": [sigma_within, sigma_between, sigma_bw, sigma_total],
                "marker": {"color": ["#4a9f6e", "#4a90d9", "#e89547", "#d94a4a"]},
                "text": [f"{sigma_within:.4f}", f"{sigma_between:.4f}", f"{sigma_bw:.4f}", f"{sigma_total:.4f}"],
                "textposition": "outside"
            }],
            "layout": {"template": "plotly_dark", "height": 280, "yaxis": {"title": "Std Dev (σ)"}}
        })

        # Histogram with within vs overall fits
        from scipy import stats as sp_stats
        x_range = np.linspace(min(data), max(data), 200)
        hist_data = [
            {"type": "histogram", "x": data.tolist(), "name": "Data", "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}}, "histnorm": "probability density"},
            {"type": "scatter", "x": x_range.tolist(), "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_within).tolist(), "mode": "lines", "name": f"Within (σ={sigma_within:.3f})", "line": {"color": "#4a90d9", "width": 2}},
            {"type": "scatter", "x": x_range.tolist(), "y": sp_stats.norm.pdf(x_range, grand_mean, sigma_total).tolist(), "mode": "lines", "name": f"Overall (σ={sigma_total:.3f})", "line": {"color": "#d94a4a", "width": 2, "dash": "dash"}},
        ]

        layout = {"template": "plotly_dark", "height": 300, "showlegend": True, "shapes": [], "annotations": []}
        if lsl is not None:
            layout["shapes"].append({"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": lsl, "y": 1.05, "yref": "paper", "text": "LSL", "showarrow": False, "font": {"color": "#e85747"}})
        if usl is not None:
            layout["shapes"].append({"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": usl, "y": 1.05, "yref": "paper", "text": "USL", "showarrow": False, "font": {"color": "#e85747"}})

        result["plots"].append({
            "title": "Within vs Overall Distribution",
            "data": hist_data,
            "layout": layout
        })

        result["summary"] = summary

    elif analysis_id == "nonnormal_capability":
        """
        Non-Normal Capability Analysis.
        Fits Normal, Lognormal, Weibull, and Exponential distributions,
        selects best fit, and computes equivalent Pp/Ppk.
        """
        from scipy import stats as sp_stats

        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None

        pos_data = data[data > 0]  # needed for lognormal/weibull

        # Fit distributions
        fits = {}

        # Normal
        mu_n, sigma_n = sp_stats.norm.fit(data)
        fits["Normal"] = {"params": (mu_n, sigma_n), "dist": sp_stats.norm, "args": (mu_n, sigma_n),
                          "ks": sp_stats.kstest(data, "norm", args=(mu_n, sigma_n))}

        # Lognormal (needs positive data)
        if len(pos_data) > 10:
            shape_ln, loc_ln, scale_ln = sp_stats.lognorm.fit(pos_data, floc=0)
            fits["Lognormal"] = {"params": (shape_ln, 0, scale_ln), "dist": sp_stats.lognorm, "args": (shape_ln, 0, scale_ln),
                                 "ks": sp_stats.kstest(pos_data, "lognorm", args=(shape_ln, 0, scale_ln))}

        # Weibull (needs positive data)
        if len(pos_data) > 10:
            shape_w, loc_w, scale_w = sp_stats.weibull_min.fit(pos_data, floc=0)
            fits["Weibull"] = {"params": (shape_w, 0, scale_w), "dist": sp_stats.weibull_min, "args": (shape_w, 0, scale_w),
                               "ks": sp_stats.kstest(pos_data, "weibull_min", args=(shape_w, 0, scale_w))}

        # Exponential (needs positive data)
        if len(pos_data) > 10:
            loc_e, scale_e = sp_stats.expon.fit(pos_data)
            fits["Exponential"] = {"params": (loc_e, scale_e), "dist": sp_stats.expon, "args": (loc_e, scale_e),
                                   "ks": sp_stats.kstest(pos_data, "expon", args=(loc_e, scale_e))}

        # Select best fit by KS p-value (highest = best fit)
        best_name = max(fits, key=lambda k: fits[k]["ks"].pvalue)
        best = fits[best_name]
        best_dist = best["dist"]
        best_args = best["args"]

        summary = f"Non-Normal Capability Analysis\n\nBest Fit Distribution: {best_name}\n\n"
        summary += f"Distribution Fit Comparison (Anderson-Darling / KS test):\n"
        summary += f"  {'Distribution':<15} {'KS Stat':>10} {'p-value':>10} {'Fit':>6}\n"
        summary += f"  {'-'*45}\n"
        for name, info in fits.items():
            marker = " <--" if name == best_name else ""
            summary += f"  {name:<15} {info['ks'].statistic:>10.4f} {info['ks'].pvalue:>10.4f} {marker}\n"

        # Compute Pp/Ppk using the fitted distribution
        if lsl is not None and usl is not None:
            p_lsl = best_dist.cdf(lsl, *best_args)
            p_usl = 1 - best_dist.cdf(usl, *best_args)

            # Equivalent Pp from total proportion out of spec
            from scipy.stats import norm as sp_norm
            total_ppm = (p_lsl + p_usl) * 1e6
            # Z-equivalent
            z_lsl = sp_norm.ppf(1 - p_lsl) if p_lsl < 1 else 0
            z_usl = sp_norm.ppf(1 - p_usl) if p_usl < 1 else 0
            ppk_equiv = min(z_lsl, z_usl) / 3 if (z_lsl > 0 and z_usl > 0) else 0

            # Pp from spec width vs distribution spread (0.135% to 99.865%)
            q_low = best_dist.ppf(0.00135, *best_args)
            q_high = best_dist.ppf(0.99865, *best_args)
            spread_6sigma = q_high - q_low
            pp_equiv = (usl - lsl) / spread_6sigma if spread_6sigma > 0 else 0

            summary += f"\nCapability Indices ({best_name} fit):\n"
            summary += f"  Pp (equivalent): {pp_equiv:.3f}\n"
            summary += f"  Ppk (equivalent): {ppk_equiv:.3f}\n"
            summary += f"  P(below LSL): {p_lsl*100:.4f}%\n"
            summary += f"  P(above USL): {p_usl*100:.4f}%\n"
            summary += f"  Total PPM: {total_ppm:.0f}\n"

        # Histogram with best-fit overlay
        x_range = np.linspace(min(data), max(data), 200)
        pdf_vals = best_dist.pdf(x_range, *best_args)

        hist_data = [
            {"type": "histogram", "x": data.tolist(), "name": "Data",
             "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}},
             "histnorm": "probability density"},
            {"type": "scatter", "x": x_range.tolist(), "y": pdf_vals.tolist(), "mode": "lines",
             "name": f"{best_name} Fit", "line": {"color": "#4a90d9", "width": 2}},
        ]

        layout = {"template": "plotly_dark", "height": 300, "showlegend": True, "shapes": [], "annotations": []}
        if lsl is not None:
            layout["shapes"].append({"type": "line", "x0": lsl, "x1": lsl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": lsl, "y": 1.05, "yref": "paper", "text": "LSL", "showarrow": False, "font": {"color": "#e85747"}})
        if usl is not None:
            layout["shapes"].append({"type": "line", "x0": usl, "x1": usl, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash", "width": 2}})
            layout["annotations"].append({"x": usl, "y": 1.05, "yref": "paper", "text": "USL", "showarrow": False, "font": {"color": "#e85747"}})

        result["plots"].append({
            "title": f"Non-Normal Capability ({best_name} Fit)",
            "data": hist_data,
            "layout": layout
        })

        # Probability plot for best fit
        sorted_d = np.sort(pos_data if best_name != "Normal" else data)
        n_pts = len(sorted_d)
        median_ranks = (np.arange(1, n_pts+1) - 0.3) / (n_pts + 0.4)
        theoretical_q = best_dist.ppf(median_ranks, *best_args)

        result["plots"].append({
            "title": f"Probability Plot ({best_name})",
            "data": [
                {"type": "scatter", "x": theoretical_q.tolist(), "y": sorted_d.tolist(),
                 "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 5}},
                {"type": "scatter", "x": [min(theoretical_q), max(theoretical_q)],
                 "y": [min(theoretical_q), max(theoretical_q)],
                 "mode": "lines", "name": "Reference", "line": {"color": "#d94a4a", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 280, "xaxis": {"title": f"Theoretical ({best_name})"}, "yaxis": {"title": "Observed"}}
        })

        result["summary"] = summary

    elif analysis_id == "moving_average":
        """
        Moving Average (MA) Chart.
        Smooths individual observations with a moving window.
        Good for detecting sustained shifts when short-term noise is high.
        """
        measurement = config.get("measurement")
        if not measurement:
            measurement = df.select_dtypes(include=[np.number]).columns[0]
        span = int(config.get("span", 5))

        data = df[measurement].dropna().values
        n = len(data)

        x_bar = np.mean(data)
        sigma = np.std(data, ddof=1)

        # Moving averages
        ma = []
        for i in range(n):
            start = max(0, i - span + 1)
            window = data[start:i + 1]
            ma.append(np.mean(window))
        ma = np.array(ma)

        # Control limits for moving average (tighten as window fills)
        ucl_arr = []
        lcl_arr = []
        for i in range(n):
            w = min(i + 1, span)
            ucl_arr.append(x_bar + 3 * sigma / np.sqrt(w))
            lcl_arr.append(x_bar - 3 * sigma / np.sqrt(w))

        # OOC detection
        ma_ooc = [i for i in range(n) if ma[i] > ucl_arr[i] or ma[i] < lcl_arr[i]]

        ma_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "markers", "name": "Individual", "marker": {"size": 4, "color": "rgba(74,159,110,0.3)"}},
            {"type": "scatter", "y": ma.tolist(), "mode": "lines+markers", "name": f"MA({span})", "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e", "width": 2}},
            {"type": "scatter", "y": [x_bar] * n, "mode": "lines", "name": "CL", "line": {"color": "#00b894", "dash": "dash"}},
            {"type": "scatter", "y": ucl_arr, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": lcl_arr, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(ma_chart_data, ma.tolist(), ma_ooc)
        result["plots"].append({
            "title": f"Moving Average Chart (span={span})",
            "data": ma_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": measurement}}
        })

        # Steady-state limits
        ucl_ss = x_bar + 3 * sigma / np.sqrt(span)
        lcl_ss = x_bar - 3 * sigma / np.sqrt(span)

        result["summary"] = f"Moving Average Chart\n\nSpan (window size): {span}\nCenter Line: {x_bar:.4f}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nSamples: {n}\nOut-of-control points: {len(ma_ooc)}\n\nThe MA chart smooths short-term noise. With span={span}, it is effective at detecting sustained shifts of {3/np.sqrt(span):.2f}σ or larger."

    elif analysis_id == "zone_chart":
        """
        Zone Chart — assigns zone scores based on Western Electric zones.
        Signals when cumulative score reaches 8 (equivalent to a zone rule violation).
        Color-coded A/B/C zones for visual pattern detection.
        """
        measurement = config.get("measurement")
        if not measurement:
            measurement = df.select_dtypes(include=[np.number]).columns[0]

        data = df[measurement].dropna().values
        n = len(data)

        x_bar = np.mean(data)
        mr = np.abs(np.diff(data))
        sigma = np.mean(mr) / 1.128 if len(mr) > 0 else np.std(data, ddof=1)

        # Zone boundaries
        zone_1s = sigma
        zone_2s = 2 * sigma
        zone_3s = 3 * sigma

        # Zone scoring
        scores = []
        cum_scores = []
        cum_score = 0
        signals = []
        side = 0  # +1 above CL, -1 below CL

        for i in range(n):
            z = (data[i] - x_bar) / sigma if sigma > 0 else 0
            current_side = 1 if z >= 0 else -1

            # Zone score: A=8, B=4, C=2, center=0
            abs_z = abs(z)
            if abs_z >= 3:
                score = 8  # Beyond Zone A — instant signal
            elif abs_z >= 2:
                score = 4  # Zone A
            elif abs_z >= 1:
                score = 2  # Zone B
            else:
                score = 0  # Zone C (reset)
                cum_score = 0

            # Reset on side change
            if i > 0 and current_side != side:
                cum_score = 0
            side = current_side

            cum_score += score
            scores.append(score)
            cum_scores.append(cum_score)

            if cum_score >= 8:
                signals.append(i)
                cum_score = 0  # Reset after signal

        # Plot with zone bands
        zone_shapes = [
            # Zone C (green) - within 1 sigma
            {"type": "rect", "y0": x_bar - zone_1s, "y1": x_bar + zone_1s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(74,159,110,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone B upper (yellow) - 1 to 2 sigma
            {"type": "rect", "y0": x_bar + zone_1s, "y1": x_bar + zone_2s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(243,156,18,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone B lower (yellow)
            {"type": "rect", "y0": x_bar - zone_2s, "y1": x_bar - zone_1s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(243,156,18,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone A upper (red) - 2 to 3 sigma
            {"type": "rect", "y0": x_bar + zone_2s, "y1": x_bar + zone_3s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(231,76,60,0.12)", "line": {"width": 0}, "layer": "below"},
            # Zone A lower (red)
            {"type": "rect", "y0": x_bar - zone_3s, "y1": x_bar - zone_2s, "x0": 0, "x1": n - 1,
             "fillcolor": "rgba(231,76,60,0.12)", "line": {"width": 0}, "layer": "below"},
        ]

        # Color data points by zone
        colors = []
        for i in range(n):
            abs_z = abs((data[i] - x_bar) / sigma) if sigma > 0 else 0
            if abs_z >= 3:
                colors.append("#e74c3c")
            elif abs_z >= 2:
                colors.append("#f39c12")
            elif abs_z >= 1:
                colors.append("#fdcb6e")
            else:
                colors.append("#4a9f6e")

        zone_chart_data = [
            {"type": "scatter", "y": data.tolist(), "mode": "lines+markers",
             "name": measurement, "marker": {"size": 7, "color": colors}, "line": {"color": "rgba(200,200,200,0.3)", "width": 1}},
            {"type": "scatter", "y": [x_bar] * n, "mode": "lines", "name": "CL", "line": {"color": "#00b894", "width": 1.5}},
            {"type": "scatter", "y": [x_bar + zone_3s] * n, "mode": "lines", "name": "UCL (3σ)", "line": {"color": "#d63031", "dash": "dash"}},
            {"type": "scatter", "y": [x_bar - zone_3s] * n, "mode": "lines", "name": "LCL (3σ)", "line": {"color": "#d63031", "dash": "dash"}},
        ]

        # Signal markers
        if signals:
            zone_chart_data.append({
                "type": "scatter", "x": signals, "y": [data[i] for i in signals],
                "mode": "markers", "name": "Signal (score≥8)",
                "marker": {"size": 12, "color": "#e74c3c", "symbol": "diamond", "line": {"color": "white", "width": 1.5}}
            })

        result["plots"].append({
            "title": "Zone Chart",
            "data": zone_chart_data,
            "layout": {
                "template": "plotly_dark", "height": 350, "showlegend": True,
                "yaxis": {"title": measurement},
                "shapes": zone_shapes,
                "annotations": [
                    {"x": n - 1, "y": x_bar + zone_1s, "text": "C", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#4a9f6e"}},
                    {"x": n - 1, "y": x_bar + zone_2s, "text": "B", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#f39c12"}},
                    {"x": n - 1, "y": x_bar + zone_3s, "text": "A", "showarrow": False, "xanchor": "right", "font": {"size": 10, "color": "#e74c3c"}},
                ]
            }
        })

        # Cumulative score chart
        result["plots"].append({
            "title": "Cumulative Zone Score",
            "data": [
                {"type": "scatter", "y": cum_scores, "mode": "lines+markers", "name": "Cum. Score",
                 "marker": {"size": 4, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "y": [8] * n, "mode": "lines", "name": "Signal Threshold",
                 "line": {"color": "#e74c3c", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 200, "showlegend": True,
                        "yaxis": {"title": "Score"}, "xaxis": {"title": "Sample"}}
        })

        result["summary"] = f"Zone Chart Analysis\n\nCenter Line: {x_bar:.4f}\nEstimated σ: {sigma:.4f}\n\nZone Boundaries:\n  C (green): ±1σ = [{x_bar - zone_1s:.4f}, {x_bar + zone_1s:.4f}]\n  B (yellow): ±2σ = [{x_bar - zone_2s:.4f}, {x_bar + zone_2s:.4f}]\n  A (red): ±3σ = [{x_bar - zone_3s:.4f}, {x_bar + zone_3s:.4f}]\n\nScoring: A=8, B=4, C=2. Signal when cumulative ≥ 8.\nSignals detected: {len(signals)}"

    elif analysis_id == "mewma":
        """
        MEWMA — Multivariate Exponentially Weighted Moving Average.
        Extends EWMA to multiple correlated quality characteristics.
        Good for detecting small sustained multivariate shifts.
        """
        from scipy.stats import chi2

        vars_list = config.get("variables", [])
        lambda_param = float(config.get("lambda", 0.1))

        if not vars_list or len(vars_list) < 2:
            # Auto-select first 2-4 numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            vars_list = num_cols[:min(4, len(num_cols))]

        X = df[vars_list].dropna().values
        n, p = X.shape

        if n < 10 or p < 2:
            result["summary"] = "MEWMA requires at least 2 variables and 10 observations."
            return result

        # Mean vector and covariance
        mu = X.mean(axis=0)
        Sigma = np.cov(X, rowvar=False, ddof=1)

        # Regularize if near-singular
        if np.linalg.cond(Sigma) > 1e10:
            Sigma += np.eye(p) * 1e-6

        # MEWMA vectors
        Z = np.zeros((n, p))
        Z[0] = lambda_param * X[0] + (1 - lambda_param) * mu
        for i in range(1, n):
            Z[i] = lambda_param * X[i] + (1 - lambda_param) * Z[i - 1]

        # T2 statistic for each MEWMA vector
        t2_values = []
        for i in range(n):
            factor = (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param) ** (2 * (i + 1)))
            Sigma_Z = factor * Sigma
            try:
                Sigma_Z_inv = np.linalg.inv(Sigma_Z)
            except np.linalg.LinAlgError:
                Sigma_Z_inv = np.linalg.pinv(Sigma_Z)
            diff = Z[i] - mu
            t2 = float(diff @ Sigma_Z_inv @ diff)
            t2_values.append(max(0, t2))

        # UCL: chi-squared approximation (asymptotic)
        ucl = chi2.ppf(1 - 0.0027, p)  # 3-sigma equivalent ARL

        # OOC
        ooc = [i for i, t2 in enumerate(t2_values) if t2 > ucl]

        mewma_chart_data = [
            {"type": "scatter", "y": t2_values, "mode": "lines+markers", "name": "MEWMA T²",
             "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
            {"type": "scatter", "y": [ucl] * n, "mode": "lines", "name": f"UCL ({ucl:.2f})",
             "line": {"color": "#d63031", "dash": "dash"}},
        ]
        _spc_add_ooc_markers(mewma_chart_data, t2_values, ooc)

        result["plots"].append({
            "title": "MEWMA Chart",
            "data": mewma_chart_data,
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True,
                        "yaxis": {"title": "T² Statistic"}, "xaxis": {"title": "Observation"}}
        })

        # Variable contribution at OOC points
        if ooc:
            first_ooc = ooc[0]
            diff = Z[first_ooc] - mu
            contributions = diff ** 2
            total_contrib = contributions.sum()
            if total_contrib > 0:
                pct_contrib = (contributions / total_contrib * 100).tolist()
            else:
                pct_contrib = [0] * p

            result["plots"].append({
                "title": f"Variable Contribution at First OOC (obs {first_ooc})",
                "data": [{"type": "bar", "x": vars_list, "y": pct_contrib,
                          "marker": {"color": "#4a9f6e"}}],
                "layout": {"template": "plotly_dark", "height": 250,
                            "yaxis": {"title": "% Contribution"}, "xaxis": {"title": "Variable"}}
            })

        result["summary"] = f"MEWMA Chart Analysis\n\nVariables: {', '.join(vars_list)} (p={p})\nλ (smoothing): {lambda_param}\nUCL (χ²): {ucl:.4f}\n\nObservations: {n}\nOut-of-control points: {len(ooc)}\n\nNote: Smaller λ increases sensitivity to small sustained shifts but also increases false alarm rate. Typical range: 0.05–0.25."

    return result


def run_visualization(df, analysis_id, config):
    """Create visualizations."""
    result = {"plots": [], "summary": ""}

    # SVEND theme colors
    theme_colors = ['#4a9f6e', '#4a9f6e', '#e89547', '#9f4a4a', '#e8c547', '#7a6a9a']

    if analysis_id == "histogram":
        var = config.get("var")
        bins = int(config.get("bins", 20))
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "histogram",
                    "x": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "opacity": 0.7,
                    "nbinsx": bins,
                    "marker": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Histogram of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "barmode": "overlay", "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Histogram of {var}",
                "data": [{"type": "histogram", "x": df[var].dropna().tolist(), "nbinsx": bins, "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "boxplot":
        var = config.get("var")
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            # Create separate box for each group with different colors
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "box",
                    "y": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "marker": {"color": theme_colors[i % len(theme_colors)]},
                    "line": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Box Plot of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Box Plot of {var}",
                "data": [{"type": "box", "y": df[var].dropna().tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}, "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "scatter":
        x_var = config.get("x")
        y_var = config.get("y")
        color_var = config.get("color")
        trendline = config.get("trendline", False)

        # Distinct colors for groups (fill, border) - semi-transparent fill with solid border
        group_colors = [
            ('rgba(74, 159, 110, 0.5)', '#4a9f6e'),   # Green
            ('rgba(232, 149, 71, 0.5)', '#e89547'),   # Orange
            ('rgba(159, 74, 74, 0.5)', '#9f4a4a'),    # Red
            ('rgba(71, 165, 232, 0.5)', '#47a5e8'),   # Blue
            ('rgba(232, 197, 71, 0.5)', '#e8c547'),   # Yellow
            ('rgba(122, 106, 154, 0.5)', '#7a6a9a'), # Purple
        ]

        data = []

        if color_var and color_var != "" and color_var != "None":
            # Create separate trace for each group
            groups = df[color_var].dropna().unique()
            for i, group in enumerate(groups):
                fill_color, border_color = group_colors[i % len(group_colors)]
                mask = df[color_var] == group
                data.append({
                    "type": "scatter",
                    "x": df.loc[mask, x_var].tolist(),
                    "y": df.loc[mask, y_var].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": fill_color,
                        "size": 8,
                        "line": {"color": border_color, "width": 1.5}
                    },
                    "name": str(group)
                })
        else:
            data.append({
                "type": "scatter",
                "x": df[x_var].tolist(),
                "y": df[y_var].tolist(),
                "mode": "markers",
                "marker": {
                    "color": "rgba(74, 159, 110, 0.5)",
                    "size": 8,
                    "line": {"color": "#4a9f6e", "width": 1.5}
                },
                "name": y_var
            })

        if trendline:
            import numpy as np
            x = df[x_var].dropna()
            y = df[y_var].loc[x.index].dropna()
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                data.append({
                    "type": "scatter",
                    "x": [float(x.min()), float(x.max())],
                    "y": [float(p(x.min())), float(p(x.max()))],
                    "mode": "lines",
                    "line": {"color": "#e8c547", "dash": "dash"},
                    "name": "Trendline"
                })

        title = f"{y_var} vs {x_var}"
        if color_var and color_var != "" and color_var != "None":
            title += f" (by {color_var})"

        result["plots"].append({
            "title": title,
            "data": data,
            "layout": {
                "height": 300,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
                "showlegend": len(data) > 1
            }
        })

    elif analysis_id == "heatmap":
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()

        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{"type": "heatmap", "z": corr.values.tolist(), "x": numeric_cols, "y": numeric_cols, "colorscale": "RdBu", "zmid": 0}],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "pareto":
        category = config.get("category") or config.get("var")
        value_col = config.get("value")

        if value_col and value_col != "":
            # Sum values by category
            counts = df.groupby(category)[value_col].sum().sort_values(ascending=False)
        else:
            # Count occurrences
            counts = df[category].value_counts()

        cumulative = counts.cumsum() / counts.sum() * 100

        result["plots"].append({
            "title": f"Pareto Chart - {category}",
            "data": [
                {"type": "bar", "x": [str(x) for x in counts.index.tolist()], "y": counts.values.tolist(), "name": "Count", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "x": [str(x) for x in counts.index.tolist()], "y": cumulative.values.tolist(), "name": "Cumulative %", "yaxis": "y2", "line": {"color": "#fdcb6e"}}
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "yaxis2": {"overlaying": "y", "side": "right", "range": [0, 100], "title": "Cumulative %"}
            }
        })

        # Find 80% cutoff
        cutoff_idx = (cumulative >= 80).idxmax() if (cumulative >= 80).any() else counts.index[-1]
        vital_few = counts.loc[:cutoff_idx]
        result["summary"] = f"Pareto Analysis\n\nTotal categories: {len(counts)}\nVital few (80%): {len(vital_few)} categories\n\nTop contributors:\n"
        for cat, val in counts.head(5).items():
            pct = val / counts.sum() * 100
            result["summary"] += f"  {cat}: {val} ({pct:.1f}%)\n"

    elif analysis_id == "matrix":
        import numpy as np
        vars_list = config.get("vars", [])
        color_var = config.get("color")

        if len(vars_list) < 2:
            result["summary"] = "Please select at least 2 variables for matrix plot."
            return result

        # Create scatter matrix
        n_vars = len(vars_list)
        fig_data = []

        for i, y_var in enumerate(vars_list):
            for j, x_var in enumerate(vars_list):
                row = n_vars - i
                col = j + 1

                if i == j:
                    # Diagonal: histogram
                    fig_data.append({
                        "type": "histogram",
                        "x": df[x_var].dropna().tolist(),
                        "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })
                else:
                    # Off-diagonal: scatter
                    fig_data.append({
                        "type": "scatter",
                        "x": df[x_var].tolist(),
                        "y": df[y_var].tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 3},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })

        # Build layout with subplots
        layout = {
            "template": "plotly_dark",
            "height": 100 + n_vars * 120,
            "showlegend": False,
        }

        # Create axis layout
        for i in range(n_vars):
            col = i + 1
            row = n_vars - i
            x_key = f"xaxis{col if col > 1 else ''}"
            y_key = f"yaxis{row if row > 1 else ''}"

            layout[x_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[i] if row == 1 else "",
                "showticklabels": row == 1
            }
            layout[y_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[n_vars - 1 - i] if col == 1 else "",
                "showticklabels": col == 1
            }

        result["plots"].append({
            "title": "Matrix Plot",
            "data": fig_data,
            "layout": layout
        })

    elif analysis_id == "timeseries":
        import numpy as np
        x_col = config.get("x")
        y_cols = config.get("y", [])
        show_markers = config.get("markers", False)

        if isinstance(y_cols, str):
            y_cols = [y_cols]

        traces = []
        for i, y_col in enumerate(y_cols):
            trace = {
                "type": "scatter",
                "x": df[x_col].astype(str).tolist(),
                "y": df[y_col].tolist(),
                "mode": "lines+markers" if show_markers else "lines",
                "name": y_col,
                "line": {"color": theme_colors[i % len(theme_colors)]}
            }
            traces.append(trace)

        result["plots"].append({
            "title": f"Time Series: {', '.join(y_cols)}",
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": x_col},
                "yaxis": {"title": "Value"},
                "showlegend": len(y_cols) > 1
            }
        })

    elif analysis_id == "probability":
        import numpy as np
        from scipy import stats

        var = config.get("var")
        dist = config.get("dist", "norm")

        x = df[var].dropna().values
        x_sorted = np.sort(x)
        n = len(x_sorted)

        # Calculate plotting positions (Hazen)
        pp = (np.arange(1, n + 1) - 0.5) / n

        # Get theoretical quantiles for chosen distribution
        if dist == "norm":
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"
        elif dist == "lognorm":
            theoretical = stats.lognorm.ppf(pp, s=1)
            dist_name = "Lognormal"
            x_sorted = np.log(x_sorted[x_sorted > 0])
            theoretical = stats.norm.ppf(pp[:len(x_sorted)])
        elif dist == "expon":
            theoretical = stats.expon.ppf(pp)
            dist_name = "Exponential"
        elif dist == "weibull":
            # Weibull with shape=2
            theoretical = stats.weibull_min.ppf(pp, c=2)
            dist_name = "Weibull"
        else:
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"

        # Fit line
        slope, intercept = np.polyfit(theoretical, x_sorted, 1)
        fit_line = slope * theoretical + intercept

        result["plots"].append({
            "title": f"Probability Plot ({dist_name}): {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": x_sorted.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": fit_line.tolist(),
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "Fit Line"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": f"Theoretical Quantiles ({dist_name})"},
                "yaxis": {"title": var}
            }
        })

        # Anderson-Darling test for normality
        if dist == "norm":
            ad_stat = stats.anderson(x)
            result["summary"] = f"Probability Plot ({dist_name})\n\nAnderson-Darling: {ad_stat.statistic:.4f}\nCritical values (15%, 10%, 5%, 2.5%, 1%):\n{ad_stat.critical_values}"

    return result


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def upload_data(request):
    """
    Upload and parse a data file for the Analysis Workbench.

    Returns columns with dtypes and a preview of the data.
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file provided"}, status=400)

    file = request.FILES["file"]
    filename = file.name.lower()

    try:
        import pandas as pd
        import numpy as np

        # Parse the file - try to detect actual format
        df = None
        parse_errors = []

        # Try based on extension first
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except Exception as e:
                parse_errors.append(f"CSV: {e}")

        elif filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except Exception as e:
                parse_errors.append(f"XLSX: {e}")
                # Maybe it's actually a CSV with wrong extension
                file.seek(0)
                try:
                    df = pd.read_csv(file)
                    parse_errors.append("(Parsed as CSV)")
                except:
                    pass

        elif filename.endswith('.xls'):
            try:
                df = pd.read_excel(file, engine='xlrd')
            except Exception as e:
                parse_errors.append(f"XLS: {e}")
                # Maybe it's a CSV or XLSX with wrong extension
                file.seek(0)
                try:
                    df = pd.read_csv(file)
                    parse_errors.append("(Parsed as CSV)")
                except:
                    file.seek(0)
                    try:
                        df = pd.read_excel(file, engine='openpyxl')
                        parse_errors.append("(Parsed as XLSX)")
                    except:
                        pass
        else:
            # Unknown extension - try all formats
            for parser, name in [(lambda f: pd.read_csv(f), 'CSV'),
                                  (lambda f: pd.read_excel(f, engine='openpyxl'), 'XLSX'),
                                  (lambda f: pd.read_excel(f, engine='xlrd'), 'XLS')]:
                try:
                    file.seek(0)
                    df = parser(file)
                    break
                except:
                    continue

        if df is None:
            return JsonResponse({"error": f"Could not parse file. Tried: {'; '.join(parse_errors) or 'all formats'}"}, status=400)

        # Save to temp storage for session use
        data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{data_id}.csv"
            df.to_csv(data_path, index=False)
        except Exception as save_err:
            # Fall back to temp directory if MEDIA_ROOT not configured
            logger.warning(f"Could not save to MEDIA_ROOT: {save_err}, using temp")
            data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{data_id}.csv"
            df.to_csv(data_path, index=False)

        # Determine column types
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"

            columns.append({
                "name": col,
                "dtype": col_type,
            })

        # Generate preview (first 100 rows)
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")

        logger.info(f"Data uploaded: {request.user.username} - {file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

        # Preload LLM in background so it's ready when user asks questions
        _preload_llm_background()

        return JsonResponse({
            "id": data_id,
            "filename": file.name,
            "rows": df.shape[0],
            "columns": columns,
            "preview": preview,
        })

    except Exception as e:
        logger.error(f"Data upload error: {e}")
        return JsonResponse({"error": f"Failed to parse file: {str(e)}"}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def execute_code(request):
    """
    Execute Python code in a sandboxed environment.

    Request body:
    {
        "code": "...",
        "data_id": "..."
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    code = body.get("code", "")
    data_id = body.get("data_id")

    if not code.strip():
        return JsonResponse({"error": "No code provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO
        import sys

        # Load data if provided
        df = None
        if data_id:
            from files.models import UploadedFile
            try:
                file_record = UploadedFile.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(file_record.file.path) if file_record.file.path.endswith('.csv') else pd.read_excel(file_record.file.path)
            except UploadedFile.DoesNotExist:
                pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Import additional libraries for the sandbox
        import scipy
        import scipy.stats
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import random
        import math
        import statistics

        # Execute in namespace with common data science libraries
        namespace = {
            "df": df,
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "stats": scipy.stats,
            "plt": plt,
            "random": random,
            "math": math,
            "statistics": statistics,
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "zip": zip,
                "enumerate": enumerate,
                "map": map,
                "filter": filter,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "type": type,
                "getattr": getattr,
                "setattr": setattr,
                "hasattr": hasattr,
                "__import__": __import__,
            }
        }

        exec(code, namespace)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Check if there are any plot objects to return
        plots = []
        if "fig" in namespace:
            # Assume it's a plotly figure
            try:
                fig = namespace["fig"]
                plots.append({
                    "title": "Output",
                    "data": fig.data,
                    "layout": fig.layout
                })
            except:
                pass

        return JsonResponse({
            "output": output or "Code executed successfully",
            "plots": plots,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def generate_code(request):
    """
    Generate Python code from natural language.

    Uses Qwen Coder by default, or Anthropic models for Enterprise users.

    Request body:
    {
        "prompt": "Run a Monte Carlo simulation...",
        "model": "qwen" | "sonnet" | "opus" | "haiku",
        "context": {
            "hypothesis": "High temperature causes defects",
            "mechanism": "..."
        }
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "Prompt is required"}, status=400)

    model = body.get("model", "qwen")
    context = body.get("context", {})

    # Build context prefix
    context_prefix = ""
    if context.get("hypothesis"):
        context_prefix = f"Context: Testing hypothesis '{context['hypothesis']}'\n"
        if context.get("mechanism"):
            context_prefix += f"Mechanism: {context['mechanism']}\n"

    # Check if using Anthropic models (Enterprise only)
    if model in ("sonnet", "opus", "haiku"):
        # Check enterprise access
        if not hasattr(request.user, 'subscription') or request.user.subscription.plan != 'enterprise':
            return JsonResponse({"error": "Anthropic models require Enterprise subscription"}, status=403)

        try:
            import anthropic

            system_prompt = """You are an expert Python code generator for data science and simulation.
Generate clean, executable Python code based on the user's request.

Rules:
- Only output Python code, no explanations or markdown
- Use numpy, pandas, scipy, matplotlib as needed
- Include print() statements for results
- Keep code concise but complete
- Add brief comments for clarity

Available libraries: numpy (np), pandas (pd), scipy, matplotlib (plt), random, math, statistics"""

            model_map = {
                "opus": "claude-opus-4-20250514",
                "sonnet": "claude-sonnet-4-20250514",
                "haiku": "claude-3-5-haiku-20241022",
            }

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model_map.get(model, "claude-sonnet-4-20250514"),
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": f"{context_prefix}Generate Python code for: {prompt}"}]
            )

            code = response.content[0].text

            # Extract code from markdown if present
            import re
            if "```python" in code:
                match = re.search(r"```python\n?([\s\S]*?)```", code)
                if match:
                    code = match.group(1)
            elif "```" in code:
                match = re.search(r"```\n?([\s\S]*?)```", code)
                if match:
                    code = match.group(1)

            return JsonResponse({"code": code.strip(), "model": model})

        except Exception as e:
            logger.exception(f"Anthropic code generation error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    # Default: Use Qwen Coder
    code_prompt = f"""{context_prefix}Generate Python code for: {prompt}

Rules:
- Only output Python code, no explanations
- Use numpy, pandas, scipy as needed
- Include print() statements for results
- Keep code concise but complete"""

    try:
        # Get Qwen Coder LLM
        from . import views as agent_views
        llm = agent_views.get_coder_llm()

        if llm is None:
            # Fallback: return a template
            code = f'''import numpy as np
import pandas as pd

# TODO: Implement - {prompt}
# Qwen Coder is loading, please try again in a moment

print("Code generation requires Qwen Coder - loading in background...")
'''
            return JsonResponse({"code": code, "note": "Qwen Coder is loading, try again shortly"})

        # Generate code with Qwen
        code = llm.generate(code_prompt, max_tokens=1024, temperature=0.2)

        # Extract code from markdown if present
        import re
        if "```python" in code:
            match = re.search(r"```python\n?([\s\S]*?)```", code)
            if match:
                code = match.group(1)
        elif "```" in code:
            match = re.search(r"```\n?([\s\S]*?)```", code)
            if match:
                code = match.group(1)

        # Clean up - remove the prompt echo if present
        if code.startswith(code_prompt[:50]):
            code = code[len(code_prompt):].strip()

        return JsonResponse({"code": code.strip(), "model": "qwen"})

    except Exception as e:
        logger.exception(f"Code generation error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_enterprise
def analyst_assistant(request):
    """
    AI assistant for data analysis questions.

    Supports multiple agent types:
    - analyst: Uses Qwen LLM to answer questions about loaded data
    - researcher: Searches the web for domain knowledge and scientific context
    - writer: Generates downloadable documents/reports

    Requires Enterprise tier.
    """
    try:
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        message = body.get("message", "")
        agent_type = body.get("agent_type", "analyst")
        selected_model = body.get("model", "default")
        context = body.get("context", {})
        data_id = context.get("data_id")
        columns = context.get("columns", [])
        data_preview = context.get("data_preview", [])

        # Load data if available
        df = None
        if data_id:
            try:
                import pandas as pd
                from io import StringIO

                if data_id.startswith("data_"):
                    data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                    else:
                        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                        data_path = data_dir / f"{data_id}.csv"
                        if data_path.exists():
                            df = pd.read_csv(data_path)
                else:
                    from .models import TriageResult
                    try:
                        triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                        df = pd.read_csv(StringIO(triage_result.cleaned_csv))
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Could not load data for analyst: {e}")

        # Get session history from context
        session_history = context.get("session_history", [])

        # Get shared LLM (non-blocking check)
        llm = None
        llm_loading = False
        cuda_available = False

        # Quick CUDA check
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.warning("CUDA not available - using keyword fallback")
        except ImportError:
            logger.warning("PyTorch not available")

        if cuda_available:
            try:
                from . import views as agent_views

                # Check if LLM is already loaded - don't block on first load
                if agent_views._shared_llm_loaded:
                    llm = agent_views._shared_llm
                    logger.info(f"LLM already loaded: type={type(llm).__name__ if llm else 'None'}")
                else:
                    # LLM not loaded yet - trigger background loading
                    llm_loading = True
                    logger.info("LLM not yet loaded - triggering background load")
                    import threading
                    def load_llm_background():
                        try:
                            agent_views.get_shared_llm()
                            logger.info("Background LLM load completed")
                        except Exception as e:
                            logger.error(f"Background LLM load failed: {e}")
                    threading.Thread(target=load_llm_background, daemon=True).start()

            except Exception as e:
                logger.error(f"Failed to check LLM status: {e}")
                import traceback
                traceback.print_exc()

        # Handle different agent types
        if agent_type == "researcher":
            # Researcher agent: web search for domain knowledge
            response, sources = generate_researcher_response(message, df, columns, data_preview)
            log_agent_action(request.user, "researcher", "research", success=True)
            return JsonResponse({"response": response, "sources": sources})

        elif agent_type == "writer":
            # Writer agent: generate downloadable documents
            response, document, filename = generate_writer_response(message, df, columns, session_history)
            log_agent_action(request.user, "writer", "write", success=True)
            return JsonResponse({"response": response, "document": document, "filename": filename})

        # Default: Analyst agent
        # Check for enterprise model selection (Opus/Sonnet/Haiku)
        if selected_model in ("opus", "sonnet", "haiku"):
            try:
                response = generate_anthropic_response(selected_model, message, df, columns, session_history)
                log_agent_action(request.user, "analyst", "question", success=True)
                return JsonResponse({"response": response, "model": selected_model})
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                response = f"Model error: {str(e)}"
                return JsonResponse({"response": response})

        if llm_loading:
            # LLM is loading in background - provide helpful response with keyword fallback
            logger.info("LLM loading in background, using enhanced fallback")
            response = generate_analyst_response(message.lower(), df, columns)
            response = "*(Qwen LLM is loading in the background - using quick response mode)*\n\n" + response
        elif llm is None:
            logger.info("Using keyword-based fallback response")
            response = generate_analyst_response(message.lower(), df, columns)
        else:
            logger.info(f"Using LLM for response, message: {message[:50]}...")
            try:
                response = generate_llm_response(llm, message, df, columns, session_history)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                import traceback
                traceback.print_exc()
                response = f"LLM error: {str(e)}"

        log_agent_action(request.user, "analyst", "question", success=True)
        return JsonResponse({"response": response})

    except Exception as e:
        logger.exception(f"Analyst assistant error: {e}")
        return JsonResponse({"response": f"Error: {str(e)}"})


def generate_anthropic_response(model, message, df, columns, session_history=None):
    """Generate analyst response using Anthropic API (Opus/Sonnet/Haiku)."""
    import anthropic
    import numpy as np

    model_map = {
        "opus": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-3-5-haiku-20241022",
    }

    # Build data context
    if df is None:
        data_context = "No dataset loaded."
    else:
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns
Numeric columns: {', '.join(numeric_cols[:10])}
Categorical columns: {', '.join(cat_cols[:10])}

Summary:
{df.describe().to_string()}

Sample data (first 5 rows):
{df.head().to_string()}
"""

    system_prompt = f"""You are an expert data analyst assistant in a Decision Science Workbench.
You help users understand their data, suggest analyses, and interpret results.

Current data context:
{data_context}

Be concise but thorough. Use markdown formatting for clarity."""

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model_map.get(model, "claude-sonnet-4-20250514"),
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": message}]
    )

    return response.content[0].text


def generate_llm_response(llm, message, df, columns, session_history=None):
    """Generate response using Qwen LLM as a lab assistant."""
    import numpy as np

    # Build data context
    if df is None:
        return "Please load a dataset first. Click the folder icon in the toolbar to load data from Triage or upload a CSV file."

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build correlation matrix for numeric columns (helps answer relationship questions)
    corr_info = ""
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            # Find top correlations
            corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.3:  # Only notable correlations
                        corr_pairs.append((col1, col2, corr_val))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if corr_pairs:
                corr_info = "\n\nNOTABLE CORRELATIONS:\n"
                for col1, col2, r in corr_pairs[:10]:
                    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.5 else "weak"
                    direction = "positive" if r > 0 else "negative"
                    corr_info += f"- {col1} ↔ {col2}: r={r:.3f} ({strength} {direction})\n"
        except Exception:
            pass

    # Build data summary for context
    data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns

Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:15])}
Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:15])}

Summary statistics:
{df.describe().to_string()}{corr_info}

Sample data (first 3 rows):
{df.head(3).to_string()}
"""

    # Build session history context
    session_context = ""
    if session_history:
        session_context = "\n\nSESSION HISTORY (analyses run this session):\n"
        for item in session_history[-10:]:
            session_context += f"- {item.get('type', 'unknown')}: {item.get('name', '')} - {item.get('summary', '')[:200]}\n"

    # Build prompt - lab assistant persona
    prompt = f"""You are a lab assistant helping a scientist analyze their data. You're knowledgeable, helpful, and speak like a colleague - not a generic chatbot.

Your role:
- Answer questions directly and specifically about THIS data
- Explain patterns, anomalies, or relationships you see
- Suggest appropriate statistical tests with reasoning
- Reference specific column names and values
- Be concise but insightful

Available analysis tools:
- Stat: Descriptive Stats, t-tests, ANOVA, Regression, Correlation, Normality, Chi-Square
- ML: Classification (RF, XGBoost, SVM), Clustering (K-Means, DBSCAN), PCA
- SPC: Control charts (I-MR, Xbar-R), Capability Analysis
- Graph: Histogram, Boxplot, Scatter, Matrix Plot, Time Series, Pareto

DATA:
{data_context}{session_context}

USER: {message}

Respond as a helpful lab assistant. Be specific to this data. If they ask about relationships, look at the correlations. If they ask what to analyze, give concrete suggestions based on the actual columns. Keep response under 250 words."""

    try:
        import concurrent.futures
        # Use a thread with timeout to prevent hanging
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.generate, prompt, max_tokens=500, temperature=0.7)
            try:
                response = future.result(timeout=30)  # 30 second timeout
                return response
            except concurrent.futures.TimeoutError:
                logger.warning("LLM generation timed out after 30s")
                return generate_analyst_response(message.lower(), df, columns) + "\n\n*(Response generated via quick mode due to timeout)*"
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        # Fallback
        return generate_analyst_response(message.lower(), df, columns)


def generate_analyst_response(message, df, columns):
    """Generate helpful response based on the question and data."""
    import numpy as np

    # No data loaded
    if df is None or len(columns) == 0:
        return "Please load a dataset first. Click the folder icon in the toolbar to load data from Triage or upload a CSV file."

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Keywords for different intents
    if any(w in message for w in ['describe', 'summary', 'overview', 'tell me about', 'what is']):
        summary = f"Your dataset has {n_rows:,} rows and {n_cols} columns.\n\n"
        summary += f"**Numeric columns ({len(numeric_cols)}):** {', '.join(numeric_cols[:5])}"
        if len(numeric_cols) > 5:
            summary += f" (+{len(numeric_cols)-5} more)"
        summary += f"\n\n**Categorical columns ({len(cat_cols)}):** {', '.join(cat_cols[:5])}"
        if len(cat_cols) > 5:
            summary += f" (+{len(cat_cols)-5} more)"

        if numeric_cols:
            summary += "\n\n**Quick stats for numeric columns:**\n"
            for col in numeric_cols[:3]:
                summary += f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n"

        summary += "\n\nUse **Stat > Descriptive Statistics** for detailed analysis."
        return summary

    if any(w in message for w in ['correlation', 'relationship', 'related', 'correlated']):
        if len(numeric_cols) < 2:
            return "You need at least 2 numeric columns to analyze correlations."

        corr_matrix = df[numeric_cols].corr()
        # Find strongest correlations
        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        response = "**Correlation Analysis:**\n\n"
        response += "Strongest relationships:\n"
        for col1, col2, corr in pairs[:5]:
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            direction = "positive" if corr > 0 else "negative"
            response += f"- {col1} & {col2}: r={corr:.3f} ({strength} {direction})\n"

        response += "\n\nTo visualize, use **Graph > Scatterplot** with these variable pairs."
        return response

    if any(w in message for w in ['predict', 'forecast', 'ml', 'machine learning', 'model']):
        response = "**ML Recommendations:**\n\n"

        if cat_cols:
            response += f"For **classification** (predicting categories like '{cat_cols[0]}'), use:\n"
            response += "- **ML > Classification** with Random Forest or XGBoost\n\n"

        if numeric_cols:
            response += f"For **regression** (predicting values like '{numeric_cols[0]}'), use:\n"
            response += "- **ML > Regression** with Random Forest or XGBoost\n\n"

        response += "For **finding patterns/groups**, use:\n"
        response += "- **ML > Clustering** to discover natural groupings\n"
        response += "- **ML > PCA** to reduce dimensions and visualize\n\n"

        response += "**Tip:** Select your target variable and features in the dialog. Start with Random Forest - it works well without tuning."
        return response

    if any(w in message for w in ['outlier', 'anomaly', 'unusual', 'extreme']):
        if not numeric_cols:
            return "No numeric columns found for outlier detection."

        response = "**Outlier Detection:**\n\n"
        for col in numeric_cols[:4]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            if len(outliers) > 0:
                response += f"- **{col}**: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)\n"

        response += "\n\nUse **Graph > Boxplot** to visualize outliers, or use **Triage** to clean them."
        return response

    if any(w in message for w in ['missing', 'null', 'empty', 'na']):
        response = "**Missing Data Analysis:**\n\n"
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            response += "No missing values found in your dataset."
        else:
            response += "Columns with missing values:\n"
            for col, count in missing.head(10).items():
                response += f"- **{col}**: {count} ({count/len(df)*100:.1f}%)\n"

            response += "\n\nUse **Triage** to handle missing values (imputation, removal)."
        return response

    if any(w in message for w in ['compare', 'difference', 'group', 'between']):
        if not cat_cols:
            return "No categorical columns found for group comparisons. Use a categorical variable to split your data."

        response = "**Group Comparisons:**\n\n"
        response += f"You can compare groups using '{cat_cols[0]}' or other categorical variables.\n\n"
        response += "**Recommended analyses:**\n"
        response += "- **Stat > Two-Sample t** - Compare means between 2 groups\n"
        response += "- **Stat > ANOVA** - Compare means across multiple groups\n"
        response += "- **Graph > Boxplot** with 'Group by' - Visualize differences\n"
        response += "- **Graph > Histogram** with 'Group by' - Compare distributions"
        return response

    if any(w in message for w in ['distribution', 'normal', 'spread', 'histogram']):
        if not numeric_cols:
            return "No numeric columns found for distribution analysis."

        response = "**Distribution Analysis:**\n\n"
        for col in numeric_cols[:3]:
            skew = df[col].skew()
            skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately symmetric"
            response += f"- **{col}**: {skew_desc} (skewness={skew:.2f})\n"

        response += "\n\n**To visualize:**\n"
        response += "- **Graph > Histogram** for distribution shape\n"
        response += "- **Stat > Normality Test** to test for normality"
        return response

    # Default response
    response = f"I can help you analyze your dataset ({n_rows:,} rows, {n_cols} columns).\n\n"
    response += "**Try asking about:**\n"
    response += "- \"Describe my data\" - Get an overview\n"
    response += "- \"Find correlations\" - Discover relationships\n"
    response += "- \"Check for outliers\" - Find unusual values\n"
    response += "- \"Missing data\" - Analyze gaps\n"
    response += "- \"How to predict X\" - ML recommendations\n"
    response += "- \"Compare groups\" - Statistical comparisons\n\n"
    response += "Or use the **Stat**, **ML**, and **Graph** menus above."
    return response


def generate_researcher_response(message, df, columns, data_preview):
    """
    Researcher agent: searches the web for domain knowledge and scientific context.
    Uses ddgs library for real web search results with intelligent query construction.
    """
    import re

    # Build context about the data for better search
    data_context = ""
    if df is not None:
        data_context = f"Dataset has {len(df)} rows with columns: {', '.join(columns[:10])}"
        if data_preview:
            sample_vals = []
            for col in columns[:5]:
                if col in df.columns:
                    vals = df[col].dropna().head(3).tolist()
                    sample_vals.append(f"{col}: {vals}")
            data_context += f"\nSample values: {'; '.join(sample_vals[:3])}"

    # Extract key technical terms from the question
    # Remove common question words
    query_clean = re.sub(r'\b(can you|could you|please|research|tell me about|what is the|why do|why does|how does|explain|relationship between|correlation between)\b', '', message.lower())
    query_clean = query_clean.strip()

    # Extract potential chemical/technical terms (capitalized words or known patterns)
    technical_terms = re.findall(r'\b[A-Za-z]{4,}\b', message)
    # Filter to likely technical terms (not common words)
    common_words = {'what', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'about', 'which', 'when', 'there', 'would', 'could', 'should', 'between', 'relationship', 'correlation', 'drinking', 'water', 'samples'}
    technical_terms = [t.lower() for t in technical_terms if t.lower() not in common_words]

    sources = []
    search_results = []

    # Try ddgs library
    try:
        from ddgs import DDGS
        ddgs = DDGS()

        # Build multiple targeted searches
        searches = []

        # If we found technical terms, search for their relationship
        if len(technical_terms) >= 2:
            # Search for the specific interaction/relationship
            term_combo = ' '.join(technical_terms[:3])
            searches.append(f'"{technical_terms[0]}" "{technical_terms[1]}" correlation co-occurrence site:epa.gov OR site:pubmed OR site:ncbi.nlm.nih.gov')
            searches.append(f'{term_combo} water contamination research')
            searches.append(f'"{technical_terms[0]}" "{technical_terms[1]}" drinking water study')
        else:
            # Fallback to cleaned query
            searches.append(f'{query_clean} EPA research')
            searches.append(f'{query_clean} scientific study')

        seen_urls = set()
        for search_query in searches:
            try:
                results = list(ddgs.text(search_query, max_results=3))
                for r in results:
                    url = r.get('href', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        search_results.append({
                            'title': r.get('title', 'Result'),
                            'snippet': r.get('body', ''),
                            'url': url
                        })
                        sources.append({'title': r.get('title', 'Source')[:60], 'url': url})
            except Exception as e:
                logger.warning(f"Search query failed: {e}")
                continue

    except ImportError:
        logger.warning("ddgs not installed, trying fallback")
        # Fallback to requests-based search
        try:
            import requests
            import urllib.parse
            from bs4 import BeautifulSoup

            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_clean + ' scientific')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(search_url, headers=headers, timeout=15)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for result in soup.select('.result')[:5]:
                    title_el = result.select_one('.result__title')
                    snippet_el = result.select_one('.result__snippet')
                    link_el = result.select_one('.result__url')

                    if title_el and snippet_el:
                        title = title_el.get_text(strip=True)
                        snippet = snippet_el.get_text(strip=True)
                        url = link_el.get('href', '') if link_el else ''

                        search_results.append({
                            'title': title,
                            'snippet': snippet,
                            'url': url
                        })
                        if url:
                            sources.append({'title': title[:60], 'url': url})

        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")

    # Use LLM to synthesize search results into intelligent response
    if search_results:
        # Try to use shared LLM for synthesis
        llm = None
        try:
            from . import views as agent_views
            if agent_views._shared_llm_loaded:
                llm = agent_views._shared_llm
        except Exception:
            pass

        if llm:
            # Build context for LLM synthesis
            search_context = "\n\n".join([
                f"Source: {r['title']}\n{r['snippet']}"
                for r in search_results[:5] if r.get('snippet')
            ])

            synthesis_prompt = f"""You are a research assistant helping analyze data. The user asked: "{message}"

Here is what web research found:

{search_context}

User's data context: {data_context}

Based on the search results and the user's data, provide a helpful synthesis that:
1. Directly answers their question about the relationship/topic
2. Explains any scientific mechanisms or reasons
3. Relates the findings to their specific data columns if relevant
4. Notes any important caveats or additional considerations

Keep response under 250 words. Be specific and scientific, not generic."""

            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm.generate, synthesis_prompt, max_tokens=400, temperature=0.7)
                    synthesized = future.result(timeout=30)

                response = f"**Research Analysis:** *{message}*\n\n"
                response += synthesized + "\n\n"

                if sources:
                    response += "**Sources:**\n"
                    for src in sources[:5]:
                        if src.get('url'):
                            response += f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"

                return response, sources

            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                # Fall through to basic response

        # Fallback: basic response without LLM
        response = f"**Research findings for:** *{message}*\n\n"

        for i, result in enumerate(search_results[:4], 1):
            if result['snippet']:
                response += f"**{result.get('title', f'Finding {i}')}**\n"
                response += f"{result['snippet']}\n\n"

        if data_context:
            response += f"---\n**Your data context:** {data_context}\n\n"

        if sources:
            response += "**Sources:**\n"
            for src in sources[:5]:
                if src.get('url'):
                    response += f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"
    else:
        response = f"I searched for information about *{message}* but didn't find specific results.\n\n"
        response += "**Suggestions:**\n"
        response += "- Try rephrasing your question with more specific terms\n"
        response += "- Ask about specific chemicals, processes, or scientific concepts\n"
        response += "- Use the Analyst agent for data-specific questions\n"

        if data_context:
            response += f"\n**Your data context:** {data_context}"

    return response, sources


def generate_writer_response(message, df, columns, session_history):
    """
    Writer agent: generates downloadable documents/reports using LLM.
    Creates intelligent markdown documents with analysis and insights.
    """
    import numpy as np
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    if df is None:
        document = "# Analysis Report\n\nNo data loaded. Please load a dataset first.\n"
        return "Please load a dataset first to generate a document.", document, filename

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build data context for LLM
    stats_summary = df.describe().to_string() if numeric_cols else "No numeric columns"

    # Session history context
    history_text = ""
    if session_history:
        history_text = "\n".join([
            f"- {item.get('name', 'Analysis')}: {item.get('summary', '')[:150]}"
            for item in session_history[-10:]
        ])

    # Try to use LLM for intelligent document generation
    llm = None
    try:
        from . import views as agent_views
        if agent_views._shared_llm_loaded:
            llm = agent_views._shared_llm
    except Exception:
        pass

    if llm:
        writer_prompt = f"""You are a technical writer creating a data analysis report. The user requested: "{message}"

DATA CONTEXT:
- Dataset: {n_rows:,} rows × {n_cols} columns
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}

SUMMARY STATISTICS:
{stats_summary}

ANALYSES PERFORMED THIS SESSION:
{history_text if history_text else "No analyses run yet"}

Write a professional markdown report that includes:
1. Executive Summary (2-3 sentences answering what the user asked for)
2. Data Overview (brief description of the dataset)
3. Key Findings (based on the statistics and any analyses performed)
4. Recommendations (next steps for analysis)

Use proper markdown formatting with headers (##), bullet points, and tables where appropriate.
Keep it concise but informative (under 500 words)."""

        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm.generate, writer_prompt, max_tokens=800, temperature=0.7)
                llm_content = future.result(timeout=45)

            document = f"# Analysis Report\n\n"
            document += f"*Generated: {timestamp}*\n\n"
            document += llm_content + "\n\n"
            document += "---\n\n"
            document += "## Appendix: Data Summary\n\n"
            document += f"**Dataset:** {n_rows:,} rows × {n_cols} columns\n\n"
            document += "| Variable | Type | Missing | Unique |\n"
            document += "|----------|------|---------|--------|\n"
            for col in df.columns[:15]:
                dtype = "Numeric" if col in numeric_cols else "Categorical"
                missing = int(df[col].isna().sum())
                unique = int(df[col].nunique())
                document += f"| {col} | {dtype} | {missing} | {unique} |\n"

            response = "I've written a detailed analysis report based on your data and session history.\n\n"
            response += "The document includes an executive summary, key findings, and recommendations.\n\n"
            response += "Click the download link below to save."

            return response, document, filename

        except Exception as e:
            logger.warning(f"LLM writer failed: {e}")

    # Fallback: static document
    document = f"# Data Analysis Report\n\n"
    document += f"*Generated: {timestamp}*\n\n"
    document += f"*Request: {message}*\n\n"

    document += "## Data Overview\n\n"
    document += f"- **Rows:** {n_rows:,}\n"
    document += f"- **Columns:** {n_cols}\n"
    document += f"- **Numeric variables:** {len(numeric_cols)}\n"
    document += f"- **Categorical variables:** {len(cat_cols)}\n\n"

    document += "### Variables\n\n"
    document += "| Variable | Type | Missing | Unique |\n"
    document += "|----------|------|---------|--------|\n"
    for col in df.columns[:20]:
        dtype = "Numeric" if col in numeric_cols else "Categorical"
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique())
        document += f"| {col} | {dtype} | {missing} | {unique} |\n"

    if numeric_cols:
        document += "\n## Summary Statistics\n\n"
        document += "| Variable | Mean | Std Dev | Min | Max |\n"
        document += "|----------|------|---------|-----|-----|\n"
        for col in numeric_cols[:10]:
            stats = df[col].describe()
            document += f"| {col} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

    if session_history:
        document += "\n## Analyses Performed\n\n"
        for item in session_history[-10:]:
            document += f"- **{item.get('name', 'Analysis')}**: {item.get('summary', '')[:100]}\n"

    response = f"I've prepared a report document for your analysis.\n\n"
    response += f"**Document includes:**\n"
    response += f"- Data overview ({n_rows:,} rows × {n_cols} columns)\n"
    response += f"- Variable listing with types and missing values\n"
    if numeric_cols:
        response += f"- Summary statistics for {len(numeric_cols)} numeric variables\n"
    response += f"\nClick the download link below to save."

    return response, document, filename


# ============================================================================
# DATA TRANSFORMATION TOOLS
# ============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def transform_data(request):
    """
    Apply data transformation tools (subset, sort, calculator, etc.)
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tool = body.get("tool")
    config = body.get("config", {})
    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data loaded"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Load data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Apply transformation
        result_df = df.copy()
        message = ""

        if tool == "calculator":
            new_col = config.get("new_col", "").strip()
            expression = config.get("expression", "").strip()

            if not new_col or not expression:
                return JsonResponse({"error": "Column name and expression required"}, status=400)

            # Safe evaluation - only allow column names and basic math
            try:
                # Replace column names with df['column'] syntax
                safe_expr = expression
                for col in df.columns:
                    safe_expr = safe_expr.replace(col, f"df['{col}']")

                # Allow basic numpy functions
                safe_locals = {
                    'df': df,
                    'np': np,
                    'log': np.log,
                    'log10': np.log10,
                    'sqrt': np.sqrt,
                    'abs': np.abs,
                    'exp': np.exp,
                    'sin': np.sin,
                    'cos': np.cos,
                    'round': np.round,
                }

                result_df[new_col] = eval(safe_expr, {"__builtins__": {}}, safe_locals)
                message = f"Created column '{new_col}'"
            except Exception as e:
                return JsonResponse({"error": f"Expression error: {str(e)}"}, status=400)

        elif tool == "subset":
            filter_col = config.get("filter_col")
            condition = config.get("condition")
            filter_value = config.get("filter_value", "")

            if condition == "isna":
                result_df = df[df[filter_col].isna()]
            elif condition == "notna":
                result_df = df[df[filter_col].notna()]
            else:
                # Try to convert value to appropriate type
                try:
                    if df[filter_col].dtype in ['int64', 'float64']:
                        filter_value = float(filter_value)
                except:
                    pass

                if condition == "eq":
                    result_df = df[df[filter_col] == filter_value]
                elif condition == "ne":
                    result_df = df[df[filter_col] != filter_value]
                elif condition == "gt":
                    result_df = df[df[filter_col] > filter_value]
                elif condition == "gte":
                    result_df = df[df[filter_col] >= filter_value]
                elif condition == "lt":
                    result_df = df[df[filter_col] < filter_value]
                elif condition == "lte":
                    result_df = df[df[filter_col] <= filter_value]
                elif condition == "contains":
                    result_df = df[df[filter_col].astype(str).str.contains(str(filter_value), na=False)]

            message = f"Filtered to {len(result_df)} rows where {filter_col} {condition} {filter_value}"

        elif tool == "sort":
            sort_col = config.get("sort_col")
            order = config.get("order", "asc")

            result_df = df.sort_values(by=sort_col, ascending=(order == "asc")).reset_index(drop=True)
            message = f"Sorted by {sort_col} ({order}ending)"

        elif tool == "transpose":
            result_df = df.set_index(df.columns[0]).T.reset_index()
            result_df.columns = ['Variable'] + list(result_df.columns[1:])
            message = f"Transposed: {len(result_df)} rows × {len(result_df.columns)} columns"

        elif tool == "stack":
            operation = config.get("operation", "melt")

            if operation == "melt":
                id_cols = config.get("id_cols", [])
                if id_cols:
                    result_df = df.melt(id_vars=id_cols)
                else:
                    result_df = df.melt()
                message = f"Unpivoted to {len(result_df)} rows"

            elif operation == "pivot":
                index_col = config.get("index")
                pivot_col = config.get("pivot_col")
                values_col = config.get("values")

                result_df = df.pivot(index=index_col, columns=pivot_col, values=values_col).reset_index()
                result_df.columns.name = None
                message = f"Pivoted to {len(result_df)} rows × {len(result_df.columns)} columns"

        elif tool == "encode":
            columns = config.get("columns", [])
            method = config.get("method", "onehot")
            drop_first = config.get("drop_first", False)

            if not columns:
                return JsonResponse({"error": "Select columns to encode"}, status=400)
            columns = [c for c in columns if c in df.columns]

            if method == "onehot":
                result_df = pd.get_dummies(result_df, columns=columns, drop_first=drop_first, dtype=int)
                n_new = len(result_df.columns) - len(df.columns)
                message = f"One-hot encoded {len(columns)} column(s) → {n_new} new dummy columns"
            elif method == "label":
                for col in columns:
                    cats = result_df[col].astype(str).unique()
                    cats.sort()
                    mapping = {v: i for i, v in enumerate(cats)}
                    result_df[col] = result_df[col].astype(str).map(mapping)
                message = f"Label encoded {len(columns)} column(s)"
            else:
                return JsonResponse({"error": f"Unknown encoding method: {method}"}, status=400)

        elif tool == "scale":
            columns = config.get("columns", [])
            method = config.get("method", "zscore")

            if not columns:
                return JsonResponse({"error": "Select columns to scale"}, status=400)
            columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

            for col in columns:
                vals = result_df[col]
                if method == "zscore":
                    std = vals.std()
                    result_df[col] = (vals - vals.mean()) / std if std > 0 else 0
                elif method == "minmax":
                    mn, mx = vals.min(), vals.max()
                    result_df[col] = (vals - mn) / (mx - mn) if mx > mn else 0
                elif method == "robust":
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    result_df[col] = (vals - vals.median()) / iqr if iqr > 0 else 0
                else:
                    return JsonResponse({"error": f"Unknown scaling method: {method}"}, status=400)

            method_names = {"zscore": "Z-score", "minmax": "Min-Max [0,1]", "robust": "Robust (IQR)"}
            message = f"{method_names.get(method, method)} scaled {len(columns)} column(s)"

        elif tool == "bin":
            column = config.get("column", "")
            method = config.get("method", "equal_width")
            n_bins = int(config.get("n_bins", 5))
            custom_bins = config.get("bins", [])
            labels = config.get("labels", [])

            if not column or column not in df.columns:
                return JsonResponse({"error": "Select a valid column to bin"}, status=400)

            new_col = f"{column}_binned"
            try:
                if method == "equal_width":
                    result_df[new_col] = pd.cut(result_df[column], bins=n_bins, labels=labels or False)
                elif method == "equal_frequency":
                    result_df[new_col] = pd.qcut(result_df[column], q=n_bins, labels=labels or False, duplicates="drop")
                elif method == "custom":
                    if len(custom_bins) < 2:
                        return JsonResponse({"error": "Provide at least 2 breakpoints"}, status=400)
                    result_df[new_col] = pd.cut(result_df[column], bins=custom_bins, labels=labels or False, include_lowest=True)
                else:
                    return JsonResponse({"error": f"Unknown binning method: {method}"}, status=400)

                result_df[new_col] = result_df[new_col].astype(str)
                message = f"Binned {column} into '{new_col}' ({method}, {n_bins if method != 'custom' else len(custom_bins)-1} bins)"
            except Exception as e:
                return JsonResponse({"error": f"Binning error: {str(e)}"}, status=400)

        else:
            return JsonResponse({"error": f"Unknown tool: {tool}"}, status=400)

        # Save transformed data
        new_data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{new_data_id}.csv"
            result_df.to_csv(data_path, index=False)
        except Exception:
            data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{new_data_id}.csv"
            result_df.to_csv(data_path, index=False)

        # Build column info
        columns = []
        for col in result_df.columns:
            dtype = result_df[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"
            columns.append({"name": col, "dtype": col_type})

        return JsonResponse({
            "data_id": new_data_id,
            "filename": f"{tool}_{new_data_id[:8]}",
            "rows": len(result_df),
            "columns": columns,
            "preview": result_df.head(100).to_dict(orient="records"),
            "message": message
        })

    except Exception as e:
        logger.exception(f"Transform error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def download_data(request):
    """
    Download current data as CSV.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        from io import StringIO
        from django.http import HttpResponse

        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Return as CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{data_id}.csv"'
        df.to_csv(response, index=False)
        return response

    except Exception as e:
        logger.exception(f"Download error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_data(request):
    """
    Run Triage (data cleaning) on loaded data.

    Uses the scrub module to clean Excel errors, missing values, etc.
    Returns cleaned data as a new dataset.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO
        from scrub import DataCleaner, CleaningConfig

        # Load the data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Get user options
        options = body.get("options", {})
        fix_excel = options.get("fix_excel", True)
        fix_missing = options.get("fix_missing", False)
        missing_action = options.get("missing_action", "impute_mean")
        fix_outliers = options.get("fix_outliers", False)
        outlier_action = options.get("outlier_action", "flag")
        fix_types = options.get("fix_types", False)

        df_clean = df.copy()
        changes = {
            "excel_errors": 0,
            "missing_filled": 0,
            "missing_dropped": 0,
            "outliers_flagged": 0,
            "outliers_removed": 0,
            "outliers_clipped": 0,
            "types_converted": 0,
        }
        original_rows = len(df_clean)

        # 1. Fix Excel errors
        if fix_excel:
            excel_errors = ['#NUM!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#N/A', '#NULL!', '#ERROR!']
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    mask = df_clean[col].astype(str).str.upper().isin(excel_errors)
                    count = mask.sum()
                    if count > 0:
                        df_clean.loc[mask, col] = np.nan
                        changes["excel_errors"] += count

        # 2. Fix types (before missing/outlier handling)
        if fix_types:
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    # Try to convert to numeric
                    try:
                        converted = pd.to_numeric(df_clean[col].str.replace(',', ''), errors='coerce')
                        # Only convert if >80% success
                        success_rate = converted.notna().sum() / max(df_clean[col].notna().sum(), 1)
                        if success_rate > 0.8:
                            df_clean[col] = converted
                            changes["types_converted"] += 1
                    except Exception:
                        pass

        # 3. Handle missing values
        if fix_missing and missing_action != "leave":
            if missing_action == "drop_rows":
                before = len(df_clean)
                df_clean = df_clean.dropna()
                changes["missing_dropped"] = before - len(df_clean)
            elif missing_action in ["impute_mean", "impute_median"]:
                for col in df_clean.columns:
                    missing_count = df_clean[col].isna().sum()
                    if missing_count > 0:
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            if missing_action == "impute_mean":
                                fill_val = df_clean[col].mean()
                            else:
                                fill_val = df_clean[col].median()
                            df_clean[col] = df_clean[col].fillna(fill_val)
                        else:
                            # Mode for categorical
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                        changes["missing_filled"] += missing_count

        # 4. Handle outliers
        if fix_outliers:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        if outlier_action == "flag":
                            df_clean[f"{col}_outlier"] = outlier_mask.astype(int)
                            changes["outliers_flagged"] += outlier_count
                        elif outlier_action == "remove":
                            df_clean = df_clean[~outlier_mask]
                            changes["outliers_removed"] += outlier_count
                        elif outlier_action == "clip":
                            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                            changes["outliers_clipped"] += outlier_count

        # Save cleaned data
        new_data_id = f"data_{uuid.uuid4().hex[:8]}"
        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{new_data_id}.csv"
        df_clean.to_csv(data_path, index=False)

        # Build column info
        columns = []
        for col in df_clean.columns:
            dtype = df_clean[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"
            columns.append({"name": col, "dtype": col_type})

        # Generate preview - convert numpy types to Python native
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        preview_df = df_clean.head(100).replace({np.nan: None})
        preview = []
        for _, row in preview_df.iterrows():
            preview.append({k: convert_numpy(v) for k, v in row.items()})

        # Convert changes to Python int
        changes_clean = {k: int(v) for k, v in changes.items()}

        return JsonResponse({
            "success": True,
            "data": {
                "id": new_data_id,
                "rows": len(df_clean),
                "columns": columns,
                "preview": preview,
            },
            "changes": changes_clean,
            "rows_removed": int(original_rows - len(df_clean)),
        })

    except Exception as e:
        logger.exception(f"Triage error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_scan(request):
    """
    Scan dataset for issues WITHOUT cleaning.

    Returns detailed report of:
    - Excel error values (#NUM!, #DIV/0!, etc.)
    - Missing values per column
    - Potential outliers
    - Type issues (strings that should be numbers, etc.)
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Load the data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Scan for issues
        issues = {
            "excel_errors": {},
            "missing": {},
            "outliers": {},
            "type_issues": [],
            "totals": {
                "excel_errors": 0,
                "missing": 0,
                "outliers": 0,
                "type_issues": 0,
            }
        }

        excel_errors = ['#NUM!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#N/A', '#NULL!', '#ERROR!']

        for col in df.columns:
            # Check for Excel errors
            if df[col].dtype == object:
                error_mask = df[col].astype(str).str.upper().isin(excel_errors)
                error_count = error_mask.sum()
                if error_count > 0:
                    issues["excel_errors"][col] = int(error_count)
                    issues["totals"]["excel_errors"] += int(error_count)

            # Check for missing values
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues["missing"][col] = {
                    "count": int(missing_count),
                    "percent": round(float(missing_count / len(df) * 100), 1)
                }
                issues["totals"]["missing"] += int(missing_count)

            # Check for outliers (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                    if outlier_count > 0:
                        issues["outliers"][col] = {
                            "count": int(outlier_count),
                            "percent": round(float(outlier_count / len(df) * 100), 1),
                            "range": f"{lower:.2f} - {upper:.2f}"
                        }
                        issues["totals"]["outliers"] += int(outlier_count)

            # Check for type issues (strings that look like numbers)
            if df[col].dtype == object:
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', ''))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                if len(sample) > 0 and numeric_count / len(sample) > 0.8:
                    issues["type_issues"].append({
                        "column": col,
                        "current": "text",
                        "suggested": "numeric",
                        "confidence": round(numeric_count / len(sample) * 100, 1)
                    })
                    issues["totals"]["type_issues"] += 1

        # Determine if data has issues
        has_issues = (
            issues["totals"]["excel_errors"] > 0 or
            issues["totals"]["missing"] > 0 or
            issues["totals"]["outliers"] > 0 or
            issues["totals"]["type_issues"] > 0
        )

        return JsonResponse({
            "success": True,
            "has_issues": has_issues,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
        })

    except Exception as e:
        logger.exception(f"Triage scan error: {e}")
        return JsonResponse({"error": str(e)}, status=500)
